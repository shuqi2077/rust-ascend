use super::{Primitive,Result,buffer,check_queue};
use crate::{Ascend,Autodiff,driver::CannError,runtime::{AscendRuntime,RotaryLayout}};
use ruda_autodiff::{checkpoint::{base::Checkpointer,strategy::CheckpointStrategy},grads::Gradients,
    ops::{Backward,Ops,OpsKind}};
use ruda_core::tensor::Metadata;
use ruda_tensor::{Backend,TensorPrimitive,api::Tensor,tensor::FloatTensor};

/// Native rotation with fixed, caller-provided Ascend tables and input-only autodiff.
pub trait RotaryBackend:Backend {
    fn rotary(input:FloatTensor<Self>,cos:FloatTensor<Ascend>,sin:FloatTensor<Ascend>,layout:RotaryLayout)
        ->Result<FloatTensor<Self>>;
}
/// Full positive/even last-axis FP32 rotation with contiguous cos/sin tables of half width.
/// Leading dimensions must match exactly. Tables are fixed inner-backend tensors;
/// their frequency construction, position selection and any scaling are supplied by the caller.
pub fn rotary<B:RotaryBackend,const D:usize>(input:Tensor<B,D>,cos:Tensor<Ascend,D>,sin:Tensor<Ascend,D>,
    layout:RotaryLayout)->Result<Tensor<B,D>> {
    let input=match input.into_primitive() {
        TensorPrimitive::Float(input)=>input,
        TensorPrimitive::QFloat(_)=>return Err(CannError::InvalidTensor("native rotary requires unquantized FP32 input".into())),
    };
    let fixed=|table:Tensor<Ascend,D>|match table.into_primitive() {
        TensorPrimitive::Float(table)=>Ok(table),
        TensorPrimitive::QFloat(_)=>Err(CannError::InvalidTensor("native rotary requires unquantized FP32 tables".into())),
    };
    B::rotary(input,fixed(cos)?,fixed(sin)?,layout).map(|output|Tensor::from_primitive(TensorPrimitive::Float(output)))
}
fn rotate(input:Primitive,cos:Primitive,sin:Primitive,layout:RotaryLayout,backward:bool)->Result<Primitive> {
    check_queue(&input,&[&cos,&sin],"rotary")?;
    let client=input.client.clone();let device=input.device.clone();
    let output=if backward {AscendRuntime::rotary_backward(&client,buffer(input),buffer(cos),buffer(sin),layout)?}
        else {AscendRuntime::rotary(&client,buffer(input),buffer(cos),buffer(sin),layout)?};
    Ok(Primitive::new(client,output.handle,Metadata::new(output.shape,output.strides),device,output.dtype))
}
impl RotaryBackend for Ascend {
    fn rotary(input:FloatTensor<Self>,cos:FloatTensor<Ascend>,sin:FloatTensor<Ascend>,layout:RotaryLayout)
        ->Result<FloatTensor<Self>> {rotate(input,cos,sin,layout,false)}
}
#[derive(Debug)]
struct RotaryBackward;
impl Backward<Ascend,1> for RotaryBackward {
    type State=(Primitive,Primitive,RotaryLayout);
    fn backward(self,ops:Ops<Self::State,1>,grads:&mut Gradients,_:&mut Checkpointer) {
        let (cos,sin,layout)=ops.state;
        let grad=grads.consume::<Ascend>(&ops.node);
        let dx=rotate(grad,cos,sin,layout,true).expect("Ascend rotary backward failed");
        if let Some(parent)=ops.parents[0].as_ref() {grads.register::<Ascend>(parent.id,dx);}
    }
}
impl<C:CheckpointStrategy> RotaryBackend for Autodiff<Ascend,C> {
    fn rotary(input:FloatTensor<Self>,cos:FloatTensor<Ascend>,sin:FloatTensor<Ascend>,layout:RotaryLayout)
        ->Result<FloatTensor<Self>> {
        let output=rotate(input.primitive,cos.clone(),sin.clone(),layout,false)?;
        Ok(match RotaryBackward.prepare::<C>([input.node]).compute_bound().stateful() {
            OpsKind::Tracked(prep)=>prep.finish((cos,sin,layout),output),
            OpsKind::UnTracked(prep)=>prep.finish(output),
        })
    }
}

use super::{Primitive,Result,buffer,check_queue};
use crate::{Ascend,Autodiff,driver::CannError,runtime::{AscendRuntime,AscendDevice,ComputeClient}};
use ruda_autodiff::{checkpoint::{base::Checkpointer,strategy::CheckpointStrategy},grads::Gradients,ops::{Backward,Ops,OpsKind}};
use ruda_core::tensor::{Metadata,Shape,DType};
use ruda_tensor::{Backend,TensorPrimitive,api::Tensor,tensor::FloatTensor};

/// Explicit native FP32 last-axis affine addition; no broadcasting of residuals.
pub trait BiasAddBackend:Backend {
    fn bias_add(input:FloatTensor<Self>,bias:FloatTensor<Self>,residual:Option<FloatTensor<Self>>)->Result<FloatTensor<Self>>;
}
fn unquantized<B:Backend>(tensor:TensorPrimitive<B>)->Result<FloatTensor<B>> {
    match tensor {TensorPrimitive::Float(t)=>Ok(t),TensorPrimitive::QFloat(_)=>Err(CannError::InvalidTensor("native bias add requires unquantized FP32".into()))}
}
/// X[...,H] + bias[H], contiguous FP32 rank 1..8, positive H and total count <=u32.
/// Empty token axes return an empty output and a zero bias derivative.
pub fn bias_add<B:BiasAddBackend,const D:usize>(input:Tensor<B,D>,bias:Tensor<B,1>)->Result<Tensor<B,D>> {
    B::bias_add(unquantized(input.into_primitive())?,unquantized(bias.into_primitive())?,None)
        .map(|out|Tensor::from_primitive(TensorPrimitive::Float(out)))
}
/// (X + residual) + bias, in that FP32 order. Residual must exactly match X.
/// Q/K/V or Linear/LoRA outputs are not given an implicit bias/residual.
pub fn residual_bias_add<B:BiasAddBackend,const D:usize>(input:Tensor<B,D>,residual:Tensor<B,D>,bias:Tensor<B,1>)->Result<Tensor<B,D>> {
    B::bias_add(unquantized(input.into_primitive())?,unquantized(bias.into_primitive())?,Some(unquantized(residual.into_primitive())?))
        .map(|out|Tensor::from_primitive(TensorPrimitive::Float(out)))
}
fn forward(input:Primitive,bias:Primitive,residual:Option<Primitive>)->Result<Primitive> {
    let mut others=vec![&bias];if let Some(value)=&residual {others.push(value);}check_queue(&input,&others,"bias add")?;
    let client=input.client.clone();let device=input.device.clone();
    let out=AscendRuntime::bias_add(&client,buffer(input),buffer(bias),residual.map(buffer))?;
    Ok(Primitive::new(client,out.handle,Metadata::new(out.shape,out.strides),device,out.dtype))
}
impl BiasAddBackend for Ascend {
    fn bias_add(input:FloatTensor<Self>,bias:FloatTensor<Self>,residual:Option<FloatTensor<Self>>)->Result<FloatTensor<Self>> {
        forward(input,bias,residual)
    }
}
#[derive(Clone)]
struct AffineState {shape:Shape,client:ComputeClient<AscendRuntime>,device:AscendDevice}
impl std::fmt::Debug for AffineState {
    fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result {
        f.debug_struct("AffineState").field("shape",&self.shape).field("device",&self.device).finish_non_exhaustive()
    }
}
#[derive(Debug)]
struct BiasAddBackward<const N:usize>;
impl<const N:usize> Backward<Ascend,N> for BiasAddBackward<N> {
    type State=AffineState;
    fn backward(self,ops:Ops<Self::State,N>,grads:&mut Gradients,_:&mut Checkpointer) {
        let AffineState {shape,client,device}=ops.state;let grad=grads.consume::<Ascend>(&ops.node);
        assert!(grad.device==device && grad.client.same_execution_queue(&client),"bias add gradient device/queue mismatch");
        assert!(grad.dtype==DType::F32 && grad.meta.shape()==&shape,"bias add gradient dtype/shape mismatch");
        if let Some(parent)=ops.parents[1].as_ref() {
            let out=AscendRuntime::bias_add_backward(&client,buffer(grad.clone()),shape).expect("Ascend bias gradient reduction failed");
            grads.register::<Ascend>(parent.id,Primitive::new(client,out.handle,Metadata::new(out.shape,out.strides),device,out.dtype));
        }
        for (index,parent) in ops.parents.into_iter().enumerate() {
            if index==1 {continue;}
            if let Some(parent)=parent {grads.register::<Ascend>(parent.id,grad.clone());}
        }
    }
}
impl<C:CheckpointStrategy> BiasAddBackend for Autodiff<Ascend,C> {
    fn bias_add(input:FloatTensor<Self>,bias:FloatTensor<Self>,residual:Option<FloatTensor<Self>>)->Result<FloatTensor<Self>> {
        let state=AffineState {shape:input.primitive.meta.shape().clone(),client:input.primitive.client.clone(),device:input.primitive.device.clone()};
        if let Some(residual)=residual {
            let output=forward(input.primitive,bias.primitive,Some(residual.primitive))?;
            Ok(match BiasAddBackward::<3>.prepare::<C>([input.node,bias.node,residual.node]).compute_bound().stateful() {
                OpsKind::Tracked(prep)=>prep.finish(state,output),OpsKind::UnTracked(prep)=>prep.finish(output),
            })
        } else {
            let output=forward(input.primitive,bias.primitive,None)?;
            Ok(match BiasAddBackward::<2>.prepare::<C>([input.node,bias.node]).compute_bound().stateful() {
                OpsKind::Tracked(prep)=>prep.finish(state,output),OpsKind::UnTracked(prep)=>prep.finish(output),
            })
        }
    }
}

use super::{Primitive,Result,buffer};
use crate::{Ascend,Autodiff,driver::CannError,runtime::{AscendRuntime,AscendDevice,ComputeClient}};
use ruda_autodiff::{checkpoint::{base::Checkpointer,strategy::CheckpointStrategy},grads::Gradients,ops::{Backward,Ops,OpsKind}};
use ruda_core::tensor::{Metadata,Shape};
use ruda_tensor::{Backend,TensorPrimitive,api::Tensor,tensor::FloatTensor};
pub trait RepeatKvBackend:Backend {
    fn repeat_kv_heads(input:FloatTensor<Self>,query_heads:u32)->Result<FloatTensor<Self>>;
}
/// X[B,Hkv,N,D] -> Y[B,Hq,N,D]; consecutive Hq/Hkv heads share one KV head.
/// Native contiguous FP32 copy with device-side gradient summation; no input values saved.
pub fn repeat_kv_heads<B:RepeatKvBackend>(input:Tensor<B,4>,query_heads:usize)->Result<Tensor<B,4>> {
    let heads=u32::try_from(query_heads).map_err(|_|CannError::InvalidTensor("query heads exceed u32".into()))?;
    let input=match input.into_primitive() {TensorPrimitive::Float(input)=>input,
        TensorPrimitive::QFloat(_)=>return Err(CannError::InvalidTensor("KV repetition requires unquantized FP32".into()))};
    <B as RepeatKvBackend>::repeat_kv_heads(input,heads).map(|out|Tensor::from_primitive(TensorPrimitive::Float(out)))
}
fn forward(input:Primitive,heads:u32)->Result<Primitive> {
    let client=input.client.clone();let device=input.device.clone();
    let out=AscendRuntime::repeat_kv_heads(&client,buffer(input),heads)?;
    Ok(Primitive::new(client,out.handle,Metadata::new(out.shape,out.strides),device,out.dtype))
}
impl RepeatKvBackend for Ascend {
    fn repeat_kv_heads(input:FloatTensor<Self>,heads:u32)->Result<FloatTensor<Self>> {forward(input,heads)}
}
#[derive(Clone)]
struct RepeatState {shape:Shape,heads:u32,client:ComputeClient<AscendRuntime>,device:AscendDevice}
impl std::fmt::Debug for RepeatState {
    fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result {
        f.debug_struct("RepeatState").field("shape",&self.shape).field("heads",&self.heads).field("device",&self.device).finish_non_exhaustive()
    }
}
#[derive(Debug)]
struct RepeatBackward;
impl Backward<Ascend,1> for RepeatBackward {
    type State=RepeatState;
    fn backward(self,ops:Ops<Self::State,1>,grads:&mut Gradients,_:&mut Checkpointer) {
        let RepeatState {shape,heads,client,device}=ops.state;let grad=grads.consume::<Ascend>(&ops.node);
        assert!(grad.device==device && grad.client.same_execution_queue(&client),"KV repetition gradient device/queue mismatch");
        let out=AscendRuntime::repeat_kv_heads_backward(&client,shape,heads,buffer(grad)).expect("Ascend KV repetition backward failed");
        if let Some(parent)=ops.parents[0].as_ref() {
            grads.register::<Ascend>(parent.id,Primitive::new(client,out.handle,Metadata::new(out.shape,out.strides),device,out.dtype));
        }
    }
}
impl<C:CheckpointStrategy> RepeatKvBackend for Autodiff<Ascend,C> {
    fn repeat_kv_heads(input:FloatTensor<Self>,heads:u32)->Result<FloatTensor<Self>> {
        let state=RepeatState {shape:input.primitive.meta.shape().clone(),heads,client:input.primitive.client.clone(),device:input.primitive.device.clone()};
        let output=forward(input.primitive,heads)?;
        Ok(match RepeatBackward.prepare::<C>([input.node]).compute_bound().stateful() {
            OpsKind::Tracked(prep)=>prep.finish(state,output),OpsKind::UnTracked(prep)=>prep.finish(output),
        })
    }
}

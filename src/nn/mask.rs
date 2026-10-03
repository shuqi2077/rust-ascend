use super::{Primitive,Result};
use crate::{Ascend,Autodiff,driver::CannError,runtime::{AscendRuntime,CausalMaskSpec}};
use ruda_autodiff::checkpoint::strategy::CheckpointStrategy;
use ruda_core::tensor::Metadata;
use ruda_tensor::{Backend,TensorPrimitive,backend::AutodiffBackend,api::Tensor,tensor::{FloatTensor,Device}};
use crate::runtime::portable::backend::Runtime;
pub trait CausalMaskBackend:Backend {
    fn causal_mask(device:&Device<Self>,spec:CausalMaskSpec)->Result<FloatTensor<Self>>;
}
/// Fixed device-generated causal mask[B,Q,K], with explicit query/key absolute starts.
/// Returns +0 for allowed keys and exact -infinity for future keys; no trainable parent.
pub fn causal_mask<B:CausalMaskBackend>(device:&Device<B>,shape:[usize;3],query_start:u64,key_start:u64)->Result<Tensor<B,3>> {
    let [batch,queries,keys]=shape.map(|n|u32::try_from(n).map_err(|_|CannError::InvalidTensor("causal dimension exceeds u32".into())));
    let spec=CausalMaskSpec {batch:batch?,queries:queries?,keys:keys?,query_start,key_start};
    <B as CausalMaskBackend>::causal_mask(device,spec).map(|out|Tensor::from_primitive(TensorPrimitive::Float(out)))
}
fn forward(device:&Device<Ascend>,spec:CausalMaskSpec)->Result<Primitive> {
    let client=AscendRuntime::client(device);let out=AscendRuntime::causal_mask(&client,spec)?;
    Ok(Primitive::new(client,out.handle,Metadata::new(out.shape,out.strides),device.clone(),out.dtype))
}
impl CausalMaskBackend for Ascend {
    fn causal_mask(device:&Device<Self>,spec:CausalMaskSpec)->Result<FloatTensor<Self>> {forward(device,spec)}
}
impl<C:CheckpointStrategy> CausalMaskBackend for Autodiff<Ascend,C> {
    fn causal_mask(device:&Device<Self>,spec:CausalMaskSpec)->Result<FloatTensor<Self>> {forward(device,spec).map(<Self as AutodiffBackend>::from_inner)}
}

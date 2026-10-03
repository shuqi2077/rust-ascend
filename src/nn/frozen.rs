use super::{Primitive,Result,buffer,check_queue,LinearBf16Fp32Backend,SiluMulBackend,linear_bf16_fp32,silu_mul};
use crate::{Ascend,Autodiff,driver::CannError,runtime::AscendRuntime};
use ruda_autodiff::{checkpoint::{base::Checkpointer,strategy::CheckpointStrategy},grads::Gradients,ops::{Backward,Ops,OpsKind}};
use ruda_core::tensor::{Metadata,Shape};
use ruda_tensor::{Backend,TensorPrimitive,api::{Tensor,Int},tensor::{FloatTensor,IntTensor,Device}};
use crate::tensor::DType;

/// Explicit input-only differentiation with a fixed, already BF16-stored weight.
pub trait FrozenLinearBf16Fp32Backend:Backend<Device=Device<Ascend>> {
    fn linear_frozen_bf16_fp32(input:FloatTensor<Self>,weight:FloatTensor<Ascend>)->Result<FloatTensor<Self>>;
}
pub trait FrozenEmbeddingBf16Fp32Backend:Backend<Device=Device<Ascend>> {
    fn embedding_frozen_bf16_fp32(weight:FloatTensor<Ascend>,indices:IntTensor<Self>)->Result<FloatTensor<Self>>;
}
/// Fixed BF16 table[V,H] and INT32/INT64 IDs[B,S] produce FP32 activations[B,S,H].
pub fn embedding_frozen_bf16_fp32<B:FrozenEmbeddingBf16Fp32Backend>(weight:Tensor<Ascend,2>,indices:Tensor<B,2,Int>)->Result<Tensor<B,3>> {
    embedding_frozen_bf16_fp32_nd::<B,2,3>(weight,indices)
}
/// IDs rank D is 1..7; output rank O is D+1. No table gradient or floating-ID conversion.
pub fn embedding_frozen_bf16_fp32_nd<B:FrozenEmbeddingBf16Fp32Backend,const D:usize,const O:usize>(weight:Tensor<Ascend,2>,indices:Tensor<B,D,Int>)->Result<Tensor<B,O>> {
    if !(1..=7).contains(&D) || D.checked_add(1)!=Some(O) {return Err(CannError::InvalidTensor("frozen embedding output rank must append one axis to rank 1..7 IDs".into()));}
    let weight=match weight.into_primitive() {TensorPrimitive::Float(weight)=>weight,
        TensorPrimitive::QFloat(_)=>return Err(CannError::InvalidTensor("frozen BF16 embedding requires unquantized BF16 table".into()))};
    <B as FrozenEmbeddingBf16Fp32Backend>::embedding_frozen_bf16_fp32(weight,indices.into_primitive())
        .map(|out|Tensor::from_primitive(TensorPrimitive::Float(out)))
}
fn embedding_forward(weight:Primitive,indices:Primitive)->Result<Primitive> {
    check_queue(&weight,&[&indices],"frozen BF16 embedding")?;
    let client=weight.client.clone();let device=weight.device.clone();
    let out=AscendRuntime::embedding_frozen_bf16_fp32(&client,buffer(weight),buffer(indices))?;
    Ok(Primitive::new(client,out.handle,Metadata::new(out.shape,out.strides),device,out.dtype))
}
impl FrozenEmbeddingBf16Fp32Backend for Ascend {
    fn embedding_frozen_bf16_fp32(weight:FloatTensor<Ascend>,indices:IntTensor<Self>)->Result<FloatTensor<Self>> {embedding_forward(weight,indices)}
}
impl<C:CheckpointStrategy> FrozenEmbeddingBf16Fp32Backend for Autodiff<Ascend,C> {
    fn embedding_frozen_bf16_fp32(weight:FloatTensor<Ascend>,indices:IntTensor<Self>)->Result<FloatTensor<Self>> {
        embedding_forward(weight,indices).map(<Self as ruda_tensor::backend::AutodiffBackend>::from_inner)
    }
}
/// FP32 X[M,K] @ fixed BF16 W[N,K]^T -> FP32 Y[M,N].
/// X and dY are rounded to BF16 on device; dX is FP32. M/N/K are positive multiples of 16.
/// Does not cast/copy W, save X, or compute dW; caller keeps W unchanged through backward.
pub fn linear_frozen_bf16_fp32<B:FrozenLinearBf16Fp32Backend>(input:Tensor<B,2>,weight:Tensor<Ascend,2>)->Result<Tensor<B,2>> {
    let input=match input.into_primitive() {TensorPrimitive::Float(input)=>input,
        TensorPrimitive::QFloat(_)=>return Err(CannError::InvalidTensor("frozen BF16 linear requires unquantized FP32 input".into()))};
    let weight=match weight.into_primitive() {TensorPrimitive::Float(weight)=>weight,
        TensorPrimitive::QFloat(_)=>return Err(CannError::InvalidTensor("frozen BF16 linear requires unquantized BF16 weight".into()))};
    <B as FrozenLinearBf16Fp32Backend>::linear_frozen_bf16_fp32(input,weight).map(|out|Tensor::from_primitive(TensorPrimitive::Float(out)))
}
fn forward(input:Primitive,weight:Primitive)->Result<Primitive> {
    check_queue(&input,&[&weight],"frozen BF16 linear")?;
    let client=input.client.clone();let device=input.device.clone();
    let out=AscendRuntime::linear_frozen_bf16_fp32(&client,buffer(input),buffer(weight))?;
    Ok(Primitive::new(client,out.handle,Metadata::new(out.shape,out.strides),device,out.dtype))
}
impl FrozenLinearBf16Fp32Backend for Ascend {
    fn linear_frozen_bf16_fp32(input:FloatTensor<Self>,weight:FloatTensor<Ascend>)->Result<FloatTensor<Self>> {forward(input,weight)}
}
#[derive(Debug)]
struct FrozenLinearBackward;
impl Backward<Ascend,1> for FrozenLinearBackward {
    type State=(Shape,Primitive);
    fn backward(self,ops:Ops<Self::State,1>,grads:&mut Gradients,_:&mut Checkpointer) {
        let (shape,weight)=ops.state;let grad=grads.consume::<Ascend>(&ops.node);check_queue(&weight,&[&grad],"frozen BF16 linear backward").expect("Ascend frozen linear gradient queue mismatch");
        let client=weight.client.clone();let device=weight.device.clone();
        let out=AscendRuntime::linear_frozen_bf16_fp32_backward(&client,shape,buffer(weight),buffer(grad)).expect("Ascend frozen BF16 linear backward failed");
        if let Some(parent)=ops.parents[0].as_ref() {grads.register::<Ascend>(parent.id,Primitive::new(client,out.handle,Metadata::new(out.shape,out.strides),device,out.dtype));}
    }
}
impl<C:CheckpointStrategy> FrozenLinearBf16Fp32Backend for Autodiff<Ascend,C> {
    fn linear_frozen_bf16_fp32(input:FloatTensor<Self>,weight:FloatTensor<Ascend>)->Result<FloatTensor<Self>> {
        let shape=input.primitive.meta.shape().clone();let output=forward(input.primitive,weight.clone())?;
        Ok(match FrozenLinearBackward.prepare::<C>([input.node]).compute_bound().stateful() {
            OpsKind::Tracked(prep)=>prep.finish((shape,weight),output),OpsKind::UnTracked(prep)=>prep.finish(output),
        })
    }
}
fn aligned(sizes:&[usize])->bool {sizes.iter().all(|&d|d!=0 && d%16==0 && d<=i32::MAX as usize)}
/// X @ fixed BF16 W^T + scale * (X @ FP32 A^T) @ FP32 B^T, in explicit BF16 compute mode.
/// Only X and the FP32 adapters may be trainable; no global freezing or weight merging is performed.
pub fn lora_frozen_linear_bf16_fp32<B:FrozenLinearBf16Fp32Backend+LinearBf16Fp32Backend>(
    input:Tensor<B,2>,weight:Tensor<Ascend,2>,down:Tensor<B,2>,up:Tensor<B,2>,scale:f32)->Result<Tensor<B,2>> {
    let [m,k]=input.dims();let [n,wk]=weight.dims();let [rank,ak]=down.dims();
    if wk!=k || ak!=k || up.dims()!=[n,rank] || !aligned(&[m,n,k,rank]) || !scale.is_finite() {
        return Err(CannError::InvalidTensor("frozen BF16 LoRA requires aligned X[M,K], W[N,K], A[R,K], B[N,R] and finite scale".into()));
    }
    let device=input.device();
    if [&input,&down,&up].iter().any(|x|x.dtype()!=DType::F32 || x.device()!=device) || weight.dtype()!=DType::BF16 || weight.device()!=device {
        return Err(CannError::InvalidTensor("frozen BF16 LoRA requires FP32 activations/adapters and fixed BF16 W on the same device".into()));
    }
    let base=linear_frozen_bf16_fp32(input.clone(),weight)?;
    let adapter=linear_bf16_fp32(linear_bf16_fp32(input,down)?,up)?;Ok(base+adapter*scale)
}
/// SwiGLU with fixed BF16-stored gate/up/down weights and FP32 activation/input gradients.
pub fn swiglu_frozen_bf16_fp32<B:FrozenLinearBf16Fp32Backend+SiluMulBackend>(input:Tensor<B,2>,gate:Tensor<Ascend,2>,
    up:Tensor<Ascend,2>,down:Tensor<Ascend,2>)->Result<Tensor<B,2>> {
    let [m,k]=input.dims();let [hidden,gk]=gate.dims();let [n,dh]=down.dims();
    if gk!=k || up.dims()!=[hidden,k] || dh!=hidden || !aligned(&[m,n,k,hidden]) {
        return Err(CannError::InvalidTensor("frozen BF16 SwiGLU requires aligned X[M,K], gate/up[H,K], down[N,H]".into()));
    }
    let device=input.device();
    if input.dtype()!=DType::F32 || [&gate,&up,&down].iter().any(|w|w.dtype()!=DType::BF16 || w.device()!=device) {
        return Err(CannError::InvalidTensor("frozen BF16 SwiGLU requires FP32 input and fixed BF16 weights on the same device".into()));
    }
    let gate=linear_frozen_bf16_fp32(input.clone(),gate)?;let up=linear_frozen_bf16_fp32(input,up)?;
    linear_frozen_bf16_fp32(silu_mul(gate,up)?,down)
}

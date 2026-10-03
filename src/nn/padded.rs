use super::{Primitive,Result,buffer,check_queue,SiluMulBackend,silu_mul};
use crate::{Ascend,Autodiff,driver::CannError,runtime::{AscendRuntime,Transpose},tensor::DType};
use ruda_autodiff::{checkpoint::{base::Checkpointer,strategy::CheckpointStrategy},grads::Gradients,ops::{Backward,Ops,OpsKind}};
use ruda_core::tensor::{Metadata,Shape};
use ruda_tensor::{Backend,TensorPrimitive,api::Tensor,tensor::{FloatTensor,Device}};

/// Opt-in BF16 compute with explicit on-device zero padding and FP32 output crop.
pub trait PaddedMatmulBf16Fp32Backend:Backend {
    fn matmul_padded_bf16_fp32(a:FloatTensor<Self>,b:FloatTensor<Self>,ta:Transpose,tb:Transpose)->Result<FloatTensor<Self>>;
}
pub trait FrozenPaddedLinearBf16Fp32Backend:Backend<Device=Device<Ascend>> {
    fn linear_frozen_padded_bf16_fp32(input:FloatTensor<Self>,weight:FloatTensor<Ascend>)->Result<FloatTensor<Self>>;
}
fn unquantized<B:Backend>(tensor:TensorPrimitive<B>)->Result<FloatTensor<B>> {
    match tensor {TensorPrimitive::Float(t)=>Ok(t),TensorPrimitive::QFloat(_)=>Err(CannError::InvalidTensor("padded linear/matmul does not dequantize inputs implicitly".into()))}
}
/// Positive M/N/K, rank-2 or matching-batch rank-3, all four transpose choices.
/// Rounds X/W/dY to BF16 on device; outputs and first-order derivatives are FP32.
/// Native dimensions round up to 16; no bias, batch broadcast or CPU fallback.
pub fn matmul_padded_bf16_fp32<B:PaddedMatmulBf16Fp32Backend,const D:usize>(a:Tensor<B,D>,b:Tensor<B,D>,ta:Transpose,tb:Transpose)->Result<Tensor<B,D>> {
    B::matmul_padded_bf16_fp32(unquantized(a.into_primitive())?,unquantized(b.into_primitive())?,ta,tb)
        .map(|out|Tensor::from_primitive(TensorPrimitive::Float(out)))
}
/// X[M,K] W[N,K]^T with explicit padding; no positive-axis 16-divisibility requirement.
pub fn linear_padded_bf16_fp32<B:PaddedMatmulBf16Fp32Backend>(input:Tensor<B,2>,weight:Tensor<B,2>)->Result<Tensor<B,2>> {
    matmul_padded_bf16_fp32(input,weight,Transpose::No,Transpose::Yes)
}
fn forward(a:Primitive,b:Primitive,ta:Transpose,tb:Transpose)->Result<[Primitive;3]> {
    check_queue(&a,&[&b],"padded BF16-compute matmul")?;let client=a.client.clone();let device=a.device.clone();
    let result=AscendRuntime::gemm_padded_bf16_fp32(&client,buffer(a),buffer(b),ta,tb)?;
    Ok(result.map(|out|Primitive::new(client.clone(),out.handle,Metadata::new(out.shape,out.strides),device.clone(),out.dtype)))
}
impl PaddedMatmulBf16Fp32Backend for Ascend {
    fn matmul_padded_bf16_fp32(a:FloatTensor<Self>,b:FloatTensor<Self>,ta:Transpose,tb:Transpose)->Result<FloatTensor<Self>> {
        forward(a,b,ta,tb).map(|[out,_,_]|out)
    }
}
#[derive(Debug)]
struct PaddedMatmulBackward;
impl Backward<Ascend,2> for PaddedMatmulBackward {
    type State=(Primitive,Primitive,Shape,Shape,Transpose,Transpose);
    fn backward(self,ops:Ops<Self::State,2>,grads:&mut Gradients,_:&mut Checkpointer) {
        let (a,b,a_shape,b_shape,ta,tb)=ops.state;let grad=grads.consume::<Ascend>(&ops.node);
        check_queue(&a,&[&b,&grad],"padded matmul backward").expect("Ascend padded matmul gradient queue mismatch");
        let client=a.client.clone();let device=a.device.clone();
        let result=AscendRuntime::gemm_padded_bf16_fp32_backward(&client,buffer(a),buffer(b),buffer(grad),a_shape,b_shape,ta,tb)
            .expect("Ascend padded BF16 matmul backward failed");
        for (parent,out) in ops.parents.into_iter().zip(result) {if let Some(parent)=parent {
            grads.register::<Ascend>(parent.id,Primitive::new(client.clone(),out.handle,Metadata::new(out.shape,out.strides),device.clone(),out.dtype));
        }}
    }
}
impl<C:CheckpointStrategy> PaddedMatmulBf16Fp32Backend for Autodiff<Ascend,C> {
    fn matmul_padded_bf16_fp32(a:FloatTensor<Self>,b:FloatTensor<Self>,ta:Transpose,tb:Transpose)->Result<FloatTensor<Self>> {
        let a_shape=a.primitive.meta.shape().clone();let b_shape=b.primitive.meta.shape().clone();
        let [out,a_saved,b_saved]=forward(a.primitive,b.primitive,ta,tb)?;
        Ok(match PaddedMatmulBackward.prepare::<C>([a.node,b.node]).compute_bound().stateful() {
            OpsKind::Tracked(prep)=>prep.finish((a_saved,b_saved,a_shape,b_shape,ta,tb),out),OpsKind::UnTracked(prep)=>prep.finish(out),
        })
    }
}
/// FP32 X and fixed BF16 W, with input-only differentiation and logical tails.
/// A weight tail creates a padded BF16 copy; aligned weights reuse the original
/// handle. Keep the fixed weight unchanged through backward. No FP32 expansion.
pub fn linear_frozen_padded_bf16_fp32<B:FrozenPaddedLinearBf16Fp32Backend>(input:Tensor<B,2>,weight:Tensor<Ascend,2>)->Result<Tensor<B,2>> {
    B::linear_frozen_padded_bf16_fp32(unquantized(input.into_primitive())?,unquantized(weight.into_primitive())?)
        .map(|out|Tensor::from_primitive(TensorPrimitive::Float(out)))
}
fn frozen_forward(input:Primitive,weight:Primitive)->Result<[Primitive;2]> {
    check_queue(&input,&[&weight],"frozen padded BF16 linear")?;let client=input.client.clone();let device=input.device.clone();
    let result=AscendRuntime::linear_frozen_padded_bf16_fp32(&client,buffer(input),buffer(weight))?;
    Ok(result.map(|out|Primitive::new(client.clone(),out.handle,Metadata::new(out.shape,out.strides),device.clone(),out.dtype)))
}
impl FrozenPaddedLinearBf16Fp32Backend for Ascend {
    fn linear_frozen_padded_bf16_fp32(input:FloatTensor<Self>,weight:FloatTensor<Ascend>)->Result<FloatTensor<Self>> {
        frozen_forward(input,weight).map(|[out,_]|out)
    }
}
#[derive(Debug)]
struct FrozenPaddedLinearBackward;
impl Backward<Ascend,1> for FrozenPaddedLinearBackward {
    type State=(Primitive,Shape,Shape);
    fn backward(self,ops:Ops<Self::State,1>,grads:&mut Gradients,_:&mut Checkpointer) {
        let (weight,input_shape,weight_shape)=ops.state;let grad=grads.consume::<Ascend>(&ops.node);
        check_queue(&weight,&[&grad],"frozen padded linear backward").expect("Ascend frozen padded gradient queue mismatch");
        let client=weight.client.clone();let device=weight.device.clone();
        let out=AscendRuntime::linear_frozen_padded_bf16_fp32_backward(&client,buffer(weight),buffer(grad),input_shape,weight_shape)
            .expect("Ascend frozen padded BF16 linear backward failed");
        if let Some(parent)=ops.parents[0].as_ref() {grads.register::<Ascend>(parent.id,Primitive::new(client,out.handle,Metadata::new(out.shape,out.strides),device,out.dtype));}
    }
}
impl<C:CheckpointStrategy> FrozenPaddedLinearBf16Fp32Backend for Autodiff<Ascend,C> {
    fn linear_frozen_padded_bf16_fp32(input:FloatTensor<Self>,weight:FloatTensor<Ascend>)->Result<FloatTensor<Self>> {
        let input_shape=input.primitive.meta.shape().clone();let weight_shape=weight.meta.shape().clone();
        let [out,weight]=frozen_forward(input.primitive,weight)?;
        Ok(match FrozenPaddedLinearBackward.prepare::<C>([input.node]).compute_bound().stateful() {
            OpsKind::Tracked(prep)=>prep.finish((weight,input_shape,weight_shape),out),OpsKind::UnTracked(prep)=>prep.finish(out),
        })
    }
}
fn positive(sizes:&[usize])->bool {sizes.iter().all(|&size|size>0)}
/// Same LoRA formula and gradient graph, with any positive rank (including 1/7/8).
/// Each intermediate is cropped before the following zero-pad, so artificial
/// channels never become model activations. Scale is explicit and finite.
pub fn lora_padded_linear_bf16_fp32<B:PaddedMatmulBf16Fp32Backend>(input:Tensor<B,2>,weight:Tensor<B,2>,down:Tensor<B,2>,up:Tensor<B,2>,scale:f32)->Result<Tensor<B,2>> {
    let [m,k]=input.dims();let [n,wk]=weight.dims();let [rank,ak]=down.dims();
    if wk!=k || ak!=k || up.dims()!=[n,rank] || !positive(&[m,n,k,rank]) || !scale.is_finite() {
        return Err(CannError::InvalidTensor("padded LoRA requires positive X[M,K], W[N,K], A[R,K], B[N,R] and finite scale".into()));
    }
    let device=input.device();if [&input,&weight,&down,&up].iter().any(|t|t.dtype()!=DType::F32 || t.device()!=device) {
        return Err(CannError::InvalidTensor("padded LoRA requires FP32 inputs/weights on the same device".into()));
    }
    let base=linear_padded_bf16_fp32(input.clone(),weight)?;
    let adapter=linear_padded_bf16_fp32(linear_padded_bf16_fp32(input,down)?,up)?;Ok(base+adapter*scale)
}
pub fn lora_frozen_padded_linear_bf16_fp32<B:FrozenPaddedLinearBf16Fp32Backend+PaddedMatmulBf16Fp32Backend>(
    input:Tensor<B,2>,weight:Tensor<Ascend,2>,down:Tensor<B,2>,up:Tensor<B,2>,scale:f32)->Result<Tensor<B,2>> {
    let [m,k]=input.dims();let [n,wk]=weight.dims();let [rank,ak]=down.dims();
    if wk!=k || ak!=k || up.dims()!=[n,rank] || !positive(&[m,n,k,rank]) || !scale.is_finite() {
        return Err(CannError::InvalidTensor("frozen padded LoRA requires positive matching X/W/A/B and finite scale".into()));
    }
    let device=input.device();if weight.dtype()!=DType::BF16 || weight.device()!=device
        || [&input,&down,&up].iter().any(|t|t.dtype()!=DType::F32 || t.device()!=device) {
        return Err(CannError::InvalidTensor("frozen padded LoRA requires fixed BF16 W and FP32 X/A/B on the same device".into()));
    }
    let base=linear_frozen_padded_bf16_fp32(input.clone(),weight)?;
    let adapter=linear_padded_bf16_fp32(linear_padded_bf16_fp32(input,down)?,up)?;Ok(base+adapter*scale)
}
pub fn swiglu_padded_bf16_fp32<B:PaddedMatmulBf16Fp32Backend+SiluMulBackend>(input:Tensor<B,2>,gate:Tensor<B,2>,up:Tensor<B,2>,down:Tensor<B,2>)->Result<Tensor<B,2>> {
    let [m,k]=input.dims();let [h,gk]=gate.dims();let [n,dh]=down.dims();
    if gk!=k || up.dims()!=[h,k] || dh!=h || !positive(&[m,n,k,h]) {return Err(CannError::InvalidTensor("padded SwiGLU requires positive matching X/gate/up/down".into()));}
    let device=input.device();if [&input,&gate,&up,&down].iter().any(|t|t.dtype()!=DType::F32 || t.device()!=device) {return Err(CannError::InvalidTensor("padded SwiGLU requires same-device FP32 inputs/weights".into()));}
    let gate=linear_padded_bf16_fp32(input.clone(),gate)?;let up=linear_padded_bf16_fp32(input,up)?;
    linear_padded_bf16_fp32(silu_mul(gate,up)?,down)
}
pub fn swiglu_frozen_padded_bf16_fp32<B:FrozenPaddedLinearBf16Fp32Backend+SiluMulBackend>(input:Tensor<B,2>,gate:Tensor<Ascend,2>,up:Tensor<Ascend,2>,down:Tensor<Ascend,2>)->Result<Tensor<B,2>> {
    let [m,k]=input.dims();let [h,gk]=gate.dims();let [n,dh]=down.dims();
    if gk!=k || up.dims()!=[h,k] || dh!=h || !positive(&[m,n,k,h]) {return Err(CannError::InvalidTensor("frozen padded SwiGLU requires positive matching X/gate/up/down".into()));}
    let device=input.device();if input.dtype()!=DType::F32 || [&gate,&up,&down].iter().any(|t|t.dtype()!=DType::BF16 || t.device()!=device) {return Err(CannError::InvalidTensor("frozen padded SwiGLU requires FP32 X/fixed BF16 weights on the same device".into()));}
    let gate=linear_frozen_padded_bf16_fp32(input.clone(),gate)?;let up=linear_frozen_padded_bf16_fp32(input,up)?;
    linear_frozen_padded_bf16_fp32(silu_mul(gate,up)?,down)
}

use super::{LinearBf16Fp32Backend,SiluMulBackend,Result,linear_bf16_fp32,silu_mul};
use crate::{driver::CannError,tensor::{DType,api::Tensor}};

/// X W^T + scale * (X A^T) B^T with FP32 storage/output and explicit BF16 linear compute.
/// W[N,K], A[R,K], B[N,R]; all M/N/K/R must be positive multiples of 16.
/// Scale and which inputs require gradients are selected by the caller; no implicit freezing.
/// No dropout, bias, merged-weight update, quantization or padding is installed here.
pub fn lora_linear_bf16_fp32<B:LinearBf16Fp32Backend>(input:Tensor<B,2>,weight:Tensor<B,2>,
    down:Tensor<B,2>,up:Tensor<B,2>,scale:f32)->Result<Tensor<B,2>> {
    let [m,k]=input.dims();let [n,wk]=weight.dims();let [rank,ak]=down.dims();let [bn,br]=up.dims();
    if wk!=k || ak!=k || bn!=n || br!=rank || !scale.is_finite()
        || [m,n,k,rank].iter().any(|&size|size==0 || size%16!=0 || size>i32::MAX as usize) {
        return Err(CannError::InvalidTensor("LoRA requires aligned X[M,K], W[N,K], A[R,K], B[N,R] and finite scale".into()));
    }
    let device=input.device();
    if [&input,&weight,&down,&up].iter().any(|tensor|tensor.dtype()!=DType::F32 || tensor.device()!=device) {
        return Err(CannError::InvalidTensor("LoRA requires FP32 input and weights on the same device".into()));
    }
    let base=linear_bf16_fp32(input.clone(),weight)?;
    let low_rank=linear_bf16_fp32(linear_bf16_fp32(input,down)?,up)?;
    Ok(base+low_rank*scale)
}

/// (SiLU(X Wgate^T) * (X Wup^T)) Wdown^T, with FP32 activation and BF16 linear compute.
/// Gate/up weights are [H,K], down is [N,H]; M/N/K/H must be positive multiples of 16.
/// Uses native common-IR gated activation and the existing RUDA gradient graph.
/// No model-specific dimensions, bias, dropout, residual or normalization are inferred.
pub fn swiglu_bf16_fp32<B:LinearBf16Fp32Backend+SiluMulBackend>(input:Tensor<B,2>,gate:Tensor<B,2>,
    up:Tensor<B,2>,down:Tensor<B,2>)->Result<Tensor<B,2>> {
    let [m,k]=input.dims();let [hidden,gk]=gate.dims();let [n,dh]=down.dims();
    if gk!=k || up.dims()!=[hidden,k] || dh!=hidden
        || [m,n,k,hidden].iter().any(|&size|size==0 || size%16!=0 || size>i32::MAX as usize) {
        return Err(CannError::InvalidTensor("SwiGLU requires aligned X[M,K], gate/up[H,K] and down[N,H]".into()));
    }
    let device=input.device();
    if [&input,&gate,&up,&down].iter().any(|tensor|tensor.dtype()!=DType::F32 || tensor.device()!=device) {
        return Err(CannError::InvalidTensor("SwiGLU requires FP32 input and weights on the same device".into()));
    }
    let gate=linear_bf16_fp32(input.clone(),gate)?;
    let up=linear_bf16_fp32(input,up)?;
    linear_bf16_fp32(silu_mul(gate,up)?,down)
}

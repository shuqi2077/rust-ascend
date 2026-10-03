use super::{MatmulBf16Fp32Backend,SoftmaxBackend,Result,matmul_bf16_fp32,softmax};
use crate::{driver::CannError,runtime::Transpose,tensor::{DType,api::Tensor}};

/// Composed attention for Q[B,M,D], K[B,N,D] and V[B,N,Dv], using explicit BF16 GEMMs.
/// Scores, Softmax, output, parameter storage and gradients are FP32. GEMM inputs and
/// upstream GEMM gradients are rounded to BF16 by the explicit matmul compute mode.
/// The optional additive FP32 mask has exact shape [B,M,N]; -infinity excludes keys.
/// Scale is supplied explicitly. No implicit causal mask, dropout, head repetition or padding.
/// This materializes scores/probabilities; it is not FlashAttention.
pub fn scaled_dot_product_attention_bf16_fp32<B:MatmulBf16Fp32Backend+SoftmaxBackend>(
    query:Tensor<B,3>,key:Tensor<B,3>,value:Tensor<B,3>,scale:f32,
    additive_mask:Option<Tensor<B,3>>)->Result<Tensor<B,3>> {
    let [batch,m,d]=query.dims();let [kb,n,kd]=key.dims();let [vb,vn,dv]=value.dims();
    if batch==0 || batch>4096 || kb!=batch || vb!=batch || kd!=d || vn!=n
        || [m,d,dv].iter().any(|&size|size==0 || size%16!=0 || size>i32::MAX as usize)
        || !(32..=4096).contains(&n) || n%32!=0 || !scale.is_finite()
        || batch.checked_mul(m).and_then(|size|size.checked_mul(n)).is_none_or(|size|size>u32::MAX as usize) {
        return Err(CannError::InvalidTensor("attention requires matching batches/contractions, aligned positive matrix dimensions, N=32..4096 divisible by 32, a finite FP32 scale and scores within u32".into()));
    }
    let device=query.device();
    if query.dtype()!=DType::F32 || key.dtype()!=DType::F32 || value.dtype()!=DType::F32
        || key.device()!=device || value.device()!=device {
        return Err(CannError::InvalidTensor("attention requires FP32 Q/K/V on the same device".into()));
    }
    if let Some(mask)=&additive_mask {
        if mask.dims()!=[batch,m,n] || mask.dtype()!=DType::F32 || mask.device()!=device {
            return Err(CannError::InvalidTensor("attention additive mask must be FP32 [B,M,N] on the same device".into()));
        }
    }
    let mut scores=matmul_bf16_fp32(query,key,Transpose::No,Transpose::Yes)?*scale;
    if let Some(mask)=additive_mask {scores=scores+mask;}
    let probabilities=softmax(scores)?;
    matmul_bf16_fp32(probabilities,value,Transpose::No,Transpose::No)
}

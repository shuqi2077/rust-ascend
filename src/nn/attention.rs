use super::{MatmulBf16Fp32Backend,SoftmaxBackend,CausalMaskBackend,Result,matmul_bf16_fp32,softmax,causal_mask};
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
    validate_dimensions(query.dims(),key.dims(),value.dims(),scale)?;
    let [batch,m,_]=query.dims();let [_,n,_]=key.dims();
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
/// Explicit causal attention using a fixed device-generated mask and absolute positions.
/// Query/key starts are provided for prefill, chunked queries or a retained KV prefix.
/// Retains the existing BF16-compute/FP32-storage matrix and materialized Softmax path.
pub fn causal_attention_bf16_fp32<B:MatmulBf16Fp32Backend+SoftmaxBackend+CausalMaskBackend>(
    query:Tensor<B,3>,key:Tensor<B,3>,value:Tensor<B,3>,scale:f32,query_start:u64,key_start:u64)
    ->Result<Tensor<B,3>> {
    validate_dimensions(query.dims(),key.dims(),value.dims(),scale)?;
    let [batch,queries,_]=query.dims();let keys=key.dims()[1];
    if query.dtype()!=DType::F32 || key.dtype()!=DType::F32 || value.dtype()!=DType::F32
        || key.device()!=query.device() || value.device()!=query.device() {
        return Err(CannError::InvalidTensor("attention requires FP32 Q/K/V on the same device".into()));
    }
    let mask=causal_mask::<B>(&query.device(),[batch,queries,keys],query_start,key_start)?;
    scaled_dot_product_attention_bf16_fp32(query,key,value,scale,Some(mask))
}
fn validate_dimensions(query:[usize;3],key:[usize;3],value:[usize;3],scale:f32)->Result<()> {
    let [batch,m,d]=query;let [kb,n,kd]=key;let [vb,vn,dv]=value;
    if batch==0 || batch>4096 || kb!=batch || vb!=batch || kd!=d || vn!=n
        || [m,d,dv].iter().any(|&size|size==0 || size%16!=0 || size>i32::MAX as usize)
        || n==0 || n>i32::MAX as usize || n%32!=0 || !scale.is_finite()
        || batch.checked_mul(m).and_then(|size|size.checked_mul(n)).is_none_or(|size|size>u32::MAX as usize) {
        return Err(CannError::InvalidTensor("attention requires matching batches/contractions, aligned positive matrix dimensions, positive N divisible by 32 within INT32_MAX, a finite FP32 scale and scores within u32".into()));
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn long_context_attention_uses_aligned_shapes_without_a_4096_key_cap() {
        for n in [32,4096,4128,8192,32768] {
            assert!(validate_dimensions([2,32,16],[2,n,16],[2,n,32],0.25).is_ok());
        }
        for n in [0,31,4097,i32::MAX as usize+1] {
            assert!(validate_dimensions([2,32,16],[2,n,16],[2,n,32],0.25).is_err());
        }
        assert!(validate_dimensions([4096,65536,16],[4096,8192,16],[4096,8192,16],0.25).is_err());
        assert!(validate_dimensions([2,32,16],[1,8192,16],[2,8192,32],0.25).is_err());
        assert!(validate_dimensions([2,32,16],[2,8192,16],[2,8192,32],f32::NAN).is_err());
    }
}

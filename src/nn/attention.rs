use super::{MatmulBf16Fp32Backend,SoftmaxBackend,CausalMaskBackend,RepeatKvBackend,Result,matmul_bf16_fp32,softmax,causal_mask,repeat_kv_heads};
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
/// Explicit grouped-query attention for Q[B,Hq,M,D], K[B,Hkv,N,D], V[B,Hkv,N,Dv].
/// Repeats consecutive KV heads on-device, then reuses the materialized attention path.
/// All tensors/masks are FP32; optional mask is exactly [B,Hq,M,N], without broadcasting.
/// Hq must be divisible by Hkv. This supports MHA/MQA/GQA, not fused FlashAttention.
pub fn grouped_query_attention_bf16_fp32<B:MatmulBf16Fp32Backend+SoftmaxBackend+RepeatKvBackend>(
    query:Tensor<B,4>,key:Tensor<B,4>,value:Tensor<B,4>,scale:f32,additive_mask:Option<Tensor<B,4>>)
    ->Result<Tensor<B,4>> {
    let flat=validate_gqa(query.dims(),key.dims(),value.dims(),scale)?;
    let [batch,heads,m,d]=query.dims();let n=key.dims()[2];let dv=value.dims()[3];let device=query.device();
    if query.dtype()!=DType::F32 || key.dtype()!=DType::F32 || value.dtype()!=DType::F32 || key.device()!=device || value.device()!=device {
        return Err(CannError::InvalidTensor("GQA requires FP32 Q/K/V on the same device".into()));
    }
    if let Some(mask)=&additive_mask {
        if mask.dims()!=[batch,heads,m,n] || mask.dtype()!=DType::F32 || mask.device()!=device {
            return Err(CannError::InvalidTensor("GQA additive mask must be FP32 [B,Hq,M,N] on the same device".into()));
        }
    }
    let key=repeat_kv_heads(key,heads)?.reshape([flat,n,d]);
    let value=repeat_kv_heads(value,heads)?.reshape([flat,n,dv]);
    let output=scaled_dot_product_attention_bf16_fp32(query.reshape([flat,m,d]),key,value,scale,
        additive_mask.map(|mask|mask.reshape([flat,m,n])))?;
    Ok(output.reshape([batch,heads,m,dv]))
}
/// GQA with a fixed exact-position causal mask; explicit starts support retained KV prefixes.
pub fn causal_grouped_query_attention_bf16_fp32<B:MatmulBf16Fp32Backend+SoftmaxBackend+RepeatKvBackend+CausalMaskBackend>(
    query:Tensor<B,4>,key:Tensor<B,4>,value:Tensor<B,4>,scale:f32,query_start:u64,key_start:u64)
    ->Result<Tensor<B,4>> {
    let flat=validate_gqa(query.dims(),key.dims(),value.dims(),scale)?;
    let [batch,heads,m,_]=query.dims();let n=key.dims()[2];
    if query.dtype()!=DType::F32 || key.dtype()!=DType::F32 || value.dtype()!=DType::F32
        || key.device()!=query.device() || value.device()!=query.device() {
        return Err(CannError::InvalidTensor("GQA requires FP32 Q/K/V on the same device".into()));
    }
    let mask=causal_mask::<B>(&query.device(),[flat,m,n],query_start,key_start)?.reshape([batch,heads,m,n]);
    grouped_query_attention_bf16_fp32(query,key,value,scale,Some(mask))
}
fn validate_gqa(query:[usize;4],key:[usize;4],value:[usize;4],scale:f32)->Result<usize> {
    let [batch,heads,m,d]=query;let [kb,kv_heads,n,kd]=key;let [vb,vh,vn,dv]=value;
    if heads==0 || kv_heads==0 || heads%kv_heads!=0 || kb!=batch || vb!=batch || vh!=kv_heads || kd!=d || vn!=n {
        return Err(CannError::InvalidTensor("GQA requires matching batches, positive divisible Q/KV heads and matching K/V lengths and Q/K widths".into()));
    }
    let flat=batch.checked_mul(heads).ok_or_else(||CannError::InvalidTensor("GQA flattened batch overflow".into()))?;
    validate_dimensions([flat,m,d],[flat,n,d],[flat,n,dv],scale)?;
    // Validate both expanded KV allocations before any device work is submitted.
    for width in [d,dv] {
        crate::runtime::RepeatKvSpec {batch:batch as u32,kv_heads:kv_heads as u32,query_heads:heads as u32,
            sequence:n as u32,width:width as u32}.elements().map_err(|e|CannError::InvalidTensor(e.to_string()))?;
    }
    Ok(flat)
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
    #[test]
    fn grouped_attention_preserves_head_ratio_batches_and_long_context() {
        for (heads,kv_heads) in [(1,1),(6,6),(6,3),(6,1),(15,3)] {
            assert_eq!(validate_gqa([2,heads,32,16],[2,kv_heads,8192,16],[2,kv_heads,8192,32],0.25).unwrap(),2*heads);
        }
        for (query,key,value) in [([2,5,32,16],[2,3,96,16],[2,3,96,16]),
            ([2,6,32,16],[2,3,96,16],[2,2,96,16]),([2,6,32,16],[2,3,96,16],[1,3,96,16]),
            ([2,6,32,16],[2,3,96,16],[2,3,32,16]),([2,0,32,16],[2,3,96,16],[2,3,96,16]),
            ([4096,2,32,16],[4096,1,96,16],[4096,1,96,16]),([0,6,32,16],[0,3,96,16],[0,3,96,16]),
            ([2,6,32,16],[2,3,31,16],[2,3,31,16])] {
            assert!(validate_gqa(query,key,value,0.25).is_err());
        }
    }
}

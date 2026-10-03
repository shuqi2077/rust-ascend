use super::{PaddedMatmulBf16Fp32Backend,SoftmaxBackend,CausalMaskBackend,RepeatKvBackend,Result,
    matmul_padded_bf16_fp32,softmax,causal_mask,repeat_kv_heads};
use crate::{driver::{CannError,tensor::{TensorLayout,DType as CannDType,deepgemm::PaddedGemmSpec}},
    runtime::{Transpose,RepeatKvSpec},tensor::{Backend,DType,api::Tensor}};
fn invalid(message:&str)->CannError {CannError::InvalidTensor(message.into())}

fn dimensions(query:[usize;3],key:[usize;3],value:[usize;3],scale:f32)->Result<()> {
    let [batch,m,d]=query;let [kb,n,kd]=key;let [vb,vn,_]=value;
    if kb!=batch || vb!=batch || kd!=d || vn!=n || !scale.is_finite()
        || batch.checked_mul(m).and_then(|count|count.checked_mul(n)).is_none_or(|count|count>u32::MAX as usize) {
        return Err(invalid("padded attention requires matching batches/contractions, finite scale and logical scores within u32"));
    }
    let layout=|shape:[usize;3]|->Result<TensorLayout> {
        let shape=shape.into_iter().map(|d|i64::try_from(d).map_err(|_|invalid("padded attention axis exceeds INT64_MAX")))
            .collect::<Result<Vec<_>>>()?;TensorLayout::contiguous(&shape,CannDType::F32)
    };
    let scores=PaddedGemmSpec::new(&layout(query)?,&layout(key)?,Transpose::No,Transpose::Yes)?;
    PaddedGemmSpec::new(&scores.output,&layout(value)?,Transpose::No,Transpose::No)?;Ok(())
}
fn same_device<B:Backend,const D:usize>(query:&Tensor<B,D>,key:&Tensor<B,D>,value:&Tensor<B,D>)->Result<()> {
    let device=query.device();
    if [query,key,value].iter().any(|t|t.dtype()!=DType::F32 || t.device()!=device) {
        return Err(invalid("padded attention requires FP32 Q/K/V on the same device"));
    }Ok(())
}
/// Materialized attention on FP32 Q[B,M,D], K[B,N,D], V[B,N,Dv].
/// All matrix axes are positive; each native BF16 GEMM pads/crops explicitly.
/// Softmax and mask operate only on logical keys, never on artificial padding.
/// Optional additive mask is exactly FP32[B,M,N]; all gradients use RUDA's graph.
/// No mask broadcasting, inferred scale, dropout, KV cache or FlashAttention.
pub fn scaled_dot_product_attention_padded_bf16_fp32<B:PaddedMatmulBf16Fp32Backend+SoftmaxBackend>(
    query:Tensor<B,3>,key:Tensor<B,3>,value:Tensor<B,3>,scale:f32,additive_mask:Option<Tensor<B,3>>)->Result<Tensor<B,3>> {
    dimensions(query.dims(),key.dims(),value.dims(),scale)?;same_device(&query,&key,&value)?;
    let [batch,m,_]=query.dims();let n=key.dims()[1];
    if let Some(mask)=&additive_mask {
        if mask.dims()!=[batch,m,n] || mask.dtype()!=DType::F32 || mask.device()!=query.device() {
            return Err(invalid("padded attention mask must be same-device FP32[B,M,N] without broadcasting"));
        }
    }
    let mut scores=matmul_padded_bf16_fp32(query,key,Transpose::No,Transpose::Yes)?*scale;
    if let Some(mask)=additive_mask {scores=scores+mask;}
    matmul_padded_bf16_fp32(softmax(scores)?,value,Transpose::No,Transpose::No)
}
/// Fixed exact-position causal mask; query_start and key_start are explicit.
pub fn causal_attention_padded_bf16_fp32<B:PaddedMatmulBf16Fp32Backend+SoftmaxBackend+CausalMaskBackend>(
    query:Tensor<B,3>,key:Tensor<B,3>,value:Tensor<B,3>,scale:f32,query_start:u64,key_start:u64)->Result<Tensor<B,3>> {
    dimensions(query.dims(),key.dims(),value.dims(),scale)?;same_device(&query,&key,&value)?;
    let [batch,m,_]=query.dims();let n=key.dims()[1];
    let mask=causal_mask::<B>(&query.device(),[batch,m,n],query_start,key_start)?;
    scaled_dot_product_attention_padded_bf16_fp32(query,key,value,scale,Some(mask))
}
fn grouped_dimensions(query:[usize;4],key:[usize;4],value:[usize;4],scale:f32)->Result<usize> {
    let [batch,heads,m,d]=query;let [kb,kv_heads,n,kd]=key;let [vb,vh,vn,dv]=value;
    if heads==0 || kv_heads==0 || heads%kv_heads!=0 || kb!=batch || vb!=batch || vh!=kv_heads || vn!=n || kd!=d {
        return Err(invalid("padded GQA requires matching batches/KV shapes and positive divisible Q/KV heads"));
    }
    let flat=batch.checked_mul(heads).ok_or_else(||invalid("padded GQA flattened batch overflow"))?;
    dimensions([flat,m,d],[flat,n,d],[flat,n,dv],scale)?;
    // The matrix contract above bounds flat to 4096 and each rounded axis to
    // INT32_MAX. Validate actual KV-repeat domains before submitting device work.
    for width in [d,dv] {
        RepeatKvSpec {batch:batch as u32,kv_heads:kv_heads as u32,query_heads:heads as u32,sequence:n as u32,width:width as u32}
            .elements().map_err(|e|CannError::InvalidTensor(e.to_string()))?;
    }Ok(flat)
}
/// MHA/MQA/GQA on Q[B,Hq,M,D], K[B,Hkv,N,D], V[B,Hkv,N,Dv].
/// Repeats consecutive KV heads on device and reduces their gradients natively.
/// Optional FP32 mask is exactly [B,Hq,M,N]. This materializes repeated KV and scores.
pub fn grouped_query_attention_padded_bf16_fp32<B:PaddedMatmulBf16Fp32Backend+SoftmaxBackend+RepeatKvBackend>(
    query:Tensor<B,4>,key:Tensor<B,4>,value:Tensor<B,4>,scale:f32,additive_mask:Option<Tensor<B,4>>)->Result<Tensor<B,4>> {
    let flat=grouped_dimensions(query.dims(),key.dims(),value.dims(),scale)?;same_device(&query,&key,&value)?;
    let [batch,heads,m,d]=query.dims();let n=key.dims()[2];let dv=value.dims()[3];
    if let Some(mask)=&additive_mask {
        if mask.dims()!=[batch,heads,m,n] || mask.dtype()!=DType::F32 || mask.device()!=query.device() {
            return Err(invalid("padded GQA mask must be same-device FP32[B,Hq,M,N] without broadcasting"));
        }
    }
    let key=repeat_kv_heads(key,heads)?.reshape([flat,n,d]);let value=repeat_kv_heads(value,heads)?.reshape([flat,n,dv]);
    let out=scaled_dot_product_attention_padded_bf16_fp32(query.reshape([flat,m,d]),key,value,scale,
        additive_mask.map(|mask|mask.reshape([flat,m,n])))?;Ok(out.reshape([batch,heads,m,dv]))
}
pub fn causal_grouped_query_attention_padded_bf16_fp32<B:PaddedMatmulBf16Fp32Backend+SoftmaxBackend+RepeatKvBackend+CausalMaskBackend>(
    query:Tensor<B,4>,key:Tensor<B,4>,value:Tensor<B,4>,scale:f32,query_start:u64,key_start:u64)->Result<Tensor<B,4>> {
    let flat=grouped_dimensions(query.dims(),key.dims(),value.dims(),scale)?;same_device(&query,&key,&value)?;
    let [batch,heads,m,_]=query.dims();let n=key.dims()[2];
    let mask=causal_mask::<B>(&query.device(),[flat,m,n],query_start,key_start)?.reshape([batch,heads,m,n]);
    grouped_query_attention_padded_bf16_fp32(query,key,value,scale,Some(mask))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn positive_attention_tails_share_native_matrix_and_logical_score_contracts() {
        for (m,n,d,dv) in [(1,1,1,1),(3,7,5,9),(17,33,7,3),(3,4097,7,9)] {
            assert!(dimensions([2,m,d],[2,n,d],[2,n,dv],0.25).is_ok());
        }
        for (query,key,value,scale) in [([0,3,5],[0,7,5],[0,7,9],0.25),([2,0,5],[2,7,5],[2,7,9],0.25),
            ([2,3,5],[2,0,5],[2,0,9],0.25),([2,3,5],[2,7,6],[2,7,9],0.25),([2,3,5],[1,7,5],[2,7,9],0.25),
            ([2,3,5],[2,7,5],[2,6,9],0.25),([2,3,5],[2,7,5],[2,7,9],f32::NAN),
            ([4096,65536,7],[4096,65536,7],[4096,65536,9],0.25)] {assert!(dimensions(query,key,value,scale).is_err());}
    }
    #[test]
    fn padded_gqa_preserves_head_grouping_and_exact_expanded_domains() {
        for (heads,kv_heads) in [(1,1),(6,6),(6,2),(5,1),(15,3)] {
            assert_eq!(grouped_dimensions([2,heads,3,7],[2,kv_heads,4097,7],[2,kv_heads,4097,9],0.25).unwrap(),2*heads);
        }
        for (query,key,value) in [([2,5,3,7],[2,3,7,7],[2,3,7,9]),([2,6,3,7],[2,2,7,7],[2,1,7,9]),
            ([2,6,3,7],[2,2,7,7],[1,2,7,9]),([2,6,3,7],[2,2,7,7],[2,2,8,9]),
            ([4096,2,3,7],[4096,1,7,7],[4096,1,7,9]),([2,0,3,7],[2,1,7,7],[2,1,7,9]),
            ([2,3,1,7],[2,1,1,7],[2,1,1,1_000_000_000])] {assert!(grouped_dimensions(query,key,value,0.25).is_err());}
    }
}

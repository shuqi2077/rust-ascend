use super::{Result,PaddedMatmulBf16Fp32Backend,FrozenPaddedLinearBf16Fp32Backend,SiluMulBackend,
    linear_padded_bf16_fp32,linear_frozen_padded_bf16_fp32,lora_padded_linear_bf16_fp32,
    lora_frozen_padded_linear_bf16_fp32,swiglu_padded_bf16_fp32,swiglu_frozen_padded_bf16_fp32};
use crate::{Ascend,driver::CannError,tensor::{Backend,DType,api::Tensor}};

fn invalid(message:&str)->CannError {CannError::InvalidTensor(message.into())}
fn projection_shape<const D:usize>(shape:[usize;D],width:usize)->Result<([usize;2],[usize;D])> {
    if !(1..=8).contains(&D) || width==0 || shape.iter().any(|&size|size==0) {
        return Err(invalid("last-axis projection requires rank 1..8 and positive input/output axes"));
    }
    let rows=shape[..D-1].iter().try_fold(1usize,|rows,&size|rows.checked_mul(size))
        .ok_or_else(||invalid("last-axis projection token count overflow"))?;
    if rows.checked_mul(shape[D-1]).is_none() || rows.checked_mul(width).is_none() {
        return Err(invalid("last-axis projection element count overflow"));
    }
    let mut output=shape;output[D-1]=width;Ok(([rows,shape[D-1]],output))
}
fn flatten<B:Backend,const D:usize>(input:Tensor<B,D>,width:usize)->Result<(Tensor<B,2>,[usize;D])> {
    let (matrix,output)=projection_shape(input.dims(),width)?;
    if input.dtype()!=DType::F32 {return Err(invalid("last-axis projection requires FP32 activation storage"));}
    Ok((input.reshape(matrix),output))
}
/// Projects FP32 X[...,K] with trainable FP32 W[N,K], preserving all leading axes.
/// Rank 1..8 and positive axes; flattened tokens use the existing padded GEMM domain.
/// Reshapes stay in RUDA's graph, including shared parameter-gradient accumulation.
pub fn linear_padded_bf16_fp32_nd<B:PaddedMatmulBf16Fp32Backend,const D:usize>(
    input:Tensor<B,D>,weight:Tensor<B,2>)->Result<Tensor<B,D>> {
    let (input,output)=flatten(input,weight.dims()[0])?;
    Ok(linear_padded_bf16_fp32(input,weight)?.reshape(output))
}
/// Fixed BF16 W[N,K], FP32 X[...,K]; only the input gets a base-weight derivative.
/// Original fixed weights must remain unchanged through backward; no FP32 expansion.
pub fn linear_frozen_padded_bf16_fp32_nd<B:FrozenPaddedLinearBf16Fp32Backend,const D:usize>(
    input:Tensor<B,D>,weight:Tensor<Ascend,2>)->Result<Tensor<B,D>> {
    let (input,output)=flatten(input,weight.dims()[0])?;
    Ok(linear_frozen_padded_bf16_fp32(input,weight)?.reshape(output))
}
/// Last-axis LoRA: X W^T + scale * (X A^T) B^T; all parameters are caller-selected.
/// W[N,K], A[R,K], B[N,R], any positive R and explicit finite scale; no dropout/bias.
pub fn lora_padded_linear_bf16_fp32_nd<B:PaddedMatmulBf16Fp32Backend,const D:usize>(
    input:Tensor<B,D>,weight:Tensor<B,2>,down:Tensor<B,2>,up:Tensor<B,2>,scale:f32)->Result<Tensor<B,D>> {
    let (input,output)=flatten(input,weight.dims()[0])?;
    Ok(lora_padded_linear_bf16_fp32(input,weight,down,up,scale)?.reshape(output))
}
/// Last-axis LoRA with fixed BF16 W and trainable FP32 adapters; same explicit scale.
pub fn lora_frozen_padded_linear_bf16_fp32_nd<B:FrozenPaddedLinearBf16Fp32Backend+PaddedMatmulBf16Fp32Backend,const D:usize>(
    input:Tensor<B,D>,weight:Tensor<Ascend,2>,down:Tensor<B,2>,up:Tensor<B,2>,scale:f32)->Result<Tensor<B,D>> {
    let (input,output)=flatten(input,weight.dims()[0])?;
    Ok(lora_frozen_padded_linear_bf16_fp32(input,weight,down,up,scale)?.reshape(output))
}
/// Last-axis (SiLU(X Wgate^T) * (X Wup^T)) Wdown^T, with trainable FP32 weights.
pub fn swiglu_padded_bf16_fp32_nd<B:PaddedMatmulBf16Fp32Backend+SiluMulBackend,const D:usize>(
    input:Tensor<B,D>,gate:Tensor<B,2>,up:Tensor<B,2>,down:Tensor<B,2>)->Result<Tensor<B,D>> {
    let (input,output)=flatten(input,down.dims()[0])?;
    Ok(swiglu_padded_bf16_fp32(input,gate,up,down)?.reshape(output))
}
/// Last-axis SwiGLU with fixed BF16 gate/up/down and FP32 activation/input derivative.
pub fn swiglu_frozen_padded_bf16_fp32_nd<B:FrozenPaddedLinearBf16Fp32Backend+SiluMulBackend,const D:usize>(
    input:Tensor<B,D>,gate:Tensor<Ascend,2>,up:Tensor<Ascend,2>,down:Tensor<Ascend,2>)->Result<Tensor<B,D>> {
    let (input,output)=flatten(input,down.dims()[0])?;
    Ok(swiglu_frozen_padded_bf16_fp32(input,gate,up,down)?.reshape(output))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn last_axis_preserves_vector_tokens_batches_and_higher_rank_prefixes() {
        assert_eq!(projection_shape([7],3).unwrap(),([1,7],[3]));
        assert_eq!(projection_shape([3,7],5).unwrap(),([3,7],[3,5]));
        assert_eq!(projection_shape([2,3,7],9).unwrap(),([6,7],[2,3,9]));
        assert_eq!(projection_shape([2,3,5,7],11).unwrap(),([30,7],[2,3,5,11]));
        assert_eq!(projection_shape([1,2,1,3,1,5,1,7],9).unwrap(),([30,7],[1,2,1,3,1,5,1,9]));
    }
    #[test]
    fn projection_rejects_rank_empty_and_count_overflows_before_reshape() {
        assert!(projection_shape([],3).is_err());assert!(projection_shape([1;9],3).is_err());
        assert!(projection_shape([2,0,7],3).is_err());assert!(projection_shape([2,3,7],0).is_err());
        assert!(projection_shape([usize::MAX,2,1],1).is_err());
        assert!(projection_shape([usize::MAX,2],1).is_err());
        assert!(projection_shape([usize::MAX,1],2).is_err());
    }
}

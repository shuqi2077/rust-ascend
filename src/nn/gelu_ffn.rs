use super::{Result,PaddedMatmulBf16Fp32Backend,FrozenPaddedLinearBf16Fp32Backend,
    linear_padded_bf16_fp32_nd,linear_frozen_padded_bf16_fp32_nd};
use crate::{Ascend,driver::CannError,tensor::{Backend,DType,api::{Tensor,activation}}};

/// Selects the existing RUDA GELU forward and autodiff definitions explicitly.
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum GeluMode {Exact,Tanh}
fn activate<B:Backend,const D:usize>(input:Tensor<B,D>,mode:GeluMode)->Tensor<B,D> {
    match mode {GeluMode::Exact=>activation::gelu(input),GeluMode::Tanh=>activation::gelu_approximate(input)}
}
fn shapes<const D:usize>(input:[usize;D],up:[usize;2],down:[usize;2],gate:Option<[usize;2]>)->Result<()> {
    if !(1..=8).contains(&D) || input.iter().any(|&size|size==0) {
        return Err(CannError::InvalidTensor("GELU FFN requires rank 1..8 and positive token/input axes".into()));
    }
    if up[0]==0 || up[1]!=input[D-1] || down[0]==0 || down[1]!=up[0] || gate.is_some_and(|shape|shape!=up) {
        return Err(CannError::InvalidTensor("GELU FFN requires X[...,K], up/gate[H,K], down[N,H] with positive matching axes".into()));
    }
    Ok(())
}
/// GELU(X Wup^T) Wdown^T with trainable FP32 weights and explicit BF16 matmul.
/// Rank 1..8, positive token axes and arbitrary positive K/H/N logical tails.
/// Leading axes, FP32 activation storage and RUDA gradients are retained.
pub fn gelu_mlp_padded_bf16_fp32_nd<B:PaddedMatmulBf16Fp32Backend,const D:usize>(
    input:Tensor<B,D>,up:Tensor<B,2>,down:Tensor<B,2>,mode:GeluMode)->Result<Tensor<B,D>> {
    shapes(input.dims(),up.dims(),down.dims(),None)?;let device=input.device();
    if input.dtype()!=DType::F32 || [&up,&down].iter().any(|t|t.dtype()!=DType::F32 || t.device()!=device) {
        return Err(CannError::InvalidTensor("GELU MLP requires same-device FP32 input and weights".into()));
    }
    linear_padded_bf16_fp32_nd(activate(linear_padded_bf16_fp32_nd(input,up)?,mode),down)
}
/// GELU(X Wgate^T) * (X Wup^T), followed by Wdown^T; all weights caller-tracked.
/// The two projected branches share RUDA's input-gradient accumulation.
pub fn geglu_padded_bf16_fp32_nd<B:PaddedMatmulBf16Fp32Backend,const D:usize>(
    input:Tensor<B,D>,gate:Tensor<B,2>,up:Tensor<B,2>,down:Tensor<B,2>,mode:GeluMode)->Result<Tensor<B,D>> {
    shapes(input.dims(),up.dims(),down.dims(),Some(gate.dims()))?;let device=input.device();
    if input.dtype()!=DType::F32 || [&gate,&up,&down].iter().any(|t|t.dtype()!=DType::F32 || t.device()!=device) {
        return Err(CannError::InvalidTensor("GEGLU requires same-device FP32 input and weights".into()));
    }
    let gate=activate(linear_padded_bf16_fp32_nd(input.clone(),gate)?,mode);
    let up=linear_padded_bf16_fp32_nd(input,up)?;
    linear_padded_bf16_fp32_nd(gate*up,down)
}
/// Fixed BF16 Wup/Wdown, FP32 activation storage and input-only differentiation.
/// Keep weights unchanged through backward; no FP32 weight expansion.
pub fn gelu_mlp_frozen_padded_bf16_fp32_nd<B:FrozenPaddedLinearBf16Fp32Backend,const D:usize>(
    input:Tensor<B,D>,up:Tensor<Ascend,2>,down:Tensor<Ascend,2>,mode:GeluMode)->Result<Tensor<B,D>> {
    shapes(input.dims(),up.dims(),down.dims(),None)?;let device=input.device();
    if input.dtype()!=DType::F32 || [&up,&down].iter().any(|t|t.dtype()!=DType::BF16 || t.device()!=device) {
        return Err(CannError::InvalidTensor("frozen GELU MLP requires same-device FP32 input and fixed BF16 weights".into()));
    }
    linear_frozen_padded_bf16_fp32_nd(activate(linear_frozen_padded_bf16_fp32_nd(input,up)?,mode),down)
}
/// GEGLU with fixed BF16 gate/up/down and shared FP32 input gradients.
/// No implicit bias, dropout, residual, normalization, LoRA or weight freezing.
pub fn geglu_frozen_padded_bf16_fp32_nd<B:FrozenPaddedLinearBf16Fp32Backend,const D:usize>(
    input:Tensor<B,D>,gate:Tensor<Ascend,2>,up:Tensor<Ascend,2>,down:Tensor<Ascend,2>,mode:GeluMode)->Result<Tensor<B,D>> {
    shapes(input.dims(),up.dims(),down.dims(),Some(gate.dims()))?;let device=input.device();
    if input.dtype()!=DType::F32 || [&gate,&up,&down].iter().any(|t|t.dtype()!=DType::BF16 || t.device()!=device) {
        return Err(CannError::InvalidTensor("frozen GEGLU requires same-device FP32 input and fixed BF16 weights".into()));
    }
    let gate=activate(linear_frozen_padded_bf16_fp32_nd(input.clone(),gate)?,mode);
    let up=linear_frozen_padded_bf16_fp32_nd(input,up)?;
    linear_frozen_padded_bf16_fp32_nd(gate*up,down)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]fn gelu_ffn_preserves_positive_token_shapes_and_checks_all_weights() {
        assert!(shapes([7],[5,7],[9,5],None).is_ok());
        assert!(shapes([2,3,7],[5,7],[9,5],Some([5,7])).is_ok());
        assert!(shapes([1,2,1,3,1,1,1,7],[5,7],[9,5],Some([5,7])).is_ok());
        assert!(shapes([],[5,7],[9,5],None).is_err());assert!(shapes([1;9],[5,1],[9,5],None).is_err());
        assert!(shapes([2,0,7],[5,7],[9,5],None).is_err());
        for (up,down,gate) in [([5,6],[9,5],None),([0,7],[9,0],None),([5,7],[0,5],None),
            ([5,7],[9,4],None),([5,7],[9,5],Some([4,7]))] {
            assert!(shapes([2,3,7],up,down,gate).is_err());
        }
    }
}

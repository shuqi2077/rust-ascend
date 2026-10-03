use super::{
    LossReduction, NllLossBackend, NllLossOptions, PiecewiseBackend, Result, SoftmaxBackend,
    TokenWindow, TokenWindowBackend, clamp, log_softmax, nll_loss_with_total_weight, token_window,
    token_window_int,
};
use crate::{
    Ascend,
    driver::CannError,
    tensor::{
        Backend, DType,
        api::{Int, Tensor},
    },
};

/// Model-independent decoder interface; projection may be dense, frozen or LoRA.
pub trait CausalLanguageModel<B: Backend> {
    fn forward_hidden(&self, tokens: Tensor<B, 2, Int>) -> Result<Tensor<B, 3>>;
    fn project(&self, hidden: Tensor<B, 2>) -> Result<Tensor<B, 2>>;
}

/// Borrow an original RUDA causal model without copying its parameters or records.
/// Its forward/projection use the selected backend; this does not turn packed or
/// paged kernels unsupported by that backend into supported kernels.
pub struct RudaCausalModel<'a, M>(pub &'a M);
impl<B: Backend, M: ruda_nn::loss::CausalLanguageModel<B>> CausalLanguageModel<B>
    for RudaCausalModel<'_, M>
{
    fn forward_hidden(&self, tokens: Tensor<B, 2, Int>) -> Result<Tensor<B, 3>> {
        Ok(ruda_nn::loss::CausalLanguageModel::forward_hidden(
            self.0, tokens,
        ))
    }
    fn project(&self, hidden: Tensor<B, 2>) -> Result<Tensor<B, 2>> {
        Ok(ruda_nn::loss::CausalLanguageModel::project(self.0, hidden))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CausalCrossEntropyConfig {
    pub token_chunk_size: usize,
    pub ignore_index: i64,
    pub shift: bool,
}
impl Default for CausalCrossEntropyConfig {
    fn default() -> Self {
        Self {
            token_chunk_size: 32,
            ignore_index: -100,
            shift: true,
        }
    }
}

/// Full-vocabulary FP32 loss sum and non-trainable FP32 unit-weight token count.
/// Count is exact within the accepted <=2^24-token domain. No host readback occurs.
pub struct CausalLoss<B: Backend> {
    pub loss_sum: Tensor<B, 1>,
    pub valid_tokens: Tensor<B, 1>,
}
impl<B: PiecewiseBackend> CausalLoss<B> {
    /// Ignored/empty windows have zero mean; use loss_sum for token-weighted accumulation.
    pub fn mean(self) -> Result<Tensor<B, 1>> {
        Ok(self.loss_sum / clamp(self.valid_tokens, 1., f32::MAX)?)
    }
}

fn domain(
    shape: [usize; 3],
    labels: [usize; 2],
    config: CausalCrossEntropyConfig,
) -> Result<(usize, usize)> {
    let [batch, time, width] = shape;
    if config.token_chunk_size == 0 || labels != [batch, time] || width == 0 {
        return Err(CannError::InvalidTensor(
            "causal loss requires matching [B,T,H]/[B,T], positive H and token chunk size".into(),
        ));
    }
    let steps = if config.shift {
        time.saturating_sub(1)
    } else {
        time
    };
    let count = batch
        .checked_mul(steps)
        .ok_or_else(|| CannError::InvalidTensor("causal token count overflow".into()))?;
    if count > 1 << 24 {
        return Err(CannError::InvalidTensor(
            "causal loss FP32 token count requires at most 2^24 logical tokens per call".into(),
        ));
    }
    Ok((steps, count))
}

impl CausalCrossEntropyConfig {
    pub fn forward_model<B, M>(
        &self,
        model: &M,
        tokens: Tensor<B, 2, Int>,
        labels: Tensor<B, 2, Int>,
    ) -> Result<CausalLoss<B>>
    where
        B: TokenWindowBackend + NllLossBackend + SoftmaxBackend,
        M: CausalLanguageModel<B>,
    {
        if tokens.dims() != labels.dims() || tokens.device() != labels.device() {
            return Err(CannError::InvalidTensor(
                "causal token/label batch, sequence and device must match".into(),
            ));
        }
        self.forward_hidden(model.forward_hidden(tokens)?, labels, |hidden| {
            model.project(hidden)
        })
    }

    /// Chunk tokens only, never vocabulary. Shift never pairs the last token of one batch
    /// with the first of another. Projection and all gradients use the existing device graph.
    /// This does not recompute backward graphs or bound total retained autodiff memory.
    pub fn forward_hidden<B, F>(
        &self,
        hidden: Tensor<B, 3>,
        labels: Tensor<B, 2, Int>,
        project: F,
    ) -> Result<CausalLoss<B>>
    where
        B: TokenWindowBackend + NllLossBackend + SoftmaxBackend,
        F: Fn(Tensor<B, 2>) -> Result<Tensor<B, 2>>,
    {
        let shape = hidden.dims();
        let (steps, tokens) = domain(shape, labels.dims(), *self)?;
        let device = hidden.device();
        if labels.device() != device
            || hidden.dtype() != DType::F32
            || !matches!(labels.dtype(), DType::I32 | DType::I64)
        {
            return Err(CannError::InvalidTensor(
                "causal loss requires same-device FP32 hidden and I32/I64 labels".into(),
            ));
        }
        let mut loss_sum = Tensor::<B, 1>::zeros([1], (&device, DType::F32));
        let mut valid_tokens = Tensor::<B, 1>::zeros([1], (&device, DType::F32));
        if tokens == 0 {
            return Ok(CausalLoss {
                loss_sum,
                valid_tokens,
            });
        }
        let labels = labels.reshape([shape[0], shape[1], 1]);
        let mut classes = None;
        let mut weight = None;
        for start in (0..tokens).step_by(self.token_chunk_size) {
            let count = self.token_chunk_size.min(tokens - start);
            let window = TokenWindow {
                time_start: 0,
                time_len: steps,
                start,
                count,
            };
            let chunk = token_window(hidden.clone(), window)?;
            let labels = token_window_int(
                labels.clone(),
                TokenWindow {
                    time_start: usize::from(self.shift),
                    ..window
                },
            )?
            .reshape([count]);
            let logits = project(chunk)?;
            let [rows, vocab] = logits.dims();
            if rows != count
                || vocab == 0
                || logits.dtype() != DType::F32
                || logits.device() != device
                || classes.is_some_and(|expected| expected != vocab)
            {
                return Err(CannError::InvalidTensor("causal projection requires consistent same-device FP32 [chunk,V] with positive V".into()));
            }
            classes = Some(vocab);
            let weight = weight
                .get_or_insert_with(|| Tensor::<Ascend, 1>::ones([vocab], (&device, DType::F32)));
            let (loss, total) = nll_loss_with_total_weight(
                log_softmax(logits)?,
                labels,
                weight.clone(),
                NllLossOptions {
                    reduction: LossReduction::Sum,
                    ignore_index: Some(self.ignore_index),
                },
            )?;
            loss_sum = loss_sum + loss;
            valid_tokens = valid_tokens + total;
        }
        Ok(CausalLoss {
            loss_sum,
            valid_tokens,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn causal_domain_handles_shift_empty_matching_axes_and_exact_fp32_counts() {
        let config = CausalCrossEntropyConfig::default();
        assert_eq!(domain([2, 5, 7], [2, 5], config).unwrap(), (4, 8));
        assert_eq!(domain([2, 1, 7], [2, 1], config).unwrap(), (0, 0));
        assert_eq!(domain([0, 5, 7], [0, 5], config).unwrap(), (4, 0));
        assert_eq!(domain([2, 0, 7], [2, 0], config).unwrap(), (0, 0));
        assert_eq!(
            domain(
                [2, 5, 7],
                [2, 5],
                CausalCrossEntropyConfig {
                    shift: false,
                    ..config
                }
            )
            .unwrap(),
            (5, 10)
        );
        assert_eq!(
            domain([1, 1 + (1 << 24), 7], [1, 1 + (1 << 24)], config)
                .unwrap()
                .1,
            1 << 24
        );
        assert!(domain([2, 5, 7], [2, 4], config).is_err());
        assert!(domain([2, 5, 0], [2, 5], config).is_err());
        assert!(
            domain(
                [2, 5, 7],
                [2, 5],
                CausalCrossEntropyConfig {
                    token_chunk_size: 0,
                    ..config
                }
            )
            .is_err()
        );
        assert!(domain([1, 2 + (1 << 24), 7], [1, 2 + (1 << 24)], config).is_err());
        assert!(domain([usize::MAX, 5, 7], [usize::MAX, 5], config).is_err());
    }
}

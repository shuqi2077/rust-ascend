use rust_ascend::{
    Autodiff, RudaAscend,
    model::module::{Module, ModuleMapper, Param},
    nn::modules::{
        EmbeddingConfig, LinearConfig, LoRALinearConfig, RmsNormConfig,
        attention::{MhaInput, MultiHeadAttentionConfig},
        loss::CausalCrossEntropyConfig,
    },
    runtime::{AscendRuntime, RuntimeOptions},
    tensor::{
        Backend, DType, FloatDType,
        api::{Bool, Int, Tensor},
    },
};

type B = Autodiff<RudaAscend>;
struct StorageDtype(DType);
impl ModuleMapper<B> for StorageDtype {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        param.map(|tensor| tensor.cast(self.0))
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let toolkit = std::env::var_os("ASCEND_HOME_PATH").ok_or("set ASCEND_HOME_PATH")?;
    let mut options = RuntimeOptions::new(toolkit);
    if let Some(value) = std::env::var_os("RUDA_CANN_LIBRARY") {
        options.acl_library = value;
    }
    if let Some(value) = std::env::var_os("RUDA_CANN_OPERATOR_LIBRARIES") {
        options.operator_libraries = std::env::split_paths(&value)
            .map(|p| p.into_os_string())
            .collect();
    }
    let device = unsafe { AscendRuntime::initialize_exclusive(options)? };
    let tokens = Tensor::<B, 2, Int>::from_data([[0, 1, 0]], &device);
    let labels = Tensor::<B, 2, Int>::from_data([[0, 1, 2]], &device);
    let mask = Tensor::<B, 3, Bool>::from_data(
        [[
            [false, true, true],
            [false, false, true],
            [false, false, false],
        ]],
        &device,
    );
    for dtype in [DType::F16, DType::BF16] {
        B::seed(&device, 2077);
        let mut mapper = StorageDtype(dtype);
        let embedding = EmbeddingConfig::new(3, 2)
            .init::<B>(&device)
            .map(&mut mapper);
        let attention = MultiHeadAttentionConfig::new(2, 1)
            .with_dropout(0.)
            .init::<B>(&device)
            .map(&mut mapper);
        let norm = RmsNormConfig::new(2).init::<B>(&device);
        let head = LoRALinearConfig::new(1, 2.)
            .init(LinearConfig::new(2, 3).init::<B>(&device).map(&mut mapper));
        let hidden = embedding.forward(tokens.clone());
        assert_eq!(hidden.dtype(), dtype);
        let hidden = attention
            .forward(MhaInput::self_attn(hidden).mask_attn(mask.clone()))
            .context;
        assert_eq!(hidden.dtype(), dtype);
        let hidden = norm.forward_with_compute_dtype(hidden, FloatDType::F32);
        let logits = head.forward(hidden);
        assert_eq!(logits.dtype(), dtype);
        let result = CausalCrossEntropyConfig::new()
            .with_token_chunk_size(1)
            .forward_logits(logits, labels.clone());
        assert_eq!(result.valid_tokens.clone().into_scalar(), 2);
        let loss = result.mean();
        assert!(loss.clone().into_scalar().is_finite());
        let gradients = loss.backward();
        for gradient in [
            embedding
                .weight
                .val()
                .grad(&gradients)
                .ok_or("missing original embedding gradient")?,
            attention
                .query
                .weight
                .val()
                .grad(&gradients)
                .ok_or("missing original attention gradient")?,
            head.adapter_b
                .weight
                .val()
                .grad(&gradients)
                .ok_or("missing original adapter gradient")?,
        ] {
            assert_eq!(gradient.dtype(), dtype);
            assert!(
                gradient
                    .cast(DType::F32)
                    .into_data()
                    .to_vec::<f32>()?
                    .iter()
                    .all(|value| value.is_finite())
            );
        }
        assert!(norm.gamma.val().grad(&gradients).is_some());
        assert!(head.base.weight.val().grad(&gradients).is_none());
        println!(
            "{dtype:?}: original RUDA embedding, causal MHA, FP32 RMSNorm, LoRA and causal loss backward passed"
        );
    }
    Ok(())
}

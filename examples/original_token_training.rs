use rust_ascend::{
    Autodiff, RudaAscend,
    model::module::Module,
    nn::modules::{
        EmbeddingConfig, LinearConfig, LoRALinearConfig, RmsNormConfig, RotaryEncodingConfig,
        attention::{MhaInput, MultiHeadAttentionConfig},
        loss::CausalCrossEntropyConfig,
    },
    optim::{AdamWStorageStep, adamw_master_tensor_step},
    runtime::{AscendRuntime, RuntimeOptions},
    tensor::{
        Backend, DType, FloatDType,
        api::{Bool, Int, Tensor},
    },
};

type B = Autodiff<RudaAscend>;

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
    for (dtype, module_dtype) in [
        (DType::F16, FloatDType::F16),
        (DType::BF16, FloatDType::BF16),
    ] {
        B::seed(&device, 2077);
        let embedding = EmbeddingConfig::new(3, 2)
            .init::<B>(&device)
            .to_dtype(module_dtype);
        let attention = MultiHeadAttentionConfig::new(2, 1)
            .with_dropout(0.)
            .init::<B>(&device)
            .to_dtype(module_dtype);
        let norm = RmsNormConfig::new(2).init::<B>(&device);
        let rotary = RotaryEncodingConfig::new(3, 2).init::<B>(&device);
        let mut head = LoRALinearConfig::new(1, 2.).init(
            LinearConfig::new(2, 3)
                .init::<B>(&device)
                .to_dtype(module_dtype),
        );
        let hidden = embedding.forward(tokens.clone());
        assert_eq!(hidden.dtype(), dtype);
        let hidden = rotary.forward_with_compute_dtype(hidden, FloatDType::F32);
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
        let gradient = head
            .adapter_b
            .weight
            .val()
            .grad(&gradients)
            .ok_or("missing adapter update gradient")?;
        let mut parameter = head.adapter_b.weight.val().inner();
        let mut master = parameter.clone().cast(DType::F32);
        let mut first = Tensor::<RudaAscend, 2>::zeros(parameter.dims(), &device);
        let mut second = Tensor::<RudaAscend, 2>::zeros(parameter.dims(), &device);
        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        adamw_master_tensor_step(
            &mut parameter,
            &mut master,
            &gradient,
            &mut first,
            &mut second,
            AdamWStorageStep {
                learning_rate: 0.001,
                beta1,
                beta2,
                epsilon: 1e-8,
                weight_decay: 0.,
                correction1: 1. - beta1,
                correction2: 1. - beta2,
                inverse_gradient_scale: 1.,
                clip_multiplier: 1.,
            },
        )?;
        head.adapter_b.weight = head
            .adapter_b
            .weight
            .map(|_| Tensor::<B, 2>::from_inner(parameter).require_grad());
        let hidden = embedding.forward(tokens.clone());
        let hidden = rotary.forward_with_compute_dtype(hidden, FloatDType::F32);
        let hidden = attention
            .forward(MhaInput::self_attn(hidden).mask_attn(mask.clone()))
            .context;
        let logits = head.forward(norm.forward_with_compute_dtype(hidden, FloatDType::F32));
        let loss = CausalCrossEntropyConfig::new()
            .forward_logits(logits, labels.clone())
            .mean();
        assert!(loss.into_scalar().is_finite());
        println!(
            "{dtype:?}: original RUDA token graph with RoPE backward, explicit FP32-master AdamW adapter update and next forward passed"
        );
    }
    Ok(())
}

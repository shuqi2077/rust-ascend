use rust_ascend::{
    Autodiff, RudaAscend,
    model::module::{Initializer, Module},
    nn::modules::{LayerNormConfig, LinearConfig, LoRALinearConfig, RmsNormConfig},
    runtime::{AscendRuntime, RuntimeOptions},
    tensor::{
        DType, FloatDType,
        api::{Tensor, activation},
    },
};

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
    type B = Autodiff<RudaAscend>;
    for (dtype, module_dtype) in [
        (DType::F16, FloatDType::F16),
        (DType::BF16, FloatDType::BF16),
    ] {
        let base = LinearConfig::new(2, 3)
            .with_bias(false)
            .with_initializer(Initializer::Ones)
            .init::<B>(&device)
            .to_dtype(module_dtype);
        let model = LoRALinearConfig::new(1, 2.).init(base);
        let input = Tensor::<B, 2>::from_data([[1., 3.], [2., 4.]], &device)
            .cast(dtype)
            .require_grad();
        let output = model.forward(input.clone());
        assert_eq!(output.dtype(), dtype);
        assert_eq!(
            output
                .clone()
                .cast(DType::F32)
                .into_data()
                .to_vec::<f32>()?,
            vec![4., 4., 4., 6., 6., 6.]
        );
        let gradients = output.sum().backward();
        let gradient = input
            .grad(&gradients)
            .ok_or("missing half input gradient")?;
        assert_eq!(gradient.dtype(), dtype);
        assert_eq!(
            gradient.cast(DType::F32).into_data().to_vec::<f32>()?,
            vec![3.; 4]
        );
        assert!(model.base.weight.val().grad(&gradients).is_none());
        let gradient = model
            .adapter_b
            .weight
            .val()
            .grad(&gradients)
            .ok_or("missing original half LoRA gradient")?;
        assert_eq!(gradient.dtype(), dtype);
        assert!(
            gradient
                .cast(DType::F32)
                .into_data()
                .to_vec::<f32>()?
                .iter()
                .all(|value| value.is_finite())
        );

        let layer = LayerNormConfig::new(2).init::<B>(&device);
        let rms = RmsNormConfig::new(2).init::<B>(&device);
        for normalized in [
            layer.forward_with_compute_dtype(input.clone(), FloatDType::F32),
            rms.forward_with_compute_dtype(input.clone(), FloatDType::F32),
        ] {
            assert_eq!(normalized.dtype(), dtype);
            let gradients = normalized.square().sum().backward();
            let gradient = input
                .grad(&gradients)
                .ok_or("missing half normalization gradient")?;
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
        let probabilities = activation::softmax(input.clone(), 1);
        assert_eq!(probabilities.dtype(), dtype);
        for sum in probabilities
            .sum_dim(1)
            .cast(DType::F32)
            .into_data()
            .to_vec::<f32>()?
        {
            assert!((sum - 1.).abs() < 0.01);
        }
        let activated = activation::relu(input.clone() - 2.);
        assert_eq!(
            activated
                .clone()
                .cast(DType::F32)
                .into_data()
                .to_vec::<f32>()?,
            vec![0., 1., 0., 2.]
        );
        let gradients = activated.sum().backward();
        assert_eq!(
            input
                .grad(&gradients)
                .ok_or("missing half ReLU gradient")?
                .cast(DType::F32)
                .into_data()
                .to_vec::<f32>()?,
            vec![0., 1., 0., 1.]
        );
    }
    println!(
        "original RUDA half Linear/LoRA, FP32 normalization and native activation paths passed"
    );
    Ok(())
}

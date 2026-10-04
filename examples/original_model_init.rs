use rust_ascend::{
    Autodiff, RudaAscend,
    nn::modules::{DropoutConfig, LinearConfig, LoRALinearConfig},
    runtime::{AscendRuntime, RuntimeOptions},
    tensor::{Backend, Distribution, api::Tensor},
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
    B::seed(&device, 2077);
    let first = LinearConfig::new(4, 3).init::<B>(&device);
    let weights = first.weight.val().into_data().to_vec::<f32>()?;
    B::seed(&device, 2077);
    let second = LinearConfig::new(4, 3).init::<B>(&device);
    assert_eq!(weights, second.weight.val().into_data().to_vec::<f32>()?);
    assert!(weights.iter().all(|value| value.is_finite()));
    let input = Tensor::<B, 2>::ones([2, 4], &device).require_grad();
    let dropout = DropoutConfig::new(0.5).init();
    B::seed(&device, 7);
    let dropped = dropout.forward(input.clone());
    let values = dropped.clone().into_data().to_vec::<f32>()?;
    assert!(values.iter().all(|&value| value == 0. || value == 2.));
    let gradients = dropped.sum().backward();
    assert_eq!(
        values,
        input
            .grad(&gradients)
            .ok_or("missing original dropout gradient")?
            .into_data()
            .to_vec::<f32>()?
    );
    let head = LoRALinearConfig::new(2, 4.).with_dropout(0.25).init(second);
    let loss = head.forward(input).square().sum();
    let gradients = loss.backward();
    assert!(head.base.weight.val().grad(&gradients).is_none());
    assert!(head.adapter_b.weight.val().grad(&gradients).is_some());
    let normal = Tensor::<B, 1>::random([8], Distribution::Normal(2., 0.), &device);
    assert_eq!(normal.into_data().to_vec::<f32>()?, vec![2.; 8]);
    println!(
        "original RUDA model initialization, seeded random generation, dropout and LoRA backward passed"
    );
    Ok(())
}

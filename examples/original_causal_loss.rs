use rust_ascend::{
    Autodiff, RudaAscend,
    nn::modules::loss::CausalCrossEntropyConfig,
    runtime::{AscendRuntime, RuntimeOptions},
    tensor::api::{Int, Tensor},
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
    let input =
        Tensor::<B, 3>::from_data([[[1., 2., -1.], [0., -0.5, 3.], [1.5, 1., 0.]]], &device)
            .require_grad();
    let labels = Tensor::<B, 2, Int>::from_data([[2, 1, -100]], &device);
    let z = 1f32.exp() + 2f32.exp() + (-1f32).exp();
    let expected_loss = z.ln() - 2.;
    let expected_gradient = [
        1f32.exp() / z,
        2f32.exp() / z - 1.,
        (-1f32).exp() / z,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
    ];
    for chunk in [1, 8] {
        let result = CausalCrossEntropyConfig::new()
            .with_token_chunk_size(chunk)
            .forward_logits(input.clone(), labels.clone());
        assert_eq!(result.valid_tokens.clone().into_scalar(), 1);
        let loss = result.mean();
        assert!((loss.clone().into_scalar() - expected_loss).abs() < 1e-5);
        let gradients = loss.backward();
        let actual = input
            .grad(&gradients)
            .ok_or("missing original causal loss gradient")?
            .into_data()
            .to_vec::<f32>()?;
        for (actual, expected) in actual.iter().zip(expected_gradient) {
            assert!((actual - expected).abs() < 1e-5);
        }
    }
    let values =
        Tensor::<B, 2>::from_data([[1., 2., 3., 4.], [5., 6., 7., 8.]], &device).require_grad();
    let selected = values.clone().slice([
        rust_ascend::tensor::Slice::full(),
        rust_ascend::tensor::Slice::new(1, Some(4), -2),
    ]);
    assert_eq!(
        selected.clone().into_data().to_vec::<f32>()?,
        vec![4., 2., 8., 6.]
    );
    let gradients = selected.sum().backward();
    assert_eq!(
        values
            .grad(&gradients)
            .ok_or("missing stepped slice gradient")?
            .into_data()
            .to_vec::<f32>()?,
        vec![0., 1., 0., 1., 0., 1., 0., 1.]
    );
    println!("original RUDA causal loss and native stepped-slice backward passed");
    Ok(())
}

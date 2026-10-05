use rust_ascend::{
    Autodiff, RudaAscend,
    runtime::{AscendRuntime, RuntimeOptions},
    tensor::{DType, api::Tensor},
};

type B = Autodiff<RudaAscend>;

fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert!(
            actual.is_finite() && (actual - expected).abs() <= 0.04 * (1. + expected.abs()),
            "actual={actual}, expected={expected}"
        );
    }
}

fn check(
    input: Tensor<B, 2>,
    forward: fn(Tensor<B, 2>) -> Tensor<B, 2>,
    reference: fn(f32) -> f32,
    derivative: fn(f32) -> f32,
) -> Result<(), Box<dyn std::error::Error>> {
    let dtype = input.dtype();
    let values = input.clone().cast(DType::F32).into_data().to_vec::<f32>()?;
    let output = forward(input.clone());
    assert_eq!(output.dtype(), dtype);
    close(
        &output
            .clone()
            .cast(DType::F32)
            .into_data()
            .to_vec::<f32>()?,
        &values.iter().copied().map(reference).collect::<Vec<_>>(),
    );
    let gradients = (output.clone() + output).sum().backward();
    let gradient = input
        .grad(&gradients)
        .ok_or("missing original RUDA gradient")?;
    assert_eq!(gradient.dtype(), dtype);
    close(
        &gradient.cast(DType::F32).into_data().to_vec::<f32>()?,
        &values
            .iter()
            .map(|&x| 2. * derivative(x))
            .collect::<Vec<_>>(),
    );
    Ok(())
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
    for dtype in [DType::F16, DType::BF16] {
        let input = Tensor::<B, 2>::from_data([[-0.75, -0.25, 0.25], [0., 0.5, 0.75]], &device)
            .cast(dtype)
            .swap_dims(0, 1)
            .detach()
            .require_grad();
        check(
            input.clone(),
            |x| x.tan(),
            f32::tan,
            |x| 1. + x.tan().powi(2),
        )?;
        check(input.clone(), |x| x.sinh(), f32::sinh, f32::cosh)?;
        check(input.clone(), |x| x.cosh(), f32::cosh, f32::sinh)?;
        check(
            input.clone(),
            |x| x.asin(),
            f32::asin,
            |x| 1. / (1. - x * x).sqrt(),
        )?;
        check(
            input.clone(),
            |x| x.acos(),
            f32::acos,
            |x| -1. / (1. - x * x).sqrt(),
        )?;
        check(
            input.clone(),
            |x| x.atan(),
            f32::atan,
            |x| 1. / (1. + x * x),
        )?;
        check(input, |x| x.atanh(), f32::atanh, |x| 1. / (1. - x * x))?;
        let input = Tensor::<B, 2>::from_data([[1.25, 1.5, 2.], [2.5, 3., 4.]], &device)
            .cast(dtype)
            .swap_dims(0, 1)
            .detach()
            .require_grad();
        check(
            input,
            |x| x.acosh(),
            f32::acosh,
            |x| 1. / (x * x - 1.).sqrt(),
        )?;
        let input = Tensor::<B, 2>::from_data([[-2.5, -1.5, -0.5], [0.5, 1.5, 2.5]], &device)
            .cast(dtype)
            .detach()
            .require_grad();
        let output = input.clone().round();
        assert_eq!(output.dtype(), dtype);
        assert_eq!(
            output
                .clone()
                .cast(DType::F32)
                .into_data()
                .to_vec::<f32>()?,
            vec![-2., -2., 0., 0., 2., 2.]
        );
        let gradients = output.sum().backward();
        assert_eq!(
            input
                .grad(&gradients)
                .ok_or("missing round gradient")?
                .cast(DType::F32)
                .into_data()
                .to_vec::<f32>()?,
            vec![0.; 6]
        );
        let empty = Tensor::<RudaAscend, 2>::empty([0, 3], &device).cast(dtype);
        assert_eq!(empty.round().dims(), [0, 3]);
        println!(
            "{dtype:?}: native half math, strided views, shared RUDA gradients and ties-to-even round passed"
        );
    }
    Ok(())
}

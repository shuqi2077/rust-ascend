//! Native model position/mask tensors and RUDA's original masked autodiff.
use rust_ascend::{
    Autodiff, RudaAscend,
    runtime::{AscendRuntime, RuntimeOptions},
    tensor::{
        DType, IntDType, TensorData,
        api::{Int, Tensor, TensorCreationOptions},
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
    // SAFETY: this standalone process owns the CANN device lifecycle.
    let device = unsafe { AscendRuntime::initialize_exclusive(options)? };
    type B = Autodiff<RudaAscend>;
    let positions = Tensor::<B, 1, Int>::arange(0..3, &device)
        .cast(IntDType::I64)
        .reshape([1, 3]);
    let offsets = Tensor::<B, 2, Int>::from_data(
        TensorData::new(vec![16_777_217i64, 16_777_218], [2, 1]),
        TensorCreationOptions::<B>::new(device.clone()).with_dtype(DType::I64),
    );
    let absolute = offsets + positions.clone();
    assert_eq!(
        absolute.into_data().to_vec::<i64>()?,
        vec![
            16_777_217, 16_777_218, 16_777_219, 16_777_218, 16_777_219, 16_777_220
        ]
    );

    let valid = positions.clone().lower_elem(2).expand([2, 3]);
    let mask = valid
        .clone()
        .bool_not()
        .bool_or(positions.greater_elem(1).expand([2, 3]));
    assert_eq!(
        valid.swap_dims(0, 1).int().into_data().to_vec::<i32>()?,
        vec![1, 1, 1, 1, 0, 0]
    );
    let input = Tensor::<B, 2>::from_data([[1., 2., 3.], [4., 5., 6.]], &device).require_grad();
    let output = input
        .clone()
        .swap_dims(0, 1)
        .mask_fill(mask.swap_dims(0, 1), 0.);
    assert_eq!(
        output.clone().into_data().to_vec::<f32>()?,
        vec![1., 4., 2., 5., 0., 0.]
    );
    let gradients = output.square().sum().backward();
    let actual = input
        .grad(&gradients)
        .ok_or("missing masked input gradient")?
        .into_data()
        .to_vec::<f32>()?;
    assert_eq!(actual, vec![2., 4., 0., 8., 10., 0.]);
    for dtype in [DType::F16, DType::BF16] {
        let input = Tensor::<B, 1>::from_data([1.25, -1.25, 2., 0.], &device).cast(dtype);
        for (output, expected) in [
            (input.clone().floor(), vec![1., -2., 2., 0.]),
            (input.clone().ceil(), vec![2., -1., 2., 0.]),
            (input.trunc(), vec![1., -1., 2., 0.]),
        ] {
            assert_eq!(output.dtype(), dtype);
            assert_eq!(
                output.cast(DType::F32).into_data().to_vec::<f32>()?,
                expected
            );
        }
    }
    println!("native integer positions, strided Bool masks and masked autodiff passed");
    Ok(())
}

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
    let device = unsafe { AscendRuntime::initialize_exclusive(options)? };
    type B = Autodiff<RudaAscend>;
    for dtype in [DType::F16, DType::BF16] {
        let input = Tensor::<B, 2>::from_data([[1., 2., 3.], [-1., 4., 2.]], &device)
            .cast(dtype)
            .detach()
            .require_grad();
        let output = input.clone().cumsum(1);
        assert_eq!(output.dtype(), dtype);
        assert_eq!(output.dims(), [2, 3]);
        assert_eq!(
            output
                .clone()
                .cast(DType::F32)
                .into_data()
                .to_vec::<f32>()?,
            vec![1., 3., 6., -1., 3., 5.]
        );
        let gradients = output.cast(DType::F32).sum().backward();
        let gradient = input
            .grad(&gradients)
            .ok_or("missing original cumulative-sum gradient")?;
        assert_eq!(gradient.dtype(), dtype);
        assert_eq!(
            gradient.cast(DType::F32).into_data().to_vec::<f32>()?,
            vec![3., 2., 1., 3., 2., 1.]
        );
        let transposed = input.swap_dims(0, 1).cumsum(0);
        assert_eq!(transposed.dims(), [3, 2]);
        assert_eq!(
            transposed.cast(DType::F32).into_data().to_vec::<f32>()?,
            vec![1., -1., 3., 3., 6., 5.]
        );
    }
    for dtype in [DType::I32, DType::I64] {
        let input = Tensor::<RudaAscend, 2, Int>::from_data(
            TensorData::new(
                vec![16_777_217i64, 1, -16_777_216, -16_777_217, -1, 16_777_216],
                [2, 3],
            ),
            TensorCreationOptions::<RudaAscend>::new(device.clone()).with_dtype(dtype),
        )
        .swap_dims(0, 1);
        let output = input.cumsum(0);
        assert_eq!(output.dtype(), dtype);
        assert_eq!(output.dims(), [3, 2]);
        assert_eq!(
            output.cast(IntDType::I64).into_data().to_vec::<i64>()?,
            vec![16_777_217, -16_777_217, 16_777_218, -16_777_218, 2, -2]
        );
    }
    println!(
        "native half cumulative sums, original backward and exact-width strided integer sums passed"
    );
    Ok(())
}

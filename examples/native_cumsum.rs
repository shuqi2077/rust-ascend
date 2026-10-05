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
    for dtype in [DType::F16, DType::BF16] {
        let input = Tensor::<B, 2>::from_data([[3., 3., 2., 1.], [4., 2., 6., 7.]], &device)
            .cast(dtype)
            .swap_dims(0, 1)
            .detach()
            .require_grad();
        let minimum = input.clone().cummin(0);
        let maximum = input.clone().cummax(0);
        for (output, expected) in [
            (minimum.clone(), vec![3., 4., 3., 2., 2., 2., 1., 2.]),
            (maximum.clone(), vec![3., 4., 3., 4., 3., 6., 3., 7.]),
        ] {
            assert_eq!(output.dtype(), dtype);
            assert_eq!(output.dims(), [4, 2]);
            assert_eq!(
                output.cast(DType::F32).into_data().to_vec::<f32>()?,
                expected
            );
        }
        let gradients =
            (minimum.cast(DType::F32).sum() + maximum.cast(DType::F32).sum()).backward();
        let gradient = input
            .grad(&gradients)
            .ok_or("missing original cumulative-extrema gradient")?;
        assert_eq!(gradient.dtype(), dtype);
        assert_eq!(
            gradient.cast(DType::F32).into_data().to_vec::<f32>()?,
            vec![2., 3., 4., 3., 1., 1., 1., 1.]
        );
        let empty = Tensor::<B, 2>::empty([2, 0], &device).cast(dtype);
        assert_eq!(empty.clone().cummin(1).dims(), [2, 0]);
        assert_eq!(empty.cummax(1).dims(), [2, 0]);
    }
    let input = Tensor::<RudaAscend, 2, Int>::from_data(
        [
            [i32::MIN, i32::MIN + 1, -7, i32::MAX],
            [i32::MAX, 0, i32::MIN, 9],
        ],
        &device,
    )
    .swap_dims(0, 1);
    for (output, expected) in [
        (
            input.clone().cummin(0),
            vec![
                i32::MIN,
                i32::MAX,
                i32::MIN,
                0,
                i32::MIN,
                i32::MIN,
                i32::MIN,
                i32::MIN,
            ],
        ),
        (
            input.cummax(0),
            vec![
                i32::MIN,
                i32::MAX,
                i32::MIN + 1,
                i32::MAX,
                -7,
                i32::MAX,
                i32::MAX,
                i32::MAX,
            ],
        ),
    ] {
        assert_eq!(output.dtype(), DType::I32);
        assert_eq!(output.dims(), [4, 2]);
        assert_eq!(output.into_data().to_vec::<i32>()?, expected);
    }
    println!(
        "native half cumulative sums/extrema, original backward, tied extrema and exact-width strided integer sums/extrema passed"
    );
    Ok(())
}

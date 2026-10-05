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
        let input = Tensor::<B, 2>::from_data([[3., 3., 1., 2.], [0., -1., 5., 5.]], &device)
            .cast(dtype)
            .detach()
            .require_grad();
        let (ascending, indices) = input.clone().sort_with_indices(1);
        assert_eq!(ascending.dtype(), dtype);
        assert_eq!(
            ascending.cast(DType::F32).into_data().to_vec::<f32>()?,
            vec![1., 2., 3., 3., -1., 0., 5., 5.]
        );
        assert_eq!(
            indices.cast(IntDType::I64).into_data().to_vec::<i64>()?,
            vec![2, 3, 0, 1, 1, 0, 2, 3]
        );
        assert_eq!(
            input
                .clone()
                .argtopk(2, 1)
                .cast(IntDType::I64)
                .into_data()
                .to_vec::<i64>()?,
            vec![0, 1, 2, 3]
        );
        let output = input.clone().topk(2, 1);
        assert_eq!(output.dtype(), dtype);
        assert_eq!(
            output
                .clone()
                .cast(DType::F32)
                .into_data()
                .to_vec::<f32>()?,
            vec![3., 3., 5., 5.]
        );
        let gradients = output.cast(DType::F32).sum().backward();
        let gradient = input
            .grad(&gradients)
            .ok_or("missing original Top-K gradient")?;
        assert_eq!(gradient.dtype(), dtype);
        assert_eq!(
            gradient.cast(DType::F32).into_data().to_vec::<f32>()?,
            vec![1., 1., 0., 0., 0., 0., 1., 1.]
        );
        let (values, indices) = input
            .clone()
            .swap_dims(0, 1)
            .sort_descending_with_indices(0);
        assert_eq!(values.dims(), [4, 2]);
        assert_eq!(
            values.cast(DType::F32).into_data().to_vec::<f32>()?,
            vec![3., 5., 3., 5., 2., 0., 1., -1.]
        );
        assert_eq!(
            indices.cast(IntDType::I64).into_data().to_vec::<i64>()?,
            vec![0, 2, 1, 3, 3, 0, 2, 1]
        );
        assert_eq!(input.clone().argtopk(0, 1).dims(), [2, 0]);
        assert_eq!(input.topk_with_indices(4, 1).0.dims(), [2, 4]);
    }
    for (dtype, large) in [
        (DType::I32, 16_777_217i64),
        (DType::I64, 9_007_199_254_740_993),
    ] {
        let input = Tensor::<B, 2, Int>::from_data(
            TensorData::new(vec![large, large - 1, large, -large], [1, 4]),
            TensorCreationOptions::<B>::new(device.clone()).with_dtype(dtype),
        );
        let (values, indices) = input.clone().sort_descending_with_indices(1);
        assert_eq!(values.dtype(), dtype);
        assert_eq!(indices.dtype(), dtype);
        assert_eq!(
            values.cast(IntDType::I64).into_data().to_vec::<i64>()?,
            vec![large, large, large - 1, -large]
        );
        assert_eq!(
            indices.cast(IntDType::I64).into_data().to_vec::<i64>()?,
            vec![0, 2, 1, 3]
        );
        assert_eq!(
            input
                .clone()
                .argtopk(2, 1)
                .cast(IntDType::I64)
                .into_data()
                .to_vec::<i64>()?,
            vec![0, 2]
        );
        assert_eq!(
            input
                .topk(2, 1)
                .cast(IntDType::I64)
                .into_data()
                .to_vec::<i64>()?,
            vec![large, large]
        );
    }
    println!(
        "native stable sort and sort-based Top-K: half gradients, strided axes and exact integer widths passed"
    );
    Ok(())
}

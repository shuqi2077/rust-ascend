use rust_ascend::{
    RudaAscend,
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
    for (dtype, large) in [
        (DType::I32, 16_777_217i64),
        (DType::I64, 9_007_199_254_740_993),
    ] {
        let input = Tensor::<RudaAscend, 2, Int>::from_data(
            TensorData::new(vec![large, -large, 7, -7], [2, 2]),
            TensorCreationOptions::<RudaAscend>::new(device.clone()).with_dtype(dtype),
        )
        .swap_dims(0, 1);
        let divisor = Tensor::<RudaAscend, 2, Int>::from_data(
            TensorData::new(vec![3i64, -3], [2, 1]),
            TensorCreationOptions::<RudaAscend>::new(device.clone()).with_dtype(dtype),
        );
        let output = input.clone() / divisor.clone();
        assert_eq!(output.dtype(), dtype);
        assert_eq!(output.dims(), [2, 2]);
        assert_eq!(
            output.cast(IntDType::I64).into_data().to_vec::<i64>()?,
            vec![large / 3, 2, large / 3, 2]
        );
        let output = input.clone().div_scalar(-3);
        assert_eq!(output.dtype(), dtype);
        assert_eq!(
            output.cast(IntDType::I64).into_data().to_vec::<i64>()?,
            vec![-large / 3, -2, large / 3, 2]
        );
        let remainder = |a: i64, b: i64| {
            let a = a as i128;
            let b = b as i128;
            (((a % b) + b) % b) as i64
        };
        let output = input.clone().remainder(divisor);
        assert_eq!(output.dtype(), dtype);
        assert_eq!(output.dims(), [2, 2]);
        assert_eq!(
            output.cast(IntDType::I64).into_data().to_vec::<i64>()?,
            vec![remainder(large, 3), 1, remainder(-large, -3), -1]
        );
        let output = input.clone().remainder_scalar(-3);
        assert_eq!(output.dtype(), dtype);
        assert_eq!(
            output.cast(IntDType::I64).into_data().to_vec::<i64>()?,
            vec![remainder(large, -3), -2, remainder(-large, -3), -1]
        );
        assert_eq!(
            input.cast(IntDType::I64).into_data().to_vec::<i64>()?,
            vec![large, 7, -large, -7]
        );
    }
    println!(
        "native strided/broadcast integer division and signed remainder beyond floating precision passed"
    );
    Ok(())
}

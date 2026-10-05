use rust_ascend::{
    RudaAscend,
    runtime::{AscendRuntime, RuntimeOptions},
    tensor::{
        DType, IntDType,
        api::{Int, Tensor},
        ops::FloatTensorOps,
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
    type B = RudaAscend;
    for dtype in [DType::F16, DType::BF16] {
        let input = Tensor::<B, 2>::from_data([[3., 3., -2., -2.], [-4., 5., 5., -4.]], &device)
            .cast(dtype);
        assert_eq!(
            input.clone().argmax(1).into_data().to_vec::<i32>()?,
            vec![0, 1]
        );
        assert_eq!(
            input.clone().argmin(1).into_data().to_vec::<i32>()?,
            vec![2, 0]
        );
        for index_dtype in [IntDType::I32, IntDType::I64] {
            let input = input.clone().swap_dims(0, 1).into_primitive().tensor();
            for (min, expected) in [(false, vec![0i64, 1]), (true, vec![2, 0])] {
                let output = if min {
                    B::float_argmin(input.clone(), 0, index_dtype)
                } else {
                    B::float_argmax(input.clone(), 0, index_dtype)
                };
                let output = Tensor::<B, 2, Int>::from_primitive(output);
                assert_eq!(output.dims(), [1, 2]);
                assert_eq!(output.dtype(), DType::from(index_dtype));
                assert_eq!(
                    output.cast(IntDType::I64).into_data().to_vec::<i64>()?,
                    expected
                );
            }
        }
    }
    println!(
        "native half argmin/argmax with strided input, first-index ties and I32/I64 outputs passed"
    );
    Ok(())
}

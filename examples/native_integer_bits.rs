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
    type B = RudaAscend;
    for dtype in [DType::I32, DType::I64] {
        let upper = if dtype == DType::I32 {
            1i64 << 25
        } else {
            1i64 << 55
        };
        let values = vec![upper + 5, -1, upper + 2, -upper + 3, 0, upper + 3];
        let input = Tensor::<B, 2, Int>::from_data(
            TensorData::new(values.clone(), [2, 3]),
            TensorCreationOptions::<B>::new(device.clone()).with_dtype(dtype),
        )
        .swap_dims(0, 1);
        let masks = [upper + 3, 15];
        let mask = Tensor::<B, 2, Int>::from_data(
            TensorData::new(masks.to_vec(), [1, 2]),
            TensorCreationOptions::<B>::new(device.clone()).with_dtype(dtype),
        );
        let transposed = [
            values[0], values[3], values[1], values[4], values[2], values[5],
        ];
        let width = if dtype == DType::I32 { 32 } else { 64 };
        let shifts = [0_i64, width - 1];
        let shift = Tensor::<B, 2, Int>::from_data(
            TensorData::new(shifts.to_vec(), [1, 2]),
            TensorCreationOptions::<B>::new(device.clone()).with_dtype(dtype),
        );
        let results = [
            (
                input.clone().bitwise_and(mask.clone()),
                transposed
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| x & masks[i % 2])
                    .collect::<Vec<_>>(),
            ),
            (
                input.clone().bitwise_or(mask.clone()),
                transposed
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| x | masks[i % 2])
                    .collect::<Vec<_>>(),
            ),
            (
                input.clone().bitwise_xor(mask),
                transposed
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| x ^ masks[i % 2])
                    .collect::<Vec<_>>(),
            ),
            (
                input.clone().bitwise_and_scalar(15),
                transposed.iter().map(|&x| x & 15).collect(),
            ),
            (
                input.clone().bitwise_or_scalar(15),
                transposed.iter().map(|&x| x | 15).collect(),
            ),
            (
                input.clone().bitwise_xor_scalar(15),
                transposed.iter().map(|&x| x ^ 15).collect(),
            ),
            (
                input.clone().bitwise_not(),
                transposed.iter().map(|&x| !x).collect(),
            ),
            (
                input.clone().bitwise_right_shift(shift),
                transposed
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| x >> shifts[i % 2])
                    .collect(),
            ),
            (
                input.clone().bitwise_right_shift_scalar(1),
                transposed.iter().map(|&x| x >> 1).collect(),
            ),
            (
                input.clone().bitwise_right_shift_scalar((width - 1) as i32),
                transposed.iter().map(|&x| x >> (width - 1)).collect(),
            ),
            (input.abs(), transposed.iter().map(|&x| x.abs()).collect()),
        ];
        for (output, expected) in results {
            assert_eq!(output.dtype(), dtype);
            assert_eq!(output.dims(), [3, 2]);
            assert_eq!(
                output.cast(IntDType::I64).into_data().to_vec::<i64>()?,
                expected
            );
        }
    }
    println!(
        "native I32/I64 bitwise and/or/xor/not, arithmetic right shift and abs: exact high bits, signed values, scalars and strided broadcast passed"
    );
    Ok(())
}

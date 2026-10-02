use super::{AscendRuntime, ComputeClient, Result, TensorBuffer,
    normalization::{buffer, check_for, layout_for, row, run}};
use rust_ascend_compiler::ascend::row_programs::RowProgram;

pub(super) fn softmax(client: &ComputeClient<AscendRuntime>, input: TensorBuffer,
    logarithmic: bool) -> Result<TensorBuffer> {
    let name = if logarithmic {"LogSoftmax"} else {"Softmax"};
    let (rows,width) = layout_for(&input.shape, &input.strides, input.dtype, name)?;
    check_for(&input, &input.shape, name)?;
    let output = buffer(client, input.shape.clone(), false);
    if rows != 0 {
        let op = if logarithmic {RowProgram::LogSoftmax} else {RowProgram::Softmax};
        run(client, row(op, rows, width, 1e-5)?, &[&input,&output])?;
    }
    Ok(output)
}

pub(super) fn softmax_backward(client: &ComputeClient<AscendRuntime>, output: TensorBuffer,
    grad: TensorBuffer, logarithmic: bool) -> Result<TensorBuffer> {
    let name = if logarithmic {"LogSoftmax backward"} else {"Softmax backward"};
    let (rows,width) = layout_for(&output.shape, &output.strides, output.dtype, name)?;
    check_for(&output, &output.shape, name)?;
    check_for(&grad, &output.shape, name)?;
    let dx = buffer(client, output.shape.clone(), false);
    if rows != 0 {
        let op = if logarithmic {RowProgram::LogSoftmaxBackward} else {RowProgram::SoftmaxBackward};
        run(client, row(op, rows, width, 1e-5)?, &[&output,&grad,&dx])?;
    }
    Ok(dx)
}

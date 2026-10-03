use super::{AscendRuntime, ComputeClient, Result, TensorBuffer,
    normalization::{buffer, check_for, layout_for, row, run}};
use ruda_core::tensor::{DType, Shape};
use rust_ascend_compiler::ascend::row_programs::RowProgram;

fn reduced_shape(shape: &Shape) -> Shape {
    let mut result=shape.to_vec();
    *result.last_mut().expect("validated last dimension")=1;
    Shape::from(result)
}

pub(super) fn reduce(client: &ComputeClient<AscendRuntime>, input: TensorBuffer, mean: bool)
    -> Result<TensorBuffer> {
    if input.shape.last().is_some_and(|&width|width>4096) {return super::wide_rows::reduction(client,input,mean);}
    let (rows,width)=layout_for(&input.shape,&input.strides,input.dtype,"row reduction")?;
    check_for(&input,&input.shape,"row reduction")?;
    let output=buffer(client,reduced_shape(&input.shape),false);
    if rows!=0 {
        run(client,row(if mean {RowProgram::Mean} else {RowProgram::Sum},rows,width,1e-5)?,&[&input,&output])?;
    }
    Ok(output)
}

pub(super) fn reduce_backward(client: &ComputeClient<AscendRuntime>, shape: Shape,
    grad: TensorBuffer, mean: bool) -> Result<TensorBuffer> {
    // The derivative needs only the input layout, never saved input values.
    let mut strides=vec![1;shape.len()];let mut stride=1usize;
    for (i,&dim) in shape.iter().enumerate().rev() {
        strides[i]=stride;stride=stride.checked_mul(dim).ok_or_else(||super::error("reduction shape overflow"))?;
    }
    if shape.last().is_some_and(|&width|width>4096) {return super::wide_rows::reduction_backward(client,shape,ruda_core::tensor::Strides::from(strides),grad,mean);}
    let (rows,width)=layout_for(&shape,&strides,DType::F32,"row reduction backward")?;
    check_for(&grad,&reduced_shape(&shape),"row reduction backward")?;
    let output=buffer(client,shape,false);
    if rows!=0 {
        run(client,row(if mean {RowProgram::MeanBackward} else {RowProgram::SumBackward},rows,width,1e-5)?,&[&grad,&output])?;
    }
    Ok(output)
}

pub(super) fn softmax(client: &ComputeClient<AscendRuntime>, input: TensorBuffer,
    logarithmic: bool) -> Result<TensorBuffer> {
    if input.shape.last().is_some_and(|&width|width>4096) {return super::wide_rows::softmax(client,input,logarithmic);}
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
    if output.shape.last().is_some_and(|&width|width>4096) {return super::wide_rows::backward(client,output,grad,logarithmic);}
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

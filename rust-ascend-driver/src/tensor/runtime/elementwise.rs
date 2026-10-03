use super::{AscendRuntime,ComputeClient,Result,TensorBuffer,error,
    normalization::{buffer,check_for,map,run}};
use rust_ascend_compiler::ascend::programs::MapProgram;

fn elements(input:&TensorBuffer)->Result<usize> {
    check_for(input,&input.shape,"SiLU Mul")?;
    let n=input.shape.iter().try_fold(1usize,|n,&dim|n.checked_mul(dim))
        .ok_or_else(||error("SiLU Mul shape overflow"))?;
    if n>u32::MAX as usize {return Err(error("SiLU Mul element count exceeds u32"));}
    Ok(n)
}

pub(super) fn silu_mul(client:&ComputeClient<AscendRuntime>,gate:TensorBuffer,up:TensorBuffer)
    ->Result<TensorBuffer> {
    let n=elements(&gate)?;check_for(&up,&gate.shape,"SiLU Mul")?;
    let output=buffer(client,gate.shape.clone(),false);
    if n!=0 {run(client,map(MapProgram::SiluMul,n)?,&[&gate,&up,&output])?;}
    Ok(output)
}

pub(super) fn silu_mul_backward(client:&ComputeClient<AscendRuntime>,gate:TensorBuffer,
    up:TensorBuffer,grad:TensorBuffer)->Result<[TensorBuffer;2]> {
    let n=elements(&gate)?;
    for input in [&up,&grad] {check_for(input,&gate.shape,"SiLU Mul backward")?;}
    let dx=buffer(client,gate.shape.clone(),false);let du=buffer(client,gate.shape.clone(),false);
    if n!=0 {run(client,map(MapProgram::SiluMulBackward,n)?,&[&gate,&up,&grad,&dx,&du])?;}
    Ok([dx,du])
}

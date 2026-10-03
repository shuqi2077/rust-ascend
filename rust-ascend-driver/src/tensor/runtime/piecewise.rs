use super::{AscendRuntime,ComputeClient,Result,TensorBuffer,error,normalization::{buffer,check_for,run}};
use ruda_core::{compiler::Compiler,ir::UIntKind,launch::ExecutionMode};
use rust_ascend_compiler::ascend::{AscendCompiler,AscendOptions,AscendTarget,piecewise_programs::{self,PiecewiseActivation}};
type Client=ComputeClient<AscendRuntime>;
fn count(shape:&[usize])->Result<u64> {
    if !(1..=8).contains(&shape.len()) {return Err(error("piecewise activation requires rank 1..8"));}
    // Check the same suffix products used to construct contiguous strides, including empty shapes.
    shape.iter().rev().try_fold(1usize,|n,&d|n.checked_mul(d))
        .filter(|&n|n<=u32::MAX as usize).map(|n|n as u64)
        .ok_or_else(||error("piecewise activation shape/stride overflow or domain exceeds u32"))
}
pub(super) fn execute(client:&Client,input:TensorBuffer,grad:Option<TensorBuffer>,activation:PiecewiseActivation)->Result<TensorBuffer> {
    let elements=count(&input.shape)?;check_for(&input,&input.shape,"piecewise activation")?;
    if let Some(grad)=&grad {check_for(grad,&input.shape,"piecewise activation backward")?;}
    let output=buffer(client,input.shape.clone(),false);
    if elements!=0 {
        let kernel=AscendCompiler.compile(piecewise_programs::definition(activation,elements,grad.is_some()).map_err(error)?,
            &AscendOptions {target:Some(AscendTarget::Ascend950DT),elements,..Default::default()},
            ExecutionMode::Checked,UIntKind::U64.into()).map_err(error)?;
        let mut tensors=vec![&input];if let Some(grad)=&grad {tensors.push(grad);}tensors.push(&output);
        run(client,kernel,&tensors)?;
    }Ok(output)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn complete_piecewise_domain_includes_empty_last_axis_but_checks_suffix_overflow() {
        for (shape,expected) in [(vec![7],7),(vec![2,3,33],198),(vec![0,7],0),(vec![3,0],0),
            (vec![1,1,1,1,1,1,1,65],65)] {assert_eq!(count(&shape).unwrap(),expected);}
        for shape in [vec![],vec![1;9],vec![u32::MAX as usize,2],vec![0,usize::MAX,2]] {assert!(count(&shape).is_err());}
    }
}

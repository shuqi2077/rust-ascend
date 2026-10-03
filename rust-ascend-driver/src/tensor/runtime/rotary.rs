use super::{AscendRuntime,ComputeClient,Result,TensorBuffer,error,
    normalization::{buffer,check_for,run}};
use ruda_core::{compiler::Compiler,ir::UIntKind,launch::ExecutionMode,tensor::Shape};
use rust_ascend_compiler::ascend::{AscendCompiler,AscendOptions,AscendTarget,rotary_programs::{self,RotaryLayout}};

fn table_shape(shape:&Shape)->Result<(Shape,usize,u32)> {
    if !(1..=8).contains(&shape.len()) {return Err(error("rotary requires rank 1..8"));}
    let width=shape[shape.len()-1];
    if width==0 || width%2!=0 || width>u32::MAX as usize {return Err(error("rotary width must be positive and even within u32"));}
    let n=shape.iter().try_fold(1usize,|n,&dim|n.checked_mul(dim)).ok_or_else(||error("rotary shape overflow"))?;
    if n>u32::MAX as usize {return Err(error("rotary element count exceeds u32"));}
    let mut table=shape.to_vec();*table.last_mut().unwrap()=width/2;
    Ok((Shape::from(table),n,width as u32))
}
pub(super) fn rotate(client:&ComputeClient<AscendRuntime>,input:TensorBuffer,cos:TensorBuffer,sin:TensorBuffer,
    layout:RotaryLayout,backward:bool)->Result<TensorBuffer> {
    let (shape,n,width)=table_shape(&input.shape)?;
    check_for(&input,&input.shape,"rotary")?;
    check_for(&cos,&shape,"rotary cos")?;check_for(&sin,&shape,"rotary sin")?;
    let output=buffer(client,input.shape.clone(),false);
    if n!=0 {
        let kernel=AscendCompiler.compile(rotary_programs::definition(width,n as u64,layout,backward).map_err(error)?,
            &AscendOptions {target:Some(AscendTarget::Ascend950DT),elements:n as u64,..Default::default()},
            ExecutionMode::Checked,UIntKind::U64.into()).map_err(error)?;
        // Two readonly aliases permit independent original/paired common-IR load layouts.
        run(client,kernel,&[&input,&input,&cos,&sin,&output])?;
    }
    Ok(output)
}

use super::{AscendRuntime,ComputeClient,Result,TensorBuffer,error,
    normalization::{buffer,check_for,run},indexing};
use ruda_core::{compiler::Compiler,ir::UIntKind,launch::ExecutionMode,tensor::Shape};
use rust_ascend_compiler::ascend::{AscendCompiler,AscendOptions,AscendTarget,rotary_programs::{self,RotaryLayout,PrefixRotarySpec}};

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
fn prefix_spec(input:&Shape,table:&Shape,width:u32,layout:RotaryLayout)->Result<PrefixRotarySpec> {
    let dimensions=|shape:&Shape|shape.iter().map(|&d|u32::try_from(d).map_err(error)).collect::<Result<Vec<_>>>();
    let spec=PrefixRotarySpec {input:dimensions(input)?,table:dimensions(table)?,rotary_width:width,layout};
    spec.elements().map_err(error)?;Ok(spec)
}
pub(super) fn prefix(client:&ComputeClient<AscendRuntime>,input:TensorBuffer,cos:TensorBuffer,sin:TensorBuffer,
    width:u32,layout:RotaryLayout,backward:bool)->Result<TensorBuffer> {
    let spec=prefix_spec(&input.shape,&cos.shape,width,layout)?;
    check_for(&input,&input.shape,"rotary prefix")?;check_for(&cos,&cos.shape,"rotary prefix cos")?;
    check_for(&sin,&cos.shape,"rotary prefix sin")?;
    let input_layout=indexing::layout(&input)?;let output=indexing::allocate(client,&input_layout);
    let (n,_)=spec.elements().map_err(error)?;
    if n!=0 {
        let kernel=AscendCompiler.compile(rotary_programs::prefix_definition(&spec,backward).map_err(error)?,
            &AscendOptions {target:Some(AscendTarget::Ascend950DT),elements:n,..Default::default()},ExecutionMode::Checked,UIntKind::U64.into()).map_err(error)?;
        run(client,kernel,&[&input,&input,&cos,&sin,&output])?;
    }
    Ok(output)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prefix_layout_accepts_shared_sequence_per_batch_and_per_head_tables() {
        for table in [[1,1,5,3],[2,1,5,3],[1,3,1,3],[2,3,5,3]] {
            assert!(prefix_spec(&Shape::new([2,3,5,11]),&Shape::new(table),6,RotaryLayout::SplitHalf).is_ok());
        }
        for table in [[1,2,5,3],[2,3,4,3],[2,3,5,6]] {
            assert!(prefix_spec(&Shape::new([2,3,5,11]),&Shape::new(table),6,RotaryLayout::SplitHalf).is_err());
        }
        assert!(prefix_spec(&Shape::new([0,3,5,11]),&Shape::new([1,1,5,3]),6,RotaryLayout::Interleaved).is_ok());
        assert!(prefix_spec(&Shape::new([2,3,5,11]),&Shape::new([5,3]),6,RotaryLayout::Interleaved).is_err());
    }
}

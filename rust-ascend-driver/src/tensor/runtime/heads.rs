use super::{AscendRuntime,ComputeClient,Result,TensorBuffer,RepeatKvSpec,error,
    normalization::{check_for,run},indexing::allocate};
use crate::tensor::{TensorLayout,DType};
use ruda_core::{compiler::Compiler,ir::UIntKind,kernel::KernelDefinition,launch::ExecutionMode,tensor::Shape};
use rust_ascend_compiler::ascend::{AscendCompiler,AscendOptions,AscendTarget,heads_programs};
type Client=ComputeClient<AscendRuntime>;
fn specification(shape:&Shape,query_heads:u32)->Result<RepeatKvSpec> {
    if shape.len()!=4 {return Err(error("KV repetition requires [B,Hkv,N,D]"));}
    let dims=shape.iter().map(|&n|u32::try_from(n).map_err(error)).collect::<Result<Vec<_>>>()?;
    let spec=RepeatKvSpec {batch:dims[0],kv_heads:dims[1],query_heads,sequence:dims[2],width:dims[3]};
    spec.elements().map_err(error)?;Ok(spec)
}
fn output(client:&Client,shape:[u32;4])->Result<TensorBuffer> {
    let layout=TensorLayout::contiguous(&shape.map(i64::from),DType::F32)?;Ok(allocate(client,&layout))
}
fn compile(kernel:KernelDefinition,elements:u64,partial:bool)->Result<rust_ascend_compiler::ascend::AscendKernel> {
    let options=AscendOptions {target:Some(AscendTarget::Ascend950DT),elements,..Default::default()};
    if partial {AscendCompiler.compile_partial_map(kernel,&options,ExecutionMode::Checked,UIntKind::U64.into())}
        else {AscendCompiler.compile(kernel,&options,ExecutionMode::Checked,UIntKind::U64.into())}.map_err(error)
}
pub(super) fn repeat(client:&Client,input:TensorBuffer,query_heads:u32)->Result<TensorBuffer> {
    let spec=specification(&input.shape,query_heads)?;check_for(&input,&input.shape,"KV repetition")?;
    if spec.query_heads==spec.kv_heads {return Ok(input);}
    let (_,n)=spec.elements().map_err(error)?;
    let out=output(client,[spec.batch,spec.query_heads,spec.sequence,spec.width])?;
    if n!=0 {run(client,compile(heads_programs::repeat_definition(spec).map_err(error)?,n,false)?,&[&input,&out])?;}
    Ok(out)
}
pub(super) fn backward(client:&Client,input_shape:Shape,query_heads:u32,grad:TensorBuffer)->Result<TensorBuffer> {
    let spec=specification(&input_shape,query_heads)?;
    check_for(&grad,&[spec.batch as usize,spec.query_heads as usize,spec.sequence as usize,spec.width as usize],"KV repetition backward")?;
    let mut groups=spec.query_heads/spec.kv_heads;
    if groups==1 {return Ok(grad);}
    let (n,_)=spec.elements().map_err(error)?;
    if n==0 {return output(client,[spec.batch,spec.kv_heads,spec.sequence,spec.width]);}
    let rows=spec.batch as u64*spec.kv_heads as u64;let width=spec.sequence as u64*spec.width as u64;
    let mut value=grad;
    while groups>1 {
        let next_groups=groups.div_ceil(2);
        let next=output(client,[spec.batch,spec.kv_heads*next_groups,spec.sequence,spec.width])?;
        let (kernel,elements)=heads_programs::reduce_definition(rows,groups,width,false).map_err(error)?;
        run(client,compile(kernel,elements,true)?,&[&value,&value,&next])?;
        if groups%2!=0 {
            let (kernel,elements)=heads_programs::reduce_definition(rows,groups,width,true).map_err(error)?;
            run(client,compile(kernel,elements,true)?,&[&value,&next])?;
        }
        // Pair and optional tail patches cover the complete next allocation before reuse.
        value=next;groups=next_groups;
    }
    Ok(value)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn kv_shapes_preserve_mha_mqa_gqa_and_empty_domains() {
        for (shape,heads) in [([2,3,8192,128],15),([2,1,96,7],9),([0,3,96,7],3),([2,3,0,7],6),([2,3,96,0],6)] {
            assert!(specification(&Shape::new(shape),heads).is_ok());
        }
        for (shape,heads) in [([2,3,96,7],5),([2,0,96,7],6),([2,3,96,7],0),([2,3,u32::MAX as usize,128],6)] {
            assert!(specification(&Shape::new(shape),heads).is_err());
        }
        assert!(specification(&Shape::new([2,3,96]),6).is_err());
    }
}

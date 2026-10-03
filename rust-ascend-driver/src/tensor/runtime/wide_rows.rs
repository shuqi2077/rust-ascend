use super::{AscendRuntime,ComputeClient,Result,TensorBuffer,error,contiguous,
    normalization::{buffer,check_for,row,run,epsilon_for,column_sum,slice}};
use ruda_core::{compiler::Compiler,ir::UIntKind,kernel::KernelDefinition,launch::ExecutionMode,tensor::{DType,Shape,Strides}};
use rust_ascend_compiler::ascend::{AscendCompiler,AscendOptions,AscendTarget,row_programs::RowProgram,
    programs::{self,MapProgram},wide_programs::{self,WideStage}};
const TILE:usize=4096;
type Client=ComputeClient<AscendRuntime>;
pub(super) fn needs_tiles(width:usize)->bool {width>4096 || width%32!=0}
pub(super) fn layout(input:&TensorBuffer)->Result<(usize,u32)> {
    let result=layout_for(&input.shape,&input.strides,input.dtype)?;
    check_for(input,&input.shape,"wide rows")?;Ok(result)
}
fn layout_for(shape:&[usize],strides:&[usize],dtype:DType)->Result<(usize,u32)> {
    if dtype!=DType::F32 || !contiguous(shape,strides) {return Err(error("wide rows require contiguous FP32"));}
    let &width=shape.last().ok_or_else(||error("wide rows require a last axis"))?;
    if width==0 || width>u32::MAX as usize {return Err(error("tiled row width must be positive within u32"));}
    let rows=shape[..shape.len()-1].iter().try_fold(1usize,|n,&d|n.checked_mul(d)).ok_or_else(||error("wide row shape overflow"))?;
    if rows.checked_mul(width).is_none_or(|n|n>u32::MAX as usize) {return Err(error("wide row element count exceeds u32"));}
    Ok((rows,width as u32))
}
fn compile(definition:KernelDefinition,elements:u64,partial:bool)->Result<rust_ascend_compiler::ascend::AscendKernel> {
    let options=AscendOptions {target:Some(AscendTarget::Ascend950DT),elements,..Default::default()};
    if partial {AscendCompiler.compile_partial_map(definition,&options,ExecutionMode::Checked,UIntKind::U64.into())}
        else {AscendCompiler.compile(definition,&options,ExecutionMode::Checked,UIntKind::U64.into())}.map_err(error)
}
fn stage(client:&Client,kind:WideStage,rows:usize,width:u32,start:u32,columns:u32,tensors:&[&TensorBuffer])->Result<()> {
    run(client,compile(wide_programs::definition(kind,rows as u64,width,start,columns).map_err(error)?,rows as u64*columns as u64,kind.partial())?,tensors)
}
fn output(client:&Client,shape:Shape)->Result<TensorBuffer> {
    let value=buffer(client,shape,false);let elements=value.shape.iter().product::<usize>() as u64;
    // Partial patches retain unwritten storage. Initialize it on-device before patching.
    run(client,compile(wide_programs::zero_definition(elements).map_err(error)?,elements,false)?,&[&value])?;Ok(value)
}
fn merge(client:&Client,a:TensorBuffer,b:TensorBuffer,rows:usize,max:bool)->Result<TensorBuffer> {
    let value=buffer(client,Shape::new([rows]),false);
    run(client,compile(wide_programs::merge_definition(rows as u64,max).map_err(error)?,rows as u64,false)?,&[&a,&b,&value])?;Ok(value)
}
fn reduce(client:&Client,value:&TensorBuffer,rows:usize,columns:u32,max:bool)->Result<TensorBuffer> {
    let stat=buffer(client,Shape::new([rows]),false);
    if columns==1 {
        run(client,compile(programs::definition(MapProgram::Copy),rows as u64,false)?,&[value,&stat])?;
    } else if columns%32==0 {
        run(client,row(if max {RowProgram::Max} else {RowProgram::Sum},rows,columns,1e-5)?,&[value,&stat])?;
    } else {
        let aligned=columns.div_ceil(32)*32;
        // Preserve the existing full logical u32 domain. Split row batches if
        // expanding a tiny tile to 32 lanes would otherwise exceed that domain.
        let batch_rows=u32::MAX as usize/aligned as usize;
        for start in (0..rows).step_by(batch_rows) {
            let count=(rows-start).min(batch_rows);
            let input=slice(value,start*columns as usize,count*columns as usize);
            let padded=buffer(client,Shape::new([count,aligned as usize]),false);
            let out=slice(&stat,start,count);
            run(client,compile(wide_programs::reduction_pad_definition(count as u64,columns,max).map_err(error)?,
                count as u64*aligned as u64,false)?,&[&input,&padded])?;
            run(client,row(if max {RowProgram::Max} else {RowProgram::Sum},count,aligned,1e-5)?,&[&padded,&out])?;
        }
    }
    Ok(stat)
}
pub(super) fn reduction(client:&Client,input:TensorBuffer,mean:bool)->Result<TensorBuffer> {
    let (rows,width)=layout(&input)?;let mut shape=input.shape.clone();*shape.last_mut().expect("validated last axis")=1;
    let out=buffer(client,shape,false);if rows==0 {return Ok(out);}
    let mut total=None;
    for start in (0..width as usize).step_by(TILE) {
        let columns=(width as usize-start).min(TILE) as u32;let tile=buffer(client,Shape::new([rows,columns as usize]),false);
        stage(client,WideStage::CopyTile,rows,width,start as u32,columns,&[&input,&tile])?;
        let current=reduce(client,&tile,rows,columns,false)?;
        total=Some(match total {Some(previous)=>merge(client,previous,current,rows,false)?,None=>current});
    }
    let total=total.ok_or_else(||error("wide reduction has no tiles"))?;
    let definition=if mean {wide_programs::mean_definition(rows as u64,width).map_err(error)?} else {programs::definition(MapProgram::Copy)};
    run(client,compile(definition,rows as u64,false)?,&[&total,&out])?;Ok(out)
}
pub(super) fn reduction_backward(client:&Client,shape:Shape,strides:Strides,grad:TensorBuffer,mean:bool)->Result<TensorBuffer> {
    let (rows,width)=layout_for(&shape,&strides,DType::F32)?;
    let mut reduced=shape.clone();*reduced.last_mut().expect("validated last axis")=1;
    check_for(&grad,&reduced,"wide reduction backward")?;
    if rows==0 {return Ok(buffer(client,shape,false));}
    let out=output(client,shape)?;
    for start in (0..width as usize).step_by(TILE) {
        let columns=(width as usize-start).min(TILE) as u32;
        stage(client,if mean {WideStage::MeanBackwardTile} else {WideStage::SumBackwardTile},rows,width,start as u32,columns,&[&grad,&out])?;
    }
    Ok(out)
}
fn tiled_sum(client:&Client,kind:WideStage,rows:usize,width:u32,inputs:&[&TensorBuffer])->Result<TensorBuffer> {
    let mut total=None;
    for start in (0..width as usize).step_by(TILE) {
        let columns=(width as usize-start).min(TILE) as u32;let tile=buffer(client,Shape::new([rows,columns as usize]),false);
        let mut bindings=inputs.to_vec();bindings.push(&tile);stage(client,kind,rows,width,start as u32,columns,&bindings)?;
        let current=reduce(client,&tile,rows,columns,false)?;
        total=Some(match total {Some(previous)=>merge(client,previous,current,rows,false)?,None=>current});
    }
    total.ok_or_else(||error("wide statistic has no tiles"))
}
fn average(client:&Client,sum:TensorBuffer,rows:usize,width:u32)->Result<TensorBuffer> {
    let value=buffer(client,Shape::new([rows]),false);
    run(client,compile(wide_programs::mean_definition(rows as u64,width).map_err(error)?,rows as u64,false)?,&[&sum,&value])?;Ok(value)
}
pub(super) fn layer_forward(client:&Client,input:TensorBuffer,weight:TensorBuffer,bias:Option<TensorBuffer>,eps:f64)->Result<[TensorBuffer;3]> {
    let (rows,width)=layout(&input)?;let eps=epsilon_for(eps,"wide LayerNorm")?;
    check_for(&weight,&[width as usize],"wide LayerNorm weight")?;
    if let Some(bias)=&bias {check_for(bias,&[width as usize],"wide LayerNorm bias")?;}
    if rows==0 {return Ok([buffer(client,input.shape,false),buffer(client,Shape::new([rows]),false),buffer(client,Shape::new([rows]),false)]);}
    let sum=tiled_sum(client,WideStage::CopyTile,rows,width,&[&input])?;let mean=average(client,sum,rows,width)?;
    // Center before squaring, preserving the original variance formula and divisor width.
    let squares=tiled_sum(client,WideStage::CenteredSquareTile,rows,width,&[&input,&mean])?;
    let rstd=buffer(client,Shape::new([rows]),false);
    run(client,compile(wide_programs::rstd_definition(rows as u64,width,eps).map_err(error)?,rows as u64,false)?,&[&squares,&rstd])?;
    let bias=match bias {Some(value)=>value,None=>output(client,Shape::new([width as usize]))?};
    let out=output(client,input.shape.clone())?;
    for start in (0..width as usize).step_by(TILE) {
        let columns=(width as usize-start).min(TILE) as u32;
        stage(client,WideStage::LayerNormTile,rows,width,start as u32,columns,&[&input,&weight,&bias,&mean,&rstd,&out])?;
    }
    Ok([out,mean,rstd])
}
pub(super) fn layer_backward(client:&Client,input:TensorBuffer,weight:TensorBuffer,grad:TensorBuffer,mean:TensorBuffer,rstd:TensorBuffer)->Result<[TensorBuffer;3]> {
    let (rows,width)=layout(&input)?;check_for(&weight,&[width as usize],"wide LayerNorm weight")?;
    check_for(&grad,&input.shape,"wide LayerNorm grad")?;check_for(&mean,&[rows],"wide LayerNorm mean")?;check_for(&rstd,&[rows],"wide LayerNorm rstd")?;
    if rows==0 {return Ok([buffer(client,input.shape,false),output(client,Shape::new([width as usize]))?,output(client,Shape::new([width as usize]))?]);}
    let sum_g=tiled_sum(client,WideStage::LayerGTile,rows,width,&[&grad,&weight])?;let mean_g=average(client,sum_g,rows,width)?;
    let sum_gy=tiled_sum(client,WideStage::LayerGYTile,rows,width,&[&input,&grad,&weight,&mean,&rstd])?;let mean_gy=average(client,sum_gy,rows,width)?;
    let dx=output(client,input.shape.clone())?;let parts=output(client,input.shape.clone())?;
    for start in (0..width as usize).step_by(TILE) {
        let columns=(width as usize-start).min(TILE) as u32;
        stage(client,WideStage::LayerBackwardTile,rows,width,start as u32,columns,&[&input,&grad,&weight,&mean,&rstd,&mean_g,&mean_gy,&dx])?;
        stage(client,WideStage::LayerWeightTile,rows,width,start as u32,columns,&[&input,&grad,&mean,&rstd,&parts])?;
    }
    let dw=column_sum(client,parts,rows,width as usize)?;let db=column_sum(client,grad,rows,width as usize)?;Ok([dx,dw,db])
}
pub(super) fn rms_forward(client:&Client,input:TensorBuffer,weight:TensorBuffer,eps:f64)->Result<[TensorBuffer;2]> {
    let (rows,width)=layout(&input)?;let eps=epsilon_for(eps,"wide RMSNorm")?;
    check_for(&weight,&[width as usize],"wide RMSNorm weight")?;
    let rstd=buffer(client,Shape::new([rows]),false);
    if rows==0 {return Ok([buffer(client,input.shape,false),rstd]);}
    let sum=tiled_sum(client,WideStage::SquareTile,rows,width,&[&input])?;
    run(client,compile(wide_programs::rstd_definition(rows as u64,width,eps).map_err(error)?,rows as u64,false)?,&[&sum,&rstd])?;
    let out=output(client,input.shape.clone())?;
    for start in (0..width as usize).step_by(TILE) {
        let columns=(width as usize-start).min(TILE) as u32;
        stage(client,WideStage::RmsNormTile,rows,width,start as u32,columns,&[&input,&weight,&rstd,&out])?;
    }
    Ok([out,rstd])
}
pub(super) fn rms_backward(client:&Client,input:TensorBuffer,weight:TensorBuffer,grad:TensorBuffer,rstd:TensorBuffer)->Result<[TensorBuffer;2]> {
    let (rows,width)=layout(&input)?;check_for(&weight,&[width as usize],"wide RMSNorm weight")?;
    check_for(&grad,&input.shape,"wide RMSNorm grad")?;check_for(&rstd,&[rows],"wide RMSNorm rstd")?;
    if rows==0 {return Ok([buffer(client,input.shape,false),output(client,Shape::new([width as usize]))?]);}
    let sum=tiled_sum(client,WideStage::RmsDotTile,rows,width,&[&input,&grad,&weight])?;
    let mean=buffer(client,Shape::new([rows]),false);
    run(client,compile(wide_programs::mean_definition(rows as u64,width).map_err(error)?,rows as u64,false)?,&[&sum,&mean])?;
    let dx=output(client,input.shape.clone())?;let parts=output(client,input.shape.clone())?;
    for start in (0..width as usize).step_by(TILE) {
        let columns=(width as usize-start).min(TILE) as u32;
        stage(client,WideStage::RmsBackwardTile,rows,width,start as u32,columns,&[&input,&grad,&weight,&rstd,&mean,&dx])?;
        stage(client,WideStage::RmsWeightTile,rows,width,start as u32,columns,&[&input,&grad,&rstd,&parts])?;
    }
    let dw=column_sum(client,parts,rows,width as usize)?;Ok([dx,dw])
}
pub(super) fn softmax(client:&Client,input:TensorBuffer,logarithmic:bool)->Result<TensorBuffer> {
    let (rows,width)=layout(&input)?;
    if rows==0 {return Ok(buffer(client,input.shape,false));}
    let mut maximum=None;
    for start in (0..width as usize).step_by(TILE) {
        let columns=(width as usize-start).min(TILE) as u32;
        let tile=buffer(client,Shape::new([rows,columns as usize]),false);
        stage(client,WideStage::CopyTile,rows,width,start as u32,columns,&[&input,&tile])?;
        let current=reduce(client,&tile,rows,columns,true)?;
        maximum=Some(match maximum {Some(previous)=>merge(client,previous,current,rows,true)?,None=>current});
    }
    let maximum=maximum.ok_or_else(||error("wide rows have no tiles"))?;
    let mut total=None;let mut exponents=vec![];
    for start in (0..width as usize).step_by(TILE) {
        let columns=(width as usize-start).min(TILE) as u32;
        let tile=buffer(client,Shape::new([rows,columns as usize]),false);
        stage(client,WideStage::ExpTile,rows,width,start as u32,columns,&[&input,&maximum,&tile])?;
        let current=reduce(client,&tile,rows,columns,false)?;
        total=Some(match total {Some(previous)=>merge(client,previous,current,rows,false)?,None=>current});
        if !logarithmic {exponents.push((start as u32,columns,tile));}
    }
    let total=total.ok_or_else(||error("wide rows have no exponent sum"))?;
    let out=output(client,input.shape.clone())?;
    if logarithmic {
        for start in (0..width as usize).step_by(TILE) {
            let columns=(width as usize-start).min(TILE) as u32;
            stage(client,WideStage::LogSoftmaxTile,rows,width,start as u32,columns,&[&input,&maximum,&total,&out])?;
        }
    } else {
        for (start,columns,tile) in exponents {stage(client,WideStage::SoftmaxTile,rows,width,start,columns,&[&tile,&total,&out])?;}
    }
    Ok(out)
}

pub(super) fn maximum(client:&Client,input:TensorBuffer)->Result<TensorBuffer> {
    let (rows,width)=layout(&input)?;
    let mut shape=input.shape.clone();*shape.last_mut().expect("validated last axis")=1;
    let out=buffer(client,shape,false);if rows==0 {return Ok(out);}
    let mut maximum=None;
    for start in (0..width as usize).step_by(TILE) {
        let columns=(width as usize-start).min(TILE) as u32;
        let tile=buffer(client,Shape::new([rows,columns as usize]),false);
        stage(client,WideStage::CopyTile,rows,width,start as u32,columns,&[&input,&tile])?;
        let current=reduce(client,&tile,rows,columns,true)?;
        maximum=Some(match maximum {Some(previous)=>merge(client,previous,current,rows,true)?,None=>current});
    }
    run(client,compile(programs::definition(MapProgram::Copy),rows as u64,false)?,
        &[&maximum.ok_or_else(||error("wide maximum has no tiles"))?,&out])?;
    Ok(out)
}
pub(super) fn backward(client:&Client,saved:TensorBuffer,grad:TensorBuffer,logarithmic:bool)->Result<TensorBuffer> {
    let (rows,width)=layout(&saved)?;check_for(&grad,&saved.shape,"wide Softmax backward")?;
    if rows==0 {return Ok(buffer(client,saved.shape,false));}
    let mut total=None;
    for start in (0..width as usize).step_by(TILE) {
        let columns=(width as usize-start).min(TILE) as u32;
        let tile=buffer(client,Shape::new([rows,columns as usize]),false);
        if logarithmic {stage(client,WideStage::CopyTile,rows,width,start as u32,columns,&[&grad,&tile])?;}
        else {stage(client,WideStage::DotTile,rows,width,start as u32,columns,&[&saved,&grad,&tile])?;}
        let current=reduce(client,&tile,rows,columns,false)?;
        total=Some(match total {Some(previous)=>merge(client,previous,current,rows,false)?,None=>current});
    }
    let total=total.ok_or_else(||error("wide backward has no tiles"))?;
    let out=output(client,saved.shape.clone())?;
    for start in (0..width as usize).step_by(TILE) {
        let columns=(width as usize-start).min(TILE) as u32;
        stage(client,if logarithmic {WideStage::LogSoftmaxBackwardTile} else {WideStage::SoftmaxBackwardTile},
            rows,width,start as u32,columns,&[&saved,&grad,&total,&out])?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn wide_rows_keep_full_domain_and_reject_invalid_layouts() {
        assert_eq!(layout_for(&[2,3,8224],&[24672,8224,1],DType::F32).unwrap(),(6,8224));
        assert_eq!(layout_for(&[1,0,4128],&[0,4128,1],DType::F32).unwrap(),(0,4128));
        for width in [1,2,7,31,33,65,4095,4097,4129,8225] {assert_eq!(layout_for(&[width],&[1],DType::F32).unwrap(),(1,width as u32));}
        assert!(layout_for(&[0],&[1],DType::F32).is_err());
        for width in [32,4096] {assert!(!needs_tiles(width));}
        for width in [1,31,33,4095,4097,4128] {assert!(needs_tiles(width));}
        assert!(layout_for(&[2,4128],&[1,2],DType::F32).is_err());
        assert!(layout_for(&[4128],&[1],DType::BF16).is_err());
        assert!(layout_for(&[u32::MAX as usize,4128],&[4128,1],DType::F32).is_err());
    }
}

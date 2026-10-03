use super::{AscendRuntime,ComputeClient,Result,TensorBuffer,WORKER,contiguous,error};
use crate::tensor::{DType as CannDType,TensorLayout};
use ruda_core::tensor::{DType,Shape,Strides};

fn dtype(value:DType)->Result<CannDType> {
    match value {DType::F32=>Ok(CannDType::F32),DType::F16=>Ok(CannDType::F16),DType::BF16=>Ok(CannDType::BF16),
        _=>Err(error("runtime Cast requires FP32/FP16/BF16 dtype"))}
}
fn layout_parts(shape:&[usize],strides:&[usize],value:DType)->Result<TensorLayout> {
    if !(1..=8).contains(&shape.len()) || !contiguous(shape,strides) {
        return Err(error("runtime Cast requires contiguous rank 1..8 tensors"));
    }
    let shape=shape.iter().map(|&d|i64::try_from(d).map_err(error)).collect::<Result<Vec<_>>>()?;
    TensorLayout::contiguous(&shape,dtype(value)?)
}
fn layout(tensor:&TensorBuffer)->Result<TensorLayout> {
    let value=layout_parts(&tensor.shape,&tensor.strides,tensor.dtype)?;
    if tensor.handle.size_in_used()<value.byte_len() as u64 {return Err(error("Cast buffer is shorter than its layout"));}
    Ok(value)
}
pub(super) fn cast(client:&ComputeClient<AscendRuntime>,input:TensorBuffer,value:DType)->Result<TensorBuffer> {
    let source=layout(&input)?;
    let target=TensorLayout::contiguous(source.shape(),dtype(value)?)?;
    let output=TensorBuffer {handle:client.empty(target.byte_len()),shape:Shape::from(input.shape.to_vec()),
        strides:Strides::from(target.strides().iter().map(|&s|s as usize).collect::<Vec<_>>()),dtype:value};
    cast_into(client,input,output.clone())?;
    Ok(output)
}
pub(super) fn cast_into(client:&ComputeClient<AscendRuntime>,input:TensorBuffer,output:TensorBuffer)->Result<()> {
    let layouts=[layout(&input)?,layout(&output)?];
    if layouts[0].shape()!=layouts[1].shape() {return Err(error("Cast output shape must match input"));}
    client.flush().map_err(error)?;
    if layouts[0].byte_len()==0 {return Ok(());}
    let guards=[&input,&output].iter().map(|t|client.get_resource(t.handle.clone()).map_err(error))
        .collect::<Result<Vec<_>>>()?;
    let resources=guards.iter().zip(&layouts).map(|(guard,layout)| {
        let mut resource=guard.resource().clone();
        if resource.byte_len()<layout.byte_len() {return Err(error("Cast resource is too short"));}
        resource.size=layout.byte_len();Ok(resource)
    }).collect::<Result<Vec<_>>>()?.try_into().map_err(|_|error("Cast binding count mismatch"))?;
    let worker=&WORKER.get().ok_or_else(||error("Ascend runtime is not initialized"))?.1;
    let result=worker.call(move |state|state.cast(layouts,resources));
    drop(guards);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cast_layout_preserves_shape_and_explicit_storage_precision() {
        for (dtype,size) in [(DType::F32,4),(DType::F16,2),(DType::BF16,2)] {
            assert_eq!(layout_parts(&[2,65],&[65,1],dtype).unwrap().byte_len(),130*size);
            assert_eq!(layout_parts(&[0,65],&[65,1],dtype).unwrap().byte_len(),0);
        }
        assert!(layout_parts(&[2,65],&[1,2],DType::F32).is_err());
        assert!(layout_parts(&[],&[],DType::F32).is_err());
        assert!(layout_parts(&[1;9],&[1;9],DType::F32).is_err());
        assert!(layout_parts(&[usize::MAX,65],&[65,1],DType::F32).is_err());
        assert!(layout_parts(&[65],&[1],DType::I32).is_err());
    }
}

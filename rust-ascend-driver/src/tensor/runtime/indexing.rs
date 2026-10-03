use super::{AscendRuntime,ComputeClient,Result,TensorBuffer,WORKER,EmbeddingOptions,error,contiguous};
use crate::tensor::{DType as CannDType,TensorLayout,embedding::{forward_layout,backward_layout}};
use ruda_core::tensor::{DType,Shape,Strides};
type Client=ComputeClient<AscendRuntime>;
fn layout_parts(shape:&[usize],strides:&[usize],dtype:DType)->Result<TensorLayout> {
    if !contiguous(shape,strides) {return Err(error("typed tensor operation requires contiguous storage"));}
    let dtype=match dtype {DType::F32=>CannDType::F32,DType::F16=>CannDType::F16,DType::BF16=>CannDType::BF16,
        DType::I32=>CannDType::I32,DType::I64=>CannDType::I64,_=>return Err(error("typed tensor storage dtype is unsupported"))};
    let shape=shape.iter().map(|&d|i64::try_from(d).map_err(error)).collect::<Result<Vec<_>>>()?;
    TensorLayout::contiguous(&shape,dtype)
}
pub(super) fn layout(value:&TensorBuffer)->Result<TensorLayout> {
    let layout=layout_parts(&value.shape,&value.strides,value.dtype)?;
    if value.handle.size_in_used()<layout.byte_len() as u64 {return Err(error("typed tensor buffer is shorter than its layout"));}
    Ok(layout)
}
pub(super) fn allocate(client:&Client,layout:&TensorLayout)->TensorBuffer {
    let dtype=match layout.dtype() {CannDType::F32=>DType::F32,CannDType::F16=>DType::F16,CannDType::BF16=>DType::BF16,
        CannDType::I32=>DType::I32,CannDType::I64=>DType::I64,_=>unreachable!("checked storage dtype")};
    TensorBuffer {handle:client.empty(layout.byte_len()),shape:Shape::from(layout.shape().iter().map(|&n|n as usize).collect::<Vec<_>>()),
        strides:Strides::from(layout.strides().iter().map(|&n|n as usize).collect::<Vec<_>>()),dtype}
}
fn launch(client:&Client,tensors:[&TensorBuffer;3],layouts:[TensorLayout;3],backward:Option<EmbeddingOptions>)->Result<()> {
    client.flush().map_err(error)?;
    let guards=tensors.iter().map(|tensor|client.get_resource(tensor.handle.clone()).map_err(error)).collect::<Result<Vec<_>>>()?;
    let resources=guards.iter().zip(&layouts).map(|(guard,layout)| {
        let mut resource=guard.resource().clone();if resource.byte_len()<layout.byte_len() {return Err(error("embedding resource is too short"));}
        resource.size=layout.byte_len();Ok(resource)
    }).collect::<Result<Vec<_>>>()?.try_into().map_err(|_|error("embedding binding count mismatch"))?;
    let worker=&WORKER.get().ok_or_else(||error("Ascend runtime is not initialized"))?.1;
    let result=worker.call(move|state|state.embedding(layouts,resources,backward));drop(guards);result
}
pub(super) fn embedding(client:&Client,weight:TensorBuffer,indices:TensorBuffer)->Result<TensorBuffer> {
    let weight_layout=layout(&weight)?;let indices_layout=layout(&indices)?;
    let target=forward_layout(&weight_layout,&indices_layout)?;let out=allocate(client,&target);
    launch(client,[&weight,&indices,&out],[weight_layout,indices_layout,target],None)?;Ok(out)
}
pub(super) fn embedding_with_saved_indices(client:&Client,weight:TensorBuffer,indices:TensorBuffer)->Result<[TensorBuffer;2]> {
    let indices_layout=layout(&indices)?;
    // Validate the full lookup before allocating or submitting its saved-index copy.
    forward_layout(&layout(&weight)?,&indices_layout)?;
    let saved=copy_contiguous(client,indices)?;
    let output=embedding(client,weight,saved.clone())?;Ok([output,saved])
}
pub(super) fn copy_contiguous(client:&Client,input:TensorBuffer)->Result<TensorBuffer> {
    let input_layout=layout(&input)?;
    let saved=allocate(client,&input_layout);client.flush().map_err(error)?;
    if input_layout.byte_len()!=0 {
        let guards=[&input,&saved].iter().map(|value|client.get_resource(value.handle.clone()).map_err(error)).collect::<Result<Vec<_>>>()?;
        let resources=guards.iter().map(|guard| {
            let mut resource=guard.resource().clone();
            if resource.byte_len()<input_layout.byte_len() {return Err(error("snapshot resource is too short"));}
            resource.size=input_layout.byte_len();Ok(resource)
        }).collect::<Result<Vec<_>>>()?.try_into().map_err(|_|error("snapshot binding count mismatch"))?;
        let worker=&WORKER.get().ok_or_else(||error("Ascend runtime is not initialized"))?.1;
        let result=worker.call(move|state|state.copy_contiguous(input_layout,resources));drop(guards);result?;
    }
    Ok(saved)
}
pub(super) fn embedding_backward(client:&Client,grad:TensorBuffer,indices:TensorBuffer,num_weights:u64,options:EmbeddingOptions)->Result<TensorBuffer> {
    let grad_layout=layout(&grad)?;let indices_layout=layout(&indices)?;
    let target=backward_layout(&grad_layout,&indices_layout,num_weights,options)?;let out=allocate(client,&target);
    launch(client,[&grad,&indices,&out],[grad_layout,indices_layout,target],Some(options))?;Ok(out)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn runtime_embedding_layout_keeps_integer_ids_without_fp32_conversion() {
        assert_eq!(layout_parts(&[2,3],&[3,1],DType::I32).unwrap().byte_len(),24);
        assert_eq!(layout_parts(&[2,3],&[3,1],DType::I64).unwrap().byte_len(),48);
        assert!(layout_parts(&[2,3],&[1,2],DType::I32).is_err());
        assert!(layout_parts(&[usize::MAX,3],&[3,1],DType::F32).is_err());
    }
}

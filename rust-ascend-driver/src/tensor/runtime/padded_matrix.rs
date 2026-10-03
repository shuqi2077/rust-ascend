use super::{AscendRuntime,ComputeClient,Result,TensorBuffer,WORKER,error,matrix};
use crate::tensor::{DType as CannDType,TensorLayout,deepgemm::{PaddedGemmSpec,Transpose}};
use ruda_core::tensor::{DType,Shape,Strides};
type Client=ComputeClient<AscendRuntime>;

pub(super) fn prefix_padding(source:&TensorLayout,target:&TensorLayout)->Result<Vec<i64>> {
    if source.shape().len()!=target.shape().len() || !(1..=8).contains(&source.shape().len())
        || source.dtype()!=target.dtype() || !matches!(source.dtype(),CannDType::F32|CannDType::BF16)
        || source.shape().iter().chain(target.shape()).any(|&size|size<=0) {
        return Err(error("prefix resize requires positive matching-rank FP32/BF16 layouts of the same dtype"));
    }
    // ACLNN pairs start with the final axis. Positive end padding adds zero;
    // negative end padding crops without doing arithmetic on retained values.
    Ok(source.shape().iter().zip(target.shape()).rev().flat_map(|(&a,&b)|[0,b-a]).collect())
}
fn allocate(client:&Client,layout:&TensorLayout)->TensorBuffer {
    TensorBuffer {handle:client.empty(layout.byte_len()),shape:Shape::from(layout.shape().iter().map(|&d|d as usize).collect::<Vec<_>>()),
        strides:Strides::from(layout.strides().iter().map(|&d|d as usize).collect::<Vec<_>>()),
        dtype:if layout.dtype()==CannDType::BF16 {DType::BF16} else {DType::F32}}
}
fn resize(client:&Client,input:TensorBuffer,target:&TensorLayout)->Result<TensorBuffer> {
    let source=matrix::layout(&input)?;prefix_padding(&source,target)?;
    if source==*target {return Ok(input);}
    let output=allocate(client,target);let layouts=[source,target.clone()];
    client.flush().map_err(error)?;
    let guards=[&input,&output].iter().map(|t|client.get_resource(t.handle.clone()).map_err(error)).collect::<Result<Vec<_>>>()?;
    let resources=guards.iter().zip(&layouts).map(|(guard,layout)| {
        let mut resource=guard.resource().clone();if resource.byte_len()<layout.byte_len() {return Err(error("prefix resize resource too short"));}
        resource.size=layout.byte_len();Ok(resource)
    }).collect::<Result<Vec<_>>>()?.try_into().map_err(|_|error("prefix resize binding count mismatch"))?;
    let worker=&WORKER.get().ok_or_else(||error("Ascend runtime is not initialized"))?.1;
    let result=worker.call(move |state|state.resize_prefix(layouts,resources));drop(guards);result?;Ok(output)
}
fn rounded(client:&Client,input:TensorBuffer,target:&TensorLayout)->Result<TensorBuffer> {
    let input=super::conversion::cast(client,input,DType::BF16)?;resize(client,input,target)
}
fn logical(shape:&Shape,dtype:CannDType)->Result<TensorLayout> {
    TensorLayout::contiguous(&shape.iter().map(|&d|i64::try_from(d).map_err(error)).collect::<Result<Vec<_>>>()?,dtype)
}
pub(super) fn forward(client:&Client,a:TensorBuffer,b:TensorBuffer,ta:Transpose,tb:Transpose)->Result<[TensorBuffer;3]> {
    let a_layout=matrix::layout(&a)?;let b_layout=matrix::layout(&b)?;
    if a.dtype!=DType::F32 || b.dtype!=DType::F32 {return Err(error("padded trainable matmul requires FP32 inputs"));}
    let spec=PaddedGemmSpec::new(&a_layout,&b_layout,ta,tb)?;
    let a=rounded(client,a,&spec.native.a)?;let b=rounded(client,b,&spec.native.b)?;
    let output=matrix::gemm(client,a.clone(),b.clone(),ta,tb,DType::F32)?;
    Ok([resize(client,output,&spec.output)?,a,b])
}
pub(super) fn backward(client:&Client,a:TensorBuffer,b:TensorBuffer,grad:TensorBuffer,
    a_shape:Shape,b_shape:Shape,ta:Transpose,tb:Transpose)->Result<[TensorBuffer;2]> {
    let x=logical(&a_shape,CannDType::F32)?;let w=logical(&b_shape,CannDType::F32)?;
    let spec=PaddedGemmSpec::new(&x,&w,ta,tb)?;
    if matrix::layout(&a)?!=spec.native.a || matrix::layout(&b)?!=spec.native.b || matrix::layout(&grad)?!=spec.output {
        return Err(error("padded matmul backward saved input or logical dY layout mismatch"));
    }
    let target=TensorLayout::contiguous(spec.native.output_layout().shape(),CannDType::BF16)?;
    let grad=rounded(client,grad,&target)?;
    let [da_spec,db_spec]=spec.backward_specs()?;
    let da=allocate(client,da_spec.output_layout());let db=allocate(client,db_spec.output_layout());
    let (da_a,da_b)=if ta==Transpose::No {(&grad,&b)} else {(&b,&grad)};
    let (db_a,db_b)=if tb==Transpose::No {(&a,&grad)} else {(&grad,&a)};
    matrix::launch(client,da_spec,da_a,da_b,&da)?;matrix::launch(client,db_spec,db_a,db_b,&db)?;
    Ok([resize(client,da,&x)?,resize(client,db,&w)?])
}
pub(super) fn frozen_forward(client:&Client,input:TensorBuffer,weight:TensorBuffer)->Result<[TensorBuffer;2]> {
    let x=matrix::layout(&input)?;let w=matrix::layout(&weight)?;
    if x.shape().len()!=2 || w.shape().len()!=2 || input.dtype!=DType::F32 || weight.dtype!=DType::BF16 {
        return Err(error("frozen padded linear requires rank-2 FP32 input and fixed BF16 weight"));
    }
    let spec=PaddedGemmSpec::new(&x,&w,Transpose::No,Transpose::Yes)?;
    let input=rounded(client,input,&spec.native.a)?;
    // Aligned weights reuse their actual handle; only a tail requires a new
    // padded BF16 allocation. No full-weight FP32 expansion is performed.
    let weight=resize(client,weight,&spec.native.b)?;
    let output=matrix::gemm(client,input,weight.clone(),Transpose::No,Transpose::Yes,DType::F32)?;
    Ok([resize(client,output,&spec.output)?,weight])
}
pub(super) fn frozen_backward(client:&Client,weight:TensorBuffer,grad:TensorBuffer,input_shape:Shape,weight_shape:Shape)->Result<TensorBuffer> {
    let input=logical(&input_shape,CannDType::F32)?;let w=logical(&weight_shape,CannDType::BF16)?;
    if input.shape().len()!=2 || w.shape().len()!=2 {return Err(error("frozen padded linear backward requires rank-2 shapes"));}
    let spec=PaddedGemmSpec::new(&input,&w,Transpose::No,Transpose::Yes)?;
    if matrix::layout(&weight)?!=spec.native.b || matrix::layout(&grad)?!=spec.output {return Err(error("frozen padded linear saved weight or dY layout mismatch"));}
    let target=TensorLayout::contiguous(spec.native.output_layout().shape(),CannDType::BF16)?;
    let grad=rounded(client,grad,&target)?;
    let output=matrix::gemm(client,grad,weight,Transpose::No,Transpose::No,DType::F32)?;
    resize(client,output,&input)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pad_and_crop_use_reverse_axis_pairs_and_preserve_storage_dtype() {
        for dtype in [CannDType::F32,CannDType::BF16] {
            let x=TensorLayout::contiguous(&[2,3,17],dtype).unwrap();let y=TensorLayout::contiguous(&[2,16,32],dtype).unwrap();
            assert_eq!(prefix_padding(&x,&y).unwrap(),[0,15,0,13,0,0]);
            assert_eq!(prefix_padding(&y,&x).unwrap(),[0,-15,0,-13,0,0]);
            assert_eq!(prefix_padding(&x,&x).unwrap(),[0,0,0,0,0,0]);
            assert_eq!(x.byte_len(),2*3*17*dtype.bytes());
        }
        let t=|s:&[i64],d|TensorLayout::contiguous(s,d).unwrap();
        assert!(prefix_padding(&t(&[3,7],CannDType::BF16),&t(&[16,16],CannDType::F32)).is_err());
        assert!(prefix_padding(&t(&[3,7],CannDType::F32),&t(&[0,16],CannDType::F32)).is_err());
        assert!(prefix_padding(&t(&[3,7],CannDType::F32),&t(&[16],CannDType::F32)).is_err());
    }
}

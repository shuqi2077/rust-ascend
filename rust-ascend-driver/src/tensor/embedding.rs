use super::*;

/// Dense embedding gradients: no padding row or frequency scaling unless requested.
#[derive(Clone,Copy,Debug,Default,PartialEq,Eq)]
pub struct EmbeddingOptions {
    pub padding_idx:Option<usize>,
    pub scale_grad_by_freq:bool,
}
impl EmbeddingOptions {
    pub(super) fn padding(self,rows:u64)->Result<u64,CannError> {
        match self.padding_idx {
            Some(index) if (index as u64)<rows=>Ok(index as u64),
            Some(_)=>Err(invalid("embedding padding index must be within the weight rows")),
            // The SDK's unsigned ABI carries the documented negative no-padding sentinel.
            None=>Ok(u64::MAX),
        }
    }
}
pub(super) type EmbeddingPlan=unsafe extern "C" fn(*const AclTensor,*const AclTensor,*const AclTensor,*mut u64,*mut *mut AclOpExecutor)->Status;
pub(super) type EmbeddingGradPlan=unsafe extern "C" fn(*const AclTensor,*const AclTensor,u64,u64,bool,*const AclTensor,*mut u64,*mut *mut AclOpExecutor)->Status;
fn indices_layout(indices:&TensorLayout)->Result<(),CannError> {
    if !(1..=7).contains(&indices.shape().len()) || !matches!(indices.dtype(),DType::I32|DType::I64) {
        return Err(invalid("embedding indices require contiguous rank 1..7 INT32/INT64 storage"));
    }
    Ok(())
}
pub(super) fn forward_layout(weight:&TensorLayout,indices:&TensorLayout)->Result<TensorLayout,CannError> {
    indices_layout(indices)?;
    if weight.shape().len()!=2 || !matches!(weight.dtype(),DType::F32|DType::F16|DType::BF16)
        || (weight.shape()[0]==0 && indices.byte_len()!=0) {
        return Err(invalid("embedding weight requires a rank-2 FP32/FP16/BF16 table; nonempty indices require weight rows"));
    }
    let mut shape=indices.shape().to_vec();shape.push(weight.shape()[1]);TensorLayout::contiguous(&shape,weight.dtype())
}
pub(super) fn backward_layout(grad:&TensorLayout,indices:&TensorLayout,num_weights:u64,options:EmbeddingOptions)->Result<TensorLayout,CannError> {
    indices_layout(indices)?;
    let rows=i64::try_from(num_weights).map_err(|_|invalid("embedding weight row count exceeds i64"))?;
    options.padding(num_weights)?;
    if grad.shape().len()!=indices.shape().len()+1 || grad.shape()[..indices.shape().len()]!=*indices.shape()
        || !matches!(grad.dtype(),DType::F32|DType::F16|DType::BF16) || (rows==0 && indices.byte_len()!=0) {
        return Err(invalid("embedding gradient must have the index shape followed by the embedding width and floating dtype"));
    }
    TensorLayout::contiguous(&[rows,*grad.shape().last().unwrap()],grad.dtype())
}
impl CannSession {
    /// ACLNN lookup. Index values must be in [0, weight rows); no host index transfer.
    pub fn embedding(self:&Rc<Self>,weight:&CannTensor,indices:&CannTensor)->Result<CannTensor,CannError> {
        self.same_session(&[weight,indices])?;
        let layout=forward_layout(weight.layout(),indices.layout())?;
        let out=self.allocate_tensor(layout.shape(),layout.dtype())?;
        if layout.byte_len()==0 {return Ok(out);}
        // SAFETY: exact CANN signatures; owned descriptors/storage outlive synchronized execution.
        unsafe {
            let plan:EmbeddingPlan=self.ops.get(c"aclnnEmbeddingGetWorkspaceSize")?;
            let run=self.ops.get(c"aclnnEmbedding")?;
            self.execute("aclnnEmbedding",run,|size,executor|plan(weight.descriptor.as_ptr(),indices.descriptor.as_ptr(),out.descriptor.as_ptr(),size,executor))?;
        }
        Ok(out)
    }
    /// ACLNN dense weight gradient, accumulating repeated IDs and optionally scaling by frequency.
    /// Padding suppresses only that row's derivative, not the forward lookup values.
    pub fn embedding_backward(self:&Rc<Self>,grad:&CannTensor,indices:&CannTensor,num_weights:u64,
        options:EmbeddingOptions)->Result<CannTensor,CannError> {
        self.same_session(&[grad,indices])?;
        let layout=backward_layout(grad.layout(),indices.layout(),num_weights,options)?;
        let out=self.zeros(layout.shape(),layout.dtype())?;
        if grad.layout().byte_len()==0 || layout.byte_len()==0 {return Ok(out);}
        let padding=options.padding(num_weights)?;
        // SAFETY: fixed SDK ABI and synchronized executor; dense output starts at zero.
        unsafe {
            let plan:EmbeddingGradPlan=self.ops.get(c"aclnnEmbeddingDenseBackwardGetWorkspaceSize")?;
            let run=self.ops.get(c"aclnnEmbeddingDenseBackward")?;
            self.execute("aclnnEmbeddingDenseBackward",run,|size,executor|plan(grad.descriptor.as_ptr(),indices.descriptor.as_ptr(),num_weights,padding,
                options.scale_grad_by_freq,out.descriptor.as_ptr(),size,executor))?;
        }
        Ok(out)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn layout(shape:&[i64],dtype:DType)->TensorLayout {TensorLayout::contiguous(shape,dtype).unwrap()}
    #[test]
    fn lookup_and_backward_layouts_preserve_integer_indices_and_storage_dtype() {
        for dtype in [DType::F32,DType::F16,DType::BF16] {for integer in [DType::I32,DType::I64] {
            let weights=layout(&[13,65],dtype);let indices=layout(&[2,3],integer);
            let output=forward_layout(&weights,&indices).unwrap();assert_eq!(output.shape(),&[2,3,65]);assert_eq!(output.dtype(),dtype);
            assert_eq!(backward_layout(&output,&indices,13,EmbeddingOptions::default()).unwrap(),weights);
            let empty=layout(&[0,3],integer);let output=forward_layout(&weights,&empty).unwrap();assert_eq!(output.byte_len(),0);
            assert_eq!(backward_layout(&output,&empty,13,EmbeddingOptions::default()).unwrap(),weights);
        }}
        assert!(forward_layout(&layout(&[3,65],DType::F32),&layout(&[2],DType::F32)).is_err());
        assert!(forward_layout(&layout(&[0,65],DType::F32),&layout(&[2],DType::I32)).is_err());
        assert!(backward_layout(&layout(&[3,2,65],DType::F32),&layout(&[2,3],DType::I32),13,EmbeddingOptions::default()).is_err());
        assert!(EmbeddingOptions {padding_idx:Some(13),scale_grad_by_freq:false}.padding(13).is_err());
        assert_eq!(EmbeddingOptions::default().padding(13).unwrap(),u64::MAX);
        assert_eq!(EmbeddingOptions {padding_idx:Some(0),scale_grad_by_freq:true}.padding(13).unwrap(),0);
    }
}

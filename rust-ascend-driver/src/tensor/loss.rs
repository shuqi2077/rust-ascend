use super::*;

/// Explicit loss reduction; weighted mean divides by the non-ignored target weight sum.
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
#[repr(i64)]
pub enum LossReduction {None=0,Mean=1,Sum=2}
/// No implicit ignore label or reduction is selected.
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub struct NllLossOptions {pub reduction:LossReduction,pub ignore_index:Option<i64>}
impl NllLossOptions {
    pub(super) fn ignore(self)->i64 {self.ignore_index.unwrap_or(-1)}
}
pub(super) type NllPlan=unsafe extern "C" fn(*const AclTensor,*const AclTensor,*const AclTensor,i64,i64,
    *mut AclTensor,*mut AclTensor,*mut u64,*mut *mut AclOpExecutor)->Status;
pub(super) type NllGradPlan=unsafe extern "C" fn(*const AclTensor,*const AclTensor,*const AclTensor,*const AclTensor,
    i64,i64,*const AclTensor,*mut AclTensor,*mut u64,*mut *mut AclOpExecutor)->Status;
pub(super) fn forward_layouts(input:&TensorLayout,target:&TensorLayout,weight:&TensorLayout,
    options:NllLossOptions)->Result<[TensorLayout;2],CannError> {
    if input.shape().len()!=2 || input.shape()[1]<=0 || !matches!(input.dtype(),DType::F32|DType::F16|DType::BF16)
        || target.shape()!=&[input.shape()[0]] || !matches!(target.dtype(),DType::I32|DType::I64)
        || weight.shape()!=&[input.shape()[1]] || weight.dtype()!=input.dtype() {
        return Err(invalid("NLLLoss requires floating input[N,C], matching integer target[N] and same-dtype class weight[C]"));
    }
    let rows=if options.reduction==LossReduction::None {input.shape()[0]} else {1};
    Ok([TensorLayout::contiguous(&[rows],input.dtype())?,TensorLayout::contiguous(&[1],input.dtype())?])
}
pub(super) fn backward_layout(grad:&TensorLayout,input:&TensorLayout,target:&TensorLayout,weight:&TensorLayout,
    total_weight:&TensorLayout,options:NllLossOptions)->Result<TensorLayout,CannError> {
    let [expected_grad,expected_total]=forward_layouts(input,target,weight,options)?;
    if grad!=&expected_grad || total_weight!=&expected_total {
        return Err(invalid("NLLLoss upstream gradient or saved total weight does not match forward reduction/dtype"));
    }
    Ok(input.clone())
}
impl CannSession {
    /// ACLNN NLLLoss from log-probabilities. Returns [loss, device total weight].
    /// Total weight is valid only for Mean/Sum; targets must be valid classes or the explicit ignore label.
    pub fn nll_loss(self:&Rc<Self>,input:&CannTensor,target:&CannTensor,weight:&CannTensor,
        options:NllLossOptions)->Result<[CannTensor;2],CannError> {
        self.same_session(&[input,target,weight])?;
        let [out_layout,total_layout]=forward_layouts(input.layout(),target.layout(),weight.layout(),options)?;
        let out=self.zeros(out_layout.shape(),out_layout.dtype())?;
        let total=self.zeros(total_layout.shape(),total_layout.dtype())?;
        // SAFETY: fixed SDK ABI, owned tensors and synchronized executor/workspace lifetimes.
        unsafe {
            let plan:NllPlan=self.ops.get(c"aclnnNLLLossGetWorkspaceSize")?;let run=self.ops.get(c"aclnnNLLLoss")?;
            self.execute("aclnnNLLLoss",run,|size,executor|plan(input.descriptor.as_ptr(),target.descriptor.as_ptr(),weight.descriptor.as_ptr(),
                options.reduction as i64,options.ignore(),out.descriptor.as_ptr(),total.descriptor.as_ptr(),size,executor))?;
        }
        Ok([out,total])
    }
    /// Dense log-probability derivative, using the forward total weight for weighted Mean.
    pub fn nll_loss_backward(self:&Rc<Self>,grad:&CannTensor,input:&CannTensor,target:&CannTensor,
        weight:&CannTensor,total_weight:&CannTensor,options:NllLossOptions)->Result<CannTensor,CannError> {
        self.same_session(&[grad,input,target,weight,total_weight])?;
        let out_layout=backward_layout(grad.layout(),input.layout(),target.layout(),weight.layout(),total_weight.layout(),options)?;
        let out=self.zeros(out_layout.shape(),out_layout.dtype())?;
        if out_layout.byte_len()==0 {return Ok(out);}
        // SAFETY: checked dense output, exact SDK signature and synchronized executor.
        unsafe {
            let plan:NllGradPlan=self.ops.get(c"aclnnNLLLossBackwardGetWorkspaceSize")?;let run=self.ops.get(c"aclnnNLLLossBackward")?;
            self.execute("aclnnNLLLossBackward",run,|size,executor|plan(grad.descriptor.as_ptr(),input.descriptor.as_ptr(),target.descriptor.as_ptr(),
                weight.descriptor.as_ptr(),options.reduction as i64,options.ignore(),total_weight.descriptor.as_ptr(),out.descriptor.as_ptr(),size,executor))?;
        }
        Ok(out)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn layout(shape:&[i64],dtype:DType)->TensorLayout {TensorLayout::contiguous(shape,dtype).unwrap()}
    #[test]
    fn nll_layouts_preserve_integer_labels_dtype_and_all_reductions() {
        for dtype in [DType::F32,DType::F16,DType::BF16] {for integer in [DType::I32,DType::I64] {
            for rows in [0,3] {for reduction in [LossReduction::None,LossReduction::Mean,LossReduction::Sum] {
                let input=layout(&[rows,65],dtype);let target=layout(&[rows],integer);let weight=layout(&[65],dtype);
                let options=NllLossOptions {reduction,ignore_index:Some(-100)};
                let [loss,total]=forward_layouts(&input,&target,&weight,options).unwrap();
                assert_eq!(loss.shape(),&[if reduction==LossReduction::None {rows} else {1}]);assert_eq!(total.shape(),&[1]);
                assert_eq!(backward_layout(&loss,&input,&target,&weight,&total,options).unwrap(),input);
            }}
        }}
        assert_eq!(NllLossOptions {reduction:LossReduction::Sum,ignore_index:None}.ignore(),-1);
    }
    #[test]
    fn nll_layouts_reject_invalid_labels_weights_and_upstream_contracts() {
        let x=layout(&[3,65],DType::F32);let labels=layout(&[3],DType::I64);let weight=layout(&[65],DType::F32);
        let options=NllLossOptions {reduction:LossReduction::Mean,ignore_index:None};
        assert!(forward_layouts(&x,&layout(&[3],DType::F32),&weight,options).is_err());
        assert!(forward_layouts(&x,&layout(&[3,1],DType::I64),&weight,options).is_err());
        assert!(forward_layouts(&x,&labels,&layout(&[64],DType::F32),options).is_err());
        assert!(forward_layouts(&x,&labels,&layout(&[65],DType::BF16),options).is_err());
        assert!(forward_layouts(&layout(&[3,0],DType::F32),&labels,&layout(&[0],DType::F32),options).is_err());
        assert!(backward_layout(&layout(&[3],DType::F32),&x,&labels,&weight,&layout(&[1],DType::F32),options).is_err());
        assert!(backward_layout(&layout(&[1],DType::F32),&x,&labels,&weight,&layout(&[1],DType::BF16),options).is_err());
    }
}

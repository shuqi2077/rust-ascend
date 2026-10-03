use super::*;
use std::ffi::CStr;

pub(super) type CastPlan = unsafe extern "C" fn(
    *const AclTensor, i32, *mut AclTensor, *mut u64, *mut *mut AclOpExecutor,
) -> Status;
type SiluGradPlan = unsafe extern "C" fn(
    *const AclTensor, *const AclTensor, *mut AclTensor, *mut u64, *mut *mut AclOpExecutor,
) -> Status;
type SoftmaxGradPlan = unsafe extern "C" fn(
    *const AclTensor, *const AclTensor, i64, *mut AclTensor,
    *mut u64, *mut *mut AclOpExecutor,
) -> Status;
type RmsGradPlan = unsafe extern "C" fn(
    *const AclTensor, *const AclTensor, *const AclTensor, *const AclTensor,
    *const AclTensor, *const AclTensor, *mut u64, *mut *mut AclOpExecutor,
) -> Status;

fn float_layout(layout: &TensorLayout) -> Result<(), CannError> {
    if !(1..=8).contains(&layout.shape().len())
        || !matches!(layout.dtype(), DType::F32 | DType::F16 | DType::BF16)
    {
        return Err(invalid("gradient operators require rank 1..8 FP32/FP16/BF16 tensors"));
    }
    Ok(())
}

fn matching_gradient(input: &TensorLayout, grad: &TensorLayout) -> Result<(), CannError> {
    float_layout(input)?;
    if input.shape() != grad.shape() || input.dtype() != grad.dtype() {
        return Err(invalid("gradient shape and dtype must match the forward tensor"));
    }
    Ok(())
}

fn rms_gradient_layouts(
    input: &TensorLayout, gamma: &TensorLayout, grad: &TensorLayout, rstd: &TensorLayout,
) -> Result<(), CannError> {
    matching_gradient(input, grad)?;
    float_layout(gamma)?;
    let rank = input.shape().len();
    let normalized = gamma.shape().len();
    if normalized > rank || input.shape()[rank - normalized..] != *gamma.shape() {
        return Err(invalid("RMSNorm gamma must match trailing input dimensions"));
    }
    let mut retained = input.shape().to_vec();
    retained[rank - normalized..].fill(1);
    if rstd.dtype() != DType::F32
        || (rstd.shape() != retained && rstd.shape() != &input.shape()[..rank - normalized])
    {
        return Err(invalid("RMSNorm rstd must contain one FP32 value per normalized group"));
    }
    Ok(())
}

impl CannSession {
    /// Explicit device-side conversion. This does not copy input values to the host.
    pub fn cast(self: &Rc<Self>, input: &CannTensor, dtype: DType) -> Result<CannTensor, CannError> {
        self.same_session(&[input])?;
        let output = self.allocate_tensor(input.layout.shape(), dtype)?;
        // SAFETY: CANN aclnnCast signatures; owned descriptors and buffers remain alive.
        unsafe {
            let plan: CastPlan = self.ops.get(c"aclnnCastGetWorkspaceSize")?;
            let run = self.ops.get(c"aclnnCast")?;
            self.execute("aclnnCast", run, |size, executor| {
                plan(input.descriptor.as_ptr(), dtype as i32, output.descriptor.as_ptr(), size, executor)
            })?;
        }
        Ok(output)
    }

    /// SiLU backward through ACLNN, preserving FP32/FP16/BF16 input dtype.
    pub fn silu_backward(
        self: &Rc<Self>, input: &CannTensor, grad_output: &CannTensor,
    ) -> Result<CannTensor, CannError> {
        matching_gradient(input.layout(), grad_output.layout())?;
        self.same_session(&[input, grad_output])?;
        let grad_input = self.allocate_tensor(input.layout.shape(), input.layout.dtype())?;
        // SAFETY: exact ACLNN SiLU backward ABI; all resources live through synchronization.
        unsafe {
            let plan: SiluGradPlan = self.ops.get(c"aclnnSiluBackwardGetWorkspaceSize")?;
            let run = self.ops.get(c"aclnnSiluBackward")?;
            self.execute("aclnnSiluBackward", run, |size, executor| {
                plan(grad_output.descriptor.as_ptr(), input.descriptor.as_ptr(),
                    grad_input.descriptor.as_ptr(), size, executor)
            })?;
        }
        Ok(grad_input)
    }

    /// LogSoftmax backward on an explicit axis, through ACLNN.
    pub fn log_softmax_backward(
        self: &Rc<Self>, output: &CannTensor, grad_output: &CannTensor, dim: i64,
    ) -> Result<CannTensor, CannError> {
        self.softmax_gradient(output, grad_output, dim, c"aclnnLogSoftmaxBackwardGetWorkspaceSize",
            c"aclnnLogSoftmaxBackward", "aclnnLogSoftmaxBackward")
    }

    /// Softmax backward on an explicit axis, through ACLNN.
    pub fn softmax_backward(
        self: &Rc<Self>, output: &CannTensor, grad_output: &CannTensor, dim: i64,
    ) -> Result<CannTensor, CannError> {
        self.softmax_gradient(output, grad_output, dim, c"aclnnSoftmaxBackwardGetWorkspaceSize",
            c"aclnnSoftmaxBackward", "aclnnSoftmaxBackward")
    }

    fn softmax_gradient(
        self: &Rc<Self>, output: &CannTensor, grad_output: &CannTensor, dim: i64,
        plan_name: &CStr, run_name: &CStr, operation: &'static str,
    ) -> Result<CannTensor, CannError> {
        matching_gradient(output.layout(), grad_output.layout())?;
        let axis = layout::axis(dim, output.layout.shape.len())? as i64;
        self.same_session(&[output, grad_output])?;
        let grad_input = self.allocate_tensor(output.layout.shape(), output.layout.dtype())?;
        // SAFETY: both private call sites use the same ACLNN gradient ABI.
        unsafe {
            let plan: SoftmaxGradPlan = self.ops.get(plan_name)?;
            let run = self.ops.get(run_name)?;
            self.execute(operation, run, |size, executor| {
                plan(grad_output.descriptor.as_ptr(), output.descriptor.as_ptr(), axis,
                    grad_input.descriptor.as_ptr(), size, executor)
            })?;
        }
        Ok(grad_input)
    }

    /// ACLNN RMSNorm backward, returning (input gradient, FP32 gamma gradient).
    /// `rstd` is the reciprocal RMS returned by the forward pass; no epsilon is recomputed.
    pub fn rms_norm_backward(
        self: &Rc<Self>, input: &CannTensor, gamma: &CannTensor,
        grad_output: &CannTensor, rstd: &CannTensor,
    ) -> Result<(CannTensor, CannTensor), CannError> {
        rms_gradient_layouts(input.layout(), gamma.layout(), grad_output.layout(), rstd.layout())?;
        self.same_session(&[input, gamma, grad_output, rstd])?;
        let dx = self.allocate_tensor(input.layout.shape(), input.layout.dtype())?;
        let dgamma = self.allocate_tensor(gamma.layout.shape(), DType::F32)?;
        // SAFETY: CANN declares const descriptor pointers for both writable outputs.
        unsafe {
            let plan: RmsGradPlan = self.ops.get(c"aclnnRmsNormGradGetWorkspaceSize")?;
            let run = self.ops.get(c"aclnnRmsNormGrad")?;
            self.execute("aclnnRmsNormGrad", run, |size, executor| {
                plan(grad_output.descriptor.as_ptr(), input.descriptor.as_ptr(),
                    rstd.descriptor.as_ptr(), gamma.descriptor.as_ptr(),
                    dx.descriptor.as_ptr(), dgamma.descriptor.as_ptr(), size, executor)
            })?;
        }
        Ok((dx, dgamma))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tensor(shape: &[i64], dtype: DType) -> TensorLayout {
        TensorLayout::contiguous(shape, dtype).unwrap()
    }
    #[test]
    fn rms_gradient_supports_three_precisions_and_multiaxis_weights() {
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let x = tensor(&[2, 3, 4], dtype);
            let gamma = tensor(&[3, 4], dtype);
            for shape in [&[2, 1, 1][..], &[2][..]] {
                assert!(rms_gradient_layouts(&x, &gamma, &x, &tensor(shape, DType::F32)).is_ok());
            }
        }
    }
    #[test]
    fn gradient_contract_rejects_wrong_shapes_and_types() {
        let x = tensor(&[2, 3, 4], DType::BF16);
        let gamma = tensor(&[4], DType::BF16);
        let rstd = tensor(&[2, 3, 1], DType::F32);
        assert!(rms_gradient_layouts(&x, &gamma, &tensor(&[2, 3, 4], DType::F16), &rstd).is_err());
        assert!(rms_gradient_layouts(&x, &tensor(&[3], DType::BF16), &x, &rstd).is_err());
        assert!(rms_gradient_layouts(&x, &gamma, &x, &tensor(&[2, 3, 1], DType::BF16)).is_err());
        assert!(rms_gradient_layouts(&x, &gamma, &x, &tensor(&[2, 2, 1], DType::F32)).is_err());
        assert!(matching_gradient(&tensor(&[], DType::F32), &tensor(&[], DType::F32)).is_err());
        assert!(matching_gradient(&tensor(&[2], DType::I32), &tensor(&[2], DType::I32)).is_err());
    }
}

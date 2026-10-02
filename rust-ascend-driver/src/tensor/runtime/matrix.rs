use super::{AscendRuntime, ComputeClient, Result, TensorBuffer, WORKER, contiguous, error};
use crate::tensor::{DType as CannDType, TensorLayout, deepgemm::{GemmKind, GemmSpec, Transpose}};
use ruda_core::tensor::{DType, Shape, Strides};

type Client = ComputeClient<AscendRuntime>;

fn layout_parts(shape: &[usize], strides: &[usize], dtype: DType) -> Result<TensorLayout> {
    let dtype = match dtype { DType::BF16 => CannDType::BF16, DType::F32 => CannDType::F32,
        _ => return Err(error("native matrix buffers require BF16 or FP32 dtype")) };
    if !contiguous(shape, strides) { return Err(error("native matrix buffers must be contiguous")); }
    let shape = shape.iter().map(|&d| i64::try_from(d).map_err(error)).collect::<Result<Vec<_>>>()?;
    TensorLayout::contiguous(&shape, dtype)
}
fn layout(tensor: &TensorBuffer) -> Result<TensorLayout> {
    let layout = layout_parts(&tensor.shape, &tensor.strides, tensor.dtype)?;
    if tensor.handle.size_in_used() < layout.byte_len() as u64 {
        return Err(error("native matrix buffer is shorter than its layout"));
    }
    Ok(layout)
}
fn output_dtype(dtype: DType) -> Result<CannDType> {
    match dtype { DType::BF16 => Ok(CannDType::BF16), DType::F32 => Ok(CannDType::F32),
        _ => Err(error("native matrix output must be BF16 or FP32")) }
}
fn allocate(client: &Client, layout: &TensorLayout) -> TensorBuffer {
    TensorBuffer { handle: client.empty(layout.byte_len()),
        shape: Shape::from(layout.shape().iter().map(|&d| d as usize).collect::<Vec<_>>()),
        strides: Strides::from(layout.strides().iter().map(|&d| d as usize).collect::<Vec<_>>()),
        dtype: if layout.dtype()==CannDType::BF16 {DType::BF16} else {DType::F32} }
}
fn launch(client: &Client, spec: GemmSpec, a: &TensorBuffer, b: &TensorBuffer, out: &TensorBuffer) -> Result<()> {
    client.flush().map_err(error)?;
    let guards = [a,b,out].iter().map(|t| client.get_resource(t.handle.clone()).map_err(error))
        .collect::<Result<Vec<_>>>()?;
    let resources = guards.iter().zip([&spec.a,&spec.b,&spec.out]).map(|(guard, layout)| {
        let mut resource = guard.resource().clone();
        if resource.byte_len() < layout.byte_len() { return Err(error("native matrix resource is too short")); }
        resource.size = layout.byte_len();
        Ok(resource)
    }).collect::<Result<Vec<_>>>()?.try_into().map_err(|_| error("native matrix binding count mismatch"))?;
    let worker = &WORKER.get().ok_or_else(|| error("Ascend runtime is not initialized"))?.1;
    let result = worker.call(move |state| state.gemm(spec, resources));
    drop(guards);
    result
}

pub(super) fn gemm(client: &Client, a: TensorBuffer, b: TensorBuffer,
    ta: Transpose, tb: Transpose, dtype: DType) -> Result<TensorBuffer> {
    let spec = GemmSpec::new(&layout(&a)?, &layout(&b)?, ta, tb, output_dtype(dtype)?)?;
    let out = allocate(client, spec.output_layout());
    launch(client, spec, &a, &b, &out)?;
    Ok(out)
}
pub(super) fn gemm_into(client: &Client, a: TensorBuffer, b: TensorBuffer,
    ta: Transpose, tb: Transpose, out: TensorBuffer) -> Result<()> {
    let spec = GemmSpec::new(&layout(&a)?, &layout(&b)?, ta, tb, output_dtype(out.dtype)?)?;
    if layout(&out)? != *spec.output_layout() { return Err(error("native matrix output layout mismatch")); }
    launch(client, spec, &a, &b, &out)
}
pub(super) fn linear_nt_backward(client: &Client, input: TensorBuffer, weight: TensorBuffer,
    grad: TensorBuffer) -> Result<[TensorBuffer; 2]> {
    let x = layout(&input)?; let w = layout(&weight)?; let dy = layout(&grad)?;
    let forward = GemmSpec::new(&x,&w,Transpose::No,Transpose::Yes,CannDType::BF16)?;
    if forward.kind()!=GemmKind::Dense || dy!=*forward.output_layout() {
        return Err(error("linear backward requires rank-2 BF16 dY matching X @ W^T"));
    }
    let dx_spec = GemmSpec::new(&dy,&w,Transpose::No,Transpose::No,CannDType::BF16)?;
    let dw_spec = GemmSpec::new(&dy,&x,Transpose::Yes,Transpose::No,CannDType::F32)?;
    let dx = allocate(client, dx_spec.output_layout());
    let dw = allocate(client, dw_spec.output_layout());
    launch(client, dx_spec, &grad, &weight, &dx)?;
    launch(client, dw_spec, &grad, &input, &dw)?;
    Ok([dx,dw])
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn matrix_layout_preserves_dtype_and_checks_strides_and_overflow() {
        assert_eq!(layout_parts(&[2,32,48], &[1536,48,1], DType::BF16).unwrap().byte_len(), 6144);
        assert_eq!(layout_parts(&[32,64], &[64,1], DType::F32).unwrap().byte_len(), 8192);
        assert!(layout_parts(&[32,48], &[1,32], DType::BF16).is_err());
        assert!(layout_parts(&[32,48], &[48,1], DType::F16).is_err());
        assert!(layout_parts(&[usize::MAX,48], &[48,1], DType::BF16).is_err());
        assert!(output_dtype(DType::F16).is_err());
    }
}

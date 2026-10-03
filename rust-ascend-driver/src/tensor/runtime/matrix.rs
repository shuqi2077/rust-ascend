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

fn mixed_linear_spec(x: &TensorLayout, w: &TensorLayout) -> Result<GemmSpec> {
    if x.shape().len()!=2 || w.shape().len()!=2 || x.dtype()!=CannDType::F32 || w.dtype()!=CannDType::F32 {
        return Err(error("BF16-compute FP32 linear requires rank-2 FP32 input and weight"));
    }
    let x=TensorLayout::contiguous(x.shape(),CannDType::BF16)?;
    let w=TensorLayout::contiguous(w.shape(),CannDType::BF16)?;
    GemmSpec::new(&x,&w,Transpose::No,Transpose::Yes,CannDType::F32)
}

pub(super) fn linear_bf16_fp32(client: &Client, input: TensorBuffer, weight: TensorBuffer)
    -> Result<[TensorBuffer;3]> {
    let spec=mixed_linear_spec(&layout(&input)?,&layout(&weight)?)?;
    // Cast allocations are independent snapshots, including when FP32 parameters alias.
    let input=super::conversion::cast(client,input,DType::BF16)?;
    let weight=super::conversion::cast(client,weight,DType::BF16)?;
    let output=allocate(client,spec.output_layout());
    launch(client,spec,&input,&weight,&output)?;
    Ok([output,input,weight])
}

fn mixed_linear_backward_specs(x: &TensorLayout, w: &TensorLayout, dy: &TensorLayout)
    -> Result<[GemmSpec;2]> {
    let forward=GemmSpec::new(x,w,Transpose::No,Transpose::Yes,CannDType::F32)?;
    if forward.kind()!=GemmKind::Dense || dy!=forward.output_layout() {
        return Err(error("BF16-compute linear backward requires saved rank-2 BF16 X/W and matching FP32 dY"));
    }
    let dy=TensorLayout::contiguous(dy.shape(),CannDType::BF16)?;
    Ok([GemmSpec::new(&dy,w,Transpose::No,Transpose::No,CannDType::F32)?,
        GemmSpec::new(&dy,x,Transpose::Yes,Transpose::No,CannDType::F32)?])
}

pub(super) fn linear_bf16_fp32_backward(client: &Client, input: TensorBuffer, weight: TensorBuffer,
    grad: TensorBuffer) -> Result<[TensorBuffer;2]> {
    let [dx_spec,dw_spec]=mixed_linear_backward_specs(&layout(&input)?,&layout(&weight)?,&layout(&grad)?)?;
    let grad=super::conversion::cast(client,grad,DType::BF16)?;
    let dx=allocate(client,dx_spec.output_layout());
    let dw=allocate(client,dw_spec.output_layout());
    launch(client,dx_spec,&grad,&weight,&dx)?;
    launch(client,dw_spec,&grad,&input,&dw)?;
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

    #[test]
    fn mixed_linear_keeps_fp32_output_and_both_gradients() {
        let t=|shape:&[i64],dtype|TensorLayout::contiguous(shape,dtype).unwrap();
        let x=t(&[32,16],CannDType::F32);let w=t(&[48,16],CannDType::F32);
        let forward=mixed_linear_spec(&x,&w).unwrap();
        assert_eq!(forward.output_layout(),&t(&[32,48],CannDType::F32));
        let [dx,dw]=mixed_linear_backward_specs(&forward.a,&forward.b,forward.output_layout()).unwrap();
        assert_eq!(dx.output_layout(),&x);assert_eq!(dw.output_layout(),&w);
        assert!(mixed_linear_spec(&t(&[2,32,16],CannDType::F32),&w).is_err());
        assert!(mixed_linear_spec(&t(&[32,16],CannDType::BF16),&w).is_err());
        assert!(mixed_linear_spec(&t(&[31,16],CannDType::F32),&w).is_err());
        assert!(mixed_linear_spec(&x,&t(&[48,32],CannDType::F32)).is_err());
        assert!(mixed_linear_backward_specs(&forward.a,&forward.b,&t(&[32,48],CannDType::BF16)).is_err());
        assert!(mixed_linear_backward_specs(&forward.a,&forward.b,&t(&[48,32],CannDType::F32)).is_err());
    }
}

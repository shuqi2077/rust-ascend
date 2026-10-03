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
pub(super) fn layout(tensor: &TensorBuffer) -> Result<TensorLayout> {
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
pub(super) fn launch(client: &Client, spec: GemmSpec, a: &TensorBuffer, b: &TensorBuffer, out: &TensorBuffer) -> Result<()> {
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

fn mixed_gemm_spec(a: &TensorLayout, b: &TensorLayout, ta: Transpose, tb: Transpose) -> Result<GemmSpec> {
    if a.dtype()!=CannDType::F32 || b.dtype()!=CannDType::F32 {
        return Err(error("BF16-compute FP32 matmul requires FP32 inputs"));
    }
    let a=TensorLayout::contiguous(a.shape(),CannDType::BF16)?;
    let b=TensorLayout::contiguous(b.shape(),CannDType::BF16)?;
    GemmSpec::new(&a,&b,ta,tb,CannDType::F32)
}
pub(super) fn gemm_bf16_fp32(client: &Client, input: TensorBuffer, weight: TensorBuffer,
    ta: Transpose, tb: Transpose) -> Result<[TensorBuffer;3]> {
    let spec=mixed_gemm_spec(&layout(&input)?,&layout(&weight)?,ta,tb)?;
    // Cast allocations are independent snapshots, including when FP32 parameters alias.
    let input=super::conversion::cast(client,input,DType::BF16)?;
    let weight=super::conversion::cast(client,weight,DType::BF16)?;
    let output=allocate(client,spec.output_layout());
    launch(client,spec,&input,&weight,&output)?;
    Ok([output,input,weight])
}

fn mixed_gemm_backward_specs(x: &TensorLayout, w: &TensorLayout, dy: &TensorLayout,
    ta: Transpose, tb: Transpose)
    -> Result<[GemmSpec;2]> {
    let forward=GemmSpec::new(x,w,ta,tb,CannDType::F32)?;
    if dy!=forward.output_layout() {
        return Err(error("BF16-compute matmul backward requires saved BF16 inputs and matching FP32 dY"));
    }
    let dy=TensorLayout::contiguous(dy.shape(),CannDType::BF16)?;
    let opposite=|t|if t==Transpose::No {Transpose::Yes} else {Transpose::No};
    let dx=if ta==Transpose::No {GemmSpec::new(&dy,w,Transpose::No,opposite(tb),CannDType::F32)?}
        else {GemmSpec::new(w,&dy,tb,Transpose::Yes,CannDType::F32)?};
    let dw=if tb==Transpose::No {GemmSpec::new(x,&dy,opposite(ta),Transpose::No,CannDType::F32)?}
        else {GemmSpec::new(&dy,x,Transpose::Yes,ta,CannDType::F32)?};
    Ok([dx,dw])
}
pub(super) fn gemm_bf16_fp32_backward(client: &Client, input: TensorBuffer, weight: TensorBuffer,
    grad: TensorBuffer, ta: Transpose, tb: Transpose) -> Result<[TensorBuffer;2]> {
    let [dx_spec,dw_spec]=mixed_gemm_backward_specs(&layout(&input)?,&layout(&weight)?,&layout(&grad)?,ta,tb)?;
    let grad=super::conversion::cast(client,grad,DType::BF16)?;
    let dx=allocate(client,dx_spec.output_layout());
    let dw=allocate(client,dw_spec.output_layout());
    let (dx_a,dx_b)=if ta==Transpose::No {(&grad,&weight)} else {(&weight,&grad)};
    let (dw_a,dw_b)=if tb==Transpose::No {(&input,&grad)} else {(&grad,&input)};
    launch(client,dx_spec,dx_a,dx_b,&dx)?;
    launch(client,dw_spec,dw_a,dw_b,&dw)?;
    Ok([dx,dw])
}

fn mixed_linear_spec(x: &TensorLayout, w: &TensorLayout) -> Result<GemmSpec> {
    if x.shape().len()!=2 || w.shape().len()!=2 {
        return Err(error("BF16-compute FP32 linear requires rank-2 input and weight"));
    }
    mixed_gemm_spec(x,w,Transpose::No,Transpose::Yes)
}
pub(super) fn linear_bf16_fp32(client: &Client, input: TensorBuffer, weight: TensorBuffer)
    -> Result<[TensorBuffer;3]> {
    mixed_linear_spec(&layout(&input)?,&layout(&weight)?)?;
    gemm_bf16_fp32(client,input,weight,Transpose::No,Transpose::Yes)
}
fn mixed_linear_backward_specs(x: &TensorLayout, w: &TensorLayout, dy: &TensorLayout)
    -> Result<[GemmSpec;2]> {
    if x.shape().len()!=2 || w.shape().len()!=2 {
        return Err(error("BF16-compute linear backward requires saved rank-2 inputs"));
    }
    mixed_gemm_backward_specs(x,w,dy,Transpose::No,Transpose::Yes)
}
pub(super) fn linear_bf16_fp32_backward(client: &Client, input: TensorBuffer, weight: TensorBuffer,
    grad: TensorBuffer) -> Result<[TensorBuffer;2]> {
    mixed_linear_backward_specs(&layout(&input)?,&layout(&weight)?,&layout(&grad)?)?;
    gemm_bf16_fp32_backward(client,input,weight,grad,Transpose::No,Transpose::Yes)
}

fn frozen_linear_spec(input:&TensorLayout,weight:&TensorLayout)->Result<GemmSpec> {
    if input.shape().len()!=2 || weight.shape().len()!=2 || input.dtype()!=CannDType::F32 || weight.dtype()!=CannDType::BF16 {
        return Err(error("frozen BF16 linear requires rank-2 FP32 input and fixed BF16 weight"));
    }
    let input=TensorLayout::contiguous(input.shape(),CannDType::BF16)?;
    GemmSpec::new(&input,weight,Transpose::No,Transpose::Yes,CannDType::F32)
}
/// The fixed weight is used directly, without a cast or independent full-weight copy.
pub(super) fn linear_frozen_bf16_fp32(client:&Client,input:TensorBuffer,weight:TensorBuffer)->Result<TensorBuffer> {
    let spec=frozen_linear_spec(&layout(&input)?,&layout(&weight)?)?;
    let input=super::conversion::cast(client,input,DType::BF16)?;
    let out=allocate(client,spec.output_layout());launch(client,spec,&input,&weight,&out)?;Ok(out)
}
fn frozen_linear_backward_spec(input:&TensorLayout,weight:&TensorLayout,grad:&TensorLayout)->Result<GemmSpec> {
    let forward=frozen_linear_spec(input,weight)?;
    if grad!=forward.output_layout() {return Err(error("frozen BF16 linear backward requires matching FP32 dY"));}
    let grad=TensorLayout::contiguous(grad.shape(),CannDType::BF16)?;
    let backward=GemmSpec::new(&grad,weight,Transpose::No,Transpose::No,CannDType::F32)?;
    if backward.output_layout()!=input {return Err(error("frozen BF16 linear input gradient shape mismatch"));}Ok(backward)
}
pub(super) fn linear_frozen_bf16_fp32_backward(client:&Client,input_shape:Shape,weight:TensorBuffer,grad:TensorBuffer)->Result<TensorBuffer> {
    let shape=input_shape.iter().map(|&d|i64::try_from(d).map_err(error)).collect::<Result<Vec<_>>>()?;
    let input=TensorLayout::contiguous(&shape,CannDType::F32)?;
    let spec=frozen_linear_backward_spec(&input,&layout(&weight)?,&layout(&grad)?)?;
    let grad=super::conversion::cast(client,grad,DType::BF16)?;
    let out=allocate(client,spec.output_layout());launch(client,spec,&grad,&weight,&out)?;Ok(out)
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

    #[test]
    fn mixed_gemm_all_transposes_restore_physical_gradient_layouts() {
        for batch in [None,Some(2)] {for ta in [Transpose::No,Transpose::Yes] {for tb in [Transpose::No,Transpose::Yes] {
            let mut a=if ta==Transpose::No {vec![32,16]} else {vec![16,32]};
            let mut b=if tb==Transpose::No {vec![16,48]} else {vec![48,16]};
            if let Some(batch)=batch {a.insert(0,batch);b.insert(0,batch);}
            let a=TensorLayout::contiguous(&a,CannDType::F32).unwrap();
            let b=TensorLayout::contiguous(&b,CannDType::F32).unwrap();
            let forward=mixed_gemm_spec(&a,&b,ta,tb).unwrap();
            let [da,db]=mixed_gemm_backward_specs(&forward.a,&forward.b,forward.output_layout(),ta,tb).unwrap();
            assert_eq!(da.output_layout(),&a);assert_eq!(db.output_layout(),&b);
            assert_eq!(da.dtype,CannDType::F32);assert_eq!(db.dtype,CannDType::F32);
        }}}
        let t=|shape:&[i64]|TensorLayout::contiguous(shape,CannDType::F32).unwrap();
        assert!(mixed_gemm_spec(&t(&[2,32,16]),&t(&[1,16,48]),Transpose::No,Transpose::No).is_err());
    }
    #[test]
    fn frozen_bf16_linear_keeps_two_byte_weights_and_only_an_fp32_input_gradient() {
        let t=|shape:&[i64],dtype|TensorLayout::contiguous(shape,dtype).unwrap();
        let x=t(&[32,16],CannDType::F32);let w=t(&[48,16],CannDType::BF16);
        let forward=frozen_linear_spec(&x,&w).unwrap();
        assert_eq!(forward.b,w);assert_eq!(forward.b.byte_len(),48*16*2);
        assert_eq!(forward.a.byte_len(),32*16*2);assert_eq!(forward.output_layout(),&t(&[32,48],CannDType::F32));
        let backward=frozen_linear_backward_spec(&x,&w,forward.output_layout()).unwrap();
        assert_eq!(backward.b,w);assert_eq!(backward.a.dtype(),CannDType::BF16);assert_eq!(backward.output_layout(),&x);
        for (input,weight) in [(t(&[32,16],CannDType::BF16),w.clone()),(x.clone(),t(&[48,16],CannDType::F32)),
            (t(&[2,32,16],CannDType::F32),w.clone()),(t(&[31,16],CannDType::F32),w.clone()),(x.clone(),t(&[48,32],CannDType::BF16))] {
            assert!(frozen_linear_spec(&input,&weight).is_err());
        }
        assert!(frozen_linear_backward_spec(&x,&w,&t(&[32,48],CannDType::BF16)).is_err());
        assert!(frozen_linear_backward_spec(&x,&w,&t(&[48,32],CannDType::F32)).is_err());
    }
}

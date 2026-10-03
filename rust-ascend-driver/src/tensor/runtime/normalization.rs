use super::{AscendRuntime, ComputeClient, Result, WORKER, contiguous, error};
use ruda_core::{compiler::Compiler, ir::UIntKind, launch::ExecutionMode, tensor::{DType, Shape, Strides}};
use ruda_runtime::runtime::normalization::TensorBuffer;
use rust_ascend_compiler::ascend::{AscendCompiler, AscendKernel, AscendOptions, AscendTarget,
    programs::{self, MapProgram}, row_programs::{self, RowProgram}};

type Client = ComputeClient<AscendRuntime>;

fn layout(shape: &[usize], strides: &[usize], dtype: DType) -> Result<(usize, u32)> {
    layout_for(shape, strides, dtype, "LayerNorm")
}
pub(super) fn layout_for(shape: &[usize], strides: &[usize], dtype: DType, operation: &str) -> Result<(usize, u32)> {
    if dtype != DType::F32 || !contiguous(shape, strides) {
        return Err(error(format!("native {operation} requires contiguous FP32 tensors")));
    }
    let &width = shape.last().ok_or_else(|| error(format!("{operation} requires a last dimension")))?;
    if !(32..=4096).contains(&width) || width % 32 != 0 {
        return Err(error(format!("{operation} width must be 32..4096 and divisible by 32")));
    }
    let rows = shape[..shape.len()-1].iter().try_fold(1usize, |n, &d| n.checked_mul(d))
        .ok_or_else(|| error(format!("{operation} shape overflow")))?;
    if rows.checked_mul(width).is_none_or(|n| n > u32::MAX as usize) {
        return Err(error(format!("{operation} element count exceeds u32")));
    }
    Ok((rows, width as u32))
}
fn check(t: &TensorBuffer, shape: &[usize]) -> Result<()> {
    check_for(t, shape, "LayerNorm")
}
pub(super) fn check_for(t: &TensorBuffer, shape: &[usize], operation: &str) -> Result<()> {
    let count = shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d))
        .and_then(|n| n.checked_mul(4)).ok_or_else(|| error(format!("{operation} buffer size overflow")))?;
    if t.dtype != DType::F32 || &t.shape[..] != shape || !contiguous(&t.shape, &t.strides)
        || t.handle.size_in_used() < count as u64 {
        return Err(error(format!("{operation} buffer shape, stride, dtype or byte length mismatch")));
    }
    Ok(())
}
fn epsilon(value: f64) -> Result<f32> {
    epsilon_for(value, "LayerNorm")
}
pub(super) fn epsilon_for(value: f64, operation: &str) -> Result<f32> {
    let v = value as f32;
    if !value.is_finite() || !v.is_finite() || v <= 0.0 {
        return Err(error(format!("{operation} epsilon must be positive and representable in FP32")));
    }
    Ok(v)
}
pub(super) fn buffer(client: &Client, shape: Shape, zero: bool) -> TensorBuffer {
    let elements: usize = shape.iter().product();
    let handle = if zero { client.create_from_slice(&vec![0; elements*4]) } else { client.empty(elements*4) };
    let mut strides = vec![0; shape.len()];
    let mut stride = 1;
    for (i, &dim) in shape.iter().enumerate().rev() { strides[i] = stride; stride *= dim; }
    TensorBuffer { handle, shape, strides: Strides::from(strides), dtype: DType::F32 }
}
pub(super) fn row(op: RowProgram, rows: usize, width: u32, eps: f32) -> Result<AscendKernel> {
    AscendCompiler.compile(row_programs::definition(op, width, eps).map_err(error)?, &AscendOptions {
        target: Some(AscendTarget::Ascend950DT), elements: rows as u64*width as u64,
        row_width: Some(width), ..Default::default()
    }, ExecutionMode::Checked, UIntKind::U64.into()).map_err(error)
}
pub(super) fn map(op: MapProgram, count: usize) -> Result<AscendKernel> {
    AscendCompiler.compile(programs::definition(op), &AscendOptions {
        target: Some(AscendTarget::Ascend950DT), elements: count as u64, ..Default::default()
    }, ExecutionMode::Checked, UIntKind::U64.into()).map_err(error)
}
pub(super) fn run(client: &Client, kernel: AscendKernel, tensors: &[&TensorBuffer]) -> Result<()> {
    if tensors.len() != kernel.bindings().len() { return Err(error("LayerNorm kernel binding mismatch")); }
    client.flush().map_err(error)?;
    let guards = tensors.iter().map(|t| client.get_resource(t.handle.clone()).map_err(error))
        .collect::<Result<Vec<_>>>()?;
    let resources = guards.iter().zip(kernel.bindings()).map(|(guard, binding)| {
        let mut resource = guard.resource().clone();
        if binding.bytes > resource.byte_len() as u64 { return Err(error("LayerNorm kernel buffer too short")); }
        resource.size = binding.bytes as usize;
        Ok(resource)
    }).collect::<Result<Vec<_>>>()?;
    let worker = &WORKER.get().ok_or_else(|| error("Ascend runtime is not initialized"))?.1;
    // ManagedResource guards retain every allocation until synchronized device completion.
    let result = worker.call(move |state| state.launch(kernel, resources));
    drop(guards);
    result
}
fn slice(t: &TensorBuffer, offset: usize, elements: usize) -> TensorBuffer {
    let bytes = elements as u64*4;
    let start = offset as u64*4;
    let size = t.handle.size_in_used();
    assert!(start <= size && bytes <= size-start);
    TensorBuffer { handle: t.handle.clone().offset_start(start).offset_end(size-start-bytes),
        shape: Shape::new([elements]), strides: Strides::from(vec![1]), dtype: DType::F32 }
}

/// Pairwise device reduction over leading rows, preserving an odd row at each level.
pub(super) fn column_sum(client: &Client, mut value: TensorBuffer, mut rows: usize, width: usize) -> Result<TensorBuffer> {
    if rows == 0 { return Ok(buffer(client, Shape::new([width]), true)); }
    while rows > 1 {
        let pairs = rows/2;
        let next_rows = rows.div_ceil(2);
        let next = buffer(client, Shape::new([next_rows, width]), false);
        let a = slice(&value, 0, pairs*width);
        let b = slice(&value, pairs*width, pairs*width);
        let out = slice(&next, 0, pairs*width);
        run(client, map(MapProgram::Add, pairs*width)?, &[&a, &b, &out])?;
        if rows % 2 != 0 {
            let tail = slice(&value, (rows-1)*width, width);
            let out = slice(&next, pairs*width, width);
            run(client, map(MapProgram::Copy, width)?, &[&tail, &out])?;
        }
        value = next;
        rows = next_rows;
    }
    Ok(slice(&value, 0, width))
}

pub(super) fn forward(client: &Client, input: TensorBuffer, weight: TensorBuffer,
    bias: Option<TensorBuffer>, eps: f64) -> Result<[TensorBuffer; 3]> {
    let (rows, width) = layout(&input.shape, &input.strides, input.dtype)?;
    let eps = epsilon(eps)?;
    check(&input, &input.shape)?;
    check(&weight, &[width as usize])?;
    if let Some(bias) = &bias { check(bias, &[width as usize])?; }
    let out = buffer(client, input.shape.clone(), false);
    let mean = buffer(client, Shape::new([rows]), false);
    let rstd = buffer(client, Shape::new([rows]), false);
    if rows != 0 {
        let bias = bias.unwrap_or_else(|| buffer(client, Shape::new([width as usize]), true));
        run(client, row(RowProgram::LayerNorm, rows, width, eps)?, &[&input, &weight, &bias, &out, &mean, &rstd])?;
    }
    Ok([out, mean, rstd])
}
pub(super) fn backward(client: &Client, input: TensorBuffer, weight: TensorBuffer,
    grad: TensorBuffer, mean: TensorBuffer, rstd: TensorBuffer) -> Result<[TensorBuffer; 3]> {
    let (rows, width) = layout(&input.shape, &input.strides, input.dtype)?;
    check(&input, &input.shape)?;
    check(&grad, &input.shape)?;
    check(&weight, &[width as usize])?;
    check(&mean, &[rows])?; check(&rstd, &[rows])?;
    let dx = buffer(client, input.shape.clone(), false);
    if rows == 0 {
        return Ok([dx, buffer(client, Shape::new([width as usize]), true), buffer(client, Shape::new([width as usize]), true)]);
    }
    run(client, row(RowProgram::LayerNormInputBackward, rows, width, 1e-5)?, &[&input, &grad, &weight, &mean, &rstd, &dx])?;
    let contributions = buffer(client, input.shape.clone(), false);
    run(client, row(RowProgram::LayerNormWeightContributions, rows, width, 1e-5)?, &[&input, &grad, &mean, &rstd, &contributions])?;
    let dw = column_sum(client, contributions, rows, width as usize)?;
    let db = column_sum(client, grad, rows, width as usize)?;
    Ok([dx, dw, db])
}

pub(super) fn rms_forward(client: &Client, input: TensorBuffer, weight: TensorBuffer,
    eps: f64) -> Result<[TensorBuffer; 2]> {
    if input.shape.last().is_some_and(|&width|width>4096) {return super::wide_rows::rms_forward(client,input,weight,eps);}
    let (rows, width) = layout_for(&input.shape, &input.strides, input.dtype, "RMSNorm")?;
    let eps = epsilon_for(eps, "RMSNorm")?;
    check_for(&input, &input.shape, "RMSNorm")?;
    check_for(&weight, &[width as usize], "RMSNorm")?;
    let out = buffer(client, input.shape.clone(), false);
    let rstd = buffer(client, Shape::new([rows]), false);
    if rows != 0 {
        run(client, row(RowProgram::RmsNorm, rows, width, eps)?, &[&input, &weight, &out, &rstd])?;
    }
    Ok([out, rstd])
}

pub(super) fn rms_backward(client: &Client, input: TensorBuffer, weight: TensorBuffer,
    grad: TensorBuffer, rstd: TensorBuffer) -> Result<[TensorBuffer; 2]> {
    if input.shape.last().is_some_and(|&width|width>4096) {return super::wide_rows::rms_backward(client,input,weight,grad,rstd);}
    let (rows, width) = layout_for(&input.shape, &input.strides, input.dtype, "RMSNorm")?;
    check_for(&input, &input.shape, "RMSNorm")?;
    check_for(&weight, &[width as usize], "RMSNorm")?;
    check_for(&grad, &input.shape, "RMSNorm")?;
    check_for(&rstd, &[rows], "RMSNorm")?;
    let dx = buffer(client, input.shape.clone(), false);
    if rows == 0 {
        return Ok([dx, buffer(client, Shape::new([width as usize]), true)]);
    }
    // Both derivatives consume the forward's saved reciprocal RMS; epsilon is not recomputed.
    run(client, row(RowProgram::RmsNormInputBackward, rows, width, 1e-5)?, &[&input, &grad, &weight, &rstd, &dx])?;
    let parts = buffer(client, input.shape.clone(), false);
    run(client, row(RowProgram::RmsNormWeightContributions, rows, width, 1e-5)?, &[&input, &grad, &rstd, &parts])?;
    let dw = column_sum(client, parts, rows, width as usize)?;
    Ok([dx, dw])
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_layout_and_epsilon_contract() {
        assert_eq!(layout(&[2, 3, 96], &[288, 96, 1], DType::F32).unwrap(), (6, 96));
        assert_eq!(layout(&[0, 64], &[64, 1], DType::F32).unwrap(), (0, 64));
        assert_eq!(layout(&[32], &[1], DType::F32).unwrap(), (1, 32));
        for width in [0, 1, 31, 33, 4097] { assert!(layout(&[width], &[1], DType::F32).is_err()); }
        assert!(layout(&[2, 32], &[1, 2], DType::F32).is_err());
        assert!(layout(&[32], &[1], DType::F16).is_err());
        assert!(layout(&[usize::MAX, 32], &[32, 1], DType::F32).is_err());
        for e in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::MAX, 1e-100] { assert!(epsilon(e).is_err()); }
        assert_eq!(epsilon(1e-5).unwrap(), 1e-5f32);
    }
    #[test]
    fn rms_layout_preserves_last_axis_and_checks_all_leading_dimensions() {
        for (shape, strides, rows, width) in [
            (vec![96], vec![1], 1, 96),
            (vec![2, 3, 96], vec![288, 96, 1], 6, 96),
            (vec![1, 0, 4096], vec![0, 4096, 1], 0, 4096),
        ] {
            assert_eq!(layout_for(&shape, &strides, DType::F32, "RMSNorm").unwrap(), (rows, width));
        }
        assert!(layout_for(&[], &[], DType::F32, "RMSNorm").is_err());
        assert!(layout_for(&[3, 96], &[1, 3], DType::F32, "RMSNorm").is_err());
        for dtype in [DType::F16, DType::BF16] {
            assert!(layout_for(&[96], &[1], dtype, "RMSNorm").is_err());
        }
        for eps in [0., -1., f64::NAN, f64::INFINITY, 1e-100] {
            assert!(epsilon_for(eps, "RMSNorm").is_err());
        }
    }
}

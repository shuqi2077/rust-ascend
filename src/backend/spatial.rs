use super::*;
use crate::runtime::{Pool2dOptions, SpatialConvOptions};

pub(super) fn same_queue(reference: &Primitive, others: &[&Primitive]) {
    for value in others {
        assert_eq!(
            reference.device, value.device,
            "spatial operator device mismatch"
        );
        assert!(
            reference.client.same_execution_queue(&value.client),
            "spatial operator queue mismatch"
        );
    }
}
pub(super) fn conv_options<const N: usize>(o: ConvOptions<N>) -> SpatialConvOptions<N> {
    SpatialConvOptions {
        stride: o.stride,
        padding: o.padding,
        dilation: o.dilation,
        groups: o.groups,
        transposed: false,
        output_padding: [0; N],
    }
}
pub(super) fn transpose_options<const N: usize>(
    o: ConvTransposeOptions<N>,
) -> SpatialConvOptions<N> {
    SpatialConvOptions {
        stride: o.stride,
        padding: o.padding,
        dilation: o.dilation,
        groups: o.groups,
        transposed: true,
        output_padding: o.padding_out,
    }
}
pub(super) fn pool_options(
    kernel: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: bool,
    count_include_pad: bool,
) -> Pool2dOptions {
    Pool2dOptions {
        kernel,
        stride,
        padding,
        dilation,
        ceil_mode,
        count_include_pad,
    }
}
pub(super) fn convolution<const N: usize>(
    x: Primitive,
    weight: Primitive,
    bias: Option<Primitive>,
    options: SpatialConvOptions<N>,
) -> Primitive {
    same_queue(&x, &[&weight]);
    if let Some(b) = &bias {
        same_queue(&x, &[b]);
    }
    let x = contiguous(x);
    let weight = contiguous(weight);
    let bias = bias.map(contiguous).map(buffer);
    let output =
        AscendRuntime::convolution(&x.client, buffer(x.clone()), buffer(weight), bias, options)
            .expect("native convolution failed");
    wrap(&x, output)
}
pub(super) fn convolution_backward<const N: usize>(
    x: Primitive,
    weight: Primitive,
    grad: Primitive,
    options: SpatialConvOptions<N>,
    weight_gradient: bool,
) -> Primitive {
    same_queue(&x, &[&weight, &grad]);
    let x = contiguous(x);
    let weight = contiguous(weight);
    let grad = contiguous(grad);
    let output = AscendRuntime::convolution_backward(
        &x.client,
        buffer(x.clone()),
        buffer(weight),
        buffer(grad),
        options,
        weight_gradient,
    )
    .expect("native convolution backward failed");
    wrap(&x, output)
}

fn adaptive_window(index: usize, input: usize, output: usize) -> (usize, usize) {
    let start = index
        .checked_mul(input)
        .expect("adaptive pool geometry overflow")
        / output;
    let end = (index + 1)
        .checked_mul(input)
        .expect("adaptive pool geometry overflow")
        .div_ceil(output);
    (start, end)
}
fn slices(batch: usize, channels: usize, h: (usize, usize), w: (usize, usize)) -> [Slice; 4] {
    [
        Slice::from(0..batch),
        Slice::from(0..channels),
        Slice::from(h.0..h.1),
        Slice::from(w.0..w.1),
    ]
}
fn pool_shape(x: &Primitive, o: &Pool2dOptions) -> Shape {
    let shape = x.meta.shape();
    assert_eq!(shape.rank(), 4, "pooling requires NCHW tensors");
    assert!(
        !o.kernel.contains(&0) && !o.stride.contains(&0) && !o.dilation.contains(&0),
        "invalid pool geometry"
    );
    for axis in 0..2 {
        let kernel = (o.kernel[axis] - 1)
            .checked_mul(o.dilation[axis])
            .and_then(|v| v.checked_add(1))
            .expect("pool kernel overflow");
        assert!(
            o.padding[axis]
                .checked_mul(2)
                .and_then(|v| v.checked_add(shape[axis + 2]))
                .is_some_and(|n| n >= kernel),
            "invalid pool extent"
        );
    }
    ruda_core::tensor::spatial::calculate_pool_output_shape(
        shape,
        &o.kernel,
        &o.stride,
        &o.padding,
        &o.dilation,
        o.ceil_mode,
    )
    .expect("invalid pool rank")
}
pub(super) fn native_pool_geometry(x: &Primitive, o: &Pool2dOptions) -> bool {
    let out = pool_shape(x, o);
    (0..2).all(|a| {
        o.padding[a] <= o.kernel[a] / 2
            && (!o.ceil_mode
                || (out[a + 2] - 1)
                    .checked_mul(o.stride[a])
                    .is_some_and(|v| v < x.meta.shape()[a + 2] + o.padding[a]))
    })
}
fn positions(
    input: &Shape,
    output: &Shape,
    o: &Pool2dOptions,
    kh: usize,
    kw: usize,
) -> (Vec<i32>, Vec<bool>) {
    let mut indices = Vec::with_capacity(output[2] * output[3]);
    let mut valid = Vec::with_capacity(indices.capacity());
    for h in 0..output[2] {
        for w in 0..output[3] {
            let ih = h as i128 * o.stride[0] as i128 + kh as i128 * o.dilation[0] as i128
                - o.padding[0] as i128;
            let iw = w as i128 * o.stride[1] as i128 + kw as i128 * o.dilation[1] as i128
                - o.padding[1] as i128;
            let inside = ih >= 0 && ih < input[2] as i128 && iw >= 0 && iw < input[3] as i128;
            valid.push(inside);
            indices.push(if inside {
                i32::try_from(ih * input[3] as i128 + iw).expect("pool indices exceed I32")
            } else {
                0
            });
        }
    }
    (indices, valid)
}
fn spatial_mask(values: Vec<bool>, device: &crate::runtime::AscendDevice) -> Primitive {
    let length = values.len();
    RudaAscend::bool_from_data(TensorData::new(values, [1, 1, length]), device)
}
pub(super) fn maximum_pool(x: Primitive, o: Pool2dOptions) -> MaxPool2dWithIndices<RudaAscend> {
    let input = x.meta.shape().clone();
    let output_shape = pool_shape(&x, &o);
    let dtype = FloatDType::try_from(x.dtype).expect("max pool requires floating input");
    let plane = input[2].checked_mul(input[3]).expect("pool plane overflow");
    assert!(
        plane as u128 <= i32::MAX as u128 + 1,
        "pool plane does not fit I32 indices"
    );
    let count = output_shape[2] * output_shape[3];
    let flat_shape = Shape::new([input[0], input[1], count]);
    let mut maximum = RudaAscend::float_full(
        flat_shape.clone(),
        Scalar::Float(f64::NEG_INFINITY),
        &x.device,
        dtype,
    );
    let mut indices = RudaAscend::int_full(flat_shape, Scalar::Int(-1), &x.device, IntDType::I32);
    let x = RudaAscend::float_reshape(x, Shape::new([input[0], input[1], plane]));
    let mut seen = vec![false; count];
    // Preserve RUDA's first-valid value, strict greater-than ties, NaN behavior,
    // and -1 empty-window indices instead of relying on ACLNN MaxPool semantics.
    for kh in 0..o.kernel[0] {
        for kw in 0..o.kernel[1] {
            let (positions, valid) = positions(&input, &output_shape, &o, kh, kw);
            if !valid.iter().any(|&v| v) {
                continue;
            }
            let first: Vec<bool> = valid
                .iter()
                .zip(&mut seen)
                .map(|(&v, seen)| {
                    let first = v && !*seen;
                    *seen |= v;
                    first
                })
                .collect();
            let selected_indices =
                RudaAscend::int_from_data(TensorData::new(positions, [count]), &x.device);
            let selected = RudaAscend::float_select(x.clone(), 2, selected_indices.clone());
            let greater =
                RudaAscend::float_greater(selected.clone(), maximum.clone(), BoolDType::U8);
            let replace = RudaAscend::bool_and(
                spatial_mask(valid, &x.device),
                RudaAscend::bool_or(spatial_mask(first, &x.device), greater),
            );
            maximum = RudaAscend::float_mask_where(maximum, replace.clone(), selected);
            let selected_indices =
                RudaAscend::int_reshape(selected_indices, Shape::new([1, 1, count]));
            indices = RudaAscend::int_mask_where(indices, replace, selected_indices);
        }
    }
    MaxPool2dWithIndices {
        output: RudaAscend::float_reshape(maximum, output_shape.clone()),
        indices: RudaAscend::int_reshape(indices, output_shape),
    }
}
pub(super) fn maximum_pool_backward(
    x: Primitive,
    grad: Primitive,
    indices: Primitive,
    o: Pool2dOptions,
) -> Primitive {
    same_queue(&x, &[&grad, &indices]);
    let expected = pool_shape(&x, &o);
    assert_eq!(
        grad.meta.shape(),
        &expected,
        "max pool gradient shape mismatch"
    );
    assert_eq!(
        indices.meta.shape(),
        &expected,
        "max pool indices shape mismatch"
    );
    assert_eq!(x.dtype, grad.dtype, "max pool gradient dtype mismatch");
    let [batch, channels, height, width] = x.meta.shape().dims();
    let dtype = FloatDType::try_from(x.dtype).expect("max pool requires floating input");
    let output = RudaAscend::float_zeros(
        Shape::new([batch, channels, height * width]),
        &x.device,
        dtype,
    );
    if height == 0 || width == 0 {
        return RudaAscend::float_reshape(output, x.meta.shape().clone());
    }
    let indices = RudaAscend::int_reshape(
        indices,
        Shape::new([batch, channels, expected[2] * expected[3]]),
    );
    let invalid = RudaAscend::int_lower_elem(indices.clone(), Scalar::Int(0), BoolDType::U8);
    let indices = RudaAscend::int_mask_fill(indices, invalid.clone(), Scalar::Int(0));
    let grad = RudaAscend::float_reshape(
        grad,
        Shape::new([batch, channels, expected[2] * expected[3]]),
    );
    let grad = RudaAscend::float_mask_fill(grad, invalid, Scalar::Float(0.0));
    let output = RudaAscend::float_scatter_add(2, output, indices, grad);
    RudaAscend::float_reshape(output, x.meta.shape().clone())
}
fn pool_divisors(input: &Shape, output: &Shape, o: &Pool2dOptions) -> Vec<f32> {
    let extent = |index: usize, axis: usize| {
        let begin = index as i128 * o.stride[axis] as i128 - o.padding[axis] as i128;
        let end = begin + o.kernel[axis] as i128;
        let pad = if o.count_include_pad {
            o.padding[axis] as i128
        } else {
            0
        };
        (end.min(input[axis + 2] as i128 + pad) - begin.max(-pad)).max(0) as usize
    };
    (0..output[2])
        .flat_map(|h| {
            (0..output[3]).map(move |w| {
                extent(h, 0)
                    .checked_mul(extent(w, 1))
                    .expect("pool divisor overflow") as f32
            })
        })
        .collect()
}
pub(super) fn average_pool(x: Primitive, o: Pool2dOptions) -> Primitive {
    let input = x.meta.shape().clone();
    let output_shape = pool_shape(&x, &o);
    let dtype = FloatDType::try_from(x.dtype).expect("average pool requires floating input");
    let count = output_shape[2] * output_shape[3];
    let mut sum =
        RudaAscend::float_zeros(Shape::new([input[0], input[1], count]), &x.device, dtype);
    let x = RudaAscend::float_reshape(x, Shape::new([input[0], input[1], input[2] * input[3]]));
    for kh in 0..o.kernel[0] {
        for kw in 0..o.kernel[1] {
            let (indices, valid) = positions(&input, &output_shape, &o, kh, kw);
            if !valid.iter().any(|&v| v) {
                continue;
            }
            let indices = RudaAscend::int_from_data(TensorData::new(indices, [count]), &x.device);
            let values = RudaAscend::float_select(x.clone(), 2, indices);
            let invalid = spatial_mask(valid.into_iter().map(|v| !v).collect(), &x.device);
            sum = RudaAscend::float_add(
                sum,
                RudaAscend::float_mask_fill(values, invalid, Scalar::Float(0.0)),
            );
        }
    }
    let divisors = RudaAscend::float_from_data(
        TensorData::new(pool_divisors(&input, &output_shape, &o), [1, 1, count]),
        &x.device,
    );
    let mean = RudaAscend::float_div(sum, RudaAscend::float_cast(divisors, dtype));
    RudaAscend::float_reshape(mean, output_shape)
}
pub(super) fn average_pool_backward(x: Primitive, grad: Primitive, o: Pool2dOptions) -> Primitive {
    same_queue(&x, &[&grad]);
    let input = x.meta.shape().clone();
    let output_shape = pool_shape(&x, &o);
    assert_eq!(
        grad.meta.shape(),
        &output_shape,
        "average pool gradient shape mismatch"
    );
    assert_eq!(x.dtype, grad.dtype, "average pool gradient dtype mismatch");
    let dtype = FloatDType::try_from(x.dtype).expect("average pool requires floating input");
    let count = output_shape[2] * output_shape[3];
    let mut output = RudaAscend::float_zeros(
        Shape::new([input[0], input[1], input[2] * input[3]]),
        &x.device,
        dtype,
    );
    if input[2] == 0 || input[3] == 0 {
        return RudaAscend::float_reshape(output, input);
    }
    let grad = RudaAscend::float_reshape(grad, Shape::new([input[0], input[1], count]));
    let divisors = RudaAscend::float_from_data(
        TensorData::new(pool_divisors(&input, &output_shape, &o), [1, 1, count]),
        &x.device,
    );
    let contribution = RudaAscend::float_div(grad, RudaAscend::float_cast(divisors, dtype));
    for kh in 0..o.kernel[0] {
        for kw in 0..o.kernel[1] {
            let (indices, valid) = positions(&input, &output_shape, &o, kh, kw);
            if !valid.iter().any(|&v| v) {
                continue;
            }
            let indices =
                RudaAscend::int_from_data(TensorData::new(indices, [1, 1, count]), &x.device);
            let indices = RudaAscend::int_expand(indices, Shape::new([input[0], input[1], count]));
            let invalid = spatial_mask(valid.into_iter().map(|v| !v).collect(), &x.device);
            let values =
                RudaAscend::float_mask_fill(contribution.clone(), invalid, Scalar::Float(0.0));
            output = RudaAscend::float_scatter_add(2, output, indices, values);
        }
    }
    RudaAscend::float_reshape(output, input)
}
pub(super) fn adaptive_pool(x: Primitive, size: [usize; 2]) -> Primitive {
    let [batch, channels, height, width] = x.meta.shape().dims();
    assert!(
        size.iter().all(|&n| n > 0) && height > 0 && width > 0,
        "adaptive pool requires positive spatial dimensions"
    );
    let dtype = FloatDType::try_from(x.dtype).expect("adaptive pool requires floating storage");
    let mut output = RudaAscend::float_empty(
        Shape::new([batch, channels, size[0], size[1]]),
        &x.device,
        dtype,
    );
    for h in 0..size[0] {
        for w in 0..size[1] {
            let hs = adaptive_window(h, height, size[0]);
            let ws = adaptive_window(w, width, size[1]);
            let values = RudaAscend::float_slice(x.clone(), &slices(batch, channels, hs, ws));
            let values = RudaAscend::float_cast(values, FloatDType::F32);
            let sum = RudaAscend::float_sum_dim(RudaAscend::float_sum_dim(values, 3), 2);
            let mean = RudaAscend::float_div_scalar(
                sum,
                Scalar::Float(((hs.1 - hs.0) * (ws.1 - ws.0)) as f64),
            );
            let mean = RudaAscend::float_cast(mean, dtype);
            output = RudaAscend::float_slice_assign(
                output,
                &slices(batch, channels, (h, h + 1), (w, w + 1)),
                mean,
            );
        }
    }
    output
}
pub(super) fn adaptive_pool_backward(x: Primitive, grad: Primitive) -> Primitive {
    same_queue(&x, &[&grad]);
    let [batch, channels, height, width] = x.meta.shape().dims();
    let [gb, gc, gh, gw] = grad.meta.shape().dims();
    assert_eq!(
        [batch, channels],
        [gb, gc],
        "adaptive pool gradient channels mismatch"
    );
    assert_eq!(x.dtype, grad.dtype, "adaptive pool gradient dtype mismatch");
    assert!(
        gh > 0 && gw > 0 && height > 0 && width > 0,
        "adaptive pool requires positive spatial dimensions"
    );
    let dtype = FloatDType::try_from(x.dtype).expect("adaptive pool requires floating storage");
    let mut output = RudaAscend::float_zeros(x.meta.shape().clone(), &x.device, FloatDType::F32);
    let grad = RudaAscend::float_cast(grad, FloatDType::F32);
    for h in 0..gh {
        for w in 0..gw {
            let hs = adaptive_window(h, height, gh);
            let ws = adaptive_window(w, width, gw);
            let region = slices(batch, channels, hs, ws);
            let value = RudaAscend::float_slice(
                grad.clone(),
                &slices(batch, channels, (h, h + 1), (w, w + 1)),
            );
            let value = RudaAscend::float_div_scalar(
                value,
                Scalar::Float(((hs.1 - hs.0) * (ws.1 - ws.0)) as f64),
            );
            let value = RudaAscend::float_expand(
                value,
                Shape::new([batch, channels, hs.1 - hs.0, ws.1 - ws.0]),
            );
            let previous = RudaAscend::float_slice(output.clone(), &region);
            output = RudaAscend::float_slice_assign(
                output,
                &region,
                RudaAscend::float_add(previous, value),
            );
        }
    }
    RudaAscend::float_cast(output, dtype)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn adaptive_bins_overlap_without_rounding_the_output_size() {
        assert_eq!(adaptive_window(0, 5, 3), (0, 2));
        assert_eq!(adaptive_window(1, 5, 3), (1, 4));
        assert_eq!(adaptive_window(2, 5, 3), (3, 5));
        assert_eq!(adaptive_window(1, 2, 4), (0, 1));
    }
    #[test]
    fn pool_padding_dilation_and_ceil_divisors_match_ruda_geometry() {
        let input = Shape::new([1, 1, 3, 3]);
        let output = Shape::new([1, 1, 3, 3]);
        let mut o = pool_options([2; 2], [2; 2], [1; 2], [1; 2], true, true);
        assert_eq!(
            pool_divisors(&input, &output, &o),
            vec![4., 4., 2., 4., 4., 2., 2., 2., 1.]
        );
        o.count_include_pad = false;
        assert_eq!(
            pool_divisors(&input, &output, &o),
            vec![1., 2., 0., 2., 4., 0., 0., 0., 0.]
        );
        let (indices, valid) = positions(&input, &output, &o, 1, 1);
        assert_eq!(indices, vec![0, 2, 0, 6, 8, 0, 0, 0, 0]);
        assert_eq!(
            valid,
            vec![true, true, false, true, true, false, false, false, false]
        );
        o.dilation = [2; 2];
        let (indices, valid) = positions(&input, &output, &o, 1, 1);
        assert_eq!(indices[0], 4);
        assert!(valid[0]);
        assert!(!valid[1]);
    }
}

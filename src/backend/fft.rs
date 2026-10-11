use super::*;

fn length(requested: usize) -> usize {
    assert!(requested > 0, "FFT transform length must be positive");
    requested
        .checked_next_power_of_two()
        .expect("FFT length overflow")
}
fn bit_reversal(n: usize) -> Vec<i64> {
    let bits = n.trailing_zeros();
    (0..n)
        .map(|i| {
            if bits == 0 {
                0
            } else {
                i64::try_from(i.reverse_bits() >> (usize::BITS - bits)).expect("FFT index overflow")
            }
        })
        .collect()
}
fn twiddles(width: usize, inverse: bool) -> (Vec<f32>, Vec<f32>) {
    (0..width / 2)
        .map(|k| {
            let angle =
                (if inverse { 2.0 } else { -2.0 }) * std::f64::consts::PI * k as f64 / width as f64;
            (angle.cos() as f32, angle.sin() as f32)
        })
        .unzip()
}
fn axis_slice(value: Primitive, dim: usize, start: usize, end: usize) -> Primitive {
    let ranges = value
        .meta
        .shape()
        .iter()
        .enumerate()
        .map(|(axis, &size)| Slice::from(if axis == dim { start..end } else { 0..size }))
        .collect::<Vec<_>>();
    RudaAscend::float_slice(value, &ranges)
}
fn resize_last(value: Primitive, length: usize, used: usize) -> Primitive {
    let last = value.meta.shape().rank() - 1;
    let value = axis_slice(value, last, 0, used);
    if length == used {
        return contiguous(value);
    }
    let mut shape = value.meta.shape().clone();
    shape[last] = length;
    let output = RudaAscend::float_zeros(shape, &value.device, FloatDType::F32);
    let ranges = value
        .meta
        .shape()
        .iter()
        .map(|&size| Slice::from(0..size))
        .collect::<Vec<_>>();
    RudaAscend::float_slice_assign(output, &ranges, value)
}
// Radix-2 butterflies execute through native tensor operators; only permutation
// indices and twiddle constants are constructed on the host.
fn transform(mut real: Primitive, mut imag: Primitive, inverse: bool) -> (Primitive, Primitive) {
    let output_shape = real.meta.shape().clone();
    let original_last = output_shape.rank() - 1;
    let n = output_shape[original_last];
    let batches = output_shape[..original_last].iter().product::<usize>();
    let original = Shape::new([batches, n]);
    let last = 1;
    real = RudaAscend::float_reshape(real, original.clone());
    imag = RudaAscend::float_reshape(imag, original.clone());
    assert!(
        n.is_power_of_two(),
        "FFT requires a power-of-two working length"
    );
    let indices = RudaAscend::int_from_data(TensorData::new(bit_reversal(n), [n]), &real.device);
    real = RudaAscend::float_select(real, last, indices.clone());
    imag = RudaAscend::float_select(imag, last, indices);
    let mut width = 2;
    while width <= n {
        let mut stage_shape = original[..last].to_vec();
        stage_shape.extend([n / width, width]);
        let axis = stage_shape.len() - 1;
        real = RudaAscend::float_reshape(real, Shape::from(stage_shape.clone()));
        imag = RudaAscend::float_reshape(imag, Shape::from(stage_shape));
        let ar = axis_slice(real.clone(), axis, 0, width / 2);
        let ai = axis_slice(imag.clone(), axis, 0, width / 2);
        let br = axis_slice(real, axis, width / 2, width);
        let bi = axis_slice(imag, axis, width / 2, width);
        let (cos, sin) = twiddles(width, inverse);
        let mut twiddle_shape = vec![1; axis + 1];
        twiddle_shape[axis] = width / 2;
        let cos =
            RudaAscend::float_from_data(TensorData::new(cos, twiddle_shape.clone()), &br.device);
        let sin = RudaAscend::float_from_data(TensorData::new(sin, twiddle_shape), &br.device);
        let tr = RudaAscend::float_sub(
            RudaAscend::float_mul(br.clone(), cos.clone()),
            RudaAscend::float_mul(bi.clone(), sin.clone()),
        );
        let ti = RudaAscend::float_add(
            RudaAscend::float_mul(br, sin),
            RudaAscend::float_mul(bi, cos),
        );
        real = RudaAscend::float_cat(
            vec![
                RudaAscend::float_add(ar.clone(), tr.clone()),
                RudaAscend::float_sub(ar, tr),
            ],
            axis,
        );
        imag = RudaAscend::float_cat(
            vec![
                RudaAscend::float_add(ai.clone(), ti.clone()),
                RudaAscend::float_sub(ai, ti),
            ],
            axis,
        );
        real = RudaAscend::float_reshape(real, original.clone());
        imag = RudaAscend::float_reshape(imag, original.clone());
        if width == n {
            break;
        }
        width *= 2;
    }
    if inverse {
        real = RudaAscend::float_div_scalar(real, Scalar::Float(n as f64));
        imag = RudaAscend::float_div_scalar(imag, Scalar::Float(n as f64));
    }
    (
        RudaAscend::float_reshape(real, output_shape.clone()),
        RudaAscend::float_reshape(imag, output_shape),
    )
}
pub(super) fn rfft(signal: Primitive, dim: usize, n: Option<usize>) -> (Primitive, Primitive) {
    assert_eq!(signal.dtype, DType::F32, "RUDA FFT requires F32 storage");
    let shape = signal.meta.shape();
    assert!(dim < shape.rank(), "rfft dimension out of bounds");
    let requested = n.unwrap_or(shape[dim]);
    let length = length(requested);
    let used = requested.min(shape[dim]);
    let last = shape.rank() - 1;
    let signal = RudaAscend::float_swap_dims(signal, dim, last);
    let real = resize_last(signal, length, used);
    let imag = RudaAscend::float_zeros(real.meta.shape().clone(), &real.device, FloatDType::F32);
    let (real, imag) = transform(real, imag, false);
    let real = axis_slice(real, last, 0, length / 2 + 1);
    let imag = axis_slice(imag, last, 0, length / 2 + 1);
    (
        RudaAscend::float_swap_dims(real, dim, last),
        RudaAscend::float_swap_dims(imag, dim, last),
    )
}
pub(super) fn irfft(real: Primitive, imag: Primitive, dim: usize, n: Option<usize>) -> Primitive {
    spatial::same_queue(&real, &[&imag]);
    assert_eq!(real.dtype, DType::F32, "RUDA FFT requires F32 storage");
    assert_eq!(real.dtype, imag.dtype, "irfft dtype mismatch");
    assert_eq!(real.meta.shape(), imag.meta.shape(), "irfft shape mismatch");
    let shape = real.meta.shape();
    assert!(
        dim < shape.rank() && shape[dim] > 0,
        "irfft requires a nonempty spectrum axis"
    );
    let requested = n.unwrap_or_else(|| {
        (shape[dim] - 1)
            .checked_mul(2)
            .expect("irfft inferred length overflow")
    });
    let length = length(requested);
    let half = length / 2 + 1;
    let used = shape[dim].min(half);
    let last = shape.rank() - 1;
    let real = resize_last(RudaAscend::float_swap_dims(real, dim, last), half, used);
    let mut imag = resize_last(RudaAscend::float_swap_dims(imag, dim, last), half, used);
    // DC and (for even N) Nyquist bins have no imaginary component in a real signal.
    for index in [0, half - 1] {
        let mut shape = imag.meta.shape().clone();
        shape[last] = 1;
        let zero = RudaAscend::float_zeros(shape, &imag.device, FloatDType::F32);
        let ranges = imag
            .meta
            .shape()
            .iter()
            .enumerate()
            .map(|(axis, &size)| {
                Slice::from(if axis == last {
                    index..index + 1
                } else {
                    0..size
                })
            })
            .collect::<Vec<_>>();
        imag = RudaAscend::float_slice_assign(imag, &ranges, zero);
    }
    let (real, imag) = if length > 2 {
        let mirror_re =
            RudaAscend::float_flip(axis_slice(real.clone(), last, 1, half - 1), &[last]);
        let mirror_im = RudaAscend::float_neg(RudaAscend::float_flip(
            axis_slice(imag.clone(), last, 1, half - 1),
            &[last],
        ));
        (
            RudaAscend::float_cat(vec![real, mirror_re], last),
            RudaAscend::float_cat(vec![imag, mirror_im], last),
        )
    } else {
        (real, imag)
    };
    let (real, _) = transform(real, imag, true);
    let real = axis_slice(real, last, 0, requested);
    RudaAscend::float_swap_dims(real, dim, last)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn legacy_lengths_and_radix_two_constants() {
        assert_eq!(length(3), 4);
        assert_eq!(length(9), 16);
        assert_eq!(bit_reversal(1), vec![0]);
        assert_eq!(bit_reversal(8), vec![0, 4, 2, 6, 1, 5, 3, 7]);
        let (real, imag) = twiddles(4, false);
        assert_eq!(real[0], 1.0);
        assert_eq!(imag[0], -0.0);
        assert!(real[1].abs() < 1e-7);
        assert_eq!(imag[1], -1.0);
    }
}

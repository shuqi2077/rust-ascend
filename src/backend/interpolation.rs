use super::*;

fn coordinate(index: usize, input: usize, output: usize, options: &InterpolateOptions) -> f32 {
    if options.align_corners {
        if matches!(options.mode, InterpolateMode::Bilinear) {
            index as f32 * ((input - 1) as f32 / (output.saturating_sub(1).max(1)) as f32)
        } else {
            index
                .checked_mul(input - 1)
                .expect("interpolation coordinate overflow") as f32
                / output.saturating_sub(1).max(1) as f32
        }
    } else {
        let value = (index as f32 + 0.5) * (input as f32 / output as f32) - 0.5;
        if matches!(options.mode, InterpolateMode::Bilinear) {
            value.clamp(0.0, (input - 1) as f32)
        } else {
            value
        }
    }
}
fn taps(mode: &InterpolateMode) -> usize {
    match mode {
        InterpolateMode::Nearest => 1,
        InterpolateMode::Bilinear => 2,
        InterpolateMode::Bicubic => 4,
        InterpolateMode::Lanczos3 => 6,
    }
}
fn position(
    index: usize,
    input: usize,
    output: usize,
    tap: usize,
    options: &InterpolateOptions,
) -> (i64, f32, bool) {
    if matches!(options.mode, InterpolateMode::Nearest) {
        let index = (index as u128 * input as u128 / output as u128) as usize;
        return (index as i64, 0.0, true);
    }
    let value = coordinate(index, input, output, options);
    let floor = value.floor();
    let position = match options.mode {
        InterpolateMode::Bilinear => {
            if tap == 0 {
                floor
            } else {
                value.ceil()
            }
        }
        InterpolateMode::Bicubic => floor + tap as f32 - 1.0,
        InterpolateMode::Lanczos3 => floor + tap as f32 - 2.0,
        _ => unreachable!(),
    };
    let valid = !matches!(options.mode, InterpolateMode::Lanczos3)
        || (position >= 0.0 && position <= (input - 1) as f32);
    let fraction = if matches!(options.mode, InterpolateMode::Lanczos3) {
        value - position
    } else {
        value - floor
    };
    (
        position.clamp(0.0, (input - 1) as f32) as i64,
        fraction,
        valid,
    )
}
fn constant(reference: &Primitive, value: f64) -> Primitive {
    scalar_like(reference, Scalar::Float(value))
}
fn cubic(t: Primitive, tap: usize) -> Primitive {
    let (x, outer) = match tap {
        0 => (RudaAscend::float_add_scalar(t, Scalar::Float(1.0)), true),
        1 => (t, false),
        2 => (RudaAscend::float_sub(constant(&t, 1.0), t), false),
        3 => (RudaAscend::float_sub(constant(&t, 2.0), t), true),
        _ => unreachable!(),
    };
    if outer {
        let value = RudaAscend::float_mul_scalar(x.clone(), Scalar::Float(-0.75));
        let value = RudaAscend::float_mul(
            RudaAscend::float_add_scalar(value, Scalar::Float(3.75)),
            x.clone(),
        );
        let value =
            RudaAscend::float_mul(RudaAscend::float_add_scalar(value, Scalar::Float(-6.0)), x);
        RudaAscend::float_add_scalar(value, Scalar::Float(3.0))
    } else {
        let value = RudaAscend::float_mul_scalar(x.clone(), Scalar::Float(1.25));
        let value = RudaAscend::float_sub_scalar(value, Scalar::Float(2.25));
        let value = RudaAscend::float_mul(RudaAscend::float_mul(value, x.clone()), x);
        RudaAscend::float_add_scalar(value, Scalar::Float(1.0))
    }
}
fn weight(fraction: Primitive, tap: usize, mode: &InterpolateMode) -> Primitive {
    match mode {
        InterpolateMode::Nearest => RudaAscend::float_ones(
            fraction.meta.shape().clone(),
            &fraction.device,
            FloatDType::F32,
        ),
        InterpolateMode::Bilinear => {
            if tap == 0 {
                RudaAscend::float_sub(constant(&fraction, 1.0), fraction)
            } else {
                fraction
            }
        }
        InterpolateMode::Bicubic => cubic(fraction, tap),
        InterpolateMode::Lanczos3 => {
            let distance = RudaAscend::float_abs(fraction.clone());
            let near =
                RudaAscend::float_lower_elem(distance.clone(), Scalar::Float(1e-7), BoolDType::U8);
            let outside =
                RudaAscend::float_greater_equal_elem(distance, Scalar::Float(3.0), BoolDType::U8);
            let pi =
                RudaAscend::float_mul_scalar(fraction, Scalar::Float(std::f32::consts::PI as f64));
            let third = RudaAscend::float_div_scalar(pi.clone(), Scalar::Float(3.0));
            let value = RudaAscend::float_div(
                RudaAscend::float_mul(
                    RudaAscend::float_sin(pi.clone()),
                    RudaAscend::float_sin(third.clone()),
                ),
                RudaAscend::float_mul(pi, third),
            );
            let value = RudaAscend::float_mask_fill(value, near, Scalar::Float(1.0));
            RudaAscend::float_mask_fill(value, outside, Scalar::Float(0.0))
        }
    }
}
struct Tap {
    indices: Primitive,
    x_weight: Primitive,
    y_weight: Primitive,
    valid: Primitive,
}
fn tap(
    reference: &Primitive,
    input: [usize; 2],
    output: [usize; 2],
    kh: usize,
    kw: usize,
    options: &InterpolateOptions,
    dtype: FloatDType,
) -> Tap {
    let count = output[0]
        .checked_mul(output[1])
        .expect("interpolation output overflow");
    let mut indices = Vec::with_capacity(count);
    let mut xs = Vec::with_capacity(count);
    let mut ys = Vec::with_capacity(count);
    let mut valid = Vec::with_capacity(count);
    for h in 0..output[0] {
        let (y, yf, yvalid) = position(h, input[0], output[0], kh, options);
        for w in 0..output[1] {
            let (x, xf, xvalid) = position(w, input[1], output[1], kw, options);
            indices.push(
                y.checked_mul(input[1] as i64)
                    .and_then(|v| v.checked_add(x))
                    .expect("interpolation index overflow"),
            );
            xs.push(xf);
            ys.push(yf);
            valid.push(xvalid && yvalid);
        }
    }
    let indices = RudaAscend::int_from_data(TensorData::new(indices, [count]), &reference.device);
    let xs = RudaAscend::float_cast(
        RudaAscend::float_from_data(TensorData::new(xs, [1, 1, count]), &reference.device),
        dtype,
    );
    let ys = RudaAscend::float_cast(
        RudaAscend::float_from_data(TensorData::new(ys, [1, 1, count]), &reference.device),
        dtype,
    );
    let valid =
        RudaAscend::bool_from_data(TensorData::new(valid, [1, 1, count]), &reference.device);
    Tap {
        indices,
        x_weight: weight(xs, kw, &options.mode),
        y_weight: weight(ys, kh, &options.mode),
        valid,
    }
}
fn check(x: &Primitive, output: [usize; 2]) -> [usize; 4] {
    let dims: [usize; 4] = x.meta.shape().dims();
    assert!(
        dims[2] > 0 && dims[3] > 0,
        "interpolation requires positive input spatial dimensions"
    );
    output[0]
        .checked_mul(output[1])
        .expect("interpolation output overflow");
    assert!(
        matches!(x.dtype, DType::F32 | DType::F16 | DType::BF16),
        "Ascend interpolation requires F32/F16/BF16"
    );
    dims
}
pub(super) fn forward(x: Primitive, size: [usize; 2], options: InterpolateOptions) -> Primitive {
    let [batch, channels, height, width] = check(&x, size);
    let dtype = FloatDType::try_from(x.dtype).expect("floating input");
    let count = size[0] * size[1];
    let x = RudaAscend::float_reshape(x, Shape::new([batch, channels, height * width]));
    if matches!(options.mode, InterpolateMode::Nearest) {
        let tap = tap(&x, [height, width], size, 0, 0, &options, FloatDType::F32);
        let values = RudaAscend::float_select(x, 2, tap.indices);
        return RudaAscend::float_reshape(values, Shape::new([batch, channels, size[0], size[1]]));
    }
    let shape = Shape::new([batch, channels, count]);
    let mut result = RudaAscend::float_zeros(shape.clone(), &x.device, dtype);
    let mut total_weight =
        RudaAscend::float_zeros(Shape::new([1, 1, count]), &x.device, FloatDType::F32);
    let n = taps(&options.mode);
    for kh in 0..n {
        let mut row = None;
        let mut row_weight = None;
        for kw in 0..n {
            let tap = tap(
                &x,
                [height, width],
                size,
                kh,
                kw,
                &options,
                if matches!(options.mode, InterpolateMode::Lanczos3) {
                    FloatDType::F32
                } else {
                    dtype
                },
            );
            let values = RudaAscend::float_select(x.clone(), 2, tap.indices);
            let contribution = if matches!(options.mode, InterpolateMode::Lanczos3) {
                let weight = RudaAscend::float_mul(tap.x_weight, tap.y_weight);
                let weight = RudaAscend::float_mask_fill(
                    weight,
                    RudaAscend::bool_not(tap.valid.clone()),
                    Scalar::Float(0.0),
                );
                total_weight = RudaAscend::float_add(total_weight, weight.clone());
                RudaAscend::float_mul(values, RudaAscend::float_cast(weight, dtype))
            } else {
                row_weight = Some(tap.y_weight.clone());
                let value = RudaAscend::float_mul(values, tap.x_weight);
                if matches!(options.mode, InterpolateMode::Bicubic) {
                    value
                } else {
                    RudaAscend::float_mul(value, tap.y_weight)
                }
            };
            let contribution = RudaAscend::float_mask_fill(
                contribution,
                RudaAscend::bool_not(tap.valid),
                Scalar::Float(0.0),
            );
            if matches!(
                options.mode,
                InterpolateMode::Lanczos3 | InterpolateMode::Bilinear
            ) {
                if matches!(options.mode, InterpolateMode::Bilinear) && kh == 0 && kw == 0 {
                    result = contribution;
                    continue;
                }
                result = RudaAscend::float_add(result, contribution);
            } else {
                row = Some(match row {
                    None => contribution,
                    Some(row) => RudaAscend::float_add(row, contribution),
                });
            }
        }
        if matches!(
            options.mode,
            InterpolateMode::Lanczos3 | InterpolateMode::Bilinear
        ) {
            continue;
        }
        let mut row = row.unwrap();
        if matches!(options.mode, InterpolateMode::Bicubic) {
            row = RudaAscend::float_mul(row, row_weight.unwrap());
        }
        if kh == 0 {
            result = row;
        } else {
            result = RudaAscend::float_add(result, row);
        }
    }
    if matches!(options.mode, InterpolateMode::Lanczos3) {
        let positive = RudaAscend::bool_not(RudaAscend::float_equal_elem(
            total_weight.clone(),
            Scalar::Float(0.0),
            BoolDType::U8,
        ));
        let inverse = RudaAscend::float_div(constant(&total_weight, 1.0), total_weight);
        let normalized =
            RudaAscend::float_mul(result.clone(), RudaAscend::float_cast(inverse, dtype));
        result = RudaAscend::float_mask_where(result, positive, normalized);
    }
    RudaAscend::float_reshape(result, Shape::new([batch, channels, size[0], size[1]]))
}
pub(super) fn backward(
    x: Primitive,
    grad: Primitive,
    size: [usize; 2],
    options: InterpolateOptions,
) -> Primitive {
    spatial::same_queue(&x, &[&grad]);
    let [batch, channels, height, width] = check(&x, size);
    assert_eq!(
        grad.meta.shape(),
        &Shape::new([batch, channels, size[0], size[1]]),
        "interpolation gradient shape mismatch"
    );
    let dtype = FloatDType::try_from(x.dtype).expect("floating input");
    let count = size[0] * size[1];
    let grad = RudaAscend::float_cast(grad, FloatDType::F32);
    let grad = RudaAscend::float_reshape(grad, Shape::new([batch, channels, count]));
    let mut result = RudaAscend::float_zeros(
        Shape::new([batch, channels, height * width]),
        &x.device,
        FloatDType::F32,
    );
    let n = taps(&options.mode);
    let mut total_weight =
        RudaAscend::float_zeros(Shape::new([1, 1, count]), &x.device, FloatDType::F32);
    if matches!(options.mode, InterpolateMode::Lanczos3) {
        for kh in 0..n {
            for kw in 0..n {
                let tap = tap(&x, [height, width], size, kh, kw, &options, FloatDType::F32);
                let weight = RudaAscend::float_mul(tap.x_weight, tap.y_weight);
                total_weight = RudaAscend::float_add(
                    total_weight,
                    RudaAscend::float_mask_fill(
                        weight,
                        RudaAscend::bool_not(tap.valid),
                        Scalar::Float(0.0),
                    ),
                );
            }
        }
    }
    for kh in 0..n {
        for kw in 0..n {
            let tap = tap(
                &x,
                [height, width],
                size,
                kh,
                kw,
                &options,
                if matches!(
                    options.mode,
                    InterpolateMode::Lanczos3 | InterpolateMode::Nearest
                ) {
                    FloatDType::F32
                } else {
                    dtype
                },
            );
            let indices = RudaAscend::int_reshape(tap.indices, Shape::new([1, 1, count]));
            let indices = RudaAscend::int_expand(indices, Shape::new([batch, channels, count]));
            let contribution = if matches!(options.mode, InterpolateMode::Lanczos3) {
                let inverse =
                    RudaAscend::float_div(constant(&total_weight, 1.0), total_weight.clone());
                let inverse =
                    RudaAscend::float_cast(RudaAscend::float_cast(inverse, dtype), FloatDType::F32);
                let scaled = RudaAscend::float_mul(grad.clone(), inverse);
                let nonzero = RudaAscend::bool_not(RudaAscend::float_equal_elem(
                    total_weight.clone(),
                    Scalar::Float(0.0),
                    BoolDType::U8,
                ));
                let scaled = RudaAscend::float_mask_where(grad.clone(), nonzero, scaled);
                let weight = RudaAscend::float_mul(tap.x_weight, tap.y_weight);
                let weight =
                    RudaAscend::float_cast(RudaAscend::float_cast(weight, dtype), FloatDType::F32);
                RudaAscend::float_mul(scaled, weight)
            } else {
                let x_weight = RudaAscend::float_cast(tap.x_weight, FloatDType::F32);
                let y_weight = RudaAscend::float_cast(tap.y_weight, FloatDType::F32);
                if matches!(options.mode, InterpolateMode::Bicubic) {
                    RudaAscend::float_mul(RudaAscend::float_mul(grad.clone(), y_weight), x_weight)
                } else {
                    RudaAscend::float_mul(RudaAscend::float_mul(grad.clone(), x_weight), y_weight)
                }
            };
            let contribution = RudaAscend::float_mask_fill(
                contribution,
                RudaAscend::bool_not(tap.valid),
                Scalar::Float(0.0),
            );
            result = RudaAscend::float_scatter_add(2, result, indices, contribution);
        }
    }
    RudaAscend::float_cast(
        RudaAscend::float_reshape(result, x.meta.shape().clone()),
        dtype,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn coordinates_keep_existing_nearest_bilinear_and_cubic_rules() {
        let nearest = InterpolateOptions::new(InterpolateMode::Nearest);
        assert_eq!(position(3, 3, 5, 0, &nearest).0, 1);
        let bilinear = InterpolateOptions::new(InterpolateMode::Bilinear).with_align_corners(false);
        assert_eq!(coordinate(0, 2, 4, &bilinear), 0.0);
        assert_eq!(coordinate(1, 2, 4, &bilinear), 0.25);
        let cubic = InterpolateOptions::new(InterpolateMode::Bicubic).with_align_corners(false);
        assert_eq!(coordinate(0, 2, 4, &cubic), -0.25);
        assert_eq!(position(0, 2, 4, 0, &cubic).0, 0);
        let lanczos = InterpolateOptions::new(InterpolateMode::Lanczos3).with_align_corners(false);
        assert!(!position(0, 2, 4, 0, &lanczos).2);
    }
}

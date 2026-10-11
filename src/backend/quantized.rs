use super::*;
use crate::runtime::portable::server::MemoryLayoutStrategy;
use ruda_core::tensor::{QuantLevel, QuantParam, QuantStore, QuantValue, params_shape};
mod formats;

fn packing(scheme: &QuantScheme, rank: usize) -> (usize, usize) {
    let dim = match scheme.store {
        QuantStore::PackedU32(dim) => dim,
        QuantStore::PackedNative(dim) if scheme.value == QuantValue::E2M1 => dim,
        QuantStore::Native if scheme.size_bits_value() == 8 => 0,
        _ => panic!("invalid native quantized storage for the requested value format"),
    };
    assert!(dim < rank, "quantized packing dimension out of bounds");
    (rank - dim - 1, scheme.size_bits_value())
}
fn parameter_dtype(scheme: &QuantScheme) -> FloatDType {
    match scheme.param {
        QuantParam::F32 => FloatDType::F32,
        QuantParam::F16 => FloatDType::F16,
        QuantParam::BF16 => FloatDType::BF16,
        _ => panic!("encoded quantization scales are not floating storage"),
    }
}
fn encoded_scale(scheme: &QuantScheme) -> bool {
    matches!(scheme.param, QuantParam::UE8M0 | QuantParam::UE4M3)
}
fn expanded_scales(
    mut scales: Primitive,
    shape: &Shape,
    scheme: &QuantScheme,
    dtype: FloatDType,
) -> Primitive {
    assert_eq!(
        scales.meta.shape(),
        &params_shape(shape, scheme.level),
        "quantization scale shape mismatch"
    );
    scales = if encoded_scale(scheme) {
        RudaAscend::float_cast(formats::decode_scale(scales, scheme.param), dtype)
    } else {
        RudaAscend::float_cast(scales, dtype)
    };
    if let QuantLevel::Block(block) = scheme.level {
        let blocks = block.to_dim_vec(shape.rank());
        for (axis, &size) in shape.iter().enumerate() {
            let indices: Vec<i64> = (0..size)
                .map(|i| {
                    i64::try_from(i / blocks[axis] as usize).expect("quantization index overflow")
                })
                .collect();
            let indices =
                RudaAscend::int_from_data(TensorData::new(indices, [size]), &scales.device);
            scales = RudaAscend::float_select(scales, axis, indices);
        }
    }
    scales
}
fn mask_shape(rank: usize, axis: usize, length: usize) -> Vec<usize> {
    let mut shape = vec![1; rank];
    shape[axis] = length;
    shape
}
fn signed_word(value: Primitive) -> Primitive {
    let negative =
        RudaAscend::int_greater_equal_elem(value.clone(), Scalar::Int(1i64 << 31), BoolDType::U8);
    let signed = RudaAscend::int_sub_scalar(value.clone(), Scalar::Int(1i64 << 32));
    RudaAscend::int_cast(
        RudaAscend::int_mask_where(value, negative, signed),
        IntDType::I32,
    )
}
pub(super) fn quantize(value: Primitive, scheme: &QuantScheme, scales: Primitive) -> Primitive {
    spatial::same_queue(&value, &[&scales]);
    let shape = value.meta.shape().clone();
    packing(scheme, shape.rank());
    let dtype = FloatDType::try_from(value.dtype).expect("quantization requires floating input");
    let scales = if encoded_scale(scheme) {
        formats::encode_scale(scales, scheme.param)
    } else {
        RudaAscend::float_cast(scales, parameter_dtype(scheme))
    };
    let expanded = expanded_scales(scales.clone(), &shape, scheme, dtype);
    let zero_scale =
        RudaAscend::float_equal_elem(expanded.clone(), Scalar::Float(0.0), BoolDType::U8);
    let zero_value = RudaAscend::float_equal_elem(value.clone(), Scalar::Float(0.0), BoolDType::U8);
    let zero = RudaAscend::bool_and(zero_scale, zero_value);
    let scaled = RudaAscend::float_div(value.clone(), expanded);
    let scaled = RudaAscend::float_mask_where(scaled, zero, value.clone());
    let (minimum, maximum) = scheme.value.range();
    let rounded = if formats::floating(scheme.value) {
        scaled
    } else {
        unary(scaled, TensorUnaryOp::Round)
    };
    let quantized = RudaAscend::float_clamp(
        rounded,
        Scalar::Float(minimum as f64),
        Scalar::Float(maximum as f64),
    );
    let quantized = if formats::floating(scheme.value) {
        formats::encode(quantized, scheme.value)
    } else {
        RudaAscend::float_into_int(quantized, IntDType::I64)
    };
    pack(quantized, scheme, scales)
}
fn pack(quantized: Primitive, scheme: &QuantScheme, scales: Primitive) -> Primitive {
    let shape = quantized.meta.shape().clone();
    let value = quantized.clone();
    let (axis, bits) = packing(scheme, shape.rank());
    let factor = scheme.num_quants();
    let words = shape[axis].div_ceil(factor);
    let mut packed_shape = shape.clone();
    packed_shape[axis] = words;
    let mut packed = RudaAscend::int_zeros(packed_shape, &value.device, IntDType::I64);
    for lane in 0..factor {
        let positions: Vec<usize> = (0..words).map(|word| word * factor + lane).collect();
        let indices: Vec<i64> = positions
            .iter()
            .map(|&p| if p < shape[axis] { p as i64 } else { 0 })
            .collect();
        let indices = RudaAscend::int_from_data(TensorData::new(indices, [words]), &value.device);
        let lane_value = RudaAscend::int_select(quantized.clone(), axis, indices);
        let invalid: Vec<bool> = positions.iter().map(|&p| p >= shape[axis]).collect();
        let invalid = RudaAscend::bool_from_data(
            TensorData::new(invalid, mask_shape(shape.rank(), axis, words)),
            &value.device,
        );
        let lane_value = RudaAscend::int_mask_fill(lane_value, invalid, Scalar::Int(0));
        let lane_value =
            RudaAscend::bitwise_and_scalar(lane_value, Scalar::Int((1i64 << bits) - 1));
        let lane_value = RudaAscend::int_mul_scalar(lane_value, Scalar::Int(1i64 << (lane * bits)));
        packed = RudaAscend::bitwise_or(packed, lane_value);
    }
    let packed = match scheme.store {
        QuantStore::PackedU32(_) => signed_word(packed),
        QuantStore::Native if !formats::floating(scheme.value) => cast(quantized, DType::I8),
        _ => cast(packed, DType::U8),
    };
    let output = ruda_kernel::tensor::allocation::empty_qtensor(
        shape,
        *scheme,
        &value.device,
        MemoryLayoutStrategy::Contiguous,
    );
    let (mut target, mut params) = output
        .quantized_handles()
        .expect("quantized allocation has no scales");
    // The signed view changes only the ACL descriptor, never the packed bits.
    if matches!(scheme.store, QuantStore::PackedU32(_)) {
        target.dtype = DType::I32;
    }
    let mut scales = scales;
    if encoded_scale(scheme) {
        params.dtype = DType::U8;
        scales.dtype = DType::U8;
    }
    AscendRuntime::tensor_copy_into(&value.client, buffer(packed), buffer(target))
        .expect("quantized payload copy failed");
    AscendRuntime::tensor_copy_into(&value.client, buffer(contiguous(scales)), buffer(params))
        .expect("quantized scales copy failed");
    output
}
pub(super) fn dequantize(value: Primitive, dtype: FloatDType) -> Primitive {
    let DType::QFloat(scheme) = value.dtype else {
        return value;
    };
    if !encoded_scale(&scheme) {
        parameter_dtype(&scheme);
    }
    let scales = value.scales().expect("quantized tensor has no scales");
    let shape = value.meta.shape().clone();
    let values = unpack(value);
    let values = if formats::floating(scheme.value) {
        RudaAscend::float_cast(formats::decode(values, scheme.value), dtype)
    } else {
        RudaAscend::int_into_float(values, dtype)
    };
    let scales = expanded_scales(scales, &shape, &scheme, dtype);
    RudaAscend::float_mul(values, scales)
}
fn unpack(value: Primitive) -> Primitive {
    let DType::QFloat(scheme) = value.dtype else {
        panic!("expected quantized tensor");
    };
    let shape = value.meta.shape().clone();
    let (axis, bits) = packing(&scheme, shape.rank());
    if !encoded_scale(&scheme) {
        parameter_dtype(&scheme);
    }
    let (mut words, _) = value
        .quantized_handles()
        .expect("quantized tensor has no scales");
    if matches!(scheme.store, QuantStore::PackedU32(_)) {
        words.dtype = DType::I32;
    }
    let last = shape.rank() - 1;
    words = RudaAscend::int_swap_dims(words, axis, last);
    let moved_shape = words.meta.shape().clone();
    let word_count = moved_shape[last];
    let outer = moved_shape[..last].iter().product::<usize>();
    words = RudaAscend::int_cast(words, IntDType::I64);
    words = RudaAscend::int_reshape(words, Shape::new([outer, word_count, 1]));
    let factor = scheme.num_quants();
    let mut lanes = Vec::with_capacity(factor);
    for lane in 0..factor {
        let raw = RudaAscend::bitwise_right_shift_scalar(
            words.clone(),
            Scalar::Int((lane * bits) as i64),
        );
        let raw = RudaAscend::bitwise_and_scalar(raw, Scalar::Int((1i64 << bits) - 1));
        if formats::floating(scheme.value) {
            lanes.push(raw);
            continue;
        }
        let negative = RudaAscend::int_greater_equal_elem(
            raw.clone(),
            Scalar::Int(1i64 << (bits - 1)),
            BoolDType::U8,
        );
        let signed = RudaAscend::int_sub_scalar(raw.clone(), Scalar::Int(1i64 << bits));
        lanes.push(RudaAscend::int_mask_where(raw, negative, signed));
    }
    let values = RudaAscend::int_cat(lanes, 2);
    let mut padded_shape = moved_shape;
    padded_shape[last] = word_count * factor;
    let values = RudaAscend::int_reshape(values, padded_shape);
    let values = RudaAscend::int_swap_dims(values, axis, last);
    let slices = shape.iter().map(|&n| Slice::from(0..n)).collect::<Vec<_>>();
    RudaAscend::int_slice(values, &slices)
}
pub(super) fn flip(value: Primitive, axes: &[usize]) -> Primitive {
    let DType::QFloat(scheme) = value.dtype else {
        panic!("expected quantized tensor");
    };
    if matches!(scheme.level, QuantLevel::Block(_)) {
        let values = RudaAscend::float_flip(dequantize(value, FloatDType::F32), axes);
        return RudaAscend::quantize_dynamic(values, &scheme);
    }
    let scales = value.scales().expect("quantized tensor has no scales");
    pack(RudaAscend::int_flip(unpack(value), axes), &scheme, scales)
}
pub(super) fn select(value: Primitive, axis: usize, indices: Primitive) -> Primitive {
    spatial::same_queue(&value, &[&indices]);
    let DType::QFloat(scheme) = value.dtype else {
        panic!("expected quantized tensor");
    };
    if matches!(scheme.level, QuantLevel::Block(_)) {
        let values = RudaAscend::float_select(dequantize(value, FloatDType::F32), axis, indices);
        return RudaAscend::quantize_dynamic(values, &scheme);
    }
    let scales = value.scales().expect("quantized tensor has no scales");
    pack(
        RudaAscend::int_select(unpack(value), axis, indices),
        &scheme,
        scales,
    )
}
pub(super) fn gather(value: Primitive, axis: usize, indices: Primitive) -> Primitive {
    spatial::same_queue(&value, &[&indices]);
    let DType::QFloat(scheme) = value.dtype else {
        panic!("expected quantized tensor");
    };
    if matches!(scheme.level, QuantLevel::Block(_)) {
        let dtype = ruda_tensor::get_device_settings::<RudaAscend>(&value.device).float_dtype;
        let values = RudaAscend::float_gather(axis, dequantize(value, dtype), indices);
        return RudaAscend::quantize_dynamic(values, &scheme);
    }
    let scales = value.scales().expect("quantized tensor has no scales");
    pack(
        RudaAscend::int_gather(axis, unpack(value), indices),
        &scheme,
        scales,
    )
}
pub(super) async fn into_data(value: Primitive) -> Result<TensorData, ExecutionError> {
    let shape = value.meta.shape().clone();
    let dtype = value.dtype;
    let (mut values, scales) = value
        .quantized_handles()
        .expect("quantized tensor has no scales");
    if matches!(
        dtype,
        DType::QFloat(QuantScheme {
            store: QuantStore::PackedU32(_),
            ..
        })
    ) {
        values.dtype = DType::I32;
    }
    let mut data = ruda_kernel::tensor::readback::into_data(contiguous(values)).await?;
    let params = ruda_kernel::tensor::readback::into_data(contiguous(scales)).await?;
    data.bytes.extend_from_byte_slice(&params.bytes);
    Ok(TensorData {
        bytes: data.bytes,
        shape,
        dtype,
    })
}
pub(super) fn slice(value: Primitive, slices: &[Slice]) -> Primitive {
    let DType::QFloat(scheme) = value.dtype else {
        panic!("expected quantized tensor");
    };
    if matches!(scheme.level, QuantLevel::Block(_)) {
        let values = RudaAscend::float_slice(dequantize(value, FloatDType::F32), slices);
        return RudaAscend::quantize_dynamic(values, &scheme);
    }
    let scales = value.scales().expect("quantized tensor has no scales");
    pack(
        RudaAscend::int_slice(unpack(value), slices),
        &scheme,
        scales,
    )
}
pub(super) fn reshape_quantized(value: Primitive, shape: Shape) -> Primitive {
    let scheme = *ruda_tensor::QTensorPrimitive::scheme(&value);
    match ruda_kernel::tensor::reshape::try_q_reshape(value, shape) {
        Ok(value) => value,
        Err((value, shape)) => RudaAscend::quantize_dynamic(
            RudaAscend::float_reshape(dequantize(value, FloatDType::F32), shape),
            &scheme,
        ),
    }
}
pub(super) fn expand(value: Primitive, shape: Shape) -> Primitive {
    let DType::QFloat(scheme) = value.dtype else {
        panic!("expected quantized tensor");
    };
    let scales = value.scales().expect("quantized tensor has no scales");
    let scales = RudaAscend::float_expand(scales, params_shape(&shape, scheme.level));
    let values = RudaAscend::int_expand(unpack(value), shape);
    pack(values, &scheme, scales)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn integer_formats_preserve_packing_axis_and_parameter_precision() {
        for (value, bits) in [
            (QuantValue::Q8F, 8),
            (QuantValue::Q8S, 8),
            (QuantValue::Q4F, 4),
            (QuantValue::Q4S, 4),
            (QuantValue::Q2F, 2),
            (QuantValue::Q2S, 2),
        ] {
            let scheme = QuantScheme::default()
                .with_value(value)
                .with_store(QuantStore::PackedU32(1));
            assert_eq!(packing(&scheme, 3), (1, bits));
            assert_eq!(parameter_dtype(&scheme), FloatDType::F32);
            assert_eq!(mask_shape(3, 1, 7), vec![1, 7, 1]);
        }
    }
}

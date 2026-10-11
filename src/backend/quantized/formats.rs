use super::*;

pub(super) fn floating(value: QuantValue) -> bool {
    matches!(
        value,
        QuantValue::E4M3 | QuantValue::E5M2 | QuantValue::E2M1
    )
}
fn decoded(value: QuantValue, bits: u8) -> f32 {
    match value {
        QuantValue::E4M3 => ruda_core::e4m3::from_bits(bits).to_f32(),
        QuantValue::E5M2 => ruda_core::e5m2::from_bits(bits).to_f32(),
        QuantValue::E2M1 => ruda_core::e2m1::from_bits(bits).to_f32(),
        _ => unreachable!("checked floating quantization format"),
    }
}
fn maximum_code(value: QuantValue) -> usize {
    match value {
        QuantValue::E4M3 => 126,
        QuantValue::E5M2 => 123,
        QuantValue::E2M1 => 7,
        _ => unreachable!("checked floating quantization format"),
    }
}
fn boundaries(value: QuantValue) -> Vec<f32> {
    let maximum = maximum_code(value);
    (0..=maximum)
        .map(|code| {
            if code == maximum {
                f32::INFINITY
            } else {
                (decoded(value, code as u8) + decoded(value, (code + 1) as u8)) * 0.5
            }
        })
        .collect()
}
fn lookup(table: Primitive, indices: Primitive) -> Primitive {
    let shape = indices.meta.shape().clone();
    let count = shape.num_elements();
    let indices = RudaAscend::int_reshape(indices, Shape::new([count]));
    let values = RudaAscend::float_select(table, 0, indices);
    RudaAscend::float_reshape(values, shape)
}
pub(super) fn encode(value: Primitive, format: QuantValue) -> Primitive {
    let value = contiguous(RudaAscend::float_cast(value, FloatDType::F32));
    let mut bits = value.clone();
    bits.dtype = DType::I32;
    let bits = RudaAscend::int_cast(bits, IntDType::I64);
    let sign = RudaAscend::bitwise_right_shift_scalar(bits.clone(), Scalar::Int(31));
    let sign = RudaAscend::bitwise_and_scalar(sign, Scalar::Int(1));
    let mut sign = RudaAscend::int_mul_scalar(sign, Scalar::Int(1i64 << (format.size_bits() - 1)));
    let shape = value.meta.shape().clone();
    let magnitude = RudaAscend::float_abs(value.clone());
    let maximum = maximum_code(format);
    let table = RudaAscend::float_from_data(
        TensorData::new(boundaries(format), [maximum + 1]),
        &value.device,
    );
    let mut low = RudaAscend::int_zeros(shape.clone(), &value.device, IntDType::I64);
    let mut high = RudaAscend::int_full(
        shape,
        Scalar::Int(maximum as i64),
        &value.device,
        IntDType::I64,
    );
    for _ in 0..usize::BITS - maximum.leading_zeros() {
        let mid = RudaAscend::int_div_scalar(
            RudaAscend::int_add(low.clone(), high.clone()),
            Scalar::Int(2),
        );
        let threshold = lookup(table.clone(), mid.clone());
        let greater =
            RudaAscend::float_greater(magnitude.clone(), threshold.clone(), BoolDType::U8);
        let equal = RudaAscend::float_equal(magnitude.clone(), threshold, BoolDType::U8);
        let odd = RudaAscend::int_equal_elem(
            RudaAscend::bitwise_and_scalar(mid.clone(), Scalar::Int(1)),
            Scalar::Int(1),
            BoolDType::U8,
        );
        let right = RudaAscend::bool_or(greater, RudaAscend::bool_and(equal, odd));
        let next = RudaAscend::int_add_scalar(mid.clone(), Scalar::Int(1));
        low = RudaAscend::int_mask_where(low, right.clone(), next);
        high = RudaAscend::int_mask_where(mid, right, high);
    }
    let nan = RudaAscend::bool_not(RudaAscend::float_equal(value.clone(), value, BoolDType::U8));
    if format == QuantValue::E2M1 {
        sign = RudaAscend::int_mask_fill(sign, nan.clone(), Scalar::Int(0));
    }
    let nan_code = match format {
        QuantValue::E5M2 => RudaAscend::bitwise_or_scalar(
            RudaAscend::bitwise_and_scalar(
                RudaAscend::bitwise_right_shift_scalar(bits, Scalar::Int(21)),
                Scalar::Int(3),
            ),
            Scalar::Int(0x7e),
        ),
        QuantValue::E4M3 => scalar_like(&low, Scalar::Int(0x7f)),
        QuantValue::E2M1 => scalar_like(&low, Scalar::Int(7)),
        _ => unreachable!("checked floating quantization format"),
    };
    RudaAscend::bitwise_or(RudaAscend::int_mask_where(low, nan, nan_code), sign)
}
pub(super) fn decode(value: Primitive, format: QuantValue) -> Primitive {
    let count = 1usize << format.size_bits();
    let table: Vec<f32> = (0..count).map(|bits| decoded(format, bits as u8)).collect();
    let table = RudaAscend::float_from_data(TensorData::new(table, [count]), &value.device);
    lookup(table, value)
}
pub(super) fn encode_scale(value: Primitive, param: QuantParam) -> Primitive {
    if param == QuantParam::UE4M3 {
        return cast(encode(value, QuantValue::E4M3), DType::U8);
    }
    assert_eq!(param, QuantParam::UE8M0);
    let value = contiguous(RudaAscend::float_cast(value, FloatDType::F32));
    let mut bits = value.clone();
    bits.dtype = DType::I32;
    let bits = RudaAscend::int_cast(bits, IntDType::I64);
    let exponent = RudaAscend::bitwise_and_scalar(
        RudaAscend::bitwise_right_shift_scalar(bits.clone(), Scalar::Int(23)),
        Scalar::Int(255),
    );
    let mantissa = RudaAscend::bitwise_and_scalar(bits.clone(), Scalar::Int(0x7fffff));
    let fractional = RudaAscend::bool_not(RudaAscend::int_equal_elem(
        mantissa.clone(),
        Scalar::Int(0),
        BoolDType::U8,
    ));
    let up = RudaAscend::int_add_scalar(exponent.clone(), Scalar::Int(1));
    let mut code = RudaAscend::int_mask_where(exponent.clone(), fractional.clone(), up);
    let too_large = RudaAscend::int_greater_elem(code.clone(), Scalar::Int(254), BoolDType::U8);
    code = RudaAscend::int_mask_fill(code, too_large, Scalar::Int(254));
    let subnormal = RudaAscend::int_equal_elem(exponent.clone(), Scalar::Int(0), BoolDType::U8);
    let above_minimum =
        RudaAscend::int_greater_elem(mantissa, Scalar::Int(0x400000), BoolDType::U8);
    let smallest = RudaAscend::bool_into_int(above_minimum, IntDType::I64);
    code = RudaAscend::int_mask_where(code, subnormal, smallest);
    let negative = RudaAscend::int_lower_elem(bits, Scalar::Int(0), BoolDType::U8);
    code = RudaAscend::int_mask_fill(code, negative, Scalar::Int(0));
    let nan = RudaAscend::bool_and(
        RudaAscend::int_equal_elem(exponent, Scalar::Int(255), BoolDType::U8),
        fractional,
    );
    cast(
        RudaAscend::int_mask_fill(code, nan, Scalar::Int(255)),
        DType::U8,
    )
}
pub(super) fn decode_scale(value: Primitive, param: QuantParam) -> Primitive {
    let mut value = value;
    value.dtype = DType::U8;
    if param == QuantParam::UE4M3 {
        return decode(cast(value, DType::I64), QuantValue::E4M3);
    }
    assert_eq!(param, QuantParam::UE8M0);
    let table = (0..=255)
        .map(|code| ruda_core::ue8m0::from_bits(code).to_f32())
        .collect::<Vec<_>>();
    let table = RudaAscend::float_from_data(TensorData::new(table, [256]), &value.device);
    lookup(table, cast(value, DType::I64))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn reference(format: QuantValue, value: f32) -> u8 {
        match format {
            QuantValue::E4M3 => ruda_core::e4m3::from_f32(value).to_bits(),
            QuantValue::E5M2 => ruda_core::e5m2::from_f32(value).to_bits(),
            QuantValue::E2M1 => ruda_core::e2m1::from_f32(value).to_bits(),
            _ => unreachable!(),
        }
    }
    #[test]
    fn all_finite_codes_and_rounding_boundaries_match_ruda_formats() {
        for format in [QuantValue::E4M3, QuantValue::E5M2, QuantValue::E2M1] {
            let table = boundaries(format);
            for bits in 0..=maximum_code(format) {
                let v = decoded(format, bits as u8);
                assert_eq!(reference(format, v), bits as u8);
                assert_eq!(
                    reference(format, -v),
                    (bits | 1 << (format.size_bits() - 1)) as u8
                );
                if bits == maximum_code(format) {
                    continue;
                }
                let midpoint = table[bits];
                assert_eq!(
                    reference(format, midpoint),
                    if bits & 1 == 0 {
                        bits as u8
                    } else {
                        (bits + 1) as u8
                    }
                );
                assert_eq!(reference(format, midpoint.next_down()), bits as u8);
                assert_eq!(reference(format, midpoint.next_up()), (bits + 1) as u8);
            }
        }
    }
}

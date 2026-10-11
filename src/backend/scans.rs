use super::*;

pub(super) enum Scan {
    Product,
    Minimum,
    Maximum,
}
pub(super) fn scan(value: Primitive, dim: usize, operation: Scan) -> Primitive {
    let shape = value.meta.shape().clone();
    assert!(
        dim < shape.rank(),
        "cumulative operation axis out of bounds"
    );
    assert!(native_float(&value) || matches!(value.dtype, DType::I32 | DType::I64));
    if shape.num_elements() == 0 {
        return contiguous(value);
    }
    let mut prefix_shape = shape.clone();
    prefix_shape[dim] = 1;
    let output = filled(shape.clone(), Scalar::Int(0), &value.device, value.dtype);
    let mut prefix = filled(prefix_shape, Scalar::Int(1), &value.device, value.dtype);
    let mut slices = vec![Slice::full(); shape.rank()];
    for index in 0..shape[dim] {
        slices[dim] = Slice::from(index..index + 1);
        let part = AscendRuntime::tensor_slice(&value.client, buffer(value.clone()), &slices)
            .expect("native scan input slice failed");
        let part = wrap(&value, part);
        prefix = match operation {
            Scan::Product => binary(prefix, part, TensorBinaryOp::Mul),
            _ if index == 0 => part,
            Scan::Minimum | Scan::Maximum => {
                let op = if matches!(operation, Scan::Minimum) {
                    TensorBinaryOp::Lower
                } else {
                    TensorBinaryOp::Greater
                };
                let mask = binary(part.clone(), prefix.clone(), op);
                mask_where(prefix, mask, part)
            }
        };
        let target = AscendRuntime::tensor_slice(&value.client, buffer(output.clone()), &slices)
            .expect("native scan output slice failed");
        AscendRuntime::tensor_copy_into(&value.client, buffer(prefix.clone()), target)
            .expect("native scan prefix copy failed");
    }
    output
}

fn matmul_shape(a: &Shape, b: &Shape) -> Shape {
    assert!((2..=8).contains(&a.rank()) && (2..=8).contains(&b.rank()));
    assert_eq!(
        a[a.rank() - 1],
        b[b.rank() - 2],
        "integer matmul inner dimensions differ"
    );
    let rank = a.rank().max(b.rank());
    let mut shape = vec![1; rank];
    for (axis, size) in shape[..rank - 2].iter_mut().enumerate() {
        let lhs = if axis + a.rank() >= rank {
            a[axis + a.rank() - rank]
        } else {
            1
        };
        let rhs = if axis + b.rank() >= rank {
            b[axis + b.rank() - rank]
        } else {
            1
        };
        assert!(
            lhs == rhs || lhs == 1 || rhs == 1,
            "integer matmul batch dimensions differ"
        );
        *size = if lhs == 1 { rhs } else { lhs };
    }
    shape[rank - 2] = a[a.rank() - 2];
    shape[rank - 1] = b[b.rank() - 1];
    shape.into()
}
pub(super) fn integer_matmul(a: Primitive, b: Primitive) -> Primitive {
    spatial::same_queue(&a, &[&b]);
    assert_eq!(a.dtype, b.dtype);
    assert!(matches!(a.dtype, DType::I32 | DType::I64));
    let shape = matmul_shape(a.meta.shape(), b.meta.shape());
    let mut result = RudaAscend::int_zeros(shape, &a.device, a.dtype.into());
    let mut a_slices = vec![Slice::full(); a.meta.shape().rank()];
    let mut b_slices = vec![Slice::full(); b.meta.shape().rank()];
    for inner in 0..a.meta.shape()[a_slices.len() - 1] {
        let a_axis = a_slices.len() - 1;
        let b_axis = b_slices.len() - 2;
        a_slices[a_axis] = Slice::from(inner..inner + 1);
        b_slices[b_axis] = Slice::from(inner..inner + 1);
        let left = RudaAscend::int_slice(a.clone(), &a_slices);
        let right = RudaAscend::int_slice(b.clone(), &b_slices);
        result = RudaAscend::int_add(result, RudaAscend::int_mul(left, right));
    }
    result
}
pub(super) fn boolean_update(
    value: Primitive,
    dim: usize,
    indices: Primitive,
    source: Primitive,
    select: bool,
) -> Primitive {
    spatial::same_queue(&value, &[&indices, &source]);
    let counts = RudaAscend::int_zeros(value.meta.shape().clone(), &value.device, IntDType::I64);
    let source = RudaAscend::bool_into_int(source, IntDType::I64);
    let counts = index_add(counts, indices, source, dim, select);
    let updated = RudaAscend::bool_not(RudaAscend::int_equal_elem(
        counts,
        Scalar::Int(0),
        BoolDType::U8,
    ));
    RudaAscend::bool_or(value, updated)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn integer_matmul_keeps_broadcast_zero_and_inner_dimensions() {
        assert_eq!(
            matmul_shape(&Shape::new([2, 1, 3, 5]), &Shape::new([4, 5, 7])),
            Shape::new([2, 4, 3, 7])
        );
        assert_eq!(
            matmul_shape(&Shape::new([0, 3, 0]), &Shape::new([1, 0, 7])),
            Shape::new([0, 3, 7])
        );
        assert_eq!(
            matmul_shape(&Shape::new([3, 0]), &Shape::new([0, 7])),
            Shape::new([3, 7])
        );
        assert!(
            std::panic::catch_unwind(|| matmul_shape(&Shape::new([3, 5]), &Shape::new([4, 7])))
                .is_err()
        );
    }
}

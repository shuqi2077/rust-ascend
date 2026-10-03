//! Native dispatch for generic RUDA modules. The original Ascend alias is unchanged.
use crate::{
    Ascend,
    runtime::{AscendRuntime, TensorBuffer},
};
use ruda_core::tensor::{
    BoolDType, FloatDType, IntDType, Metadata, Shape, Slice,
    quantization::QuantScheme,
};
use ruda_tensor::{
    Backend, Distribution, ExecutionError, Scalar, TensorData,
    backend::{BackendTypes, DTypeUsageSet},
    ops::*,
    tensor::{
        BoolTensor, Device, FloatTensor, IntTensor, QuantizedTensor,
        quantization::QuantizationParametersPrimitive,
    },
};
use ruda_tensor_device::RudaTensor;

type Primitive = RudaTensor<AscendRuntime>;

pub(crate) trait AscendTensorBackend:
    Backend<
        Device = crate::runtime::AscendDevice,
        FloatTensorPrimitive = Primitive,
        IntTensorPrimitive = Primitive,
    >
{
}
impl AscendTensorBackend for Ascend {}
impl AscendTensorBackend for RudaAscend {}

/// Backend for original RUDA modules with native FP32 matmul and axis reductions.
/// Unsupported upstream kernels still fail explicitly; there is no CPU fallback.
/// Use `Autodiff<RudaAscend>` for RUDA's original module/optimizer/record contracts.
#[derive(Clone, Default, Debug)]
pub struct RudaAscend;

impl BackendTypes for RudaAscend {
    type Device = <Ascend as BackendTypes>::Device;
    type FloatTensorPrimitive = <Ascend as BackendTypes>::FloatTensorPrimitive;
    type FloatElem = <Ascend as BackendTypes>::FloatElem;
    type IntTensorPrimitive = <Ascend as BackendTypes>::IntTensorPrimitive;
    type IntElem = <Ascend as BackendTypes>::IntElem;
    type BoolTensorPrimitive = <Ascend as BackendTypes>::BoolTensorPrimitive;
    type BoolElem = <Ascend as BackendTypes>::BoolElem;
    type QuantizedTensorPrimitive = <Ascend as BackendTypes>::QuantizedTensorPrimitive;
}
impl Backend for RudaAscend {
    fn name(device: &Self::Device) -> String {
        format!("RUDA {}", Ascend::name(device))
    }
    fn seed(device: &Self::Device, seed: u64) {
        Ascend::seed(device, seed)
    }
    fn sync(device: &Self::Device) -> Result<(), ExecutionError> {
        Ascend::sync(device)
    }
    fn dtype_usage(device: &Self::Device, dtype: ruda_tensor::DType) -> DTypeUsageSet {
        Ascend::dtype_usage(device, dtype)
    }
    fn device_count(type_id: u16) -> usize {
        Ascend::device_count(type_id)
    }
    fn memory_cleanup(device: &Self::Device) {
        Ascend::memory_cleanup(device)
    }
    fn memory_persistent_allocations<
        Output: Send,
        Input: Send,
        Func: Fn(Input) -> Output + Send,
    >(
        device: &Self::Device,
        input: Input,
        func: Func,
    ) -> Output {
        Ascend::memory_persistent_allocations(device, input, func)
    }
    fn staging<'a, Iter>(data: Iter, device: &Self::Device)
    where
        Iter: Iterator<Item = &'a mut TensorData>,
    {
        Ascend::staging(data, device)
    }
}

macro_rules! forward {
    ($trait:ident; $(fn $method:ident($($argument:ident: $ty:ty),* $(,)?) -> $output:ty;)*) => {
        $(fn $method($($argument: $ty),*) -> $output {
            <Ascend as $trait<Ascend>>::$method($($argument),*)
        })*
    };
}
fn buffer(value: Primitive) -> TensorBuffer {
    TensorBuffer {
        shape: value.meta.shape().clone(),
        strides: value.meta.strides().clone(),
        dtype: value.dtype,
        handle: value.handle,
    }
}
fn wrap(reference: &Primitive, value: TensorBuffer) -> Primitive {
    Primitive::new(
        reference.client.clone(),
        value.handle,
        Metadata::new(value.shape, value.strides),
        reference.device.clone(),
        value.dtype,
    )
}
fn contiguous(value: Primitive) -> Primitive {
    ruda_kernel::tensor::contiguous::into_contiguous(value)
}
fn axis_op(value: Primitive, dim: usize, op: u8) -> Primitive {
    let last = value
        .meta
        .shape()
        .len()
        .checked_sub(1)
        .expect("a tensor needs an axis");
    assert!(dim <= last, "axis out of bounds");
    let value = contiguous(Ascend::float_swap_dims(value, dim, last));
    let output = match op {
        0 => AscendRuntime::sum_last(&value.client, buffer(value.clone())),
        1 => AscendRuntime::mean_last(&value.client, buffer(value.clone())),
        2 => AscendRuntime::max_last(&value.client, buffer(value.clone())),
        3 => AscendRuntime::softmax(&value.client, buffer(value.clone())),
        4 => AscendRuntime::log_softmax(&value.client, buffer(value.clone())),
        _ => unreachable!(),
    }
    .expect("native Ascend axis operation failed");
    Ascend::float_swap_dims(wrap(&value, output), dim, last)
}
fn piecewise(value: Primitive, op: crate::runtime::PiecewiseActivation) -> Primitive {
    let value = contiguous(value);
    let output = AscendRuntime::piecewise_activation(&value.client, buffer(value.clone()), op)
        .expect("native Ascend piecewise operation failed");
    wrap(&value, output)
}

impl FloatTensorOps<Self> for RudaAscend {
    forward! { FloatTensorOps;
        fn float_from_data(data: TensorData, device: &Device<Self>) -> FloatTensor<Self>;
        fn float_random( shape: Shape, distribution: Distribution, device: &Device<Self>, dtype: FloatDType, ) -> FloatTensor<Self>;
        fn float_into_data( tensor: FloatTensor<Self>, ) -> impl Future<Output = Result<TensorData, ExecutionError>> + Send;
        fn float_device(tensor: &FloatTensor<Self>) -> Device<Self>;
        fn float_to_device(tensor: FloatTensor<Self>, device: &Device<Self>) -> FloatTensor<Self>;
        fn float_into_int(tensor: FloatTensor<Self>, out_dtype: IntDType) -> IntTensor<Self>;
        fn float_empty(shape: Shape, device: &Device<Self>, dtype: FloatDType) -> FloatTensor<Self>;
        fn float_add(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self>;
        fn float_add_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self>;
        fn float_sub(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self>;
        fn float_sub_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self>;
        fn float_mul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self>;
        fn float_mul_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self>;
        fn float_div(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self>;
        fn float_div_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self>;
        fn float_remainder(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self>;
        fn float_remainder_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self>;
        fn float_cross(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>, dim: usize) -> FloatTensor<Self>;
        fn float_recip(tensor: FloatTensor<Self>) -> FloatTensor<Self>;
        fn float_swap_dims(tensor: FloatTensor<Self>, dim1: usize, dim2: usize) -> FloatTensor<Self>;
        fn float_permute(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self>;
        fn float_flip(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self>;
        fn float_reshape(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self>;
        fn float_gather(dim: usize, tensor: FloatTensor<Self>, indices: IntTensor<Self>) -> FloatTensor<Self>;
        fn float_scatter_add( dim: usize, tensor: FloatTensor<Self>, indices: IntTensor<Self>, value: FloatTensor<Self>, ) -> FloatTensor<Self>;
        fn float_select(tensor: FloatTensor<Self>, dim: usize, indices: IntTensor<Self>) -> FloatTensor<Self>;
        fn float_select_add( tensor: FloatTensor<Self>, dim: usize, indices: IntTensor<Self>, value: FloatTensor<Self>, ) -> FloatTensor<Self>;
        fn float_slice(tensor: FloatTensor<Self>, slices: &[Slice]) -> FloatTensor<Self>;
        fn float_slice_assign( tensor: FloatTensor<Self>, slices: &[Slice], value: FloatTensor<Self>, ) -> FloatTensor<Self>;
        fn float_mask_where( tensor: FloatTensor<Self>, mask: BoolTensor<Self>, value: FloatTensor<Self>, ) -> FloatTensor<Self>;
        fn float_mask_fill( tensor: FloatTensor<Self>, mask: BoolTensor<Self>, value: Scalar, ) -> FloatTensor<Self>;
        fn float_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>, out_dtype: BoolDType) -> BoolTensor<Self>;
        fn float_equal_elem(lhs: FloatTensor<Self>, rhs: Scalar, out_dtype: BoolDType) -> BoolTensor<Self>;
        fn float_greater( lhs: FloatTensor<Self>, rhs: FloatTensor<Self>, out_dtype: BoolDType, ) -> BoolTensor<Self>;
        fn float_greater_elem(lhs: FloatTensor<Self>, rhs: Scalar, out_dtype: BoolDType) -> BoolTensor<Self>;
        fn float_greater_equal( lhs: FloatTensor<Self>, rhs: FloatTensor<Self>, out_dtype: BoolDType, ) -> BoolTensor<Self>;
        fn float_greater_equal_elem( lhs: FloatTensor<Self>, rhs: Scalar, out_dtype: BoolDType, ) -> BoolTensor<Self>;
        fn float_lower(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>, out_dtype: BoolDType) -> BoolTensor<Self>;
        fn float_lower_elem(lhs: FloatTensor<Self>, rhs: Scalar, out_dtype: BoolDType) -> BoolTensor<Self>;
        fn float_lower_equal( lhs: FloatTensor<Self>, rhs: FloatTensor<Self>, out_dtype: BoolDType, ) -> BoolTensor<Self>;
        fn float_lower_equal_elem( lhs: FloatTensor<Self>, rhs: Scalar, out_dtype: BoolDType, ) -> BoolTensor<Self>;
        fn float_cumsum(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self>;
        fn float_cumprod(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self>;
        fn float_cummin(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self>;
        fn float_cummax(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self>;
        fn float_exp(tensor: FloatTensor<Self>) -> FloatTensor<Self>;
        fn float_log(tensor: FloatTensor<Self>) -> FloatTensor<Self>;
        fn float_log1p(tensor: FloatTensor<Self>) -> FloatTensor<Self>;
        fn float_powf(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self>;
        fn float_powf_scalar_impl(tensor: FloatTensor<Self>, value: Scalar) -> FloatTensor<Self>;
        fn float_sqrt(tensor: FloatTensor<Self>) -> FloatTensor<Self>;
        fn float_abs(tensor: FloatTensor<Self>) -> FloatTensor<Self>;
        fn float_cos(tensor: FloatTensor<Self>) -> FloatTensor<Self>;
        fn float_sin(tensor: FloatTensor<Self>) -> FloatTensor<Self>;
        fn float_tan(tensor: FloatTensor<Self>) -> FloatTensor<Self>;
        fn float_cosh(tensor: FloatTensor<Self>) -> FloatTensor<Self>;
        fn float_sinh(tensor: FloatTensor<Self>) -> FloatTensor<Self>;
        fn float_tanh(tensor: FloatTensor<Self>) -> FloatTensor<Self>;
        fn float_acos(tensor: FloatTensor<Self>) -> FloatTensor<Self>;
        fn float_acosh(tensor: FloatTensor<Self>) -> FloatTensor<Self>;
        fn float_asin(tensor: FloatTensor<Self>) -> FloatTensor<Self>;
        fn float_asinh(tensor: FloatTensor<Self>) -> FloatTensor<Self>;
        fn float_atan(tensor: FloatTensor<Self>) -> FloatTensor<Self>;
        fn float_atanh(tensor: FloatTensor<Self>) -> FloatTensor<Self>;
        fn float_atan2(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self>;
        fn float_round(tensor: FloatTensor<Self>) -> FloatTensor<Self>;
        fn float_floor(tensor: FloatTensor<Self>) -> FloatTensor<Self>;
        fn float_ceil(tensor: FloatTensor<Self>) -> FloatTensor<Self>;
        fn float_trunc(tensor: FloatTensor<Self>) -> FloatTensor<Self>;
        fn float_erf(tensor: FloatTensor<Self>) -> FloatTensor<Self>;
        fn float_argmax(tensor: FloatTensor<Self>, dim: usize, out_dtype: IntDType) -> IntTensor<Self>;
        fn float_argtopk( tensor: FloatTensor<Self>, dim: usize, k: usize, out_dtype: IntDType, ) -> IntTensor<Self>;
        fn float_argmin(tensor: FloatTensor<Self>, dim: usize, out_dtype: IntDType) -> IntTensor<Self>;
        fn float_expand(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self>;
        fn float_unfold(tensor: FloatTensor<Self>, dim: usize, size: usize, step: usize) -> FloatTensor<Self>;
    }

    fn float_matmul(a: Primitive, b: Primitive) -> Primitive {
        assert_eq!(a.device, b.device, "matmul device mismatch");
        assert!(
            a.client.same_execution_queue(&b.client),
            "matmul execution queue mismatch"
        );
        let a = contiguous(a);
        let b = contiguous(b);
        let out = AscendRuntime::matmul_fp32(&a.client, buffer(a.clone()), buffer(b))
            .expect("native FP32 matmul failed");
        wrap(&a, out)
    }
    fn float_cast(value: Primitive, dtype: FloatDType) -> Primitive {
        if value.dtype == dtype.into() {
            return value;
        }
        let value = contiguous(value);
        let out = AscendRuntime::cast(&value.client, buffer(value.clone()), dtype.into())
            .expect("native Ascend cast failed");
        wrap(&value, out)
    }
    fn float_sum(value: Primitive) -> Primitive {
        let elements = value.meta.shape().num_elements();
        if elements == 0 {
            return Self::float_zeros(Shape::new([1]), &value.device, value.dtype.into());
        }
        let value = Ascend::float_reshape(value, Shape::new([elements]));
        axis_op(value, 0, 0)
    }
    fn float_sum_dim(value: Primitive, dim: usize) -> Primitive {
        assert!(dim < value.meta.shape().len(), "axis out of bounds");
        if value.meta.shape()[dim] == 0 {
            let mut shape = value.meta.shape().to_vec();
            shape[dim] = 1;
            return Self::float_zeros(shape.into(), &value.device, value.dtype.into());
        }
        axis_op(value, dim, 0)
    }
    fn float_mean_dim(value: Primitive, dim: usize) -> Primitive {
        axis_op(value, dim, 1)
    }
    fn float_max_dim(value: Primitive, dim: usize) -> Primitive {
        axis_op(value, dim, 2)
    }
    fn float_max(value: Primitive) -> Primitive {
        let elements = value.meta.shape().num_elements();
        axis_op(Ascend::float_reshape(value, Shape::new([elements])), 0, 2)
    }
    fn float_clamp(value: Primitive, min: Scalar, max: Scalar) -> Primitive {
        piecewise(
            value,
            crate::runtime::PiecewiseActivation::Clamp {
                min: min.elem(),
                max: max.elem(),
            },
        )
    }
    fn float_clamp_min(value: Primitive, min: Scalar) -> Primitive {
        Self::float_clamp(value, min, f32::INFINITY.into())
    }
    fn float_clamp_max(value: Primitive, max: Scalar) -> Primitive {
        Self::float_clamp(value, f32::NEG_INFINITY.into(), max)
    }
    fn float_repeat_dim(value: Primitive, dim: usize, times: usize) -> Primitive {
        Ascend::float_repeat_dim(value, dim, times)
    }
}

impl IntTensorOps<Self> for RudaAscend {
    forward! { IntTensorOps;
        fn int_empty(shape: Shape, device: &Device<Self>, dtype: IntDType) -> IntTensor<Self>;
        fn int_into_data( tensor: IntTensor<Self>, ) -> impl Future<Output = Result<TensorData, ExecutionError>> + Send;
        fn int_from_data(data: TensorData, device: &Device<Self>) -> IntTensor<Self>;
        fn int_device(tensor: &IntTensor<Self>) -> Device<Self>;
        fn int_to_device(tensor: IntTensor<Self>, device: &Device<Self>) -> IntTensor<Self>;
        fn int_reshape(tensor: IntTensor<Self>, shape: Shape) -> IntTensor<Self>;
        fn int_slice(tensor: IntTensor<Self>, slices: &[Slice]) -> IntTensor<Self>;
        fn int_slice_assign( tensor: IntTensor<Self>, slices: &[Slice], value: IntTensor<Self>, ) -> IntTensor<Self>;
        fn int_into_float(tensor: IntTensor<Self>, out_dtype: FloatDType) -> FloatTensor<Self>;
        fn int_mask_where( tensor: IntTensor<Self>, mask: BoolTensor<Self>, value: IntTensor<Self>, ) -> IntTensor<Self>;
        fn int_mask_fill(tensor: IntTensor<Self>, mask: BoolTensor<Self>, value: Scalar) -> IntTensor<Self>;
        fn int_gather(dim: usize, tensor: IntTensor<Self>, indices: IntTensor<Self>) -> IntTensor<Self>;
        fn int_scatter_add( dim: usize, tensor: IntTensor<Self>, indices: IntTensor<Self>, value: IntTensor<Self>, ) -> IntTensor<Self>;
        fn int_select(tensor: IntTensor<Self>, dim: usize, indices: IntTensor<Self>) -> IntTensor<Self>;
        fn int_select_add( tensor: IntTensor<Self>, dim: usize, indices: IntTensor<Self>, value: IntTensor<Self>, ) -> IntTensor<Self>;
        fn int_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>, out_dtype: BoolDType) -> BoolTensor<Self>;
        fn int_equal_elem(lhs: IntTensor<Self>, rhs: Scalar, out_dtype: BoolDType) -> BoolTensor<Self>;
        fn int_greater(lhs: IntTensor<Self>, rhs: IntTensor<Self>, out_dtype: BoolDType) -> BoolTensor<Self>;
        fn int_greater_elem(lhs: IntTensor<Self>, rhs: Scalar, out_dtype: BoolDType) -> BoolTensor<Self>;
        fn int_greater_equal( lhs: IntTensor<Self>, rhs: IntTensor<Self>, out_dtype: BoolDType, ) -> BoolTensor<Self>;
        fn int_greater_equal_elem( lhs: IntTensor<Self>, rhs: Scalar, out_dtype: BoolDType, ) -> BoolTensor<Self>;
        fn int_lower(lhs: IntTensor<Self>, rhs: IntTensor<Self>, out_dtype: BoolDType) -> BoolTensor<Self>;
        fn int_lower_elem(lhs: IntTensor<Self>, rhs: Scalar, out_dtype: BoolDType) -> BoolTensor<Self>;
        fn int_lower_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>, out_dtype: BoolDType) -> BoolTensor<Self>;
        fn int_lower_equal_elem(lhs: IntTensor<Self>, rhs: Scalar, out_dtype: BoolDType) -> BoolTensor<Self>;
        fn int_add(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self>;
        fn int_add_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self>;
        fn int_sub(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self>;
        fn int_sub_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self>;
        fn int_mul(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self>;
        fn int_mul_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self>;
        fn int_div(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self>;
        fn int_div_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self>;
        fn int_remainder(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self>;
        fn int_remainder_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self>;
        fn int_matmul(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self>;
        fn int_sum(tensor: IntTensor<Self>) -> IntTensor<Self>;
        fn int_sum_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self>;
        fn int_prod(tensor: IntTensor<Self>) -> IntTensor<Self>;
        fn int_prod_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self>;
        fn int_mean_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self>;
        fn int_cumsum(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self>;
        fn int_cumprod(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self>;
        fn int_cummin(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self>;
        fn int_cummax(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self>;
        fn int_argmax(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self>;
        fn int_argtopk(tensor: IntTensor<Self>, dim: usize, k: usize) -> IntTensor<Self>;
        fn int_argmin(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self>;
        fn int_abs(tensor: IntTensor<Self>) -> IntTensor<Self>;
        fn int_swap_dims(tensor: IntTensor<Self>, dim1: usize, dim2: usize) -> IntTensor<Self>;
        fn int_permute(tensor: IntTensor<Self>, axes: &[usize]) -> IntTensor<Self>;
        fn int_flip(tensor: IntTensor<Self>, axes: &[usize]) -> IntTensor<Self>;
        fn int_random( shape: Shape, distribution: Distribution, device: &Device<Self>, dtype: IntDType, ) -> IntTensor<Self>;
        fn int_expand(tensor: IntTensor<Self>, shape: Shape) -> IntTensor<Self>;
        fn bitwise_and(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self>;
        fn bitwise_and_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self>;
        fn bitwise_or(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self>;
        fn bitwise_or_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self>;
        fn bitwise_xor(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self>;
        fn bitwise_xor_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self>;
        fn bitwise_not(tensor: IntTensor<Self>) -> IntTensor<Self>;
        fn bitwise_left_shift(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self>;
        fn bitwise_left_shift_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self>;
        fn bitwise_right_shift(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self>;
        fn bitwise_right_shift_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self>;
        fn int_cast(tensor: IntTensor<Self>, dtype: IntDType) -> IntTensor<Self>;
        fn int_unfold(tensor: IntTensor<Self>, dim: usize, size: usize, step: usize) -> IntTensor<Self>;
    }
}

impl BoolTensorOps<Self> for RudaAscend {
    forward! { BoolTensorOps;
        fn bool_empty(shape: Shape, device: &Device<Self>, dtype: BoolDType) -> BoolTensor<Self>;
        fn bool_zeros(shape: Shape, device: &Device<Self>, dtype: BoolDType) -> BoolTensor<Self>;
        fn bool_ones(shape: Shape, device: &Device<Self>, dtype: BoolDType) -> BoolTensor<Self>;
        fn bool_into_data( tensor: BoolTensor<Self>, ) -> impl Future<Output = Result<TensorData, ExecutionError>> + Send;
        fn bool_from_data(data: TensorData, device: &Device<Self>) -> BoolTensor<Self>;
        fn bool_into_int(tensor: BoolTensor<Self>, out_dtype: IntDType) -> IntTensor<Self>;
        fn bool_into_float(tensor: BoolTensor<Self>, out_dtype: FloatDType) -> FloatTensor<Self>;
        fn bool_device(tensor: &BoolTensor<Self>) -> Device<Self>;
        fn bool_to_device(tensor: BoolTensor<Self>, device: &Device<Self>) -> BoolTensor<Self>;
        fn bool_reshape(tensor: BoolTensor<Self>, shape: Shape) -> BoolTensor<Self>;
        fn bool_slice(tensor: BoolTensor<Self>, slices: &[Slice]) -> BoolTensor<Self>;
        fn bool_slice_assign( tensor: BoolTensor<Self>, slices: &[Slice], value: BoolTensor<Self>, ) -> BoolTensor<Self>;
        fn bool_mask_where( tensor: BoolTensor<Self>, mask: BoolTensor<Self>, value: BoolTensor<Self>, ) -> BoolTensor<Self>;
        fn bool_mask_fill(tensor: BoolTensor<Self>, mask: BoolTensor<Self>, value: Scalar) -> BoolTensor<Self>;
        fn bool_gather(dim: usize, tensor: BoolTensor<Self>, indices: IntTensor<Self>) -> BoolTensor<Self>;
        fn bool_scatter_or( dim: usize, tensor: BoolTensor<Self>, indices: IntTensor<Self>, value: BoolTensor<Self>, ) -> BoolTensor<Self>;
        fn bool_select(tensor: BoolTensor<Self>, dim: usize, indices: IntTensor<Self>) -> BoolTensor<Self>;
        fn bool_select_or( tensor: BoolTensor<Self>, dim: usize, indices: IntTensor<Self>, value: BoolTensor<Self>, ) -> BoolTensor<Self>;
        fn bool_equal(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self>;
        fn bool_equal_elem(lhs: BoolTensor<Self>, rhs: Scalar) -> BoolTensor<Self>;
        fn bool_not(tensor: BoolTensor<Self>) -> BoolTensor<Self>;
        fn bool_and(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self>;
        fn bool_or(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self>;
        fn bool_swap_dims(tensor: BoolTensor<Self>, dim1: usize, dim2: usize) -> BoolTensor<Self>;
        fn bool_permute(tensor: BoolTensor<Self>, axes: &[usize]) -> BoolTensor<Self>;
        fn bool_flip(tensor: BoolTensor<Self>, axes: &[usize]) -> BoolTensor<Self>;
        fn bool_expand(tensor: BoolTensor<Self>, shape: Shape) -> BoolTensor<Self>;
        fn bool_unfold(tensor: BoolTensor<Self>, dim: usize, size: usize, step: usize) -> BoolTensor<Self>;
    }
}

impl QTensorOps<Self> for RudaAscend {
    forward! { QTensorOps;
        fn q_from_data(data: TensorData, device: &Device<Self>) -> QuantizedTensor<Self>;
        fn dequantize(tensor: QuantizedTensor<Self>, dtype: FloatDType) -> FloatTensor<Self>;
        fn q_device(tensor: &QuantizedTensor<Self>) -> Device<Self>;
        fn q_to_device(tensor: QuantizedTensor<Self>, device: &Device<Self>) -> QuantizedTensor<Self>;
        fn q_reshape(tensor: QuantizedTensor<Self>, shape: Shape) -> QuantizedTensor<Self>;
        fn q_into_data( tensor: QuantizedTensor<Self>, ) -> impl Future<Output = Result<TensorData, ExecutionError>> + Send;
        fn q_expand(tensor: QuantizedTensor<Self>, shape: Shape) -> QuantizedTensor<Self>;
        fn q_swap_dims(tensor: QuantizedTensor<Self>, dim1: usize, dim2: usize) -> QuantizedTensor<Self>;
        fn q_permute(tensor: QuantizedTensor<Self>, axes: &[usize]) -> QuantizedTensor<Self>;
        fn q_flip(tensor: QuantizedTensor<Self>, axes: &[usize]) -> QuantizedTensor<Self>;
        fn q_select( tensor: QuantizedTensor<Self>, dim: usize, indices: IntTensor<Self>, ) -> QuantizedTensor<Self>;
        fn q_slice(tensor: QuantizedTensor<Self>, slices: &[Slice]) -> QuantizedTensor<Self>;
    }

    fn quantize(
        value: Primitive,
        scheme: &QuantScheme,
        qparams: QuantizationParametersPrimitive<Self>,
    ) -> Primitive {
        Ascend::quantize(
            value,
            scheme,
            QuantizationParametersPrimitive {
                scales: qparams.scales,
            },
        )
    }
}

impl ModuleOps<Self> for RudaAscend {
    forward! { ModuleOps;
        fn conv2d( x: FloatTensor<Self>, weight: FloatTensor<Self>, bias: Option<FloatTensor<Self>>, options: ConvOptions<2>, ) -> FloatTensor<Self>;
        fn deform_conv2d( x: FloatTensor<Self>, offset: FloatTensor<Self>, weight: FloatTensor<Self>, mask: Option<FloatTensor<Self>>, bias: Option<FloatTensor<Self>>, options: DeformConvOptions<2>, ) -> FloatTensor<Self>;
        fn conv3d( x: FloatTensor<Self>, weight: FloatTensor<Self>, bias: Option<FloatTensor<Self>>, options: ConvOptions<3>, ) -> FloatTensor<Self>;
        fn conv_transpose2d( x: FloatTensor<Self>, weight: FloatTensor<Self>, bias: Option<FloatTensor<Self>>, options: ConvTransposeOptions<2>, ) -> FloatTensor<Self>;
        fn conv_transpose3d( x: FloatTensor<Self>, weight: FloatTensor<Self>, bias: Option<FloatTensor<Self>>, options: ConvTransposeOptions<3>, ) -> FloatTensor<Self>;
        fn avg_pool2d( x: FloatTensor<Self>, kernel_size: [usize; 2], stride: [usize; 2], padding: [usize; 2], count_include_pad: bool, ceil_mode: bool, ) -> FloatTensor<Self>;
        fn avg_pool2d_backward( x: FloatTensor<Self>, grad: FloatTensor<Self>, kernel_size: [usize; 2], stride: [usize; 2], padding: [usize; 2], count_include_pad: bool, ceil_mode: bool, ) -> FloatTensor<Self>;
        fn adaptive_avg_pool2d(x: FloatTensor<Self>, output_size: [usize; 2]) -> FloatTensor<Self>;
        fn adaptive_avg_pool2d_backward(x: FloatTensor<Self>, grad: FloatTensor<Self>) -> FloatTensor<Self>;
        fn max_pool2d( x: FloatTensor<Self>, kernel_size: [usize; 2], stride: [usize; 2], padding: [usize; 2], dilation: [usize; 2], ceil_mode: bool, ) -> FloatTensor<Self>;
        fn interpolate( x: FloatTensor<Self>, output_size: [usize; 2], options: InterpolateOptions, ) -> FloatTensor<Self>;
        fn interpolate_backward( x: FloatTensor<Self>, grad: FloatTensor<Self>, output_size: [usize; 2], options: InterpolateOptions, ) -> FloatTensor<Self>;
        fn rfft( signal: FloatTensor<Self>, dim: usize, n: Option<usize>, ) -> (FloatTensor<Self>, FloatTensor<Self>);
        fn irfft( spectrum_re: FloatTensor<Self>, spectrum_im: FloatTensor<Self>, dim: usize, n: Option<usize>, ) -> FloatTensor<Self>;
    }

    fn deform_conv2d_backward(
        x: Primitive,
        offset: Primitive,
        weight: Primitive,
        mask: Option<Primitive>,
        bias: Option<Primitive>,
        output_grad: Primitive,
        options: DeformConvOptions<2>,
    ) -> DeformConv2dBackward<Self> {
        let result =
            Ascend::deform_conv2d_backward(x, offset, weight, mask, bias, output_grad, options);
        DeformConv2dBackward {
            x_grad: result.x_grad,
            offset_grad: result.offset_grad,
            weight_grad: result.weight_grad,
            mask_grad: result.mask_grad,
            bias_grad: result.bias_grad,
        }
    }
    fn max_pool2d_with_indices(
        x: Primitive,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
    ) -> MaxPool2dWithIndices<Self> {
        let out =
            Ascend::max_pool2d_with_indices(x, kernel_size, stride, padding, dilation, ceil_mode);
        MaxPool2dWithIndices {
            output: out.output,
            indices: out.indices,
        }
    }
    fn max_pool2d_with_indices_backward(
        x: Primitive,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
        output_grad: Primitive,
        indices: Primitive,
    ) -> MaxPool2dBackward<Self> {
        let out = Ascend::max_pool2d_with_indices_backward(
            x,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            output_grad,
            indices,
        );
        MaxPool2dBackward { x_grad: out.x_grad }
    }
    fn attention(
        query: Primitive,
        key: Primitive,
        value: Primitive,
        mask: Option<Primitive>,
        attn_bias: Option<Primitive>,
        options: AttentionModuleOptions,
    ) -> Primitive {
        ruda_tensor::ops::attention::attention_fallback::<Self>(
            query, key, value, mask, attn_bias, options,
        )
    }
    fn embedding(weight: Primitive, indices: Primitive) -> Primitive {
        assert_eq!(weight.device, indices.device, "embedding device mismatch");
        assert!(
            weight.client.same_execution_queue(&indices.client),
            "embedding queue mismatch"
        );
        let weight = contiguous(weight);
        let output =
            AscendRuntime::embedding(&weight.client, buffer(weight.clone()), buffer(indices))
                .expect("native embedding failed");
        wrap(&weight, output)
    }
    fn embedding_backward(weight: Primitive, grad: Primitive, indices: Primitive) -> Primitive {
        assert_eq!(
            weight.device, indices.device,
            "embedding backward device mismatch"
        );
        assert_eq!(
            weight.device, grad.device,
            "embedding backward gradient device mismatch"
        );
        assert!(
            weight.client.same_execution_queue(&grad.client)
                && weight.client.same_execution_queue(&indices.client),
            "embedding backward queue mismatch"
        );
        let grad = contiguous(grad);
        let output = AscendRuntime::embedding_backward(
            &weight.client,
            buffer(grad),
            buffer(indices),
            weight.meta.shape()[0] as u64,
            crate::runtime::EmbeddingOptions::default(),
        )
        .expect("native embedding backward failed");
        wrap(&weight, output)
    }
}

impl TransactionOps<Self> for RudaAscend {}
impl ActivationOps<Self> for RudaAscend {
    fn softmax(value: Primitive, dim: usize) -> Primitive {
        axis_op(value, dim, 3)
    }
    fn log_softmax(value: Primitive, dim: usize) -> Primitive {
        axis_op(value, dim, 4)
    }
    fn relu(value: Primitive) -> Primitive {
        piecewise(value, crate::runtime::PiecewiseActivation::Relu)
    }
    fn relu_backward(value: Primitive, grad: Primitive) -> Primitive {
        assert_eq!(value.device, grad.device, "ReLU gradient device mismatch");
        assert!(
            value.client.same_execution_queue(&grad.client),
            "ReLU gradient queue mismatch"
        );
        let value = contiguous(value);
        let grad = contiguous(grad);
        let out = AscendRuntime::piecewise_activation_backward(
            &value.client,
            buffer(value.clone()),
            buffer(grad),
            crate::runtime::PiecewiseActivation::Relu,
        )
        .expect("native ReLU backward failed");
        wrap(&value, out)
    }
}

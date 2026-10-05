//! ACLNN dispatch for typed model tensors, including strided and broadcast views.
use super::{AscendRuntime, ComputeClient, Result, TensorBuffer, WORKER, error};
use super::{
    descriptor::Descriptor,
    worker::{AscendResource, State},
};
use crate::tensor::{
    DType as CannDType, ScalarValue, TensorLayout,
    ffi::{AclIntArray, AclOpExecutor, AclScalar, AclTensor},
    layout::broadcast,
};
use ruda_core::tensor::{BoolStore, DType, IntDType, Shape, Slice, Strides};
use std::ffi::{CStr, c_void};

/// Same-dtype tensor operations. Comparisons/logical operations produce Bool(U8).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorBinaryOp {
    Equal,
    Greater,
    GreaterEqual,
    Lower,
    LowerEqual,
    And,
    Or,
    Add,
    Sub,
    Mul,
    Div,
    IntDiv,
    Remainder,
    Atan2,
    Pow,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    RightShift,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorUnaryOp {
    Neg,
    Abs,
    Exp,
    Log,
    Log1p,
    Sqrt,
    Recip,
    Sin,
    Cos,
    Tanh,
    Tan,
    Cosh,
    Sinh,
    Acos,
    Acosh,
    Asin,
    Asinh,
    Atan,
    Atanh,
    Erf,
    Relu,
    Floor,
    Ceil,
    Trunc,
    Round,
    BitwiseNot,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorReduceOp {
    Sum,
    Prod,
    Max,
    Min,
    Mean,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TensorRandomDistribution {
    Uniform { low: f64, high: f64 },
    Normal { mean: f32, std: f32 },
    Bernoulli { probability: f64 },
}

#[derive(Clone)]
enum Operation {
    Binary(TensorBinaryOp),
    ScalarBinary(TensorBinaryOp, ScalarValue),
    Unary(TensorUnaryOp),
    Clamp(ScalarValue, ScalarValue),
    Softmax {
        dim: usize,
        log: bool,
    },
    ReluBackward,
    Cross(usize),
    Not,
    Copy,
    Cast(CannDType),
    Where,
    Reduce {
        dim: usize,
        op: TensorReduceOp,
    },
    ArgReduce {
        dim: usize,
        min: bool,
        dtype: CannDType,
    },
    Cumsum(usize),
    Cummin(usize),
    Sort {
        dim: usize,
        descending: bool,
    },
    Gather(usize),
    Select(usize),
    ScatterAdd(usize),
    SelectAdd(usize),
    CopyTo(TensorLayout),
    Random {
        layout: TensorLayout,
        distribution: TensorRandomDistribution,
        seed: i64,
        offset: i64,
    },
    Fill(TensorLayout, ScalarValue),
    Arange {
        layout: TensorLayout,
        start: i64,
        end: i64,
        step: usize,
    },
}

fn dtype(value: DType) -> Result<CannDType> {
    Ok(match value {
        DType::F32 => CannDType::F32,
        DType::F16 => CannDType::F16,
        DType::BF16 => CannDType::BF16,
        DType::I32 => CannDType::I32,
        DType::I64 => CannDType::I64,
        DType::Bool(BoolStore::U8 | BoolStore::Native) => CannDType::Bool,
        _ => {
            return Err(error(
                "native model tensors require FP32/FP16/BF16/I32/I64 or byte Bool",
            ));
        }
    })
}
fn ruda_dtype(value: CannDType) -> DType {
    match value {
        CannDType::F32 => DType::F32,
        CannDType::F16 => DType::F16,
        CannDType::BF16 => DType::BF16,
        CannDType::I32 => DType::I32,
        CannDType::I64 => DType::I64,
        CannDType::Bool => DType::Bool(BoolStore::U8),
        _ => unreachable!("checked model dtype"),
    }
}
fn shape(value: &[usize]) -> Result<Vec<i64>> {
    if !(1..=8).contains(&value.len()) {
        return Err(error("native model tensors require rank 1..8"));
    }
    value
        .iter()
        .map(|&n| i64::try_from(n).map_err(error))
        .collect()
}
pub(super) fn layout(value: &TensorBuffer) -> Result<TensorLayout> {
    let dims = shape(&value.shape)?;
    let strides = value
        .strides
        .iter()
        .map(|&n| i64::try_from(n).map_err(error))
        .collect::<Result<Vec<_>>>()?;
    let layout = TensorLayout::strided(&dims, &strides, dtype(value.dtype)?)?;
    if value.handle.size_in_used() < layout.byte_len() as u64 {
        return Err(error("native tensor view exceeds its allocation"));
    }
    Ok(layout)
}
pub(super) fn allocate(
    client: &ComputeClient<AscendRuntime>,
    layout: &TensorLayout,
) -> TensorBuffer {
    TensorBuffer {
        handle: client.empty(layout.byte_len()),
        shape: Shape::from(
            layout
                .shape()
                .iter()
                .map(|&n| n as usize)
                .collect::<Vec<_>>(),
        ),
        strides: Strides::from(
            layout
                .strides()
                .iter()
                .map(|&n| n as usize)
                .collect::<Vec<_>>(),
        ),
        dtype: ruda_dtype(layout.dtype()),
    }
}
fn numeric(value: CannDType) -> bool {
    matches!(
        value,
        CannDType::F32 | CannDType::F16 | CannDType::BF16 | CannDType::I32 | CannDType::I64
    )
}
fn floating(value: CannDType) -> bool {
    matches!(value, CannDType::F32 | CannDType::F16 | CannDType::BF16)
}
fn output_layout(operation: &Operation, inputs: &[TensorLayout]) -> Result<TensorLayout> {
    let unary = || {
        inputs
            .first()
            .filter(|_| inputs.len() == 1)
            .ok_or_else(|| error("native unary tensor binding count mismatch"))
    };
    match operation {
        Operation::Binary(op) => {
            if inputs.len() != 2 || inputs[0].dtype() != inputs[1].dtype() {
                return Err(error("native binary inputs must have the same dtype"));
            }
            let kind = inputs[0].dtype();
            let logical = matches!(op, TensorBinaryOp::And | TensorBinaryOp::Or);
            let bitwise = matches!(
                op,
                TensorBinaryOp::BitwiseAnd
                    | TensorBinaryOp::BitwiseOr
                    | TensorBinaryOp::BitwiseXor
                    | TensorBinaryOp::RightShift
            );
            let arithmetic = matches!(
                op,
                TensorBinaryOp::Add
                    | TensorBinaryOp::Sub
                    | TensorBinaryOp::Mul
                    | TensorBinaryOp::Div
                    | TensorBinaryOp::IntDiv
                    | TensorBinaryOp::Remainder
                    | TensorBinaryOp::Atan2
                    | TensorBinaryOp::Pow
                    | TensorBinaryOp::BitwiseAnd
                    | TensorBinaryOp::BitwiseOr
                    | TensorBinaryOp::BitwiseXor
                    | TensorBinaryOp::RightShift
            );
            if (logical && kind != CannDType::Bool)
                || (arithmetic && !numeric(kind))
                || (matches!(
                    op,
                    TensorBinaryOp::Div | TensorBinaryOp::Pow | TensorBinaryOp::Atan2
                ) && !floating(kind))
                || (*op == TensorBinaryOp::IntDiv
                    && !matches!(kind, CannDType::I32 | CannDType::I64))
                || (bitwise && !matches!(kind, CannDType::I32 | CannDType::I64))
            {
                return Err(error("native binary tensor dtype contract mismatch"));
            }
            let kind = if arithmetic { kind } else { CannDType::Bool };
            TensorLayout::contiguous(&broadcast(inputs[0].shape(), inputs[1].shape())?, kind)
        }
        Operation::ScalarBinary(op, _) => {
            let input = unary()?;
            if !floating(input.dtype())
                || !matches!(
                    op,
                    TensorBinaryOp::Add
                        | TensorBinaryOp::Sub
                        | TensorBinaryOp::Mul
                        | TensorBinaryOp::Div
                        | TensorBinaryOp::Pow
                )
            {
                return Err(error("native scalar arithmetic requires floating tensors"));
            }
            TensorLayout::contiguous(input.shape(), input.dtype())
        }
        Operation::Unary(
            op @ (TensorUnaryOp::Neg | TensorUnaryOp::Abs | TensorUnaryOp::BitwiseNot),
        ) => {
            let input = unary()?;
            if !numeric(input.dtype())
                || (*op == TensorUnaryOp::BitwiseNot
                    && !matches!(input.dtype(), CannDType::I32 | CannDType::I64))
            {
                return Err(error("native integer unary operation dtype mismatch"));
            }
            TensorLayout::contiguous(input.shape(), input.dtype())
        }
        Operation::Unary(_) | Operation::Clamp(_, _) | Operation::Softmax { .. } => {
            let input = unary()?;
            if !floating(input.dtype()) {
                return Err(error("native floating operation dtype mismatch"));
            }
            if let Operation::Softmax { dim, .. } = operation {
                if *dim >= input.shape().len() {
                    return Err(error("softmax axis out of bounds"));
                }
            }
            TensorLayout::contiguous(input.shape(), input.dtype())
        }
        Operation::Cross(dim) => {
            if inputs.len() != 2
                || inputs[0].dtype() != inputs[1].dtype()
                || !floating(inputs[0].dtype())
                || inputs[0].shape().len() != inputs[1].shape().len()
                || inputs[0].shape().get(*dim) != Some(&3)
                || inputs[1].shape().get(*dim) != Some(&3)
            {
                return Err(error(
                    "cross requires same-width floating inputs and a size-three axis",
                ));
            }
            TensorLayout::contiguous(
                &broadcast(inputs[0].shape(), inputs[1].shape())?,
                inputs[0].dtype(),
            )
        }
        Operation::ReluBackward => {
            if inputs.len() != 2
                || !floating(inputs[0].dtype())
                || inputs[0].dtype() != inputs[1].dtype()
                || inputs[0].shape() != inputs[1].shape()
            {
                return Err(error(
                    "ReLU backward requires matching floating input and gradient",
                ));
            }
            TensorLayout::contiguous(inputs[0].shape(), inputs[0].dtype())
        }
        Operation::Not => {
            let input = unary()?;
            if input.dtype() != CannDType::Bool {
                return Err(error("logical not requires byte Bool"));
            }
            TensorLayout::contiguous(input.shape(), CannDType::Bool)
        }
        Operation::Copy => {
            let input = unary()?;
            TensorLayout::contiguous(input.shape(), input.dtype())
        }
        Operation::CopyTo(target) => {
            let input = unary()?;
            if input.dtype() != target.dtype() || input.shape() != target.shape() {
                return Err(error("slice assignment shape/dtype mismatch"));
            }
            Ok(target.clone())
        }
        Operation::Cast(kind) => TensorLayout::contiguous(unary()?.shape(), *kind),
        Operation::Where => {
            if inputs.len() != 3
                || inputs[0].dtype() != CannDType::Bool
                || inputs[1].dtype() != inputs[2].dtype()
            {
                return Err(error("where requires byte Bool and same-dtype values"));
            }
            let values = broadcast(inputs[1].shape(), inputs[2].shape())?;
            TensorLayout::contiguous(&broadcast(inputs[0].shape(), &values)?, inputs[1].dtype())
        }
        Operation::Reduce { dim, op } => {
            let input = unary()?;
            if !numeric(input.dtype())
                || *dim >= input.shape().len()
                || (*op == TensorReduceOp::Mean && !floating(input.dtype()))
            {
                return Err(error("native reduction axis/dtype mismatch"));
            }
            if input.shape()[*dim] == 0 && matches!(op, TensorReduceOp::Max | TensorReduceOp::Min) {
                return Err(error("min/max cannot reduce an empty axis"));
            }
            let mut shape = input.shape().to_vec();
            shape[*dim] = 1;
            TensorLayout::contiguous(&shape, input.dtype())
        }
        Operation::Cummin(dim) => {
            let input = unary()?;
            if *dim >= input.shape().len()
                || !matches!(
                    input.dtype(),
                    CannDType::F32 | CannDType::F16 | CannDType::BF16 | CannDType::I32
                )
            {
                return Err(error(
                    "native cumulative minimum requires FP32/FP16/BF16/I32 and a valid axis",
                ));
            }
            TensorLayout::contiguous(input.shape(), input.dtype())
        }
        Operation::Sort { dim, .. } => {
            let input = unary()?;
            if !numeric(input.dtype()) || *dim >= input.shape().len() {
                return Err(error(
                    "native sort requires a numeric tensor and a valid axis",
                ));
            }
            TensorLayout::contiguous(input.shape(), input.dtype())
        }
        Operation::Cumsum(dim) => {
            let input = unary()?;
            if !numeric(input.dtype()) || *dim >= input.shape().len() {
                return Err(error("cumulative sum axis/dtype mismatch"));
            }
            TensorLayout::contiguous(input.shape(), input.dtype())
        }
        Operation::ArgReduce { dim, dtype, .. } => {
            let input = unary()?;
            if !numeric(input.dtype())
                || !matches!(dtype, CannDType::I32 | CannDType::I64)
                || *dim >= input.shape().len()
            {
                return Err(error("arg-reduction axis/dtype mismatch"));
            }
            if input.shape()[*dim] == 0 {
                return Err(error("arg-reduction cannot reduce an empty axis"));
            }
            let mut shape = input.shape().to_vec();
            shape[*dim] = 1;
            TensorLayout::contiguous(&shape, *dtype)
        }
        Operation::Gather(dim)
        | Operation::Select(dim)
        | Operation::ScatterAdd(dim)
        | Operation::SelectAdd(dim) => {
            let update = matches!(
                operation,
                Operation::ScatterAdd(_) | Operation::SelectAdd(_)
            );
            let select = matches!(operation, Operation::Select(_) | Operation::SelectAdd(_));
            if inputs.len() != if update { 3 } else { 2 } {
                return Err(error("native indexing binding count mismatch"));
            }
            let input = &inputs[0];
            let indices = &inputs[1];
            if *dim >= input.shape().len()
                || !matches!(indices.dtype(), CannDType::I32 | CannDType::I64)
            {
                return Err(error("native indexing axis/index dtype mismatch"));
            }
            let shape = if select {
                if indices.shape().len() != 1 {
                    return Err(error("index_select needs one-dimensional indices"));
                }
                let mut shape = input.shape().to_vec();
                shape[*dim] = indices.shape()[0];
                shape
            } else {
                if indices.shape().len() != input.shape().len()
                    || indices
                        .shape()
                        .iter()
                        .zip(input.shape())
                        .enumerate()
                        .any(|(i, (&n, &d))| i != *dim && n > d)
                {
                    return Err(error("gather/scatter index shape mismatch"));
                }
                indices.shape().to_vec()
            };
            if input.shape()[*dim] == 0 && !shape.contains(&0) {
                return Err(error("cannot index a nonempty result from an empty axis"));
            }
            if update {
                if !numeric(input.dtype())
                    || inputs[2].dtype() != input.dtype()
                    || inputs[2].shape() != shape
                {
                    return Err(error("index-add/scatter-add source shape/dtype mismatch"));
                }
                TensorLayout::contiguous(input.shape(), input.dtype())
            } else {
                TensorLayout::contiguous(&shape, input.dtype())
            }
        }
        Operation::Random {
            layout: target,
            distribution,
            offset,
            ..
        } => {
            if !inputs.is_empty()
                || !matches!(
                    target.dtype(),
                    CannDType::F32 | CannDType::F16 | CannDType::BF16
                )
                || *offset < 0
                || *offset % 4 != 0
            {
                return Err(error(
                    "native random tensors require floating storage and a nonnegative four-aligned offset",
                ));
            }
            let valid = match distribution {
                TensorRandomDistribution::Uniform { low, high } => {
                    low.is_finite() && high.is_finite() && low <= high
                }
                TensorRandomDistribution::Normal { mean, std } => {
                    mean.is_finite() && std.is_finite() && *std >= 0.
                }
                TensorRandomDistribution::Bernoulli { probability } => {
                    (0. ..=1.).contains(probability)
                }
            };
            if !valid {
                return Err(error("invalid native random distribution parameters"));
            }
            Ok(target.clone())
        }
        Operation::Fill(target, _) | Operation::Arange { layout: target, .. } => {
            if !inputs.is_empty() {
                return Err(error("native tensor initialization has unexpected inputs"));
            }
            Ok(target.clone())
        }
    }
}

fn execute(
    client: &ComputeClient<AscendRuntime>,
    operation: Operation,
    inputs: &[TensorBuffer],
) -> Result<TensorBuffer> {
    let mut layouts = inputs.iter().map(layout).collect::<Result<Vec<_>>>()?;
    let target = output_layout(&operation, &layouts)?;
    let output = allocate(client, &target);
    submit(client, operation, inputs, &output, &mut layouts, target)?;
    Ok(output)
}
fn submit(
    client: &ComputeClient<AscendRuntime>,
    operation: Operation,
    inputs: &[TensorBuffer],
    output: &TensorBuffer,
    layouts: &mut Vec<TensorLayout>,
    target: TensorLayout,
) -> Result<()> {
    submit_outputs(
        client,
        operation,
        inputs,
        std::slice::from_ref(output),
        layouts,
        &[target],
    )
}
fn submit_outputs(
    client: &ComputeClient<AscendRuntime>,
    operation: Operation,
    inputs: &[TensorBuffer],
    outputs: &[TensorBuffer],
    layouts: &mut Vec<TensorLayout>,
    targets: &[TensorLayout],
) -> Result<()> {
    client.flush().map_err(error)?;
    if outputs.len() != targets.len() || outputs.is_empty() {
        return Err(error("native tensor output binding count mismatch"));
    }
    if targets.iter().all(|target| target.byte_len() == 0) {
        return Ok(());
    }
    layouts.extend_from_slice(targets);
    let guards = inputs
        .iter()
        .chain(outputs)
        .map(|value| client.get_resource(value.handle.clone()).map_err(error))
        .collect::<Result<Vec<_>>>()?;
    let resources = guards
        .iter()
        .zip(layouts.iter())
        .map(|(guard, layout)| {
            let mut resource = guard.resource().clone();
            if resource.byte_len() < layout.byte_len() {
                return Err(error("native tensor resource is too short"));
            }
            resource.size = layout.byte_len();
            Ok(resource)
        })
        .collect::<Result<Vec<_>>>()?;
    let result = WORKER
        .get()
        .ok_or_else(|| error("Ascend runtime is not initialized"))?
        .1
        .call({
            let layouts = layouts.clone();
            move |state| state.typed_tensor(operation, layouts, resources)
        });
    drop(guards);
    result
}

struct SlicePlan {
    shape: Shape,
    strides: Strides,
    offset: usize,
    reversed: Vec<usize>,
}
fn slice_plan(
    dims: &[usize],
    strides: &[usize],
    dtype: DType,
    slices: &[Slice],
) -> Result<SlicePlan> {
    if dims.len() != strides.len()
        || slices.len() > dims.len()
        || slices.iter().any(|s| s.step == 0)
    {
        return Err(error("slice rank/step mismatch"));
    }
    let mut shape = Vec::with_capacity(dims.len());
    let mut output_strides = Vec::with_capacity(dims.len());
    let mut reversed = Vec::new();
    let mut offset = 0usize;
    for (axis, (&size, &stride)) in dims.iter().zip(strides).enumerate() {
        let slice = slices.get(axis).copied().unwrap_or_else(Slice::full);
        let bounds = slice.to_range(size);
        let step = slice.step.unsigned_abs();
        let count = bounds.end.saturating_sub(bounds.start).div_ceil(step);
        shape.push(count);
        output_strides.push(
            stride
                .checked_mul(step)
                .ok_or_else(|| error("slice stride overflow"))?,
        );
        if count != 0 {
            let first = if slice.step < 0 {
                reversed.push(axis);
                bounds.end - 1 - (count - 1) * step
            } else {
                bounds.start
            };
            offset = offset
                .checked_add(
                    first
                        .checked_mul(stride)
                        .ok_or_else(|| error("slice offset overflow"))?,
                )
                .ok_or_else(|| error("slice offset overflow"))?;
        }
    }
    if shape.contains(&0) {
        offset = 0;
    }
    let offset = offset
        .checked_mul(dtype.size())
        .ok_or_else(|| error("slice byte offset overflow"))?;
    Ok(SlicePlan {
        shape: shape.into(),
        strides: output_strides.into(),
        offset,
        reversed,
    })
}

impl AscendRuntime {
    /// Same-width floating cross products, with strided and broadcast batch axes.
    pub fn tensor_cross(
        client: &ComputeClient<Self>,
        lhs: TensorBuffer,
        rhs: TensorBuffer,
        dim: usize,
    ) -> Result<TensorBuffer> {
        execute(client, Operation::Cross(dim), &[lhs, rhs])
    }
    /// FP32/FP16/BF16/I32 cumulative minima, including strided views and empty axes.
    pub fn tensor_cummin(
        client: &ComputeClient<Self>,
        input: TensorBuffer,
        dim: usize,
    ) -> Result<TensorBuffer> {
        let operation = Operation::Cummin(dim);
        let mut layouts = vec![layout(&input)?];
        let values_layout = output_layout(&operation, &layouts)?;
        let indices_layout = TensorLayout::contiguous(values_layout.shape(), CannDType::I64)?;
        let values = allocate(client, &values_layout);
        let indices = allocate(client, &indices_layout);
        submit_outputs(
            client,
            operation,
            &[input],
            &[values.clone(), indices],
            &mut layouts,
            &[values_layout, indices_layout],
        )?;
        Ok(values)
    }
    /// Stable native axis sort. Values keep their dtype; indices are I64.
    /// Equal values retain their original axis coordinates in either direction.
    pub fn tensor_sort(
        client: &ComputeClient<Self>,
        input: TensorBuffer,
        dim: usize,
        descending: bool,
    ) -> Result<(TensorBuffer, TensorBuffer)> {
        let operation = Operation::Sort { dim, descending };
        let mut layouts = vec![layout(&input)?];
        let values_layout = output_layout(&operation, &layouts)?;
        let indices_layout = TensorLayout::contiguous(values_layout.shape(), CannDType::I64)?;
        let values = allocate(client, &values_layout);
        let indices = allocate(client, &indices_layout);
        submit_outputs(
            client,
            operation,
            &[input],
            &[values.clone(), indices.clone()],
            &mut layouts,
            &[values_layout, indices_layout],
        )?;
        Ok((values, indices))
    }
    /// Inclusive cumulative sum along an axis, preserving input shape and dtype.
    pub fn tensor_cumsum(
        client: &ComputeClient<Self>,
        input: TensorBuffer,
        dim: usize,
    ) -> Result<TensorBuffer> {
        execute(client, Operation::Cumsum(dim), &[input])
    }
    pub fn tensor_arg_reduce(
        client: &ComputeClient<Self>,
        input: TensorBuffer,
        dim: usize,
        out_dtype: IntDType,
        min: bool,
    ) -> Result<TensorBuffer> {
        execute(
            client,
            Operation::ArgReduce {
                dim,
                min,
                dtype: dtype(out_dtype.into())?,
            },
            &[input],
        )
    }
    pub fn tensor_unary(
        client: &ComputeClient<Self>,
        input: TensorBuffer,
        op: TensorUnaryOp,
    ) -> Result<TensorBuffer> {
        execute(client, Operation::Unary(op), &[input])
    }
    pub fn tensor_scalar(
        client: &ComputeClient<Self>,
        input: TensorBuffer,
        value: ScalarValue,
        op: TensorBinaryOp,
    ) -> Result<TensorBuffer> {
        execute(client, Operation::ScalarBinary(op, value), &[input])
    }
    pub fn tensor_clamp(
        client: &ComputeClient<Self>,
        input: TensorBuffer,
        min: ScalarValue,
        max: ScalarValue,
    ) -> Result<TensorBuffer> {
        execute(client, Operation::Clamp(min, max), &[input])
    }
    pub fn tensor_softmax(
        client: &ComputeClient<Self>,
        input: TensorBuffer,
        dim: usize,
        log: bool,
    ) -> Result<TensorBuffer> {
        execute(client, Operation::Softmax { dim, log }, &[input])
    }
    pub fn tensor_relu_backward(
        client: &ComputeClient<Self>,
        input: TensorBuffer,
        grad: TensorBuffer,
    ) -> Result<TensorBuffer> {
        execute(client, Operation::ReluBackward, &[input, grad])
    }
    pub fn tensor_random(
        client: &ComputeClient<Self>,
        dims: Shape,
        target: DType,
        distribution: TensorRandomDistribution,
        seed: i64,
        offset: i64,
    ) -> Result<TensorBuffer> {
        let layout = TensorLayout::contiguous(&shape(&dims)?, dtype(target)?)?;
        execute(
            client,
            Operation::Random {
                layout,
                distribution,
                seed,
                offset,
            },
            &[],
        )
    }
    pub fn tensor_reduce(
        client: &ComputeClient<Self>,
        input: TensorBuffer,
        dim: usize,
        op: TensorReduceOp,
    ) -> Result<TensorBuffer> {
        let source = layout(&input)?;
        let target = output_layout(&Operation::Reduce { dim, op }, &[source.clone()])?;
        if source.shape()[dim] == 0 {
            let value = match op {
                TensorReduceOp::Prod => ScalarValue::I64(1),
                TensorReduceOp::Mean => ScalarValue::F64(f64::NAN),
                _ => ScalarValue::I64(0),
            };
            return execute(client, Operation::Fill(target, value), &[]);
        }
        execute(client, Operation::Reduce { dim, op }, &[input])
    }
    pub fn tensor_gather(
        client: &ComputeClient<Self>,
        input: TensorBuffer,
        indices: TensorBuffer,
        dim: usize,
    ) -> Result<TensorBuffer> {
        execute(client, Operation::Gather(dim), &[input, indices])
    }
    pub fn tensor_select(
        client: &ComputeClient<Self>,
        input: TensorBuffer,
        indices: TensorBuffer,
        dim: usize,
    ) -> Result<TensorBuffer> {
        execute(client, Operation::Select(dim), &[input, indices])
    }
    pub fn tensor_scatter_add(
        client: &ComputeClient<Self>,
        input: TensorBuffer,
        indices: TensorBuffer,
        source: TensorBuffer,
        dim: usize,
    ) -> Result<TensorBuffer> {
        let operation = Operation::ScatterAdd(dim);
        output_layout(
            &operation,
            &[layout(&input)?, layout(&indices)?, layout(&source)?],
        )?;
        if indices.shape.contains(&0) {
            return Self::materialize(client, input);
        }
        execute(client, operation, &[input, indices, source])
    }
    pub fn tensor_select_add(
        client: &ComputeClient<Self>,
        input: TensorBuffer,
        indices: TensorBuffer,
        source: TensorBuffer,
        dim: usize,
    ) -> Result<TensorBuffer> {
        let operation = Operation::SelectAdd(dim);
        output_layout(
            &operation,
            &[layout(&input)?, layout(&indices)?, layout(&source)?],
        )?;
        if indices.shape.contains(&0) {
            return Self::materialize(client, input);
        }
        execute(client, operation, &[input, indices, source])
    }
    pub fn tensor_slice(
        client: &ComputeClient<Self>,
        input: TensorBuffer,
        slices: &[Slice],
    ) -> Result<TensorBuffer> {
        layout(&input)?;
        let plan = slice_plan(&input.shape, &input.strides, input.dtype, slices)?;
        if plan.shape.contains(&0) {
            return Self::tensor_full(client, plan.shape, input.dtype, ScalarValue::I64(0));
        }
        let mut output = input;
        for &axis in &plan.reversed {
            let count = plan.shape[axis];
            let bound = slices[axis].to_range(output.shape[axis]);
            let indices = Self::tensor_arange(
                client,
                0,
                i64::try_from(count).map_err(error)?,
                1,
                DType::I64,
            )?;
            let scale = Self::tensor_full(
                client,
                Shape::new([1]),
                DType::I64,
                ScalarValue::I64(slices[axis].step as i64),
            )?;
            let indices = Self::tensor_binary(client, indices, scale, TensorBinaryOp::Mul)?;
            let offset = Self::tensor_full(
                client,
                Shape::new([1]),
                DType::I64,
                ScalarValue::I64(i64::try_from(bound.end - 1).map_err(error)?),
            )?;
            let indices = Self::tensor_binary(client, indices, offset, TensorBinaryOp::Add)?;
            output = Self::tensor_select(client, output, indices, axis)?;
        }
        let mut positive = slices.to_vec();
        for &axis in &plan.reversed {
            positive[axis] = Slice::full();
        }
        let plan = slice_plan(&output.shape, &output.strides, output.dtype, &positive)?;
        let view = TensorBuffer {
            handle: output.handle.offset_start(plan.offset as u64),
            shape: plan.shape,
            strides: plan.strides,
            dtype: output.dtype,
        };
        Self::materialize(client, view)
    }
    pub fn tensor_slice_assign(
        client: &ComputeClient<Self>,
        input: TensorBuffer,
        slices: &[Slice],
        value: TensorBuffer,
    ) -> Result<TensorBuffer> {
        layout(&input)?;
        layout(&value)?;
        let plan = slice_plan(&input.shape, &input.strides, input.dtype, slices)?;
        if value.shape != plan.shape || value.dtype != input.dtype {
            return Err(error(
                "slice assignment requires the exact selected shape and dtype",
            ));
        }
        let output = Self::materialize(client, input)?;
        if plan.shape.contains(&0) {
            return Ok(output);
        }
        let plan = slice_plan(&output.shape, &output.strides, output.dtype, slices)?;
        let mut value = value;
        for &axis in &plan.reversed {
            value = Self::tensor_flip(client, value, &[axis])?;
        }
        let target = TensorBuffer {
            handle: output.handle.clone().offset_start(plan.offset as u64),
            shape: plan.shape,
            strides: plan.strides,
            dtype: output.dtype,
        };
        let mut layouts = vec![layout(&value)?];
        let target_layout = layout(&target)?;
        let operation = Operation::CopyTo(target_layout.clone());
        output_layout(&operation, &layouts)?;
        submit(
            client,
            operation,
            &[value],
            &target,
            &mut layouts,
            target_layout,
        )?;
        Ok(output)
    }
    pub fn tensor_flip(
        client: &ComputeClient<Self>,
        mut input: TensorBuffer,
        axes: &[usize],
    ) -> Result<TensorBuffer> {
        layout(&input)?;
        for &axis in axes {
            if axis >= input.shape.len() {
                return Err(error("flip axis out of bounds"));
            }
            let mut slices = vec![Slice::full(); input.shape.len()];
            slices[axis].step = -1;
            input = Self::tensor_slice(client, input, &slices)?;
        }
        Ok(input)
    }
    /// Materialize a strided/broadcast view in its original dtype on the device.
    pub fn materialize(client: &ComputeClient<Self>, input: TensorBuffer) -> Result<TensorBuffer> {
        execute(client, Operation::Copy, &[input])
    }
    /// Explicit device conversion, including integer and byte-Bool storage.
    pub fn tensor_cast(
        client: &ComputeClient<Self>,
        input: TensorBuffer,
        target: DType,
    ) -> Result<TensorBuffer> {
        let mut output = execute(client, Operation::Cast(dtype(target)?), &[input])?;
        output.dtype = target;
        Ok(output)
    }
    /// Native comparisons, byte-Bool logic and exact-width numeric add/sub/mul.
    pub fn tensor_binary(
        client: &ComputeClient<Self>,
        a: TensorBuffer,
        b: TensorBuffer,
        op: TensorBinaryOp,
    ) -> Result<TensorBuffer> {
        execute(client, Operation::Binary(op), &[a, b])
    }
    pub fn tensor_not(client: &ComputeClient<Self>, input: TensorBuffer) -> Result<TensorBuffer> {
        execute(client, Operation::Not, &[input])
    }
    /// True selects `when_true`, false selects `when_false`; all three broadcast.
    pub fn tensor_where(
        client: &ComputeClient<Self>,
        condition: TensorBuffer,
        when_true: TensorBuffer,
        when_false: TensorBuffer,
    ) -> Result<TensorBuffer> {
        execute(
            client,
            Operation::Where,
            &[condition, when_true, when_false],
        )
    }
    /// Fill on device; only the single caller-supplied scalar is host resident.
    pub fn tensor_full(
        client: &ComputeClient<Self>,
        dims: Shape,
        target: DType,
        value: ScalarValue,
    ) -> Result<TensorBuffer> {
        let layout = TensorLayout::contiguous(&shape(&dims)?, dtype(target)?)?;
        let mut output = execute(client, Operation::Fill(layout, value), &[])?;
        output.dtype = target;
        Ok(output)
    }
    /// Signed integer positions in [start,end), with a positive, nonzero step.
    pub fn tensor_arange(
        client: &ComputeClient<Self>,
        start: i64,
        end: i64,
        step: usize,
        target: DType,
    ) -> Result<TensorBuffer> {
        let target = arange_layout(start, end, step, dtype(target)?)?;
        execute(
            client,
            Operation::Arange {
                layout: target,
                start,
                end,
                step,
            },
            &[],
        )
    }
}
fn arange_layout(start: i64, end: i64, step: usize, kind: CannDType) -> Result<TensorLayout> {
    if step == 0 || !matches!(kind, CannDType::I32 | CannDType::I64) {
        return Err(error(
            "integer arange requires a positive step and I32/I64 storage",
        ));
    }
    i64::try_from(step).map_err(|_| error("ACLNN arange step exceeds signed 64-bit range"))?;
    let length = (end as i128 - start as i128).max(0);
    let count = (length + step as i128 - 1) / step as i128;
    TensorLayout::contiguous(&[i64::try_from(count).map_err(error)?], kind)
}

type BinaryPlan = unsafe extern "C" fn(
    *const AclTensor,
    *const AclTensor,
    *mut AclTensor,
    *mut u64,
    *mut *mut AclOpExecutor,
) -> i32;
type UnaryPlan = unsafe extern "C" fn(
    *const AclTensor,
    *mut AclTensor,
    *mut u64,
    *mut *mut AclOpExecutor,
) -> i32;
fn binary_symbols(op: TensorBinaryOp) -> (&'static CStr, &'static CStr) {
    match op {
        TensorBinaryOp::Equal => (c"aclnnEqTensorGetWorkspaceSize", c"aclnnEqTensor"),
        TensorBinaryOp::Greater => (c"aclnnGtTensorGetWorkspaceSize", c"aclnnGtTensor"),
        TensorBinaryOp::GreaterEqual => (c"aclnnGeTensorGetWorkspaceSize", c"aclnnGeTensor"),
        TensorBinaryOp::Lower => (c"aclnnLtTensorGetWorkspaceSize", c"aclnnLtTensor"),
        TensorBinaryOp::LowerEqual => (c"aclnnLeTensorGetWorkspaceSize", c"aclnnLeTensor"),
        TensorBinaryOp::And => (c"aclnnLogicalAndGetWorkspaceSize", c"aclnnLogicalAnd"),
        TensorBinaryOp::Or => (c"aclnnLogicalOrGetWorkspaceSize", c"aclnnLogicalOr"),
        TensorBinaryOp::Mul => (c"aclnnMulGetWorkspaceSize", c"aclnnMul"),
        TensorBinaryOp::Add => (c"aclnnAddGetWorkspaceSize", c"aclnnAdd"),
        TensorBinaryOp::Sub => (c"aclnnSubGetWorkspaceSize", c"aclnnSub"),
        TensorBinaryOp::Div => (c"aclnnDivGetWorkspaceSize", c"aclnnDiv"),
        TensorBinaryOp::IntDiv => (c"aclnnDivModGetWorkspaceSize", c"aclnnDivMod"),
        TensorBinaryOp::Remainder => (
            c"aclnnRemainderTensorTensorGetWorkspaceSize",
            c"aclnnRemainderTensorTensor",
        ),
        TensorBinaryOp::Atan2 => (c"aclnnAtan2GetWorkspaceSize", c"aclnnAtan2"),
        TensorBinaryOp::BitwiseAnd => (
            c"aclnnBitwiseAndTensorGetWorkspaceSize",
            c"aclnnBitwiseAndTensor",
        ),
        TensorBinaryOp::BitwiseOr => (
            c"aclnnBitwiseOrTensorGetWorkspaceSize",
            c"aclnnBitwiseOrTensor",
        ),
        TensorBinaryOp::BitwiseXor => (
            c"aclnnBitwiseXorTensorGetWorkspaceSize",
            c"aclnnBitwiseXorTensor",
        ),
        TensorBinaryOp::RightShift => (c"aclnnRightShiftGetWorkspaceSize", c"aclnnRightShift"),
        TensorBinaryOp::Pow => (
            c"aclnnPowTensorTensorGetWorkspaceSize",
            c"aclnnPowTensorTensor",
        ),
    }
}
fn unary_symbols(op: TensorUnaryOp) -> (&'static CStr, &'static CStr) {
    match op {
        TensorUnaryOp::Neg => (c"aclnnNegGetWorkspaceSize", c"aclnnNeg"),
        TensorUnaryOp::Abs => (c"aclnnAbsGetWorkspaceSize", c"aclnnAbs"),
        TensorUnaryOp::Exp => (c"aclnnExpGetWorkspaceSize", c"aclnnExp"),
        TensorUnaryOp::Log => (c"aclnnLogGetWorkspaceSize", c"aclnnLog"),
        TensorUnaryOp::Log1p => (c"aclnnLog1pGetWorkspaceSize", c"aclnnLog1p"),
        TensorUnaryOp::Sqrt => (c"aclnnSqrtGetWorkspaceSize", c"aclnnSqrt"),
        TensorUnaryOp::Recip => (c"aclnnReciprocalGetWorkspaceSize", c"aclnnReciprocal"),
        TensorUnaryOp::Sin => (c"aclnnSinGetWorkspaceSize", c"aclnnSin"),
        TensorUnaryOp::Cos => (c"aclnnCosGetWorkspaceSize", c"aclnnCos"),
        TensorUnaryOp::Tanh => (c"aclnnTanhGetWorkspaceSize", c"aclnnTanh"),
        TensorUnaryOp::Tan => (c"aclnnTanGetWorkspaceSize", c"aclnnTan"),
        TensorUnaryOp::Cosh => (c"aclnnCoshGetWorkspaceSize", c"aclnnCosh"),
        TensorUnaryOp::Sinh => (c"aclnnSinhGetWorkspaceSize", c"aclnnSinh"),
        TensorUnaryOp::Acos => (c"aclnnAcosGetWorkspaceSize", c"aclnnAcos"),
        TensorUnaryOp::Acosh => (c"aclnnAcoshGetWorkspaceSize", c"aclnnAcosh"),
        TensorUnaryOp::Asin => (c"aclnnAsinGetWorkspaceSize", c"aclnnAsin"),
        TensorUnaryOp::Asinh => (c"aclnnAsinhGetWorkspaceSize", c"aclnnAsinh"),
        TensorUnaryOp::Atan => (c"aclnnAtanGetWorkspaceSize", c"aclnnAtan"),
        TensorUnaryOp::Atanh => (c"aclnnAtanhGetWorkspaceSize", c"aclnnAtanh"),
        TensorUnaryOp::Erf => (c"aclnnErfGetWorkspaceSize", c"aclnnErf"),
        TensorUnaryOp::Relu => (c"aclnnReluGetWorkspaceSize", c"aclnnRelu"),
        TensorUnaryOp::Floor => (c"aclnnFloorGetWorkspaceSize", c"aclnnFloor"),
        TensorUnaryOp::Ceil => (c"aclnnCeilGetWorkspaceSize", c"aclnnCeil"),
        TensorUnaryOp::Trunc => (c"aclnnTruncGetWorkspaceSize", c"aclnnTrunc"),
        TensorUnaryOp::Round => (c"aclnnRoundGetWorkspaceSize", c"aclnnRound"),
        TensorUnaryOp::BitwiseNot => (c"aclnnBitwiseNotGetWorkspaceSize", c"aclnnBitwiseNot"),
    }
}
impl State {
    fn typed_tensor(
        &mut self,
        operation: Operation,
        layouts: Vec<TensorLayout>,
        resources: Vec<AscendResource>,
    ) -> Result<()> {
        let output_count = if matches!(&operation, Operation::Sort { .. } | Operation::Cummin(_)) {
            2
        } else {
            1
        };
        let output_index = layouts
            .len()
            .checked_sub(output_count)
            .ok_or_else(|| error("missing native tensor output"))?;
        if resources.len() != layouts.len()
            || resources
                .iter()
                .zip(&layouts)
                .any(|(r, l)| r.size != l.byte_len())
            || output_layout(&operation, &layouts[..output_index])? != layouts[output_index]
            || (output_count == 2
                && TensorLayout::contiguous(layouts[output_index].shape(), CannDType::I64)?
                    != layouts[output_index + 1])
        {
            return Err(error("native tensor layout/resource contract mismatch"));
        }
        let addresses = resources
            .iter()
            .map(|r| self.pointer(r).map(|p| p as usize))
            .collect::<Result<Vec<_>>>()?;
        ranges(&addresses, &layouts)?;
        if output_count == 2 {
            ranges(&addresses[..output_index + 1], &layouts[..output_index + 1])?;
        }
        self.session.bind()?;
        let kind = layouts[0].dtype() as i32;
        let descriptors = layouts
            .into_iter()
            .zip(addresses)
            .map(|(layout, address)| {
                // SAFETY: checked reachable spans; all guarded inputs remain alive through
                // synchronized execution. The fresh contiguous output does not alias them.
                unsafe { Descriptor::view(&self.session, layout, address as *mut c_void) }
            })
            .collect::<Result<Vec<_>>>()?;
        let handle = |i: usize| descriptors[i].handle.as_ptr();
        let out = handle(output_index);
        // SAFETY: every symbol below has the documented ACLNN ABI, not an IR fallback.
        unsafe {
            match operation {
                Operation::Cummin(dim) => {
                    type Plan = unsafe extern "C" fn(
                        *const AclTensor,
                        i64,
                        *mut AclTensor,
                        *mut AclTensor,
                        *mut u64,
                        *mut *mut AclOpExecutor,
                    ) -> i32;
                    let plan: Plan = self.session.ops.get(c"aclnnCumminGetWorkspaceSize")?;
                    let run = self.session.ops.get(c"aclnnCummin")?;
                    self.session.execute("aclnnCummin", run, |size, executor| {
                        plan(
                            handle(0),
                            dim as i64,
                            out,
                            handle(output_index + 1),
                            size,
                            executor,
                        )
                    })
                }
                Operation::Sort { dim, descending } => {
                    type Plan = unsafe extern "C" fn(
                        *const AclTensor,
                        bool,
                        i64,
                        bool,
                        *mut AclTensor,
                        *mut AclTensor,
                        *mut u64,
                        *mut *mut AclOpExecutor,
                    ) -> i32;
                    let plan: Plan = self.session.ops.get(c"aclnnSortGetWorkspaceSize")?;
                    let run = self.session.ops.get(c"aclnnSort")?;
                    self.session.execute("aclnnSort", run, |size, executor| {
                        plan(
                            handle(0),
                            true,
                            dim as i64,
                            descending,
                            out,
                            handle(output_index + 1),
                            size,
                            executor,
                        )
                    })
                }
                Operation::Cross(dim) => {
                    type Plan = unsafe extern "C" fn(
                        *const AclTensor,
                        *const AclTensor,
                        i64,
                        *mut AclTensor,
                        *mut u64,
                        *mut *mut AclOpExecutor,
                    ) -> i32;
                    let plan: Plan = self.session.ops.get(c"aclnnLinalgCrossGetWorkspaceSize")?;
                    let run = self.session.ops.get(c"aclnnLinalgCross")?;
                    self.session
                        .execute("native cross product", run, |size, executor| {
                            plan(handle(0), handle(1), dim as i64, out, size, executor)
                        })
                }
                Operation::Binary(op) => {
                    let (plan_name, run_name) = binary_symbols(op);
                    let run = self.session.ops.get(run_name)?;
                    if op == TensorBinaryOp::IntDiv {
                        type Plan = unsafe extern "C" fn(
                            *const AclTensor,
                            *const AclTensor,
                            i32,
                            *mut AclTensor,
                            *mut u64,
                            *mut *mut AclOpExecutor,
                        ) -> i32;
                        let plan: Plan = self.session.ops.get(plan_name)?;
                        self.session.execute(
                            "native truncating integer division",
                            run,
                            |size, executor| plan(handle(0), handle(1), 1, out, size, executor),
                        )
                    } else if matches!(op, TensorBinaryOp::Add | TensorBinaryOp::Sub) {
                        type Plan = unsafe extern "C" fn(
                            *const AclTensor,
                            *const AclTensor,
                            *const AclScalar,
                            *mut AclTensor,
                            *mut u64,
                            *mut *mut AclOpExecutor,
                        ) -> i32;
                        let alpha = self.session.scalar(&mut ScalarValue::I64(1))?;
                        let plan: Plan = self.session.ops.get(plan_name)?;
                        self.session
                            .execute("native add/sub", run, |size, executor| {
                                plan(
                                    handle(0),
                                    handle(1),
                                    alpha.handle.as_ptr(),
                                    out,
                                    size,
                                    executor,
                                )
                            })
                    } else {
                        let plan: BinaryPlan = self.session.ops.get(plan_name)?;
                        self.session
                            .execute("native compare/logic/mul", run, |size, executor| {
                                plan(handle(0), handle(1), out, size, executor)
                            })
                    }
                }
                Operation::ScalarBinary(op, mut value) => {
                    let value = self.session.scalar(&mut value)?;
                    if matches!(op, TensorBinaryOp::Add | TensorBinaryOp::Sub) {
                        type Plan = unsafe extern "C" fn(
                            *const AclTensor,
                            *const AclScalar,
                            *const AclScalar,
                            *mut AclTensor,
                            *mut u64,
                            *mut *mut AclOpExecutor,
                        ) -> i32;
                        let alpha = self.session.scalar(&mut ScalarValue::I64(1))?;
                        let (plan_name, run_name) = if op == TensorBinaryOp::Add {
                            (c"aclnnAddsGetWorkspaceSize", c"aclnnAdds")
                        } else {
                            (c"aclnnSubsGetWorkspaceSize", c"aclnnSubs")
                        };
                        let plan: Plan = self.session.ops.get(plan_name)?;
                        let run = self.session.ops.get(run_name)?;
                        self.session.execute(
                            "native floating add/sub scalar",
                            run,
                            |size, executor| {
                                plan(
                                    handle(0),
                                    value.handle.as_ptr(),
                                    alpha.handle.as_ptr(),
                                    out,
                                    size,
                                    executor,
                                )
                            },
                        )
                    } else {
                        type Plan = unsafe extern "C" fn(
                            *const AclTensor,
                            *const AclScalar,
                            *mut AclTensor,
                            *mut u64,
                            *mut *mut AclOpExecutor,
                        ) -> i32;
                        let (plan_name, run_name) = match op {
                            TensorBinaryOp::Mul => (c"aclnnMulsGetWorkspaceSize", c"aclnnMuls"),
                            TensorBinaryOp::Div => (c"aclnnDivsGetWorkspaceSize", c"aclnnDivs"),
                            TensorBinaryOp::Pow => (
                                c"aclnnPowTensorScalarGetWorkspaceSize",
                                c"aclnnPowTensorScalar",
                            ),
                            _ => return Err(error("unsupported floating scalar operation")),
                        };
                        let plan: Plan = self.session.ops.get(plan_name)?;
                        let run = self.session.ops.get(run_name)?;
                        self.session.execute(
                            "native floating scalar operation",
                            run,
                            |size, executor| {
                                plan(handle(0), value.handle.as_ptr(), out, size, executor)
                            },
                        )
                    }
                }
                Operation::Unary(op) => {
                    let (plan_name, run_name) = unary_symbols(op);
                    let plan: UnaryPlan = self.session.ops.get(plan_name)?;
                    let run = self.session.ops.get(run_name)?;
                    self.session.execute(
                        "native floating unary operation",
                        run,
                        |size, executor| plan(handle(0), out, size, executor),
                    )
                }
                Operation::Clamp(mut min, mut max) => {
                    type Plan = unsafe extern "C" fn(
                        *const AclTensor,
                        *const AclScalar,
                        *const AclScalar,
                        *mut AclTensor,
                        *mut u64,
                        *mut *mut AclOpExecutor,
                    ) -> i32;
                    let min = self.session.scalar(&mut min)?;
                    let max = self.session.scalar(&mut max)?;
                    let plan: Plan = self.session.ops.get(c"aclnnClampGetWorkspaceSize")?;
                    let run = self.session.ops.get(c"aclnnClamp")?;
                    self.session.execute("aclnnClamp", run, |size, executor| {
                        plan(
                            handle(0),
                            min.handle.as_ptr(),
                            max.handle.as_ptr(),
                            out,
                            size,
                            executor,
                        )
                    })
                }
                Operation::Softmax { dim, log } => {
                    type Plan = unsafe extern "C" fn(
                        *const AclTensor,
                        i64,
                        *mut AclTensor,
                        *mut u64,
                        *mut *mut AclOpExecutor,
                    ) -> i32;
                    let (plan_name, run_name) = if log {
                        (c"aclnnLogSoftmaxGetWorkspaceSize", c"aclnnLogSoftmax")
                    } else {
                        (c"aclnnSoftmaxGetWorkspaceSize", c"aclnnSoftmax")
                    };
                    let plan: Plan = self.session.ops.get(plan_name)?;
                    let run = self.session.ops.get(run_name)?;
                    self.session
                        .execute("native softmax/log-softmax", run, |size, executor| {
                            plan(handle(0), dim as i64, out, size, executor)
                        })
                }
                Operation::ReluBackward => {
                    type Plan = unsafe extern "C" fn(
                        *const AclTensor,
                        *const AclTensor,
                        *const AclScalar,
                        *mut AclTensor,
                        *mut u64,
                        *mut *mut AclOpExecutor,
                    ) -> i32;
                    let threshold = self.session.scalar(&mut ScalarValue::F64(0.))?;
                    let plan: Plan = self
                        .session
                        .ops
                        .get(c"aclnnThresholdBackwardGetWorkspaceSize")?;
                    let run = self.session.ops.get(c"aclnnThresholdBackward")?;
                    self.session
                        .execute("aclnnThresholdBackward(ReLU)", run, |size, executor| {
                            plan(
                                handle(1),
                                handle(0),
                                threshold.handle.as_ptr(),
                                out,
                                size,
                                executor,
                            )
                        })
                }
                Operation::Cumsum(dim) => {
                    type Plan = unsafe extern "C" fn(
                        *const AclTensor,
                        i64,
                        i32,
                        *mut AclTensor,
                        *mut u64,
                        *mut *mut AclOpExecutor,
                    ) -> i32;
                    let plan: Plan = self.session.ops.get(c"aclnnCumsumGetWorkspaceSize")?;
                    let run = self.session.ops.get(c"aclnnCumsum")?;
                    self.session.execute("aclnnCumsum", run, |size, executor| {
                        plan(handle(0), dim as i64, kind, out, size, executor)
                    })
                }
                Operation::ArgReduce { dim, min, .. } => {
                    type Plan = unsafe extern "C" fn(
                        *const AclTensor,
                        i64,
                        bool,
                        *mut AclTensor,
                        *mut u64,
                        *mut *mut AclOpExecutor,
                    ) -> i32;
                    let (plan_name, run_name) = if min {
                        (c"aclnnArgMinGetWorkspaceSize", c"aclnnArgMin")
                    } else {
                        (c"aclnnArgMaxGetWorkspaceSize", c"aclnnArgMax")
                    };
                    let plan: Plan = self.session.ops.get(plan_name)?;
                    let run = self.session.ops.get(run_name)?;
                    self.session
                        .execute("native argmin/argmax", run, |size, executor| {
                            plan(handle(0), dim as i64, true, out, size, executor)
                        })
                }
                Operation::Not => {
                    let plan: UnaryPlan =
                        self.session.ops.get(c"aclnnLogicalNotGetWorkspaceSize")?;
                    let run = self.session.ops.get(c"aclnnLogicalNot")?;
                    self.session
                        .execute("aclnnLogicalNot", run, |size, executor| {
                            plan(handle(0), out, size, executor)
                        })
                }
                Operation::Copy | Operation::CopyTo(_) => {
                    type Plan = unsafe extern "C" fn(
                        *mut AclTensor,
                        *const AclTensor,
                        *mut u64,
                        *mut *mut AclOpExecutor,
                    ) -> i32;
                    let plan: Plan = self.session.ops.get(c"aclnnInplaceCopyGetWorkspaceSize")?;
                    let run = self.session.ops.get(c"aclnnInplaceCopy")?;
                    self.session
                        .execute("aclnnInplaceCopy", run, |size, executor| {
                            plan(out, handle(0), size, executor)
                        })
                }
                Operation::Cast(kind) => {
                    let plan: crate::tensor::backward::CastPlan =
                        self.session.ops.get(c"aclnnCastGetWorkspaceSize")?;
                    let run = self.session.ops.get(c"aclnnCast")?;
                    self.session
                        .execute("aclnnCast(typed)", run, |size, executor| {
                            plan(handle(0), kind as i32, out, size, executor)
                        })
                }
                Operation::Where => {
                    type Plan = unsafe extern "C" fn(
                        *const AclTensor,
                        *const AclTensor,
                        *const AclTensor,
                        *mut AclTensor,
                        *mut u64,
                        *mut *mut AclOpExecutor,
                    ) -> i32;
                    let plan: Plan = self.session.ops.get(c"aclnnSWhereGetWorkspaceSize")?;
                    let run = self.session.ops.get(c"aclnnSWhere")?;
                    self.session.execute("aclnnSWhere", run, |size, executor| {
                        plan(handle(0), handle(1), handle(2), out, size, executor)
                    })
                }
                Operation::Reduce { dim, op } => match op {
                    TensorReduceOp::Sum | TensorReduceOp::Mean => {
                        type Plan = unsafe extern "C" fn(
                            *const AclTensor,
                            *const AclIntArray,
                            bool,
                            i32,
                            *mut AclTensor,
                            *mut u64,
                            *mut *mut AclOpExecutor,
                        ) -> i32;
                        let dims = self.session.int_array(&[dim as i64])?;
                        let (plan_name, run_name) = if op == TensorReduceOp::Sum {
                            (c"aclnnReduceSumGetWorkspaceSize", c"aclnnReduceSum")
                        } else {
                            (c"aclnnMeanGetWorkspaceSize", c"aclnnMean")
                        };
                        let plan: Plan = self.session.ops.get(plan_name)?;
                        let run = self.session.ops.get(run_name)?;
                        self.session
                            .execute("native sum/mean", run, |size, executor| {
                                plan(
                                    handle(0),
                                    dims.handle.as_ptr(),
                                    true,
                                    kind,
                                    out,
                                    size,
                                    executor,
                                )
                            })
                    }
                    TensorReduceOp::Prod => {
                        type Plan = unsafe extern "C" fn(
                            *const AclTensor,
                            i64,
                            bool,
                            i32,
                            *mut AclTensor,
                            *mut u64,
                            *mut *mut AclOpExecutor,
                        ) -> i32;
                        let plan: Plan = self.session.ops.get(c"aclnnProdDimGetWorkspaceSize")?;
                        let run = self.session.ops.get(c"aclnnProdDim")?;
                        self.session.execute("aclnnProdDim", run, |size, executor| {
                            plan(handle(0), dim as i64, true, kind, out, size, executor)
                        })
                    }
                    TensorReduceOp::Max | TensorReduceOp::Min => {
                        type Plan = unsafe extern "C" fn(
                            *const AclTensor,
                            *const AclIntArray,
                            bool,
                            *mut AclTensor,
                            *mut u64,
                            *mut *mut AclOpExecutor,
                        ) -> i32;
                        let dims = self.session.int_array(&[dim as i64])?;
                        let (plan_name, run_name) = if op == TensorReduceOp::Max {
                            (c"aclnnAmaxGetWorkspaceSize", c"aclnnAmax")
                        } else {
                            (c"aclnnAminGetWorkspaceSize", c"aclnnAmin")
                        };
                        let plan: Plan = self.session.ops.get(plan_name)?;
                        let run = self.session.ops.get(run_name)?;
                        self.session
                            .execute("aclnnAmax/Amin", run, |size, executor| {
                                plan(handle(0), dims.handle.as_ptr(), true, out, size, executor)
                            })
                    }
                },
                Operation::Gather(dim) | Operation::Select(dim) => {
                    type Plan = unsafe extern "C" fn(
                        *const AclTensor,
                        i64,
                        *const AclTensor,
                        *mut AclTensor,
                        *mut u64,
                        *mut *mut AclOpExecutor,
                    ) -> i32;
                    let (plan_name, run_name) = if matches!(operation, Operation::Gather(_)) {
                        (c"aclnnGatherGetWorkspaceSize", c"aclnnGather")
                    } else {
                        (c"aclnnIndexSelectGetWorkspaceSize", c"aclnnIndexSelect")
                    };
                    let plan: Plan = self.session.ops.get(plan_name)?;
                    let run = self.session.ops.get(run_name)?;
                    self.session
                        .execute("aclnnGather/IndexSelect", run, |size, executor| {
                            plan(handle(0), dim as i64, handle(1), out, size, executor)
                        })
                }
                Operation::ScatterAdd(dim) => {
                    type Plan = unsafe extern "C" fn(
                        *const AclTensor,
                        i64,
                        *const AclTensor,
                        *const AclTensor,
                        *mut AclTensor,
                        *mut u64,
                        *mut *mut AclOpExecutor,
                    ) -> i32;
                    let plan: Plan = self.session.ops.get(c"aclnnScatterAddGetWorkspaceSize")?;
                    let run = self.session.ops.get(c"aclnnScatterAdd")?;
                    self.session
                        .execute("aclnnScatterAdd", run, |size, executor| {
                            plan(
                                handle(0),
                                dim as i64,
                                handle(1),
                                handle(2),
                                out,
                                size,
                                executor,
                            )
                        })
                }
                Operation::SelectAdd(dim) => {
                    type Plan = unsafe extern "C" fn(
                        *const AclTensor,
                        i64,
                        *const AclTensor,
                        *const AclTensor,
                        *const AclScalar,
                        *mut AclTensor,
                        *mut u64,
                        *mut *mut AclOpExecutor,
                    ) -> i32;
                    let alpha = self.session.scalar(&mut ScalarValue::I64(1))?;
                    let plan: Plan = self.session.ops.get(c"aclnnIndexAddGetWorkspaceSize")?;
                    let run = self.session.ops.get(c"aclnnIndexAdd")?;
                    self.session
                        .execute("aclnnIndexAdd", run, |size, executor| {
                            plan(
                                handle(0),
                                dim as i64,
                                handle(1),
                                handle(2),
                                alpha.handle.as_ptr(),
                                out,
                                size,
                                executor,
                            )
                        })
                }
                Operation::Fill(_, mut value) => {
                    type Plan = unsafe extern "C" fn(
                        *mut AclTensor,
                        *const AclScalar,
                        *mut u64,
                        *mut *mut AclOpExecutor,
                    ) -> i32;
                    let value = self.session.scalar(&mut value)?;
                    let plan: Plan = self
                        .session
                        .ops
                        .get(c"aclnnInplaceFillScalarGetWorkspaceSize")?;
                    let run = self.session.ops.get(c"aclnnInplaceFillScalar")?;
                    self.session
                        .execute("aclnnInplaceFillScalar", run, |size, executor| {
                            plan(out, value.handle.as_ptr(), size, executor)
                        })
                }
                Operation::Random {
                    distribution,
                    seed,
                    offset,
                    ..
                } => match distribution {
                    TensorRandomDistribution::Uniform { low, high } => {
                        type Plan = unsafe extern "C" fn(
                            *const AclTensor,
                            f64,
                            f64,
                            u64,
                            u64,
                            *mut u64,
                            *mut *mut AclOpExecutor,
                        ) -> i32;
                        let plan: Plan = self
                            .session
                            .ops
                            .get(c"aclnnInplaceUniformGetWorkspaceSize")?;
                        let run = self.session.ops.get(c"aclnnInplaceUniform")?;
                        self.session
                            .execute("aclnnInplaceUniform", run, |size, executor| {
                                plan(out, low, high, seed as u64, offset as u64, size, executor)
                            })
                    }
                    TensorRandomDistribution::Normal { mean, std } => {
                        type Plan = unsafe extern "C" fn(
                            *const AclTensor,
                            f32,
                            f32,
                            i64,
                            i64,
                            *mut u64,
                            *mut *mut AclOpExecutor,
                        ) -> i32;
                        let plan: Plan = self
                            .session
                            .ops
                            .get(c"aclnnInplaceNormalGetWorkspaceSize")?;
                        let run = self.session.ops.get(c"aclnnInplaceNormal")?;
                        self.session
                            .execute("aclnnInplaceNormal", run, |size, executor| {
                                plan(out, mean, std, seed, offset, size, executor)
                            })
                    }
                    TensorRandomDistribution::Bernoulli { probability } => {
                        type Plan = unsafe extern "C" fn(
                            *const AclTensor,
                            *const AclScalar,
                            i64,
                            i64,
                            *mut u64,
                            *mut *mut AclOpExecutor,
                        ) -> i32;
                        let probability =
                            self.session.scalar(&mut ScalarValue::F64(probability))?;
                        let plan: Plan = self
                            .session
                            .ops
                            .get(c"aclnnInplaceBernoulliGetWorkspaceSize")?;
                        let run = self.session.ops.get(c"aclnnInplaceBernoulli")?;
                        self.session
                            .execute("aclnnInplaceBernoulli", run, |size, executor| {
                                plan(
                                    out,
                                    probability.handle.as_ptr(),
                                    seed,
                                    offset,
                                    size,
                                    executor,
                                )
                            })
                    }
                },
                Operation::Arange {
                    start, end, step, ..
                } => {
                    type Plan = unsafe extern "C" fn(
                        *const AclScalar,
                        *const AclScalar,
                        *const AclScalar,
                        *mut AclTensor,
                        *mut u64,
                        *mut *mut AclOpExecutor,
                    ) -> i32;
                    let start = self.session.scalar(&mut ScalarValue::I64(start))?;
                    let end = self.session.scalar(&mut ScalarValue::I64(end))?;
                    let step = self.session.scalar(&mut ScalarValue::I64(step as i64))?;
                    let plan: Plan = self.session.ops.get(c"aclnnArangeGetWorkspaceSize")?;
                    let run = self.session.ops.get(c"aclnnArange")?;
                    self.session.execute("aclnnArange", run, |size, executor| {
                        plan(
                            start.handle.as_ptr(),
                            end.handle.as_ptr(),
                            step.handle.as_ptr(),
                            out,
                            size,
                            executor,
                        )
                    })
                }
            }
        }
    }
}
fn ranges(addresses: &[usize], layouts: &[TensorLayout]) -> Result<()> {
    let ends = addresses
        .iter()
        .zip(layouts)
        .map(|(&p, layout)| {
            if p == 0 || p % layout.dtype().bytes() != 0 {
                return Err(error("native tensor address is null or unaligned"));
            }
            p.checked_add(layout.byte_len())
                .ok_or_else(|| error("native tensor address overflow"))
        })
        .collect::<Result<Vec<_>>>()?;
    let out = addresses.len() - 1;
    for i in 0..out {
        if layouts[i].byte_len() != 0 && addresses[out] < ends[i] && addresses[i] < ends[out] {
            return Err(error("native tensor output overlaps a readonly input"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cumulative_minimum_retains_compute_width_and_view_shape() {
        for dtype in [
            CannDType::F32,
            CannDType::F16,
            CannDType::BF16,
            CannDType::I32,
        ] {
            let input = TensorLayout::strided(&[4, 2], &[1, 4], dtype).unwrap();
            for dim in [0, 1] {
                assert_eq!(
                    output_layout(&Operation::Cummin(dim), &[input.clone()]).unwrap(),
                    TensorLayout::contiguous(&[4, 2], dtype).unwrap()
                );
            }
            assert!(output_layout(&Operation::Cummin(2), &[input]).is_err());
            let empty = TensorLayout::contiguous(&[2, 0], dtype).unwrap();
            assert_eq!(
                output_layout(&Operation::Cummin(1), &[empty.clone()]).unwrap(),
                empty
            );
        }
        for dtype in [CannDType::I64, CannDType::Bool] {
            assert!(
                output_layout(
                    &Operation::Cummin(0),
                    &[TensorLayout::contiguous(&[4], dtype).unwrap()]
                )
                .is_err()
            );
        }
    }
    #[test]
    fn native_integer_bits_preserve_broadcast_shape_and_exact_width() {
        for dtype in [CannDType::I32, CannDType::I64] {
            let input = TensorLayout::strided(&[3, 2], &[1, 3], dtype).unwrap();
            let mask = TensorLayout::contiguous(&[3, 1], dtype).unwrap();
            for op in [
                TensorBinaryOp::BitwiseAnd,
                TensorBinaryOp::BitwiseOr,
                TensorBinaryOp::BitwiseXor,
                TensorBinaryOp::RightShift,
            ] {
                let output =
                    output_layout(&Operation::Binary(op), &[input.clone(), mask.clone()]).unwrap();
                assert_eq!(output, TensorLayout::contiguous(&[3, 2], dtype).unwrap());
            }
            for op in [
                TensorUnaryOp::BitwiseNot,
                TensorUnaryOp::Abs,
                TensorUnaryOp::Neg,
            ] {
                assert_eq!(
                    output_layout(&Operation::Unary(op), &[input.clone()]).unwrap(),
                    TensorLayout::contiguous(&[3, 2], dtype).unwrap()
                );
            }
        }
        let float = TensorLayout::contiguous(&[2, 3], CannDType::F32).unwrap();
        assert!(
            output_layout(
                &Operation::Unary(TensorUnaryOp::BitwiseNot),
                &[float.clone()]
            )
            .is_err()
        );
        assert!(
            output_layout(
                &Operation::Binary(TensorBinaryOp::BitwiseAnd),
                &[float.clone(), float]
            )
            .is_err()
        );
    }
    #[test]
    fn native_sort_accepts_strided_numeric_axes_and_preserves_value_width() {
        for dtype in [
            CannDType::F32,
            CannDType::F16,
            CannDType::BF16,
            CannDType::I32,
            CannDType::I64,
        ] {
            let input = TensorLayout::strided(&[3, 2], &[1, 3], dtype).unwrap();
            for dim in [0, 1] {
                for descending in [false, true] {
                    let output =
                        output_layout(&Operation::Sort { dim, descending }, &[input.clone()])
                            .unwrap();
                    assert_eq!(output, TensorLayout::contiguous(&[3, 2], dtype).unwrap());
                    let indices = TensorLayout::contiguous(output.shape(), CannDType::I64).unwrap();
                    assert_eq!(indices.shape(), output.shape());
                }
            }
            assert!(
                output_layout(
                    &Operation::Sort {
                        dim: 2,
                        descending: false
                    },
                    &[input]
                )
                .is_err()
            );
            let empty = TensorLayout::contiguous(&[3, 0], dtype).unwrap();
            assert_eq!(
                output_layout(
                    &Operation::Sort {
                        dim: 1,
                        descending: true
                    },
                    &[empty.clone()]
                )
                .unwrap(),
                empty
            );
        }
        let boolean = TensorLayout::contiguous(&[2, 3], CannDType::Bool).unwrap();
        assert!(
            output_layout(
                &Operation::Sort {
                    dim: 1,
                    descending: false
                },
                &[boolean]
            )
            .is_err()
        );
    }
    #[test]
    fn cross_preserves_batch_broadcasts_and_requires_size_three_axis() {
        for kind in [CannDType::F32, CannDType::F16, CannDType::BF16] {
            let left = TensorLayout::strided(&[3, 2], &[1, 3], kind).unwrap();
            let right = TensorLayout::contiguous(&[3, 1], kind).unwrap();
            assert_eq!(
                output_layout(&Operation::Cross(0), &[left.clone(), right.clone()]).unwrap(),
                TensorLayout::contiguous(&[3, 2], kind).unwrap()
            );
            assert!(output_layout(&Operation::Cross(1), &[left.clone(), right.clone()]).is_err());
            assert!(output_layout(&Operation::Cross(2), &[left, right.clone()]).is_err());
            let empty = TensorLayout::contiguous(&[3, 0], kind).unwrap();
            assert_eq!(
                output_layout(&Operation::Cross(0), &[empty.clone(), right]).unwrap(),
                empty
            );
            let short = TensorLayout::contiguous(&[1, 2], kind).unwrap();
            let vector = TensorLayout::contiguous(&[3, 2], kind).unwrap();
            assert!(output_layout(&Operation::Cross(0), &[short, vector]).is_err());
        }
        let integer = TensorLayout::contiguous(&[3], CannDType::I32).unwrap();
        assert!(output_layout(&Operation::Cross(0), &[integer.clone(), integer]).is_err());
    }
    #[test]
    fn atan2_and_remainder_preserve_broadcast_layout_and_compute_width() {
        for kind in [
            CannDType::F32,
            CannDType::F16,
            CannDType::BF16,
            CannDType::I32,
            CannDType::I64,
            CannDType::Bool,
        ] {
            let left = TensorLayout::strided(&[3, 2], &[1, 3], kind).unwrap();
            let right = TensorLayout::contiguous(&[3, 1], kind).unwrap();
            for op in [TensorBinaryOp::Atan2, TensorBinaryOp::Remainder] {
                let result = output_layout(&Operation::Binary(op), &[left.clone(), right.clone()]);
                let supported = if op == TensorBinaryOp::Atan2 {
                    floating(kind)
                } else {
                    numeric(kind)
                };
                if supported {
                    assert_eq!(
                        result.unwrap(),
                        TensorLayout::contiguous(&[3, 2], kind).unwrap()
                    );
                    let empty = TensorLayout::contiguous(&[0, 2], kind).unwrap();
                    let scalar = TensorLayout::contiguous(&[1], kind).unwrap();
                    assert_eq!(
                        output_layout(&Operation::Binary(op), &[empty.clone(), scalar]).unwrap(),
                        empty
                    );
                } else {
                    assert!(result.is_err());
                }
            }
        }
    }
    #[test]
    fn integer_division_preserves_broadcast_shape_and_integer_width() {
        for dtype in [CannDType::I32, CannDType::I64] {
            let left = TensorLayout::strided(&[3, 2], &[1, 3], dtype).unwrap();
            let right = TensorLayout::contiguous(&[3, 1], dtype).unwrap();
            let output = output_layout(
                &Operation::Binary(TensorBinaryOp::IntDiv),
                &[left.clone(), right],
            )
            .unwrap();
            assert_eq!(output.shape(), &[3, 2]);
            assert_eq!(output.dtype(), dtype);
            let float = TensorLayout::contiguous(&[3, 2], CannDType::F32).unwrap();
            assert!(
                output_layout(&Operation::Binary(TensorBinaryOp::IntDiv), &[left, float]).is_err()
            );
        }
        let float = TensorLayout::contiguous(&[2], CannDType::F16).unwrap();
        assert!(
            output_layout(
                &Operation::Binary(TensorBinaryOp::IntDiv),
                &[float.clone(), float]
            )
            .is_err()
        );
    }
    #[test]
    fn cumsum_preserves_shape_dtype_and_accepts_empty_and_strided_inputs() {
        for dtype in [
            CannDType::F32,
            CannDType::F16,
            CannDType::BF16,
            CannDType::I32,
            CannDType::I64,
        ] {
            let input = TensorLayout::strided(&[3, 2], &[1, 3], dtype).unwrap();
            for dim in [0, 1] {
                let output = output_layout(&Operation::Cumsum(dim), &[input.clone()]).unwrap();
                assert_eq!(output.shape(), &[3, 2]);
                assert_eq!(output.dtype(), dtype);
                assert_eq!(output.strides(), &[2, 1]);
            }
            assert!(output_layout(&Operation::Cumsum(2), &[input]).is_err());
            let empty = TensorLayout::contiguous(&[2, 0], dtype).unwrap();
            let output = output_layout(&Operation::Cumsum(1), &[empty]).unwrap();
            assert_eq!(output.shape(), &[2, 0]);
            assert_eq!(output.byte_len(), 0);
        }
        let boolean = TensorLayout::contiguous(&[2, 3], CannDType::Bool).unwrap();
        assert!(output_layout(&Operation::Cumsum(1), &[boolean]).is_err());
    }
    #[test]
    fn arg_reduction_preserves_axis_and_requested_integer_width() {
        for input_dtype in [CannDType::F16, CannDType::BF16, CannDType::I64] {
            let input = TensorLayout::contiguous(&[2, 5, 3], input_dtype).unwrap();
            for out_dtype in [CannDType::I32, CannDType::I64] {
                for min in [true, false] {
                    let output = output_layout(
                        &Operation::ArgReduce {
                            dim: 1,
                            min,
                            dtype: out_dtype,
                        },
                        &[input.clone()],
                    )
                    .unwrap();
                    assert_eq!(output.shape(), &[2, 1, 3]);
                    assert_eq!(output.dtype(), out_dtype);
                }
            }
        }
        let empty = TensorLayout::contiguous(&[2, 0], CannDType::F16).unwrap();
        assert!(
            output_layout(
                &Operation::ArgReduce {
                    dim: 1,
                    min: false,
                    dtype: CannDType::I64
                },
                &[empty]
            )
            .is_err()
        );
        let input = TensorLayout::contiguous(&[2, 5], CannDType::F16).unwrap();
        assert!(
            output_layout(
                &Operation::ArgReduce {
                    dim: 2,
                    min: false,
                    dtype: CannDType::I64
                },
                &[input.clone()]
            )
            .is_err()
        );
        assert!(
            output_layout(
                &Operation::ArgReduce {
                    dim: 1,
                    min: false,
                    dtype: CannDType::F32
                },
                &[input]
            )
            .is_err()
        );
    }
    #[test]
    fn half_arithmetic_activation_and_reduction_keep_dtypes_and_shapes() {
        for dtype in [CannDType::F16, CannDType::BF16] {
            let input = TensorLayout::contiguous(&[2, 5], dtype).unwrap();
            let scalar = ScalarValue::F64(1. / 4097.);
            for op in [
                TensorBinaryOp::Add,
                TensorBinaryOp::Sub,
                TensorBinaryOp::Mul,
                TensorBinaryOp::Div,
                TensorBinaryOp::Pow,
            ] {
                assert_eq!(
                    output_layout(
                        &Operation::ScalarBinary(op, scalar.clone()),
                        &[input.clone()]
                    )
                    .unwrap(),
                    input
                );
                let rhs = TensorLayout::contiguous(&[1, 5], dtype).unwrap();
                assert_eq!(
                    output_layout(&Operation::Binary(op), &[input.clone(), rhs]).unwrap(),
                    input
                );
            }
            for op in [
                Operation::Unary(TensorUnaryOp::Sqrt),
                Operation::Unary(TensorUnaryOp::Floor),
                Operation::Unary(TensorUnaryOp::Ceil),
                Operation::Unary(TensorUnaryOp::Trunc),
                Operation::Unary(TensorUnaryOp::Round),
                Operation::Unary(TensorUnaryOp::Tan),
                Operation::Unary(TensorUnaryOp::Cosh),
                Operation::Unary(TensorUnaryOp::Sinh),
                Operation::Unary(TensorUnaryOp::Acos),
                Operation::Unary(TensorUnaryOp::Acosh),
                Operation::Unary(TensorUnaryOp::Asin),
                Operation::Unary(TensorUnaryOp::Asinh),
                Operation::Unary(TensorUnaryOp::Atan),
                Operation::Unary(TensorUnaryOp::Atanh),
                Operation::Softmax { dim: 1, log: true },
                Operation::Clamp(ScalarValue::F64(0.), ScalarValue::F64(1.)),
            ] {
                assert_eq!(output_layout(&op, &[input.clone()]).unwrap(), input);
            }
            let mean = output_layout(
                &Operation::Reduce {
                    dim: 1,
                    op: TensorReduceOp::Mean,
                },
                &[input.clone()],
            )
            .unwrap();
            assert_eq!(mean.shape(), &[2, 1]);
            assert_eq!(mean.dtype(), dtype);
            assert_eq!(
                output_layout(&Operation::ReluBackward, &[input.clone(), input.clone()]).unwrap(),
                input
            );
        }
        let int = TensorLayout::contiguous(&[2, 5], CannDType::I64).unwrap();
        assert!(
            output_layout(
                &Operation::Binary(TensorBinaryOp::Div),
                &[int.clone(), int.clone()]
            )
            .is_err()
        );
        assert!(
            output_layout(
                &Operation::Reduce {
                    dim: 1,
                    op: TensorReduceOp::Mean
                },
                &[int]
            )
            .is_err()
        );
    }
    #[test]
    fn native_random_contracts_use_floating_storage_and_aligned_stream_offsets() {
        let target = TensorLayout::contiguous(&[2, 3], CannDType::BF16).unwrap();
        let operation = |distribution, offset| Operation::Random {
            layout: target.clone(),
            distribution,
            seed: -1,
            offset,
        };
        for distribution in [
            TensorRandomDistribution::Uniform { low: -2., high: 3. },
            TensorRandomDistribution::Normal { mean: 0., std: 1. },
            TensorRandomDistribution::Bernoulli { probability: 0.5 },
        ] {
            assert_eq!(
                output_layout(&operation(distribution, 4), &[]).unwrap(),
                target
            );
            assert!(output_layout(&operation(distribution, 1), &[]).is_err());
        }
        for distribution in [
            TensorRandomDistribution::Uniform { low: 3., high: -2. },
            TensorRandomDistribution::Normal { mean: 0., std: -1. },
            TensorRandomDistribution::Bernoulli {
                probability: f64::NAN,
            },
        ] {
            assert!(output_layout(&operation(distribution, 0), &[]).is_err());
        }
    }
    #[test]
    fn typed_indexing_and_reduction_keep_original_dtypes_and_shapes() {
        let layout = |shape: &[i64], dtype| TensorLayout::contiguous(shape, dtype).unwrap();
        let values = layout(&[2, 5, 3], CannDType::I64);
        let indices = layout(&[4], CannDType::I32);
        let selected =
            output_layout(&Operation::Select(1), &[values.clone(), indices.clone()]).unwrap();
        assert_eq!(selected.shape(), &[2, 4, 3]);
        assert_eq!(selected.dtype(), CannDType::I64);
        let updated = output_layout(
            &Operation::SelectAdd(1),
            &[values.clone(), indices, selected],
        )
        .unwrap();
        assert_eq!(updated, values);
        let indices = layout(&[2, 1, 3], CannDType::I64);
        assert_eq!(
            output_layout(&Operation::Gather(1), &[values.clone(), indices])
                .unwrap()
                .shape(),
            &[2, 1, 3]
        );
        for op in [
            TensorReduceOp::Sum,
            TensorReduceOp::Prod,
            TensorReduceOp::Max,
            TensorReduceOp::Min,
        ] {
            let reduced =
                output_layout(&Operation::Reduce { dim: 1, op }, &[values.clone()]).unwrap();
            assert_eq!(reduced.shape(), &[2, 1, 3]);
            assert_eq!(reduced.dtype(), CannDType::I64);
        }
        assert!(
            output_layout(
                &Operation::Gather(1),
                &[values, layout(&[3, 1, 3], CannDType::I64)]
            )
            .is_err()
        );
        let empty = layout(&[2, 0, 3], CannDType::I32);
        assert!(
            output_layout(
                &Operation::Reduce {
                    dim: 1,
                    op: TensorReduceOp::Max
                },
                &[empty.clone()]
            )
            .is_err()
        );
        assert_eq!(
            output_layout(
                &Operation::Reduce {
                    dim: 1,
                    op: TensorReduceOp::Sum
                },
                &[empty]
            )
            .unwrap()
            .shape(),
            &[2, 1, 3]
        );
    }
    #[test]
    fn stepped_slice_views_and_reversed_assignment_match_ruda_ranges() {
        let plan = slice_plan(
            &[3, 8],
            &[8, 1],
            DType::I64,
            &[Slice::new(1, Some(3), 1), Slice::new(1, Some(8), -3)],
        )
        .unwrap();
        assert_eq!(&*plan.shape, &[2, 3]);
        assert_eq!(&*plan.strides, &[8, 3]);
        assert_eq!(plan.offset, 9 * 8);
        assert_eq!(plan.reversed, vec![1]);
        for row in 0..2 {
            for col in 0..3 {
                let physical =
                    plan.offset / 8 + row * plan.strides[0] + (2 - col) * plan.strides[1];
                assert_eq!(physical, (1 + row) * 8 + 7 - col * 3);
            }
        }
        let plan = slice_plan(
            &[3, 8],
            &[1, 3],
            DType::F32,
            &[Slice::new(-2, None, 1), Slice::new(1, Some(-1), 2)],
        )
        .unwrap();
        assert_eq!(&*plan.shape, &[2, 3]);
        assert_eq!(&*plan.strides, &[1, 6]);
        assert_eq!(plan.offset, 16);
        assert!(plan.reversed.is_empty());
        assert_eq!(
            slice_plan(&[0, 8], &[8, 1], DType::I64, &[])
                .unwrap()
                .offset,
            0
        );
        assert!(
            slice_plan(
                &[3],
                &[1],
                DType::I32,
                &[Slice {
                    start: 0,
                    end: None,
                    step: 0
                }]
            )
            .is_err()
        );
    }
    #[test]
    fn typed_broadcast_masks_and_integer_precision_contracts() {
        let int = |dims: &[i64]| TensorLayout::contiguous(dims, CannDType::I64).unwrap();
        let mask = TensorLayout::contiguous(&[1, 3, 1], CannDType::Bool).unwrap();
        let compare = output_layout(
            &Operation::Binary(TensorBinaryOp::Greater),
            &[int(&[2, 1, 5]), int(&[1, 3, 1])],
        )
        .unwrap();
        assert_eq!(compare.shape(), &[2, 3, 5]);
        assert_eq!(compare.dtype(), CannDType::Bool);
        let select = output_layout(&Operation::Where, &[mask, int(&[2, 1, 5]), int(&[1])]).unwrap();
        assert_eq!(select.shape(), &[2, 3, 5]);
        assert_eq!(select.dtype(), CannDType::I64);
        assert!(
            output_layout(
                &Operation::Binary(TensorBinaryOp::And),
                &[int(&[1]), int(&[1])]
            )
            .is_err()
        );
        assert!(
            output_layout(
                &Operation::Binary(TensorBinaryOp::Equal),
                &[int(&[2]), int(&[3])]
            )
            .is_err()
        );
        assert_eq!(
            output_layout(
                &Operation::Binary(TensorBinaryOp::Add),
                &[int(&[0, 5]), int(&[1, 5])]
            )
            .unwrap()
            .byte_len(),
            0
        );
    }
    #[test]
    fn native_views_bound_reachable_storage_and_reject_output_aliases() {
        let transposed = TensorLayout::strided(&[3, 2], &[1, 3], CannDType::I64).unwrap();
        assert_eq!(transposed.byte_len(), 48);
        let expanded = TensorLayout::strided(&[100, 3], &[0, 1], CannDType::Bool).unwrap();
        assert_eq!(expanded.byte_len(), 3);
        assert_eq!(
            TensorLayout::strided(&[0, 3], &[0, i64::MAX], CannDType::I64)
                .unwrap()
                .byte_len(),
            0
        );
        assert!(TensorLayout::strided(&[3], &[i64::MAX], CannDType::I64).is_err());
        assert!(ranges(&[1024, 1024], &[transposed.clone(), transposed.clone()]).is_err());
        assert!(ranges(&[1024, 2048], &[transposed.clone(), transposed]).is_ok());
    }
    #[test]
    fn integer_range_counts_without_float_conversion_or_signed_overflow() {
        assert_eq!(
            arange_layout(-5, 6, 3, CannDType::I64).unwrap().shape(),
            &[4]
        );
        assert_eq!(
            arange_layout(5, -5, 2, CannDType::I64).unwrap().byte_len(),
            0
        );
        assert_eq!(
            arange_layout(i64::MAX - 5, i64::MAX, 2, CannDType::I64)
                .unwrap()
                .shape(),
            &[3]
        );
        assert!(arange_layout(0, 5, 0, CannDType::I32).is_err());
        assert!(arange_layout(i64::MIN, i64::MAX, 1, CannDType::I64).is_err());
    }
}

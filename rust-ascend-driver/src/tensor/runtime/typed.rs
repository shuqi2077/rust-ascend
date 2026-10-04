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
use ruda_core::tensor::{BoolStore, DType, Shape, Slice, Strides};
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
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorReduceOp {
    Sum,
    Prod,
    Max,
    Min,
}

#[derive(Clone)]
enum Operation {
    Binary(TensorBinaryOp),
    Not,
    Copy,
    Cast(CannDType),
    Where,
    Reduce {
        dim: usize,
        op: TensorReduceOp,
    },
    Gather(usize),
    Select(usize),
    ScatterAdd(usize),
    SelectAdd(usize),
    CopyTo(TensorLayout),
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
fn layout(value: &TensorBuffer) -> Result<TensorLayout> {
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
fn allocate(client: &ComputeClient<AscendRuntime>, layout: &TensorLayout) -> TensorBuffer {
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
            let arithmetic = matches!(
                op,
                TensorBinaryOp::Add | TensorBinaryOp::Sub | TensorBinaryOp::Mul
            );
            if (logical && kind != CannDType::Bool) || (arithmetic && !numeric(kind)) {
                return Err(error("native binary tensor dtype contract mismatch"));
            }
            let kind = if arithmetic { kind } else { CannDType::Bool };
            TensorLayout::contiguous(&broadcast(inputs[0].shape(), inputs[1].shape())?, kind)
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
            if !numeric(input.dtype()) || *dim >= input.shape().len() {
                return Err(error("native reduction axis/dtype mismatch"));
            }
            if input.shape()[*dim] == 0 && matches!(op, TensorReduceOp::Max | TensorReduceOp::Min) {
                return Err(error("min/max cannot reduce an empty axis"));
            }
            let mut shape = input.shape().to_vec();
            shape[*dim] = 1;
            TensorLayout::contiguous(&shape, input.dtype())
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
    client.flush().map_err(error)?;
    if target.byte_len() == 0 {
        return Ok(());
    }
    layouts.push(target);
    let guards = inputs
        .iter()
        .chain(std::iter::once(output))
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
    pub fn tensor_reduce(
        client: &ComputeClient<Self>,
        input: TensorBuffer,
        dim: usize,
        op: TensorReduceOp,
    ) -> Result<TensorBuffer> {
        let source = layout(&input)?;
        let target = output_layout(&Operation::Reduce { dim, op }, &[source.clone()])?;
        if source.shape()[dim] == 0 {
            let value = if op == TensorReduceOp::Prod { 1 } else { 0 };
            return execute(
                client,
                Operation::Fill(target, ScalarValue::I64(value)),
                &[],
            );
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
    }
}
impl State {
    fn typed_tensor(
        &mut self,
        operation: Operation,
        layouts: Vec<TensorLayout>,
        resources: Vec<AscendResource>,
    ) -> Result<()> {
        let output_index = layouts
            .len()
            .checked_sub(1)
            .ok_or_else(|| error("missing native tensor output"))?;
        if resources.len() != layouts.len()
            || resources
                .iter()
                .zip(&layouts)
                .any(|(r, l)| r.size != l.byte_len())
            || output_layout(&operation, &layouts[..output_index])? != layouts[output_index]
        {
            return Err(error("native tensor layout/resource contract mismatch"));
        }
        let addresses = resources
            .iter()
            .map(|r| self.pointer(r).map(|p| p as usize))
            .collect::<Result<Vec<_>>>()?;
        ranges(&addresses, &layouts)?;
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
                Operation::Binary(op) => {
                    let (plan_name, run_name) = binary_symbols(op);
                    let run = self.session.ops.get(run_name)?;
                    if matches!(op, TensorBinaryOp::Add | TensorBinaryOp::Sub) {
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
                    TensorReduceOp::Sum => {
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
                        let plan: Plan = self.session.ops.get(c"aclnnReduceSumGetWorkspaceSize")?;
                        let run = self.session.ops.get(c"aclnnReduceSum")?;
                        self.session
                            .execute("aclnnReduceSum", run, |size, executor| {
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

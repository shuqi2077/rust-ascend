//! Specialize RUDA's packed scalar/metadata ABI before device lowering.
use super::{Result, invalid, unsupported};
use ruda_core::{
    arguments::{Info, Metadata as Layout},
    ir::*,
    kernel::KernelDefinition,
};
use std::collections::HashMap;
enum MetaValue { Scalar(Variable), Array(Vec<Variable>,Variable) }

/// Resolve launch-time scalar and tensor metadata values using RUDA's own ABI
/// layout. FP32 scalar bit patterns are preserved through IR reinterpretation.
pub fn specialize(
    mut kernel: KernelDefinition,
    words: &[u64],
    dynamic_word_offset: usize,
    address: StorageType,
) -> Result<KernelDefinition> {
    if !matches!(
        address,
        StorageType::Scalar(ElemType::UInt(UIntKind::U32 | UIntKind::U64))
    ) {
        return Err(unsupported("metadata address type must be u32 or u64"));
    }
    if !kernel.tensor_maps.is_empty() {
        return Err(unsupported(
            "packed arguments do not support tensor maps",
        ));
    }
    for group in &kernel.scalars {
        if !matches!(
            group.ty,
            StorageType::Scalar(
                ElemType::Float(FloatKind::F32 | FloatKind::F16 | FloatKind::BF16)
                    | ElemType::UInt(UIntKind::U8 | UIntKind::U16 | UIntKind::U32 | UIntKind::U64)
                    | ElemType::Int(IntKind::I8 | IntKind::I16 | IntKind::I32 | IntKind::I64)
                    | ElemType::Bool
            )
        ) {
            return Err(unsupported("packed scalar type"));
        }
        if group.count > words.len().saturating_mul(8) / group.ty.size() {
            return Err(invalid("scalar count exceeds argument buffer"));
        }
    }
    let ext: Vec<_> = kernel
        .buffers
        .iter()
        .filter(|b| b.has_extended_meta)
        .map(|b| b.id)
        .collect();
    let info = Info::new(
        &kernel.scalars,
        Layout::new(kernel.buffers.len() as u32, ext.len() as u32),
        address,
    );
    let bytes: Vec<_> = words.iter().flat_map(|v| v.to_le_bytes()).collect();
    if info.dynamic_meta_offset > bytes.len()
        || dynamic_word_offset.checked_mul(8) != Some(info.dynamic_meta_offset)
    {
        return Err(invalid(
            "packed metadata boundary differs from the RUDA ABI",
        ));
    }
    let read = |offset: usize, ty: StorageType| -> Result<u64> {
        let end = offset
            .checked_add(ty.size())
            .ok_or_else(|| invalid("argument offset overflow"))?;
        let src = bytes
            .get(offset..end)
            .ok_or_else(|| invalid("truncated packed argument"))?;
        let mut bits = [0u8; 8];
        bits[..src.len()].copy_from_slice(src);
        Ok(u64::from_le_bytes(bits))
    };
    let mut values = HashMap::new();
    let mut prefix = Vec::new();
    let mut next = u32::MAX;
    fn used(scope: &Scope, id: u32) -> bool {
        scope.instructions.iter().any(|i| {
            i.out
                .is_some_and(|v| matches!(v.kind, VariableKind::LocalConst { id: n } if n == id))
                || match &i.operation {
                    Operation::Branch(Branch::If(b)) => used(&b.scope, id),
                    Operation::Branch(Branch::IfElse(b)) => {
                        used(&b.scope_if, id) || used(&b.scope_else, id)
                    }
                    Operation::Branch(Branch::Loop(b)) => used(&b.scope,id),
                    Operation::Branch(Branch::RangeLoop(b)) => used(&b.scope,id),
                    Operation::Branch(Branch::Switch(b)) => used(&b.scope_default,id)||b.cases.iter().any(|(_,s)|used(s,id)),
                    _ => false,
                }
        })
    }
    for field in &info.scalars {
        for index in 0..field.size {
            let bits = read(field.offset + index * field.ty.size(), field.ty)?;
            let source = Variable::new(
                VariableKind::GlobalScalar(index as u32),
                Type::new(field.ty),
            );
            let value = match field.ty.elem_type() {
                ElemType::Float(FloatKind::F32 | FloatKind::F16 | FloatKind::BF16) => {
                    while used(&kernel.body, next) {
                        next = next
                            .checked_sub(1)
                            .ok_or_else(|| invalid("local id exhaustion"))?;
                    }
                    let local = Variable::new(VariableKind::LocalConst { id: next }, source.ty);
                    next = next
                        .checked_sub(1)
                        .ok_or_else(|| invalid("local id exhaustion"))?;
                    prefix.push(Instruction::new(
                        Operator::Reinterpret(UnaryOperator {
                            input: Variable::constant(
                                ConstantValue::UInt(bits),
                                Type::new(if field.ty.size()==2 {UIntKind::U16.into()}else{UIntKind::U32.into()}),
                            ),
                        }),
                        local,
                    ));
                    local
                }
                ElemType::Int(IntKind::I32) => {
                    Variable::constant(ConstantValue::Int(bits as i32 as i64), source.ty)
                }
                ElemType::Int(IntKind::I8) => Variable::constant(ConstantValue::Int(bits as i8 as i64),source.ty),
                ElemType::Int(IntKind::I16) => Variable::constant(ConstantValue::Int(bits as i16 as i64),source.ty),
                ElemType::Bool => Variable::constant(ConstantValue::Bool(bits!=0),source.ty),
                ElemType::Int(IntKind::I64) => {
                    Variable::constant(ConstantValue::Int(bits as i64), source.ty)
                }
                _ => Variable::constant(ConstantValue::UInt(bits), source.ty),
            };
            values.insert(source, value);
        }
    }
    let meta_base = info.sized_meta.map_or(0, |field| field.offset);
    let resolve_meta = |op: &Metadata| -> Result<MetaValue> {
        let (var, dim) = match op {
            Metadata::Length { var } | Metadata::BufferLength { var } | Metadata::Rank { var } => {
                (*var, None)
            }
            Metadata::Shape { var, dim } | Metadata::Stride { var, dim } => (*var, Some(*dim)),
        };
        let id = match var.kind {
            VariableKind::GlobalInputArray(id) | VariableKind::GlobalOutputArray(id) => id,
            _ => return Err(unsupported("metadata on non-global storage")),
        };
        let buffer = kernel
            .buffers
            .iter()
            .position(|b| b.id == id)
            .ok_or_else(|| invalid("unknown metadata buffer"))? as u32;
        let index = match op {
            Metadata::Length { .. } => info.metadata.len_index(buffer),
            Metadata::BufferLength { .. } => info.metadata.buffer_len_index(buffer),
            _ => {
                let extended = ext
                    .iter()
                    .position(|&n| n == id)
                    .ok_or_else(|| invalid("buffer has no extended metadata"))?
                    as u32;
                match op {
                    Metadata::Rank { .. } => info.metadata.rank_index(extended),
                    _ => {
                        let rank = read(
                            meta_base
                                + info.metadata.rank_index(extended) as usize * address.size(),
                            address,
                        )?;
                        let field = if matches!(op, Metadata::Shape { .. }) {
                            info.metadata.shape_offset_index(extended)
                        } else {
                            info.metadata.stride_offset_index(extended)
                        };
                        let base = read(meta_base + field as usize * address.size(), address)?;
                        let at=|dim:u64|->Result<Variable>{
                            let offset = base.checked_add(dim)
                            .and_then(|n| usize::try_from(n).ok())
                            .and_then(|n| n.checked_mul(address.size()))
                            .and_then(|n| n.checked_add(info.dynamic_meta_offset))
                            .ok_or_else(|| invalid("metadata offset overflow"))?;
                            Ok(Variable::constant(
                            ConstantValue::UInt(read(offset, address)?),
                            Type::new(address),
                            ))
                        };
                        let dim=dim.unwrap();
                        if let VariableKind::Constant(ConstantValue::UInt(n))=dim.kind {
                            if n>=rank{return Err(invalid("metadata dimension exceeds rank"));}
                            return Ok(MetaValue::Scalar(at(n)?));
                        }
                        if rank>bytes.len() as u64 / address.size() as u64{return Err(invalid("metadata rank exceeds argument buffer"));}
                        return Ok(MetaValue::Array((0..rank).map(at).collect::<Result<Vec<_>>>()?,dim));
                    }
                }
            }
        };
        Ok(MetaValue::Scalar(Variable::constant(
            ConstantValue::UInt(read(meta_base + index as usize * address.size(), address)?),
            Type::new(address),
        )))
    };
    fn scope(
        scope: &mut Scope,
        values: &mut HashMap<Variable, Variable>,
        meta: &impl Fn(&Metadata) -> Result<MetaValue>,
    ) -> Result<()> {
        let replace = |v: Variable, values: &HashMap<Variable, Variable>| {
            values.get(&v).copied().unwrap_or(v)
        };
        let mut instructions=std::mem::take(&mut scope.instructions);
        for i in &mut instructions {
            match &mut i.operation {
                Operation::Branch(Branch::If(b)) => {
                    b.cond = replace(b.cond, values);
                    scope_child(&mut b.scope, values, meta)?;
                }
                Operation::Branch(Branch::IfElse(b)) => {
                    b.cond = replace(b.cond, values);
                    scope_child(&mut b.scope_if, values, meta)?;
                    scope_child(&mut b.scope_else, values, meta)?;
                }
                Operation::Branch(Branch::Loop(b))=>scope_child(&mut b.scope,values,meta)?,
                Operation::Branch(Branch::RangeLoop(b))=>{
                    b.start=replace(b.start,values);b.end=replace(b.end,values);
                    b.step=b.step.map(|s|replace(s,values));
                    scope_child(&mut b.scope,values,meta)?;
                },
                Operation::Branch(Branch::Switch(b))=>{
                    b.value=replace(b.value,values);
                    for (value,child) in &mut b.cases {*value=replace(*value,values);scope_child(child,values,meta)?;}
                    scope_child(&mut b.scope_default,values,meta)?;
                },
                Operation::NonSemantic(_) => {}
                operation => {
                    if let Some(args) = operation.args() {
                        let args: Vec<_> = args.into_iter().map(|v| replace(v, values)).collect();
                        *operation = Operation::from_code_and_args(operation.op_code(), &args)
                            .ok_or_else(|| invalid("IR reflection failed"))?;
                    }
                }
            }
            if let Operation::Metadata(op) = &i.operation {
                let out = i.out.ok_or_else(|| invalid("metadata result missing"))?;
                let var=match op {Metadata::Length{var}|Metadata::BufferLength{var}|Metadata::Rank{var}|Metadata::Shape{var,..}|Metadata::Stride{var,..}=>*var};
                if !matches!(var.kind,VariableKind::GlobalInputArray(_)|VariableKind::GlobalOutputArray(_)) {
                    values.remove(&out);continue;
                }
                match meta(op)? {
                    MetaValue::Scalar(value)=>{
                        let value=Variable::new(value.kind,out.ty);
                        i.operation=Operation::Copy(value);
                        if matches!(out.kind,VariableKind::LocalConst{..}|VariableKind::Versioned{..}){values.insert(out,value);}
                    },
                    MetaValue::Array(data,index)=>{
                        let array=*scope.create_const_array(Type::new(out.storage_type()),data);
                        i.operation=Operator::Index(IndexOperator{list:array,index,vector_size:0,unroll_factor:1}).into();
                    },
                }
            } else if let (Some(out), Operation::Copy(value)) = (i.out, &i.operation) {
                if matches!(value.kind, VariableKind::Constant(_)) && matches!(out.kind,VariableKind::LocalConst{..}|VariableKind::Versioned{..}) {
                    values.insert(out, *value);
                } else {
                    values.remove(&out);
                }
            } else if let Some(out) = i.out {
                values.remove(&out);
            }
        }
        scope.instructions=instructions;
        Ok(())
    }
    fn scope_child(
        child: &mut Scope,
        values: &HashMap<Variable, Variable>,
        meta: &impl Fn(&Metadata) -> Result<MetaValue>,
    ) -> Result<()> {
        scope(child, &mut values.clone(), meta)
    }
    scope(&mut kernel.body, &mut values, &resolve_meta)?;
    prefix.append(&mut kernel.body.instructions);
    kernel.body.instructions = prefix;
    kernel.scalars.clear();
    for (index, b) in kernel.buffers.iter_mut().enumerate() {
        let bytes = info.sized_meta.as_ref().ok_or_else(|| invalid("missing buffer metadata"))?.offset
            + info.metadata.buffer_len_index(index as u32) as usize * address.size();
        b.size = Some(usize::try_from(read(bytes, address)?).map_err(|_| unsupported("buffer length exceeds host address space"))?);
        b.has_extended_meta = false;
    }
    Ok(kernel)
}

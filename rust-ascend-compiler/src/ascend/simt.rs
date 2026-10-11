use super::*;
use ruda_core::{ir::*, kernel::KernelArg};
use std::{
    collections::{HashMap, HashSet},
    fmt::Write,
};

fn scalar(storage: StorageType) -> Result<&'static str> {
    let elem = match storage {
        StorageType::Scalar(e) => e,
        _ => return Err(unsupported("SIMT packed/opaque storage")),
    };
    Ok(match elem {
        ElemType::Float(FloatKind::F32) => "float",
        ElemType::Float(FloatKind::F16) => "half",
        ElemType::Float(FloatKind::BF16) => "bfloat16_t",
        ElemType::Int(IntKind::I8) => "int8_t",
        ElemType::Int(IntKind::I16) => "int16_t",
        ElemType::Int(IntKind::I32) => "int32_t",
        ElemType::Int(IntKind::I64) => "int64_t",
        ElemType::UInt(UIntKind::U8) => "uint8_t",
        ElemType::UInt(UIntKind::U16) => "uint16_t",
        ElemType::UInt(UIntKind::U32) => "uint32_t",
        ElemType::UInt(UIntKind::U64) => "uint64_t",
        ElemType::Bool => "bool",
        _ => return Err(unsupported(format!("SIMT element {elem:?}"))),
    })
}
fn ty(t: Type) -> Result<String> {
    match t {
        Type::Scalar(s) => Ok(scalar(s)?.into()),
        Type::Vector(s, 1) => Ok(scalar(s)?.into()),
        Type::Vector(s, n) if (2..=32).contains(&n) => Ok(format!("ruda_vec<{}, {n}>", scalar(s)?)),
        _ => Err(unsupported(format!("SIMT type {t:?}"))),
    }
}
fn lanes(t: Type) -> usize {
    match t {
        Type::Vector(_, n) => n,
        _ => 1,
    }
}
fn is_array(v: Variable) -> bool {
    matches!(
        v.kind,
        VariableKind::GlobalInputArray(_)
            | VariableKind::GlobalOutputArray(_)
            | VariableKind::LocalArray { .. }
            | VariableKind::SharedArray { .. }
            | VariableKind::ConstantArray { .. }
    )
}
fn variables(scope: &Scope, values: &mut Vec<Variable>) {
    values.extend(scope.locals.iter().copied());
    for i in &scope.instructions {
        values.extend(i.operation.args().unwrap_or_default());
        values.extend(i.out);
        match &i.operation {
            Operation::Branch(Branch::If(b)) => variables(&b.scope, values),
            Operation::Branch(Branch::IfElse(b)) => {
                variables(&b.scope_if, values);
                variables(&b.scope_else, values);
            }
            Operation::Branch(Branch::RangeLoop(b)) => {
                values.push(b.i);
                variables(&b.scope, values);
            }
            Operation::Branch(Branch::Loop(b)) => variables(&b.scope, values),
            Operation::Branch(Branch::Switch(b)) => {
                variables(&b.scope_default, values);
                for (_, s) in &b.cases {
                    variables(s, values);
                }
            }
            _ => {}
        }
    }
}
fn local_name(v: Variable) -> Result<String> {
    Ok(match v.kind {
        VariableKind::LocalMut { id } => format!("m{id}"),
        VariableKind::LocalConst { id } => format!("c{id}"),
        VariableKind::Versioned { id, version } => format!("v{id}_{version}"),
        VariableKind::LocalArray { id, .. }
        | VariableKind::SharedArray { id, .. }
        | VariableKind::Shared { id }
        | VariableKind::ConstantArray { id, .. } => format!("a{id}"),
        _ => return Err(invalid("SIMT expected a local variable")),
    })
}
struct Printer<'a> {
    buffers: &'a [KernelArg],
    grid: [u32; 3],
    dim: [u32; 3],
    address: StorageType,
    source: String,
    declared: HashSet<Variable>,
    arrays: HashSet<Variable>,
    array_types: HashMap<String, Type>,
    lengths: HashMap<Variable, String>,
    mode: ExecutionMode,
    temporary: usize,
    shared_bytes: usize,
}
impl Printer<'_> {
    fn line(&mut self, s: impl AsRef<str>) {
        writeln!(self.source, "    {}", s.as_ref()).unwrap();
    }
    fn expr(&self, v: Variable) -> Result<String> {
        ty(v.ty)?;
        Ok(match v.kind {
            VariableKind::GlobalInputArray(id) | VariableKind::GlobalOutputArray(id) => {
                let b = self
                    .buffers
                    .iter()
                    .find(|b| b.id == id)
                    .ok_or_else(|| invalid("SIMT unknown buffer"))?;
                if b.ty != v.ty {
                    return Err(invalid("SIMT buffer type mismatch"));
                }
                format!("b{id}")
            }
            VariableKind::Constant(c) => {
                let literal = match c {
                    ConstantValue::Bool(v) => v.to_string(),
                    ConstantValue::Int(n) => format!(
                        "static_cast<{}>(0x{:016x}ULL)",
                        scalar(v.storage_type())?,
                        n as u64
                    ),
                    ConstantValue::UInt(n) => {
                        format!("static_cast<{}>(0x{n:016x}ULL)", scalar(v.storage_type())?)
                    }
                    ConstantValue::Float(n) => format!(
                        "static_cast<{}>(ruda_bits<float>(uint32_t(0x{:08x}U)))",
                        scalar(v.storage_type())?,
                        (n as f32).to_bits()
                    ),
                };
                if lanes(v.ty) > 1 {
                    format!(
                        "{}{{{{{}}}}}",
                        ty(v.ty)?,
                        vec![literal; lanes(v.ty)].join(", ")
                    )
                } else {
                    literal
                }
            }
            VariableKind::Builtin(b) => self.builtin(b)?,
            _ => {
                if !self.declared.contains(&v) && !self.arrays.contains(&v) {
                    return Err(invalid(format!("undefined SIMT variable {v:?}")));
                }
                let name = local_name(v)?;
                if self.array_types.get(&name).is_some_and(|t| *t != v.ty) {
                    format!("reinterpret_cast<{}*>({name})", ty(v.ty)?)
                } else {
                    name
                }
            }
        })
    }
    fn builtin(&self, b: Builtin) -> Result<String> {
        let u = scalar(self.address)?;
        let unit = "(uint64_t(threadIdx.z) * blockDim.y * blockDim.x + uint64_t(threadIdx.y) * blockDim.x + threadIdx.x)";
        let block = "(uint64_t(blockIdx.z) * gridDim.y * gridDim.x + uint64_t(blockIdx.y) * gridDim.x + blockIdx.x)";
        let units = u64::from(self.dim[0]) * u64::from(self.dim[1]) * u64::from(self.dim[2]);
        let blocks = u64::from(self.grid[0]) * u64::from(self.grid[1]) * u64::from(self.grid[2]);
        let e=match b {
            Builtin::UnitPos => unit.into(), Builtin::UnitPosX => "threadIdx.x".into(),
            Builtin::UnitPosY => "threadIdx.y".into(), Builtin::UnitPosZ => "threadIdx.z".into(),
            Builtin::RudaPos => block.into(), Builtin::RudaPosX => "blockIdx.x".into(),
            Builtin::RudaPosY => "blockIdx.y".into(), Builtin::RudaPosZ => "blockIdx.z".into(),
            Builtin::RudaDim => format!("{units}ULL"), Builtin::RudaCount => format!("{blocks}ULL"),
            Builtin::RudaDimX => "blockDim.x".into(), Builtin::RudaDimY => "blockDim.y".into(), Builtin::RudaDimZ => "blockDim.z".into(),
            Builtin::RudaCountX => "gridDim.x".into(), Builtin::RudaCountY => "gridDim.y".into(), Builtin::RudaCountZ => "gridDim.z".into(),
            Builtin::AbsolutePos => "((uint64_t(blockIdx.z) * blockDim.z + threadIdx.z) * uint64_t(gridDim.x) * blockDim.x * gridDim.y * blockDim.y + (uint64_t(blockIdx.y) * blockDim.y + threadIdx.y) * uint64_t(gridDim.x) * blockDim.x + uint64_t(blockIdx.x) * blockDim.x + threadIdx.x)".into(),
            Builtin::AbsolutePosX => "(uint64_t(blockIdx.x) * blockDim.x + threadIdx.x)".into(),
            Builtin::AbsolutePosY => "(uint64_t(blockIdx.y) * blockDim.y + threadIdx.y)".into(),
            Builtin::AbsolutePosZ => "(uint64_t(blockIdx.z) * blockDim.z + threadIdx.z)".into(),
            Builtin::PlaneDim => "32U".into(), Builtin::UnitPosPlane => format!("({unit} % 32U)"),
            Builtin::PlanePos => format!("({unit} / 32U)"), Builtin::PlaneCount => format!("{}U",units.div_ceil(32)),
            _ => return Err(unsupported(format!("SIMT builtin {b:?}"))),
        };
        Ok(format!("static_cast<{u}>({e})"))
    }
    fn lane(&self, v: Variable, i: usize) -> Result<String> {
        if matches!(v.kind, VariableKind::Constant(_)) {
            return self.expr(Variable::new(v.kind, Type::Scalar(v.storage_type())));
        }
        let value = self.expr(v)?;
        if lanes(v.ty) == 1 {
            Ok(value)
        } else if i < lanes(v.ty) {
            Ok(format!("{value}.lane[{i}]"))
        } else {
            Err(invalid("SIMT vector lane mismatch"))
        }
    }
    fn declare(&mut self, v: Variable) -> Result<()> {
        if self.declared.contains(&v) {
            return Ok(());
        }
        let n = local_name(v)?;
        match v.kind {
            VariableKind::LocalMut { .. }
            | VariableKind::LocalConst { .. }
            | VariableKind::Versioned { .. } => {
                self.line(format!("{} {n};", ty(v.ty)?));
                self.declared.insert(v);
            }
            VariableKind::Shared { .. } => {
                self.shared_bytes = self
                    .shared_bytes
                    .checked_add(v.ty.size())
                    .ok_or_else(|| invalid("SIMT shared size overflow"))?;
                self.line(format!("__ubuf__ {} {n};", ty(v.ty)?));
                self.declared.insert(v);
            }
            VariableKind::LocalArray {
                length,
                unroll_factor,
                ..
            }
            | VariableKind::SharedArray {
                length,
                unroll_factor,
                ..
            } => {
                if self.arrays.insert(v) {
                    let length = length
                        .checked_mul(unroll_factor)
                        .ok_or_else(|| invalid("SIMT array size overflow"))?;
                    self.lengths.insert(v, format!("{length}ULL"));
                    if self.array_types.contains_key(&n) {
                        return Ok(());
                    }
                    self.array_types.insert(n.clone(), v.ty);
                    let shared = matches!(v.kind, VariableKind::SharedArray { .. });
                    if shared {
                        self.shared_bytes = self
                            .shared_bytes
                            .checked_add(
                                length
                                    .checked_mul(v.ty.size())
                                    .ok_or_else(|| invalid("SIMT shared size overflow"))?,
                            )
                            .ok_or_else(|| invalid("SIMT shared size overflow"))?;
                    }
                    let alignment = match v.kind {
                        VariableKind::SharedArray {
                            alignment: Some(a), ..
                        } => {
                            if !a.is_power_of_two() {
                                return Err(invalid(
                                    "SIMT shared alignment must be a power of two",
                                ));
                            }
                            format!("__align__({a}) ")
                        }
                        _ => String::new(),
                    };
                    self.line(format!(
                        "{alignment}{}{} {n}[{length}];",
                        if shared { "__ubuf__ " } else { "" },
                        ty(v.ty)?
                    ));
                    self.lengths.insert(v, format!("{length}ULL"));
                }
            }
            _ => return Err(unsupported(format!("SIMT declaration {v:?}"))),
        }
        Ok(())
    }
    fn assign(
        &mut self,
        out: Variable,
        make: impl Fn(&Self, usize) -> Result<String>,
    ) -> Result<()> {
        self.declare(out)?;
        for i in 0..lanes(out.ty) {
            let rhs = make(self, i)?;
            self.line(format!(
                "{} = static_cast<{}>({rhs});",
                self.lane(out, i)?,
                scalar(out.storage_type())?
            ));
        }
        Ok(())
    }
    fn length(&self, v: Variable) -> Result<String> {
        if let Some(s) = self.lengths.get(&v) {
            return Ok(s.clone());
        }
        if !is_array(v) && lanes(v.ty) > 1 {
            return Ok(format!("{}ULL", lanes(v.ty)));
        }
        match v.kind {
            VariableKind::GlobalInputArray(id) | VariableKind::GlobalOutputArray(id) => self
                .buffers
                .iter()
                .find(|b| b.id == id)
                .and_then(|b| b.size)
                .map(|n| format!("{n}ULL"))
                .ok_or_else(|| invalid("SIMT buffer length is not specialized")),
            _ => Err(unsupported("SIMT length of this value")),
        }
    }
    fn child(&mut self, scope: Scope) -> Result<()> {
        let declared = self.declared.clone();
        let arrays = self.arrays.clone();
        let lengths = self.lengths.clone();
        self.scope(scope)?;
        self.declared = declared;
        self.arrays = arrays;
        self.lengths = lengths;
        Ok(())
    }
    fn scope(&mut self, mut scope: Scope) -> Result<()> {
        let processed = scope.process(std::iter::empty::<&dyn Processor>());
        for (v, values) in scope.const_arrays {
            if self.arrays.insert(v) {
                let values = values
                    .iter()
                    .map(|v| self.expr(*v))
                    .collect::<Result<Vec<_>>>()?
                    .join(", ");
                let length = match v.kind {
                    VariableKind::ConstantArray { length, .. } => length,
                    _ => return Err(invalid("invalid constant array")),
                };
                self.line(format!(
                    "const {} {}[{length}] = {{{values}}};",
                    ty(v.ty)?,
                    local_name(v)?
                ));
                self.lengths.insert(v, format!("{length}ULL"));
            }
        }
        for v in processed.variables {
            self.declare(v)?;
        }
        for inst in processed.instructions {
            let inst = inst.lower_typed_memory().map_err(invalid)?;
            for v in inst
                .operation
                .args()
                .unwrap_or_default()
                .into_iter()
                .chain(inst.out)
            {
                if matches!(
                    v.kind,
                    VariableKind::LocalArray { .. }
                        | VariableKind::SharedArray { .. }
                        | VariableKind::Shared { .. }
                ) {
                    self.declare(v)?;
                }
            }
            self.instruction(inst)?;
        }
        Ok(())
    }
    fn instruction(&mut self, inst: Instruction) -> Result<()> {
        let out = inst.out;
        let destination = || out.ok_or_else(|| invalid("SIMT operation has no destination"));
        match inst.operation {
            Operation::Copy(v) => self.assign(destination()?, |p, i| p.lane(v, i)),
            Operation::Arithmetic(a) => self.arithmetic(a, destination()?),
            Operation::Comparison(c) => {
                if let Comparison::IsNan(ref o) | Comparison::IsInf(ref o) = c {
                    let sign = if matches!(c, Comparison::IsNan(_)) {
                        "isnan"
                    } else {
                        "isinf"
                    };
                    return self.assign(destination()?, |p, i| {
                        Ok(format!("{sign}(float({}))", p.lane(o.input, i)?))
                    });
                }
                let (op, s) = match c {
                    Comparison::Equal(o) => (o, "=="),
                    Comparison::NotEqual(o) => (o, "!="),
                    Comparison::Lower(o) => (o, "<"),
                    Comparison::LowerEqual(o) => (o, "<="),
                    Comparison::Greater(o) => (o, ">"),
                    Comparison::GreaterEqual(o) => (o, ">="),
                    _ => unreachable!(),
                };
                self.assign(destination()?, |p, i| {
                    Ok(format!(
                        "({} {s} {})",
                        p.lane(op.lhs, i)?,
                        p.lane(op.rhs, i)?
                    ))
                })
            }
            Operation::Bitwise(b) => self.bitwise(b, destination()?),
            Operation::Plane(p) => self.plane(p, destination()?),
            Operation::Operator(o) => self.operator(o, out),
            Operation::Metadata(m) => match m {
                Metadata::Length { var } | Metadata::BufferLength { var } => {
                    let length = self.length(var)?;
                    self.assign(destination()?, |_, _| Ok(length.clone()))
                }
                _ => Err(unsupported(
                    "SIMT metadata must be specialized by the runtime",
                )),
            },
            Operation::Branch(b) => self.branch(b),
            Operation::Synchronization(s) => {
                self.line(match s {
                    Synchronization::SyncRuda | Synchronization::SyncStorage => {
                        "asc_syncthreads();"
                    }
                    Synchronization::SyncPlane => "cooperative_groups::coalesced_threads().sync();",
                    _ => return Err(unsupported("SIMT async proxy fence")),
                });
                Ok(())
            }
            Operation::NonSemantic(_) | Operation::Marker(_) => Ok(()),
            _ => Err(unsupported(format!("SIMT operation {:?}", inst.operation))),
        }
    }
    fn branch(&mut self, b: Branch) -> Result<()> {
        match b {
            Branch::If(b) => {
                self.line(format!("if ({}) {{", self.expr(b.cond)?));
                self.child(b.scope)?;
                self.line("}");
            }
            Branch::IfElse(b) => {
                self.line(format!("if ({}) {{", self.expr(b.cond)?));
                self.child(b.scope_if)?;
                self.line("} else {");
                self.child(b.scope_else)?;
                self.line("}");
            }
            Branch::Switch(b) => {
                self.line(format!("switch ({}) {{", self.expr(b.value)?));
                for (v, s) in b.cases {
                    self.line(format!("case {}: {{", self.expr(v)?));
                    self.child(s)?;
                    self.line("break; }");
                }
                self.line("default: {");
                self.child(b.scope_default)?;
                self.line("break; } }");
            }
            Branch::RangeLoop(b) => {
                let start = self.expr(b.start)?;
                let end = self.expr(b.end)?;
                let step = match b.step {
                    Some(s) => self.expr(s)?,
                    None => "1".into(),
                };
                let declared = self.declared.clone();
                self.declared.insert(b.i);
                self.line(format!(
                    "for ({} {} = {start}; {} {} {end}; {} += {step}) {{",
                    ty(b.i.ty)?,
                    local_name(b.i)?,
                    local_name(b.i)?,
                    if b.inclusive { "<=" } else { "<" },
                    local_name(b.i)?
                ));
                self.child(b.scope)?;
                self.line("}");
                self.declared = declared;
            }
            Branch::Loop(b) => {
                self.line("while (true) {");
                self.child(b.scope)?;
                self.line("}");
            }
            Branch::Return => self.line("return;"),
            Branch::Break => self.line("break;"),
            Branch::Unreachable => self.line("__builtin_unreachable();"),
        }
        Ok(())
    }
    fn arithmetic(&mut self, a: Arithmetic, out: Variable) -> Result<()> {
        match &a {
            Arithmetic::SaturatingAdd(o) | Arithmetic::SaturatingSub(o) => {
                if !out.ty.is_int() {
                    return Err(invalid("SIMT saturation requires integers"));
                }
                let sub = matches!(a, Arithmetic::SaturatingSub(_));
                return self.assign(out, |p, i| {
                    Ok(format!(
                        "ruda_saturate<{}>({}, {}, {})",
                        scalar(out.storage_type())?,
                        p.lane(o.lhs, i)?,
                        p.lane(o.rhs, i)?,
                        sub
                    ))
                });
            }
            Arithmetic::MulHi(o) => {
                if !out.ty.is_int() {
                    return Err(invalid("SIMT mul_hi requires integers"));
                }
                return self.assign(out, |p, i| {
                    Ok(format!(
                        "ruda_mul_hi<{}>({}, {})",
                        scalar(out.storage_type())?,
                        p.lane(o.lhs, i)?,
                        p.lane(o.rhs, i)?
                    ))
                });
            }
            _ => {}
        }
        let binary = match &a {
            Arithmetic::Add(o) => Some((o, "+")),
            Arithmetic::Sub(o) => Some((o, "-")),
            Arithmetic::Mul(o) => Some((o, "*")),
            Arithmetic::Div(o) => Some((o, "/")),
            _ => None,
        };
        if let Some((o, s)) = binary {
            return self.assign(out, |p, i| {
                let left = p.lane(o.lhs, i)?;
                let right = p.lane(o.rhs, i)?;
                if out.ty.is_int() && s != "/" {
                    Ok(format!("(uint64_t({left}) {s} uint64_t({right}))"))
                } else {
                    Ok(format!("({left} {s} {right})"))
                }
            });
        }
        let binary = match &a {
            Arithmetic::Modulo(o) => Some((o, "mod")),
            Arithmetic::Remainder(o) => Some((o, "rem")),
            Arithmetic::Min(o) => Some((o, "min")),
            Arithmetic::Max(o) => Some((o, "max")),
            Arithmetic::Powf(o) => Some((o, "powf")),
            Arithmetic::Powi(o) => Some((o, "powi")),
            Arithmetic::ArcTan2(o) => Some((o, "atan2f")),
            Arithmetic::Hypot(o) => Some((o, "hypotf")),
            Arithmetic::Rhypot(o) => Some((o, "rhypot")),
            _ => None,
        };
        if let Some((o, s)) = binary {
            return self.assign(out, |p, i| {
                let x = p.lane(o.lhs, i)?;
                let y = p.lane(o.rhs, i)?;
                Ok(match s {
                    "mod" | "rem" if out.ty.is_float() => {
                        if s == "mod" {
                            format!("fmodf(float({x}), float({y}))")
                        } else {
                            format!("(float({x}) - floorf(float({x}) / float({y})) * float({y}))")
                        }
                    }
                    "mod" => format!("ruda_mod<{}>({x}, {y})", scalar(out.storage_type())?),
                    "rem" => format!("ruda_rem<{}>({x}, {y})", scalar(out.storage_type())?),
                    "min" | "max" if out.ty.is_float() => {
                        format!("f{}f(float({x}), float({y}))", s)
                    }
                    "min" => format!("({x} < {y} ? {x} : {y})"),
                    "max" => format!("({x} > {y} ? {x} : {y})"),
                    "powi" => format!("ruda_powi({}, {y})", x),
                    "rhypot" => format!("(1.0f / hypotf(float({x}), float({y})))"),
                    _ => format!("{s}(float({x}), float({y}))"),
                })
            });
        }
        let magnitude = matches!(a, Arithmetic::Magnitude(_));
        match a {
            Arithmetic::Fma(o)
                if matches!(
                    out.elem_type(),
                    ElemType::Float(FloatKind::F16 | FloatKind::BF16)
                ) =>
            {
                if [o.a, o.b, o.c]
                    .iter()
                    .any(|v| v.elem_type() != out.elem_type())
                {
                    return Err(invalid("SIMT half FMA requires matching operand precision"));
                }
                self.assign(out, |p, i| {
                    Ok(format!(
                        "__hfma({}, {}, {})",
                        p.lane(o.a, i)?,
                        p.lane(o.b, i)?,
                        p.lane(o.c, i)?
                    ))
                })
            }
            Arithmetic::Fma(o) if out.ty.is_float() => self.assign(out, |p, i| {
                Ok(format!(
                    "fmaf(float({}), float({}), float({}))",
                    p.lane(o.a, i)?,
                    p.lane(o.b, i)?,
                    p.lane(o.c, i)?
                ))
            }),
            Arithmetic::Fma(o) if out.ty.is_int() => self.assign(out, |p, i| {
                Ok(format!(
                    "(uint64_t({}) * uint64_t({}) + uint64_t({}))",
                    p.lane(o.a, i)?,
                    p.lane(o.b, i)?,
                    p.lane(o.c, i)?
                ))
            }),
            Arithmetic::Clamp(o) => self.assign(out, |p, i| {
                let x = p.lane(o.input, i)?;
                let lo = p.lane(o.min_value, i)?;
                let hi = p.lane(o.max_value, i)?;
                Ok(format!("({x} < {lo} ? {lo} : ({x} > {hi} ? {hi} : {x}))"))
            }),
            Arithmetic::Dot(o) => {
                if lanes(o.lhs.ty) != lanes(o.rhs.ty) || lanes(out.ty) != 1 {
                    return Err(invalid("SIMT dot dimensions"));
                }
                let t = scalar(out.storage_type())?;
                self.declare(out)?;
                self.line(format!("{} = {t}(0);", self.expr(out)?));
                for i in 0..lanes(o.lhs.ty) {
                    let x = self.lane(o.lhs, i)?;
                    let y = self.lane(o.rhs, i)?;
                    let z = self.expr(out)?;
                    self.line(if out.ty.is_int() {
                        format!(
                            "{z} = static_cast<{t}>(uint64_t({z}) + uint64_t({x}) * uint64_t({y}));"
                        )
                    } else {
                        format!("{z} += {x} * {y};")
                    });
                }
                Ok(())
            }
            Arithmetic::VectorSum(o) => {
                self.declare(out)?;
                self.line(format!(
                    "{} = {}(0);",
                    self.expr(out)?,
                    scalar(out.storage_type())?
                ));
                for i in 0..lanes(o.input.ty) {
                    let z = self.expr(out)?;
                    let x = self.lane(o.input, i)?;
                    self.line(if out.ty.is_int() {
                        format!(
                            "{z} = static_cast<{}>(uint64_t({z}) + uint64_t({x}));",
                            scalar(out.storage_type())?
                        )
                    } else {
                        format!("{z} += {x};")
                    });
                }
                Ok(())
            }
            Arithmetic::Magnitude(o) | Arithmetic::Normalize(o) => {
                if !out.ty.is_float() {
                    return Err(invalid("SIMT magnitude/normalize requires floats"));
                }
                if magnitude && lanes(out.ty) != 1 {
                    return Err(invalid("SIMT magnitude result must be scalar"));
                }
                let id = self.temporary;
                self.temporary += 1;
                self.line(format!("float ruda_norm_{id}=0.0f;"));
                for i in 0..lanes(o.input.ty) {
                    let x = self.lane(o.input, i)?;
                    self.line(format!("ruda_norm_{id} += float({x}) * float({x});"));
                }
                self.line(format!("ruda_norm_{id}=sqrtf(ruda_norm_{id});"));
                self.assign(out, |p, i| {
                    if magnitude {
                        Ok(format!("ruda_norm_{id}"))
                    } else {
                        Ok(format!("(float({}) / ruda_norm_{id})", p.lane(o.input, i)?))
                    }
                })
            }
            other => {
                let (o, s) = match other {
                    Arithmetic::Neg(o) => (o, "neg"),
                    Arithmetic::Abs(o) => (o, "abs"),
                    Arithmetic::Recip(o) => (o, "recip"),
                    Arithmetic::InverseSqrt(o) => (o, "rsqrt"),
                    Arithmetic::Exp(o) => (o, "expf"),
                    Arithmetic::Log(o) => (o, "logf"),
                    Arithmetic::Log1p(o) => (o, "log1pf"),
                    Arithmetic::Sin(o) => (o, "sinf"),
                    Arithmetic::Cos(o) => (o, "cosf"),
                    Arithmetic::Tan(o) => (o, "tanf"),
                    Arithmetic::Tanh(o) => (o, "tanhf"),
                    Arithmetic::Sinh(o) => (o, "sinhf"),
                    Arithmetic::Cosh(o) => (o, "coshf"),
                    Arithmetic::ArcSin(o) => (o, "asinf"),
                    Arithmetic::ArcCos(o) => (o, "acosf"),
                    Arithmetic::ArcTan(o) => (o, "atanf"),
                    Arithmetic::ArcSinh(o) => (o, "asinhf"),
                    Arithmetic::ArcCosh(o) => (o, "acoshf"),
                    Arithmetic::ArcTanh(o) => (o, "atanhf"),
                    Arithmetic::Sqrt(o) => (o, "sqrtf"),
                    Arithmetic::Erf(o) => (o, "erff"),
                    Arithmetic::Floor(o) => (o, "floorf"),
                    Arithmetic::Ceil(o) => (o, "ceilf"),
                    Arithmetic::Trunc(o) => (o, "truncf"),
                    Arithmetic::Round(o) => (o, "rintf"),
                    Arithmetic::Degrees(o) => (o, "degrees"),
                    Arithmetic::Radians(o) => (o, "radians"),
                    _ => return Err(unsupported(format!("SIMT arithmetic {other:?}"))),
                };
                if !out.ty.is_float() && !matches!(s, "neg" | "abs") {
                    return Err(invalid(
                        "SIMT transcendental/rounding operation requires floating storage",
                    ));
                }
                self.assign(out, |p, i| {
                    let x = p.lane(o.input, i)?;
                    Ok(match s {
                        "neg" if out.ty.is_int() => format!("(uint64_t(0) - uint64_t({x}))"),
                        "neg" => format!("(-{x})"),
                        "abs" if out.ty.is_int() => {
                            format!("({x} < 0 ? uint64_t(0)-uint64_t({x}) : uint64_t({x}))")
                        }
                        "abs" => format!("fabsf(float({x}))"),
                        "recip" => format!("(1.0f / float({x}))"),
                        "rsqrt" => format!("(1.0f / sqrtf(float({x})))"),
                        "degrees" => format!("(float({x}) * 57.29577951308232f)"),
                        "radians" => format!("(float({x}) * 0.017453292519943295f)"),
                        _ => format!("{s}(float({x}))"),
                    })
                })
            }
        }
    }
    fn bitwise(&mut self, b: Bitwise, out: Variable) -> Result<()> {
        let binary = match &b {
            Bitwise::BitwiseAnd(o) => Some((o, "&")),
            Bitwise::BitwiseOr(o) => Some((o, "|")),
            Bitwise::BitwiseXor(o) => Some((o, "^")),
            Bitwise::ShiftLeft(o) => Some((o, "<<")),
            Bitwise::ShiftRight(o) => Some((o, ">>")),
            _ => None,
        };
        if let Some((o, s)) = binary {
            return self.assign(out, |p, i| {
                let x = p.lane(o.lhs, i)?;
                let y = p.lane(o.rhs, i)?;
                Ok(if s == "<<" {
                    format!("(uint64_t({x}) << {y})")
                } else {
                    format!("({x} {s} {y})")
                })
            });
        }
        let (o, s) = match b {
            Bitwise::BitwiseNot(o) => (o, "not"),
            Bitwise::CountOnes(o) => (o, "popcount"),
            Bitwise::LeadingZeros(o) => (o, "clz"),
            Bitwise::TrailingZeros(o) => (o, "ctz"),
            Bitwise::FindFirstSet(o) => (o, "ffs"),
            Bitwise::ReverseBits(o) => (o, "reverse"),
            _ => unreachable!(),
        };
        let bits = o.input.storage_type().size() * 8;
        self.assign(out, |p, i| {
            let raw = p.lane(o.input, i)?;
            let x = format!(
                "uint64_t(typename std::make_unsigned<{}>::type({raw}))",
                scalar(o.input.storage_type())?
            );
            Ok(match s {
                "not" => format!("(~{x})"),
                "reverse" => format!("ruda_reverse<uint64_t>({x}, {bits})"),
                "clz" => format!(
                    "({x} == 0 ? {bits} : __builtin_clzll(uint64_t({x})) - {})",
                    64 - bits
                ),
                "ctz" => format!("({x} == 0 ? {bits} : __builtin_ctzll(uint64_t({x})))"),
                "ffs" => format!("__builtin_ffsll(uint64_t({x}))"),
                _ => format!("__builtin_popcountll(uint64_t({x}))"),
            })
        })
    }
    fn plane(&mut self, op: Plane, out: Variable) -> Result<()> {
        let name = match &op {
            Plane::All(_) => "asc_all",
            Plane::Any(_) => "asc_any",
            Plane::Ballot(_) => "asc_ballot",
            Plane::ShuffleXor(_) => "asc_shfl_xor",
            Plane::ShuffleUp(_) => "asc_shfl_up",
            Plane::ShuffleDown(_) => "asc_shfl_down",
            Plane::Broadcast(_) | Plane::Shuffle(_) => "asc_shfl",
            Plane::Sum(_) => "asc_reduce_add",
            Plane::Min(_) => "asc_reduce_min",
            Plane::Max(_) => "asc_reduce_max",
            _ => "",
        };
        match op {
            Plane::Elect => self.assign(out, |_, _| {
                Ok("(cooperative_groups::coalesced_threads().thread_rank() == 0)".into())
            }),
            Plane::All(o) | Plane::Any(o) | Plane::Ballot(o) => {
                self.assign(out, |p, i| Ok(format!("{name}({})", p.lane(o.input, i)?)))
            }
            Plane::Broadcast(o)
            | Plane::Shuffle(o)
            | Plane::ShuffleXor(o)
            | Plane::ShuffleUp(o)
            | Plane::ShuffleDown(o) => {
                if !matches!(
                    o.lhs.elem_type(),
                    ElemType::Float(FloatKind::F32)
                        | ElemType::Int(IntKind::I32 | IntKind::I64)
                        | ElemType::UInt(UIntKind::U32 | UIntKind::U64)
                ) {
                    return Err(unsupported("SIMT shuffle element dtype"));
                }
                self.assign(out, |p, i| {
                    Ok(format!("{name}({}, {})", p.lane(o.lhs, i)?, p.expr(o.rhs)?))
                })
            }
            Plane::Sum(o) | Plane::Min(o) | Plane::Max(o) => {
                if !matches!(
                    o.input.elem_type(),
                    ElemType::Float(FloatKind::F32 | FloatKind::F16)
                        | ElemType::Int(IntKind::I32)
                        | ElemType::UInt(UIntKind::U32)
                ) {
                    return Err(unsupported("SIMT hardware reduction element dtype"));
                }
                self.assign(out, |p, i| Ok(format!("{name}({})", p.lane(o.input, i)?)))
            }
            _ => Err(unsupported(format!("SIMT plane operation {op:?}"))),
        }
    }
    fn operator(&mut self, o: Operator, out: Option<Variable>) -> Result<()> {
        let dst = || out.ok_or_else(|| invalid("SIMT missing result"));
        let unchecked = matches!(
            o,
            Operator::UncheckedIndex(_) | Operator::UncheckedIndexAssign(_)
        );
        match o {
            Operator::Cast(o) => self.assign(dst()?, |p, i| p.lane(o.input, i)),
            Operator::Reinterpret(o) => {
                let v = dst()?;
                if v.ty.size() != o.input.ty.size() {
                    return Err(invalid("SIMT bitcast size mismatch"));
                }
                self.declare(v)?;
                self.line(format!(
                    "{} = ruda_bits<{}>({});",
                    self.expr(v)?,
                    ty(v.ty)?,
                    self.expr(o.input)?
                ));
                Ok(())
            }
            Operator::Select(o) => self.assign(dst()?, |p, i| {
                Ok(format!(
                    "({} ? {} : {})",
                    p.lane(o.cond, i)?,
                    p.lane(o.then, i)?,
                    p.lane(o.or_else, i)?
                ))
            }),
            Operator::And(o) => self.assign(dst()?, |p, i| {
                Ok(format!("({} && {})", p.lane(o.lhs, i)?, p.lane(o.rhs, i)?))
            }),
            Operator::Or(o) => self.assign(dst()?, |p, i| {
                Ok(format!("({} || {})", p.lane(o.lhs, i)?, p.lane(o.rhs, i)?))
            }),
            Operator::Not(o) => {
                self.assign(dst()?, |p, i| Ok(format!("(!{})", p.lane(o.input, i)?)))
            }
            Operator::InitVector(o) => {
                let v = dst()?;
                if lanes(v.ty) != o.inputs.len() {
                    return Err(invalid("SIMT vector initializer length"));
                }
                self.assign(v, |p, i| p.expr(o.inputs[i]))
            }
            Operator::Index(o) | Operator::UncheckedIndex(o) => {
                let v = dst()?;
                let array = self.expr(o.list)?;
                let index = self.expr(o.index)?;
                let guarded = self.mode == ExecutionMode::Checked && !unchecked;
                if o.list.elem_type() != v.elem_type() {
                    return Err(invalid("SIMT index element dtype mismatch"));
                }
                let (read, length) =
                    if !is_array(o.list) && lanes(o.list.ty) > 1 && lanes(v.ty) == 1 {
                        (format!("{array}.lane[{index}]"), self.length(o.list)?)
                    } else if is_array(o.list) {
                        let width = if o.vector_size == 0 {
                            lanes(o.list.ty)
                        } else {
                            o.vector_size
                        };
                        if width != lanes(v.ty) {
                            return Err(invalid("SIMT index vector width mismatch"));
                        }
                        let read = if o.list.ty == v.ty {
                            format!("{array}[{index}]")
                        } else {
                            format!("reinterpret_cast<const {}*>({array})[{index}]", ty(v.ty)?)
                        };
                        (
                            read,
                            format!(
                                "({} * {}ULL / {}ULL)",
                                self.length(o.list)?,
                                lanes(o.list.ty),
                                width
                            ),
                        )
                    } else {
                        return Err(unsupported("SIMT index source"));
                    };
                self.declare(v)?;
                self.line(format!(
                    "{} = {};",
                    self.expr(v)?,
                    if guarded {
                        format!("({index} < {length} ? {read} : {}{{}})", ty(v.ty)?)
                    } else {
                        read
                    }
                ));
                Ok(())
            }
            Operator::IndexAssign(o) | Operator::UncheckedIndexAssign(o) => {
                let v = dst()?;
                let array = self.expr(v)?;
                let index = self.expr(o.index)?;
                let value = self.expr(o.value)?;
                if !matches!(
                    v.kind,
                    VariableKind::GlobalInputArray(_)
                        | VariableKind::GlobalOutputArray(_)
                        | VariableKind::LocalArray { .. }
                        | VariableKind::SharedArray { .. }
                        | VariableKind::LocalMut { .. }
                        | VariableKind::LocalConst { .. }
                ) {
                    return Err(unsupported("SIMT index assignment destination"));
                }
                if let VariableKind::GlobalInputArray(id) | VariableKind::GlobalOutputArray(id) =
                    v.kind
                {
                    if self.buffers.iter().find(|b| b.id == id).unwrap().visibility
                        != Visibility::ReadWrite
                    {
                        return Err(invalid("SIMT write to readonly buffer"));
                    }
                }
                if !is_array(v) && lanes(v.ty) > 1 && lanes(o.value.ty) == 1 {
                    if self.mode == ExecutionMode::Checked && !unchecked {
                        self.line(format!("if ({index} < {}ULL) {{", lanes(v.ty)));
                    }
                    self.line(format!("{array}.lane[{index}] = {value};"));
                    if self.mode == ExecutionMode::Checked && !unchecked {
                        self.line("}");
                    }
                } else {
                    if !is_array(v) || v.elem_type() != o.value.elem_type() {
                        return Err(invalid("SIMT index assignment dtype mismatch"));
                    }
                    let width = if o.vector_size == 0 {
                        lanes(v.ty)
                    } else {
                        o.vector_size
                    };
                    if width != lanes(o.value.ty) {
                        return Err(unsupported(
                            "SIMT store needs an explicitly matching value vector width",
                        ));
                    }
                    let target = if v.ty == o.value.ty {
                        array
                    } else {
                        format!("reinterpret_cast<{}*>({array})", ty(o.value.ty)?)
                    };
                    let guarded = self.mode == ExecutionMode::Checked && !unchecked;
                    if guarded {
                        self.line(format!(
                            "if ({index} < ({} * {}ULL / {}ULL)) {{",
                            self.length(v)?,
                            lanes(v.ty),
                            width
                        ));
                    }
                    self.line(format!("{target}[{index}] = {value};"));
                    if guarded {
                        self.line("}");
                    }
                }
                Ok(())
            }
            Operator::NativeAddress(o) => {
                let x = self.expr(o.lhs)?;
                let y = self.expr(o.rhs)?;
                self.assign(dst()?, |_, _| {
                    Ok(format!("reinterpret_cast<uint64_t>(&{x}[{y}])"))
                })
            }
            Operator::NativeLoad(o) => {
                let v = dst()?;
                let ptr = self.expr(o.input)?;
                self.declare(v)?;
                self.line(format!(
                    "{} = *reinterpret_cast<{}*>({ptr});",
                    self.expr(v)?,
                    ty(v.ty)?
                ));
                Ok(())
            }
            Operator::NativeStore(o) => {
                self.line(format!(
                    "*reinterpret_cast<{}*>({}) = {};",
                    ty(o.rhs.ty)?,
                    self.expr(o.lhs)?,
                    self.expr(o.rhs)?
                ));
                Ok(())
            }
            Operator::CopyMemory(o) => {
                let v = dst()?;
                let input_index = self.expr(o.in_index)?;
                let output_index = self.expr(o.out_index)?;
                self.copy(o.input, v, input_index, output_index)
            }
            Operator::CopyMemoryBulk(o) => {
                let v = dst()?;
                let id = self.temporary;
                self.temporary += 1;
                let input_index = format!(
                    "(uint64_t({}) + uint64_t({}) + ruda_copy_{id})",
                    self.expr(o.in_index)?,
                    self.expr(o.offset_input)?
                );
                let output_index = format!(
                    "(uint64_t({}) + uint64_t({}) + ruda_copy_{id})",
                    self.expr(o.out_index)?,
                    self.expr(o.offset_out)?
                );
                self.line(format!(
                    "for (uint64_t ruda_copy_{id}=0; ruda_copy_{id}<{}ULL; ++ruda_copy_{id}) {{",
                    o.len
                ));
                self.copy(o.input, v, input_index, output_index)?;
                self.line("}");
                Ok(())
            }
        }
    }
    fn copy(
        &mut self,
        input: Variable,
        out: Variable,
        input_index: String,
        output_index: String,
    ) -> Result<()> {
        if !is_array(input)
            || !is_array(out)
            || input.ty != out.ty
            || matches!(out.kind, VariableKind::ConstantArray { .. })
        {
            return Err(invalid("SIMT copy requires compatible writable arrays"));
        }
        if let VariableKind::GlobalInputArray(id) | VariableKind::GlobalOutputArray(id) = out.kind {
            if self
                .buffers
                .iter()
                .find(|b| b.id == id)
                .is_none_or(|b| b.visibility != Visibility::ReadWrite)
            {
                return Err(invalid("SIMT copy to readonly binding"));
            }
        }
        let src = self.expr(input)?;
        let dst = self.expr(out)?;
        let checked = self.mode == ExecutionMode::Checked;
        if checked {
            self.line(format!("if ({output_index} < {}) {{", self.length(out)?));
        }
        let value = if checked {
            format!(
                "({input_index} < {} ? {src}[{input_index}] : {}{{}})",
                self.length(input)?,
                ty(input.ty)?
            )
        } else {
            format!("{src}[{input_index}]")
        };
        self.line(format!("{dst}[{output_index}] = {value};"));
        if checked {
            self.line("}");
        }
        Ok(())
    }
}

const HELPERS: &str = r#"
#include <stdint.h>
#include <math.h>
#include <type_traits>
#include <limits>
#include "simt_api/asc_simt.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_bf16.h"
#include "simt_api/cooperative_groups.h"
#include "acl/acl.h"
template<class T, unsigned N> struct ruda_vec { T lane[N]; };
template<class T, class U> __aicore__ inline T ruda_bits(U value) {
    static_assert(sizeof(T)==sizeof(U)); T result; __builtin_memcpy(&result,&value,sizeof(T)); return result;
}
template<class T> __aicore__ inline T ruda_reverse(T value,unsigned bits) {
    T result=0; for(unsigned i=0;i<bits;++i){ result=(result<<1)|(value&1); value>>=1; } return result;
}
template<class T,class E> __aicore__ inline T ruda_powi(T value,E exponent) {
    using U=typename std::make_unsigned<E>::type;
    bool negative=std::is_signed<E>::value && exponent<0;
    U power=negative ? U(0)-U(exponent) : U(exponent);
    if(negative) value=T(1)/value;
    T result=T(1); bool first=true;
    while(power){ if(power&1){result=first ? value : T(result*value);first=false;} power>>=1;if(power)value=T(value*value); }
    return result;
}
template<class T> __aicore__ inline T ruda_mod(T x,T y) {
    if constexpr(std::is_signed<T>::value) {
        if(x==std::numeric_limits<T>::min() && y==T(-1)) return T(0);
    }
    return x%y;
}
template<class T> __aicore__ inline T ruda_rem(T x,T y) {
    T r=ruda_mod(x,y);
    if constexpr(std::is_signed<T>::value) {if(r!=0 && ((r<0)!=(y<0)))r=T(r+y);}
    return r;
}
template<class T> __aicore__ inline T ruda_saturate(T a,T b,bool subtract) {
    T value; bool overflow=subtract ? __builtin_sub_overflow(a,b,&value) : __builtin_add_overflow(a,b,&value);
    if(!overflow)return value;
    if constexpr(std::is_signed<T>::value) {
        return b<0 ? (subtract ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min())
                   : (subtract ? std::numeric_limits<T>::min() : std::numeric_limits<T>::max());
    } else {return subtract ? T(0) : std::numeric_limits<T>::max();}
}
template<class T> __aicore__ inline T ruda_mul_hi(T x,T y) {
    using U=typename std::make_unsigned<T>::type;
    constexpr unsigned bits=sizeof(T)*8;
    uint64_t a=U(x),b=U(y),result;
    if constexpr(bits<64) {result=(a*b)>>bits;}
    else {
        uint64_t a0=uint32_t(a),a1=a>>32,b0=uint32_t(b),b1=b>>32;
        uint64_t p=a0*b0,q=a1*b0+(p>>32),carry=q>>32;
        q=uint32_t(q)+a0*b1;result=a1*b1+carry+(q>>32);
    }
    if constexpr(std::is_signed<T>::value) {if(x<0)result-=b;if(y<0)result-=a;}
    return T(result);
}
"#;

pub(super) fn compile(
    mut kernel: KernelDefinition,
    o: &AscendOptions,
    target: AscendTarget,
    grid: [u32; 3],
    mode: ExecutionMode,
    address: StorageType,
) -> Result<AscendKernel> {
    if mode == ExecutionMode::Validate {
        return Err(unsupported("SIMT Validate diagnostics"));
    }
    if kernel.options.debug_symbols {
        return Err(unsupported("SIMT debug symbols"));
    }
    if !matches!(
        address,
        StorageType::Scalar(ElemType::UInt(UIntKind::U32 | UIntKind::U64))
    ) {
        return Err(unsupported("SIMT address type"));
    }
    kernel.validate_typed_memory().map_err(invalid)?;
    kernel.body.lower_pointer_values().map_err(invalid)?;
    let errors = kernel.body.pop_errors();
    if !errors.is_empty() {
        return Err(invalid(errors.join("; ")));
    }
    if !kernel.scalars.is_empty()
        || !kernel.tensor_maps.is_empty()
        || kernel.options.cluster_dim.is_some()
    {
        return Err(unsupported(
            "SIMT requires specialized scalar arguments and no tensor maps/clusters",
        ));
    }
    let name = &kernel.options.kernel_name;
    if name.is_empty()
        || !(name.as_bytes()[0].is_ascii_alphabetic() || name.as_bytes()[0] == b'_')
        || !name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
    {
        return Err(invalid("invalid SIMT kernel name"));
    }
    let dim = [kernel.ruda_dim.x, kernel.ruda_dim.y, kernel.ruda_dim.z];
    let threads = dim
        .iter()
        .try_fold(1u64, |a, &b| a.checked_mul(b.into()))
        .ok_or_else(|| invalid("SIMT thread count overflow"))?;
    if threads == 0 || threads > 2048 {
        return Err(unsupported(
            "SIMT thread block must contain 1..2048 threads",
        ));
    }
    let blocks = grid
        .iter()
        .try_fold(1u64, |a, &b| a.checked_mul(b.into()))
        .ok_or_else(|| invalid("SIMT grid overflow"))?;
    if blocks > 65535 {
        return Err(unsupported("CANN SIMT grid exceeds 65535 thread blocks"));
    }
    blocks
        .checked_mul(threads)
        .ok_or_else(|| invalid("SIMT logical thread domain overflow"))?;
    let mut ids = HashSet::new();
    let bindings = kernel
        .buffers
        .iter()
        .map(|b| {
            if !ids.insert(b.id) {
                return Err(invalid("SIMT duplicate binding id"));
            }
            ty(b.ty)?;
            let count = b
                .size
                .ok_or_else(|| invalid("SIMT needs each binding's logical length"))?;
            let bytes = (count as u64)
                .checked_mul(b.ty.size() as u64)
                .ok_or_else(|| invalid("SIMT binding size overflow"))?;
            Ok(AscendBinding {
                id: b.id,
                writable: b.visibility == Visibility::ReadWrite,
                bytes,
                storage: b.ty.storage_type(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let params = kernel
        .buffers
        .iter()
        .map(|b| Ok(format!("{}* b{}", ty(b.ty)?, b.id)))
        .collect::<Result<Vec<_>>>()?
        .join(", ");
    let args = kernel
        .buffers
        .iter()
        .enumerate()
        .map(|(i, b)| Ok(format!("reinterpret_cast<{}*>(buffers[{i}])", ty(b.ty)?)))
        .collect::<Result<Vec<_>>>()?
        .join(", ");
    let mut vars = Vec::new();
    variables(&kernel.body, &mut vars);
    let mut p = Printer {
        buffers: &kernel.buffers,
        grid,
        dim,
        address,
        source: HELPERS.into(),
        declared: HashSet::new(),
        arrays: HashSet::new(),
        array_types: HashMap::new(),
        lengths: HashMap::new(),
        mode,
        temporary: 0,
        shared_bytes: 0,
    };
    p.line(format!(
        "__global__ __launch_bounds__({threads}) void ruda_simt_{name}({params}) {{"
    ));
    let mut arrays = vars
        .into_iter()
        .filter(|v| {
            matches!(
                v.kind,
                VariableKind::SharedArray { .. }
                    | VariableKind::Shared { .. }
                    | VariableKind::LocalArray { .. }
            )
        })
        .map(|v| {
            let count = match v.kind {
                VariableKind::SharedArray {
                    length,
                    unroll_factor,
                    ..
                }
                | VariableKind::LocalArray {
                    length,
                    unroll_factor,
                    ..
                } => length
                    .checked_mul(unroll_factor)
                    .ok_or_else(|| invalid("SIMT array capacity overflow"))?,
                _ => 1,
            };
            Ok((
                count
                    .checked_mul(v.ty.size())
                    .ok_or_else(|| invalid("SIMT array capacity overflow"))?,
                v,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    arrays.sort_by_key(|(bytes, _)| std::cmp::Reverse(*bytes));
    for (_, v) in arrays {
        if matches!(
            v.kind,
            VariableKind::SharedArray { .. }
                | VariableKind::Shared { .. }
                | VariableKind::LocalArray { .. }
        ) {
            p.declare(v)?;
        }
    }
    p.scope(kernel.body)?;
    p.line("}");
    if p.shared_bytes > o.ub_limit_bytes as usize {
        return Err(unsupported(
            "SIMT shared allocation exceeds configured UB limit",
        ));
    }
    p.line("extern \"C\" int32_t ruda_ascend_launch(void* stream, void** buffers) {");
    if blocks != 0 {
        p.line(format!("ruda_simt_{name}<<<dim3({},{},{}), dim3({},{},{}), 0, static_cast<aclrtStream>(stream)>>>({args});",grid[0],grid[1],grid[2],dim[0],dim[1],dim[2]));
    }
    p.line("return aclrtGetLastError(ACL_RT_THREAD_LEVEL); }");
    Ok(AscendKernel {
        source: p.source,
        entrypoint: name.clone(),
        target,
        elements: blocks,
        row_width: None,
        block_dim: grid[0],
        tile_elements: threads as u32,
        ub_bytes: p.shared_bytes as u32,
        temporary_slots: 0,
        bindings,
        initialized_outputs: true,
        simt_grid: Some(grid),
        simt_dim: Some(dim),
    })
}

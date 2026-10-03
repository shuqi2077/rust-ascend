//! Validate common RUDA IR before scheduling vector instructions. No raw C++ IR.
use super::{Result, invalid, unsupported};
use ruda_core::{ir::*, kernel::{KernelArg, KernelDefinition, Visibility}};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use super::index::Index;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Unary { Neg, Abs, Exp, Log, Sqrt, Rsqrt, Recip, Erf, Tanh, Sin, Cos }
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Binary { Add, Sub, Mul, Div, Max }
#[derive(Clone, Copy, Debug)]
pub(super) enum Node { Input(usize), UniformInput(usize,u64), Constant(u32), IndexFloat(usize), IndexSelect(usize,usize,usize), Unary(Unary, usize), Binary(Binary, usize, usize) }
impl Node { pub fn inputs(&self) -> Vec<usize> { match self {
    Self::Unary(_, a) => vec![*a], Self::Binary(_, a,b)|Self::IndexSelect(_,a,b) => vec![*a,*b], _ => vec![] } } }
#[derive(Clone,Copy,Debug)]
pub(super) enum IndexCompare {Eq,Ne,Lt,Le,Gt,Ge}
#[derive(Clone,Debug)]
pub(super) struct IndexPredicate {pub comparison:IndexCompare,pub lhs:Rc<Index>,pub rhs:Rc<Index>}
impl IndexPredicate {
    pub fn cce(&self)->String {
        let operation=match self.comparison {IndexCompare::Eq=>"==",IndexCompare::Ne=>"!=",IndexCompare::Lt=>"<",IndexCompare::Le=>"<=",IndexCompare::Gt=>">",IndexCompare::Ge=>">="};
        format!("({} {operation} {})",self.lhs.cce(),self.rhs.cce())
    }
    #[cfg(test)]
    pub fn eval(&self,lane:u64)->bool {
        let a=self.lhs.eval(lane);let b=self.rhs.eval(lane);
        match self.comparison {IndexCompare::Eq=>a==b,IndexCompare::Ne=>a!=b,IndexCompare::Lt=>a<b,IndexCompare::Le=>a<=b,IndexCompare::Gt=>a>b,IndexCompare::Ge=>a>=b}
    }
}
#[derive(Debug)]
pub(super) struct Program {
    pub name: String, pub bindings: Vec<KernelArg>, pub nodes: Vec<Node>,
    pub stores: Vec<(usize, usize)>,
    pub store_indices: HashMap<usize,Rc<Index>>,
    pub load_indices: HashMap<usize, Rc<Index>>,
    pub index_values: Vec<Rc<Index>>,
    pub predicates:Vec<IndexPredicate>,
}
#[derive(Clone, Debug)]
enum Value { Lane, Length, Inside, Outside, Predicate(usize), Index(u64), Mapped(Rc<Index>), Vector(usize) }
fn f32_type() -> Type { Type::scalar(ElemType::Float(FloatKind::F32)) }
fn is_index(ty: Type) -> bool { ty == Type::scalar(ElemType::UInt(UIntKind::U32)) || ty == Type::scalar(ElemType::UInt(UIntKind::U64)) }
fn valid_local(v: Variable) -> bool { matches!(v.kind, VariableKind::LocalMut{..} | VariableKind::LocalConst{..} | VariableKind::Versioned{..}) }
fn ident(s:&str)->bool { !["for","while","if","else","return","float","int","void","class","template","auto","const","extern","union","struct","namespace","operator","new","delete"].contains(&s) && !s.is_empty() && s.len()<=128 && s.as_bytes()[0].is_ascii_alphabetic() && s.bytes().all(|c|c.is_ascii_alphanumeric()||c==b'_') }

struct Lower { p: Program, elements:u64, partial_stores:bool, values: HashMap<Variable,Value>, loads: HashMap<usize,usize>,
    uniform_loads: HashMap<(usize,u64),usize>, loaded: HashSet<usize>, constants: HashMap<u32,usize>, wrote: HashSet<usize> }
impl Lower {
    fn resolve(&self, v: Variable) -> Result<Value> {
        if matches!(v.kind, VariableKind::Builtin(Builtin::AbsolutePosX | Builtin::AbsolutePos)) && is_index(v.ty) { return Ok(Value::Lane); }
        if let VariableKind::Constant(ConstantValue::UInt(n))=v.kind {
            if is_index(v.ty) { return Ok(if n==self.elements {Value::Length}else{Value::Index(n)}); }
        }
        self.values.get(&v).cloned().ok_or_else(|| invalid(format!("undefined or unsupported operand {v:?}")))
    }
    fn index(&self, v: Variable) -> Result<Rc<Index>> {
        Ok(match self.resolve(v)? {
            Value::Lane => Rc::new(Index::Lane), Value::Length => Rc::new(Index::Constant(self.elements)),
            Value::Index(n) => Rc::new(Index::Constant(n)), Value::Mapped(index) => index,
            _ => return Err(unsupported("non-index layout operand")),
        })
    }
    fn vector(&mut self,v:Variable)->Result<usize> {
        if v.ty != f32_type() { return Err(unsupported(format!("non-FP32 arithmetic operand {v:?}"))); }
        if let VariableKind::Constant(ConstantValue::Float(x)) = v.kind {
            let y=x as f32;
            if !y.is_finite() { return Err(unsupported("non-finite compile-time constants")); }
            let bits=y.to_bits();
            if let Some(&id)=self.constants.get(&bits) { return Ok(id); }
            let id=self.p.nodes.len(); self.p.nodes.push(Node::Constant(bits)); self.constants.insert(bits,id);return Ok(id);
        }
        match self.resolve(v)? { Value::Vector(n)=>Ok(n), _=>Err(unsupported("lane indices/metadata cannot become floating point data")) }
    }
    fn assign(&mut self,out:Variable,value:Value)->Result<()> {
        if !valid_local(out) { return Err(invalid(format!("not a local destination: {out:?}"))); }
        let ty_ok=match value { Value::Vector(_)=>out.ty==f32_type(),Value::Outside|Value::Inside|Value::Predicate(_)=>out.ty==Type::scalar(ElemType::Bool),_=>is_index(out.ty) };
        if !ty_ok { return Err(invalid("IR output type does not match operation")); }
        if self.values.contains_key(&out) && !matches!(out.kind,VariableKind::LocalMut{..}) { return Err(invalid("immutable local assigned twice")); }
        self.values.insert(out,value);Ok(())
    }
    fn array(&self,v:Variable,writable:bool)->Result<usize> {
        let id=match v.kind {
            VariableKind::GlobalInputArray(id)|VariableKind::GlobalOutputArray(id)=>id,
            _=>return Err(unsupported("expected global buffer")),
        };
        let i=self.p.bindings.iter().position(|b|b.id==id).ok_or_else(||invalid("unknown buffer id"))?;
        let b=&self.p.bindings[i];
        if b.ty!=v.ty || (writable && b.visibility!=Visibility::ReadWrite) { return Err(invalid("buffer visibility/type mismatch")); }
        Ok(i)
    }
    fn length_array(&self,v:Variable)->Result<()> {
        match v.kind { VariableKind::GlobalInputArray(_)=>{self.array(v,false)?;},VariableKind::GlobalOutputArray(_)=>{self.array(v,true)?;},_=>return Err(unsupported("length of non-global array")) }; Ok(())
    }
    fn add(&mut self,out:Variable,node:Node)->Result<()> { let n=self.p.nodes.len();self.p.nodes.push(node);self.assign(out,Value::Vector(n)) }
    fn instruction(&mut self,i:&Instruction)->Result<()> {
        if !i.modes.fp_math_mode.is_empty() { return Err(unsupported("fast-math modes must be lowered explicitly, not silently dropped")); }
        let out=i.out;
        match &i.operation {
            Operation::NonSemantic(_) => Ok(()),
            Operation::Copy(v) => {
                let dst=out.ok_or_else(||invalid("copy has no destination"))?;
                let value=if dst.ty==f32_type(){Value::Vector(self.vector(*v)?)}else{self.resolve(*v)?};
                self.assign(dst,value)
            },
            Operation::Metadata(Metadata::Length{var}|Metadata::BufferLength{var})=>{
                self.length_array(*var)?;
                let size=self.p.bindings[self.array(*var,false)?].size.map(|n|n as u64).unwrap_or(self.elements);
                self.assign(out.ok_or_else(||invalid("metadata output missing"))?,if size==self.elements{Value::Length}else{Value::Index(size)})
            },
            Operation::Comparison(comparison)=>{
                let (kind,op)=match comparison {Comparison::Equal(op)=>(IndexCompare::Eq,op),Comparison::NotEqual(op)=>(IndexCompare::Ne,op),
                    Comparison::Lower(op)=>(IndexCompare::Lt,op),Comparison::LowerEqual(op)=>(IndexCompare::Le,op),
                    Comparison::Greater(op)=>(IndexCompare::Gt,op),Comparison::GreaterEqual(op)=>(IndexCompare::Ge,op),
                    _=>return Err(unsupported("only unsigned layout/index comparisons are supported"))};
                let dst=out.ok_or_else(||invalid("comparison output missing"))?;
                if matches!(kind,IndexCompare::Lt|IndexCompare::Ge)
                    && matches!((self.resolve(op.lhs)?,self.resolve(op.rhs)?),(Value::Lane,Value::Length)) {
                    return self.assign(dst,if matches!(kind,IndexCompare::Lt) {Value::Inside} else {Value::Outside});
                }
                if !is_index(op.lhs.ty) || op.lhs.ty!=op.rhs.ty {return Err(unsupported("comparison requires matching unsigned index types"));}
                let predicate=IndexPredicate {comparison:kind,lhs:self.index(op.lhs)?,rhs:self.index(op.rhs)?};
                let id=self.p.predicates.len();self.p.predicates.push(predicate);self.assign(dst,Value::Predicate(id))
            },
            Operation::Operator(Operator::Select(op))=>{
                let cond=self.resolve(op.cond)?;let dst=out.ok_or_else(||invalid("select output missing"))?;
                match cond {
                    Value::Inside=>{let value=self.vector(op.then)?;self.assign(dst,Value::Vector(value))},
                    Value::Outside=>{let value=self.vector(op.or_else)?;self.assign(dst,Value::Vector(value))},
                    Value::Predicate(id)=>{
                        let a=self.vector(op.then)?;let b=self.vector(op.or_else)?;self.add(dst,Node::IndexSelect(id,a,b))
                    },
                    _=>Err(unsupported("select condition must be a checked unsigned index predicate")),
                }
            },
            Operation::Operator(Operator::Not(op))=>{
                let value=match self.resolve(op.input)?{Value::Inside=>Value::Outside,Value::Outside=>Value::Inside,_=>return Err(unsupported("non-domain boolean negation"))};
                self.assign(out.ok_or_else(||invalid("not output missing"))?,value)
            },
            Operation::Operator(Operator::And(op))=>{
                let value=match (self.resolve(op.lhs)?,self.resolve(op.rhs)?){
                    (Value::Inside,Value::Inside)=>Value::Inside,
                    (Value::Outside,Value::Outside)=>Value::Outside,
                    _=>return Err(unsupported("conjunction requires identical domain predicates")),
                };
                self.assign(out.ok_or_else(||invalid("and output missing"))?,value)
            },
            Operation::Operator(Operator::Cast(op))=>{
                let dst=out.ok_or_else(||invalid("cast output missing"))?;
                if dst.ty==op.input.ty {let value=if dst.ty==f32_type(){Value::Vector(self.vector(op.input)?)}else{self.resolve(op.input)?};return self.assign(dst,value);}
                if is_index(dst.ty)&&is_index(op.input.ty){
                    let mut value=self.resolve(op.input)?;
                    if let Value::Mapped(index)=&value {if dst.ty==Type::new(UIntKind::U32.into()) && index.bounds(self.elements)?.1>u32::MAX as u64{return Err(unsupported("narrowing layout index may wrap"));}}
                    if let Value::Index(n)=value {let n=if dst.ty==Type::new(UIntKind::U32.into()){n as u32 as u64}else{n};value=if n==self.elements{Value::Length}else{Value::Index(n)};}
                    return self.assign(dst,value);
                }
                if dst.ty==f32_type() && is_index(op.input.ty) {
                    let index=self.index(op.input)?;
                    if index.bounds(self.elements)?.1>(1u64<<24) {
                        return Err(unsupported("integer-to-FP32 index cast must be provably exact (0..=2^24)"));
                    }
                    let id=self.p.index_values.len();self.p.index_values.push(index);
                    return self.add(dst,Node::IndexFloat(id));
                }
                Err(unsupported("non-identity data cast"))
            },
            Operation::Operator(Operator::Reinterpret(op))=>{
                let dst=out.ok_or_else(||invalid("reinterpret output missing"))?;
                if dst.ty!=f32_type() || op.input.ty!=Type::new(UIntKind::U32.into()){return Err(unsupported("reinterpret must preserve FP32 scalar bits"));}
                let VariableKind::Constant(ConstantValue::UInt(bits))=op.input.kind else{return Err(unsupported("nonconstant reinterpret"));};
                self.add(dst,Node::Constant(bits as u32))
            },
            Operation::Branch(Branch::If(branch))=>{
                match self.resolve(branch.cond)? {
                    Value::Inside => {
                        // Every dispatched lane is in [0,elements). Keep the original
                        // guarded operations and their order; do not lower data branches.
                        for instruction in &branch.scope.instructions {self.instruction(instruction)?;}
                        Ok(())
                    },
                    Value::Outside => {
                        let ops:Vec<_>=branch.scope.instructions.iter().filter(|i|!matches!(i.operation,Operation::NonSemantic(_))).collect();
                        if ops.len()!=1 || !matches!(ops[0].operation,Operation::Branch(Branch::Return)) { return Err(unsupported("only a canonical tail-guard return is allowed")); }
                        Ok(())
                    },
                    _ => Err(unsupported("data-dependent branch")),
                }
            },
            Operation::Operator(Operator::Index(op)|Operator::UncheckedIndex(op))=>{
                if op.vector_size!=0 || op.unroll_factor!=1 { return Err(unsupported("loads must index one scalar")); }
                let a=self.array(op.list,false)?;
                let index=self.index(op.index)?;
                let size=self.p.bindings[a].size.map(|n|n as u64).unwrap_or(self.elements);
                if self.elements!=0 && index.bounds(self.elements)?.1>=size{return Err(invalid("layout load may exceed the bound buffer"));}
                if self.p.bindings[a].visibility==Visibility::ReadWrite && *index!=Index::Lane{return Err(unsupported("mapped in-place reads require scatter dependency analysis"));}
                if self.wrote.contains(&a){return Err(unsupported("load after store needs explicit ordering lowering"));}
                self.loaded.insert(a);
                if let Index::Constant(offset)=*index {
                    // Readonly uniform slots may have different constant offsets in one binding.
                    // Each is loaded on-device, not specialized from its value on the host.
                    let id=if let Some(&id)=self.uniform_loads.get(&(a,offset)) {id} else {
                        let id=self.p.nodes.len();self.p.nodes.push(Node::UniformInput(a,offset));
                        self.uniform_loads.insert((a,offset),id);id
                    };
                    return self.assign(out.ok_or_else(||invalid("load output missing"))?,Value::Vector(id));
                }
                if self.p.load_indices.get(&a).is_some_and(|old|old!=&index){return Err(unsupported("multiple distinct layouts for one input binding"));}
                self.p.load_indices.insert(a,index);
                let id=if let Some(&n)=self.loads.get(&a){n}else{let n=self.p.nodes.len();self.p.nodes.push(Node::Input(a));self.loads.insert(a,n);n};
                self.assign(out.ok_or_else(||invalid("load output missing"))?,Value::Vector(id))
            },
            Operation::Operator(Operator::IndexAssign(op)|Operator::UncheckedIndexAssign(op))=>{
                if op.vector_size!=0 || op.unroll_factor!=1 { return Err(unsupported("stores must index one scalar")); }
                let a=self.array(out.ok_or_else(||invalid("store output missing"))?,true)?;
                let index=self.index(op.index)?;
                if *index!=Index::Lane {
                    if !self.partial_stores {return Err(unsupported("mapped stores require explicit partial-map compilation"));}
                    if self.elements>1 && !index.injective() {return Err(unsupported("mapped stores must be provably injective"));}
                    if self.loaded.contains(&a) {return Err(unsupported("mapped stores cannot read their output binding"));}
                }
                let size=self.p.bindings[a].size.map(|n|n as u64).unwrap_or(self.elements);
                if self.elements!=0 && index.bounds(self.elements)?.1>=size {return Err(invalid("store may exceed the bound output"));}
                if !self.wrote.insert(a){return Err(unsupported("multiple stores to one output"));}
                self.p.store_indices.insert(a,index);
                let v=self.vector(op.value)?;self.p.stores.push((a,v));Ok(())
            },
            Operation::Arithmetic(a)=>{
                let dst=out.ok_or_else(||invalid("arithmetic output missing"))?;
                if is_index(dst.ty) {
                    let (kind,op)=match a{Arithmetic::Add(op)=>('+',op),Arithmetic::Sub(op)=>('-',op),Arithmetic::Mul(op)=>('*',op),Arithmetic::Div(op)=>('/',op),Arithmetic::Modulo(op)=>('%',op),_=>return Err(unsupported("layout index operation"))};
                    let max=if dst.ty==Type::new(UIntKind::U32.into()){u32::MAX as u64}else{u64::MAX};
                    let index=Index::binary(kind,self.index(op.lhs)?,self.index(op.rhs)?,self.elements,max)?;
                    let value=match index.as_ref(){Index::Lane=>Value::Lane,Index::Constant(n) if *n==self.elements=>Value::Length,Index::Constant(n)=>Value::Index(*n),_=>Value::Mapped(index)};
                    return self.assign(dst,value);
                }
                let binary=match a { Arithmetic::Add(op)=>Some((Binary::Add,op)),Arithmetic::Sub(op)=>Some((Binary::Sub,op)),Arithmetic::Mul(op)=>Some((Binary::Mul,op)),Arithmetic::Div(op)=>Some((Binary::Div,op)),Arithmetic::Max(op)=>Some((Binary::Max,op)),_=>None };
                if let Some((kind,op))=binary {let lhs=self.vector(op.lhs)?;let rhs=self.vector(op.rhs)?;return self.add(dst,Node::Binary(kind,lhs,rhs));}
                if let Arithmetic::Powi(op)=a {
                    let (negative,mut exponent)=match op.rhs.kind {
                        VariableKind::Constant(ConstantValue::Int(n)) if matches!(op.rhs.ty,Type::Scalar(StorageType::Scalar(ElemType::Int(IntKind::I32|IntKind::I64))))=>(n<0,n.unsigned_abs()),
                        VariableKind::Constant(ConstantValue::UInt(n)) if is_index(op.rhs.ty)=>(false,n),
                        _=>return Err(unsupported("FP32 Powi requires a specialized constant signed/unsigned integer exponent")),
                    };
                    let mut base=self.vector(op.lhs)?;
                    if negative {let node=self.p.nodes.len();self.p.nodes.push(Node::Unary(Unary::Recip,base));base=node;}
                    let mut result=None;
                    while exponent!=0 {
                        if exponent&1!=0 {result=Some(if let Some(previous)=result {
                            let node=self.p.nodes.len();self.p.nodes.push(Node::Binary(Binary::Mul,previous,base));node
                        } else {base});}
                        exponent>>=1;
                        if exponent!=0 {let node=self.p.nodes.len();self.p.nodes.push(Node::Binary(Binary::Mul,base,base));base=node;}
                    }
                    let result=match result {Some(node)=>node,None=>self.vector(Variable::constant(ConstantValue::Float(1.),f32_type()))?};
                    return self.assign(dst,Value::Vector(result));
                }
                let (kind,op)=match a { Arithmetic::Neg(op)=>(Unary::Neg,op),Arithmetic::Abs(op)=>(Unary::Abs,op),Arithmetic::Exp(op)=>(Unary::Exp,op),Arithmetic::Log(op)=>(Unary::Log,op),Arithmetic::Sqrt(op)=>(Unary::Sqrt,op),Arithmetic::InverseSqrt(op)=>(Unary::Rsqrt,op),Arithmetic::Recip(op)=>(Unary::Recip,op),Arithmetic::Erf(op)=>(Unary::Erf,op),Arithmetic::Tanh(op)=>(Unary::Tanh,op),Arithmetic::Sin(op)=>(Unary::Sin,op),Arithmetic::Cos(op)=>(Unary::Cos,op),_=>return Err(unsupported(format!("arithmetic {a:?}"))) };
                let input=self.vector(op.input)?;self.add(dst,Node::Unary(kind,input))
            },
            other=>Err(unsupported(format!("operation {other:?}; no fallback to another kernel/compiler"))),
        }
    }
}

#[cfg(test)]
pub(super) fn lower(k:KernelDefinition,elements:u64)->Result<Program> {
    lower_map(k,elements,false)
}
pub(super) fn lower_map(mut k:KernelDefinition,elements:u64,partial_stores:bool)->Result<Program> {
    if !ident(&k.options.kernel_name) {return Err(invalid("kernel entry must be a short ASCII identifier beginning with a letter"));}
    if k.options.debug_symbols || k.options.cluster_dim.is_some() {return Err(unsupported("debug symbols/clusters"));}
    if k.ruda_dim.x==0 || k.ruda_dim.y!=1 || k.ruda_dim.z!=1 {return Err(unsupported("only nonzero one-dimensional logical lane launch is accepted"));}
    if !k.tensor_maps.is_empty()||!k.scalars.is_empty()||!k.body.const_arrays.is_empty(){return Err(unsupported("tensor maps, runtime scalars and constant arrays"));}
    let errors=k.body.pop_errors();if !errors.is_empty(){return Err(invalid(errors.join("; ")));}
    if k.buffers.is_empty()||k.buffers.len()>8{return Err(unsupported("expected 1..8 total buffers"));}
    let mut ids=HashSet::new();let mut inputs=0;let mut outputs=0;
    for b in &k.buffers {
        if !ids.insert(b.id){return Err(invalid("duplicate kernel buffer id"));}
        if b.size.is_some_and(|n|n as u64>u32::MAX as u64){return Err(unsupported("buffer length exceeds u32"));}
        if b.ty!=f32_type(){return Err(unsupported("only scalar FP32 buffers; no implicit precision changes"));}
        if b.has_extended_meta|| (!partial_stores && b.visibility==Visibility::ReadWrite && b.size.is_some_and(|n|n as u64!=elements)){return Err(unsupported("extended metadata or contiguous output size mismatch"));}
        if b.visibility==Visibility::Read {inputs+=1}else{outputs+=1}
    }
    if inputs>7||outputs==0||outputs>4{return Err(unsupported("at most eight total buffers/four outputs, and at least one output"));}
    // Read actual scope instructions. Unused local declarations carry no effects;
    // every operation, operand and referenced special storage is checked below.
    if k.body.instructions.len()>4096{return Err(unsupported("instruction limit exceeded"));}
    let mut l=Lower{p:Program{name:k.options.kernel_name,bindings:k.buffers,nodes:vec![],stores:vec![],store_indices:HashMap::new(),load_indices:HashMap::new(),index_values:vec![],predicates:vec![]},elements,partial_stores,values:HashMap::new(),loads:HashMap::new(),uniform_loads:HashMap::new(),loaded:HashSet::new(),constants:HashMap::new(),wrote:HashSet::new()};
    for (n,i) in k.body.instructions.iter().enumerate(){l.instruction(i).map_err(|e|{
        let text=format!("instruction {n}: {e}");
        if matches!(e,ruda_core::compiler::CompilationError::UnsupportedInstruction{..}){unsupported(text)}else{invalid(text)}
    })?;}
    if l.wrote.len()!=outputs{return Err(invalid("every declared output must be written exactly once"));}
    if l.loaded.len()>8{return Err(unsupported("at most eight loaded buffers, including in-place inputs"));}
    if l.p.bindings.iter().enumerate().any(|(i,b)|b.visibility==Visibility::Read&&!l.loaded.contains(&i)){return Err(unsupported("unused input bindings must be removed before lowering"));}
    Ok(l.p)
}

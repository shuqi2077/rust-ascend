//! Position-aware causal mask authored in RUDA's unsigned-index/select IR.
use super::{Result,invalid};
use ruda_core::{ir::*,kernel::{KernelArg,KernelDefinition,KernelOptions,Visibility},launch::RudaDim};

/// Mask[B,Q,K] is -infinity when key_start+key > query_start+query, otherwise +0.
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub struct CausalMaskSpec {pub batch:u32,pub queries:u32,pub keys:u32,pub query_start:u64,pub key_start:u64}
impl CausalMaskSpec {
    pub fn elements(self)->Result<u64> {
        if self.queries==0 || self.keys==0 {return Err(invalid("causal mask requires positive query/key lengths"));}
        if self.query_start.checked_add(self.queries as u64-1).is_none() || self.key_start.checked_add(self.keys as u64-1).is_none() {
            return Err(invalid("causal absolute position overflow"));
        }
        let elements=(self.batch as u64).checked_mul(self.queries as u64).and_then(|n|n.checked_mul(self.keys as u64))
            .ok_or_else(||invalid("causal mask shape overflow"))?;
        if elements>u32::MAX as u64 {return Err(invalid("causal mask domain exceeds u32"));}Ok(elements)
    }
}
pub fn definition(spec:CausalMaskSpec)->Result<KernelDefinition> {
    let n=spec.elements()?;let f=Type::new(FloatKind::F32.into());let u=Type::new(UIntKind::U64.into());
    let mut kernel=KernelDefinition {buffers:vec![KernelArg {id:0,visibility:Visibility::ReadWrite,ty:f,size:Some(n as usize),has_extended_meta:false}],
        tensor_maps:vec![],scalars:vec![],ruda_dim:RudaDim::new_1d(64),body:Scope::root(false),options:KernelOptions {kernel_name:"ruda_cann_causal_mask".into(),..Default::default()}};
    let mut id=0;
    let mut emit=|operation:Operation,ty:Type| {
        let out=Variable::new(VariableKind::LocalConst{id},ty);id+=1;
        kernel.body.instructions.push(Instruction::new(operation,out));out
    };
    let constant=|n:u64|Variable::constant(ConstantValue::UInt(n),u);
    let lane=Variable::builtin(Builtin::AbsolutePosX,UIntKind::U64.into());
    let row=emit(Arithmetic::Div(BinaryOperator {lhs:lane,rhs:constant(spec.keys as u64)}).into(),u);
    let row=emit(Arithmetic::Modulo(BinaryOperator {lhs:row,rhs:constant(spec.queries as u64)}).into(),u);
    let column=emit(Arithmetic::Modulo(BinaryOperator {lhs:lane,rhs:constant(spec.keys as u64)}).into(),u);
    let query=emit(Arithmetic::Add(BinaryOperator {lhs:row,rhs:constant(spec.query_start)}).into(),u);
    let key=emit(Arithmetic::Add(BinaryOperator {lhs:column,rhs:constant(spec.key_start)}).into(),u);
    let cond=emit(Comparison::Greater(BinaryOperator {lhs:key,rhs:query}).into(),Type::scalar(ElemType::Bool));
    let negative_infinity=emit(Operator::Reinterpret(UnaryOperator {input:Variable::constant(ConstantValue::UInt(0xff800000),Type::new(UIntKind::U32.into()))}).into(),f);
    let mask=emit(Operator::Select(Select {cond,then:negative_infinity,or_else:Variable::constant(ConstantValue::Float(0.),f)}).into(),f);
    kernel.body.instructions.push(Instruction::new(Operator::IndexAssign(IndexAssignOperator {index:lane,value:mask,vector_size:0,unroll_factor:1}),
        Variable::new(VariableKind::GlobalOutputArray(0),f)));Ok(kernel)
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ascend::{AscendCompiler,AscendOptions,AscendTarget,lower::{self,Node}};
    use ruda_core::{compiler::Compiler,launch::ExecutionMode};
    #[test]
    fn causal_masks_match_prefill_cached_and_large_absolute_positions() {
        for (batch,queries,keys) in [(0,3,7),(1,3,7),(2,16,32),(2,32,96)] {
            for (query_start,key_start) in [(0,0),(29,0),(9,7),(1u64<<40,1u64<<40),(u64::MAX-96,u64::MAX-96)] {
                let spec=CausalMaskSpec {batch,queries,keys,query_start,key_start};let n=spec.elements().unwrap();
                let ir=definition(spec).unwrap();let p=lower::lower(ir.clone(),n).unwrap();let mut values:Vec<Vec<f32>>=vec![];
                for node in &p.nodes {values.push(match *node {
                    Node::Constant(bits)=>vec![f32::from_bits(bits);n as usize],
                    Node::IndexSelect(i,a,b)=>(0..n as usize).map(|lane|if p.predicates[i].eval(lane as u64) {values[a][lane]} else {values[b][lane]}).collect(),
                    _=>panic!("causal mask unexpectedly needs floating arithmetic or data input"),
                });}
                let out=&values[p.stores[0].1];
                for b in 0..batch as usize {for q in 0..queries as usize {for k in 0..keys as usize {
                    let bits=if key_start+k as u64>query_start+q as u64 {f32::NEG_INFINITY.to_bits()} else {0};
                    assert_eq!(out[(b*queries as usize+q)*keys as usize+k].to_bits(),bits);
                }}}
                let compiled=AscendCompiler.compile(ir,&AscendOptions {target:Some(AscendTarget::Ascend950DT),elements:n,..Default::default()},ExecutionMode::Checked,UIntKind::U64.into()).unwrap();
                assert!(!compiled.source().contains("static_cast<float>"));assert!(compiled.source().contains("0xff800000U"));
                assert_eq!(compiled.bindings()[0].bytes,n*4);
            }
        }
    }
    #[test]
    fn causal_domain_and_position_boundaries_are_checked_without_a_fp32_index_limit() {
        let large=CausalMaskSpec {batch:1,queries:8192,keys:8192,query_start:1u64<<40,key_start:0};
        AscendCompiler.compile(definition(large).unwrap(),&AscendOptions {target:Some(AscendTarget::Ascend950DT),elements:large.elements().unwrap(),..Default::default()},ExecutionMode::Checked,UIntKind::U64.into()).unwrap();
        for spec in [CausalMaskSpec {queries:0,..large},CausalMaskSpec {keys:0,..large},CausalMaskSpec {queries:65536,keys:65536,..large},
            CausalMaskSpec {query_start:u64::MAX,..large},CausalMaskSpec {key_start:u64::MAX,..large}] {assert!(definition(spec).is_err());}
    }
}

//! Native FP32 last-axis bias broadcast with an optional same-shape residual.
use super::{Result,invalid};
use ruda_core::{ir::*,kernel::{KernelArg,KernelDefinition,KernelOptions,Visibility},launch::RudaDim};

#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub struct BiasAddSpec {pub rows:u64,pub width:u32,pub residual:bool}
impl BiasAddSpec {
    pub fn elements(self)->Result<u64> {
        if self.width==0 {return Err(invalid("bias add requires a positive last-axis width"));}
        self.rows.checked_mul(self.width as u64).filter(|&n|n<=u32::MAX as u64)
            .ok_or_else(||invalid("bias add complete domain exceeds u32"))
    }
}
fn f()->Type {Type::new(FloatKind::F32.into())}
fn u()->Type {Type::new(UIntKind::U64.into())}
pub fn definition(spec:BiasAddSpec)->Result<KernelDefinition> {
    let elements=spec.elements()?;let mut sizes=vec![elements,spec.width as u64];
    if spec.residual {sizes.push(elements);}sizes.push(elements);
    let output=sizes.len() as u32-1;
    let mut kernel=KernelDefinition {buffers:sizes.iter().enumerate().map(|(id,&size)|KernelArg {id:id as u32,
        visibility:if id as u32==output {Visibility::ReadWrite} else {Visibility::Read},ty:f(),size:Some(size as usize),has_extended_meta:false}).collect(),
        tensor_maps:vec![],scalars:vec![],ruda_dim:RudaDim::new_1d(64),body:Scope::root(false),
        options:KernelOptions {kernel_name:(if spec.residual {"ruda_cann_residual_bias_add"} else {"ruda_cann_bias_add"}).into(),..Default::default()}};
    let lane=Variable::builtin(Builtin::AbsolutePosX,UIntKind::U64.into());let mut next=0;
    let mut op=|operation:Operation,ty:Type| {
        let out=Variable::new(VariableKind::LocalConst{id:next},ty);next+=1;
        kernel.body.instructions.push(Instruction::new(operation,out));out
    };
    let read=|id,index|Operator::Index(IndexOperator {list:Variable::new(VariableKind::GlobalInputArray(id),f()),
        index,vector_size:0,unroll_factor:1});
    let mut value=op(read(0,lane).into(),f());
    if spec.residual {
        let residual=op(read(2,lane).into(),f());value=op(Arithmetic::Add(BinaryOperator {lhs:value,rhs:residual}).into(),f());
    }
    let column=op(Arithmetic::Modulo(BinaryOperator {lhs:lane,rhs:Variable::constant(ConstantValue::UInt(spec.width as u64),u())}).into(),u());
    let bias=op(read(1,column).into(),f());let value=op(Arithmetic::Add(BinaryOperator {lhs:value,rhs:bias}).into(),f());
    kernel.body.instructions.push(Instruction::new(Operator::IndexAssign(IndexAssignOperator {index:lane,value,vector_size:0,unroll_factor:1}),
        Variable::new(VariableKind::GlobalOutputArray(output),f())));Ok(kernel)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ascend::{AscendCompiler,AscendOptions,AscendTarget,lower::{self,Node,Binary}};
    use ruda_core::{compiler::Compiler,launch::ExecutionMode};
    #[test]
    fn native_broadcast_and_residual_order_cover_tails_and_empty_rows() {
        for rows in [0,1,2,7] {for width in [1,7,31,32,33,4097] {for residual in [false,true] {
            let spec=BiasAddSpec {rows,width,residual};let n=spec.elements().unwrap();let kernel=definition(spec).unwrap();
            let ordered=rows==1 && width==1;
            let x:Vec<f32>=(0..n).map(|i|if ordered {1e20} else {(i%13) as f32/8.-0.5}).collect();
            let bias:Vec<f32>=(0..width).map(|i|if ordered {1.} else {(i%7) as f32/16.-0.125}).collect();
            let r:Vec<f32>=(0..n).map(|i|if ordered {-1e20} else {(i%5) as f32/4.-0.25}).collect();let inputs=[&x,&bias,&r];
            let plan=lower::lower_map(kernel.clone(),n,false).unwrap();let mut values:Vec<Vec<f32>>=vec![];
            for node in &plan.nodes {values.push(match *node {
                Node::Input(id)=>(0..n).map(|lane|inputs[id][plan.load_indices[&id].eval(lane) as usize]).collect(),
                Node::UniformInput(id,offset)=>vec![inputs[id][offset as usize];n as usize],
                Node::Binary(Binary::Add,a,b)=>values[a].iter().zip(&values[b]).map(|(&a,&b)|a+b).collect(),
                _=>panic!("unexpected bias-add operation"),
            });}
            let (id,value)=plan.stores[0];assert_eq!(plan.store_indices[&id].bounds(n).unwrap(),if n==0 {(0,0)} else {(0,n-1)});
            for lane in 0..n as usize {assert_eq!(values[value][lane],(if residual {x[lane]+r[lane]} else {x[lane]})+bias[lane%width as usize]);}
            let compiled=AscendCompiler.compile(kernel,&AscendOptions {target:Some(AscendTarget::Ascend950DT),elements:n,..Default::default()},
                ExecutionMode::Checked,UIntKind::U64.into()).unwrap();
            assert_eq!(compiled.bindings()[1].bytes,width as u64*4);assert!(!compiled.bindings()[1].writable);
            assert_eq!(compiled.bindings().last().unwrap().bytes,n*4);
        }}}
        // The documented order must not be reassociated by the IR producer.
        let plan=lower::lower_map(definition(BiasAddSpec {rows:1,width:1,residual:true}).unwrap(),1,false).unwrap();
        let additions:Vec<_>=plan.nodes.iter().filter(|node|matches!(node,Node::Binary(Binary::Add,_,_))).collect();assert_eq!(additions.len(),2);
    }
    #[test]
    fn bias_domain_keeps_positive_width_and_complete_u32_bound() {
        for spec in [BiasAddSpec {rows:1,width:0,residual:false},BiasAddSpec {rows:u64::MAX,width:2,residual:true},
            BiasAddSpec {rows:2,width:u32::MAX,residual:false}] {assert!(definition(spec).is_err());}
        for spec in [BiasAddSpec {rows:0,width:u32::MAX,residual:true},BiasAddSpec {rows:1,width:u32::MAX,residual:false}] {
            assert!(definition(spec).is_ok());
        }
    }
}

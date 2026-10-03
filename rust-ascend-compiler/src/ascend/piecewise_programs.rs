//! Explicit FP32 piecewise maps using local predicates, not global Bool buffers.
use super::{Result,invalid};
use ruda_core::{ir::*,kernel::{KernelArg,KernelDefinition,KernelOptions,Visibility},launch::RudaDim};

#[derive(Clone,Copy,Debug,PartialEq)]
pub enum PiecewiseActivation {Relu,Clamp {min:f32,max:f32}}
fn f()->Type {Type::new(FloatKind::F32.into())}
pub fn definition(activation:PiecewiseActivation,elements:u64,backward:bool)->Result<KernelDefinition> {
    if elements>u32::MAX as u64 {return Err(invalid("piecewise activation domain exceeds u32"));}
    let output=if backward {2} else {1};
    let mut kernel=KernelDefinition {buffers:(0..=output).map(|id|KernelArg {id,
        visibility:if id==output {Visibility::ReadWrite} else {Visibility::Read},ty:f(),size:Some(elements as usize),has_extended_meta:false}).collect(),
        tensor_maps:vec![],scalars:vec![],ruda_dim:RudaDim::new_1d(64),body:Scope::root(false),
        options:KernelOptions {kernel_name:match (activation,backward) {
            (PiecewiseActivation::Relu,false)=>"ruda_cann_relu",(PiecewiseActivation::Relu,true)=>"ruda_cann_relu_backward",
            (PiecewiseActivation::Clamp {..},false)=>"ruda_cann_clamp",(PiecewiseActivation::Clamp {..},true)=>"ruda_cann_clamp_backward",
        }.into(),..Default::default()}};
    let lane=Variable::builtin(Builtin::AbsolutePosX,UIntKind::U64.into());let mut next=0;
    let mut op=|operation:Operation,ty:Type| {
        let out=Variable::new(VariableKind::LocalConst{id:next},ty);next+=1;
        kernel.body.instructions.push(Instruction::new(operation,out));out
    };
    let read=|id|Operator::Index(IndexOperator {list:Variable::new(VariableKind::GlobalInputArray(id),f()),index:lane,vector_size:0,unroll_factor:1});
    let bits=|value:f32|Operator::Reinterpret(UnaryOperator {input:Variable::constant(ConstantValue::UInt(value.to_bits() as u64),Type::new(UIntKind::U32.into()))});
    let x=op(read(0).into(),f());let grad=if backward {Some(op(read(1).into(),f()))} else {None};
    let zero=op(bits(0.).into(),f());let bool_type=Type::new(ElemType::Bool);
    let result=match activation {
        PiecewiseActivation::Relu=>{
            let mask=op(Comparison::LowerEqual(BinaryOperator {lhs:x,rhs:zero}).into(),bool_type);
            op(Operator::Select(Select {cond:mask,then:zero,or_else:grad.unwrap_or(x)}).into(),f())
        },
        PiecewiseActivation::Clamp {min,max}=>{
            let max=op(bits(max).into(),f());let min=op(bits(min).into(),f());
            let hi=op(Comparison::Greater(BinaryOperator {lhs:x,rhs:max}).into(),bool_type);
            let upper=op(Operator::Select(Select {cond:hi,then:max,or_else:x}).into(),f());
            let lo=op(Comparison::Lower(BinaryOperator {lhs:upper,rhs:min}).into(),bool_type);
            if let Some(grad)=grad {
                let upper_grad=op(Operator::Select(Select {cond:hi,then:zero,or_else:grad}).into(),f());
                op(Operator::Select(Select {cond:lo,then:zero,or_else:upper_grad}).into(),f())
            } else {op(Operator::Select(Select {cond:lo,then:min,or_else:upper}).into(),f())}
        },
    };
    kernel.body.instructions.push(Instruction::new(Operator::IndexAssign(IndexAssignOperator {index:lane,value:result,vector_size:0,unroll_factor:1}),
        Variable::new(VariableKind::GlobalOutputArray(output),f())));Ok(kernel)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ascend::{AscendCompiler,AscendOptions,AscendTarget,tests::evaluate};
    use ruda_core::{compiler::Compiler,launch::ExecutionMode};
    #[test]
    fn relu_and_ordered_clamp_keep_nan_zero_bounds_and_grad_bits() {
        let values=[f32::NEG_INFINITY,-2.,-1.,-0.,0.,1.,2.,f32::INFINITY,f32::from_bits(0x7fc12345)];
        for activation in [PiecewiseActivation::Relu,PiecewiseActivation::Clamp {min:-1.,max:1.},
            PiecewiseActivation::Clamp {min:2.,max:-1.},PiecewiseActivation::Clamp {min:f32::NEG_INFINITY,max:f32::INFINITY},
            PiecewiseActivation::Clamp {min:f32::NAN,max:1.},PiecewiseActivation::Clamp {min:-1.,max:f32::NAN},
            PiecewiseActivation::Clamp {min:-0.,max:0.}] {
            for count in [0usize,1,7,65,257] {for backward in [false,true] {
                let x:Vec<f32>=(0..count).map(|i|values[i%values.len()]).collect();
                let grad:Vec<f32>=(0..count).map(|i|if i%3==0 {-0.} else {(i%7) as f32-3.}).collect();
                let kernel=definition(activation,count as u64,backward).unwrap();let actual=evaluate(kernel.clone(),count,[&x,&grad,&[]]);
                for lane in 0..count {let x=x[lane];let grad=grad[lane];let expected=match activation {
                    PiecewiseActivation::Relu=>if x<=0. {0.} else if backward {grad} else {x},
                    PiecewiseActivation::Clamp {min,max}=>{
                        let hi=x>max;let upper=if hi {max} else {x};let lo=upper<min;
                        if backward {if hi || lo {0.} else {grad}} else if lo {min} else {upper}
                    },
                };assert_eq!(actual[0][lane].to_bits(),expected.to_bits());}
                let compiled=AscendCompiler.compile(kernel,&AscendOptions {target:Some(AscendTarget::Ascend950DT),elements:count as u64,..Default::default()},
                    ExecutionMode::Checked,UIntKind::U64.into()).unwrap();
                assert_eq!(compiled.bindings().len(),if backward {3} else {2});assert!(compiled.source().contains("AscendC::Compare("));
            }}
        }
        assert!(definition(PiecewiseActivation::Relu,u32::MAX as u64+1,false).is_err());
    }
}

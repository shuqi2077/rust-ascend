//! Explicit FP32 piecewise maps using local predicates, not global Bool buffers.
use super::{Result,invalid};
use ruda_core::{ir::*,kernel::{KernelArg,KernelDefinition,KernelOptions,Visibility},launch::RudaDim};

#[derive(Clone,Copy,Debug,PartialEq)]
pub enum PiecewiseActivation {Relu,Clamp {min:f32,max:f32},LeakyRelu {negative_slope:f32},HardSigmoid {alpha:f32,beta:f32},
    Elu {alpha:f32},Celu {alpha:f32},Selu}
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
            (PiecewiseActivation::LeakyRelu {..},false)=>"ruda_cann_leaky_relu",(PiecewiseActivation::LeakyRelu {..},true)=>"ruda_cann_leaky_relu_backward",
            (PiecewiseActivation::HardSigmoid {..},false)=>"ruda_cann_hard_sigmoid",(PiecewiseActivation::HardSigmoid {..},true)=>"ruda_cann_hard_sigmoid_backward",
            (PiecewiseActivation::Elu {..},false)=>"ruda_cann_elu",(PiecewiseActivation::Elu {..},true)=>"ruda_cann_elu_backward",
            (PiecewiseActivation::Celu {..},false)=>"ruda_cann_celu",(PiecewiseActivation::Celu {..},true)=>"ruda_cann_celu_backward",
            (PiecewiseActivation::Selu,false)=>"ruda_cann_selu",(PiecewiseActivation::Selu,true)=>"ruda_cann_selu_backward",
        }.into(),..Default::default()}};
    let lane=Variable::builtin(Builtin::AbsolutePosX,UIntKind::U64.into());let mut next=0;
    let mut op=|operation:Operation,ty:Type| {
        let out=Variable::new(VariableKind::LocalConst{id:next},ty);next+=1;
        kernel.body.instructions.push(Instruction::new(operation,out));out
    };
    let read=|id|Operator::Index(IndexOperator {list:Variable::new(VariableKind::GlobalInputArray(id),f()),index:lane,vector_size:0,unroll_factor:1});
    let bits=|value:f32|Operator::Reinterpret(UnaryOperator {input:Variable::constant(ConstantValue::UInt(value.to_bits() as u64),Type::new(UIntKind::U32.into()))});
    let x=op(read(0).into(),f());let grad=if backward {Some(op(read(1).into(),f()))} else {None};
    let zero=op(bits(0.).into(),f());let bool_type=Type::scalar(ElemType::Bool);
    let result=match activation {
        PiecewiseActivation::Relu=>{
            let mask=op(Comparison::LowerEqual(BinaryOperator {lhs:x,rhs:zero}).into(),bool_type);
            op(Operator::Select(Select {cond:mask,then:zero,or_else:grad.unwrap_or(x)}).into(),f())
        },
        PiecewiseActivation::LeakyRelu {negative_slope}=>{
            let slope=op(bits(negative_slope).into(),f());
            let mask=op(Comparison::Lower(BinaryOperator {lhs:x,rhs:zero}).into(),bool_type);
            if let Some(grad)=grad {
                // Retain RUDA MaskWhere -> MulScalar -> shared-parent accumulation,
                // including non-finite arithmetic in the masked scalar branch.
                let direct=op(Operator::Select(Select {cond:mask,then:zero,or_else:grad}).into(),f());
                let negative=op(Operator::Select(Select {cond:mask,then:grad,or_else:zero}).into(),f());
                let scaled=op(Arithmetic::Mul(BinaryOperator {lhs:negative,rhs:slope}).into(),f());
                op(Arithmetic::Add(BinaryOperator {lhs:scaled,rhs:direct}).into(),f())
            } else {
                let scaled=op(Arithmetic::Mul(BinaryOperator {lhs:x,rhs:slope}).into(),f());
                op(Operator::Select(Select {cond:mask,then:scaled,or_else:x}).into(),f())
            }
        },
        PiecewiseActivation::Elu {..}|PiecewiseActivation::Celu {..}|PiecewiseActivation::Selu=>{
            const SELU_ALPHA:f64=1.6732632423543772848170429916717;
            const SELU_GAMMA:f64=1.0507009873554804934193349852946;
            let selu=matches!(activation,PiecewiseActivation::Selu);let celu=matches!(activation,PiecewiseActivation::Celu {..});
            let coefficient=match activation {PiecewiseActivation::Elu {alpha}|PiecewiseActivation::Celu {alpha}=>alpha,
                PiecewiseActivation::Selu=>(SELU_ALPHA*SELU_GAMMA) as f32,_=>unreachable!()};
            let coefficient_node=op(bits(coefficient).into(),f());
            let mask=op((if selu {Comparison::GreaterEqual(BinaryOperator {lhs:x,rhs:zero})}
                else {Comparison::LowerEqual(BinaryOperator {lhs:x,rhs:zero})}).into(),bool_type);
            let exponent=if celu {op(Arithmetic::Div(BinaryOperator {lhs:x,rhs:coefficient_node}).into(),f())} else {x};
            let exponential=op(Arithmetic::Exp(UnaryOperator {input:exponent}).into(),f());
            if let Some(grad)=grad {
                let negative_grad=op(Operator::Select(Select {cond:mask,then:if selu {zero} else {grad},or_else:if selu {grad} else {zero}}).into(),f());
                let direct_grad=op(Operator::Select(Select {cond:mask,then:if selu {grad} else {zero},or_else:if selu {zero} else {grad}}).into(),f());
                let scaled_grad=op(Arithmetic::Mul(BinaryOperator {lhs:negative_grad,rhs:coefficient_node}).into(),f());
                let mut negative_grad=op(Arithmetic::Mul(BinaryOperator {lhs:scaled_grad,rhs:exponential}).into(),f());
                if celu {
                    // RUDA DivScalar backward specializes the FP32 reciprocal, then multiplies.
                    let reciprocal=op(bits(1f32/coefficient).into(),f());
                    negative_grad=op(Arithmetic::Mul(BinaryOperator {lhs:negative_grad,rhs:reciprocal}).into(),f());
                }
                let direct_grad=if selu {let gamma=op(bits(SELU_GAMMA as f32).into(),f());
                    op(Arithmetic::Mul(BinaryOperator {lhs:direct_grad,rhs:gamma}).into(),f())} else {direct_grad};
                op(Arithmetic::Add(BinaryOperator {lhs:negative_grad,rhs:direct_grad}).into(),f())
            } else {
                let one=op(bits(1.).into(),f());let shifted=op(Arithmetic::Sub(BinaryOperator {lhs:exponential,rhs:one}).into(),f());
                let negative=op(Arithmetic::Mul(BinaryOperator {lhs:shifted,rhs:coefficient_node}).into(),f());
                let positive=if selu {let gamma=op(bits(SELU_GAMMA as f32).into(),f());
                    op(Arithmetic::Mul(BinaryOperator {lhs:x,rhs:gamma}).into(),f())} else {x};
                op(Operator::Select(Select {cond:mask,then:if selu {positive} else {negative},or_else:if selu {negative} else {positive}}).into(),f())
            }
        },
        PiecewiseActivation::Clamp {..}|PiecewiseActivation::HardSigmoid {..}=>{
            let (input,min,max,alpha)=match activation {
                PiecewiseActivation::Clamp {min,max}=>(x,min,max,None),
                PiecewiseActivation::HardSigmoid {alpha,beta}=>{
                    let alpha=op(bits(alpha).into(),f());let beta=op(bits(beta).into(),f());
                    let scaled=op(Arithmetic::Mul(BinaryOperator {lhs:x,rhs:alpha}).into(),f());
                    let affine=op(Arithmetic::Add(BinaryOperator {lhs:scaled,rhs:beta}).into(),f());
                    (affine,0.,1.,Some(alpha))
                },_=>unreachable!(),
            };
            let max=op(bits(max).into(),f());let min=op(bits(min).into(),f());
            let hi=op(Comparison::Greater(BinaryOperator {lhs:input,rhs:max}).into(),bool_type);
            let upper=op(Operator::Select(Select {cond:hi,then:max,or_else:input}).into(),f());
            let lo=op(Comparison::Lower(BinaryOperator {lhs:upper,rhs:min}).into(),bool_type);
            if let Some(grad)=grad {
                let upper_grad=op(Operator::Select(Select {cond:hi,then:zero,or_else:grad}).into(),f());
                let lower_grad=op(Operator::Select(Select {cond:lo,then:zero,or_else:upper_grad}).into(),f());
                if let Some(alpha)=alpha {op(Arithmetic::Mul(BinaryOperator {lhs:lower_grad,rhs:alpha}).into(),f())} else {lower_grad}
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
                    _=>unreachable!(),
                };assert_eq!(actual[0][lane].to_bits(),expected.to_bits());}
                let compiled=AscendCompiler.compile(kernel,&AscendOptions {target:Some(AscendTarget::Ascend950DT),elements:count as u64,..Default::default()},
                    ExecutionMode::Checked,UIntKind::U64.into()).unwrap();
                assert_eq!(compiled.bindings().len(),if backward {3} else {2});assert!(compiled.source().contains("AscendC::Compare("));
            }}
        }
        assert!(definition(PiecewiseActivation::Relu,u32::MAX as u64+1,false).is_err());
    }
    #[test]
    fn leaky_relu_retains_strict_negative_mask_and_shared_parent_backward() {
        let values=[f32::NEG_INFINITY,-2.,-0.,0.,2.,f32::INFINITY,f32::from_bits(0x7fc12345)];
        for slope in [0.1f32,0.,-0.,-0.5,f32::INFINITY,f32::NEG_INFINITY,f32::NAN] {
            for count in [0usize,1,7,65,257] {for backward in [false,true] {
                let x:Vec<f32>=(0..count).map(|i|values[i%values.len()]).collect();
                let grad:Vec<f32>=(0..count).map(|i|if i%3==0 {-0.} else {(i%7) as f32-3.}).collect();
                let kernel=definition(PiecewiseActivation::LeakyRelu {negative_slope:slope},count as u64,backward).unwrap();
                let actual=evaluate(kernel.clone(),count,[&x,&grad,&[]]);
                for lane in 0..count {let negative=x[lane]<0.;let expected=if backward {
                    let direct=if negative {0.} else {grad[lane]};let scaled=(if negative {grad[lane]} else {0.})*slope;
                    scaled+direct
                } else if negative {x[lane]*slope} else {x[lane]};
                    let result=actual[0][lane];assert!(result.to_bits()==expected.to_bits() || result.is_nan() && expected.is_nan());
                }
                let compiled=AscendCompiler.compile(kernel,&AscendOptions {target:Some(AscendTarget::Ascend950DT),elements:count as u64,..Default::default()},
                    ExecutionMode::Checked,UIntKind::U64.into()).unwrap();
                assert!(compiled.source().contains("AscendC::CMPMODE::LT"));assert!(compiled.source().contains("AscendC::Mul("));
                assert_eq!(compiled.source().contains("AscendC::Add("),backward);
            }}
        }
    }
    #[test]
    fn hard_sigmoid_affine_order_and_clipped_gradients_match_default_graph() {
        let values=[f32::NEG_INFINITY,-6.,-2.,-0.,0.,2.,6.,f32::INFINITY,f32::NAN];
        for (alpha,beta) in [(0.25f32,0.5f32),(-0.25,0.5),(0.,0.5),(f32::INFINITY,0.),(0.5,f32::NAN)] {
            for count in [0usize,1,7,65,257] {for backward in [false,true] {
                let x:Vec<f32>=(0..count).map(|i|values[i%values.len()]).collect();
                let grad:Vec<f32>=(0..count).map(|i|if i%3==0 {-0.} else {(i%7) as f32-3.}).collect();
                let kernel=definition(PiecewiseActivation::HardSigmoid {alpha,beta},count as u64,backward).unwrap();
                let actual=evaluate(kernel.clone(),count,[&x,&grad,&[]]);
                for lane in 0..count {let affine=x[lane]*alpha+beta;let hi=affine>1.;let upper=if hi {1.} else {affine};let lo=upper<0.;
                    let expected=if backward {(if hi || lo {0.} else {grad[lane]})*alpha} else if lo {0.} else {upper};
                    let result=actual[0][lane];assert!(result.to_bits()==expected.to_bits() || result.is_nan() && expected.is_nan());
                }
                let compiled=AscendCompiler.compile(kernel,&AscendOptions {target:Some(AscendTarget::Ascend950DT),elements:count as u64,..Default::default()},
                    ExecutionMode::Checked,UIntKind::U64.into()).unwrap();
                assert!(compiled.source().find("AscendC::Mul(").unwrap()<compiled.source().find("AscendC::Add(").unwrap());
                assert!(compiled.source().contains("AscendC::CMPMODE::GT"));assert!(compiled.source().contains("AscendC::CMPMODE::LT"));
            }}
        }
    }
    #[test]
    fn exponential_piecewise_maps_keep_source_masks_coefficients_and_backward_order() {
        const ALPHA:f64=1.6732632423543772848170429916717;const GAMMA:f64=1.0507009873554804934193349852946;
        let values=[f32::NEG_INFINITY,-2.,-0.,0.,2.,90.,f32::INFINITY,f32::from_bits(0x7fc12345)];
        let mut activations=vec![PiecewiseActivation::Selu];
        for alpha in [0.1f32,1.,-0.5,0.,f32::INFINITY,f32::NAN] {
            activations.push(PiecewiseActivation::Elu {alpha});activations.push(PiecewiseActivation::Celu {alpha});
        }
        for activation in activations {for count in [0usize,1,7,65,257] {for backward in [false,true] {
            let x:Vec<f32>=(0..count).map(|i|values[i%values.len()]).collect();
            let grad:Vec<f32>=(0..count).map(|i|if i%3==0 {-0.} else {(i%7) as f32-3.}).collect();
            let kernel=definition(activation,count as u64,backward).unwrap();let actual=evaluate(kernel.clone(),count,[&x,&grad,&[]]);
            for lane in 0..count {
                let (coefficient,exponential,negative,positive_scale)=match activation {
                    PiecewiseActivation::Elu {alpha}=>(alpha,x[lane].exp(),x[lane]<=0.,None),
                    PiecewiseActivation::Celu {alpha}=>(alpha,(x[lane]/alpha).exp(),x[lane]<=0.,None),
                    PiecewiseActivation::Selu=>((ALPHA*GAMMA) as f32,x[lane].exp(),!(x[lane]>=0.),Some(GAMMA as f32)),_=>unreachable!(),
                };
                let expected=if backward {
                    let direct=if negative {0.} else {grad[lane]};let direct=if let Some(gamma)=positive_scale {direct*gamma} else {direct};
                    let masked=if negative {grad[lane]} else {0.};let mut scaled=(masked*coefficient)*exponential;
                    if matches!(activation,PiecewiseActivation::Celu {..}) {scaled*=1f32/coefficient;}scaled+direct
                } else if negative {(exponential-1.)*coefficient} else if let Some(gamma)=positive_scale {x[lane]*gamma} else {x[lane]};
                let result=actual[0][lane];assert!(result.to_bits()==expected.to_bits() || result.is_nan() && expected.is_nan());
            }
            let compiled=AscendCompiler.compile(kernel,&AscendOptions {target:Some(AscendTarget::Ascend950DT),elements:count as u64,..Default::default()},
                ExecutionMode::Checked,UIntKind::U64.into()).unwrap();
            assert!(compiled.source().contains("AscendC::Exp("));assert_eq!(compiled.source().contains("AscendC::Div("),matches!(activation,PiecewiseActivation::Celu {..}));
            assert!(compiled.source().contains(if matches!(activation,PiecewiseActivation::Selu) {"CMPMODE::GE"} else {"CMPMODE::LE"}));
        }}}
    }
}

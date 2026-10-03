//! Table-driven rotary position encoding in common RUDA IR; no frequency policy.
use super::{Result,invalid};
use ruda_core::{ir::*,kernel::{KernelArg,KernelDefinition,KernelOptions,Visibility},launch::RudaDim};

#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum RotaryLayout {Interleaved,SplitHalf}
struct Builder {kernel:KernelDefinition,next:u32}
impl Builder {
    fn local(&mut self,ty:Type)->Variable {let id=self.next;self.next+=1;Variable::new(VariableKind::LocalConst{id},ty)}
    fn op(&mut self,operation:impl Into<Operation>,ty:Type)->Variable {
        let out=self.local(ty);self.kernel.body.instructions.push(Instruction::new(operation,out));out
    }
    fn arithmetic(&mut self,kind:fn(BinaryOperator)->Arithmetic,a:Variable,b:Variable,ty:Type)->Variable {
        self.op(kind(BinaryOperator{lhs:a,rhs:b}),ty)
    }
    fn read(&mut self,id:u32,index:Variable)->Variable {
        self.op(Operator::Index(IndexOperator{list:Variable::new(VariableKind::GlobalInputArray(id),fp32()),
            index,vector_size:0,unroll_factor:1}),fp32())
    }
}
fn fp32()->Type {Type::new(FloatKind::F32.into())}
fn uint()->Type {Type::new(UIntKind::U64.into())}
fn integer(value:u64)->Variable {Variable::constant(ConstantValue::UInt(value),uint())}
fn float(value:f64)->Variable {Variable::constant(ConstantValue::Float(value),fp32())}

/// Full last-axis rotation with positive even width and explicit tables containing one value per pair.
/// Bindings: [X, X aliased readonly for paired indexing, cos[N/2], sin[N/2], Y[N]].
/// Backward consumes dY instead of X and applies the transpose Jacobian; tables are fixed.
pub fn definition(width:u32,elements:u64,layout:RotaryLayout,backward:bool)->Result<KernelDefinition> {
    if width==0 || width%2!=0 || elements>u32::MAX as u64 || elements%u64::from(width)!=0 {
        return Err(invalid("rotary requires positive even width, complete rows and elements within u32"));
    }
    let half=u64::from(width)/2;
    let mut builder=Builder {kernel:KernelDefinition {buffers:vec![],tensor_maps:vec![],scalars:vec![],
        ruda_dim:RudaDim::new_1d(64),body:Scope::root(false),options:KernelOptions {
            kernel_name:format!("ruda_cann_rotary_{}_{}",if layout==RotaryLayout::Interleaved {"interleaved"} else {"split"},
                if backward {"backward"} else {"forward"}),..Default::default()}},next:0};
    for (id,size) in [elements,elements,elements/2,elements/2,elements].into_iter().enumerate() {
        builder.kernel.buffers.push(KernelArg {id:id as u32,visibility:if id==4 {Visibility::ReadWrite} else {Visibility::Read},
            ty:fp32(),size:Some(size as usize),has_extended_meta:false});
    }
    let lane=Variable::builtin(Builtin::AbsolutePosX,UIntKind::U64.into());
    let row=builder.arithmetic(Arithmetic::Div,lane,integer(width as u64),uint());
    let col=builder.arithmetic(Arithmetic::Modulo,lane,integer(width as u64),uint());
    let (paired_col,pair,side)=match layout {
        RotaryLayout::Interleaved=>{
            let pair=builder.arithmetic(Arithmetic::Div,col,integer(2),uint());
            let base=builder.arithmetic(Arithmetic::Mul,pair,integer(2),uint());
            let plus_one=builder.arithmetic(Arithmetic::Add,col,integer(1),uint());
            let other_side=builder.arithmetic(Arithmetic::Modulo,plus_one,integer(2),uint());
            let paired=builder.arithmetic(Arithmetic::Add,base,other_side,uint());
            let side=builder.arithmetic(Arithmetic::Modulo,col,integer(2),uint());
            (paired,pair,side)
        },
        RotaryLayout::SplitHalf=>{
            let shifted=builder.arithmetic(Arithmetic::Add,col,integer(half),uint());
            let paired=builder.arithmetic(Arithmetic::Modulo,shifted,integer(width as u64),uint());
            let pair=builder.arithmetic(Arithmetic::Modulo,col,integer(half),uint());
            let side=builder.arithmetic(Arithmetic::Div,col,integer(half),uint());
            (paired,pair,side)
        },
    };
    let row_base=builder.arithmetic(Arithmetic::Mul,row,integer(width as u64),uint());
    let paired=builder.arithmetic(Arithmetic::Add,row_base,paired_col,uint());
    let table_base=builder.arithmetic(Arithmetic::Mul,row,integer(half),uint());
    let table=builder.arithmetic(Arithmetic::Add,table_base,pair,uint());
    let side=builder.op(Operator::Cast(UnaryOperator{input:side}),fp32());
    let side=builder.arithmetic(Arithmetic::Mul,side,float(2.),fp32());
    let sign=builder.arithmetic(Arithmetic::Sub,side,float(1.),fp32());
    let x=builder.read(0,lane);let paired=builder.read(1,paired);
    let cos=builder.read(2,table);let sin=builder.read(3,table);
    let direct=builder.arithmetic(Arithmetic::Mul,x,cos,fp32());
    let rotated=builder.arithmetic(Arithmetic::Mul,paired,sin,fp32());
    let rotated=builder.arithmetic(Arithmetic::Mul,rotated,sign,fp32());
    let output=builder.arithmetic(if backward {Arithmetic::Sub} else {Arithmetic::Add},direct,rotated,fp32());
    builder.kernel.body.instructions.push(Instruction::new(Operator::IndexAssign(IndexAssignOperator {
        index:lane,value:output,vector_size:0,unroll_factor:1}),Variable::new(VariableKind::GlobalOutputArray(4),fp32())));
    Ok(builder.kernel)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ascend::{AscendCompiler,AscendOptions,AscendTarget,lower::{self,Node,Unary,Binary}};
    use ruda_core::{compiler::Compiler,launch::ExecutionMode};
    fn evaluate(kernel:KernelDefinition,elements:usize,inputs:[&[f32];4])->Vec<f32> {
        let p=lower::lower(kernel,elements as u64).unwrap();let mut values:Vec<Vec<f32>>=vec![];
        for node in &p.nodes {
            let value=match *node {
                Node::Input(id)=>(0..elements as u64).map(|lane|inputs[id][p.load_indices[&id].eval(lane) as usize]).collect(),
                Node::Constant(bits)=>vec![f32::from_bits(bits);elements],
                Node::IndexFloat(id)=>(0..elements as u64).map(|lane|p.index_values[id].eval(lane) as f32).collect(),
                Node::Binary(op,a,b)=>values[a].iter().zip(&values[b]).map(|(&a,&b)|match op {
                    Binary::Add=>a+b,Binary::Sub=>a-b,Binary::Mul=>a*b,Binary::Div=>a/b}).collect(),
                Node::Unary(op,a)=>values[a].iter().map(|&a|match op {Unary::Neg=>-a,Unary::Abs=>a.abs(),
                    Unary::Exp=>a.exp(),Unary::Log=>a.ln(),Unary::Sqrt=>a.sqrt(),Unary::Rsqrt=>a.sqrt().recip(),Unary::Recip=>a.recip()}).collect(),
            };
            values.push(value);
        }
        values[p.stores[0].1].clone()
    }
    #[test]
    fn rotary_ir_matches_pairwise_forward_and_transpose_jacobian() {
        for width in [2,6,64,96] {for rows in [0,1,3] {for layout in [RotaryLayout::Interleaved,RotaryLayout::SplitHalf] {
            let n=rows*width;let half=width/2;
            let x:Vec<f32>=(0..n).map(|i|(i%19) as f32*0.25-2.).collect();
            let dy:Vec<f32>=(0..n).map(|i|(i%7) as f32*0.125-0.25).collect();
            let cos:Vec<f32>=(0..n/2).map(|i|0.7+(i%3) as f32*0.125).collect();
            let sin:Vec<f32>=(0..n/2).map(|i|-0.3+(i%5) as f32*0.0625).collect();
            for backward in [false,true] {
                let input=if backward {&dy} else {&x};
                let output=evaluate(definition(width as u32,n as u64,layout,backward).unwrap(),n,[input,input,&cos,&sin]);
                for row in 0..rows {for pair in 0..half {
                    let (a,b)=if layout==RotaryLayout::Interleaved {(row*width+pair*2,row*width+pair*2+1)}
                        else {(row*width+pair,row*width+half+pair)};
                    let c=cos[row*half+pair] as f64;let s=sin[row*half+pair] as f64;
                    let sign=if backward {-1.} else {1.};
                    assert!((output[a] as f64-(input[a] as f64*c-sign*input[b] as f64*s)).abs()<2e-6);
                    assert!((output[b] as f64-(input[b] as f64*c+sign*input[a] as f64*s)).abs()<2e-6);
                }}
            }
        }}}
    }
    #[test]
    fn rotary_compiles_common_ir_and_preserves_table_lengths() {
        for layout in [RotaryLayout::Interleaved,RotaryLayout::SplitHalf] {for backward in [false,true] {
            let ir=definition(6,18,layout,backward).unwrap();
            let output=AscendCompiler.compile(ir.clone(),&AscendOptions {target:Some(AscendTarget::Ascend950DT),
                elements:18,..Default::default()},ExecutionMode::Checked,UIntKind::U64.into()).unwrap();
            assert_eq!(output.bindings().iter().map(|b|b.bytes).collect::<Vec<_>>(),[72,72,36,36,72]);
            assert!(output.source().contains("static_cast<float>"));assert!(output.source().contains("S_V"));
            assert!(!output.source().contains("aclnn"));
            #[cfg(feature="ptx")] {
                use crate::ptx::*;
                assert!(PtxCompiler.compile(ir,&PtxCompilationOptions {target:Some(PtxTarget {version:(8,0),sm:75})},
                    ExecutionMode::Checked,UIntKind::U64.into()).unwrap().source.contains(".entry"));
            }
        }}
        for (width,n) in [(0,0),(3,9),(6,7),(2,u32::MAX as u64+1)] {assert!(definition(width,n,RotaryLayout::SplitHalf,false).is_err());}
    }
}

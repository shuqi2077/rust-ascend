//! Table-driven rotary position encoding in common RUDA IR; no frequency policy.
use super::{Result,invalid};
use ruda_core::{ir::*,kernel::{KernelArg,KernelDefinition,KernelOptions,Visibility},launch::RudaDim};

#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum RotaryLayout {Interleaved,SplitHalf}
/// Rotate only the first P values; fixed cos/sin tables broadcast singleton leading axes.
#[derive(Clone,Debug,PartialEq,Eq)]
pub struct PrefixRotarySpec {pub input:Vec<u32>,pub table:Vec<u32>,pub rotary_width:u32,pub layout:RotaryLayout}
impl PrefixRotarySpec {
    pub fn elements(&self)->Result<(u64,u64)> {
        if !(1..=8).contains(&self.input.len()) || self.table.len()!=self.input.len() {
            return Err(invalid("rotary prefix requires equal input/table ranks within 1..8"));
        }
        let rank=self.input.len();let p=self.rotary_width;
        if p==0 || p%2!=0 || p>self.input[rank-1] || self.table[rank-1]!=p/2
            || self.input[..rank-1].iter().zip(&self.table[..rank-1]).any(|(&input,&table)|table!=1 && table!=input) {
            return Err(invalid("rotary prefix requires positive even P<=D, table last axis P/2 and singleton or matching leading axes"));
        }
        let count=|shape:&[u32]|shape.iter().try_fold(1u64,|n,&d|n.checked_mul(d as u64)).filter(|&n|n<=u32::MAX as u64)
            .ok_or_else(||invalid("rotary prefix complete domain exceeds u32"));
        Ok((count(&self.input)?,count(&self.table)?))
    }
}
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

/// Bindings [X, readonly paired X alias, cos, sin, Y]; tail values keep their exact bits.
/// Both rotation and its transpose Jacobian use caller tables without expansion or frequency policy.
pub fn prefix_definition(spec:&PrefixRotarySpec,backward:bool)->Result<KernelDefinition> {
    let (elements,tables)=spec.elements()?;let rank=spec.input.len();let width=spec.input[rank-1] as u64;
    let p=spec.rotary_width as u64;let half=p/2;
    let mut b=Builder {kernel:KernelDefinition {buffers:[elements,elements,tables,tables,elements].into_iter().enumerate().map(|(id,size)|KernelArg {
        id:id as u32,visibility:if id==4 {Visibility::ReadWrite} else {Visibility::Read},ty:fp32(),size:Some(size as usize),has_extended_meta:false}).collect(),
        tensor_maps:vec![],scalars:vec![],ruda_dim:RudaDim::new_1d(64),body:Scope::root(false),options:KernelOptions {
            kernel_name:format!("ruda_cann_rotary_prefix_{}_{}",if spec.layout==RotaryLayout::Interleaved {"interleaved"} else {"split"},if backward {"backward"} else {"forward"}),
            ..Default::default()}},next:0};
    let lane=Variable::builtin(Builtin::AbsolutePosX,UIntKind::U64.into());
    let row=b.arithmetic(Arithmetic::Div,lane,integer(width),uint());
    let col=b.arithmetic(Arithmetic::Modulo,lane,integer(width),uint());
    // Tail lanes use valid prefix/table addresses too: Select does not suppress operand loads.
    let prefix_col=b.arithmetic(Arithmetic::Modulo,col,integer(p),uint());
    let (paired_col,pair,side)=match spec.layout {
        RotaryLayout::Interleaved=>{
            let pair=b.arithmetic(Arithmetic::Div,prefix_col,integer(2),uint());
            let base=b.arithmetic(Arithmetic::Mul,pair,integer(2),uint());
            let shifted=b.arithmetic(Arithmetic::Add,prefix_col,integer(1),uint());
            let other=b.arithmetic(Arithmetic::Modulo,shifted,integer(2),uint());
            let paired=b.arithmetic(Arithmetic::Add,base,other,uint());
            let side=b.arithmetic(Arithmetic::Modulo,prefix_col,integer(2),uint());(paired,pair,side)
        },
        RotaryLayout::SplitHalf=>{
            let shifted=b.arithmetic(Arithmetic::Add,prefix_col,integer(half),uint());
            let paired=b.arithmetic(Arithmetic::Modulo,shifted,integer(p),uint());
            let pair=b.arithmetic(Arithmetic::Modulo,prefix_col,integer(half),uint());
            let side=b.arithmetic(Arithmetic::Div,prefix_col,integer(half),uint());(paired,pair,side)
        },
    };
    let base=b.arithmetic(Arithmetic::Mul,row,integer(width),uint());
    let paired=b.arithmetic(Arithmetic::Add,base,paired_col,uint());
    let mut table=pair;
    if elements!=0 {
        let mut input_stride=width;let mut table_stride=half;
        for axis in (0..rank-1).rev() {
            if spec.table[axis]!=1 {
                let coordinate=b.arithmetic(Arithmetic::Div,lane,integer(input_stride),uint());
                let coordinate=b.arithmetic(Arithmetic::Modulo,coordinate,integer(spec.input[axis] as u64),uint());
                let offset=b.arithmetic(Arithmetic::Mul,coordinate,integer(table_stride),uint());
                table=b.arithmetic(Arithmetic::Add,table,offset,uint());
            }
            input_stride=input_stride.checked_mul(spec.input[axis] as u64).ok_or_else(||invalid("rotary input stride overflow"))?;
            table_stride=table_stride.checked_mul(spec.table[axis] as u64).ok_or_else(||invalid("rotary table stride overflow"))?;
        }
    }
    let x=b.read(0,lane);let other=b.read(1,paired);let cos=b.read(2,table);let sin=b.read(3,table);
    let side=b.op(Operator::Cast(UnaryOperator {input:side}),fp32());
    let side=b.arithmetic(Arithmetic::Mul,side,float(2.),fp32());let sign=b.arithmetic(Arithmetic::Sub,side,float(1.),fp32());
    let direct=b.arithmetic(Arithmetic::Mul,x,cos,fp32());let rotation=b.arithmetic(Arithmetic::Mul,other,sin,fp32());
    let rotation=b.arithmetic(Arithmetic::Mul,rotation,sign,fp32());
    let rotated=b.arithmetic(if backward {Arithmetic::Sub} else {Arithmetic::Add},direct,rotation,fp32());
    let cond=b.op(Comparison::Lower(BinaryOperator {lhs:col,rhs:integer(p)}),Type::scalar(ElemType::Bool));
    let output=b.op(Operator::Select(Select {cond,then:rotated,or_else:x}),fp32());
    b.kernel.body.instructions.push(Instruction::new(Operator::IndexAssign(IndexAssignOperator {index:lane,value:output,vector_size:0,unroll_factor:1}),
        Variable::new(VariableKind::GlobalOutputArray(4),fp32())));Ok(b.kernel)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ascend::{AscendCompiler,AscendOptions,AscendTarget,lower::{self,Node,Unary,Binary}};
    use ruda_core::{compiler::Compiler,launch::ExecutionMode};
    fn evaluate(kernel:KernelDefinition,elements:usize,inputs:[&[f32];4])->Vec<f32> {
        let p=lower::lower(kernel,elements as u64).unwrap();let mut values:Vec<Vec<f32>>=vec![];
        if elements==0 {return vec![];}
        for node in &p.nodes {
            let value=match *node {
                Node::Input(id)=>(0..elements as u64).map(|lane|inputs[id][p.load_indices[&id].eval(lane) as usize]).collect(),
                Node::UniformInput(id,offset)=>vec![inputs[id][offset as usize];elements],
                Node::Constant(bits)=>vec![f32::from_bits(bits);elements],
                Node::IndexFloat(id)=>(0..elements as u64).map(|lane|p.index_values[id].eval(lane) as f32).collect(),
                Node::IndexSelect(id,a,b)=>(0..elements).map(|lane|if p.predicates[id].eval(lane as u64) {values[a][lane]} else {values[b][lane]}).collect(),
                Node::Binary(op,a,b)=>values[a].iter().zip(&values[b]).map(|(&a,&b)|match op {
                    Binary::Add=>a+b,Binary::Sub=>a-b,Binary::Mul=>a*b,Binary::Div=>a/b,Binary::Max=>a.max(b)}).collect(),
                Node::Unary(op,a)=>values[a].iter().map(|&a|match op {Unary::Neg=>-a,Unary::Abs=>a.abs(),
                    Unary::Exp=>a.exp(),Unary::Log=>a.ln(),Unary::Sqrt=>a.sqrt(),Unary::Rsqrt=>a.sqrt().recip(),Unary::Recip=>a.recip(),Unary::Erf=>super::super::tests::erf_reference(a),Unary::Tanh=>a.tanh(),Unary::Sin=>a.sin(),Unary::Cos=>a.cos()}).collect(),
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
    #[test]
    fn prefix_rotation_broadcasts_table_coordinates_and_keeps_tail_bits() {
        for (input,table,p) in [
            (vec![11],vec![3],6),(vec![2,5,11],vec![1,5,3],6),
            (vec![2,3,5,11],vec![1,1,5,3],6),(vec![2,3,5,11],vec![2,1,5,3],6),
            (vec![2,3,5,11],vec![1,3,1,3],6),(vec![2,3,5,11],vec![2,3,5,3],6),
            (vec![2,1,1,1,1,3,5,11],vec![1,1,1,1,1,1,5,3],6),
            (vec![0,3,5,11],vec![1,1,5,3],6),(vec![2,3,0,11],vec![1,1,0,3],6),
            (vec![2,3,5,6],vec![1,1,5,3],6),
        ] {for layout in [RotaryLayout::Interleaved,RotaryLayout::SplitHalf] {
            let spec=PrefixRotarySpec {input:input.clone(),table:table.clone(),rotary_width:p,layout};
            let (n,t)=spec.elements().unwrap();let n=n as usize;let width=*input.last().unwrap() as usize;let half=p as usize/2;
            let cos:Vec<f32>=(0..t).map(|i|0.7+(i%3) as f32/8.).collect();
            let sin:Vec<f32>=(0..t).map(|i|-0.3+(i%5) as f32/16.).collect();
            for backward in [false,true] {
                let x:Vec<f32>=(0..n).map(|i|if i%width>=p as usize {
                    f32::from_bits([0x80000000,0x7fc01234,0x7f800000,0xff800000][i%4])
                } else {(i%17) as f32/8.-0.75}).collect();
                let ir=prefix_definition(&spec,backward).unwrap();let y=evaluate(ir.clone(),n,[&x,&x,&cos,&sin]);
                let compiled=AscendCompiler.compile(ir,&AscendOptions {target:Some(AscendTarget::Ascend950DT),elements:n as u64,..Default::default()},
                    ExecutionMode::Checked,UIntKind::U64.into()).unwrap();
                assert_eq!(compiled.bindings().iter().map(|b|b.bytes).collect::<Vec<_>>(),[n as u64*4,n as u64*4,t*4,t*4,n as u64*4]);
                for row in 0..n/width {
                    let mut remaining=row;let mut coordinates=vec![0;input.len()-1];
                    for axis in (0..coordinates.len()).rev() {coordinates[axis]=remaining%input[axis] as usize;remaining/=input[axis] as usize;}
                    let table_row=table[..table.len()-1].iter().zip(coordinates).fold(0usize,|row,(&dim,c)|row*dim as usize+if dim==1 {0} else {c});
                    for pair in 0..half {
                        let (a,b)=if layout==RotaryLayout::Interleaved {(row*width+pair*2,row*width+pair*2+1)} else {(row*width+pair,row*width+half+pair)};
                        let c=cos[table_row*half+pair] as f64;let s=sin[table_row*half+pair] as f64;let sign=if backward {-1.} else {1.};
                        for (i,expected) in [(a,x[a] as f64*c-sign*x[b] as f64*s),(b,x[b] as f64*c+sign*x[a] as f64*s)] {
                            assert!((y[i] as f64-expected).abs()<3e-6+3e-6*expected.abs());
                        }
                    }
                    for column in p as usize..width {assert_eq!(y[row*width+column].to_bits(),x[row*width+column].to_bits());}
                }
            }
        }}
    }
    #[test]
    fn prefix_rejects_invalid_broadcast_and_preserves_full_rotation() {
        let spec=PrefixRotarySpec {input:vec![2,3,5,11],table:vec![1,1,5,3],rotary_width:6,layout:RotaryLayout::Interleaved};
        for invalid in [PrefixRotarySpec {input:vec![],..spec.clone()},PrefixRotarySpec {table:vec![5,3],..spec.clone()},
            PrefixRotarySpec {table:vec![1,2,5,3],..spec.clone()},PrefixRotarySpec {table:vec![1,1,5,6],..spec.clone()},
            PrefixRotarySpec {rotary_width:0,..spec.clone()},PrefixRotarySpec {rotary_width:5,..spec.clone()},
            PrefixRotarySpec {rotary_width:12,..spec.clone()},PrefixRotarySpec {input:vec![2,3,u32::MAX,11],table:vec![1,1,u32::MAX,3],..spec.clone()}] {
            assert!(prefix_definition(&invalid,false).is_err());
        }
        for layout in [RotaryLayout::Interleaved,RotaryLayout::SplitHalf] {for backward in [false,true] {
            let x:Vec<f32>=(0..18).map(|i|i as f32/8.).collect();let cos=vec![0.75;9];let sin=vec![0.25;9];
            let spec=PrefixRotarySpec {input:vec![3,6],table:vec![3,3],rotary_width:6,layout};
            let expected=evaluate(definition(6,18,layout,backward).unwrap(),18,[&x,&x,&cos,&sin]);
            let actual=evaluate(prefix_definition(&spec,backward).unwrap(),18,[&x,&x,&cos,&sin]);
            assert_eq!(actual.iter().map(|x|x.to_bits()).collect::<Vec<_>>(),expected.iter().map(|x|x.to_bits()).collect::<Vec<_>>());
        }}
        let large=PrefixRotarySpec {input:vec![2,3,8192,128],table:vec![1,1,8192,32],rotary_width:64,..spec};
        let n=large.elements().unwrap().0;
        AscendCompiler.compile(prefix_definition(&large,false).unwrap(),&AscendOptions {target:Some(AscendTarget::Ascend950DT),elements:n,..Default::default()},
            ExecutionMode::Checked,UIntKind::U64.into()).unwrap();
    }
}

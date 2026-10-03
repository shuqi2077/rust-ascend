//! Common-IR tile operations for multi-pass wide-row normalization.
use super::{Result,invalid};
use ruda_core::{ir::*,kernel::{KernelArg,KernelDefinition,KernelOptions,Visibility},launch::RudaDim};

#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum WideStage {CopyTile,ExpTile,SoftmaxTile,LogSoftmaxTile,DotTile,SoftmaxBackwardTile,LogSoftmaxBackwardTile,SumBackwardTile,MeanBackwardTile,
    SquareTile,RmsNormTile,RmsDotTile,RmsBackwardTile,RmsWeightTile,
    CenteredSquareTile,LayerNormTile,LayerGTile,LayerGYTile,LayerBackwardTile,LayerWeightTile}
impl WideStage {
    pub fn partial(self)->bool {matches!(self,Self::SoftmaxTile|Self::LogSoftmaxTile|Self::SoftmaxBackwardTile|Self::LogSoftmaxBackwardTile|Self::SumBackwardTile|Self::MeanBackwardTile
        |Self::RmsNormTile|Self::RmsBackwardTile|Self::RmsWeightTile|Self::LayerNormTile|Self::LayerBackwardTile|Self::LayerWeightTile)}
    fn name(self)->&'static str {match self {Self::CopyTile=>"copy",Self::ExpTile=>"exp",Self::SoftmaxTile=>"softmax",
        Self::LogSoftmaxTile=>"log_softmax",Self::DotTile=>"dot",Self::SoftmaxBackwardTile=>"softmax_backward",Self::LogSoftmaxBackwardTile=>"log_softmax_backward",
        Self::SumBackwardTile=>"sum_backward",Self::MeanBackwardTile=>"mean_backward",Self::SquareTile=>"square",
        Self::RmsNormTile=>"rms_norm",Self::RmsDotTile=>"rms_dot",Self::RmsBackwardTile=>"rms_backward",Self::RmsWeightTile=>"rms_weight",
        Self::CenteredSquareTile=>"centered_square",Self::LayerNormTile=>"layer_norm",Self::LayerGTile=>"layer_g",Self::LayerGYTile=>"layer_gy",
        Self::LayerBackwardTile=>"layer_backward",Self::LayerWeightTile=>"layer_weight"}}
}
fn f()->Type {Type::new(FloatKind::F32.into())}
fn u()->Type {Type::new(UIntKind::U64.into())}
fn integer(value:u64)->Variable {Variable::constant(ConstantValue::UInt(value),u())}
struct Builder {kernel:KernelDefinition,next:u32}
impl Builder {
    fn new(name:String,sizes:&[u64])->Self {
        Self {kernel:KernelDefinition {buffers:sizes.iter().enumerate().map(|(id,&size)|KernelArg {id:id as u32,
            visibility:if id+1==sizes.len() {Visibility::ReadWrite} else {Visibility::Read},ty:f(),size:Some(size as usize),has_extended_meta:false}).collect(),
            tensor_maps:vec![],scalars:vec![],ruda_dim:RudaDim::new_1d(64),body:Scope::root(false),options:KernelOptions {kernel_name:name,..Default::default()}},next:0}
    }
    fn op(&mut self,operation:impl Into<Operation>,ty:Type)->Variable {
        let out=Variable::new(VariableKind::LocalConst{id:self.next},ty);self.next+=1;
        self.kernel.body.instructions.push(Instruction::new(operation,out));out
    }
    fn binary(&mut self,kind:fn(BinaryOperator)->Arithmetic,a:Variable,b:Variable,ty:Type)->Variable {
        self.op(kind(BinaryOperator {lhs:a,rhs:b}),ty)
    }
    fn unary(&mut self,kind:fn(UnaryOperator)->Arithmetic,input:Variable)->Variable {self.op(kind(UnaryOperator {input}),f())}
    fn read(&mut self,id:u32,index:Variable)->Variable {self.op(Operator::Index(IndexOperator {
        list:Variable::new(VariableKind::GlobalInputArray(id),f()),index,vector_size:0,unroll_factor:1}),f())}
    fn write(&mut self,index:Variable,value:Variable) {
        let id=self.kernel.buffers.len() as u32-1;
        self.kernel.body.instructions.push(Instruction::new(Operator::IndexAssign(IndexAssignOperator {
            index,value,vector_size:0,unroll_factor:1}),Variable::new(VariableKind::GlobalOutputArray(id),f())));
    }
}
pub fn definition(stage:WideStage,rows:u64,width:u32,start:u32,columns:u32)->Result<KernelDefinition> {
    let full=rows.checked_mul(width as u64).ok_or_else(||invalid("wide row domain overflow"))?;
    let tile=rows.checked_mul(columns as u64).ok_or_else(||invalid("wide tile domain overflow"))?;
    if width==0 || width%32!=0 || columns==0 || columns>4096 || columns%32!=0
        || start.checked_add(columns).is_none_or(|end|end>width) || full>u32::MAX as u64 {
        return Err(invalid("wide tiles require aligned positive width/columns, columns<=4096, an in-row range and full domain within u32"));
    }
    let sizes=match stage {
        WideStage::CopyTile|WideStage::SquareTile=>vec![full,tile],WideStage::ExpTile=>vec![full,rows,tile],
        WideStage::SoftmaxTile=>vec![tile,rows,full],WideStage::LogSoftmaxTile=>vec![full,rows,rows,full],
        WideStage::DotTile=>vec![full,full,tile],WideStage::SoftmaxBackwardTile|WideStage::LogSoftmaxBackwardTile=>vec![full,full,rows,full],
        WideStage::SumBackwardTile|WideStage::MeanBackwardTile=>vec![rows,full],
        WideStage::RmsNormTile=>vec![full,width as u64,rows,full],WideStage::RmsDotTile=>vec![full,full,width as u64,tile],
        WideStage::RmsBackwardTile=>vec![full,full,width as u64,rows,rows,full],WideStage::RmsWeightTile=>vec![full,full,rows,full],
        WideStage::CenteredSquareTile=>vec![full,rows,tile],WideStage::LayerNormTile=>vec![full,width as u64,width as u64,rows,rows,full],
        WideStage::LayerGTile=>vec![full,width as u64,tile],WideStage::LayerGYTile=>vec![full,full,width as u64,rows,rows,tile],
        WideStage::LayerBackwardTile=>vec![full,full,width as u64,rows,rows,rows,rows,full],WideStage::LayerWeightTile=>vec![full,full,rows,rows,full],
    };
    let mut builder=Builder::new(format!("ruda_cann_wide_{}",stage.name()),&sizes);
    let lane=Variable::builtin(Builtin::AbsolutePosX,UIntKind::U64.into());
    let row=builder.binary(Arithmetic::Div,lane,integer(columns as u64),u());
    let col=builder.binary(Arithmetic::Modulo,lane,integer(columns as u64),u());
    let base=builder.binary(Arithmetic::Mul,row,integer(width as u64),u());
    let index=builder.binary(Arithmetic::Add,base,col,u());
    let index=builder.binary(Arithmetic::Add,index,integer(start as u64),u());
    let shared=builder.binary(Arithmetic::Add,col,integer(start as u64),u());
    let output=match stage {
        WideStage::CopyTile=>builder.read(0,index),
        WideStage::SquareTile=>{let x=builder.read(0,index);builder.binary(Arithmetic::Mul,x,x,f())},
        WideStage::CenteredSquareTile=>{
            let x=builder.read(0,index);let mean=builder.read(1,row);let centered=builder.binary(Arithmetic::Sub,x,mean,f());
            builder.binary(Arithmetic::Mul,centered,centered,f())
        },
        WideStage::LayerNormTile=>{
            let x=builder.read(0,index);let weight=builder.read(1,shared);let bias=builder.read(2,shared);let mean=builder.read(3,row);let r=builder.read(4,row);
            let centered=builder.binary(Arithmetic::Sub,x,mean,f());let normalized=builder.binary(Arithmetic::Mul,centered,r,f());
            let affine=builder.binary(Arithmetic::Mul,normalized,weight,f());builder.binary(Arithmetic::Add,affine,bias,f())
        },
        WideStage::LayerGTile=>{
            let grad=builder.read(0,index);let weight=builder.read(1,shared);builder.binary(Arithmetic::Mul,grad,weight,f())
        },
        WideStage::LayerGYTile=>{
            let x=builder.read(0,index);let grad=builder.read(1,index);let weight=builder.read(2,shared);let mean=builder.read(3,row);let r=builder.read(4,row);
            let centered=builder.binary(Arithmetic::Sub,x,mean,f());let normalized=builder.binary(Arithmetic::Mul,centered,r,f());
            let g=builder.binary(Arithmetic::Mul,grad,weight,f());builder.binary(Arithmetic::Mul,g,normalized,f())
        },
        WideStage::LayerBackwardTile=>{
            let x=builder.read(0,index);let grad=builder.read(1,index);let weight=builder.read(2,shared);let mean=builder.read(3,row);let r=builder.read(4,row);
            let mean_g=builder.read(5,row);let mean_gy=builder.read(6,row);
            let centered=builder.binary(Arithmetic::Sub,x,mean,f());let normalized=builder.binary(Arithmetic::Mul,centered,r,f());
            let g=builder.binary(Arithmetic::Mul,grad,weight,f());let correction=builder.binary(Arithmetic::Mul,normalized,mean_gy,f());
            let centered=builder.binary(Arithmetic::Sub,g,mean_g,f());let result=builder.binary(Arithmetic::Sub,centered,correction,f());
            builder.binary(Arithmetic::Mul,result,r,f())
        },
        WideStage::LayerWeightTile=>{
            let x=builder.read(0,index);let grad=builder.read(1,index);let mean=builder.read(2,row);let r=builder.read(3,row);
            let centered=builder.binary(Arithmetic::Sub,x,mean,f());let normalized=builder.binary(Arithmetic::Mul,centered,r,f());
            builder.binary(Arithmetic::Mul,normalized,grad,f())
        },
        WideStage::RmsNormTile=>{
            let x=builder.read(0,index);let weight=builder.read(1,shared);let r=builder.read(2,row);
            let normalized=builder.binary(Arithmetic::Mul,x,r,f());builder.binary(Arithmetic::Mul,normalized,weight,f())
        },
        WideStage::RmsDotTile=>{
            let x=builder.read(0,index);let grad=builder.read(1,index);let weight=builder.read(2,shared);
            let g=builder.binary(Arithmetic::Mul,grad,weight,f());builder.binary(Arithmetic::Mul,g,x,f())
        },
        WideStage::RmsBackwardTile=>{
            let x=builder.read(0,index);let grad=builder.read(1,index);let weight=builder.read(2,shared);
            let r=builder.read(3,row);let mean=builder.read(4,row);
            let g=builder.binary(Arithmetic::Mul,grad,weight,f());let rr=builder.binary(Arithmetic::Mul,r,r,f());
            let correction=builder.binary(Arithmetic::Mul,mean,rr,f());let correction=builder.binary(Arithmetic::Mul,x,correction,f());
            let centered=builder.binary(Arithmetic::Sub,g,correction,f());builder.binary(Arithmetic::Mul,centered,r,f())
        },
        WideStage::RmsWeightTile=>{
            let x=builder.read(0,index);let grad=builder.read(1,index);let r=builder.read(2,row);
            let normalized=builder.binary(Arithmetic::Mul,x,r,f());builder.binary(Arithmetic::Mul,normalized,grad,f())
        },
        WideStage::ExpTile=>{
            let x=builder.read(0,index);let max=builder.read(1,row);
            let shifted=builder.binary(Arithmetic::Sub,x,max,f());builder.unary(Arithmetic::Exp,shifted)
        },
        WideStage::SoftmaxTile=>{
            let exp=builder.read(0,lane);let sum=builder.read(1,row);builder.binary(Arithmetic::Div,exp,sum,f())
        },
        WideStage::LogSoftmaxTile=>{
            let x=builder.read(0,index);let max=builder.read(1,row);let sum=builder.read(2,row);
            let shifted=builder.binary(Arithmetic::Sub,x,max,f());let log=builder.unary(Arithmetic::Log,sum);
            builder.binary(Arithmetic::Sub,shifted,log,f())
        },
        WideStage::DotTile=>{
            let y=builder.read(0,index);let grad=builder.read(1,index);builder.binary(Arithmetic::Mul,y,grad,f())
        },
        WideStage::SoftmaxBackwardTile=>{
            let y=builder.read(0,index);let grad=builder.read(1,index);let dot=builder.read(2,row);
            let centered=builder.binary(Arithmetic::Sub,grad,dot,f());builder.binary(Arithmetic::Mul,y,centered,f())
        },
        WideStage::LogSoftmaxBackwardTile=>{
            let y=builder.read(0,index);let grad=builder.read(1,index);let sum=builder.read(2,row);
            let exp=builder.unary(Arithmetic::Exp,y);let term=builder.binary(Arithmetic::Mul,exp,sum,f());
            builder.binary(Arithmetic::Sub,grad,term,f())
        },
        WideStage::SumBackwardTile|WideStage::MeanBackwardTile=>{
            let grad=builder.read(0,row);
            if stage==WideStage::MeanBackwardTile {builder.binary(Arithmetic::Div,grad,Variable::constant(ConstantValue::Float(width as f32 as f64),f()),f())}
            else {grad}
        },
    };
    builder.write(if stage.partial() {index} else {lane},output);
    Ok(builder.kernel)
}
pub fn merge_definition(elements:u64,max:bool)->Result<KernelDefinition> {
    if elements>u32::MAX as u64 {return Err(invalid("wide statistic domain exceeds u32"));}
    let mut builder=Builder::new(format!("ruda_cann_wide_merge_{}",if max {"max"} else {"sum"}),&[elements;3]);
    let lane=Variable::builtin(Builtin::AbsolutePosX,UIntKind::U64.into());
    let a=builder.read(0,lane);let b=builder.read(1,lane);
    let result=builder.binary(if max {Arithmetic::Max} else {Arithmetic::Add},a,b,f());builder.write(lane,result);
    Ok(builder.kernel)
}
pub fn zero_definition(elements:u64)->Result<KernelDefinition> {
    if elements>u32::MAX as u64 {return Err(invalid("wide output domain exceeds u32"));}
    let mut builder=Builder::new("ruda_cann_wide_zero".into(),&[elements]);
    builder.write(Variable::builtin(Builtin::AbsolutePosX,UIntKind::U64.into()),Variable::constant(ConstantValue::Float(0.),f()));
    Ok(builder.kernel)
}
/// Divide completed row sums once by the full width, not by individual tile widths.
pub fn mean_definition(rows:u64,width:u32)->Result<KernelDefinition> {
    if width==0 || rows>u32::MAX as u64 {return Err(invalid("mean requires positive width and a row count within u32"));}
    let mut builder=Builder::new("ruda_cann_wide_mean".into(),&[rows;2]);
    let lane=Variable::builtin(Builtin::AbsolutePosX,UIntKind::U64.into());let value=builder.read(0,lane);
    let result=builder.binary(Arithmetic::Div,value,Variable::constant(ConstantValue::Float(width as f32 as f64),f()),f());
    builder.write(lane,result);Ok(builder.kernel)
}
pub fn rstd_definition(rows:u64,width:u32,epsilon:f32)->Result<KernelDefinition> {
    if width==0 || rows>u32::MAX as u64 || !epsilon.is_finite() || epsilon<=0. {return Err(invalid("RMS statistics require a positive width and finite positive epsilon"));}
    let mut builder=Builder::new("ruda_cann_wide_rstd".into(),&[rows;2]);let lane=Variable::builtin(Builtin::AbsolutePosX,UIntKind::U64.into());
    let sum=builder.read(0,lane);let mean=builder.binary(Arithmetic::Div,sum,Variable::constant(ConstantValue::Float(width as f32 as f64),f()),f());
    let shifted=builder.binary(Arithmetic::Add,mean,Variable::constant(ConstantValue::Float(epsilon as f64),f()),f());
    let r=builder.unary(Arithmetic::InverseSqrt,shifted);builder.write(lane,r);Ok(builder.kernel)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ascend::{AscendCompiler,AscendOptions,AscendTarget,lower::{self,Node,Unary,Binary}};
    use ruda_core::{compiler::Compiler,launch::ExecutionMode};
    fn options(n:u64)->AscendOptions {AscendOptions {target:Some(AscendTarget::Ascend950DT),elements:n,..Default::default()}}
    // Test-only interpretation of the lowered IR, including actual mapped destination indices.
    fn evaluate(kernel:KernelDefinition,n:usize,inputs:&[&[f32]],destination:&mut [f32],partial:bool) {
        let p=lower::lower_map(kernel,n as u64,partial).unwrap();let mut values:Vec<Vec<f32>>=vec![];
        if n==0 {return;}
        for node in &p.nodes {
            values.push(match *node {
                Node::Input(id)=>(0..n as u64).map(|lane|inputs[id][p.load_indices[&id].eval(lane) as usize]).collect(),
                Node::UniformInput(id,offset)=>vec![inputs[id][offset as usize];n],
                Node::Constant(bits)=>vec![f32::from_bits(bits);n],
                Node::IndexFloat(id)=>(0..n as u64).map(|lane|p.index_values[id].eval(lane) as f32).collect(),
                Node::Binary(op,a,b)=>values[a].iter().zip(&values[b]).map(|(&a,&b)|match op {
                    Binary::Add=>a+b,Binary::Sub=>a-b,Binary::Mul=>a*b,Binary::Div=>a/b,Binary::Max=>a.max(b)}).collect(),
                Node::Unary(op,a)=>values[a].iter().map(|&a|match op {Unary::Neg=>-a,Unary::Abs=>a.abs(),
                    Unary::Exp=>a.exp(),Unary::Log=>a.ln(),Unary::Sqrt=>a.sqrt(),Unary::Rsqrt=>a.sqrt().recip(),Unary::Recip=>a.recip()}).collect(),
            });
        }
        for &(binding,value) in &p.stores {for lane in 0..n {
            destination[p.store_indices[&binding].eval(lane as u64) as usize]=values[value][lane];
        }}
    }
    fn tile(kind:WideStage,rows:usize,width:usize,start:usize,columns:usize,inputs:&[&[f32]],output:&mut [f32]) {
        evaluate(definition(kind,rows as u64,width as u32,start as u32,columns as u32).unwrap(),rows*columns,inputs,output,kind.partial());
    }
    #[test]
    fn every_wide_stage_compiles_with_explicit_output_contract_and_tail_bounds() {
        let stages=[WideStage::CopyTile,WideStage::ExpTile,WideStage::SoftmaxTile,WideStage::LogSoftmaxTile,
            WideStage::DotTile,WideStage::SoftmaxBackwardTile,WideStage::LogSoftmaxBackwardTile,WideStage::SumBackwardTile,WideStage::MeanBackwardTile,
            WideStage::SquareTile,WideStage::RmsNormTile,WideStage::RmsDotTile,WideStage::RmsBackwardTile,WideStage::RmsWeightTile,
            WideStage::CenteredSquareTile,WideStage::LayerNormTile,WideStage::LayerGTile,WideStage::LayerGYTile,WideStage::LayerBackwardTile,WideStage::LayerWeightTile];
        for width in [4128u32,8192,8224] {for rows in [0u64,1,2] {for start in (0..width).step_by(4096) {
            let columns=(width-start).min(4096);let n=rows*columns as u64;
            for stage in stages {
                let k=definition(stage,rows,width,start,columns).unwrap();
                if stage.partial() && rows!=0 {assert!(AscendCompiler.compile(k.clone(),&options(n),ExecutionMode::Checked,UIntKind::U64.into()).is_err());}
                let compiled=if stage.partial() {AscendCompiler.compile_partial_map(k,&options(n),ExecutionMode::Checked,UIntKind::U64.into())}
                    else {AscendCompiler.compile(k,&options(n),ExecutionMode::Checked,UIntKind::U64.into())}.unwrap();
                assert_eq!(compiled.bindings().last().unwrap().bytes,if stage.partial() {rows*width as u64*4} else {n*4});
                if stage.partial() && rows!=0 {
                    assert!(compiled.requires_initialized_outputs());
                    if start!=0 || rows>1 {
                        assert!(compiled.source().contains("[lane], run_copy)"));
                        assert!(!compiled.source().contains("scatter_cell.SetValue"));
                    }
                }
                assert!(compiled.ub_bytes()<=131072);
            }
        }}}
        assert!(definition(WideStage::CopyTile,1,4128,4096,64).is_err());
        assert!(definition(WideStage::CopyTile,u32::MAX as u64,4128,0,32).is_err());
        let mut bad=definition(WideStage::SoftmaxTile,2,4128,4096,32).unwrap();
        if let Operation::Operator(Operator::IndexAssign(op))=&mut bad.body.instructions.last_mut().unwrap().operation {op.index=integer(0);}
        assert!(AscendCompiler.compile_partial_map(bad,&options(64),ExecutionMode::Checked,UIntKind::U64.into()).is_err());
        let mut bad=definition(WideStage::SoftmaxTile,2,4128,4096,32).unwrap();bad.buffers.last_mut().unwrap().size=Some(64);
        assert!(AscendCompiler.compile_partial_map(bad,&options(64),ExecutionMode::Checked,UIntKind::U64.into()).is_err());
        assert!(AscendCompiler.compile(merge_definition(3,true).unwrap(),&options(3),ExecutionMode::Checked,UIntKind::U64.into()).unwrap().source().contains("AscendC::Max("));
        assert!(AscendCompiler.compile(zero_definition(65).unwrap(),&options(65),ExecutionMode::Checked,UIntKind::U64.into()).is_ok());
        assert!(AscendCompiler.compile(mean_definition(3,8224).unwrap(),&options(3),ExecutionMode::Checked,UIntKind::U64.into()).is_ok());
        assert!(AscendCompiler.compile(rstd_definition(3,8224,1e-3).unwrap(),&options(3),ExecutionMode::Checked,UIntKind::U64.into()).is_ok());
    }
    #[test]
    fn aligned_row_copy_and_broadcast_use_runs_but_strided_memory_keeps_scalar_sync() {
        let kernel=definition(WideStage::LogSoftmaxTile,3,8224,4096,4096).unwrap();
        let compiled=AscendCompiler.compile_partial_map(kernel,&options(3*4096),ExecutionMode::Checked,UIntKind::U64.into()).unwrap();
        assert!(compiled.source().contains("AscendC::DataCopyPad(load0[lane]"));
        assert!(compiled.source().contains("AscendC::Duplicate(load1[lane], run_value, run_aligned)"));
        assert!(compiled.source().contains("4096ULL - ((offset + lane) % 4096ULL)"));
        assert!(!compiled.source().contains("load0.SetValue(lane, gather_cell.GetValue(0))"));
        for scatter in [false,true] {
            let sizes=if scatter {[65,130]} else {[130,65]};let mut builder=Builder::new("ruda_strided_copy".into(),&sizes);
            let lane=Variable::builtin(Builtin::AbsolutePosX,UIntKind::U64.into());
            let stride=builder.binary(Arithmetic::Mul,lane,integer(2),u());
            let value=builder.read(0,if scatter {lane} else {stride});builder.write(if scatter {stride} else {lane},value);
            let compiled=if scatter {AscendCompiler.compile_partial_map(builder.kernel,&options(65),ExecutionMode::Checked,UIntKind::U64.into())}
                else {AscendCompiler.compile(builder.kernel,&options(65),ExecutionMode::Checked,UIntKind::U64.into())}.unwrap();
            if scatter {
                assert!(compiled.source().contains("scatter_cell.SetValue"));
                assert!(compiled.source().contains("HardEvent::S_MTE3"));assert!(compiled.source().contains("HardEvent::MTE3_S"));
            } else {assert!(compiled.source().contains("load0.SetValue(lane, gather_cell.GetValue(0))"));}
        }
    }
    #[test]
    fn wide_softmax_and_backward_tiles_match_independent_fp64_equations() {
        for width in [4128usize,8192,8224] {let rows=2;let n=rows*width;
            let x:Vec<f32>=(0..n).map(|i|-1000.+(i%47) as f32/8.).collect();
            let grad:Vec<f32>=(0..n).map(|i|(i%13) as f32/7.-0.6).collect();
            let mut maximum=vec![f32::NEG_INFINITY;rows];
            for start in (0..width).step_by(4096) {
                let columns=(width-start).min(4096);let mut copied=vec![0.;rows*columns];
                tile(WideStage::CopyTile,rows,width,start,columns,&[&x],&mut copied);
                let current:Vec<_>=copied.chunks(columns).map(|row|row.iter().copied().fold(f32::NEG_INFINITY,f32::max)).collect();
                let mut merged=vec![0.;rows];evaluate(merge_definition(rows as u64,true).unwrap(),rows,&[&maximum,&current],&mut merged,false);maximum=merged;
            }
            let mut total=vec![0.;rows];let mut tiles=vec![];
            for start in (0..width).step_by(4096) {
                let columns=(width-start).min(4096);let mut exp=vec![0.;rows*columns];
                tile(WideStage::ExpTile,rows,width,start,columns,&[&x,&maximum],&mut exp);
                let current:Vec<_>=exp.chunks(columns).map(|row|row.iter().sum::<f32>()).collect();
                let mut merged=vec![0.;rows];evaluate(merge_definition(rows as u64,false).unwrap(),rows,&[&total,&current],&mut merged,false);total=merged;
                tiles.push((start,columns,exp));
            }
            for logarithmic in [false,true] {
                let mut y=vec![f32::NAN;n];
                for (start,columns,exp) in &tiles {
                    if logarithmic {tile(WideStage::LogSoftmaxTile,rows,width,*start,*columns,&[&x,&maximum,&total],&mut y);}
                    else {tile(WideStage::SoftmaxTile,rows,width,*start,*columns,&[exp,&total],&mut y);}
                }
                let mut dot=vec![0.;rows];
                for (start,columns,_) in &tiles {
                    let mut product=vec![0.;rows*columns];
                    if logarithmic {tile(WideStage::CopyTile,rows,width,*start,*columns,&[&grad],&mut product);}
                    else {tile(WideStage::DotTile,rows,width,*start,*columns,&[&y,&grad],&mut product);}
                    for row in 0..rows {dot[row]+=product[row*columns..(row+1)*columns].iter().sum::<f32>();}
                }
                let mut dx=vec![f32::NAN;n];
                for (start,columns,_) in &tiles {
                    tile(if logarithmic {WideStage::LogSoftmaxBackwardTile} else {WideStage::SoftmaxBackwardTile},rows,width,*start,*columns,&[&y,&grad,&dot],&mut dx);
                }
                for row in 0..rows {
                    let data=&x[row*width..(row+1)*width];let max=data.iter().map(|&x|x as f64).fold(f64::NEG_INFINITY,f64::max);
                    let sum=data.iter().map(|&x|(x as f64-max).exp()).sum::<f64>();
                    let reference_dot=(0..width).map(|c|grad[row*width+c] as f64*if logarithmic {1.} else {(data[c] as f64-max).exp()/sum}).sum::<f64>();
                    for c in 0..width {let i=row*width+c;let probability=(data[c] as f64-max).exp()/sum;
                        let expected=if logarithmic {data[c] as f64-max-sum.ln()} else {probability};
                        let expected_dx=if logarithmic {grad[i] as f64-probability*reference_dot} else {probability*(grad[i] as f64-reference_dot)};
                        assert!((y[i] as f64-expected).abs()<2e-5+2e-5*expected.abs(),"forward width={width} lane={i}");
                        assert!((dx[i] as f64-expected_dx).abs()<2e-5+2e-5*expected_dx.abs(),"backward width={width} lane={i}");
                    }
                }
            }
        }
    }
    #[test]
    fn wide_sum_mean_and_broadcast_gradients_keep_the_full_width_divisor() {
        for width in [4128usize,8192,8224] {for rows in [1usize,3] {
            let x:Vec<f32>=(0..rows*width).map(|i|(i%19) as f32/8.-0.75).collect();
            let grad:Vec<f32>=(0..rows).map(|i|i as f32*0.25-0.5).collect();let mut sum=vec![0.;rows];
            for start in (0..width).step_by(4096) {
                let columns=(width-start).min(4096);let mut copied=vec![0.;rows*columns];
                tile(WideStage::CopyTile,rows,width,start,columns,&[&x],&mut copied);
                let current:Vec<_>=copied.chunks(columns).map(|row|row.iter().sum::<f32>()).collect();
                let mut merged=vec![0.;rows];evaluate(merge_definition(rows as u64,false).unwrap(),rows,&[&sum,&current],&mut merged,false);sum=merged;
            }
            let mut mean=vec![0.;rows];evaluate(mean_definition(rows as u64,width as u32).unwrap(),rows,&[&sum],&mut mean,false);
            for row in 0..rows {
                let reference=x[row*width..(row+1)*width].iter().map(|&x|x as f64).sum::<f64>();
                assert_eq!(sum[row] as f64,reference);
                assert!((mean[row] as f64-reference/width as f64).abs()<1e-6);
            }
            for averaged in [false,true] {
                let mut output=vec![f32::NAN;rows*width];
                for start in (0..width).step_by(4096) {
                    tile(if averaged {WideStage::MeanBackwardTile} else {WideStage::SumBackwardTile},rows,width,start,(width-start).min(4096),&[&grad],&mut output);
                }
                for (i,&x) in output.iter().enumerate() {
                    let expected=grad[i/width] as f64/if averaged {width as f64} else {1.};
                    assert!((x as f64-expected).abs()<1e-7);
                }
            }
        }}
    }
    #[test]
    fn wide_rms_norm_saved_statistics_and_affine_gradients_match_fp64() {
        for width in [4128usize,8192,8224] {let rows=3;let eps=1e-3f32;
            let x:Vec<f32>=(0..rows*width).map(|i|(i%31) as f32/7.-1.).collect();
            let weight:Vec<f32>=(0..width).map(|i|0.5+(i%7) as f32/9.).collect();
            let grad:Vec<f32>=(0..rows*width).map(|i|(i%11) as f32/5.-0.7).collect();
            let accumulated=|kind,inputs:&[&[f32]]| {
                let mut sum=vec![0.;rows];
                for start in (0..width).step_by(4096) {
                    let columns=(width-start).min(4096);let mut values=vec![0.;rows*columns];
                    tile(kind,rows,width,start,columns,inputs,&mut values);
                    for row in 0..rows {sum[row]+=values[row*columns..(row+1)*columns].iter().sum::<f32>();}
                }
                sum
            };
            let squares=accumulated(WideStage::SquareTile,&[&x]);let mut rstd=vec![0.;rows];
            evaluate(rstd_definition(rows as u64,width as u32,eps).unwrap(),rows,&[&squares],&mut rstd,false);
            let dots=accumulated(WideStage::RmsDotTile,&[&x,&grad,&weight]);let mut mean=vec![0.;rows];
            evaluate(mean_definition(rows as u64,width as u32).unwrap(),rows,&[&dots],&mut mean,false);
            let mut y=vec![f32::NAN;x.len()];let mut dx=y.clone();let mut parts=y.clone();
            for start in (0..width).step_by(4096) {
                let columns=(width-start).min(4096);
                tile(WideStage::RmsNormTile,rows,width,start,columns,&[&x,&weight,&rstd],&mut y);
                tile(WideStage::RmsBackwardTile,rows,width,start,columns,&[&x,&grad,&weight,&rstd,&mean],&mut dx);
                tile(WideStage::RmsWeightTile,rows,width,start,columns,&[&x,&grad,&rstd],&mut parts);
            }
            let mut expected_dw=vec![0.;width];
            for row in 0..rows {
                let data=&x[row*width..(row+1)*width];let r=(data.iter().map(|&x|(x as f64).powi(2)).sum::<f64>()/width as f64+eps as f64).sqrt().recip();
                let dot=(0..width).map(|c|grad[row*width+c] as f64*weight[c] as f64*data[c] as f64).sum::<f64>()/width as f64;
                assert!((rstd[row] as f64-r).abs()<2e-5);
                for c in 0..width {let i=row*width+c;let normalized=data[c] as f64*r;
                    let expected_y=normalized*weight[c] as f64;
                    let expected_dx=r*(grad[i] as f64*weight[c] as f64-data[c] as f64*r*r*dot);
                    expected_dw[c]+=grad[i] as f64*normalized;
                    assert!((y[i] as f64-expected_y).abs()<2e-4);
                    assert!((dx[i] as f64-expected_dx).abs()<2e-4);
                }
            }
            for c in 0..width {
                let observed=(0..rows).map(|row|parts[row*width+c] as f64).sum::<f64>();
                assert!((observed-expected_dw[c]).abs()<2e-4);
            }
        }
    }
    #[test]
    fn wide_layer_norm_centers_variance_and_uses_saved_affine_backward_statistics() {
        for width in [4128usize,8192,8224] {let rows=3;let eps=1e-3f32;
            let x:Vec<f32>=(0..rows*width).map(|i|(i%31) as f32/7.-1.).collect();
            let weight:Vec<f32>=(0..width).map(|i|0.5+(i%7) as f32/9.).collect();
            let bias:Vec<f32>=(0..width).map(|i|(i%5) as f32/8.-0.25).collect();
            let grad:Vec<f32>=(0..rows*width).map(|i|(i%11) as f32/5.-0.7).collect();
            let accumulated=|kind,inputs:&[&[f32]]| {
                let mut sum=vec![0.;rows];
                for start in (0..width).step_by(4096) {
                    let columns=(width-start).min(4096);let mut values=vec![0.;rows*columns];
                    tile(kind,rows,width,start,columns,inputs,&mut values);
                    for row in 0..rows {sum[row]+=values[row*columns..(row+1)*columns].iter().sum::<f32>();}
                }
                sum
            };
            let average=|sum:&[f32]| {let mut mean=vec![0.;rows];evaluate(mean_definition(rows as u64,width as u32).unwrap(),rows,&[sum],&mut mean,false);mean};
            let mean=average(&accumulated(WideStage::CopyTile,&[&x]));
            let squares=accumulated(WideStage::CenteredSquareTile,&[&x,&mean]);let mut rstd=vec![0.;rows];
            evaluate(rstd_definition(rows as u64,width as u32,eps).unwrap(),rows,&[&squares],&mut rstd,false);
            let mean_g=average(&accumulated(WideStage::LayerGTile,&[&grad,&weight]));
            let mean_gy=average(&accumulated(WideStage::LayerGYTile,&[&x,&grad,&weight,&mean,&rstd]));
            let mut y=vec![f32::NAN;x.len()];let mut dx=y.clone();let mut parts=y.clone();
            for start in (0..width).step_by(4096) {
                let columns=(width-start).min(4096);
                tile(WideStage::LayerNormTile,rows,width,start,columns,&[&x,&weight,&bias,&mean,&rstd],&mut y);
                tile(WideStage::LayerBackwardTile,rows,width,start,columns,&[&x,&grad,&weight,&mean,&rstd,&mean_g,&mean_gy],&mut dx);
                tile(WideStage::LayerWeightTile,rows,width,start,columns,&[&x,&grad,&mean,&rstd],&mut parts);
            }
            let mut expected_dw=vec![0.;width];
            for row in 0..rows {
                let data=&x[row*width..(row+1)*width];let m=data.iter().map(|&x|x as f64).sum::<f64>()/width as f64;
                let variance=data.iter().map(|&x|(x as f64-m).powi(2)).sum::<f64>()/width as f64;let r=(variance+eps as f64).sqrt().recip();
                let mean_g=(0..width).map(|c|grad[row*width+c] as f64*weight[c] as f64).sum::<f64>()/width as f64;
                let mean_gy=(0..width).map(|c|grad[row*width+c] as f64*weight[c] as f64*(data[c] as f64-m)*r).sum::<f64>()/width as f64;
                assert!((mean[row] as f64-m).abs()<2e-5);assert!((rstd[row] as f64-r).abs()<2e-5);
                for c in 0..width {let i=row*width+c;let normalized=(data[c] as f64-m)*r;
                    let expected_y=normalized*weight[c] as f64+bias[c] as f64;
                    let expected_dx=(grad[i] as f64*weight[c] as f64-mean_g-normalized*mean_gy)*r;
                    expected_dw[c]+=grad[i] as f64*normalized;
                    assert!((y[i] as f64-expected_y).abs()<2e-4);
                    assert!((dx[i] as f64-expected_dx).abs()<2e-4);
                }
            }
            for c in 0..width {
                let observed=(0..rows).map(|row|parts[row*width+c] as f64).sum::<f64>();
                assert!((observed-expected_dw[c]).abs()<2e-4);
            }
        }
    }
}

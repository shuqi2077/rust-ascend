//! Grouped-query KV head repetition and batched transpose-Jacobian reduction.
use super::{Result,invalid};
use ruda_core::{ir::*,kernel::{KernelArg,KernelDefinition,KernelOptions,Visibility},launch::RudaDim};

#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub struct RepeatKvSpec {pub batch:u32,pub kv_heads:u32,pub query_heads:u32,pub sequence:u32,pub width:u32}
impl RepeatKvSpec {
    /// Consecutive query heads share one KV head; both complete domains must fit u32.
    pub fn elements(self)->Result<(u64,u64)> {
        if self.kv_heads==0 || self.query_heads==0 || self.query_heads%self.kv_heads!=0 {
            return Err(invalid("KV repetition requires positive heads and query_heads divisible by kv_heads"));
        }
        let count=|heads:u32|[self.batch,heads,self.sequence,self.width].into_iter()
            .try_fold(1u64,|n,d|n.checked_mul(d as u64)).filter(|&n|n<=u32::MAX as u64)
            .ok_or_else(||invalid("KV repetition complete domain exceeds u32"));
        Ok((count(self.kv_heads)?,count(self.query_heads)?))
    }
}
fn f()->Type {Type::new(FloatKind::F32.into())}
fn u()->Type {Type::new(UIntKind::U64.into())}
fn integer(n:u64)->Variable {Variable::constant(ConstantValue::UInt(n),u())}
struct Builder {kernel:KernelDefinition,next:u32}
impl Builder {
    fn new(name:&str,sizes:&[u64])->Self {
        Self {kernel:KernelDefinition {buffers:sizes.iter().enumerate().map(|(id,&size)|KernelArg {id:id as u32,
            visibility:if id+1==sizes.len() {Visibility::ReadWrite} else {Visibility::Read},ty:f(),size:Some(size as usize),has_extended_meta:false}).collect(),
            tensor_maps:vec![],scalars:vec![],ruda_dim:RudaDim::new_1d(64),body:Scope::root(false),
            options:KernelOptions {kernel_name:name.into(),..Default::default()}},next:0}
    }
    fn op(&mut self,operation:impl Into<Operation>,ty:Type)->Variable {
        let out=Variable::new(VariableKind::LocalConst{id:self.next},ty);self.next+=1;
        self.kernel.body.instructions.push(Instruction::new(operation,out));out
    }
    fn index(&mut self,kind:fn(BinaryOperator)->Arithmetic,a:Variable,b:Variable)->Variable {self.op(kind(BinaryOperator {lhs:a,rhs:b}),u())}
    fn read(&mut self,id:u32,index:Variable)->Variable {self.op(Operator::Index(IndexOperator {
        list:Variable::new(VariableKind::GlobalInputArray(id),f()),index,vector_size:0,unroll_factor:1}),f())}
    fn write(&mut self,index:Variable,value:Variable) {
        self.kernel.body.instructions.push(Instruction::new(Operator::IndexAssign(IndexAssignOperator {index,value,vector_size:0,unroll_factor:1}),
            Variable::new(VariableKind::GlobalOutputArray(self.kernel.buffers.len() as u32-1),f())));
    }
}
/// X[B,Hkv,N,D] -> Y[B,Hq,N,D], with kv_head = query_head / (Hq/Hkv).
pub fn repeat_definition(spec:RepeatKvSpec)->Result<KernelDefinition> {
    let (input,output)=spec.elements()?;let mut b=Builder::new("ruda_cann_repeat_kv_heads",&[input,output]);
    let lane=Variable::builtin(Builtin::AbsolutePosX,UIntKind::U64.into());
    let index=if output==0 {lane} else {
        let width=spec.sequence as u64*spec.width as u64;
        let head=b.index(Arithmetic::Div,lane,integer(width));
        let kv_head=b.index(Arithmetic::Div,head,integer((spec.query_heads/spec.kv_heads) as u64));
        let base=b.index(Arithmetic::Mul,kv_head,integer(width));
        let column=b.index(Arithmetic::Modulo,lane,integer(width));
        b.index(Arithmetic::Add,base,column)
    };
    let value=b.read(0,index);b.write(lane,value);Ok(b.kernel)
}
/// One reduction level over [rows,groups,width], preserving the odd final group.
/// Pair bindings are [X, X readonly alias, next]; tail bindings are [X, next].
/// The two disjoint injective patches together write every next element exactly once.
pub fn reduce_definition(rows:u64,groups:u32,width:u64,tail:bool)->Result<(KernelDefinition,u64)> {
    if groups<2 || width==0 || (tail && groups%2==0) {return Err(invalid("KV reduction requires groups>=2, positive width and an odd group for tail"));}
    let count=|groups:u32|rows.checked_mul(groups as u64).and_then(|n|n.checked_mul(width)).filter(|&n|n<=u32::MAX as u64)
        .ok_or_else(||invalid("KV reduction domain exceeds u32"));
    let input=count(groups)?;let pairs=groups/2;let next=groups.div_ceil(2);let output=count(next)?;
    let stride=(groups as u64).checked_mul(width).ok_or_else(||invalid("KV reduction row stride overflow"))?;
    let next_stride=(next as u64).checked_mul(width).ok_or_else(||invalid("KV reduction output stride overflow"))?;
    let pair_width=(pairs as u64).checked_mul(width).ok_or_else(||invalid("KV reduction pair span overflow"))?;
    let span=if tail {width} else {pair_width};let elements=rows.checked_mul(span).ok_or_else(||invalid("KV reduction launch overflow"))?;
    let mut b=Builder::new(if tail {"ruda_cann_repeat_kv_tail"} else {"ruda_cann_repeat_kv_pairs"},
        &if tail {vec![input,output]} else {vec![input,input,output]});
    let lane=Variable::builtin(Builtin::AbsolutePosX,UIntKind::U64.into());
    let row=b.index(Arithmetic::Div,lane,integer(span));let column=b.index(Arithmetic::Modulo,lane,integer(span));
    let source_base=b.index(Arithmetic::Mul,row,integer(stride));
    let mut source=b.index(Arithmetic::Add,source_base,column);
    let output_base=b.index(Arithmetic::Mul,row,integer(next_stride));
    let mut destination=b.index(Arithmetic::Add,output_base,column);
    let value=if tail {
        source=b.index(Arithmetic::Add,source,integer(stride-width));
        destination=b.index(Arithmetic::Add,destination,integer(pair_width));
        b.read(0,source)
    } else {
        let right=b.index(Arithmetic::Add,source,integer(pair_width));
        let left=b.read(0,source);let right=b.read(1,right);
        b.op(Arithmetic::Add(BinaryOperator {lhs:left,rhs:right}),f())
    };
    b.write(destination,value);Ok((b.kernel,elements))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ascend::{AscendCompiler,AscendOptions,AscendTarget,lower::{self,Node,Binary}};
    use ruda_core::{compiler::Compiler,launch::ExecutionMode};
    fn evaluate(kernel:KernelDefinition,elements:u64,partial:bool,inputs:&[&[f32]],out:&mut [f32]) {
        let p=lower::lower_map(kernel.clone(),elements,partial).unwrap();let mut values:Vec<Vec<f32>>=vec![];
        for node in &p.nodes {
            values.push(match *node {
                Node::Input(id)=>(0..elements).map(|lane|inputs[id][p.load_indices[&id].eval(lane) as usize]).collect(),
                Node::UniformInput(id,offset)=>vec![inputs[id][offset as usize];elements as usize],
                Node::Binary(Binary::Add,a,b)=>values[a].iter().zip(&values[b]).map(|(&a,&b)|a+b).collect(),
                _=>panic!("unexpected operation in KV repetition"),
            });
        }
        let (id,value)=p.stores[0];
        for lane in 0..elements {out[p.store_indices[&id].eval(lane) as usize]=values[value][lane as usize];}
        let options=AscendOptions {target:Some(AscendTarget::Ascend950DT),elements,..Default::default()};
        let compiled=if partial {AscendCompiler.compile_partial_map(kernel,&options,ExecutionMode::Checked,UIntKind::U64.into())}
            else {AscendCompiler.compile(kernel,&options,ExecutionMode::Checked,UIntKind::U64.into())}.unwrap();
        assert!(!compiled.source().contains("static_cast<float>"));
    }
    #[test]
    fn repeat_mapping_preserves_batches_heads_odd_shapes_and_special_bits() {
        for batch in [0,1,2] {for heads in [1,3] {for group in [1,2,3,5,8] {for (sequence,width) in [(0,7),(3,0),(3,7),(2,32)] {
            let spec=RepeatKvSpec {batch,kv_heads:heads,query_heads:heads*group,sequence,width};let (input,output)=spec.elements().unwrap();
            let x:Vec<f32>=(0..input).map(|i|f32::from_bits([0,0x80000000,0x7fc01234,0x7f800000,0xff800000,0x3f800000][i as usize%6])).collect();
            let mut y=vec![f32::NAN;output as usize];evaluate(repeat_definition(spec).unwrap(),output,false,&[&x],&mut y);
            for b in 0..batch as usize {for h in 0..spec.query_heads as usize {for n in 0..sequence as usize {for d in 0..width as usize {
                let src=((b*heads as usize+h/group as usize)*sequence as usize+n)*width as usize+d;
                let dst=((b*spec.query_heads as usize+h)*sequence as usize+n)*width as usize+d;
                assert_eq!(y[dst].to_bits(),x[src].to_bits());
            }}}}
        }}}}
    }
    #[test]
    fn batched_transpose_jacobian_sums_all_replicas_without_crossing_kv_heads() {
        for rows in [1,6] {for width in [1,7,32,96] {for initial_groups in [1,2,3,5,9,16] {
            let mut value:Vec<f32>=(0..rows*initial_groups*width).map(|i|(i%29) as f32/8.-1.25).collect();
            let original=value.clone();let mut groups=initial_groups as u32;
            while groups>1 {
                let mut next=vec![f32::NAN;rows*groups.div_ceil(2) as usize*width];
                let (kernel,n)=reduce_definition(rows as u64,groups,width as u64,false).unwrap();evaluate(kernel,n,true,&[&value,&value],&mut next);
                if groups%2!=0 {let (kernel,n)=reduce_definition(rows as u64,groups,width as u64,true).unwrap();evaluate(kernel,n,true,&[&value],&mut next);}
                assert!(next.iter().all(|x|x.is_finite()));value=next;groups=groups.div_ceil(2);
            }
            for r in 0..rows {for col in 0..width {
                let expected=(0..initial_groups).map(|g|original[(r*initial_groups+g)*width+col] as f64).sum::<f64>();
                assert_eq!(value[r*width+col] as f64,expected);
            }}
        }}}
        for groups in [0,1,2,4] {assert!(reduce_definition(2,groups,32,true).is_err());}
        assert!(reduce_definition(2,3,0,false).is_err());
    }
    #[test]
    fn repeat_rejects_invalid_head_ratio_and_overflow_not_long_context() {
        let spec=RepeatKvSpec {batch:2,kv_heads:3,query_heads:15,sequence:8192,width:128};assert!(repeat_definition(spec).is_ok());
        for spec in [RepeatKvSpec {kv_heads:0,..spec},RepeatKvSpec {query_heads:0,..spec},RepeatKvSpec {query_heads:14,..spec},
            RepeatKvSpec {sequence:u32::MAX,..spec}] {assert!(repeat_definition(spec).is_err());}
        assert!(reduce_definition(u32::MAX as u64,3,32,false).is_err());
    }
}

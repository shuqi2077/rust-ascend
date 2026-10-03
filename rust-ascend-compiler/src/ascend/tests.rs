use super::*;
use super::{lower::{Node,Binary,Unary},programs::{MapProgram,definition}};
use ruda_core::{ir::*,kernel::*,launch::RudaDim};
fn opt(n:u64)->AscendOptions{AscendOptions{target:Some(AscendTarget::Ascend950DT),elements:n,..Default::default()}}
fn compile(k:KernelDefinition,n:u64)->Result<AscendKernel>{AscendCompiler.compile(k,&opt(n),ExecutionMode::Checked,UIntKind::U64.into())}
fn f()->Type{Type::scalar(ElemType::Float(FloatKind::F32))}
fn v(id:u32)->Variable{Variable::new(VariableKind::LocalMut{id},f())}
#[cfg_attr(unix,link(name="m"))]
unsafe extern "C" {#[link_name="erf"] fn c_erf(x:f64)->f64;}
pub(super) fn erf_reference(x:f32)->f32 {unsafe {c_erf(x as f64) as f32}}
fn activation_kernel(make:impl FnOnce(Variable)->Arithmetic)->KernelDefinition {
    let mut kernel=definition(MapProgram::Copy);let input=kernel.body.instructions[0].out.unwrap();let output=v(999);
    kernel.body.instructions.insert(1,Instruction::new(make(input),output));
    if let Operation::Operator(Operator::IndexAssign(store))=&mut kernel.body.instructions[2].operation {store.value=output;}
    kernel
}
#[test]fn native_math_workspace_is_distinct_budgeted_and_reused() {
    for (kind,factor) in [(0,3),(1,1)] {
        let mut kernel=activation_kernel(|input|if kind==0 {Arithmetic::Erf(UnaryOperator{input})} else {Arithmetic::Tanh(UnaryOperator{input})});
        for tile in [8,64,256,4096] {
            let mut options=opt(65);options.tile_elements=tile;
            let compiled=AscendCompiler.compile(kernel.clone(),&options,ExecutionMode::Checked,UIntKind::U64.into()).unwrap();
            let p=lower::lower(kernel.clone(),65).unwrap();let a=plan::allocate(&p,true).unwrap();
            let workspace=factor*(tile as usize*4).max(256);
            assert_eq!(compiled.ub_bytes() as usize,(p.bindings.len()+a.slots)*tile as usize*4+workspace);
            assert!(compiled.source().contains(&format!("pipe.InitBuffer(math_workspace, {workspace}U)")));
            assert!(compiled.source().contains(if kind==0 {"Erf<float, false, ruda_erf_config>"} else {"Tanh<float, false, ruda_tanh_config>"}));
            assert!(compiled.source().contains(if kind==0 {"SUBSECTION_POLYNOMIAL_APPROXIMATION"} else {"SUBSECTION_COMPENSATION"}));
            options.ub_limit_bytes=compiled.ub_bytes()-1;
            assert!(AscendCompiler.compile(kernel.clone(),&options,ExecutionMode::Checked,UIntKind::U64.into()).is_err());
        }
        if kind==0 {
            kernel.body.instructions.insert(2,Instruction::new(Arithmetic::Tanh(UnaryOperator{input:v(999)}),v(998)));
            if let Operation::Operator(Operator::IndexAssign(store))=&mut kernel.body.instructions[3].operation {store.value=v(998);}
            assert_eq!(plan::math_workspace(&lower::lower(kernel,65).unwrap(),8).unwrap(),768);
        }
    }
    assert_eq!(plan::math_workspace(&lower::lower(definition(MapProgram::Copy),65).unwrap(),8).unwrap(),0);
}
#[test]fn specialized_integer_powers_preserve_parity_and_inverse_first() {
    let input=[-2.,-1.,-0.,0.,0.5,2.,1e20,f32::INFINITY,f32::NEG_INFINITY,f32::NAN];
    for exponent in [-3i64,-2,-1,0,1,2,3,7] {
        let rhs=Variable::constant(ConstantValue::Int(exponent),Type::scalar(ElemType::Int(IntKind::I64)));
        let kernel=activation_kernel(|lhs|Arithmetic::Powi(BinaryOperator{lhs,rhs}));
        compile(kernel.clone(),input.len() as u64).unwrap();
        let actual=evaluate(kernel,input.len(),[&input,&[],&[]]);
        for (&a,&x) in actual[0].iter().zip(&input) {
            let expected=if exponent<0 {x.recip().powi(-exponent as i32)} else {x.powi(exponent as i32)};
            assert!(a.to_bits()==expected.to_bits() || a.is_nan() && expected.is_nan() || (a-expected).abs()<=expected.abs()*2e-6);
        }
        if exponent==-2 {assert!(actual[0][6]>0.);}
    }
    for rhs in [Variable::constant(ConstantValue::Int(i64::MIN),Type::scalar(ElemType::Int(IntKind::I64))),
        Variable::constant(ConstantValue::UInt(u64::MAX),Type::scalar(ElemType::UInt(UIntKind::U64)))] {
        let kernel=activation_kernel(|lhs|Arithmetic::Powi(BinaryOperator{lhs,rhs}));
        assert!(lower::lower(kernel.clone(),3).unwrap().nodes.len()<=130);
        let result=evaluate(kernel.clone(),3,[&[-1.,1.,0.],&[],&[]]);
        assert_eq!(result[0],if matches!(rhs.kind,VariableKind::Constant(ConstantValue::Int(_))) {vec![1.,1.,f32::INFINITY]} else {vec![-1.,1.,0.]});
        compile(kernel,3).unwrap();
    }
}
#[test]fn integer_power_does_not_guess_dynamic_or_float_exponents() {
    for rhs in [v(999),Variable::constant(ConstantValue::Float(3.),f())] {
        assert!(compile(activation_kernel(|lhs|Arithmetic::Powi(BinaryOperator{lhs,rhs})),1).is_err());
    }
}
#[test]fn full_range_trigonometry_uses_sdk_workspace_and_one_shared_buffer() {
    for (make,call) in [(Arithmetic::Sin as fn(UnaryOperator)->Arithmetic,"Sin<float, false, ruda_sin_config>"),
        (Arithmetic::Cos as fn(UnaryOperator)->Arithmetic,"Cos<float, false, ruda_cos_config>")] {
        let mut kernel=activation_kernel(|input|make(UnaryOperator{input}));
        for tile in [8u32,24,32,64,256,4096] {
            let mut options=opt(65);options.tile_elements=tile;
            let compiled=AscendCompiler.compile(kernel.clone(),&options,ExecutionMode::Checked,UIntKind::U64.into()).unwrap();
            let workspace=tile.div_ceil(32) as usize*32*8+32;
            let p=lower::lower(kernel.clone(),65).unwrap();let slots=plan::allocate(&p,true).unwrap().slots;
            assert_eq!(compiled.ub_bytes() as usize,(p.bindings.len()+slots)*tile as usize*4+workspace);
            assert!(compiled.source().contains(call));assert!(compiled.source().contains("RADIAN_REDUCTION"));
            assert!(compiled.source().contains(&format!("pipe.InitBuffer(math_workspace, {workspace}U)")));
            options.ub_limit_bytes=compiled.ub_bytes()-1;
            assert!(AscendCompiler.compile(kernel.clone(),&options,ExecutionMode::Checked,UIntKind::U64.into()).is_err());
        }
        kernel.body.instructions.insert(2,Instruction::new(Arithmetic::Cos(UnaryOperator{input:v(999)}),v(998)));
        kernel.body.instructions.insert(3,Instruction::new(Arithmetic::Tanh(UnaryOperator{input:v(998)}),v(997)));
        if let Operation::Operator(Operator::IndexAssign(store))=&mut kernel.body.instructions[4].operation {store.value=v(997);}
        let p=lower::lower(kernel.clone(),65).unwrap();assert_eq!(plan::math_workspace(&p,8).unwrap(),288);
        let compiled=compile(kernel.clone(),65).unwrap();assert_eq!(compiled.source().matches("TPosition::VECCALC> math_workspace;").count(),1);
        kernel.body.instructions.insert(4,Instruction::new(Arithmetic::Erf(UnaryOperator{input:v(997)}),v(996)));
        if let Operation::Operator(Operator::IndexAssign(store))=&mut kernel.body.instructions[5].operation {store.value=v(996);}
        assert_eq!(plan::math_workspace(&lower::lower(kernel,65).unwrap(),8).unwrap(),768);
    }
}
#[test]fn real_common_ir_generates_vector_primitives(){let k=compile(definition(MapProgram::Add),1025).unwrap();assert!(k.source().contains("AscendC::Add("));assert!(k.source().contains("DataCopyPad"));assert!(k.source().contains("count * uint32_t(sizeof(float))"));assert!(!k.source().contains("aclnn"));assert!(!k.source().contains("deep_gemm"));assert_eq!(k.bindings().len(),3);assert_eq!(k.bindings()[2].bytes,4100);}
#[test]fn every_algorithm_is_a_common_kernel(){for p in [MapProgram::Copy,MapProgram::Add,MapProgram::Mul,MapProgram::Silu,MapProgram::SiluMul,MapProgram::SiluBackward,MapProgram::SiluMulBackward]{let k=compile(definition(p),513).unwrap();assert_eq!(k.bindings().len(),p.input_count()+p.output_count());assert!(k.ub_bytes()<=131072);}}
#[test]fn missing_target_is_not_guessed(){assert!(AscendCompiler.compile(definition(MapProgram::Add),&AscendOptions::default(),ExecutionMode::Checked,UIntKind::U64.into()).is_err());}
#[test]fn map_unchecked_has_the_same_proven_element_domain(){let a=AscendCompiler.compile(definition(MapProgram::Add),&opt(16),ExecutionMode::Unchecked,UIntKind::U64.into()).unwrap();assert_eq!(a.source(),compile(definition(MapProgram::Add),16).unwrap().source());assert!(AscendCompiler.compile(definition(MapProgram::Add),&opt(16),ExecutionMode::Validate,UIntKind::U64.into()).is_err());}
#[test]fn logical_u32_indices_keep_native_gm_pointers(){let a=AscendCompiler.compile(definition(MapProgram::Add),&opt(16),ExecutionMode::Checked,UIntKind::U32.into()).unwrap();assert_eq!(a.source(),compile(definition(MapProgram::Add),16).unwrap().source());}
#[test]fn limits_are_checked(){for tile in [0,1,7,15,4097]{let mut o=opt(16);o.tile_elements=tile;assert!(AscendCompiler.compile(definition(MapProgram::Add),&o,ExecutionMode::Checked,UIntKind::U64.into()).is_err());}let mut o=opt(16);o.ub_limit_bytes=32;assert!(AscendCompiler.compile(definition(MapProgram::Silu),&o,ExecutionMode::Checked,UIntKind::U64.into()).is_err());}
#[test]fn code_injection_in_entry_rejected(){for name in ["", "1x", "x(){", "x\n", "x::y"]{let mut k=definition(MapProgram::Add);k.options.kernel_name=name.into();assert!(compile(k,16).is_err());}}
#[test]fn unsupported_buffer_precision_is_not_cast(){let mut k=definition(MapProgram::Add);k.buffers[0].ty=Type::scalar(ElemType::Float(FloatKind::F16));assert!(compile(k,16).is_err());}
#[test]fn output_is_not_loadable_as_an_input(){let mut k=definition(MapProgram::Add);if let Operation::Operator(Operator::Index(op))=&mut k.body.instructions[0].operation{op.list=Variable::new(VariableKind::GlobalOutputArray(2),f());}assert!(compile(k,16).is_err());}
#[test]fn duplicate_and_missing_stores_are_rejected(){let mut k=definition(MapProgram::Add);k.body.instructions.push(k.body.instructions.last().unwrap().clone());assert!(compile(k,16).is_err());let mut k=definition(MapProgram::Add);k.body.instructions.pop();assert!(compile(k,16).is_err());}
#[test]fn constant_loads_broadcast_and_out_of_bounds_fail(){let mut k=definition(MapProgram::Add);if let Operation::Operator(Operator::Index(op))=&mut k.body.instructions[0].operation{op.index=0u32.into();}assert!(compile(k.clone(),16).unwrap().source().contains("gather_cell"));if let Operation::Operator(Operator::Index(op))=&mut k.body.instructions[0].operation{op.index=16u32.into();}assert!(compile(k,16).is_err());}
#[test]fn undefined_local_rejected(){let mut k=definition(MapProgram::Add);if let Operation::Arithmetic(Arithmetic::Add(op))=&mut k.body.instructions[2].operation{op.lhs=v(999);}assert!(compile(k,16).is_err());}
#[test]fn unsupported_matrix_warp_sync_not_scalarized(){let mut k=definition(MapProgram::Add);k.body.instructions.insert(0,Instruction::no_out(Branch::Return));assert!(compile(k,16).is_err());let mut k=definition(MapProgram::Add);k.ruda_dim=RudaDim::new_2d(32,4);assert!(compile(k,16).is_err());}
#[test]fn canonical_length_guard_is_accepted(){let mut k=definition(MapProgram::Add);let arr=Variable::new(VariableKind::GlobalInputArray(0),f());let len=Variable::new(VariableKind::LocalMut{id:50},Type::scalar(ElemType::UInt(UIntKind::U32)));let outside=Variable::new(VariableKind::LocalMut{id:51},Type::scalar(ElemType::Bool));let index=Variable::builtin(Builtin::AbsolutePosX,UIntKind::U32.into());let mut child=Scope::root(false);child.instructions.push(Instruction::no_out(Branch::Return));let mut prefix=vec![Instruction::new(Metadata::Length{var:arr},len),Instruction::new(Comparison::GreaterEqual(BinaryOperator{lhs:index,rhs:len}),outside),Instruction::no_out(Branch::If(Box::new(If{cond:outside,scope:child})))];prefix.append(&mut k.body.instructions);k.body.instructions=prefix;assert!(compile(k,17).is_ok());}
#[test]fn canonical_inside_guard_keeps_the_same_vector_operations(){
    for n in [0,1,65,1025] {
        let flat=definition(MapProgram::SiluMulBackward);
        let mut guarded=flat.clone();
        let len=Variable::new(VariableKind::LocalMut{id:50},Type::new(UIntKind::U64.into()));
        let inside=Variable::new(VariableKind::LocalMut{id:51},Type::scalar(ElemType::Bool));
        let mut child=Scope::root(false);child.instructions=std::mem::take(&mut guarded.body.instructions);
        guarded.body.instructions=vec![
            Instruction::new(Metadata::Length{var:Variable::new(VariableKind::GlobalInputArray(0),f())},len),
            Instruction::new(Comparison::Lower(BinaryOperator{lhs:Variable::builtin(Builtin::AbsolutePosX,UIntKind::U64.into()),rhs:len}),inside),
            Instruction::no_out(Branch::If(Box::new(If{cond:inside,scope:child}))),
        ];
        assert_eq!(compile(guarded.clone(),n).unwrap().source(),compile(flat,n).unwrap().source());
        if let Operation::Branch(Branch::If(branch))=&mut guarded.body.instructions[2].operation {
            branch.scope.instructions.insert(0,Instruction::no_out(Branch::Return));
        }
        assert!(compile(guarded,n).is_err());
    }
}
#[test]fn input_sizes_and_extended_metadata_are_checked(){let mut k=definition(MapProgram::Add);k.buffers[0].size=Some(18);assert_eq!(compile(k.clone(),17).unwrap().bindings()[0].bytes,72);k.buffers[0].size=Some(16);assert!(compile(k.clone(),17).is_err());k.buffers[0].size=Some(18);k.buffers[0].has_extended_meta=true;assert!(compile(k,17).is_err());}
#[test]fn signed_zero_constant_bits_survive(){let mut k=definition(MapProgram::Copy);let value=Variable::constant(ConstantValue::Float(-0.0),f());k.body.instructions.insert(1,Instruction::new(Arithmetic::Add(BinaryOperator{lhs:v(0),rhs:value}),v(10))); // use the actual loaded value type/kind below
if let Some(load)=k.body.instructions[0].out{if let Operation::Arithmetic(Arithmetic::Add(op))=&mut k.body.instructions[1].operation{op.lhs=load;}}
assert!(compile(k,9).unwrap().source().contains("0x80000000U"));}
#[test]fn zero_elements_have_explicit_no_work_body(){let k=compile(definition(MapProgram::Copy),0).unwrap();assert!(k.source().contains("elements == 0"));assert_eq!(k.bindings()[0].bytes,0);}
#[test]fn reuse_lowers_ub_without_overwriting_live_operands(){let k=definition(MapProgram::SiluMulBackward);let p=lower::lower(k.clone(),257).unwrap();let reused=plan::allocate(&p,true).unwrap();let fresh=plan::allocate(&p,false).unwrap();assert!(reused.slots<fresh.slots);for(i,node)in p.nodes.iter().enumerate(){for src in node.inputs(){if let(Some(x),Some(y))=(reused.node_slots[i],reused.node_slots[src]){assert_ne!(x,y);}}}let mut o=opt(257);o.reuse_temporaries=false;let old=AscendCompiler.compile(k.clone(),&o,ExecutionMode::Checked,UIntKind::U64.into()).unwrap();let new=compile(k,257).unwrap();assert!(new.ub_bytes()<old.ub_bytes());}
#[test]fn same_ir_can_use_ptx_backend(){
    #[cfg(feature="ptx")] {
        use crate::ptx::*;
        let k=definition(MapProgram::SiluMul);
        let p=PtxCompiler.compile(k.clone(),&PtxCompilationOptions{target:Some(PtxTarget{version:(8,0),sm:75})},ExecutionMode::Checked,UIntKind::U64.into()).unwrap();
        let a=compile(k,1025).unwrap();assert!(p.source.contains(".entry"));assert!(a.source().contains("AscendC::Mul"));
    }
}
#[test]fn index_to_fp32_cast_requires_an_exact_integer_domain(){
    let mut kernel=definition(MapProgram::Copy);
    let output=v(999);
    let lane=Variable::builtin(Builtin::AbsolutePosX,UIntKind::U64.into());
    kernel.body.instructions.insert(1,Instruction::new(Operator::Cast(UnaryOperator{input:lane}),output));
    if let Operation::Operator(Operator::IndexAssign(store))=&mut kernel.body.instructions[2].operation {store.value=output;}
    let compiled=compile(kernel.clone(),65).unwrap();
    assert!(compiled.source().contains("static_cast<float>((offset + lane))"));
    assert!(compiled.source().contains("ruda_map_sync<AscendC::HardEvent::S_V>"));
    assert!(compile(kernel.clone(),(1<<24)+1).is_ok());
    assert!(compile(kernel,(1<<24)+2).is_err());
}
/// Test-only reference of checked SSA expressions, never linked to a device executor.
fn reference(op:MapProgram,x:&[f32],y:&[f32],dy:&[f32])->Vec<Vec<f32>>{
    evaluate(definition(op),x.len(),[x,y,dy])
}
pub(super) fn evaluate(kernel:KernelDefinition,elements:usize,inputs:[&[f32];3])->Vec<Vec<f32>>{
    let p=lower::lower(kernel,elements as u64).unwrap();let mut nodes:Vec<Vec<f32>>=vec![];
    for node in &p.nodes{let z:Vec<f32>=match *node{
        Node::Input(i)=>(0..elements as u64).map(|lane|inputs[i][p.load_indices[&i].eval(lane) as usize]).collect(),
        Node::UniformInput(i,offset)=>vec![inputs[i][offset as usize];elements],
        Node::Constant(bits)=>vec![f32::from_bits(bits);elements],
        Node::IndexFloat(i)=>(0..elements as u64).map(|lane|p.index_values[i].eval(lane) as f32).collect(),
        Node::IndexSelect(i,a,b)=>(0..elements).map(|lane|if p.predicates[i].eval(lane as u64) {nodes[a][lane]} else {nodes[b][lane]}).collect(),
        Node::Compare(comparison,a,b)=>(0..elements).map(|lane|if comparison.float_eval(nodes[a][lane],nodes[b][lane]) {1.} else {0.}).collect(),
        Node::DataSelect(mask,a,b)=>(0..elements).map(|lane|if nodes[mask][lane]!=0. {nodes[a][lane]} else {nodes[b][lane]}).collect(),
        Node::Unary(u,a)=>nodes[a].iter().map(|&v|match u{Unary::Neg=>-v,Unary::Abs=>v.abs(),Unary::Exp=>v.exp(),Unary::Log=>v.ln(),Unary::Sqrt=>v.sqrt(),Unary::Rsqrt=>1.0/v.sqrt(),Unary::Recip=>1.0/v,Unary::Erf=>erf_reference(v),Unary::Tanh=>v.tanh(),Unary::Sin=>v.sin(),Unary::Cos=>v.cos()}).collect(),
        Node::Binary(b,a,c)=>nodes[a].iter().zip(&nodes[c]).map(|(&v,&w)|match b{Binary::Add=>v+w,Binary::Sub=>v-w,Binary::Mul=>v*w,Binary::Div=>v/w,Binary::Max=>v.max(w)}).collect()};nodes.push(z);}
    p.stores.iter().map(|&(_,v)|nodes[v].clone()).collect()
}
#[test]fn readonly_uniform_slots_are_dynamic_and_keep_distinct_offsets(){
    let mut kernel=definition(MapProgram::Add);kernel.buffers[1].size=Some(9);
    if let Operation::Operator(Operator::Index(read))=&mut kernel.body.instructions[1].operation {read.index=0u64.into();}
    let scalar=v(998);let result=v(999);let sum=kernel.body.instructions[2].out.unwrap();
    kernel.body.instructions.insert(3,Instruction::new(Operator::Index(IndexOperator {
        list:Variable::new(VariableKind::GlobalInputArray(1),f()),index:8u64.into(),vector_size:0,unroll_factor:1}),scalar));
    kernel.body.instructions.insert(4,Instruction::new(Arithmetic::Mul(BinaryOperator {lhs:sum,rhs:scalar}),result));
    if let Operation::Operator(Operator::IndexAssign(write))=&mut kernel.body.instructions[5].operation {write.value=result;}
    for elements in [1,65] {
        let compiled=compile(kernel.clone(),elements).unwrap();
        assert!(compiled.source().contains("g1[0ULL]"));assert!(compiled.source().contains("g1[8ULL]"));
        assert_eq!(compiled.bindings()[1].bytes,36);
        let x:Vec<f32>=(0..elements).map(|i|i as f32*0.125).collect();
        for (shift,scale) in [(-0.25,0.5),(1.5,-2.)] {
            let mut parameters=vec![0f32;9];parameters[0]=shift;parameters[8]=scale;
            let observed=evaluate(kernel.clone(),elements as usize,[&x,&parameters,&[]]);
            assert_eq!(observed[0],x.iter().map(|&x|(x+shift)*scale).collect::<Vec<_>>());
        }
    }
    kernel.buffers[1].size=Some(8);assert!(compile(kernel,1).is_err());
}
#[test]fn unsigned_index_predicates_select_exact_bits_for_all_six_comparisons(){
    let spec=mask_programs::CausalMaskSpec {batch:2,queries:3,keys:7,query_start:(1u64<<40)+11,key_start:(1u64<<40)+9};
    let original=mask_programs::definition(spec).unwrap();
    let comparisons:[fn(BinaryOperator)->Comparison;6]=[Comparison::Equal,Comparison::NotEqual,Comparison::Lower,Comparison::LowerEqual,Comparison::Greater,Comparison::GreaterEqual];
    for (kind,comparison) in comparisons.into_iter().enumerate() {
        let mut kernel=original.clone();
        for instruction in &mut kernel.body.instructions {if let Operation::Comparison(Comparison::Greater(op))=&instruction.operation {instruction.operation=comparison(op.clone()).into();}}
        let output=evaluate(kernel.clone(),42,[&[],&[],&[]]);
        for i in 0..42 {let a=spec.key_start+(i%7) as u64;let b=spec.query_start+(i/7%3) as u64;
            let selected=match kind {0=>a==b,1=>a!=b,2=>a<b,3=>a<=b,4=>a>b,_=>a>=b};
            assert_eq!(output[0][i].to_bits(),if selected {0xff800000} else {0});
        }
        let source=compile(kernel,42).unwrap();assert!(!source.source().contains("static_cast<float>"));
    }
    let mut invalid=original;
    for instruction in &mut invalid.body.instructions {if let Operation::Comparison(Comparison::Greater(op))=&mut instruction.operation {
        op.lhs=Variable::constant(ConstantValue::Float(0.),f());
    }}
    assert!(compile(invalid,42).is_err());
}
#[test]fn fp32_predicates_select_bits_with_padded_capacity_and_live_mask() {
    let pairs=[(-0.,0.),(0.,-0.),(-1.,1.),(1.,-1.),(f32::INFINITY,f32::INFINITY),
        (f32::NEG_INFINITY,f32::INFINITY),(f32::from_bits(0x7fc12345),0.),(0.,f32::from_bits(0x7fc23456))];
    let comparisons:[fn(BinaryOperator)->Comparison;6]=[Comparison::Equal,Comparison::NotEqual,Comparison::Lower,
        Comparison::LowerEqual,Comparison::Greater,Comparison::GreaterEqual];
    for (kind,comparison) in comparisons.into_iter().enumerate() {for elements in [0usize,1,7,63,65,257] {
        let mut kernel=definition(MapProgram::Add);let lhs=kernel.body.instructions[0].out.unwrap();let rhs=kernel.body.instructions[1].out.unwrap();
        let mask=Variable::new(VariableKind::LocalConst{id:991},Type::scalar(ElemType::Bool));
        let copied=Variable::new(VariableKind::LocalConst{id:992},Type::scalar(ElemType::Bool));
        kernel.body.instructions[2]=Instruction::new(comparison(BinaryOperator {lhs,rhs}),mask);
        kernel.body.instructions.insert(3,Instruction::new(Operation::Copy(mask),copied));
        kernel.body.instructions.insert(4,Instruction::new(Arithmetic::Add(BinaryOperator {lhs,rhs}),v(993)));
        kernel.body.instructions.insert(5,Instruction::new(Operator::Select(Select {cond:copied,then:lhs,or_else:rhs}),v(994)));
        if let Operation::Operator(Operator::IndexAssign(store))=&mut kernel.body.instructions[6].operation {store.value=v(994);}
        let a:Vec<f32>=(0..elements).map(|i|pairs[i%pairs.len()].0).collect();let b:Vec<f32>=(0..elements).map(|i|pairs[i%pairs.len()].1).collect();
        let output=evaluate(kernel.clone(),elements,[&a,&b,&[]]);
        for lane in 0..elements {let x=a[lane];let y=b[lane];let cond=match kind {0=>x==y,1=>x!=y,2=>x<y,3=>x<=y,4=>x>y,_=>x>=y};
            assert_eq!(output[0][lane].to_bits(),(if cond {x} else {y}).to_bits());}
        let p=lower::lower(kernel.clone(),elements as u64).unwrap();let alloc=plan::allocate(&p,true).unwrap();
        let compare=p.nodes.iter().position(|n|matches!(n,Node::Compare(..))).unwrap();
        let select=p.nodes.iter().position(|n|matches!(n,Node::DataSelect(..))).unwrap();
        for i in compare+1..=select {assert_ne!(alloc.node_slots[compare],alloc.node_slots[i]);}
        for tile in [8u32,24,64,72,256,4096] {
            let mut options=opt(elements as u64);options.tile_elements=tile;
            let compiled=AscendCompiler.compile(kernel.clone(),&options,ExecutionMode::Checked,UIntKind::U64.into()).unwrap();
            assert_eq!(compiled.tile_elements(),tile);assert_eq!(compiled.ub_bytes() as usize,(p.bindings.len()+alloc.slots)*tile.div_ceil(64) as usize*64*4);
            assert!(compiled.source().contains("AscendC::Compare("));assert!(compiled.source().contains("AscendC::Select("));
            assert!(compiled.source().contains("(count + 63U) / 64U * 64U"));assert!(compiled.source().contains("VSEL_TENSOR_TENSOR_MODE"));
            options.ub_limit_bytes=compiled.ub_bytes()-1;
            assert!(AscendCompiler.compile(kernel.clone(),&options,ExecutionMode::Checked,UIntKind::U64.into()).is_err());
        }
        let mut branched=kernel.clone();let mut child=Scope::root(false);child.instructions.push(Instruction::no_out(Branch::Return));
        branched.body.instructions.insert(3,Instruction::no_out(Branch::If(Box::new(If {cond:mask,scope:child}))));
        assert!(compile(branched,elements as u64).is_err());
        let mut global_bool=kernel;global_bool.buffers[0].ty=Type::scalar(ElemType::Bool);
        assert!(compile(global_bool,elements as u64).is_err());
    }}
}
#[test]fn local_boolean_composition_retains_nan_truth_tables_and_index_precision() {
    let pairs=[(-1.,1.),(-1.,-1.),(1.,1.),(1.,-1.),(-0.,0.),(f32::NEG_INFINITY,f32::INFINITY),
        (f32::from_bits(0x7fc12345),1.),(-1.,f32::from_bits(0x7fc23456))];
    for elements in [0usize,1,7,63,65,257] {for kind in 0..8 {
        let mut kernel=definition(MapProgram::Add);let lhs=kernel.body.instructions[0].out.unwrap();let rhs=kernel.body.instructions[1].out.unwrap();
        let local=|id|Variable::new(VariableKind::LocalConst {id},Type::scalar(ElemType::Bool));
        let p=local(991);let q=local(992);let negated=local(993);let condition=local(994);let index=local(995);
        let zero=Variable::constant(ConstantValue::Float(0.),f());
        let truth=Variable::constant(ConstantValue::Bool(true),Type::scalar(ElemType::Bool));
        let falsity=Variable::constant(ConstantValue::Bool(false),Type::scalar(ElemType::Bool));
        kernel.body.instructions.truncate(2);
        kernel.body.instructions.push(Instruction::new(Comparison::Lower(BinaryOperator {lhs,rhs:zero}),p));
        kernel.body.instructions.push(Instruction::new(Comparison::Greater(BinaryOperator {lhs:rhs,rhs:zero}),q));
        kernel.body.instructions.push(Instruction::new(Operator::Not(UnaryOperator {input:p}),negated));
        let lane=Variable::builtin(Builtin::AbsolutePosX,UIntKind::U64.into());
        let offset=Variable::constant(ConstantValue::UInt((1u64<<40)+11),Type::new(UIntKind::U64.into()));
        let shifted=Variable::new(VariableKind::LocalConst {id:996},Type::new(UIntKind::U64.into()));
        kernel.body.instructions.push(Instruction::new(Arithmetic::Add(BinaryOperator {lhs:lane,rhs:offset}),shifted));
        let bound=Variable::constant(ConstantValue::UInt((1u64<<40)+11+(elements/2) as u64),Type::new(UIntKind::U64.into()));
        kernel.body.instructions.push(Instruction::new(Comparison::Lower(BinaryOperator {lhs:shifted,rhs:bound}),index));
        let operation=match kind {
            0=>Operator::Not(UnaryOperator {input:p}),1=>Operator::And(BinaryOperator {lhs:p,rhs:q}),
            2=>Operator::Or(BinaryOperator {lhs:p,rhs:q}),3=>Operator::Or(BinaryOperator {lhs:negated,rhs:q}),
            4=>Operator::And(BinaryOperator {lhs:p,rhs:index}),5=>Operator::Or(BinaryOperator {lhs:index,rhs:q}),
            6=>Operator::And(BinaryOperator {lhs:p,rhs:truth}),_=>Operator::Or(BinaryOperator {lhs:q,rhs:falsity}),
        };
        kernel.body.instructions.push(Instruction::new(operation,condition));
        kernel.body.instructions.push(Instruction::new(Operator::Select(Select {cond:condition,then:lhs,or_else:rhs}),v(997)));
        let output=Variable::new(VariableKind::GlobalOutputArray(2),f());
        kernel.body.instructions.push(Instruction::new(Operator::IndexAssign(IndexAssignOperator {index:lane,value:v(997),vector_size:0,unroll_factor:1}),output));
        let a:Vec<f32>=(0..elements).map(|i|pairs[i%pairs.len()].0).collect();let b:Vec<f32>=(0..elements).map(|i|pairs[i%pairs.len()].1).collect();
        let actual=evaluate(kernel.clone(),elements,[&a,&b,&[]]);
        for i in 0..elements {let p=a[i]<0.;let q=b[i]>0.;let index=i<elements/2;
            let select=match kind {0=>!p,1=>p&&q,2=>p||q,3=>!p||q,4=>p&&index,5=>index||q,6=>p,_=>q};
            assert_eq!(actual[0][i].to_bits(),(if select {a[i]} else {b[i]}).to_bits());
        }
        for tile in [8,64,256] {let mut options=opt(elements as u64);options.tile_elements=tile;
            let compiled=AscendCompiler.compile(kernel.clone(),&options,ExecutionMode::Checked,UIntKind::U64.into()).unwrap();
            assert!(compiled.source().contains("AscendC::Compare("));assert!(compiled.source().contains("AscendC::Select("));
            if matches!(kind,4|5) {assert!(compiled.source().contains("1099511627787ULL"));}
        }
        let mut child=Scope::root(false);child.instructions.push(Instruction::no_out(Branch::Return));
        kernel.body.instructions.push(Instruction::no_out(Branch::If(Box::new(If {cond:condition,scope:child}))));
        assert!(compile(kernel,elements as u64).is_err());
    }}
}
#[test]fn backward_equations_match_finite_differences(){let x=[-2.0,-0.3,0.2,1.7];let up=[1.2,0.7,-0.9,2.0];let dy=[0.5,-1.0,0.2,0.8];let grads=reference(MapProgram::SiluMulBackward,&x,&up,&dy);let h=0.001;for j in 0..4{let mut hi=x;let mut lo=x;hi[j]+=h;lo[j]-=h;let a=reference(MapProgram::SiluMul,&hi,&up,&[]);let b=reference(MapProgram::SiluMul,&lo,&up,&[]);let finite=(a[0][j]-b[0][j])/(2.0*h)*dy[j];assert!((grads[0][j]-finite).abs()<0.001);let expected=dy[j]*x[j]/(1.0+(-x[j]).exp());assert!((grads[1][j]-expected).abs()<1e-6);}}

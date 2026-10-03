//! Logical-tail SDPA and MHA/MQA/GQA with native BF16 matmuls and RUDA gradients.
use rust_ascend::{Ascend,Autodiff,nn,runtime::{AscendRuntime,RuntimeOptions},tensor::{DType,TensorData,api::Tensor}};
type AD=Autodiff<Ascend>;
fn bf16(value:f32)->f64 {
    let bits=value.to_bits();f32::from_bits(bits.wrapping_add(0x7fff+((bits>>16)&1))&0xffff0000) as f64
}
fn close(actual:&[f32],expected:&[f64],name:&str)->Result<(),Box<dyn std::error::Error>> {
    if actual.len()!=expected.len() {return Err(format!("{name}: length mismatch").into());}
    for (i,(&a,&b)) in actual.iter().zip(expected).enumerate() {
        if !a.is_finite() || !b.is_finite() || (a as f64-b).abs()>2e-3+3e-3*b.abs() {return Err(format!("{name}[{i}]: {a} != {b}").into());}
    }Ok(())
}
struct Reference {output:Vec<f64>,query:Vec<f64>,key:Vec<f64>,value:Vec<f64>,mask:Vec<f64>}
fn reference(shape:[usize;7],q:&[f32],k:&[f32],v:&[f32],dy:&[f32],mask:&[f32])->Reference {
    let [batch,heads,kv_heads,m,n,d,dv]=shape;let group=heads/kv_heads;
    let mut out=Reference {output:vec![0.;dy.len()],query:vec![0.;q.len()],key:vec![0.;k.len()],value:vec![0.;v.len()],mask:vec![0.;mask.len()]};
    for b in 0..batch {for h in 0..heads {let kv=h/group;
        let qi=|row,col|((b*heads+h)*m+row)*d+col;let ki=|row,col|((b*kv_heads+kv)*n+row)*d+col;
        let vi=|row,col|((b*kv_heads+kv)*n+row)*dv+col;let yi=|row,col|((b*heads+h)*m+row)*dv+col;
        let mi=|row,col|((b*heads+h)*m+row)*n+col;
        for row in 0..m {
            let scores:Vec<f64>=(0..n).map(|j|0.25*(0..d).map(|c|bf16(q[qi(row,c)])*bf16(k[ki(j,c)])).sum::<f64>()+mask[mi(row,j)] as f64).collect();
            let max=scores.iter().copied().fold(f64::NEG_INFINITY,f64::max);
            let exps:Vec<f64>=scores.iter().map(|s|(s-max).exp()).collect();let total=exps.iter().sum::<f64>();
            let p:Vec<f64>=exps.iter().map(|e|e/total).collect();
            let dp:Vec<f64>=(0..n).map(|j|(0..dv).map(|c|bf16(dy[yi(row,c)])*bf16(v[vi(j,c)])).sum()).collect();
            let dot=p.iter().zip(&dp).map(|(p,g)|p*g).sum::<f64>();
            for j in 0..n {
                let ds=p[j]*(dp[j]-dot);out.mask[mi(row,j)]=ds;
                let bp=bf16(p[j] as f32);let bds=bf16((0.25*ds) as f32);
                for c in 0..dv {out.output[yi(row,c)]+=bp*bf16(v[vi(j,c)]);out.value[vi(j,c)]+=bp*bf16(dy[yi(row,c)]);}
                for c in 0..d {out.query[qi(row,c)]+=bds*bf16(k[ki(j,c)]);out.key[ki(j,c)]+=bds*bf16(q[qi(row,c)]);}
            }
        }
    }}out
}
fn values(shape:[usize;7])->[Vec<f32>;4] {
    let [b,h,kv,m,n,d,dv]=shape;
    [(0..b*h*m*d).map(|i|(i%13) as f32/16.-0.375).collect(),
     (0..b*kv*n*d).map(|i|(i%17) as f32/16.-0.5).collect(),
     (0..b*kv*n*dv).map(|i|(i%11) as f32/8.-0.5).collect(),
     (0..b*h*m*dv).map(|i|(i%7) as f32/8.-0.25).collect()]
}
fn mask_values(shape:[usize;7],mode:u8)->Vec<f32> {
    let [b,h,_,m,n,_,_]=shape;
    (0..b*h*m*n).map(|i|match mode {
        0=>0.,1=>(i%5) as f32/32.-0.0625,
        _=>if i%n>n-m+i/n%m {f32::NEG_INFINITY} else {0.},
    }).collect()
}
fn main()->Result<(),Box<dyn std::error::Error>> {
    let toolkit=std::env::var_os("ASCEND_HOME_PATH").ok_or("set ASCEND_HOME_PATH")?;let mut options=RuntimeOptions::new(toolkit);
    if let Some(path)=std::env::var_os("RUDA_CANN_LIBRARY") {options.acl_library=path;}
    if let Some(paths)=std::env::var_os("RUDA_CANN_OPERATOR_LIBRARIES") {options.operator_libraries=std::env::split_paths(&paths).map(|p|p.into_os_string()).collect();}
    // SAFETY: this standalone executable is the sole ACL context owner.
    let device=unsafe {AscendRuntime::initialize_exclusive(options)?};
    let mut cases=0;
    for (b,m,n,d,dv) in [(1,1,1,1,1),(2,3,7,5,9),(1,17,33,7,3)] {for mode in 0..3 {
        let shape=[b,1,1,m,n,d,dv];let [q,k,v,dy]=values(shape);let mask_data=mask_values(shape,mode);
        let query=Tensor::<AD,3>::from_data(TensorData::new(q.clone(),[b,m,d]),(&device,DType::F32)).require_grad();
        let key=Tensor::<AD,3>::from_data(TensorData::new(k.clone(),[b,n,d]),(&device,DType::F32)).require_grad();
        let value=Tensor::<AD,3>::from_data(TensorData::new(v.clone(),[b,n,dv]),(&device,DType::F32)).require_grad();
        let mask=Tensor::<AD,3>::from_data(TensorData::new(mask_data.clone(),[b,m,n]),(&device,DType::F32)).require_grad();
        let output=if mode==2 {nn::causal_attention_padded_bf16_fp32(query.clone(),key.clone(),value.clone(),0.25,(1u64<<40)+(n-m) as u64,1u64<<40)?}
            else {nn::scaled_dot_product_attention_padded_bf16_fp32(query.clone(),key.clone(),value.clone(),0.25,if mode==1 {Some(mask.clone())} else {None})?};
        assert_eq!(output.dims(),[b,m,dv]);assert_eq!(output.dtype(),DType::F32);
        let doubled:Vec<f32>=dy.iter().map(|x|2.*x).collect();let expected=reference(shape,&q,&k,&v,&doubled,&mask_data);
        close(output.clone().into_data().as_slice::<f32>()?,&expected.output,"SDPA output")?;
        let upstream=Tensor::<AD,3>::from_data(TensorData::new(dy,[b,m,dv]),(&device,DType::F32));
        let gradients=((output.clone()+output)*upstream).backward();
        close(query.grad(&gradients).ok_or("missing SDPA dQ")?.into_data().as_slice::<f32>()?,&expected.query,"SDPA shared dQ")?;
        close(key.grad(&gradients).ok_or("missing SDPA dK")?.into_data().as_slice::<f32>()?,&expected.key,"SDPA shared dK")?;
        close(value.grad(&gradients).ok_or("missing SDPA dV")?.into_data().as_slice::<f32>()?,&expected.value,"SDPA shared dV")?;
        if mode==1 {close(mask.grad(&gradients).ok_or("missing SDPA mask gradient")?.into_data().as_slice::<f32>()?,&expected.mask,"SDPA dMask")?;}
        cases+=1;println!("ASCEND_PADDED_ATTENTION_CASE shape={shape:?} mode={mode} passed=true");
    }}
    for shape in [[1,4,4,1,1,1,1],[2,3,1,3,7,5,9],[1,6,2,17,33,7,3],[2,10,2,3,31,5,7],[1,9,3,3,4097,7,9]] {for mode in [1,2] {
        let [b,h,kv,m,n,d,dv]=shape;let [q,k,v,dy]=values(shape);let mask_data=mask_values(shape,mode);
        let query=Tensor::<AD,4>::from_data(TensorData::new(q.clone(),[b,h,m,d]),(&device,DType::F32)).require_grad();
        let key=Tensor::<AD,4>::from_data(TensorData::new(k.clone(),[b,kv,n,d]),(&device,DType::F32)).require_grad();
        let value=Tensor::<AD,4>::from_data(TensorData::new(v.clone(),[b,kv,n,dv]),(&device,DType::F32)).require_grad();
        let mask=Tensor::<AD,4>::from_data(TensorData::new(mask_data.clone(),[b,h,m,n]),(&device,DType::F32)).require_grad();
        let output=if mode==2 {nn::causal_grouped_query_attention_padded_bf16_fp32(query.clone(),key.clone(),value.clone(),0.25,(1u64<<40)+(n-m) as u64,1u64<<40)?}
            else {nn::grouped_query_attention_padded_bf16_fp32(query.clone(),key.clone(),value.clone(),0.25,Some(mask.clone()))?};
        assert_eq!(output.dims(),[b,h,m,dv]);assert_eq!(output.dtype(),DType::F32);
        let doubled:Vec<f32>=dy.iter().map(|x|2.*x).collect();let expected=reference(shape,&q,&k,&v,&doubled,&mask_data);
        close(output.clone().into_data().as_slice::<f32>()?,&expected.output,"GQA output")?;
        let upstream=Tensor::<AD,4>::from_data(TensorData::new(dy,[b,h,m,dv]),(&device,DType::F32));
        let gradients=((output.clone()+output)*upstream).backward();
        close(query.grad(&gradients).ok_or("missing GQA dQ")?.into_data().as_slice::<f32>()?,&expected.query,"GQA shared dQ")?;
        close(key.grad(&gradients).ok_or("missing GQA dK")?.into_data().as_slice::<f32>()?,&expected.key,"GQA shared dK")?;
        close(value.grad(&gradients).ok_or("missing GQA dV")?.into_data().as_slice::<f32>()?,&expected.value,"GQA shared dV")?;
        if mode==1 {close(mask.grad(&gradients).ok_or("missing GQA mask gradient")?.into_data().as_slice::<f32>()?,&expected.mask,"GQA dMask")?;}
        cases+=1;println!("ASCEND_PADDED_GQA_CASE shape={shape:?} mode={mode} passed=true");
    }}
    let q=Tensor::<Ascend,3>::from_data(TensorData::new(vec![1.],[1,1,1]),(&device,DType::F32));
    let k=Tensor::<Ascend,3>::from_data(TensorData::new(vec![1.;3],[1,3,1]),(&device,DType::F32));
    let v=Tensor::<Ascend,3>::from_data(TensorData::new(vec![2.;3],[1,3,1]),(&device,DType::F32));
    let output=nn::scaled_dot_product_attention_padded_bf16_fp32(q.clone(),k.clone(),v.clone(),0.25,None)?;
    close(output.into_data().as_slice::<f32>()?,&[2.],"plain logical-key attention")?;
    let mask=Tensor::<Ascend,3>::from_data(TensorData::new(vec![f32::NEG_INFINITY;3],[1,1,3]),(&device,DType::F32));
    let output=nn::scaled_dot_product_attention_padded_bf16_fp32(q,k,v,0.25,Some(mask))?;
    assert!(output.into_data().as_slice::<f32>()?.iter().all(|x|x.is_nan()));
    println!("ASCEND_PADDED_ATTENTION_DEVICE_OK gradient_cases={cases} plain_cases=2");Ok(())
}

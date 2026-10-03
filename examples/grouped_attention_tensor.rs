//! Native KV repetition/gradient reduction and composed MHA/MQA/GQA on RUDA tensors.
use rust_ascend::{Ascend,Autodiff,nn,runtime::{AscendRuntime,RuntimeOptions},tensor::{DType,TensorData,api::Tensor}};
type AD=Autodiff<Ascend>;
fn bf16(value:f32)->f64 {
    let bits=value.to_bits();f32::from_bits(bits.wrapping_add(0x7fff+((bits>>16)&1))&0xffff0000) as f64
}
fn close(actual:&[f32],expected:&[f64],name:&str)->Result<(),Box<dyn std::error::Error>> {
    if actual.len()!=expected.len() {return Err(format!("{name}: length mismatch").into());}
    for (i,(&a,&b)) in actual.iter().zip(expected).enumerate() {
        if !a.is_finite() || (a as f64-b).abs()>2e-3+3e-3*b.abs() {return Err(format!("{name}[{i}]: {a} != {b}").into());}
    }Ok(())
}
struct Reference {output:Vec<f64>,query:Vec<f64>,key:Vec<f64>,value:Vec<f64>,mask:Vec<f64>}
fn reference(shape:[usize;7],q:&[f32],k:&[f32],v:&[f32],dy:&[f32],mask:&[f32])->Reference {
    let [batch,heads,kv_heads,m,n,d,dv]=shape;let group=heads/kv_heads;
    let mut out=Reference {output:vec![0.;dy.len()],query:vec![0.;q.len()],key:vec![0.;k.len()],value:vec![0.;v.len()],mask:vec![0.;mask.len()]};
    for b in 0..batch {for h in 0..heads {let kv=h/group;
        let qi=|row,col|((b*heads+h)*m+row)*d+col;
        let ki=|row,col|((b*kv_heads+kv)*n+row)*d+col;
        let vi=|row,col|((b*kv_heads+kv)*n+row)*dv+col;
        let yi=|row,col|((b*heads+h)*m+row)*dv+col;
        let mi=|row,col|((b*heads+h)*m+row)*n+col;
        for row in 0..m {
            let scores:Vec<f64>=(0..n).map(|j|0.25*(0..d).map(|c|bf16(q[qi(row,c)])*bf16(k[ki(j,c)])).sum::<f64>()+mask[mi(row,j)] as f64).collect();
            let max=scores.iter().copied().fold(f64::NEG_INFINITY,f64::max);
            let exps:Vec<f64>=scores.iter().map(|s|(s-max).exp()).collect();let total=exps.iter().sum::<f64>();
            let probabilities:Vec<f64>=exps.iter().map(|e|e/total).collect();
            let dp:Vec<f64>=(0..n).map(|j|(0..dv).map(|c|bf16(dy[yi(row,c)])*bf16(v[vi(j,c)])).sum()).collect();
            let dot=probabilities.iter().zip(&dp).map(|(p,g)|p*g).sum::<f64>();
            for j in 0..n {
                let ds=probabilities[j]*(dp[j]-dot);out.mask[mi(row,j)]=ds;
                let p=bf16(probabilities[j] as f32);let ds=bf16((0.25*ds) as f32);
                for c in 0..dv {
                    out.output[yi(row,c)]+=p*bf16(v[vi(j,c)]);
                    out.value[vi(j,c)]+=p*bf16(dy[yi(row,c)]);
                }
                for c in 0..d {
                    out.query[qi(row,c)]+=ds*bf16(k[ki(j,c)]);
                    out.key[ki(j,c)]+=ds*bf16(q[qi(row,c)]);
                }
            }
        }
    }}out
}
fn main()->Result<(),Box<dyn std::error::Error>> {
    let toolkit=std::env::var_os("ASCEND_HOME_PATH").ok_or("set ASCEND_HOME_PATH")?;let mut options=RuntimeOptions::new(toolkit);
    if let Some(path)=std::env::var_os("RUDA_CANN_LIBRARY") {options.acl_library=path;}
    if let Some(paths)=std::env::var_os("RUDA_CANN_OPERATOR_LIBRARIES") {options.operator_libraries=std::env::split_paths(&paths).map(|p|p.into_os_string()).collect();}
    // SAFETY: this standalone executable is the sole ACL context owner.
    let device=unsafe {AscendRuntime::initialize_exclusive(options)?};
    for group in [1,2,3,5,9] {
        let (batch,kv_heads,n,d)=(2,3,5,7);let heads=kv_heads*group;
        let x:Vec<f32>=(0..batch*kv_heads*n*d).map(|i|i as f32/8.-1.).collect();
        let dy:Vec<f32>=(0..batch*heads*n*d).map(|i|(i%11) as f32/8.-0.25).collect();
        let input=Tensor::<AD,4>::from_data(TensorData::new(x.clone(),[batch,kv_heads,n,d]),(&device,DType::F32)).require_grad();
        let output=nn::repeat_kv_heads(input.clone(),heads)?;let values=output.clone().into_data();
        let mut expected=vec![0.;dy.len()];let mut dx=vec![0.;x.len()];
        for b in 0..batch {for h in 0..heads {for p in 0..n*d {
            let src=(b*kv_heads+h/group)*n*d+p;let dst=(b*heads+h)*n*d+p;
            expected[dst]=x[src] as f64;dx[src]+=dy[dst] as f64;
        }}}
        close(values.as_slice::<f32>()?,&expected,"repeat")?;
        let upstream=Tensor::<AD,4>::from_data(TensorData::new(dy,[batch,heads,n,d]),(&device,DType::F32));
        let gradients=(output.clone()*upstream.clone()+output*upstream).backward();
        for g in &mut dx {*g*=2.;}
        close(input.grad(&gradients).ok_or("missing repeat gradient")?.into_data().as_slice::<f32>()?,&dx,"repeat shared dX")?;
        println!("ASCEND_REPEAT_KV_TENSOR_CASE group={group} passed=true");
    }
    for shape in [[0,3,5,7],[2,3,0,7],[2,3,5,0]] {
        let input=Tensor::<AD,4>::from_data(TensorData::new(Vec::<f32>::new(),shape),(&device,DType::F32)).require_grad();
        let output=nn::repeat_kv_heads(input.clone(),9)?;assert!(output.clone().into_data().as_slice::<f32>()?.is_empty());
        assert!(input.grad(&output.backward()).ok_or("missing empty repeat gradient")?.into_data().as_slice::<f32>()?.is_empty());
    }
    for (kv_heads,group,n,causal) in [(3,1,32,false),(1,3,96,false),(3,2,32,false),(3,5,96,false),(3,3,96,true)] {
        let (batch,m,d,dv)=(2,32,16,16);let heads=kv_heads*group;
        let q:Vec<f32>=(0..batch*heads*m*d).map(|i|(i%13) as f32/16.-0.375).collect();
        let k:Vec<f32>=(0..batch*kv_heads*n*d).map(|i|(i%17) as f32/16.-0.5).collect();
        let v:Vec<f32>=(0..batch*kv_heads*n*dv).map(|i|(i%11) as f32/8.-0.5).collect();
        let dy:Vec<f32>=(0..batch*heads*m*dv).map(|i|(i%7) as f32/8.-0.25).collect();
        let query=Tensor::<AD,4>::from_data(TensorData::new(q.clone(),[batch,heads,m,d]),(&device,DType::F32)).require_grad();
        let key=Tensor::<AD,4>::from_data(TensorData::new(k.clone(),[batch,kv_heads,n,d]),(&device,DType::F32)).require_grad();
        let value=Tensor::<AD,4>::from_data(TensorData::new(v.clone(),[batch,kv_heads,n,dv]),(&device,DType::F32)).require_grad();
        let upstream=Tensor::<AD,4>::from_data(TensorData::new(dy.clone(),[batch,heads,m,dv]),(&device,DType::F32));
        let prefix=n-m;
        let mask_values:Vec<f32>=(0..batch*heads*m*n).map(|i|if causal {
            if i%n>prefix+i/n%m {f32::NEG_INFINITY} else {0.}
        } else {(i%5) as f32/32.-0.0625}).collect();
        let mask=Tensor::<AD,4>::from_data(TensorData::new(mask_values.clone(),[batch,heads,m,n]),(&device,DType::F32)).require_grad();
        let output=if causal {nn::causal_grouped_query_attention_bf16_fp32(query.clone(),key.clone(),value.clone(),0.25,(1u64<<40)+prefix as u64,1u64<<40)?}
            else {nn::grouped_query_attention_bf16_fp32(query.clone(),key.clone(),value.clone(),0.25,Some(mask.clone()))?};
        let reference=reference([batch,heads,kv_heads,m,n,d,dv],&q,&k,&v,&dy,&mask_values);
        close(output.clone().into_data().as_slice::<f32>()?,&reference.output,"GQA output")?;
        let gradients=(output*upstream).backward();
        close(query.grad(&gradients).ok_or("missing GQA dQ")?.into_data().as_slice::<f32>()?,&reference.query,"GQA dQ")?;
        close(key.grad(&gradients).ok_or("missing GQA dK")?.into_data().as_slice::<f32>()?,&reference.key,"GQA dK")?;
        close(value.grad(&gradients).ok_or("missing GQA dV")?.into_data().as_slice::<f32>()?,&reference.value,"GQA dV")?;
        if !causal {close(mask.grad(&gradients).ok_or("missing GQA mask gradient")?.into_data().as_slice::<f32>()?,&reference.mask,"GQA dMask")?;}
        println!("ASCEND_GROUPED_ATTENTION_TENSOR_CASE kv_heads={kv_heads} group={group} keys={n} causal={causal} passed=true");
    }
    println!("ASCEND_GROUPED_ATTENTION_TENSOR_DEVICE_OK repeat_cases=8 attention_cases=5");Ok(())
}

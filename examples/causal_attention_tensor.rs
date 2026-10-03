//! Exact device causal masks and position-aware composed attention with RUDA gradients.
use rust_ascend::{Ascend,Autodiff,nn,runtime::{AscendRuntime,RuntimeOptions},
    tensor::{DType,TensorData,api::Tensor}};
type AD=Autodiff<Ascend>;
fn bf16(value:f32)->f64 {
    let bits=value.to_bits();f32::from_bits((bits.wrapping_add(0x7fff+((bits>>16)&1)))&0xffff0000) as f64
}
fn close(actual:&[f32],expected:&[f64],name:&str)->Result<(),Box<dyn std::error::Error>> {
    if actual.len()!=expected.len() {return Err(format!("{name}: length mismatch").into());}
    for (i,(&a,&b)) in actual.iter().zip(expected).enumerate() {
        if !a.is_finite() || (a as f64-b).abs()>3e-4+3e-4*b.abs() {return Err(format!("{name}[{i}]: {a} != {b}").into());}
    }Ok(())
}
fn main()->Result<(),Box<dyn std::error::Error>> {
    let toolkit=std::env::var_os("ASCEND_HOME_PATH").ok_or("set ASCEND_HOME_PATH")?;
    let mut options=RuntimeOptions::new(toolkit);
    if let Some(path)=std::env::var_os("RUDA_CANN_LIBRARY") {options.acl_library=path;}
    if let Some(paths)=std::env::var_os("RUDA_CANN_OPERATOR_LIBRARIES") {
        options.operator_libraries=std::env::split_paths(&paths).map(|p|p.into_os_string()).collect();
    }
    // SAFETY: this standalone executable is the sole ACL context owner.
    let device=unsafe {AscendRuntime::initialize_exclusive(options)?};
    for (queries,keys,query_start,key_start) in [(3usize,7usize,0u64,0u64),(3,7,5,0),(3,7,1u64<<40,1u64<<40)] {
        let mask=nn::causal_mask::<Ascend>(&device,[2,queries,keys],query_start,key_start)?.into_data();
        let values=mask.as_slice::<f32>()?;
        for b in 0..2 {for row in 0..queries {for col in 0..keys {
            let bits=if key_start+col as u64>query_start+row as u64 {f32::NEG_INFINITY.to_bits()} else {0};
            assert_eq!(values[(b*queries+row)*keys+col].to_bits(),bits);
        }}}
    }
    let (batch,m,d,dv)=(2,32,16,16);
    for n in [32,96] {for cached in [false,true] {
        let query_start=if cached {(n-m) as u64} else {0};
        let query=Tensor::<AD,3>::from_data(TensorData::new(vec![0f32;batch*m*d],[batch,m,d]),(&device,DType::F32)).require_grad();
        let key=Tensor::<AD,3>::from_data(TensorData::new(vec![0f32;batch*n*d],[batch,n,d]),(&device,DType::F32)).require_grad();
        let v:Vec<f32>=(0..batch*n*dv).map(|i|(i%7) as f32/8.-0.375).collect();
        let dy:Vec<f32>=(0..batch*m*dv).map(|i|(i%3) as f32/4.-0.25).collect();
        let value=Tensor::<AD,3>::from_data(TensorData::new(v.clone(),[batch,n,dv]),(&device,DType::F32)).require_grad();
        let upstream=Tensor::<AD,3>::from_data(TensorData::new(dy.clone(),[batch,m,dv]),(&device,DType::F32));
        let output=nn::causal_attention_bf16_fp32(query.clone(),key.clone(),value.clone(),0.25,query_start,0)?;
        let observed=output.clone().into_data();let gradients=(output*upstream).backward();
        let mut y=vec![0.;batch*m*dv];let mut gv=vec![0.;v.len()];
        for b in 0..batch {for row in 0..m {
            let active=((query_start+row as u64+1) as usize).min(n);let probability=bf16(1f32/active as f32);
            for col in 0..active {for c in 0..dv {
                y[(b*m+row)*dv+c]+=probability*bf16(v[(b*n+col)*dv+c]);
                gv[(b*n+col)*dv+c]+=probability*bf16(dy[(b*m+row)*dv+c]);
            }}
        }}
        close(observed.as_slice::<f32>()?,&y,"causal output")?;
        close(query.grad(&gradients).ok_or("missing causal Q gradient")?.into_data().as_slice::<f32>()?,&vec![0.;batch*m*d],"dQ")?;
        close(key.grad(&gradients).ok_or("missing causal K gradient")?.into_data().as_slice::<f32>()?,&vec![0.;batch*n*d],"dK")?;
        close(value.grad(&gradients).ok_or("missing causal V gradient")?.into_data().as_slice::<f32>()?,&gv,"dV")?;
        println!("ASCEND_CAUSAL_ATTENTION_TENSOR_CASE keys={n} cached={cached} passed=true");
    }}
    println!("ASCEND_CAUSAL_ATTENTION_TENSOR_DEVICE_OK mask_cases=3 attention_cases=4");Ok(())
}

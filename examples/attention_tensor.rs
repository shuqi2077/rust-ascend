//! Composed native-matrix/FP32-Softmax attention and Q/K/V/mask gradients through RUDA.
use rust_ascend::{Ascend,Autodiff,nn,runtime::{AscendRuntime,RuntimeOptions},
    tensor::{DType,TensorData,api::Tensor}};
type AD=Autodiff<Ascend>;
fn close(actual:&[f32],expected:&[f64],name:&str)->Result<(),Box<dyn std::error::Error>> {
    if actual.len()!=expected.len() {return Err(format!("{name}: length mismatch").into());}
    for (i,(&a,&b)) in actual.iter().zip(expected).enumerate() {
        if !a.is_finite() || (a as f64-b).abs()>5e-4+5e-4*b.abs() {
            return Err(format!("{name}[{i}]: {a} != {b}").into());
        }
    }
    Ok(())
}
fn main()->Result<(),Box<dyn std::error::Error>> {
    let toolkit=std::env::var_os("ASCEND_HOME_PATH").ok_or("set ASCEND_HOME_PATH")?;
    let mut options=RuntimeOptions::new(toolkit);
    if let Some(path)=std::env::var_os("RUDA_CANN_LIBRARY") {options.acl_library=path;}
    if let Some(paths)=std::env::var_os("RUDA_CANN_OPERATOR_LIBRARIES") {
        options.operator_libraries=std::env::split_paths(&paths).map(|p|p.into_os_string()).collect();
    }
    // SAFETY: this executable exclusively owns the process-wide ACL context.
    let device=unsafe {AscendRuntime::initialize_exclusive(options)?};
    let (batch,m,d,dv)=(2,32,16,16);let scale=0.25f32;
    for n in [32,8192] {for masked in [false,true] {
        // Orthogonal nonzero Q/K give zero scores, but nonzero Q and K derivatives.
        // Power-of-two active widths give exactly representable BF16 probabilities.
        let mut q=vec![0f32;batch*m*d];let mut k=vec![0f32;batch*n*d];
        let mut v=vec![0f32;batch*n*dv];let mut dy=vec![0f32;batch*m*dv];
        let mut mask=vec![0f32;batch*m*n];let active=if masked {n/2} else {n};
        for b in 0..batch {
            for row in 0..m {
                q[(b*m+row)*d]=((row%3) as f32-1.)*0.5;
                dy[(b*m+row)*dv]=((row%3) as f32-1.)*0.25;
                if masked {for col in active..n {mask[(b*m+row)*n+col]=f32::NEG_INFINITY;}}
            }
            for col in 0..n {
                k[(b*n+col)*d+1]=((col%5) as f32-2.)*0.25;
                v[(b*n+col)*dv]=((col%5) as f32-2.)*0.125;
            }
        }
        let query=Tensor::<AD,3>::from_data(TensorData::new(q.clone(),[batch,m,d]),(&device,DType::F32)).require_grad();
        let key=Tensor::<AD,3>::from_data(TensorData::new(k.clone(),[batch,n,d]),(&device,DType::F32)).require_grad();
        let value=Tensor::<AD,3>::from_data(TensorData::new(v.clone(),[batch,n,dv]),(&device,DType::F32)).require_grad();
        let upstream=Tensor::<AD,3>::from_data(TensorData::new(dy.clone(),[batch,m,dv]),(&device,DType::F32));
        let additive=masked.then(||Tensor::<AD,3>::from_data(TensorData::new(mask.clone(),[batch,m,n]),(&device,DType::F32)).require_grad());
        let output=nn::scaled_dot_product_attention_bf16_fp32(query.clone(),key.clone(),value.clone(),scale,additive.clone())?;
        let observed=output.clone().into_data();let gradients=(output*upstream).backward();
        let mut y=vec![0.;batch*m*dv];let mut gq=vec![0.;q.len()];let mut gk=vec![0.;k.len()];
        let mut gv=vec![0.;v.len()];let mut gm=vec![0.;batch*m*n];
        let p=1./active as f64;
        for b in 0..batch {for row in 0..m {
            let mean_v=(0..active).map(|col|v[(b*n+col)*dv] as f64).sum::<f64>()*p;
            let gradient=dy[(b*m+row)*dv] as f64;y[(b*m+row)*dv]=mean_v;
            for col in 0..active {
                let ds=p*gradient*(v[(b*n+col)*dv] as f64-mean_v);
                gm[(b*m+row)*n+col]=ds;
                gq[(b*m+row)*d+1]+=ds*scale as f64*k[(b*n+col)*d+1] as f64;
                gk[(b*n+col)*d]+=ds*scale as f64*q[(b*m+row)*d] as f64;
                gv[(b*n+col)*dv]+=p*gradient;
            }
        }}
        close(observed.as_slice::<f32>()?,&y,"attention output")?;
        close(query.grad(&gradients).ok_or("missing Q gradient")?.into_data().as_slice::<f32>()?,&gq,"dQ")?;
        close(key.grad(&gradients).ok_or("missing K gradient")?.into_data().as_slice::<f32>()?,&gk,"dK")?;
        close(value.grad(&gradients).ok_or("missing V gradient")?.into_data().as_slice::<f32>()?,&gv,"dV")?;
        if let Some(mask)=additive {close(mask.grad(&gradients).ok_or("missing mask gradient")?.into_data().as_slice::<f32>()?,&gm,"dMask")?;}
        println!("ASCEND_ATTENTION_TENSOR_CASE n={n} masked={masked} passed=true");
    }}
    println!("ASCEND_ATTENTION_TENSOR_DEVICE_OK cases=4");
    Ok(())
}

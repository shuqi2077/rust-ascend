//! Explicit BF16 compute with FP32 parameters, outputs and RUDA autodiff gradients.
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
    for (m,n,k) in [(16,32,16),(32,48,32)] {
        // Inputs and the shared upstream derivative are exactly representable in BF16.
        let x:Vec<f32>=(0..m*k).map(|i|((i%17) as f32-8.)*0.125).collect();
        let w:Vec<f32>=(0..n*k).map(|i|((i%13) as f32-6.)*0.25).collect();
        let dy:Vec<f32>=(0..m*n).map(|i|((i%11) as f32-5.)*0.0625).collect();
        let input=Tensor::<AD,2>::from_data(TensorData::new(x.clone(),[m,k]),(&device,DType::F32)).require_grad();
        let weight=Tensor::<AD,2>::from_data(TensorData::new(w.clone(),[n,k]),(&device,DType::F32)).require_grad();
        let upstream=Tensor::<AD,2>::from_data(TensorData::new(dy,[m,n]),(&device,DType::F32));
        let output=nn::linear_bf16_fp32(input.clone(),weight.clone())?;
        if output.dims()!=[m,n] || output.dtype()!=DType::F32 {return Err("linear output metadata mismatch".into());}
        let observed=output.clone().into_data();
        let gradients=(output.clone()*upstream.clone()+output*upstream).backward();
        let dx=input.grad(&gradients).ok_or("missing input gradient")?;
        let dw=weight.grad(&gradients).ok_or("missing weight gradient")?;
        if dx.dims()!=[m,k] || dw.dims()!=[n,k] || dx.dtype()!=DType::F32 || dw.dtype()!=DType::F32 {
            return Err("linear gradient metadata mismatch".into());
        }
        let mut y=vec![0.;m*n];let mut gx=vec![0.;m*k];let mut gw=vec![0.;n*k];
        for row in 0..m {for col in 0..n {for q in 0..k {
            y[row*n+col]+=x[row*k+q] as f64*w[col*k+q] as f64;
            let gradient=2.*(((row*n+col)%11) as f64-5.)*0.0625;
            gx[row*k+q]+=gradient*w[col*k+q] as f64;
            gw[col*k+q]+=gradient*x[row*k+q] as f64;
        }}}
        close(observed.as_slice::<f32>()?,&y,"Y")?;
        close(dx.into_data().as_slice::<f32>()?,&gx,"shared dX")?;
        close(dw.into_data().as_slice::<f32>()?,&gw,"shared dW")?;
        let plain_x=Tensor::<Ascend,2>::from_data(TensorData::new(x.clone(),[m,k]),(&device,DType::F32));
        let plain_w=Tensor::<Ascend,2>::from_data(TensorData::new(w.clone(),[n,k]),(&device,DType::F32));
        close(nn::linear_bf16_fp32(plain_x,plain_w)?.into_data().as_slice::<f32>()?,&y,"plain Y")?;
        let untracked_x=Tensor::<AD,2>::from_data(TensorData::new(x,[m,k]),(&device,DType::F32));
        let untracked_w=Tensor::<AD,2>::from_data(TensorData::new(w,[n,k]),(&device,DType::F32));
        close(nn::linear_bf16_fp32(untracked_x,untracked_w)?.into_data().as_slice::<f32>()?,&y,"untracked Y")?;
        println!("ASCEND_LINEAR_TENSOR_CASE m={m} n={n} k={k} passed=true");
    }
    println!("ASCEND_LINEAR_TENSOR_DEVICE_OK cases=2");
    Ok(())
}

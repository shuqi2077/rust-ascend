//! RUDA's original fused FP32 AdamW storage kernel executed on Ascend.
use rust_ascend::{core::tensor::{DType, Shape, Strides}, optim::{AdamWStorageStep, adamw_step},
    runtime::{AscendRuntime, ComputeClient, RuntimeOptions, TensorBuffer, portable::backend::Runtime}};

fn upload(client:&ComputeClient<AscendRuntime>, values:&[f32])->TensorBuffer {
    let bytes:Vec<_>=values.iter().flat_map(|v|v.to_ne_bytes()).collect();
    TensorBuffer {handle:client.create_from_slice(&bytes),shape:Shape::new([1,values.len()]),
        strides:Strides::from(vec![values.len(),1]),dtype:DType::F32}
}
fn read(client:&ComputeClient<AscendRuntime>, tensor:&TensorBuffer)->Result<Vec<f32>,Box<dyn std::error::Error>> {
    Ok(client.read_one(tensor.handle.clone())?.chunks_exact(4)
        .map(|b|f32::from_ne_bytes(b.try_into().unwrap())).collect())
}
fn close(actual:&[f32], expected:&[f64], name:&str)->Result<(),Box<dyn std::error::Error>> {
    if actual.len()!=expected.len() {return Err(format!("{name}: length mismatch").into());}
    for (i,(&a,&b)) in actual.iter().zip(expected).enumerate() {
        if !a.is_finite() || (a as f64-b).abs()>2e-5+2e-5*b.abs() {
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
    // SAFETY: this standalone executable owns the process-wide ACL lifecycle.
    let device=unsafe {AscendRuntime::initialize_exclusive(options)?};
    let client=AscendRuntime::client(&device);
    for n in [0,1,65,1025] {
        let initial:Vec<f32>=(0..n).map(|i|(i%23) as f32/8.-1.).collect();
        let parameter=upload(&client,&initial);let first=upload(&client,&vec![0.;n]);let second=upload(&client,&vec![0.;n]);
        let mut p:Vec<f64>=initial.iter().map(|&v|v as f64).collect();let mut m=vec![0.;n];let mut v=vec![0.;n];
        for update in 1..=5 {
            let beta1=0.9f32;let beta2=0.95f32;
            let step=AdamWStorageStep {learning_rate:0.01,beta1,beta2,epsilon:1e-5,weight_decay:0.02,
                correction1:(1.-(beta1 as f64).powi(update)) as f32,
                correction2:(1.-(beta2 as f64).powi(update)) as f32,
                inverse_gradient_scale:0.125,clip_multiplier:0.75};
            let gradient:Vec<f32>=(0..n).map(|i|((i%17) as f32-8.)*0.0625+update as f32*0.001).collect();
            let grad=upload(&client,&gradient);
            adamw_step(&client,&parameter,&grad,&first,&second,step)?;
            for i in 0..n {
                let g=gradient[i] as f64*step.inverse_gradient_scale as f64*step.clip_multiplier as f64;
                m[i]=beta1 as f64*m[i]+(1.-beta1 as f64)*g;
                v[i]=beta2 as f64*v[i]+(1.-beta2 as f64)*g*g;
                p[i]=p[i]*(1.-step.learning_rate as f64*step.weight_decay as f64)
                    -step.learning_rate as f64*(m[i]/step.correction1 as f64)
                    /((v[i]/step.correction2 as f64).sqrt()+step.epsilon as f64);
            }
            close(&read(&client,&parameter)?,&p,"parameter")?;
            close(&read(&client,&first)?,&m,"first moment")?;
            close(&read(&client,&second)?,&v,"second moment")?;
            if read(&client,&grad)?!=gradient {return Err("AdamW modified the gradient buffer".into());}
        }
        println!("ASCEND_ADAMW_STORAGE_CASE elements={n} updates=5 passed=true");
    }
    client.flush()?;
    println!("ASCEND_ADAMW_STORAGE_DEVICE_OK cases=4");
    Ok(())
}

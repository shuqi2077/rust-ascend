//! Native SwiGLU graph: the zero gate has a nonzero derivative and trainable gate weights.
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
    let device=unsafe {AscendRuntime::initialize_exclusive(options)?};let width=16;let elements=width*width;
    let x:Vec<f32>=(0..elements).map(|i|((i%7) as f32-3.)*0.125).collect();
    let dy:Vec<f32>=(0..elements).map(|i|((i%7) as f32-3.)*0.0625).collect();
    let mut u=vec![0f32;elements];let mut d=u.clone();
    for i in 0..width {u[i*width+i]=0.5;d[i*width+i]=0.25;}
    let tensor=|values:Vec<f32>|Tensor::<AD,2>::from_data(TensorData::new(values,[width,width]),(&device,DType::F32)).require_grad();
    let input=tensor(x.clone());let gate=tensor(vec![0.;elements]);let up=tensor(u);let down=tensor(d);
    let upstream=Tensor::<AD,2>::from_data(TensorData::new(dy.clone(),[width,width]),(&device,DType::F32));
    let output=nn::swiglu_bf16_fp32(input.clone(),gate.clone(),up.clone(),down.clone())?;
    close(output.clone().into_data().as_slice::<f32>()?,&vec![0.;elements],"SwiGLU Y")?;
    let gradients=(output*upstream).backward();let mut expected=vec![0.;elements];
    for row in 0..width {for col in 0..width {for sample in 0..width {
        expected[row*width+col]+=0.0625*dy[sample*width+row] as f64*x[sample*width+row] as f64*x[sample*width+col] as f64;
    }}}
    close(gate.grad(&gradients).ok_or("missing gate gradient")?.into_data().as_slice::<f32>()?,&expected,"SwiGLU dGate")?;
    for (name,tensor) in [("dX",input),("dUp",up),("dDown",down)] {
        close(tensor.grad(&gradients).ok_or("missing SwiGLU gradient")?.into_data().as_slice::<f32>()?,&vec![0.;elements],name)?;
    }
    println!("ASCEND_SWIGLU_TENSOR_DEVICE_OK input_and_all_weight_gradients=true");
    Ok(())
}

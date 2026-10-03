//! Native last-axis reductions and their RUDA autodiff gradients on Ascend.
use rust_ascend::{Ascend, Autodiff, nn, runtime::{AscendRuntime,RuntimeOptions},
    tensor::{DType,TensorData,api::Tensor}};
type AD=Autodiff<Ascend>;

fn close(actual:&[f32],expected:&[f64],name:&str)->Result<(),Box<dyn std::error::Error>> {
    if actual.len()!=expected.len() {return Err(format!("{name}: length mismatch").into());}
    for (i,(&a,&b)) in actual.iter().zip(expected).enumerate() {
        if !a.is_finite() || (a as f64-b).abs()>5e-5+5e-5*b.abs() {
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
    // SAFETY: this standalone executable exclusively owns ACL initialization.
    let device=unsafe {AscendRuntime::initialize_exclusive(options)?};
    let cases=[(0,32),(1,32),(3,96),(7,256),(33,4096),(0,4128),(1,4128),(3,8192),(2,8224)];
    for (rows,width) in cases {
        let x:Vec<f32>=(0..rows*width).map(|i|(i%23) as f32/8.-1.).collect();
        let up:Vec<f32>=(0..rows).map(|r|r as f32*0.125-0.5).collect();
        let input=Tensor::<AD,3>::from_data(TensorData::new(x.clone(),[1,rows,width]),(&device,DType::F32)).require_grad();
        let upstream=Tensor::<AD,3>::from_data(TensorData::new(up.clone(),[1,rows,1]),(&device,DType::F32));
        let sum=nn::sum_last(input.clone())?;let mean=nn::mean_last(input.clone())?;
        assert_eq!(sum.dims(),[1,rows,1]);assert_eq!(mean.dims(),[1,rows,1]);
        let expected_sum:Vec<f64>=x.chunks(width).map(|row|row.iter().map(|&v|v as f64).sum()).collect();
        let expected_mean:Vec<f64>=expected_sum.iter().map(|&v|v/width as f64).collect();
        close(sum.clone().into_data().as_slice::<f32>()?,&expected_sum,"sum")?;
        close(mean.clone().into_data().as_slice::<f32>()?,&expected_mean,"mean")?;
        let gradients=(sum*upstream.clone()+mean*upstream).backward();
        let dx=input.grad(&gradients).ok_or("missing reduction gradient")?.into_data();
        let expected_dx:Vec<f64>=up.iter().flat_map(|&g|vec![g as f64*(1.+1./width as f64);width]).collect();
        close(dx.as_slice::<f32>()?,&expected_dx,"shared sum/mean dX")?;
        let plain=Tensor::<Ascend,3>::from_data(TensorData::new(x.clone(),[1,rows,width]),(&device,DType::F32));
        close(nn::sum_last(plain)?.into_data().as_slice::<f32>()?,&expected_sum,"plain sum")?;
        let untracked=Tensor::<AD,3>::from_data(TensorData::new(x,[1,rows,width]),(&device,DType::F32));
        close(nn::mean_last(untracked)?.into_data().as_slice::<f32>()?,&expected_mean,"untracked mean")?;
        println!("ASCEND_REDUCTION_TENSOR_CASE rows={rows} width={width} passed=true");
    }
    println!("ASCEND_REDUCTION_TENSOR_DEVICE_OK cases={}",cases.len());
    Ok(())
}

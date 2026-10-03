//! Explicit native ReLU/Clamp with local FP32 predicates and RUDA graph gradients.
use rust_ascend::{Ascend,Autodiff,nn,runtime::{AscendRuntime,RuntimeOptions,AscendDevice},
    tensor::{DType,TensorData,api::Tensor}};
type AD=Autodiff<Ascend>;
fn expected(x:f32,activation:nn::PiecewiseActivation)->(f32,bool) {
    match activation {
        nn::PiecewiseActivation::Relu=>(if x<=0. {0.} else {x},x<=0.),
        nn::PiecewiseActivation::Clamp {min,max}=>{
            let hi=x>max;let upper=if hi {max} else {x};let lo=upper<min;
            (if lo {min} else {upper},hi||lo)
        },
    }
}
fn apply<B:nn::PiecewiseBackend,const D:usize>(input:Tensor<B,D>,activation:nn::PiecewiseActivation)->Result<Tensor<B,D>,rust_ascend::driver::CannError> {
    match activation {nn::PiecewiseActivation::Relu=>nn::relu(input),nn::PiecewiseActivation::Clamp {min,max}=>nn::clamp(input,min,max)}
}
fn bits(actual:&[f32],expected:&[f32],name:&str)->Result<(),Box<dyn std::error::Error>> {
    if actual.len()!=expected.len() {return Err(format!("{name}: length mismatch").into());}
    for (i,(&a,&b)) in actual.iter().zip(expected).enumerate() {
        if a.to_bits()!=b.to_bits() {return Err(format!("{name}[{i}]: {:#x} != {:#x}",a.to_bits(),b.to_bits()).into());}
    }Ok(())
}
fn modes()->[nn::PiecewiseActivation;7] {
    [nn::PiecewiseActivation::Relu,nn::PiecewiseActivation::Clamp {min:-1.,max:1.},
        nn::PiecewiseActivation::Clamp {min:2.,max:-1.},
        nn::PiecewiseActivation::Clamp {min:f32::NEG_INFINITY,max:f32::INFINITY},
        nn::PiecewiseActivation::Clamp {min:f32::NAN,max:1.},nn::PiecewiseActivation::Clamp {min:-1.,max:f32::NAN},
        nn::PiecewiseActivation::Clamp {min:-0.,max:0.}]
}
fn case<const D:usize>(device:&AscendDevice,shape:[usize;D])->Result<(),Box<dyn std::error::Error>> {
    let values=[-2.,-1.,-0.,0.,1.,2.,0.5];let count=shape.iter().product();
    let x:Vec<f32>=(0..count).map(|i|values[i%values.len()]).collect();
    let dy:Vec<f32>=(0..count).map(|i|if i%3==0 {-0.} else {(i%11) as f32/8.-0.375}).collect();
    for activation in modes() {
        let input=Tensor::<AD,D>::from_data(TensorData::new(x.clone(),shape),(device,DType::F32)).require_grad();
        let output=apply(input.clone(),activation)?;assert_eq!(output.dims(),shape);assert_eq!(output.dtype(),DType::F32);
        let reference:Vec<f32>=x.iter().map(|&x|expected(x,activation).0).collect();
        bits(output.clone().into_data().as_slice::<f32>()?,&reference,"piecewise output")?;
        let upstream=Tensor::<AD,D>::from_data(TensorData::new(dy.clone(),shape),(device,DType::F32));
        let gradients=((output.clone()+output)*upstream).backward();
        let reference:Vec<f32>=x.iter().zip(&dy).map(|(&x,&g)|if expected(x,activation).1 {0.} else {2.*g}).collect();
        let dx=input.grad(&gradients).ok_or("missing piecewise gradient")?;assert_eq!(dx.dims(),shape);
        // Arithmetic used for shared graph accumulation may normalize signed zero, unlike Select itself.
        let actual=dx.into_data();let actual=actual.as_slice::<f32>()?;
        if !actual.iter().zip(&reference).all(|(&a,&b)|a==b) {return Err("piecewise shared gradient mismatch".into());}
        println!("ASCEND_PIECEWISE_TENSOR_CASE shape={shape:?} activation={activation:?} passed=true");
    }Ok(())
}
fn main()->Result<(),Box<dyn std::error::Error>> {
    let toolkit=std::env::var_os("ASCEND_HOME_PATH").ok_or("set ASCEND_HOME_PATH")?;let mut options=RuntimeOptions::new(toolkit);
    if let Some(path)=std::env::var_os("RUDA_CANN_LIBRARY") {options.acl_library=path;}
    if let Some(paths)=std::env::var_os("RUDA_CANN_OPERATOR_LIBRARIES") {options.operator_libraries=std::env::split_paths(&paths).map(|p|p.into_os_string()).collect();}
    // SAFETY: this standalone executable is the sole ACL context owner.
    let device=unsafe {AscendRuntime::initialize_exclusive(options)?};
    case(&device,[7])?;case(&device,[2,3,33])?;case(&device,[1,1,1,1,1,1,1,65])?;case(&device,[0,7])?;case(&device,[3,0])?;
    let values=[f32::NEG_INFINITY,-2.,-1.,-0.,0.,1.,2.,f32::INFINITY,f32::from_bits(0x7fc12345)];
    for activation in modes() {
        let input=Tensor::<Ascend,1>::from_data(TensorData::new(values.to_vec(),[values.len()]),(&device,DType::F32));
        let reference:Vec<f32>=values.iter().map(|&x|expected(x,activation).0).collect();
        bits(apply(input,activation)?.into_data().as_slice::<f32>()?,&reference,"nonfinite piecewise output")?;
    }
    println!("ASCEND_PIECEWISE_TENSOR_DEVICE_OK gradient_cases=35 plain_cases=7");Ok(())
}

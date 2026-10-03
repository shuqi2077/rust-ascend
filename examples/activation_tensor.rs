//! Existing RUDA Erf/Tanh/integer powers and exact/approximate GELU on Ascend.
use rust_ascend::{Ascend,Autodiff,runtime::{AscendRuntime,RuntimeOptions,AscendDevice},
    tensor::{DType,TensorData,api::Tensor}};
type AD=Autodiff<Ascend>;
#[cfg_attr(unix,link(name="m"))]
unsafe extern "C" {#[link_name="erf"] fn c_erf(x:f64)->f64;}
fn erf(x:f32)->f32 {unsafe {c_erf(x as f64) as f32}}
fn close(actual:&[f32],expected:&[f32],name:&str)->Result<(),Box<dyn std::error::Error>> {
    if actual.len()!=expected.len() {return Err(format!("{name}: length mismatch").into());}
    for (i,(&a,&b)) in actual.iter().zip(expected).enumerate() {
        if !(a==b || a.is_nan() && b.is_nan() || a.is_finite() && b.is_finite() && (a-b).abs()<=3e-5*(1.+b.abs())) {
            return Err(format!("{name}[{i}]: {a} != {b}").into());
        }
    }Ok(())
}
fn case<const D:usize>(device:&AscendDevice,shape:[usize;D])->Result<(),Box<dyn std::error::Error>> {
    let count=shape.iter().product();
    let x:Vec<f32>=(0..count).map(|i|(i%29) as f32/4.-3.5).collect();
    let dy:Vec<f32>=(0..count).map(|i|(i%11) as f32/8.-0.375).collect();
    for mode in 0..5 {
        let input=Tensor::<AD,D>::from_data(TensorData::new(x.clone(),shape),(device,DType::F32)).require_grad();
        let output=match mode {0=>input.clone().erf(),1=>input.clone().tanh(),2=>input.clone().powf_scalar(3.),
            3=>ruda_nn::Gelu::new().forward(input.clone()),_=>ruda_nn::Gelu::new_approximate().forward(input.clone())};
        assert_eq!(output.dims(),shape);assert_eq!(output.dtype(),DType::F32);
        let expected:Vec<f32>=x.iter().map(|&x|match mode {0=>erf(x),1=>x.tanh(),2=>x*x*x,
            3=>x*(1.+erf(x/core::f32::consts::SQRT_2))/2.,
            _=>0.5*x*(1.+((2./core::f32::consts::PI).sqrt()*(x+0.044715*x*x*x)).tanh())}).collect();
        close(output.clone().into_data().as_slice::<f32>()?,&expected,"activation output")?;
        let upstream=Tensor::<AD,D>::from_data(TensorData::new(dy.clone(),shape),(device,DType::F32));
        let gradients=((output.clone()+output)*upstream).backward();
        let expected:Vec<f32>=x.iter().zip(&dy).map(|(&x,&g)|2.*g*match mode {
            0=>2./core::f32::consts::PI.sqrt()*(-x*x).exp(),1=>1.-x.tanh().powi(2),2=>3.*x*x,
            // Preserve the pinned RUDA exact-GELU backward definition.
            3=>{let x3=x*x*x;let t=(0.0356774*x3+0.797885*x).tanh();
                0.5*t+(0.0535161*x3+0.398942*x)*(1.-t*t)+0.5},
            _=>{let c=(2./core::f32::consts::PI).sqrt();let t=(c*(x+0.044715*x*x*x)).tanh();
                0.5*(1.+t)+0.5*x*(1.-t*t)*c*(1.+3.*0.044715*x*x)}
        }).collect();
        let dx=input.grad(&gradients).ok_or("missing activation gradient")?;assert_eq!(dx.dims(),shape);
        close(dx.into_data().as_slice::<f32>()?,&expected,"activation shared gradient")?;
        println!("ASCEND_ACTIVATION_TENSOR_CASE shape={shape:?} mode={mode} passed=true");
    }Ok(())
}
fn main()->Result<(),Box<dyn std::error::Error>> {
    let toolkit=std::env::var_os("ASCEND_HOME_PATH").ok_or("set ASCEND_HOME_PATH")?;let mut options=RuntimeOptions::new(toolkit);
    if let Some(path)=std::env::var_os("RUDA_CANN_LIBRARY") {options.acl_library=path;}
    if let Some(paths)=std::env::var_os("RUDA_CANN_OPERATOR_LIBRARIES") {options.operator_libraries=std::env::split_paths(&paths).map(|p|p.into_os_string()).collect();}
    // SAFETY: this standalone executable is the sole ACL context owner.
    let device=unsafe {AscendRuntime::initialize_exclusive(options)?};
    case(&device,[7])?;case(&device,[2,3,33])?;case(&device,[3,4097])?;case(&device,[0,7])?;
    let values=[-0.,0.,f32::INFINITY,f32::NEG_INFINITY,f32::NAN,-1.,1.];
    let input=Tensor::<Ascend,1>::from_data(TensorData::new(values.to_vec(),[7]),(&device,DType::F32));
    close(input.clone().erf().into_data().as_slice::<f32>()?,&values.map(erf),"nonfinite Erf")?;
    close(input.tanh().into_data().as_slice::<f32>()?,&values.map(f32::tanh),"nonfinite Tanh")?;
    let values:[f32;5]=[-2.,-0.5,0.5,2.,1e20];
    let input=Tensor::<Ascend,1>::from_data(TensorData::new(values.to_vec(),[5]),(&device,DType::F32));
    close(input.powf_scalar(-3.).into_data().as_slice::<f32>()?,&values.map(|x|x.recip().powi(3)),"negative integer power")?;
    println!("ASCEND_ACTIVATION_TENSOR_DEVICE_OK gradient_cases=20 plain_cases=3");Ok(())
}

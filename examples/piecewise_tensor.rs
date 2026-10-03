//! Explicit native piecewise activations with local FP32 predicates and RUDA graph gradients.
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
        nn::PiecewiseActivation::LeakyRelu {negative_slope}=>(if x<0. {x*negative_slope} else {x},false),
        nn::PiecewiseActivation::HardSigmoid {alpha,beta}=>{
            let value=x*alpha+beta;let hi=value>1.;let upper=if hi {1.} else {value};let lo=upper<0.;
            (if lo {0.} else {upper},hi||lo)
        },
        _=>unreachable!("exponential activation references live in exp_piecewise_tensor"),
    }
}
fn apply<B:nn::PiecewiseBackend,const D:usize>(input:Tensor<B,D>,activation:nn::PiecewiseActivation)->Result<Tensor<B,D>,rust_ascend::driver::CannError> {
    match activation {nn::PiecewiseActivation::Relu=>nn::relu(input),nn::PiecewiseActivation::Clamp {min,max}=>nn::clamp(input,min,max),
        nn::PiecewiseActivation::LeakyRelu {negative_slope}=>nn::leaky_relu(input,negative_slope),
        nn::PiecewiseActivation::HardSigmoid {alpha,beta}=>nn::hard_sigmoid(input,alpha,beta),
        nn::PiecewiseActivation::Elu {alpha}=>nn::elu(input,alpha),nn::PiecewiseActivation::Celu {alpha}=>nn::celu(input,alpha),
        nn::PiecewiseActivation::Selu=>nn::selu(input)}
}
fn bits(actual:&[f32],expected:&[f32],name:&str)->Result<(),Box<dyn std::error::Error>> {
    if actual.len()!=expected.len() {return Err(format!("{name}: length mismatch").into());}
    for (i,(&a,&b)) in actual.iter().zip(expected).enumerate() {
        if a.to_bits()!=b.to_bits() {return Err(format!("{name}[{i}]: {:#x} != {:#x}",a.to_bits(),b.to_bits()).into());}
    }Ok(())
}
fn modes()->[nn::PiecewiseActivation;13] {
    [nn::PiecewiseActivation::Relu,nn::PiecewiseActivation::Clamp {min:-1.,max:1.},
        nn::PiecewiseActivation::Clamp {min:2.,max:-1.},
        nn::PiecewiseActivation::Clamp {min:f32::NEG_INFINITY,max:f32::INFINITY},
        nn::PiecewiseActivation::Clamp {min:f32::NAN,max:1.},nn::PiecewiseActivation::Clamp {min:-1.,max:f32::NAN},
        nn::PiecewiseActivation::Clamp {min:-0.,max:0.},nn::PiecewiseActivation::LeakyRelu {negative_slope:0.1},
        nn::PiecewiseActivation::LeakyRelu {negative_slope:0.},nn::PiecewiseActivation::LeakyRelu {negative_slope:-0.5},
        nn::PiecewiseActivation::HardSigmoid {alpha:0.25,beta:0.5},nn::PiecewiseActivation::HardSigmoid {alpha:-0.25,beta:0.5},
        nn::PiecewiseActivation::HardSigmoid {alpha:0.,beta:0.5}]
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
        let reference:Vec<f32>=x.iter().zip(&dy).map(|(&x,&g)|match activation {
            nn::PiecewiseActivation::LeakyRelu {negative_slope}=>{
                let direct=if x<0. {0.} else {2.*g};let negative=if x<0. {2.*g} else {0.};negative*negative_slope+direct
            },nn::PiecewiseActivation::HardSigmoid {alpha,..}=>(if expected(x,activation).1 {0.} else {2.*g})*alpha,
            _=>if expected(x,activation).1 {0.} else {2.*g},
        }).collect();
        let dx=input.grad(&gradients).ok_or("missing piecewise gradient")?;assert_eq!(dx.dims(),shape);
        // Arithmetic used for shared graph accumulation may normalize signed zero, unlike Select itself.
        let actual=dx.into_data();let actual=actual.as_slice::<f32>()?;
        if !actual.iter().zip(&reference).all(|(&a,&b)|a==b) {return Err("piecewise shared gradient mismatch".into());}
        println!("ASCEND_PIECEWISE_TENSOR_CASE shape={shape:?} activation={activation:?} passed=true");
    }
    let input=Tensor::<AD,D>::from_data(TensorData::new(x.clone(),shape),(device,DType::F32)).require_grad();
    let output=nn::hard_swish(input.clone())?;assert_eq!(output.dims(),shape);assert_eq!(output.dtype(),DType::F32);
    let activation=nn::PiecewiseActivation::HardSigmoid {alpha:1f32/6.,beta:0.5};
    let reference:Vec<f32>=x.iter().map(|&x|x*expected(x,activation).0).collect();
    bits(output.clone().into_data().as_slice::<f32>()?,&reference,"hard swish output")?;
    let upstream=Tensor::<AD,D>::from_data(TensorData::new(dy.clone(),shape),(device,DType::F32));
    let gradients=((output.clone()+output)*upstream).backward();
    let reference:Vec<f32>=x.iter().zip(&dy).map(|(&x,&g)|{
        let (sigmoid,clipped)=expected(x,activation);let direct=2.*g*sigmoid;
        let through_sigmoid=(if clipped {0.} else {2.*g*x})*(1f32/6.);through_sigmoid+direct
    }).collect();
    let dx=input.grad(&gradients).ok_or("missing hard swish gradient")?;assert_eq!(dx.dims(),shape);
    let actual=dx.into_data();let actual=actual.as_slice::<f32>()?;
    assert!(actual.iter().zip(&reference).all(|(&a,&b)|a==b || (a-b).abs()<=2e-6*(1.+b.abs())));
    println!("ASCEND_HARD_SWISH_TENSOR_CASE shape={shape:?} passed=true");Ok(())
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
        let output=apply(input,activation)?.into_data();let actual=output.as_slice::<f32>()?;
        if matches!(activation,nn::PiecewiseActivation::LeakyRelu {..}|nn::PiecewiseActivation::HardSigmoid {..}) {
            assert!(actual.iter().zip(&reference).all(|(&a,&b)|a==b || a.is_nan() && b.is_nan()));
        } else {bits(actual,&reference,"nonfinite piecewise output")?;}
    }
    println!("ASCEND_PIECEWISE_TENSOR_DEVICE_OK gradient_cases=70 plain_cases=13");Ok(())
}

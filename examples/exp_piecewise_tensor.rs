//! Native FP32 ELU/CELU/SELU with RUDA's branch boundaries and ordered backward arithmetic.
use rust_ascend::{Ascend,Autodiff,nn,runtime::{AscendRuntime,RuntimeOptions,AscendDevice},
    tensor::{DType,TensorData,api::Tensor}};
type AD=Autodiff<Ascend>;
fn modes()->Vec<nn::PiecewiseActivation> {
    let mut modes=vec![nn::PiecewiseActivation::Selu];
    for alpha in [0.1f32,1.,-0.5,0.,f32::INFINITY,f32::NAN] {
        modes.push(nn::PiecewiseActivation::Elu {alpha});modes.push(nn::PiecewiseActivation::Celu {alpha});
    }modes
}
fn reference(x:f32,g:f32,activation:nn::PiecewiseActivation)->(f32,f32) {
    const ALPHA:f64=1.6732632423543772848170429916717;const GAMMA:f64=1.0507009873554804934193349852946;
    match activation {
        nn::PiecewiseActivation::Elu {alpha}=>{
            let negative=x<=0.;let exp=x.exp();
            let output=if negative {(exp-1.)*alpha} else {x};
            let masked=if negative {g} else {0.};let direct=if negative {0.} else {g};
            (output,(masked*alpha)*exp+direct)
        },
        nn::PiecewiseActivation::Celu {alpha}=>{
            let negative=x<=0.;let exp=(x/alpha).exp();
            let output=if negative {(exp-1.)*alpha} else {x};
            let masked=if negative {g} else {0.};let direct=if negative {0.} else {g};
            (output,((masked*alpha)*exp)*(1f32/alpha)+direct)
        },
        nn::PiecewiseActivation::Selu=>{
            let positive=x>=0.;let exp=x.exp();let coefficient=(ALPHA*GAMMA) as f32;let gamma=GAMMA as f32;
            let output=if positive {x*gamma} else {(exp-1.)*coefficient};
            let masked=if positive {0.} else {g};let direct=if positive {g} else {0.};
            (output,(masked*coefficient)*exp+direct*gamma)
        },_=>unreachable!("this example covers exponential piecewise activations"),
    }
}
fn apply<B:nn::PiecewiseBackend,const D:usize>(input:Tensor<B,D>,activation:nn::PiecewiseActivation)->Result<Tensor<B,D>,rust_ascend::driver::CannError> {
    match activation {nn::PiecewiseActivation::Elu {alpha}=>nn::elu(input,alpha),
        nn::PiecewiseActivation::Celu {alpha}=>nn::celu(input,alpha),nn::PiecewiseActivation::Selu=>nn::selu(input),
        _=>unreachable!("this example covers exponential piecewise activations")}
}
fn close(actual:&[f32],expected:&[f32],name:&str)->Result<(),Box<dyn std::error::Error>> {
    if actual.len()!=expected.len() {return Err(format!("{name}: length mismatch").into());}
    for (i,(&a,&b)) in actual.iter().zip(expected).enumerate() {
        if !(a==b || a.is_nan() && b.is_nan() || a.is_finite() && b.is_finite() && (a-b).abs()<=3e-5*(1.+b.abs())) {
            return Err(format!("{name}[{i}]: {a:?} != {b:?}").into());
        }
    }Ok(())
}
fn values()->[f32;11] { [f32::NEG_INFINITY,-2.,-1.,-0.,0.,0.5,1.,2.,90.,f32::INFINITY,f32::from_bits(0x7fc12345)] }
fn case<const D:usize>(device:&AscendDevice,shape:[usize;D])->Result<(),Box<dyn std::error::Error>> {
    let values=values();let count=shape.iter().product();
    let x:Vec<f32>=(0..count).map(|i|values[i%values.len()]).collect();
    let dy:Vec<f32>=(0..count).map(|i|if i%3==0 {-0.} else {(i%11) as f32/8.-0.375}).collect();
    for activation in modes() {
        let input=Tensor::<AD,D>::from_data(TensorData::new(x.clone(),shape),(device,DType::F32)).require_grad();
        let output=apply(input.clone(),activation)?;assert_eq!(output.dims(),shape);assert_eq!(output.dtype(),DType::F32);
        let expected:Vec<f32>=x.iter().map(|&x|reference(x,0.,activation).0).collect();
        close(output.clone().into_data().as_slice::<f32>()?,&expected,"exponential piecewise output")?;
        let upstream=Tensor::<AD,D>::from_data(TensorData::new(dy.clone(),shape),(device,DType::F32));
        let gradients=((output.clone()+output)*upstream).backward();
        let expected:Vec<f32>=x.iter().zip(&dy).map(|(&x,&g)|reference(x,2.*g,activation).1).collect();
        let dx=input.grad(&gradients).ok_or("missing exponential piecewise gradient")?;assert_eq!(dx.dims(),shape);
        close(dx.into_data().as_slice::<f32>()?,&expected,"exponential piecewise shared gradient")?;
        println!("ASCEND_EXP_PIECEWISE_TENSOR_CASE shape={shape:?} activation={activation:?} passed=true");
    }Ok(())
}
fn main()->Result<(),Box<dyn std::error::Error>> {
    let toolkit=std::env::var_os("ASCEND_HOME_PATH").ok_or("set ASCEND_HOME_PATH")?;let mut options=RuntimeOptions::new(toolkit);
    if let Some(path)=std::env::var_os("RUDA_CANN_LIBRARY") {options.acl_library=path;}
    if let Some(paths)=std::env::var_os("RUDA_CANN_OPERATOR_LIBRARIES") {options.operator_libraries=std::env::split_paths(&paths).map(|p|p.into_os_string()).collect();}
    // SAFETY: this standalone executable is the sole ACL context owner.
    let device=unsafe {AscendRuntime::initialize_exclusive(options)?};
    case(&device,[7])?;case(&device,[2,3,33])?;case(&device,[1,1,1,1,1,1,1,65])?;case(&device,[0,7])?;case(&device,[3,0])?;
    let values=values();
    for activation in modes() {
        let input=Tensor::<Ascend,1>::from_data(TensorData::new(values.to_vec(),[values.len()]),(&device,DType::F32));
        let expected:Vec<f32>=values.iter().map(|&x|reference(x,0.,activation).0).collect();
        close(apply(input,activation)?.into_data().as_slice::<f32>()?,&expected,"plain exponential piecewise output")?;
    }
    println!("ASCEND_EXP_PIECEWISE_TENSOR_DEVICE_OK gradient_cases=65 plain_cases=13");Ok(())
}

//! Native FP32 last-axis bias and ordered residual fusion with RUDA gradients.
use rust_ascend::{Ascend,Autodiff,nn,runtime::{AscendRuntime,RuntimeOptions,AscendDevice},
    tensor::{DType,TensorData,api::Tensor}};
type AD=Autodiff<Ascend>;
fn equal(actual:&[f32],expected:&[f32],name:&str)->Result<(),Box<dyn std::error::Error>> {
    if actual.len()!=expected.len() {return Err(format!("{name}: length mismatch").into());}
    for (i,(&a,&b)) in actual.iter().zip(expected).enumerate() {
        if !(a==b || a.is_nan() && b.is_nan()) {return Err(format!("{name}[{i}]: {a} != {b}").into());}
    }Ok(())
}
fn case<const D:usize>(device:&AscendDevice,shape:[usize;D])->Result<(),Box<dyn std::error::Error>> {
    let width=shape[D-1];let count=shape.iter().product();
    let x:Vec<f32>=(0..count).map(|i|(i%13) as f32/8.-0.5).collect();
    let r:Vec<f32>=(0..count).map(|i|(i%5) as f32/4.-0.25).collect();
    let b:Vec<f32>=(0..width).map(|i|(i%7) as f32/16.-0.125).collect();
    let dy:Vec<f32>=(0..count).map(|i|(i%11) as f32/8.-0.375).collect();
    for mode in 0..3 {
        let input=Tensor::<AD,D>::from_data(TensorData::new(x.clone(),shape),(device,DType::F32)).require_grad();
        let residual=Tensor::<AD,D>::from_data(TensorData::new(r.clone(),shape),(device,DType::F32)).require_grad();
        let bias=Tensor::<AD,1>::from_data(TensorData::new(b.clone(),[width]),(device,DType::F32)).require_grad();
        let output=match mode {0=>nn::bias_add(input.clone(),bias.clone())?,
            1=>nn::residual_bias_add(input.clone(),residual.clone(),bias.clone())?,
            _=>nn::residual_bias_add(input.clone(),input.clone(),bias.clone())?};
        assert_eq!(output.dims(),shape);assert_eq!(output.dtype(),DType::F32);
        let expected:Vec<f32>=(0..count).map(|i|(match mode {0=>x[i],1=>x[i]+r[i],_=>x[i]+x[i]})+b[i%width]).collect();
        equal(output.clone().into_data().as_slice::<f32>()?,&expected,"native bias output")?;
        let upstream=Tensor::<AD,D>::from_data(TensorData::new(dy.clone(),shape),(device,DType::F32));
        let gradients=((output.clone()+output)*upstream).backward();
        let dx=input.grad(&gradients).ok_or("missing bias-add dX")?;assert_eq!(dx.dims(),shape);
        let factor=if mode==2 {4.} else {2.};let expected:Vec<f32>=dy.iter().map(|&g|factor*g).collect();
        equal(dx.into_data().as_slice::<f32>()?,&expected,"native bias shared dX")?;
        if mode==1 {equal(residual.grad(&gradients).ok_or("missing residual gradient")?.into_data().as_slice::<f32>()?,
            &dy.iter().map(|&g|2.*g).collect::<Vec<_>>(),"native residual shared dR")?;}
        let mut db=vec![0.;width];for (i,&g) in dy.iter().enumerate() {db[i%width]+=2.*g;}
        equal(bias.grad(&gradients).ok_or("missing bias gradient")?.into_data().as_slice::<f32>()?,&db,"native device bias sum")?;
        println!("ASCEND_BIAS_TENSOR_CASE shape={shape:?} mode={mode} passed=true");
    }
    // A fixed bias must not force an unused bias-gradient reduction.
    let input=Tensor::<AD,D>::from_data(TensorData::new(x,shape),(device,DType::F32)).require_grad();
    let fixed=Tensor::<AD,1>::from_data(TensorData::new(b,[width]),(device,DType::F32));
    let output=nn::bias_add(input.clone(),fixed)?;let upstream=Tensor::<AD,D>::from_data(TensorData::new(dy.clone(),shape),(device,DType::F32));
    equal(input.grad(&(output*upstream).backward()).ok_or("missing fixed-bias input gradient")?.into_data().as_slice::<f32>()?,&dy,"fixed-bias dX")?;Ok(())
}
fn main()->Result<(),Box<dyn std::error::Error>> {
    let toolkit=std::env::var_os("ASCEND_HOME_PATH").ok_or("set ASCEND_HOME_PATH")?;let mut options=RuntimeOptions::new(toolkit);
    if let Some(path)=std::env::var_os("RUDA_CANN_LIBRARY") {options.acl_library=path;}
    if let Some(paths)=std::env::var_os("RUDA_CANN_OPERATOR_LIBRARIES") {options.operator_libraries=std::env::split_paths(&paths).map(|p|p.into_os_string()).collect();}
    // SAFETY: this standalone executable is the sole ACL context owner.
    let device=unsafe {AscendRuntime::initialize_exclusive(options)?};
    case(&device,[1])?;case(&device,[7])?;case(&device,[3,31])?;case(&device,[2,3,33])?;case(&device,[5,4097])?;
    case(&device,[2,0,7])?;case(&device,[1,2,1,3,1,1,1,7])?;
    let plain=|value|Tensor::<Ascend,1>::from_data(TensorData::new(vec![value],[1]),(&device,DType::F32));
    equal(nn::residual_bias_add(plain(1e20),plain(-1e20),plain(1.))?.into_data().as_slice::<f32>()?,&[1.],"ordered FP32 additions")?;
    let data=[0.,-0.,f32::NAN,f32::INFINITY,f32::NEG_INFINITY,1.,-1.];
    let x=Tensor::<Ascend,1>::from_data(TensorData::new(data.to_vec(),[7]),(&device,DType::F32));
    let bias=Tensor::<Ascend,1>::from_data(TensorData::new(vec![1.;7],[7]),(&device,DType::F32));
    equal(nn::bias_add(x,bias)?.into_data().as_slice::<f32>()?,&data.map(|x|x+1.),"nonfinite bias addition")?;
    println!("ASCEND_BIAS_TENSOR_DEVICE_OK gradient_cases=28 plain_cases=2");Ok(())
}

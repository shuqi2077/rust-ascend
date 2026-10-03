//! Native full-range Sin/Cos, shared RUDA gradients and device-generated RoPE tables.
use rust_ascend::{Ascend,Autodiff,nn,runtime::{AscendRuntime,RuntimeOptions,AscendDevice,RotaryLayout},
    tensor::{DType,TensorData,api::Tensor}};
type AD=Autodiff<Ascend>;
fn close(actual:&[f32],expected:&[f64],name:&str)->Result<(),Box<dyn std::error::Error>> {
    if actual.len()!=expected.len() {return Err(format!("{name}: length mismatch").into());}
    for (i,(&a,&b)) in actual.iter().zip(expected).enumerate() {
        if !(a as f64==b || a.is_nan() && b.is_nan() || a.is_finite() && b.is_finite() && (a as f64-b).abs()<=3e-5*(1.+b.abs())) {
            return Err(format!("{name}[{i}]: {a} != {b}").into());
        }
    }Ok(())
}
fn case<const D:usize>(device:&AscendDevice,shape:[usize;D])->Result<(),Box<dyn std::error::Error>> {
    let count=shape.iter().product();
    let samples:[f32;12]=[-0.,0.,-3.,0.5,3.,65505.,-65505.,1e10,-1e10,1e20,-1e20,f32::MAX];
    let x:Vec<f32>=(0..count).map(|i|samples[i%samples.len()]).collect();
    let dy:Vec<f32>=(0..count).map(|i|(i%11) as f32/8.-0.375).collect();
    let input=Tensor::<AD,D>::from_data(TensorData::new(x.clone(),shape),(device,DType::F32)).require_grad();
    let output=input.clone().sin()*0.75+input.clone().cos()*0.25;
    assert_eq!(output.dims(),shape);assert_eq!(output.dtype(),DType::F32);
    close(output.clone().into_data().as_slice::<f32>()?,&x.iter().map(|&x|0.75*(x as f64).sin()+0.25*(x as f64).cos()).collect::<Vec<_>>(),"trig output")?;
    let upstream=Tensor::<AD,D>::from_data(TensorData::new(dy.clone(),shape),(device,DType::F32));
    let gradients=((output.clone()+output)*upstream).backward();
    let expected:Vec<f64>=x.iter().zip(&dy).map(|(&x,&g)|2.*g as f64*(0.75*(x as f64).cos()-0.25*(x as f64).sin())).collect();
    close(input.grad(&gradients).ok_or("missing Sin/Cos gradient")?.into_data().as_slice::<f32>()?,&expected,"trig shared gradient")?;
    println!("ASCEND_TRIG_TENSOR_CASE shape={shape:?} passed=true");Ok(())
}
fn main()->Result<(),Box<dyn std::error::Error>> {
    let toolkit=std::env::var_os("ASCEND_HOME_PATH").ok_or("set ASCEND_HOME_PATH")?;let mut options=RuntimeOptions::new(toolkit);
    if let Some(path)=std::env::var_os("RUDA_CANN_LIBRARY") {options.acl_library=path;}
    if let Some(paths)=std::env::var_os("RUDA_CANN_OPERATOR_LIBRARIES") {options.operator_libraries=std::env::split_paths(&paths).map(|p|p.into_os_string()).collect();}
    // SAFETY: this standalone executable is the sole ACL context owner.
    let device=unsafe {AscendRuntime::initialize_exclusive(options)?};
    case(&device,[1])?;case(&device,[7])?;case(&device,[2,3,33])?;case(&device,[4097])?;case(&device,[0,7])?;
    let values:[f32;7]=[-0.,0.,f32::INFINITY,f32::NEG_INFINITY,f32::NAN,-f32::MAX,f32::MAX];
    let input=Tensor::<Ascend,1>::from_data(TensorData::new(values.to_vec(),[7]),(&device,DType::F32));
    close(input.clone().sin().into_data().as_slice::<f32>()?,&values.map(|x|(x as f64).sin()),"nonfinite Sin")?;
    close(input.cos().into_data().as_slice::<f32>()?,&values.map(|x|(x as f64).cos()),"nonfinite Cos")?;
    let (batch,heads,sequence,width,prefix)=(2,3,5,11,6);let half=prefix/2;
    let angles:Vec<f32>=(0..sequence*half).map(|i|65505.+i as f32*0.5).collect();
    let angle=Tensor::<Ascend,4>::from_data(TensorData::new(angles.clone(),[1,1,sequence,half]),(&device,DType::F32));
    let cos=angle.clone().cos();let sin=angle.sin();let n=batch*heads*sequence*width;
    let x:Vec<f32>=(0..n).map(|i|(i%19) as f32/8.-1.).collect();
    for layout in [RotaryLayout::Interleaved,RotaryLayout::SplitHalf] {
        let input=Tensor::<Ascend,4>::from_data(TensorData::new(x.clone(),[batch,heads,sequence,width]),(&device,DType::F32));
        let output=nn::rotary_prefix(input,cos.clone(),sin.clone(),prefix,layout)?;
        let mut expected:Vec<f64>=x.iter().map(|&x|x as f64).collect();
        for row in 0..batch*heads*sequence {for pair in 0..half {
            let (a,b)=if layout==RotaryLayout::Interleaved {(row*width+pair*2,row*width+pair*2+1)} else {(row*width+pair,row*width+half+pair)};
            let angle=angles[(row%sequence)*half+pair] as f64;
            expected[a]=x[a] as f64*angle.cos()-x[b] as f64*angle.sin();
            expected[b]=x[b] as f64*angle.cos()+x[a] as f64*angle.sin();
        }}
        close(output.into_data().as_slice::<f32>()?,&expected,"device-generated prefix RoPE")?;
    }
    println!("ASCEND_TRIG_TENSOR_DEVICE_OK gradient_cases=5 plain_cases=4");Ok(())
}

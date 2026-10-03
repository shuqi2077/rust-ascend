//! Padded GELU MLP/GEGLU with trainable or fixed BF16 weights and RUDA gradients.
use rust_ascend::{Ascend,Autodiff,nn,runtime::{AscendRuntime,RuntimeOptions,AscendDevice},
    tensor::{DType,TensorData,api::{Tensor,activation}}};
type AD=Autodiff<Ascend>;
#[cfg_attr(unix,link(name="m"))]
unsafe extern "C" {#[link_name="erf"] fn c_erf(x:f64)->f64;}
fn bf16(x:f32)->f32 {let bits=x.to_bits();f32::from_bits(bits.wrapping_add(0x7fff+((bits>>16)&1))&0xffff0000)}
fn linear(x:&[f32],w:&[f32],rows:usize,k:usize,n:usize)->Vec<f32> {
    (0..rows*n).map(|i|(0..k).map(|j|bf16(x[i/n*k+j])*bf16(w[i%n*k+j])).sum()).collect()
}
fn gelu(x:f32,mode:nn::GeluMode)->f32 {
    match mode {nn::GeluMode::Exact=>x*(1.+unsafe {c_erf(x as f64/core::f64::consts::SQRT_2)} as f32)*0.5,
        nn::GeluMode::Tanh=>x*0.5*(1.+((2./core::f32::consts::PI).sqrt()*(x+0.044715*x*x*x)).tanh())}
}
fn close(a:&[f32],b:&[f32],name:&str)->Result<(),Box<dyn std::error::Error>> {
    if a.len()!=b.len() {return Err(format!("{name}: length mismatch").into());}
    for (i,(&a,&b)) in a.iter().zip(b).enumerate() {
        if !a.is_finite() || !b.is_finite() || (a-b).abs()>5e-4*(1.+b.abs()) {return Err(format!("{name}[{i}]: {a} != {b}").into());}
    }Ok(())
}
fn run<const D:usize>(device:&AscendDevice,shape:[usize;D],gated:bool,frozen:bool,mode:nn::GeluMode,direct:bool)
    ->Result<Vec<Vec<f32>>,Box<dyn std::error::Error>> {
    let k=shape[D-1];let rows=shape[..D-1].iter().product::<usize>();let (h,n)=(5,9);
    let mut out_shape=shape;out_shape[D-1]=n;
    let x:Vec<f32>=(0..rows*k).map(|i|((i%13) as f32-6.)/64.).collect();
    let u:Vec<f32>=(0..h*k).map(|i|((i%11) as f32-5.)/16.).collect();
    let g:Vec<f32>=(0..h*k).map(|i|((i%7) as f32-3.)/16.).collect();
    let d:Vec<f32>=(0..n*h).map(|i|((i%5) as f32-2.)/8.).collect();
    let dy:Vec<f32>=(0..rows*n).map(|i|((i%7) as f32-3.)/8.).collect();
    let input=Tensor::<AD,D>::from_data(TensorData::new(x.clone(),shape),(device,DType::F32)).require_grad();
    let train=|values:Vec<f32>,dims|Tensor::<AD,2>::from_data(TensorData::new(values,dims),(device,DType::F32)).require_grad();
    let fixed=|values:Vec<f32>,dims|Tensor::<Ascend,2>::from_data(TensorData::new(values,dims),(device,DType::BF16));
    let up=train(u.clone(),[h,k]);let gate=train(g.clone(),[h,k]);let down=train(d.clone(),[n,h]);
    let fixed_up=fixed(u.clone(),[h,k]);let fixed_gate=fixed(g.clone(),[h,k]);let fixed_down=fixed(d.clone(),[n,h]);
    let before=[fixed_up.clone().into_data().bytes.to_vec(),fixed_gate.clone().into_data().bytes.to_vec(),fixed_down.clone().into_data().bytes.to_vec()];
    let output=if direct {match (gated,frozen) {
        (false,false)=>nn::gelu_mlp_padded_bf16_fp32_nd(input.clone(),up.clone(),down.clone(),mode)?,
        (true,false)=>nn::geglu_padded_bf16_fp32_nd(input.clone(),gate.clone(),up.clone(),down.clone(),mode)?,
        (false,true)=>nn::gelu_mlp_frozen_padded_bf16_fp32_nd(input.clone(),fixed_up.clone(),fixed_down.clone(),mode)?,
        (true,true)=>nn::geglu_frozen_padded_bf16_fp32_nd(input.clone(),fixed_gate.clone(),fixed_up.clone(),fixed_down.clone(),mode)?,
    }} else {
        let project=|x:Tensor<AD,D>,w:Tensor<AD,2>,f:Tensor<Ascend,2>|if frozen {nn::linear_frozen_padded_bf16_fp32_nd(x,f)} else {nn::linear_padded_bf16_fp32_nd(x,w)};
        let value=project(input.clone(),if gated {gate.clone()} else {up.clone()},if gated {fixed_gate.clone()} else {fixed_up.clone()})?;
        let value=match mode {nn::GeluMode::Exact=>activation::gelu(value),nn::GeluMode::Tanh=>activation::gelu_approximate(value)};
        let value=if gated {value*project(input.clone(),up.clone(),fixed_up.clone())?} else {value};
        project(value,down.clone(),fixed_down.clone())?
    };
    assert_eq!(output.dims(),out_shape);assert_eq!(output.dtype(),DType::F32);
    let mut hidden=linear(&x,if gated {&g} else {&u},rows,k,h);for value in &mut hidden {*value=gelu(*value,mode);}
    if gated {let up=linear(&x,&u,rows,k,h);for (a,b) in hidden.iter_mut().zip(up) {*a*=b;}}
    let expected=linear(&hidden,&d,rows,h,n);let observed=output.clone().into_data().as_slice::<f32>()?.to_vec();
    close(&observed,&expected,"independent GELU FFN forward")?;
    let upstream=Tensor::<AD,D>::from_data(TensorData::new(dy,out_shape),(device,DType::F32));
    let gradients=((output.clone()+output)*upstream).backward();
    let dx=input.grad(&gradients).ok_or("missing GELU FFN dX")?;assert_eq!(dx.dims(),shape);
    let mut result=vec![observed,dx.into_data().as_slice::<f32>()?.to_vec()];
    if !frozen {for weight in [&up,&down] {let gradient=weight.grad(&gradients).ok_or("missing GELU FFN weight gradient")?;
        assert_eq!(gradient.dims(),weight.dims());result.push(gradient.into_data().as_slice::<f32>()?.to_vec());}
        if gated {let gradient=gate.grad(&gradients).ok_or("missing GEGLU gate gradient")?;assert_eq!(gradient.dims(),[h,k]);result.push(gradient.into_data().as_slice::<f32>()?.to_vec());}
    } else {assert_eq!(before,[fixed_up.into_data().bytes.to_vec(),fixed_gate.into_data().bytes.to_vec(),fixed_down.into_data().bytes.to_vec()]);}
    Ok(result)
}
fn case<const D:usize>(device:&AscendDevice,shape:[usize;D])->Result<(),Box<dyn std::error::Error>> {
    for gated in [false,true] {for frozen in [false,true] {for mode in [nn::GeluMode::Exact,nn::GeluMode::Tanh] {
        let direct=run(device,shape,gated,frozen,mode,true)?;let separate=run(device,shape,gated,frozen,mode,false)?;
        assert_eq!(direct.len(),separate.len());for (i,(a,b)) in direct.iter().zip(separate).enumerate() {close(a,&b,&format!("GELU FFN composition component {i}"))?;}
        println!("ASCEND_GELU_FFN_TENSOR_CASE shape={shape:?} gated={gated} frozen={frozen} mode={mode:?} passed=true");
    }}}Ok(())
}
fn main()->Result<(),Box<dyn std::error::Error>> {
    let toolkit=std::env::var_os("ASCEND_HOME_PATH").ok_or("set ASCEND_HOME_PATH")?;let mut options=RuntimeOptions::new(toolkit);
    if let Some(path)=std::env::var_os("RUDA_CANN_LIBRARY") {options.acl_library=path;}
    if let Some(paths)=std::env::var_os("RUDA_CANN_OPERATOR_LIBRARIES") {options.operator_libraries=std::env::split_paths(&paths).map(|p|p.into_os_string()).collect();}
    // SAFETY: this standalone executable is the sole ACL context owner.
    let device=unsafe {AscendRuntime::initialize_exclusive(options)?};
    case(&device,[7])?;case(&device,[2,3,7])?;case(&device,[1,2,1,3,1,1,1,7])?;
    println!("ASCEND_GELU_FFN_TENSOR_DEVICE_OK cases=24");Ok(())
}

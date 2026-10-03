//! Last-axis Linear/LoRA/SwiGLU preserve token ranks and RUDA parameter gradients.
use rust_ascend::{Ascend,Autodiff,nn,runtime::{AscendRuntime,RuntimeOptions,AscendDevice},
    tensor::{DType,TensorData,api::Tensor}};
type AD=Autodiff<Ascend>;
fn bf16(x:f32)->f64 {let bits=x.to_bits();f32::from_bits(bits.wrapping_add(0x7fff+((bits>>16)&1))&0xffff0000) as f64}
fn close(actual:&[f32],expected:&[f64],name:&str)->Result<(),Box<dyn std::error::Error>> {
    if actual.len()!=expected.len() {return Err(format!("{name}: length mismatch").into());}
    for (i,(&a,&b)) in actual.iter().zip(expected).enumerate() {
        if !a.is_finite() || !b.is_finite() || (a as f64-b).abs()>5e-4+5e-4*b.abs() {return Err(format!("{name}[{i}]: {a} != {b}").into());}
    }Ok(())
}
fn run<const D:usize>(device:&AscendDevice,shape:[usize;D],mode:u8,frozen:bool,nd:bool)
    ->Result<Vec<Vec<f32>>,Box<dyn std::error::Error>> {
    let k=shape[D-1];let m=shape[..D-1].iter().product::<usize>();let (n,hidden,rank)=(9,5,3);
    let mut output_shape=shape;output_shape[D-1]=n;
    let x:Vec<f32>=(0..m*k).map(|i|((i%13) as f32-6.)/53.).collect();
    let wshape=if mode==2 {[hidden,k]} else {[n,k]};
    let ashape=if mode==2 {[hidden,k]} else {[rank,k]};let bshape=if mode==2 {[n,hidden]} else {[n,rank]};
    let w:Vec<f32>=(0..wshape[0]*wshape[1]).map(|i|((i%11) as f32-5.)/16.).collect();
    let a:Vec<f32>=(0..ashape[0]*ashape[1]).map(|i|((i%7) as f32-3.)/16.).collect();
    let b:Vec<f32>=(0..bshape[0]*bshape[1]).map(|i|((i%5) as f32-2.)/8.).collect();
    let dy:Vec<f32>=(0..m*n).map(|i|((i%7) as f32-3.)/8.).collect();
    let input=Tensor::<AD,D>::from_data(TensorData::new(x.clone(),shape),(device,DType::F32)).require_grad();
    let train=|data:Vec<f32>,dims|Tensor::<AD,2>::from_data(TensorData::new(data,dims),(device,DType::F32)).require_grad();
    let fixed=|data:Vec<f32>,dims|Tensor::<Ascend,2>::from_data(TensorData::new(data,dims),(device,DType::BF16));
    let weight=train(w.clone(),wshape);let down=train(a.clone(),ashape);let up=train(b.clone(),bshape);
    let fixed_w=fixed(w.clone(),wshape);let fixed_a=fixed(a,ashape);let fixed_b=fixed(b,bshape);
    let before=[fixed_w.clone().into_data().bytes.to_vec(),fixed_a.clone().into_data().bytes.to_vec(),fixed_b.clone().into_data().bytes.to_vec()];
    let output=if nd {match (frozen,mode) {
        (false,0)=>nn::linear_padded_bf16_fp32_nd(input.clone(),weight.clone())?,
        (false,1)=>nn::lora_padded_linear_bf16_fp32_nd(input.clone(),weight.clone(),down.clone(),up.clone(),0.25)?,
        (false,_)=>nn::swiglu_padded_bf16_fp32_nd(input.clone(),weight.clone(),down.clone(),up.clone())?,
        (true,0)=>nn::linear_frozen_padded_bf16_fp32_nd(input.clone(),fixed_w.clone())?,
        (true,1)=>nn::lora_frozen_padded_linear_bf16_fp32_nd(input.clone(),fixed_w.clone(),down.clone(),up.clone(),0.25)?,
        (true,_)=>nn::swiglu_frozen_padded_bf16_fp32_nd(input.clone(),fixed_w.clone(),fixed_a.clone(),fixed_b.clone())?,
    }} else {let flat=input.clone().reshape([m,k]);let out=match (frozen,mode) {
        (false,0)=>nn::linear_padded_bf16_fp32(flat,weight.clone())?,
        (false,1)=>nn::lora_padded_linear_bf16_fp32(flat,weight.clone(),down.clone(),up.clone(),0.25)?,
        (false,_)=>nn::swiglu_padded_bf16_fp32(flat,weight.clone(),down.clone(),up.clone())?,
        (true,0)=>nn::linear_frozen_padded_bf16_fp32(flat,fixed_w.clone())?,
        (true,1)=>nn::lora_frozen_padded_linear_bf16_fp32(flat,fixed_w.clone(),down.clone(),up.clone(),0.25)?,
        (true,_)=>nn::swiglu_frozen_padded_bf16_fp32(flat,fixed_w.clone(),fixed_a.clone(),fixed_b.clone())?,
    };out.reshape(output_shape)};
    assert_eq!(output.dims(),output_shape);assert_eq!(output.dtype(),DType::F32);
    let y=output.clone().into_data().as_slice::<f32>()?.to_vec();
    let upstream=Tensor::<AD,D>::from_data(TensorData::new(dy.clone(),output_shape),(device,DType::F32));
    let gradients=((output.clone()+output)*upstream).backward();
    let dx=input.grad(&gradients).ok_or("missing last-axis input gradient")?;assert_eq!(dx.dims(),shape);assert_eq!(dx.dtype(),DType::F32);
    let dx=dx.into_data().as_slice::<f32>()?.to_vec();let mut results=vec![y,dx];
    if !frozen {let dw=weight.grad(&gradients).ok_or("missing last-axis weight gradient")?;assert_eq!(dw.dims(),wshape);results.push(dw.into_data().as_slice::<f32>()?.to_vec());}
    if mode!=0 && (!frozen || mode==1) {for (parameter,dims) in [(down,ashape),(up,bshape)] {
        let grad=parameter.grad(&gradients).ok_or("missing last-axis adapter/gate gradient")?;assert_eq!(grad.dims(),dims);
        results.push(grad.into_data().as_slice::<f32>()?.to_vec());
    }}
    assert_eq!([fixed_w.into_data().bytes.to_vec(),fixed_a.into_data().bytes.to_vec(),fixed_b.into_data().bytes.to_vec()],before);
    if mode==0 {
        let mut y=vec![0.;m*n];let mut dx=vec![0.;m*k];let mut dw=vec![0.;n*k];
        for row in 0..m {for col in 0..n {for q in 0..k {
            y[row*n+col]+=bf16(x[row*k+q])*bf16(w[col*k+q]);
            dx[row*k+q]+=bf16(2.*dy[row*n+col])*bf16(w[col*k+q]);
            dw[col*k+q]+=bf16(2.*dy[row*n+col])*bf16(x[row*k+q]);
        }}}
        close(&results[0],&y,"last-axis independent linear Y")?;close(&results[1],&dx,"last-axis independent shared dX")?;
        if !frozen {close(&results[2],&dw,"last-axis independent shared dW")?;}
    }Ok(results)
}
fn case<const D:usize>(device:&AscendDevice,shape:[usize;D])->Result<(),Box<dyn std::error::Error>> {
    for frozen in [false,true] {for mode in 0..3 {
        let nd=run(device,shape,mode,frozen,true)?;let flat=run(device,shape,mode,frozen,false)?;assert_eq!(nd.len(),flat.len());
        for (i,(actual,expected)) in nd.iter().zip(flat).enumerate() {
            close(actual,&expected.iter().map(|&x|x as f64).collect::<Vec<_>>(),&format!("last-axis/flat component {i}"))?;
        }
        println!("ASCEND_PROJECTION_TENSOR_CASE shape={shape:?} mode={mode} frozen={frozen} passed=true");
    }}Ok(())
}
fn main()->Result<(),Box<dyn std::error::Error>> {
    let toolkit=std::env::var_os("ASCEND_HOME_PATH").ok_or("set ASCEND_HOME_PATH")?;let mut options=RuntimeOptions::new(toolkit);
    if let Some(path)=std::env::var_os("RUDA_CANN_LIBRARY") {options.acl_library=path;}
    if let Some(paths)=std::env::var_os("RUDA_CANN_OPERATOR_LIBRARIES") {options.operator_libraries=std::env::split_paths(&paths).map(|p|p.into_os_string()).collect();}
    // SAFETY: this standalone executable is the sole ACL context owner.
    let device=unsafe {AscendRuntime::initialize_exclusive(options)?};
    case(&device,[7])?;case(&device,[3,7])?;case(&device,[2,3,7])?;case(&device,[2,1,3,7])?;case(&device,[1,2,1,1,3,1,1,7])?;
    println!("ASCEND_PROJECTION_TENSOR_DEVICE_OK cases=30");Ok(())
}

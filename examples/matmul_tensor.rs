//! Dense and batched NN/NT/TN/TT BF16 compute through the RUDA autodiff graph.
use rust_ascend::{Ascend,Autodiff,nn,runtime::{AscendDevice,AscendRuntime,RuntimeOptions,Transpose},
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
fn case<const D:usize>(device:&AscendDevice,batches:usize,ta:Transpose,tb:Transpose)
    ->Result<(),Box<dyn std::error::Error>> {
    let (m,n,k)=(32,48,16);
    let mut ashape=if ta==Transpose::No {vec![m,k]} else {vec![k,m]};
    let mut bshape=if tb==Transpose::No {vec![k,n]} else {vec![n,k]};
    let mut oshape=vec![m,n];
    if D==3 {ashape.insert(0,batches);bshape.insert(0,batches);oshape.insert(0,batches);}
    let ashape:[usize;D]=ashape.try_into().map_err(|_|"input rank mismatch")?;
    let bshape:[usize;D]=bshape.try_into().map_err(|_|"input rank mismatch")?;
    let oshape:[usize;D]=oshape.try_into().map_err(|_|"output rank mismatch")?;
    let a:Vec<f32>=(0..batches*m*k).map(|i|((i%17) as f32-8.)*0.125).collect();
    let b:Vec<f32>=(0..batches*k*n).map(|i|((i%13) as f32-6.)*0.25).collect();
    let dy:Vec<f32>=(0..batches*m*n).map(|i|((i%11) as f32-5.)*0.0625).collect();
    let x=Tensor::<AD,D>::from_data(TensorData::new(a.clone(),ashape),(device,DType::F32)).require_grad();
    let w=Tensor::<AD,D>::from_data(TensorData::new(b.clone(),bshape),(device,DType::F32)).require_grad();
    let upstream=Tensor::<AD,D>::from_data(TensorData::new(dy.clone(),oshape),(device,DType::F32));
    let output=nn::matmul_bf16_fp32(x.clone(),w.clone(),ta,tb)?;
    if output.dims()!=oshape || output.dtype()!=DType::F32 {return Err("matmul output metadata mismatch".into());}
    let observed=output.clone().into_data();
    let gradients=(output*upstream).backward();
    let dx=x.grad(&gradients).ok_or("missing A gradient")?;
    let dw=w.grad(&gradients).ok_or("missing B gradient")?;
    if dx.dims()!=ashape || dw.dims()!=bshape || dx.dtype()!=DType::F32 || dw.dtype()!=DType::F32 {
        return Err("matmul gradient metadata mismatch".into());
    }
    let mut y=vec![0.;batches*m*n];let mut gx=vec![0.;a.len()];let mut gw=vec![0.;b.len()];
    for batch in 0..batches {for row in 0..m {for col in 0..n {for q in 0..k {
        let ai=batch*m*k+if ta==Transpose::No {row*k+q} else {q*m+row};
        let bi=batch*k*n+if tb==Transpose::No {q*n+col} else {col*k+q};
        let yi=batch*m*n+row*n+col;
        y[yi]+=a[ai] as f64*b[bi] as f64;
        gx[ai]+=dy[yi] as f64*b[bi] as f64;
        gw[bi]+=dy[yi] as f64*a[ai] as f64;
    }}}}
    close(observed.as_slice::<f32>()?,&y,"Y")?;
    close(dx.into_data().as_slice::<f32>()?,&gx,"dA")?;
    close(dw.into_data().as_slice::<f32>()?,&gw,"dB")?;
    let plain_x=Tensor::<Ascend,D>::from_data(TensorData::new(a,ashape),(device,DType::F32));
    let plain_w=Tensor::<Ascend,D>::from_data(TensorData::new(b,bshape),(device,DType::F32));
    close(nn::matmul_bf16_fp32(plain_x,plain_w,ta,tb)?.into_data().as_slice::<f32>()?,&y,"plain Y")?;
    println!("ASCEND_MATMUL_TENSOR_CASE rank={D} batches={batches} ta={ta:?} tb={tb:?} passed=true");
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
    let device=unsafe {AscendRuntime::initialize_exclusive(options)?};
    for ta in [Transpose::No,Transpose::Yes] {for tb in [Transpose::No,Transpose::Yes] {
        case::<2>(&device,1,ta,tb)?;case::<3>(&device,2,ta,tb)?;
    }}
    println!("ASCEND_MATMUL_TENSOR_DEVICE_OK cases=8");
    Ok(())
}

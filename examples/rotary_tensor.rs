//! Caller-supplied fixed rotary tables, both pairing layouts and RUDA input gradients.
use rust_ascend::{Ascend,Autodiff,nn::{self,RotaryLayout},runtime::{AscendRuntime,RuntimeOptions},
    tensor::{DType,TensorData,api::Tensor}};
type AD=Autodiff<Ascend>;
fn close(actual:&[f32],expected:&[f64],name:&str)->Result<(),Box<dyn std::error::Error>> {
    if actual.len()!=expected.len() {return Err(format!("{name}: length mismatch").into());}
    for (i,(&a,&b)) in actual.iter().zip(expected).enumerate() {
        if !a.is_finite() || (a as f64-b).abs()>2e-5+1e-4*b.abs() {
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
    // SAFETY: this executable exclusively owns the process-wide ACL context.
    let device=unsafe {AscendRuntime::initialize_exclusive(options)?};let mut cases=0;
    for (rows,width) in [(0,6),(1,2),(3,6),(2,96)] {for layout in [RotaryLayout::Interleaved,RotaryLayout::SplitHalf] {
        let n=rows*width;let half=width/2;
        let x:Vec<f32>=(0..n).map(|i|(i%19) as f32*0.25-2.).collect();
        let dy:Vec<f32>=(0..n).map(|i|(i%7) as f32*0.125-0.25).collect();
        let angles:Vec<f64>=(0..n/2).map(|i|(i%23) as f64*0.125).collect();
        let cos:Vec<f32>=angles.iter().map(|&a|a.cos() as f32).collect();
        let sin:Vec<f32>=angles.iter().map(|&a|a.sin() as f32).collect();
        let input=Tensor::<AD,2>::from_data(TensorData::new(x.clone(),[rows,width]),(&device,DType::F32)).require_grad();
        let upstream=Tensor::<AD,2>::from_data(TensorData::new(dy.clone(),[rows,width]),(&device,DType::F32));
        let c=Tensor::<Ascend,2>::from_data(TensorData::new(cos.clone(),[rows,half]),(&device,DType::F32));
        let s=Tensor::<Ascend,2>::from_data(TensorData::new(sin.clone(),[rows,half]),(&device,DType::F32));
        let output=nn::rotary(input.clone(),c.clone(),s.clone(),layout)?;
        let observed=output.clone().into_data();let gradients=(output*upstream).backward();
        let mut y=vec![0.;n];let mut dx=vec![0.;n];
        for row in 0..rows {for pair in 0..half {
            let (a,b)=if layout==RotaryLayout::Interleaved {(row*width+pair*2,row*width+pair*2+1)}
                else {(row*width+pair,row*width+half+pair)};
            let c=cos[row*half+pair] as f64;let s=sin[row*half+pair] as f64;
            y[a]=x[a] as f64*c-x[b] as f64*s;y[b]=x[b] as f64*c+x[a] as f64*s;
            dx[a]=dy[a] as f64*c+dy[b] as f64*s;dx[b]=dy[b] as f64*c-dy[a] as f64*s;
        }}
        close(observed.as_slice::<f32>()?,&y,"rotary Y")?;
        close(input.grad(&gradients).ok_or("missing rotary input gradient")?.into_data().as_slice::<f32>()?,&dx,"rotary dX")?;
        let plain=Tensor::<Ascend,2>::from_data(TensorData::new(x,[rows,width]),(&device,DType::F32));
        close(nn::rotary(plain,c,s,layout)?.into_data().as_slice::<f32>()?,&y,"plain rotary Y")?;
        cases+=1;println!("ASCEND_ROTARY_TENSOR_CASE rows={rows} width={width} layout={layout:?} passed=true");
    }}
    println!("ASCEND_ROTARY_TENSOR_DEVICE_OK cases={cases}");
    Ok(())
}

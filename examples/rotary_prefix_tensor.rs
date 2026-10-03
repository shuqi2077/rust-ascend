//! Prefix RoPE with compact singleton-axis tables and input-only RUDA autodiff.
use rust_ascend::{Ascend,Autodiff,nn,runtime::{AscendRuntime,RuntimeOptions,RotaryLayout},tensor::{DType,TensorData,api::Tensor}};
type AD=Autodiff<Ascend>;
fn check(actual:&[f32],expected:&[f64],name:&str)->Result<(),Box<dyn std::error::Error>> {
    if actual.len()!=expected.len() {return Err(format!("{name}: length mismatch").into());}
    for (i,(&a,&b)) in actual.iter().zip(expected).enumerate() {
        if !a.is_finite() || (a as f64-b).abs()>3e-5+3e-5*b.abs() {return Err(format!("{name}[{i}]: {a} != {b}").into());}
    }Ok(())
}
fn main()->Result<(),Box<dyn std::error::Error>> {
    let toolkit=std::env::var_os("ASCEND_HOME_PATH").ok_or("set ASCEND_HOME_PATH")?;let mut options=RuntimeOptions::new(toolkit);
    if let Some(path)=std::env::var_os("RUDA_CANN_LIBRARY") {options.acl_library=path;}
    if let Some(paths)=std::env::var_os("RUDA_CANN_OPERATOR_LIBRARIES") {options.operator_libraries=std::env::split_paths(&paths).map(|p|p.into_os_string()).collect();}
    // SAFETY: this standalone executable is the sole ACL context owner.
    let device=unsafe {AscendRuntime::initialize_exclusive(options)?};
    let (batch,heads,sequence,width,prefix)=(2,3,5,11,6);let half=prefix/2;
    for shape in [[1,1,sequence,half],[batch,1,sequence,half],[1,heads,1,half],[batch,heads,sequence,half]] {
        let elements=shape.iter().product::<usize>();
        let cos:Vec<f32>=(0..elements).map(|i|0.75+(i%3) as f32/16.).collect();
        let sin:Vec<f32>=(0..elements).map(|i|0.25-(i%5) as f32/32.).collect();
        let cos_tensor=Tensor::<Ascend,4>::from_data(TensorData::new(cos.clone(),shape),(&device,DType::F32));
        let sin_tensor=Tensor::<Ascend,4>::from_data(TensorData::new(sin.clone(),shape),(&device,DType::F32));
        for layout in [RotaryLayout::Interleaved,RotaryLayout::SplitHalf] {
            let n=batch*heads*sequence*width;let x:Vec<f32>=(0..n).map(|i|(i%19) as f32/8.-1.).collect();
            let dy:Vec<f32>=(0..n).map(|i|(i%13) as f32/8.-0.5).collect();
            let input=Tensor::<AD,4>::from_data(TensorData::new(x.clone(),[batch,heads,sequence,width]),(&device,DType::F32)).require_grad();
            let upstream=Tensor::<AD,4>::from_data(TensorData::new(dy.clone(),[batch,heads,sequence,width]),(&device,DType::F32));
            let output=nn::rotary_prefix(input.clone(),cos_tensor.clone(),sin_tensor.clone(),prefix,layout)?;
            let mut expected:Vec<f64>=x.iter().map(|&x|x as f64).collect();let mut dx:Vec<f64>=dy.iter().map(|&x|2.*x as f64).collect();
            for b in 0..batch {for h in 0..heads {for s in 0..sequence {
                let row=(b*heads+h)*sequence+s;let coordinate=|c,dim|if dim==1 {0} else {c};
                let table_row=(coordinate(b,shape[0])*shape[1]+coordinate(h,shape[1]))*shape[2]+coordinate(s,shape[2]);
                for pair in 0..half {
                    let (a,b)=if layout==RotaryLayout::Interleaved {(row*width+pair*2,row*width+pair*2+1)} else {(row*width+pair,row*width+half+pair)};
                    let c=cos[table_row*half+pair] as f64;let s=sin[table_row*half+pair] as f64;
                    expected[a]=x[a] as f64*c-x[b] as f64*s;expected[b]=x[b] as f64*c+x[a] as f64*s;
                    dx[a]=2.*(dy[a] as f64*c+dy[b] as f64*s);dx[b]=2.*(dy[b] as f64*c-dy[a] as f64*s);
                }
            }}}
            let observed=output.clone().into_data();check(observed.as_slice::<f32>()?,&expected,"prefix output")?;
            for row in 0..n/width {for col in prefix..width {assert_eq!(observed.as_slice::<f32>()?[row*width+col].to_bits(),x[row*width+col].to_bits());}}
            let gradients=(output.clone()*upstream.clone()+output*upstream).backward();
            check(input.grad(&gradients).ok_or("missing prefix gradient")?.into_data().as_slice::<f32>()?,&dx,"prefix shared dX")?;
            println!("ASCEND_ROTARY_PREFIX_TENSOR_CASE tables={shape:?} layout={layout:?} passed=true");
        }
    }
    for shape in [[0,3,5,11],[2,3,0,11]] {
        let table=[1,1,shape[2],half];let values=vec![0.75;table.iter().product()];
        let cos=Tensor::<Ascend,4>::from_data(TensorData::new(values.clone(),table),(&device,DType::F32));
        let sin=Tensor::<Ascend,4>::from_data(TensorData::new(values,table),(&device,DType::F32));
        let input=Tensor::<AD,4>::from_data(TensorData::new(Vec::<f32>::new(),shape),(&device,DType::F32)).require_grad();
        let output=nn::rotary_prefix(input.clone(),cos,sin,prefix,RotaryLayout::SplitHalf)?;
        assert!(output.clone().into_data().as_slice::<f32>()?.is_empty());
        assert!(input.grad(&output.backward()).ok_or("missing empty prefix gradient")?.into_data().as_slice::<f32>()?.is_empty());
    }
    println!("ASCEND_ROTARY_PREFIX_TENSOR_DEVICE_OK cases=10");Ok(())
}

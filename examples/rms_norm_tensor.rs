//! Native RMSNorm using RUDA tensors, a RUDA module's parameters and RUDA autodiff.
use rust_ascend::{Ascend, Autodiff, nn, runtime::{AscendRuntime, RuntimeOptions},
    tensor::{DType, TensorData, api::Tensor}};
use ruda_nn::RmsNormConfig;

type AD = Autodiff<Ascend>;

fn close(actual: &[f32], expected: &[f64], name: &str) -> Result<(), Box<dyn std::error::Error>> {
    if actual.len()!=expected.len() {return Err(format!("{name}: length mismatch").into());}
    for (i,(&a,&b)) in actual.iter().zip(expected).enumerate() {
        if !a.is_finite() || (a as f64-b).abs()>5e-4+5e-4*b.abs() {
            return Err(format!("{name}[{i}]: {a} != {b}").into());
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let toolkit=std::env::var_os("ASCEND_HOME_PATH").ok_or("set ASCEND_HOME_PATH")?;
    let mut options=RuntimeOptions::new(toolkit);
    if let Some(path)=std::env::var_os("RUDA_CANN_LIBRARY") {options.acl_library=path;}
    if let Some(paths)=std::env::var_os("RUDA_CANN_OPERATOR_LIBRARIES") {
        options.operator_libraries=std::env::split_paths(&paths).map(|p|p.into_os_string()).collect();
    }
    // SAFETY: standalone executable; no other library owns the ACL context.
    let device=unsafe {AscendRuntime::initialize_exclusive(options)?};
    for (rows,width) in [(0,32),(1,32),(3,96),(7,256),(33,4096)] {
        let eps=1e-3;
        let weight:Vec<f32>=(0..width).map(|i|0.5+(i%7) as f32/9.).collect();
        let mut layer=RmsNormConfig::new(width).with_epsilon(eps).init::<AD>(&device);
        layer.gamma=layer.gamma.map(|_| Tensor::<AD,1>::from_data(TensorData::new(weight.clone(),[width]),
            (&device,DType::F32)).require_grad());
        let x:Vec<f32>=(0..rows*width).map(|i|(i%31) as f32/7.-1.).collect();
        let upstream:Vec<f32>=(0..rows*width).map(|i|(i%11) as f32/5.-0.7).collect();
        let input=Tensor::<AD,3>::from_data(TensorData::new(x.clone(),[1,rows,width]),
            (&device,DType::F32)).require_grad();
        let dy=Tensor::<AD,3>::from_data(TensorData::new(upstream.clone(),[1,rows,width]),(&device,DType::F32));
        let y=nn::rms_norm(input.clone(),layer.gamma.val(),layer.epsilon)?;
        let observed=y.clone().into_data();
        let gradients=(y.clone()*dy.clone()+y*dy).backward();
        let dx=input.grad(&gradients).ok_or("missing RMSNorm input gradient")?.into_data();
        let dw=layer.gamma.val().grad(&gradients).ok_or("missing RMSNorm weight gradient")?.into_data();
        let mut expected=vec![0.;x.len()]; let mut expected_dx=expected.clone();
        let mut expected_dw=vec![0.;width];
        for row in 0..rows {
            let offset=row*width;
            let a=&x[offset..offset+width];
            let r=(a.iter().map(|&v|(v as f64).powi(2)).sum::<f64>()/width as f64+eps).sqrt().recip();
            let mean_gx=(0..width).map(|c|2.*upstream[offset+c] as f64*weight[c] as f64*a[c] as f64)
                .sum::<f64>()/width as f64;
            for c in 0..width {
                let i=offset+c; let normalized=a[c] as f64*r;
                expected[i]=normalized*weight[c] as f64;
                expected_dx[i]=r*(2.*upstream[i] as f64*weight[c] as f64-a[c] as f64*r*r*mean_gx);
                expected_dw[c]+=2.*upstream[i] as f64*normalized;
            }
        }
        close(observed.as_slice::<f32>()?,&expected,"Y")?;
        close(dx.as_slice::<f32>()?,&expected_dx,"dX")?;
        close(dw.as_slice::<f32>()?,&expected_dw,"dWeight")?;
        // The plain backend follows the same native forward without graph registration.
        let plain_x=Tensor::<Ascend,3>::from_data(TensorData::new(x,[1,rows,width]),(&device,DType::F32));
        let plain_w=Tensor::<Ascend,1>::from_data(TensorData::new(weight,[width]),(&device,DType::F32));
        let plain=nn::rms_norm(plain_x,plain_w,eps)?.into_data();
        close(plain.as_slice::<f32>()?,&expected,"plain Y")?;
        println!("ASCEND_RMS_NORM_TENSOR_CASE rows={rows} width={width} passed=true");
    }
    println!("ASCEND_RMS_NORM_TENSOR_DEVICE_OK cases=5");
    Ok(())
}

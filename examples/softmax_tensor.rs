//! Last-axis native Softmax and LogSoftmax, reusing RUDA tensors and autodiff.
use rust_ascend::{Ascend, Autodiff, nn, runtime::{AscendRuntime, RuntimeOptions},
    tensor::{DType, TensorData, api::Tensor}};
type AD = Autodiff<Ascend>;

fn close(actual:&[f32], expected:&[f64], name:&str)->Result<(),Box<dyn std::error::Error>> {
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
    // SAFETY: this standalone executable is the sole ACL context owner.
    let device=unsafe {AscendRuntime::initialize_exclusive(options)?};
    for (rows,width) in [(0,32),(1,32),(3,96),(7,256),(33,4096)] {for logarithmic in [false,true] {
        let x:Vec<f32>=(0..rows*width).map(|i|(i%31) as f32/4.-1000.).collect();
        let upstream:Vec<f32>=(0..rows*width).map(|i|(i%11) as f32/5.-0.7).collect();
        let input=Tensor::<AD,3>::from_data(TensorData::new(x.clone(),[1,rows,width]),
            (&device,DType::F32)).require_grad();
        let dy=Tensor::<AD,3>::from_data(TensorData::new(upstream.clone(),[1,rows,width]),(&device,DType::F32));
        let y=if logarithmic {nn::log_softmax(input.clone())?} else {nn::softmax(input.clone())?};
        let observed=y.clone().into_data();
        let gradients=(y.clone()*dy.clone()+y*dy).backward();
        let dx=input.grad(&gradients).ok_or("missing native Softmax input gradient")?.into_data();
        let mut expected=vec![0.;x.len()];let mut expected_dx=expected.clone();
        for row in 0..rows {
            let offset=row*width;let a=&x[offset..offset+width];
            let max=a.iter().map(|&v|v as f64).fold(f64::NEG_INFINITY,f64::max);
            let exp:Vec<_>=a.iter().map(|&v|(v as f64-max).exp()).collect();let sum=exp.iter().sum::<f64>();
            let dot=(0..width).map(|c|2.*upstream[offset+c] as f64*if logarithmic {1.} else {exp[c]/sum}).sum::<f64>();
            for c in 0..width {
                let i=offset+c;let probability=exp[c]/sum;
                expected[i]=if logarithmic {a[c] as f64-max-sum.ln()} else {probability};
                expected_dx[i]=if logarithmic {2.*upstream[i] as f64-probability*dot}
                    else {probability*(2.*upstream[i] as f64-dot)};
            }
        }
        close(observed.as_slice::<f32>()?,&expected,"Y")?;
        close(dx.as_slice::<f32>()?,&expected_dx,"dX")?;
        let plain=Tensor::<Ascend,3>::from_data(TensorData::new(x.clone(),[1,rows,width]),(&device,DType::F32));
        let plain=if logarithmic {nn::log_softmax(plain)?} else {nn::softmax(plain)?};
        close(plain.into_data().as_slice::<f32>()?,&expected,"plain Y")?;
        let untracked=Tensor::<AD,3>::from_data(TensorData::new(x,[1,rows,width]),(&device,DType::F32));
        let untracked=if logarithmic {nn::log_softmax(untracked)?} else {nn::softmax(untracked)?};
        close(untracked.into_data().as_slice::<f32>()?,&expected,"untracked Y")?;
        println!("ASCEND_SOFTMAX_TENSOR_CASE rows={rows} width={width} logarithmic={logarithmic} passed=true");
    }}
    println!("ASCEND_SOFTMAX_TENSOR_DEVICE_OK cases=10");
    Ok(())
}

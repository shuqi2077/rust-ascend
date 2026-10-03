//! Native gated activation and both derivatives through the existing RUDA graph.
use rust_ascend::{Ascend,Autodiff,nn,runtime::{AscendRuntime,RuntimeOptions},
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
    // SAFETY: the executable exclusively owns the process-wide ACL context.
    let device=unsafe {AscendRuntime::initialize_exclusive(options)?};
    for n in [0,1,65,1025] {
        let x:Vec<f32>=(0..n).map(|i|(i%41) as f32*0.6-12.).collect();
        let u:Vec<f32>=(0..n).map(|i|(i%13) as f32*0.125-0.5).collect();
        let dy:Vec<f32>=(0..n).map(|i|(i%7) as f32*0.25-0.75).collect();
        let gate=Tensor::<AD,2>::from_data(TensorData::new(x.clone(),[1,n]),(&device,DType::F32)).require_grad();
        let up=Tensor::<AD,2>::from_data(TensorData::new(u.clone(),[1,n]),(&device,DType::F32)).require_grad();
        let upstream=Tensor::<AD,2>::from_data(TensorData::new(dy.clone(),[1,n]),(&device,DType::F32));
        let output=nn::silu_mul(gate.clone(),up.clone())?;
        let observed=output.clone().into_data();
        let gradients=(output*upstream.clone()).backward();
        let dx=gate.grad(&gradients).ok_or("missing gate gradient")?.into_data();
        let du=up.grad(&gradients).ok_or("missing up gradient")?.into_data();
        let mut expected=vec![0.;n];let mut expected_dx=expected.clone();let mut expected_du=expected.clone();
        let mut expected_shared=expected.clone();
        for i in 0..n {
            let a=x[i] as f64;let s=1./(1.+(-a).exp());let derivative=s*(1.+a*(1.-s));
            expected[i]=a*s*u[i] as f64;
            expected_dx[i]=dy[i] as f64*u[i] as f64*derivative;
            expected_du[i]=dy[i] as f64*a*s;
            expected_shared[i]=dy[i] as f64*(a*derivative+a*s);
        }
        close(observed.as_slice::<f32>()?,&expected,"Y")?;
        close(dx.as_slice::<f32>()?,&expected_dx,"dGate")?;
        close(du.as_slice::<f32>()?,&expected_du,"dUp")?;
        let shared_gate=Tensor::<AD,2>::from_data(TensorData::new(x.clone(),[1,n]),(&device,DType::F32)).require_grad();
        let shared=(nn::silu_mul(shared_gate.clone(),shared_gate.clone())?*upstream).backward();
        close(shared_gate.grad(&shared).ok_or("missing shared gate gradient")?.into_data().as_slice::<f32>()?,&expected_shared,"shared node gradient")?;
        let plain_x=Tensor::<Ascend,2>::from_data(TensorData::new(x.clone(),[1,n]),(&device,DType::F32));
        let plain_u=Tensor::<Ascend,2>::from_data(TensorData::new(u.clone(),[1,n]),(&device,DType::F32));
        close(nn::silu_mul(plain_x,plain_u)?.into_data().as_slice::<f32>()?,&expected,"plain Y")?;
        let untracked_x=Tensor::<AD,2>::from_data(TensorData::new(x,[1,n]),(&device,DType::F32));
        let untracked_u=Tensor::<AD,2>::from_data(TensorData::new(u,[1,n]),(&device,DType::F32));
        close(nn::silu_mul(untracked_x,untracked_u)?.into_data().as_slice::<f32>()?,&expected,"untracked Y")?;
        println!("ASCEND_SILU_MUL_TENSOR_CASE elements={n} passed=true");
    }
    println!("ASCEND_SILU_MUL_TENSOR_DEVICE_OK cases=4");
    Ok(())
}

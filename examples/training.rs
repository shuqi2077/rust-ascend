//! Two native RMSNorm-affine training updates on explicit synthetic inputs.
//! Device forward/backward/AdamW execute through RUDA; FP64 host math is a reference only.
use rust_ascend::{Ascend,Autodiff,nn,optim::{AdamWStorageStep,adamw_tensor_step},
    runtime::{AscendRuntime,RuntimeOptions},tensor::{DType,TensorData,api::Tensor}};
type AD=Autodiff<Ascend>;

fn close(actual:&[f32],expected:&[f64],name:&str)->Result<(),Box<dyn std::error::Error>> {
    if actual.len()!=expected.len() {return Err(format!("{name}: length mismatch").into());}
    for (i,(&a,&b)) in actual.iter().zip(expected).enumerate() {
        if !a.is_finite() || (a as f64-b).abs()>1e-4+1e-4*b.abs() {
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
    // SAFETY: this standalone executable exclusively owns the process ACL lifecycle.
    let device=unsafe {AscendRuntime::initialize_exclusive(options)?};
    let rows=3;let width=96;let eps=1e-3;
    let x:Vec<f32>=(0..rows*width).map(|i|(i%29) as f32/8.-1.).collect();
    let target:Vec<f32>=(0..rows*width).map(|i|(i%13) as f32/32.-0.125).collect();
    let initial:Vec<f32>=(0..width).map(|i|0.75+(i%7) as f32/32.).collect();
    let input=Tensor::<AD,2>::from_data(TensorData::new(x.clone(),[rows,width]),(&device,DType::F32));
    let teacher=Tensor::<AD,2>::from_data(TensorData::new(target.clone(),[rows,width]),(&device,DType::F32));
    let mut parameter=Tensor::<Ascend,1>::from_data(TensorData::new(initial.clone(),[width]),(&device,DType::F32));
    let mut first=Tensor::<Ascend,1>::from_data(TensorData::new(vec![0f32;width],[width]),(&device,DType::F32));
    let mut second=Tensor::<Ascend,1>::from_data(TensorData::new(vec![0f32;width],[width]),(&device,DType::F32));
    let mut p:Vec<f64>=initial.iter().map(|&v|v as f64).collect();let mut m=vec![0.;width];let mut v=vec![0.;width];
    for update in 1..=2 {
        let weight=Tensor::<AD,1>::from_inner(parameter.clone()).require_grad();
        let y=nn::rms_norm(input.clone(),weight.clone(),eps)?;
        let residual=y-teacher.clone();
        let loss=nn::mean_last(residual.clone()*residual)?;
        let observed_loss=loss.clone().into_data();
        let gradients=loss.backward();
        let grad=weight.grad(&gradients).ok_or("missing trainable weight gradient")?;
        let mut loss_ref=vec![0.;rows];let mut dw=vec![0.;width];
        for row in 0..rows {
            let offset=row*width;
            let r=(x[offset..offset+width].iter().map(|&z|(z as f64).powi(2)).sum::<f64>()/width as f64+eps).sqrt().recip();
            for col in 0..width {
                let index=offset+col;let normalized=x[index] as f64*r;
                let difference=normalized*p[col]-target[index] as f64;
                loss_ref[row]+=difference*difference/width as f64;
                dw[col]+=2.*difference*normalized/width as f64;
            }
        }
        close(observed_loss.as_slice::<f32>()?,&loss_ref,"row loss")?;
        close(grad.clone().into_data().as_slice::<f32>()?,&dw,"weight gradient")?;
        let beta1=0.9f32;let beta2=0.95f32;
        let step=AdamWStorageStep {learning_rate:0.01,beta1,beta2,epsilon:1e-5,weight_decay:0.02,
            correction1:(1.-(beta1 as f64).powi(update)) as f32,
            correction2:(1.-(beta2 as f64).powi(update)) as f32,inverse_gradient_scale:1.,clip_multiplier:1.};
        // The actual update consumes the device gradient; readbacks above only check it.
        adamw_tensor_step(&mut parameter,&grad,&mut first,&mut second,step)?;
        for col in 0..width {
            m[col]=beta1 as f64*m[col]+(1.-beta1 as f64)*dw[col];
            v[col]=beta2 as f64*v[col]+(1.-beta2 as f64)*dw[col]*dw[col];
            p[col]=p[col]*(1.-step.learning_rate as f64*step.weight_decay as f64)
                -step.learning_rate as f64*(m[col]/step.correction1 as f64)
                /((v[col]/step.correction2 as f64).sqrt()+step.epsilon as f64);
        }
        close(parameter.clone().into_data().as_slice::<f32>()?,&p,"updated weight")?;
        close(first.clone().into_data().as_slice::<f32>()?,&m,"first moment")?;
        close(second.clone().into_data().as_slice::<f32>()?,&v,"second moment")?;
        println!("ASCEND_TRAINING_STEP update={update} total=2 row_loss={:?} passed=true",observed_loss.as_slice::<f32>()?);
    }
    println!("ASCEND_TRAINING_DEVICE_OK updates=2");
    Ok(())
}

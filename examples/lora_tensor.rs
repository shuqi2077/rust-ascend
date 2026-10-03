//! Native mixed-precision LoRA and both adapter gradients, followed by an FP32 AdamW update.
use rust_ascend::{Ascend,Autodiff,nn,optim::{AdamWStorageStep,adamw_tensor_step},
    runtime::{AscendRuntime,RuntimeOptions},tensor::{DType,TensorData,api::Tensor}};
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
fn main()->Result<(),Box<dyn std::error::Error>> {
    let toolkit=std::env::var_os("ASCEND_HOME_PATH").ok_or("set ASCEND_HOME_PATH")?;
    let mut options=RuntimeOptions::new(toolkit);
    if let Some(path)=std::env::var_os("RUDA_CANN_LIBRARY") {options.acl_library=path;}
    if let Some(paths)=std::env::var_os("RUDA_CANN_OPERATOR_LIBRARIES") {
        options.operator_libraries=std::env::split_paths(&paths).map(|p|p.into_os_string()).collect();
    }
    // SAFETY: this executable exclusively owns the process-wide ACL context.
    let device=unsafe {AscendRuntime::initialize_exclusive(options)?};
    let (m,n,k,rank)=(16,32,32,16);let scale=0.125f32;
    let x:Vec<f32>=(0..m*k).map(|i|((i%7) as f32-3.)*0.125).collect();
    let dy:Vec<f32>=(0..m*n).map(|i|((i%7) as f32-3.)*0.0625).collect();
    let mut w=vec![0f32;n*k];let mut a=vec![0f32;rank*k];let mut b=vec![0f32;n*rank];
    for col in 0..n {w[col*k+col]=0.25;b[col*rank+col%rank]=0.25;}
    for row in 0..rank {a[row*k+row]=0.5;}
    let input=Tensor::<AD,2>::from_data(TensorData::new(x.clone(),[m,k]),(&device,DType::F32)).require_grad();
    // Base weight is deliberately not require_grad; the public API does not freeze it implicitly.
    let base=Tensor::<AD,2>::from_data(TensorData::new(w.clone(),[n,k]),(&device,DType::F32));
    let mut down=Tensor::<Ascend,2>::from_data(TensorData::new(a.clone(),[rank,k]),(&device,DType::F32));
    let mut up=Tensor::<Ascend,2>::from_data(TensorData::new(b.clone(),[n,rank]),(&device,DType::F32));
    let down_node=Tensor::<AD,2>::from_inner(down.clone()).require_grad();
    let up_node=Tensor::<AD,2>::from_inner(up.clone()).require_grad();
    let upstream=Tensor::<AD,2>::from_data(TensorData::new(dy.clone(),[m,n]),(&device,DType::F32));
    let output=nn::lora_linear_bf16_fp32(input.clone(),base.clone(),down_node.clone(),up_node.clone(),scale)?;
    let observed=output.clone().into_data();let gradients=(output*upstream).backward();
    if base.grad(&gradients).is_some() {return Err("frozen base received a gradient".into());}
    let ga=down_node.grad(&gradients).ok_or("missing LoRA A gradient")?;
    let gb=up_node.grad(&gradients).ok_or("missing LoRA B gradient")?;
    let mut y=vec![0.;m*n];let mut gx=vec![0.;m*k];let mut g_a=vec![0.;rank*k];let mut g_b=vec![0.;n*rank];
    for row in 0..m {for col in 0..n {
        y[row*n+col]=x[row*k+col] as f64*0.25+x[row*k+col%rank] as f64*0.5*0.25*scale as f64;
        gx[row*k+col]+=dy[row*n+col] as f64*0.25;
        let dh=dy[row*n+col] as f64*scale as f64*0.25;
        gx[row*k+col%rank]+=dh*0.5;
        for r in 0..rank {
            g_b[col*rank+r]+=dy[row*n+col] as f64*scale as f64*x[row*k+r] as f64*0.5;
            if r==col%rank {for q in 0..k {g_a[r*k+q]+=dh*x[row*k+q] as f64;}}
        }
    }}
    close(observed.as_slice::<f32>()?,&y,"LoRA Y")?;
    close(input.grad(&gradients).ok_or("missing LoRA input gradient")?.into_data().as_slice::<f32>()?,&gx,"LoRA dX")?;
    close(ga.clone().into_data().as_slice::<f32>()?,&g_a,"LoRA dA")?;
    close(gb.clone().into_data().as_slice::<f32>()?,&g_b,"LoRA dB")?;
    let step=AdamWStorageStep {learning_rate:0.01,beta1:0.9,beta2:0.95,epsilon:1e-5,weight_decay:0.,
        correction1:1.-0.9f32,correction2:1.-0.95f32,inverse_gradient_scale:1.,clip_multiplier:1.};
    let mut am=Tensor::<Ascend,2>::from_data(TensorData::new(vec![0f32;rank*k],[rank,k]),(&device,DType::F32));
    let mut av=Tensor::<Ascend,2>::from_data(TensorData::new(vec![0f32;rank*k],[rank,k]),(&device,DType::F32));
    let mut bm=Tensor::<Ascend,2>::from_data(TensorData::new(vec![0f32;n*rank],[n,rank]),(&device,DType::F32));
    let mut bv=Tensor::<Ascend,2>::from_data(TensorData::new(vec![0f32;n*rank],[n,rank]),(&device,DType::F32));
    adamw_tensor_step(&mut down,&ga,&mut am,&mut av,step)?;
    adamw_tensor_step(&mut up,&gb,&mut bm,&mut bv,step)?;
    let updated=|initial:&[f32],gradient:&[f64]|initial.iter().zip(gradient).map(|(&p,&g)|
        p as f64-step.learning_rate as f64*g/(g.abs()+step.epsilon as f64)).collect::<Vec<_>>();
    close(down.into_data().as_slice::<f32>()?,&updated(&a,&g_a),"updated LoRA A")?;
    close(up.into_data().as_slice::<f32>()?,&updated(&b,&g_b),"updated LoRA B")?;
    if base.into_data().as_slice::<f32>()?!=w {return Err("LoRA update modified frozen base".into());}
    println!("ASCEND_LORA_TENSOR_DEVICE_OK input_and_adapter_gradients=true frozen_base=true updates=1");
    Ok(())
}

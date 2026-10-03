//! Direct BF16-stored frozen base weights with FP32 activation gradients and LoRA updates.
use rust_ascend::{Ascend,Autodiff,nn,optim::{AdamWStorageStep,adamw_tensor_step},runtime::{AscendRuntime,RuntimeOptions},tensor::{DType,TensorData,api::{Tensor,Int}}};
type AD=Autodiff<Ascend>;
fn bf16(value:f32)->f64 {let bits=value.to_bits();f32::from_bits(bits.wrapping_add(0x7fff+((bits>>16)&1))&0xffff0000) as f64}
fn close(actual:&[f32],expected:&[f64],name:&str)->Result<(),Box<dyn std::error::Error>> {
    if actual.len()!=expected.len() {return Err(format!("{name}: length mismatch").into());}
    for (i,(&a,&b)) in actual.iter().zip(expected).enumerate() {
        if !a.is_finite() || (a as f64-b).abs()>5e-4+5e-4*b.abs() {return Err(format!("{name}[{i}]: {a} != {b}").into());}
    }Ok(())
}
fn main()->Result<(),Box<dyn std::error::Error>> {
    let toolkit=std::env::var_os("ASCEND_HOME_PATH").ok_or("set ASCEND_HOME_PATH")?;let mut options=RuntimeOptions::new(toolkit);
    if let Some(path)=std::env::var_os("RUDA_CANN_LIBRARY") {options.acl_library=path;}
    if let Some(paths)=std::env::var_os("RUDA_CANN_OPERATOR_LIBRARIES") {options.operator_libraries=std::env::split_paths(&paths).map(|p|p.into_os_string()).collect();}
    // SAFETY: this standalone executable is the sole ACL context owner.
    let device=unsafe {AscendRuntime::initialize_exclusive(options)?};
    let (vocab,embedding_width)=(7,65);let table:Vec<f32>=(0..vocab*embedding_width).map(|i|(i%23) as f32/16.-0.5).collect();
    let embedding=Tensor::<Ascend,2>::from_data(TensorData::new(table.clone(),[vocab,embedding_width]),(&device,DType::BF16));
    let bytes=embedding.clone().into_data().bytes.to_vec();assert_eq!(bytes.len(),vocab*embedding_width*2);
    let ids=vec![2i32,0,2,5,0,2];let indices=Tensor::<AD,2,Int>::from_data(TensorData::new(ids.clone(),[2,3]),(&device,DType::I32));
    let output=nn::embedding_frozen_bf16_fp32(embedding.clone(),indices)?;assert_eq!(output.dtype(),DType::F32);
    let expected:Vec<f64>=ids.iter().flat_map(|&row|table[row as usize*embedding_width..(row as usize+1)*embedding_width].iter().map(|&x|x as f64)).collect();
    close(output.into_data().as_slice::<f32>()?,&expected,"frozen BF16 embedding")?;
    let indices=Tensor::<Ascend,1,Int>::from_data([2i64,5],(&device,DType::I64));
    let output=nn::embedding_frozen_bf16_fp32_nd::<Ascend,1,2>(embedding.clone(),indices)?;
    let expected:Vec<f64>=[2usize,5].iter().flat_map(|&row|table[row*embedding_width..(row+1)*embedding_width].iter().map(|&x|x as f64)).collect();
    close(output.into_data().as_slice::<f32>()?,&expected,"frozen BF16 INT64 embedding")?;
    let indices=Tensor::<AD,2,Int>::from_data(TensorData::new(Vec::<i32>::new(),[2,0]),(&device,DType::I32));
    assert!(nn::embedding_frozen_bf16_fp32(embedding.clone(),indices)?.into_data().as_slice::<f32>()?.is_empty());
    assert_eq!(embedding.into_data().bytes.to_vec(),bytes);
    for (m,n,k) in [(16,32,32),(32,48,16)] {
        let x:Vec<f32>=(0..m*k).map(|i|(i%17) as f32*0.037-0.25).collect();
        let w:Vec<f32>=(0..n*k).map(|i|(i%13) as f32*0.019-0.125).collect();
        let dy:Vec<f32>=(0..m*n).map(|i|(i%11) as f32*0.017-0.0625).collect();
        let weight=Tensor::<Ascend,2>::from_data(TensorData::new(w.clone(),[n,k]),(&device,DType::BF16));
        assert_eq!(weight.dtype(),DType::BF16);let bytes=weight.clone().into_data().bytes.to_vec();assert_eq!(bytes.len(),n*k*2);
        let input=Tensor::<AD,2>::from_data(TensorData::new(x.clone(),[m,k]),(&device,DType::F32)).require_grad();
        let upstream=Tensor::<AD,2>::from_data(TensorData::new(dy.clone(),[m,n]),(&device,DType::F32));
        let output=nn::linear_frozen_bf16_fp32(input.clone(),weight.clone())?;
        let mut y=vec![0.;m*n];let mut dx=vec![0.;m*k];
        for row in 0..m {for col in 0..n {for c in 0..k {
            y[row*n+col]+=bf16(x[row*k+c])*bf16(w[col*k+c]);
            dx[row*k+c]+=bf16(2.*dy[row*n+col])*bf16(w[col*k+c]);
        }}}
        close(output.clone().into_data().as_slice::<f32>()?,&y,"frozen linear Y")?;
        let gradients=(output.clone()*upstream.clone()+output*upstream).backward();
        close(input.grad(&gradients).ok_or("missing frozen linear dX")?.into_data().as_slice::<f32>()?,&dx,"frozen linear shared dX")?;
        let plain=Tensor::<Ascend,2>::from_data(TensorData::new(x,[m,k]),(&device,DType::F32));
        close(nn::linear_frozen_bf16_fp32(plain,weight.clone())?.into_data().as_slice::<f32>()?,&y,"plain frozen linear")?;
        assert_eq!(weight.into_data().bytes.to_vec(),bytes);
        println!("ASCEND_FROZEN_BF16_LINEAR_CASE m={m} n={n} k={k} passed=true");
    }
    let (m,n,k,rank)=(16,32,32,16);let scale=0.125f32;
    let x:Vec<f32>=(0..m*k).map(|i|((i%7) as f32-3.)/8.).collect();
    let dy:Vec<f32>=(0..m*n).map(|i|((i%7) as f32-3.)/16.).collect();
    let mut w=vec![0f32;n*k];let mut a=vec![0f32;rank*k];let mut b=vec![0f32;n*rank];
    for col in 0..n {w[col*k+col]=0.25;b[col*rank+col%rank]=0.25;}for row in 0..rank {a[row*k+row]=0.5;}
    let weight=Tensor::<Ascend,2>::from_data(TensorData::new(w,[n,k]),(&device,DType::BF16));let bytes=weight.clone().into_data().bytes.to_vec();
    let input=Tensor::<AD,2>::from_data(TensorData::new(x.clone(),[m,k]),(&device,DType::F32)).require_grad();
    let mut down=Tensor::<Ascend,2>::from_data(TensorData::new(a.clone(),[rank,k]),(&device,DType::F32));
    let mut up=Tensor::<Ascend,2>::from_data(TensorData::new(b.clone(),[n,rank]),(&device,DType::F32));
    let down_node=Tensor::<AD,2>::from_inner(down.clone()).require_grad();let up_node=Tensor::<AD,2>::from_inner(up.clone()).require_grad();
    let upstream=Tensor::<AD,2>::from_data(TensorData::new(dy.clone(),[m,n]),(&device,DType::F32));
    let output=nn::lora_frozen_linear_bf16_fp32(input.clone(),weight.clone(),down_node.clone(),up_node.clone(),scale)?;
    let mut y=vec![0.;m*n];let mut gx=vec![0.;m*k];let mut ga=vec![0.;rank*k];let mut gb=vec![0.;n*rank];
    for row in 0..m {for col in 0..n {
        y[row*n+col]=x[row*k+col] as f64*0.25+x[row*k+col%rank] as f64*0.5*0.25*scale as f64;
        gx[row*k+col]+=dy[row*n+col] as f64*0.25;let dh=dy[row*n+col] as f64*scale as f64*0.25;
        gx[row*k+col%rank]+=dh*0.5;
        for r in 0..rank {gb[col*rank+r]+=dy[row*n+col] as f64*scale as f64*x[row*k+r] as f64*0.5;
            if r==col%rank {for c in 0..k {ga[r*k+c]+=dh*x[row*k+c] as f64;}}
        }
    }}
    close(output.clone().into_data().as_slice::<f32>()?,&y,"frozen LoRA Y")?;let gradients=(output*upstream).backward();
    let grad_a=down_node.grad(&gradients).ok_or("missing frozen LoRA dA")?;let grad_b=up_node.grad(&gradients).ok_or("missing frozen LoRA dB")?;
    close(input.grad(&gradients).ok_or("missing frozen LoRA dX")?.into_data().as_slice::<f32>()?,&gx,"frozen LoRA dX")?;
    close(grad_a.clone().into_data().as_slice::<f32>()?,&ga,"frozen LoRA dA")?;close(grad_b.clone().into_data().as_slice::<f32>()?,&gb,"frozen LoRA dB")?;
    let zero=|shape:[usize;2]|Tensor::<Ascend,2>::from_data(TensorData::new(vec![0f32;shape.iter().product()],shape),(&device,DType::F32));
    let mut am=zero([rank,k]);let mut av=zero([rank,k]);let mut bm=zero([n,rank]);let mut bv=zero([n,rank]);
    let step=AdamWStorageStep {learning_rate:0.01,beta1:0.9,beta2:0.95,epsilon:1e-5,weight_decay:0.,correction1:1.-0.9f32,correction2:1.-0.95f32,inverse_gradient_scale:1.,clip_multiplier:1.};
    adamw_tensor_step(&mut down,&grad_a,&mut am,&mut av,step)?;adamw_tensor_step(&mut up,&grad_b,&mut bm,&mut bv,step)?;
    let updated=|initial:&[f32],gradient:&[f64]|initial.iter().zip(gradient).map(|(&p,&g)|p as f64-step.learning_rate as f64*g/(g.abs()+step.epsilon as f64)).collect::<Vec<_>>();
    close(down.into_data().as_slice::<f32>()?,&updated(&a,&ga),"updated frozen LoRA A")?;close(up.into_data().as_slice::<f32>()?,&updated(&b,&gb),"updated frozen LoRA B")?;
    assert_eq!(weight.into_data().bytes.to_vec(),bytes);
    let width=16;let n=width*width;let mut diagonal=vec![0f32;n];for i in 0..width {diagonal[i*width+i]=0.5;}
    let gate=Tensor::<Ascend,2>::from_data(TensorData::new(diagonal.clone(),[width,width]),(&device,DType::BF16));
    for value in &mut diagonal {*value*=0.5;}
    let up=Tensor::<Ascend,2>::from_data(TensorData::new(diagonal.clone(),[width,width]),(&device,DType::BF16));
    let down=Tensor::<Ascend,2>::from_data(TensorData::new(diagonal,[width,width]),(&device,DType::BF16));
    let x:Vec<f32>=(0..n).map(|i|((i%7) as f32-3.)/8.).collect();let dy:Vec<f32>=(0..n).map(|i|((i%7) as f32-3.)/16.).collect();
    let input=Tensor::<AD,2>::from_data(TensorData::new(x.clone(),[width,width]),(&device,DType::F32)).require_grad();
    let upstream=Tensor::<AD,2>::from_data(TensorData::new(dy.clone(),[width,width]),(&device,DType::F32));
    let output=nn::swiglu_frozen_bf16_fp32(input.clone(),gate,up,down)?;
    let mut y=vec![0.;n];let mut dx=vec![0.;n];
    for i in 0..n {
        let g=x[i] as f64*0.5;let u=x[i] as f64*0.25;let sigmoid=1./(1.+(-g).exp());let silu=g*sigmoid;
        y[i]=bf16((silu*u) as f32)*0.25;
        let dh=bf16(dy[i])*0.25;let dg=dh*u*sigmoid*(1.+g*(1.-sigmoid));let du=dh*silu;
        dx[i]=bf16(dg as f32)*0.5+bf16(du as f32)*0.25;
    }
    close(output.clone().into_data().as_slice::<f32>()?,&y,"frozen SwiGLU Y")?;
    let gradients=(output*upstream).backward();close(input.grad(&gradients).ok_or("missing frozen SwiGLU dX")?.into_data().as_slice::<f32>()?,&dx,"frozen SwiGLU dX")?;
    println!("ASCEND_FROZEN_BF16_TENSOR_DEVICE_OK embedding_cases=3 linear_cases=2 lora_updates=1 swiglu_cases=1");Ok(())
}

//! Device matrix tails and arbitrary-rank LoRA, with independent logical-shape references.
use rust_ascend::{Ascend,Autodiff,nn,runtime::{AscendRuntime,RuntimeOptions,Transpose},
    tensor::{DType,TensorData,api::Tensor}};
type AD=Autodiff<Ascend>;
fn bf16(x:f32)->f64 {let bits=x.to_bits();f32::from_bits(bits.wrapping_add(0x7fff+((bits>>16)&1))&0xffff0000) as f64}
fn close(actual:&[f32],expected:&[f64],name:&str)->Result<(),Box<dyn std::error::Error>> {
    if actual.len()!=expected.len() {return Err(format!("{name}: length mismatch").into());}
    for (i,(&a,&b)) in actual.iter().zip(expected).enumerate() {
        if !a.is_finite() || (a as f64-b).abs()>5e-4+5e-4*b.abs() {return Err(format!("{name}[{i}]: {a} != {b}").into());}
    }Ok(())
}
fn matrix_case<const D:usize>(device:&rust_ascend::runtime::AscendDevice,batch:Option<usize>,m:usize,n:usize,k:usize,
    ta:Transpose,tb:Transpose)->Result<(),Box<dyn std::error::Error>> {
    let batches=batch.unwrap_or(1);let mut ashape=if ta==Transpose::No {vec![m,k]} else {vec![k,m]};
    let mut bshape=if tb==Transpose::No {vec![k,n]} else {vec![n,k]};let mut oshape=vec![m,n];
    if let Some(batch)=batch {ashape.insert(0,batch);bshape.insert(0,batch);oshape.insert(0,batch);}
    let ashape:[usize;D]=ashape.try_into().map_err(|_|"matrix case rank")?;
    let bshape:[usize;D]=bshape.try_into().map_err(|_|"matrix case rank")?;
    let oshape:[usize;D]=oshape.try_into().map_err(|_|"matrix case rank")?;
    let a:Vec<f32>=(0..batches*m*k).map(|i|((i%17) as f32-8.)/59.).collect();
    let b:Vec<f32>=(0..batches*k*n).map(|i|((i%13) as f32-6.)/43.).collect();
    let dy:Vec<f32>=(0..batches*m*n).map(|i|((i%11) as f32-5.)/37.).collect();
    let ia=|batch:usize,row:usize,q:usize|batch*m*k+if ta==Transpose::No {row*k+q} else {q*m+row};
    let ib=|batch:usize,q:usize,col:usize|batch*k*n+if tb==Transpose::No {q*n+col} else {col*k+q};
    let mut y=vec![0.;dy.len()];let mut da=vec![0.;a.len()];let mut db=vec![0.;b.len()];
    for batch in 0..batches {for row in 0..m {for col in 0..n {let o=batch*m*n+row*n+col;
        for q in 0..k {let ai=ia(batch,row,q);let bi=ib(batch,q,col);
            y[o]+=bf16(a[ai])*bf16(b[bi]);da[ai]+=bf16(dy[o])*bf16(b[bi]);db[bi]+=bf16(dy[o])*bf16(a[ai]);
        }
    }}}
    let plain_a=Tensor::<Ascend,D>::from_data(TensorData::new(a.clone(),ashape),(device,DType::F32));
    let plain_b=Tensor::<Ascend,D>::from_data(TensorData::new(b.clone(),bshape),(device,DType::F32));
    let plain=nn::matmul_padded_bf16_fp32(plain_a.clone(),plain_b.clone(),ta,tb)?;
    assert_eq!(plain.dims(),oshape);assert_eq!(plain.dtype(),DType::F32);close(plain.into_data().as_slice::<f32>()?,&y,"padded plain Y")?;
    let x=Tensor::<AD,D>::from_inner(plain_a).require_grad();let w=Tensor::<AD,D>::from_inner(plain_b).require_grad();
    let upstream=Tensor::<AD,D>::from_data(TensorData::new(dy,oshape),(device,DType::F32));
    let output=nn::matmul_padded_bf16_fp32(x.clone(),w.clone(),ta,tb)?;
    let gradients=((output.clone()+output)*upstream).backward();
    let dx=x.grad(&gradients).ok_or("missing padded dA")?;let dw=w.grad(&gradients).ok_or("missing padded dB")?;
    assert_eq!(dx.dims(),ashape);assert_eq!(dw.dims(),bshape);assert_eq!(dx.dtype(),DType::F32);assert_eq!(dw.dtype(),DType::F32);
    for g in &mut da {*g*=2.;}for g in &mut db {*g*=2.;}
    close(dx.into_data().as_slice::<f32>()?,&da,"padded shared dA")?;close(dw.into_data().as_slice::<f32>()?,&db,"padded shared dB")?;Ok(())
}
fn lora_case(device:&rust_ascend::runtime::AscendDevice,rank:usize)->Result<(),Box<dyn std::error::Error>> {
    let (m,n,k)=(3,19,35);let scale=0.25f32;
    let x:Vec<f32>=(0..m*k).map(|i|((i%7) as f32-3.)/64.).collect();let w:Vec<f32>=(0..n*k).map(|i|((i%5) as f32-2.)/16.).collect();
    let a:Vec<f32>=(0..rank*k).map(|i|((i%9) as f32-4.)/16.).collect();let b:Vec<f32>=(0..n*rank).map(|i|((i%5) as f32-2.)/8.).collect();
    let dy:Vec<f32>=(0..m*n).map(|i|((i%7) as f32-3.)/8.).collect();
    let input=Tensor::<AD,2>::from_data(TensorData::new(x.clone(),[m,k]),(device,DType::F32)).require_grad();
    let weight=Tensor::<Ascend,2>::from_data(TensorData::new(w.clone(),[n,k]),(device,DType::BF16));
    let bytes=weight.clone().into_data().bytes.to_vec();assert_eq!(bytes.len(),n*k*2);
    let down=Tensor::<AD,2>::from_data(TensorData::new(a.clone(),[rank,k]),(device,DType::F32)).require_grad();
    let up=Tensor::<AD,2>::from_data(TensorData::new(b.clone(),[n,rank]),(device,DType::F32)).require_grad();
    let upstream=Tensor::<AD,2>::from_data(TensorData::new(dy.clone(),[m,n]),(device,DType::F32));
    let mut h=vec![0.;m*rank];for row in 0..m {for r in 0..rank {for q in 0..k {h[row*rank+r]+=bf16(x[row*k+q])*bf16(a[r*k+q]);}}}
    let mut y=vec![0.;m*n];let mut gx=vec![0.;m*k];let mut ga=vec![0.;rank*k];let mut gb=vec![0.;n*rank];let mut dh=vec![0.;m*rank];
    for row in 0..m {for col in 0..n {
        let o=row*n+col;for q in 0..k {y[o]+=bf16(x[row*k+q])*bf16(w[col*k+q]);gx[row*k+q]+=bf16(dy[o])*bf16(w[col*k+q]);}
        let mut adapter=0.;for r in 0..rank {adapter+=bf16(h[row*rank+r] as f32)*bf16(b[col*rank+r]);
            dh[row*rank+r]+=bf16(dy[o]*scale)*bf16(b[col*rank+r]);gb[col*rank+r]+=bf16(dy[o]*scale)*bf16(h[row*rank+r] as f32);
        }y[o]+=adapter*scale as f64;
    }}
    for row in 0..m {for r in 0..rank {for q in 0..k {
        gx[row*k+q]+=bf16(dh[row*rank+r] as f32)*bf16(a[r*k+q]);ga[r*k+q]+=bf16(dh[row*rank+r] as f32)*bf16(x[row*k+q]);
    }}}
    let output=nn::lora_frozen_padded_linear_bf16_fp32(input.clone(),weight.clone(),down.clone(),up.clone(),scale)?;
    assert_eq!(output.dims(),[m,n]);close(output.clone().into_data().as_slice::<f32>()?,&y,"tail frozen LoRA Y")?;
    let gradients=(output*upstream).backward();close(input.grad(&gradients).ok_or("missing tail LoRA dX")?.into_data().as_slice::<f32>()?,&gx,"tail LoRA dX")?;
    close(down.grad(&gradients).ok_or("missing tail LoRA dA")?.into_data().as_slice::<f32>()?,&ga,"tail LoRA dA")?;
    close(up.grad(&gradients).ok_or("missing tail LoRA dB")?.into_data().as_slice::<f32>()?,&gb,"tail LoRA dB")?;
    assert_eq!(weight.into_data().bytes.to_vec(),bytes);Ok(())
}
fn main()->Result<(),Box<dyn std::error::Error>> {
    let toolkit=std::env::var("ASCEND_HOME_PATH")?;
    let device=unsafe {AscendRuntime::initialize_exclusive(RuntimeOptions::new(toolkit))?};
    let mut cases=0;
    for (m,n,k) in [(1,1,1),(3,17,7),(17,33,65),(16,32,48)] {for ta in [Transpose::No,Transpose::Yes] {for tb in [Transpose::No,Transpose::Yes] {
        matrix_case::<2>(&device,None,m,n,k,ta,tb)?;matrix_case::<3>(&device,Some(2),m,n,k,ta,tb)?;cases+=2;
    }}}
    for rank in [1,7,8] {lora_case(&device,rank)?;}
    // A logical infinite activation creates NaNs in artificial output channels
    // (Inf*0). Cropping before the next layer must discard those channels.
    let x=Tensor::<Ascend,2>::from_data(TensorData::new(vec![f32::INFINITY],[1,1]),(&device,DType::F32));
    let one=||Tensor::<Ascend,2>::from_data(TensorData::new(vec![1f32],[1,1]),(&device,DType::F32));
    let out=nn::lora_padded_linear_bf16_fp32(x,one(),one(),one(),1.)?;
    assert_eq!(out.into_data().as_slice::<f32>()?,&[f32::INFINITY]);
    println!("ASCEND_PADDED_MATRIX_TENSOR_DEVICE_OK matrix_cases={cases} frozen_lora_ranks=1,7,8 nonfinite_chain=1");Ok(())
}

//! Integer device lookup and dense weight gradients, including repeated IDs and padding.
use rust_ascend::{Ascend,Autodiff,nn,runtime::{AscendRuntime,RuntimeOptions},
    tensor::{DType,TensorData,api::{Tensor,Int}}};
type AD=Autodiff<Ascend>;
fn close(actual:&[f32],expected:&[f32],name:&str)->Result<(),Box<dyn std::error::Error>> {
    if actual.len()!=expected.len() {return Err(format!("{name}: length mismatch").into());}
    for (i,(&a,&b)) in actual.iter().zip(expected).enumerate() {
        if !a.is_finite() || (a-b).abs()>1e-5+1e-5*b.abs() {return Err(format!("{name}[{i}]: {a} != {b}").into());}
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
    let (vocab,width)=(7,65);
    let weights:Vec<f32>=(0..vocab*width).map(|i|i as f32/8.-2.).collect();
    let ids=vec![2i32,0,2,5,0,2];
    let upstream:Vec<f32>=(0..ids.len()*width).map(|i|(i%17) as f32/8.-0.5).collect();
    for padding_idx in [None,Some(0)] {for scale_grad_by_freq in [false,true] {
        let settings=nn::EmbeddingOptions {padding_idx,scale_grad_by_freq};
        let weight=Tensor::<AD,2>::from_data(TensorData::new(weights.clone(),[vocab,width]),(&device,DType::F32)).require_grad();
        let indices=Tensor::<AD,2,Int>::from_data(TensorData::new(ids.clone(),[2,3]),(&device,DType::I32));
        let y=nn::embedding(weight.clone(),indices,settings)?;
        let expected:Vec<f32>=ids.iter().flat_map(|&row|weights[row as usize*width..(row as usize+1)*width].iter().copied()).collect();
        close(y.clone().into_data().as_slice::<f32>()?,&expected,"lookup")?;
        let dy=Tensor::<AD,3>::from_data(TensorData::new(upstream.clone(),[2,3,width]),(&device,DType::F32));
        let gradients=(y.clone()*dy.clone()+y*dy).backward();
        let actual=weight.grad(&gradients).ok_or("missing embedding weight gradient")?.into_data();
        let mut dw=vec![0.;vocab*width];
        for (position,&id) in ids.iter().enumerate() {
            let row=id as usize;if padding_idx==Some(row) {continue;}
            for c in 0..width {dw[row*width+c]+=2.*upstream[position*width+c];}
        }
        if scale_grad_by_freq {for row in 0..vocab {
            let count=ids.iter().filter(|&&id|id as usize==row).count();
            if count!=0 {for value in &mut dw[row*width..(row+1)*width] {*value/=count as f32;}}
        }}
        close(actual.as_slice::<f32>()?,&dw,"weight gradient")?;
        println!("ASCEND_EMBEDDING_TENSOR_CASE padding={padding_idx:?} frequency={scale_grad_by_freq} passed=true");
    }}
    let plain=Tensor::<Ascend,2>::from_data(TensorData::new(weights.clone(),[vocab,width]),(&device,DType::F32));
    let one=Tensor::<Ascend,1,Int>::from_data([2i64,5],(&device,DType::I64));
    let result=nn::embedding_nd::<Ascend,1,2>(plain,one,nn::EmbeddingOptions::default())?;
    let expected:Vec<f32>=[2usize,5].iter().flat_map(|&row|weights[row*width..(row+1)*width].iter().copied()).collect();
    close(result.into_data().as_slice::<f32>()?,&expected,"INT64 plain lookup")?;
    for shape in [[0,3],[2,0]] {
        let weight=Tensor::<AD,2>::from_data(TensorData::new(weights.clone(),[vocab,width]),(&device,DType::F32)).require_grad();
        let indices=Tensor::<AD,2,Int>::from_data(TensorData::new(Vec::<i32>::new(),shape),(&device,DType::I32));
        let out=nn::embedding(weight.clone(),indices,nn::EmbeddingOptions::default())?;
        assert!(out.clone().into_data().as_slice::<f32>()?.is_empty());
        let gradients=out.backward();
        close(weight.grad(&gradients).ok_or("missing empty embedding gradient")?.into_data().as_slice::<f32>()?,&vec![0.;vocab*width],"empty weight gradient")?;
    }
    println!("ASCEND_EMBEDDING_TENSOR_DEVICE_OK cases=7");Ok(())
}

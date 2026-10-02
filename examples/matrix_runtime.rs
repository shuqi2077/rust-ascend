//! Native BF16 matrix kernels, JIT-built from Rust and called through ComputeClient.
use rust_ascend::{core::tensor::{DType, Shape, Strides},
    runtime::{AscendRuntime, ComputeClient, RuntimeOptions, TensorBuffer, Transpose, portable::backend::Runtime}};

fn upload(client: &ComputeClient<AscendRuntime>, shape: &[usize], values: &[f32]) -> TensorBuffer {
    // The generated test values are exactly representable in BF16.
    let bytes: Vec<_> = values.iter().flat_map(|v| ((v.to_bits()>>16) as u16).to_ne_bytes()).collect();
    let mut strides = vec![0; shape.len()]; let mut stride = 1;
    for (i, &d) in shape.iter().enumerate().rev() { strides[i]=stride; stride*=d; }
    TensorBuffer { handle: client.create_from_slice(&bytes), shape: Shape::from(shape.to_vec()),
        strides: Strides::from(strides), dtype: DType::BF16 }
}
fn close(client: &ComputeClient<AscendRuntime>, tensor: TensorBuffer, expected: &[f64], name: &str)
    -> Result<(), Box<dyn std::error::Error>> {
    let bytes = client.read_one(tensor.handle)?;
    let size = if tensor.dtype==DType::BF16 {2} else {4};
    if bytes.len()!=expected.len()*size {return Err(format!("{name}: byte length mismatch").into());}
    for (i,(chunk,&want)) in bytes.chunks_exact(size).zip(expected).enumerate() {
        let actual = if size==2 {f32::from_bits((u16::from_ne_bytes(chunk.try_into().unwrap()) as u32)<<16)}
            else {f32::from_ne_bytes(chunk.try_into().unwrap())} as f64;
        let relative = if size==2 {4e-3} else {5e-4};
        if !actual.is_finite() || (actual-want).abs()>5e-4+relative*want.abs() {
            return Err(format!("{name}[{i}]: {actual} != {want}").into());
        }
    }
    Ok(())
}
fn reference(a:&[f32], b:&[f32], batches:usize, m:usize, n:usize, k:usize,
    ta:Transpose, tb:Transpose)->Vec<f64> {
    let mut out=vec![0.;batches*m*n];
    for batch in 0..batches {for row in 0..m {for col in 0..n {
        out[batch*m*n+row*n+col]=(0..k).map(|q| {
            let ai=batch*m*k+if ta==Transpose::No {row*k+q} else {q*m+row};
            let bi=batch*k*n+if tb==Transpose::No {q*n+col} else {col*k+q};
            a[ai] as f64*b[bi] as f64
        }).sum();
    }}}
    out
}
fn main()->Result<(),Box<dyn std::error::Error>> {
    let toolkit=std::env::var_os("ASCEND_HOME_PATH").ok_or("set ASCEND_HOME_PATH")?;
    let mut options=RuntimeOptions::new(toolkit);
    if let Some(path)=std::env::var_os("RUDA_CANN_LIBRARY") {options.acl_library=path;}
    if let Some(paths)=std::env::var_os("RUDA_CANN_OPERATOR_LIBRARIES") {
        options.operator_libraries=std::env::split_paths(&paths).map(|p|p.into_os_string()).collect();
    }
    // SAFETY: this standalone executable is the sole owner of the ACL context.
    let device=unsafe {AscendRuntime::initialize_exclusive(options)?};
    let client=AscendRuntime::client(&device);
    let (m,n,k)=(32,48,16); let mut cases=0;
    for batches in [1,2] {for ta in [Transpose::No,Transpose::Yes] {for tb in [Transpose::No,Transpose::Yes] {
        let mut ashape=if ta==Transpose::No {vec![m,k]} else {vec![k,m]};
        let mut bshape=if tb==Transpose::No {vec![k,n]} else {vec![n,k]};
        if batches!=1 {ashape.insert(0,batches);bshape.insert(0,batches);}
        let a:Vec<f32>=(0..batches*m*k).map(|i|((i%17) as f32-8.)*0.125).collect();
        let b:Vec<f32>=(0..batches*k*n).map(|i|((i%13) as f32-6.)*0.25).collect();
        let want=reference(&a,&b,batches,m,n,k,ta,tb);
        let da=upload(&client,&ashape,&a);let db=upload(&client,&bshape,&b);
        for dtype in [DType::BF16,DType::F32] {
            let output=AscendRuntime::gemm(&client,da.clone(),db.clone(),ta,tb,dtype)?;
            let shape=if batches==1 {vec![m,n]} else {vec![batches,m,n]};
            if output.shape!=Shape::from(shape) {return Err("matrix output shape mismatch".into());}
            close(&client,output.clone(),&want,"GEMM")?;
            AscendRuntime::gemm_into(&client,da.clone(),db.clone(),ta,tb,output.clone())?;
            close(&client,output,&want,"GEMM reuse")?;
            cases+=1;
            println!("ASCEND_MATRIX_RUNTIME_CASE batches={batches} ta={ta:?} tb={tb:?} dtype={dtype:?} passed=true");
        }
    }}}
    let x:Vec<f32>=(0..m*k).map(|i|((i%17) as f32-8.)*0.125).collect();
    let weight:Vec<f32>=(0..n*k).map(|i|((i%13) as f32-6.)*0.25).collect();
    let dy:Vec<f32>=(0..m*n).map(|i|((i%11) as f32-5.)*0.0625).collect();
    let [dx,dw]=AscendRuntime::linear_nt_backward(&client,upload(&client,&[m,k],&x),
        upload(&client,&[n,k],&weight),upload(&client,&[m,n],&dy))?;
    if &dx.shape[..]!=[m,k] || &dw.shape[..]!=[n,k] {return Err("linear gradient shape mismatch".into());}
    close(&client,dx,&reference(&dy,&weight,1,m,k,n,Transpose::No,Transpose::No),"dX")?;
    close(&client,dw,&reference(&dy,&x,1,n,k,m,Transpose::Yes,Transpose::No),"dWeight")?;
    client.flush()?;
    println!("ASCEND_MATRIX_RUNTIME_DEVICE_OK cases={cases} linear_backward=true");
    Ok(())
}

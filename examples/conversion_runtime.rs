//! Explicit CANN device conversions between RUDA buffers, never a host conversion path.
use rust_ascend::{core::tensor::{DType,Shape,Strides},runtime::{AscendRuntime,ComputeClient,
    RuntimeOptions,TensorBuffer,portable::backend::Runtime}};

fn upload(client:&ComputeClient<AscendRuntime>,values:&[f32])->TensorBuffer {
    let bytes:Vec<_>=values.iter().flat_map(|v|v.to_ne_bytes()).collect();
    TensorBuffer {handle:client.create_from_slice(&bytes),shape:Shape::new([1,values.len()]),
        strides:Strides::from(vec![values.len(),1]),dtype:DType::F32}
}
fn check(client:&ComputeClient<AscendRuntime>,value:TensorBuffer,expected:&[f32])->Result<(),Box<dyn std::error::Error>> {
    let restored=AscendRuntime::cast(client,value,DType::F32)?;
    let bytes=client.read_one(restored.handle)?;
    let actual:Vec<_>=bytes.chunks_exact(4).map(|b|f32::from_ne_bytes(b.try_into().unwrap())).collect();
    if actual!=expected {return Err("device Cast differs from exactly representable values".into());}
    Ok(())
}
fn main()->Result<(),Box<dyn std::error::Error>> {
    let toolkit=std::env::var_os("ASCEND_HOME_PATH").ok_or("set ASCEND_HOME_PATH")?;
    let mut options=RuntimeOptions::new(toolkit);
    if let Some(path)=std::env::var_os("RUDA_CANN_LIBRARY") {options.acl_library=path;}
    if let Some(paths)=std::env::var_os("RUDA_CANN_OPERATOR_LIBRARIES") {
        options.operator_libraries=std::env::split_paths(&paths).map(|p|p.into_os_string()).collect();
    }
    // SAFETY: standalone executable, sole owner of the process-wide ACL lifecycle.
    let device=unsafe {AscendRuntime::initialize_exclusive(options)?};
    let client=AscendRuntime::client(&device);let mut cases=0;
    for n in [0,1,65,1025] {
        // Explicit test data, exactly representable in all three dtypes.
        let values:Vec<f32>=(0..n).map(|i|((i%33) as f32-16.)*0.125).collect();
        let base=upload(&client,&values);
        for source in [DType::F32,DType::F16,DType::BF16] {
            let input=AscendRuntime::cast(&client,base.clone(),source)?;
            for target in [DType::F32,DType::F16,DType::BF16] {
                let output=AscendRuntime::cast(&client,input.clone(),target)?;
                if output.shape!=base.shape || output.dtype!=target {return Err("Cast metadata mismatch".into());}
                check(&client,output.clone(),&values)?;
                AscendRuntime::cast_into(&client,input.clone(),output.clone())?;
                check(&client,output,&values)?;
                check(&client,input.clone(),&values)?;
                cases+=1;
                println!("ASCEND_CONVERSION_RUNTIME_CASE elements={n} source={source:?} target={target:?} passed=true");
            }
            if n!=0 && AscendRuntime::cast_into(&client,input.clone(),input).is_ok() {
                return Err("overlapping Cast storage was accepted".into());
            }
        }
    }
    client.flush()?;
    println!("ASCEND_CONVERSION_RUNTIME_DEVICE_OK cases={cases}");
    Ok(())
}

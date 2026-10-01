use rust_ascend::{
    Ascend, Autodiff,
    runtime::{AscendRuntime, RuntimeOptions},
    tensor::{DType, api::Tensor},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let toolkit = std::env::var_os("ASCEND_HOME_PATH").ok_or("set ASCEND_HOME_PATH")?;
    let mut options = RuntimeOptions::new(toolkit);
    if let Some(path) = std::env::var_os("RUDA_CANN_LIBRARY") {
        options.acl_library = path;
    }
    if let Some(paths) = std::env::var_os("RUDA_CANN_OPERATOR_LIBRARIES") {
        options.operator_libraries = std::env::split_paths(&paths)
            .map(|p| p.into_os_string())
            .collect();
    }
    // Standalone process; no other ACL owner or torch_npu is initialized.
    let device = unsafe { AscendRuntime::initialize_exclusive(options)? };
    let a = Tensor::<Ascend, 1>::from_data([1.0f32, 2.0, 3.0, 4.0], (&device, DType::F32));
    let b = Tensor::<Ascend, 1>::from_data([4.0f32, 3.0, 2.0, 1.0], (&device, DType::F32));
    let out = (a + b) * 2.0;
    let data = out.into_data();
    if data.as_slice::<f32>()? != [10.0f32; 4] {
        return Err("tensor arithmetic mismatch".into());
    }
    let x =
        Tensor::<Autodiff<Ascend>, 1>::from_data([1.0f32, 2.0, 3.0, 4.0], (&device, DType::F32))
            .require_grad();
    let y = x.clone() * x.clone();
    let grads = y.backward();
    let dx = x.grad(&grads).ok_or("missing gradient")?.into_data();
    if dx.as_slice::<f32>()? != [2.0f32, 4.0, 6.0, 8.0] {
        return Err("autodiff gradient mismatch".into());
    }
    println!("ASCEND_RUDA_TENSOR_AUTODIFF_DEVICE_OK");
    Ok(())
}

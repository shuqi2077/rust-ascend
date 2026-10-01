use rust_ascend::{
    compiler::{
        AscendCompiler,
        programs::{self, MapProgram},
    },
    core::{
        ir::{StorageType, UIntKind},
        kernel::KernelDefinition,
    },
    runtime::{
        AscendRuntime, RuntimeOptions,
        portable::{
            backend::Runtime,
            id::KernelId,
            kernel::{KernelMetadata, KernelTask, RudaKernel},
            server::{KernelArguments, RudaCount},
        },
    },
};

struct Add;
impl KernelMetadata for Add {
    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
    fn address_type(&self) -> StorageType {
        UIntKind::U64.into()
    }
}
impl RudaKernel for Add {
    fn define(&self) -> KernelDefinition {
        programs::definition(MapProgram::Add)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let toolkit = std::env::var_os("ASCEND_HOME_PATH")
        .ok_or("set ASCEND_HOME_PATH to the installed CANN SDK")?;
    let mut options = RuntimeOptions::new(toolkit);
    if let Some(path) = std::env::var_os("RUDA_CANN_LIBRARY") {
        options.acl_library = path;
    }
    if let Some(paths) = std::env::var_os("RUDA_CANN_OPERATOR_LIBRARIES") {
        options.operator_libraries = std::env::split_paths(&paths)
            .map(|p| p.into_os_string())
            .collect();
    }
    // Standalone executable: no torch_npu or other ACL owner is initialized.
    let device = unsafe { AscendRuntime::initialize_exclusive(options)? };
    let client = AscendRuntime::client(&device);
    for n in [0usize, 1, 65, 1025] {
        let x: Vec<f32> = (0..n).map(|i| i as f32 * 0.25).collect();
        let y: Vec<f32> = (0..n).map(|i| -(i as f32) * 0.125).collect();
        let encode = |v: &[f32]| v.iter().flat_map(|f| f.to_ne_bytes()).collect::<Vec<u8>>();
        let a = client.create_from_slice(&encode(&x));
        let b = client.create_from_slice(&encode(&y));
        let out = client.empty(n * 4);
        for _ in 0..2 {
            client.launch(
                Box::new(KernelTask::<AscendCompiler, _>::new(Add)),
                RudaCount::Static(n.div_ceil(64) as u32, 1, 1),
                KernelArguments::new()
                    .with_buffer(a.clone().binding())
                    .with_buffer(b.clone().binding())
                    .with_buffer(out.clone().binding()),
            );
            let bytes = client.read_one(out.clone())?;
            let got: Vec<_> = bytes
                .chunks_exact(4)
                .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
                .collect();
            let expected: Vec<_> = x.iter().zip(&y).map(|(a, b)| a + b).collect();
            if got != expected {
                return Err(format!("runtime add mismatch for {n} elements").into());
            }
        }
    }
    client.flush()?;
    println!("ASCEND_COMPUTE_CLIENT_DEVICE_OK");
    Ok(())
}

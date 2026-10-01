use rust_ascend::{
    compiler::{AscendCompiler, AscendOptions, AscendTarget, row_programs::{self, RowProgram}},
    core::{compiler::Compiler, ir::UIntKind, launch::ExecutionMode},
    driver::{CannDevice, tensor::{CannSession, common_ir::CannProgram}},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let forward_dir = args.next().ok_or("usage: layer_norm FORWARD_ARTIFACT BACKWARD_ARTIFACT")?;
    let backward_dir = args.next().ok_or("missing backward artifact directory")?;
    let options = AscendOptions {
        target: Some(AscendTarget::Ascend950DT), elements: 3 * 96, row_width: Some(96),
        ..Default::default()
    };
    let compile = |op| AscendCompiler.compile(
        row_programs::definition(op, 96, 1e-5)?, &options,
        ExecutionMode::Checked, UIntKind::U64.into(),
    );
    let acl = std::env::var("RUDA_CANN_LIBRARY").unwrap_or_else(|_| "libascendcl.so".into());
    let opapi = std::env::var("RUDA_CANN_OPAPI").unwrap_or_else(|_| "libopapi.so".into());
    // SAFETY: standalone process with exclusive ACL ownership and trusted artifacts.
    let session = unsafe { CannSession::open_exclusive(CannDevice::new(0)?, acl, opapi)? };
    let forward = unsafe { CannProgram::load(&session, compile(RowProgram::LayerNorm)?, forward_dir)? };
    let backward = unsafe { CannProgram::load(&session, compile(RowProgram::LayerNormInputBackward)?, backward_dir)? };
    let values: Vec<f32> = (0..288).map(|i| (i % 17) as f32 / 8.0 - 1.0).collect();
    let x = session.from_f32(&[3, 96], &values)?;
    let weight = session.from_f32(&[96], &[1.0; 96])?;
    let bias = session.from_f32(&[96], &[0.0; 96])?;
    let outputs = forward.run(&[&x, &weight, &bias])?;
    let upstream: Vec<f32> = (0..288).map(|i| (i % 7) as f32 / 4.0).collect();
    let dy = session.from_f32(&[3, 96], &upstream)?;
    // Saved device mean and rstd go directly to the input-backward kernel.
    let dx = backward.run(&[&x, &dy, &weight, &outputs[1], &outputs[2]])?;
    println!("y={:?}", &outputs[0].to_f32()?[..8]);
    println!("mean={:?}, rstd={:?}", outputs[1].to_f32()?, outputs[2].to_f32()?);
    println!("dx={:?}", &dx[0].to_f32()?[..8]);
    Ok(())
}

use rust_ascend::{
    compiler::{AscendCompiler, AscendOptions, AscendTarget, row_programs::{self, RowProgram}},
    core::{compiler::Compiler, ir::UIntKind, launch::ExecutionMode},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ir = row_programs::definition(RowProgram::Softmax, 64, 1e-5)?;
    let kernel = AscendCompiler.compile(
        ir,
        &AscendOptions {
            target: Some(AscendTarget::Ascend950DT),
            elements: 3 * 64,
            row_width: Some(64),
            ..Default::default()
        },
        ExecutionMode::Checked,
        UIntKind::U64.into(),
    )?;
    print!("{}", kernel.source());
    Ok(())
}

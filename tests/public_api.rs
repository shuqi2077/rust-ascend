use rust_ascend::{
    compiler::{AscendCompiler, AscendOptions, AscendTarget, row_programs::{self, RowProgram}},
    core::{compiler::Compiler, ir::UIntKind, launch::ExecutionMode},
    driver::tensor::common_ir::CannProgram,
};

#[test]
fn root_entry_exposes_compiler_driver_and_kernels() {
    let ir = row_programs::definition(RowProgram::RmsNorm, 64, 1e-5).unwrap();
    let compiled = AscendCompiler.compile(ir, &AscendOptions {
        target: Some(AscendTarget::Ascend950DT), elements: 192, row_width: Some(64),
        ..Default::default()
    }, ExecutionMode::Checked, UIntKind::U64.into()).unwrap();
    assert_eq!(compiled.bindings()[3].bytes, 12);
    assert!(compiled.source().contains("AscendC::ReduceSum"));
    let _program: Option<CannProgram> = None;
    let specs = rust_ascend::kernels::Spec::all();
    assert_eq!(specs.len(), 18);
    assert!(rust_ascend::kernels::emit(specs[0]).unwrap().contains("asc_mmad"));
}

#[test]
fn compute_client_rms_norm_exposes_forward_and_both_gradients() {
    use rust_ascend::runtime::{AscendRuntime, ComputeClient, TensorBuffer};
    use rust_ascend::driver::CannError;
    let _: fn(&ComputeClient<AscendRuntime>, TensorBuffer, TensorBuffer, f64)
        -> Result<[TensorBuffer; 2], CannError> = AscendRuntime::rms_norm;
    let _: fn(&ComputeClient<AscendRuntime>, TensorBuffer, TensorBuffer, TensorBuffer, TensorBuffer)
        -> Result<[TensorBuffer; 2], CannError> = AscendRuntime::rms_norm_backward;
}

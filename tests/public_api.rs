use rust_ascend::{
    compiler::{AscendCompiler, AscendOptions, AscendTarget, row_programs::{self, RowProgram}},
    core::{compiler::Compiler, ir::UIntKind, launch::ExecutionMode},
    driver::tensor::common_ir::CannProgram,
};

#[test]
fn causal_training_exposes_device_windows_loss_sum_and_total_weight() {
    use rust_ascend::{Ascend,Autodiff,nn,driver::CannError,
        tensor::api::{Tensor,Int},runtime::{AscendRuntime,ComputeClient,TensorBuffer}};
    let _:fn(Tensor<Ascend,3>,nn::TokenWindow)->Result<Tensor<Ascend,2>,CannError>=nn::token_window::<Ascend>;
    let _:fn(Tensor<Autodiff<Ascend>,3>,nn::TokenWindow)->Result<Tensor<Autodiff<Ascend>,2>,CannError>
        =nn::token_window::<Autodiff<Ascend>>;
    let _:fn(Tensor<Autodiff<Ascend>,3,Int>,nn::TokenWindow)->Result<Tensor<Autodiff<Ascend>,2,Int>,CannError>
        =nn::token_window_int::<Autodiff<Ascend>>;
    let _:fn(&ComputeClient<AscendRuntime>,TensorBuffer,nn::TokenWindow)->Result<TensorBuffer,CannError>
        =AscendRuntime::token_window;
    let config=nn::CausalCrossEntropyConfig::default();
    assert_eq!(config.token_chunk_size,32);assert_eq!(config.ignore_index,-100);assert!(config.shift);
}

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

#[test]
fn compute_client_matrix_exposes_native_gemm_and_linear_gradients() {
    use rust_ascend::runtime::{AscendRuntime, ComputeClient, TensorBuffer, Transpose};
    use rust_ascend::core::tensor::DType;
    use rust_ascend::driver::CannError;
    let _: fn(&ComputeClient<AscendRuntime>, TensorBuffer, TensorBuffer, Transpose, Transpose, DType)
        -> Result<TensorBuffer, CannError> = AscendRuntime::gemm;
    let _: fn(&ComputeClient<AscendRuntime>, TensorBuffer, TensorBuffer, Transpose, Transpose, TensorBuffer)
        -> Result<(), CannError> = AscendRuntime::gemm_into;
    let _: fn(&ComputeClient<AscendRuntime>, TensorBuffer, TensorBuffer, TensorBuffer)
        -> Result<[TensorBuffer; 2], CannError> = AscendRuntime::linear_nt_backward;
}

#[test]
fn native_rms_norm_accepts_ruda_tensor_and_autodiff_backends() {
    use rust_ascend::{Ascend, Autodiff, driver::CannError, nn, tensor::api::Tensor};
    let _: fn(Tensor<Ascend, 3>, Tensor<Ascend, 1>, f64) -> Result<Tensor<Ascend, 3>, CannError>
        = nn::rms_norm::<Ascend, 3>;
    let _: fn(Tensor<Autodiff<Ascend>, 3>, Tensor<Autodiff<Ascend>, 1>, f64)
        -> Result<Tensor<Autodiff<Ascend>, 3>, CannError> = nn::rms_norm::<Autodiff<Ascend>, 3>;
}

#[test]
fn native_softmax_exposes_runtime_and_ruda_autodiff_entries() {
    use rust_ascend::{Ascend, Autodiff, driver::CannError, nn,
        runtime::{AscendRuntime, ComputeClient, TensorBuffer}, tensor::api::Tensor};
    let _: fn(&ComputeClient<AscendRuntime>, TensorBuffer) -> Result<TensorBuffer,CannError> = AscendRuntime::softmax;
    let _: fn(&ComputeClient<AscendRuntime>, TensorBuffer) -> Result<TensorBuffer,CannError> = AscendRuntime::log_softmax;
    let _: fn(&ComputeClient<AscendRuntime>, TensorBuffer, TensorBuffer) -> Result<TensorBuffer,CannError> = AscendRuntime::softmax_backward;
    let _: fn(&ComputeClient<AscendRuntime>, TensorBuffer, TensorBuffer) -> Result<TensorBuffer,CannError> = AscendRuntime::log_softmax_backward;
    let _: fn(Tensor<Ascend,3>) -> Result<Tensor<Ascend,3>,CannError> = nn::softmax::<Ascend,3>;
    let _: fn(Tensor<Autodiff<Ascend>,3>) -> Result<Tensor<Autodiff<Ascend>,3>,CannError> = nn::softmax::<Autodiff<Ascend>,3>;
    let _: fn(Tensor<Ascend,3>) -> Result<Tensor<Ascend,3>,CannError> = nn::log_softmax::<Ascend,3>;
    let _: fn(Tensor<Autodiff<Ascend>,3>) -> Result<Tensor<Autodiff<Ascend>,3>,CannError> = nn::log_softmax::<Autodiff<Ascend>,3>;
}

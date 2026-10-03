use rust_ascend::{
    compiler::{AscendCompiler, AscendOptions, AscendTarget, row_programs::{self, RowProgram}},
    core::{compiler::Compiler, ir::UIntKind, launch::ExecutionMode},
    driver::tensor::common_ir::CannProgram,
};

#[test]
fn generic_ruda_modules_and_training_contracts_share_ascend_primitives() {
    use rust_ascend::{RudaAscend, Ascend, Autodiff, nn,
        tensor::{Backend, backend::AutodiffBackend, api::Tensor}};
    fn assert_backend<B: Backend>() {}
    fn assert_autodiff<B: AutodiffBackend>() {}
    fn assert_chunk_loss<B: nn::TokenWindowBackend + nn::NllLossBackend +
        nn::SoftmaxBackend + nn::PiecewiseBackend>() {}
    assert_backend::<RudaAscend>();
    assert_autodiff::<Autodiff<RudaAscend>>();
    assert_chunk_loss::<RudaAscend>();
    assert_chunk_loss::<Autodiff<RudaAscend>>();
    let _: fn(Tensor<RudaAscend,3>, nn::TokenWindow) -> Result<Tensor<RudaAscend,2>, rust_ascend::driver::CannError>
        = nn::token_window::<RudaAscend>;
    let _: Option<rust_ascend::data_parallel::DataParallel<Autodiff<RudaAscend>>> = None;
    let _: Option<rust_ascend::data_parallel::DataParallel<Autodiff<Ascend>>> = None;
    let _: Option<nn::modules::Mhc<Autodiff<RudaAscend>>> = None;
    let _: Option<rust_ascend::collective::tensor_device::TensorDevice<RudaAscend>> = None;
}

#[cfg(feature = "models")]
#[test]
fn original_model_loader_and_causal_adapter_are_available() {
    use rust_ascend::{RudaAscend, Autodiff, models, nn, tensor::api::{Tensor,Int}};
    type B = Autodiff<RudaAscend>;
    fn forward(model: &models::LlamaForCausalLm<B>, tokens: Tensor<B,2,Int>, labels: Tensor<B,2,Int>)
        -> Result<nn::CausalLoss<B>, rust_ascend::driver::CannError> {
        nn::CausalCrossEntropyConfig::default().forward_model(&nn::RudaCausalModel(model), tokens, labels)
    }
    let _ = forward;
    let _loader = |path: &std::path::Path, device: &rust_ascend::runtime::AscendDevice|
        models::load_huggingface_llama::<RudaAscend>(path, device);
}

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

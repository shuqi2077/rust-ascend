use rust_ascend::{
    Autodiff, RudaAscend,
    model::{
        module::{Initializer, Module},
        record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
    },
    nn::modules::{LinearConfig, LoRALinear, LoRALinearConfig},
    optim::{
        AdamWConfig, AdamWState, Fp32MasterOptimizer, Fp32MasterState, GradientsParams, Optimizer,
    },
    runtime::{AscendRuntime, RuntimeOptions},
    tensor::{Backend, DType, FloatDType, api::Tensor},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let toolkit = std::env::var_os("ASCEND_HOME_PATH").ok_or("set ASCEND_HOME_PATH")?;
    let mut options = RuntimeOptions::new(toolkit);
    if let Some(value) = std::env::var_os("RUDA_CANN_LIBRARY") {
        options.acl_library = value;
    }
    if let Some(value) = std::env::var_os("RUDA_CANN_OPERATOR_LIBRARIES") {
        options.operator_libraries = std::env::split_paths(&value)
            .map(|p| p.into_os_string())
            .collect();
    }
    let device = unsafe { AscendRuntime::initialize_exclusive(options)? };
    type B = Autodiff<RudaAscend>;
    for (dtype, module_dtype) in [
        (DType::F16, FloatDType::F16),
        (DType::BF16, FloatDType::BF16),
    ] {
        B::seed(&device, 2077);
        let base = LinearConfig::new(2, 2)
            .with_bias(false)
            .with_initializer(Initializer::Ones)
            .init::<B>(&device)
            .to_dtype(module_dtype);
        let mut model = LoRALinearConfig::new(1, 2.).init(base);
        let base_id = model.base.weight.id;
        let a_id = model.adapter_a.weight.id;
        let b_id = model.adapter_b.weight.id;
        let mut optimizer =
            Fp32MasterOptimizer::new(AdamWConfig::new().build()).init::<B, LoRALinear<B>>();
        for _ in 0..3 {
            let input = Tensor::<B, 2>::from_data([[1., 2.], [2., 1.]], &device).cast(dtype);
            let prediction = model.forward(input);
            assert_eq!(prediction.dtype(), dtype);
            let loss = prediction.cast(DType::F32).square().mean();
            assert!(loss.clone().into_scalar().is_finite());
            let gradients = GradientsParams::from_grads(loss.backward(), &model);
            model = optimizer.step(0.001, model, gradients);
            assert_eq!(model.base.weight.id, base_id);
            assert_eq!(model.adapter_a.weight.id, a_id);
            assert_eq!(model.adapter_b.weight.id, b_id);
            assert_eq!(model.adapter_a.weight.val().dtype(), dtype);
            assert_eq!(model.adapter_b.weight.val().dtype(), dtype);
            assert_eq!(
                model
                    .base
                    .weight
                    .val()
                    .cast(DType::F32)
                    .into_data()
                    .to_vec::<f32>()?,
                vec![1.; 4]
            );
            let record = optimizer.to_record();
            assert_eq!(record.len(), 2);
            assert!(!record.contains_key(&base_id));
            for id in [a_id, b_id] {
                let state: Fp32MasterState<RudaAscend, 2, AdamWState<RudaAscend, 2>> =
                    record[&id].clone().into_state();
                assert_eq!(state.master.dtype(), DType::F32);
                assert_eq!(
                    state.inner.as_ref().unwrap().momentum.moment_1.dtype(),
                    DType::F32
                );
            }
            let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
            let bytes = <BinBytesRecorder<FullPrecisionSettings> as Recorder<B>>::record(
                &recorder,
                record,
                (),
            )?;
            let restored = <BinBytesRecorder<FullPrecisionSettings> as Recorder<B>>::load(
                &recorder, bytes, &device,
            )?;
            optimizer = optimizer.load_record(restored);
        }
        println!(
            "{dtype:?}: original LoRA module, FP32-master AdamW and optimizer record continuation passed"
        );
    }
    Ok(())
}

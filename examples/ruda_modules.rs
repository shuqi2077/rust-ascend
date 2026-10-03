//! Original RUDA mHC, Linear, Muon/AdamW and combined training-record continuation.
//! This short example uses explicit test parameters, not random initialization.
use rust_ascend::{
    Autodiff, RudaAscend,
    model::{
        module::{Initializer, Module, Param},
        record::{BinBytesRecorder, FullPrecisionSettings},
    },
    nn::modules::{Linear, LinearConfig, Mhc, MhcConfig},
    optim::{GradientsAccumulator, GradientsParams, MuonAdamW, MuonAdamWConfig, Optimizer},
    runtime::{AscendRuntime, RuntimeOptions},
    tensor::{Backend, TensorData, api::Tensor},
    training::TrainingRecord,
};

#[derive(Module, Debug)]
struct Connection<B: Backend> {
    connection: Mhc<B>,
    branch: Linear<B>,
}
type AD = Autodiff<RudaAscend>;
type Model = Connection<AD>;
type Optim = MuonAdamW<Model, AD>;
type Saved = TrainingRecord<AD, Model, Optim, f64, u64>;

fn gradients(model: &Model, input: Tensor<AD, 4>) -> GradientsParams {
    let output = model.connection.forward(input, |x| model.branch.forward(x));
    GradientsParams::from_grads(output.square().mean().backward(), model)
}

fn compare<const D: usize>(
    a: Tensor<AD, D>,
    b: Tensor<AD, D>,
) -> Result<(), Box<dyn std::error::Error>> {
    if a.dims() != b.dims() {
        return Err("restored parameter shape changed".into());
    }
    let a = a.into_data().to_vec::<f32>()?;
    let b = b.into_data().to_vec::<f32>()?;
    for (a, b) in a.iter().zip(&b) {
        if !a.is_finite() || !b.is_finite() || (a - b).abs() > 2e-5 + 2e-5 * b.abs() {
            return Err("checkpoint continuation changed the next parameter update".into());
        }
    }
    Ok(())
}

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
    // SAFETY: standalone process owns its initialized device and CANN lifecycle.
    let device = unsafe { AscendRuntime::initialize_exclusive(options)? };
    let mut connection = MhcConfig::new(3).with_streams(2).init::<AD>(&device);
    // The caller supplies the mapping; the original mHC algorithm/config are unchanged.
    connection.mapping = Param::from_tensor(Tensor::from_data(
        TensorData::new(
            (0..48)
                .map(|i| (i % 11) as f32 / 100. - 0.05)
                .collect::<Vec<_>>(),
            [6, 8],
        ),
        &device,
    ));
    let branch = LinearConfig::new(3, 3)
        .with_initializer(Initializer::Constant { value: 0.1 })
        .init::<AD>(&device);
    let mut model = Connection { connection, branch };
    // Roles are explicit: only the branch's hidden matrix goes to Muon.
    let config = MuonAdamWConfig::new();
    let selected = [model.branch.weight.id];
    let mut optimizer = config.init(&model, &selected)?;
    let scheduler = 0.001f64;
    let input = Tensor::<AD, 4>::from_data(
        TensorData::new(
            (0..24)
                .map(|i| (i % 7) as f32 / 8. - 0.375)
                .collect::<Vec<_>>(),
            [1, 4, 2, 3],
        ),
        &device,
    );
    let grads = gradients(&model, input.clone());
    model = optimizer.step(scheduler, model, grads);
    let mut pending = GradientsAccumulator::new();
    pending.accumulate(&model, gradients(&model, input));
    let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
    let bytes =
        Saved::capture(&model, &optimizer, &scheduler, &pending, 1u64)?.save(&recorder, ())?;
    let fresh_optimizer = config.init(&model, &selected)?;
    let saved = Saved::load(&recorder, bytes, &device)?;
    let mut restored = saved.restore(model.clone(), fresh_optimizer, scheduler, &device)?;
    if restored.state != 1 {
        return Err("caller continuation state changed".into());
    }
    let expected = optimizer.step(scheduler, model, pending.grads());
    let actual = restored.optimizer.step(
        restored.scheduler,
        restored.model,
        restored.accumulator.grads(),
    );
    compare(
        expected.connection.mapping.val(),
        actual.connection.mapping.val(),
    )?;
    compare(
        expected.connection.alpha.val(),
        actual.connection.alpha.val(),
    )?;
    compare(expected.connection.bias.val(), actual.connection.bias.val())?;
    compare(expected.branch.weight.val(), actual.branch.weight.val())?;
    compare(
        expected.branch.bias.unwrap().val(),
        actual.branch.bias.unwrap().val(),
    )?;
    println!("original RUDA mHC + Muon/AdamW + pending-gradient record continuation passed");
    Ok(())
}

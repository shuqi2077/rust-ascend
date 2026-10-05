//! One process per NPU. Set RUDA_RANK, RUDA_WORLD_SIZE, RUDA_NPU_DEVICE and
//! RUDA_RENDEZVOUS (rank-zero host:port) equally where appropriate on every rank.
//! RUDA_RENDEZVOUS_ID is the same 32 hex digits on all ranks. CANN/HCCL must be configured.
use ruda_optim::SgdConfig;
use rust_ascend::{
    Autodiff, RudaAscend,
    collective::{
        rank::{TcpRendezvousServer, UniqueId, communicator::RankCommunicator},
        tensor_device::{TensorDevice, TensorDeviceError},
    },
    data_parallel::{DataParallel, MissingGradientPolicy},
    distributed::HcclCommunicator,
    model::module::{Module, Param, ParamId},
    optim::{Fp32MasterOptimizer, GradientsParams, Optimizer},
    runtime::{AscendRuntime, RuntimeOptions},
    tensor::{
        Backend, Bool, DType, FloatDType, Int, TensorCreationOptions, TensorData, api::Tensor,
    },
};
use std::{net::SocketAddr, thread, time::Duration};
type B = Autodiff<RudaAscend>;

#[derive(Module, Debug)]
struct Replica<T: Backend> {
    weight: Param<Tensor<T, 1>>,
    counter: Param<Tensor<T, 1, Int>>,
    flags: Param<Tensor<T, 1, Bool>>,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rank: u32 = std::env::var("RUDA_RANK")?.parse()?;
    let world: u32 = std::env::var("RUDA_WORLD_SIZE")?.parse()?;
    if world < 2 || rank >= world {
        return Err("requires at least two ranks, each with one NPU".into());
    }
    let address: SocketAddr = std::env::var("RUDA_RENDEZVOUS")?.parse()?;
    let hex = std::env::var("RUDA_RENDEZVOUS_ID")?;
    if hex.len() != 32 || !hex.is_ascii() {
        return Err("RUDA_RENDEZVOUS_ID must contain 32 hex digits".into());
    }
    let mut id = [0u8; 16];
    for (i, byte) in id.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16)?;
    }
    let id = UniqueId::from_bytes(id);
    let mut options =
        RuntimeOptions::new(std::env::var_os("ASCEND_HOME_PATH").ok_or("set ASCEND_HOME_PATH")?);
    options.device = std::env::var("RUDA_NPU_DEVICE")?.parse()?;
    if let Some(value) = std::env::var_os("RUDA_CANN_LIBRARY") {
        options.acl_library = value;
    }
    if let Some(value) = std::env::var_os("RUDA_CANN_OPERATOR_LIBRARIES") {
        options.operator_libraries = std::env::split_paths(&value)
            .map(|p| p.into_os_string())
            .collect();
    }
    let device = unsafe { AscendRuntime::initialize_exclusive(options)? };
    let coordinator = if rank == 0 {
        let server = TcpRendezvousServer::bind(address, id, world)?;
        Some(thread::spawn(move || server.run()))
    } else {
        None
    };
    let control = RankCommunicator::connect(
        || Ok::<_, TensorDeviceError>(TensorDevice::<RudaAscend>::new(device.clone())),
        address,
        id,
        rank,
        world,
        Duration::from_secs(120),
        "native-hccl-training",
    )?;
    let library = std::env::var_os("RUDA_HCCL_LIBRARY").unwrap_or_else(|| "libhccl.so".into());
    let communicator = unsafe { HcclCommunicator::initialize(control, library)? };
    let root = world - 1;
    for dtype in [FloatDType::F16, FloatDType::BF16] {
        let model = Replica {
            weight: Param::from_tensor(Tensor::<B, 1>::full([2], rank + 1, &device).cast(dtype)),
            counter: Param::initialized(
                ParamId::new(),
                Tensor::<B, 1, Int>::from_data(
                    TensorData::from([9_007_199_254_740_993_i64 + rank as i64]),
                    TensorCreationOptions::<B>::new(device.clone()).with_dtype(DType::I64),
                ),
            ),
            flags: Param::initialized(
                ParamId::new(),
                Tensor::<B, 1, Bool>::from_data([rank == root, rank != root], &device),
            ),
        };
        let (ddp, model) = DataParallel::<B, HcclCommunicator>::initialize_with_buffers(
            communicator.clone(),
            model,
            root,
        )?;
        assert_eq!(
            model.counter.val().into_data().to_vec::<i64>()?,
            vec![9_007_199_254_740_993_i64 + root as i64]
        );
        assert_eq!(
            model.flags.val().into_data().to_vec::<u8>()?,
            vec![1, 0]
        );
        let local_weight = rank as u64 + 1;
        let loss =
            model.weight.val().cast(FloatDType::F32).sum() * (local_weight * local_weight) as f32;
        let gradients = GradientsParams::from_grads(loss.backward(), &model);
        let gradients = ddp.reduce_fp32(
            &model,
            gradients,
            local_weight,
            MissingGradientPolicy::Error,
        )?;
        assert_eq!(
            gradients.global_weight,
            world as u64 * (world as u64 + 1) / 2
        );
        let expected_gradient = (2 * world + 1) as f32 / 3.;
        for value in gradients
            .gradients
            .get::<RudaAscend, 1>(model.weight.id)
            .ok_or("missing reduced gradient")?
            .into_data()
            .to_vec::<f32>()?
        {
            assert!((value - expected_gradient).abs() < 0.01);
        }
        let mut optimizer =
            Fp32MasterOptimizer::new(SgdConfig::new().build()).init::<B, Replica<B>>();
        let model = optimizer.step(0.125, model, gradients.gradients);
        let expected = world as f32 - 0.125 * expected_gradient;
        for value in model
            .weight
            .val()
            .cast(FloatDType::F32)
            .into_data()
            .to_vec::<f32>()?
        {
            assert!((value - expected).abs() < 0.03);
        }
        assert_eq!(model.weight.val().dtype(), dtype.into());
        assert_eq!(
            model.counter.val().into_data().to_vec::<i64>()?,
            vec![9_007_199_254_740_993_i64 + root as i64]
        );
        println!(
            "rank {rank}/{world} {dtype:?}: native HCCL broadcast, weighted FP32 reduction and original master SGD passed"
        );
    }
    drop(communicator);
    B::sync(&device)?;
    if let Some(coordinator) = coordinator {
        coordinator
            .join()
            .map_err(|_| "rendezvous thread panicked")??;
    }
    Ok(())
}

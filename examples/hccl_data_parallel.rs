//! One process per NPU. Set RUDA_RANK, RUDA_WORLD_SIZE, RUDA_NPU_DEVICE and
//! RUDA_RENDEZVOUS (rank-zero host:port) equally where appropriate on every rank.
//! RUDA_RENDEZVOUS_ID is the same 32 hex digits on all ranks. CANN/HCCL must be configured.
use ruda_optim::SgdConfig;
use rust_ascend::{
    Autodiff, RudaAscend,
    collective::{
        ReduceOperation,
        rank::{TcpRendezvousServer, UniqueId, communicator::RankCommunicator},
        tensor_device::{TensorDevice, TensorDeviceError},
    },
    data_parallel::{DataParallel, MissingGradientPolicy},
    distributed::HcclCommunicator,
    model::module::{Module, Param, ParamId},
    optim::{Fp32MasterOptimizer, GradientsParams, Optimizer},
    runtime::{AscendDevice, AscendRuntime, HcclReduceOp, RuntimeOptions},
    tensor::{
        Backend, DType, FloatDType, IntDType, TensorData,
        api::{Bool, Int, Tensor, TensorCreationOptions},
    },
};
use std::{net::SocketAddr, thread, time::Duration};
type B = Autodiff<RudaAscend>;

#[derive(Module, Debug)]
struct Replica<B: Backend> {
    weight: Param<Tensor<B, 1>>,
    counter: Param<Tensor<B, 1, Int>>,
    flags: Param<Tensor<B, 1, Bool>>,
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
        let server = TcpRendezvousServer::bind(address, id, world as usize)?;
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
    check_sharded_collectives(&communicator, &device)?;
    check_sharded_gradients(&communicator, &device)?;
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
        assert_eq!(model.flags.val().into_data().to_vec::<u8>()?, vec![1, 0]);
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

fn check_sharded_gradients(
    communicator: &HcclCommunicator,
    device: &AscendDevice,
) -> Result<(), Box<dyn std::error::Error>> {
    use rust_ascend::distributed::{all_gather, reduce_scatter_mean, reduce_scatter_sum};
    let rank = communicator.rank() as usize;
    let world = communicator.world_size() as usize;
    let rank_sum = (world * (world + 1) / 2) as f32;
    for dtype in [FloatDType::F32, FloatDType::F16, FloatDType::BF16] {
        let input = Tensor::<B, 2>::from_data([[rank as f32 + 1., 2.], [3., 4.]], device)
            .cast(dtype)
            .swap_dims(0, 1)
            .detach()
            .require_grad();
        let output = all_gather(input.clone(), communicator.clone())?;
        let weights = Tensor::<B, 2>::full([2 * world, 2], rank + 1, device).cast(dtype);
        let gradients = (output * weights).sum().backward();
        let gradient = input
            .grad(&gradients)
            .ok_or("missing all-gather gradient")?;
        assert_eq!(gradient.dtype(), dtype.into());
        assert_eq!(gradient.dims(), [2, 2]);
        for actual in gradient.cast(FloatDType::F32).into_data().to_vec::<f32>()? {
            assert!((actual - rank_sum).abs() <= 0.02 * (1. + rank_sum.abs()));
        }
        for mean in [false, true] {
            let input = Tensor::<B, 2>::full([2, 2 * world], rank + 1, device)
                .cast(dtype)
                .swap_dims(0, 1)
                .detach()
                .require_grad();
            let output = if mean {
                reduce_scatter_mean(input.clone(), communicator.clone())?
            } else {
                reduce_scatter_sum(input.clone(), communicator.clone())?
            };
            assert_eq!(output.dims(), [2, 2]);
            let weights = Tensor::<B, 2>::full([2, 2], rank + 1, device).cast(dtype);
            let gradients = (output * weights).sum().backward();
            let gradient = input
                .grad(&gradients)
                .ok_or("missing reduce-scatter gradient")?;
            assert_eq!(gradient.dtype(), dtype.into());
            assert_eq!(gradient.dims(), [2 * world, 2]);
            let expected = (0..world)
                .flat_map(|rank| {
                    let value = (rank + 1) as f32 / if mean { world as f32 } else { 1. };
                    [value; 4]
                })
                .collect::<Vec<_>>();
            let values = gradient.cast(FloatDType::F32).into_data().to_vec::<f32>()?;
            for (actual, expected) in values.iter().zip(&expected) {
                assert!((actual - expected).abs() <= 0.02 * (1. + expected.abs()));
            }
        }
    }
    println!("rank {rank}/{world}: native HCCL sharded tensors with shared RUDA backward passed");
    Ok(())
}

fn check_sharded_collectives(
    communicator: &HcclCommunicator,
    device: &AscendDevice,
) -> Result<(), Box<dyn std::error::Error>> {
    let rank = communicator.rank();
    let world = communicator.world_size() as usize;
    for dtype in [FloatDType::F32, FloatDType::F16, FloatDType::BF16] {
        let input = Tensor::<RudaAscend, 2>::from_data([[rank as f32 + 1., 2.], [3., 4.]], device)
            .cast(dtype)
            .swap_dims(0, 1);
        let gathered =
            Tensor::<RudaAscend, 2>::from_primitive(rust_ascend::tensor::TensorPrimitive::Float(
                communicator.all_gather_float(input.clone().into_primitive().tensor())?,
            ));
        assert_eq!(gathered.dims(), [2 * world, 2]);
        assert_eq!(gathered.dtype(), dtype.into());
        let expected = (0..world)
            .flat_map(|rank| [rank as f32 + 1., 3., 2., 4.])
            .collect::<Vec<_>>();
        assert_eq!(
            gathered
                .clone()
                .cast(FloatDType::F32)
                .into_data()
                .to_vec::<f32>()?,
            expected
        );
        for operation in [ReduceOperation::Sum, ReduceOperation::Mean] {
            let shard = Tensor::<RudaAscend, 2>::from_primitive(
                rust_ascend::tensor::TensorPrimitive::Float(
                    communicator.reduce_scatter_float(
                        gathered.clone().into_primitive().tensor(),
                        operation,
                    )?,
                ),
            );
            assert_eq!(shard.dims(), [2, 2]);
            assert_eq!(shard.dtype(), dtype.into());
            let scale = if operation == ReduceOperation::Sum {
                world as f32
            } else {
                1.
            };
            let expected = [rank as f32 + 1., 3., 2., 4.].map(|value| value * scale);
            assert_eq!(
                shard.cast(FloatDType::F32).into_data().to_vec::<f32>()?,
                expected.to_vec()
            );
        }
        assert_eq!(
            gathered.cast(FloatDType::F32).into_data().to_vec::<f32>()?,
            expected
        );
        assert_eq!(
            input.cast(FloatDType::F32).into_data().to_vec::<f32>()?,
            vec![rank as f32 + 1., 3., 2., 4.]
        );
    }
    for dtype in [IntDType::I32, IntDType::I64] {
        let large = if dtype == IntDType::I64 {
            9_007_199_254_740_993_i64
        } else {
            16_777_217_i64
        };
        let input = Tensor::<RudaAscend, 2, Int>::from_data(
            TensorData::new(vec![large + rank as i64, -3, 7, 2], [2, 2]),
            TensorCreationOptions::<RudaAscend>::new(device.clone()).with_dtype(dtype.into()),
        )
        .swap_dims(0, 1);
        let gathered = Tensor::<RudaAscend, 2, Int>::from_primitive(
            communicator.all_gather_int(input.clone().into_primitive())?,
        );
        assert_eq!(gathered.dims(), [2 * world, 2]);
        assert_eq!(gathered.dtype(), dtype.into());
        let expected = (0..world)
            .flat_map(|rank| [large + rank as i64, 7, -3, 2])
            .collect::<Vec<_>>();
        assert_eq!(
            gathered
                .clone()
                .cast(IntDType::I64)
                .into_data()
                .to_vec::<i64>()?,
            expected
        );
        let shard = Tensor::<RudaAscend, 2, Int>::from_primitive(
            communicator
                .reduce_scatter_int(gathered.clone().into_primitive(), HcclReduceOp::Minimum)?,
        );
        assert_eq!(shard.dims(), [2, 2]);
        assert_eq!(shard.dtype(), dtype.into());
        assert_eq!(
            shard.cast(IntDType::I64).into_data().to_vec::<i64>()?,
            vec![large + rank as i64, 7, -3, 2]
        );
        assert_eq!(
            gathered.cast(IntDType::I64).into_data().to_vec::<i64>()?,
            expected
        );
        assert_eq!(
            input.cast(IntDType::I64).into_data().to_vec::<i64>()?,
            vec![large + rank as i64, 7, -3, 2]
        );
        let empty = Tensor::<RudaAscend, 2, Int>::empty([2, 0], device).cast(dtype);
        let gathered = communicator.all_gather_int(empty.into_primitive())?;
        assert_eq!(
            Tensor::<RudaAscend, 2, Int>::from_primitive(gathered.clone()).dims(),
            [2 * world, 0]
        );
        let shard = communicator.reduce_scatter_int(gathered, HcclReduceOp::Sum)?;
        assert_eq!(
            Tensor::<RudaAscend, 2, Int>::from_primitive(shard).dims(),
            [2, 0]
        );
    }
    println!(
        "rank {rank}/{world}: native sharded collectives preserve dtype, rank order, views, empty axes and input snapshots"
    );
    Ok(())
}

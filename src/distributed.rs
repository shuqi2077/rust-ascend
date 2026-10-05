//! Native HCCL payload transport for RUDA's original replicated training contracts.
use crate::{
    RudaAscend,
    backend::{buffer, wrap},
    runtime::{AscendDevice, HcclContext, HcclReduceOp, HcclRootInfo},
};
use ruccl::{
    ReduceOperation,
    rank::communicator::RankCommunicator,
    tensor_device::{TensorDevice, TensorDeviceError},
};
use ruda_optim::data_parallel::{DataParallelCommunicator, DataParallelError};
use ruda_tensor::{DType, TensorMetadata, ops::FloatTensorOps};
use ruda_tensor_device::RudaTensor;
use std::ffi::OsStr;

type Primitive = RudaTensor<crate::runtime::AscendRuntime>;
type Control = RankCommunicator<TensorDevice<RudaAscend>>;

pub use ruda_autodiff::collective::{
    all_gather, all_gather_dim, reduce_scatter_mean, reduce_scatter_mean_dim, reduce_scatter_sum,
    reduce_scatter_sum_dim,
};

impl ruda_tensor::collective::TensorCollective<RudaAscend> for HcclCommunicator {
    type Error = TensorDeviceError;

    fn world_size(&self) -> u32 {
        HcclCommunicator::world_size(self)
    }

    fn all_gather_float(&self, value: Primitive) -> Result<Primitive, Self::Error> {
        HcclCommunicator::all_gather_float(self, value)
    }

    fn reduce_scatter_sum(&self, value: Primitive) -> Result<Primitive, Self::Error> {
        self.reduce_scatter_float(value, ReduceOperation::Sum)
    }
}

/// HCCL broadcasts/reductions on NPU memory, with ruCCL TCP for training metadata only.
/// Use one process per NPU; all ranks must enter operations in matching order.
#[derive(Clone, Debug)]
pub struct HcclCommunicator {
    control: Control,
    native: crate::runtime::HcclCommunicator,
}
impl HcclCommunicator {
    /// Initialize HCCL using the existing rank/device assignment and TCP rendezvous.
    /// Tensor payloads do not pass through this control communicator.
    ///
    /// # Safety
    /// `library` must identify a trusted HCCL library compatible with the CANN SDK.
    pub unsafe fn initialize(
        control: Control,
        library: impl AsRef<OsStr>,
    ) -> Result<Self, DataParallelError> {
        let prepared = unsafe { HcclContext::open(control.execution().device(), library) }
            .and_then(|context| {
                let bytes = if control.rank() == 0 {
                    context.root_info()?.as_bytes().to_vec()
                } else {
                    Vec::new()
                };
                Ok((context, bytes))
            });
        let payload = match &prepared {
            Ok((_, bytes)) => {
                let mut payload = vec![0];
                payload.extend_from_slice(bytes);
                payload
            }
            Err(error) => failure_bytes(error),
        };
        let replies = control.all_gather_bytes(payload)?;
        check_replies(&replies, control.world_size())?;
        let root = HcclRootInfo::from_bytes(&replies[0][1..]).map_err(collective_error)?;
        let (context, _) = prepared.map_err(collective_error)?;
        let native = context.initialize(root, control.rank(), control.world_size());
        let payload = match &native {
            Ok(_) => vec![0],
            Err(error) => failure_bytes(error),
        };
        let replies = control.all_gather_bytes(payload)?;
        check_replies(&replies, control.world_size())?;
        Ok(Self {
            control,
            native: native.map_err(collective_error)?,
        })
    }
    pub fn rank(&self) -> u32 {
        self.native.rank()
    }
    pub fn world_size(&self) -> u32 {
        self.native.world_size()
    }
    pub fn device(&self) -> &AscendDevice {
        self.native.device()
    }
    /// Gather floating tensors along axis zero, in rank order, on NPU memory.
    pub fn all_gather_float(&self, value: Primitive) -> Result<Primitive, TensorDeviceError> {
        self.check_float(&value)?;
        let output = self
            .native
            .all_gather(&value.client, buffer(value.clone()))
            .map_err(collective_error)?;
        Ok(wrap(&value, output))
    }
    /// Gather I32/I64 tensors without floating-point conversion or host staging.
    pub fn all_gather_int(&self, value: Primitive) -> Result<Primitive, TensorDeviceError> {
        self.check_int(&value)?;
        let output = self
            .native
            .all_gather(&value.client, buffer(value.clone()))
            .map_err(collective_error)?;
        Ok(wrap(&value, output))
    }
    /// Reduce floating tensors and return this rank's equal axis-zero shard.
    pub fn reduce_scatter_float(
        &self,
        value: Primitive,
        operation: ReduceOperation,
    ) -> Result<Primitive, TensorDeviceError> {
        self.check_float(&value)?;
        let output = self
            .native
            .reduce_scatter(&value.client, buffer(value.clone()), HcclReduceOp::Sum)
            .map_err(collective_error)?;
        let output = wrap(&value, output);
        Ok(if operation == ReduceOperation::Mean {
            RudaAscend::float_div_scalar(output, (self.world_size() as f32).into())
        } else {
            output
        })
    }
    /// Reduce I32/I64 tensors using native sum/product/min/max and return a rank shard.
    pub fn reduce_scatter_int(
        &self,
        value: Primitive,
        operation: HcclReduceOp,
    ) -> Result<Primitive, TensorDeviceError> {
        self.check_int(&value)?;
        let output = self
            .native
            .reduce_scatter(&value.client, buffer(value.clone()), operation)
            .map_err(collective_error)?;
        Ok(wrap(&value, output))
    }
    fn check_float(&self, value: &Primitive) -> Result<(), TensorDeviceError> {
        self.check_device(value)?;
        if !matches!(value.dtype(), DType::F32 | DType::F16 | DType::BF16) {
            return Err(TensorDeviceError::UnsupportedDType(value.dtype()));
        }
        Ok(())
    }
    fn check_int(&self, value: &Primitive) -> Result<(), TensorDeviceError> {
        self.check_device(value)?;
        if !matches!(value.dtype(), DType::I32 | DType::I64) {
            return Err(TensorDeviceError::UnsupportedDType(value.dtype()));
        }
        Ok(())
    }
    fn check_device(&self, value: &Primitive) -> Result<(), TensorDeviceError> {
        if &value.device != self.device() {
            Err(TensorDeviceError::DeviceMismatch)
        } else {
            Ok(())
        }
    }
}
fn failure_bytes(error: &impl std::fmt::Display) -> Vec<u8> {
    let mut bytes = vec![1];
    bytes.extend_from_slice(error.to_string().as_bytes());
    bytes
}
fn check_replies(replies: &[Vec<u8>], world_size: u32) -> Result<(), DataParallelError> {
    if replies.len() != world_size as usize {
        return Err(DataParallelError::Contract(
            "HCCL rank metadata count mismatch".into(),
        ));
    }
    for (rank, reply) in replies.iter().enumerate() {
        if reply.first() != Some(&0) {
            return Err(DataParallelError::Collective(TensorDeviceError::Data(
                format!(
                    "HCCL rank {rank}: {}",
                    String::from_utf8_lossy(reply.get(1..).unwrap_or_default())
                ),
            )));
        }
    }
    Ok(())
}
fn collective_error(error: impl std::fmt::Display) -> TensorDeviceError {
    TensorDeviceError::Data(error.to_string())
}

impl DataParallelCommunicator<RudaAscend> for HcclCommunicator {
    fn rank(&self) -> u32 {
        self.rank()
    }
    fn world_size(&self) -> u32 {
        self.world_size()
    }
    fn device(&self) -> &AscendDevice {
        self.device()
    }
    fn all_gather_bytes(&self, payload: Vec<u8>) -> Result<Vec<Vec<u8>>, DataParallelError> {
        self.control.all_gather_bytes(payload)
    }
    fn broadcast_float(&self, value: Primitive, root: u32) -> Result<Primitive, TensorDeviceError> {
        self.check_device(&value)?;
        if !matches!(value.dtype(), DType::F32 | DType::F16 | DType::BF16) {
            return Err(TensorDeviceError::UnsupportedDType(value.dtype()));
        }
        let output = self
            .native
            .broadcast(&value.client, buffer(value.clone()), root)
            .map_err(collective_error)?;
        Ok(wrap(&value, output))
    }
    fn broadcast_int(&self, value: Primitive, root: u32) -> Result<Primitive, TensorDeviceError> {
        self.check_device(&value)?;
        if !matches!(value.dtype(), DType::I32 | DType::I64) {
            return Err(TensorDeviceError::UnsupportedDType(value.dtype()));
        }
        let output = self
            .native
            .broadcast(&value.client, buffer(value.clone()), root)
            .map_err(collective_error)?;
        Ok(wrap(&value, output))
    }
    fn all_reduce_float(
        &self,
        value: Primitive,
        operation: ReduceOperation,
    ) -> Result<Primitive, TensorDeviceError> {
        self.check_device(&value)?;
        if !matches!(value.dtype(), DType::F32 | DType::F16 | DType::BF16) {
            return Err(TensorDeviceError::UnsupportedDType(value.dtype()));
        }
        let output = self
            .native
            .all_reduce(&value.client, buffer(value.clone()), HcclReduceOp::Sum)
            .map_err(collective_error)?;
        let output = wrap(&value, output);
        Ok(if operation == ReduceOperation::Mean {
            RudaAscend::float_div_scalar(output, (self.world_size() as f32).into())
        } else {
            output
        })
    }
}

//! RUDA compute-client integration for Ascend's checked common-IR compiler.
//! CANN objects stay on one owning thread; callers exchange allocation IDs only.
mod build;
mod worker;
mod normalization;
mod matrix;
mod rows;
use crate::CannError;
use ruda_core::{
    backtrace::BackTrace,
    bytes::Bytes,
    device::{Device, DeviceId, DeviceService, ServerUtilitiesHandle},
    future::DynFut,
    ir::{
        AddressType, DeviceProperties, FloatKind, HardwareProperties, MemoryDeviceProperties,
        TargetProperties,
        features::{Features, TypeUsage},
    },
    profile::{ProfileDuration, TimingMethod},
    stream_id::StreamId,
    tensor::{Shape, Strides},
};
pub use ruda_runtime::runtime as portable;
use ruda_runtime::runtime::{
    allocator::ContiguousMemoryLayoutPolicy,
    backend::Runtime,
    logging::ServerLogger,
    memory_management::{
        ManagedMemoryHandle, MemoryAllocationMode, MemoryConfiguration, MemoryManagement,
        MemoryManagementOptions, MemoryUsage,
    },
    server::*,
    storage::{ComputeStorage, ManagedResource, StorageHandle, StorageId, StorageUtilization},
    timestamp_profiler::TimestampProfiler,
};
pub use ruda_runtime::runtime::{client::ComputeClient, compiler::RudaTask};
pub use ruda_runtime::runtime::normalization::TensorBuffer;
pub use crate::tensor::deepgemm::Transpose;
use rust_ascend_compiler::ascend::{AscendCompiler, AscendOptions, AscendTarget};
use std::{
    ffi::OsString,
    path::PathBuf,
    sync::{Arc, Mutex, OnceLock},
    time::Duration,
};
pub use worker::AscendResource;
use worker::Worker;

type Result<T> = std::result::Result<T, CannError>;
fn error(e: impl std::fmt::Display) -> CannError {
    CannError::InvalidTensor(e.to_string())
}
fn server_error(e: impl std::fmt::Display) -> ServerError {
    ServerError::Generic {
        reason: e.to_string(),
        backtrace: BackTrace::capture(),
    }
}
fn io_error(e: impl std::fmt::Display) -> IoError {
    IoError::Unknown {
        description: e.to_string(),
        backtrace: BackTrace::capture(),
    }
}

#[derive(Debug, Clone)]
pub struct RuntimeOptions {
    pub device: u16,
    pub toolkit: PathBuf,
    pub acl_library: OsString,
    pub operator_libraries: Vec<OsString>,
    pub compile_timeout: Duration,
}
impl RuntimeOptions {
    pub fn new(toolkit: impl Into<PathBuf>) -> Self {
        Self {
            device: 0,
            toolkit: toolkit.into(),
            acl_library: "libascendcl.so".into(),
            operator_libraries: vec![
                "libnnopbase.so".into(),
                "libopapi_math.so".into(),
                "libopapi_nn.so".into(),
            ],
            compile_timeout: Duration::from_secs(120),
        }
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub struct AscendDevice {
    ordinal: u16,
}
impl AscendDevice {
    pub fn ordinal(&self) -> u16 {
        self.ordinal
    }
}
impl ruda_tensor::DeviceOps for AscendDevice {}
impl Device for AscendDevice {
    fn from_id(id: DeviceId) -> Self {
        assert_eq!(id.type_id, 0, "unsupported Ascend device type");
        Self {
            ordinal: id.index_id,
        }
    }
    fn to_id(&self) -> DeviceId {
        DeviceId {
            type_id: 0,
            index_id: self.ordinal,
        }
    }
}

static WORKER: OnceLock<(u16, Worker)> = OnceLock::new();
static INITIALIZE: Mutex<()> = Mutex::new(());

#[derive(Debug, Clone)]
pub struct AscendRuntime;
impl AscendRuntime {
    /// Native FP32 Softmax along the last dimension (width 32..4096, multiple of 32).
    pub fn softmax(client: &ComputeClient<Self>, input: TensorBuffer) -> Result<TensorBuffer> {
        rows::softmax(client, input, false)
    }

    /// Native FP32 LogSoftmax along the last dimension, preserving all leading dimensions.
    pub fn log_softmax(client: &ComputeClient<Self>, input: TensorBuffer) -> Result<TensorBuffer> {
        rows::softmax(client, input, true)
    }

    /// Native Softmax input gradient from saved forward output and upstream gradient.
    pub fn softmax_backward(client: &ComputeClient<Self>, output: TensorBuffer, grad: TensorBuffer)
        -> Result<TensorBuffer> {
        rows::softmax_backward(client, output, grad, false)
    }

    /// Native LogSoftmax input gradient from saved log-probabilities and upstream gradient.
    pub fn log_softmax_backward(client: &ComputeClient<Self>, output: TensorBuffer, grad: TensorBuffer)
        -> Result<TensorBuffer> {
        rows::softmax_backward(client, output, grad, true)
    }

    /// Native BF16 GEMM on contiguous rank-2 or matching rank-3 TensorBuffers.
    /// M/N/K must be positive multiples of 16. Output may be BF16 or FP32.
    /// Rust-authored matrix kernels are JIT-built and cached on the owning device thread.
    pub fn gemm(client: &ComputeClient<Self>, a: TensorBuffer, b: TensorBuffer,
        ta: Transpose, tb: Transpose, dtype: ruda_core::tensor::DType) -> Result<TensorBuffer> {
        matrix::gemm(client, a, b, ta, tb, dtype)
    }

    /// Execute GEMM into reusable output storage. Overlapping input/output is rejected.
    pub fn gemm_into(client: &ComputeClient<Self>, a: TensorBuffer, b: TensorBuffer,
        ta: Transpose, tb: Transpose, out: TensorBuffer) -> Result<()> {
        matrix::gemm_into(client, a, b, ta, tb, out)
    }

    /// Native rank-2 linear backward for Y = X W^T. Returns [BF16 dX, FP32 dWeight].
    /// X, W and dY must be BF16. This is an explicit runtime call, not autograd registration.
    pub fn linear_nt_backward(client: &ComputeClient<Self>, input: TensorBuffer,
        weight: TensorBuffer, grad: TensorBuffer) -> Result<[TensorBuffer; 2]> {
        matrix::linear_nt_backward(client, input, weight, grad)
    }

    /// Native common-IR RMSNorm on contiguous FP32 tensors. Returns [output, rstd].
    /// The last dimension must be 32..4096 and divisible by 32.
    pub fn rms_norm(client: &ComputeClient<Self>, input: TensorBuffer,
        weight: TensorBuffer, epsilon: f64) -> Result<[TensorBuffer; 2]> {
        normalization::rms_forward(client, input, weight, epsilon)
    }

    /// Native first-order [input gradient, weight gradient], using saved rstd.
    /// The weight gradient sums all leading rows on-device; empty batches yield zeros.
    pub fn rms_norm_backward(client: &ComputeClient<Self>, input: TensorBuffer,
        weight: TensorBuffer, grad: TensorBuffer, rstd: TensorBuffer) -> Result<[TensorBuffer; 2]> {
        normalization::rms_backward(client, input, weight, grad, rstd)
    }

    /// Initialize one process-owned 950DT device before calling `client`.
    /// The registered worker is retained for the process lifetime.
    ///
    /// # Safety
    /// This runtime must exclusively own ACL initialization and the selected
    /// device; do not combine it with torch_npu or an externally initialized ACL
    /// process. SDK binaries and compiler paths must be trusted and ABI-compatible.
    pub unsafe fn initialize_exclusive(options: RuntimeOptions) -> Result<AscendDevice> {
        let _guard = INITIALIZE.lock().map_err(error)?;
        if WORKER.get().is_some() {
            return Err(error("Ascend runtime already initialized"));
        }
        let ordinal = options.device;
        let worker = unsafe { Worker::start(options)? };
        WORKER
            .set((ordinal, worker))
            .map_err(|_| error("Ascend runtime already initialized"))?;
        Ok(AscendDevice { ordinal })
    }
}
impl Runtime for AscendRuntime {
    type Compiler = AscendCompiler;
    type Server = AscendServer;
    type Device = AscendDevice;
    fn has_native_layer_norm() -> bool { true }
    fn layer_norm(
        client: &ComputeClient<Self>, input: portable::normalization::TensorBuffer,
        weight: portable::normalization::TensorBuffer, bias: Option<portable::normalization::TensorBuffer>,
        epsilon: f64,
    ) -> [portable::normalization::TensorBuffer; 3] {
        normalization::forward(client, input, weight, bias, epsilon).expect("Ascend LayerNorm failed")
    }
    fn layer_norm_backward(
        client: &ComputeClient<Self>, input: portable::normalization::TensorBuffer,
        weight: portable::normalization::TensorBuffer, grad: portable::normalization::TensorBuffer,
        mean: portable::normalization::TensorBuffer, rstd: portable::normalization::TensorBuffer,
    ) -> [portable::normalization::TensorBuffer; 3] {
        normalization::backward(client, input, weight, grad, mean, rstd).expect("Ascend LayerNorm backward failed")
    }
    fn client(device: &Self::Device) -> ComputeClient<Self> {
        ComputeClient::load(device)
    }
    fn name(_: &ComputeClient<Self>) -> &'static str {
        "ascend-950dt"
    }
    fn max_ruda_count() -> (u32, u32, u32) {
        (u32::MAX, 1, 1)
    }
    fn can_read_tensor(shape: &Shape, strides: &Strides) -> bool {
        contiguous(shape, strides)
    }
    fn target_properties() -> TargetProperties {
        TargetProperties {
            mma: Default::default(),
        }
    }
    fn enumerate_devices(type_id: u16, info: &u16) -> Vec<DeviceId> {
        if type_id == 0 {
            vec![DeviceId {
                type_id: 0,
                index_id: *info,
            }]
        } else {
            vec![]
        }
    }
}

#[derive(Debug)]
pub struct AscendStorage {
    worker: Worker,
}
impl ComputeStorage for AscendStorage {
    type Resource = AscendResource;
    fn alignment(&self) -> usize {
        32
    }
    fn get(&mut self, handle: &StorageHandle) -> AscendResource {
        AscendResource {
            id: handle.id,
            offset: handle.offset().try_into().expect("offset exceeds usize"),
            size: handle.size().try_into().expect("size exceeds usize"),
        }
    }
    fn alloc(&mut self, size: u64) -> std::result::Result<StorageHandle, IoError> {
        let id = StorageId::new();
        let bytes = usize::try_from(size).map_err(io_error)?;
        self.worker
            .call(move |state| state.allocate(id, bytes))
            .map_err(io_error)?;
        Ok(StorageHandle::new(
            id,
            StorageUtilization { offset: 0, size },
        ))
    }
    fn dealloc(&mut self, id: StorageId) {
        // Failure retains the device allocation; it is never released while busy.
        let _ = self.worker.call(move |state| state.free(id));
    }
    fn flush(&mut self) {}
}

pub struct AscendServer {
    worker: Worker,
    memory: MemoryManagement<AscendStorage>,
    utilities: Arc<ServerUtilities<Self>>,
    timestamps: TimestampProfiler,
    errors: Vec<ServerError>,
}
impl std::fmt::Debug for AscendServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AscendServer").finish()
    }
}
impl DeviceService for AscendServer {
    fn init(device: DeviceId) -> Self {
        let (ordinal, worker) = WORKER
            .get()
            .expect("call unsafe AscendRuntime::initialize_exclusive before client()");
        assert_eq!(
            device,
            DeviceId {
                type_id: 0,
                index_id: *ordinal
            },
            "device was not initialized"
        );
        let memory_properties = MemoryDeviceProperties {
            max_page_size: worker.total_memory,
            alignment: 32,
        };
        // These are compiler-supported logical limits, not measured hardware occupancy.
        let hardware = HardwareProperties {
            load_width: 32,
            plane_size_min: 32,
            plane_size_max: 32,
            max_bindings: 8,
            max_shared_memory_size: 0,
            max_ruda_count: AscendRuntime::max_ruda_count(),
            max_units_per_ruda: 1024,
            max_ruda_dim: (1024, 1, 1),
            num_streaming_multiprocessors: None,
            num_cpu_cores: None,
            num_tensor_cores: None,
            min_tensor_cores_dim: None,
            max_vector_size: 1,
        };
        let mut properties = DeviceProperties::new(
            Features::default(),
            memory_properties.clone(),
            hardware,
            TimingMethod::System,
        );
        properties.register_address_type(AddressType::U64);
        properties.register_address_type(AddressType::U32);
        properties.register_type_usage(FloatKind::F32, TypeUsage::Buffer);
        let logger = Arc::new(ServerLogger::default());
        let utilities = Arc::new(ServerUtilities::new(
            properties,
            logger.clone(),
            *ordinal,
            ContiguousMemoryLayoutPolicy::new(32),
        ));
        let memory = MemoryManagement::from_configuration(
            AscendStorage {
                worker: worker.clone(),
            },
            &memory_properties,
            MemoryConfiguration::default(),
            logger,
            MemoryManagementOptions::new("Ascend HBM"),
        );
        Self {
            worker: worker.clone(),
            memory,
            utilities,
            timestamps: TimestampProfiler::default(),
            errors: vec![],
        }
    }
    fn utilities(&self) -> ServerUtilitiesHandle {
        self.utilities.clone()
    }
}
impl ServerCommunication for AscendServer {
    const SERVER_COMM_ENABLED: bool = false;
}

impl AscendServer {
    fn resource(&mut self, binding: Binding) -> std::result::Result<AscendResource, ServerError> {
        let start = binding.offset_start.unwrap_or(0);
        let size = binding
            .size
            .checked_sub(start)
            .and_then(|n| n.checked_sub(binding.offset_end.unwrap_or(0)))
            .ok_or_else(|| server_error("invalid buffer slice"))?;
        let mut resource =
            self.memory
                .get_resource(binding.memory, Some(start), binding.offset_end)?;
        if size > resource.size as u64 {
            return Err(server_error("binding exceeds managed allocation"));
        }
        resource.size = size.try_into().map_err(server_error)?;
        Ok(resource)
    }
    fn take_errors(&mut self) -> std::result::Result<(), ServerError> {
        if self.errors.is_empty() {
            return Ok(());
        }
        let err = ServerError::ServerUnhealthy {
            errors: std::mem::take(&mut self.errors),
            backtrace: BackTrace::capture(),
        };
        self.timestamps
            .error(ProfileError::Server(Box::new(err.clone())));
        Err(err)
    }
    fn record(&mut self, result: std::result::Result<(), ServerError>) {
        if let Err(e) = result {
            self.errors.push(e);
        }
    }
    fn read_inner(
        &mut self,
        descriptors: Vec<CopyDescriptor>,
    ) -> std::result::Result<Vec<Bytes>, ServerError> {
        self.take_errors()?;
        let mut out = Vec::with_capacity(descriptors.len());
        for desc in descriptors {
            let size = copy_size(&desc)?;
            let mut resource = self.resource(desc.handle)?;
            if size > resource.size {
                return Err(server_error("copy exceeds resource"));
            }
            resource.size = size;
            out.push(Bytes::from_bytes_vec(
                self.worker
                    .call(move |s| s.read(resource))
                    .map_err(server_error)?,
            ));
        }
        Ok(out)
    }
    fn launch_inner(
        &mut self,
        task: Box<dyn RudaTask<AscendCompiler>>,
        count: RudaCount,
        arguments: KernelArguments,
        mode: ExecutionMode,
    ) -> std::result::Result<(), ServerError> {
        if !arguments.tensor_maps.is_empty() {
            return Err(server_error(
                "Ascend common IR does not yet lower tensor maps",
            ));
        }
        let elements = arguments
            .buffers
            .iter()
            .map(|b| {
                b.size
                    .checked_sub(b.offset_start.unwrap_or(0))
                    .and_then(|n| n.checked_sub(b.offset_end.unwrap_or(0)))
                    .ok_or_else(|| server_error("invalid kernel binding"))
            })
            .collect::<std::result::Result<Vec<_>, _>>()?
            .into_iter()
            .max()
            .unwrap_or(0);
        if elements % 4 != 0 {
            return Err(server_error("FP32 kernel domain must have whole elements"));
        }
        let mut options = AscendOptions {
            target: Some(AscendTarget::Ascend950DT),
            elements: elements / 4,
            ..Default::default()
        };
        let compiled = if arguments.info.data.is_empty() {
            task.compile(&mut AscendCompiler, &options, mode, task.address_type())
        } else {
            use ruda_core::compiler::Compiler;
            let definition=task.kernel_definition().ok_or_else(||server_error("packed arguments require a public KernelDefinition"))?;
            let dim=definition.ruda_dim;
            rust_ascend_compiler::ascend::arguments::specialize(definition,&arguments.info.data,
                arguments.info.dynamic_metadata_offset,task.address_type()).and_then(|definition| {
                options.elements=definition.buffers.iter().filter(|b| b.visibility==ruda_core::kernel::Visibility::ReadWrite)
                    .filter_map(|b|b.size).map(|n|n as u64).max().unwrap_or(0);
                AscendCompiler.compile(definition,&options,mode,task.address_type()).map(|repr| {
                    ruda_runtime::runtime::kernel::CompiledKernel::<AscendCompiler> {
                        entrypoint_name:repr.entrypoint().into(), source:repr.source().into(), repr:Some(repr),
                        debug_name:Some(task.name()),ruda_dim:dim,debug_info:None,
                    }
                })
            })
        }.map_err(|e| ServerError::Launch(LaunchError::CompilationError(e)))?;
        let grid = match count {
            RudaCount::Static(x, y, z) => (x, y, z),
            RudaCount::Dynamic(binding) => {
                let mut resource = self.resource(binding)?;
                if resource.size < 12 {
                    return Err(server_error(
                        "indirect grid buffer is shorter than 12 bytes",
                    ));
                }
                resource.size = 12;
                let bytes = self
                    .worker
                    .call(move |s| s.read(resource))
                    .map_err(server_error)?;
                let u = |n| u32::from_le_bytes(bytes[n..n + 4].try_into().unwrap());
                (u(0), u(4), u(8))
            }
        };
        let repr = compiled
            .repr
            .ok_or_else(|| server_error("Ascend task must return its compiled binding contract"))?;
        validate_grid(grid, compiled.ruda_dim, &repr)?;
        let resources = arguments
            .buffers
            .into_iter()
            .map(|b| self.resource(b))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        self.worker
            .call(move |state| state.launch(repr, resources))
            .map_err(server_error)
    }
}

impl ComputeServer for AscendServer {
    type Kernel = Box<dyn RudaTask<AscendCompiler>>;
    type Storage = AscendStorage;
    type MemoryLayoutPolicy = ContiguousMemoryLayoutPolicy;
    type Info = u16;
    fn logger(&self) -> Arc<ServerLogger> {
        self.utilities.logger.clone()
    }
    fn utilities(&self) -> Arc<ServerUtilities<Self>> {
        self.utilities.clone()
    }
    fn initialize_memory(&mut self, memory: ManagedMemoryHandle, size: u64, _: StreamId) {
        let result = self
            .memory
            .reserve(size)
            .and_then(|reserved| self.memory.bind(reserved, memory, 0))
            .map_err(Into::into);
        self.record(result);
    }
    fn read(
        &mut self,
        descriptors: Vec<CopyDescriptor>,
        _: StreamId,
    ) -> DynFut<std::result::Result<Vec<Bytes>, ServerError>> {
        let result = self.read_inner(descriptors);
        Box::pin(async move { result })
    }
    fn write(&mut self, descriptors: Vec<(CopyDescriptor, Bytes)>, _: StreamId) {
        if !self.errors.is_empty() {
            return;
        }
        let result = (|| {
            for (desc, bytes) in descriptors {
                let size = copy_size(&desc)?;
                if size != bytes.len() {
                    return Err(server_error("copy shape/data length mismatch"));
                }
                let mut resource = self.resource(desc.handle)?;
                if size > resource.size {
                    return Err(server_error("copy exceeds resource"));
                }
                resource.size = size;
                let bytes = bytes.to_vec();
                self.worker
                    .call(move |s| s.write(resource, bytes))
                    .map_err(server_error)?;
            }
            Ok(())
        })();
        self.record(result);
    }
    fn sync(&mut self, stream: StreamId) -> DynFut<std::result::Result<(), ServerError>> {
        let result = self.flush(stream);
        Box::pin(async move { result })
    }
    fn get_resource(
        &mut self,
        binding: Binding,
        _: StreamId,
    ) -> std::result::Result<ManagedResource<AscendResource>, ServerError> {
        self.take_errors()?;
        let memory = binding.memory.clone();
        let resource = self.resource(binding)?;
        Ok(ManagedResource::new(memory, resource))
    }
    unsafe fn launch(
        &mut self,
        kernel: Self::Kernel,
        count: RudaCount,
        arguments: KernelArguments,
        mode: ExecutionMode,
        _: StreamId,
    ) {
        if !self.errors.is_empty() {
            return;
        }
        let result = self.launch_inner(kernel, count, arguments, mode);
        self.record(result);
    }
    fn flush(&mut self, _: StreamId) -> std::result::Result<(), ServerError> {
        let result = self.worker.call(|s| s.sync()).map_err(server_error);
        self.record(result);
        self.take_errors()
    }
    fn memory_usage(&mut self, _: StreamId) -> std::result::Result<MemoryUsage, ServerError> {
        Ok(self.memory.memory_usage())
    }
    fn memory_cleanup(&mut self, _: StreamId) {
        self.memory.cleanup(true);
    }
    fn start_profile(
        &mut self,
        stream: StreamId,
    ) -> std::result::Result<ProfilingToken, ServerError> {
        self.flush(stream)?;
        Ok(self.timestamps.start())
    }
    fn end_profile(
        &mut self,
        stream: StreamId,
        token: ProfilingToken,
    ) -> std::result::Result<ProfileDuration, ProfileError> {
        if let Err(e) = self.flush(stream) {
            self.timestamps.error(ProfileError::Server(Box::new(e)));
        }
        self.timestamps.stop(token)
    }
    fn allocation_mode(&mut self, mode: MemoryAllocationMode, _: StreamId) {
        self.memory.mode(mode);
    }
}

fn contiguous(shape: &[usize], strides: &[usize]) -> bool {
    if shape.len() != strides.len() {
        return false;
    }
    if shape.contains(&0) {
        return true;
    }
    let mut expected = 1usize;
    for (&dim, &stride) in shape.iter().zip(strides).rev() {
        if dim > 1 && stride != expected {
            return false;
        }
        let Some(next) = expected.checked_mul(dim) else {
            return false;
        };
        expected = next;
    }
    true
}
fn copy_size(desc: &CopyDescriptor) -> std::result::Result<usize, ServerError> {
    if !contiguous(&desc.shape, &desc.strides) {
        return Err(IoError::UnsupportedStrides {
            backtrace: BackTrace::capture(),
        }
        .into());
    }
    if desc.elem_size == 0 {
        return Err(server_error("zero-sized tensor element"));
    }
    if desc.shape.contains(&0) {
        return Ok(0);
    }
    desc.shape
        .iter()
        .try_fold(desc.elem_size, |n, &d| n.checked_mul(d))
        .ok_or_else(|| server_error("copy size overflow"))
}
fn validate_grid(
    grid: (u32, u32, u32),
    dim: RudaDim,
    kernel: &rust_ascend_compiler::ascend::AscendKernel,
) -> std::result::Result<(), ServerError> {
    if grid.1 != 1 || grid.2 != 1 || dim.y != 1 || dim.z != 1 || dim.x == 0 {
        return Err(server_error(
            "Ascend common IR requires a one-dimensional logical grid",
        ));
    }
    let logical_units = match kernel.row_width() {
        Some(width) => kernel.elements() / u64::from(width) * 32,
        None => kernel.elements(),
    };
    if u64::from(grid.0) != logical_units.div_ceil(u64::from(dim.x)) {
        return Err(server_error(
            "launch grid differs from the complete compiled logical domain",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn implements_portable_runtime_contract() {
        fn runtime<R: Runtime>() {}
        fn server<S: ComputeServer>() {}
        runtime::<AscendRuntime>();
        server::<AscendServer>();
    }
    #[test]
    fn contiguous_layout_edges() {
        assert!(contiguous(&[], &[]));
        assert!(contiguous(&[2, 1, 3], &[3, 99, 1]));
        assert!(contiguous(&[0, 3], &[9, 9]));
        assert!(!contiguous(&[2, 3], &[1, 2]));
        assert!(!contiguous(&[usize::MAX, 2], &[2, 1]));
        assert!(!contiguous(&[1], &[]));
    }
    #[test]
    fn logical_grid_cannot_silently_change_work() {
        use ruda_core::{compiler::Compiler, ir::UIntKind};
        use rust_ascend_compiler::ascend::programs::{MapProgram, definition};
        let kernel = AscendCompiler
            .compile(
                definition(MapProgram::Add),
                &AscendOptions {
                    target: Some(AscendTarget::Ascend950DT),
                    elements: 65,
                    ..Default::default()
                },
                ExecutionMode::Checked,
                UIntKind::U64.into(),
            )
            .unwrap();
        let dim = RudaDim::new_1d(64);
        assert!(validate_grid((2, 1, 1), dim, &kernel).is_ok());
        for grid in [(1, 1, 1), (3, 1, 1), (2, 2, 1), (0, 1, 1)] {
            assert!(validate_grid(grid, dim, &kernel).is_err());
        }
    }
}

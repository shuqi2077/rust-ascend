//! HCCL handles and device collectives owned by the existing CANN worker.
use super::worker::{AscendResource, Worker};
use super::{AscendDevice, AscendRuntime, ComputeClient, Result, TensorBuffer, WORKER, error};
use crate::{
    CannError, CannLibrary,
    tensor::{CannSession, DType, TensorLayout},
};
use std::{
    collections::HashMap,
    ffi::{OsStr, c_void},
    ptr::NonNull,
    rc::Rc,
    sync::Arc,
};

const ROOT_INFO_BYTES: usize = 4108;

/// Opaque HCCL rendezvous data. Transmit these bytes unchanged between ranks.
#[derive(Clone)]
#[repr(C)]
pub struct HcclRootInfo {
    internal: [u8; ROOT_INFO_BYTES],
}
impl HcclRootInfo {
    pub fn as_bytes(&self) -> &[u8] {
        &self.internal
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Ok(Self {
            internal: bytes
                .try_into()
                .map_err(|_| error("HCCL root info must contain 4108 bytes"))?,
        })
    }
}
impl std::fmt::Debug for HcclRootInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HcclRootInfo")
            .field("bytes", &ROOT_INFO_BYTES)
            .finish()
    }
}

/// Reductions supported by the native HCCL entry point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum HcclReduceOp {
    Sum = 0,
    Product = 1,
    Maximum = 2,
    Minimum = 3,
}

type GetRoot = unsafe extern "C" fn(*mut HcclRootInfo) -> i32;
type Init = unsafe extern "C" fn(u32, *const HcclRootInfo, u32, *mut *mut c_void) -> i32;
type Destroy = unsafe extern "C" fn(*mut c_void) -> i32;
type Broadcast = unsafe extern "C" fn(*mut c_void, u64, i32, u32, *mut c_void, *mut c_void) -> i32;
type AllReduce =
    unsafe extern "C" fn(*mut c_void, *mut c_void, u64, i32, i32, *mut c_void, *mut c_void) -> i32;

struct Api {
    _library: CannLibrary,
    root: GetRoot,
    init: Init,
    destroy: Destroy,
    broadcast: Broadcast,
    all_reduce: AllReduce,
}
impl Api {
    unsafe fn load(library: &OsStr) -> Result<Self> {
        // SAFETY: caller supplies a trusted HCCL library with the CANN C ABI.
        let library = unsafe { CannLibrary::load_from(library)? };
        Ok(Self {
            root: unsafe { *library.symbol::<GetRoot>(c"HcclGetRootInfo")? },
            init: unsafe { *library.symbol::<Init>(c"HcclCommInitRootInfo")? },
            destroy: unsafe { *library.symbol::<Destroy>(c"HcclCommDestroy")? },
            broadcast: unsafe { *library.symbol::<Broadcast>(c"HcclBroadcast")? },
            all_reduce: unsafe { *library.symbol::<AllReduce>(c"HcclAllReduce")? },
            _library: library,
        })
    }
}
fn status(operation: &'static str, code: i32) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(error(format!("{operation} failed with HCCL status {code}")))
    }
}
fn datatype(dtype: DType) -> Result<i32> {
    Ok(match dtype {
        DType::I32 => 2,
        DType::F16 => 3,
        DType::F32 => 4,
        DType::I64 => 5,
        DType::U8 | DType::Bool => 7,
        DType::BF16 => 11,
        _ => {
            return Err(error(
                "HCCL tensor requires FP32/FP16/BF16/I32/I64 or byte Bool",
            ));
        }
    })
}

struct Entry {
    api: Rc<Api>,
    session: Rc<CannSession>,
    handle: Option<NonNull<c_void>>,
    world_size: u32,
}
impl Drop for Entry {
    fn drop(&mut self) {
        if let Some(handle) = self.handle {
            // Keep the library/context alive if vendor work cannot be drained or destroyed.
            if self.session.synchronize().is_err()
                || unsafe { (self.api.destroy)(handle.as_ptr()) } != 0
            {
                std::mem::forget(self.api.clone());
                std::mem::forget(self.session.clone());
            }
        }
    }
}
#[derive(Default)]
pub(super) struct Registry {
    entries: HashMap<u64, Entry>,
    next: u64,
}
impl Registry {
    unsafe fn open(&mut self, session: Rc<CannSession>, library: &OsStr) -> Result<u64> {
        session.bind()?;
        let next = self
            .next
            .checked_add(1)
            .ok_or_else(|| error("HCCL context ID overflow"))?;
        let api = Rc::new(unsafe { Api::load(library)? });
        self.entries.insert(
            next,
            Entry {
                api,
                session,
                handle: None,
                world_size: 0,
            },
        );
        self.next = next;
        Ok(next)
    }
    fn entry(&self, id: u64) -> Result<&Entry> {
        self.entries
            .get(&id)
            .ok_or_else(|| error("HCCL context was released"))
    }
    fn root(&self, id: u64) -> Result<HcclRootInfo> {
        let entry = self.entry(id)?;
        entry.session.bind()?;
        let mut root = HcclRootInfo {
            internal: [0; ROOT_INFO_BYTES],
        };
        status("HcclGetRootInfo", unsafe { (entry.api.root)(&mut root) })?;
        Ok(root)
    }
    fn initialize(
        &mut self,
        id: u64,
        root: HcclRootInfo,
        rank: u32,
        world_size: u32,
    ) -> Result<()> {
        if world_size == 0 || rank >= world_size {
            return Err(error("invalid HCCL rank/world size"));
        }
        let entry = self
            .entries
            .get_mut(&id)
            .ok_or_else(|| error("HCCL context was released"))?;
        if entry.handle.is_some() {
            return Err(error("HCCL context is already initialized"));
        }
        entry.session.bind()?;
        let mut handle = std::ptr::null_mut();
        status("HcclCommInitRootInfo", unsafe {
            (entry.api.init)(world_size, &root, rank, &mut handle)
        })?;
        entry.handle =
            Some(NonNull::new(handle).ok_or(CannError::NullHandle("HcclCommInitRootInfo"))?);
        entry.world_size = world_size;
        Ok(())
    }
    fn collective(
        &self,
        id: u64,
        layout: &TensorLayout,
        addresses: &[usize],
        operation: Operation,
    ) -> Result<()> {
        let entry = self.entry(id)?;
        let handle = entry
            .handle
            .ok_or_else(|| error("HCCL communicator is not initialized"))?;
        let dtype = datatype(layout.dtype())?;
        if let Operation::AllReduce(HcclReduceOp::Product) = operation {
            if layout.dtype() == DType::BF16 {
                return Err(error("HCCL product does not support BF16"));
            }
        }
        if let Operation::Broadcast(root) = operation {
            if root >= entry.world_size {
                return Err(error("HCCL broadcast root is outside the world"));
            }
        }
        let count = u64::try_from(layout.byte_len() / layout.dtype().bytes()).map_err(error)?;
        if count == 0 {
            return Ok(());
        }
        entry.session.bind()?;
        entry.session.quiescent.set(false);
        let (name, launch_code) = match operation {
            Operation::Broadcast(root) => ("HcclBroadcast", unsafe {
                (entry.api.broadcast)(
                    addresses[0] as *mut c_void,
                    count,
                    dtype,
                    root,
                    handle.as_ptr(),
                    entry.session.stream,
                )
            }),
            Operation::AllReduce(op) => ("HcclAllReduce", unsafe {
                (entry.api.all_reduce)(
                    addresses[0] as *mut c_void,
                    addresses[1] as *mut c_void,
                    count,
                    dtype,
                    op as i32,
                    handle.as_ptr(),
                    entry.session.stream,
                )
            }),
        };
        // Drain even after a failed launch; guarded buffers outlive device completion.
        let sync_code = unsafe {
            entry
                .session
                .api
                .aclrtSynchronizeStream(entry.session.stream)
        };
        entry.session.quiescent.set(sync_code == 0);
        if launch_code != 0 || sync_code != 0 {
            return Err(CannError::Completion {
                operation: name,
                launch_code,
                sync_code,
            });
        }
        Ok(())
    }
}

struct Registration {
    id: u64,
    device: AscendDevice,
    worker: Worker,
}
impl Drop for Registration {
    fn drop(&mut self) {
        let id = self.id;
        let _ = self.worker.call(move |state| {
            state.hccl.entries.remove(&id);
            Ok(())
        });
    }
}
impl std::fmt::Debug for Registration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HcclRegistration")
            .field("id", &self.id)
            .field("device", &self.device)
            .finish()
    }
}

/// A loaded HCCL context on the process's existing NPU worker.
#[derive(Debug)]
pub struct HcclContext {
    registration: Arc<Registration>,
}
impl HcclContext {
    /// # Safety
    /// `library` must be a trusted HCCL library compatible with the loaded CANN SDK.
    pub unsafe fn open(device: &AscendDevice, library: impl AsRef<OsStr>) -> Result<Self> {
        let (ordinal, worker) = WORKER
            .get()
            .ok_or_else(|| error("Ascend runtime is not initialized"))?;
        if *ordinal != device.ordinal() {
            return Err(error("HCCL device differs from the process NPU"));
        }
        let library = library.as_ref().to_os_string();
        let id = worker
            .call(move |state| unsafe { state.hccl.open(state.session.clone(), &library) })?;
        Ok(Self {
            registration: Arc::new(Registration {
                id,
                device: device.clone(),
                worker: worker.clone(),
            }),
        })
    }
    /// Call on one rank, then transmit the bytes to all other ranks unchanged.
    pub fn root_info(&self) -> Result<HcclRootInfo> {
        let id = self.registration.id;
        self.registration
            .worker
            .call(move |state| state.hccl.root(id))
    }
    /// All ranks enter with the same root info and distinct contiguous rank IDs.
    pub fn initialize(
        self,
        root: HcclRootInfo,
        rank: u32,
        world_size: u32,
    ) -> Result<HcclCommunicator> {
        let id = self.registration.id;
        self.registration
            .worker
            .call(move |state| state.hccl.initialize(id, root, rank, world_size))?;
        Ok(HcclCommunicator {
            registration: self.registration,
            rank,
            world_size,
        })
    }
}

/// Synchronous device-native collectives; no tensor payload is staged through host RAM.
#[derive(Debug, Clone)]
pub struct HcclCommunicator {
    registration: Arc<Registration>,
    rank: u32,
    world_size: u32,
}
#[derive(Clone, Copy)]
enum Operation {
    Broadcast(u32),
    AllReduce(HcclReduceOp),
}
impl HcclCommunicator {
    pub fn rank(&self) -> u32 {
        self.rank
    }
    pub fn world_size(&self) -> u32 {
        self.world_size
    }
    pub fn device(&self) -> &AscendDevice {
        &self.registration.device
    }
    /// Return an independent dense broadcast result, preserving the caller's input/view.
    /// All ranks must pass identical shape/dtype and root, including for empty tensors.
    pub fn broadcast(
        &self,
        client: &ComputeClient<AscendRuntime>,
        value: TensorBuffer,
        root: u32,
    ) -> Result<TensorBuffer> {
        if root >= self.world_size {
            return Err(error("HCCL broadcast root is outside the world"));
        }
        self.execute(client, value, Operation::Broadcast(root))
    }
    /// Return an independent dense reduction result. All ranks use matching contracts.
    pub fn all_reduce(
        &self,
        client: &ComputeClient<AscendRuntime>,
        value: TensorBuffer,
        op: HcclReduceOp,
    ) -> Result<TensorBuffer> {
        self.execute(client, value, Operation::AllReduce(op))
    }
    fn execute(
        &self,
        client: &ComputeClient<AscendRuntime>,
        value: TensorBuffer,
        operation: Operation,
    ) -> Result<TensorBuffer> {
        let source_layout = super::typed::layout(&value)?;
        datatype(source_layout.dtype())?;
        if matches!(operation, Operation::AllReduce(_))
            && matches!(source_layout.dtype(), DType::U8 | DType::Bool)
        {
            return Err(error(
                "HCCL all-reduce requires a floating or I32/I64 tensor",
            ));
        }
        if matches!(operation, Operation::AllReduce(HcclReduceOp::Product))
            && source_layout.dtype() == DType::BF16
        {
            return Err(error("HCCL product does not support BF16"));
        }
        let dense = TensorLayout::contiguous(source_layout.shape(), source_layout.dtype())?;
        let source = if matches!(operation, Operation::AllReduce(_)) && source_layout == dense {
            value
        } else {
            AscendRuntime::materialize(client, value)?
        };
        let layout = super::typed::layout(&source)?;
        let (output, buffers) = match operation {
            Operation::Broadcast(_) => (source.clone(), vec![source]),
            Operation::AllReduce(_) => {
                let output = super::typed::allocate(client, &layout);
                (output.clone(), vec![source, output])
            }
        };
        client.flush().map_err(error)?;
        if layout.byte_len() == 0 {
            return Ok(output);
        }
        let guards = buffers
            .iter()
            .map(|buffer| client.get_resource(buffer.handle.clone()).map_err(error))
            .collect::<Result<Vec<_>>>()?;
        let resources = guards
            .iter()
            .map(|guard| {
                let mut resource: AscendResource = guard.resource().clone();
                if resource.byte_len() < layout.byte_len() {
                    return Err(error("HCCL allocation is too short"));
                }
                resource.size = layout.byte_len();
                Ok(resource)
            })
            .collect::<Result<Vec<_>>>()?;
        let id = self.registration.id;
        let result = self.registration.worker.call(move |state| {
            let addresses = resources
                .iter()
                .map(|resource| state.pointer(resource).map(|ptr| ptr as usize))
                .collect::<Result<Vec<_>>>()?;
            state.hccl.collective(id, &layout, &addresses, operation)
        });
        if matches!(&result, Err(CannError::Completion { sync_code, .. }) if *sync_code != 0) {
            std::mem::forget(guards);
        }
        result?;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn root_info_wire_and_dtype_abi() {
        let bytes = (0..ROOT_INFO_BYTES)
            .map(|i| (i % 251) as u8)
            .collect::<Vec<_>>();
        assert_eq!(std::mem::size_of::<HcclRootInfo>(), 4108);
        assert_eq!(HcclRootInfo::from_bytes(&bytes).unwrap().as_bytes(), bytes);
        assert!(HcclRootInfo::from_bytes(&bytes[..4107]).is_err());
        for (dtype, code) in [
            (DType::I32, 2),
            (DType::F16, 3),
            (DType::F32, 4),
            (DType::I64, 5),
            (DType::Bool, 7),
            (DType::BF16, 11),
        ] {
            assert_eq!(datatype(dtype).unwrap(), code);
        }
        assert!(datatype(DType::F64).is_err());
    }
}

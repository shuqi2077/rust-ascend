use super::{Result, RuntimeOptions, build::Toolchain, error};
use crate::{
    check_status,
    sys::*,
    tensor::{Buffer, CannSession, common_ir::CannProgram, deepgemm::{DeepGemm, GemmSpec, native::NativeApi}},
};
use ruda_runtime::runtime::storage::StorageId;
use rust_ascend_compiler::ascend::AscendKernel;
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    ffi::c_void,
    rc::Rc,
    sync::{Arc, mpsc},
    thread,
};

type Request = Box<dyn FnOnce(&mut State) + Send>;

#[derive(Clone)]
pub(super) struct Worker {
    sender: Arc<mpsc::Sender<Request>>,
    pub total_memory: u64,
}
impl std::fmt::Debug for Worker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AscendWorker")
            .field("total_memory", &self.total_memory)
            .finish()
    }
}
impl Worker {
    /// Caller owns the process-wide CANN lifecycle, as required by initialize_exclusive.
    pub unsafe fn start(options: RuntimeOptions) -> Result<Self> {
        let (sender, receiver) = mpsc::channel::<Request>();
        let (ready, result) = mpsc::sync_channel(1);
        thread::Builder::new()
            .name("rust-ascend-device".into())
            .spawn(move || {
                let initialized = (|| {
                    let tools = Toolchain::new(&options.toolkit, options.compile_timeout)?;
                    let session = unsafe {
                        CannSession::open_exclusive_libraries(
                            crate::CannDevice::new(options.device as u32)?,
                            &options.acl_library,
                            &options.operator_libraries,
                        )?
                    };
                    NativeApi::resolve(&session)?.check_soc()?;
                    let mut free = 0;
                    let mut total = 0;
                    check_status("aclrtGetMemInfo", unsafe {
                        session
                            .api
                            .aclrtGetMemInfo(ACL_HBM_MEM, &mut free, &mut total)
                    })?;
                    let gemm_artifacts = tempfile::Builder::new().prefix("rust-ascend-gemm-").tempdir().map_err(error)?;
                    let gemm = unsafe { DeepGemm::load(&session, gemm_artifacts.path())? };
                    let state = State {
                        session,
                        tools,
                        buffers: HashMap::new(),
                        programs: HashMap::new(),
                        gemm,
                        gemm_artifacts,
                    };
                    Ok((state, total as u64))
                })();
                match initialized {
                    Ok((mut state, total)) => {
                        if ready.send(Ok(total)).is_err() {
                            return;
                        }
                        while let Ok(request) = receiver.recv() {
                            request(&mut state);
                        }
                    }
                    Err(e) => {
                        let _ = ready.send(Err(e));
                    }
                }
            })
            .map_err(error)?;
        let total_memory = result.recv().map_err(error)??;
        Ok(Self {
            sender: Arc::new(sender),
            total_memory,
        })
    }

    pub fn call<T: Send + 'static>(
        &self,
        f: impl FnOnce(&mut State) -> Result<T> + Send + 'static,
    ) -> Result<T> {
        let (send, receive) = mpsc::sync_channel(1);
        self.sender
            .send(Box::new(move |state| {
                let _ = send.send(f(state));
            }))
            .map_err(|_| error("Ascend device worker disconnected"))?;
        receive.recv().map_err(error)?
    }
}

#[derive(Debug, Clone)]
pub struct AscendResource {
    pub(super) id: StorageId,
    pub(super) offset: usize,
    pub(super) size: usize,
}
impl AscendResource {
    pub fn byte_len(&self) -> usize {
        self.size
    }
}

pub(super) struct State {
    session: Rc<CannSession>,
    tools: Toolchain,
    buffers: HashMap<StorageId, Buffer>,
    programs: HashMap<String, CannProgram>,
    gemm: DeepGemm,
    gemm_artifacts: tempfile::TempDir,
}
impl State {
    pub fn allocate(&mut self, id: StorageId, size: usize) -> Result<()> {
        self.buffers.insert(id, self.session.allocate(size)?);
        Ok(())
    }
    pub fn free(&mut self, id: StorageId) -> Result<()> {
        self.session.synchronize()?;
        self.buffers
            .remove(&id)
            .ok_or_else(|| error("unknown device allocation"))?;
        Ok(())
    }
    fn pointer(&self, resource: &AscendResource) -> Result<*mut c_void> {
        let buffer = self
            .buffers
            .get(&resource.id)
            .ok_or_else(|| error("unknown device allocation"))?;
        if resource
            .offset
            .checked_add(resource.size)
            .is_none_or(|end| end > buffer.bytes)
        {
            return Err(error("device resource exceeds allocation"));
        }
        Ok(buffer.data.as_ptr().wrapping_byte_add(resource.offset))
    }
    pub fn read(&mut self, resource: AscendResource) -> Result<Vec<u8>> {
        self.session.synchronize()?;
        let pointer = self.pointer(&resource)?;
        let mut bytes = vec![0u8; resource.size];
        if !bytes.is_empty() {
            check_status("aclrtMemcpy", unsafe {
                self.session.api.aclrtMemcpy(
                    bytes.as_mut_ptr().cast(),
                    bytes.len(),
                    pointer,
                    bytes.len(),
                    ACL_MEMCPY_DEVICE_TO_HOST,
                )
            })?;
        }
        Ok(bytes)
    }
    pub fn write(&mut self, resource: AscendResource, bytes: Vec<u8>) -> Result<()> {
        self.session.bind()?;
        if resource.size != bytes.len() {
            return Err(error("host/device copy byte count mismatch"));
        }
        let pointer = self.pointer(&resource)?;
        if !bytes.is_empty() {
            check_status("aclrtMemcpy", unsafe {
                self.session.api.aclrtMemcpy(
                    pointer,
                    bytes.len(),
                    bytes.as_ptr().cast(),
                    bytes.len(),
                    ACL_MEMCPY_HOST_TO_DEVICE,
                )
            })?;
        }
        Ok(())
    }
    pub fn sync(&mut self) -> Result<()> {
        self.session.synchronize()
    }
    pub fn launch(&mut self, compiled: AscendKernel, resources: Vec<AscendResource>) -> Result<()> {
        let mut hash = Sha256::new();
        hash.update(compiled.source().as_bytes());
        hash.update(compiled.build_contract().as_bytes());
        let key = format!("{:x}", hash.finalize());
        if !self.programs.contains_key(&key) {
            let artifact = self.tools.build(&compiled)?;
            let program = unsafe { CannProgram::load(&self.session, compiled, artifact.path())? };
            self.programs.insert(key.clone(), program);
        }
        let addresses = resources
            .iter()
            .map(|r| self.pointer(r).map(|p| (p, r.size)))
            .collect::<Result<Vec<_>>>()?;
        // All buffers and the program belong to this worker/session and stay alive
        // until run_addresses synchronizes. No pointers cross the channel.
        unsafe { self.programs[&key].run_addresses(addresses) }
    }

    pub fn gemm(&mut self, spec: GemmSpec, resources: [AscendResource; 3]) -> Result<()> {
        let key = spec.key();
        if !self.gemm_artifacts.path().join(&key).is_dir() {
            self.tools.build_gemm(&key, self.gemm_artifacts.path())?;
        }
        let [a,b,out] = resources;
        let addresses = [(self.pointer(&a)?,a.size), (self.pointer(&b)?,b.size),
            (self.pointer(&out)?,out.size)];
        // SAFETY: resource ranges belong to this worker. The caller retains their
        // ManagedResource guards until the provider's synchronous execution returns.
        unsafe { self.gemm.run_addresses(&spec, addresses) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn channel_and_resources_are_send_sync_without_moving_cann_handles() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Worker>();
        assert_send_sync::<AscendResource>();
    }
}

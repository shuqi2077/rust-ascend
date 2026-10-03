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
    pub fn copy_embedding_indices(&mut self,layout:crate::tensor::TensorLayout,resources:[AscendResource;2])->Result<()> {
        if resources.iter().any(|r|r.size!=layout.byte_len()) || !matches!(layout.dtype(),crate::tensor::DType::I32|crate::tensor::DType::I64) {
            return Err(error("embedding index snapshot contract mismatch"));
        }
        let addresses=[self.pointer(&resources[0])? as usize,self.pointer(&resources[1])? as usize];
        cast_ranges(addresses,&[layout.clone(),layout.clone()])?;
        if layout.byte_len()==0 {return Ok(());}
        self.session.bind()?;
        // SAFETY: exact-size, disjoint device allocations retained by caller guards;
        // synchronous device-to-device copy preserves integer bits and never visits host RAM.
        check_status("aclrtMemcpy(embedding IDs)",unsafe {self.session.api.aclrtMemcpy(addresses[1] as *mut c_void,layout.byte_len(),
            addresses[0] as *const c_void,layout.byte_len(),ACL_MEMCPY_DEVICE_TO_DEVICE)})
    }
    pub fn embedding(&mut self,layouts:[crate::tensor::TensorLayout;3],resources:[AscendResource;3],backward:Option<crate::tensor::EmbeddingOptions>)->Result<()> {
        use crate::tensor::embedding::{forward_layout,backward_layout,EmbeddingPlan,EmbeddingGradPlan};
        if resources.iter().zip(&layouts).any(|(r,l)|r.size!=l.byte_len()) {return Err(error("embedding layout/resource contract mismatch"));}
        let expected=if let Some(options)=backward {
            if layouts[2].shape().len()!=2 {return Err(error("embedding dense output must have rank two"));}
            backward_layout(&layouts[0],&layouts[1],layouts[2].shape()[0] as u64,options)?
        } else {forward_layout(&layouts[0],&layouts[1])?};
        if expected!=layouts[2] {return Err(error("embedding output does not match its checked contract"));}
        let addresses=[self.pointer(&resources[0])? as usize,self.pointer(&resources[1])? as usize,self.pointer(&resources[2])? as usize];
        embedding_ranges(addresses,&layouts)?;self.session.bind()?;
        if backward.is_some() && layouts[2].byte_len()!=0 {
            type Memset=unsafe extern "C" fn(*mut c_void,usize,i32,usize)->i32;
            // SAFETY: checked dense gradient storage is initialized on the device.
            let memset=unsafe {self.session.api.library().symbol::<Memset>(c"aclrtMemset")?};
            check_status("aclrtMemset(embedding gradient)",unsafe {memset(addresses[2] as *mut c_void,layouts[2].byte_len(),0,layouts[2].byte_len())})?;
        }
        if layouts[0].byte_len()==0 || layouts[1].byte_len()==0 || layouts[2].byte_len()==0 {return Ok(());}
        let [a,ids,out]=layouts;
        let rows=out.shape()[0] as u64;
        // SAFETY: typed descriptors borrow guarded allocations until the synchronized ACLNN executor returns.
        let a=unsafe {super::descriptor::Descriptor::new(&self.session,a,addresses[0] as *mut c_void)?};
        let ids=unsafe {super::descriptor::Descriptor::new(&self.session,ids,addresses[1] as *mut c_void)?};
        let out=unsafe {super::descriptor::Descriptor::new(&self.session,out,addresses[2] as *mut c_void)?};
        unsafe {
            if let Some(options)=backward {
                let padding=options.padding(rows)?;
                let plan:EmbeddingGradPlan=self.session.ops.get(c"aclnnEmbeddingDenseBackwardGetWorkspaceSize")?;
                let run=self.session.ops.get(c"aclnnEmbeddingDenseBackward")?;
                self.session.execute("aclnnEmbeddingDenseBackward",run,|size,executor|plan(a.handle.as_ptr(),ids.handle.as_ptr(),rows,padding,options.scale_grad_by_freq,out.handle.as_ptr(),size,executor))
            } else {
                let plan:EmbeddingPlan=self.session.ops.get(c"aclnnEmbeddingGetWorkspaceSize")?;let run=self.session.ops.get(c"aclnnEmbedding")?;
                self.session.execute("aclnnEmbedding",run,|size,executor|plan(a.handle.as_ptr(),ids.handle.as_ptr(),out.handle.as_ptr(),size,executor))
            }
        }
    }
    pub fn cast(&mut self,layouts:[crate::tensor::TensorLayout;2],resources:[AscendResource;2])->Result<()> {
        if layouts[0].shape()!=layouts[1].shape() || resources.iter().zip(&layouts).any(|(r,l)|r.size!=l.byte_len()) {
            return Err(error("Cast layout/resource contract mismatch"));
        }
        let addresses=[self.pointer(&resources[0])? as usize,self.pointer(&resources[1])? as usize];
        cast_ranges(addresses,&layouts)?;
        if layouts[0].byte_len()==0 {return Ok(());}
        let [input,output]=layouts;
        let target=output.dtype();
        // SAFETY: both managed allocations belong to this worker. The caller retains
        // their guards until the existing synchronized ACLNN executor returns.
        let input=unsafe {super::descriptor::Descriptor::new(&self.session,input,addresses[0] as *mut c_void)?};
        let output=unsafe {super::descriptor::Descriptor::new(&self.session,output,addresses[1] as *mut c_void)?};
        // The target dtype remains explicit; no native-IR or CPU fallback is selected.
        unsafe {
            let plan:crate::tensor::backward::CastPlan=self.session.ops.get(c"aclnnCastGetWorkspaceSize")?;
            let run=self.session.ops.get(c"aclnnCast")?;
            self.session.execute("aclnnCast",run,|size,executor| {
                plan(input.handle.as_ptr(),target as i32,output.handle.as_ptr(),size,executor)
            })
        }
    }

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

fn embedding_ranges(addresses:[usize;3],layouts:&[crate::tensor::TensorLayout;3])->Result<()> {
    let mut ends=[0usize;3];
    for i in 0..3 {
        if addresses[i]==0 || addresses[i]%layouts[i].dtype().bytes()!=0 {return Err(error("embedding address is null or unaligned"));}
        ends[i]=addresses[i].checked_add(layouts[i].byte_len()).ok_or_else(||error("embedding address range overflow"))?;
    }
    for input in 0..2 {if layouts[input].byte_len()!=0 && layouts[2].byte_len()!=0 && addresses[2]<ends[input] && addresses[input]<ends[2] {
        return Err(error("embedding output overlaps readonly input"));
    }}
    Ok(())
}
fn cast_ranges(addresses:[usize;2],layouts:&[crate::tensor::TensorLayout;2])->Result<()> {
    let mut ends=[0;2];
    for i in 0..2 {
        if addresses[i]==0 || addresses[i]%layouts[i].dtype().bytes()!=0 {return Err(error("Cast address is null or unaligned"));}
        ends[i]=addresses[i].checked_add(layouts[i].byte_len()).ok_or_else(||error("Cast address range overflow"))?;
    }
    if layouts[0].byte_len()!=0 && addresses[0]<ends[1] && addresses[1]<ends[0] {
        return Err(error("Cast input and output storage overlap"));
    }
    Ok(())
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
    #[test]
    fn cast_ranges_reject_overlap_alignment_and_overflow_before_any_execution() {
        use crate::tensor::{DType,TensorLayout};
        let layouts=[TensorLayout::contiguous(&[65],DType::BF16).unwrap(),
            TensorLayout::contiguous(&[65],DType::F32).unwrap()];
        assert!(cast_ranges([4096,8192],&layouts).is_ok());
        assert!(cast_ranges([4096,4096],&layouts).is_err());
        assert!(cast_ranges([4096,4224],&layouts).is_err());
        assert!(cast_ranges([4097,8192],&layouts).is_err());
        assert!(cast_ranges([4096,8194],&layouts).is_err());
        assert!(cast_ranges([0,8192],&layouts).is_err());
        assert!(cast_ranges([usize::MAX-1,8192],&layouts).is_err());
    }
    #[test]
    fn embedding_ranges_preserve_readonly_aliases_and_disjoint_typed_output() {
        use crate::tensor::{DType,TensorLayout};
        let layouts=[TensorLayout::contiguous(&[7,65],DType::F32).unwrap(),
            TensorLayout::contiguous(&[2,3],DType::I64).unwrap(),
            TensorLayout::contiguous(&[2,3,65],DType::F32).unwrap()];
        assert!(embedding_ranges([4096,8192,12288],&layouts).is_ok());
        assert!(embedding_ranges([4096,4096,12288],&layouts).is_ok());
        for addresses in [[4096,8192,4096],[4096,8192,8192],[4096,8193,12288],
            [4096,8192,12289],[0,8192,12288],[usize::MAX-3,8192,12288]] {
            assert!(embedding_ranges(addresses,&layouts).is_err());
        }
    }
}

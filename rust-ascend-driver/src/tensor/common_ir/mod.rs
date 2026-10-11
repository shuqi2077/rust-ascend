//! Explicit synchronous execution of `AscendCompiler` results on CannSession.
//! No generic Ruda Runtime/Backend or PyTorch device is claimed. No ACLNN math
//! fallback exists here. The existing session is reused for allocations/streams.
mod artifact;
use super::{CannSession,CannTensor,DType,layout::invalid,deepgemm::native::{NativeApi,Kernel}};
use crate::{CannError,CannLibrary,check_status};
use ruda_core::ir::{StorageType,ElemType,FloatKind,IntKind,UIntKind};
use rust_ascend_compiler::ascend::AscendKernel;
use std::{cell::Cell,ffi::c_void,path::Path,rc::Rc};

#[derive(Clone,Copy,Debug,Default,PartialEq,Eq)]
pub struct ProgramStats{pub launches:u64,pub synchronizations:u64,pub empty_calls:u64}
/// One checked IR program and one cached native module. Thread-confined by Rc.
/// Outputs may be reused via run_into; input/output aliasing is rejected.
pub struct CannProgram{compiled:AscendKernel,kernel:ProgramImage,stats:Cell<ProgramStats>,poisoned:Cell<bool>}
type SimtLaunch=unsafe extern "C" fn(*mut c_void,*mut *mut c_void)->i32;
enum ProgramImage { Vector(Kernel), Simt(SimtImage) }
struct SimtImage { library:Option<CannLibrary>, launch:SimtLaunch, session:Rc<CannSession> }
impl Drop for SimtImage {
    fn drop(&mut self) {
        if !self.session.releasable() {
            if let Some(library)=self.library.take(){std::mem::forget(library);}
            std::mem::forget(self.session.clone());
        }
    }
}
impl ProgramImage {
    fn session(&self)->&Rc<CannSession>{match self {Self::Vector(k)=>&k.api.session,Self::Simt(k)=>&k.session}}
}
fn dtype(storage:StorageType)->Result<DType,CannError>{
    Ok(match storage {
        StorageType::Scalar(ElemType::Float(FloatKind::F32))=>DType::F32,
        StorageType::Scalar(ElemType::Float(FloatKind::F16))=>DType::F16,
        StorageType::Scalar(ElemType::Float(FloatKind::BF16))=>DType::BF16,
        StorageType::Scalar(ElemType::Int(IntKind::I8))=>DType::I8,
        StorageType::Scalar(ElemType::Int(IntKind::I16))=>DType::I16,
        StorageType::Scalar(ElemType::Int(IntKind::I32))=>DType::I32,
        StorageType::Scalar(ElemType::Int(IntKind::I64))=>DType::I64,
        StorageType::Scalar(ElemType::UInt(UIntKind::U8))=>DType::U8,
        StorageType::Scalar(ElemType::UInt(UIntKind::U16))=>DType::U16,
        StorageType::Scalar(ElemType::UInt(UIntKind::U32))=>DType::U32,
        StorageType::Scalar(ElemType::UInt(UIntKind::U64))=>DType::U64,
        StorageType::Scalar(ElemType::Bool)=>DType::Bool,
        _=>return Err(invalid("unsupported common-IR binding dtype")),
    })
}
impl CannProgram{
    /// # Safety
    /// The artifact directory and CANN libraries must be trusted. The contract
    /// and digests are corruption/staleness checks, not a binary safety sandbox.
    /// CannSession's exclusive context/stream/lifecycle obligations still apply.
    pub unsafe fn load(session:&Rc<CannSession>,compiled:AscendKernel,artifact_dir:impl AsRef<Path>)->Result<Self,CannError>{
        if cfg!(not(target_pointer_width="64"))||cfg!(not(target_endian="little")){return Err(invalid("common IR ABI requires a 64-bit little-endian host"));}
        let image=artifact::read(artifact_dir.as_ref(),&compiled)?;
        let api=NativeApi::resolve(session)?;api.check_soc()?;
        let kernel=if compiled.simt_grid().is_some(){
            session.bind()?;
            let library=unsafe{CannLibrary::load_from(artifact_dir.as_ref().join(compiled.artifact_name()))}?;
            let launch=unsafe{*library.symbol::<SimtLaunch>(c"ruda_ascend_launch")?};
            ProgramImage::Simt(SimtImage{library:Some(library),launch,session:session.clone()})
        }else{ProgramImage::Vector(Kernel::load(api,image,compiled.entrypoint())?)};
        Ok(Self{compiled,kernel,stats:Cell::new(ProgramStats::default()),poisoned:Cell::new(false)})
    }
    pub fn compiled(&self)->&AscendKernel{&self.compiled}
    pub fn stats(&self)->ProgramStats{self.stats.get()}
    /// Allocating convenience API; outputs are flattened because the IR observes
    /// only a contiguous binding domain. Row statistics are allocated by their own
    /// compiled byte length, not by the full activation length. No implicit tensor broadcasting.
    pub fn run(&self,inputs:&[&CannTensor])->Result<Vec<CannTensor>,CannError>{
        if self.compiled.requires_initialized_outputs(){return Err(invalid("in-place IR requires initialized output tensors; use run_into"));}
        let mut out=Vec::new();for b in self.compiled.bindings(){if b.writable{out.push(self.kernel.session().allocate_tensor(&[(b.bytes / b.storage.size() as u64) as i64],dtype(b.storage)?)?);}}
        let mut refs:Vec<_>=out.iter_mut().collect();self.run_into(inputs,&mut refs)?;Ok(out)
    }
    pub fn run_into(&self,inputs:&[&CannTensor],outputs:&mut[&mut CannTensor])->Result<(),CannError>{
        if self.poisoned.get(){return Err(invalid("program is poisoned after a launch/completion failure; synchronize and recreate"));}
        let mut all:Vec<&CannTensor>=inputs.to_vec();all.extend(outputs.iter().map(|t|&**t));
        self.kernel.session().same_session(&all)?;
        let expected_inputs=self.compiled.bindings().iter().filter(|b|!b.writable).count();
        let expected_outputs=self.compiled.bindings().len()-expected_inputs;
        if inputs.len()!=expected_inputs||outputs.len()!=expected_outputs{return Err(invalid("common-IR input/output count mismatch"));}
        let mut read=inputs.iter();let mut write=outputs.iter();let mut addresses=Vec::<*mut c_void>::new();let mut ranges=Vec::new();
        for b in self.compiled.bindings(){
            let t:&CannTensor=if b.writable{&**write.next().ok_or_else(||invalid("missing output"))?}else{read.next().ok_or_else(||invalid("missing input"))?};
            if t.layout.dtype()!=dtype(b.storage)? || t.layout.byte_len() as u64!=b.bytes{return Err(invalid("buffer dtype/byte length does not match compiled binding domain"));}
            let start=t.buffer.data.as_ptr() as usize;
            let end=start.checked_add(t.layout.byte_len()).ok_or_else(||invalid("device address overflow"))?;
            ranges.push((start,end,b.writable));addresses.push(t.buffer.data.as_ptr());
        }
        validate_ranges(&ranges)?;
        self.launch_addresses(addresses)
    }
    /// Runtime-owned buffers remain allocated on the same worker through completion.
    #[cfg(feature = "runtime")]
    pub(super) unsafe fn run_addresses(&self, addresses:Vec<(*mut c_void,usize)>)->Result<(),CannError>{
        if addresses.len()!=self.compiled.bindings().len(){return Err(invalid("runtime binding count mismatch"));}
        let mut ranges=Vec::new();
        for ((p,n),b) in addresses.iter().zip(self.compiled.bindings()){
            if *n as u64!=b.bytes || (*p as usize)%b.storage.size()!=0{return Err(invalid("runtime binding size/alignment mismatch"));}
            let start=*p as usize;
            ranges.push((start,start.checked_add(*n).ok_or_else(||invalid("runtime address overflow"))?,b.writable));
        }
        validate_ranges(&ranges)?;
        self.launch_addresses(addresses.into_iter().map(|(p,_)|p).collect())
    }
    fn launch_addresses(&self,mut addresses:Vec<*mut c_void>)->Result<(),CannError>{
        if self.poisoned.get(){return Err(invalid("program is poisoned after a launch/completion failure"));}
        self.kernel.session().bind()?;
        if self.compiled.elements()==0{let mut st=self.stats.get();st.empty_calls+=1;self.stats.set(st);return Ok(());}
        // Keep BOTH host arrays stable and alive until this stream completes.
        let mut argv:Vec<*mut c_void>=addresses.iter_mut().map(|p|(p as *mut *mut c_void).cast()).collect();
        let session=self.kernel.session();session.quiescent.set(false);
        // SAFETY: exact generated signature is one GM pointer per binding;
        // image, arrays, session and tensors remain alive through synchronization.
        let launch=unsafe{match &self.kernel {
            ProgramImage::Vector(k)=>(k.api.launch)(k.function.as_ptr(),self.compiled.block_dim(),session.stream,std::ptr::null_mut(),argv.as_mut_ptr()),
            ProgramImage::Simt(k)=>(k.launch)(session.stream,addresses.as_mut_ptr()),
        }};
        let sync=unsafe{session.api.aclrtSynchronizeStream(session.stream)};
        session.quiescent.set(sync==0);let mut st=self.stats.get();st.launches+=1;st.synchronizations+=1;self.stats.set(st);
        if launch!=0||sync!=0{self.poisoned.set(true);}
        if sync!=0{return Err(CannError::Completion{operation:"common IR Ascend kernel",launch_code:launch,sync_code:sync});}
        check_status("common IR kernel launch",launch)
    }
}
fn validate_ranges(ranges:&[(usize,usize,bool)])->Result<(),CannError>{
    for(i,&(a,b,w))in ranges.iter().enumerate(){for&(c,d,x)in &ranges[..i]{if (w||x)&&a<d&&c<b{return Err(invalid("writable common-IR buffer overlaps another binding"));}}}Ok(())
}
#[cfg(test)]mod tests{use super::*;
#[test]fn read_alias_allowed_write_alias_rejected(){assert!(validate_ranges(&[(100,200,false),(100,200,false),(300,400,true)]).is_ok());assert!(validate_ranges(&[(100,200,false),(199,201,true)]).is_err());assert!(validate_ranges(&[(100,200,true),(150,250,true)]).is_err());assert!(validate_ranges(&[(100,200,true),(200,250,true)]).is_ok());}
}

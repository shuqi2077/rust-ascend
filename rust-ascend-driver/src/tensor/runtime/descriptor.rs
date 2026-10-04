//! Worker-local ACL descriptor borrowing managed device storage, never freeing it.
use crate::{CannError,tensor::{CannSession,TensorLayout,ffi::AclTensor}};
use std::{ffi::c_void,ptr::NonNull,rc::Rc};

pub(super) struct Descriptor {
    session:Rc<CannSession>,
    _layout:TensorLayout,
    _storage: Vec<i64>,
    pub handle:NonNull<AclTensor>,
}
impl Descriptor {
    /// Caller retains the allocation's ManagedResource guard through synchronized execution.
    pub unsafe fn new(session:&Rc<CannSession>,layout:TensorLayout,pointer:*mut c_void)->Result<Self,CannError> {
        let storage = layout.shape().to_vec();
        unsafe { Self::with_storage(session, layout, pointer, storage) }
    }
    /// Borrow a view whose reachable storage is a checked flat span.
    pub unsafe fn view(session:&Rc<CannSession>,layout:TensorLayout,pointer:*mut c_void)->Result<Self,CannError> {
        let storage = vec![i64::try_from(layout.byte_len() / layout.dtype().bytes())
            .map_err(|_| CannError::InvalidTensor("descriptor storage length overflow".into()))?];
        unsafe { Self::with_storage(session, layout, pointer, storage) }
    }
    unsafe fn with_storage(session:&Rc<CannSession>,layout:TensorLayout,pointer:*mut c_void,storage:Vec<i64>)->Result<Self,CannError> {
        session.bind()?;
        // SAFETY: checked layout and allocation range; metadata remains owned below.
        let handle=unsafe {(session.ops.create_tensor)(layout.shape().as_ptr(),layout.shape().len() as u64,
            layout.dtype() as i32,layout.strides().as_ptr(),0,2,storage.as_ptr(),
            storage.len() as u64,pointer)};
        let handle=NonNull::new(handle).ok_or(CannError::NullHandle("aclCreateTensor"))?;
        Ok(Self {session:session.clone(),_layout:layout,_storage:storage,handle})
    }
}
impl Drop for Descriptor {
    fn drop(&mut self) {
        if self.session.releasable() {
            // SAFETY: this descriptor is unique and device work has completed.
            let _=unsafe {(self.session.ops.destroy_tensor)(self.handle.as_ptr())};
        } else {
            std::mem::forget(self.session.clone());
        }
    }
}

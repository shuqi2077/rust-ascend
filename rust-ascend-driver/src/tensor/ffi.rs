use crate::{CannError, CannLibrary};
use std::ffi::{CStr, c_void};

#[repr(C)]
pub struct AclTensor {
    _opaque: [u8; 0],
}
#[repr(C)]
pub struct AclScalar {
    _opaque: [u8; 0],
}
#[repr(C)]
pub struct AclIntArray {
    _opaque: [u8; 0],
}
#[repr(C)]
pub struct AclOpExecutor {
    _opaque: [u8; 0],
}
pub type Status = i32;
pub type Run =
    unsafe extern "C" fn(*mut c_void, u64, *mut AclOpExecutor, crate::sys::AclStream) -> Status;
pub type CreateTensor = unsafe extern "C" fn(
    *const i64,
    u64,
    i32,
    *const i64,
    i64,
    i32,
    *const i64,
    u64,
    *mut c_void,
) -> *mut AclTensor;
pub type DestroyTensor = unsafe extern "C" fn(*const AclTensor) -> Status;
pub type ExecutorAction = unsafe extern "C" fn(*mut AclOpExecutor) -> Status;

pub struct OperatorApi {
    libraries: Vec<CannLibrary>,
    pub create_tensor: CreateTensor,
    pub destroy_tensor: DestroyTensor,
    pub destroy_executor: ExecutorAction,
}

impl OperatorApi {
    pub unsafe fn load_libraries(libraries: Vec<CannLibrary>) -> Result<Self, CannError> {
        if libraries.is_empty() {
            return Err(super::layout::invalid("at least one CANN operator library is required"));
        }
        // SAFETY: the caller supplied the matching CANN operator library.
        unsafe {
            Ok(Self {
                create_tensor: Self::resolve(&libraries, c"aclCreateTensor")?,
                destroy_tensor: Self::resolve(&libraries, c"aclDestroyTensor")?,
                destroy_executor: Self::resolve(&libraries, c"aclDestroyAclOpExecutor")?,
                libraries,
            })
        }
    }
    unsafe fn resolve<T: Copy>(libraries: &[CannLibrary], name: &CStr) -> Result<T, CannError> {
        let mut errors = Vec::new();
        for library in libraries {
            match unsafe { library.symbol::<T>(name) } {
                Ok(symbol) => return Ok(*symbol),
                Err(error) => errors.push(error.to_string()),
            }
        }
        Err(CannError::Symbol {
            name: name.to_string_lossy().into_owned(),
            message: errors.join("; "),
        })
    }
    pub unsafe fn get<T: Copy>(&self, name: &CStr) -> Result<T, CannError> {
        // SAFETY: call sites specify the signature of the named SDK function.
        unsafe { Self::resolve(&self.libraries, name) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_operator_libraries_are_rejected_without_initializing_acl() {
        assert!(unsafe { OperatorApi::load_libraries(Vec::new()) }.is_err());
    }
}

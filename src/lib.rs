//! Ascend CANN runtime, Rust common-IR compiler and BF16 device programs.
pub mod nn;
pub mod optim;
pub mod distributed;
mod backend;
pub use backend::RudaAscend;
pub use ruda_model as model;
pub use ruda_store as store;
#[cfg(feature = "models")]
pub use rullm as models;
pub use ruda_optim::training;
pub use ruda_optim::data_parallel;
pub use ruccl as collective;
pub use ruda_core as core;
pub use rust_ascend_compiler::ascend as compiler;
pub use rust_ascend_driver as driver;
pub use rust_ascend_driver::runtime;
pub use ruda_tensor as tensor;
pub use ruda_autodiff::Autodiff;
pub type Ascend = ruda_tensor_device::DeviceBackend<runtime::AscendRuntime, f32, i32, u8>;
pub use rust_ascend_kernels as kernels;

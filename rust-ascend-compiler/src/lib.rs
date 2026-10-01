#[cfg(feature = "ascend")]
pub mod ascend;

#[cfg(feature = "ptx")]
pub use upstream_compiler::ptx;

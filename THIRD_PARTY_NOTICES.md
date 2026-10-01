# Third-party notices

The Ascend compiler, CANN driver, build tools and tests are extracted from
[RUDA](https://github.com/shuqi2077/RUDA/tree/38746646f90bbba0523cb51ec9429817b96c94ad),
under the Apache License 2.0. Their paths, crate names and workspace integration
are adapted for this independent repository. RUDA shared components remain
external dependencies under their respective licenses.

The direct-store BF16 algorithm, persistent scheduler, tile offsets and pipeline
protocol in `rust-ascend-kernels` are translated/adapted from DeepGEMM-Ascend,
copyright 2026 DeepSeek, MIT license. Source commit:
`8491bbb4b8c02a094a2318965f50c70438a3e73c`.

The Rust port retains its MIT license, upstream notice and source digests in
`rust-ascend-kernels/LICENSE` and `rust-ascend-kernels/UPSTREAM.json`.
Original C++ device headers are not redistributed. Generated CCE still requires
Bisheng. CANN/DeepJIT/torch_npu binaries and toolchains are not redistributed.

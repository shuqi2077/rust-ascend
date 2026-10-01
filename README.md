# rust-ascend

Rust 编写的昇腾 CANN 驱动、公共 IR 编译器和设备内核。

## 使用

```bash
cargo add rust-ascend --git https://github.com/shuqi2077/rust-ascend.git
```

需要 Rust 2024 工具链。编译器与主机测试不需要安装 CANN；生成设备二进制和运行设备算子需要目标机器上的 CANN、Bisheng、驱动及昇腾设备。

```rust
use rust_ascend::{
    compiler::{AscendCompiler, AscendOptions, AscendTarget},
    driver::tensor::{CannSession, CannTensor},
};
```

| 入口 | 内容 |
|---|---|
| `rust_ascend::compiler` | 公共 Kernel IR → CCE 编译；FP32 逐元素、行归约和归一化 |
| `rust_ascend::driver` | AscendCL/ACLNN 动态加载、设备张量、原生内核执行 |
| `rust_ascend::kernels` | Rust BF16 矩阵设备程序与 CCE 生成 |
| `rust_ascend::core` | RUDA 公共 IR 与编译器接口 |

ACLNN 路径另提供 `cast`、`silu_backward`、`softmax_backward`、`log_softmax_backward` 和 `rms_norm_backward`。反向接口支持 FP32/FP16/BF16；RMSNorm 返回输入梯度及 FP32 权重梯度。调用示例见 [gradients](examples/gradients.rs)。

`CannSession::open_exclusive_libraries` / `attach_libraries` 可显式传入 CANN 9 的 `libnnopbase.so`、`libopapi_math.so`、`libopapi_nn.so` 等拆分库；原有单库接口保留。

公共 IR 依赖 `ruda-core`。可选的 `rust-ascend-compiler/ptx` 使用 RUDA PTX 编译器检查同一 IR 的兼容性，不改变默认昇腾执行路径。

## 生成与执行

```bash
git clone https://github.com/shuqi2077/rust-ascend.git
cd rust-ascend

# 输出 Softmax CCE 源码，不执行设备计算。
cargo run --locked --release --example softmax

# 生成 RMSNorm 的 CCE、IR 和绑定契约。
python tools/ascend/build_common_ir.py --emit-only \
  --op rms_norm --elements 12288 --row-width 4096 --out ./target/rmsnorm-source

# 生成 BF16 矩阵内核源码。
python tools/ascend/build_deepgemm.py --emit-only --out ./target/bf16-source
```

移除 `--emit-only` 并提供 `--toolkit "$ASCEND_HOME_PATH"`，即可调用 CANN 工具链编译设备产物。输出目录必须是新目录。

设备执行接口与完整调用见：

- [公共逐元素程序](rust-ascend-driver/examples/common_ir_validate.rs)
- [行归约、Softmax、RMSNorm 及部分反向程序](rust-ascend-driver/examples/common_rows_validate.rs)
- [BF16 矩阵程序](rust-ascend-driver/examples/deepgemm_validate.rs)

## 支持范围

- 公共编译器：连续 FP32 逐元素程序；行宽为 32～4096、且为 32 的倍数。
- 行计算：sum/mean/max、Softmax/LogSoftmax、RMSNorm，以及 Softmax/LogSoftmax 和 RMSNorm 的输入梯度。
- BF16 矩阵：direct-store Dense/Batched NN/NT/TN/TT、对齐的 MGrouped NT，BF16/FP32 输出。
- 设备代码目标为 Ascend950DT / dav-c310；不自动推断或替换目标型号。
- Rust 程序生成 CCE，再由 Bisheng 编译为设备机器码，不是直接 Rust → 昇腾 ISA。
- 不包含完整 PyTorch 昇腾后端、公共 IR 中的 RMSNorm 权重梯度、通用低精度行计算或任意 stride/广播。

## 测试入口

```bash
cargo test --locked --release --workspace --all-features --all-targets
python -m pip install -r tools/ascend/requirements-test.txt
python -m pytest tools/ascend/tests -q
```

配置真实 CANN 环境后，设备验证入口为：

```bash
python tools/ascend/validate.py --toolkit "$ASCEND_HOME_PATH" --output ./target/bf16-device
python tools/ascend/validate_common_ir.py --toolkit "$ASCEND_HOME_PATH" --output ./target/map-device
python tools/ascend/validate_common_rows.py --toolkit "$ASCEND_HOME_PATH" --output ./target/row-device
```

GitHub Actions 执行主机编译、测试和 CCE 源码生成；CANN 设备编译和数值验证使用上述设备入口。

## 许可

编译器、驱动和统一入口使用 [Apache-2.0](LICENSE)。BF16 设备程序保留 [MIT](rust-ascend-kernels/LICENSE) 许可和上游声明，详见 [第三方说明](THIRD_PARTY_NOTICES.md)。

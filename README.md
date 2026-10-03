# rust-ascend

Rust 编写的昇腾 CANN 驱动、公共 IR 编译器和设备内核。

## 使用

```bash
cargo add rust-ascend
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
| `rust_ascend::runtime` | RUDA `Runtime` / `ComputeServer` / `ComputeStorage` 接口、设备工作线程与 CCE 即时编译 |

ACLNN 路径另提供 `cast`、`silu_backward`、`softmax_backward`、`log_softmax_backward` 和 `rms_norm_backward`。反向接口支持 FP32/FP16/BF16；RMSNorm 返回输入梯度及 FP32 权重梯度。调用示例见 [gradients](examples/gradients.rs)。

`CannSession::open_exclusive_libraries` / `attach_libraries` 可显式传入 CANN 9 的 `libnnopbase.so`、`libopapi_math.so`、`libopapi_nn.so` 等拆分库；原有单库接口保留。

公共 IR 依赖 `ruda-core`。可选的 `rust-ascend-compiler/ptx` 使用 RUDA PTX 编译器检查同一 IR 的兼容性，不改变默认昇腾执行路径。

`AscendRuntime::initialize_exclusive(RuntimeOptions::new(toolkit))` 初始化进程独占的 950DT 运行时，随后通过 RUDA `ComputeClient` 分配、上传、启动 `RudaTask<AscendCompiler>`、读取及同步。CANN 会话、设备内存和模块固定在设备线程；公共客户端只传递资源句柄。程序由安装的 Bisheng / ld.lld 编译、加载，并在进程内按源码和绑定契约缓存。每次编译或链接默认限时 120 秒，可通过 `compile_timeout` 配置。此入口不可与 `torch_npu` 或其他 ACL 初始化方共用；现有借用会话的 `attach` 接口不变。

运行时示例：`cargo run --release --example runtime`，需要 `ASCEND_HOME_PATH`。示例直接使用 RUDA 计算客户端，不需要预先生成算子目录。运行时按 RUDA 参数 ABI 解析标量和 metadata，再按本次参数特化公共 IR；FP32 标量保留原始位模式，生成代码参与模块缓存键。支持 u32/u64 逻辑索引和同一绑定上的原地逐元素读写；任意布局与完整模型所需的指令覆盖仍不包含在当前编译范围内。

## RUDA 张量与自动求导

`Ascend` 复用 `ruda-tensor-device::DeviceBackend`，`Autodiff<Ascend>` 复用 RUDA 自动求导。当前公共编译路径支持 FP32 逐元素运算的连续输入，以及可证明索引范围的广播、转置和带间隔输入；输出要求连续布局，非连续原地写回不在此范围内。连续输入保留整块搬运，非连续输入在设备端按索引搬运后执行向量运算。创建张量时显式指定 `DType::F32`；整数、布尔和通用低精度运算不由此入口提供。未覆盖的 IR 返回错误，不切换到 CPU 或其他数学后端。

```rust
use rust_ascend::{Ascend, Autodiff, tensor::{DType, api::Tensor}};

// device 来自 AscendRuntime::initialize_exclusive。
let x = Tensor::<Autodiff<Ascend>, 1>::from_data(
    [1.0f32, 2.0, 3.0, 4.0], (&device, DType::F32),
).require_grad();
let y = x.clone() * x.clone();
let gradients = y.backward();
let dx = x.grad(&gradients).unwrap();
```

完整进程初始化与调用见 [tensor 示例](examples/tensor.rs)：`cargo run --release --example tensor`。

公共 map 编译器对可严格证明的 32 字节对齐连续区间使用分段整块搬运，行广播在每段只读取一次并在设备本地展开；宽行归一化的输入块、共享 weight 和输出 patch 复用这一通路。非连续步长、无法证明的区间或不满足本地对齐的索引仍按原设备逐元素路径执行。索引边界、FP32 计算、尾部 padding 与未写输出保留语义不变。

## 整数索引 Embedding

`CannSession::embedding` / `embedding_backward` 和 `AscendRuntime` 的对应接口使用 ACLNN 设备端查表与 dense 权重反向。连续权重支持 FP32／FP16／BF16；索引保留 INT32／INT64，rank 为 1～7，输出形状为索引形状追加权重宽度。索引值由调用方保证在 `[0, vocabulary_size)` 内，不搬回主机、不经 FP32 转换，也不隐式 clamp。支持空索引和非 32 对齐宽度。

`nn::embedding(weight, indices, options)` 接收 FP32 `weight[V,H]` 与整数 `indices[B,S]`，返回 `[B,S,H]`；`embedding_nd` 提供任意上述索引 rank。`Autodiff<Ascend>` 接入 RUDA 图，反向累积重复 ID 的 dense 权重梯度；`EmbeddingOptions` 显式选择 padding 行和按频次缩放，默认两者均不启用。padding 只屏蔽该行梯度，不改写前向表值。已跟踪的前向保存独立的设备到设备整数索引快照，反向不依赖后来修改的原索引。

调用见 [embedding_tensor 示例](examples/embedding_tensor.rs)：`cargo run --locked --release --example embedding_tensor`。

## 计算客户端原生 BF16 矩阵

`AscendRuntime::gemm` / `gemm_into` 使用 RUDA `ComputeClient` 的 `TensorBuffer`，在设备线程按需生成、编译和缓存已有 Rust BF16 矩阵内核，无需预先准备算子目录。支持连续 BF16 输入、BF16／FP32 输出，rank-2 Dense 和 batch 数相同的 rank-3 Batched，以及 NN／NT／TN／TT 四种转置组合。M／N／K 必须为正且是 16 的倍数，不做 batch 广播。

`AscendRuntime::linear_nt_backward` 为 rank-2 `Y = X W^T` 返回 `[BF16 dX, FP32 dWeight]`，输入、weight 和 `dY` 为 BF16。这些是显式原生运行时接口，不把公共 FP32 IR 编译器改成 BF16 编译器，也不自动注册张量 matmul 的自动求导。调用见 [matrix_runtime 示例](examples/matrix_runtime.rs)：`cargo run --locked --release --example matrix_runtime`。

## 显式设备精度转换

`AscendRuntime::cast` / `cast_into` 接收 RUDA `ComputeClient` 的连续 rank-1～8 `TensorBuffer`，在 CANN 设备端显式转换 FP32／FP16／BF16，保留形状；后者写入已有、与输入不重叠的输出缓冲区。支持空张量。输入值不搬回主机，设备描述符、workspace 和执行完成均由同一设备线程管理。

此入口复用已有 ACLNN Cast，不冒充公共 IR 内核，不隐式转换矩阵或优化器输入，也不修改通用张量 cast 的调度。调用见 [conversion_runtime 示例](examples/conversion_runtime.rs)：`cargo run --locked --release --example conversion_runtime`。

## BF16 计算、FP32 存储的线性层

`nn::linear_bf16_fp32(input, weight)` 显式计算 `Y = X W^T`，接收连续 FP32 `X[M,K]` 与 `W[N,K]`，在设备端转换为 BF16 后执行已有 Rust 原生矩阵内核，返回 FP32 输出。`Autodiff<Ascend>` 保存独立的 BF16 输入快照；反向将 FP32 上游梯度显式转换为 BF16，再由原生矩阵内核生成 FP32 `dX` 与 `dWeight`，接入 RUDA 共享图梯度累积和 FP32 参数优化器。精度转换使用 ACLNN Cast，矩阵乘不调用 ACLNN Matmul。

M／N／K 必须为正且为 16 的倍数；此入口不含 bias、batch 广播或隐式 padding，不改变通用张量 `matmul` 的调度。对应运行时接口为 `AscendRuntime::linear_bf16_fp32` / `linear_bf16_fp32_backward`。调用见 [linear_tensor 示例](examples/linear_tensor.rs)：`cargo run --locked --release --example linear_tensor`。

`nn::matmul_bf16_fp32(a, b, ta, tb)` 将相同显式精度模式扩展到 rank-2 Dense 与 batch 数相同的 rank-3 Batched，支持 NN／NT／TN／TT。反向保持两个输入各自的物理形状，不创建完整转置副本；FP32 梯度通过 RUDA 图累积。对应运行时接口为 `gemm_bf16_fp32` / `gemm_bf16_fp32_backward`，仍要求 M／N／K 为正且为 16 的倍数，不做 batch 广播。调用见 [matmul_tensor 示例](examples/matmul_tensor.rs)：`cargo run --locked --release --example matmul_tensor`。

## 显式组合 Attention

`nn::scaled_dot_product_attention_bf16_fp32(q, k, v, scale, additive_mask)` 接收 FP32 `Q[B,M,D]`、`K[B,N,D]`、`V[B,N,Dv]`，组合原生 BF16 矩阵乘、FP32 scale／可选加性 mask、原生 FP32 Softmax 与第二次原生矩阵乘；输出及 Q／K／V／可训练 mask 的梯度接入 RUDA 自动求导。scale 显式提供，mask 为同设备 FP32 `[B,M,N]`，可用负无穷屏蔽 key，不做 mask 广播。

该入口物化 score 和 probability 矩阵，不是 FlashAttention；没有隐式 causal mask、dropout、GQA head 重复或 padding。各 batch 数必须相同，M／D／Dv 为正且为 16 的倍数，N 为正且为 32 的倍数，各矩阵轴不超过 INT32_MAX，score 总元素数不超过 u32；N 超过 4096 时接入原生分块 Softmax。矩阵计算与反向使用前述显式 BF16 精度模式。调用见 [attention_tensor 示例](examples/attention_tensor.rs)：`cargo run --locked --release --example attention_tensor`。

## 设备端因果 Mask 与位置偏移

`CausalMaskSpec` 显式提供 batch、query／key 长度与两个 u64 绝对起始位置。`AscendRuntime::causal_mask` / `nn::causal_mask` 在设备端生成 FP32 `[B,Q,K]`：`key_start + key > query_start + query` 时写入精确负无穷，否则写入正零。比较使用公共 IR 的 UInt32／UInt64 索引和 FP32 Select，不先转换成浮点位置；支持超过 `2^24` 的位置、空 batch、非对齐 query／key 长度，总输出元素数不超过 u32。

`nn::causal_attention_bf16_fp32` 显式组合这一固定 mask 与已有 Attention，起始位置适用于 prefill 或带 KV 前缀的 query chunk；仍使用 BF16 计算／FP32 存储、原生 FP32 Softmax 及 RUDA Q／K／V 自动求导，矩阵尺寸限制不变。不自动构建或更新 KV cache，也不改变原有可选 additive mask 接口。

调用见 [causal_attention_tensor 示例](examples/causal_attention_tensor.rs)：`cargo run --locked --release --example causal_attention_tensor`。

## GQA 与 KV 头梯度归并

`nn::repeat_kv_heads(input, query_heads)` 将连续 FP32 `[B,Hkv,N,D]` 变为 `[B,Hq,N,D]`，每个 KV 头连续重复 `Hq/Hkv` 次；Hq／Hkv 为正且可整除，支持 MHA、MQA、GQA、奇数重复次数和空张量。公共 IR 使用整数索引复制，保留 FP32 位模式；反向在设备端按 KV 头分组逐级成对求和，奇数末组原样传递，不使用原子累加，不保存输入值或将梯度搬回主机。完整输入／输出元素数不超过 u32。运行时对应 `AscendRuntime::repeat_kv_heads` / `repeat_kv_heads_backward`。

`nn::grouped_query_attention_bf16_fp32` 接收 `Q[B,Hq,M,D]`、`K[B,Hkv,N,D]`、`V[B,Hkv,N,Dv]`，接入上述原生头重复与梯度归并，再复用 BF16 计算／FP32 存储的 Attention。可选同设备 FP32 additive mask 必须为 `[B,Hq,M,N]`，其梯度也接入 RUDA 图；不做 mask 广播。`nn::causal_grouped_query_attention_bf16_fp32` 另提供显式 query／key 绝对起始位置，使用固定的设备端因果 mask。

这些组合入口沿用 Attention 的矩阵对齐和域限制，且展平后的 `B*Hq` 不超过 4096；物化重复的 K／V、score 和 probability，不是融合 FlashAttention，不自动维护 KV cache、添加 dropout 或 padding。调用见 [grouped_attention_tensor 示例](examples/grouped_attention_tensor.rs)：`cargo run --locked --release --example grouped_attention_tensor`。

## 原生旋转位置编码

`nn::rotary(input, cos, sin, RotaryLayout)` 使用公共 Rust IR 执行 FP32 全末轴旋转，支持 `Interleaved` 相邻配对与 `SplitHalf` 前后半轴配对。输入为连续 rank-1～8，末轴宽度为正偶数；cos／sin 是同设备固定 `Tensor<Ascend, D>`，前导维度与输入一致，末轴宽度减半。频率、位置、base 与缩放策略由调用方明确提供，不隐式生成或广播表；支持空前导 batch。

`Autodiff<Ascend>` 保存 cos／sin，不保存输入值，通过原生转置 Jacobian 计算输入梯度；不计算固定表的梯度。运行时对应 `AscendRuntime::rotary` / `rotary_backward`。调用见 [rotary_tensor 示例](examples/rotary_tensor.rs)：`cargo run --locked --release --example rotary_tensor`。

`nn::rotary_prefix(input, cos, sin, P, layout)` 扩展为正偶数 `P<=D` 的前缀旋转，其余末轴值在前向和反向中直接保留，不参与浮点运算。输入和固定表均为连续 FP32、rank-1～8；表的 rank 与输入相同、末轴为 P/2，前导轴可为 1 或与输入对应轴相同。公共整数索引直接读取紧凑广播表，不先展开表或转换位置索引为 FP32；例如输入 `[B,H,N,D]` 可使用 `[1,1,N,P/2]` 的共享序列表或 `[B,1,N,P/2]` 的逐 batch 表。支持两种配对布局、空前导轴及 RUDA 输入梯度，共享图梯度正常累积。旧 `rotary` 入口不变；频率、位置及缩放仍由调用方提供。

运行时对应 `rotary_prefix` / `rotary_prefix_backward`，公共 IR 配置为 `PrefixRotarySpec`。调用见 [rotary_prefix_tensor 示例](examples/rotary_prefix_tensor.rs)：`cargo run --locked --release --example rotary_prefix_tensor`。

## LoRA 与 SwiGLU 组合

`nn::bias_add(input, bias)` 显式执行 FP32 `X[...,H] + bias[H]`；`nn::residual_bias_add(input, residual, bias)` 以 `(X + residual) + bias` 的 FP32 顺序融合两次加法，residual 必须与 X 完全同形状。使用公共 Rust IR 原生向量内核，支持连续 FP32 rank 1～8、任意正 H、空 token 轴及不超过 u32 的总元素数，不做 residual 广播或类型转换。反向不保存激活值：输入及 residual 的梯度直接传递，bias 梯度在设备端按全部 token 成对求和，支持共享输入／图的梯度累积。固定 bias 不触发无用的 bias 梯度归约，不改变 Linear／LoRA 的无 bias 默认行为。调用见 [bias_tensor 示例](examples/bias_tensor.rs)：`cargo run --locked --release --example bias_tensor`。

`nn::lora_linear_bf16_fp32(input, weight, down, up, scale)` 计算 `X W^T + scale * (X A^T) B^T`，使用 FP32 参数／输出和显式 BF16 线性计算，复用 RUDA 自动求导及共享输入梯度累积。权重形状为 `W[N,K]`、`A[R,K]`、`B[N,R]`；M／N／K／R 为正且为 16 的倍数。scale 显式提供，base 或 adapter 是否冻结由调用方的 `require_grad` 决定，不自动修改权重、合并 adapter、添加 dropout 或 padding。原生前向 → 两个 adapter 梯度 → FP32 AdamW 调用见 [lora_tensor 示例](examples/lora_tensor.rs)：`cargo run --locked --release --example lora_tensor`。

`nn::swiglu_bf16_fp32(input, gate, up, down)` 计算 `(SiLU(X Wgate^T) * (X Wup^T)) Wdown^T`，组合原生线性计算、FP32 SiLU 门控乘法与 RUDA 求导。gate／up 为 `[H,K]`，down 为 `[N,H]`；M／N／K／H 为正且为 16 的倍数。不推断模型维度、bias、dropout、residual 或 normalization。调用见 [swiglu_tensor 示例](examples/swiglu_tensor.rs)：`cargo run --locked --release --example swiglu_tensor`。

## 显式矩阵 padding 与任意正 LoRA rank

`nn::matmul_padded_bf16_fp32(a, b, ta, tb)` 与 `nn::linear_padded_bf16_fp32(input, weight)` 接收连续 FP32 参数，支持正 M／N／K 不为 16 倍数的逻辑尺寸。设备端用 ACLNN Cast 转成 BF16，再用 ACLNN ConstantPadNd 补零到 16 对齐，矩阵计算仍执行本仓库的 Rust 原生 GEMM；FP32 输出和两侧梯度裁回原始物理形状。Matmul 支持 rank-2／相同 batch 的 rank-3 和四种转置组合，不做 batch 广播；对齐后的矩阵轴不超过 INT32_MAX，batch 为 1～4096，沿用原生调度器的 tile 域限制。

`nn::lora_padded_linear_bf16_fp32` 和 `nn::swiglu_padded_bf16_fp32` 复用这一入口，LoRA rank 可为任意正数，包括 1、7、8。每层先裁回逻辑激活，再为下一层补零，padding 通道不参与模型激活。输入、参数及其梯度保持 FP32 存储，计算精度仍是显式 BF16 模式。

`nn::linear_frozen_padded_bf16_fp32`、`nn::lora_frozen_padded_linear_bf16_fp32`、`nn::swiglu_frozen_padded_bf16_fp32` 接收已有固定 BF16 权重、FP32 输入和 adapter；只对固定权重路径计算输入梯度。对齐权重复用原存储，不对齐权重需要额外的补零 BF16 缓冲区；不展开成 FP32，也不保存线性层输入值。调用方须保持固定权重到反向结束。可训练路径保存独立的补零 BF16 输入快照。以上入口均显式分配 padding／裁剪缓冲区，不改变原有无 padding 接口，不含 bias、dropout 或权重合并。

调用见 [padded_matrix_tensor 示例](examples/padded_matrix_tensor.rs)：`cargo run --locked --release --example padded_matrix_tensor`。

`nn::linear_padded_bf16_fp32_nd`、`nn::lora_padded_linear_bf16_fp32_nd`、`nn::swiglu_padded_bf16_fp32_nd` 直接处理 FP32 `X[...,K]`，支持 rank 1～8 和正前导维度，包括 `[K]`、`[B,S,K]` 及更高维 token 布局，输出仅将最后一维改为 N。对应 `linear_frozen_padded_bf16_fp32_nd`、`lora_frozen_padded_linear_bf16_fp32_nd`、`swiglu_frozen_padded_bf16_fp32_nd` 使用固定 BF16 基座权重。前导 token 轴展平成矩阵 M 后复用原生 padding 路径，再恢复原输出形状；RUDA reshape 图负责恢复输入梯度，参数梯度按全部 token 累加。展平后的 M 仍受原生 GEMM 域限制；不支持空 token 轴，不隐式添加 bias、dropout、head 分组或模型配置。调用见 [projection_tensor 示例](examples/projection_tensor.rs)：`cargo run --locked --release --example projection_tensor`。

`nn::scaled_dot_product_attention_padded_bf16_fp32`、`nn::causal_attention_padded_bf16_fp32`、`nn::grouped_query_attention_padded_bf16_fp32`、`nn::causal_grouped_query_attention_padded_bf16_fp32` 将这一矩阵路径接入 Attention。逻辑 M／N／D／Dv 可为任意正数，不再要求 16／32 对齐；Q／K／V、输出、Softmax 和梯度为 FP32，矩阵计算显式使用 BF16。第一次 GEMM 先裁回逻辑 score，再加 mask 和执行原生 Softmax，因此补零 key 不进入归一化；第二次 GEMM 及反向同样裁回逻辑形状。scale、完整形状的可选 additive mask、causal 绝对起始位置及 MHA／MQA／GQA 的 head 分组保持显式，支持可训练 mask 和共享图梯度累积。

这些入口沿用对齐后矩阵轴、batch／`B*Hq`、原生 tile 调度和逻辑 score 总元素数的域限制。仍物化 score／probability 及 GQA 重复的 K／V，并额外分配矩阵 padding 缓冲区；不是 FlashAttention，不添加 dropout、mask 广播或 KV cache。调用见 [padded_attention_tensor 示例](examples/padded_attention_tensor.rs)：`cargo run --locked --release --example padded_attention_tensor`。

## 直接使用冻结 BF16 权重

`nn::embedding_frozen_bf16_fp32` / `embedding_frozen_bf16_fp32_nd` 直接使用固定 BF16 `[V,H]` 表和设备端 INT32／INT64 ID，输出 FP32 激活。二维 ID 输出 `[B,S,H]`，ND 入口支持 rank-1～7 ID 并追加 H 轴；ID 须在 `[0,V)`，支持重复 ID 与空张量。不把整张表展开为 FP32，不保存 ID 反向快照，不计算固定表或整数 ID 的梯度。

`nn::linear_frozen_bf16_fp32(input, weight)` 接收 FP32 `Tensor<B,2>` 的 `[M,K]` 输入与固定 BF16 `Tensor<Ascend,2>` 的 `[N,K]` 权重。X 与上游 dY 在设备端转为 BF16，原生矩阵内核输出 FP32 Y 和 dX；M／N／K 为正且为 16 的倍数。权重直接以每元素 2 字节的现有存储参与 GEMM，不转为 FP32，不另存一份完整权重或输入快照。反向只需要原输入形状与同一份固定 BF16 权重，调用方须保持该权重在反向完成前不变；不计算权重梯度。

`nn::lora_frozen_linear_bf16_fp32` 将同一份固定 BF16 基座与 FP32 A／B adapter 组合，支持 X、A、B 的梯度及已有 FP32 AdamW 更新。`nn::swiglu_frozen_bf16_fp32` 则使用固定 BF16 gate／up／down 权重、FP32 激活和输入梯度。上述入口显式选择固定权重类型，不改变原有 `linear_bf16_fp32`、LoRA、SwiGLU 的可训练权重行为，也不全局冻结模块、合并 adapter、添加 bias／dropout 或 padding。

运行时对应 `linear_frozen_bf16_fp32` / `linear_frozen_bf16_fp32_backward`。完整调用见 [frozen_bf16_tensor 示例](examples/frozen_bf16_tensor.rs)：`cargo run --locked --release --example frozen_bf16_tensor`。

## 原生 RMSNorm 公共 IR

`AscendRuntime::rms_norm` 通过 RUDA `ComputeClient` 接收 `TensorBuffer` 的输入和共享 weight，返回 `[Y, rstd]`。`AscendRuntime::rms_norm_backward` 接收输入、weight、`dY` 及前向保存的 `rstd`，返回 `[dX, dWeight]`。`RowProgram::RmsNormWeightContributions` 生成逐元素 weight 梯度贡献，再在设备端归约所有前导行；空 batch 的 weight 梯度为零。

此接口使用公共 Rust IR 和 CCE，不调用 ACLNN RMSNorm。支持连续 FP32、最后一维宽度为正，总元素数不超过 u32。非 32 对齐或超过 4096 的宽度使用设备端分块平方和及全行 reciprocal RMS；反向复用前向统计，分块计算输入梯度及共享 weight 梯度。完整运行时调用见 [rms_norm_runtime 示例](examples/rms_norm_runtime.rs)：`cargo run --locked --release --example rms_norm_runtime`。

非对齐行只在完成计算的归约 tile 尾部补中性值：求和补正零，最大值补负无穷；输入、weight 和激活保持原逻辑形状，均值、方差和梯度统计始终除以原行宽。补尾程序使用公共 IR 的安全索引与选择操作，再执行已有原生行归约，不调用 ACLNN；单列归约直接设备复制，扩展后的 tile 域过大时按行分批。

`rust_ascend::nn::rms_norm(input, weight, epsilon)` 接收现有 RUDA `Tensor<Ascend, D>` 或 `Tensor<Autodiff<Ascend>, D>`，复用 RUDA 的计算图、梯度存储和共享图梯度累积，原生计算输入与 weight 的梯度。已有 `ruda_nn::RmsNorm` 可将 `gamma.val()` 和 `epsilon` 传给此入口；不修改该模块原有的 `forward` 方法。完整调用见 [rms_norm_tensor 示例](examples/rms_norm_tensor.rs)：`cargo run --locked --release --example rms_norm_tensor`。

## 原生 Softmax 与 LogSoftmax

`AscendRuntime::softmax` / `log_softmax` 及其 `*_backward` 接收 RUDA `TensorBuffer`，归一化连续 FP32 输入的最后一维；宽度为正，总元素数不超过 u32。32 对齐且不超过 4096 的宽度使用原行内核；其余宽度以最多 4096 列分块，在设备端合并全行最大值及指数和，再写回逻辑形状的归一化结果。反向同样分块合并全行统计，使用前向保存的输出；支持任意前导维度和空 batch，不调用 ACLNN。分块 Softmax 保留一份完整 FP32 指数工作区，LogSoftmax 不保留该完整工作区。

`rust_ascend::nn::softmax` / `log_softmax` 接收 `Tensor<Ascend, D>` 或 `Tensor<Autodiff<Ascend>, D>`，原生反向接入 RUDA 的现有计算图和梯度累积。调用见 [softmax_tensor 示例](examples/softmax_tensor.rs)：`cargo run --locked --release --example softmax_tensor`。

## 分类损失与整数标签

`CannSession::nll_loss` / `nll_loss_backward` 与 `AscendRuntime` 的对应接口使用 ACLNN NLLLoss，接收连续 FP32／FP16／BF16 的 `[N,C]` log-probabilities、INT32／INT64 `[N]` 标签及同精度 `[C]` class weight。前向返回 loss 和设备端 total weight；反向复用前向 total weight。`NllLossOptions` 要求显式选择 None／Mean／Sum 和可选 ignore index，标签须为有效类别或该 ignore 值。None 返回 `[N]`，Mean／Sum 返回 `[1]`；加权 Mean 除以未忽略标签的权重和。

`nn::nll_loss` 把 FP32 输入反向接入 RUDA 图，class weight 为固定的 inner-backend 张量。已跟踪的前向独立保存设备端整数标签与 class weight 快照，不将标签或归约计数搬回主机。`nn::cross_entropy` / `weighted_cross_entropy` 将原生 FP32 LogSoftmax 与这一 NLLLoss 路径组合，支持 logits 梯度及共享图累积；类别数为正，不再要求 32 对齐。调用方自行决定标签位移，不隐式 shift、label smoothing 或 soft targets。

调用见 [cross_entropy_tensor 示例](examples/cross_entropy_tensor.rs)：`cargo run --locked --release --example cross_entropy_tensor`。`AscendRuntime::copy_contiguous` 也可独立保存连续 FP32／FP16／BF16／INT32／INT64 设备张量，保留原存储位模式。

## 原生 SiLU 门控乘法

`AscendRuntime::silu_mul` / `silu_mul_backward` 使用已有公共逐元素 IR，计算 `SiLU(gate) * up` 与两路输入梯度。输入为形状相同的连续 FP32 缓冲区，支持任意元素数、非对齐尾部与空张量；不做隐式广播或低精度转换。

`rust_ascend::nn::silu_mul` 接收 `Tensor<Ascend, D>` 或 `Tensor<Autodiff<Ascend>, D>`，接入 RUDA 求导图与共享节点梯度累积。调用见 [silu_mul_tensor 示例](examples/silu_mul_tensor.rs)：`cargo run --locked --release --example silu_mul_tensor`。

## 原生 Sum 与 Mean

`AscendRuntime::sum_last` / `mean_last` 归约连续 FP32 输入的最后一维，保留该维且长度变为 1；宽度为正，总元素数不超过 u32。非 32 对齐或超过 4096 列时在设备端分块归约并合并全行和，Mean 最后除以完整逻辑行宽。对应反向在设备端广播每行上游梯度，Mean 再除以行宽；支持空 batch，不保存输入值，不调用 ACLNN。

`rust_ascend::nn::sum_last` / `mean_last` 接收 `Tensor<Ascend, D>` 或 `Tensor<Autodiff<Ascend>, D>`，接入现有 RUDA 计算图和梯度累积。这是显式原生入口，不改变张量原有 `sum_dim` / `mean_dim` 的调度。调用见 [reduction_tensor 示例](examples/reduction_tensor.rs)：`cargo run --locked --release --example reduction_tensor`。

## 原生 LayerNorm 公共 IR

`RowProgram::LayerNorm` 接收连续 FP32 的 `X[rows,width]`、`weight[width]`、`bias[width]`，输出 `Y[rows,width]`、`mean[rows]` 和 `rstd[rows]`。方差按行宽计算，使用中心化平方和。

`RowProgram::LayerNormInputBackward` 接收 `X`、`dY`、`weight`、保存的 `mean`、`rstd`，返回 `dX`。`LayerNormWeightContributions` 计算逐元素权重梯度贡献，再通过设备端逐级成对求和生成 weight 梯度；bias 梯度由 `dY` 按同样方式求和。前向统计量保留在设备端传给反向。上述路径使用公共 Rust IR 和 CCE 向量指令，不调用 ACLNN LayerNorm。

RUDA 的现有 `ruda_nn::LayerNorm` 已接入 `Ascend` 和 `Autodiff<Ascend>`，包括输入、weight、可选 bias 的梯度及共享计算图梯度累积。输入为连续 FP32，归一化最后一维，宽度为正，总元素数不超过 u32；支持任意数量的前导维度和空 batch。非 32 对齐或超过 4096 的宽度在设备端分块计算全行均值，再以中心化平方和计算方差，反向复用保存的 mean／rstd 并合并全行梯度统计。不支持的 dtype／布局直接报错。

当前接入使用 Cargo.toml 中固定 Git 提交的 RUDA 依赖，尚不包含在已发布的 crates.io 0.1.0 中。从本仓库运行完整张量／自动求导示例：

```bash
cargo run --locked --release --example layer_norm_tensor
```

在配置 CANN 的编译／设备机器上构建三行、宽度 96 的示例，输出目录必须尚不存在：

```bash
python tools/ascend/build_common_ir.py --toolkit "$ASCEND_HOME_PATH" --op layer_norm --elements 288 --row-width 96 --out ./target/layernorm-forward
python tools/ascend/build_common_ir.py --toolkit "$ASCEND_HOME_PATH" --op layer_norm_input_backward --elements 288 --row-width 96 --out ./target/layernorm-backward
cargo run --locked --release --example layer_norm -- ./target/layernorm-forward ./target/layernorm-backward
```

完整设备前向 → 保存统计量 → 输入反向调用见 [layer_norm 示例](examples/layer_norm.rs)。

## 原位 AdamW 存储更新

`rust_ascend::optim::adamw_step` 通过 RUDA `ComputeClient` 执行已有 `ruda_optim::fused_adamw::storage::adamw_scaled` 的公共 IR，不另写优化器公式。参数、梯度和一阶／二阶矩为形状相同的连续 FP32 缓冲区；参数和矩原位更新、地址不变，梯度保持只读。支持非对齐尾部和空张量。

`AdamWStorageStep` 显式提供学习率、beta、epsilon、weight decay、两个 bias correction、逆梯度 scale 和 clip multiplier。FP32 unscale 后再应用 clip；epsilon 加在 bias-corrected 平方根外。函数同步完成后返回，不隐式生成全局 step，不做 AMSGrad、低精度 master cast 或自动梯度检查。

上述九个标量以原始 FP32 位模式上传为 36 字节设备参数缓冲区，内核从该缓冲区读取；同一元素数的更新不再因学习率、bias correction 或缩放值改变而生成不同源码／模块缓存键。参数、梯度与矩的布局和更新语义不变，不将梯度搬回主机。

调用见 [adamw_storage 示例](examples/adamw_storage.rs)：`cargo run --locked --release --example adamw_storage`。

`optim::adamw_tensor_step` 可直接更新现有 `Tensor<Ascend, D>` 参数与 FP32 矩；梯度可使用 `Autodiff<Ascend>` 返回的 inner tensor，不需要将梯度搬回主机。它原位修改已有存储及其外部别名，不构建优化器求导图。原生 RMSNorm → Mean loss → 自动求导 → AdamW 的训练调用见 [training 示例](examples/training.rs)：`cargo run --locked --release --example training`。

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
- 行计算：sum/mean/max、Softmax/LogSoftmax、RMSNorm、LayerNorm，以及对应归一化操作的输入梯度；RMSNorm 和 LayerNorm 另提供共享 weight 梯度。运行时 Sum/Mean、Softmax/LogSoftmax、RMSNorm、LayerNorm 及反向支持任意正行宽的原生分块路径；直接调用完整行编译模式仍遵循 32～4096 列、32 对齐限制。
- BF16 矩阵：direct-store Dense/Batched NN/NT/TN/TT、对齐的 MGrouped NT，BF16/FP32 输出。
- 设备代码目标为 Ascend950DT / dav-c310；不自动推断或替换目标型号。
- Rust 程序生成 CCE，再由 Bisheng 编译为设备机器码，不是直接 Rust → 昇腾 ISA。
- 不包含完整 PyTorch 昇腾后端、通用低精度行计算或任意 stride/广播。

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

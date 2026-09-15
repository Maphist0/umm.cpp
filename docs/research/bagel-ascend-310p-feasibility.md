# BAGEL-7B-MoT 在 Ascend 310P 上的纯 NPU 推理可行性调研

日期：2026-09-15

调研分支：`ascend-310p-bagel-npu`
基线：umm.cpp `d227cb92736e7cfaa15dc81834731dd63a73f76c`

## 1. 结论

在“所有神经网络计算图均由 Ascend NPU 执行，CPU 只负责分词、调度、文件 I/O、随机数和 PNG 编码”的定义下，BAGEL-7B-MoT 已经在当前机器的 Ascend 310P3 上跑通。

如果“纯 NPU”要求连 CFG 合成、Euler 更新、随机噪声和图片预处理都不得在 CPU 上执行，则当前 umm.cpp/stable-diffusion.cpp 架构不能直接满足，需要额外把这些宿主端张量运算重写为 ggml 计算图。这不是首阶段建议目标。

当前分支已经补齐显式后端选择、310P CANN 兼容修改、跨设备 KV staging、BAGEL layer split 边界和 CFG 空前缀处理。F16 模型包已完成 text、image、think-image、understand、think-understand、edit、think-edit 七种 CLI 模式的严格 NPU 验证。双卡 diffusion 强制切分和三路编辑 CFG 也已通过。

尚未完成的性能/质量扩展项是 512/1024 分辨率、50 steps 和官方 PyTorch/CUDA 同参数对照；它们不影响当前 256 分辨率全流程可运行的结论。

### 1.1 2026-09-15 实测结果

模型包位于 `models/BAGEL-7B-MoT-F16-UMM`：

| 组件 | 文件大小 | 转换后类型 |
| --- | ---: | --- |
| understanding | 15,237,885,440 bytes | F16/F32（二维主权重 F16，一维参数 F32） |
| generation | 13,109,120,896 bytes | F16 |
| vision | 886,289,472 bytes | F16/F32 |
| VAE | 335,304,388 bytes | 原始 safetensors，主体 F32 |

所有运行均设置 `GGML_SCHED_STRICT_ACCEL=1`。该模式会在神经网络计算图落到 CPU 时直接拒绝执行，因此成功退出可作为无 CPU 算子回退的验收证据。CPU 仍负责分词、调度、图像文件读写和 CFG/Euler 等宿主侧控制逻辑。

| 模式/场景 | 设备布局 | 结果 |
| --- | --- | --- |
| text | understanding=CANN0 | 成功，29/29 层 offload |
| understand | understanding=CANN0, vision=CANN1 | 成功，SigLIP 和 LLM 均在 NPU |
| think-understand | understanding=CANN0, vision=CANN1 | 成功 |
| image, CFG=1 | understanding=CANN0, diffusion=CANN1, VAE=CANN0 | 成功 |
| image, CFG=4 | understanding=CANN0, diffusion=CANN1, VAE=CANN0 | 成功 |
| image, CFG=4，强制双卡 | diffusion=CANN1&CANN2，各限 14 GiB | 成功，30 个 segment 按 15/15 切分 |
| image, 8 steps，强制双卡 | 同上 | 成功，采样 13.55 秒，输出语义清晰的苹果图 |
| edit, CFG=1 | LLM=CANN0, vision/VAE=CANN1, diffusion=CANN2 | 成功 |
| edit, CFG=4, image-CFG=1.5 | 同上 | 成功，conditional/without-text/without-image 三槽均执行 |
| think-image / think-edit | LLM=CANN0，其余模块按上述布局 | 成功 |

代表输出：`outputs/bagel-f16-256-8step-cfg4-split-strict.png`。该图为有效的 256×256 RGB PNG，能清晰呈现提示词中的红苹果、叶片与木桌，没有 NaN、花屏或纯色退化。测试结束后 8 张 310P3 均为 `Health: OK`，且没有残留 NPU 进程。

本轮定位并修复了三个 BAGEL 特有问题：

1. stable-diffusion.cpp 的通用名字转换会把 `text_encoders.llm.token_embd/blk.*` 错转回 Hugging Face 名字，导致 generation runner 找不到共享 understanding 权重。BAGEL 现在保留 llama.cpp canonical 名字。
2. Runner 构造阶段直接读取两个视觉边界 token 的 embedding，与 layer-split 延迟装载冲突。现在在执行图内用 `GET_ROWS` 从 NPU 权重提取，支持延迟装载和多卡分配。
3. BAGEL 的 without-text 前缀可以合法地包含 0 个 token。旧逻辑把对应的空 `SDCondition` 当作条件不存在，跳过 CFG 分支，最终报 `Diffusion model sampling failed`。现在以 external KV slot 的 active 状态判断分支是否存在。

## 2. 当前 BAGEL 接入范围

umm.cpp 当前主线已经包含：

- `scripts/convert-model.py`：识别官方 `BagelForConditionalGeneration`，输出 `understanding.gguf`、`generation.gguf`、`vision.gguf`、`vae.safetensors`；
- `src/model-workflow.cpp`：实现 `text`、`image`、`think-image`、`understand`、`think-understand`、`edit` 和 `think-edit` 的 BAGEL 提示词及状态机；
- `src/llama-cpp-adapter.cpp`：识别 `bagel.branch=understanding`，追加视觉 embedding，导出/导入 KV prefix；
- `src/sd-cpp-adapter.cpp`：通过 llama.cpp 的 MTMD/SigLIP 路径编码 BAGEL 图像；
- stable-diffusion.cpp 子仓库：实现 BAGEL generation expert、共享 understanding 权重、外部 KV conditioning、三路 CFG、图像 latent prefix 和 Flux VAE；
- llama.cpp 子仓库：实现 BAGEL understanding GGUF 转换及 BAGEL 视觉 projector。

README 对 BAGEL 的状态仍标记为实验性：转换、CPU handoff、图构建和小型视觉 forward 已检查，但完整推理和图像质量没有验证。平台说明仍是仅在 CUDA 上验证。

## 3. 官方模型与权重规模

官方配置为 28 层、hidden size 3584、28 个 attention heads、4 个 KV heads、head dim 128。视觉编码器输入 980、patch size 14、hidden size 1152；VAE latent channel 为 16，downsample 为 8，latent patch size 为 2。模型为 7B active parameters、约 14B total parameters。

通过 safetensors 头部元数据统计得到：

| 组件 | 原始精度 | 原始大小 |
| --- | --- | ---: |
| understanding/shared LLM | BF16 | 14.185 GiB |
| generation expert | BF16 | 12.155 GiB |
| generation bridge | BF16 | 0.054 GiB |
| SigLIP/connector/position | BF16 | 0.814 GiB |
| VAE encoder + decoder | F32 | 0.313 GiB |

由此估算：

- 全部主干转换为 F16：模型包约 27.5 GiB；
- understanding/vision 为 F16、generation 为 F32、VAE 保持 F32：约 39.7 GiB；
- stable-diffusion.cpp 的 BAGEL image context 会再次加载 understanding/shared 权重，因此进程总设备驻留量还要加上 llama.cpp 自己持有的 understanding 权重，但可以放在不同 NPU。

本机当前有约 187 GiB 可用磁盘。官方源权重约 29.55 GB，保留源权重并生成 F16 和 generation-F32 两套模型包预计仍可容纳，但实施前应再次检查空间。

官方资料：

- https://github.com/ByteDance-Seed/Bagel
- https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT
- https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT/raw/main/config.json
- https://arxiv.org/abs/2505.14683

## 4. Ascend 310P 兼容性分析

### 4.1 数据类型

当前 llama.cpp CANN 后端在 `ASCEND_310P` 条件下：

- `MUL_MAT` 不接受 BF16；
- `GET_ROWS` 不接受 BF16；
- Q4/Q8 矩阵乘不支持；
- F16 和 F32 是可用路径。

所以不能使用当前转换器的默认 BAGEL 输出。初次转换应使用：

```text
--outtype f16 --generation-outtype f32
```

`--generation-outtype` 需要从已验证的 U1.5 Ascend 分支迁回 BAGEL 分支。还应生成一套 F16 generation 包，用于判断 BAGEL 是否像 U1.5 一样存在 F16 数值失真。如果 F16 与参考结果一致，应优先使用 F16 以降低显存和跨设备传输；若出现 NaN、重复纹理、颜色异常或结构崩坏，再使用 F32 generation。

### 4.2 Flash Attention

310P 的当前 CANN 后端明确不支持 `GGML_OP_FLASH_ATTN_EXT`。以下三处必须在选择 CANN 时自动关闭 Flash Attention：

1. llama.cpp understanding context；
2. MTMD/SigLIP vision context；
3. stable-diffusion.cpp 的 diffusion/VAE context。

当前 BAGEL vision adapter 强制 `CLIP_FLASH_ATTN_TYPE_ENABLED`，会导致 MTMD 将不支持的 attention 放回 CPU。该处必须改为基于后端选择，CANN 使用 disabled。

### 4.3 BAGEL generation graph

BAGEL generation graph 使用的主要操作包括 F16/F32 matmul、add、mul、SILU、RMS norm、concat、contiguous、slice/view、get_rows、timestep embedding、NeoX RoPE 和 attention。按 CANN `supports_op` 声明，这些操作在满足 F16/F32、连续布局和完整 128 维 RoPE 的条件下都有 NPU 路径。

当前 `bagel.h` 没有 `mark_graph_cut`，多设备 layer split 无法识别 28 个 transformer 层。应在以下位置建立边界：

- latent/time/position 输入投影后的 prelude；
- 每个 transformer layer 输出之后；
- final norm 和 `llm2vae` 之前或之后的尾部。

每层同时使用 understanding/shared 权重处理两个边界 token，并使用 generation expert 权重处理 image tokens。切分器必须把同一层的两组权重分配到同一设备，避免每层内跨 NPU 来回传输。

U1.5 分支中“参数已经在第一次 CFG pass 分配后，后续 pass 复用原设备映射”的 layer-split 修复是通用修复，BAGEL 的两路/三路 CFG 同样需要它。

### 4.4 KV prefix 跨设备交接

文生图方向：

```text
llama.cpp / CANN0 KV
  -> host-owned staging descriptors
  -> stable-diffusion.cpp primary generation NPU
  -> per-layer persistent cache
```

不能把 CANN0 上的 `ggml_tensor*` 直接交给另一个 CANN backend。U1.5 分支已经实现 device-to-host staging，这部分可复用。

编辑方向还多一条反向链：

```text
VAE encoder + BAGEL generation graph 产生 image KV
  -> host staging
  -> llama.cpp import_prefix
  -> 后续 reasoning / prompt KV
```

这条路径在同一 CUDA 默认设备上可能工作，但在独立 Ascend 设备上必须验证 tensor buffer 的所属设备、同步时机和生命周期。

### 4.5 Vision encoder

vision.gguf 约 0.814 GiB，转换器已固定为 F16。SigLIP 图的卷积、matmul、norm、GELU、pool、reshape 等均有 CANN 实现。

风险来自 980×980、14×14 patch 对应最多 70×70，即 4900 tokens；禁用 Flash Attention 后注意力工作区可能很大。需要：

- 暴露 `vision_backend`，不要依赖“第一个 GPU”；
- CANN 自动关闭 MTMD flash attention；
- 用严格调度输出 unsupported-op 清单；
- 先用官方预处理后的最小有效输入，再测试 980×980 最大路径；
- 必要时把 vision 放在独立 NPU，或在编码结束后卸载 vision context。

### 4.6 VAE

BAGEL 使用真实 Flux 风格 VAE，而 U1.5 使用 FakeVAE，因此 U1.5 成功不能证明 BAGEL VAE 已兼容 310P。VAE 权重虽只有约 0.313 GiB，但 1024×1024 解码的 activation 较大。

CANN 声明支持其主要卷积展开、matmul、group norm、nearest upscale、attention 和逐元素操作。仍需在 strict 模式分别验证 encoder 和 decoder，重点排查：

- 非连续 tensor 触发 CPU fallback；
- F32 convolution 的 shape/stride 限制；
- upscale 的轴比例限制；
- 1024 解码工作区 OOM。

stable-diffusion.cpp 支持按模块指定后端，因此可以把 VAE 放在单独设备，并在需要时启用 VAE tiling。

## 5. 推荐设备布局

### 首次跑通：3 张 NPU

```text
CANN0: llama.cpp understanding + BAGEL vision encoder
CANN1 & CANN2: BAGEL diffusion layer split
CANN0 或 CANN2: VAE（按实测余量决定）
```

建议 stable-diffusion.cpp backend spec 形态：

```text
diffusion=CANN1&CANN2,vae=CANN0
```

如果 CANN0 的 vision/LLM 计算缓冲不足，则使用 4 张 NPU，将 VAE 移到 CANN3。

### 收敛目标：2 张 NPU

```text
CANN0: understanding + vision + VAE + 少量 diffusion 层
CANN1: 大部分 diffusion 层
```

可参考 U1.5 已验证的非均匀显存预算，让 CANN0 只承担少量 generation 层。F16 generation 更可能在两卡下稳定；F32 generation 在 1024 分辨率下可能需要三卡。

## 6. 实施步骤

### 阶段 A：建立可复现基线

1. 保持本分支基于最新 umm.cpp `origin/main`；
2. 为 llama.cpp 和 stable-diffusion.cpp 分别建立 BAGEL Ascend 子分支；
3. 复用 U1.5 分支中的通用 310P 修复，逐个挑选，不直接合并 U1 模型专用逻辑；
4. 使用现有 CANN 8.5.0 / openEuler 24.03 容器构建，宿主机只挂载代码、模型、输出和设备；
5. 将 strict accelerator 检查作为验收条件。

### 阶段 B：模型转换与静态审计

1. 下载官方 29.55 GB checkpoint；
2. 增加并验证 `--generation-outtype f16|f32`；
3. 生成 F16 understanding、F16 vision、F32 VAE，以及 F16/F32 两套 generation；
4. 校验四个组件的 manifest、tensor 数量、shape、dtype 和大小；
5. 在不执行完整模型的情况下构建 BAGEL 图，收集 CANN unsupported-op 清单。

### 阶段 C：打通文本和视觉

1. 给 llama adapter 增加显式 understanding device；
2. CANN 下禁用 llama Flash Attention；
3. 使用 F16 understanding，在 CANN0 运行 16/128 token 文本测试；
4. 给 BAGEL vision adapter 增加显式 device；
5. CANN 下禁用 MTMD Flash Attention；
6. 运行单图 SigLIP forward，确认所有图节点位于 NPU；
7. 完成 `understand` 和 `think-understand`。

### 阶段 D：打通文生图

1. 把 U1.5 的 root-level backend/max-VRAM 参数迁移到最新 session/CLI；
2. 在 BAGEL 28 层图中加入 graph-cut 标记；
3. 合并 stable-diffusion.cpp 的重复 CFG pass 参数映射修复；
4. 复用 llama KV host staging；
5. 先以 CFG=1、256×256、1 step 验证单 conditioning pass；
6. 再以 CFG=4 验证 conditional/without-text 两个 slot 和重复图执行；
7. 逐级测试 512×512、1024×1024，以及 8/50 steps；
8. 对比 F16 与 F32 generation 的 NaN、范围、统计量和参考图片质量。

### 阶段 E：VAE 与编辑

1. 单独验证 VAE decoder 的 256/512/1024 输出；
2. 单独验证 VAE encoder；
3. 打通 generation NPU 到 llama NPU 的 image-prefix 反向 KV staging；
4. 验证 `edit` 的 conditional、without-text、without-image 三个 slot；
5. 验证 `think-edit` 的 reasoning token 重放和位置连续性；
6. 必要时加入 VAE tiling 或独立 VAE 设备。

### 阶段 F：最终验收

每种模式至少覆盖：

- `text`；
- `image`，CFG=1 与 CFG>1；
- `think-image`；
- `understand` / `think-understand`；
- `edit` / `think-edit`。

验收证据包括：

- `GGML_SCHED_STRICT_ACCEL=1` 下成功退出；
- 日志中没有 CPU graph split 或 unsupported op；
- LLM、vision、diffusion、VAE 的参数和计算 buffer 均记录为 CANN；
- 生成 PNG 尺寸、通道、数值范围有效；
- NPU 测试后健康，无残留进程；
- 50-step 1024×1024 输出通过人工质量检查；
- 至少与官方 PyTorch 或 CUDA 路径做一组同 seed/参数的定性比较。

## 7. 风险排序

| 风险 | 概率 | 影响 | 处理策略 |
| --- | --- | --- | --- |
| BF16 或 Flash Attention 导致 CPU fallback | 高 | 阻断纯 NPU | F16/F32 转换，CANN 自动禁用 FA，strict 模式 |
| BAGEL 图缺少逐层切分边界 | 高 | 多卡无法装载或仅用首卡 | 为 28 层添加 graph-cut 标记 |
| 1024 无 FA attention 工作区过大 | 中高 | OOM | 三卡首跑、降低尺寸、精确预算、分段 |
| F16 generation 数值质量异常 | 中 | 图片失真 | 同时准备 F32 generation，做参考对比 |
| VAE 特定 shape/stride 不受 CANN 支持 | 中 | 最终编解码回退 CPU | strict 单测、修正 layout、必要时 tiling |
| 编辑模式反向 KV 跨设备生命周期错误 | 中高 | 编辑失败或崩溃 | 明确 host ownership、同步和导入复制 |
| SigLIP 4900-token 无 FA 内存过大 | 中 | 图像理解 OOM | 独立 NPU、逐尺寸测试、编码后释放 |

## 8. 预计代码改动位置

umm.cpp：

- `include/session.h`
- `include/llama-cpp-adapter.h`
- `include/sd-cpp-adapter.h`
- `src/session.cpp`
- `src/cli.cpp`
- `src/llama-cpp-adapter.cpp`
- `src/sd-cpp-adapter.cpp`
- `scripts/convert-model.py`
- Docker/CANN 构建文档

llama.cpp 子仓库：

- 复用经过筛选的通用 310P CANN 修复；
- 必要时补充 BAGEL Qwen3/MTMD 的严格后端测试；
- 不预计需要改 BAGEL 模型数学结构。

stable-diffusion.cpp 子仓库：

- `src/model/diffusion/bagel.h`：graph-cut、跨设备 cache；
- `src/core/layer_split_partition.cpp`：复用 CFG 重复 pass 映射修复；
- 必要时修正 VAE 的 CANN layout；
- 不改变 BAGEL guidance 数学公式。

## 9. 建议的首个里程碑

首个里程碑只要求：F16 understanding + F32 generation，三张 NPU，256×256，1 step，CFG=1，所有神经网络图严格在 CANN 上运行并输出有效 PNG。

该里程碑成功后，再依次增加 CFG、分辨率、步数、vision、VAE encoder 和编辑流程。这样每次只引入一个新的计算路径，容易定位 310P 的算子、内存或跨设备问题。

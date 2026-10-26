# DeepSeek V4 系列在 A3 / A5 上的矩阵切分：完整说明与配置指南

研究日期：2026-10-26。

分析对象：DeepSeek-V4-Flash、DeepSeek-V4-Pro、DeepSeek-V4.1-Flash。代码基线：`vllm-ascend @ 365692e28ee33c5bff29e62cb0e67cc697f336ad`；上游对照：本地 `vllm @ ffcabb8c7ab3e066e1c77b55d7673e5d08cf26cc`。本报告结合三个 agent 的专项分析，按当前代码重新组织。

**本文解释当前实现、配置条件和已知边界。本次未运行 A3 / A5 实机测试。“有实现”“有单元测试”“有模型部署配方”“完成专项硬件验收”分别表示不同强度的证据。**

## 1. 先明确：常规 TP、细粒度 TP、PCP 分别是什么

这三个概念容易混在一起。先按“使用哪个并行维度、影响哪些模块”区分：

| 机制 | 使用的分组 | 主要参数 | 对本文模块的影响 |
| --- | --- | --- | --- |
| 常规 TP | 标准 TP 组 | `--tensor-parallel-size T` | 在普通路径下，三款模型的 O、Token Embedding、LM head 自动随 T 分片 |
| 细粒度 TP | 沿 DP 轴构建的模块子组，固定标准 TP 索引 | `additional_config.finegrained_tp_config` | 对 O、Embedding、LM head、Dense MLP 分别设置切分数 K |
| PCP | 额外的上下文并行组 | `--prefill-context-parallel-size P` | 分配长提示词计算；可同时对 O、Embedding、LM head 做 PCP 权重分片 |

**`finegrained_tp_config` 一共有四项。OTP 是其中 O 投影的通信组；PCP 不在这四项里面。**

| 模块 | 细粒度配置键 | 通信组 | 默认值 |
| --- | --- | --- | ---: |
| Attention O 投影 | `oproj_tensor_parallel_size` | OTP | 0 |
| Token Embedding | `embedding_tensor_parallel_size` | Embed-TP | 0 |
| LM head | `lmhead_tensor_parallel_size` | LMTP | 0 |
| 可匹配的 Dense MLP 路径 | `mlp_tensor_parallel_size` | MLP-TP | 0 |

四项均沿 DP 轴建组，可以分别设置 K。例：标准 TP1、DP8、Embedding K4，表示每四个 DP rank 合作存储和计算一份 Embedding，共两组；模型的注意力并没有因此整体变成 TP4。[C1][C2]

PCP 的权重开关位于 `additional_config` 顶层：

| PCP 权重开关 | 控制对象 | 默认值 |
| --- | --- | --- |
| `enable_pcp_o_proj_weight_sharding` | O；DeepSeek 中是 `wo_a + wo_b` | true |
| `enable_pcp_embedding_lmhead_weight_sharding` | Token Embedding 与 LM head，二者共同控制 | true |

只有 PCP>1 且进入相应实现时，这些 PCP 开关才产生跨 PCP 分片。它们不是四个细粒度整数项的别名。

### 1.1 同时配置时，分片倍数怎样计算

设标准 TP 为 T、某模块细粒度大小为 K、PCP 为 P：

| 路径 | Embedding / LM head 每 rank 的词表权重行数 |
| --- | --- |
| 纯标准 TP | 约 `V_padded / T` |
| 启用该模块细粒度 TP | 约 `V_padded / K`，替代该模块的标准 TP 布局 |
| 启用 PCP 词表权重分片 | 约 `V_padded / (T×P)` |

例如原来 T4，改成细粒度 K2，会让每卡这块权重从 1/4 增到 1/2；K4 不再额外省权重；K8 才比原 T4 再少一半，并且 DP 必须能被 8 整除。不能把 T4、K4 理解成 16 份。[C3]

## 2. 三款模型与 A3 / A5 的支持范围

### 2.1 模型结构与普通 TP 上限

| 模型 | hidden H | 注意力头 N | head dim d | O 分组 G | O 低秩 r | 词表 V | 当前普通路径 TP 上限 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| V4 Flash | 4096 | 64 | 512 | 8 | 1024 | 129280 | TP8 |
| V4 Pro | 7168 | 128 | 512 | 16 | 1024 | 129280 | TP16 |
| V4.1 Flash | 5120 | 64 | 512 | 8 | 1024 | 129280 | TP8 |

结构数值来自官方原始配置：[V4 Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/raw/main/config.json)、[V4 Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/raw/main/config.json)、[V4.1 Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/config.json)。0731、0813、1017 和量化版必须检查实际权重配置，不应只按模型名套数值。

这些 TP 上限在当前平台校验中不区分 A3 / A5；还要满足维度、量化和模型其他组件的约束。MRv1 的旧 DSA-CP 存在独立拓扑例外，本文的普通 TP 上限不把该例外混入。[C4]

### 2.2 三个重点模块的细粒度支持矩阵

| 模型 / 硬件 | O 细粒度 TP | OTP 通信 buffer 跨层复用 | Token Embedding 细粒度 TP | LM head 细粒度 TP | 当前可见证据 |
| --- | --- | --- | --- | --- | --- |
| V4 Flash / A3 | 有实现；仅合法 PD Decode | 已实现；有效 OTP 下自动复用 | 有公共实现 | 有公共实现 | 有 A3 模型配方；未找到三个细粒度项完整专项验收 |
| V4 Flash / A5 | 有实现，包括 FP8 OTP 分支 | 已实现；有效 OTP 下自动复用 | 有公共实现 | 有公共实现 | 有 A5 模型 YAML、A5 FP8 O 单测；不等于三项组合 E2E |
| V4 Pro / A3 | 与 V4 Flash 共用模型路径 | 已实现；有效 OTP 下自动复用 | 有公共实现 | 有公共实现 | 有 A3 多节点部署资料；细粒度组合需专项测 |
| V4 Pro / A5 | 同一 O 路径；K 可按 G=16 评估 | 已实现；有效 OTP 下自动复用 | 有公共实现 | 有公共实现 | 有 950DT 原生混合精度及 PD 配方；细粒度组合需专项测 |
| V4.1 Flash / A3 | 复用公共 O projector；仅合法 PD Decode | 已实现；有效 OTP 下自动复用 | 有公共实现；不控制 Engram 表 | 有公共实现 | 有 W8A8 A3 配方；其中 TP2/TP4 布局不能直接叠 OTP |
| V4.1 Flash / A5 | 复用公共 O projector；目标权重需专项验证 | 已实现；有效 OTP 下自动复用 | 已接入细粒度切分；未找到该模型在 A5 上的专项实测证据 | 已接入细粒度切分；存在采样兼容限制，未找到该模型在 A5 上的完整组合实测证据 | A5 强制 MRv2；有 TP1/DP4/EP4 回归例子，非三项细粒度生产验收 |

这里“公共实现”表示模型构造了对应层，并由 Ascend 注册替换成支持相应布局的实现；不是仅凭通用文档推断。[C3][C5]

“未找到专项实测证据”表示本报告收集到的证据覆盖有限，不表示代码不支持，也不表示需要额外开发才能开启。“采样兼容限制”指 LM head 切分需要同时满足 logits 汇集、采样与投机解码路径的约束，详见第 4.4 节。

“OTP 通信 buffer 跨层复用”是独立确认的源码能力：V4 Flash / Pro 与 V4.1 Flash 均把模型级 `OProjBufferPool` 交给各层，没有针对 A3 / A5 禁用这套复用机制。有效 OTP 下，相同 shape、dtype、device 的顺序层复用 All-to-All 的发送 / 接收 buffer 和 Reduce-Scatter 输出 buffer，无需额外共享开关。该列表示已有实现；跨层复用单测不等于各模型在 A3 / A5 上的完整实机验收。[C5][C8]

第四项 MLP 是框架级能力，但这三款模型的主干层使用 MoE。其 Routed Experts、Shared Experts 和可匹配的普通 MLP 必须分别分析，详见第 5 节，不能把第四项直接画成“三款模型全部 FFN 都已验证可切”。

### 2.3 两类硬件的额外边界

- A3 / A5 没有针对上述四个细粒度字段的统一硬件禁用条件；真正差异来自 runner、算子、量化、内存容量和通信拓扑。
- 仓库把 910C 系列归为 A3、`ascend950*` 归为 A5。部署仍需区分 950PR / 950DT、逻辑设备数和互联，不能照搬 A3 的设备数量。
- V4.1 A5 **必须** `VLLM_USE_V2_MODEL_RUNNER=1`。A3 主教程使用 MRv1，不能把其 runner 设置直接复制到 A5。
- V4.1 启用 Engram 时平台要求 TP=1/2/4/8、PP=DCP=1，并拒绝 Ascend Engram 的 `embedding_across_dp` 配置。它与 Token Embedding 的细粒度开关独立。[C4][C6]

## 3. 常规 TP：三个重点模块会自动切分

### 原理与开启方式

在普通 TP 路径下，只要 T>1 且模型配置合法，三款模型的 O、Token Embedding、主 LM head 都自动随 T 切分，不需要开启 `finegrained_tp_config`。

```bash
--tensor-parallel-size 4
```

| 模块 | 普通 TP 如何切 |
| --- | --- |
| O 投影 | 按注意力头 / O 组分配，`wo_a` 和 `wo_b` 配套切分，每 rank 处理 G/T 个 O 组 |
| Token Embedding | 按词表行分片，每 rank 约 `[V_padded/T,H]`，查表贡献归约后得到完整向量 |
| LM head | 按输出词表分片，每 rank 计算局部词表 logits，再按采样路径汇聚 |

### 开启条件与叠加边界

- T=1 无实际标准 TP 切分；合法 T 需满足第 2 节上限及模型维度整除。
- Embedding / LM head 的细粒度字段保持 0，才能沿用这两个模块的默认标准 TP 计划。
- 如果 PCP>1 且默认词表分片开启，Embedding / LM head 改为 TP×PCP 布局。
- O 细粒度 K>1 与标准 T>1 不兼容，配置会拒绝。PCP 的 O 权重分片则有自己的条件，见第 6 节。
- 标准 TP 的 O 分片可用于一体部署、Prefill 和 Decode，不受“细粒度 OTP 仅限 PD Decode”的限制。
- 标准 TP 只说明并行层如何布局，不表示所有权重都切分；模型仍可能包含 ReplicatedLinear、复制的 draft head 等。[C3][C5]

## 4. 细粒度 O、Embedding、LM head：逐项说明

以下命令均为**合并到已成功启动的目标模型基线中的配置增量**，在目标 Linux / CANN / torch_npu 环境使用。模型路径、量化、tokenizer、网卡和多节点启动方式沿用基线。一条服务命令只保留一份合并后的 `--additional-config`。

### 4.1 三项及 MLP 的共同条件

- 模型被正确识别为 MoE；非 MoE 配置会被拒绝。
- K 为非负整数；非零 K 必须整除全局 DP，即 `DP % K == 0`。多节点不能用 local DP 代替全局 DP。
- 实际分片使用 K>1。Embedding / LM head / MLP 的代码以 K>0 进入细粒度路径；K=1 不是“恢复默认”。O 则以 K>1 启用。
- 四项默认 0，仅关闭对应细粒度计划，不关闭常规 TP。
- 官方细粒度指南将 PP>1 列为不支持组合。本文按 **PP=1 的文档支持边界**推荐；本次未找到覆盖四项的统一 PP 拒绝校验，不能保证误配一定在启动时报错。
- 子组各 rank 必须使用一致的相关容量、投机长度等配置，并在执行 collective 时共同参与。空闲 rank 也可能需要 dummy 计算。
- EP 是常见 MoE 部署拓扑，示例会启用；不能把“存在某个单独字段”当作整个 MoE / DP / EP 拓扑已合法。[C1][C2][C7]

### 4.2 O 投影细粒度 TP：`oproj_tensor_parallel_size`

**切什么、如何计算**

DeepSeek 的 O 不是一个简单完整矩阵，而是两级、分组低秩投影：

```text
注意力输出 [B,N,d]
  → 按 G 组重排 [B,G,N*d/G]
  → 每组 wo_a 降到 r 维
  → 拼接 [B,G*r]
  → wo_b 映射回 [B,H]
```

用行向量写法：`Z_g=A_g W_a,g`，`Y=Σ_g Z_g W_b,g`。把完整 O 组分给 K 个设备，每个设备保存相应的 `wo_a` 和 `wo_b` 权重。DP rank 的请求不同，所以还需要交换请求数据：

```text
本 rank 注意力输出按 O 组分块并填充
  → all_to_all，送到相应 O 组权重 owner
  → 本地 wo_a + wo_b，得到部分和
  → reduce_scatter，求和并返回原请求所属 DP rank
```

两级权重都切；不是仅切最后一个 `wo_b`。量化在 all-to-all 之后执行，通信 buffer 保持输入 dtype。A5 的低精度 GEMM不等于整个通信也采用 FP8。[C8]

**怎么开启**

在满足下方全部条件的 D 节点上，设置：

```bash
--tensor-parallel-size 1 \
--data-parallel-size 8 \
--prefill-context-parallel-size 1 \
--enable-expert-parallel \
--max-num-seqs 64 \
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":512}' \
--additional-config '{
  "scheduler_config":{"recompute_scheduler_enable":true},
  "finegrained_tp_config":{"oproj_tensor_parallel_size":8}
}'
```

还必须配置 PD connector 与 `PreemptOffloadConnector`，完整结构见第 7.2 节。上述 512 是演示值，不是固定推荐值。

**开启条件、互斥与失败行为**

| 条件 / 组合 | 当前行为 |
| --- | --- |
| K>1，且 K 整除 DP | 实际建立 OTP 组；不合法的 DP 整除配置拒绝 |
| K 整除 `n_local_groups` | O 构造 / 运行校验；在标准 TP1 下等价于 K 整除 G |
| 标准 TP>1 | 硬拒绝；O 组 reshape 与跨 DP 权重轴尚不能直接叠加 |
| 一体部署、P 节点、无 `kv_consumer` | 不在支持范围，配置拒绝 |
| 未开启 recompute scheduler | 硬拒绝 |
| connector 链没有 PreemptOffloadConnector | 硬拒绝；需通过 MultiConnector 与 PD connector 组合 |
| eager / `--enforce-eager` | 不支持；运行时也检查是否落入非图步骤 |
| PCP>1 | 硬拒绝 |
| PP>1 | 文档支持边界之外；推荐 PP1 |
| 投机解码 | 不一概禁止，但须放大图容量，并满足 draft / runner / 模型自身限制 |
| 与 Embedding / LM head 细粒度同时开 | 可以，逐项满足各自条件；K 可不同 |

按本文官方原始配置，Flash / V4.1 的实际 K 候选是 2/4/8，Pro 是 2/4/8/16，还须整除实际 DP。不能只因为 DP16 就给 Flash 设置 OTP16。

最大 Decode 步按下式评估：

```text
max_step = min(max_num_batched_tokens,
               max_num_seqs * (1 + num_speculative_tokens))
```

capture 上限不足时，启动代码会告警并把 O / MLP 细粒度值重置为 0，服务仍可能启动成功。运行时超过静态容量或落入 eager 步会显式失败。投机 token=5、max_num_seqs=64 时可能需要覆盖 384 tokens，不能仍按 64 配置。

**通信 buffer 如何跨层复用**

V4 Flash / Pro 和 V4.1 Flash 在模型构造时创建一个 `OProjBufferPool`，交给各层 O 投影实现，并在 KV profiling 前预留地址稳定的通信空间。池按 `(shape, dtype, device)` 缓存：兼容层直接取得同一组 `_oproj_send_buf`、`_oproj_recv_buf` 和 `_oproj_rs_out_buf`；布局不同则分别分配。由于各层顺序执行，同一 rank 内可以跨层复用这些临时空间，并保持图捕获 / 回放地址稳定。[C5][C8]

无需单独设置共享开关。`oproj_tensor_parallel_size > 1` 且通过本节全部条件、未被启动校验重置为 0 时，自动预留并复用；未启用 OTP 时不分配这套专用 buffer。这里共享的是通信临时空间，各层 O 权重仍分别保存，各 rank 也各有自己的 buffer。假设 L 层所需布局完全相同，这三类 buffer 从逐层分配的 L 组降为一组，但不代表整个 O 计算的所有中间张量都只保留一份。单测明确检查同一模型各层引用相同 buffer、不同模型分别分配，以及不兼容布局分别分配。[C8]

这部分共享 buffer 的显存仍需纳入净收益。开启此路径后，某些 offload 恢复失败会中止请求，避免返回 P 端重算造成 KV 精度变化。[C8][C9]

### 4.3 Token Embedding 细粒度 TP：`embedding_tensor_parallel_size`

**切什么、如何计算**

权重为 `E[V,H]`，按词表行分片。每个 rank 保存约 `[V_padded/K,H]`，H 维保持完整。

DP rank 的 token 不同，因此先把 token ID 填充到统一容量后 all-gather；每卡查自己的词表片，非本地 ID 的贡献为零；随后 reduce-scatter，把完整向量返回原 token owner。

```text
本 rank token IDs
  → 固定容量 padding + all_gather
  → 本地词表分片查表、非本地 ID 置零
  → reduce_scatter
  → 本 rank 的完整 H 维 Embedding
```

固定容量覆盖 `max(potential_max_tokens,max_num_batched_tokens)`，可处理 Prefill；不是只为 Decode 小批次设计。[C3]

**怎么开启**

```bash
--tensor-parallel-size 1 \
--data-parallel-size 8 \
--enable-expert-parallel \
--additional-config '{
  "finegrained_tp_config":{"embedding_tensor_parallel_size":8}
}'
```

**开启条件、互斥与边界**

| 条件 / 组合 | 当前行为 |
| --- | --- |
| MoE；K 整除 DP；padding 后词表能按分片数划分 | 必须满足；词表使用 divide 等检查 |
| K=0 | 不采用该细粒度计划，仍可使用标准 TP |
| K=1 | 采用 size1 的细粒度计划；可能把原标准 TP 分片变成完整权重 |
| 标准 TP>1 | 可同时配置；该模块改按 K 分，不是 T×K |
| eager / graph、Prefill / Decode、一体 / PD | 公共实现支持；不要求 recompute 或 PreemptOffload |
| PCP>1 且 PCP 词表权重分片为 true | 与本项硬互斥 |
| 关闭 PCP 词表分片后再叠细粒度 | 不作为已验证通用组合推荐；跨 DP 组布局与 PCP 拓扑仍需验证。本文细粒度示例统一 PCP1 |
| PP>1 | 文档不支持；推荐 PP1 |
| MTP / DSpark | 不天然互斥；需核实 draft 是共享目标 Embedding，还是独立构造了其他分片计划 |
| V4.1 Engram 表 | 不受本项控制；CPU offload、主机共享内存和 Engram head 分片另行配置 |

主要收益是整张词表的常驻内存减少。Embedding 是查表，整表缩小到 1/K 不表示每个 token 的实际读取量也减少 K 倍。Prefill 容量设得很大时，固定容量通信也可能带来额外成本。[C3][C6]

### 4.4 LM head 细粒度 TP：`lmhead_tensor_parallel_size`

**切什么、如何计算**

逻辑权重为 `W[V,H]`，输入 `X[B,H]`，logits 为 `XWᵀ`。按输出词表切分，每 rank 保存约 `[V_padded/K,H]`。

```text
本 rank 待采样隐藏状态 [B,H]
  → padding 到统一容量 + all_gather
  → 本地 GEMM，计算所有请求的局部词表 logits
  → all_to_all，按请求 owner 重排
  → 本 rank 请求的完整词表 logits [B,V]
  → 正常采样
```

空闲 DP rank 必须参与同形状的 dummy logits collective；否则活跃 rank 可能挂起。最终采样前仍恢复本 rank 的完整 V 维 logits，因此峰值 logits 内存与总计算量不会自动下降 K 倍。[C10]

**怎么开启**

```bash
--tensor-parallel-size 1 \
--data-parallel-size 8 \
--enable-expert-parallel \
--additional-config '{
  "finegrained_tp_config":{"lmhead_tensor_parallel_size":8}
}'
```

**开启条件、互斥与 runner 差异**

| 条件 / 特性 | 当前边界 |
| --- | --- |
| MoE；K 整除 DP；词表分片维度合法 | 必须满足 |
| K=0 / K=1 | 0 保留默认计划；1 进入 size1 细粒度计划，可能增加原 TP 部署的权重占用 |
| 标准 TP>1 | 可同时配置，head 按 K 分而非 T×K |
| eager / graph、Prefill / Decode、一体 / PD | 公共实现支持，不要求 O 路径的 recompute/offload 条件 |
| `enable_batch_sharded_sampling` | 与细粒度 LM head 硬互斥 |
| PCP>1 且 PCP 词表分片 true | 与本项硬互斥；本文采用 PCP1 |
| PP>1 | 文档不支持；推荐 PP1 |
| prompt logprobs、fixed-token prompt scoring | MRv2 明确拒绝；MRv1 未找到同等保护，额外未对齐 logits 调用有风险，不作为支持组合推荐 |
| 正常生成阶段的 logprobs | 不应与 prompt logprobs 混为一谈；前述限制不是对所有生成 logprobs 的统一禁令 |

投机解码需要继续区分 MRv1 / MRv2，不能写成“开了 MTP 就不能切 head”，也不能把某个 runner 的检查推广给全部路径：

| 投机组合 | 当前判断 |
| --- | --- |
| MTP / EAGLE、DFlash、DSpark 的相关公共路径 | 有行数 padding / trim 的适配；仍需按模型、runner、实际 draft head 验证 |
| MRv2 的概率式 draft sampling | `draft_sample_method=probabilistic` 在相应适配路径明确拒绝 |
| MRv2 `use_local_argmax_reduction` | 明确拒绝；局部词表 argmax 不能替代完整跨 DP 词表结果 |
| MRv2 `enable_adaptive_verification` | 明确拒绝；行数对齐与估计器 buffer 尚不兼容 |
| DFlash2 | MRv2 明确拒绝；MRv1 不能仅因没有相同报错就认定支持，其候选 TopK 路径未满足这里的跨 DP 词表通信契约 |
| MRv1 的概率 / argmax / adaptive 等组合 | 有不同实现与控制路径，不能套用 MRv2 全部硬禁令，也不据此给出未经验证的支持承诺 |

`disable_tp=True` 的复制式 draft / Markov head 应保持复制，不能因名字中包含 head 就进入目标 head 的 LMTP 通信。旧文档出现过 `enable_reduce_sample`，但当前生产代码未找到对应控制，不列为当前有效的硬互斥参数。[C10][C11]

## 5. 第四项：MLP 细粒度 TP 应怎样理解

### 5.1 原理与开启方式

对于可匹配的 Dense FFN，例如 `gate_up_proj → activation → down_proj`，按 FFN 中间维度切分：

```text
本 rank hidden states
  → MLP-TP 组 all_gather
  → 本地 gate/up 中间维分片
  → 本地 activation 与 down 分片，得到部分和
  → reduce_scatter 返回原请求 owner
```

与 Embedding / LM head 不同，MLP collective 形状随实际 token 数变化，各 rank 每一步必须满足对齐要求。因此它与 O 一起被约束在图调度的 PD Decode 场景。[C12]

在有适配 Dense MLP 路径的 MoE 模型 D 节点上：

```bash
--tensor-parallel-size 1 \
--data-parallel-size 8 \
--prefill-context-parallel-size 1 \
--enable-expert-parallel \
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":512}' \
--additional-config '{
  "scheduler_config":{"recompute_scheduler_enable":true},
  "finegrained_tp_config":{"mlp_tensor_parallel_size":8}
}'
```

同样必须配置第 7.2 节的 PD + PreemptOffload connector 链。此例展示框架入口，不能作为三款 V4 模型的通用已验收配方。

### 5.2 开启条件与互斥

- MoE、K 整除 DP，还须满足具体 MLP 的中间维度等权重分片条件。
- 真正 K>1 分片时要求标准 TP1、PCP1、合法 PD Decode、graph、recompute scheduler、PreemptOffload；PP1 按文档支持边界处理。
- capture 上限不足会与 O 一起被自动清零；运行时 eager 步会被保护性拒绝。
- 可与其他三个细粒度模块配置同时存在，但每项各自满足条件。
- K=1 也会进入 MLP 细粒度选择，且当前 recompute 检查对 MLP 使用 K>0；不要用 1 作为“关闭”。关闭用 0。
- 不能把本项当成 Routed Experts 的 EP 配置；量化方法、shared-expert 布局与层识别仍需单独核实。[C1][C9][C12]

### 5.3 三款 V4 模型为什么不能直接照搬 R1 的 MLP 结论

R1 的典型用途是切前几层独立 Dense MLP。当前三款目标模型的 decoder 主干都构造 MoE，不能直接套用“R1 前三层省多少显存”的数字。

还必须核对代码实际命中的层：当前 hook 根据 `gate_up_proj/down_proj` 前缀以及 `is_moe_layer(prefix)` 判断，而后者读取 `first_k_dense_replace`、`moe_layer_freq` 等配置，不是直接检查模块类型。

官方原始配置没有 `first_k_dense_replace`。若运行时配置对象也没有补该字段，当前判断默认 infinity，可能让独立 Shared Experts 的 gate/up/down 进入 MLP hook；Shared Experts 被融合时又可能不存在这些独立层。`enable_shared_expert_dp` 还会按 EP 与标准 TP>1 归一，而真正 MLP K>1 要求标准 TP1，所以不能只凭命令写了该 boolean 就判定实际布局。

因此，本报告对三款模型给出的判断是：**MLP 是框架支持的第四项；目标模型实际切到哪些 MLP / Shared Expert 层需看运行时配置和对象，不把所有 MoE 专家或所有 Shared Experts 统一标成已支持、已验收。** 这是静态发现的条件路径，尚无本次硬件数值与性能验证。[C12]

对本文三款目标模型，建议该项默认保持 0；只有确认运行时 HF 配置、实际 custom op、Shared Experts 融合方式，并完成通信与精度对照后再开启。参数校验通过不能代替这一验证。

## 6. PCP 权重切分：两个开关分别说明

### 6.1 PCP 本身的共同前置条件与叠加限制

PCP 用额外 rank 分担长提示词的 token 计算；并行规模会额外乘 P，它不是沿 DP 复用 rank 的细粒度 TP。

例如 PP1、标准 TP1、DP2、PCP8，逻辑并行规模为 16 个设备 rank；不能把它和“TP1、DP8、某模块细粒度 K8”的 8 个 rank 当成同一个拓扑。

```bash
export VLLM_USE_V2_MODEL_RUNNER=1

# 将下列参数合并到已验证的目标模型部署。
--prefill-context-parallel-size 8
```

| 条件 / 组合 | PCP 的当前边界 |
| --- | --- |
| MRv1 | 不支持 PCP，平台拒绝；使用 MRv2 |
| 旧 `enable_dsa_cp=true` | 与 PCP 硬互斥；旧 DSA-CP 本身也不支持 MRv2 |
| encoder-decoder、LoRA | 当前 PCP manager 明确拒绝 |
| 整体 `cudagraph_mode=FULL` | PCP 不支持；可选 eager、PIECEWISE、FULL_AND_PIECEWISE、FULL_DECODE_ONLY 等受支持模式 |
| 投机解码 | manager 仅接受相应 MTP / Eagle3 / DSpark 方法；具体 backend 继续限制，V4 DSA 主要为 MTP/DSpark，V4.1 DSA 为 DSpark |
| adaptive verification | PCP 投机路径不支持；动态 draft 长度不在当前验证范围 |
| PD 分离 | 文档要求 PCP 仅放在 P 节点，D 节点 PCP1；不要因为 backend 内部有 decode 分片实现就推定 PD D 端可开 PCP |
| PP + PCP | 需模型支持；MooncakeConnectorV1 / MooncakeHybridConnector 有限制，PP+PCP 使用相应 MooncakeConnectorV2 配方 |
| KV Cache Pool + PCP | 文档范围为 AscendStoreConnector、`use_layerwise=false` |
| DCP + PCP | MLA/SFA/GQA 有各自支持表；不能套用到本文 DeepSeek DSA/V4.1。本文不推荐给这三款模型叠 DCP；V4.1 另有明确 DCP1 要求 |

上述限制有些来自平台 / manager 的硬检查，有些是 backend / connector 的文档支持范围，并不保证所有误配都同样在同一阶段报错。[C6][C13]

### 6.2 PCP O 权重分片

**开关与原理**

```bash
--prefill-context-parallel-size 8 \
--additional-config '{"enable_pcp_o_proj_weight_sharding":true}'
```

boolean 默认 true。DeepSeek 使用共享的 DSA-PCP 两级 O 实现；常驻权重按 PCP O 组分片。

| 执行类型 | O 的工作方式 |
| --- | --- |
| 含 Prefill 的步骤 | 可临时 all-gather TP 本地完整 O 权重，为本地 token 计算，随后恢复常驻 shard |
| 分片式 Decode | 聚合 token，使用本地 O 组计算，再 reduce-scatter |
| 复制式 Decode 的通用实现 | 本地 O 组计算后 PCP all-reduce；不据此宣称目标模型的任意 DCP 拓扑已支持 |

PCP>1、DCP1 会选择相应分片式 Decode 策略。它可以用于一体 PCP 服务中的 Decode，不代表 PD 分离的 D 端允许 PCP>1。

**条件、互斥与回退**

- 标准 TP 与 PCP 的 O 组布局必须合法，尤其 `G/T` 能被 P 整除；例如 Flash G8、T4 时 P8 不合法，T1/P8 才满足该项结构条件。
- `wo_a`、`wo_b` 的具体权重方法都必须支持 weight switch。不能由模型标为 W8A8、MXFP8 或 BF16 就一概断定兼容，应检查实际方法能力。
- 初始化阶段发现 O 组无法整除或方法不支持时，会告警并关闭 PCP O 分片，回退普通 O 权重。后续 loader / shape 错误不保证自动回退，可能直接失败。
- 不能与 O 细粒度 K>1 同开；后者已要求 PCP1。
- 不要求细粒度 OTP 的“仅 PD D、recompute、PreemptOffload”条件；必须满足 PCP 自己的条件。
- 完整权重 scratch pool 可能在 warmup 分配并继续常驻。不能把“常驻每层权重已分片”解释成“运行期间不存在任何完整 O buffer”。

关闭时：`"enable_pcp_o_proj_weight_sharding":false`。这只关闭 O 的 PCP 权重分片，不关闭 PCP 本身，也不关闭词表分片。[C14]

量化方面，当前 BF16、W8A8 dynamic/static、MXFP8、FP8 block、部分 MXFP4 方法具备 weight-switch 基础能力；[W8A8FP8Dynamic](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/vllm_ascend/quantization/methods/w8a8/w8a8fp8_dynamic.py#L42) 明确不支持。此处只是在说明权重切换接口，不能据此给某个 checkpoint 的 O BMM 兼容性打勾：仍需 `wo_a` / `wo_b` 两侧方法、实际分片 shape、量化 pack/scale 对齐和对应 kernel 同时成立。

### 6.3 PCP Token Embedding 与 LM head 权重分片

**开关与原理**

```bash
--tensor-parallel-size 1 \
--prefill-context-parallel-size 8 \
--additional-config '{"enable_pcp_embedding_lmhead_weight_sharding":true}'
```

boolean 默认 true，并同时控制这两个模块；没有在该开关下分别设置两个 PCP K 的接口。权重布局按 TP×PCP 切词表。

| 模块 | 计算与通信 |
| --- | --- |
| Token Embedding | PCP all-gather token IDs → 本地词表查表 → PCP reduce-scatter → 必要的 TP all-reduce |
| LM head | token 行在采样前恢复全局顺序 → 局部词表 GEMM → 先 PCP、再 TP 聚合词表 logits |

PCP head 路径不使用细粒度 LMTP 的 DP all-to-all。[C3][C10]

**条件、互斥与关闭方式**

- 满足第 6.1 节 PCP 条件；padding 后词表须能按 TP×PCP 切分，实际 dtype / quant method 也须能正确加载对应片段。
- 与 `embedding_tensor_parallel_size>0`、`lmhead_tensor_parallel_size>0` 分别互斥，构造时拒绝。即使细粒度值为 1，也可能触发互斥。
- 与 `enable_batch_sharded_sampling` 硬互斥，这是联合开关的当前配置限制。
- `disable_tp=True` 的复制式 head 优先保持复制，不会因 PCP boolean=true 就强行分片。
- 可与 PCP O 权重分片一起使用，二者为独立 boolean。
- 关闭用 `"enable_pcp_embedding_lmhead_weight_sharding":false`；Token Embedding 与 LM head 退回各自普通 TP 布局，PCP 本身继续存在。
- 不把细粒度 LMTP 的 prompt / draft 限制自动照搬到 PCP head；PCP 的限制应按自己的采样、runner 和 backend 路径判断。[C3][C7][C10]

## 7. 实际部署怎样组合

### 7.1 一体部署：先评估两个边界模块

对于已验证的 TP1/DP8 MoE 基线，可先尝试：

```bash
--tensor-parallel-size 1 \
--data-parallel-size 8 \
--prefill-context-parallel-size 1 \
--enable-expert-parallel \
--additional-config '{
  "finegrained_tp_config":{
    "embedding_tensor_parallel_size":8,
    "lmhead_tensor_parallel_size":8
  }
}'
```

不需要额外开启 O 路径的 recompute / PreemptOffload。保留模型所需 runner、量化、KV、投机等基线设置，并检查第 4.4 节采样互斥。这里 DP8 不是 V4 Pro 能在任意八卡机器装下的保证。

### 7.2 PD Decode：O 加两个边界模块

只在 TP1、PCP1、PP1 的合法 Decode 拓扑使用；以下为示例配置：

```bash
--tensor-parallel-size 1 \
--data-parallel-size 8 \
--prefill-context-parallel-size 1 \
--pipeline-parallel-size 1 \
--enable-expert-parallel \
--max-num-seqs 64 \
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":512}' \
--additional-config '{
  "scheduler_config":{"recompute_scheduler_enable":true},
  "finegrained_tp_config":{
    "oproj_tensor_parallel_size":8,
    "embedding_tensor_parallel_size":8,
    "lmhead_tensor_parallel_size":8
  }
}'
```

将现有 PD connector 与 offload 合并成 `--kv-transfer-config`，结构示例如下：

```json
{
  "kv_connector":"MultiConnector",
  "kv_role":"kv_consumer",
  "kv_connector_extra_config":{
    "connectors":[
      {
        "kv_connector":"MooncakeConnectorV2",
        "kv_role":"kv_consumer",
        "kv_port":"30100",
        "kv_connector_extra_config":{
          "prefill":{"dp_size":1,"tp_size":1},
          "decode":{"dp_size":8,"tp_size":1}
        }
      },
      {
        "kv_connector":"PreemptOffloadConnector",
        "kv_role":"kv_consumer",
        "kv_connector_extra_config":{"cpu_bytes_to_use_per_rank":17179869184}
      }
    ]
  }
}
```

P/D topology、端口、connector 版本和 CPU offload 预算必须按已有部署修改；16 GiB/rank 是演示值。P 端保持匹配的 producer 配置。若实际 P 端用了 PCP/PP，应按该 connector 的 schema 与模型配方补齐 topology，不能原样照抄这里的 TP1 示例。

MLP 项没有放进三款 V4 模型的组合示例；完成第 5 节实际命中层与精度验证后，才有依据加入对应 K。

### 7.3 长上下文 PCP：使用另一套权重布局

在合法的一体 PCP 服务或 PD Prefill 节点：

```bash
export VLLM_USE_V2_MODEL_RUNNER=1

--tensor-parallel-size 1 \
--prefill-context-parallel-size 8 \
--additional-config '{
  "enable_pcp_o_proj_weight_sharding":true,
  "enable_pcp_embedding_lmhead_weight_sharding":true
}'
```

这里不设置四项细粒度 K，不能保留旧 `enable_dsa_cp=true`。根据 backend 选择 eager / 受支持的 graph 模式，不使用整体 FULL。本例只是切分参数，完整部署继续沿用该模型的 PCP 配方。

## 8. 互斥关系速查

下表只汇总本报告范围内的主要边界；每项的 runner / backend / connector 条件以对应章节为准。没有硬检查不等于已支持。

| 组合 | O 细粒度 | Embedding 细粒度 | LM head 细粒度 | MLP 细粒度 | PCP 权重分片 |
| --- | --- | --- | --- | --- | --- |
| 标准 TP>1 | 禁止 | 可配，改用 K | 可配，改用 K | 禁止 | 可配，检查组 / 维度 / 量化 |
| PCP>1 | 禁止 | 与默认 PCP 词表分片互斥；不推荐绕过后混用 | 同左 | 禁止 | 必要条件 |
| PP>1 | 文档不支持 | 文档不支持 | 文档不支持 | 文档不支持 | 模型和 connector 条件支持；V4.1 Engram 仍要求 PP1 |
| eager | 禁止 | 可 | 可 | 禁止 | 可；PCP 不支持整体 FULL 图 |
| Prefill / 一体部署 | 禁止 | 可 | 可 | 禁止 | 可，满足 PCP 条件 |
| PD Decode | 必须为此场景 | 可 | 可 | 必须为此场景 | 文档要求 D 端 PCP1 |
| batch-sharded sampling | 非本模块专属禁令 | 非本模块专属禁令 | 硬互斥 | 非本模块专属禁令 | PCP 词表联合开关硬互斥 |
| 投机解码 | 按最大步扩 capture | 核实 draft Embedding | 详见 runner 相关限制 | 核实步对齐 / 路径 | 按 PCP backend / 方法支持表 |
| 旧 DSA-CP | 不作为叠加推荐 | 另需专门验证 | 另需专门验证 | 不作为叠加推荐 | 硬互斥 |

## 9. 上游 vLLM 与 GPU：支持边界怎样表述

**当前核查的上游 vLLM 没有 Ascend 这套 `finegrained_tp_config` 及三个主模块独立跨 DP 切分入口。GPU 硬件可以实现这样的方案，当前差别在推理框架的软件实现。**

| 能力 | 当前本地上游 vLLM | vLLM Ascend |
| --- | --- | --- |
| 常规 TP | 有 | 有 |
| 本文四项细粒度配置、分组和执行适配 | 未发现该机制 | 有 |
| 通用 additional_config 容器 | 有；不代表会执行其中任意扩展键 | 用来消费 Ascend 扩展配置 |
| Embedding 自定义 parallel_group、部分 linear 的 tp_size 覆盖 | 有底层接口 | 配套实现模块计划、跨 DP 请求交换与调度 |
| V4.1 Engram 跨 DP 相关分片 | 有独立实现 | 有自己的支持策略；不等同于三个主矩阵开关 |

vLLM 主要服务 GPU 推理，包括 CUDA / ROCm，也能通过插件支持 Ascend 等设备。准确说法是“当前上游 GPU 路径未提供本文这套现成功能”，不是“GPU 不能切这些矩阵”——普通 TP 已经可以切。

全仓搜索配置键后，还检查了上游配置、通信组、模型构造和 logits 路径，未发现等价接入。上游 `additional_config` 能携带 JSON，并不表示传入 Ascend 的键就生效；加载 Ascend 插件后获得的功能来自插件。[C15]

## 10. 显存收益、性能与生效验证

### 10.1 估算时明确比较基线

相对标准 TP1 全量副本，某权重理想 K 分片节省约 `完整字节数×(1−1/K)`；若原来标准 TP=T，则比较的是 `完整字节数×(1/T−1/K)`。

若输入 Embedding 与 LM head 分别为 BF16、未绑定权重、忽略 padding，二者 K8 相对 TP1 的理论节省为：

| 模型 | 单块完整权重 GiB | 两块合计 GiB | 两块 K8 理论节省 GiB/rank |
| --- | ---: | ---: | ---: |
| V4 Flash | 0.9863 | 1.9727 | 1.7261 |
| V4 Pro | 1.7261 | 3.4521 | 3.0206 |
| V4.1 Flash | 1.2329 | 2.4658 | 2.1576 |

O 两级投影每层理论参数量是 `N*d*r + H*G*r`，理想 K 分片后权重约为 1/K。实际应计入 scales、通信 buffer、权重聚合 scratch pool、图内存和 KV 容量变化。

模型名字中的 W8A8 / W4A8 / MXFP4 不保证所有层采用同样位宽。Embedding、LM head、O 必须逐层检查实际 dtype 与量化 metadata。A5 已有低精度权重时，进一步切分的净收益需要重新测量。[C16]

省显存也不保证加速：O 是逐层通信，LM head 需要为其他 DP 请求计算局部词表，Embedding 使用固定容量交换。仓库现有细粒度性能表测的是 **A2、DeepSeek-R1-W8A8**；其中 O TP8 在节省权重的同时使 TPOT 增加约 1.5ms。不能拿它当本文三模型在 A3/A5 的实测结果。[C7]

### 10.2 怎样确认真正生效

1. 记录模型权重版本、插件 / vLLM commit、CANN、torch_npu、固件、设备型号与逻辑卡数量。
2. 检查 `finegrained_tp_config enabled` 中实际列出的字段和大小，同时查 O/MLP 的自动关闭告警。
3. PCP O 查不支持分片的 fallback 告警；boolean=true 不代表没有回退。
4. 在调试环境确认实际层的分片计划、weight shape、dtype；MLP 还要确认具体命中了哪些层，不能只看配置对象。
5. 分别测试 baseline、单项开关与组合，固定负载 / KV / 投机 / capture 配置，测显存、KV 容量、TTFT、TPOT、吞吐与通信时间。
6. 覆盖空闲 DP rank、不均衡 batch、长 Prefill、graph 最大步、投机验证、抢占恢复和长时间运行。O/MLP 仅在合法 D 节点测试。
7. 验证任务输出与数据集精度；HTTP 200 只说明请求成功，不能证明权重已分片或精度已通过。

```bash
rg 'finegrained_tp_config enabled|Disabling oproj_tensor_parallel_size|DSA-PCP O-projection weight sharding is unavailable' serve.log
```

本次有代码与单元测试内容核查，没有执行 pytest 或 NPU E2E。现有 A3/A5 模型配方多未显式开启这几个细粒度项，不能据其存在声称专项组合已经验收。未修改推理源码、未提交或推送。

## 11. 关键代码与文档索引

链接按研究时行号定位。Ascend 固定版本：[365692e28](https://github.com/vllm-project/vllm-ascend/tree/365692e28ee33c5bff29e62cb0e67cc697f336ad)。部分教程仍保留“V4 不支持独立 TP”“V4.1 仅 MRv1”等旧表述，本文结合当前平台校验和硬件范围解释，不把旧描述推广到所有路径。

- **C1 配置定义与校验**：[四项字段](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/vllm_ascend/ascend_config.py#L1334)、[TP/PD/graph/PCP/capture 与 DP 校验](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/vllm_ascend/ascend_config.py#L1364)、[启用判断](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/vllm_ascend/utils.py#L973)。
- **C2 DP 轴模块组**：[parallel_state.py](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/vllm_ascend/distributed/parallel_state.py#L173)。
- **C3 词表三种布局与 Embedding 通信**：[布局优先级与互斥](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/vllm_ascend/ops/vocab_parallel_embedding.py#L93)、[分片分配](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/vllm_ascend/ops/vocab_parallel_embedding.py#L245)、[Embedding交换](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/vllm_ascend/ops/vocab_parallel_embedding.py#L307)。
- **C4 模型和硬件校验**：[普通 TP 上限](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/vllm_ascend/platform.py#L1905)、[硬件无关上限测试](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/tests/ut/test_platform.py#L2253)、[A5 V4.1 强制 MRv2](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/vllm_ascend/platform.py#L1002)、[设备识别](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/vllm_ascend/device/hardware.py#L24)。
- **C5 模型接入**：[V4 O](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/vllm_ascend/models/deepseek_v4/model.py#L500)、[V4 模型级 OTP 池与预留](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/vllm_ascend/models/deepseek_v4/model.py#L889)、[V4.1 O](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/vllm_ascend/models/deepseek_v41/model.py#L381)、[V4.1 模型级 OTP 池与预留](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/vllm_ascend/models/deepseek_v41/model.py#L979)、[公共算子注册](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/vllm_ascend/utils.py#L860)。
- **C6 Engram**：[平台约束](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/vllm_ascend/platform.py#L1785)、[独立表构造](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/vllm_ascend/models/deepseek_v41/model.py#L1058)、[V4.1 A3配方](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/docs/source/tutorials/models/DeepSeek-V4.1-Flash.md#L44)、[A5回归](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/docs/source/tutorials/models/DeepSeek-V4.1-Flash.md#L1898)。
- **C7 细粒度官方指南与采样互斥**：[Fine_grained_TP.md](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/docs/source/user_guide/feature_guide/Fine_grained_TP.md#L1)、[batch-sharded sampling互斥](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/vllm_ascend/ascend_config.py#L961)。
- **C8 O 核心实现**：[跨层 buffer 池](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/vllm_ascend/attention/dsa_v1.py#L98)、[buffer / 整除 / 通信](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/vllm_ascend/attention/dsa_v1.py#L2149)、[跨层共享单测](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/tests/ut/attention/test_dsa_v1.py#L2975)、[不兼容布局单测](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/tests/ut/attention/test_dsa_v1.py#L3038)、[A5 MXFP8 OTP加载](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/vllm_ascend/quantization/methods/w8a8/w8a8_mxfp8.py#L665)、[A5 FP8 OTP单测](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/tests/ut/attention/test_dsa_v1.py#L3119)。
- **C9 O/MLP 执行保护**：[recompute要求](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/vllm_ascend/ascend_config.py#L830)、[PreemptOffload要求](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/vllm_ascend/platform.py#L1215)、[MRv2 graph guard](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/vllm_ascend/worker/v2/model_runner.py#L800)、[offload失败处理](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/vllm_ascend/core/recompute_scheduler.py#L134)。
- **C10 LM head/logits**：[PCP与细粒度通信](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/vllm_ascend/ops/vocab_parallel_embedding.py#L525)、[MRv2 prompt限制](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/vllm_ascend/worker/v2/model_runner.py#L365)、[idle rank](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/vllm_ascend/worker/v2/model_runner.py#L690)、[行数对齐](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/vllm_ascend/worker/v2/model_runner.py#L1097)。
- **C11 投机路径**：[MRv2限制](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/vllm_ascend/worker/v2/spec_decode/lmhead_tp_utils.py#L33)、[MRv2 DSpark](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/vllm_ascend/worker/v2/spec_decode/dspark/speculator.py#L106)、[MRv1 proposer](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/vllm_ascend/worker/v1/spec_decode/llm_base_proposer.py#L1506)、[MRv1 DFlash2](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/vllm_ascend/worker/v1/spec_decode/dflash2_proposer.py#L49)。
- **C12 MLP与Shared Experts**：[MLP通信](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/vllm_ascend/ops/linear_op.py#L263)、[hook选择](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/vllm_ascend/ops/linear_op.py#L410)、[层识别](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/vllm_ascend/ops/linear_op.py#L479)、[shared DP归一](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/vllm_ascend/ascend_config.py#L752)、[V4 MoE构造](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/vllm_ascend/models/deepseek_v4/model.py#L760)、[V4.1 MoE构造](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/vllm_ascend/models/deepseek_v41/model.py#L821)。
- **C13 PCP前置条件**：[平台MRv2限制](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/vllm_ascend/platform.py#L1822)、[DSA-CP互斥](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/vllm_ascend/ascend_config.py#L711)、[PCP manager限制](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/vllm_ascend/worker/v2/pcp_manager.py#L318)、[官方叠加范围](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/docs/source/user_guide/feature_guide/context_parallel.md#L103)、[Hybrid connector边界](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py#L1684)、[V4.1不支持DCP](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/vllm_ascend/attention/dsa_v41.py#L1605)。
- **C14 PCP O**：[默认开关](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/vllm_ascend/ascend_config.py#L583)、[资格检查与回退](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/vllm_ascend/attention/context_parallel/dsa_cp.py#L2584)、[运行策略及权重池](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/vllm_ascend/attention/context_parallel/dsa_cp.py#L2718)、[decode分片选择](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/vllm_ascend/attention/context_parallel/common_cp.py#L13)、[weight switch](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/vllm_ascend/weight_switch/linear.py#L458)。
- **C15 上游对照**：[additional_config容器](https://github.com/vllm-project/vllm/blob/ffcabb8c7ab3e066e1c77b55d7673e5d08cf26cc/vllm/config/vllm.py#L425)、[Embedding自定义组接口](https://github.com/vllm-project/vllm/blob/ffcabb8c7ab3e066e1c77b55d7673e5d08cf26cc/vllm/model_executor/layers/vocab_parallel_embedding.py#L252)、[ParallelLMHead](https://github.com/vllm-project/vllm/blob/ffcabb8c7ab3e066e1c77b55d7673e5d08cf26cc/vllm/model_executor/layers/vocab_parallel_embedding.py#L598)、[标准logits gather](https://github.com/vllm-project/vllm/blob/ffcabb8c7ab3e066e1c77b55d7673e5d08cf26cc/vllm/model_executor/layers/logits_processor.py#L128)、[V4 O](https://github.com/vllm-project/vllm/blob/ffcabb8c7ab3e066e1c77b55d7673e5d08cf26cc/vllm/models/deepseek_v4/attention.py#L212)、[V4.1 O](https://github.com/vllm-project/vllm/blob/ffcabb8c7ab3e066e1c77b55d7673e5d08cf26cc/vllm/models/deepseek_v41/attention.py#L295)、[Engram配置](https://github.com/vllm-project/vllm/blob/ffcabb8c7ab3e066e1c77b55d7673e5d08cf26cc/vllm/config/engram.py#L40)。
- **C16 验证与量化**：[层级量化方法](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/vllm_ascend/quantization/configs/modelslim_config.py#L763)、[Embedding/head单测](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/tests/ut/ops/test_vocab_parallel_embedding.py#L342)、[MRv2细粒度单测](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/tests/ut/worker/test_model_runner_v2_finegrained_tp.py#L279)、[硬件验证要求](https://github.com/vllm-project/vllm-ascend/blob/365692e28ee33c5bff29e62cb0e67cc697f336ad/docs/source/developer_guide/contribution/testing.md#L419)。

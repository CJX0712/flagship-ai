<div align="center">

# 🚢 Flagship AI

**零依赖纯 JavaScript 手写 GPT 引擎 — 从字符到生成，每一行都可验证**

[![Version](https://img.shields.io/badge/version-v15.11-blue)](#)
[![Self-check](https://img.shields.io/badge/selfcheck-12%2F12-brightgreen)](#自检套件12-项全绿)
[![Layer Tests](https://img.shields.io/badge/tests-191%2F191%20passing-brightgreen)](#十层分解架构落地状态)
[![ISA](https://img.shields.io/badge/ISA-1024%2F1024-blue)](#十层分解架构落地状态)
[![Architecture](https://img.shields.io/badge/architecture-L1%E2%86%92L10%20ten--layer-blueviolet)](#十层分解架构落地状态)
[![Regression](https://img.shields.io/badge/regression-R1%E2%80%93R10%20%CE%94%3D0-brightgreen)](#十层分解架构落地状态)
[![CI](https://github.com/CJX0712/flagship-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/CJX0712/flagship-ai/actions/workflows/ci.yml)
[![Dependencies](https://img.shields.io/badge/dependencies-0-success)](#)
[![Node](https://img.shields.io/badge/node-%E2%89%A514-blue)](https://nodejs.org)
[![Browser](https://img.shields.io/badge/browser-file%3A%2F%2F%20direct-orange)](#-浏览器一键运行)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)
[![Author](https://img.shields.io/badge/author-%E6%99%A8%E6%98%9F-purple)](https://github.com/CJX0712)

*Char/BPE Tokenizer · Decoder-only Transformer · GPT-2 ⇄ LLaMA ⇄ LLaMA-GQA ⇄ Mixtral-MoE ⇄ SWA ⇄ Gemma 架构可切换 · INT8 权重量化推理 · MoE 混合专家 · 滑动窗口/层交错稀疏注意力 · QK-norm/logit soft-cap · Attention Sink/StreamingLLM 无限上下文 · YaRN 长度外推 · Hand-written Backpropagation · AdamW · KV Cache · Temperature/top-k/top-p Sampling · 十层分解架构（L3–L10 实拆 + 1024 槽位微指令注册中心 1024 项已实现）· 浏览器与 Node 双端同码*

</div>

🌐 **语言 / Language**：[English](README.en.md) · 中文（当前）

---

## 📑 目录

- [这是什么（v3.0 → v15.11 版本演进）](#这是什么)
- [自检套件（12 项全绿）](#自检套件12-项全绿)
- [性能](#性能)
  - [一键基准报告](#一键基准报告)
- [快速开始](#快速开始)
  - [Node（验证 + API）](#node验证--api)
  - [浏览器（一键运行）](#浏览器一键运行)
- [API 速查](#api-速查)
- [项目结构](#项目结构)
- [设计原则](#设计原则)
- [已知限制](#已知限制)
- [许可](#许可)

---

## 这是什么

一个 **零依赖、单文件** 的 GPT 语言模型完整实现：分词 → 前向 → 手写反向传播 → AdamW → 采样生成，Node 与浏览器双端运行。

**设计信条：零重写，集大成。** 架构不发明，直接站在世界顶级成果之上——

### v3.0 新增（现代 LLM 标准配方，一行切换）

| 能力 | 复用来源 | 说明 |
|---|---|---|
| **模块化架构开关** | GPT-2 / LLaMA / Mistral | `norm`（LN↔RMSNorm）、`ffn`（GELU↔SwiGLU）、`pos`（learned↔RoPE）可独立组合 |
| **架构预设** | `applyPreset(cfg,'llama')` | 一行切到 `RMSNorm + SwiGLU + RoPE`，每个模块均经中心差分梯度检验 |
| **BPE 分词器** | GPT-2 / tiktoken | 字节级（覆盖任意 UTF-8）与字符级（中文语料更省 token）双基元，往返无损 |
| **RoPE 长度外推** | LLaMA | 相对位置不变性，注意力分数只依赖相对距离，支持超长生成 + 滑动窗口精确一致 |

### v4.0 新增（GQA 分组查询注意力，LLaMA-2/3 标准）

| 能力 | 复用来源 | 说明 |
|---|---|---|
| **GQA** | LLaMA-2/3 | `nKvHead` 开关（`<nHead` 启用），Q 头→KV 头分组映射 `⌊h/(nHead/nKvHead)⌋`，KV cache 同步压缩 |

### v5.0 新增（INT8 权重量化推理，weight-only PTQ）

| 能力 | 复用来源 | 说明 |
|---|---|---|
| **INT8 权重量化** | LLM.int8 / AWQ 谱系 | `model.quantize()` 将线性权重转 {Int8, scale} 常驻内存；推理时在矩阵乘内联解量化，权重体积 ≈ 1/8，模型存档 ≈ 1/7（实测 5.6x），对解码无损（logits maxΔ<0.03）。原训练模型不受影响 |

### v6.0 新增（工程化基准收口）

| 能力 | 说明 |
|---|---|
| **统一基准报告 `bench.js`** | `npm run bench` 一键聚合多架构梯度检验 / RoPE 不变性 / GQA / BPE / INT8 量化 / KV cache 加速比，自动生成 [`docs/BENCHMARK.md`](docs/BENCHMARK.md)；`npm run bench:full` 追加**参数对齐全现代架构对照**（GPT-2/LLaMA/GQA/SWA/Gemma/MoE 六套架构等参数预算下 llама 准确率 47.6% > gpt2 34.9%） |
| **CI 门禁闭环** | GitHub Actions（Node 20/22）依次跑 verify(12/12) → probe-v3/v4/v5 → bench，把"可复现可验证"固化成工程纪律 |

### v7.0 新增（MoE 混合专家，Mixtral/DeepSeek 核心稀疏架构）

| 能力 | 复用来源 | 说明 |
|---|---|---|
| **MoE 稀疏 FFN** | Mixtral / DeepSeek / Switch | `nExp>1` 时每层 FFN 替换为 N 个独立专家 + 1 个 top-k 路由网络；每 token 仅激活 `topK` 个专家（稀疏计算），手写前/反向含路由梯度与负载均衡辅助损失 |
| **top-k 门控** | Switch / GShard | `softmax(logits)` 后取 top-k 专家，按门控概率加权求和；离散选择不回传，连续门控概率经 softmax 反向精确回传 |
| **负载均衡辅助损失** | Switch Transformer | `auxWeight · N·Σ_e f_e·P_e`（`f_e`=路由占比冻结不回传，`P_e`=平均门控概率），抑制专家坍缩：实测坍缩 aux=3.59 → 均衡 aux=1.02 |
| **`mixtral` 预设** | Mixtral 8x7B 配方 | `applyPreset(cfg,'mixtral')` 一行得到 `RMSNorm+SwiGLU+RoPE+GQA(nKvHead=2)+MoE(4专家/top2)` |

> MoE 梯度检验在 top-k 选择稳定（大间隔路由）前提下，路由与专家权重中心差分 err/tol = **1.1e-3**；辅助损失梯度 err/tol = **8.3e-6**；未选中专家梯度恒为 0（离散路由正确性）。详见 `scripts/probe-moe.js`（7 项全绿）。

### v8.0 新增（MoE × INT8 组合：稀疏专家的权重量化）

| 能力 | 说明 |
|---|---|
| **MoE 权重量化** | `model.quantize()` 现已把**路由 `wg` 与专家 `e*.w1/w2`** 一并转 INT8（偏置 `bg/b*/ln` 保持 Float64）；稀疏混合专家模型同样获得 ≈ 1/8 权重体积 |
| **组合无损验证** | `scripts/probe-moe-quant.js` 14 项：logits maxΔ<0.1 / KV-cache 末位 Δ=0 / 存档往返 Δ=0 / in-RAM 线性 8x、整体 6.90x / JSON 存档 5.67x / 量化后路由不坍缩（最大专家占用 63%） |
| **CI 基准产物** | GitHub Actions 新增 `probe-moe-quant` 步骤，并把 `docs/BENCHMARK.md` 作为 artifact 上传（每次 CI 运行可下载） |

> 修复：v7.0 的 `quantize()` 用 `Object.create` 复制模型时漏拷 MoE 字段（`useMoE/nExp/topK/auxWeight`），导致量化副本误走稠密路径报错；v8.0 补全，组合才闭环。

### v9.0 新增（滑动窗口注意力 SWA，Mistral/Gemma 稀疏注意力）

| 能力 | 复用来源 | 说明 |
|---|---|---|
| **滑动窗口注意力** | Mistral 7B / Gemma | `window>0` 时查询 i 仅 attend 键 `[i-window+1, i]`（窗外分数掩码 -1e9，softmax 后概率精确为 0）；`window=0`（默认）即标准全因果，旧模型逐位兼容 |
| **全路径一致** | — | `forward` T×T 掩码、`forwardStep` 单 token 窗口切片、生成（缓存/重算）、量化、存档全链路同一窗口语义；RoPE 相对位置保证 KV 缓存逐位等价 |
| **`swa` 预设** | Mistral 配方 | `applyPreset(cfg,'swa')` 一行得到 `RMSNorm+SwiGLU+RoPE+GQA(nKvHead=2)+window=64` |

> SWA 梯度检验（RoPE+SwiGLU+RMSNorm+GQA+window=4 全组合）中心差分 err/tol = **5.2e-4**；窗口局部性：距末位超出 `(window-1)·层数` 感受野的扰动对末位 logits **零影响**（逐位 0），窗内扰动必须可见；KV 缓存增量 vs 全序列重算 maxΔ=**0**。详见 `scripts/probe-swa.js`（8 项全绿）。

### v10.0 新增（层交错稀疏注意力，Gemma 2/3 局部/全局模式）

| 能力 | 复用来源 | 说明 |
|---|---|---|
| **逐层窗口** | Gemma 2/3 | `window` 支持三种写法：标量（所有层同窗）、数组（逐层窗口，长度须=nLayer）、哨兵 `{alt:N}`（偶数层全因果、奇数层窗口 N 的 1:1 交错）；统一归一化为 `model.windows` 逐层数组 |
| **局部/全局混合深度** | Gemma 2/3 | 滑动窗口层管局部依赖 O(T·w)，全局层保长程信息；静态掩码逐层应用，反向传播经缓存 `Ps` 自动继承（零新增代码） |
| **`gemma` 预设** | Gemma 配方 | `applyPreset(cfg,'gemma')` 一行得到 `RMSNorm+SwiGLU+RoPE+GQA(nKvHead:2)+局部/全局层交错(window:64)` |

> 层交错梯度检验（windows=[4,0,8] 混合）err/tol = **5.2e-4**；逐层窗口独立生效（[4,0]/[0,4]/[4,4]/[0,0] 四配置 logits 两两互异）；数组全 0 ≡ 不传 window（零回归逐位一致）；KV 缓存逐步一致 maxΔ=**0**；存档（数组/哨兵）与 INT8 量化后 windows 无损。详见 `scripts/probe-gemma.js`（12 项全绿）。

### v11.0 新增（Gemma 2 稳定化组件：QK-norm + logit soft-cap）

| 能力 | 复用来源 | 说明 |
|---|---|---|
| **QK-norm** | Gemma 2/3 / Chameleon / ViT-22B | `qkNorm:true` 时对每个头的 Q/K 向量做 RMSNorm（RoPE **前**，增益按 head_dim 跨头共享）；稳定注意力 logits 尺度，抑制训练发散。归一后每头 RMS ≤ 1（恒等式 `ss/dh+eps=1/rstd²` 实测误差 6.4e-16） |
| **logit soft-cap** | Gemma 2（cap=30） | `logitCap:N` 时最终 logits 套 `N·tanh(x/N)`，输出**严格有界** `\|logits\|≤N`；反向乘 `sech²(x/N)=1−(lg/N)²`。无帽模型训练后 max\|logit\|=6.05，cap=1.5 后压到 1.46（削峰生效） |
| **`gemma` 预设升级** | Gemma 2 完整配方 | `applyPreset(cfg,'gemma')` = `RMSNorm+SwiGLU+RoPE+GQA(nKvHead:2)+层交错(alt:64)+QK-norm+logitCap:30` |

> QK-norm 与 soft-cap 梯度检验（含增益 qg/kg）err/tol 分别 = **1.4e-3** / **6.5e-3**；双开关默认全关 = 逐位零回归（关闭时不产生 qg/kg 参数）。soft-cap 反向曾因误写 `tanh(lg/cap)`（二次 tanh）被梯度检验抓出（err/tol=3.69），修正为 `1−(lg/cap)²` 后闭环。详见 `scripts/probe-gemma2.js`（10 项全绿）。



### v12.0 新增（KV Cache INT8 量化缓存，推理期显存压缩）

| 能力 | 复用来源 | 说明 |
|---|---|---|
| **KV INT8 量化** | LLM.int8 / 推理优化谱系 | `kvQuant:8` 时 K/V 缓存存 `Int8Array`，per-(token,KV头) 对称 absmax 量化（scale=max\|x\|/127）；写入即量化、无需校准集；误差硬界 `|dequant−x| ≤ absmax/254`（相对 ≤ 1/254 ≈ 0.39%） |
| **显存压缩** | — | KV 缓存由 Float64(8B) → Int8(1B) + per-(token,head) scale(8B)；典型 L=4/B=64/KV头=2/dh=16 下 Float64 128KB → INT8+scale 24KB（**5.33x**），对解码无损（logits maxΔ<5e-4） |
| **全组合兼容** | — | 与 INT8 权重量化、SWA、层交错窗口、GQA、MoE、QK-norm、logit soft-cap 任意组合均经探针验证；`kvQuant=0`（默认）逐位等价于旧版 Float64 缓存 |
| **kvShift 同步** | — | 滑动窗口平移时 scale 与码字同步 `copyWithin`，解量化不错位 |

> KV INT8 与 F32 缓存逐步末位 logits 差异有界（占动态范围 <0.02%）；贪心解码 INT8-KV 与 F32-KV 生成 token 逐字一致；端到端采样无 NaN/Inf、同种子可复现。详见 `scripts/probe-kvquant.js`（11 项全绿）。

### v13.0 新增（Attention Sink / StreamingLLM，推理期无限上下文）

| 能力 | 复用来源 | 说明 |
|---|---|---|
| **Attention Sink** | StreamingLLM（Xiao et al., 2023） | `attentionSink:K`（K>0）时 KV 缓存**恒驻首 K 个锚点 token**，其余为滚动窗口（容量 = K + 滚动窗口 R）；注意力永远包含锚点，故可丢弃中段、支持无限长度流式推理 |
| **滚动窗口淘汰** | — | 缓存满（K+R）后每步淘汰最旧的非锚点 token（仅平移窗口区 `[K, n)`，`copyWithin` 原生 memmove，O(容量·D) 无重算），锚点 `[0, K)` 原地不动 |
| **显存/长度解耦** | — | 缓存容量与序列长度脱钩：生成任意长文本，缓存恒为 K+R（如 4+16=20 token），显存恒定、延迟恒定 |
| **零干预训练** | — | StreamingLLM 仅作用于推理增量路径（`forwardStep`/`kvShift`），`forward` 全序列训练不受影响，**反向传播/梯度检验零改动** |
| **全组合兼容** | — | 与 KV INT8 量化、SWA 滑动窗口、INT8 权重量化、GQA、MoE、QK-norm、soft-cap 任意组合均经探针验证；`attentionSink=0`（默认）逐位等价于旧版滑动窗口 |
| **kvShift scale 同步** | — | 锚点感知淘汰时 K/V 与（INT8 下）scale 同步平移，解量化不错位 |

> 核心不变量：远超容量后 `slot[0..K)` 的 K/V 与初次写入**逐位一致**（锚点永不被淘汰）；StreamingLLM 末位 logits 与纯滑动窗口**明显不同**（证明锚点确实被 attend，maxΔ=2.5e-2）；greedy 300 token 同种子可复现、无 NaN/Inf；× KV INT8 末位 maxΔ<2e-4。详见 `scripts/probe-streaming.js`（10 项全绿）。

### v14.0 新增（YaRN 长度外推，长上下文）

| 能力 | 复用来源 | 说明 |
|---|---|---|
| **NTK-by-parts RoPE 频率插值** | YaRN（Peng et al., 2023） | `ropeScale:{type:'yarn',factor}` 时按 HF 同式对 RoPE 频率做逐维插值：高频维（小 p）保留、低频维（大 p）插值，`inv_freq /= factor^mask`（`mask∈[0,1]` 由 correction range + smoothstep 决定）；`type:'ntk'` 为全局基频缩放特例 |
| **注意力温度（mscale）** | YaRN | softmax 前 logits 乘 `mscale = 0.1·ln(factor)/ln(2)`，补偿频率插值带来的幅度变化；`factor≤1` 或 `originalMaxLength` 省略时取训练上下文长度 |
| **长上下文外推** | — | 频率被压缩后，远超 `blockSize` 的绝对位置不会产生 NaN/Inf，支持在比训练更长序列上推理（与 SWA/StreamingLLM 叠加 = 完整长上下文栈） |
| **零干预训练** | — | 仅作用于 RoPE 旋转（前后向同频）与注意力 logits 缩放（线性因子前后向一致），`forward` 全序列训练 + 反向传播 + 中心差分梯度检验**零改动**（`factor>1` 下 maxErr/Tol≈2e-3） |
| **全组合兼容** | — | 与 KV INT8、SWA、Attention Sink/StreamingLLM、INT8 权重、GQA、MoE、QK-norm、soft-cap 任意组合均经探针验证；`ropeScale` 缺省或 `factor≤1` 逐位等价于旧版 RoPE |

> 核心不变量（详见 `scripts/probe-yarn.js` 5 项全绿）：`factor=1` 与缺省配置逐位等价（Δ=0）；`forward` 与 `forwardStep` 在 YaRN 下逐位一致（Δ=0）；`factor>1` 中心差分梯度检验 maxErr/Tol≤1；`mscale` 公式精确（`0.1·ln(4)/ln(2)=0.2`）；`freqScale∈[1/factor,1]` 且非递减；`pos≫blockSize`（如 3000）输出有限且同输入确定性 Δ=0。

### v15.11 修复（L10 A–E 五域谓词纪律：640 项全部「先调 run，再验性质」+ 变异门禁）

**发现的系统性缺陷**：门禁只调 `op.predicate(...args)`，**从不调 `op.run`**（`tests/lib/op-harness.js:43`、`scripts/probe-microisa.js:106`）。A–E 五域 640 项谓词中，存在大量「把 run 的定义式内联复算、或直调 run 内部的底层 helper」的自比断言 —— run 若被改坏，门禁完全失明（假绿）。**E 域最严重：128/128 项谓词全部直接调 `NS.L7.shape.*` 或本地 helper，从不调本 op 的 run**，即整域彻底假绿。

**修复范围**（A-scalar-000~003、B-vector-000~003、C-matrix-000~003、D-attention-000~003、E-shape-000~003 共 20 文件 640 项，+6046 / −3752 行）：

| 域 | 原缺陷 | 修复手法 |
|---|---|---|
| **E（形状与索引）** | 128/128 谓词直调 `NS.L7.shape.*`（size / linearize / contiguousStrides / broadcastShapes / permute / invertPerm），run 从未被调用 | 全部经 `runOf('Exxx')` 调 run；oracle 取本文件内第一性原理实现（`prod` 维度乘积、`suffixProds` 后缀积、`dotIdx` 本地点积、`bcastLocal` 本地尾部对齐广播、`invLocal` 本地逆置换、`Math.floor(d/2)` 与 run 的位运算写法区分）+ 逐元素独立重建 |
| **D（注意力）** | 谓词「独立校验」但比对对象是本地 helper（det2 / isSym / colVar / vcat）—— 与 run 同源，run 损坏仍失明 | 全部改 `runOf` 路由；跨 op 互引（D001 裸点积、D013 因果、D025 softmax、D030 LSE）；幂等经 run 双重调用（D016↔D013） |
| **C（矩阵）** | C035 / C062 谓词是 `meq(o,o)` 自比较（**恒真假绿**）；C104–C106、C120 的「独立校验」比对对象仍是本地 helper | 改内联索引 oracle / 独立三重循环 / 前向 vs 后向递推（方向相反的独立实现）；C106 第二子句逻辑反向亦修正 |
| **B（向量）** | B106 / B108 谓词误传 2 参给 3 参 run（z=undefined → 崩溃）；多处仅校验长度或单边不等式 | 修传参；长度类判据升级为「钉死内容逐位比对」 |
| **A（标量）** | A107 只断言「非有限」，不钉符号与 NaN 分支 | 强化为 `a>0 ⇒ +∞`、`a<0 ⇒ −∞`、`a=0 ⇒ NaN` 三分支精确刻画 |

**新增变异测试门禁（`mutcheck`）——证明断言非空转**：光靠「全部通过」无法证明门禁有牙齿。`mutcheck` 对每个 impl op 采集原 run 下谓词为 true 的输入集，然后把 `op.run` 替换为四种损坏版本（`shift` 每次调用加不同常数 / `neg` 逐元素取负 / `rev` 数组反序 / `flip` 布尔翻转、NaN→0、字符串加后缀），重跑谓词 —— 若四种扰动全部漏网，即判为**弱判据（潜在假绿）**并列出。这道门禁额外暴露了 **16 项此前全绿但判据过松的谓词**，全部补独立 oracle 钉死：

| 弱判据 | 原判据 | 钉死方式 |
|---|---|---|
| A107 | `isFinite(run(a)) === false` | 三分支精确（+∞ / −∞ / NaN） |
| B078 / B104 | 只验单边不等式（超可加 / 三角不等式），返回常量也过 | 钉死 `minᵢ(xᵢ+yᵢ)` / `maxᵢ|xᵢ−zᵢ|` |
| B121 / B123 | 只验长度 | 钉死拼接 / 切片的逐位内容 |
| C015 / C121 | 只验长度（转置 / 垂直拼接） | 钉死逐位转置关系 / 逐位拼接内容 |
| C059 | 只验 idx/val 长度 | 钉死每行 top-2 次序统计值 + idx↔val 自洽 |
| C065 / C080 | 只验输出长度（matmul） | 独立三重循环钉死乘积 |
| C075 | 只验 `fro(P) == √ΣP²`（对**任意** P 都成立，不校验 matmul 本身） | 补独立三重循环钉死乘积，再验 Frobenius 自洽 |
| D006 | 只验 Cauchy–Schwarz 上界（不钉值） | 钉死 `\|⟨q,k⟩\|` |
| D018 | 只验掩码对称性 | 独立 `\|i−j\|>w` 可见性 oracle 钉死掩码 |
| D019 | 只验单调性符号 | 闭式可见格点计数差钉死值 |
| D051 / D055 | 只验对称性 / 有限性 | 钉死 `−slope·\|i−j\|` |

**废弃 6 处恒真式假绿**（D 域）：D074 `nH*hd===nH*hd`、D075 `d!==d` 恒假循环、D078、D087 `(base+pos*d)===base+pos*d`、D089 `(i+1)-i!==1`、D094 —— 全部改为 run 耦合断言 + 定点锚。

**修复一处 run 本体潜伏缺陷（D079）**：`cacheAppend(cache, L, cap, d, row)` 误以 5 参调用 2 参的 `cacheAppend` —— 因谓词从不调 run，该崩溃从未被发现。改为 `cacheAppend({data: cache, len: L, cap, d}, row)`。

**校验脚本自身两次扩能（消除伪红误判）**：① 边缘用例池原为定长 4×4 / 8 维，与 op 声明的 `matrixDims` / `vecDims` 冲突会 OOB 读出 `undefined` → 伪红；改为按声明维度自适应伸缩（平铺 / 截断）。② 组合上限 400 会把尾部输入截断为 `undefined` → 伪红；提至 5000。**纪律沉淀：先证明失败是真实缺陷而非工具伪红，再改代码。**

**验证**：双模式校验（4 种子 × 80 例随机 + 边缘用例手工构造）覆盖 640 项 —— A `44971/0`、B `47781/0`、C `42601/0`、D `80493/0`、E `42380/0`，合计 **258226 例 0 失败**；变异门禁 **640 项 0 弱判据**。全门禁：verify.js 12/12、lint 72 文件 0 违例、isa 1024/1024、probe-microisa 8/8、tests/L10 8 文件 8/8、13 组探针全绿、demo.html 重建两次 md5 一致。

### v15.10 新增（ALiBi 线性距离偏置 · 第 7 套位置表示）

第 7 套位置/注意力写法：**不加位置嵌入、不做旋转**，而是把「距离惩罚」直接打进注意力打分里。

| 能力 | 复用来源 | 说明 |
|---|---|---|
| **ALiBi 偏置** | Press et al., 2022（ALiBi）· BLOOM / MPT | softmax 前打分加 `−slope[h]·|i−j|`；无参数、不依赖 Q/K ⇒ `softmaxBwd` 已含偏置，**注意力反向零改动** |
| **斜率闭式** | ALiBi `get_slopes` | `slope[h] = 2^(−8h/n)`，`n` = 不小于头数的最小 2 的幂；**0-based**（`slope[0]=1` 惩罚最重），与 L10 微指令 μD045 / μD053 参考实现逐位一致 |
| **取代位置编码** | BLOOM / MPT 的发表设计 | `alibi:true` 时强制 `useRope=false` 且**不创建 `posEmb`**，省 `blockSize·dModel` 个参数；`applyPreset(cfg,'bloom')` 一行启用（LayerNorm + GELU + ALiBi） |
| **双端逐位一致（含淘汰后）** | — | 缓存槽位原本与绝对位置解耦，`kvShift` 后映射会丢失（RoPE 因只依赖相对距离而幸存，ALiBi 不是）⇒ 新增 `kv.pos` 记录每槽绝对位置，并与 K/V（及 INT8 scale）同步平移 |
| **关闭零回归** | — | ALiBi 默认关闭；关闭时 `slopes=null`、`kv.pos` 不分配 ⇒ 与 legacy（未改动的老代码）前向与增量路径 **Δ=0** |

> 核心不变量（`scripts/probe-alibi.js` 13 项全绿）：斜率与 μD045/μD053 `run` 逐位一致；零 qkv 下 `log(pⱼ/p₀)=slope·j`（证明偏置真的进了 softmax）；`forward ≡ forwardStep` 在短序列/滑动窗口/**500 token 淘汰后（cap=16）**/锚点+滚动窗口下均 Δ=0；**整体平移不变**（同一序列从绝对位置 1000 起跑逐位相同）——这条能抓住"拿槽位当位置"的实现；中心差分梯度检验 err/tol=4.68e-3 ≤ 1；并把引擎斜率喂给 μD045–μD056 / μD112 / μD113 / μD114 / μD117 / μD119 的 `run` 与 `predicate` 逐项通过。已做过**故障注入自检**：把 `kv.pos[slot]` 改成 `slot` 后 5c/5d 立即转红，证明这两项不是空跑。

> ⚠️ 下标约定说明：HF `transformers` 的 `get_slopes` 是 **1-based**（`slope[h]=2^(−8(h+1)/n)`），本项目按任务要求以**微指令登记中心的已证数学**为准取 **0-based**。二者是同一几何级数错一格的关系，选 0-based 才能与 μD045/μD053 的 `run` 输出做逐位对拷。

### v15.9 新增（bench:full 参数对齐全现代架构对照）

把 `npm run bench:full` 从「gpt2 vs llama 两项对照」升级为**六套现代架构等参数预算横向对照**：以 gpt2 参数量为基准、二分各预设 `ffnHidden` 把 **GPT-2 / LLaMA / LLaMA-GQA / SWA / Gemma / Mixtral-MoE** 拉到相等参数预算（对齐偏差均 < 5K），再同语料/同种子/同步数训练，报 final loss 与准确率，并给出「等参数预算最优（准确率 / 最低 loss）」结论。

| 架构（ffnHidden 对齐后） | final loss | 准确率 | 参数 |
|---|---|---|---|
| gpt2 (256) | 1.910 | 34.9% | 124,480 |
| **llama (181)** | 0.859 | **47.6%** | 124,244 |
| llama-gqa (203) | 0.831 | 44.4% | 124,460 |
| swa (203) | 0.831 | 44.4% | 124,460 |
| gemma (202) | **0.802** | 44.4% | 124,136 |
| mixtral/MoE (50) | 1.260 | 38.1% | 124,200 |

**结论**：等参数预算下现代 RMSNorm+SwiGLU+RoPE 配方（llama 47.6%）显著优于 vanilla GPT-2（34.9%），gemma 拿到最低 loss；MoE 在小专家 + 微语料下暂不占优（诚实呈现）。swa 与 llama-gqa 结果逐位相同——window=64=blockSize 时滑窗退化为全注意力，本身即一条正确性佐证。对齐用的二分搜索对 GQA（KV 头少→需更大 ffnHidden）与 MoE（专家多→需更小 ffnHidden）均收敛。

### v15.8 新增（CI 真实浏览器渲染/交互门禁 + demo 截图）

把 demo 从「代码层可验证」升级为「真实浏览器层可验证」——用 Playwright headless Chromium 直开 `demo.html`（file://，与用户双击同一条路径），新增一条独立于 `probe-demo`（结构校验）与 `test:layers`（Node 单测）的可视化验证通路：

| 能力 | 说明 |
|---|---|
| **真实渲染门禁** | `scripts/probe-demo-browser.js`（`npm run test:demo-browser`）在 headless Chromium 加载 demo，断言十层栈装配完成、`window.FlagshipNS.L10.stats()` = 1024/1024（八域各 128） |
| **交互断言** | 微指令浏览器 DOM 渲染（八域摘要卡 8/8 + 列表首屏 60 条 + 计数 1024/1024）；点「A」域筛选 → 计数变 128/1024、首项 key=μA001、chip `aria-pressed=true`；点首项展开七件套详情 8 段 |
| **批量不变量** | 触发「批量试运行」：1024 项 × 8 种子 = 8192 次谓词求值，断言 `✓ 1024/1024 项通过`、失败 0（实测 ~0.8s） |
| **截图留痕** | 产出 `docs/demo-screenshot.png`（下方头图 + CI artifact），把整条微指令不变量验证通路可视化 |
| **CI 集成** | GitHub Actions 新增步骤：按 `package-lock` 哈希缓存 Playwright 浏览器（免每次重下 ~130MB Chromium）→ `playwright install chromium` → 跑门禁 → 上传截图 artifact |
| **零运行时依赖不变** | Playwright 仅入 `devDependencies`，`dependencies` 仍为 0（CI 零依赖断言照常通过）；Chromium 解析顺序：`PW_CHROMIUM_EXECUTABLE` → Playwright 托管 → 本机 ms-playwright 缓存扫描 |

### v15.7 新增（demo 视觉/交互升级 + L10 微指令浏览器）

十层架构与 1024 项微指令全量落地后，把 demo 从「能用」升级为「可展示」：

| 能力 | 说明 |
|---|---|
| **设计系统** | CSS 变量驱动的配色/间距/圆角/阴影/字体栈；按钮 primary·ghost·禁用态；输入与下拉统一风格；`:focus-visible` 键盘焦点可见；hover 过渡 |
| **信息架构** | 配置项按语义分组（① 语料与模型 ② 架构结构 ③ 训练 ④ 注意力与上下文 ⑤ 采样 ⑥ 量化与外推 ⑦ 图表与进度 ⑧ 代码/日志 ⑨ 微指令浏览器 ⑩ 响应式），每项配可见说明与默认值提示 |
| **状态与反馈** | Header 徽章组（v15.7 / L10 微指令 1024/1024 / 十层架构 L1–L10 / zero-dependency）；模型信息升级为结构化状态卡；构建与训练按钮 loading 态；错误以内联提示呈现（不阻塞） |
| **训练可视化** | `<canvas>` loss 图表加坐标轴、网格线、曲线配色与 legend；进度条与进度文本样式重做 |
| **L10 微指令浏览器** | 展示注册中心全貌（八域明细 + 合计 1024/1024）；按域筛选（全部 + A–H 九按钮）；关键字搜索（匹配 op.name / op.key）；点击展开七件套（直觉/数学/伪代码/复杂度/论文/坑/不变量 + 输入域）；**「试运行」在浏览器内真实执行该 op 的 predicate**（参数由 `NS.L9.rng.mulberry32` + `NS.L10.genScalar/genVector/genMatrix/genShape` 确定性生成）；**「批量试运行」对全部 1024 项 × 8 种子跑不变量并汇报通过/失败** |
| **响应式** | 宽屏多栏、窄屏单栏；小屏不破版 |

**浏览器内批量试运行实测**：1024 项 × 8 种子 = 8192 次谓词求值 **全绿**（1271ms）——即 demo 页面本身成为一条独立的微指令不变量验证通路，与 `npm run test:layers` 互为交叉验证。

**零依赖约束全程保持**：无 CDN / 外链字体 / 外链图片 / ES module，图表继续用原生 canvas 手绘，`demo.html` 仍可 file:// 直开。

### v15.7.1 修复（L10 F/G 域谓词纪律：224 项全部「先调 run，再验性质」）

**发现的系统性缺陷**：门禁只调 `op.predicate(...args)`，**从不调 `op.run`**（`tests/lib/op-harness.js:43`、`scripts/probe-microisa.js:106`）。但 F 域 96 项 + G 域 128 项中有大量谓词把 run 的定义式**内联复算或硬编码在定点上自比**，等于「run 等于它自己」的恒真断言 —— run 若被改坏，门禁完全失明（假绿）。

**修复范围与手法**（F-norm-000/001/002、G-act-000/001/002、G-actloss-003 共 6 文件 224 项）：

| 原缺陷类型 | 实例 | 修复 |
|---|---|---|
| 恒真自比（参考与 run 同式） | G099 参考 `S.softplus` 与 run 表达式逐字相同；G113 参考 `argmax===k?1:0` 即 run 定义式 | 改异构参考（tanh 形式 softplus、两遍扫描 argmax） |
| 恒真但被误当 oracle | G103 `rc(r,-tanh(-x))`（sin 亦满足）；G105 `rc(r,1-σ(-x))` | 换 exp 形式 / tanh 形式真 oracle |
| 纯界断言（无 oracle） | G104/G111/G119/G120/G127 只查区间，返回 0.5 也过 | 加 log 反向还原 / softplus 还原 / 构造极值 |
| 参考逐字复制 run | G114/G115/G116/G117/G121/G122–G124 | 反向累加 + 不同结合方式 |
| vacuous 放行分支 | G097 `x>700` 裸 `return true` | 改为断言 `r === MAX_VALUE` |
| 恒被自比蒙混 | G107/G128 只 `r>=0 && kl(p,p)==0` | 加分离对数式参考 + one-hot 构造 |
| 退化域未加守卫 | F073/F087/F078/G033/G036/G088/G089/G103 | 守卫而非放宽容差，并同步改正文档声称 |

**同时修复 3 处真实数值缺陷**（run 本体，非谓词放宽）：

- **G117 温度缩放 CE**：原 `softmax` 后 `log(max(p,1e-12))`，温度极小（T∈[1e-6,10]，z/T 可达 1e7）时分量下溢为 0，损失被错误截断在 27.63。改为 log 域 `-(z_t/T − LSE(z/T))`，与 G101/G110 路径统一。
- **G120 对比损失**：原用 `‖x‖²+‖y‖²−2x·y` 展开式求 `‖x−y‖`，x≈y 时相消放大。改为直接 `Σ(x_i−y_i)²`。
- **G121 triplet loss**：同理，`d_ap/d_an` 均改为直接差平方。

**一处文档纠错**：G118 的 `kk = safeIdx(k,n)` 取模后夹取到 `[1,n-1]`（不是 `[1,n]`），「kk=n 时退化为完整 CE」在当前域下不可达 —— 已改为 `kk=1 ⇒ max(z)−z_t` 的构造断言并在文档标明真实取值范围。

**验证**：独立校验脚本双模式（4 种子 × 80 例随机 + 边缘用例手工构造：one-hot/全常数/次正规数/近并列/极端温度）覆盖 224 项，F 域 `total=44311 fail=0`、G 域 `total=46827 fail=0`。全门禁（并入 v15.8 主线后复核）：verify.js 12/12、node --test 191/191、lint 72 文件 0 违例、isa 1024/1024、probe-microisa 8/8、regression R1–R10 Δ=0、11 组探针全绿、真实浏览器门禁（1024×8 谓词）全绿。

### v15.6 新增（L10 微指令 A–E 五域补满：八域 1024 槽位全落实，864 → 1024）

八域 × 128 槽位**全部落实**（此前 A–E 各 96 + F/G/H 各 128 = 864）。本批为 A/B/C/D/E 五域各补第 4 批 32 项（X097–X128），共 160 项：

| 域 | 指令 | 代表不变量 |
|---|---|---|
| **A 标量** | A097–A128 | sign 奇/乘性/原点、copysign、fmod 符号随被除数、clamp 夹取/保序/两定义一致、NaN 传播（0·∞、∞−∞）、isFinite/isNaN、ulp 为 2 的幂、nextafter 可逆、exp2/log2 换底、cbrt 奇、atan2 四象限/对跖、度弧度往返、三参 hypot、min/max 交换·结合·幂等·对偶·次超可加 |
| **B 向量** | B097–B128 | Lp(p=3) 齐次/非负/三角、加权 L2、余弦距离有界 [0,2] 且自为 0、切比雪夫距离对称/三角、汉明距离对称/三角、L1 距离、cumprod 末项即总积、cummax/cummin 单调且末项即极值、argmax 选中最大值、one-hot 恰一个 1、gather 单热桥接、concat 长度可加且偏移切片还原、slice 半开长度/内容、pad 保原值、rotate 复合还原 |
| **C 矩阵** | C097–C128 | det 2×2 公式、det 转置不变、det(I)=1、对角/三角阵 det 为对角线积、det 乘积性（2×2 一般 + 3×3 固定对角因子）、2×2 迹=特征值之和、对称/反对称判定、列均值/列 softmax/列 argmax、行 LSE 上下界、列方差非负·平移不变·等于转置行方差、垂直/水平拼接形状内容、行切片、矩形双转置、对角提取、pad 补零 |
| **E 形状** | E097–E128 | squeeze 去 1 维无残留且幂等、unsqueeze 保 size 且与 squeeze 互逆、expand 尾形广播 size/单元素复制、flatten 指定轴区间保 size、split 与 flatten 互逆、moveaxis 保 size、ravel/unravel 往返、sub2ind/ind2sub 行主序往返与越界抛、负索引归一、slice 偏移/半开长度公式、翻转轴镜像恒等式、contiguous 后缀积准则、广播三形状结合律、共轭置换仍为置换、空维移除互逆 |

**D 域（D097–D128）**在 D001–D096 原子操作之上覆盖**组装 / 位置编码 / KV Cache 三层交叉的复合判据**：

D 域由 96 补满至 128，在 D001–D096 原子操作之上覆盖**组装 / 位置编码 / KV Cache 三层交叉的复合判据**：

| 子域 | 指令 | 代表不变量 |
|---|---|---|
| **组装语义** | D097–D108 | 缩放打分 CS 有界、softmax 掩位归零、输出是 V 凸组合、因果行支撑=i+1、等分打分权重 1/n、低温 one-hot 守卫、V 平移⇒输出平移、V 缩放⇒输出缩放、端到端=lse 参考、因果首行逐位=V 首行、query 行局部性、常列守恒 |
| **RoPE 深度性质** | D109–D111, D115–D116, D118, D120 | 打分=q 与相对旋转 k 内积、步进角严格有序、基向量旋转形式、旋转只触碰前 2⌊d/2⌋ 维、频率公比恒定、**相邻位置 Lipschitz 逐对弦长界**、行分布整体平移不变 |
| **ALiBi 与复合** | D112–D114, D117, D119 | 等分打分几何衰减、RoPE×ALiBi 联合平移不变、斜率累积积闭式、行距离和闭式、行最小偏置闭式 |
| **GQA×Cache 复合** | D121–D128 | 头间写入独立、追加后 QK 打分前缀逐位不变、组内共享 KV 一致、滚动长度守恒、增量 softmax 旧权重等比缩放、输出递推闭式、滚动后锚点逐位不变、增量解码=全量重算逐位一致 |

**D118 Lipschitz 界的收紧**（变异测试倒逼）：相邻位置打分变化界由粗式 `2sin(f₀/2)·‖q‖·‖k‖` 收紧为逐对弦长 `Σᵢ2sin(fᵢ/2)·‖qᵢ‖·‖kᵢ‖`——随机输入下粗式松约 2 倍（仅最高频对有满弦长而 ‖q‖ 含全部 4 对），连「频率翻倍」级实现 bug 都漏判；对应变异 twin（漏因子 2）468/800 被判负，证明紧式有牙齿。

**QA 三件套**（交付前全过）：① 32 项 × 2500 种子确定性扫描（80,000 条断言）0 失败；② 32 个变异 twin 全部被判负（每项注入一个合理漏洞，证明断言非空转）；③ 条件守卫分支激活率统计（如 D127 淘汰路径 706/800、D102 保序守卫 800/800），证明守卫被真实行使而非摆设。

**修复一处真实的数值 bug（F084）**：F-norm-002.js 的 `gnNormRef`（GroupNorm 参考路径）原用简化方差公式 `v = Σx²/n − μ²`。该式数学等价但**数值病态**——「均值大、方差小」时两个相近大数相减触发**灾难性抵消**，方差精度崩塌（本例 μ≈5、组方差极小），导致用病态参考路径去验证稳定的两遍法实现时，在 2500 种子扫描中出现假红（seed 1450）。修复：参考路径改用 **Welford 在线算法**（单遍增量更新均值与 M2），既数值稳定、又与两遍法算法结构完全不同，保留交叉验证力。修复后 1024 项 × 2500 种子扫描 0 失败。**纪律沉淀：参考路径必须独立且数值稳定——不可用病态公式（简化方差/E·[X²]−μ² 类）充当验证基准。**

### v15.5 新增（L10 微指令 H 域：采样与量化 128 项 + F/G 补满 128 + 三处 latent 修复）

补齐八域最后一域 **H（采样与量化）** 并将 F/G 由 96 补满至 128，微指令实现数 **672 → 864**（A–E 各 96 + F/G/H 各 128）。F097–F128（logsumexp 高级/数值稳定、矩阵/批/高级归一化）、G097–G128（数值稳定交叉熵、温度缩放、Gumbel-softmax、加权 CE 等）与 H 全域由 T06 一次落地：

| 子域 | 指令 | 代表不变量 |
|---|---|---|
| **H 确定性采样·top-k/p** | H001–H032 | argmax 唯一、categorical 采样分布一致、top-k 截断 Σ=1、nucleus 累积覆盖、Gumbel-max 重参数 |
| **H 温度·舍入** | H033–H064 | 温度除 logits、温度后 softmax Σ=1、温度保序、低温锐化熵不增、各类舍入（四舍五入/向上/向下/截断/银行家） |
| **H INT8 对称·仿射** | H065–H096 | 对称量化 round(x/s)、仿射量化逐通道 scale/zero、反量化误差界、饱和裁剪、per-channel 守恒 |
| **H 量化性质·高级** | H097–H128 | 量化单调、zero-point 往返、动态/静态量化、端到端 KL 有界、混合精度 |

**三处 latent predicate 失败已修复**（G087 gumbel-softmax 单调 / G093 温度缩放概率单调 / H041 温度保序）：极端温度（如 T=0.0167）或大幅值 logits（如 9232）下，多个相异 logit 经 x/T 后都坠入 softmax 浮点下溢尾部（二者均≈0），严格输出序比较对数学上应为严格大于的一对失效（双零 tie）。修复为「缩放 logit 精确序 + 可解析性守卫」——softmax(z/T)_i > softmax(z/T)_j ⇔ (z_i−m)/T > (z_j−m)/T（T>0 恒精确），仅当 si−sj>1e-6 且 si>−740（未坠入下溢尾部）时才断言严格输出序：既消除假阳性，又保留对真正乱序/常量输出回归的检测力。2500 种子确定性扫描复验 0 失败。

### v15.4 新增（L10 微指令 G 域：激活与损失 96 项）

补齐八域中的 **G 域（激活与损失）**，微指令实现数 **576 → 672**，覆盖激活函数族与损失函数族的全链路原子操作：

| 子域 | 指令 | 代表不变量 |
|---|---|---|
| **基础激活** | G001–G032 | sigmoid 单调·界·对合·logit、tanh 奇/界/双曲、ReLU 非负·正区恒等·稀疏、LeakyReLU 分情况、PReLU 仿射、ELU 连续·负区有界·阈值、Softplus 凸·正·非负·上界(ln2+max(0,x))、Softsign 奇·界·定点、HardTanh 三分段·饱和 |
| **现代激活** | G033–G048 | GELU/Swish/Mish 正支单调增·非负·上界、SELU 缩放·不动点·与 ELU 关系、HardSigmoid 三定点·界、LogSumExp 平滑最大值·上界·坐标平移、Softmax 解析归一·行和为 1·坐标单调、LogSoftmax 与 log(softmax) 一致 |
| **损失函数** | G049–G096 | MSE 对称·非负·常数差、MAE 分段梯度、Huber 连续·分界、SmoothL1 连续、BCE 0/1 极值·凸、BCE-logits 数值稳定·与 BCE 一致、CE 非负·one-hot 退化、CE-logits 平移不变·数值稳定、KL 非负·0 当且仅当同分布、Focal 减权易分·γ=0 退化 CE |

**关键的数学纪律**（与 F 域一脉相承）：
- **解析性质优先于浮点拟合**：softmax 行和=1、KL≥0、CE≥0、softplus 上界等为精确恒等式/不等式，predicate 走独立参考路径（硬编码定点 / 不同公式 / 解析推导），不靠放宽容差造假绿。
- **真实数值 bug 已修复**：`ceLogits` 的 log-sum-exp 原误写为 `log(m + Σexp(zᵢ−m))`（漏外层 m），致 LSE 偏小、CE-logits 出现负值；修正为 `m + log(Σexp(zᵢ−m))` 后，CE-logits 严格≥0 且平移不变。独立校验脚本（4 种子 × 80 例 = 30720 次）全过。

> 修复：独立校验在接入主链路前抓出 1 处真实数值 bug（CE-logits 的 logsumexp 漏外层 m）+ 多处性质/容差错误（softplus 上界、gelu/mish 非全局单调、softmax 坐标单调、bce/ce 的 0·log0 保护等）。

### v15.3 新增（L10 微指令 F 域：归一化 96 项）

补齐八域中的 **F 域（归一化）**，微指令实现数 **480 → 576**，覆盖归一化族的全链路原子操作：

| 子域 | 指令 | 代表不变量 |
|---|---|---|
| **均值** | F001–F012 | 定义/反向累加一致、与顺序无关、线性、齐次、平移、矩阵按行·按列、极值界、与求和关系 |
| **方差与标准差** | F013–F024 | 总体方差两遍法/简化公式一致、平移不变、齐次（a²）、样本方差 Bessel 校正、标准差非负、切比雪夫样本界 |
| **L2 范数与单位化** | F025–F032 | 定义/反向累加一致、齐次（|a|）、三角不等式、非负、单位化长度=1、与缩放无关、保持方向 |
| **LayerNorm** | F033–F048 | 无仿射零均值、方差解析 Var/(Var+ε)、平移不变、仿射 γ·β 作用、矩阵逐行独立、退化情形、符号保持 |
| **RMSNorm** | F049–F064 | 定义/rms 部分/输出 rms、矩阵、与 LN 零均值相等、仿射 γ·β、符号保持、范数保持、与 LN 差异、平移改变输出 |
| **BatchNorm** | F065–F080 | 按列均值/方差/前向/零均值/单位方差/仿射/平移不变/跨样本耦合/与 LN 区别/常数通道→β/全零→β/转置 LN 对偶 |
| **GroupNorm/IN/WS** | F081–F089 | 定义/零均值/单位方差/仿射/G=1→InstanceNorm/G=D→β/与 BN 区别/权重标准化≡LN 按行 |
| **LRN 与经典归一化** | F090–F096 | LRN 定义/单调/有限、min-max/z-score/L1/L∞ 归一化 |

**关键的「非尺度齐次」纪律**：LayerNorm/RMSNorm/BatchNorm 的分母含数据相关方差，整体缩放后输出**不**满足简单齐次式。本批所有涉及缩放的不变量一律走「独立参考路径交叉验证」或「解析期望」（如 BatchNorm 单位方差=σ²/(σ²+ε)），绝不写伪齐性式——这是 F 域与 B/C 域最本质的数学差异。

> 修复：独立校验脚本在接入主链路前抓出 2 处 bug——F031 域误用 `nonZero`（含负，单位化方向翻转）改 `smallPos`；F083 predicate 错误复用输出均值当作输入均值，改为独立计算输入组均值。两处修复后经 4 种子 × 80 例（30720 次）抽样全过。

**不变量设计的两条纪律**（本批落地时被验证有效）：
1. **交叉验证**：每条 `predicate` 都走与 `run` 不同的参考路径（如 run 正向累加、predicate 反向累加），不搞「自证」。
2. **前置条件式放行**：涉及头数/长度合法性的项，inputs 不合法时显式放行并在 `invariant` 文本写明前置条件——不靠放宽容差制造假绿。

### v15.2 新增（L10 微指令 D 域：注意力基元 96 项）

补齐八域中的 **D 域（注意力基元）**，微指令实现数 **384 → 480**，覆盖注意力计算全链路的原子操作：

| 子域 | 指令 | 代表不变量 |
|---|---|---|
| **打分与缩放** | D001–D012 | 点积交换/齐次/分配、Cauchy–Schwarz 上界、自点积=‖q‖²、缩放除乘一致、温度单调、YaRN mscale 公式 |
| **掩码** | D013–D024 | 因果/滑动窗口/padding 构造正确、掩码幂等、窗口单调、与有限偏置可交换、安全 softmax 全掩回退均匀 |
| **softmax·LSE** | D025–D032 | 行和为一、平移不变、正值、安全版=朴素版、保序、lse 平移公式、softmax=exp(s−lse) |
| **RoPE** | D033–D044 | 逆频率公式、保范数、R(−p)∘R(p)=I、相对位置不变性、旋转复合、线性、逐对不交叉、det=1 |
| **ALiBi** | D045–D056 | 斜率几何级数、单调、偏置=−slope·|i−j|、对角为零、非正、对称、非 2 幂头数取最近幂 |
| **温度与信息量** | D057–D064 | 温度除乘一致、熵上界 log n、熵随温度单调、均匀分布取上界、CE=−log p[t]、CE≥0、KL≥0、JS 对称 |
| **GQA/MQA** | D065–D078 | q→kv 头映射、组大小、MQA/MHA 退化、值域合法、映射单调、覆盖性、重复 KV 逐位一致、拼接维度守恒 |
| **KV Cache** | D079–D096 | 追加递增、行读取一致、容量截断、滚动丢弃最旧非锚点、**sink 永不淘汰**、K/V 长度同步、清空、写不覆盖、行地址、INT8 往返硬界、长位置外推有限 |

同时为 D 域新增两个输入域标签 `vecMask`（掩码 0/1 向量）与 `vecProb`（归一化为概率单纯形）、`factor1`（外推因子 ≥1，避免 mscale 开负根）。

### v15.1 新增（十层分解架构实拆 + 浏览器双端同码）

引擎从「单体」拆成十层可验证积木，并把「双端一致」从口号变成**逐位可测**的硬门禁：

| 能力 | 说明 |
|---|---|
| **L3 架构组装** | `src/L3-arch/blocks.js` 重组装 forward/backward，替代 legacy 单体；模型壳仍走 legacy（构造器 RNG 顺序冻结），L3 仅覆写前/反向 → R2/R3/R4/R5 全部 Δ=0 |
| **L4 神经算子** | `src/L4-ops/*`：norm / activation / positional / softmax / linear / attention / moe / loss / registry 十件算子目录 |
| **L5 自动微分** | `src/L5-autograd/*`：VJP 注册中心 + tape 契约 + 梯度检验工厂（Path A） |
| **L6 后端运行时** | `src/L6-backend/*`：kernel 注册与派发 + CPU(JS) exact 通道 + profile |
| **L7 张量与内存** | `src/L7-tensor/*`：tensor / shape / arena（指针碰撞分配）/ kv-cache |
| **L8 数值内核** | `src/L8-kernel/*`：matmul / reduce / quantize —— 与 legacy **逐位一致**（Δ=0），R9 契约 |
| **L9 数值基础** | `src/L9-numbase/*`：rng（mulberry32 与 v11 同款）/ fp-ops / special / assert |
| **L10 微指令** | `src/L10-microisa/*`：**1024 槽位**注册中心（八域 × 128），**1024 项已实现**（八域各 128），每项七件套（直觉/数学/伪代码/复杂度/论文/常见坑/不变量）+ `run` + `predicate` |
| **浏览器双端同码** | `demo.html` 内联**完整十层栈**（mini-require 运行时 + 72 个模块），浏览器执行的就是 Node 侧同一份 `src/index.js` 装配代码 → 前向 logits 与 Node **逐位一致**（maxΔ=0，280 logits） |
| **不注水** | 未实现槽位显式 `status='stub'` 且 `run=null`；`npm run isa` 逐域报进度（A 128/128 · B 128/128 · C 128/128 · D 128/128 · E 128/128 · F 128/128 · G 128/128 · H 128/128 · 合计 1024/1024），不拿桩充数 |

> **修复（历史事故）**：v12.0–v14.0 期间 `build-demo.js` 误把 6 行 Node shim（`module.exports = require(...)`）内联进 `demo.html`，浏览器 classic script 执行到 `require` 即 `ReferenceError` —— 连续三个版本 demo 全坏却无人察觉（Node 侧测试全绿掩盖了浏览器断裂）。v15.1 三处根治：① 构建期硬校验（shim 特征检测 + `GPTModel` 存在性断言，不符即 exit 1）；② 改为内联完整十层栈，浏览器与 Node 同码；③ 新增 `scripts/probe-demo.js` 无头门禁，把该类事故变成 CI 可检测失败。

**v15.1 新增门禁**（`npm run test:all` 已纳入）：

| 门禁 | 命令 | 项数 |
|---|---|---|
| 微指令门禁 | `npm run test:L10` | 8 项全绿（1024 容量 · 1024 实现计数诚实且与 `isa.js` 交叉一致 · 1024 项 × 2 种子 × 20 例不变量抽查 · stub 诚实标记 · RNG 确定性 · L8/L9 与 legacy 逐位一致 · 引擎零影响 K15 34 键 · L3–L10 全挂载） |
| 浏览器装配门禁 | `npm run test:demo` | 9 项全绿（脚本块无异常 · 34 键一致 · L3–L10 挂载 · L10 容量双端一致 · 自检套件就位 · 微指令可运行 · **前向双端逐位一致** · 浏览器侧确定性） |
| 真实浏览器门禁 | `npm run test:demo-browser` | Playwright headless Chromium 直开 demo.html：十层栈装配 · L10 1024/1024 · 微指令浏览器 DOM/筛选/七件套 · 批量 1024×8 谓词全绿 · 产出截图 |
| 层 lint | `npm run lint` | `src/L*` 浏览器红线 0 违例 |
| 性质测试 | `npm run test:layers` | A/B/C/D/E/F/G/H 八域每项 ≥120 例 fast-check 任意子谓词检验 |
| ALiBi 门禁 | `npm run test:alibi` | **13 项全绿**（斜率闭式 ≡ μD045/μD053 · 偏置矩阵五性质 · 零 qkv 复原 `log(pⱼ/p₀)=slope·j` · 几何衰减 μD112 · 关闭时与 legacy Δ=0 · 有效性非摆设 · 双端逐位一致 4 场景（含 500 token 淘汰后与锚点+滚动）· 中心差分梯度检验 ≤1 · 微指令交叉验证） |

| 组件 | 复用来源 | 论文 |
|---|---|---|
| Decoder-only 结构（pre-LN + 残差 + 权重共享输出头） | GPT-2 / nanoGPT | [Radford et al., 2019](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) · [Karpathy, 2022](https://github.com/karpathy/nanoGPT) |
| 单文件极简实现风格 | picoGPT | [Larroy, 2023](https://github.com/jaymody/picoGPT) |
| AdamW 优化器（解耦权重衰减 + 梯度裁剪 + warmup + cosine 衰减） | AdamW | [Loshchilov & Hutter, 2019](https://arxiv.org/abs/1711.05101) |
| LayerNorm / GELU 激活 | BERT/GPT-2 标准件 | [Ba et al., 2016](https://arxiv.org/abs/1607.06450) · [Hendrycks & Gimpel, 2016](https://arxiv.org/abs/1606.08415) |

不借助 PyTorch/TensorFlow，反向传播的每一条梯度都**手写**，并接受**中心差分数值梯度检验**的拷问。

## 自检套件（12 项全绿）

`node src/verify.js` 一键运行，全部不变量独立可交叉验证：

| # | 检查项 | 实测结果 | 判据 |
|---|---|---|---|
| 1 | RNG 确定性与均匀性 | 同种子逐位一致，取值 [0,1) | mulberry32 可复现 |
| 2 | Softmax 行和=1 且全正 | 行和偏差 **2.22e-16**（机器精度） | =1 不变量 |
| 3 | LayerNorm 归一化不变量 | xhat 均值 8.33e-17，Y=g·xhat+b 偏差 **0** | 均值 0 / 方差 1（含 eps 衰减） |
| 4 | GELU 导数数值一致性 | 解析 vs 中心差分 **5.85e-10** | tanh 近似精确导数 |
| 5 | 因果掩码不变量 | 过去位置 logits 变化 **0**，上三角注意力=0 | 未来不可见 |
| 6 | **中心差分梯度检验** | 395 参数，最大 err/tol = **4.22e-4**（阈值 1.0） | 解析梯度 ≡ 数值梯度 |
| 7 | 过拟合：小语料可背熟 | loss **2.917 → 0.0011** | 容量证明，阈值 <0.1 |
| 8 | 随机批次训练收敛 | 前 30 步均值 2.463 → 后 30 步 0.036 | loss 单调下降 |
| 9 | 生成确定性与健壮性 | 同种子一致 ✓ 贪心 ✓ 120 字长 prompt ✓ 空 prompt ✓ | 边界全过 |
| 10 | Tokenizer 编解码往返 | 往返一致，未知字符跳过 | encode∘decode = id |
| 11 | **KV Cache 与全序列前向逐位一致** | 末位 logits 差异 **0**；10 步贪心逐位一致；40 步超长窗口容量恒定 | 增量推理 ≡ 全量推理 |
| 12 | 模型持久化 save/load 无损往返 | 参数差 **0**，logits 差 **0**，再序列化一致，词表恢复 | 存档可完整复现 |

> 检查 6 的 5 个跳过参数是 Q/K 偏置——softmax 的行移位不变性使其梯度**恒为零**（PyTorch 行为一致），此为数学事实而非缺陷。

**架构梯度检验扩展**：`scripts/probe-v3.js` 对 5 种架构组合（GPT-2、LLaMA、仅 RMSNorm、仅 SwiGLU、仅 RoPE）逐一做中心差分梯度检验，最大 err/tol 均 < 3e-3，`scripts/probe-v3.js` 同时验证 BPE 往返无损、RoPE 相对位置不变性（差异 1.33e-15）与 KV cache 逐位一致。

## 性能

KV Cache 增量解码是唯一有效的提速手段：每步只算 1 个新 token，历史 K/V 直接复用。

| 配置 | 生成 64 token | 生成 200 token | 吞吐 |
|---|---|---|---|
| 37 万参数 · blockSize 96 | **28.5x**（33ms vs 942ms） | **58.8x**（115ms vs 6759ms） | 1739 token/s |
| blockSize 256 · dModel 128 | **27.3x**（79ms vs 2153ms） | **95.3x**（223ms vs 21253ms） | 897 token/s |

浏览器端实测（12 万参数）：**80 token / 27ms = 2996 token/s**。

**两条被实测否决的优化路线**（不做无数据支撑的优化）：

| 路线 | 实测 | 结论 |
|---|---|---|
| 手写 4 路循环展开 | 32.7µs vs 30.2µs（更慢） | V8 已把朴素循环优化到标量极限 ~900M MAC/s |
| Int8 计算量化（llama.cpp 路线） | 0.92x（无加速） | JS 无 SIMD intrinsics，量化计算收益来自 C++ SIMD，JS 拿不到 |
| **INT8 权重量化（v5.0，仅权重压缩）** | **存档 ≈ 5.6x、内存 ≈ 6.9x** | weight-only PTQ：权重常驻 Int8 在矩阵乘内联解量化，**计算速度持平但体积/内存大降**，对解码无损（logits maxΔ<0.03） |

### 一键基准报告

`npm run bench` 聚合所有核心度量，自动生成 [`docs/BENCHMARK.md`](docs/BENCHMARK.md)（多架构梯度检验、RoPE 不变性、GQA、MoE、BPE、INT8 量化、KV cache 加速比）；`npm run bench:full` 追加**参数对齐全现代架构对照**——以 gpt2 参数量为基准、二分各预设 `ffnHidden` 把六套架构（GPT-2 / LLaMA / LLaMA-GQA / SWA / Gemma / Mixtral-MoE）拉到**等参数预算**，再同语料/同种子/同步数训练，横向比较 loss 与准确率。CI 默认跑快速版，并把报告作为 artifact 上传（每次运行可下载）。一键全探针：`npm run test:all`。

**等参数预算架构对照实测**（`bench:full`，约 12.4 万参数对齐，同语料/同种子/300 步）：

| 架构 | final loss | 准确率 |
|---|---|---|
| gpt2 | 1.910 | 34.9% |
| **llama** | 0.859 | **47.6%** |
| llama-gqa | 0.831 | 44.4% |
| swa | 0.831 | 44.4% |
| gemma | **0.802** | 44.4% |
| mixtral (MoE) | 1.260 | 38.1% |

结论：等参数预算下现代 RMSNorm+SwiGLU+RoPE 配方（llama 47.6%）显著优于 vanilla GPT-2（34.9%）；gemma 拿到最低 loss。swa 与 llama-gqa 逐位相同（window=64=blockSize 时滑窗退化为全注意力）——本身即一条正确性佐证。

## 快速开始

### Node（验证 + API）

```bash
git clone https://github.com/CJX0712/flagship-ai.git
cd flagship-ai
node src/verify.js   # 12 项自检，约 10 秒
```

10 行训练 + 生成：

```js
const F = require('./src/index.js');
const { CORPUS } = require('./src/corpus.js');

// 字符级分词（经典、中文友好，词表 ≈291）
const tok = new F.Tokenizer(CORPUS);
const model = new F.GPTModel({ vocabSize: tok.vocabSize,
  blockSize: 64, nLayer: 2, nHead: 4, dModel: 64, seed: 42 });

const r = F.train(model, tok.encode(CORPUS),
  { steps: 300, lr: 3e-3, batch: 32, warmup: 10, seed: 0 });
console.log('final loss:', r.finalLoss);                // ≈ 1.93

const out = F.generate(model, tok, '旗舰之道', {
  maxTokens: 100, temperature: 0.8, topK: 40, seed: 7 }); // KV cache 默认开启
console.log(out.text);

// ── v3.0：一行切到现代架构（RMSNorm + SwiGLU + RoPE）──
const llama = new F.GPTModel(F.applyPreset(
  { vocabSize: tok.vocabSize, blockSize: 64, nLayer: 2, nHead: 4, dModel: 64 }, 'llama'));

// ── v4.0：GQA 分组查询注意力（LLaMA-2/3 标准，KV 头数 < 注意力头数）──
const gqa = new F.GPTModel(F.applyPreset(
  { vocabSize: tok.vocabSize, blockSize: 64, nLayer: 2, nHead: 4, dModel: 64 }, 'llama-gqa')); // nKvHead=2
// 也可直接传 nKvHead（须满足 nHead % nKvHead === 0）；留空 = nKvHead=nHead 即 MHA
const gqa2 = new F.GPTModel({ vocabSize: tok.vocabSize, blockSize: 64, nLayer: 2,
  nHead: 4, nKvHead: 1, dModel: 64, norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope' });

// ── v7.0：MoE 混合专家（每层 N 个专家 + top-k 路由 + 负载均衡辅助损失）──
const moe = new F.GPTModel({ vocabSize: tok.vocabSize, blockSize: 64, nLayer: 2,
  nHead: 4, nKvHead: 2, dModel: 64, norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope',
  nExp: 4, topK: 2, auxWeight: 0.01 });                 // 也可 applyPreset(cfg,'mixtral')
const moeOut = F.generate(moe, tok, '旗舰', { maxTokens: 40, temperature: 0.9 });
// 每 token 只跑 2/4 个专家；训练时 aux 损失自动均衡专家负载

// ── v9.0：滑动窗口注意力（查询 i 只 attend 最近 window 个键，Mistral/Gemma）──
const swa = new F.GPTModel({ vocabSize: tok.vocabSize, blockSize: 64, nLayer: 2,
  nHead: 4, nKvHead: 2, dModel: 64, norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope',
  window: 32 });                                        // 也可 applyPreset(cfg,'swa')
const swaOut = F.generate(swa, tok, '旗舰', { maxTokens: 40, temperature: 0.9 });
// window=0（默认）= 全因果；RoPE 下 KV 缓存增量与全序列重算逐位一致

// ── v10.0：层交错稀疏注意力（Gemma 2/3：偶数层全局、奇数层滑动窗口）──
const gemma = new F.GPTModel({ vocabSize: tok.vocabSize, blockSize: 64, nLayer: 4,
  nHead: 4, nKvHead: 2, dModel: 64, norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope',
  window: { alt: 32 } });                                // 也可 applyPreset(cfg,'gemma')
// 逐层显式写法：window: [0, 32, 0, 32]（长度须 = nLayer）；model.windows 为归一化逐层数组

// ── v11.0：Gemma 2 稳定化组件（QK-norm + logit soft-cap）──
const gemma2 = new F.GPTModel({ vocabSize: tok.vocabSize, blockSize: 64, nLayer: 4,
  nHead: 4, nKvHead: 2, dModel: 64, norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope',
  window: { alt: 32 }, qkNorm: true, logitCap: 30 });    // gemma 预设已含这两项
// QK-norm：每头 Q/K  RMSNorm（RoPE 前）；logitCap：logits = 30·tanh(x/30)，输出严格有界

// ── v3.0：BPE 分词器（字符级基元，中文语料更省 token）──
const bpe = new F.BPETokenizer(CORPUS, 320, { byteLevel: false }); // 词表 317 · 往返无损
console.log('bpe tokens:', bpe.encode('旗舰 AI 锻造世界顶级模型').length);

// 持久化：参数 + 词表 一并存档，导入后可完整复现（BPE 亦无损）
const json = F.saveModel(model, tok);
const model2 = F.loadModel(json);
const tok2 = F.loadTokenizer(json);

// ── v5.0：INT8 权重量化推理（weight-only PTQ，内存/存档 ≈ 1/7，对解码无损）──
const mInt8 = model.quantize();                 // 只读导出，原 model 不受影响
const json8 = F.saveModel(mInt8, tok);           // 存档自动以 {q, scale} 编码
const m8 = F.loadModel(json8);                   // 载入即 INT8 常驻
console.log('量化存档体积比:', (json.length / json8.length).toFixed(2) + 'x'); // ≈ 5.6x
const out8 = F.generate(m8, tok, '旗舰', { maxTokens: 40, temperature: 0.9 });
```

### 浏览器（一键运行）

无构建、无服务器：**双击 `demo.html`**（file:// 协议直开）。单文件内联全部代码，支持：

- 📝 语料实时编辑 → 一键构建模型（词表/参数量即时统计）
- 🧪 一键自检（12 项逐项点亮，与 Node 结果一致）
- 🔬 **L10 微指令浏览器**（八域 A–H × 128 = 1024 项全貌；按域筛选 / 关键字搜索 / 展开七件套 / **浏览器内试运行不变量** / 批量 1024×8 谓词全绿）
- 📈 训练（分块不卡 UI，loss 实时曲线，断点续训与直训**逐位等价**）
- ✨ 生成（温度 / top-k / 种子可调，KV cache 加速，实时显示 token/s）
- 💾 导出/导入模型（参数 + 词表，JSON 无损）

**L10 微指令浏览器实测截图**（由 `npm run test:demo-browser` 在 headless Chromium 真实渲染 + 交互 + 批量试运行后生成，CI 每次运行上传同款 artifact）：

![Flagship AI — L10 微指令浏览器（1024/1024，批量试运行 8192 次谓词求值全绿）](docs/demo-screenshot.png)

## API 速查

| 导出 | 签名 | 说明 |
|---|---|---|
| `Tokenizer` | `new Tokenizer(text)` → `.encode(str)`, `.decode(ids)`, `.vocabSize` | 字符级，未知字符跳过 |
| `BPETokenizer` | `new BPETokenizer(text, vocabSize?, {minPairFreq, byteLevel})` | BPE 分词；`byteLevel:true`=256 字节基元（GPT-2 标准），`false`=字符级（中文更省 token）；`.encode`/`.decode`/`.vocabSize` 接口同 `Tokenizer` |
| `GPTModel` | `new GPTModel(cfg)` → `.forward(tokens)`, `.backward(cache, dlogits)`, `.params`, `.grads`, `.paramCount()`, `.evaluate(tokens)` | cfg: `vocabSize/blockSize/nLayer/nHead/dModel/seed` + 架构开关 `norm('ln'\|'rmsnorm')` / `ffn('gelu'\|'swiglu')` / `pos('learned'\|'rope')` / `ffnHidden` / `nKvHead`（KV 头数，默认=`nHead` 即 MHA；`<nHead` 启用 GQA） + MoE 开关 `nExp`（专家数，默认 1=稠密）/ `topK`（激活专家数，默认 1）/ `auxWeight`（负载均衡权重，MoE 默认 0.01） + 稀疏注意力开关 `window`（标量=所有层同窗；数组=逐层窗口，长度须=nLayer；`{alt:N}`=偶数层全局/奇数层窗口 N；默认 0=全因果） + 稳定化开关 `qkNorm`（Q/K 逐头 RMSNorm，RoPE 前，默认 false）/ `logitCap`（logit 软帽 `cap·tanh(x/cap)`，默认 0=不启用）/ `kvQuant`（KV 缓存 INT8 量化，8=启用；0=关闭，默认）/ `attentionSink`（Attention Sink 锚点数，K>0=恒驻首 K 个 token 支持无限上下文流式推理；0=关闭，默认）/ `rollingWindow`（滚动窗口大小，sink>0 时与 SWA `window` 同为滚动边界，默认 256）/ `ropeScale`（YaRN 长度外推：`{type:'yarn'|'ntk', factor>1, originalMaxLength}`；`type:'yarn'`=NTK-by-parts RoPE 频率插值 + 注意力温度 mscale，`'ntk'`=全局基频缩放；`factor≤1` 或省略=恒等） |
| `applyPreset` / `ARCH_PRESETS` | `applyPreset(cfg, 'gpt2'\|'llama'\|'llama-gqa'\|'mixtral'\|'swa'\|'gemma'\|'bloom')` | 一键套用预设：`gpt2={ln,gelu,learned}`、`llama={rmsnorm,swiglu,rope}`、`llama-gqa={rmsnorm,swiglu,rope,nKvHead:2}`、`mixtral={rmsnorm,swiglu,rope,nKvHead:2,nExp:4,topK:2,auxWeight:0.01}`、`swa={rmsnorm,swiglu,rope,nKvHead:2,window:64}`、`gemma={rmsnorm,swiglu,rope,nKvHead:2,window:{alt:64},qkNorm:true,logitCap:30}`、`bloom={ln,gelu,alibi:true}`（ALiBi 取代位置编码，无 posEmb / 无 RoPE）（preset 会覆盖同名 cfg 项，需改回请在其后再赋值） |
| `model.quantize()` | `model.quantize()` → 量化副本（含 `.q`/`.scale`） | INT8 weight-only PTQ：线性权重转 Int8 常驻（稠密 wqkv/wproj/w1/w2、MoE 路由 wg 与专家 e*.w1/w2、嵌入），推理时内联解量化；原模型不变；`saveModel` 自动编码为 `{q,s}` |
| `train` | `train(model, tokens, opts)` → `{history, finalLoss, state}` | opts: `steps/lr/batch/warmup/seed/wd/gradClip`；`state+stepOffset+totalSteps` 支持断点续训 |
| `generate` | `generate(model, tok, prompt, opts)` → `{text, ids}` | opts: `maxNew`(别名 `maxTokens`) / `temperature` / `topK` / `topP` / `seed` / `useCache` |
| `initKVCache` | `initKVCache(model)` → `{K, V, Ks, Vs, n, cap, sink}` | 每层 K/V 预分配 `cap×D`（cap = `attentionSink>0 ? K+R : blockSize`；KV INT8 时附 per-(token,head) `Ks/Vs` scale） |
| `forwardStep` | `forwardStep(model, tokenId, pos, kv)` → `{logits}` | 单 token 增量前向，O(T·D)；注意力含锚点 `[0,sink)` + 窗口 `[jLo,n)` |
| `kvShift` | `kvShift(model, kv)` | 缓存淘汰：sink>0 时恒驻锚点、仅平移窗口区；否则整体平移丢弃最旧 token（`copyWithin` 原生 memmove，K/V 与 INT8 scale 同步） |
| `saveModel` / `loadModel` / `loadTokenizer` | `saveModel(model, tok?)` → JSON | 参数与词表无损持久化（自动携带 tokenizer 类型与 BPE merges） |
| `globalThis.FlagshipAI`（层命名空间） | `FlagshipAI.L10.stats()` / `L10.ops[key].run(...)` / `L10.ops[key].predicate(...)` / `L10.display(key)` / `L9.rng.mulberry32(seed)` / `L8.matmul.matmul(...)` | 十层命名空间（Node：`require('src/index.js')` 后可读；浏览器：`window.FlagshipNS`）。**不影响** 34 键公共 API 导出清单（K15 契约） |

## 项目结构

```
flagship-ai/
├── src/
│   ├── index.js             # 十层装配入口：按 __loadOrder 加载 L10→L1→L0 并组装 34 键公共 API（L0-boot/assemble 为唯一映射真源）
│   ├── namespace.js         # 装配真源：__loadOrder + 域定义（Node 与浏览器共读）
│   ├── verify.js            # 12 项自检套件（Node + 浏览器双端）
│   ├── corpus.js            # 默认中文语料（可随意替换）
│   ├── legacy/              # v11 引擎本体（零回归 oracle，R1–R10 逐位对照）
│   ├── L10-microisa/        # 微指令层：1024 槽位注册中心 + 七件套微指令（1024 项已实现）
│   ├── L9-numbase/          # 数值基础：rng / fp-ops / special / assert
│   ├── L8-kernel/           # 数值内核：matmul / reduce / quantize（与 legacy 逐位一致）
│   ├── L7-tensor/           # 张量与内存：tensor / shape / arena / kv-cache
│   ├── L6-backend/          # 后端运行时：kernel 注册派发 / CPU(JS) exact 通道
│   ├── L5-autograd/         # 自动微分：VJP 注册中心 / tape 契约 / 梯度检验工厂
│   ├── L4-ops/              # 神经算子：norm/activation/positional/softmax/linear/attention/moe/loss
│   ├── L3-arch/             # 架构组装：forward/backward 重组装
│   ├── L2-pipeline/         # 编排管线：tokenizer/train/generate/sampler/evaluate/persistence/config
│   ├── L1-app/              # 应用层：demo-app（已接入 __loadOrder）
│   └── L0-boot/             # 引导装配：assemble（34 键映射 + K15 逐名门禁 + 注册中心自举，已接入）
├── tests/                   # node:test + fast-check 性质测试（L3–L10 各层）
├── scripts/probe-alibi.js   # v15.10 ALiBi 专项：斜率闭式 / 偏置矩阵 / 几何衰减 / 关闭 Δ=0 / 双端逐位（含淘汰后）/ 梯度检验 / 微指令交叉验证
├── scripts/probe-demo.js    # 浏览器装配门禁（9 项，含前向双端逐位一致）
├── scripts/build-demo.js    # 把十层栈+套件+应用内联进 demo.html（mini-require 双端同码）
├── scripts/isa.js           # 微指令覆盖率 CLI（逐域 impl/capacity 断言）
├── scripts/probe-v3.js      # v3 对照实验：BPE / 多架构梯度检验 / RoPE 不变性
├── scripts/probe-v4-gqa.js  # v4 验证：GQA 梯度检验 / KV 一致性 / 持久化 / 参数节省
├── scripts/probe-v5-quant.js # v5 验证：INT8 量化保真 / 存档体积 / in-RAM 压缩
├── scripts/probe-moe.js      # v7 验证：MoE 梯度检验 / 负载均衡 / 存档往返
├── scripts/probe-moe-quant.js # v8 验证：MoE × INT8 组合保真 / 压缩 / 不坍缩
├── scripts/probe-swa.js      # v9 验证：SWA 梯度检验 / 窗口局部性 / KV 一致性 / 零回归
├── scripts/probe-gemma.js    # v10 验证：层交错梯度检验 / 逐层生效 / alt 展开 / 存档量化往返
├── scripts/probe-gemma2.js   # v11 验证：QK-norm 恒等式+梯度 / soft-cap 有界+梯度 / 零回归
├── scripts/probe-kvquant.js # v12 验证：KV INT8 量化保真 / 误差硬界 / kvShift 同步 / 全组合 / 端到端生成
├── scripts/probe-streaming.js # v13 验证：Attention Sink / StreamingLLM 锚点不淘汰 / 滚动窗口 / ×KV-INT8 / ×SWA / 确定性
├── scripts/probe-yarn.js      # v14 验证：YaRN 频率插值 / mscale / 梯度检验 / 长上下文外推 / 零回归
├── scripts/probe-microisa.js # v15 门禁：L10 1024 槽位 / 1024 实现诚实计数 / 不变量抽查 / 引擎零影响
├── scripts/probe-demo.js     # v15 门禁：demo.html 浏览器装配无头验证 + 前向双端逐位一致
├── scripts/bench.js         # 统一基准报告：聚合度量 → docs/BENCHMARK.md（`npm run bench`）
├── demo.html                # 单文件浏览器演示（构建产物，双击即用）
├── docs/ARCHITECTURE.md     # 架构文档：公式 / 复杂度 / 反向传播推导 / 十层与双端装配（§9）
├── docs/BENCHMARK.md        # 自动生成的基准报告（bench.js 产出）
└── package.json
```

## 设计原则

1. **零依赖** — 不装任何包，`Float64Array` 手写全部矩阵运算
2. **可验证** — 每个算法至少 1 条硬不变量：softmax 行和、LN 归一化、因果掩码、数值梯度检验、过拟合收敛
3. **双端一致** — Node 与浏览器共享同一份代码，同一份验证
4. **教学友好** — 单文件可通读，注释标注论文出处，梯度推导见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## 已知限制

- 教学级规模（dModel 64 / 2 层 ≈ 12 万参数），面向学习与验证，非生产推理
- 纯 JS 单线程，无 GPU；已实测排除循环展开与 Int8 **计算**量化（V8 无 SIMD 收益），但 v5.0 的 INT8 **权重**量化（仅压缩）仍带来 ≈ 7x 体积/内存下降
- 量化后计算速度持平（无 SIMD 加速），收益在内存与序列化体积，对解码质量无损
- `pos:'learned'`（GPT-2 绝对位置嵌入）下，序列超出 blockSize 后转入**滑动窗口**（丢弃最旧 token），与"全量重算最后 blockSize 个 token"不再逐位等价 —— 这是 GPT-2 位置嵌入的固有约束
- `pos:'rope'`（RoPE）天然支持长度外推，滑动窗口下与全量推理**逐位一致**（注意力分数只依赖相对距离），更适合超长生成
- 字节级 BPE 在中文小语料下压缩率反而不如字符级（每汉字 3 字节需先"合并回来"）；中文场景建议 `byteLevel:false`

## 许可

[MIT](LICENSE) © 2026 晨星 (Chenxing)

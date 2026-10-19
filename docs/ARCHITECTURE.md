# Flagship AI 架构文档

> 零依赖纯 JS 手写 GPT：公式 · 复杂度 · 反向传播推导 · 参照实现对照
> 作者：晨星 · License: MIT

---

## 1. 分层总览

```
┌─────────────────────────────────────────────────────────┐
│ 应用层   app.js（语料编辑/训练/曲线/生成 UI）              │
├─────────────────────────────────────────────────────────┤
│ 生成层   generate：greedy / temperature / top-k / top-p  │
│          + KV Cache 增量解码（滑动窗口 memmove）          │
├─────────────────────────────────────────────────────────┤
│ 持久化   saveModel / loadModel / loadTokenizer（JSON）    │
├─────────────────────────────────────────────────────────┤
│ 优化层   AdamW(β1=0.9, β2=0.95) + 解耦权重衰减            │
│          + 全局梯度范数裁剪 + warmup + cosine 衰减        │
├─────────────────────────────────────────────────────────┤
│ 模型层   GPTModel：N×Block(Attn→残差, FFN→残差)           │
│          + 最终 LayerNorm + 权重共享输出头                │
├─────────────────────────────────────────────────────────┤
│ 算子层   matmul / softmax / LayerNorm / GELU / 交叉熵     │
│          （全部手写，Float64Array 行优先）                 │
├─────────────────────────────────────────────────────────┤
│ 数基层   mulberry32 RNG + Box-Muller 正态采样             │
└─────────────────────────────────────────────────────────┘
```

**数据流**：`text → Tokenizer.encode → GPTModel.forward → crossEntropy → GPTModel.backward → AdamW.step →（循环）→ generate`

---

## 2. 模型结构（GPT-2 标准）

### 2.1 前向计算

设词表大小 `V`，上下文长度 `T`，模型宽度 `D`，层数 `L`，头数 `H`（每头 `dh = D/H`）。

```
x₀ = TokEmb(t) + PosEmb(0..T-1)            # (T×D)，两者均 0.02·N(0,1) 初始化

重复 L 次（pre-LN 结构）：
  x ← x + MHA(LN₁(x))                      # 注意力块 + 残差
  x ← x + FFN(LN₂(x))                      # 前馈块 + 残差，FFN 隐层 4D

h  = LN_f(x)                               # 最终 LayerNorm
logits = h @ Wteᵀ                          # (T×V)，与输入嵌入共享权重（weight tying）
```

### 2.2 因果多头自注意力

单头内（第 `h` 头取 QKV 切片）：

```
scores[i][j] = (Q_i · K_j) / √dh           # 缩放点积
scores[i][j] = -∞   若 j > i               # 因果掩码：未来不可见
P = softmax(scores, axis=-1)
ctx_i = Σ_{j≤i} P[i][j] · V_j              # 只聚合过去与自身
```

多头并行后拼接投影回 `D`。**掩码不变量**（自检第 5 项）：修改位置 `t` 的输入，位置 `< t` 的 logits 必须逐位不变。

### 2.3 前馈网络（GPT-2 标准）

```
FFN(x) = GELU(x @ W₁ + b₁) @ W₂ + b₂      # D → 4D → D
```

GELU 采用 GPT-2 的 tanh 近似：

```
GELU(x) = 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715x³)))
```

### 2.4 LayerNorm（eps = 1e-5）

```
xhat = (x - μ) / √(σ² + eps)
y    = g ⊙ xhat + b
```

> 数值注意：因分母含 eps，输出方差理论值为 `var/(var+eps) < 1`。小方差行偏差可达 ~3e-6（自检第 3 项实测），这是**正确的数学行为**，判据按此放宽。

### 2.5 初始化（GPT-2 标准做法）

| 参数 | 初始化 |
|---|---|
| 嵌入 / 一般矩阵 | `N(0, 0.02²)` |
| 残差路径输出投影（attn.out / ffn.w2） | `N(0, (0.02/√(2L))²)` — 随深度缩放，保护残差流方差 |
| LayerNorm g/b | 1 / 0 |
| 偏置 | 0 |

### 2.6 权重共享输出头（weight tying）

`logits = h @ Wteᵀ`：输入嵌入与输出投影共享同一矩阵（GPT-1/2 标准做法），省 `V×D` 参数并提升小语料下的收敛。

### 2.7 模块化架构开关（v3.0）

归一化 / 前馈 / 位置编码三者解耦，可独立组合；`applyPreset` 提供两套已验证配方：

```
cfg.norm: 'ln'（LayerNorm，GPT-2）      | 'rmsnorm'（RMSNorm，LLaMA/Mistral）
cfg.ffn:  'gelu'（GPT-2）               | 'swiglu'（SiLU 门控，LLaMA/Mistral）
cfg.pos:  'learned'（可学习位置嵌入）    | 'rope'（旋转位置编码，LLaMA/Mistral）
cfg.ffnHidden: 默认 4·D；LLaMA 惯例用 ⌊2/3·4D⌋ 可显式传入
```

| 预设 | norm | ffn | pos | 备注 |
|---|---|---|---|---|
| `gpt2` | ln | gelu | learned | 项目默认，向后兼容 |
| `llama` | rmsnorm | swiglu | rope | 现代 LLM 标准配方 |

> 每个开关分支均**手写下反向**并经中心差分梯度检验（见 3.4 节）。

### 2.8 BPE 分词器（v3.0）

复刻 GPT-2 / tiktoken 的字节对编码，但基元可切换：

- **字节级**（`byteLevel:true`，GPT-2 标准）：初始词表 = 256 个字节（经 GPT-2 字节映射避免半截 UTF-8），覆盖任意 UTF-8。
- **字符级**（`byteLevel:false`）：初始词表 = 语料全部字符。中文语料下显著更省 token——每汉字 3 字节的 UTF-8 在字节级下必须先"合并回来"才不亏，`scripts/probe-v3.js` 实测中文语料字符级相对字节级 **1.14x** 压缩率。

```
算法：逐词块（正则预切分）统计相邻对频次 → 每轮合并最高频对 → 直到达到 vocabSize
编码：词块内按 merge-rank 贪心合并（等价 tiktoken 的 get_pairs 递归）
解码：字节级先拼全部基元再一次性 UTF-8 解码 → 跨 token 多字节汉字无损
```

不变量：`encode∘decode = id`（含中文 / emoji / 标点 / 跨 token 汉字），`scripts/probe-v3.js` 第 1 项逐例验证。

### 2.9 RMSNorm / SwiGLU / RoPE（v3.0）

**RMSNorm**（省去均值中心化，LLaMA）：

```
xhat = x / √(mean(x²) + eps)
y    = g ⊙ xhat          # 无偏置项 b
```

**SwiGLU**（SiLU 门控前馈，LLaMA）：

```
FFN(x) = (SiLU(x @ W₁ + b₁) ⊙ (x @ W₃ + b₃)) @ W₂ + b₂   # 隐层扩为 2·FH
SiLU(z) = z · σ(z) = z / (1 + e^{-z})
```

相比 GPT-2 的 `GELU → D×4D×D`，SwiGLU 用 `2·FH` 宽门控（`FH` 默认 4D，对齐时 ⌊8D/3⌋），参数略多但收敛更好（见 3.4 对照）。

**RoPE**（旋转位置编码，相对位置不变性）：

```
对每头 q,k 按奇偶维度配对 (x_{2i}, x_{2i+1})，绕原点旋转 θ_m·pos：
  [q̂_{2i}, q̂_{2i+1}] = R(θ_m · pos) · [q_{2i}, q_{2i+1}]
  R(φ) = [[cos φ, -sin φ], [sin φ, cos φ]]   θ_m = base^{-2m/D}
```

关键性质：**注意力分数只依赖相对距离**（旋转矩阵复合 `R(φ_j-φ_i)`）。因此：

- 天然支持长度外推（无需 learnable 位置嵌入）；
- 滑动窗口下 KV cache 与全量重算**逐位一致**（`scripts/probe-v3.js` 第 3 项：整体平移 5 位后分数最大差异 **1.33e-15**）；
- 反向即反向旋转（`ropeApply(..., -1)`），与正向共用同一组 θ。

---

## 3. 反向传播（全部手写）

### 3.1 损失与起点

交叉熵对 logits 的梯度（softmax 减 targets，`/n` 平均）：

```
∂L/∂logits_i = (softmax(logits_i) - onehot(y_i)) / n
```

> 工程细节：`crossEntropy` 对**全部 T 行**分配梯度并补零，仅前 `T-1` 行承载损失（下一 token 预测）——`backward` 按 T 行索引，行数不齐会越界读 undefined。

### 3.2 关键推导（自检第 6 项曾抓出的三处易错点）

**① 注意力 dK 的累加方向**。`ctx_i = Σ_{j≤i} P[i][j]·V_j` 对 K_j 的依赖经 `P[i][j]` 来自**所有未来位置 i ≥ j**：

```
dK_j = Σ_{i≥j} dS[i][j] · Q_i        # 不是 j≤i，方向相反
dQ_i = Σ_{j≤i} dS[i][j] · K_j
```

**② dV 按列累加**。`ctx[i] = Σ_j P[i][j]·V[j]` ⇒ `dV[j] = Σ_i P[i][j]·dctx[i]`（按列散列，不是按行）。

**③ 权重共享输出头**。`logits = h @ Wteᵀ` 的反向两条路都要走：

```
dWte += dlogitsᵀ @ h        # Wte 作为输出投影的梯度
dh   += dlogits @ Wte       # 流回主干再经 LN_f 反传
```

**Q/K 偏置梯度恒零**：softmax 行减去任意常数结果不变 ⇒ `b_q`、`b_k` 的解析梯度与数值梯度**同时恒为 0**（检验中显式跳过，与 PyTorch 行为一致）。

### 3.3 验证金标准

默认 GPT-2 配置下，395 个抽样参数做中心差分检验（ε=1e-5）：

```
num_i = (L(θ+εeᵢ) − L(θ−εeᵢ)) / 2ε
判据：|ana − num| ≤ atol + rtol·max(|ana|, |num|)，atol=1e-7, rtol=1e-4
实测最大 err/tol = 4.22e-4（GPT-2）
```

### 3.4 多架构梯度检验（v3.0）

`scripts/probe-v3.js` 对 5 种架构组合逐一做中心差分检验，最大 err/tol 均 < 3e-3，证明每个模块分支的反向都正确：

| 架构 | 检验分量（跳过零梯度） | 最大 err/tol |
|---|---|---|
| GPT-2（LN+GELU+learned） | 1243（跳 45） | 2.86e-3 |
| LLaMA（RMS+SwiGLU+RoPE） | 1068（跳 0） | 1.91e-3 |
| 仅 RMSNorm | 1083（跳 45） | 1.43e-3 |
| 仅 SwiGLU | 1243（跳 45） | 9.42e-4 |
| 仅 RoPE | 1228（跳 0） | 1.24e-3 |

**参数对齐架构对照**（同语料/同种子/同步数，小语料过拟合）：

| 配置 | 参数量 | loss | 准确率 |
|---|---|---|---|
| gpt2 | 124,480 | 1.910 | 34.9% |
| llama(4D) | 153,344 | 0.779 | 39.7% |
| llama(对齐 4D→FH=181) | 124,244 | 0.859 | **47.6%** |

结论：在**参数对齐**条件下，LLaMA 配方（RMSNorm+SwiGLU+RoPE）收敛更快、准确率更高——与业界"现代架构优于经典 GPT-2 配方"的共识一致。

---

## 4. 优化器与调度

### 4.1 AdamW（解耦权重衰减）

```
m ← β₁m + (1-β₁)g          β₁ = 0.9
v ← β₂v + (1-β₂)g²         β₂ = 0.95   # GPT-3 风格
m̂ = m/(1-β₁ᵗ),  v̂ = v/(1-β₂ᵗ)
θ ← θ - lr·(m̂/(√v̂+ε) + wd·θ)           # wd 不进动量——解耦
```

### 4.2 梯度裁剪与调度

- **全局范数裁剪**（clip=1.0）：`g ← g · min(1, clip/‖g‖₂)`，‖g‖₂ 为全部参数梯度平方和的根。
- **学习率**：线性 warmup（`step+1)/warmup`）→ cosine 衰减至峰值 10%。

### 4.3 断点续训（浏览器分块训练）

`state = { m, v, t, rngA }`（Adam 一二阶矩 + 时间步 + **批次采样 RNG 状态**）。续训传 `state + stepOffset + totalSteps`，实测 **5 块续训与直训 30 步逐位一致（参数差 0）**——关键在于 LR 调度必须按 `totalSteps` 计算，而非每块局部步数。

---

## 5. 生成（自回归采样）

```
prompt 编码 → 预热 KV cache → 取末位 logits
  temperature:  logits ← logits / T
  top-k:        保留概率最高的 k 个，其余置 -Infinity
  top-p:        按累积概率截断长尾（在 top-k 之后施加）
  softmax → 按分布采样 → 单步 forwardStep 推进缓存 → 重复
```

确定性：同种子 ⇒ 同输出（自检第 9 项）。`temperature <= 0` 时取 argmax 贪心解码。细节见 6.5 节。

---

## 6. 复杂度

| 环节 | 复杂度 | 说明 |
|---|---|---|
| 前向/反向 | `O(T·D² + T²·D)` | 矩阵乘主导 + 注意力 T² 项 |
| 训练单步 | `O(参数量)` | Adam 状态与参数同规模（m、v 各一份） |
| 生成（无缓存） | `O(N·T·D² + N·T²·D)` | 每步全序列重算 |
| **生成（KV cache）** | **`O(T²·D + N·T·D + N·D²)`** | 每步只算 1 个新 token，注意力仅 O(T·D) |
| 参数量 | `≈ 12(V+2D)D + L·(12D² + 2·4D²)` | 默认配置（V=291, D=64, L=2, H=4）实测 **122,816** |

---

## 6.5 KV Cache 增量推理

### 设计

每层维护 `K`、`V` 两个 `blockSize×D` 缓冲。单步只处理新 token：

```
x = TokEmb(t) + PosEmb(pos)
每层: q,k,v = LN₁(x)·Wqkv     → k,v 写入缓冲第 pos 行
      scores = q·K[0..pos] / √dh  → softmax → ctx = P·V[0..pos]
      ⋮ FFN
logits = LN_f(x)·Wteᵀ
```

历史 K/V 不重算 —— 它们携带的是写入时刻的绝对位置信息，后续步骤直接复用。

### 两个语义边界（必须明确）

| 场景 | 行为 | 与全量重算的关系 |
|---|---|---|
| 序列长度 < blockSize | 纯增量，零重算 | **严格逐位相等**（实测 logits 差异 0） |
| 序列长度 ≥ blockSize（pos='learned'） | 滑动窗口：丢弃最旧 token（`copyWithin` memmove），新 token 落在饱和位置 B-1 | **不再等价** —— 绝对位置嵌入的固有约束 |
| 序列长度 ≥ blockSize（pos='rope'） | 滑动窗口：丢弃最旧 token | **逐位一致** —— RoPE 注意力只依赖相对距离，窗口平移不改变分数 |

`pos='learned'` 第二条不可消除：GPT-2 的位置嵌入是绝对位置，窗口一旦滑动，剩余 token 的位置索引就变了，要严格等价必须全量重算。业界（llama.cpp / Mistral sliding window）同样采用近似。`pos='rope'` 则天然规避此问题——这也是现代架构更适合超长生成的原因。

### 一个被实测抓出的性能陷阱

首版滑窗策略：缓存满时**全量预热最近 B-1 个 token**。结果每 2 步就重建一次，200 token 生成耗时 3155ms（计算本身只需 ~70ms）—— 比不用缓存只快 2.19x，几乎白做。

根因是重建后 `pos = B-2`，只够走 1 步就又满。改为 **memmove 平移**（`copyWithin`，O(B·D)，无重算）后：

| 指标 | 修正前 | 修正后 |
|---|---|---|
| 200 token 生成（B=96, 37万参数） | 3155ms | **115ms** |
| 相对全量重算加速 | 2.19x | **58.8x** |
| 单 token 吞吐 | 63/s | **1739/s** |

教训：增量推理的收益完全取决于"每步固定开销 / 每 token 计算量"的比值。任何 O(B) 级别的重复计算都会吃掉全部收益。

### 两条被实测否决的优化路线

优化必须有数据支撑，以下两条经 micro-benchmark 否决，未进主代码：

| 路线 | 实测结果 | 否决原因 |
|---|---|---|
| 手写 4 路循环展开 | 32.70µs vs 朴素 30.20µs（**更慢**） | V8 已把 `i-p-j` 朴素循环优化到标量极限 ~900–1040M MAC/s，手工展开干扰了它的优化 |
| Int8 权重量化（llama.cpp 路线） | 0.92x（**无加速**） | JS 无法访问 SIMD intrinsics；Int8Array 加载还要符号扩展。量化的收益在 JS 里拿不到，反而丢精度 |

**JS 中 Float64 matmul 的真实天花板约 1 G MAC/s**（单核标量）。要突破只能靠 Wasm SIMD，那就不再是"零依赖单文件"了——本项目选择守住这个边界。

---

## 6.6 模型持久化

存档格式（JSON，无损双精度）：

```json
{ "format": "flagship-ai", "version": 2,
  "cfg": { "vocabSize": 291, "blockSize": 64, "nLayer": 2, "nHead": 4, "dModel": 64, "seed": 42,
           "norm": "ln", "ffn": "gelu", "pos": "learned", "ffnHidden": 256 },
  "tokKind": "char" | "bpe",
  "merges": [["a","b"], ...], "bpeByteLevel": true | null,
  "params": { "tokEmb": [...], "posEmb": [...], "l0.wqkv": [...], ... },
  "vocab": ["\n", "，", "一", ...] }
```

`vocab` 与分词器元数据（`tokKind` / `merges` / `bpeByteLevel`）必须一并存档：字符级/字节级分词器的字符→id 映射丢失后，参数再多也无法正确编解码。验证项 12 显式检查词表恢复（30 字符往返 encode/decode 一致）与 BPE merge 往返无损。


---

## 7. 参照实现对照

| 本项目 | 参照 | 一致点 |
|---|---|---|
| `GPTModel` 前向 | nanoGPT `model.py` | pre-LN、残差、因果掩码、权重共享、0.02 初始化与 1/√(2L) 缩放 |
| `train` | nanoGPT `train.py` | AdamW 超参、warmup+cosine、梯度裁剪 |
| `generate` | GPT-2 采样逻辑 | temperature / top-k / top-p、贪心 |
| 单文件风格 | picoGPT | 零依赖、可通读、公式即代码 |
| 验证方法论 | 各 forge 系列硬不变量 | 行和=1、正交/因果掩码、数值梯度检验、过拟合收敛 |

## 8. 为什么这些不变量是"硬金标准"

- **softmax 行和 = 1**：归一化定义，偏差 >1e-15 即实现有错（实测 2.22e-16）。
- **因果掩码**：语言模型的合法性根基，违反即信息泄漏，训练 loss 再低也是错的。
- **数值梯度检验**：对手写反传的**端到端拷问**，任何一条链式法则写错都会在此暴露（开发过程中它先后抓出 dK 方向反、dV 按行/按列错、共享头漏 dh 两处、dlogits 行数越界共 5 处真 bug）。
- **过拟合收敛**：loss 2.917→0.0011 证明模型容量与优化链路端到端无阻。
- **续训等价性**：分块 vs 直训逐位一致，证明状态管理（Adam 矩 + RNG + 调度）无泄漏。
- **KV cache 逐位一致**：增量推理与全量前向的末位 logits 差异必须为 **0**（不是"接近"）。这条把"缓存写错位置""漏拼接历史 K/V""掩码范围错"全部锁死。
- **存档无损**：参数、logits、词表三者往返后必须逐位相等 —— 存档格式任何隐式精度损失都会在此暴露。

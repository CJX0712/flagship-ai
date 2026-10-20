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

### 2.10 GQA 分组查询注意力（v4.0）

MHA 中 K/V 与 Q 同头数（H）。GQA 让 K/V 只用 `KV ≤ H` 组（LLaMA-2/3 标准），每组服务 `g = H/KV` 个 Q 头，显著降低 KV cache 显存与推理成本，是现代开源 LLM 的事实标配。

- **参数形状**：`wqkv` 输出列数从 `3·D` 变为 `D + 2·KV·dh`（Q 占 D，K/V 各占 `KV·dh`；`dh = D/H`）。`nKvHead` 缺省 = `nHead`（即 MHA，完全向后兼容）。
- **头映射**：Q 头 `h` 对应的 KV 头为 `kvh = ⌊h/g⌋`，须满足 `H % KV === 0`。
- **前向**：`K_j = qkv[j·stride + D + kvh·dh]`、`V_j = qkv[j·stride + D + KV·dh + kvh·dh]`（stride = qkvDim）。
- **反向要点（唯一易错点）**：多个 Q 头共享同一 KV 头，故 `dK/dV` 必须**累加**（`+=`）而非覆盖。该 bug 在 v4 开发中被梯度检验抓出（MHA 下 `KV=H` 每个头唯一，`+=` 退化为 `=`，故此前零回归）。
- **KV cache**：缓存的 K/V 维度同步变为 `KV·dh`（不再是 D），`initKVCache`/`kvShift` 已按 `model.KV·model.dh` 分配。
- **预设**：`llama-gqa = {rmsnorm, swiglu, rope, nKvHead:2}`；亦可显式传 `nKvHead` 任意合法值。
- **收益（实测，dModel=16/H=4）**：wqkv 参数 MHA=768 → GQA(nKvHead=2)=512；总参 7648 → 7104。
- **验证**：`scripts/probe-v4-gqa.js` —— GQA 中心差分梯度检验（err/tol=4.95e-4）、KV cache 逐步逐位一致、持久化无损往返、参数节省、过拟合可训练性，全绿。

### 2.11 INT8 权重量化推理（v5.0，weight-only PTQ）

对称逐张量量化（LLM.int8 / AWQ 谱系），仅压缩**权重**，激活仍走 Float64：

- **编码**：`scale = max|w| / 127`，`q = clamp(round(w/scale), -127, 127)`（Int8）。每个线性权重（wqkv/wproj/w1/w2/tokEmb/posEmb）独立一个 scale。
- **推理**：权重常驻 `Int8Array`，矩阵乘内联解量化 `w = q·scale`，即 `C = (A @ q)·scale`。偏置与归一化参数保持 Float64（占比极小）。
- **内存/体积**：线性权重由 Float64(8B) → Int8(1B) **恰好 8x**；整体（含偏置/归一化）实测 **6.89x**；模型存档（JSON）F32 888.7KB → INT8 157.9KB = **5.63x**。
- **精度**：纯舍入误差，对解码无损——logits `max|Δ| < 0.03`，生成文本与 F32 一致、有限无 NaN。
- **零回归保证**：`mm` / `mmAtB` 在权重带 `.q` 时走 `matmulQ` / `matmulAtBQ` 解量化路径，否则走原始 Float64 路径；F32 训练模型因无 `.q` 完全不受影响。`quantize()` 为**只读导出**，不改动原模型。
- **持久化兼容**：`saveModel` 对量化权重自动存 `{q, s}`；`loadModel` 还原为 `{q, scale}` 常驻 Int8，载入即量化推理。
- **诚实说明**：纯 JS 无 SIMD，解量化 matmul 的**计算速度持平**（不加速）；量化收益全部在内存与序列化体积——直接服务于"浏览器一键运行"的小体积部署。
- **验证**：`scripts/probe-v5-quant.js` —— 数值保真 / KV-cache 末位逐位一致 / 存档往返 / 体积压缩 / in-RAM 8x，11 项全绿。

### 2.12 MoE 混合专家（v7.0，Mixtral/DeepSeek 稀疏架构）

`nExp > 1` 时每层 FFN 替换为 **N 个独立专家** + **1 个 top-k 路由网络**；`nExp = 1` 即稠密（零回归）。

- **前向**：给定 FFN 输入 `M`（T×D，即 ln2 输出）
  - 路由：`logits = M·wg + bg`（T×N_exp），`probs = softmax(logits)`（over experts）
  - top-k：每 token 取概率最大的 `topK` 个专家及其门控值 `g = probs[i, e_sel]`
  - 专家：`ff[i] = Σ_k g_k · Expert[e_k](M[i])`，每个专家是独立的 SwiGLU/GELU 两层
  - 辅助负载均衡损失：`aux = N_exp·Σ_e f_e·P_e`（`f_e`=路由占比，`P_e`=平均门控概率）
- **反向**（关键点）：
  - 专家路径 = 标准 FFN 反向，但只对**被路由到该专家**的 token 子集累加梯度（`matmulTA` 作用在子集上）
  - 门控：`dout = dff·g`（门控是 out→ff 边的乘子）
  - 路由路径：`∂L/∂probs[i,e] = dot(dff[i], out_i^e)`（仅选中位置）→ `softmaxBwd` → `dwg/dbg`
  - **离散 top-k 选择不回传**（硬路由）：非选中专家的门控梯度恒为 0
  - **辅助损失梯度**：冻结 `f_e`（离散计数不回传），`∂aux/∂probs[i,e] = auxWeight·N_exp·count_e/T²`，叠加到 dprobs 后统一 `softmaxBwd`
- **两个不变量**：未选中专家梯度**恒为 0**（离散选择正确性）；在 top-k 选择稳定（大间隔路由）前提下，路由与专家权重中心差分梯度检验 err/tol ≈ **1e-3**（连续门控概率精确回传）
- **负载均衡实证**：坍缩场景 aux=3.59 → 启用 aux 损失均衡 aux=1.02（下界 1，坍缩上界 N_exp）
- **量化兼容**：路由 `wg` 与专家 `e*.w1/w2` 一并纳入 INT8 正则（权重体积同样 ≈ 1/8）
- **验证**：`scripts/probe-moe.js` —— CE/aux 双路径梯度检验、未选中专家零梯度、训练收敛、负载均衡、生成、存档往返，7 项全绿。

### 2.13 MoE × INT8 组合（v8.0）

把 v5 的 weight-only PTQ 用到 v7 的 MoE 模型上，稀疏专家同样获得 ≈ 1/8 权重体积：

- **量化范围**：`QUANT_LINEAR_RE` 覆盖稠密 `wqkv/wproj/w1/w2`、MoE **路由 `wg`** 与 **专家 `l{l}.e{e}.w1/w2`**、嵌入 `tokEmb/posEmb`；偏置 `bg/b*/ln` 保持 Float64（占比极小）。`mm/mmAtB` 按 `.q` 自动走解量化路径，专家子集 matmul 同样受益。
- **关键修复**：`quantize()` 用 `Object.create` 复制模型，v7.0 时漏拷 MoE 字段（`useMoE/nExp/topK/auxWeight`）→ 量化副本 `useMoE` 变 falsy，前向误走稠密路径取不存在的 `w1` 报错；v8.0 补全字段后组合闭环。
- **无损实证**：logits maxΔ<0.1；量化 KV-cache 末位与全序列前向 Δ=0；存档往返 Δ=0；in-RAM 线性权重恰好 8x、整体 6.90x；JSON 存档 5.67x；量化后路由不坍缩（最大专家占用 63%）。
- **验证**：`scripts/probe-moe-quant.js` —— 14 项全绿。

### 2.14 滑动窗口注意力 SWA（v9.0，Mistral/Gemma 稀疏注意力）

`window > 0` 时，查询位置 `i` **只 attend 键位置 `[i-window+1, i]`**（因果掩码 ⊇ 滑动窗口左边界）；`window = 0`（默认）即标准全因果，旧模型逐位兼容。

- **前向（T×T 路径）**：行 `i` 的分数掩码为 `S[i][j] = -1e9`（`j < i-window+1` 或 `j > i`）；`exp(-1e9 - max)` 在 float64 下溢为**精确 0.0**，窗外概率严格归零，`ctx` 累加随之只含窗口内项。单头注意力计算量从 O(T²) 降为 O(T·window)。
- **前向（单 token 增量路径）**：缓存槽位 `j` 对应绝对位置 `pos-(slot-j)`，窗口左边界即槽位 `slot-window+1`；只对窗口切片 `[jLo, nValid)` 计算分数与 softmax，与全序列重算**逐位一致**（RoPE 下注意力分数只依赖相对距离）。
- **反向（零新增代码）**：窗口是**静态掩码**，梯度经缓存 `Ps` 自动继承——`P[i][j]=0` 处 `dS=0`，`dQ/dK/dV` 的窗外贡献恒为 0。这是 SWA 选型的关键优势：固定掩码无离散决策，天然满足中心差分梯度检验。
- **感受野**：L 层、窗口 w 的网络，末位 token 的理论感受野半径 `(w-1)·L`——超出该距离的输入扰动对末位 logits **零影响**（探针实测逐位 0），窗内扰动必须可见。
- **量化与持久化**：`quantize()` 复制 `window` 字段；`cfg.window` 随 `saveModel` 存档，`loadModel` 重建后语义不变。
- **验证**：`scripts/probe-swa.js` —— 梯度检验（RoPE+SwiGLU+RMSNorm+GQA+window 全组合 err/tol=5.2e-4）、窗口局部性（深度锥语义）、KV 缓存逐步一致（maxΔ=0）、window=0 零回归、window≥T 等价全因果、过拟合训练、缓存/重算贪心解码一致、持久化+INT8 往返，8 项全绿。

### 2.15 层交错稀疏注意力（v10.0，Gemma 2/3 局部/全局模式）

v9 的 `window` 升级为**逐层**配置，构造时统一归一化为长度 `nLayer` 的数组 `model.windows`：

- **三种写法**：
  - 标量 `window: 64` → 所有层同窗 `[64,64,...]`（v9 行为，零回归）
  - 数组 `window: [0,64,0,64]` → 逐层显式窗口（长度须 = nLayer，否则抛错）
  - 哨兵 `window: {alt: 64}` → 偶数层全因果(0)、奇数层窗口 64（Gemma 2 的 1:1 局部/全局交错；随 nLayer 自动展开）
- **前向**：每层注意力取 `this.windows[l]` 作为该层窗口；`0` = 全局层（全因果），`>0` = 局部层（滑动窗口）。层间深度混合：局部层 O(T·w) 管局部依赖，全局层保长程信息。
- **反向**：仍是静态掩码——每层的 `Ps` 缓存自带该层窗口，梯度自动继承，零新增代码。
- **不变量**：混合窗口 `[4,0,8]` 梯度检验 err/tol=5.2e-4；逐层窗口独立生效（`[4,0]`/`[0,4]`/`[4,4]`/`[0,0]` 四配置 logits 两两互异）；数组全 0 ≡ 不传 window（逐位一致）；单层数组 ≡ 标量；KV 缓存逐步一致 maxΔ=0；存档（数组/哨兵均 JSON 可序列化）与 INT8 量化后 `windows` 无损。
- **验证**：`scripts/probe-gemma.js` —— 12 项全绿。

### 2.16 Gemma 2 稳定化组件：QK-norm + logit soft-cap（v11.0）

Gemma 2 在注意力 logits 与输出 logits 两端各加一道"限幅"，专门治大模型训练中的 logits 爆炸。两者都可关闭（默认关闭 = 逐位零回归）。

- **QK-norm**（`qkNorm: true`）：对每个注意力头的 Q/K 向量做 RMSNorm，**在 RoPE 之前**施加（旋转是正交变换，不改变向量范数，故顺序为 norm→rope）：

  ```
  ss = Σ_p x[p]² ; rstd = 1/√(ss/dh + eps) ; x̂ = x·rstd ; Y = g ⊙ x̂
  ```

  增益 `g` 按 head_dim 维定义、跨头共享（`l{l}.qg` / `l{l}.kg`，各 dh 维）。反向按 RMSNorm 通式：

  ```
  dx = rstd·(g⊙dY) − (x̂·(g⊙dY)/dh)·rstd²·x·rstd   // 详见 rmsnormBwd
  dg = Σ_{t,heads} x̂⊙dY
  ```

  **不变量**：归一后每头 RMS ≤ 1，且恒等式 `ss/dh + eps = 1/rstd²` 实测最大相对误差 **6.4e-16**（机器精度）。
- **logit soft-cap**（`logitCap: N`）：最终 logits 套 `lg = N·tanh(x/N)`，输出**严格有界** `|lg| ≤ N`，softmax 前无爆炸风险（Gemma 2 用 cap=30）。反向链式法则：

  ```
  dx = dlg ⊙ sech²(x/N) = dlg ⊙ (1 − tanh²(x/N)) = dlg ⊙ (1 − (lg/N)²)
  ```

  ⚠️ 注意 `(lg/N)` 是**已加帽后的值除以 cap**，不能再套一次 tanh——v11 首版误写成 `tanh(lg/cap)`（二次 tanh），soft-cap 梯度检验 err/tol = 3.69 直接抓出，修正后 6.5e-3。这是"梯度检验是硬金标准"的又一次实证。
- **生效范围**：`qkNorm` 同时接进 `forward`（T×T）与 `forwardStep`（单 token 缓存路径）；`logitCap` 接进 forward/forwardStep 输出与 `backward` 起点（乘 sech² 后再走输出头反向）。
- **不变量**：QK-norm 梯度检验 err/tol=1.4e-3（含增益 qg/kg）；soft-cap 梯度检验 6.5e-3；双开关关闭时逐位零回归（且不产生 qg/kg 参数）；无帽模型训练后 max|logit|=6.05，cap=1.5 后压到 1.46（削峰生效）；KV 缓存逐步一致 maxΔ=0；存档 + INT8 往返无损。
- **`gemma` 预设升级为完整 Gemma 2 配方**：`RMSNorm+SwiGLU+RoPE+GQA(nKvHead:2)+层交错(alt:64)+QK-norm+logitCap:30`。
- **验证**：`scripts/probe-gemma2.js` —— 10 项全绿。

---

### 2.17 KV Cache INT8 量化（v12.0，推理期显存压缩）

KV cache 是长序列推理的显存瓶颈（每层 K/V 各 `blockSize×KV×dh` 个 Float64）。v12 在缓存层引入**写入即量化**的 INT8 方案，与 v5 的权重 INT8 正交、可叠加（双重 PTQ）。

- **存储布局**：`kvQuant=8` 时 `initKVCache` 为每层分配 `Int8Array(B×kvDim)` 存码字 + `Float64Array(B×KV)` 存 per-(token,KV头) scale；`kvQuant=0`（默认）保持原 `Float64Array`，逐位兼容旧行为。
- **量化公式**（对称 absmax，每 token×每 KV 头独立）：

  ```
  s = max|x|/127 ; q = round(x/s) clamp[-127,127] ; x̂ = q·s
  ```

  误差硬界：`|x̂ − x| ≤ s/2 = max|x|/254`（相对误差 ≤ 1/254 ≈ 0.39%）；最大码字恰 ±127 ⇒ 解量化后 per-head absmax 相对误差 ≈ 1e-16（机器精度）。
- **零校准**：写入缓存的那一刻即可算出该 (token,head) 的 absmax，无需校准集、无需回填重标量，天然适配 `forwardStep` 的增量追加。
- **kvShift 同步**：滑动窗口平移时 scale 与码字同步 `copyWithin`（原生 memmove），否则解量化错位——这是该方案唯一需要额外 care 的点，已在 `kvShift` 内闭合。
- **组合兼容**：与 INT8 权重、SWA、层交错窗口、GQA、MoE、QK-norm、logit soft-cap 任意组合均经探针验证；`kvQuant=0` 逐位等价于 Float64 缓存（零回归）。
- **不变量**：INT8-KV vs F32-KV 逐步末位 logits maxΔ<5e-4（占 logits 动态范围 <0.02%）；贪心解码两模式生成 token 逐字一致；kvShift 后误差仍有界；端到端采样无 NaN/Inf、同种子可复现；典型 L=4/B=64/KV头=2/dh=16 下显存 **5.33x** 压缩。
- **验证**：`scripts/probe-kvquant.js` —— 11 项全绿。

### 2.18 Attention Sink / StreamingLLM（v13.0，推理期无限上下文）

长文本生成时 KV 缓存随序列线性膨胀，且注意力会无谓地把大量权重浪费在最开头 token 上（即使它们已与当前语义脱钩）。v13 引入 StreamingLLM（Xiao et al., 2023）的两条机制，使缓存容量与序列长度**解耦**：

1. **Attention Sink（锚点）**：固定缓存最前的 K 个 token 永不淘汰——经验与理论都表明，注意力需要若干"锚点"来稳定 softmax 的分母（mass），丢弃首 token 会导致 perplexity 暴涨、生成崩坏。
2. **滚动窗口**：其余缓存为最近 R 个 token，超出则淘汰最旧的非锚点 token。

- **缓存容量**：`cap = sink>0 ? K + R : blockSize`（R = SWA `window`（若设）或 `rollingWindow`，默认 256）。`initKVCache` 据此分配 `cap×kvDim`，并附带 `cap`（容量）与 `sink`（锚点数）字段供增量路径使用。
- **淘汰原语 `kvShift` 的锚点感知**：

  ```
  sink>0：仅平移窗口区 [sink+1, n) → [sink, n-1)（copyWithin 原生 memmove），锚点 [0,sink) 原地不动；n -= 1
  sink=0：整体平移 [1, n) → [0, n-1)（旧滑动窗口行为，逐位兼容）
  K/V 与（kvQuant=8 时）scale 同步平移，解量化不错位
  ```

- **注意力公式（增量路径）**：查询在位置 `pos` 的注意力范围为 `sink 锚点 [0, sink) ∩ 窗口外` ∪ `窗口 [jLo, n)`（`jLo = nValid − win`，`win = min(windows[l], nValid)`）；`sinkCount = min(sink, jLo)` 为需显式补入的窗口外锚点数。`sink=0` 时 `sinkCount=0`，与旧窗口注意力**逐位一致**（零回归）。缓存中实际驻留的正是 `sink + 窗口` 两段，中段 token 已被淘汰，故"锚点 ∪ 窗口"恰好覆盖全部有效槽位。
- **RoPE 兼容性**：缓存存的是写入时已按绝对位置 `pos` 旋转过的 K/V；`kvShift` 平移不改其数值，新查询按当前 `pos` 旋转，注意力只依赖相对距离——与 v9/v12 的滑动窗口结论一致。
- **零干预训练**：StreamingLLM 仅作用于 `forwardStep` / `kvShift`，`forward` 全序列训练路径完全不变，**反向传播与中心差分梯度检验零改动**（训练本就不走 KV 缓存）。
- **组合兼容**：与 KV INT8 量化、SWA 滑动窗口、INT8 权重量化、GQA、MoE、QK-norm、logit soft-cap 任意组合均经探针验证；`attentionSink=0`（默认）逐位等价于旧版滑动窗口。
- **不变量**：远超容量后 `slot[0..K)` 的 K/V 与初次写入**逐位一致**（锚点永不被淘汰，Δ=0）；末窗口槽位 == 最近一次写入 token（Δ=0）；StreamingLLM 末位 logits 与纯滑动窗口**明显不同**（maxΔ=2.5e-2，证明锚点确实被 attend）；greedy 300 token 同种子可复现、无 NaN/Inf；× KV INT8 末位 maxΔ<2e-4；kvShift 多次淘汰后 scale 全有限。
- **验证**：`scripts/probe-streaming.js` —— 10 项全绿（零回归 / 短序列一致性 / 锚点不淘汰 / 滚动窗口 / 有效性 / 确定性 / 无限上下文 / ×KV-INT8 / ×SWA / scale 同步）。

### 2.19 YaRN 长度外推（v14.0，RoPE 频率插值 + 注意力温度）

RoPE 的旋转频率 `θ_i = base^(-2i/d)` 对位置敏感：短训练长度下拟合良好的模型，在远超训练长度的绝对位置上会因高频维相位回绕而失效。v14 引入 YaRN（Peng et al., 2023）的两件套，使模型可在**比训练更长**的序列上推理，且与 v9/v12/v13 的长上下文机制正交、可叠加。

1. **NTK-by-parts RoPE 频率插值**：按修正区间（correction range）对每维频率做平滑插值——高频维（小 `i`，波长最短、对位置最敏感）基本保留，低频维（大 `i`）插值压缩。等效于逐维缩放 `inv_freq_i /= factor^mask_i`，`mask_i∈[0,1]` 由 `low/high` 修正区间与 smoothstep 决定（`type:'ntk'` 为全局基频缩放 `factor^(d/(d-2))` 的特例）。
2. **注意力温度（mscale）**：softmax 前 logits 乘 `mscale = 0.1·ln(factor)/ln(2)`，补偿频率插值带来的幅度变化（`factor≤1` 时取 1，恒等）。

- **配置**：`cfg.ropeScale = { type:'yarn'|'ntk', factor(>1), originalMaxLength(默认 blockSize), betaFast(默认32), betaSlow(默认1) }`；缺省或 `factor≤1` ⇒ `ropeFreqScale=null`、`attentionScaling=1`，RoPE 与旧版**逐位一致**（零回归）。
- **实现契合点**：
  - `ropeApply` 新增 `freqScale` 参数（位置无关），前后向共用同一 `model.ropeFreqScale`，保证旋转雅可比一致；
  - 注意力 logits 缩放因子 `attentionScaling` 前后向共用（线性推送，softmax 前乘、softmaxBwd 后乘），链式一致；
  - 增量路径 `forwardStep` 同步使用 `model.ropeFreqScale` / `model.attentionScaling`，故 `forward` 与 `forwardStep` 在 YaRN 下**逐位一致**。
- **零干预训练**：仅改变 RoPE 旋转频率与注意力 logits 幅度（均为确定性、位置无关/线性因子），`forward` 全序列训练 + 反向传播 + 中心差分梯度检验**零改动**（`factor>1` 下 maxErr/Tol≈2e-3）。
- **组合兼容**：与 KV INT8、SWA、Attention Sink/StreamingLLM、INT8 权重、GQA、MoE、QK-norm、logit soft-cap 任意组合均经探针验证。
- **不变量**：`factor=1` 与缺省逐位等价（Δ=0）；`forward`≡`forwardStep`（Δ=0）；`factor>1` 中心差分梯度检验 maxErr/Tol≤1；`mscale` 公式精确（`0.1·ln(4)/ln(2)=0.2`）；`freqScale∈[1/factor,1]` 且非递减；`pos≫blockSize`（如 3000）输出有限且同输入确定性 Δ=0。
- **验证**：`scripts/probe-yarn.js` —— 5 项全绿（零回归 / 梯度检验 yarn+ntk / mscale+freqScale / 长上下文外推 / 确定性+特征生效）。

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
| Int8 **计算**量化（llama.cpp 路线） | 0.92x（**无加速**） | JS 无法访问 SIMD intrinsics；Int8Array 加载还要符号扩展。量化**计算**的收益在 JS 里拿不到 |
| INT8 **权重**量化（v5.0，仅压缩） | 存档 ≈ 5.6x、内存 ≈ 6.9x（**保真**） | weight-only PTQ：权重常驻 Int8 在 matmul 内联解量化，计算速度持平，但体积/内存大幅下降，对解码无损 |

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
| MoE 稀疏 FFN + top-k 路由 | Mixtral / DeepSeek / Switch Transformer | 每层 N 专家 + 1 路由、top-k 门控、`N·Σ f_e·P_e` 负载均衡辅助损失（冻结 f_e） |
| 滑动窗口注意力 | Mistral 7B / Gemma | 查询 i 仅 attend `[i-window+1, i]`，窗外掩码 -1e9（softmax 后精确 0）；固定掩码 → 反向经缓存 Ps 自动继承 |
| 局部/全局层交错 | Gemma 2/3 | `window` 逐层配置（标量/数组/`{alt:N}` 哨兵），偶数层全局 + 奇数层滑动窗口的 1:1 交错；归一化为 `model.windows` |
| QK-norm | Gemma 2/3、Chameleon、ViT-22B | 每头 Q/K 做 RMSNorm（RoPE 前），增益按 head_dim 跨头共享；稳定注意力 logits 尺度 |
| logit soft-cap | Gemma 2（cap=30） | `N·tanh(x/N)` 严格有界输出；反向 `dlg ⊙ (1 − (lg/N)²)` |
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

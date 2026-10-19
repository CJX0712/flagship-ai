<div align="center">

# 🚢 Flagship AI

**零依赖纯 JavaScript 手写 GPT 引擎 — 从字符到生成，每一行都可验证**

[![Version](https://img.shields.io/badge/version-v3.0-blue)](#)
[![Tests](https://img.shields.io/badge/tests-12%2F12%20passing-brightgreen)](#-自检套件-12-项全绿)
[![CI](https://github.com/CJX0712/flagship-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/CJX0712/flagship-ai/actions/workflows/ci.yml)
[![Dependencies](https://img.shields.io/badge/dependencies-0-success)](#)
[![Node](https://img.shields.io/badge/node-%E2%89%A514-blue)](https://nodejs.org)
[![Browser](https://img.shields.io/badge/browser-file%3A%2F%2F%20direct-orange)](#-浏览器一键运行)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)
[![Author](https://img.shields.io/badge/author-%E6%99%A8%E6%98%9F-purple)](https://github.com/CJX0712)

*Char/BPE Tokenizer · Decoder-only Transformer · GPT-2 ⇄ LLaMA 架构可切换 · Hand-written Backpropagation · AdamW · KV Cache · Temperature/top-k/top-p Sampling*

</div>

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
| Int8 权重量化（llama.cpp 路线） | 0.92x（无加速） | JS 无 SIMD intrinsics，量化收益来自 C++ SIMD，JS 拿不到 |


## 快速开始

### Node（验证 + API）

```bash
git clone https://github.com/CJX0712/flagship-ai.git
cd flagship-ai
node src/verify.js   # 10 项自检，约 10 秒
```

10 行训练 + 生成：

```js
const F = require('./src/flagship-engine.js');
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

// ── v3.0：BPE 分词器（字符级基元，中文语料更省 token）──
const bpe = new F.BPETokenizer(CORPUS, 320, { byteLevel: false }); // 词表 317 · 往返无损
console.log('bpe tokens:', bpe.encode('旗舰 AI 锻造世界顶级模型').length);

// 持久化：参数 + 词表 一并存档，导入后可完整复现（BPE 亦无损）
const json = F.saveModel(model, tok);
const model2 = F.loadModel(json);
const tok2 = F.loadTokenizer(json);
```

### 浏览器（一键运行）

无构建、无服务器：**双击 `demo.html`**（file:// 协议直开）。单文件内联全部代码，支持：

- 📝 语料实时编辑 → 一键构建模型（词表/参数量即时统计）
- 🧪 一键自检（10 项逐项点亮，与 Node 结果一致）
- 📈 训练（分块不卡 UI，loss 实时曲线，断点续训与直训**逐位等价**）
- ✨ 生成（温度 / top-k / 种子可调，KV cache 加速，实时显示 token/s）
- 💾 导出/导入模型（参数 + 词表，JSON 无损）

## API 速查

| 导出 | 签名 | 说明 |
|---|---|---|
| `Tokenizer` | `new Tokenizer(text)` → `.encode(str)`, `.decode(ids)`, `.vocabSize` | 字符级，未知字符跳过 |
| `BPETokenizer` | `new BPETokenizer(text, vocabSize?, {minPairFreq, byteLevel})` | BPE 分词；`byteLevel:true`=256 字节基元（GPT-2 标准），`false`=字符级（中文更省 token）；`.encode`/`.decode`/`.vocabSize` 接口同 `Tokenizer` |
| `GPTModel` | `new GPTModel(cfg)` → `.forward(tokens)`, `.backward(cache, dlogits)`, `.params`, `.grads`, `.paramCount()`, `.evaluate(tokens)` | cfg: `vocabSize/blockSize/nLayer/nHead/dModel/seed` + 架构开关 `norm('ln'\|'rmsnorm')` / `ffn('gelu'\|'swiglu')` / `pos('learned'\|'rope')` / `ffnHidden` |
| `applyPreset` / `ARCH_PRESETS` | `applyPreset(cfg, 'gpt2'\|'llama')` | 一键套用预设：`gpt2={ln,gelu,learned}`、`llama={rmsnorm,swiglu,rope}` |
| `train` | `train(model, tokens, opts)` → `{history, finalLoss, state}` | opts: `steps/lr/batch/warmup/seed/wd/gradClip`；`state+stepOffset+totalSteps` 支持断点续训 |
| `generate` | `generate(model, tok, prompt, opts)` → `{text, ids}` | opts: `maxNew`(别名 `maxTokens`) / `temperature` / `topK` / `topP` / `seed` / `useCache` |
| `initKVCache` | `initKVCache(model)` → `{K, V, n}` | 每层 K/V 预分配 `blockSize×D` |
| `forwardStep` | `forwardStep(model, tokenId, pos, kv)` → `{logits}` | 单 token 增量前向，O(T·D) |
| `kvShift` | `kvShift(model, kv)` | 滑动窗口：丢弃最旧 token（`copyWithin` 原生 memmove） |
| `saveModel` / `loadModel` / `loadTokenizer` | `saveModel(model, tok?)` → JSON | 参数与词表无损持久化（自动携带 tokenizer 类型与 BPE merges） |

## 项目结构

```
flagship-ai/
├── src/
│   ├── flagship-engine.js   # 核心引擎（唯一必需文件，零依赖）
│   ├── verify.js            # 12 项自检套件（Node + 浏览器双端）
│   ├── corpus.js            # 默认中文语料（可随意替换）
│   └── app.js               # demo 应用逻辑（字符级 / BPE / 架构选择）
├── scripts/build-demo.js    # 把引擎+套件+应用内联进 demo.html
├── scripts/probe-v3.js      # v3 对照实验：BPE / 多架构梯度检验 / RoPE 不变性
├── demo.html                # 单文件浏览器演示（构建产物，双击即用）
├── docs/ARCHITECTURE.md     # 架构文档：公式 / 复杂度 / 反向传播推导
└── package.json
```

## 设计原则

1. **零依赖** — 不装任何包，`Float64Array` 手写全部矩阵运算
2. **可验证** — 每个算法至少 1 条硬不变量：softmax 行和、LN 归一化、因果掩码、数值梯度检验、过拟合收敛
3. **双端一致** — Node 与浏览器共享同一份代码，同一份验证
4. **教学友好** — 单文件可通读，注释标注论文出处，梯度推导见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## 已知限制

- 教学级规模（dModel 64 / 2 层 ≈ 12 万参数），面向学习与验证，非生产推理
- 纯 JS 单线程，无 GPU；已实测排除循环展开与 Int8 量化（V8 无 SIMD 收益）
- `pos:'learned'`（GPT-2 绝对位置嵌入）下，序列超出 blockSize 后转入**滑动窗口**（丢弃最旧 token），与"全量重算最后 blockSize 个 token"不再逐位等价 —— 这是 GPT-2 位置嵌入的固有约束
- `pos:'rope'`（RoPE）天然支持长度外推，滑动窗口下与全量推理**逐位一致**（注意力分数只依赖相对距离），更适合超长生成
- 字节级 BPE 在中文小语料下压缩率反而不如字符级（每汉字 3 字节需先"合并回来"）；中文场景建议 `byteLevel:false`

## 许可

[MIT](LICENSE) © 2026 晨星 (Chenxing)

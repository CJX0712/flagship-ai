<div align="center">

# 🚢 Flagship AI

**零依赖纯 JavaScript 手写 GPT 引擎 — 从字符到生成，每一行都可验证**

[![Tests](https://img.shields.io/badge/tests-10%2F10%20passing-brightgreen)](#-自检套件-10-项全绿)
[![Dependencies](https://img.shields.io/badge/dependencies-0-success)](#)
[![Node](https://img.shields.io/badge/node-%E2%89%A514-blue)](https://nodejs.org)
[![Browser](https://img.shields.io/badge/browser-file%3A%2F%2F%20direct-orange)](#-浏览器一键运行)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)
[![Author](https://img.shields.io/badge/author-%E6%99%A8%E6%98%9F-purple)](https://github.com/CJX0712)

*Char-level Tokenizer · Decoder-only Transformer · Hand-written Backpropagation · AdamW · Temperature/top-k Sampling*

</div>

---

## 这是什么

一个 **约 900 行、零依赖、单文件** 的 GPT 语言模型完整实现：分词 → 前向 → 手写反向传播 → AdamW → 采样生成，Node 与浏览器双端运行。

**设计信条：零重写，集大成。** 架构不发明，直接站在世界顶级成果之上——

| 组件 | 复用来源 | 论文 |
|---|---|---|
| Decoder-only 结构（pre-LN + 残差 + 权重共享输出头） | GPT-2 / nanoGPT | [Radford et al., 2019](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) · [Karpathy, 2022](https://github.com/karpathy/nanoGPT) |
| 单文件极简实现风格 | picoGPT | [Larroy, 2023](https://github.com/jaymody/picoGPT) |
| AdamW 优化器（解耦权重衰减 + 梯度裁剪 + warmup + cosine 衰减） | AdamW | [Loshchilov & Hutter, 2019](https://arxiv.org/abs/1711.05101) |
| LayerNorm / GELU 激活 | BERT/GPT-2 标准件 | [Ba et al., 2016](https://arxiv.org/abs/1607.06450) · [Hendrycks & Gimpel, 2016](https://arxiv.org/abs/1606.08415) |

不借助 PyTorch/TensorFlow，反向传播的每一条梯度都**手写**，并接受**中心差分数值梯度检验**的拷问。

## 自检套件（10 项全绿）

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

> 检查 6 的 5 个跳过参数是 Q/K 偏置——softmax 的行移位不变性使其梯度**恒为零**（PyTorch 行为一致），此为数学事实而非缺陷。

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

const tok = new F.Tokenizer(CORPUS);                    // 字符级分词（词表 291）
const model = new F.GPTModel({ vocabSize: tok.vocabSize,
  blockSize: 64, nLayer: 2, nHead: 4, dModel: 64, seed: 42 });

const r = F.train(model, tok.encode(CORPUS),
  { steps: 300, lr: 3e-3, batch: 32, warmup: 10, seed: 0 });
console.log('final loss:', r.finalLoss);                // ≈ 1.93

const out = F.generate(model, tok, '旗舰之道', {
  maxTokens: 100, temperature: 0.8, topK: 40, seed: 7 });
console.log(out.text);
```

### 浏览器（一键运行）

无构建、无服务器：**双击 `demo.html`**（file:// 协议直开）。单文件内联全部代码，支持：

- 📝 语料实时编辑 → 一键构建模型（词表/参数量即时统计）
- 🧪 一键自检（10 项逐项点亮，与 Node 结果一致）
- 📈 训练（分块不卡 UI，loss 实时曲线，断点续训与直训**逐位等价**）
- ✨ 生成（温度 / top-k / 种子可调）

## API 速查

| 导出 | 签名 | 说明 |
|---|---|---|
| `Tokenizer` | `new Tokenizer(text)` → `.encode(str)`, `.decode(ids)`, `.vocabSize` | 字符级，未知字符跳过 |
| `GPTModel` | `new GPTModel(cfg)` → `.forward(tokens)`, `.backward(cache, dlogits)`, `.params`, `.grads` | cfg: `vocabSize/blockSize/nLayer/nHead/dModel/seed` |
| `train` | `train(model, tokens, opts)` → `{history, finalLoss, state}` | opts: `steps/lr/batch/warmup/seed/wd/gradClip`；`state+stepOffset+totalSteps` 支持断点续训 |
| `generate` | `generate(model, tok, prompt, opts)` → `{text}` | opts: `maxTokens/temperature/topK/topP/seed/greedy` |

## 项目结构

```
flagship-ai/
├── src/
│   ├── flagship-engine.js   # 核心引擎（唯一必需文件，零依赖）
│   ├── verify.js            # 10 项自检套件（Node + 浏览器双端）
│   ├── corpus.js            # 默认中文语料（可随意替换）
│   └── app.js               # demo 应用逻辑
├── scripts/build-demo.js    # 把引擎+套件+应用内联进 demo.html
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
- 字符级 tokenizer，无 BPE（BPE 属工程加速，不改变架构本质）
- 纯 JS 单线程，无 GPU 加速

## 许可

[MIT](LICENSE) © 2026 晨星 (Chenxing)

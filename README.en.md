<div align="center">

# 🚢 Flagship AI

**A zero-dependency, single-file GPT engine in pure JavaScript — from characters to generation, every line verifiable**

[![Version](https://img.shields.io/badge/version-v15.10-blue)](#)
[![Self-check](https://img.shields.io/badge/selfcheck-12%2F12-brightgreen)](#verification)
[![Tests](https://img.shields.io/badge/tests-191%2F191%20passing-brightgreen)](#verification)
[![ISA](https://img.shields.io/badge/ISA-1024%2F1024-blue)](#capabilities)
[![Architecture](https://img.shields.io/badge/architecture-L1%E2%86%92L10%20ten--layer-blueviolet)](#project-structure)
[![Regression](https://img.shields.io/badge/regression-R1%E2%80%93R10%20%CE%94%3D0-brightgreen)](#verification)
[![CI](https://github.com/CJX0712/flagship-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/CJX0712/flagship-ai/actions/workflows/ci.yml)
[![Dependencies](https://img.shields.io/badge/dependencies-0-success)](#)
[![Node](https://img.shields.io/badge/node-%E2%89%A514-blue)](https://nodejs.org)
[![Browser](https://img.shields.io/badge/browser-file%3A%2F%2F%20direct-orange)](#browser-one-double-click)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)
[![Author](https://img.shields.io/badge/author-Chenxing-purple)](https://github.com/CJX0712)

*Char/BPE Tokenizer · Decoder-only Transformer · switchable GPT-2 ⇄ LLaMA ⇄ LLaMA-GQA ⇄ Mixtral-MoE ⇄ SWA ⇄ Gemma · INT8 weight-only quantized inference · MoE (mixture-of-experts) · sliding-window / interleaved sparse attention · QK-norm / logit soft-cap · Attention Sink / StreamingLLM infinite context · YaRN length extrapolation · hand-written backpropagation · AdamW · KV cache (incl. INT8) · temperature / top-k / top-p sampling · ten-layer decomposition (L3–L10) + a 1024-slot micro-ISA registry (1024 implemented) · bit-identical Node ⇄ browser*

**English** · [中文说明 →](README.md)

</div>

---

## What this is

A **zero-dependency, single-file** GPT language model, implemented end to end — tokenize → forward → hand-written backpropagation → AdamW → sampled generation — running identically in **Node and the browser**.

**Design creed: zero rewrite, stand on giants.** Nothing here is invented from scratch; every component composes a world-class published result — RoPE (LLaMA), GQA (LLaMA-2/3), SwiGLU + RMSNorm (LLaMA), MoE + top-k routing + load-balancing loss (Mixtral / DeepSeek), sliding-window & interleaved sparse attention (Mistral / Gemma 2/3), QK-norm + logit soft-cap (Gemma 2), Attention Sink / StreamingLLM, YaRN (NTK-by-parts RoPE) — each backed by a hard, independently checkable invariant rather than a claim.

## Capabilities

| Area | What's implemented | Backed by |
|---|---|---|
| **Tokenizers** | Char-level + BPE (`byteLevel` on/off), lossless round-trip | GPT-2 byte-level BPE |
| **Backbone** | Decoder-only Transformer, hand-written forward + backward, AdamW (warmup, wd, grad-clip, resume) | GPT-2 |
| **Norm / FFN / Pos** | `ln`\|`rmsnorm` · `gelu`\|`swiglu` · `learned`\|`rope` · **ALiBi** (`alibi:true`, replaces positional encoding entirely) — one-line presets | LLaMA modern recipe, ALiBi / BLOOM / MPT |
| **ALiBi (v15.10)** | Linear distance bias `−slope[h]·\|i−j\|` added to attention logits, `slope[h]=2^(−8h/n)` (0-based, `n` = next power of two); parameter-free ⇒ backwards needs no change. `kv.pos` tracks absolute positions across `kvShift` eviction so streaming stays bit-exact | Press et al., 2022 |
| **Attention** | MHA / **GQA** (`nKvHead`) · **sliding-window** & **interleaved** sparse (`window`) | LLaMA-2/3, Mistral, Gemma 2/3 |
| **Stabilizers** | Per-head **QK-norm**, **logit soft-cap** (`cap·tanh(x/cap)`) | Gemma 2 |
| **MoE** | N experts + **top-k** routing + **load-balancing aux loss** | Mixtral / DeepSeek |
| **Quantization** | **INT8 weight-only PTQ** (≈7× smaller, decode-lossless) · **KV-cache INT8** | llama.cpp-style PTQ |
| **Long context** | **Attention Sink / StreamingLLM** (bounded cache, infinite stream) · **YaRN** length extrapolation | StreamingLLM, YaRN 2023 |
| **KV cache** | Incremental decode, bit-exact vs full recompute; INT8 variant with eviction (`kvShift`) | — |
| **Sampling** | temperature / top-k / top-p / greedy, seeded & deterministic | — |
| **Ten-layer arch** | L3–L10 physically decomposed (arch / ops / autograd / backend / tensor / kernel / numbase / micro-ISA) | — |
| **Micro-ISA (L10)** | **1024-slot registry, 8 domains A–H × 128, all 1024 implemented** — each op ships intuition / math / pseudocode / complexity / reference / pitfall / invariant + `run` + `predicate` | — |
| **Dual-end** | Node and browser share one codebase; forward is **bit-identical** across both | — |

Full version-by-version changelog (v3.0 → v15.10) lives in the [Chinese README](README.md).

## Verification

Every number below is produced by a gate in CI (`Node 20/22`, run serially — parallel `node` triggers environment-level false reds here).

### Self-check suite — 12/12 ✅ (`node src/verify.js`)

| # | Check | Measured | Criterion |
|---|---|---|---|
| 1 | RNG determinism & uniformity | bit-identical per seed, range [0,1) | mulberry32 reproducible |
| 2 | Softmax row-sum = 1, all positive | row-sum error **2.22e-16** (machine ε) | = 1 |
| 3 | LayerNorm invariant | x̂ mean 8.33e-17, Y=g·x̂+b error **0** | mean 0 / var 1 (with ε) |
| 4 | GELU derivative numeric match | analytic vs central-diff **5.85e-10** | exact tanh-approx derivative |
| 5 | Causal-mask invariant | past logits change **0**, upper-tri attn = 0 | future invisible |
| 6 | **Central-difference gradient check** | 395 params, max err/tol = **4.22e-4** (threshold 1.0) | analytic ≡ numeric |
| 7 | Overfit: memorize tiny corpus | loss **2.917 → 0.0011** | capacity proof, < 0.1 |
| 8 | Random-batch training converges | first-30 mean 2.463 → last-30 0.036 | monotone decrease |
| 9 | Generation determinism & robustness | same-seed ✓ greedy ✓ 120-char prompt ✓ empty prompt ✓ | all edges pass |
| 10 | Tokenizer round-trip | encode∘decode = id, unknown chars skipped | lossless |
| 11 | **KV cache ≡ full-sequence forward** | last-logit Δ **0**; 10-step greedy bit-exact; 40-step over-window capacity constant | incremental ≡ full |
| 12 | Model persistence save/load | param Δ **0**, logits Δ **0**, re-serialize identical, vocab restored | archive fully reproducible |

> The 5 skipped params in check 6 are Q/K biases — softmax's row-shift invariance makes their gradient identically zero (matches PyTorch). A mathematical fact, not a defect.

### Gate matrix (all green)

| Gate | Command | Result |
|---|---|---|
| Layer lint (no dual-end-unsafe syntax) | `npm run lint` | 72 files, 0 violations |
| Micro-ISA coverage | `npm run isa` | 1024/1024 (8 × 128) |
| Self-check invariants | `npm test` | 12/12 |
| Layer unit tests | `npm run test:layers` | 191/191 |
| Zero-regression vs v11 | `npm run regression` | R1–R10 maxΔ = 0 |
| All feature probes | `npm run test:all` | 9/9 |
| Browser assembly (headless, bit-exact) | `npm run test:demo` | 9/9 |
| **Real-browser render + interaction** | `npm run test:demo-browser` | 1024×8 predicates green |
| Unified benchmark | `npm run bench` | 25/25 |

### One-command benchmark report

`npm run bench` aggregates all core metrics into [`docs/BENCHMARK.md`](docs/BENCHMARK.md) (multi-architecture gradient checks, RoPE invariance, GQA, MoE, BPE, INT8 quantization, KV-cache speedup). `npm run bench:full` adds a **parameter-aligned architecture comparison** — it binary-searches each preset's `ffnHidden` to pull **six architectures** (GPT-2 / LLaMA / LLaMA-GQA / SWA / Gemma / Mixtral-MoE) to an **equal parameter budget**, then trains them on the same corpus / seed / steps and compares loss & accuracy. CI runs the fast version and uploads the report as an artifact.

**Parameter-aligned comparison** (`bench:full`, ~124K params aligned, same corpus/seed/300 steps):

| Architecture | final loss | accuracy |
|---|---|---|
| gpt2 | 1.910 | 34.9% |
| **llama** | 0.859 | **47.6%** |
| llama-gqa | 0.831 | 44.4% |
| swa | 0.831 | 44.4% |
| gemma | **0.802** | 44.4% |
| mixtral (MoE) | 1.260 | 38.1% |

At equal parameter budget the modern RMSNorm+SwiGLU+RoPE recipe (llama 47.6%) clearly beats vanilla GPT-2 (34.9%); gemma reaches the lowest loss. `swa` and `llama-gqa` come out bit-identical (when window = blockSize the sliding window degenerates to full attention) — itself a correctness signal.

## Performance

KV-cache incremental decoding is the one effective speedup: each step computes only the new token and reuses cached K/V.

| Config | gen 64 tokens | gen 200 tokens | throughput |
|---|---|---|---|
| 370K params · blockSize 96 | **28.5×** (33ms vs 942ms) | **58.8×** (115ms vs 6759ms) | 1739 tok/s |
| blockSize 256 · dModel 128 | **27.3×** (79ms vs 2153ms) | **95.3×** (223ms vs 21253ms) | 897 tok/s |

Browser (120K params): **80 tokens / 27ms = 2996 tok/s**.

**Two optimizations rejected by measurement** (no data-free tuning):

| Route | Measured | Verdict |
|---|---|---|
| Hand-written 4× loop unrolling | 32.7µs vs 30.2µs (slower) | V8 already optimizes naive loops to the scalar limit ~900M MAC/s |
| Int8 **compute** quantization (llama.cpp route) | 0.92× (no speedup) | JS has no SIMD intrinsics; the gain there comes from C++ SIMD |
| **INT8 weight-only quantization (v5.0)** | **archive ≈ 5.6×, RAM ≈ 6.9×** | weights resident as Int8, dequantized inline in matmul — compute speed flat but size/RAM way down, decode-lossless (logits maxΔ < 0.03) |

## Quickstart

### Node (verify + API)

```bash
git clone https://github.com/CJX0712/flagship-ai.git
cd flagship-ai
node src/verify.js   # 12 self-checks, ~10s
```

Train + generate in ~10 lines:

```js
const F = require('./src/index.js');
const { CORPUS } = require('./src/corpus.js');

// Char-level tokenizer (classic, Chinese-friendly, vocab ≈ 291)
const tok = new F.Tokenizer(CORPUS);
const model = new F.GPTModel({ vocabSize: tok.vocabSize,
  blockSize: 64, nLayer: 2, nHead: 4, dModel: 64, seed: 42 });

const r = F.train(model, tok.encode(CORPUS),
  { steps: 300, lr: 3e-3, batch: 32, warmup: 10, seed: 0 });
console.log('final loss:', r.finalLoss);                // ≈ 1.93

const out = F.generate(model, tok, '旗舰之道', {
  maxTokens: 100, temperature: 0.8, topK: 40, seed: 7 }); // KV cache on by default
console.log(out.text);

// ── One line to the modern architecture (RMSNorm + SwiGLU + RoPE) ──
const llama = new F.GPTModel(F.applyPreset(
  { vocabSize: tok.vocabSize, blockSize: 64, nLayer: 2, nHead: 4, dModel: 64 }, 'llama'));

// ── GQA grouped-query attention (LLaMA-2/3: KV heads < attention heads) ──
const gqa = new F.GPTModel(F.applyPreset(
  { vocabSize: tok.vocabSize, blockSize: 64, nLayer: 2, nHead: 4, dModel: 64 }, 'llama-gqa')); // nKvHead=2
// Or pass nKvHead directly (needs nHead % nKvHead === 0); omitted = nKvHead=nHead (MHA)
const gqa2 = new F.GPTModel({ vocabSize: tok.vocabSize, blockSize: 64, nLayer: 2,
  nHead: 4, nKvHead: 1, dModel: 64, norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope' });

// ── MoE mixture-of-experts (N experts + top-k routing + load-balancing aux loss) ──
const moe = new F.GPTModel({ vocabSize: tok.vocabSize, blockSize: 64, nLayer: 2,
  nHead: 4, nKvHead: 2, dModel: 64, norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope',
  nExp: 4, topK: 2, auxWeight: 0.01 });                 // or applyPreset(cfg,'mixtral')
const moeOut = F.generate(moe, tok, '旗舰', { maxTokens: 40, temperature: 0.9 });
// Only 2/4 experts run per token; the aux loss balances expert load during training

// ── Sliding-window attention (query i attends only the last `window` keys; Mistral/Gemma) ──
const swa = new F.GPTModel({ vocabSize: tok.vocabSize, blockSize: 64, nLayer: 2,
  nHead: 4, nKvHead: 2, dModel: 64, norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope',
  window: 32 });                                        // or applyPreset(cfg,'swa')
const swaOut = F.generate(swa, tok, '旗舰', { maxTokens: 40, temperature: 0.9 });
// window=0 (default) = full causal; under RoPE, KV-cache incremental == full recompute bit-for-bit

// ── Interleaved sparse attention (Gemma 2/3: even layers global, odd layers windowed) ──
const gemma = new F.GPTModel({ vocabSize: tok.vocabSize, blockSize: 64, nLayer: 4,
  nHead: 4, nKvHead: 2, dModel: 64, norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope',
  window: { alt: 32 } });                                // or applyPreset(cfg,'gemma')
// Per-layer explicit form: window: [0, 32, 0, 32] (length must = nLayer); model.windows is the normalized per-layer array

// ── Gemma-2 stabilizers (QK-norm + logit soft-cap) ──
const gemma2 = new F.GPTModel({ vocabSize: tok.vocabSize, blockSize: 64, nLayer: 4,
  nHead: 4, nKvHead: 2, dModel: 64, norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope',
  window: { alt: 32 }, qkNorm: true, logitCap: 30 });    // the gemma preset already includes both
// QK-norm: per-head Q/K RMSNorm (before RoPE); logitCap: logits = 30·tanh(x/30), strictly bounded output

// ── BPE tokenizer (char-level primitives; fewer tokens on Chinese corpora) ──
const bpe = new F.BPETokenizer(CORPUS, 320, { byteLevel: false }); // vocab 317 · lossless round-trip
console.log('bpe tokens:', bpe.encode('旗舰 AI 锻造世界顶级模型').length);

// Persistence: params + vocab saved together; a reload fully reproduces (BPE too, lossless)
const json = F.saveModel(model, tok);
const model2 = F.loadModel(json);
const tok2 = F.loadTokenizer(json);

// ── INT8 weight-only quantized inference (weight-only PTQ, RAM/archive ≈ 1/7, decode-lossless) ──
const mInt8 = model.quantize();                 // read-only export; the original model is untouched
const json8 = F.saveModel(mInt8, tok);           // the archive auto-encodes as {q, scale}
const m8 = F.loadModel(json8);                   // loads as resident INT8
console.log('quantized archive ratio:', (json.length / json8.length).toFixed(2) + 'x'); // ≈ 5.6x
const out8 = F.generate(m8, tok, '旗舰', { maxTokens: 40, temperature: 0.9 });
```

### Browser (one double-click)

No build, no server: **double-click `demo.html`** (opens directly over `file://`). A single file with all code inlined, supporting:

- 📝 Live corpus editing → one-click model build (instant vocab / param count)
- 🧪 One-click self-check (all 12 items light up, matching Node)
- 🔬 **L10 micro-ISA browser** (all 8 domains A–H × 128 = 1024 ops; filter by domain / keyword search / expand the seven-piece doc / **run each op's invariant in-browser** / batch-run all 1024 × 8 seeds)
- 📈 Training (chunked, non-blocking UI; live loss curve; resume-and-continue is **bit-identical** to straight-through)
- ✨ Generation (temperature / top-k / seed adjustable; KV-cache accelerated; live token/s)
- 💾 Export / import model (params + vocab, lossless JSON)

**L10 micro-ISA browser, live screenshot** (rendered + interacted + batch-run by `npm run test:demo-browser` in headless Chromium; CI uploads the same artifact every run):

![Flagship AI — L10 micro-ISA browser (1024/1024, batch-run 8192 predicate evaluations all green)](docs/demo-screenshot.png)

## API cheat-sheet

| Export | Signature | Notes |
|---|---|---|
| `Tokenizer` | `new Tokenizer(text)` → `.encode(str)`, `.decode(ids)`, `.vocabSize` | char-level; unknown chars skipped |
| `BPETokenizer` | `new BPETokenizer(text, vocabSize?, {minPairFreq, byteLevel})` | BPE; `byteLevel:true` = 256 byte primitives (GPT-2), `false` = char-level (fewer tokens on Chinese); same `.encode`/`.decode`/`.vocabSize` |
| `GPTModel` | `new GPTModel(cfg)` → `.forward(tokens)`, `.backward(cache, dlogits)`, `.params`, `.grads`, `.paramCount()`, `.evaluate(tokens)` | cfg: `vocabSize/blockSize/nLayer/nHead/dModel/seed` + switches `norm('ln'\|'rmsnorm')` / `ffn('gelu'\|'swiglu')` / `pos('learned'\|'rope')` / `ffnHidden` / `nKvHead` (KV heads, default `=nHead` i.e. MHA; `<nHead` enables GQA) + MoE `nExp` (default 1 = dense) / `topK` (default 1) / `auxWeight` (default 0.01) + sparse attention `window` (scalar = same window all layers; array = per-layer, length must = nLayer; `{alt:N}` = even layers global / odd layers window N; default 0 = full causal) + stabilizers `qkNorm` (per-head Q/K RMSNorm before RoPE, default false) / `logitCap` (`cap·tanh(x/cap)`, default 0 = off) / `kvQuant` (KV-cache INT8, 8 = on, default 0) / `attentionSink` (K>0 = pin first K tokens for infinite-context streaming, default 0) / `rollingWindow` (rolling bound with sink, default 256) / `ropeScale` (YaRN: `{type:'yarn'\|'ntk', factor>1, originalMaxLength}`; `'yarn'` = NTK-by-parts RoPE interpolation + attention-temperature mscale, `'ntk'` = global base-frequency scaling; `factor≤1` or omitted = identity) |
| `applyPreset` / `ARCH_PRESETS` | `applyPreset(cfg, 'gpt2'\|'llama'\|'llama-gqa'\|'mixtral'\|'swa'\|'gemma')` | `gpt2={ln,gelu,learned}` · `llama={rmsnorm,swiglu,rope}` · `llama-gqa={rmsnorm,swiglu,rope,nKvHead:2}` · `mixtral={rmsnorm,swiglu,rope,nKvHead:2,nExp:4,topK:2,auxWeight:0.01}` · `swa={rmsnorm,swiglu,rope,nKvHead:2,window:64}` · `gemma={rmsnorm,swiglu,rope,nKvHead:2,window:{alt:64},qkNorm:true,logitCap:30}` (a preset overrides same-named cfg keys — reassign after it to change back) |
| `model.quantize()` | `model.quantize()` → quantized copy (with `.q`/`.scale`) | INT8 weight-only PTQ (dense wqkv/wproj/w1/w2, MoE router wg and experts e*.w1/w2, embeddings); dequantized inline at inference; original model unchanged; `saveModel` auto-encodes as `{q,s}` |
| `train` | `train(model, tokens, opts)` → `{history, finalLoss, state}` | opts: `steps/lr/batch/warmup/seed/wd/gradClip`; `state + stepOffset + totalSteps` for resume |
| `generate` | `generate(model, tok, prompt, opts)` → `{text, ids}` | opts: `maxNew` (alias `maxTokens`) / `temperature` / `topK` / `topP` / `seed` / `useCache` |
| `initKVCache` | `initKVCache(model)` → `{K, V, Ks, Vs, n, cap, sink}` | per-layer K/V preallocated `cap×D` (cap = `attentionSink>0 ? K+R : blockSize`; INT8 adds per-(token,head) `Ks/Vs` scales) |
| `forwardStep` | `forwardStep(model, tokenId, pos, kv)` → `{logits}` | single-token incremental forward, O(T·D); attention over anchors `[0,sink)` + window `[jLo,n)` |
| `kvShift` | `kvShift(model, kv)` | cache eviction: with sink>0 anchors stay and only the window slides; otherwise the whole cache shifts, dropping the oldest token (`copyWithin` native memmove; K/V and INT8 scales move together) |
| `saveModel` / `loadModel` / `loadTokenizer` | `saveModel(model, tok?)` → JSON | lossless param + vocab persistence (carries tokenizer type and BPE merges) |
| `globalThis.FlagshipAI` (layer namespace) | `FlagshipAI.L10.stats()` / `L10.ops[key].run(...)` / `L10.ops[key].predicate(...)` / `L10.display(key)` / `L9.rng.mulberry32(seed)` / `L8.matmul.matmul(...)` | ten-layer namespace (Node: readable after `require('src/index.js')`; browser: `window.FlagshipNS`). Does **not** affect the 34-key public API surface (K15 contract) |

## Project structure

```
flagship-ai/
├── src/
│   ├── index.js             # ten-layer assembly entry: loads L10→L1→L0 by __loadOrder, assembles the 34-key public API (L0-boot/assemble is the single mapping source)
│   ├── namespace.js         # assembly source of truth: __loadOrder + domain defs (shared by Node & browser)
│   ├── verify.js            # 12-item self-check suite (Node + browser)
│   ├── corpus.js            # default Chinese corpus (swap freely)
│   ├── legacy/              # v11 engine body (zero-regression oracle, R1–R10 bit-exact)
│   ├── L10-microisa/        # micro-ISA layer: 1024-slot registry + seven-piece ops (1024 implemented)
│   ├── L9-numbase/          # numerics base: rng / fp-ops / special / assert
│   ├── L8-kernel/           # numeric kernels: matmul / reduce / quantize (bit-exact with legacy)
│   ├── L7-tensor/           # tensors & memory: tensor / shape / arena / kv-cache
│   ├── L6-backend/          # backend runtime: kernel registry dispatch / CPU(JS) exact channel
│   ├── L5-autograd/         # autodiff: VJP registry / tape contract / gradient-check factory
│   ├── L4-ops/              # neural ops: norm/activation/positional/softmax/linear/attention/moe/loss
│   ├── L3-arch/             # architecture assembly: forward/backward reassembly
│   ├── L2-pipeline/         # orchestration: tokenizer/train/generate/sampler/evaluate/persistence/config
│   ├── L1-app/              # application: demo-app (wired into __loadOrder)
│   └── L0-boot/             # bootstrap: assemble (34-key mapping + K15 name gate + registry self-bootstrap)
├── tests/                   # node:test + fast-check property tests (L3–L10)
├── scripts/build-demo.js    # inlines the ten-layer stack + suite + app into demo.html (mini-require, dual-end same code)
├── scripts/isa.js           # micro-ISA coverage CLI (per-domain impl/capacity assertions)
├── scripts/probe-*.js       # per-version feature probes (v3 GQA/quant/MoE/SWA/Gemma/KV-INT8/StreamingLLM/YaRN) + demo/microisa gates
├── scripts/bench.js         # unified benchmark: aggregates metrics → docs/BENCHMARK.md (`npm run bench`)
├── demo.html                # single-file browser demo (build artifact — double-click to run)
├── docs/ARCHITECTURE.md     # architecture doc: formulas / complexity / backprop derivation / ten-layer & dual-end assembly (§9)
├── docs/BENCHMARK.md        # auto-generated benchmark report (bench.js output)
└── package.json
```

## Design principles

1. **Zero-dependency** — no packages; all matrix math hand-written on `Float64Array`
2. **Verifiable** — every algorithm carries ≥1 hard invariant: softmax row-sum, LayerNorm normalization, causal mask, numeric gradient check, overfit convergence
3. **Dual-end consistent** — Node and browser share one codebase and one verification suite
4. **Teaching-friendly** — single-file readable, comments cite sources, gradient derivations in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## Known limitations

- Teaching scale (dModel 64 / 2 layers ≈ 120K params) — for learning & verification, not production inference
- Pure-JS single thread, no GPU; loop unrolling and Int8 **compute** quantization were measured and ruled out (no SIMD gain in V8), but INT8 **weight** quantization (compression only) still yields ≈ 7× smaller archive/RAM
- Quantized compute speed is flat (no SIMD); the win is memory & serialized size, decode-quality-lossless
- Under `pos:'learned'` (GPT-2 absolute embeddings), sequences past `blockSize` fall back to a **sliding window** (drop oldest token), no longer bit-equivalent to "recompute the last blockSize tokens" — an inherent constraint of learned absolute positions
- Under `pos:'rope'`, length extrapolation is natural and the sliding window stays **bit-identical** to full inference (attention scores depend only on relative distance) — better for very long generation
- Byte-level BPE can compress worse than char-level on small Chinese corpora (each Han char is 3 bytes that must be merged back); prefer `byteLevel:false` for Chinese

## License

[MIT](LICENSE) © 2026 Chenxing (晨星)

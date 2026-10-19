'use strict';
/* ============================================================================
 * 旗舰 AI · 统一基准报告 (bench.js)
 * ----------------------------------------------------------------------------
 * 聚合各模块核心度量，一键产出 docs/BENCHMARK.md：
 *   - 多架构手写反向传播梯度检验 (err/tol ≤ 1)
 *   - RoPE 相对位置不变性
 *   - GQA 梯度检验 + 参数节省
 *   - MoE（混合专家）梯度检验 + 稀疏参数量
 *   - BPE 分词器往返无损 + 压缩率
 *   - INT8 量化保真 / 体积压缩 / in-RAM 压缩
 *   - KV cache 增量推理加速比
 *   - [--full] 参数对齐架构对照 (loss / 准确率)
 *
 * 运行：node scripts/bench.js            (快速指标 + 推理速度)
 *       node scripts/bench.js --full     (追加训练对照表)
 * 作者：晨星  ·  License: MIT
 * ========================================================================*/
const fs = require('fs');
const path = require('path');
const F = require('../src/flagship-engine.js');
const { CORPUS } = require('../src/corpus.js');

const FULL = process.argv.includes('--full');
const now = new Date();
const rows = [];           // 汇总指标行 {group, name, value, target, pass}
function rec(group, name, value, target, pass) { rows.push({ group, name, value, target, targetPass: pass }); }

/* ---------- 1. 多架构梯度检验 ---------- */
function gradCheck(model, toks, label) {
  const V = model.cfg.vocabSize, T = Math.min(toks.length, model.cfg.blockSize);
  const seq = toks.slice(0, T), targets = seq.slice(1);
  model.zeroGrads();
  const { logits, cache } = model.forward(seq);
  model.backward(cache, F.crossEntropy(logits, targets, T - 1, V, T).dlogits);
  const e = 1e-5, atol = 1e-7, rtol = 1e-4;
  let worst = 0, checked = 0, skipped = 0;
  for (const key in model.params) {
    const p = model.params[key].d, g = model.grads[key];
    const cnt = Math.min(p.length, 60);
    for (let s = 0; s < cnt; s++) {
      const i = Math.floor(s * p.length / cnt), orig = p[i];
      p[i] = orig + e; const lp = F.crossEntropy(model.forward(seq).logits, targets, T - 1, V, T).loss;
      p[i] = orig - e; const lm = F.crossEntropy(model.forward(seq).logits, targets, T - 1, V, T).loss;
      p[i] = orig;
      const num = (lp - lm) / (2 * e), ana = g[i];
      if (Math.abs(ana) < 1e-12 && Math.abs(num) < 1e-12) { skipped++; continue; }
      const err = Math.abs(ana - num) / (atol + rtol * Math.max(Math.abs(ana), Math.abs(num)));
      if (err > worst) worst = err;
      checked++;
    }
  }
  const pass = worst < 1;
  rec('反向传播', label + ' 梯度检验', `err/tol=${worst.toExponential(2)} (${checked}分量/${skipped}跳过)`, '≤ 1', pass);
  return pass;
}
const gbase = { vocabSize: 120, blockSize: 12, nLayer: 2, nHead: 4, dModel: 32, seed: 7 };
const seq = Array.from({ length: 12 }, (_, i) => i % 120);
let allPass = true;
allPass = gradCheck(new F.GPTModel(gbase), seq, 'GPT-2') && allPass;
allPass = gradCheck(new F.GPTModel(F.applyPreset(gbase, 'llama')), seq, 'LLaMA') && allPass;
allPass = gradCheck(new F.GPTModel(Object.assign({}, gbase, { norm: 'rmsnorm' })), seq, '仅 RMSNorm') && allPass;
allPass = gradCheck(new F.GPTModel(Object.assign({}, gbase, { ffn: 'swiglu' })), seq, '仅 SwiGLU') && allPass;
allPass = gradCheck(new F.GPTModel(Object.assign({}, gbase, { pos: 'rope' })), seq, '仅 RoPE') && allPass;

/* ---------- 2. RoPE 相对位置不变性 ---------- */
(function () {
  const D = 32, H = 4, dh = D / H, L = 4, shift = 5;
  const rng = F.mulberry32(1), raw = new Float64Array(L * 3 * D);
  for (let i = 0; i < raw.length; i++) raw[i] = F.randn(rng);
  const A = Float64Array.from(raw), B = Float64Array.from(raw);
  F.ropeApply(A, L, D, H, dh, 0, 1); F.ropeApply(B, L, D, H, dh, shift, 1);
  let maxDiff = 0;
  for (let a = 0; a < L; a++) for (let b = 0; b < L; b++) {
    let s1 = 0, s2 = 0;
    for (let p = 0; p < dh; p++) { s1 += A[a * 3 * D + p] * A[b * 3 * D + D + p]; s2 += B[a * 3 * D + p] * B[b * 3 * D + D + p]; }
    maxDiff = Math.max(maxDiff, Math.abs(s1 - s2));
  }
  rec('位置编码', 'RoPE 相对位置不变性', `Δ=${maxDiff.toExponential(2)}`, '< 1e-12', maxDiff < 1e-12);
})();

/* ---------- 3. GQA 梯度检验 + 参数节省 ---------- */
(function () {
  const V = 11, T = 6, H = 4, KV = 2;
  const model = new F.GPTModel({ vocabSize: V, blockSize: 7, nLayer: 1, nHead: H, nKvHead: KV, dModel: 16, norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope', seed: 7 });
  const rng = F.mulberry32(5), tokens = [], targets = [];
  for (let i = 0; i < T; i++) tokens.push(Math.floor(rng() * V));
  for (let i = 1; i < T; i++) targets.push(tokens[i]);
  const { logits, cache } = model.forward(tokens);
  model.backward(cache, F.crossEntropy(logits, targets, T - 1, V, T).dlogits);
  let worst = 0, checked = 0, skipped = 0, all = [];
  for (const k in model.params) for (let i = 0; i < model.params[k].d.length; i++) all.push([k, i]);
  const srng = F.mulberry32(99);
  for (let i = all.length - 1; i > 0; i--) { const j = Math.floor(srng() * (i + 1)); [all[i], all[j]] = [all[j], all[i]]; }
  const cand = all.slice(0, 500);
  for (const [k, i] of cand) {
    const p = model.params[k].d, orig = p[i];
    p[i] = orig + 1e-5; const lp = F.crossEntropy(model.forward(tokens).logits, targets, T - 1, V, T).loss;
    p[i] = orig - 1e-5; const lm = F.crossEntropy(model.forward(tokens).logits, targets, T - 1, V, T).loss;
    p[i] = orig;
    const num = (lp - lm) / 2e-5, ana = model.grads[k][i];
    if (Math.abs(ana) < 1e-12 && Math.abs(num) < 1e-12) { skipped++; continue; }
    worst = Math.max(worst, Math.abs(ana - num) / (1e-7 + 1e-4 * Math.max(Math.abs(ana), Math.abs(num))));
    checked++;
  }
  rec('注意力', 'GQA 梯度检验', `err/tol=${worst.toExponential(2)} (${checked}分量)`, '≤ 1', worst <= 1);
  const cfg = { vocabSize: 50, blockSize: 16, nLayer: 2, nHead: 4, dModel: 16, seed: 1 };
  const m1 = new F.GPTModel(cfg), m2 = new F.GPTModel(Object.assign({}, cfg, { nKvHead: 2 }));
  rec('注意力', 'GQA 参数节省', `MHA=${m1.paramCount()} → GQA=${m2.paramCount()}`, 'GQA < MHA', m2.paramCount() < m1.paramCount());
})();

/* ---------- 3.5 MoE（混合专家）梯度检验 + 稀疏参数量 ---------- */
(function () {
  // 强制清晰路由间隙（bg 大间隔 + wg=0）保证 top-k 选择在扰动下稳定，从而中心差分有效
  const moe = new F.GPTModel({ vocabSize: 40, blockSize: 12, nLayer: 1, nHead: 2, dModel: 16, norm: 'rmsnorm', ffn: 'swiglu', nExp: 4, topK: 2, seed: 7, auxWeight: 0 });
  moe.params['l0.bg'].d.set([8, 4, 0, -4]);
  moe.params['l0.wg'].d.fill(0);
  moe.auxWeight = 0;
  const seq2 = Array.from({ length: 8 }, (_, i) => (i * 3) % 40);
  gradCheck(moe, seq2, 'MoE(路由+专家)');
  const dense = new F.GPTModel({ vocabSize: 40, blockSize: 12, nLayer: 1, nHead: 2, dModel: 16 });
  rec('混合专家', 'MoE 稀疏参数量', `稠密=${dense.paramCount()} → MoE(4专家/top2)=${moe.paramCount()}`, 'MoE>稠密', moe.paramCount() > dense.paramCount());
})();

/* ---------- 4. BPE 往返 + 压缩率 ---------- */
(function () {
  const bpe = new F.BPETokenizer(CORPUS, 800);
  const samples = ['朝辞白帝彩云间', '旗舰 AI 锻造世界顶级模型', 'Hello, world! 123 🚀 emoji', '混合 English 与中文 + punctuation!!!'];
  let ok = true;
  for (const s of samples) ok = ok && bpe.decode(bpe.encode(s)) === s;
  rec('分词器', 'BPE 往返无损', ok ? '中文/emoji/标点/多字节 OK' : '有损', '无损', ok);
  const charTok = new F.Tokenizer(CORPUS), charLen = charTok.encode(CORPUS).length;
  const bl = new F.BPETokenizer(CORPUS, 900, { byteLevel: true }).encode(CORPUS).length;
  const cl = new F.BPETokenizer(CORPUS, 900, { byteLevel: false }).encode(CORPUS).length;
  rec('分词器', 'BPE 压缩率(字符级 V≈900)', `字节级 ${charLen / bl}x / 字符级 ${charLen / cl}x`, '字符级 ≥ 1', charLen / cl >= 1);
})();

/* ---------- 5. INT8 量化保真 / 体积 / in-RAM ---------- */
(function () {
  const tok = new F.Tokenizer(CORPUS);
  const m = new F.GPTModel(F.applyPreset({ vocabSize: tok.vocabSize, blockSize: 48, nLayer: 2, nHead: 4, dModel: 32 }, 'llama'));
  F.train(m, tok.encode(CORPUS), { steps: 120, lr: 3e-3, batch: 24, warmup: 8, seed: 0 });
  const ids = tok.encode(CORPUS.slice(0, 40));
  const lF = m.forward(ids).logits, mq = m.quantize(), lQ = mq.forward(ids).logits;
  let md = 0; for (let i = 0; i < lF.length; i++) md = Math.max(md, Math.abs(lF[i] - lQ[i]));
  rec('量化', 'INT8 logits 保真', `max|Δ|=${md.toExponential(2)}`, '< 0.1', md < 0.1);
  const jsonF = F.saveModel(m, tok), jsonQ = F.saveModel(mq, tok);
  const ratio = jsonF.length / jsonQ.length;
  rec('量化', '存档体积压缩', `F32=${(jsonF.length / 1024).toFixed(1)}KB → INT8=${(jsonQ.length / 1024).toFixed(1)}KB (${ratio.toFixed(2)}x)`, '> 2.5x', ratio > 2.5);
  let linF32 = 0, linQ = 0, totalF32 = 0, qb = 0;
  for (const k in m.params) {
    const f32 = m.params[k].d.length * 8; totalF32 += f32;
    if (mq.params[k].q) { linF32 += f32; linQ += mq.params[k].q.length; qb += mq.params[k].q.length; } else qb += f32;
  }
  rec('量化', 'in-RAM 线性权重压缩', `${linF32 / 8}B → ${linQ}B (8x)`, '= 8x', linQ * 8 === linF32);
  rec('量化', '模型整体 in-RAM 压缩', `${(totalF32 / qb).toFixed(2)}x`, '> 4x', totalF32 / qb > 4);
})();

/* ---------- 6. KV cache 增量推理加速比 ---------- */
function kvSpeed(model, N, runs) {
  const V = model.cfg.vocabSize, rng = F.mulberry32(3);
  const ids = Array.from({ length: N }, () => Math.floor(rng() * V));
  let inc = Infinity, full = Infinity;
  for (let r = 0; r < runs; r++) {
    let t = process.hrtime.bigint();
    const kv = F.initKVCache(model);
    for (let i = 0; i < ids.length; i++) F.forwardStep(model, ids[i], i, kv);
    inc = Math.min(inc, Number(process.hrtime.bigint() - t) / 1e6);
    t = process.hrtime.bigint();
    let ctx = [];
    for (let i = 0; i < ids.length; i++) { ctx.push(ids[i]); model.forward(ctx); }
    full = Math.min(full, Number(process.hrtime.bigint() - t) / 1e6);
  }
  return full / inc;
}
(function () {
  const model = new F.GPTModel(F.applyPreset({ vocabSize: 65, blockSize: 300, nLayer: 2, nHead: 4, dModel: 64 }, 'gpt2'));
  const speedup = kvSpeed(model, 256, 5);
  rec('推理', 'KV cache 增量加速 (N=256)', `${speedup.toFixed(1)}x`, '> 10x', speedup > 10);
})();

/* ---------- 7. [--full] 参数对齐架构对照 ---------- */
if (FULL) {
  const bigBpe = new F.BPETokenizer(CORPUS, 900, { byteLevel: false });
  const idsB = bigBpe.encode(CORPUS);
  const baseCfg = { vocabSize: bigBpe.vocabSize, blockSize: 64, nLayer: 2, nHead: 4, dModel: 64, seed: 42 };
  const gpt2 = new F.GPTModel(baseCfg);
  const target = gpt2.paramCount();
  let lo = 8, hi = 4 * 64;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    const pc = new F.GPTModel(F.applyPreset(Object.assign({}, baseCfg, { ffnHidden: mid }), 'llama')).paramCount();
    if (pc <= target) lo = mid; else hi = mid - 1;
  }
  const llamaEq = new F.GPTModel(F.applyPreset(Object.assign({}, baseCfg, { ffnHidden: lo }), 'llama'));
  rec('对照', '参数对齐 gpt2 参数量', target.toLocaleString(), '-', true);
  rec('对照', '参数对齐 llama(对齐) 参数量', llamaEq.paramCount().toLocaleString() + ` (ffnHidden=${lo})`, '≈ gpt2', Math.abs(llamaEq.paramCount() - target) < 5000);
  for (const [m, label] of [[gpt2, 'gpt2'], [new F.GPTModel(F.applyPreset(baseCfg, 'llama')), 'llama(4D)'], [llamaEq, 'llama(对齐)']]) {
    const r = F.train(m, idsB, { steps: 300, lr: 3e-3, batch: 32, warmup: 15, seed: 0, logEvery: 1e9 });
    const acc = m.evaluate(idsB.slice(0, 64)).acc;
    rec('对照', `${label} loss`, `${r.history[0].toFixed(2)} → ${r.finalLoss.toFixed(3)} · 准确率 ${(acc * 100).toFixed(1)}%`, '-', true);
  }
}

/* ---------- 渲染 Markdown ---------- */
function renderMarkdown() {
  const groups = [];
  for (const r of rows) { let g = groups.find(x => x.name === r.group); if (!g) { g = { name: r.group, items: [] }; groups.push(g); } g.items.push(r); }
  const L = [];
  L.push(`# 旗舰 AI · 基准报告 (BENCHMARK)`);
  L.push('');
  L.push(`> 自动生成于 ${now.toISOString()} · Node ${process.version} · ${FULL ? '含训练对照(--full)' : '快速指标'}`);
  L.push('> 引擎：零依赖纯 JS 手写 GPT · 复现命令 `npm run bench' + (FULL ? ' -- --full' : '') + '`');
  L.push('');
  const passCount = rows.filter(r => r.targetPass).length;
  L.push(`**总览：${passCount}/${rows.length} 项达标**${passCount === rows.length ? ' ✅' : ' ⚠️'}`);
  L.push('');
  for (const g of groups) {
    L.push(`## ${g.name}`);
    L.push('');
    L.push('| 指标 | 实测 | 目标 | 达标 |');
    L.push('|---|---|---|---|');
    for (const r of g.items) L.push(`| ${r.name} | ${r.value} | ${r.target} | ${r.targetPass ? '✅' : '❌'} |`);
    L.push('');
  }
  L.push('---');
  L.push('');
  L.push('### 方法论');
  L.push('- **梯度检验**：中心差分 `(L(θ+ε)−L(θ−ε))/2ε`，判据 `|ana−num| ≤ 1e-7 + 1e-4·max(|ana|,|num|)`，每项抽 ≤60 参数（GQA 500）。');
  L.push('- **RoPE 不变性**：整体平移后任意两位置注意力分数差异应恰为 0（旋转的正交性）。');
  L.push('- **量化保真**：对称逐张量 INT8（weight-only PTQ），误差纯为舍入，理论有界、解码无损。');
  L.push('- **KV 加速比**：生成 N token 时增量(forwardStep)总耗时 vs 每次全量重算(forward)总耗时之比（5 次取最优）。');
  L.push('');
  L.push('*作者：晨星 · License: MIT*');
  return L.join('\n');
}

const md = renderMarkdown();
const outPath = path.join(__dirname, '..', 'docs', 'BENCHMARK.md');
fs.writeFileSync(outPath, md);
console.log(md);
console.log(`\n报告已写入 ${path.relative(process.cwd(), outPath)}`);
process.exitCode = allPass && rows.every(r => r.targetPass) ? 0 : 1;

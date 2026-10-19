/* ============================================================================
 * 旗舰 AI · v9 SWA（滑动窗口注意力）专项验证
 * ----------------------------------------------------------------------------
 *  1. SWA 中心差分梯度检验（窗口掩码下游走反向传播，解析 vs 数值，err/tol ≤ 1）
 *  2. 窗口局部性（深度锥语义：窗外扰动对末位 logits 零影响，窗内扰动必须可见）
 *  3. SWA KV Cache 逐步一致性（forwardStep 窗口切片 == forward 全序列末位，逐位）
 *  4. window=0 零回归（与不传 window 逐位一致）
 *  5. window ≥ T 等价全因果（窗口覆盖全序列时行为不变）
 *  6. SWA 可训练性（小语料过拟合 loss -> <0.1）
 *  7. SWA 生成：KV Cache / 全序列重算 贪心解码逐位一致
 *  8. SWA 持久化 + INT8 量化往返（cfg.window 保留，量化副本 window 不丢）
 *
 * 运行：node scripts/probe-swa.js   （全通过 exit 0）
 * 作者：晨星  ·  License: MIT
 * ========================================================================*/
const F = require('../src/flagship-engine.js');

/* 1. 中心差分梯度检验：RoPE + SwiGLU + RMSNorm + GQA + 滑动窗口全组合 */
function gradCheckSWA() {
  const V = 11, T = 8, W = 4;
  const model = new F.GPTModel({
    vocabSize: V, blockSize: 12, nLayer: 2, nHead: 4, nKvHead: 2, window: W,
    dModel: 16, norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope', seed: 7,
  });
  const rng = F.mulberry32(5);
  const tokens = [];
  for (let i = 0; i < T; i++) tokens.push(Math.floor(rng() * V));
  const targets = [];
  for (let i = 1; i < T; i++) targets.push(tokens[i]);
  const { logits, cache } = model.forward(tokens);
  const { dlogits } = F.crossEntropy(logits, targets, T - 1, V, T);
  model.backward(cache, dlogits);
  const lossOf = () => F.crossEntropy(model.forward(tokens).logits, targets, T - 1, V).loss;

  const all = [];
  for (const key in model.params) {
    const p = model.params[key].d;
    for (let i = 0; i < p.length; i++) all.push([key, i]);
  }
  const srng = F.mulberry32(99);
  let cand = all;
  if (all.length > 500) {
    cand = all.slice();
    for (let i = cand.length - 1; i > 0; i--) {
      const j = Math.floor(srng() * (i + 1));
      [cand[i], cand[j]] = [cand[j], cand[i]];
    }
    cand = cand.slice(0, 500);
  }
  const e = 1e-5;
  let maxRatio = 0, checked = 0, skipped = 0;
  for (const [key, i] of cand) {
    const p = model.params[key].d;
    const orig = p[i];
    p[i] = orig + e; const lp = lossOf();
    p[i] = orig - e; const lm = lossOf();
    p[i] = orig;
    const num = (lp - lm) / (2 * e);
    const ana = model.grads[key][i];
    if (Math.abs(ana) < 1e-12 && Math.abs(num) < 1e-12) { skipped++; continue; }
    const err = Math.abs(ana - num);
    const tol = 1e-7 + 1e-4 * Math.max(Math.abs(ana), Math.abs(num));
    const ratio = err / tol;
    if (ratio > maxRatio) maxRatio = ratio;
    checked++;
  }
  return { pass: checked > 100 && maxRatio <= 1, detail: `SWA 梯度检验 ${checked} 参数(跳过 ${skipped} 零梯度) 最大 err/tol=${maxRatio.toExponential(2)}` };
}

/* 2. 窗口局部性：L 层窗口 w 的感受野为 (w-1)·L+1，
 *    距末位超过该距离的 token 扰动对末位 logits 必须零影响（逐位），
 *    落入感受野的扰动必须可见。 */
function locality() {
  const V = 11, T = 12, W = 4, L = 2;
  const mk = () => new F.GPTModel({
    vocabSize: V, blockSize: 16, nLayer: L, nHead: 4, window: W,
    dModel: 16, norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope', seed: 11,
  });
  const rng = F.mulberry32(21);
  const base = [];
  for (let i = 0; i < T; i++) base.push(Math.floor(rng() * V));
  const baseLogits = mk().forward(base).logits;
  const reach = (W - 1) * L; // 感受野半径（层）
  const lastOff = (T - 1) * V;
  let farOk = true, farMax = 0, nearOk = true, nearMin = Infinity;
  for (let p = 0; p < T - 1; p++) {
    const pert = base.slice();
    pert[p] = (pert[p] + 5) % V; // 换成另一个 token
    const lg = mk().forward(pert).logits;
    let d = 0;
    for (let j = 0; j < V; j++) d = Math.max(d, Math.abs(lg[lastOff + j] - baseLogits[lastOff + j]));
    if (T - 1 - p > reach) { if (d !== 0) farOk = false; if (d > farMax) farMax = d; }
    else { if (d <= 1e-9) nearOk = false; if (d < nearMin) nearMin = d; }
  }
  return {
    pass: farOk && nearOk,
    detail: `窗外扰动(距末位>${reach})末位最大差异=${farMax}(要求0)；窗内扰动最小差异=${nearMin === Infinity ? 'N/A' : nearMin.toExponential(2)}(要求>1e-9)`,
  };
}

/* 3. KV Cache 逐步一致性：窗口下 forwardStep 增量 == forward 全序列重算（逐位） */
function kvStepwiseSWA() {
  const V = 13, B = 48, W = 5;
  const model = new F.GPTModel({
    vocabSize: V, blockSize: B, nLayer: 2, nHead: 4, nKvHead: 2, window: W,
    dModel: 16, norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope', seed: 3,
  });
  const seq = [1, 3, 5, 7, 9, 11, 2, 4, 6, 8, 10, 0, 12, 5, 3, 7, 1, 9, 4, 2];
  const kv = F.initKVCache(model);
  let maxDiff = 0;
  for (let i = 0; i < seq.length; i++) {
    const step = F.forwardStep(model, seq[i], i, kv).logits;
    const full = model.forward(seq.slice(0, i + 1)).logits;
    const off = i * V;
    for (let j = 0; j < V; j++) maxDiff = Math.max(maxDiff, Math.abs(step[j] - full[off + j]));
    if (maxDiff > 1e-9) break;
  }
  return { pass: maxDiff === 0, detail: `SWA 逐步 forwardStep 窗口切片 vs forward 全序列 末位 logits 最大差异 ${maxDiff}（要求 0）` };
}

/* 4. window=0 零回归：显式 0 与不传 window 必须逐位一致 */
function windowZeroRegression() {
  const cfg = { vocabSize: 13, blockSize: 12, nLayer: 2, nHead: 4, dModel: 16, seed: 5 };
  const m0 = new F.GPTModel(cfg);
  const m1 = new F.GPTModel(Object.assign({}, cfg, { window: 0 }));
  const seq = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
  const l0 = m0.forward(seq).logits, l1 = m1.forward(seq).logits;
  let maxDiff = 0;
  for (let i = 0; i < l0.length; i++) maxDiff = Math.max(maxDiff, Math.abs(l0[i] - l1[i]));
  const w0 = m1.window === 0 && m0.window === 0;
  return { pass: maxDiff === 0 && w0, detail: `window=0 vs 不传 window 前向最大差异 ${maxDiff}（要求 0）` };
}

/* 5. window ≥ T 等价全因果：窗口覆盖整条序列时行为与全因果逐位一致 */
function windowGeT() {
  const cfg = { vocabSize: 13, blockSize: 16, nLayer: 2, nHead: 4, dModel: 16, seed: 5, norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope' };
  const mFull = new F.GPTModel(Object.assign({}, cfg, { window: 0 }));
  const mBig = new F.GPTModel(Object.assign({}, cfg, { window: 100 }));
  const seq = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
  const l0 = mFull.forward(seq).logits, l1 = mBig.forward(seq).logits;
  let maxDiff = 0;
  for (let i = 0; i < l0.length; i++) maxDiff = Math.max(maxDiff, Math.abs(l0[i] - l1[i]));
  return { pass: maxDiff === 0, detail: `window=100 vs window=0（T=12）前向最大差异 ${maxDiff}（要求 0）` };
}

/* 6. 可训练性：小语料过拟合（窗口 8，感受野覆盖全诗） */
function trainableSWA() {
  const POEM = '朝辞白帝彩云间，千里江陵一日还。两岸猿声啼不住，轻舟已过万重山。';
  const tok = new F.Tokenizer(POEM);
  const ids = tok.encode(POEM);
  const model = new F.GPTModel({
    vocabSize: tok.vocabSize, blockSize: ids.length, nLayer: 2, nHead: 4, window: 8,
    dModel: 32, norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope', seed: 1,
  });
  const { history } = F.train(model, ids, { steps: 300, lr: 3e-3, warmup: 20, wd: 0, fixedBatch: ids, logEvery: 1e9 });
  const last = history[history.length - 1];
  return { pass: last < 0.1, detail: `SWA 过拟合 初始 ${history[0].toFixed(2)} -> 最终 loss ${last.toFixed(4)}（阈值 <0.1）` };
}

/* 7. 生成：KV Cache 与全序列重算在贪心解码下逐位一致 */
function generateSWA() {
  const POEM = '朝辞白帝彩云间，千里江陵一日还。两岸猿声啼不住，轻舟已过万重山。';
  const tok = new F.Tokenizer(POEM);
  const ids = tok.encode(POEM);
  const model = new F.GPTModel({
    vocabSize: tok.vocabSize, blockSize: 40, nLayer: 2, nHead: 4, window: 8,
    dModel: 32, norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope', seed: 1,
  });
  F.train(model, ids, { steps: 150, lr: 3e-3, warmup: 10, wd: 0, fixedBatch: ids, logEvery: 1e9 });
  const g1 = F.generate(model, tok, '朝辞白', { maxNew: 24, temperature: 0, seed: 7, useCache: true });
  const g2 = F.generate(model, tok, '朝辞白', { maxNew: 24, temperature: 0, seed: 7, useCache: false });
  const same = g1.ids.length === g2.ids.length && g1.ids.every((v, i) => v === g2.ids[i]);
  const finite = g1.ids.every(v => Number.isInteger(v) && v >= 0 && v < tok.vocabSize);
  return {
    pass: same && finite && g1.text.length > 0,
    detail: `贪心解码 缓存/重算 ids 一致=${same}；生成 ${g1.ids.length} token（如「${g1.text.slice(0, 12)}…」）`,
  };
}

/* 8. 持久化 + INT8 量化往返：window 配置全程不丢 */
function persistSWA() {
  const cfg = {
    vocabSize: 13, blockSize: 12, nLayer: 2, nHead: 4, nKvHead: 2, window: 6,
    dModel: 16, norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope', seed: 5,
  };
  const model = new F.GPTModel(cfg);
  const seq = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
  const l1 = model.forward(seq).logits;

  const r = F.loadModel(F.saveModel(model, null));
  const l2 = r.forward(seq).logits;
  let d1 = 0;
  for (let i = 0; i < l1.length; i++) d1 = Math.max(d1, Math.abs(l1[i] - l2[i]));

  const q = model.quantize();
  const l3 = q.forward(seq).logits;
  let d2 = 0;
  for (let i = 0; i < l1.length; i++) d2 = Math.max(d2, Math.abs(l1[i] - l3[i]));

  const presetOk = F.ARCH_PRESETS.swa && F.applyPreset({}, 'swa').window === 64;
  const allFinite = Array.from(l3).every(Number.isFinite);
  return {
    pass: r.window === 6 && d1 === 0 && q.window === 6 && allFinite && d2 < 1 && presetOk,
    detail: `save/load window=${r.window} 前向差异 ${d1}；quantize window=${q.window} F32/INT8 差异 ${d2.toFixed(4)}；swa 预设=${presetOk}`,
  };
}

const CHECKS = [
  { name: 'SWA 中心差分梯度检验', run: gradCheckSWA },
  { name: 'SWA 窗口局部性（深度锥语义）', run: locality },
  { name: 'SWA KV Cache 逐步一致性', run: kvStepwiseSWA },
  { name: 'window=0 零回归', run: windowZeroRegression },
  { name: 'window ≥ T 等价全因果', run: windowGeT },
  { name: 'SWA 可训练性（过拟合）', run: trainableSWA },
  { name: 'SWA 生成：缓存/重算一致', run: generateSWA },
  { name: 'SWA 持久化 + INT8 量化往返', run: persistSWA },
];

function safeRun(fn) {
  try { const r = fn() || {}; return { pass: r.pass !== false, detail: r.detail || '' }; }
  catch (e) { return { pass: false, detail: '异常: ' + (e && e.message) }; }
}

const results = CHECKS.map(c => ({ name: c.name, ...safeRun(c.run) }));
const W = 50;
console.log('🔬 旗舰 AI · v9 SWA（滑动窗口注意力）专项验证（Node ' + process.version + '）');
console.log('─'.repeat(W));
for (let i = 0; i < results.length; i++) {
  const r = results[i];
  console.log(` ${r.pass ? '✅' : '❌'} ${String(i + 1).padStart(2)}. ${r.name}`);
  console.log(`      ${r.detail}`);
}
console.log('─'.repeat(W));
const pass = results.filter(r => r.pass).length;
console.log(`结果：${pass}/${results.length} 通过 ${pass === results.length ? '✅ 全绿' : '❌ 存在失败'}`);
process.exitCode = pass === results.length ? 0 : 1;

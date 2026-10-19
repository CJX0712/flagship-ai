/* ============================================================================
 * 旗舰 AI · v10 层交错稀疏注意力（Gemma 2/3 局部/全局模式）专项验证
 * ----------------------------------------------------------------------------
 *  1. 混合窗口梯度检验（windows=[4,0,8]，局部/全局层交错，解析 vs 数值）
 *  2. 逐层窗口独立生效（[4,0]/[0,4]/[4,4]/[0,0] 四配置 logits 互异）
 *  3. alt 哨兵展开正确性（{alt:64} → 偶数层全局、奇数层窗口 64）
 *  4. 数组全 0 ≡ 标量 0 零回归（逐位一致）
 *  5. 单层数组 ≡ 标量（逐位一致）
 *  6. 数组长度校验（长度 ≠ nLayer 必须抛错）
 *  7. 混合窗口 KV Cache 逐步一致（forwardStep == forward 末位，逐位）
 *  8. 生成：KV Cache / 全序列重算贪心解码逐位一致
 *  9. 持久化往返（数组 + 哨兵两种写法，存档后 windows 与 logits 无损）
 * 10. INT8 量化保留逐层 windows
 * 11. 可训练性（gemma 预设过拟合 loss -> <0.1）
 * 12. applyPreset('gemma') 一行得到层交错配置
 *
 * 运行：node scripts/probe-gemma.js   （全通过 exit 0）
 * 作者：晨星  ·  License: MIT
 * ========================================================================*/
const F = require('../src/flagship-engine.js');

/* 1. 中心差分梯度检验：RoPE + SwiGLU + RMSNorm + GQA + 逐层混合窗口（局部/全局交错） */
function gradCheckInterleave() {
  const V = 11, T = 8;
  const model = new F.GPTModel({
    vocabSize: V, blockSize: 12, nLayer: 3, nHead: 4, nKvHead: 2, window: [4, 0, 8],
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
  return { pass: checked > 100 && maxRatio <= 1, detail: `层交错梯度检验 ${checked} 参数(跳过 ${skipped} 零梯度) 最大 err/tol=${maxRatio.toExponential(2)}` };
}

/* 2. 逐层窗口独立生效：四种窗口配置的 logits 必须两两互异 */
function perLayerIndependent() {
  const base = { vocabSize: 13, blockSize: 16, nLayer: 2, nHead: 4, dModel: 16, seed: 5, norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope' };
  const seq = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
  const cfgs = { '[4,0]': [4, 0], '[0,4]': [0, 4], '[4,4]': [4, 4], '[0,0]': [0, 0] };
  const lg = {};
  for (const name in cfgs) {
    const m = new F.GPTModel(Object.assign({}, base, { window: cfgs[name] }));
    lg[name] = Array.from(m.forward(seq).logits);
  }
  const names = Object.keys(lg);
  let allDiff = true, minGap = Infinity;
  for (let a = 0; a < names.length; a++) {
    for (let b = a + 1; b < names.length; b++) {
      let d = 0;
      for (let i = 0; i < lg[names[a]].length; i++) d = Math.max(d, Math.abs(lg[names[a]][i] - lg[names[b]][i]));
      if (d <= 1e-9) allDiff = false;
      if (d < minGap) minGap = d;
    }
  }
  return { pass: allDiff, detail: `[4,0]/[0,4]/[4,4]/[0,0] 四配置两两最小差异 ${minGap === Infinity ? 'N/A' : minGap.toExponential(2)}（要求 >1e-9，证明窗口逐层独立生效）` };
}

/* 3. alt 哨兵展开：{alt:N} → 偶数层全因果、奇数层窗口 N */
function altExpansion() {
  const base = { vocabSize: 13, blockSize: 16, nHead: 4, dModel: 16, norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope' };
  const m4 = new F.GPTModel(Object.assign({}, base, { nLayer: 4, window: { alt: 64 } }));
  const m1 = new F.GPTModel(Object.assign({}, base, { nLayer: 1, window: { alt: 64 } }));
  const m5 = new F.GPTModel(Object.assign({}, base, { nLayer: 5, window: { alt: 7 } }));
  const ok4 = JSON.stringify(m4.windows) === JSON.stringify([0, 64, 0, 64]);
  const ok1 = JSON.stringify(m1.windows) === JSON.stringify([0]);
  const ok5 = JSON.stringify(m5.windows) === JSON.stringify([0, 7, 0, 7, 0]);
  const cfgKept = m4.window && m4.window.alt === 64;
  return {
    pass: ok4 && ok1 && ok5 && cfgKept,
    detail: `nLayer=4 -> [${m4.windows}]；nLayer=1 -> [${m1.windows}]；nLayer=5,alt=7 -> [${m5.windows}]；cfg 哨兵保留=${cfgKept}`,
  };
}

/* 4. 数组全 0 ≡ 标量 0（零回归，逐位一致） */
function arrayZeroRegression() {
  const cfg = { vocabSize: 13, blockSize: 12, nLayer: 2, nHead: 4, dModel: 16, seed: 5 };
  const m0 = new F.GPTModel(cfg);
  const m1 = new F.GPTModel(Object.assign({}, cfg, { window: [0, 0] }));
  const seq = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
  const l0 = m0.forward(seq).logits, l1 = m1.forward(seq).logits;
  let maxDiff = 0;
  for (let i = 0; i < l0.length; i++) maxDiff = Math.max(maxDiff, Math.abs(l0[i] - l1[i]));
  return { pass: maxDiff === 0, detail: `window=[0,0] vs 不传 window 前向最大差异 ${maxDiff}（要求 0）` };
}

/* 5. 单层数组 ≡ 标量（逐位一致） */
function singleLayerArray() {
  const cfg = { vocabSize: 13, blockSize: 12, nLayer: 1, nHead: 4, dModel: 16, seed: 5, norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope' };
  const mS = new F.GPTModel(Object.assign({}, cfg, { window: 7 }));
  const mA = new F.GPTModel(Object.assign({}, cfg, { window: [7] }));
  const seq = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
  const l0 = mS.forward(seq).logits, l1 = mA.forward(seq).logits;
  let maxDiff = 0;
  for (let i = 0; i < l0.length; i++) maxDiff = Math.max(maxDiff, Math.abs(l0[i] - l1[i]));
  return { pass: maxDiff === 0 && mA.windows[0] === 7, detail: `window:[7] vs window:7（nLayer=1）前向最大差异 ${maxDiff}（要求 0）` };
}

/* 6. 数组长度校验：长度 ≠ nLayer 必须抛错 */
function lengthValidation() {
  const cfg = { vocabSize: 13, blockSize: 12, nLayer: 2, nHead: 4, dModel: 16, seed: 5 };
  let threw = false;
  try { new F.GPTModel(Object.assign({}, cfg, { window: [4] })); }
  catch (e) { threw = /window 数组长度/.test(e.message); }
  let threw2 = false;
  try { new F.GPTModel(Object.assign({}, cfg, { window: [4, 0, 4] })); }
  catch (e) { threw2 = /window 数组长度/.test(e.message); }
  const ok = new F.GPTModel(Object.assign({}, cfg, { window: [4, 0] })).windows.length === 2;
  return { pass: threw && threw2 && ok, detail: `[4] 与 [4,0,4]（nLayer=2）均抛错=${threw && threw2}；[4,0] 正常构建=${ok}` };
}

/* 7. 混合窗口 KV Cache 逐步一致：forwardStep 增量 == forward 全序列末位（逐位） */
function kvStepwiseMixed() {
  const V = 13, B = 48;
  const model = new F.GPTModel({
    vocabSize: V, blockSize: B, nLayer: 3, nHead: 4, nKvHead: 2, window: [5, 0, 3],
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
  return { pass: maxDiff === 0, detail: `层交错逐步 forwardStep vs forward 全序列 末位 logits 最大差异 ${maxDiff}（要求 0）` };
}

/* 8. 生成：KV Cache 与全序列重算在贪心解码下逐位一致 */
function generateInterleave() {
  const POEM = '朝辞白帝彩云间，千里江陵一日还。两岸猿声啼不住，轻舟已过万重山。';
  const tok = new F.Tokenizer(POEM);
  const ids = tok.encode(POEM);
  const model = new F.GPTModel({
    vocabSize: tok.vocabSize, blockSize: 40, nLayer: 4, nHead: 4, nKvHead: 2,
    window: [0, 8, 0, 8], dModel: 32, norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope', seed: 1,
  });
  F.train(model, ids, { steps: 150, lr: 3e-3, warmup: 10, wd: 0, fixedBatch: ids, logEvery: 1e9 });
  const g1 = F.generate(model, tok, '朝辞白', { maxNew: 24, temperature: 0, seed: 7, useCache: true });
  const g2 = F.generate(model, tok, '朝辞白', { maxNew: 24, temperature: 0, seed: 7, useCache: false });
  const same = g1.ids.length === g2.ids.length && g1.ids.every((v, i) => v === g2.ids[i]);
  return {
    pass: same && g1.text.length > 0,
    detail: `贪心解码 缓存/重算 ids 一致=${same}；生成 ${g1.ids.length} token（如「${g1.text.slice(0, 12)}…」）`,
  };
}

/* 9. 持久化往返：数组与哨兵两种写法存档后 windows 与 logits 均无损 */
function persistInterleave() {
  const base = { vocabSize: 13, blockSize: 12, nLayer: 4, nHead: 4, nKvHead: 2, dModel: 16, norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope', seed: 5 };
  const seq = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
  const cases = [
    { name: '数组 [6,0,6,0]', cfg: { window: [6, 0, 6, 0] }, expect: [6, 0, 6, 0] },
    { name: '哨兵 {alt:8}', cfg: { window: { alt: 8 } }, expect: [0, 8, 0, 8] },
  ];
  const details = [];
  let allOk = true;
  for (const c of cases) {
    const model = new F.GPTModel(Object.assign({}, base, c.cfg));
    const l1 = model.forward(seq).logits;
    const r = F.loadModel(F.saveModel(model, null));
    const l2 = r.forward(seq).logits;
    let d = 0;
    for (let i = 0; i < l1.length; i++) d = Math.max(d, Math.abs(l1[i] - l2[i]));
    const wOk = JSON.stringify(r.windows) === JSON.stringify(c.expect);
    if (!(wOk && d === 0)) allOk = false;
    details.push(`${c.name}: windows=[${r.windows}] 差异 ${d}`);
  }
  return { pass: allOk, detail: details.join('；') };
}

/* 10. INT8 量化保留逐层 windows */
function quantizeInterleave() {
  const model = new F.GPTModel({
    vocabSize: 13, blockSize: 12, nLayer: 3, nHead: 4, nKvHead: 2, window: [4, 0, 8],
    dModel: 16, norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope', seed: 5,
  });
  const q = model.quantize();
  const seq = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
  const l1 = model.forward(seq).logits, l3 = q.forward(seq).logits;
  let d = 0;
  for (let i = 0; i < l1.length; i++) d = Math.max(d, Math.abs(l1[i] - l3[i]));
  const finite = Array.from(l3).every(Number.isFinite);
  const wOk = JSON.stringify(q.windows) === JSON.stringify([4, 0, 8]);
  return { pass: wOk && finite && d < 1, detail: `quantize windows=[${q.windows}]（保留=${wOk}）；F32/INT8 差异 ${d.toFixed(4)}` };
}

/* 11. 可训练性：gemma 预设（alt 交错）过拟合 */
function trainableGemma() {
  const POEM = '朝辞白帝彩云间，千里江陵一日还。两岸猿声啼不住，轻舟已过万重山。';
  const tok = new F.Tokenizer(POEM);
  const ids = tok.encode(POEM);
  const model = new F.GPTModel(F.applyPreset(
    { vocabSize: tok.vocabSize, blockSize: ids.length, nLayer: 4, nHead: 4, dModel: 32, seed: 1 }, 'gemma'));
  Object.assign(model.cfg, { window: { alt: 8 } }); // 演示窗口取 8（预设 64 对 28 字小语料过宽）
  const m = new F.GPTModel(model.cfg);
  const { history } = F.train(m, ids, { steps: 300, lr: 3e-3, warmup: 20, wd: 0, fixedBatch: ids, logEvery: 1e9 });
  const last = history[history.length - 1];
  return { pass: last < 0.1 && JSON.stringify(m.windows) === JSON.stringify([0, 8, 0, 8]), detail: `gemma 过拟合 初始 ${history[0].toFixed(2)} -> 最终 loss ${last.toFixed(4)}；windows=[${m.windows}]（阈值 <0.1）` };
}

/* 12. applyPreset('gemma') 一行得到层交错配置 */
function presetGemma() {
  const cfg = F.applyPreset({ vocabSize: 50, blockSize: 16, nLayer: 4, nHead: 4, dModel: 32 }, 'gemma');
  const m = new F.GPTModel(cfg);
  const fieldsOk = cfg.norm === 'rmsnorm' && cfg.ffn === 'swiglu' && cfg.pos === 'rope' && cfg.nKvHead === 2;
  const wOk = JSON.stringify(m.windows) === JSON.stringify([0, 64, 0, 64]);
  return { pass: fieldsOk && wOk, detail: `gemma 预设字段=${fieldsOk}；windows=[${m.windows}]（偶数层全局/奇数层窗口 64）` };
}

const CHECKS = [
  { name: '层交错混合窗口梯度检验', run: gradCheckInterleave },
  { name: '逐层窗口独立生效', run: perLayerIndependent },
  { name: 'alt 哨兵展开正确性', run: altExpansion },
  { name: '数组全 0 零回归', run: arrayZeroRegression },
  { name: '单层数组 ≡ 标量', run: singleLayerArray },
  { name: '数组长度校验', run: lengthValidation },
  { name: '层交错 KV Cache 逐步一致', run: kvStepwiseMixed },
  { name: '生成：缓存/重算一致', run: generateInterleave },
  { name: '持久化往返（数组+哨兵）', run: persistInterleave },
  { name: 'INT8 量化保留逐层 windows', run: quantizeInterleave },
  { name: 'gemma 预设可训练性', run: trainableGemma },
  { name: "applyPreset('gemma')", run: presetGemma },
];

function safeRun(fn) {
  try { const r = fn() || {}; return { pass: r.pass !== false, detail: r.detail || '' }; }
  catch (e) { return { pass: false, detail: '异常: ' + (e && e.message) }; }
}

const results = CHECKS.map(c => ({ name: c.name, ...safeRun(c.run) }));
const W = 50;
console.log('🔬 旗舰 AI · v10 层交错稀疏注意力（Gemma 2/3）专项验证（Node ' + process.version + '）');
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

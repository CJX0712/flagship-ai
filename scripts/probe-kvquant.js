/* ============================================================================
 * 旗舰 AI · v12 KV Cache INT8 量化专项验证
 * ----------------------------------------------------------------------------
 *  1. 零回归（kvQuant=0 ≡ 默认 Float64 缓存，逐位一致）
 *  2. 量化误差硬界：|dequant(x) − x| ≤ absmax_head/254（相对 ≤ 1/254 ≈ 0.39%）
 *  3. 每头 absmax 无损：最大码字恰为 ±127 ⇒ 解量化后 per-head absmax 相对误差 ~1e-16
 *  4. INT8 KV vs F32 KV 逐步 logits 差异有界（远小于 logits 自身动态范围）
 *  5. 贪心解码：INT8 KV 与 F32 KV 生成 token 序列完全一致
 *  6. kvShift（超 blockSize 平移）后 scale 与码字同步，误差仍有界
 *  7. 与 INT8 权重量化组合（weight-only PTQ + KV INT8 双重量化）
 *  8. 与 SWA 滑动窗口组合
 *  9. 与层交错窗口 + GQA 组合
 * 10. 全栈组合：MoE + GQA + 层交错 + QK-norm + soft-cap + KV INT8
 * 11. 缓存显存实测（Float64 vs Int8+scale 字节数比）
 * 12. 端到端生成（INT8 KV 生成非空、无 NaN/Inf、可重复）
 *
 * 运行：node scripts/probe-kvquant.js   （全通过 exit 0）
 * 作者：晨星  ·  License: MIT
 * ========================================================================*/
const F = require('../src/flagship-engine.js');

const POEM = '朝辞白帝彩云间，千里江陵一日还。两岸猿声啼不住，轻舟已过万重山。';

/** 用同一份权重跑两遍缓存：Float64 真值 vs INT8 量化，返回两者的末位 logits 与缓存 */
function runPair(model, ids) {
  model.kvQuant = 0;
  const kvA = F.initKVCache(model);
  let lgA = null;
  for (let i = 0; i < ids.length; i++) lgA = F.forwardStep(model, ids[i], i, kvA).logits;

  model.kvQuant = 8;
  const kvB = F.initKVCache(model);
  let lgB = null;
  for (let i = 0; i < ids.length; i++) lgB = F.forwardStep(model, ids[i], i, kvB).logits;

  return { lgA, lgB, kvA, kvB };
}

function maxAbsDiff(a, b) {
  let m = 0;
  for (let i = 0; i < a.length; i++) { const d = Math.abs(a[i] - b[i]); if (d > m) m = d; }
  return m;
}

/* 1. 零回归：kvQuant=0 时 forwardStep 与 forward 全序列末位逐位一致 */
function offRegression() {
  const V = 17, T = 9;
  const model = new F.GPTModel({
    vocabSize: V, blockSize: 16, nLayer: 2, nHead: 4, nKvHead: 2, dModel: 32,
    norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope', kvQuant: 0, seed: 3,
  });
  const rng = F.mulberry32(11);
  const ids = [];
  for (let i = 0; i < T; i++) ids.push(Math.floor(rng() * V));
  // forward 返回 T×V logits，取末行（末位）与逐步 forwardStep 比对
  const fullRaw = model.forward(ids).logits;
  const full = fullRaw.slice((T - 1) * V, T * V);
  const kv = F.initKVCache(model);
  let step = null;
  for (let i = 0; i < T; i++) step = F.forwardStep(model, ids[i], i, kv).logits;
  const d = maxAbsDiff(full, step);
  return {
    pass: d === 0 && model.kvQuant === 0 && kv.q8 === false,
    detail: `kvQuant=0 逐步 forwardStep vs forward 末位 logits 最大差异 ${d}（要求 0）· 缓存为 Float64（q8=${kv.q8}）`,
  };
}

/* 2+3. 量化误差硬界与每头 absmax 无损
 *  取第 0 层缓存（输入 X 与 FFN 路径完全不受缓存存储影响，故 L0 的 K/V 真值可比）；
 *  解量化 INT8 缓存与 Float64 真值逐元素比对，断言 |err| ≤ absmax_head/254。 */
function quantErrorBound() {
  const V = 17, T = 10, H = 4, KV = 2, dh = 8;
  const model = new F.GPTModel({
    vocabSize: V, blockSize: 16, nLayer: 3, nHead: H, nKvHead: KV, dModel: 32,
    norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope', seed: 5,
  });
  const rng = F.mulberry32(23);
  const ids = [];
  for (let i = 0; i < T; i++) ids.push(Math.floor(rng() * V));
  const { kvA, kvB } = runPair(model, ids);
  const kvDim = KV * dh;
  let worstRatio = 0, worstAbsRel = 0, worstHeadRel = 0, nZero = 0;
  for (let i = 0; i < T; i++) {
    for (let hh = 0; hh < KV; hh++) {
      // 真值的 per-head absmax（Float64 缓存）
      let ab = 0;
      for (let p = 0; p < dh; p++) { const v = Math.abs(kvA.K[0][i * kvDim + hh * dh + p]); if (v > ab) ab = v; }
      if (ab === 0) { nZero++; continue; }
      const sc = kvB.Ks[0][i * KV + hh];
      // 2) 逐元素误差 / (absmax/254) 必须 ≤ 1
      for (let p = 0; p < dh; p++) {
        const truth = kvA.K[0][i * kvDim + hh * dh + p];
        const deq = kvB.K[0][i * kvDim + hh * dh + p] * sc;
        const ratio = Math.abs(deq - truth) / (ab / 254);
        if (ratio > worstRatio) worstRatio = ratio;
      }
      // 3) 解量化后 per-head absmax 应与真值 absmax 相等（最大码字恰 ±127）
      let abD = 0;
      for (let p = 0; p < dh; p++) { const v = Math.abs(kvB.K[0][i * kvDim + hh * dh + p] * sc); if (v > abD) abD = v; }
      const rel = Math.abs(abD - ab) / ab;
      if (rel > worstHeadRel) worstHeadRel = rel;
      // scale 定义自洽：sc 应等于 ab/127
      const scRel = Math.abs(sc - ab / 127) / (ab / 127);
      if (scRel > worstAbsRel) worstAbsRel = scRel;
    }
  }
  const pass = worstRatio <= 1 + 1e-9 && worstHeadRel < 1e-12 && worstAbsRel < 1e-12;
  return {
    pass,
    detail: `逐元素误差/理论界(absmax/254) 最大 ${worstRatio.toFixed(4)}（要求 ≤1）· 解量化后 per-head absmax 相对误差 ${worstHeadRel.toExponential(2)} · scale=absmax/127 自洽误差 ${worstAbsRel.toExponential(2)}（全零头 ${nZero} 个）`,
  };
}

/* 4. INT8 KV vs F32 KV 逐步末位 logits 差异有界 */
function logitsBounded() {
  const V = 17, T = 12;
  const model = new F.GPTModel({
    vocabSize: V, blockSize: 24, nLayer: 3, nHead: 4, nKvHead: 2, dModel: 32,
    norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope', seed: 9,
  });
  const rng = F.mulberry32(41);
  const ids = [];
  for (let i = 0; i < T; i++) ids.push(Math.floor(rng() * V));
  const { lgA, lgB } = runPair(model, ids);
  const d = maxAbsDiff(lgA, lgB);
  let span = -Infinity, lo = Infinity;
  for (let i = 0; i < lgA.length; i++) { if (lgA[i] > span) span = lgA[i]; if (lgA[i] < lo) lo = lgA[i]; }
  const rel = d / (span - lo);
  const finite = lgB.every(Number.isFinite);
  return {
    pass: d < 0.05 && rel < 0.02 && finite,
    detail: `INT8 KV vs F32 KV 末位 logits maxΔ=${d.toExponential(3)}（要求 <0.05）· 占 logits 动态范围 ${(rel * 100).toFixed(3)}%（要求 <2%）· 无 NaN/Inf=${finite}`,
  };
}

/* 5. 贪心解码：INT8 KV 与 F32 KV 生成 token 序列完全一致 */
function greedyIdentical() {
  const tok = new F.BPETokenizer(POEM, 120, { byteLevel: false });
  const base = { vocabSize: tok.vocabSize, blockSize: 64, nLayer: 2, nHead: 4, nKvHead: 2, dModel: 48, norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope', seed: 17 };
  const m = new F.GPTModel(Object.assign({}, base));
  const ids = tok.encode(POEM);
  F.train(m, ids, { steps: 400, lr: 3e-3, warmup: 20, wd: 0, fixedBatch: ids, logEvery: 1e9 });
  const opt = { maxNew: 30, temperature: 0, seed: 99 };
  m.kvQuant = 0; const gA = F.generate(m, tok, '朝辞', opt);
  m.kvQuant = 8; const gB = F.generate(m, tok, '朝辞', opt);
  const same = gA.ids.length === gB.ids.length && gA.ids.every((v, i) => v === gB.ids[i]) && gB.ids.length > 1;
  return { pass: same, detail: `贪心解码 F32-KV ${JSON.stringify(gA.text)} vs INT8-KV ${JSON.stringify(gB.text)} · ${gB.ids.length} token 逐字一致=${same}` };
}

/* 6. kvShift：写满 blockSize 后平移，scale 与码字同步，误差仍有界 */
function shiftConsistent() {
  const V = 13, B = 8;
  const model = new F.GPTModel({
    vocabSize: V, blockSize: B, nLayer: 2, nHead: 4, nKvHead: 2, dModel: 32,
    norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope', seed: 13,
  });
  const rng = F.mulberry32(7);
  const ids = [];
  for (let i = 0; i < B + 4; i++) ids.push(Math.floor(rng() * V)); // 超长 → 触发 kvShift
  model.kvQuant = 8;
  const kv = F.initKVCache(model);
  let last = null;
  for (let i = 0; i < ids.length; i++) {
    if (i >= B) F.kvShift(model, kv);
    last = F.forwardStep(model, ids[i], i, kv).logits;
  }
  // 对照：Float64 缓存同样流程
  model.kvQuant = 0;
  const kvA = F.initKVCache(model);
  let lastA = null;
  for (let i = 0; i < ids.length; i++) {
    if (i >= B) F.kvShift(model, kvA);
    lastA = F.forwardStep(model, ids[i], i, kvA).logits;
  }
  const d = maxAbsDiff(last, lastA);
  const finite = last.every(Number.isFinite) && kv.Ks[0].every(Number.isFinite);
  return {
    pass: d < 0.05 && finite && kv.n === B,
    detail: `${ids.length} token（blockSize=${B}，触发 ${ids.length - B} 次 kvShift）：INT8 vs F32 末位 logits maxΔ=${d.toExponential(3)}（要求 <0.05）· scale 无 NaN=${finite} · kv.n=${kv.n}`,
  };
}

/* 7. 与 INT8 权重量化组合（weight-only PTQ + KV INT8 双重量化） */
function withWeightQuant() {
  const V = 17, T = 10;
  const model = new F.GPTModel({
    vocabSize: V, blockSize: 20, nLayer: 2, nHead: 4, nKvHead: 2, dModel: 32,
    norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope', seed: 21,
  });
  const rng = F.mulberry32(31);
  const ids = [];
  for (let i = 0; i < T; i++) ids.push(Math.floor(rng() * V));
  // 基准：FP32 权重 + F32 KV
  model.kvQuant = 0;
  const kvRef = F.initKVCache(model);
  let lgRef = null;
  for (let i = 0; i < T; i++) lgRef = F.forwardStep(model, ids[i], i, kvRef).logits;

  const q = model.quantize(); // INT8 权重（应携带 kvQuant=0）
  const keptField = q.kvQuant === 0;
  q.kvQuant = 0;
  const kvQ0 = F.initKVCache(q);
  let lgQ0 = null;
  for (let i = 0; i < T; i++) lgQ0 = F.forwardStep(q, ids[i], i, kvQ0).logits;

  q.kvQuant = 8;
  const kvQ8 = F.initKVCache(q);
  let lgQ8 = null;
  for (let i = 0; i < T; i++) lgQ8 = F.forwardStep(q, ids[i], i, kvQ8).logits;

  const dW = maxAbsDiff(lgRef, lgQ0);   // 仅权重量化引入的误差
  const dBoth = maxAbsDiff(lgRef, lgQ8); // 权重 + KV 双重量化
  return {
    pass: keptField && dW < 0.5 && dBoth < 0.5 && dBoth < dW * 3 + 1e-6,
    detail: `INT8 权重 maxΔ=${dW.toExponential(3)}；权重+KV 双 INT8 maxΔ=${dBoth.toExponential(3)}（叠加仍在同量级）· quantize 保留 kvQuant 字段=${keptField}`,
  };
}

/* 8. 与 SWA 滑动窗口组合 */
function withSWA() {
  const V = 15, T = 14;
  const model = new F.GPTModel({
    vocabSize: V, blockSize: 24, nLayer: 2, nHead: 4, nKvHead: 2, dModel: 32,
    norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope', window: 4, seed: 27,
  });
  const rng = F.mulberry32(13);
  const ids = [];
  for (let i = 0; i < T; i++) ids.push(Math.floor(rng() * V));
  const { lgA, lgB } = runPair(model, ids);
  const d = maxAbsDiff(lgA, lgB);
  return { pass: d < 0.05, detail: `SWA(window=4) + KV INT8：末位 logits maxΔ=${d.toExponential(3)}（要求 <0.05）· windows=[${model.windows}]` };
}

/* 9. 与层交错窗口 + GQA 组合 */
function withInterleave() {
  const V = 15, T = 14;
  const model = new F.GPTModel({
    vocabSize: V, blockSize: 24, nLayer: 4, nHead: 4, nKvHead: 2, dModel: 32,
    norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope', window: { alt: 6 }, seed: 33,
  });
  const rng = F.mulberry32(19);
  const ids = [];
  for (let i = 0; i < T; i++) ids.push(Math.floor(rng() * V));
  const { lgA, lgB } = runPair(model, ids);
  const d = maxAbsDiff(lgA, lgB);
  return { pass: d < 0.05, detail: `层交错 windows=[${model.windows}] + GQA(2 KV头) + KV INT8：maxΔ=${d.toExponential(3)}（要求 <0.05）` };
}

/* 10. 全栈组合：MoE + GQA + 层交错 + QK-norm + soft-cap + KV INT8 */
function fullStack() {
  const tok = new F.BPETokenizer(POEM, 120, { byteLevel: false });
  const m = new F.GPTModel({
    vocabSize: tok.vocabSize, blockSize: 64, nLayer: 4, nHead: 4, nKvHead: 2, dModel: 48,
    norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope', window: { alt: 8 },
    nExp: 4, topK: 2, qkNorm: true, logitCap: 30, seed: 37,
  });
  const ids = tok.encode(POEM).slice(0, 20);
  const { lgA, lgB } = runPair(m, ids);
  const d = maxAbsDiff(lgA, lgB);
  let mx = 0;
  for (let i = 0; i < lgB.length; i++) mx = Math.max(mx, Math.abs(lgB[i]));
  const bounded = mx <= 30 + 1e-9; // soft-cap 在 KV 量化下仍严格有界
  const finite = lgB.every(Number.isFinite);
  return {
    pass: d < 0.05 && bounded && finite,
    detail: `MoE(4专家/top2)+GQA+层交错[${m.windows}]+QK-norm+soft-cap(30)+KV INT8：maxΔ=${d.toExponential(3)} · max|logit|=${mx.toFixed(3)}≤30 · 有限=${finite}`,
  };
}

/* 11. 缓存显存实测：Float64 vs Int8+scale 字节数比 */
function memoryFootprint() {
  const L = 4, B = 64, KVh = 2, dh = 16;
  const model = new F.GPTModel({
    vocabSize: 20, blockSize: B, nLayer: L, nHead: 8, nKvHead: KVh, dModel: 128, seed: 1,
  });
  const kvDim = KVh * dh;
  const f64 = L * 2 * B * kvDim * 8;                 // K、V 各 Float64Array
  const i8 = L * 2 * (B * kvDim * 1 + B * KVh * 8); // 码字 1 字节 + per-(token,head) scale
  const ratio = f64 / i8;
  // 实际分配校验
  model.kvQuant = 8;
  const kv8 = F.initKVCache(model);
  const realI8 = kv8.K[0].BYTES_PER_ELEMENT === 1 && kv8.Ks[0].BYTES_PER_ELEMENT === 8;
  model.kvQuant = 0;
  const kv0 = F.initKVCache(model);
  const realF64 = kv0.K[0].BYTES_PER_ELEMENT === 8 && kv0.Ks.length === 0 && kv0.q8 === false;
  return {
    pass: ratio > 3 && realI8 && realF64,
    detail: `L=${L} B=${B} KV头=${KVh} dh=${dh}（kvDim=${kvDim}）：Float64 ${(f64 / 1024).toFixed(1)} KB → INT8+scale ${(i8 / 1024).toFixed(1)} KB，省 ${ratio.toFixed(2)}x · 实际类型 Int8Array/Float64Array 校验=${realI8 && realF64}`,
  };
}

/* 12. 端到端生成：INT8 KV 生成非空、无 NaN、两次一致（确定性） */
function generateEndToEnd() {
  const tok = new F.BPETokenizer(POEM, 120, { byteLevel: false });
  const m = new F.GPTModel({
    vocabSize: tok.vocabSize, blockSize: 64, nLayer: 2, nHead: 4, nKvHead: 2, dModel: 48,
    norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope', window: 8, kvQuant: 8, seed: 41,
  });
  const ids = tok.encode(POEM);
  F.train(m, ids, { steps: 500, lr: 3e-3, warmup: 20, wd: 0, fixedBatch: ids, logEvery: 1e9 });
  const a = F.generate(m, tok, '朝辞白帝', { maxNew: 40, temperature: 0.9, seed: 5 });
  const b = F.generate(m, tok, '朝辞白帝', { maxNew: 40, temperature: 0.9, seed: 5 });
  const identical = a.ids.length === b.ids.length && a.ids.every((v, i) => v === b.ids[i]);
  const ok = a.ids.length > 4 && identical && a.text.length > 0 && !/undefined|NaN/.test(a.text);
  return { pass: ok, detail: `INT8 KV 采样生成 ${JSON.stringify(a.text)} · ${a.ids.length} token · 同种子可复现=${identical}` };
}

const CHECKS = [
  { name: 'kvQuant=0 零回归（逐位）', run: offRegression },
  { name: '量化误差硬界 + per-head absmax 无损', run: quantErrorBound },
  { name: 'INT8 KV vs F32 KV logits 差异有界', run: logitsBounded },
  { name: '贪心解码 INT8 KV ≡ F32 KV', run: greedyIdentical },
  { name: 'kvShift 后 scale 同步 + 误差有界', run: shiftConsistent },
  { name: '× INT8 权重量化（双重 PTQ）', run: withWeightQuant },
  { name: '× SWA 滑动窗口', run: withSWA },
  { name: '× 层交错窗口 + GQA', run: withInterleave },
  { name: '× 全栈（MoE+GQA+交错+QK-norm+cap）', run: fullStack },
  { name: '缓存显存实测（Float64 vs Int8+scale）', run: memoryFootprint },
  { name: '端到端生成（INT8 KV 确定性）', run: generateEndToEnd },
];

function safeRun(fn) {
  try { const r = fn() || {}; return { pass: r.pass !== false, detail: r.detail || '' }; }
  catch (e) { return { pass: false, detail: '异常: ' + (e && e.message) }; }
}

const results = CHECKS.map(c => ({ name: c.name, ...safeRun(c.run) }));
const W = 52;
console.log('🔬 旗舰 AI · v12 KV Cache INT8 量化验证（Node ' + process.version + '）');
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

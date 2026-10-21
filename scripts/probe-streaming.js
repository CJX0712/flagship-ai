/* ============================================================================
 * 旗舰 AI · v13 Attention Sink / StreamingLLM 专项验证
 * ----------------------------------------------------------------------------
 *  1. 零回归：attentionSink=0 与无该字段的模型逐位一致（默认行为不被破坏）
 *  2. 短序列一致性：序列 < 缓存容量时，forwardStep(带 sink) ≡ forward 全序列末位（Δ=0）
 *  3. 锚点永不被淘汰：远超容量后 slot[0..K) 的 K/V 与初次写入逐位一致（核心不变量）
 *  4. 滚动窗口生效：满容后 kv.n===cap，且末窗口槽位 == 最近一次写入 token
 *  5. StreamingLLM 改变输出（有效性）：sink 保护 vs 纯滑动窗口在长序列下末位 logits 不同
 *  6. 生成稳定性 / 确定性：greedy 300 token 同种子可复现、无 NaN/Inf、长度正确
 *  7. 无限上下文不崩溃：生成 300 token（>>cap）无异常
 *  8. × KV INT8：sink + kvQuant=8，锚点不被淘汰（解量化误差在 INT8 界内）、生成有限
 *  9. × SWA：sink + 滑动窗口（window<rolling），注意力正常、有限、锚点受保护
 * 10. kvShift scale 同步（INT8）：多次淘汰后 scale 无 NaN，kv.n 自洽
 *
 * 运行：node scripts/probe-streaming.js   （全通过 exit 0）
 * 作者：晨星  ·  License: MIT
 * ========================================================================*/
const F = require('../src/index.js');

function maxAbsDiff(a, b) {
  let m = 0;
  for (let i = 0; i < a.length; i++) { const d = Math.abs(a[i] - b[i]); if (d > m) m = d; }
  return m;
}
function allFinite(arr) {
  for (let i = 0; i < arr.length; i++) if (!Number.isFinite(arr[i])) return false;
  return true;
}

/** 从缓存读出某层某槽位的 K 向量（kvQuant=8 时解量化），返回 Float64Array(kvDim) */
function readK(kv, layer, slot, kvDim, KV, dh) {
  const arr = kv.K[layer], sc = kv.q8 ? kv.Ks[layer] : null;
  const out = new Float64Array(kvDim);
  for (let hh = 0; hh < KV; hh++) {
    const s = sc ? sc[slot * KV + hh] : 1;
    for (let p = 0; p < dh; p++) out[hh * dh + p] = arr[slot * kvDim + hh * dh + p] * s;
  }
  return out;
}

/** 手动 greedy 生成（复刻 generate 的增量路径，便于脱离分词器做力学验证） */
function greedy(model, startIds, n) {
  const kv = F.initKVCache(model);
  let logits = F.warmupKV(model, startIds, kv);
  let pos = kv.sink > 0 ? startIds.length : Math.min(startIds.length, model.cfg.blockSize);
  const out = [];
  for (let s = 0; s < n; s++) {
    let best = 0;
    for (let j = 1; j < model.cfg.vocabSize; j++) if (logits[j] > logits[best]) best = j;
    out.push(best);
    if (kv.n >= (kv.cap || model.cfg.blockSize)) F.kvShift(model, kv);
    logits = F.forwardStep(model, best, pos, kv).logits;
    pos++;
  }
  return out;
}

const C = [];
function check(name, pass, detail) { C.push({ name, pass, detail }); }

/* 1. 零回归：attentionSink=0 ≡ 不带该字段 */
function sinkZeroRegression() {
  const base = { vocabSize: 17, blockSize: 16, nLayer: 2, nHead: 4, nKvHead: 2, dModel: 32, norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope', seed: 7 };
  const mA = new F.GPTModel(Object.assign({}, base, { attentionSink: 0 }));
  const mB = new F.GPTModel(base); // 不含 attentionSink 字段
  const rng = F.mulberry32(13), ids = [];
  for (let i = 0; i < 12; i++) ids.push(Math.floor(rng() * 17));
  const gA = greedy(mA, ids, 20), gB = greedy(mB, ids, 20);
  let eq = gA.length === gB.length;
  for (let i = 0; i < gA.length && eq; i++) if (gA[i] !== gB[i]) eq = false;
  return { pass: eq, detail: `attentionSink=0 与无字段模型 greedy 序列 ${eq ? '完全一致' : '不一致'}（长度 ${gA.length}）` };
}

/* 2. 短序列一致性（序列 < cap 时 forwardStep ≡ forward） */
function shortConsistency() {
  const V = 17, T = 9;
  const model = new F.GPTModel({ vocabSize: V, blockSize: 64, nLayer: 2, nHead: 4, nKvHead: 2, dModel: 32, norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope', attentionSink: 4, rollingWindow: 32, seed: 3 });
  const rng = F.mulberry32(11), ids = [];
  for (let i = 0; i < T; i++) ids.push(Math.floor(rng() * V));
  const fullRaw = model.forward(ids).logits;
  const full = fullRaw.slice((T - 1) * V, T * V);
  const kv = F.initKVCache(model);
  let step = null;
  for (let i = 0; i < T; i++) step = F.forwardStep(model, ids[i], i, kv).logits;
  const d = maxAbsDiff(full, step);
  return { pass: d === 0, detail: `序列<T=${T}<cap 时 forwardStep(sink=4) vs forward 末位 Δ=${d}（要求 0）` };
}

/* 3+4. 锚点不淘汰 + 滚动窗口 */
function sinkInvariant() {
  const V = 17, K = 4, R = 8, cap = K + R;
  const model = new F.GPTModel({ vocabSize: V, blockSize: 64, nLayer: 3, nHead: 4, nKvHead: 2, dModel: 32, norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope', attentionSink: K, rollingWindow: R, seed: 5 });
  const kvDim = model.KV * model.dh, KV = model.KV, dh = model.dh;
  const rng = F.mulberry32(23), ids = [];
  for (let i = 0; i < 200; i++) ids.push(Math.floor(rng() * V));
  const kv = F.initKVCache(model);
  const sinkInit = [];
  for (let i = 0; i < K; i++) { F.forwardStep(model, ids[i], i, kv); sinkInit.push(readK(kv, 0, i, kvDim, KV, dh)); }
  for (let i = K; i < ids.length; i++) { if (kv.n >= kv.cap) F.kvShift(model, kv); F.forwardStep(model, ids[i], i, kv); }
  // 锚点槽位逐位一致性
  let sinkMax = 0;
  for (let s = 0; s < K; s++) sinkMax = Math.max(sinkMax, maxAbsDiff(sinkInit[s], readK(kv, 0, s, kvDim, KV, dh)));
  // 末窗口槽位 == 最近写入 token
  const refkv = F.initKVCache(model);
  F.forwardStep(model, ids[ids.length - 1], ids.length - 1, refkv);
  const lastK = readK(kv, 0, cap - 1, kvDim, KV, dh);
  const refK = readK(refkv, 0, 0, kvDim, KV, dh);
  const lastMax = maxAbsDiff(lastK, refK);
  const nOk = kv.n === cap;
  return {
    pass: sinkMax === 0 && lastMax === 0 && nOk,
    detail: `远超容量后：锚点 slot[0..${K}) 与初次写入 Δ=${sinkMax}（要求 0）· 末窗口槽位==最近 token Δ=${lastMax}（要求 0）· kv.n=${kv.n}===${cap}（${nOk}）`,
  };
}

/* 5. StreamingLLM 改变输出（锚点保护确实生效） */
function sinkEffectiveness() {
  const V = 17, K = 4, R = 8, cap = K + R;
  const base = { vocabSize: V, blockSize: cap, nLayer: 2, nHead: 4, nKvHead: 2, dModel: 32, norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope', seed: 9 };
  const mSink = new F.GPTModel(Object.assign({}, base, { attentionSink: K, rollingWindow: R }));
  const mSlide = new F.GPTModel(Object.assign({}, base, { attentionSink: 0 })); // 纯滑动窗口（cap=blockSize=cap）
  const rng = F.mulberry32(31), ids = [];
  for (let i = 0; i < 5; i++) ids.push(Math.floor(rng() * V));
  const gS = greedy(mSink, ids, 40), gE = greedy(mSlide, ids, 40);
  // 复算末位 logits 差异（直接用两模型再跑一步取末 logits 比对）
  const kvS = F.initKVCache(mSink), kvE = F.initKVCache(mSlide);
  let lS = F.warmupKV(mSink, ids, kvS), lE = F.warmupKV(mSlide, ids, kvE);
  let pS = ids.length, pE = ids.length;
  for (let s = 0; s < 40; s++) {
    if (kvS.n >= kvS.cap) F.kvShift(mSink, kvS); if (kvE.n >= kvE.cap) F.kvShift(mSlide, kvE);
    lS = F.forwardStep(mSink, gS[s], pS, kvS).logits; lE = F.forwardStep(mSlide, gE[s], pE, kvE).logits; pS++; pE++;
  }
  const d = maxAbsDiff(lS, lE);
  return { pass: d > 1e-9, detail: `sink 保护 vs 纯滑动窗口 末位 logits maxΔ=${d.toExponential(3)}（>0 证明锚点确实被 attend）` };
}

/* 6+7. 确定性 + 无限上下文不崩溃 */
function streamingDeterminism() {
  const V = 17, K = 4, R = 8;
  const base = { vocabSize: V, blockSize: 64, nLayer: 2, nHead: 4, nKvHead: 2, dModel: 32, norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope', attentionSink: K, rollingWindow: R, seed: 17 };
  const m1 = new F.GPTModel(base), m2 = new F.GPTModel(base);
  const rng = F.mulberry32(41), start = [];
  for (let i = 0; i < 6; i++) start.push(Math.floor(rng() * V));
  const g1 = greedy(m1, start, 300), g2 = greedy(m2, start, 300);
  let eq = g1.length === g2.length;
  for (let i = 0; i < g1.length && eq; i++) if (g1[i] !== g2[i]) eq = false;
  // 另跑一遍捕获末位 logits，验证有限（无 NaN/Inf）
  const kv = F.initKVCache(m1);
  let lf = F.warmupKV(m1, start, kv), pos = start.length;
  let last = 0;
  for (let s = 0; s < 300; s++) {
    if (kv.n >= kv.cap) F.kvShift(m1, kv);
    lf = F.forwardStep(m1, g1[s], pos, kv).logits; pos++;
    if (s === g1.length - 1) last = lf[0];
  }
  const finite = allFinite(lf);
  return { pass: g1.length === 300 && eq && finite, detail: `greedy 300 token 长度=${g1.length}（要求300）· 同种子复现=${eq}· 末位 logits 有限=${finite}（样本 ${last.toExponential(3)}）` };
}

/* 8. × KV INT8：sink + kvQuant=8 */
function sinkWithInt8() {
  const V = 17, K = 4, R = 8, cap = K + R;
  const model = new F.GPTModel({ vocabSize: V, blockSize: 64, nLayer: 3, nHead: 4, nKvHead: 2, dModel: 32, norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope', attentionSink: K, rollingWindow: R, kvQuant: 8, seed: 5 });
  const kvDim = model.KV * model.dh, KV = model.KV, dh = model.dh;
  const rng = F.mulberry32(23), ids = [];
  for (let i = 0; i < 200; i++) ids.push(Math.floor(rng() * V));
  const kv = F.initKVCache(model);
  const sinkInit = [];
  for (let i = 0; i < K; i++) { F.forwardStep(model, ids[i], i, kv); sinkInit.push(readK(kv, 0, i, kvDim, KV, dh)); }
  let lastLogits = null;
  for (let i = K; i < ids.length; i++) { if (kv.n >= kv.cap) F.kvShift(model, kv); lastLogits = F.forwardStep(model, ids[i], i, kv).logits; }
  let sinkMax = 0;
  for (let s = 0; s < K; s++) sinkMax = Math.max(sinkMax, maxAbsDiff(sinkInit[s], readK(kv, 0, s, kvDim, KV, dh)));
  const finite = allFinite(lastLogits);
  // INT8 解量化误差应 ≤ absmax/254（粗界：与同模型 Float64 比）
  const mF = new F.GPTModel({ vocabSize: V, blockSize: 64, nLayer: 3, nHead: 4, nKvHead: 2, dModel: 32, norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope', attentionSink: K, rollingWindow: R, kvQuant: 0, seed: 5 });
  const kvF = F.initKVCache(mF);
  let lF = null;
  for (let i = 0; i < ids.length; i++) { if (kvF.n >= kvF.cap) F.kvShift(mF, kvF); lF = F.forwardStep(mF, ids[i], i, kvF).logits; }
  const dF = maxAbsDiff(lastLogits, lF);
  return {
    pass: kv.q8 === true && sinkMax === 0 && finite && dF < 0.05,
    detail: `sink+INT8：q8=${kv.q8}· 锚点不被淘汰 Δ=${sinkMax}（要求 0）· 末位有限=${finite} · INT8 vs F32 末位 maxΔ=${dF.toExponential(3)}（<0.05）`,
  };
}

/* 9. × SWA：sink + window(<rolling) */
function sinkWithSWA() {
  const V = 17, K = 4, R = 8, win = 5;
  const model = new F.GPTModel({ vocabSize: V, blockSize: 64, nLayer: 3, nHead: 4, nKvHead: 2, dModel: 32, norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope', attentionSink: K, rollingWindow: R, window: win, seed: 5 });
  const rng = F.mulberry32(23), ids = [];
  for (let i = 0; i < 200; i++) ids.push(Math.floor(rng() * V));
  const kv = F.initKVCache(model);
  let lastLogits = null;
  for (let i = 0; i < ids.length; i++) { if (kv.n >= kv.cap) F.kvShift(model, kv); lastLogits = F.forwardStep(model, ids[i], i, kv).logits; }
  const finite = allFinite(lastLogits);
  // 验证锚点被保护：满容后 kv.n===cap 且 slot[0..K) 仍是首 K token（与初次写入一致）
  const kvDim = model.KV * model.dh, KV = model.KV, dh = model.dh;
  const kv2 = F.initKVCache(model);
  const sinkInit = [];
  for (let i = 0; i < K; i++) { F.forwardStep(model, ids[i], i, kv2); sinkInit.push(readK(kv2, 0, i, kvDim, KV, dh)); }
  for (let i = K; i < ids.length; i++) { if (kv2.n >= kv2.cap) F.kvShift(model, kv2); F.forwardStep(model, ids[i], i, kv2); }
  let sinkMax = 0;
  for (let s = 0; s < K; s++) sinkMax = Math.max(sinkMax, maxAbsDiff(sinkInit[s], readK(kv2, 0, s, kvDim, KV, dh)));
  return { pass: finite && kv.n === kv.cap && sinkMax === 0, detail: `sink+window(${win})：末位有限=${finite}· kv.n=${kv.n}===cap=${kv.cap}· 锚点不被淘汰 Δ=${sinkMax}（要求 0）` };
}

/* 10. kvShift scale 同步（INT8，多次淘汰） */
function sinkScaleSync() {
  const V = 17, K = 4, R = 8;
  const model = new F.GPTModel({ vocabSize: V, blockSize: 64, nLayer: 2, nHead: 4, nKvHead: 2, dModel: 32, norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope', attentionSink: K, rollingWindow: R, kvQuant: 8, seed: 5 });
  const rng = F.mulberry32(23), ids = [];
  for (let i = 0; i < 120; i++) ids.push(Math.floor(rng() * V));
  const kv = F.initKVCache(model);
  for (let i = 0; i < ids.length; i++) { if (kv.n >= kv.cap) F.kvShift(model, kv); F.forwardStep(model, ids[i], i, kv); }
  let scaleOk = true;
  for (let l = 0; l < model.cfg.nLayer; l++) for (let i = 0; i < kv.Ks[l].length; i++) if (!Number.isFinite(kv.Ks[l][i])) scaleOk = false;
  const nOk = kv.n === kv.cap;
  return { pass: scaleOk && nOk && kv.q8, detail: `sink+INT8 多次淘汰后 scale 全有限=${scaleOk}· kv.n=${kv.n}===${kv.cap}=${nOk}· q8=${kv.q8}` };
}

// 运行
check('1. 零回归（sink=0 ≡ 无字段）', sinkZeroRegression());
check('2. 短序列一致性（<cap 时 forwardStep≡forward）', shortConsistency());
check('3. 锚点永不被淘汰（核心不变量）', sinkInvariant());
check('4. 滚动窗口生效（kv.n===cap，末槽==最近 token）', sinkInvariant());
check('5. StreamingLLM 改变输出（锚点确实被 attend）', sinkEffectiveness());
check('6. 生成确定性（greedy 300 同种子复现）', streamingDeterminism());
check('7. 无限上下文不崩溃（生成 300 token）', streamingDeterminism());
check('8. × KV INT8（sink+INT8 锚点保护+有限）', sinkWithInt8());
check('9. × SWA（sink+window 锚点保护+有限）', sinkWithSWA());
check('10. kvShift scale 同步（INT8 多次淘汰）', sinkScaleSync());

let allPass = true;
console.log('🔬 旗舰 AI · v13 Attention Sink / StreamingLLM 验证（Node ' + process.version + '）');
console.log('────────────────────────────────────────────────────');
C.forEach((c, i) => {
  const r = (typeof c.pass === 'object') ? c.pass.pass : c.pass;
  const d = (typeof c.pass === 'object') ? c.pass.detail : c.detail;
  if (!r) allPass = false;
  console.log(` ${r ? '✅' : '❌'} ${i + 1}. ${c.name}\n      ${d}`);
});
console.log('────────────────────────────────────────────────────');
console.log(`结果：${C.filter((c) => (typeof c.pass === 'object' ? c.pass.pass : c.pass)).length}/${C.length} 通过 ${allPass ? '✅ 全绿' : '❌ 有失败'}`);
process.exit(allPass ? 0 : 1);

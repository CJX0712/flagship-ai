/* ============================================================================
 * 旗舰 AI · v11 Gemma 2 稳定化组件（QK-norm + logit soft-cap）专项验证
 * ----------------------------------------------------------------------------
 *  1. QK-norm 恒等式（硬不变量：ss/dh + eps = 1/rstd²，RoPE 保范使缓存可直接验证）
 *  2. QK-norm 梯度检验（含增益 qg/kg，解析 vs 数值，err/tol ≤ 1）
 *  3. logit soft-cap 有界性（|logits| ≤ cap 精确成立；对照组证明 cap 真的生效）
 *  4. logit soft-cap 梯度检验（sech² 链式法则，err/tol ≤ 1）
 *  5. 双开关零回归（qkNorm:false + logitCap:0 ≡ 不传，逐位一致）
 *  6. KV Cache 逐步一致（QK-norm + soft-cap + 层交错窗口全组合，maxΔ=0）
 *  7. 生成：KV Cache / 全序列重算贪心解码逐位一致
 *  8. 持久化 + INT8 量化往返（qg/kg 存档无损，量化后恒等式与有界性仍成立）
 *  9. 可训练性（gemma 全配方预设过拟合 loss -> <0.1）
 * 10. applyPreset('gemma') 全配方字段（qkNorm/logitCap/alt 窗口）
 *
 * 运行：node scripts/probe-gemma2.js   （全通过 exit 0）
 * 作者：晨星  ·  License: MIT
 * ========================================================================*/
const F = require('../src/flagship-engine.js');

const POEM = '朝辞白帝彩云间，千里江陵一日还。两岸猿声啼不住，轻舟已过万重山。';

/* 1. QK-norm 恒等式：前向 q̂ = q·rstd（增益初值 1），RoPE 保范 ⇒ 缓存 qkv 满足
 *    ss/dh + eps = 1/rstd²，其中 ss = Σ_p (q̂_p/rstd)²。硬不变量（浮点误差量级）。 */
function qkNormIdentity() {
  const V = 13, T = 10, H = 4, KV = 2, dh = 4;
  const model = new F.GPTModel({
    vocabSize: V, blockSize: 12, nLayer: 2, nHead: H, nKvHead: KV, qkNorm: true,
    dModel: 16, norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope', seed: 7,
  });
  const rng = F.mulberry32(5);
  const tokens = [];
  for (let i = 0; i < T; i++) tokens.push(Math.floor(rng() * V));
  const { cache } = model.forward(tokens);
  const qkvDim = model.qkvDim, eps = 1e-5;
  let maxRel = 0, rmsMin = Infinity, rmsMax = 0;
  for (let l = 0; l < 2; l++) {
    const c = cache.layers[l], qkv = c.qkv;
    const segs = [
      { nH: H, off: 0, rstd: c.qk.qr, tag: 'Q' },
      { nH: KV, off: model.cfg.dModel, rstd: c.qk.kr, tag: 'K' },
    ];
    for (const s of segs) {
      for (let i = 0; i < T; i++) {
        for (let h = 0; h < s.nH; h++) {
          const rs = s.rstd[i * s.nH + h];
          let ss = 0, n2 = 0;
          for (let p = 0; p < dh; p++) {
            const qh = qkv[i * qkvDim + s.off + h * dh + p]; // 归一+旋转后（保范）
            ss += (qh / rs) * (qh / rs);
            n2 += qh * qh;
          }
          const rel = Math.abs(ss / dh + eps - 1 / (rs * rs)) / (1 / (rs * rs));
          if (rel > maxRel) maxRel = rel;
          const rms = Math.sqrt(n2 / dh);
          if (rms < rmsMin) rmsMin = rms;
          if (rms > rmsMax) rmsMax = rms;
        }
      }
    }
  }
  // 对照组：不启用 QK-norm 时 RMS 明显偏离 1
  const m0 = new F.GPTModel({ vocabSize: V, blockSize: 12, nLayer: 1, nHead: H, nKvHead: KV, dModel: 16, norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope', seed: 7 });
  const c0 = m0.forward(tokens).cache.layers[0];
  let ctrlMax = 0;
  for (let i = 0; i < T; i++) {
    for (let h = 0; h < H; h++) {
      let n2 = 0;
      for (let p = 0; p < dh; p++) { const q = c0.qkv[i * qkvDim + h * dh + p]; n2 += q * q; }
      ctrlMax = Math.max(ctrlMax, Math.abs(Math.sqrt(n2 / dh) - 1));
    }
  }
  return {
    pass: maxRel < 1e-9 && rmsMax < 1.000001 && rmsMin > 0.95 && ctrlMax > 0.05,
    detail: `恒等式最大相对误差 ${maxRel.toExponential(2)}（<1e-9 硬不变量）；归一后每头 RMS∈[${rmsMin.toFixed(6)}, ${rmsMax.toFixed(6)}]（≤1，下限受 eps·dh/ss 影响）；对照组(无QK-norm)最大偏离 ${ctrlMax.toFixed(3)}`,
  };
}

/* 2. QK-norm 中心差分梯度检验（RoPE+SwiGLU+RMSNorm+GQA+QK-norm 全组合） */
function gradCheckQKNorm() {
  const V = 11, T = 8;
  const model = new F.GPTModel({
    vocabSize: V, blockSize: 12, nLayer: 2, nHead: 4, nKvHead: 2, qkNorm: true,
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
  let maxRatio = 0, checked = 0, skipped = 0, gainChecked = 0;
  for (const [key, i] of cand) {
    const p = model.params[key].d;
    const orig = p[i];
    p[i] = orig + e; const lp = lossOf();
    p[i] = orig - e; const lm = lossOf();
    p[i] = orig;
    const num = (lp - lm) / (2 * e);
    const ana = model.grads[key][i];
    if (key.endsWith('.qg') || key.endsWith('.kg')) gainChecked++;
    if (Math.abs(ana) < 1e-12 && Math.abs(num) < 1e-12) { skipped++; continue; }
    const err = Math.abs(ana - num);
    const tol = 1e-7 + 1e-4 * Math.max(Math.abs(ana), Math.abs(num));
    const ratio = err / tol;
    if (ratio > maxRatio) maxRatio = ratio;
    checked++;
  }
  return { pass: checked > 100 && maxRatio <= 1 && gainChecked > 0, detail: `QK-norm 梯度检验 ${checked} 参数(含增益 ${gainChecked} 个, 跳过 ${skipped} 零梯度) 最大 err/tol=${maxRatio.toExponential(2)}` };
}

/* 3. logit soft-cap 有界性：小语料训练后 logits 确实变大（对照组），
 *    cap=1.5 时 |logits| ≤ 1.5 精确成立且贴近上界（证明 cap 真的在削）。 */
function softCapBounded() {
  const tok = new F.Tokenizer(POEM);
  const ids = tok.encode(POEM);
  const mk = (cap) => new F.GPTModel({
    vocabSize: tok.vocabSize, blockSize: ids.length, nLayer: 2, nHead: 4, dModel: 32,
    norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope', seed: 1, logitCap: cap,
  });
  const mFree = mk(0), mCap = mk(1.5);
  F.train(mFree, ids, { steps: 120, lr: 3e-3, warmup: 10, wd: 0, fixedBatch: ids, logEvery: 1e9 });
  F.train(mCap, ids, { steps: 120, lr: 3e-3, warmup: 10, wd: 0, fixedBatch: ids, logEvery: 1e9 });
  const lf = mFree.forward(ids).logits, lc = mCap.forward(ids).logits;
  let freeMax = 0, capMax = 0;
  for (let i = 0; i < lf.length; i++) { freeMax = Math.max(freeMax, Math.abs(lf[i])); capMax = Math.max(capMax, Math.abs(lc[i])); }
  return {
    pass: freeMax > 1.5 && capMax <= 1.5 + 1e-12 && capMax > 1.4,
    detail: `无帽 max|logit|=${freeMax.toFixed(3)}(>1.5 证明 logits 确实变大)；cap=1.5 后 max|logit|=${capMax.toFixed(6)}（≤1.5+1e-12 且 >1.4 证明削峰生效）`,
  };
}

/* 4. logit soft-cap 梯度检验（sech² 链式法则） */
function gradCheckSoftCap() {
  const V = 11, T = 8;
  const model = new F.GPTModel({
    vocabSize: V, blockSize: 12, nLayer: 2, nHead: 4, dModel: 16, logitCap: 5,
    norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope', seed: 7,
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
  return { pass: checked > 100 && maxRatio <= 1, detail: `soft-cap 梯度检验 ${checked} 参数(跳过 ${skipped} 零梯度) 最大 err/tol=${maxRatio.toExponential(2)}` };
}

/* 5. 双开关零回归：显式全关 ≡ 不传（逐位一致） */
function bothOffRegression() {
  const cfg = { vocabSize: 13, blockSize: 12, nLayer: 2, nHead: 4, nKvHead: 2, dModel: 16, seed: 5, norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope' };
  const m0 = new F.GPTModel(cfg);
  const m1 = new F.GPTModel(Object.assign({}, cfg, { qkNorm: false, logitCap: 0 }));
  const seq = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
  const l0 = m0.forward(seq).logits, l1 = m1.forward(seq).logits;
  let maxDiff = 0;
  for (let i = 0; i < l0.length; i++) maxDiff = Math.max(maxDiff, Math.abs(l0[i] - l1[i]));
  const noParams = !m0.params['l0.qg'] && !m0.params['l0.kg'];
  return { pass: maxDiff === 0 && noParams, detail: `qkNorm:false+logitCap:0 vs 不传 前向最大差异 ${maxDiff}（要求 0）；关闭时不产生 qg/kg 参数=${noParams}` };
}

/* 6. KV Cache 逐步一致：QK-norm + soft-cap + 层交错窗口全组合 */
function kvStepwiseFull() {
  const V = 13, B = 48;
  const model = new F.GPTModel({
    vocabSize: V, blockSize: B, nLayer: 3, nHead: 4, nKvHead: 2, qkNorm: true, logitCap: 20,
    window: [5, 0, 3], dModel: 16, norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope', seed: 3,
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
  return { pass: maxDiff === 0, detail: `全组合逐步 forwardStep vs forward 末位 logits 最大差异 ${maxDiff}（要求 0）` };
}

/* 7. 生成：KV Cache 与全序列重算贪心解码逐位一致 */
function generateFull() {
  const tok = new F.Tokenizer(POEM);
  const ids = tok.encode(POEM);
  const model = new F.GPTModel(F.applyPreset(
    { vocabSize: tok.vocabSize, blockSize: 40, nLayer: 4, nHead: 4, dModel: 32, seed: 1 }, 'gemma'));
  Object.assign(model.cfg, { window: { alt: 8 } });
  const m = new F.GPTModel(model.cfg); // gemma 全配方（alt:8 适配小语料）
  F.train(m, ids, { steps: 150, lr: 3e-3, warmup: 10, wd: 0, fixedBatch: ids, logEvery: 1e9 });
  const g1 = F.generate(m, tok, '朝辞白', { maxNew: 24, temperature: 0, seed: 7, useCache: true });
  const g2 = F.generate(m, tok, '朝辞白', { maxNew: 24, temperature: 0, seed: 7, useCache: false });
  const same = g1.ids.length === g2.ids.length && g1.ids.every((v, i) => v === g2.ids[i]);
  return { pass: same && g1.text.length > 0, detail: `贪心解码 缓存/重算 ids 一致=${same}；生成 ${g1.ids.length} token（如「${g1.text.slice(0, 12)}…」）` };
}

/* 8. 持久化 + INT8 量化往返：qg/kg 存档无损；量化后恒等式与有界性仍成立 */
function persistFull() {
  const model = new F.GPTModel({
    vocabSize: 13, blockSize: 12, nLayer: 2, nHead: 4, nKvHead: 2, qkNorm: true, logitCap: 10,
    dModel: 16, norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope', seed: 5,
  });
  const seq = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
  const l1 = model.forward(seq).logits;

  const r = F.loadModel(F.saveModel(model, null));
  const l2 = r.forward(seq).logits;
  let dLoad = 0;
  for (let i = 0; i < l1.length; i++) dLoad = Math.max(dLoad, Math.abs(l1[i] - l2[i]));
  const gainOk = !!(r.qkNorm === true && r.logitCap === 10 && r.params['l0.qg'] && r.params['l0.kg']);

  const q = model.quantize();
  const l3 = q.forward(seq).logits;
  let dQ = 0, boundOk = true;
  for (let i = 0; i < l1.length; i++) {
    dQ = Math.max(dQ, Math.abs(l1[i] - l3[i]));
    if (Math.abs(l3[i]) > 10 + 1e-9) boundOk = false;
  }
  // 量化后 QK-norm 恒等式仍成立（qg/kg 保持 Float64）
  const c = q.forward(seq).cache.layers[0];
  const rs = c.qk.qr[0];
  let ss = 0;
  const dh = 4;
  for (let p = 0; p < dh; p++) { const qh = c.qkv[p]; ss += (qh / rs) * (qh / rs); }
  const rel = Math.abs(ss / dh + 1e-5 - 1 / (rs * rs)) / (1 / (rs * rs));
  return {
    pass: dLoad === 0 && gainOk && dQ < 1 && boundOk && rel < 1e-9,
    detail: `save/load 差异 ${dLoad}，qkNorm/logitCap/qg/kg 保留=${gainOk}；量化 F32/INT8 差异 ${dQ.toFixed(4)}，量化后有界=${boundOk}，恒等式误差 ${rel.toExponential(2)}`,
  };
}

/* 9. 可训练性：gemma 全配方预设（alt:8 适配小语料）过拟合 */
function trainableFull() {
  const tok = new F.Tokenizer(POEM);
  const ids = tok.encode(POEM);
  const cfg = F.applyPreset({ vocabSize: tok.vocabSize, blockSize: ids.length, nLayer: 4, nHead: 4, dModel: 32, seed: 1 }, 'gemma');
  Object.assign(cfg, { window: { alt: 8 } });
  const m = new F.GPTModel(cfg);
  const { history } = F.train(m, ids, { steps: 300, lr: 3e-3, warmup: 20, wd: 0, fixedBatch: ids, logEvery: 1e9 });
  const last = history[history.length - 1];
  return { pass: last < 0.1 && m.qkNorm && m.logitCap === 30, detail: `gemma 全配方过拟合 初始 ${history[0].toFixed(2)} -> 最终 loss ${last.toFixed(4)}（阈值 <0.1；qkNorm=${m.qkNorm}, logitCap=${m.logitCap}）` };
}

/* 10. applyPreset('gemma') 全配方字段 */
function presetFull() {
  const cfg = F.applyPreset({ vocabSize: 50, blockSize: 16, nLayer: 4, nHead: 4, dModel: 32 }, 'gemma');
  const m = new F.GPTModel(cfg);
  const ok = cfg.norm === 'rmsnorm' && cfg.ffn === 'swiglu' && cfg.pos === 'rope' && cfg.nKvHead === 2
    && cfg.qkNorm === true && cfg.logitCap === 30 && JSON.stringify(m.windows) === JSON.stringify([0, 64, 0, 64]);
  return { pass: ok, detail: `gemma 全配方：qkNorm=${cfg.qkNorm} · logitCap=${cfg.logitCap} · windows=[${m.windows}] · GQA(nKvHead=${cfg.nKvHead})` };
}

const CHECKS = [
  { name: 'QK-norm 恒等式（RMS≈1 硬不变量）', run: qkNormIdentity },
  { name: 'QK-norm 中心差分梯度检验', run: gradCheckQKNorm },
  { name: 'logit soft-cap 有界性', run: softCapBounded },
  { name: 'logit soft-cap 梯度检验', run: gradCheckSoftCap },
  { name: '双开关零回归', run: bothOffRegression },
  { name: '全组合 KV Cache 逐步一致', run: kvStepwiseFull },
  { name: '生成：缓存/重算一致', run: generateFull },
  { name: '持久化 + INT8 量化往返', run: persistFull },
  { name: 'gemma 全配方可训练性', run: trainableFull },
  { name: "applyPreset('gemma') 全配方", run: presetFull },
];

function safeRun(fn) {
  try { const r = fn() || {}; return { pass: r.pass !== false, detail: r.detail || '' }; }
  catch (e) { return { pass: false, detail: '异常: ' + (e && e.message) }; }
}

const results = CHECKS.map(c => ({ name: c.name, ...safeRun(c.run) }));
const W = 50;
console.log('🔬 旗舰 AI · v11 Gemma 2 稳定化组件（QK-norm + logit soft-cap）验证（Node ' + process.version + '）');
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

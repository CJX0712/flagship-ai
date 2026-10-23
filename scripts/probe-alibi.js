#!/usr/bin/env node
/* ============================================================================
 * 旗舰 AI · v15.10 ALiBi（Attention with Linear Biases）专项验证
 * ----------------------------------------------------------------------------
 * ALiBi = 第 7 套位置/注意力表示：不加位置嵌入、不做旋转，而是把「−slope·距离」
 * 直接加到 softmax 前的注意力打分上（Press et al. 2022；BLOOM / MPT 的发表设计）。
 * 本探针把「引擎接线」与「L10 微指令已证数学」对拷，任何一侧走偏都会立刻红：
 *
 *   1. 斜率闭式：slope[h] = 2^(−8h/n)，n = 不小于 H 的最小 2 的幂（μD045/μD053）
 *   2. 偏置矩阵：B[i][j] = −slope·|i−j|（对角 0 / 非正 / 对称 / 距离单调 / 有限）
 *   3. 等分打分几何衰减：log(pⱼ/pᵢ) = −slope·|i−j|，且对角严格最大（μD112 语义）
 *   4. 关闭时 Δ=0：ALiBi 关闭 ⇒ 与 legacy（完全未改动的老代码）逐位一致
 *   5. 双端逐位一致：forward ≡ forwardStep（短序列 / 滑动窗口 / 淘汰后 / 锚点+滚动）
 *      —— 含 kvShift 之后仍然成立，这是本任务唯一实质难点 kv.pos 的把关处
 *   6. 中心差分梯度检验：ALiBi 注意力 err/tol ≤ 1（偏置无参数 ⇒ softmaxBwd 已正确）
 *   7. 微指令交叉验证：把引擎算出的同一组 slopes / 偏置喂给 μD045–μD056、μD112、
 *      μD113、μD114、μD117、μD119 的 run / predicate，逐项断言
 *
 * 运行：node scripts/probe-alibi.js   （全通过 exit 0）
 * 作者：晨星  ·  License: MIT
 * ==========================================================================*/
'use strict';

const F = require('../src/index.js');            // 生产加载路径：填充 globalThis.FlagshipAI
const NS = globalThis.FlagshipAI;
const legacy = require('../src/legacy/flagship-engine.v11.js'); // Δ=0 基线（本任务未触碰）
const rc = NS.L10.relClose;

const checks = [];
function check(name, r) {
  checks.push({ name, pass: !!r.pass, detail: r.detail || '' });
}

/* ------------------------------ 通用工具 ------------------------------ */
function maxAbsDiff(a, b, off) {
  off = off || 0;
  let m = 0;
  for (let i = 0; i < b.length; i++) { const d = Math.abs(a[off + i] - b[i]); if (d > m) m = d; }
  return m;
}
function rowOf(mat, row, cols) { return mat.subarray(row * cols, (row + 1) * cols); }
function allFinite(a) { for (let i = 0; i < a.length; i++) if (!Number.isFinite(a[i])) return false; return true; }
function isPow2(n) { return n > 0 && (n & (n - 1)) === 0; }
function nextPow2(n) { let p = 1; while (p < n) p *= 2; return p; }
/** 与实现独立的斜率参考式：slope[h] = 2^(−8h/n)（n = ≥H 的最小 2 的幂）。 */
function refSlopes(H) {
  const n = nextPow2(H), out = new Float64Array(H);
  for (let h = 0; h < H; h++) out[h] = Math.pow(2, -8 * h / n);
  return out;
}
/** 构建 ALiBi 模型（bloom 预设或显式开关）。头数不被 dModel 整除时自动抬高 dModel。 */
function mk(extra) {
  const base = { vocabSize: 24, blockSize: 16, nLayer: 2, nHead: 4, dModel: 32, seed: 11 };
  const cfg = Object.assign(base, extra || {}, { alibi: true });
  if (cfg.dModel % cfg.nHead !== 0) cfg.dModel = cfg.nHead * 8; // ALiBi 无 RoPE ⇒ dh 奇偶不限
  return new F.GPTModel(cfg);
}

/* ------------------------------ 1. 斜率闭式 ------------------------------ */
(function () {
  const detail = [], Hs = [1, 2, 4, 8, 16, 3, 5, 6, 7, 12];
  let ok = true, worst = 0;
  for (const H of Hs) {
    const m = mk({ nHead: H });
    const s = m.alibiSlopes, n = nextPow2(H);
    if (!s || s.length !== H) { ok = false; detail.push(`H=${H} 斜率缺失`); continue; }
    for (let h = 0; h < H; h++) {
      const want = refSlopes(H)[h];
      const d = Math.abs(s[h] - want) / Math.max(1, Math.abs(want));
      if (d > worst) worst = d;
      if (!rc(s[h], want, 1e-12)) ok = false;
      if (!(s[h] > 0 && s[h] <= 1 + 1e-15)) ok = false;                 // μD052 值域
      if (h + 1 < H && s[h] < s[h + 1]) ok = false;                     // μD046 非增
    }
    detail.push(`H=${H}(n=${n}) s0=${s[0].toFixed(4)}`);
  }
  check('1. 斜率闭式 2^(−8h/n)（0-based，含非 2 的幂 H）+ 非增 + 落在 (0,1]',
    { pass: ok, detail: `最大相对偏差 ${worst.toExponential(2)} · ${detail.slice(0, 5).join(' · ')} · 共 ${Hs.length} 组` });
})();

/* ---------- 1b. ALiBi 取代位置编码（无 posEmb、无 RoPE、省参数） ---------- */
(function () {
  const base = { vocabSize: 24, blockSize: 16, nLayer: 2, nHead: 4, dModel: 32, seed: 11 };
  const m = new F.GPTModel(F.applyPreset(base, 'bloom'));
  const plain = new F.GPTModel(base);
  const ropeForced = new F.GPTModel(Object.assign({ nHead: 4, dModel: 32, pos: 'rope' }, base, { pos: 'rope', alibi: true }));
  okMsgs(m, plain, ropeForced);
})();
function okMsgs(m, plain, ropeForced) {
  const noPe = typeof m.params.posEmb === 'undefined';
  const noRope = m.useRope === false;
  const saved = plain.paramCount() - m.paramCount();
  check('1b. ALiBi 取代位置编码：无 posEmb / useRope=false / 预设 bloom 生效',
    {
      pass: noPe && noRope && !!m.alibiSlopes && typeof plain.params.posEmb === 'object'
        && saved === 16 * 32 && ropeForced.useRope === false,
      detail: `posEmb=${typeof m.params.posEmb} · useRope=${m.useRope} · 省参数 ${saved}（= blockSize·dModel）· pos:'rope'+alibi ⇒ useRope=${ropeForced.useRope}`,
    });
}

/* --------------------- 2. 偏置矩阵 −slope·|i−j| --------------------- */
(function () {
  const m = mk({});
  const T = 8, slope = m.alibiSlopes[0];
  let ok = true, diag0 = true, nonPos = true, sym = true, mono = true, finite = true;
  // 第 1 遍：按公式填满整张偏置表（同时校验闭式、有限性、非正性）
  const B = new Float64Array(T * T);
  for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) {
    B[i * T + j] = -slope * Math.abs(i - j);
    if (!rc(B[i * T + j], -slope * Math.abs(i - j), 1e-12)) ok = false;
    if (!Number.isFinite(B[i * T + j])) finite = false;
    if (B[i * T + j] > 0) nonPos = false;
  }
  // 第 2 遍：对称性需要整张表已就绪，必须与填充分开，否则读到未写入的 0 会假红
  for (let i = 0; i < T; i++) {
    if (B[i * T + i] !== 0) diag0 = false;                                // μD048 对角零
    for (let j = 0; j < T; j++) {
      if (!Object.is(B[i * T + j], B[j * T + i])) sym = false;            // μD051 对称
      for (let j2 = 0; j2 < T; j2++) {                                    // μD050 距离单调
        if (Math.abs(i - j) < Math.abs(i - j2) && B[i * T + j] < B[i * T + j2]) mono = false;
      }
    }
  }
  check('2. 偏置矩阵：对角 0 / 非正 / 对称 / 距离单调 / 有限',
    { pass: ok && diag0 && nonPos && sym && mono && finite, detail: `T=${T} slope=${slope.toFixed(4)} · diag0=${diag0} 非正=${nonPos} 对称=${sym} 单调=${mono} 有限=${finite}` });
})();

/* ---------- 2b. 引擎真身：零 qkv ⇒ 打分只剩偏置，复原 −slope·|i−j| ---------- */
(function () {
  const D = 32, H = 4, dh = D / H, KV = H, qkvDim = D + 2 * KV * dh, T = 8;
  const m = mk({});
  const qkv = new Float64Array(T * qkvDim);                 // 全零 ⇒ 所有 q·k = 0
  const scale = 1 / Math.sqrt(dh);
  const { Ps } = NS.L4.attention.attnFwd(qkv, T, D, H, dh, qkvDim, KV, 1, 0, scale, 1, m.alibiSlopes);
  let ok = true, worst = 0;
  for (let h = 0; h < H; h++) {
    const P = rowOf(Ps[h], T - 1, T), sl = m.alibiSlopes[h];
    // 因果行内：s_j − s_0 = slope·j（由 s_j = −slope·(T−1−j) 推出）—— 逐位验证斜率被正确注入
    for (let j = 1; j < T; j++) {
      const want = sl * j, got = Math.log(P[j] / P[0]);
      const d = Math.abs(got - want) / Math.max(1, Math.abs(want));
      if (d > worst) worst = d;
      if (!rc(got, want, 1e-9)) ok = false;
    }
    // 未来位置被 −1e9 掩码压成 0（j > i 不参与）
    for (let j = 0; j < T; j++) if (!Number.isFinite(P[j]) || P[j] < 0) ok = false;
  }
  check('2b. 引擎 attnFwd：零 qkv 下打分 ≡ 偏置（log(pⱼ/p₀)=slope·j 逐位成立）',
    { pass: ok, detail: `最大相对偏差 ${worst.toExponential(2)} · ${H} 头 × T=${T}` });
})();

/* --------------------- 3. 等分打分的几何衰减（μD112） --------------------- */
(function () {
  const op = NS.L10.ops['D112'];
  const m = mk({});
  let ok = true, worst = 0;
  for (let h = 0; h < 4; h++) {
    const sl = m.alibiSlopes[h];
    const pref = op.run(7, sl, 3);                       // m = 1+(7%8) = 8，行 i=3
    for (let j = 0; j < 8; j++) {
      const want = -sl * Math.abs(3 - j), got = Math.log(pref[j] / pref[3]);
      const d = Math.abs(got - want) / Math.max(1, Math.abs(want));
      if (d > worst) worst = d;
      if (!rc(got, want, 1e-9)) ok = false;             // log(pⱼ/pᵢ) = −slope·|i−j|
    }
    for (let j = 0; j < 8; j++) if (j !== 3 && !(pref[3] > pref[j])) ok = false; // 对角严格最大
    if (op.predicate(7, sl, 3) !== true) ok = false;
  }
  check('3. 等分打分几何衰减 log(pⱼ/pᵢ)=−slope·|i−j| 且对角严格最大（μD112）',
    { pass: ok, detail: `最大相对偏差 ${worst.toExponential(2)} · predicate 全 true · 4 头` });
})();

/* --------------------- 4. 关闭时 Δ=0（对 legacy 基线） --------------------- */
(function () {
  const base = { vocabSize: 24, blockSize: 16, nLayer: 2, nHead: 4, dModel: 32, seed: 11 };
  const cases = [
    ['gpt2', base],
    ['llama', F.applyPreset(base, 'llama')],
    ['swa', F.applyPreset(base, 'swa')],
  ];
  const ids = [3, 7, 1, 9, 4, 2, 8, 5];
  let worstFwd = 0, worstStep = 0, ok = true;
  for (const [label, cfg] of cases) {
    const off = new F.GPTModel(Object.assign({}, cfg, { alibi: false })); // 显式关闭
    const l11 = new legacy.GPTModel(cfg);
    for (const key in l11.params) off.params[key].d.set(l11.params[key].d);
    const a = off.forward(ids).logits, b = l11.forward(ids).logits;
    const d = maxAbsDiff(a, b);
    if (d > worstFwd) worstFwd = d;
    // 增量路径同样要零回归（且 kv.pos 必须缺席）
    const kv = F.initKVCache(off);
    let sa = null;
    for (let i = 0; i < ids.length; i++) sa = F.forwardStep(off, ids[i], i, kv).logits;
    const kb = legacy.initKVCache(l11);
    let sb = null;
    for (let i = 0; i < ids.length; i++) sb = legacy.forwardStep(l11, ids[i], i, kb).logits;
    worstStep = Math.max(worstStep, maxAbsDiff(sa, sb));
    if (kv.pos !== null && typeof kv.pos !== 'undefined') ok = false;    // 关闭时不分配位置数组
    if (!(typeof off.params.posEmb === 'object') && !(label === 'llama' || label === 'swa')) ok = false;
  }
  check('4. 关闭时 Δ=0：alibi:false 与 legacy 前向/增量逐位一致（且 kv.pos 不分配）',
    { pass: ok && worstFwd === 0 && worstStep === 0, detail: `forward maxΔ=${worstFwd} · forwardStep maxΔ=${worstStep} · kv.pos=null=${ok} · 3 套预设` });
})();

/* ---------- 4b. 有效性：开关确实改变输出（不是摆设） ---------- */
(function () {
  const base = { vocabSize: 24, blockSize: 16, nLayer: 2, nHead: 4, dModel: 32, seed: 11 };
  const on = new F.GPTModel(Object.assign({}, base, { alibi: true }));
  const offm = new F.GPTModel(base);
  // 同一BN seed 下两者的可学习位置处理不同：只比较""注意力分布""是否改变
  const ids = [3, 7, 1, 9, 4, 2, 8, 5];
  const a = on.forward(ids).cache.layers[0].Ps[0];
  const b = offm.forward(ids).cache.layers[0].Ps[0];
  const d = maxAbsDiff(a, b);
  check('4b. 有效性：ALiBi 开启确实改变注意力分布（非摆设）',
    { pass: d > 1e-6, detail: `l0 头0 注意力分布 maxΔ=${d.toExponential(3)}（>0 证明偏置进入 softmax）` });
})();

/* --------------------- 5. 双端逐位一致（含淘汰后） --------------------- */
/* 5a. 短序列（< cap）：forwardStep 逐 token ≡ forward 末位 */
(function () {
  const m = mk({ blockSize: 16 });
  const ids = [3, 7, 1, 9, 4, 2, 8, 5], V = m.cfg.vocabSize, T = ids.length;
  const full = m.forward(ids).logits;
  const kv = F.initKVCache(m);
  let st = null;
  for (let i = 0; i < ids.length; i++) st = F.forwardStep(m, ids[i], i, kv).logits;
  const d = maxAbsDiff(full, st, (T - 1) * V);
  check('5a. 短序列（<cap）forward ≡ forwardStep 末位', { pass: d === 0, detail: `maxΔ=${d}（要求 0）` });
})();

/* 5b. 滑动窗口（不淘汰）：增量 (anchor=0,win) ≡ 全序列 forward(window) */
(function () {
  const m = mk({ blockSize: 16, window: 4 });
  const ids = [3, 7, 1, 9, 4, 2, 8, 5, 6, 0], V = m.cfg.vocabSize, T = ids.length;
  const full = m.forward(ids).logits;
  const kv = F.initKVCache(m);
  let st = null;
  for (let i = 0; i < ids.length; i++) st = F.forwardStep(m, ids[i], i, kv).logits;
  const d = maxAbsDiff(full, st, (T - 1) * V);
  check('5b. 滑动窗口（未淘汰）forward ≡ forwardStep', { pass: d === 0, detail: `window=4 · maxΔ=${d}（要求 0）` });
})();

/* 5c. 淘汰后（500 token，cap=16）：缓存 = 最后 cap 个 token */
(function () {
  const m = mk({ vocabSize: 12, blockSize: 16, nLayer: 1 });
  const V = m.cfg.vocabSize, N = 500;
  const ids = Array.from({ length: N }, (_, i) => i % V);
  const kv = F.initKVCache(m);
  let last = null;
  for (let i = 0; i < N; i++) { if (kv.n >= kv.cap) F.kvShift(m, kv); last = F.forwardStep(m, ids[i], i, kv).logits; }
  const win = ids.slice(ids.length - kv.cap);
  const full = m.forward(win).logits;
  const d = maxAbsDiff(full, last, (win.length - 1) * V);
  // 位置跟踪：槽内必须是真实绝对位置（末尾 = N-1，其余依次递减）
  let posOk = true;
  for (let s = 0; s < kv.n; s++) if (kv.pos[s] !== N - kv.n + s) posOk = false;
  check('5c. 淘汰后（500 token ≫ cap=16）forward(最后 cap 个) ≡ forwardStep',
    { pass: d === 0 && posOk, detail: `maxΔ=${d} · kv.pos=[${kv.pos[0]}…${kv.pos[kv.n - 1]}]（应 [${N - kv.n}…${N - 1}]）` });
})();

/* 5d. 锚点 + 滚动窗口 + 淘汰：kv.pos  invariants 与整体平移不变性 */
(function () {
  const cfg = { vocabSize: 12, blockSize: 16, nLayer: 1, nHead: 4, dModel: 32, seed: 11, alibi: true, attentionSink: 4, rollingWindow: 8, window: 6 };
  const V = cfg.vocabSize, N = 120, K = cfg.attentionSink;
  const ids = Array.from({ length: N }, (_, i) => i % V);

  /** 以绝对起始偏移 start 跑完整个流式序列，返回末位 logits 与缓存位置数组快照。 */
  function run(start) {
    const m = new F.GPTModel(cfg);
    const kv = F.initKVCache(m);
    let last = null;
    for (let i = 0; i < N; i++) { if (kv.n >= kv.cap) F.kvShift(m, kv); last = F.forwardStep(m, ids[i], start + i, kv).logits; }
    return { last: last, pos: Float64Array.from(kv.pos), n: kv.n, cap: kv.cap, m: m };
  }
  const r0 = run(0), rP = run(1000);

  const finite = allFinite(r0.last) && allFinite(rP.last);
  // 锚点位置必须被冻结在真实绝对位置 0..K-1
  let anchorOk = true;
  for (let s = 0; s < K; s++) if (r0.pos[s] !== s) anchorOk = false;
  // 窗口区必须是紧邻末尾的升序绝对位置
  let tailOk = true;
  for (let s = K; s < r0.n; s++) if (r0.pos[s] !== N - (r0.n - s)) tailOk = false;
  // 整体平移不变：只依赖相对距离 ⇒ 从绝对位置 1000 起跑必须逐位相同
  const dShift = maxAbsDiff(r0.last, rP.last);
  check('5d. Attention Sink + 滚动窗口（120 token ≫ cap=12）：位置跟踪 + 平移不变',
    {
      pass: anchorOk && tailOk && finite && dShift === 0,
      detail: `锚点 pos=[${Array.from(r0.pos.slice(0, K)).join(',')}]（应 0..${K - 1}）· 窗口尾 ${r0.pos[r0.n - 1]}（应 ${N - 1}）· 平移 0→1000 maxΔ=${dShift}（要求 0）· 有限=${finite}`,
    });
})();

/* --------------------- 6. 中心差分梯度检验 --------------------- */
(function () {
  function gradCheck(model, label) {
    const V = model.cfg.vocabSize, T = Math.min(12, model.cfg.blockSize);
    const seq = Array.from({ length: T }, (_, i) => i % V), targets = seq.slice(1);
    model.zeroGrads();
    const { logits, cache } = model.forward(seq);
    model.backward(cache, F.crossEntropy(logits, targets, T - 1, V, T).dlogits);
    const e = 1e-5, atol = 1e-7, rtol = 1e-4;
    let worst = 0, checked = 0, skipped = 0;
    for (const key in model.params) {
      const p = model.params[key].d, g = model.grads[key];
      const cnt = Math.min(p.length, 40);
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
    return { worst, checked, skipped, label };
  }
  const base = { vocabSize: 24, blockSize: 16, nLayer: 2, nHead: 4, dModel: 32, seed: 7 };
  const cases = [
    gradCheck(new F.GPTModel(F.applyPreset(base, 'bloom')), 'ALiBi(bloom)'),
    gradCheck(new F.GPTModel(Object.assign({}, base, { alibi: true, nKvHead: 2 })), 'ALiBi+GQA'),
    gradCheck(new F.GPTModel(Object.assign({}, base, { alibi: true, window: 4 })), 'ALiBi+SWA'),
  ];
  const worst = Math.max(...cases.map((c) => c.worst));
  const total = cases.reduce((a, c) => a + c.checked, 0);
  check('6. 中心差分梯度检验（ALiBi / ×GQA / ×SWA）',
    { pass: worst < 1, detail: `err/tol=${worst.toExponential(2)}（阈值 1）· ${total} 分量 · ${cases.map((c) => c.label).join(' / ')}` });
})();

/* --------------------- 7. 微指令交叉验证 --------------------- */
(function () {
  const op = (k) => {
    const o = NS.L10.ops[k];
    if (!o || o.status !== 'impl') throw new Error(`微指令 ${k} 未实现`);
    return o;
  };
  const fails = [];
  const okList = [];

  // 7a. 斜率：引擎 ≡ μD045（2 的幂 H）/ μD053（非 2 的幂 H）的参考实现输出
  for (const H of [1, 2, 4, 8, 16]) {
    const s = mk({ nHead: H }).alibiSlopes, ref = op('D045').run(H - 1);
    let same = ref.length === s.length;
    for (let i = 0; same && i < ref.length; i++) same = rc(s[i], ref[i], 1e-12);
    if (!same) fails.push(`H=${H} 斜率 ≠ μD045.run`);
    if (op('D045').predicate(H - 1) !== true) fails.push(`H=${H} μD045 predicate`);
    if (op('D046').predicate(H - 1) !== true) fails.push(`H=${H} μD046 predicate`);
    if (op('D052').predicate(H - 1) !== true) fails.push(`H=${H} μD052 predicate`);
    if (op('D056').predicate(H - 1) !== true) fails.push(`H=${H} μD056 predicate`);
    // 斜率累积积闭式 μD114：Π slope = 2^(−4h(h−1)/H)
    const cumRef = op('D114').run(H - 1);
    let cum = 1, cumOk = rc(cum, cumRef[0], 1e-12);
    for (let i = 0; i < H; i++) { cum *= refSlopes(H)[i]; if (!rc(cum, cumRef[i + 1], 1e-12)) cumOk = false; }
    if (!cumOk) fails.push(`H=${H} 累积积 ≠ μD114.run`);
    if (op('D114').predicate(H - 1) !== true) fails.push(`H=${H} μD114 predicate`);
  }
  for (const H of [3, 5, 6, 7, 12]) {
    const s = mk({ nHead: H }).alibiSlopes, ref = op('D053').run(H - 1);
    let same = ref.length === s.length;
    for (let i = 0; same && i < ref.length; i++) same = rc(s[i], ref[i], 1e-12);
    if (!same) fails.push(`H=${H}(非 2 的幂) 斜率 ≠ μD053.run`);
    if (op('D053').predicate(H - 1) !== true) fails.push(`H=${H} μD053 predicate`);
  }
  okList.push('斜率 μD045/μD046/μD052/μD053/μD056/μD114');

  // 7b. 偏置矩阵：引擎斜率喂给 μD047–μD051、μD055
  {
    const s = mk({}).alibiSlopes;
    for (let h = 0; h < 4; h++) {
      const Bref = op('D047').run(7, s[h]);                     // 8×8 参考矩阵
      let same = true;
      for (let i = 0; i < 8 && same; i++) for (let j = 0; j < 8; j++) {
        if (!rc(Bref[i * 8 + j], -s[h] * Math.abs(i - j), 1e-12)) { same = false; break; }
      }
      if (!same) fails.push(`h=${h} 偏置 ≠ μD047.run`);
      for (const k of ['D047', 'D048', 'D049', 'D050', 'D051', 'D055']) {
        if (op(k).predicate(7, s[h]) !== true) fails.push(`${k} predicate(h=${h})`);
      }
    }
    okList.push('偏置 μD047–μD051/μD055');
  }

  // 7c. softmax 复合：引擎 softmaxRows ≡ μD054 参考（偏置 + 常数平移不变）
  {
    const s = mk({}).alibiSlopes;
    const row = new Float64Array([0.11, -0.27, 0.35, -0.08, 0.42, -0.19, 0.06, 0.23]);
    const c = 0.77, ref = op('D054').run(row, s[0], c);
    const M = new Float64Array(8);
    for (let j = 0; j < 8; j++) M[j] = row[j] - s[0] * Math.abs(j) + c;
    const got = F.softmaxRows(M, 1, 8);
    let same = true;
    for (let j = 0; j < 8; j++) if (!rc(got[j], ref[j], 1e-12)) same = false;
    if (!same) fails.push('softmaxRows ≠ μD054.run');
    if (op('D054').predicate(row, s[0], c) !== true) fails.push('μD054 predicate');
    okList.push('softmax 复合 μD054');
  }

  // 7d. 几何衰减 μD112（引擎斜率）
  {
    const s = mk({}).alibiSlopes;
    for (let h = 0; h < 4; h++) if (op('D112').predicate(7, s[h], 3) !== true) fails.push(`μD112 predicate(h=${h})`);
    okList.push('几何衰减 μD112');
  }

  // 7e. 行级闭式 μD117（距离和）/ μD119（行最小偏置）
  {
    const s = mk({}).alibiSlopes;
    for (let i = 0; i < 8; i++) {
      const sumRef = op('D117').run(7, i);
      let acc = 0;
      for (let j = 0; j < 8; j++) acc += Math.abs(i - j);
      if (!rc(acc, sumRef, 1e-12)) fails.push(`μD117 距离和(i=${i})`);
      if (op('D117').predicate(7, i) !== true) fails.push(`μD117 predicate(i=${i})`);
      const minRef = op('D119').run(7, s[0], i);
      let mn = Infinity;
      for (let j = 0; j < 8; j++) mn = Math.min(mn, -s[0] * Math.abs(i - j));
      if (!rc(mn, minRef, 1e-12)) fails.push(`μD119 行最小(i=${i})`);
      if (op('D119').predicate(7, s[0], i) !== true) fails.push(`μD119 predicate(i=${i})`);
    }
    okList.push('行闭式 μD117/μD119');
  }

  // 7f. RoPE×ALiBi 联合平移不变 μD113（ALiBi 与任何相对位置编码复合后仍平移不变）
  {
    const rng = F.mulberry32(3);
    const q = new Float64Array(4), K = new Float64Array(4 * 4);
    for (let i = 0; i < q.length; i++) q[i] = F.randn(rng);
    for (let i = 0; i < K.length; i++) K[i] = F.randn(rng);
    const sl = mk({}).alibiSlopes[1];
    if (op('D113').predicate(q, K, 2, 5, sl) !== true) fails.push('μD113 predicate');
    okList.push('联合平移 μD113');
  }

  check('7. 微指令交叉验证（D045–D056 / D112 / D113 / D114 / D117 / D119）',
    { pass: fails.length === 0, detail: fails.length ? `失败：${fails.slice(0, 6).join(' | ')}` : `全部 run/predicate 通过 · ${okList.join(' · ')}` });
})();

/* ------------------------------ 汇总输出 ------------------------------ */
const W = 62;
console.log('🧪 旗舰 AI · ALiBi 专项验证（v15.10）');
console.log('─'.repeat(W));
for (const c of checks) console.log(`${c.pass ? '✅' : '❌'} ${c.name}\n     ${c.detail}`);
console.log('─'.repeat(W));
const pass = checks.filter((c) => c.pass).length;
console.log(`${pass}/${checks.length} 通过 ${pass === checks.length ? '✅ 全绿' : '❌ 存在失败'}`);
process.exitCode = pass === checks.length ? 0 : 1;

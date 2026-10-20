#!/usr/bin/env node
'use strict';
/* 旗舰 AI v12 · 零回归 Harness（头等公民）
 * 同进程内加载 v11(legacy) 与 v12(index)，逐元素 Object.is 比对。
 *
 * 用法：
 *   node scripts/regression-v11.js                  # 全量比对（默认）
 *   node scripts/regression-v11.js --filter=R4      # 只跑某用例
 *   node scripts/regression-v11.js --emit-golden    # 生成 data/golden-v11.node<major>.json
 *
 * 判据：Δ=0 用例用 Object.is 逐元素全等；容差用例用 maxΔ < tol。
 * Phase 0：v12 === legacy，所有用例必然全绿；后续 Phase 拆层后逐位守护 R1–R10。
 *
 * 同进程是硬要求：Math.cos/pow/tanh 等超越函数在跨 V8 版本间不保证逐位一致，
 * 跨 Node 版本比对会产生假阳性。golden 按 Node major 分文件即为此。
 */
const path = require('path');
const fs = require('fs');
const { execFileSync } = require('child_process');

const V11 = require('../src/legacy/flagship-engine.v11.js');
const V12 = require('../src/index.js');

const ARGS = process.argv.slice(2);
const FILTER = ARGS.find((a) => a.startsWith('--filter='));
const EMIT_GOLDEN = ARGS.includes('--emit-golden');
const NODE_MAJOR = process.versions.node.split('.')[0];

// ----------------------------------------------------------------- 工具
function assertBitExact(a, b, ctx) {
  if (typeof a === 'number' && typeof b === 'number') return Object.is(a, b) ? 0 : Infinity;
  if (!a || !b || a.length !== b.length) return Infinity;
  let maxD = 0;
  for (let i = 0; i < a.length; i++) {
    if (!Object.is(a[i], b[i])) {
      const d = Math.abs(a[i] - b[i]);
      if (d > maxD) maxD = d;
    }
  }
  return maxD;
}
function maxDelta(a, b) {
  if (typeof a === 'number' && typeof b === 'number') return Math.abs(a - b);
  if (!a || !b || a.length !== b.length) return Infinity;
  let md = 0;
  for (let i = 0; i < a.length; i++) md = Math.max(md, Math.abs(a[i] - b[i]));
  return md;
}
const CFG = { vocabSize: 65, blockSize: 16, nLayer: 2, nHead: 4, dModel: 64 };
const PRESETS = ['gpt2', 'llama', 'mixtral', 'swa', 'gemma'];
const SEEDS = [1, 2, 3];

function buildModel(api, preset, cfg, seed) {
  const c = api.applyPreset(
    { vocabSize: cfg.vocabSize, blockSize: cfg.blockSize, nLayer: cfg.nLayer, nHead: cfg.nHead, dModel: cfg.dModel, seed },
    preset
  );
  return new api.GPTModel(c);
}
function mkTokens(api, seed, n) {
  const rng = api.mulberry32(seed >>> 0);
  const a = new Array(n);
  for (let i = 0; i < n; i++) a[i] = Math.floor(rng() * CFG.vocabSize);
  return a;
}

// ----------------------------------------------------------------- 用例
const results = [];
function check(name, fn) {
  if (FILTER && !name.startsWith(FILTER.split('=')[1])) return;
  let pass = true, detail = '';
  try {
    const r = fn() || {};
    pass = r.pass !== false; detail = r.detail || '';
  } catch (e) {
    pass = false; detail = '异常: ' + (e && e.message);
  }
  results.push({ name, pass, detail });
  console.log(` ${pass ? '✅' : '❌'} ${name.padEnd(34)} ${detail}`);
}

check('R1 params Δ=0', () => {
  let md = 0;
  for (const p of PRESETS) for (const s of SEEDS) {
    const m11 = buildModel(V11, p, CFG, s), m12 = buildModel(V12, p, CFG, s);
    for (const k in m11.params) md = Math.max(md, assertBitExact(m11.params[k].d, m12.params[k].d, k));
  }
  return { pass: md === 0, detail: 'params maxΔ=' + md };
});

check('R2 forward logits Δ=0', () => {
  let md = 0;
  for (const p of PRESETS) for (const s of SEEDS) {
    const m11 = buildModel(V11, p, CFG, s), m12 = buildModel(V12, p, CFG, s);
    const seq = mkTokens(V11, 1000 + s, CFG.blockSize).slice(0, CFG.blockSize - 1);
    const f11 = m11.forward(seq), f12 = m12.forward(seq);
    md = Math.max(md, assertBitExact(f11.logits, f12.logits));
  }
  return { pass: md === 0, detail: 'logits maxΔ=' + md };
});

check('R3 backward grads Δ=0', () => {
  let md = 0;
  for (const p of PRESETS) for (const s of SEEDS) {
    const m11 = buildModel(V11, p, CFG, s), m12 = buildModel(V12, p, CFG, s);
    const seq = mkTokens(V11, 2000 + s, CFG.blockSize).slice(0, CFG.blockSize - 1);
    const targets = seq.slice(1);
    const f11 = m11.forward(seq), f12 = m12.forward(seq);
    const ce11 = V11.crossEntropy(f11.logits, targets, seq.length - 1, CFG.vocabSize, seq.length);
    const ce12 = V12.crossEntropy(f12.logits, targets, seq.length - 1, CFG.vocabSize, seq.length);
    m11.zeroGrads(); m12.zeroGrads();
    m11.backward(f11.cache, ce11.dlogits); m12.backward(f12.cache, ce12.dlogits);
    for (const k in m11.grads) md = Math.max(md, assertBitExact(m11.grads[k], m12.grads[k]));
  }
  return { pass: md === 0, detail: 'grads maxΔ=' + md };
});

check('R4 train loss 序列 + 末态 params Δ=0', () => {
  let md = 0, mdParams = 0;
  for (const p of PRESETS) {
    const s = 7;
    const m11 = buildModel(V11, p, CFG, s), m12 = buildModel(V12, p, CFG, s);
    const toks = mkTokens(V11, 9999, 500);
    const h11 = [], h12 = [];
    V11.train(m11, toks, { steps: 200, seed: 42, onStep: (step, loss) => h11.push(loss) });
    V12.train(m12, toks, { steps: 200, seed: 42, onStep: (step, loss) => h12.push(loss) });
    for (let i = 0; i < h11.length; i++) md = Math.max(md, Math.abs(h11[i] - h12[i]));
    for (const k in m11.params) mdParams = Math.max(mdParams, assertBitExact(m11.params[k].d, m12.params[k].d));
  }
  return { pass: md === 0 && mdParams === 0, detail: 'loss maxΔ=' + md + ' · params maxΔ=' + mdParams };
});

check('R5 generate ids+text Δ=0', () => {
  let ok = true, firstDiff = '';
  for (const p of PRESETS) for (const s of SEEDS) for (const useCache of [true, false]) {
    const m11 = buildModel(V11, p, CFG, s), m12 = buildModel(V12, p, CFG, s);
    const tok = new V11.Tokenizer('abcdefghijklmnopqrstuvwxyz 0123456789');
    const g11 = V11.generate(m11, tok, 'abc', { maxNew: 20, seed: s, useCache });
    const g12 = V12.generate(m12, tok, 'abc', { maxNew: 20, seed: s, useCache });
    if (g11.text !== g12.text || JSON.stringify(g11.ids) !== JSON.stringify(g12.ids)) {
      ok = false;
      if (!firstDiff) firstDiff = `${p}/s${s}/cache${useCache}: "${g11.text}" vs "${g12.text}"`;
    }
  }
  return { pass: ok, detail: ok ? '完全一致' : firstDiff };
});

check('R6 KV 增量 vs 全量重算 Δ=0', () => {
  let ok = true, firstDiff = '';
  for (const p of PRESETS) for (const s of SEEDS) {
    const m11 = buildModel(V11, p, CFG, s), m12 = buildModel(V12, p, CFG, s);
    const tok = new V11.Tokenizer('abcdefghijklmnopqrstuvwxyz 0123456789');
    // 注意：learned-pos（gpt2）超 blockSize 后窗口滑动为有界近似，非逐位等价；
    // 故 KV 增量 vs 全量比对限制在 < blockSize 内（3+10=13<16），保证严格等价。
    const g11 = V11.generate(m11, tok, 'abc', { maxNew: 10, seed: s, useCache: true });
    const g12 = V12.generate(m12, tok, 'abc', { maxNew: 10, seed: s, useCache: false });
    if (g11.text !== g12.text) { ok = false; if (!firstDiff) firstDiff = `${p}/s${s}: cache vs full mismatch`; }
  }
  return { pass: ok, detail: ok ? '增量≡全量' : firstDiff };
});

check('R7 存档往返 Δ=0', () => {
  let md = 0;
  for (const p of PRESETS) {
    const m11 = buildModel(V11, p, CFG, 11);
    const tok = new V11.Tokenizer('abcdefghijklmnopqrstuvwxyz');
    const seq = mkTokens(V11, 500, CFG.blockSize).slice(0, CFG.blockSize - 1);
    const f1 = m11.forward(seq);
    // 路径 1：v11 存档 → v12 恢复（跨实现互认；T04 切换后 v12.loadModel 恢复出的
    //   必须是十层 L3 模型而非 legacy 模型，故本路径是 L3 接管存档闭环的真门禁）
    const j1 = V11.saveModel(m11, tok);
    const m12a = V12.loadModel(j1);
    md = Math.max(md, assertBitExact(f1.logits, m12a.forward(seq).logits));
    // 路径 2：v12 存档 → v12 恢复（纯 v12 闭环）
    const j2 = V12.saveModel(m11, tok);
    const m12b = V12.loadModel(j2);
    md = Math.max(md, assertBitExact(f1.logits, m12b.forward(seq).logits));
    // 词表恢复：chars 逐位一致（saveModel 一并存词表的既定契约）
    const t2 = V12.loadTokenizer(j2);
    if (!t2 || t2.vocabSize !== tok.vocabSize || JSON.stringify(t2.chars) !== JSON.stringify(tok.chars)) {
      return { pass: false, detail: 'loadTokenizer 词表恢复不一致' };
    }
  }
  return { pass: md === 0, detail: 'logits maxΔ=' + md };
});

check('R8 quantize logits maxΔ<0.03', () => {
  let md = 0;
  for (const p of ['mixtral', 'gemma', 'swa']) {
    const m11 = buildModel(V11, p, CFG, 13); m11.quantize();
    const m12 = buildModel(V12, p, CFG, 13); m12.quantize();
    const seq = mkTokens(V11, 600, CFG.blockSize).slice(0, CFG.blockSize - 1);
    const f11 = m11.forward(seq), f12 = m12.forward(seq);
    md = Math.max(md, maxDelta(f11.logits, f12.logits));
  }
  return { pass: md < 0.03, detail: 'quant maxΔ=' + md.toFixed(5) };
});

check('R9 逐算子 Δ=0', () => {
  const rng = V11.mulberry32(77 >>> 0);
  const R = () => rng() * 2 - 1;
  function rmat(r, c) { const a = new Float64Array(r * c); for (let i = 0; i < a.length; i++) a[i] = R(); return a; }
  let md = 0;
  const A = rmat(4, 3), B = rmat(3, 5);
  md = Math.max(md, assertBitExact(V11.matmul(A, B, 4, 3, 5), V12.matmul(A, B, 4, 3, 5)));
  const S = rmat(3, 4);
  md = Math.max(md, assertBitExact(V11.softmaxRows(S, 3, 4), V12.softmaxRows(S, 3, 4)));
  const g = rmat(1, 4), b = rmat(1, 4);
  md = Math.max(md, assertBitExact(V11.layernormFwd(S, g, b, 3, 4).Y, V12.layernormFwd(S, g, b, 3, 4).Y));
  md = Math.max(md, assertBitExact(V11.rmsnormFwd(S, g, b, 3, 4).Y, V12.rmsnormFwd(S, g, b, 3, 4).Y));
  md = Math.max(md, assertBitExact(V11.geluFwd(S), V12.geluFwd(S)));
  md = Math.max(md, assertBitExact(V11.siluF(S), V12.siluF(S)));
  const D = 8, H = 2, dh = 4, XX = rmat(4 * 3 * D, 1);
  const XX11 = Float64Array.from(XX), XX12 = Float64Array.from(XX);
  V11.ropeApply(XX11, 4, D, H, dh, 0, 1); // 原地修改输入（探针同样直接比输入）
  V12.ropeApply(XX12, 4, D, H, dh, 0, 1);
  md = Math.max(md, assertBitExact(XX11, XX12));
  const logits = rmat(4, 5), targets = [0, 1, 2, 3];
  const ce1 = V11.crossEntropy(logits, targets, 4, 5, 4), ce2 = V12.crossEntropy(logits, targets, 4, 5, 4);
  md = Math.max(md, assertBitExact(ce1.dlogits, ce2.dlogits), ce1.loss === ce2.loss ? 0 : Infinity);
  const r1 = V11.mulberry32(123), r2 = V12.mulberry32(123);
  for (let i = 0; i < 100; i++) md = Math.max(md, r1() === r2() ? 0 : Infinity);
  return { pass: md === 0, detail: 'maxΔ=' + md };
});

check('R10 8 probe 全绿（断言快照）', () => {
  const probes = ['probe-v3', 'probe-v4-gqa', 'probe-v5-quant', 'probe-moe', 'probe-moe-quant', 'probe-swa', 'probe-gemma', 'probe-gemma2'];
  let fail = 0, failed = [];
  for (const pr of probes) {
    try { execFileSync('node', [path.join(__dirname, pr + '.js')], { stdio: 'pipe' }); }
    catch (e) { fail++; failed.push(pr); }
  }
  return { pass: fail === 0, detail: fail === 0 ? '8/8 探针 exit 0' : '失败: ' + failed.join(',') };
});

// ----------------------------------------------------------------- 汇总 / golden
const passCount = results.filter((r) => r.pass).length;
console.log('─'.repeat(46));
console.log(`零回归门禁：R1–R10 共 ${passCount}/${results.length} 通过 ${passCount === results.length ? '✅ 全绿' : '❌ 存在失败'}`);

if (EMIT_GOLDEN) {
  const golden = {
    node: NODE_MAJOR,
    generatedAt: new Date().toISOString(),
    results: results.map((r) => ({ name: r.name, pass: r.pass, detail: r.detail })),
  };
  const f = path.join(__dirname, '..', 'data', `golden-v11.node${NODE_MAJOR}.json`);
  fs.writeFileSync(f, JSON.stringify(golden, null, 2));
  console.log(`📸 golden 快照已生成：${f}`);
}

process.exitCode = passCount === results.length ? 0 : 1;

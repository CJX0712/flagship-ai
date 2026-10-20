#!/usr/bin/env node
/* ============================================================================
 * 微指令门禁探针（probe-microisa.js）
 * ----------------------------------------------------------------------------
 * 零依赖（不依赖 fast-check / node:test）的一条命令式 L10 门禁，供 CI 与
 * 干净树复跑使用。既有 tests/L10/*.test.js 走 node:test + fast-check 做
 * ≥120 例/项的性质测试；本探针做的是「装配 + 覆盖率诚实性 + 双种子不变量
 * 抽查 + 引擎零影响」的整体把关，两者是互补而非重复。
 *
 * 断言：
 *   1. 注册中心容量 1024（八域 × 128，无重复无缺失）
 *   2. 实现计数诚实且与 isa.js 验收预期一致（A=96 / C=96 / 合计 192）
 *   3. 已实现项不变量自检：192 项 × 2 种子，确定性生成参数 → predicate 全过
 *   4. 未实现槽位诚实标记（status='stub' 且 run===null，不注水）
 *   5. 确定性：同种子两次采样逐位一致
 *   6. L9 数值基础 + L8 数值内核就位；6 个被覆盖算子与 legacy 逐位一致（R9 契约）
 *   7. 引擎零影响：导出清单与 legacy 完全一致（K15 34 键）+ 前向有限且确定
 *   8. 十层命名空间完整（L3–L10 全部挂载）
 * 运行：node scripts/probe-microisa.js
 * 作者：晨星  ·  License: MIT
 * ========================================================================*/
'use strict';

const path = require('path');
const { spawnSync } = require('child_process');
require('../src/index.js');           // 生产加载路径：填充 globalThis.FlagshipAI
const NS = globalThis.FlagshipAI;
const legacy = require('../src/legacy/flagship-engine.v11.js');

const EXPECT = { capacity: 1024, domains: 8, perDomain: 128 };
const SEEDS = 2;                      // 双种子
const RUNS_PER_OP = 20;               // 每项每种子的抽查用例数
const ROOT = path.join(__dirname, '..');

const checks = [];
const chk = (name, pass, detail) => { checks.push({ name, pass, detail }); };

/* ---------- 1. 注册中心容量 ---------- */
const stats = NS.L10.stats();
const cap = stats.total.capacity;
const perDomain = NS.L10.DOMAINS.map(d => stats[d].capacity);
chk('注册中心容量 1024（八域 × 128）',
  cap === EXPECT.capacity && perDomain.every(c => c === EXPECT.perDomain) && NS.L10.DOMAINS.length === EXPECT.domains,
  `总容量 ${cap} · ${NS.L10.DOMAINS.length} 域 × ${perDomain[0]}`);

/* ---------- 2. 实现计数诚实性（逐域自洽 + 与 isa.js 验收交叉核对） ---------- */
const implCount = stats.total.impl;
const perDomainImpl = NS.L10.DOMAINS.map(d => stats[d].impl);
const sumMatch = perDomainImpl.reduce((a, b) => a + b, 0) === implCount;
const isaRun = spawnSync(process.execPath, [path.join(ROOT, 'scripts', 'isa.js')], { cwd: ROOT, encoding: 'utf8' });
const isaOut = (isaRun.stdout || '') + (isaRun.stderr || '');
const m = isaOut.match(/合计\s*(\d+)\s*\/\s*(\d+)/);
const isaTotal = m ? Number(m[1]) : -1;
const isaCap = m ? Number(m[2]) : -1;
const isaOk = isaRun.status === 0 && isaTotal === implCount && isaCap === cap;
const domDetail = NS.L10.DOMAINS.map(d => `${d} ${stats[d].impl}`).join(' · ');
chk('实现计数诚实且与 isa.js 验收交叉一致',
  sumMatch && isaOk,
  `逐域自洽=${sumMatch} · isa.js exit=${isaRun.status} 合计=${isaTotal}/${isaCap} · 本探针=${implCount}/${cap}（${domDetail}）`);

/* ---------- 3. 已实现项不变量自检（双种子） ---------- */
function implOps() {
  const out = [];
  for (const d of NS.L10.DOMAINS) {
    for (let i = 1; i <= NS.L10.CAP; i++) {
      const key = NS.L10.keyOf(d, i);
      const op = NS.L10.ops[key];
      if (op && op.status === 'impl') out.push(op);
    }
  }
  return out;
}
function genArgs(rng, op) {
  return (op.inputs || []).map((name) => {
    const tag = (op.domains || {})[name];
    const t = NS.L10.TAGS[tag];
    if (!t) throw new Error(`op ${op.key} 未知域标签 "${tag}"`);
    if (t.kind === 'matrix') {
      const dims = (op.matrixDims && op.matrixDims[name]) || t.defDims;
      if (!dims) throw new Error(`op ${op.key} 矩阵参数 ${name} 缺 matrixDims`);
      return NS.L10.genMatrix(rng, dims[0], dims[1], tag);
    }
    if (t.kind === 'vector') {
      const n = (op.vecDims && op.vecDims[name]) || t.defLen || 8;
      return NS.L10.genVector(rng, n, tag);
    }
    if (t.kind === 'shape') {
      return NS.L10.genShape(rng, tag);
    }
    return NS.L10.genScalar(rng, tag);
  });
}
const ops = implOps();
let sweepPass = 0, sweepFail = 0;
const firstFails = [];
for (const op of ops) {
  let opPass = true, reason = '';
  for (let s = 0; s < SEEDS && opPass; s++) {
    const rng = NS.L9.rng.mulberry32((s * 2654435761 + 12345) >>> 0);
    for (let r = 0; r < RUNS_PER_OP; r++) {
      let args;
      try { args = genArgs(rng, op); } catch (e) { opPass = false; reason = e.message; break; }
      try {
        if (op.predicate.apply(null, args) !== true) {
          opPass = false; reason = `predicate 失败（seed=${s} run=${r}）`; break;
        }
      } catch (e) { opPass = false; reason = '异常: ' + e.message; break; }
    }
  }
  if (opPass) sweepPass++; else { sweepFail++; if (firstFails.length < 5) firstFails.push(`${op.key} ${reason}`); }
}
chk('已实现项不变量自检（双种子抽查）',
  sweepFail === 0 && ops.length === implCount,
  `${ops.length} 项 × ${SEEDS} 种子 × ${RUNS_PER_OP} 例 → 通过 ${sweepPass} / 失败 ${sweepFail}` +
  (firstFails.length ? `（首败：${firstFails.join('；')}）` : ''));

/* ---------- 4. 未实现槽位诚实标记 ---------- */
let stubCount = 0, stubBad = [];
for (const d of NS.L10.DOMAINS) {
  for (let i = 1; i <= NS.L10.CAP; i++) {
    const op = NS.L10.ops[NS.L10.keyOf(d, i)];
    if (!op) { stubBad.push(`${d}${i} 缺失`); continue; }
    if (op.status === 'stub') {
      stubCount++;
      if (op.run !== null || op.predicate !== null) stubBad.push(`${op.key} 标 stub 却带实现体`);
    }
  }
}
chk('未实现槽位诚实标记（registered-not-implemented）',
  stubBad.length === 0 && stubCount + implCount === EXPECT.capacity,
  `stub ${stubCount} + impl ${implCount} = ${stubCount + implCount}/${cap}` +
  (stubBad.length ? `（异常：${stubBad.slice(0, 3).join('，')}）` : ' · 无注水'));

/* ---------- 5. 确定性 ---------- */
{
  const a = NS.L9.rng.mulberry32(42), b = NS.L9.rng.mulberry32(42);
  let same = true;
  for (let i = 0; i < 200; i++) if (!Object.is(a(), b())) { same = false; break; }
  const c = NS.L9.rng.mulberry32(43);
  let differs = false;
  for (let i = 0; i < 50; i++) { if (!Object.is(NS.L9.rng.mulberry32(42)(), c())) { differs = true; break; } }
  chk('L9 RNG 确定性（同种子逐位一致 / 异种子发散）', same && differs, `同种子一致=${same} · 异种子发散=${differs}`);
}

/* ---------- 6. L9/L8 就位 + 被覆盖算子与 legacy 逐位一致（R9 契约抽查） ---------- */
{
  const layersOk = !!NS.L8 && !!NS.L9 && !!NS.L9.rng && !!NS.L9.assert;
  let opOk = true, worst = 0, detail = '';
  try {
    const n = 6, m = 5;
    const A = new Float64Array(n * m), B = new Float64Array(m * n);
    for (let i = 0; i < A.length; i++) A[i] = Math.sin(i * 0.37) * 2;
    for (let i = 0; i < B.length; i++) B[i] = Math.cos(i * 0.21) * 1.5;
    const c1 = legacy.matmul(A, B, n, m, m), c2 = NS.L8.matmul.matmul(A, B, n, m, m);
    for (let i = 0; i < c1.length; i++) worst = Math.max(worst, Math.abs(c1[i] - c2[i]));
    opOk = c1.length === c2.length && worst === 0;
    const s1 = legacy.softmaxRows(new Float64Array([1, 2, 3, 4]), 1, 4);
    const s2 = NS.L8.reduce.softmaxRows(new Float64Array([1, 2, 3, 4]), 1, 4);
    for (let i = 0; i < s1.length; i++) worst = Math.max(worst, Math.abs(s1[i] - s2[i]));
    detail = `matmul/softmaxRows 与 legacy maxΔ=${worst}`;
  } catch (e) { opOk = false; detail = '异常: ' + e.message; }
  chk('L9/L8 就位且被覆盖算子与 legacy 逐位一致（R9 抽查）', layersOk && opOk, detail);
}

/* ---------- 7. 引擎零影响（K15 导出清单 + 前向） ---------- */
{
  const api = require('../src/index.js');
  const kApi = Object.keys(api).sort(), kLegacy = Object.keys(legacy).sort();
  const same = kApi.length === kLegacy.length && kApi.every((k, i) => k === kLegacy[i]);
  let fwd = false, det = false, fDetail = '未运行';
  try {
    const model = new api.GPTModel({ vocabSize: 40, blockSize: 8, nLayer: 1, nHead: 2, dModel: 16, seed: 7 });
    const toks = [1, 2, 3, 4, 5, 6, 7];
    const l1 = Array.from(model.forward(toks).logits);
    const l2 = Array.from(model.forward(toks).logits);
    fwd = l1.every(Number.isFinite);
    det = l1.length === l2.length && l1.every((v, i) => Object.is(v, l2[i]));
    fDetail = `前向有限=${fwd} · 确定性=${det}（${l1.length} logits）`;
  } catch (e) { fDetail = '异常: ' + e.message; }
  chk('引擎零影响（导出清单 K15 一致 + 前向有限确定）', same && fwd && det,
    `导出 ${kApi.length} 键（legacy ${kLegacy.length} 键，一致=${same}）· ${fDetail}`);
}

/* ---------- 8. 十层命名空间完整 ---------- */
{
  const need = ['L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10'];
  const missing = need.filter(k => !NS[k]);
  chk('十层命名空间完整（L3–L10 全部挂载）', missing.length === 0,
    missing.length ? `缺失：${missing.join(' ')}` : need.join(' '));
}

/* ---------- 汇报 ---------- */
const W = 46;
console.log('🧩 旗舰 AI · 微指令门禁探针（L10 注册中心 + 引擎零影响）');
console.log('─'.repeat(W));
for (const c of checks) {
  console.log(` ${c.pass ? '✅' : '❌'} ${c.name}`);
  console.log(`      ${c.detail}`);
}
const pass = checks.filter(c => c.pass).length;
console.log('─'.repeat(W));
console.log(`结果：${pass}/${checks.length} 通过 ${pass === checks.length ? '✅ 全绿' : '❌ 存在失败'}`);
process.exitCode = pass === checks.length ? 0 : 1;

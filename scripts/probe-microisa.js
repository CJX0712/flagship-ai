#!/usr/bin/env node
/* 旗舰 AI · L10 微指令层 + L9 数值基础层验证探针（Phase 1 CI 门禁）
 * 八项检查：
 *   1. 注册中心完整性：1024 槽位、八域 ×128、无重复
 *   2. 实现进度诚实核算：stats() 与期望逐域一致（不注水）
 *   3. 全部已实现微指令 selfTest 通过（seed=1）
 *   4. 同批 selfTest 双种子复跑（seed=2，防种子过拟合）
 *   5. 未实现项行为正确：抛错含正确 key、check 报 stub
 *   6. 确定性：stats/check 重复调用逐位一致
 *   7. L9 数值基础层注入完整且自检全绿
 *   8. 引擎零影响：公共入口导出 ≡ v11 引擎本体（R3），层加载不改引擎行为
 */
'use strict';
const path = require('path');
const F = require('../src/index.js');
const NS = globalThis.FlagshipAI;
const L10 = NS.L10;

const EXPECTED = { A: 28, B: 16, C: 12, D: 10, E: 13, F: 11, G: 15, H: 15 };
const EXPECTED_TOTAL = Object.values(EXPECTED).reduce((a, b) => a + b, 0);

const results = [];
function check(name, fn) {
  let pass = true, detail = '';
  try { const r = fn() || {}; pass = r.pass !== false; detail = r.detail || ''; }
  catch (e) { pass = false; detail = '异常: ' + (e && e.message); }
  results.push({ name, pass, detail });
  console.log(` ${pass ? '✅' : '❌'} ${name.padEnd(30)} ${detail}`);
}

/* 1. 注册中心完整性 */
check('注册中心 1024 槽位', () => {
  const keys = L10.keys();
  if (keys.length !== 1024) return { pass: false, detail: `keys=${keys.length}` };
  const seen = Object.create(null);
  for (const k of keys) { if (seen[k]) return { pass: false, detail: '重复 key: ' + k }; seen[k] = 1; }
  for (const d of Object.keys(EXPECTED)) {
    for (let n = 0; n < 128; n++) if (!seen[d + String(n).padStart(3, '0')]) return { pass: false, detail: `缺槽位 ${d}${n}` };
  }
  return { pass: true, detail: '八域 ×128 = 1024，无重复无缺失' };
});

/* 2. 实现进度诚实核算 */
check('实现进度诚实核算', () => {
  const s = L10.stats();
  if (s.total !== 1024 || s.done !== EXPECTED_TOTAL) return { pass: false, detail: `total=${s.total} done=${s.done}（期望 ${EXPECTED_TOTAL}）` };
  for (const d of Object.keys(EXPECTED)) {
    if (s.byDomain[d].done !== EXPECTED[d]) return { pass: false, detail: `${d} 域 done=${s.byDomain[d].done}（期望 ${EXPECTED[d]}）` };
  }
  return { pass: true, detail: `${EXPECTED_TOTAL}/1024 逐域一致（未实现项显式 stub）` };
});

/* 3+4. 双种子 selfTest */
for (const seed of [1, 2]) {
  check(`selfTest 全绿（seed=${seed}）`, () => {
    let pass = 0; const failed = [];
    for (const key of L10.keys()) {
      const op = L10.get(key);
      if (op.impl !== 'done') continue;
      const r = L10.check(key, seed);
      if (r.pass) pass++; else failed.push(`μ${key}: ${r.detail}`);
    }
    if (failed.length) return { pass: false, detail: failed.slice(0, 3).join(' | ') };
    return { pass: true, detail: `${pass}/${EXPECTED_TOTAL} 项不变量全过` };
  });
}

/* 5. 未实现项行为 */
check('未实现项行为正确', () => {
  for (const key of ['A100', 'B127', 'H127']) {
    try { L10.run(key, []); return { pass: false, detail: key + ' 未抛错' }; }
    catch (e) { if (!String(e.message).includes('μ' + key)) return { pass: false, detail: key + ' 抛错信息未含正确 key: ' + e.message }; }
  }
  const r = L10.check('A100', 1);
  if (r.pass !== false || r.impl !== 'stub') return { pass: false, detail: 'stub 的 check 应报 stub' };
  return { pass: true, detail: '抛错含正确 key + check 报 stub' };
});

/* 6. 确定性 */
check('确定性（重复调用一致）', () => {
  const s1 = JSON.stringify(L10.stats()), s2 = JSON.stringify(L10.stats());
  if (s1 !== s2) return { pass: false, detail: 'stats 两次不一致' };
  const r1 = JSON.stringify(L10.check('A000', 5)), r2 = JSON.stringify(L10.check('A000', 5));
  if (r1 !== r2) return { pass: false, detail: 'check 两次不一致' };
  return { pass: true, detail: 'stats/check 重复逐位一致' };
});

/* 7. L9 数值基础层 */
check('L9 数值基础层', () => {
  if (!NS.L9) return { pass: false, detail: 'NS.L9 缺失' };
  const need = ['mulberry32', 'randn', 'rngSplit', 'twoSum', 'kahanStep', 'ulpDiff', 'frexp', 'safeExp', 'logsumexpPair', 'assert', 'assertBitExact', 'sentinel', 'budget', 'runSelfTests'];
  for (const k of need) if (typeof NS.L9[k] !== 'function') return { pass: false, detail: 'L9 缺函数: ' + k };
  const st = NS.L9.runSelfTests(1);
  const bad = st.filter((r) => !r.pass);
  if (bad.length) return { pass: false, detail: 'L9 自检失败: ' + bad.map((b) => b.name).join(',') };
  return { pass: true, detail: `${need.length} 个符号就位 + ${st.length} 项自检全绿` };
});

/* 8. 引擎零影响（R3） */
check('引擎零影响（R3 兼容）', () => {
  const legacy = require('../src/flagship-engine.js');
  if (F !== legacy) return { pass: false, detail: '公共入口导出被包装（非引擎本体）' };
  const kIdx = Object.keys(F).sort().join(','), kLeg = Object.keys(legacy).sort().join(',');
  if (kIdx !== kLeg) return { pass: false, detail: '导出清单与 v11 不一致' };
  if (F === NS) return { pass: false, detail: '导出不应是命名空间本体' };
  const m = new F.GPTModel({ vocabSize: 20, blockSize: 8, nLayer: 1, nHead: 2, dModel: 16, seed: 3 });
  const lg = m.forward([1, 2, 3, 4, 5]).logits;
  if (!lg.every(Number.isFinite)) return { pass: false, detail: '前向出现非有限值' };
  return { pass: true, detail: `导出 ≡ v11（${Object.keys(F).length} 个 API）+ 前向有限` };
});

/* 汇总 */
const passCount = results.filter((r) => r.pass).length;
console.log('─'.repeat(56));
const s = L10.stats();
console.log(`L10 微指令探针：${passCount}/${results.length} 通过 · 已实现 ${s.done}/${s.total} ${passCount === results.length ? '✅ 全绿' : '❌ 存在失败'}`);
process.exitCode = passCount === results.length ? 0 : 1;

#!/usr/bin/env node
/* 旗舰 AI · L10 微指令 CLI（PRD P0-4）
 * 用法：
 *   node scripts/isa.js --stats        八域进度总览（done/total + 进度条）
 *   node scripts/isa.js --check        跑全部已实现微指令 selfTest（exit 0/1）
 *   node scripts/isa.js --check=A      只跑某域（A–H）
 *   node scripts/isa.js --run=A000     直接执行某项（调试用）
 */
'use strict';
require('../src/index.js'); // 按 __loadOrder 加载十层（副作用注入 globalThis.FlagshipAI）
const L10 = globalThis.FlagshipAI.L10;

const args = process.argv.slice(2);
const mode = (args.find((a) => a.startsWith('--')) || '--stats').slice(2);
const arg = args.find((a) => !a.startsWith('--'));

if (mode === 'stats') {
  const s = L10.stats();
  const bar = (d, t) => {
    const w = 20, f = Math.round((d / t) * w);
    return '█'.repeat(f) + '░'.repeat(w - f);
  };
  console.log('L10 · 微指令层（Micro-ISA）八域进度');
  console.log('─'.repeat(56));
  for (const k of Object.keys(s.byDomain)) {
    const d = s.byDomain[k];
    console.log(`  ${k}  ${d.name.padEnd(12, '　')} ${String(d.done).padStart(3)}/${d.total}  ${bar(d.done, d.total)}`);
  }
  console.log('─'.repeat(56));
  console.log(`  合计 ${s.done}/${s.total}（${((s.done / s.total) * 100).toFixed(1)}%）· 未实现项显式标记 registered-not-implemented`);
  process.exitCode = 0;
} else if (mode === 'check') {
  const seed = 1;
  let pass = 0, fail = 0;
  const failed = [];
  for (const key of L10.keys()) {
    const op = L10.get(key);
    if (op.impl !== 'done') continue;
    if (arg && op.domain !== arg) continue;
    const r = L10.check(key, seed);
    if (r.pass) { pass++; console.log(`  ✅ μ${key} ${op.name.padEnd(22)} ${r.detail}`); }
    else { fail++; failed.push(key); console.log(`  ❌ μ${key} ${op.name.padEnd(22)} ${r.detail}`); }
  }
  console.log('─'.repeat(56));
  console.log(`微指令自检：${pass}/${pass + fail} 通过${fail === 0 ? ' ✅ 全绿' : ' ❌ 失败: ' + failed.join(',')}`);
  process.exitCode = fail === 0 ? 0 : 1;
} else if (mode === 'run') {
  if (!arg) { console.error('--run 需要 key，如 --run=A000'); process.exitCode = 1; }
  else {
    try { console.log(JSON.stringify(L10.run(arg, args.slice(2).map((a) => +a)))); }
    catch (e) { console.error('执行失败: ' + e.message); process.exitCode = 1; }
  }
} else {
  console.error(`未知模式: --${mode}（可用：--stats / --check[=域] / --run=KEY）`);
  process.exitCode = 1;
}

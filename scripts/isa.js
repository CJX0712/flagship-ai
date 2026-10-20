#!/usr/bin/env node
'use strict';
/* 旗舰 AI v12 · 微指令登记中心覆盖率（isa = Instruction Set Archive）
 * 经生产加载路径 src/index.js 加载十层，再读取 globalThis.FlagshipAI.L10.stats()。
 *
 * 验收标准 ④：输出形如
 *   A 96/128 · C 96/128 · 合计 192/1024
 * 并对 A/C 已落实项数与总项数做断言（Phase 1 期望值），不符则 exit 1。
 *
 * 用法：node scripts/isa.js            # 打印覆盖率
 *       node scripts/isa.js --stats    # 同上（兼容文档写法）
 */
const EXPECT = { A: 96, C: 96, total: 192, capacity: 1024 };

require('../src/index.js'); // 生产加载路径：填充 globalThis.FlagshipAI
const NS = globalThis.FlagshipAI;
const s = NS.L10.stats();

const line = `A ${s.A.impl}/${s.A.capacity} · C ${s.C.impl}/${s.C.capacity} · 合计 ${s.total.impl}/${s.total.capacity}`;
console.log(line);

// 分域明细（便于核查其余 6 域 stub 进度）
const order = NS.L10.DOMAINS;
for (const d of order) {
  if (d === 'A' || d === 'C') continue;
  console.log(`  ${d} ${s[d].impl}/${s[d].capacity}`);
}

let ok = true;
if (s.A.impl !== EXPECT.A) { ok = false; console.error(`❌ A 域已落实 ${s.A.impl}，期望 ${EXPECT.A}`); }
if (s.C.impl !== EXPECT.C) { ok = false; console.error(`❌ C 域已落实 ${s.C.impl}，期望 ${EXPECT.C}`); }
if (s.total.impl !== EXPECT.total) { ok = false; console.error(`❌ 合计 ${s.total.impl}，期望 ${EXPECT.total}`); }
if (s.total.capacity !== EXPECT.capacity) { ok = false; console.error(`❌ 槽位总容量 ${s.total.capacity}，期望 ${EXPECT.capacity}`); }

if (ok) {
  console.log('✅ isa 覆盖率符合 Phase 1 (T02) 验收预期');
  process.exitCode = 0;
} else {
  process.exitCode = 1;
}

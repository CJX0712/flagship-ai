#!/usr/bin/env node
'use strict';
/* 旗舰 AI v12 · 微指令登记中心覆盖率（isa = Instruction Set Archive）
 * 经生产加载路径 src/index.js 加载十层，再读取 globalThis.FlagshipAI.L10.stats()。
 *
 * 验收标准 ④：输出形如
 *   A 96/128 · B 96/128 · C 96/128 · D 96/128 · E 96/128 · 合计 480/1024
 * 断言策略（T05 收官）：
 *   - A/B/C/D/E 五域为固定契约，各须恰为 96，不符 exit 1（防误删 / 漏实现）；
 *   - 合计固定 480，槽位总容量恒 1024；
 *   - 落地期间曾对 E 域用「浮动总量」以容忍并行进度，E 完工后已收紧为定值。
 *
 * 用法：node scripts/isa.js            # 打印覆盖率
 *       node scripts/isa.js --stats    # 同上（兼容文档写法）
 */
const EXPECT = { A: 96, B: 96, C: 96, D: 96, E: 96, total: 480, capacity: 1024 };
const FIXED_DOMAINS = ['A', 'B', 'C', 'D', 'E'];

require('../src/index.js'); // 生产加载路径：填充 globalThis.FlagshipAI
const NS = globalThis.FlagshipAI;
const s = NS.L10.stats();

const line = FIXED_DOMAINS.map(d => `${d} ${s[d].impl}/${s[d].capacity}`).join(' · ') +
  ` · 合计 ${s.total.impl}/${s.total.capacity}`;
console.log(line);

// 分域明细（便于核查其余 4 域 stub 进度）
for (const d of NS.L10.DOMAINS) {
  if (FIXED_DOMAINS.includes(d)) continue;
  console.log(`  ${d} ${s[d].impl}/${s[d].capacity}`);
}

let ok = true;
for (const d of FIXED_DOMAINS) {
  if (s[d].impl !== EXPECT[d]) { ok = false; console.error(`❌ ${d} 域已落实 ${s[d].impl}，期望 ${EXPECT[d]}`); }
}
if (s.total.impl !== EXPECT.total) { ok = false; console.error(`❌ 合计 ${s.total.impl}，期望 ${EXPECT.total}`); }
if (s.total.capacity !== EXPECT.capacity) { ok = false; console.error(`❌ 槽位总容量 ${s.total.capacity}，期望 ${EXPECT.capacity}`); }

if (ok) {
  console.log(`✅ isa 覆盖率符合 T02（A/C 域）+ T04（B/E 域）+ T05（D 域）验收预期：${EXPECT.total}/1024`);
  process.exitCode = 0;
} else {
  process.exitCode = 1;
}

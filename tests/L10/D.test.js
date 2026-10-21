'use strict';
/* L10 域 D（注意力基元）微指令不变量性质测试
 * 验收 ⑤：对 D 域已落实项各跑 ≥100 用例（此处 120）全绿。
 * 与 scripts/probe-microisa.js 互补：此处由 fast-check 驱动种子做随机模糊测试，
 * 探针负责装配 / 覆盖率诚实性 / 引擎零影响的整体把关。
 */
const { test } = require('node:test');
const assert = require('node:assert');
const { implOps, checkOp } = require('../lib/op-harness.js');

const DOMAIN = 'D';
const RUNS = 120;
const ops = implOps(DOMAIN);

test(`L10 域 ${DOMAIN} 微指令不变量（fast-check ${RUNS} 例/项，共 ${ops.length} 项）`, () => {
  assert.ok(ops.length > 0, `${DOMAIN} 域应有已落实微指令`);
  const failed = [];
  for (const op of ops) {
    const r = checkOp(op, RUNS);
    if (!r.pass) failed.push(`${op.key}(${op.name || ''}): ${r.reason || 'predicate 失败'}`);
  }
  assert.strictEqual(failed.length, 0, `失败项(${failed.length}):\n` + failed.join('\n'));
});

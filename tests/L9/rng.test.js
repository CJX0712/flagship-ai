'use strict';
/* L9 数值基础 · 确定性伪随机（rng）
 * 验收 ①：tests/L9 全绿；验收 ② R9：mulberry32/randn 与 legacy 逐位一致。
 */
const { test } = require('node:test');
const assert = require('node:assert');
require('../../src/index.js');
const NS = globalThis.FlagshipAI;
const V11 = require('../../src/legacy/flagship-engine.v11.js');

const rng = NS.L9.rng;

test('mulberry32 · 同种子完全可复现', () => {
  const a = rng.mulberry32(42), b = rng.mulberry32(42);
  for (let i = 0; i < 100; i++) assert.ok(Object.is(a(), b()), `第 ${i} 个样本不一致`);
});

test('R9 逐算子 · mulberry32 与 legacy 逐位一致', () => {
  const a = rng.mulberry32(123), b = V11.mulberry32(123);
  for (let i = 0; i < 100; i++) assert.ok(Object.is(a(), b()), `第 ${i} 个样本 Δ≠0`);
});

test('mulberry32 · 输出落在 [0,1)', () => {
  const a = rng.mulberry32(7);
  for (let i = 0; i < 1000; i++) { const v = a(); assert.ok(v >= 0 && v < 1, `越界: ${v}`); }
});

test('randn · 有限且同种子可复现', () => {
  const r1 = rng.mulberry32(2024), r2 = rng.mulberry32(2024);
  for (let i = 0; i < 200; i++) {
    const x = rng.randn(r1), y = rng.randn(r2);
    assert.ok(Number.isFinite(x), 'randn 非有限');
    assert.ok(Object.is(x, y), `randn 第 ${i} 个样本不一致`);
  }
});

test('randn · 抽样均值接近 0（大样本统计）', () => {
  const r = rng.mulberry32(555);
  let sum = 0; const N = 20000;
  for (let i = 0; i < N; i++) sum += rng.randn(r);
  const mean = sum / N;
  assert.ok(Math.abs(mean) < 0.05, `均值偏离过大: ${mean}`);
});

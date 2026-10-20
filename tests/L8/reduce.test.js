'use strict';
/* L8 数值内核 · 归约算子（softmaxRows / rowSum / topkRows）
 * 验收 ①：tests/L8 全绿。
 */
const { test } = require('node:test');
const assert = require('node:assert');
require('../../src/index.js');
const NS = globalThis.FlagshipAI;
const R = NS.L8.reduce;

function bitExact(a, b) {
  if (!a || !b || a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (!Object.is(a[i], b[i])) return false;
  return true;
}
function rndMatrix(rng, r, c) {
  const a = new Float64Array(r * c);
  for (let i = 0; i < a.length; i++) a[i] = rng() * 2 - 1;
  return a;
}
const rng = NS.L9.rng.mulberry32(99);

test('softmaxRows · 每行和为 1（容差 1e-12）', () => {
  const S = rndMatrix(rng, 5, 7);
  const P = R.softmaxRows(S, 5, 7);
  for (let i = 0; i < 5; i++) {
    let s = 0; for (let j = 0; j < 7; j++) s += P[i * 7 + j];
    assert.ok(Math.abs(s - 1) < 1e-12, `第 ${i} 行和=${s}`);
  }
});

test('softmaxRows · 数值稳定（大偏移不改变结果）', () => {
  const S = rndMatrix(rng, 3, 4);
  const big = new Float64Array(S.length);
  for (let i = 0; i < S.length; i++) big[i] = S[i] + 1e6; // 整体平移
  const P1 = R.softmaxRows(S, 3, 4);
  const P2 = R.softmaxRows(big, 3, 4);
  assert.ok(bitExact(P1, P2), 'softmaxRows 对整体平移应不变（数值稳定）');
});

test('softmaxRows · 与朴素实现一致', () => {
  const S = rndMatrix(rng, 4, 6);
  const naive = new Float64Array(24);
  for (let i = 0; i < 4; i++) {
    let mx = -Infinity; for (let j = 0; j < 6; j++) mx = Math.max(mx, S[i * 6 + j]);
    let s = 0; for (let j = 0; j < 6; j++) { const e = Math.exp(S[i * 6 + j] - mx); naive[i * 6 + j] = e; s += e; }
    for (let j = 0; j < 6; j++) naive[i * 6 + j] *= (1 / s);
  }
  assert.ok(bitExact(R.softmaxRows(S, 4, 6), naive), 'softmaxRows 与朴素不一致');
});

test('rowSum · 按列求和正确', () => {
  const Y = new Float64Array([1, 2, 3, 4, 5, 6]); // 2×3
  const s = R.rowSum(Y, 2, 3);
  assert.deepStrictEqual(Array.from(s), [5, 7, 9]); // 列和
});

test('topkRows · 返回每行最大 k 个的降序下标', () => {
  const P = new Float64Array([0.1, 0.6, 0.3, 0.2, 0.2, 0.2, 0.9, 0.05]); // 2×4
  const { idx } = R.topkRows(P, 2, 4, 2);
  // 第 0 行: [0.1,0.6,0.3,0.2] 最大两个是 0.6(索引1),0.3(索引2) -> idx [1,2]（按值降序）
  assert.deepStrictEqual(Array.from(idx.slice(0, 2)), [1, 2]);
  // 第 1 行: [0.2,0.2,0.9,0.05] 最大两个是 0.9(索引2),0.2(索引1,稳定排序取并列末位) -> idx [2,1]
  assert.deepStrictEqual(Array.from(idx.slice(2, 4)), [2, 1]);
});

test('softmaxBwd · ŷ ⊙ (dy − Σ dy·ŷ) 形状正确且有限', () => {
  const S = rndMatrix(rng, 3, 4);
  const P = R.softmaxRows(S, 3, 4);
  const dP = rndMatrix(rng, 3, 4);
  const dS = R.softmaxBwd(dP, P, 3, 4);
  assert.strictEqual(dS.length, 12);
  for (let i = 0; i < 12; i++) assert.ok(Number.isFinite(dS[i]), `dS[${i}] 非有限`);
});

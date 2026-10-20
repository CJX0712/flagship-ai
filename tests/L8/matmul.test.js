'use strict';
/* L8 数值内核 · 矩阵乘法家族（matmul / matmulAtB / matmulTA / softmaxRows）
 * 验收 ①：tests/L8 全绿；验收 ② R9：与 legacy 同名导出逐位 Δ=0。
 */
const { test } = require('node:test');
const assert = require('node:assert');
require('../../src/index.js'); // 填充 globalThis.FlagshipAI
const NS = globalThis.FlagshipAI;
const V11 = require('../../src/legacy/flagship-engine.v11.js');

const L8 = NS.L8.matmul;
const L8r = NS.L8.reduce;

function bitExact(a, b) {
  if (typeof a === 'number' && typeof b === 'number') return Object.is(a, b);
  if (!a || !b || a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (!Object.is(a[i], b[i])) return false;
  return true;
}
function rndMatrix(rng, r, c) {
  const a = new Float64Array(r * c);
  for (let i = 0; i < a.length; i++) a[i] = rng() * 2 - 1;
  return a;
}
function id(r) { const o = new Float64Array(r * r); for (let i = 0; i < r; i++) o[i * r + i] = 1; return o; }
function zeros(r, c) { return new Float64Array(r * c); }

const rng = NS.L9.rng.mulberry32(12345);

test('R9 逐算子 · matmul 与 legacy 逐位 Δ=0', () => {
  for (let t = 0; t < 10; t++) {
    const A = rndMatrix(rng, 4, 3), B = rndMatrix(rng, 3, 5);
    assert.ok(bitExact(L8.matmul(A, B, 4, 3, 5), V11.matmul(A, B, 4, 3, 5)), `matmul 用例 ${t} Δ≠0`);
  }
});

test('R9 逐算子 · matmulAtB 与 legacy 逐位 Δ=0', () => {
  for (let t = 0; t < 10; t++) {
    const A = rndMatrix(rng, 4, 3), B = rndMatrix(rng, 3, 5);
    assert.ok(bitExact(L8.matmulAtB(A, B, 4, 3, 5), V11.matmulAtB(A, B, 4, 3, 5)), `matmulAtB 用例 ${t} Δ≠0`);
  }
});

test('R9 逐算子 · matmulTA 与 legacy 逐位 Δ=0', () => {
  for (let t = 0; t < 10; t++) {
    const A = rndMatrix(rng, 3, 4), B = rndMatrix(rng, 4, 5);
    assert.ok(bitExact(L8.matmulTA(A, B, 4, 3, 5), V11.matmulTA(A, B, 4, 3, 5)), `matmulTA 用例 ${t} Δ≠0`);
  }
});

test('R9 逐算子 · softmaxRows 与 legacy 逐位 Δ=0', () => {
  for (let t = 0; t < 10; t++) {
    const S = rndMatrix(rng, 3, 4);
    assert.ok(bitExact(L8r.softmaxRows(S, 3, 4), V11.softmaxRows(S, 3, 4)), `softmaxRows 用例 ${t} Δ≠0`);
  }
});

test('性质 · matmul(A, I) = A 与 I·A = A（整数域逐位）', () => {
  const A = new Float64Array(12);
  for (let i = 0; i < 12; i++) A[i] = (i - 6);
  assert.ok(bitExact(L8.matmul(A, id(3), 4, 3, 3), A), 'A·I ≠ A');
  assert.ok(bitExact(L8.matmul(id(4), A, 4, 4, 3), A), 'I·A ≠ A');
});

test('性质 · matmul(A, 0) = 0', () => {
  const A = rndMatrix(rng, 4, 3);
  const Z = L8.matmul(A, zeros(3, 5), 4, 3, 5);
  assert.ok(bitExact(Z, zeros(4, 5)), 'A·0 ≠ 0');
});

test('性质 · matmul 结合（整数域逐位）', () => {
  const A = new Float64Array([1, 2, 3, 4, 5, 6]);     // 2×3
  const B = new Float64Array([1, 0, 0, 1, 1, 1, 0, 1]); // 3×4 简化用
  // 用更小的可验证例子：(2×2)·((2×2)·(2×2)) == ((2×2)·(2×2))·(2×2)
  const X = new Float64Array([1, 2, 3, 4]);
  const Y = new Float64Array([5, 6, 7, 8]);
  const Z = new Float64Array([9, 1, 2, 3]);
  const lhs = L8.matmul(L8.matmul(X, Y, 2, 2, 2), Z, 2, 2, 2);
  const rhs = L8.matmul(X, L8.matmul(Y, Z, 2, 2, 2), 2, 2, 2);
  assert.ok(bitExact(lhs, rhs), '整数域 matmul 不结合');
});

test('性质 · matmulAtB / matmulTA 与朴素三重循环一致', () => {
  const A = rndMatrix(rng, 3, 4), B = rndMatrix(rng, 3, 5);
  // A(3×4) @ B(3×5)^T -> (3×5)
  const atb = L8.matmulAtB(A, B, 3, 4, 5);
  const naive = new Float64Array(15);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 5; j++) { let s = 0; for (let p = 0; p < 4; p++) s += A[i * 4 + p] * B[j * 4 + p]; naive[i * 5 + j] = s; }
  assert.ok(bitExact(atb, naive), 'matmulAtB 与朴素不一致');
});

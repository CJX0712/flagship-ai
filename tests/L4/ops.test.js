'use strict';
/* L4 神经算子 · 逐算子 Δ=0 比对（T03 验收 ①）
 * 对每个 L4 算子，用 legacy 导出的同名函数在确定性随机输入下 Object.is 全等。
 * 覆盖：layernorm / rmsnorm / gelu / silu / swiglu / rope / softmax / crossEntropy / matmul 族。
 * （qkNorm / topk / moe 未由 legacy 直接导出，其逐位等价由模型级 R2/R3 + tests/L3 覆盖：
 *   gemma 预设走 qkNorm、mixtral 预设走 moe+topk，前向/反向均 Δ=0。）
 */
const { test } = require('node:test');
const assert = require('node:assert');
require('../../src/index.js');
const NS = globalThis.FlagshipAI;
const V11 = require('../../src/legacy/flagship-engine.v11.js');
const L4 = NS.L4, L8 = NS.L8;

function bitExact(a, b) {
  if (!a || !b || a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (!Object.is(a[i], b[i])) return false;
  return true;
}
const rng = NS.L9.rng.mulberry32(2024 >>> 0);
function rnd(n) { const a = new Float64Array(n); for (let i = 0; i < n; i++) a[i] = rng() * 2 - 1; return a; }

test('layernormFwd Δ=0', () => {
  const S = rnd(12), g = rnd(4), b = rnd(4);
  const y1 = V11.layernormFwd(S, g, b, 3, 4), y2 = L4.norm.layernormFwd(S, g, b, 3, 4);
  assert.ok(bitExact(y1.Y, y2.Y) && bitExact(y1.xhat, y2.xhat) && bitExact(y1.rstd, y2.rstd));
});
test('layernormBwd Δ=0', () => {
  const S = rnd(12), g = rnd(4), b = rnd(4), dY = rnd(12);
  const f = V11.layernormFwd(S, g, b, 3, 4);
  const r1 = V11.layernormBwd(dY, f.xhat, f.rstd, g, 3, 4), r2 = L4.norm.layernormBwd(dY, f.xhat, f.rstd, g, 3, 4);
  assert.ok(bitExact(r1.dX, r2.dX) && bitExact(r1.dg, r2.dg) && bitExact(r1.db, r2.db));
});
test('rmsnormFwd Δ=0', () => {
  const S = rnd(12), g = rnd(4), b = rnd(4);
  const y1 = V11.rmsnormFwd(S, g, b, 3, 4), y2 = L4.norm.rmsnormFwd(S, g, b, 3, 4);
  assert.ok(bitExact(y1.Y, y2.Y) && bitExact(y1.xhat, y2.xhat) && bitExact(y1.rstd, y2.rstd));
});
test('rmsnormBwd Δ=0', () => {
  const S = rnd(12), g = rnd(4), b = rnd(4), dY = rnd(12);
  const f = V11.rmsnormFwd(S, g, b, 3, 4);
  const r1 = V11.rmsnormBwd(dY, f.xhat, f.rstd, g, 3, 4), r2 = L4.norm.rmsnormBwd(dY, f.xhat, f.rstd, g, 3, 4);
  assert.ok(bitExact(r1.dX, r2.dX) && bitExact(r1.dg, r2.dg) && r1.db === r2.db);
});
test('geluFwd / geluBwd Δ=0', () => {
  const X = rnd(20), dY = rnd(20);
  assert.ok(bitExact(V11.geluFwd(X), L4.activation.geluFwd(X)));
  assert.ok(bitExact(V11.geluBwd(dY, X), L4.activation.geluBwd(dY, X)));
});
test('siluF Δ=0（逐元素标量）', () => {
  const X = rnd(32);
  for (let i = 0; i < X.length; i++) assert.ok(Object.is(V11.siluF(X[i]), L4.activation.siluF(X[i])));
});
test('swigluFwd / swigluBwd Δ=0', () => {
  const pre = rnd(2 * 6 * 4), dout = rnd(6 * 4); // r=6, h=4
  assert.ok(bitExact(V11.swigluFwd(pre, 6, 4), L4.activation.swigluFwd(pre, 6, 4)));
  assert.ok(bitExact(V11.swigluBwd(dout, pre, 6, 4), L4.activation.swigluBwd(dout, pre, 6, 4)));
});
test('ropeApply Δ=0（就地旋转）', () => {
  const D = 8, H = 2, dh = 4;
  const X = rnd(4 * 3 * D);
  const a = Float64Array.from(X), b = Float64Array.from(X);
  V11.ropeApply(a, 4, D, H, dh, 0, 1, 2, 3 * D, null);
  L4.positional.ropeApply(b, 4, D, H, dh, 0, 1, 2, 3 * D, null);
  assert.ok(bitExact(a, b));
});
test('softmaxRows / softmaxBwd Δ=0', () => {
  const S = rnd(15), dP = rnd(15);
  const P = V11.softmaxRows(S, 3, 5);
  assert.ok(bitExact(P, L4.softmax.softmaxRows(S, 3, 5)));
  assert.ok(bitExact(V11.softmaxBwd(dP, P, 3, 5), L4.softmax.softmaxBwd(dP, P, 3, 5)));
});
test('crossEntropy Δ=0', () => {
  const logits = rnd(4 * 5), targets = [0, 1, 2, 3];
  const c1 = V11.crossEntropy(logits, targets, 4, 5, 4), c2 = L4.loss.crossEntropy(logits, targets, 4, 5, 4);
  assert.ok(Object.is(c1.loss, c2.loss) && bitExact(c1.dlogits, c2.dlogits));
});
test('matmul / matmulAtB / matmulTA Δ=0', () => {
  const A = rnd(12), B = rnd(15);
  assert.ok(bitExact(V11.matmul(A, B, 4, 3, 5), L8.matmul.matmul(A, B, 4, 3, 5)));
  assert.ok(bitExact(V11.matmulAtB(A, B, 4, 3, 5), L8.matmul.matmulAtB(A, B, 4, 3, 5)));
  assert.ok(bitExact(V11.matmulTA(A, B, 4, 3, 5), L8.matmul.matmulTA(A, B, 4, 3, 5)));
});

'use strict';
/* L9 数值基础 · 特殊函数（special）
 * 验收 ①：tests/L9 全绿。
 */
const { test } = require('node:test');
const assert = require('node:assert');
require('../../src/index.js');
const NS = globalThis.FlagshipAI;
const S = NS.L9.special;

function relClose(a, b, tol) { tol = tol || 1e-12; return Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b)); }

test('sigmoid · 对称恒等式 σ(-x)=1-σ(x) 与 σ(0)=0.5', () => {
  assert.ok(relClose(S.sigmoid(0), 0.5, 1e-15), 'σ(0)≠0.5');
  for (const x of [-3, -1, -0.2, 0.2, 1, 3]) {
    assert.ok(relClose(S.sigmoid(-x), 1 - S.sigmoid(x), 1e-15), `σ 对称失败 @${x}`);
    assert.ok(S.sigmoid(x) > 0 && S.sigmoid(x) < 1, `σ 越界 @${x}`);
  }
});

test('softplus · 非负且大正输入趋近 x', () => {
  assert.ok(S.softplus(-5) > 0, 'softplus 负数非正');
  assert.ok(relClose(S.softplus(20), 20, 1e-9), 'softplus(20) 不趋近 20');
});

test('relu · relu(0)=0 且 x>0 恒等', () => {
  assert.strictEqual(S.relu(0), 0);
  assert.strictEqual(S.relu(4.5), 4.5);
  assert.strictEqual(S.relu(-2), 0);
});

test('elu / selu · 有限且 x>=0 退化为 x', () => {
  assert.strictEqual(S.elu(0), 0);
  assert.strictEqual(S.elu(3), 3);
  assert.strictEqual(S.selu(0), 0);
  assert.ok(Number.isFinite(S.elu(-3)));
  assert.ok(Number.isFinite(S.selu(-3)));
});

test('gelu · 有限、gelu(0)=0、单调增（x≥0 抽样）', () => {
  // 注：gelu(x)=x·Φ(x)（及 tanh 近似）在 x≈-0.7 处存在浅极小值，整体非全局单调；
  // 在 x≥0 区域 g'(x)=Φ(x)+x·φ(x)>0，严格单调增。此处验证该可靠区间。
  assert.ok(relClose(S.gelu(0), 0, 1e-15), 'gelu(0)≠0');
  let prev = -Infinity;
  for (let x = 0; x <= 5; x += 0.1) { const g = S.gelu(x); assert.ok(g >= prev - 1e-9, 'gelu 非单调'); prev = g; }
});

test('swish · swish(0)=0 且有限', () => {
  assert.ok(relClose(S.swish(0), 0, 1e-15), 'swish(0)≠0');
  assert.ok(Number.isFinite(S.swish(-4)));
});

test('logsumexp2 · 不小于 max(a,b) 且与朴素一致', () => {
  for (const [a, b] of [[1, 2], [-3, -1], [100, 0], [-50, -50]]) {
    const r = S.logsumexp2(a, b);
    assert.ok(r >= Math.max(a, b) - 1e-12, `logsumexp2 < max @${a},${b}`);
    const naive = Math.log(Math.exp(a) + Math.exp(b));
    assert.ok(relClose(r, naive, 1e-12), `logsumexp2 与朴素不一致 @${a},${b}`);
  }
});

test('erf · 奇函数、erf(0)=0、|erf|<1', () => {
  assert.ok(relClose(S.erf(0), 0, 1e-7), 'erf(0)≠0（A&S 近似误差 ~1e-9）');
  for (const x of [-2, -0.5, 0.5, 2]) assert.ok(relClose(S.erf(-x), -S.erf(x), 1e-9), `erf 非奇 @${x}`);
  assert.ok(Math.abs(S.erf(3)) < 1, '|erf|>1');
});

test('softmaxStable · 非负且和=1', () => {
  const v = new Float64Array([2, -1, 0.5, 7, -3]);
  const p = S.softmaxStable(v);
  let s = 0; for (let i = 0; i < p.length; i++) { assert.ok(p[i] >= 0, '负概率'); s += p[i]; }
  assert.ok(relClose(s, 1, 1e-12), `和=${s}`);
});

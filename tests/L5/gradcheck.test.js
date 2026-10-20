'use strict';
/* L5 自动微分 · 中心差分梯度检验（gradcheck）
 * 验收 ⑥：解析 VJP vs 有限差分，err_i = |ana−num| / max(1,|num|)，
 * maxErr ≤ tol（4.22e-4）。每个用例构造标量损失 loss = Σ fwd(x)·w
 * （w 为固定上游梯度），解析梯度取对应 L4 backward 的 VJP 约定形状。
 * 输入一律经 NS.L9.rng.mulberry32 确定性生成（K6：注入 RNG 为唯一熵源）。
 */
const { test } = require('node:test');
const assert = require('node:assert');
require('../../src/index.js');
const NS = globalThis.FlagshipAI;
const GC = NS.L5.gradcheck;
const TOL = GC.DEFAULT_TOL; // 4.22e-4

function randArr(rng, n, lo, hi) {
  const a = new Float64Array(n);
  for (let i = 0; i < n; i++) a[i] = lo + rng() * (hi - lo);
  return a;
}
function dotSum(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
function assertGradOk(r) {
  assert.strictEqual(r.pass, true, `${r.name} 未通过（maxErr=${r.maxErr}）`);
  assert.ok(r.maxErr <= TOL, `${r.name} maxErr=${r.maxErr} 超过 tol=${TOL}`);
}

/* ---------- 各算子用例（确定性种子 => 可复现） ---------- */

function opGelu() {
  const rng = NS.L9.rng.mulberry32(1234);
  const n = 24;
  const x = randArr(rng, n, -1, 1), w = randArr(rng, n, -1, 1);
  const { geluFwd, geluBwd } = NS.L4.activation;
  const lossFn = (xx) => dotSum(geluFwd(xx), w);
  const anaGradFn = (xx) => geluBwd(w, xx);
  return GC.checkNumeric('gelu', lossFn, anaGradFn, x);
}

function opSilu() {
  const rng = NS.L9.rng.mulberry32(4321);
  const n = 20;
  const x = randArr(rng, n, -1.5, 1.5), w = randArr(rng, n, -1, 1);
  const siluF = NS.L4.activation.siluF;
  const lossFn = (xx) => { let s = 0; for (let i = 0; i < n; i++) s += siluF(xx[i]) * w[i]; return s; };
  // d/dx [x·σ(x)] = σ(x) + x·σ(x)·(1−σ(x))；注意 siluF(x)=x·σ(x) 并非 σ(x)，需另算
  const anaGradFn = (xx) => {
    const d = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const sg = 1 / (1 + Math.exp(-xx[i]));
      d[i] = w[i] * (sg + xx[i] * sg * (1 - sg));
    }
    return d;
  };
  return GC.checkNumeric('silu', lossFn, anaGradFn, x);
}

function opSwiglu() {
  const rng = NS.L9.rng.mulberry32(8888);
  const r = 3, h = 4;
  const pre = randArr(rng, r * 2 * h, -1, 1);
  const w = randArr(rng, r * h, -1, 1);
  const { swigluFwd, swigluBwd } = NS.L4.activation;
  const lossFn = (p) => dotSum(swigluFwd(p, r, h), w);
  const anaGradFn = (p) => swigluBwd(w, p, r, h);
  return GC.checkNumeric('swiglu', lossFn, anaGradFn, pre);
}

function opLayernorm() { // dX
  const rng = NS.L9.rng.mulberry32(777);
  const r = 3, c = 5;
  const X = randArr(rng, r * c, -1, 1), g = randArr(rng, c, 0.5, 1.5), b = randArr(rng, c, -0.5, 0.5);
  const w = randArr(rng, r * c, -1, 1);
  const { layernormFwd, layernormBwd } = NS.L4.norm;
  const lossFn = (xx) => dotSum(layernormFwd(xx, g, b, r, c).Y, w);
  const anaGradFn = (xx) => {
    const f = layernormFwd(xx, g, b, r, c);           // 复用前向的 xhat/rstd
    return layernormBwd(w, f.xhat, f.rstd, g, r, c).dX;
  };
  return GC.checkNumeric('layernorm.dX', lossFn, anaGradFn, X);
}

function opLayernormG() { // dg（xhat/rstd 与 g 无关，前向只跑一次）
  const rng = NS.L9.rng.mulberry32(778);
  const r = 3, c = 5;
  const X = randArr(rng, r * c, -1, 1), g0 = randArr(rng, c, 0.5, 1.5), b = randArr(rng, c, -0.5, 0.5);
  const w = randArr(rng, r * c, -1, 1);
  const { layernormFwd, layernormBwd } = NS.L4.norm;
  const f = layernormFwd(X, g0, b, r, c);
  const lossFn = (gg) => dotSum(layernormFwd(X, gg, b, r, c).Y, w);
  const anaGradFn = (gg) => layernormBwd(w, f.xhat, f.rstd, gg, r, c).dg;
  return GC.checkNumeric('layernorm.dg', lossFn, anaGradFn, g0);
}

function opLayernormB() { // db
  const rng = NS.L9.rng.mulberry32(779);
  const r = 3, c = 5;
  const X = randArr(rng, r * c, -1, 1), g = randArr(rng, c, 0.5, 1.5), b0 = randArr(rng, c, -0.5, 0.5);
  const w = randArr(rng, r * c, -1, 1);
  const { layernormFwd, layernormBwd } = NS.L4.norm;
  const f = layernormFwd(X, g, b0, r, c);
  const lossFn = (bb) => dotSum(layernormFwd(X, g, bb, r, c).Y, w);
  const anaGradFn = () => layernormBwd(w, f.xhat, f.rstd, g, r, c).db;
  return GC.checkNumeric('layernorm.db', lossFn, anaGradFn, b0);
}

function opRmsnorm() { // dX（db 恒 null，只查 dX/dg）
  const rng = NS.L9.rng.mulberry32(555);
  const r = 3, c = 5;
  const X = randArr(rng, r * c, -1, 1), g = randArr(rng, c, 0.5, 1.5);
  const w = randArr(rng, r * c, -1, 1);
  const { rmsnormFwd, rmsnormBwd } = NS.L4.norm;
  const lossFn = (xx) => dotSum(rmsnormFwd(xx, g, null, r, c).Y, w);
  const anaGradFn = (xx) => {
    const f = rmsnormFwd(xx, g, null, r, c);
    const out = rmsnormBwd(w, f.xhat, f.rstd, g, r, c);
    assert.strictEqual(out.db, null, 'rmsnorm 无偏置：db 应为 null');
    return out.dX;
  };
  return GC.checkNumeric('rmsnorm.dX', lossFn, anaGradFn, X);
}

function opRmsnormG() { // dg
  const rng = NS.L9.rng.mulberry32(556);
  const r = 3, c = 5;
  const X = randArr(rng, r * c, -1, 1), g0 = randArr(rng, c, 0.5, 1.5);
  const w = randArr(rng, r * c, -1, 1);
  const { rmsnormFwd, rmsnormBwd } = NS.L4.norm;
  const f = rmsnormFwd(X, g0, null, r, c);
  const lossFn = (gg) => dotSum(rmsnormFwd(X, gg, null, r, c).Y, w);
  const anaGradFn = (gg) => rmsnormBwd(w, f.xhat, f.rstd, gg, r, c).dg;
  return GC.checkNumeric('rmsnorm.dg', lossFn, anaGradFn, g0);
}

function opMatmulA() { // dA = W @ B^T（matmul 精确线性 => 近机器精度）
  const rng = NS.L9.rng.mulberry32(2468);
  const r = 2, k = 3, c = 2;
  const A = randArr(rng, r * k, -1, 1), B = randArr(rng, k * c, -1, 1), W = randArr(rng, r * c, -1, 1);
  const { matmul, matmulAtB } = NS.L8.matmul;
  const lossFn = (aa) => dotSum(matmul(aa, B, r, k, c), W);
  const anaGradFn = () => matmulAtB(W, B, r, c, k); // W(r×c) @ B(k×c)^T -> r×k
  return GC.checkNumeric('matmul.dA', lossFn, anaGradFn, A);
}

function opMatmulB() { // dB = A^T @ W
  const rng = NS.L9.rng.mulberry32(2469);
  const r = 2, k = 3, c = 2;
  const A = randArr(rng, r * k, -1, 1), B = randArr(rng, k * c, -1, 1), W = randArr(rng, r * c, -1, 1);
  const { matmul, matmulTA } = NS.L8.matmul;
  const lossFn = (bb) => dotSum(matmul(A, bb, r, k, c), W);
  const anaGradFn = () => matmulTA(A, W, r, k, c); // A(r×k)^T @ W(r×c) -> k×c
  return GC.checkNumeric('matmul.dB', lossFn, anaGradFn, B);
}

function opSoftmax() {
  const rng = NS.L9.rng.mulberry32(31337);
  const r = 3, c = 4;
  const S = randArr(rng, r * c, -2, 2), w = randArr(rng, r * c, -1, 1);
  const { softmaxRows, softmaxBwd } = NS.L8.reduce;
  const P0 = softmaxRows(S, r, c);
  const lossFn = (ss) => dotSum(softmaxRows(ss, r, c), w);
  const anaGradFn = () => softmaxBwd(w, P0, r, c); // dS = P ⊙ (w − Σ w·P)
  return GC.checkNumeric('softmax', lossFn, anaGradFn, S);
}

function opAttention() { // 经 registry 取用的 attention VJP（集成路径）
  const rng = NS.L9.rng.mulberry32(2024);
  const T = 3, D = 4, H = 2, dh = 2, KV = 2, g = 1, qkvDim = D + 2 * KV * dh; // 12
  const scale = 1 / Math.sqrt(dh), attnScale = 1, win = 0;
  const qkv = randArr(rng, T * qkvDim, -0.8, 0.8), dctx = randArr(rng, T * D, -1, 1);
  const attnFwd = NS.L4.attention.attnFwd;
  const attnBwd = NS.L5.registry.get('attention');
  assert.strictEqual(typeof attnBwd, 'function', "registry 应已登记 'attention' VJP");
  const lossFn = (q) => {
    const { ctx } = attnFwd(q, T, D, H, dh, qkvDim, KV, g, win, scale, attnScale);
    return dotSum(ctx, dctx);
  };
  const anaGradFn = (q) => {
    const { Ps } = attnFwd(q, T, D, H, dh, qkvDim, KV, g, win, scale, attnScale);
    return attnBwd({ Ps, qkv: q }, dctx, T, D, H, dh, qkvDim, KV, g, scale, attnScale);
  };
  return GC.checkNumeric('attention', lossFn, anaGradFn, qkv);
}

/* ---------- 测试用例 ---------- */

test('centralDiff · f(x)=Σx² 与 Σx³ 差分导数精确', () => {
  const rng = NS.L9.rng.mulberry32(7);
  const x = randArr(rng, 8, -1, 1);
  const f2 = (v) => { let s = 0; for (let i = 0; i < v.length; i++) s += v[i] * v[i]; return s; };
  for (let i = 0; i < x.length; i++) {
    assert.ok(Math.abs(GC.centralDiff(f2, x, i) - 2 * x[i]) < 1e-6, `f2 idx=${i}`);
  }
  const f3 = (v) => { let s = 0; for (let i = 0; i < v.length; i++) s += v[i] * v[i] * v[i]; return s; };
  for (let i = 0; i < x.length; i++) {
    assert.ok(Math.abs(GC.centralDiff(f3, x, i) - 3 * x[i] * x[i]) < 1e-5, `f3 idx=${i}`);
  }
});

test('centralDiff · 默认步长 h=1e-6 且与 exp 解析导数一致', () => {
  const x = new Float64Array([0.3, -0.7, 1.1]);
  const f = (v) => Math.exp(v[0]) + v[1] * v[2];
  assert.strictEqual(GC.centralDiff(f, x, 0), GC.centralDiff(f, x, 0, 1e-6)); // 默认即 1e-6
  assert.ok(Math.abs(GC.centralDiff(f, x, 0) - Math.exp(0.3)) < 1e-9);
  assert.ok(Math.abs(GC.centralDiff(f, x, 1) - x[2]) < 1e-9);
  assert.ok(Math.abs(GC.centralDiff(f, x, 2) - x[1]) < 1e-9);
});

test('checkNumeric · 返回契约字段；解析梯度错误/缺元素时必须 FAIL', () => {
  const x = new Float64Array([0.5, -1.5, 2.0]);
  const lossFn = (v) => v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
  const ok = GC.checkNumeric('square', lossFn, (v) => new Float64Array([2 * v[0], 2 * v[1], 2 * v[2]]), x);
  assert.strictEqual(ok.name, 'square');
  assert.strictEqual(ok.pass, true);
  assert.strictEqual(ok.tol, TOL);
  assert.strictEqual(ok.n, 3);
  assert.ok(ok.maxErr <= TOL);
  const wrong = GC.checkNumeric('square-wrong', lossFn, () => new Float64Array([0, 0, 0]), x);
  assert.strictEqual(wrong.pass, false);
  assert.ok(wrong.maxErr > TOL);
  const short = GC.checkNumeric('square-short', lossFn, () => new Float64Array([1, 2]), x);
  assert.strictEqual(short.pass, false); // ana 缺元素 -> err=Infinity -> FAIL（安全方向）
});

test('gelu · anaGrad=geluBwd(w,x) 与数值差分一致', () => { assertGradOk(opGelu()); });

test('silu · 解析导数 σ+xσ(1−σ) 与数值差分一致', () => { assertGradOk(opSilu()); });

test('swiglu · anaGrad=swigluBwd(w,pre,r,h) 与数值差分一致', () => { assertGradOk(opSwiglu()); });

test('layernorm · dX / dg / db 三项与数值差分一致', () => {
  assertGradOk(opLayernorm());
  assertGradOk(opLayernormG());
  assertGradOk(opLayernormB());
});

test('rmsnorm · dX / dg 与数值差分一致（db 恒 null）', () => {
  assertGradOk(opRmsnorm());
  assertGradOk(opRmsnormG());
});

test('matmul · dA=W·B^T 与 dB=A^T·W 与数值差分一致（近机器精度）', () => {
  assertGradOk(opMatmulA());
  assertGradOk(opMatmulB());
});

test('softmax · dS=P⊙(w−ΣwP) 与数值差分一致', () => { assertGradOk(opSoftmax()); });

test('attention · registry VJP（attnBwd）dqkv 与数值差分一致', () => { assertGradOk(opAttention()); });

test('checkAll · 全算子汇总 allPass 且表格含每例 maxErr', () => {
  const report = [
    opGelu(), opSilu(), opSwiglu(),
    opLayernorm(), opLayernormG(), opLayernormB(),
    opRmsnorm(), opRmsnormG(),
    opMatmulA(), opMatmulB(),
    opSoftmax(), opAttention(),
  ];
  const { allPass, text } = GC.checkAll(report);
  assert.strictEqual(allPass, true);
  assert.ok(text.includes('ALL PASS'));
  for (const r of report) assert.ok(text.includes(r.name), `汇总表应含 ${r.name}`);
  for (const r of report) assertGradOk(r);
});

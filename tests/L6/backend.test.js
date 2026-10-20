'use strict';
/* L6 后端运行时 · kernel 注册表 + 后端抽象 + 默认 CPU(JS) 后端
 * 验收要点：
 *   ① 默认 profile === 'exact'（零回归精确通道为默认数值路径）；
 *   ② 默认活跃后端为 cpu，setBackend('cpu') 可用，未知后端抛 '[L6]' 错；
 *   ③ cpu 后端内核与 NS.L8 为同一函数引用（Object.is），输出逐位一致；
 *   ④ registry register/get/has/list/dispatch 契约；
 *   ⑤ setProfile('fast') 后内核仍为 exact 引用（P0 fast 档复用 exact 内核）；
 *   ⑥ fallback 可用性排序；bootstrap 幂等且 L4 惰性补登（加载前/后调用均安全）。
 */
const { test } = require('node:test');
const assert = require('node:assert');
require('../../src/index.js');
const NS = globalThis.FlagshipAI;

// 确定性伪随机（禁 Math.random；与 tests/L8 同源约定）
const rng = NS.L9.rng.mulberry32(99);
function rnd(r, c) {
  const a = new Float64Array(r * c);
  for (let i = 0; i < a.length; i++) a[i] = rng() * 2 - 1;
  return a;
}
function bitExact(a, b) {
  if (!a || !b || a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (!Object.is(a[i], b[i])) return false;
  return true;
}

// L4 晚于 L6 加载：显式自举以补登 L4 内核（幂等；L4 缺席时安全跳过）。
NS.L6.kernel.bootstrap();

test('默认 profile 为 exact', () => {
  assert.strictEqual(NS.L6.backend.getProfile(), 'exact');
});

test('默认活跃后端为 cpu，setBackend(cpu) 保持活跃', () => {
  const b = NS.L6.backend.getBackend();
  assert.ok(b, '加载后应有默认活跃后端');
  assert.strictEqual(b.name, 'cpu');
  assert.strictEqual(NS.L6.cpu.backend, b, 'NS.L6.cpu.backend 应为已注册的 cpu 对象');
  NS.L6.backend.setBackend('cpu');
  assert.strictEqual(NS.L6.backend.getBackend(), b);
});

test('未知后端 setBackend 抛 [L6] 前缀错误', () => {
  assert.throws(() => NS.L6.backend.setBackend('webgpu'), /\[L6\]/);
  assert.strictEqual(NS.L6.backend.getBackend().name, 'cpu', '抛错后活跃后端不应被改动');
});

test("kernel('matmul') 与 NS.L8.matmul.matmul 为同一函数引用", () => {
  assert.ok(Object.is(NS.L6.backend.kernel('matmul'), NS.L8.matmul.matmul));
  assert.ok(Object.is(NS.L6.cpu.backend.kernels.matmul, NS.L8.matmul.matmul));
});

test("kernel('matmul') 输出与 NS.L8.matmul.matmul 逐位一致（4×5 @ 5×3）", () => {
  const A = rnd(4, 5), B = rnd(5, 3);
  const viaBackend = NS.L6.backend.kernel('matmul')(A, B, 4, 5, 3);
  const viaL8 = NS.L8.matmul.matmul(A, B, 4, 5, 3);
  assert.ok(bitExact(viaBackend, viaL8), '逐位不一致');
});

test("kernel('matmul') 输出逐位一致（6×2 @ 2×7，第二形状）", () => {
  const A = rnd(6, 2), B = rnd(2, 7);
  const viaBackend = NS.L6.backend.kernel('matmul')(A, B, 6, 2, 7);
  const viaL8 = NS.L8.matmul.matmul(A, B, 6, 2, 7);
  assert.ok(bitExact(viaBackend, viaL8), '逐位不一致');
});

test("kernel('softmaxRows') 同一引用且输出逐位一致", () => {
  assert.ok(Object.is(NS.L6.backend.kernel('softmaxRows'), NS.L8.reduce.softmaxRows));
  const S = rnd(5, 7);
  const viaBackend = NS.L6.backend.kernel('softmaxRows')(S, 5, 7);
  const viaL8 = NS.L8.reduce.softmaxRows(S, 5, 7);
  assert.ok(bitExact(viaBackend, viaL8), '逐位不一致');
});

test("kernel('mm')/kernel('mmAtB') 同一引用；量化与非量化路径均逐位一致", () => {
  assert.ok(Object.is(NS.L6.backend.kernel('mm'), NS.L8.matmul.mm));
  assert.ok(Object.is(NS.L6.backend.kernel('mmAtB'), NS.L8.matmul.mmAtB));
  const A = rnd(4, 5);
  const W = { d: rnd(5, 3) };
  assert.ok(bitExact(
    NS.L6.backend.kernel('mm')(A, W, 4, 5, 3),
    NS.L8.matmul.mm(A, W, 4, 5, 3)), '非量化路径逐位不一致');
  const q = new Int8Array(15);
  for (let i = 0; i < q.length; i++) q[i] = (i % 7) - 3;
  const Wq = { q, scale: 0.03 };
  assert.ok(bitExact(
    NS.L6.backend.kernel('mm')(A, Wq, 4, 5, 3),
    NS.L8.matmul.mm(A, Wq, 4, 5, 3)), '量化路径逐位不一致');
});

test('registry register/get/has/list 契约（含覆盖再注册）', () => {
  const K = NS.L6.kernel;
  const f1 = (a, b) => a + b;
  const f2 = (a, b) => a - b;
  assert.strictEqual(K.register('test.add', f1), f1);
  assert.ok(Object.is(K.get('test.add'), f1));
  assert.strictEqual(K.has('test.add'), true);
  assert.ok(K.list().indexOf('test.add') >= 0, 'list 应包含新登记项');
  assert.ok(K.list().indexOf('matmul') >= 0, 'list 应包含 L8 内核');
  K.register('test.add', f2); // 显式覆盖是注册表的预期用法
  assert.ok(Object.is(K.get('test.add'), f2), '覆盖再注册应生效');
});

test('registry get 未登记返回 null；dispatch 未登记抛 [L6] 错', () => {
  const K = NS.L6.kernel;
  assert.strictEqual(K.get('nope.missing'), null);
  assert.strictEqual(K.has('nope.missing'), false);
  assert.throws(() => K.dispatch('nope.missing'), /\[L6\]/);
});

test("dispatch('matmul', ...) 与 NS.L8 直调逐位一致", () => {
  const A = rnd(3, 4), B = rnd(4, 2);
  const viaDispatch = NS.L6.kernel.dispatch('matmul', A, B, 3, 4, 2);
  const viaL8 = NS.L8.matmul.matmul(A, B, 3, 4, 2);
  assert.ok(bitExact(viaDispatch, viaL8), '逐位不一致');
});

test("setProfile('fast') 后 getProfile==='fast' 且内核仍为 exact 引用（P0）", () => {
  NS.L6.backend.setProfile('fast');
  assert.strictEqual(NS.L6.backend.getProfile(), 'fast');
  // P0：fast 档复用 exact 内核（fast/blocked 路径 deferred to P2，diff 恒 0）
  assert.ok(Object.is(NS.L6.backend.kernel('matmul'), NS.L8.matmul.matmul));
  assert.ok(Object.is(NS.L6.backend.kernel('softmaxRows'), NS.L8.reduce.softmaxRows));
  NS.L6.backend.setProfile('exact');
  assert.strictEqual(NS.L6.backend.getProfile(), 'exact');
  assert.throws(() => NS.L6.backend.setProfile('turbo'), /\[L6\]/);
});

test('fallback 按可用性排序取第一个已注册后端', () => {
  const cpu = NS.L6.backend.fallback(['cpu']);
  assert.ok(cpu && cpu.name === 'cpu', "fallback(['cpu']) 应返回 cpu");
  assert.strictEqual(NS.L6.backend.getBackend(), cpu, 'fallback 应激活所选后端');
  const skipped = NS.L6.backend.fallback(['webgpu', 'cpu']);
  assert.ok(skipped && skipped.name === 'cpu', '应跳过未注册的 webgpu 落到 cpu');
  assert.strictEqual(NS.L6.backend.fallback([]), null, '无一可用应返回 null');
  assert.throws(() => NS.L6.backend.fallback('cpu'), /\[L6\]/);
});

test('bootstrap 后 L4 内核已登记且为 NS.L4 同一引用', () => {
  const K = NS.L6.kernel;
  const pairs = [
    ['layernormFwd', NS.L4.norm.layernormFwd],
    ['rmsnormFwd', NS.L4.norm.rmsnormFwd],
    ['geluFwd', NS.L4.activation.geluFwd],
    ['swigluFwd', NS.L4.activation.swigluFwd],
    ['ropeApply', NS.L4.positional.ropeApply],
    ['crossEntropy', NS.L4.loss.crossEntropy],
    ['moeForward', NS.L4.moe.moeForward],
    ['attnFwd', NS.L4.attention.attnFwd],
  ];
  for (const [name, fn] of pairs) {
    assert.strictEqual(K.has(name), true, name + ' 应已登记');
    assert.ok(Object.is(K.get(name), fn), name + ' 应为 NS.L4 同一引用');
  }
  // L8 引用不得被 L4 同名门面顶替
  assert.ok(Object.is(K.get('softmaxRows'), NS.L8.reduce.softmaxRows));
  assert.ok(Object.is(K.get('mm'), NS.L8.matmul.mm));
});

test('cpu 后端经 bootstrap 桥接镜像 L4 内核', () => {
  assert.ok(Object.is(NS.L6.backend.kernel('rmsnormFwd'), NS.L4.norm.rmsnormFwd));
  assert.ok(Object.is(NS.L6.backend.kernel('geluFwd'), NS.L4.activation.geluFwd));
  assert.ok(Object.is(NS.L6.backend.kernel('attnFwd'), NS.L4.attention.attnFwd));
  // L8 引用保持恒等
  assert.ok(Object.is(NS.L6.backend.kernel('matmul'), NS.L8.matmul.matmul));
});

test('bootstrap 幂等：重复调用不改变登记集合与引用', () => {
  const K = NS.L6.kernel;
  const before = K.list().slice();
  K.bootstrap();
  K.bootstrap();
  assert.deepStrictEqual(K.list(), before, '重复 bootstrap 不应改变登记集合');
  assert.ok(Object.is(K.get('rmsnormFwd'), NS.L4.norm.rmsnormFwd));
  assert.ok(Object.is(K.get('matmul'), NS.L8.matmul.matmul));
});

test('bootstrap 在 L4 缺席时安全（守卫跳过 L4 段，L8 内核保持可用）', () => {
  const saved = NS.L4;
  NS.L4 = undefined; // 模拟 L4 尚未加载的时序
  try {
    NS.L6.kernel.bootstrap(); // 不得抛错
    assert.strictEqual(NS.L6.kernel.has('matmul'), true, 'L8 内核应保持可用');
    assert.ok(Object.is(NS.L6.backend.kernel('matmul'), NS.L8.matmul.matmul));
  } finally {
    NS.L4 = saved;
    NS.L6.kernel.bootstrap(); // 复原补登
  }
  assert.ok(Object.is(NS.L6.kernel.get('rmsnormFwd'), NS.L4.norm.rmsnormFwd));
});

test('cpu 后端 caps 自省元数据', () => {
  assert.deepStrictEqual(NS.L6.cpu.backend.caps, {
    simd: false, threads: false, deterministic: true,
  });
  assert.strictEqual(NS.L6.cpu.backend.profile, 'exact');
});

test('addInto / rowSumInto / topkRows 同一引用且逐位一致', () => {
  const B6 = NS.L6.backend;
  assert.ok(Object.is(B6.kernel('addInto'), NS.L8.reduce.addInto));
  assert.ok(Object.is(B6.kernel('rowSumInto'), NS.L8.reduce.rowSumInto));
  assert.ok(Object.is(B6.kernel('topkRows'), NS.L8.reduce.topkRows));
  // addInto（就地累加）
  const d1 = rnd(6), d2 = Float64Array.from(d1), src = rnd(6);
  B6.kernel('addInto')(d1, src);
  NS.L8.reduce.addInto(d2, src);
  assert.ok(bitExact(d1, d2), 'addInto 逐位不一致');
  // rowSumInto
  const Y = rnd(3, 4);
  const s1 = new Float64Array(4), s2 = new Float64Array(4);
  B6.kernel('rowSumInto')(s1, Y, 3, 4);
  NS.L8.reduce.rowSumInto(s2, Y, 3, 4);
  assert.ok(bitExact(s1, s2), 'rowSumInto 逐位不一致');
  // topkRows
  const P = rnd(4, 5);
  const r1 = B6.kernel('topkRows')(P, 4, 5, 2);
  const r2 = NS.L8.reduce.topkRows(P, 4, 5, 2);
  assert.ok(bitExact(r1.idx, r2.idx) && bitExact(r1.val, r2.val), 'topkRows 逐位不一致');
});

test('dispatch 对 L4 内核同样可用（补登后）', () => {
  const X = rnd(2, 6), g = rnd(1, 6), b = rnd(1, 6);
  const viaDispatch = NS.L6.kernel.dispatch('layernormFwd', X, g, b, 2, 6);
  const viaL4 = NS.L4.norm.layernormFwd(X, g, b, 2, 6);
  assert.ok(bitExact(viaDispatch.Y, viaL4.Y), 'layernormFwd 逐位不一致');
});

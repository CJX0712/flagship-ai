'use strict';
/* L5 自动微分 · VJP 注册中心（registry）
 * 验收：bootstrap 后 8 个算子 VJP 全部就位，且与 L4 backward 为逐函数同一
 * 引用（Path A：登记而非重写 → 与 v11 反向逐位一致，R3 Δ=0 可证）。
 */
const { test } = require('node:test');
const assert = require('node:assert');
require('../../src/index.js');
const NS = globalThis.FlagshipAI;
const REG = NS.L5.registry;

// list() 输出为字典序，恰好这 8 个（bootstrap 之后、任何自定义登记之前）
const EXPECTED_SORTED = ['attention', 'gelu', 'layernorm', 'moe', 'qknorm', 'rmsnorm', 'softmax', 'swiglu'];

test('bootstrap · 幂等且恰好登记 8 个 VJP', () => {
  assert.strictEqual(REG.bootstrap(), true);   // L4 已在场（index.js 先加载 L4）
  assert.strictEqual(REG.bootstrap(), true);   // 幂等：重复调用不抛不重复
  assert.deepStrictEqual(REG.list(), EXPECTED_SORTED);
});

test('bootstrap · get(name) 与 L4 backward 为同一函数引用', () => {
  const pairs = [
    ['layernorm', NS.L4.norm.layernormBwd],
    ['rmsnorm',   NS.L4.norm.rmsnormBwd],
    ['qknorm',    NS.L4.norm.qkNormBwd],
    ['gelu',      NS.L4.activation.geluBwd],
    ['swiglu',    NS.L4.activation.swigluBwd],
    ['softmax',   NS.L4.softmax.softmaxBwd],
    ['moe',       NS.L4.moe.moeBackward],
    ['attention', NS.L4.attention.attnBwd],
  ];
  for (const [name, ref] of pairs) {
    assert.strictEqual(REG.has(name), true, `has('${name}') 应为 true`);
    assert.strictEqual(REG.get(name), ref, `get('${name}') 应与 L4 backward 同一引用`);
  }
});

test('get/has · 未登记名称返回 null / false（不抛）', () => {
  assert.strictEqual(REG.has('no-such-op'), false);
  assert.strictEqual(REG.get('no-such-op'), null);
  assert.strictEqual(REG.has(''), false);
  assert.strictEqual(REG.has('LAYERNORM'), false); // 大小写敏感
});

test('register · 自定义 VJP 可登记并取回；非法参数抛 TypeError', () => {
  const custom = (cache, dY) => dY;
  assert.strictEqual(REG.register('test.custom.vjp', custom), custom);
  assert.strictEqual(REG.get('test.custom.vjp'), custom);
  assert.strictEqual(REG.has('test.custom.vjp'), true);
  assert.ok(REG.list().includes('test.custom.vjp'));
  assert.throws(() => REG.register('bad', null), TypeError);
  assert.throws(() => REG.register('bad', 42), TypeError);
  assert.throws(() => REG.register('', custom), TypeError);
  assert.throws(() => REG.register(123, custom), TypeError);
});

test('bootstrap · L4 缺席时安全返回 false，恢复后仍可用', () => {
  const saved = NS.L4;
  NS.L4 = null;
  try {
    assert.strictEqual(REG.bootstrap(), false);          // 不抛、不破坏既有登记
    assert.strictEqual(REG.has('gelu'), true);           // 已登记项不受影响
    assert.strictEqual(REG.get('rmsnorm'), saved.norm.rmsnormBwd);
  } finally {
    NS.L4 = saved;                                        // 恢复现场（零副作用）
  }
  assert.strictEqual(REG.bootstrap(), true);
  assert.strictEqual(REG.has('swiglu'), true);
});

test('VJP 调用 · registry 取用的 backward 与直调逐位一致', () => {
  const rng = NS.L9.rng.mulberry32(5);
  const r = 2, c = 4;
  const X = new Float64Array(r * c), g = new Float64Array(c), b = new Float64Array(c), dY = new Float64Array(r * c);
  for (let i = 0; i < X.length; i++) X[i] = rng() * 2 - 1;
  for (let j = 0; j < c; j++) { g[j] = 0.5 + rng(); b[j] = rng() * 2 - 1; dY[j] = rng() * 2 - 1; dY[c + j] = rng() * 2 - 1; }
  const f = NS.L4.norm.layernormFwd(X, g, b, r, c);
  const viaReg = REG.get('layernorm')(dY, f.xhat, f.rstd, g, r, c);
  const direct = NS.L4.norm.layernormBwd(dY, f.xhat, f.rstd, g, r, c);
  assert.ok(NS.L9.assert.assertBitExact(viaReg.dX, direct.dX), 'dX 应逐位一致');
  assert.ok(NS.L9.assert.assertBitExact(viaReg.dg, direct.dg), 'dg 应逐位一致');
  assert.ok(NS.L9.assert.assertBitExact(viaReg.db, direct.db), 'db 应逐位一致');
  // gelu 标量 VJP
  const w = new Float64Array(c);
  for (let j = 0; j < c; j++) w[j] = rng() * 2 - 1;
  const dVia = REG.get('gelu')(w, X), dDir = NS.L4.activation.geluBwd(w, X);
  assert.ok(NS.L9.assert.assertBitExact(dVia, dDir), 'gelu dX 应逐位一致');
});

test('tape 契约 · spec/emptyLayer/validateCache 自洽（与 registry 同批加载）', () => {
  const spec = NS.L5.tape.spec();
  assert.deepStrictEqual(spec.cache.keys,
    ['T', 'dh', 'H', 'D', 'L', 'layers', 'lnf', 'Hf', 'toks', 'logits']);
  assert.deepStrictEqual(spec.layer.keys,
    ['ln1', 'A', 'qkv', 'Ps', 'ctx', 'xMid', 'ln2', 'M', 'pre1', 'h1', 'moe', 'qk']);
  const el = NS.L5.tape.emptyLayer();
  assert.deepStrictEqual(Object.keys(el), spec.layer.keys);
  assert.ok(Object.values(el).every((v) => v === null));
  const ec = NS.L5.tape.emptyCache();
  assert.deepStrictEqual(Object.keys(ec), spec.cache.keys);
  ec.layers.push(el);
  ec.L = ec.layers.length; // 层数与 L 对齐（validateCache 的附加校验）
  assert.deepStrictEqual(NS.L5.tape.validateCache(ec), []);
  const bad = NS.L5.tape.emptyCache();
  delete bad.Hf;
  bad.layers.push({ ln1: null });
  assert.deepStrictEqual(NS.L5.tape.validateCache(bad), ['Hf', 'layers[0].A', 'layers[0].qkv', 'layers[0].Ps',
    'layers[0].ctx', 'layers[0].xMid', 'layers[0].ln2', 'layers[0].M', 'layers[0].pre1', 'layers[0].h1',
    'layers[0].moe', 'layers[0].qk', 'layers.length(1)!==L(0)']);
});

/* ============================================================================
 * L10 微指令 · A 域（标量算术与初等函数）· 第 3/3 批（A065–A096）
 * ----------------------------------------------------------------------------
 * 涵盖：三角函数收尾（A065–A072）、双曲函数（A073–A080）、激活函数（A081–A090）、
 *      取整/稳定性（A091–A096）。激活函数复用 L9.special（零重写）。
 * ==========================================================================*/
;(function (NS) {
  'use strict';
  NS.L10 = NS.L10 || {};
  const SP = NS.L9.special;
  function reg(i, o) { NS.L10.register('A', i, o); }
  const rc = (a, b, t) => NS.L10.relClose(a, b, t);

  // ── 三角函数收尾（A065–A072）──────────────────────────────────────────
  reg(65, {
    name: '三角 · 倍角 cos2x', intuition: 'cos(2x) = cos²x − sin²x（允许浮点误差）。',
    math: 'cos 2x = cos²x − sin²x', pseudocode: 'return Math.cos(2*x)', complexity: 'O(1)',
    ref: '三角倍角公式', commonPitfall: '需容差。',
    invariant: 'relClose(cos 2x, cos²x−sin²x, 1e-9)',
    inputs: ['x'], domains: { x: 'trig' },
    run: (x) => Math.cos(2 * x), predicate: (x) => rc(Math.cos(2 * x), Math.cos(x) ** 2 - Math.sin(x) ** 2, 1e-9),
  });
  reg(66, {
    name: '三角 · sin 周期 2π', intuition: 'sin(x+2π) = sin x（允许浮点误差）。',
    math: 'sin(x+2π) = sin x', pseudocode: 'return Math.sin(x+2*Math.PI)', complexity: 'O(1)',
    ref: '三角函数周期性', commonPitfall: '需容差；x 增 2π 后参数变大但仍精确。',
    invariant: 'relClose(sin(x+2π), sin x, 1e-9)',
    inputs: ['x'], domains: { x: 'angle' },
    run: (x) => Math.sin(x + 2 * Math.PI), predicate: (x) => rc(Math.sin(x + 2 * Math.PI), Math.sin(x), 1e-9),
  });
  reg(67, {
    name: '三角 · cos 周期 2π', intuition: 'cos(x+2π) = cos x（允许浮点误差）。',
    math: 'cos(x+2π) = cos x', pseudocode: 'return Math.cos(x+2*Math.PI)', complexity: 'O(1)',
    ref: '三角函数周期性', commonPitfall: '需容差。',
    invariant: 'relClose(cos(x+2π), cos x, 1e-9)',
    inputs: ['x'], domains: { x: 'angle' },
    run: (x) => Math.cos(x + 2 * Math.PI), predicate: (x) => rc(Math.cos(x + 2 * Math.PI), Math.cos(x), 1e-9),
  });
  reg(68, {
    name: '三角 · sin 有界', intuition: '|sin x| ≤ 1。',
    math: '|sin x| ≤ 1', pseudocode: 'return Math.sin(x)', complexity: 'O(1)',
    ref: '三角函数有界性', commonPitfall: '无', invariant: '|sin x| ≤ 1',
    inputs: ['x'], domains: { x: 'angle' },
    run: (x) => Math.sin(x), predicate: (x) => Math.abs(Math.sin(x)) <= 1,
  });
  reg(69, {
    name: '三角 · cos 有界', intuition: '|cos x| ≤ 1。',
    math: '|cos x| ≤ 1', pseudocode: 'return Math.cos(x)', complexity: 'O(1)',
    ref: '三角函数有界性', commonPitfall: '无', invariant: '|cos x| ≤ 1',
    inputs: ['x'], domains: { x: 'angle' },
    run: (x) => Math.cos(x), predicate: (x) => Math.abs(Math.cos(x)) <= 1,
  });
  reg(70, {
    name: '三角 · atan2 与 y 轴', intuition: 'atan2(y,0) = sign(y)·π/2（允许浮点误差）。',
    math: 'atan2(y,0) = sign(y)·π/2', pseudocode: 'return Math.atan2(y,0)', complexity: 'O(1)',
    ref: 'atan2 定义', commonPitfall: 'y 不可为 0。',
    invariant: '对 y≠0：relClose(atan2(y,0), sign(y)·π/2, 1e-9)',
    inputs: ['y'], domains: { y: 'nonZero' },
    run: (y) => Math.atan2(y, 0), predicate: (y) => rc(Math.atan2(y, 0), Math.sign(y) * Math.PI / 2, 1e-9),
  });
  reg(71, {
    name: '三角 · atan(tan x)=x', intuition: '反正切还原（x∈[-1.5,1.5]⊂(-π/2,π/2)）。',
    math: 'atan(tan x) = x', pseudocode: 'return Math.atan(Math.tan(x))', complexity: 'O(1)',
    ref: '反三角函数', commonPitfall: '需容差。',
    invariant: 'relClose(atan(tan x), x, 1e-9)',
    inputs: ['x'], domains: { x: 'trig' },
    run: (x) => Math.atan(Math.tan(x)), predicate: (x) => rc(Math.atan(Math.tan(x)), x, 1e-9),
  });
  reg(72, {
    name: '三角 · tan 奇函数', intuition: 'tan(-x) = -tan(x)（允许浮点误差）。',
    math: 'tan(-x) = -tan x', pseudocode: 'return Math.tan(-x)', complexity: 'O(1)',
    ref: '三角函数奇偶性', commonPitfall: '需容差；x 远离 π/2。',
    invariant: 'relClose(tan(-x), -tan x, 1e-9)',
    inputs: ['x'], domains: { x: 'trig' },
    run: (x) => Math.tan(-x), predicate: (x) => rc(Math.tan(-x), -Math.tan(x), 1e-9),
  });

  // ── 双曲函数（A073–A080）─────────────────────────────────────────────
  reg(73, {
    name: '双曲 · sinh 奇函数', intuition: 'sinh(-x) = -sinh(x)（允许浮点误差）。',
    math: 'sinh(-x) = -sinh x', pseudocode: 'return Math.sinh(-x)', complexity: 'O(1)',
    ref: '双曲函数奇偶性', commonPitfall: '需容差。',
    invariant: 'relClose(sinh(-x), -sinh x, 1e-9)',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => Math.sinh(-x), predicate: (x) => rc(Math.sinh(-x), -Math.sinh(x), 1e-9),
  });
  reg(74, {
    name: '双曲 · cosh 偶函数', intuition: 'cosh(-x) = cosh(x)（允许浮点误差）。',
    math: 'cosh(-x) = cosh x', pseudocode: 'return Math.cosh(-x)', complexity: 'O(1)',
    ref: '双曲函数奇偶性', commonPitfall: '需容差。',
    invariant: 'relClose(cosh(-x), cosh x, 1e-9)',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => Math.cosh(-x), predicate: (x) => rc(Math.cosh(-x), Math.cosh(x), 1e-9),
  });
  reg(75, {
    name: '双曲 · 恒等式 cosh²−sinh²=1', intuition: '双曲勾股恒等式；直接作差在大 x 下因相消误差失效，改用稳定形式 cosh x = √(sinh²x+1)（允许浮点误差）。',
    math: 'cosh²x − sinh²x = 1', pseudocode: 'return Math.cosh(x)**2-Math.sinh(x)**2', complexity: 'O(1)',
    ref: '双曲恒等式', commonPitfall: '直接 cosh²−sinh² 在大 x 下浮点相消误差可达 1e-3；改用 √(sinh²+1)。',
    invariant: 'relClose(cosh x, √(sinh²x+1), 1e-12)',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => Math.cosh(x) ** 2 - Math.sinh(x) ** 2, predicate: (x) => { const c = Math.cosh(x), s = Math.sinh(x); return rc(c, Math.sqrt(s * s + 1), 1e-12); },
  });
  reg(76, {
    name: '双曲 · sinh 定义式', intuition: 'sinh x = (e^x − e^{−x})/2（允许浮点误差）。',
    math: 'sinh x = (e^x−e^{−x})/2', pseudocode: 'return Math.sinh(x)', complexity: 'O(1)',
    ref: '双曲函数定义', commonPitfall: '需容差。',
    invariant: 'relClose(sinh x, (e^x−e^{−x})/2, 1e-9)',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => Math.sinh(x), predicate: (x) => rc(Math.sinh(x), (Math.exp(x) - Math.exp(-x)) / 2, 1e-9),
  });
  reg(77, {
    name: '双曲 · cosh 定义式', intuition: 'cosh x = (e^x + e^{−x})/2（允许浮点误差）。',
    math: 'cosh x = (e^x+e^{−x})/2', pseudocode: 'return Math.cosh(x)', complexity: 'O(1)',
    ref: '双曲函数定义', commonPitfall: '需容差。',
    invariant: 'relClose(cosh x, (e^x+e^{−x})/2, 1e-9)',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => Math.cosh(x), predicate: (x) => rc(Math.cosh(x), (Math.exp(x) + Math.exp(-x)) / 2, 1e-9),
  });
  reg(78, {
    name: '双曲 · tanh 奇函数', intuition: 'tanh(-x) = -tanh(x)（允许浮点误差）。',
    math: 'tanh(-x) = -tanh x', pseudocode: 'return Math.tanh(-x)', complexity: 'O(1)',
    ref: '双曲函数奇偶性', commonPitfall: '需容差。',
    invariant: 'relClose(tanh(-x), -tanh x, 1e-9)',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => Math.tanh(-x), predicate: (x) => rc(Math.tanh(-x), -Math.tanh(x), 1e-9),
  });
  reg(79, {
    name: '双曲 · tanh 有界', intuition: '|tanh x| < 1（有限 x）。',
    math: '|tanh x| < 1', pseudocode: 'return Math.tanh(x)', complexity: 'O(1)',
    ref: '双曲函数有界性', commonPitfall: '无', invariant: '|tanh x| < 1',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => Math.tanh(x), predicate: (x) => Math.abs(Math.tanh(x)) < 1,
  });
  reg(80, {
    name: '双曲 · tanh(0)=0', intuition: 'tanh 0 逐位为 0。',
    math: 'tanh 0 = 0', pseudocode: 'return Math.tanh(0)', complexity: 'O(1)',
    ref: '双曲函数', commonPitfall: '无', invariant: 'Object.is(tanh 0, 0)',
    inputs: [], domains: {},
    run: () => Math.tanh(0), predicate: () => Object.is(Math.tanh(0), 0),
  });

  // ── 激活函数（A081–A090，复用 L9.special）────────────────────────────
  reg(81, {
    name: '激活 · ReLU 幂等', intuition: 'ReLU 对自身幂等（逐位）。',
    math: 'ReLU(ReLU(a)) = ReLU(a)', pseudocode: 'return SP.relu(SP.relu(a))', complexity: 'O(1)',
    ref: 'Nair & Hinton 2010', commonPitfall: '无', invariant: 'Object.is(ReLU(ReLU(a)), ReLU(a))',
    inputs: ['a'], domains: { a: 'finite' },
    run: (a) => SP.relu(SP.relu(a)), predicate: (a) => Object.is(SP.relu(SP.relu(a)), SP.relu(a)),
  });
  reg(82, {
    name: '激活 · ReLU 非负', intuition: 'ReLU 输出恒非负。',
    math: 'ReLU(a) ≥ 0', pseudocode: 'return SP.relu(a)', complexity: 'O(1)',
    ref: 'Nair & Hinton 2010', commonPitfall: '无', invariant: 'ReLU(a) ≥ 0',
    inputs: ['a'], domains: { a: 'finite' },
    run: (a) => SP.relu(a), predicate: (a) => SP.relu(a) >= 0,
  });
  reg(83, {
    name: '激活 · Sigmoid 区间', intuition: 'Sigmoid 在适中输入下输出严格落在 (0,1)（|a|≤10 内不会饱和到精确 0/1）。',
    math: '0 < σ(a) < 1', pseudocode: 'return SP.sigmoid(a)', complexity: 'O(1)',
    ref: 'Goodfellow 2016', commonPitfall: '|a| 过大会浮点饱和到精确 0/1，故域限定 small（±10）。', invariant: '对 |a|≤10：0 < σ(a) < 1',
    inputs: ['a'], domains: { a: 'small' },
    run: (a) => SP.sigmoid(a), predicate: (a) => { const s = SP.sigmoid(a); return s > 0 && s < 1; },
  });
  reg(84, {
    name: '激活 · Sigmoid 对称', intuition: 'σ(−a) = 1 − σ(a)（允许浮点误差）。',
    math: 'σ(−a) = 1 − σ(a)', pseudocode: 'return SP.sigmoid(-a)', complexity: 'O(1)',
    ref: 'Goodfellow 2016', commonPitfall: '需容差。',
    invariant: 'relClose(σ(−a), 1−σ(a), 1e-9)',
    inputs: ['a'], domains: { a: 'finite' },
    run: (a) => SP.sigmoid(-a), predicate: (a) => rc(SP.sigmoid(-a), 1 - SP.sigmoid(a), 1e-9),
  });
  reg(85, {
    name: '激活 · Sigmoid(0)=½', intuition: 'σ(0) 逐位为 0.5。',
    math: 'σ(0) = 0.5', pseudocode: 'return SP.sigmoid(0)', complexity: 'O(1)',
    ref: 'Goodfellow 2016', commonPitfall: '无', invariant: 'Object.is(σ(0), 0.5)',
    inputs: [], domains: {},
    run: () => SP.sigmoid(0), predicate: () => Object.is(SP.sigmoid(0), 0.5),
  });
  reg(86, {
    name: '激活 · Softplus 非负', intuition: 'Softplus 为 ReLU 平滑近似，恒非负。',
    math: 'softplus(a) ≥ 0', pseudocode: 'return SP.softplus(a)', complexity: 'O(1)',
    ref: 'Dugas et al. 2001', commonPitfall: '无', invariant: 'softplus(a) ≥ 0',
    inputs: ['a'], domains: { a: 'finite' },
    run: (a) => SP.softplus(a), predicate: (a) => SP.softplus(a) >= 0,
  });
  reg(87, {
    name: '激活 · Softplus(0)=ln2', intuition: 'softplus(0) 逐位为 ln2。',
    math: 'softplus(0) = ln 2', pseudocode: 'return SP.softplus(0)', complexity: 'O(1)',
    ref: 'Dugas et al. 2001', commonPitfall: '无', invariant: 'Object.is(softplus(0), ln 2)',
    inputs: [], domains: {},
    run: () => SP.softplus(0), predicate: () => Object.is(SP.softplus(0), Math.log(2)),
  });
  reg(88, {
    name: '激活 · Swish(0)=0', intuition: 'Swish(0) 逐位为 0。',
    math: 'swish(0) = 0', pseudocode: 'return SP.swish(0)', complexity: 'O(1)',
    ref: 'Ramachandran 2017', commonPitfall: '无', invariant: 'Object.is(swish(0), 0)',
    inputs: [], domains: {},
    run: () => SP.swish(0), predicate: () => Object.is(SP.swish(0), 0),
  });
  reg(89, {
    name: '激活 · ELU(0)=0', intuition: 'ELU(0) 逐位为 0。',
    math: 'elu(0) = 0', pseudocode: 'return SP.elu(0)', complexity: 'O(1)',
    ref: 'Clevert 2015', commonPitfall: '无', invariant: 'Object.is(elu(0), 0)',
    inputs: [], domains: {},
    run: () => SP.elu(0), predicate: () => Object.is(SP.elu(0), 0),
  });
  reg(90, {
    name: '激活 · SELU(0)=0', intuition: 'SELU(0) 逐位为 0。',
    math: 'selu(0) = 0', pseudocode: 'return SP.selu(0)', complexity: 'O(1)',
    ref: 'Klambauer 2017', commonPitfall: '无', invariant: 'Object.is(selu(0), 0)',
    inputs: [], domains: {},
    run: () => SP.selu(0), predicate: () => Object.is(SP.selu(0), 0),
  });

  // ── 取整 / 稳定性（A091–A096）────────────────────────────────────────
  reg(91, {
    name: '取整 · floor 幂等', intuition: 'floor 对自身幂等（逐位）。',
    math: '⌊⌊a⌋⌋ = ⌊a⌋', pseudocode: 'return Math.floor(Math.floor(a))', complexity: 'O(1)',
    ref: 'IEEE 754', commonPitfall: '无', invariant: 'Object.is(⌊⌊a⌋⌋, ⌊a⌋)',
    inputs: ['a'], domains: { a: 'finite' },
    run: (a) => Math.floor(Math.floor(a)), predicate: (a) => Object.is(Math.floor(Math.floor(a)), Math.floor(a)),
  });
  reg(92, {
    name: '取整 · ceil 幂等', intuition: 'ceil 对自身幂等（逐位）。',
    math: '⌈⌈a⌉⌉ = ⌈a⌉', pseudocode: 'return Math.ceil(Math.ceil(a))', complexity: 'O(1)',
    ref: 'IEEE 754', commonPitfall: '无', invariant: 'Object.is(⌈⌈a⌉⌉, ⌈a⌉)',
    inputs: ['a'], domains: { a: 'finite' },
    run: (a) => Math.ceil(Math.ceil(a)), predicate: (a) => Object.is(Math.ceil(Math.ceil(a)), Math.ceil(a)),
  });
  reg(93, {
    name: '取整 · trunc 幂等', intuition: 'trunc 对自身幂等（逐位）。',
    math: 'trunc(trunc(a)) = trunc(a)', pseudocode: 'return Math.trunc(Math.trunc(a))', complexity: 'O(1)',
    ref: 'IEEE 754', commonPitfall: '无', invariant: 'Object.is(trunc(trunc(a)), trunc(a))',
    inputs: ['a'], domains: { a: 'finite' },
    run: (a) => Math.trunc(Math.trunc(a)), predicate: (a) => Object.is(Math.trunc(Math.trunc(a)), Math.trunc(a)),
  });
  reg(94, {
    name: '取整 · floor ≤ x ≤ ceil', intuition: 'floor 与 ceil 夹住原数（逐位比较）。',
    math: '⌊a⌋ ≤ a ≤ ⌈a⌉', pseudocode: 'return Math.floor(a)', complexity: 'O(1)',
    ref: 'IEEE 754', commonPitfall: '无', invariant: '⌊a⌋ ≤ a ≤ ⌈a⌉',
    inputs: ['a'], domains: { a: 'finite' },
    run: (a) => Math.floor(a), predicate: (a) => Math.floor(a) <= a && a <= Math.ceil(a),
  });
  reg(95, {
    name: '取整 · round 输出整数', intuition: 'round 的结果必为整数。',
    math: 'isInteger(round(a))', pseudocode: 'return Math.round(a)', complexity: 'O(1)',
    ref: 'IEEE 754', commonPitfall: '半值舍去方向不同于 floor(a+0.5)，故只验证整数性。',
    invariant: 'Number.isInteger(round(a))',
    inputs: ['a'], domains: { a: 'finite' },
    run: (a) => Math.round(a), predicate: (a) => Number.isInteger(Math.round(a)),
  });
  reg(96, {
    name: '取整 · round 整数不动', intuition: '整数经 round 后逐位不变。',
    math: 'round(a) = a (a∈ℤ)', pseudocode: 'return Math.round(a)', complexity: 'O(1)',
    ref: 'IEEE 754', commonPitfall: '无', invariant: '对整数 a：Object.is(round(a), a)',
    inputs: ['a'], domains: { a: 'int' },
    run: (a) => Math.round(a), predicate: (a) => Object.is(Math.round(a), a),
  });
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

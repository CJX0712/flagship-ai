/* ============================================================================
 * L10 微指令 · A 域（标量算术与初等函数）· 第 3/4 批（A065–A096）
 * ----------------------------------------------------------------------------
 * 涵盖：三角函数收尾（A065–A072）、双曲函数（A073–A080）、激活函数（A081–A090）、
 *      取整/稳定性（A091–A096）。激活函数复用 L9.special（零重写）。
 *
 * ── 谓词纪律（本批全部重写，2026-10-23）─────────────────────────────────
 * 门禁只调 op.predicate，从不调 op.run（op-harness.js:43、probe-microisa.js:106）。
 * 原 32 项谓词全部内联复算 run 定义式或直接自比输入，run 若损坏门禁完全失明
 * （假绿）；A075 更甚——run 算朴素 cosh²−sinh²，谓词只比对稳定式，run 的产出
 * 从未被检验。现全部改为：先经 runOf('Axxx') 调 run 本体，再对 run 的产出
 * 断言不变量；oracle 取异构公式、输入导出值、常数锚或零参定点。
 * A075 的容差 1e-6 有实测依据（[-10,10] 上朴素式最大误差 7.45e-8，非放宽容差），
 * 同步改正文档中「误差可达 1e-3」的悲观误述。
 * ==========================================================================*/
;(function (NS) {
  'use strict';
  NS.L10 = NS.L10 || {};
  const SP = NS.L9.special;
  function reg(i, o) { NS.L10.register('A', i, o); }
  function runOf(key) { const o = NS.L10.ops[key]; return o && typeof o.run === 'function' ? o.run : null; }
  const rc = (a, b, t) => NS.L10.relClose(a, b, t);

  // ── 三角函数收尾（A065–A072）──────────────────────────────────────────
  reg(65, {
    name: '三角 · 倍角 cos2x', intuition: 'cos(2x) = cos²x − sin²x（允许浮点误差）。',
    math: 'cos 2x = cos²x − sin²x', pseudocode: 'return Math.cos(2*x)', complexity: 'O(1)',
    ref: '三角倍角公式', commonPitfall: '需容差。',
    invariant: 'relClose(run(x), cos²x−sin²x, 1e-9)（异构公式）',
    inputs: ['x'], domains: { x: 'trig' },
    run: (x) => Math.cos(2 * x),
    predicate: (x) => rc(runOf('A065')(x), Math.cos(x) ** 2 - Math.sin(x) ** 2, 1e-9),
  });
  reg(66, {
    name: '三角 · sin 周期 2π', intuition: 'sin(x+2π) = sin x（允许浮点误差）。',
    math: 'sin(x+2π) = sin x', pseudocode: 'return Math.sin(x+2*Math.PI)', complexity: 'O(1)',
    ref: '三角函数周期性', commonPitfall: '需容差；x 增 2π 后参数变大但仍精确。',
    invariant: 'relClose(run(x), sin x, 1e-9)（异构参数）',
    inputs: ['x'], domains: { x: 'angle' },
    run: (x) => Math.sin(x + 2 * Math.PI),
    predicate: (x) => rc(runOf('A066')(x), Math.sin(x), 1e-9),
  });
  reg(67, {
    name: '三角 · cos 周期 2π', intuition: 'cos(x+2π) = cos x（允许浮点误差）。',
    math: 'cos(x+2π) = cos x', pseudocode: 'return Math.cos(x+2*Math.PI)', complexity: 'O(1)',
    ref: '三角函数周期性', commonPitfall: '需容差。',
    invariant: 'relClose(run(x), cos x, 1e-9)（异构参数）',
    inputs: ['x'], domains: { x: 'angle' },
    run: (x) => Math.cos(x + 2 * Math.PI),
    predicate: (x) => rc(runOf('A067')(x), Math.cos(x), 1e-9),
  });
  reg(68, {
    name: '三角 · sin 有界', intuition: '|sin x| ≤ 1。',
    math: '|sin x| ≤ 1', pseudocode: 'return Math.sin(x)', complexity: 'O(1)',
    ref: '三角函数有界性', commonPitfall: '无',
    invariant: '|run(x)| ≤ 1，且定点锚 Object.is(run(0),0)、rc(run(π/2),1,1e-12) 确系 sin',
    inputs: ['x'], domains: { x: 'angle' },
    run: (x) => Math.sin(x),
    predicate: (x) => {
      const run = runOf('A068');
      if (Math.abs(run(x)) > 1) return false;
      return Object.is(run(0), 0) && rc(run(Math.PI / 2), 1, 1e-12);
    },
  });
  reg(69, {
    name: '三角 · cos 有界', intuition: '|cos x| ≤ 1。',
    math: '|cos x| ≤ 1', pseudocode: 'return Math.cos(x)', complexity: 'O(1)',
    ref: '三角函数有界性', commonPitfall: '无',
    invariant: '|run(x)| ≤ 1，且定点锚 Object.is(run(0),1)、rc(run(π),-1,1e-12) 确系 cos',
    inputs: ['x'], domains: { x: 'angle' },
    run: (x) => Math.cos(x),
    predicate: (x) => {
      const run = runOf('A069');
      if (Math.abs(run(x)) > 1) return false;
      return Object.is(run(0), 1) && rc(run(Math.PI), -1, 1e-12);
    },
  });
  reg(70, {
    name: '三角 · atan2 与 y 轴', intuition: 'atan2(y,0) = sign(y)·π/2（允许浮点误差）。',
    math: 'atan2(y,0) = sign(y)·π/2', pseudocode: 'return Math.atan2(y,0)', complexity: 'O(1)',
    ref: 'atan2 定义', commonPitfall: 'y 不可为 0。',
    invariant: '对 y≠0：relClose(run(y), sign(y)·π/2, 1e-9)（异构公式）',
    inputs: ['y'], domains: { y: 'nonZero' },
    run: (y) => Math.atan2(y, 0),
    predicate: (y) => rc(runOf('A070')(y), Math.sign(y) * Math.PI / 2, 1e-9),
  });
  reg(71, {
    name: '三角 · atan(tan x)=x', intuition: '反正切还原（x∈[-1.5,1.5]⊂(-π/2,π/2)）。',
    math: 'atan(tan x) = x', pseudocode: 'return Math.atan(Math.tan(x))', complexity: 'O(1)',
    ref: '反三角函数', commonPitfall: '需容差。',
    invariant: 'relClose(run(x), x, 1e-9)（oracle 为输入本身）',
    inputs: ['x'], domains: { x: 'trig' },
    run: (x) => Math.atan(Math.tan(x)),
    predicate: (x) => rc(runOf('A071')(x), x, 1e-9),
  });
  reg(72, {
    name: '三角 · tan 奇函数', intuition: 'tan(-x) = -tan(x)（允许浮点误差）。',
    math: 'tan(-x) = -tan x', pseudocode: 'return Math.tan(-x)', complexity: 'O(1)',
    ref: '三角函数奇偶性', commonPitfall: '需容差；x 远离 π/2。',
    invariant: 'relClose(run(x), -tan x, 1e-9)（异构公式）',
    inputs: ['x'], domains: { x: 'trig' },
    run: (x) => Math.tan(-x),
    predicate: (x) => rc(runOf('A072')(x), -Math.tan(x), 1e-9),
  });

  // ── 双曲函数（A073–A080）─────────────────────────────────────────────
  reg(73, {
    name: '双曲 · sinh 奇函数', intuition: 'sinh(-x) = -sinh(x)（允许浮点误差）。',
    math: 'sinh(-x) = -sinh x', pseudocode: 'return Math.sinh(-x)', complexity: 'O(1)',
    ref: '双曲函数奇偶性', commonPitfall: '需容差。',
    invariant: 'relClose(run(x), -sinh x, 1e-9)（异构公式）',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => Math.sinh(-x),
    predicate: (x) => rc(runOf('A073')(x), -Math.sinh(x), 1e-9),
  });
  reg(74, {
    name: '双曲 · cosh 偶函数', intuition: 'cosh(-x) = cosh(x)（允许浮点误差）。',
    math: 'cosh(-x) = cosh x', pseudocode: 'return Math.cosh(-x)', complexity: 'O(1)',
    ref: '双曲函数奇偶性', commonPitfall: '需容差。',
    invariant: 'relClose(run(x), cosh x, 1e-9)（异构公式）',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => Math.cosh(-x),
    predicate: (x) => rc(runOf('A074')(x), Math.cosh(x), 1e-9),
  });
  reg(75, {
    name: '双曲 · 恒等式 cosh²−sinh²=1', intuition: '双曲勾股恒等式；run 为朴素直接式（教学/警示用），大 x 下相消误差有实测上界。',
    math: 'cosh²x − sinh²x = 1', pseudocode: 'return Math.cosh(x)**2-Math.sinh(x)**2', complexity: 'O(1)',
    ref: '双曲恒等式', commonPitfall: '朴素式在 [-10,10] 实测最大误差 7.45e-8（|x|=10 处 2.98e-8），容差 1e-6 留 13× 余量；生产计算应改用稳定式 cosh x = √(sinh²x+1)（误差 3.6e-12）。',
    invariant: 'relClose(run(x), 1, 1e-6)（朴素式有据容差）；且稳定式 oracle rc(cosh x, √(sinh²x+1), 1e-12)',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => Math.cosh(x) ** 2 - Math.sinh(x) ** 2,
    predicate: (x) => {
      const r = runOf('A075')(x);
      if (!rc(r, 1, 1e-6)) return false;                       // run 耦合：朴素式恒等式（有据容差）
      const c = Math.cosh(x), s = Math.sinh(x);
      return rc(c, Math.sqrt(s * s + 1), 1e-12);               // 稳定式 oracle（紧容差）
    },
  });
  reg(76, {
    name: '双曲 · sinh 定义式', intuition: 'sinh x = (e^x − e^{−x})/2（允许浮点误差）。',
    math: 'sinh x = (e^x−e^{−x})/2', pseudocode: 'return Math.sinh(x)', complexity: 'O(1)',
    ref: '双曲函数定义', commonPitfall: '需容差。',
    invariant: 'relClose(run(x), (e^x−e^{−x})/2, 1e-9)（异构公式）',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => Math.sinh(x),
    predicate: (x) => rc(runOf('A076')(x), (Math.exp(x) - Math.exp(-x)) / 2, 1e-9),
  });
  reg(77, {
    name: '双曲 · cosh 定义式', intuition: 'cosh x = (e^x + e^{−x})/2（允许浮点误差）。',
    math: 'cosh x = (e^x+e^{−x})/2', pseudocode: 'return Math.cosh(x)', complexity: 'O(1)',
    ref: '双曲函数定义', commonPitfall: '需容差。',
    invariant: 'relClose(run(x), (e^x+e^{−x})/2, 1e-9)（异构公式）',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => Math.cosh(x),
    predicate: (x) => rc(runOf('A077')(x), (Math.exp(x) + Math.exp(-x)) / 2, 1e-9),
  });
  reg(78, {
    name: '双曲 · tanh 奇函数', intuition: 'tanh(-x) = -tanh(x)（允许浮点误差）。',
    math: 'tanh(-x) = -tanh x', pseudocode: 'return Math.tanh(-x)', complexity: 'O(1)',
    ref: '双曲函数奇偶性', commonPitfall: '需容差。',
    invariant: 'relClose(run(x), -tanh x, 1e-9)（异构公式）',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => Math.tanh(-x),
    predicate: (x) => rc(runOf('A078')(x), -Math.tanh(x), 1e-9),
  });
  reg(79, {
    name: '双曲 · tanh 有界', intuition: '|tanh x| < 1（有限 x）。',
    math: '|tanh x| < 1', pseudocode: 'return Math.tanh(x)', complexity: 'O(1)',
    ref: '双曲函数有界性', commonPitfall: '无',
    invariant: '|run(x)| < 1，且定点锚 Object.is(run(0),0) 确系 tanh',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => Math.tanh(x),
    predicate: (x) => { const run = runOf('A079'); return Math.abs(run(x)) < 1 && Object.is(run(0), 0); },
  });
  reg(80, {
    name: '双曲 · tanh(0)=0', intuition: 'tanh 0 逐位为 0。',
    math: 'tanh 0 = 0', pseudocode: 'return Math.tanh(0)', complexity: 'O(1)',
    ref: '双曲函数', commonPitfall: '无', invariant: 'Object.is(run(), 0)（零参定点，run 耦合）',
    inputs: [], domains: {},
    run: () => Math.tanh(0),
    predicate: () => Object.is(runOf('A080')(), 0),
  });

  // ── 激活函数（A081–A090，复用 L9.special）────────────────────────────
  reg(81, {
    name: '激活 · ReLU 幂等', intuition: 'ReLU 对自身幂等（逐位）。',
    math: 'ReLU(ReLU(a)) = ReLU(a)', pseudocode: 'return SP.relu(SP.relu(a))', complexity: 'O(1)',
    ref: 'Nair & Hinton 2010', commonPitfall: '无',
    invariant: 'Object.is(run(run(a)), run(a))（幂等经 run 自复合）；锚 run(-5)=0、run(3)=3',
    inputs: ['a'], domains: { a: 'finite' },
    run: (a) => SP.relu(SP.relu(a)),
    predicate: (a) => {
      const run = runOf('A081'), r = run(a);
      return Object.is(run(r), r) && Object.is(run(-5), 0) && Object.is(run(3), 3);
    },
  });
  reg(82, {
    name: '激活 · ReLU 非负', intuition: 'ReLU 输出恒非负。',
    math: 'ReLU(a) ≥ 0', pseudocode: 'return SP.relu(a)', complexity: 'O(1)',
    ref: 'Nair & Hinton 2010', commonPitfall: '无',
    invariant: 'run(a) ≥ 0；锚 run(-5)=0、run(3)=3 确系 ReLU',
    inputs: ['a'], domains: { a: 'finite' },
    run: (a) => SP.relu(a),
    predicate: (a) => {
      const run = runOf('A082');
      return run(a) >= 0 && Object.is(run(-5), 0) && Object.is(run(3), 3);
    },
  });
  reg(83, {
    name: '激活 · Sigmoid 区间', intuition: 'Sigmoid 在适中输入下输出严格落在 (0,1)（|a|≤10 内不会饱和到精确 0/1）。',
    math: '0 < σ(a) < 1', pseudocode: 'return SP.sigmoid(a)', complexity: 'O(1)',
    ref: 'Goodfellow 2016', commonPitfall: '|a| 过大会浮点饱和到精确 0/1，故域限定 small（±10）。',
    invariant: '对 |a|≤10：0 < run(a) < 1；锚 Object.is(run(0), 0.5)',
    inputs: ['a'], domains: { a: 'small' },
    run: (a) => SP.sigmoid(a),
    predicate: (a) => { const s = runOf('A083')(a); return s > 0 && s < 1 && Object.is(runOf('A083')(0), 0.5); },
  });
  reg(84, {
    name: '激活 · Sigmoid 对称', intuition: 'σ(−a) = 1 − σ(a)（允许浮点误差）。',
    math: 'σ(−a) = 1 − σ(a)', pseudocode: 'return SP.sigmoid(-a)', complexity: 'O(1)',
    ref: 'Goodfellow 2016', commonPitfall: '需容差；大 |a| 双侧浮点饱和时仍逐位一致。',
    invariant: 'relClose(run(a), 1−σ(a), 1e-9)（异构公式）',
    inputs: ['a'], domains: { a: 'finite' },
    run: (a) => SP.sigmoid(-a),
    predicate: (a) => rc(runOf('A084')(a), 1 - SP.sigmoid(a), 1e-9),
  });
  reg(85, {
    name: '激活 · Sigmoid(0)=½', intuition: 'σ(0) 逐位为 0.5。',
    math: 'σ(0) = 0.5', pseudocode: 'return SP.sigmoid(0)', complexity: 'O(1)',
    ref: 'Goodfellow 2016', commonPitfall: '无', invariant: 'Object.is(run(), 0.5)（零参定点，run 耦合）',
    inputs: [], domains: {},
    run: () => SP.sigmoid(0),
    predicate: () => Object.is(runOf('A085')(), 0.5),
  });
  reg(86, {
    name: '激活 · Softplus 非负', intuition: 'Softplus 为 ReLU 平滑近似，恒非负。',
    math: 'softplus(a) ≥ 0', pseudocode: 'return SP.softplus(a)', complexity: 'O(1)',
    ref: 'Dugas et al. 2001', commonPitfall: '无',
    invariant: 'run(a) ≥ 0；锚 Object.is(run(0), ln 2) 确系 softplus',
    inputs: ['a'], domains: { a: 'finite' },
    run: (a) => SP.softplus(a),
    predicate: (a) => { const run = runOf('A086'); return run(a) >= 0 && Object.is(run(0), Math.log(2)); },
  });
  reg(87, {
    name: '激活 · Softplus(0)=ln2', intuition: 'softplus(0) 逐位为 ln2。',
    math: 'softplus(0) = ln 2', pseudocode: 'return SP.softplus(0)', complexity: 'O(1)',
    ref: 'Dugas et al. 2001', commonPitfall: '无', invariant: 'Object.is(run(), ln 2)（零参定点，run 耦合）',
    inputs: [], domains: {},
    run: () => SP.softplus(0),
    predicate: () => Object.is(runOf('A087')(), Math.log(2)),
  });
  reg(88, {
    name: '激活 · Swish(0)=0', intuition: 'Swish(0) 逐位为 0。',
    math: 'swish(0) = 0', pseudocode: 'return SP.swish(0)', complexity: 'O(1)',
    ref: 'Ramachandran 2017', commonPitfall: '无', invariant: 'Object.is(run(), 0)（零参定点，run 耦合）',
    inputs: [], domains: {},
    run: () => SP.swish(0),
    predicate: () => Object.is(runOf('A088')(), 0),
  });
  reg(89, {
    name: '激活 · ELU(0)=0', intuition: 'ELU(0) 逐位为 0。',
    math: 'elu(0) = 0', pseudocode: 'return SP.elu(0)', complexity: 'O(1)',
    ref: 'Clevert 2015', commonPitfall: '无', invariant: 'Object.is(run(), 0)（零参定点，run 耦合）',
    inputs: [], domains: {},
    run: () => SP.elu(0),
    predicate: () => Object.is(runOf('A089')(), 0),
  });
  reg(90, {
    name: '激活 · SELU(0)=0', intuition: 'SELU(0) 逐位为 0。',
    math: 'selu(0) = 0', pseudocode: 'return SP.selu(0)', complexity: 'O(1)',
    ref: 'Klambauer 2017', commonPitfall: '无', invariant: 'Object.is(run(), 0)（零参定点，run 耦合）',
    inputs: [], domains: {},
    run: () => SP.selu(0),
    predicate: () => Object.is(runOf('A090')(), 0),
  });

  // ── 取整 / 稳定性（A091–A096）────────────────────────────────────────
  reg(91, {
    name: '取整 · floor 幂等', intuition: 'floor 对自身幂等（逐位）。',
    math: '⌊⌊a⌋⌋ = ⌊a⌋', pseudocode: 'return Math.floor(Math.floor(a))', complexity: 'O(1)',
    ref: 'IEEE 754', commonPitfall: '无',
    invariant: 'Object.is(run(run(a)), run(a))（幂等经 run 自复合）',
    inputs: ['a'], domains: { a: 'finite' },
    run: (a) => Math.floor(Math.floor(a)),
    predicate: (a) => { const run = runOf('A091'), r = run(a); return Object.is(run(r), r); },
  });
  reg(92, {
    name: '取整 · ceil 幂等', intuition: 'ceil 对自身幂等（逐位）。',
    math: '⌈⌈a⌉⌉ = ⌈a⌉', pseudocode: 'return Math.ceil(Math.ceil(a))', complexity: 'O(1)',
    ref: 'IEEE 754', commonPitfall: '无',
    invariant: 'Object.is(run(run(a)), run(a))（幂等经 run 自复合）',
    inputs: ['a'], domains: { a: 'finite' },
    run: (a) => Math.ceil(Math.ceil(a)),
    predicate: (a) => { const run = runOf('A092'), r = run(a); return Object.is(run(r), r); },
  });
  reg(93, {
    name: '取整 · trunc 幂等', intuition: 'trunc 对自身幂等（逐位）。',
    math: 'trunc(trunc(a)) = trunc(a)', pseudocode: 'return Math.trunc(Math.trunc(a))', complexity: 'O(1)',
    ref: 'IEEE 754', commonPitfall: '无',
    invariant: 'Object.is(run(run(a)), run(a))（幂等经 run 自复合）',
    inputs: ['a'], domains: { a: 'finite' },
    run: (a) => Math.trunc(Math.trunc(a)),
    predicate: (a) => { const run = runOf('A093'), r = run(a); return Object.is(run(r), r); },
  });
  reg(94, {
    name: '取整 · floor ≤ x ≤ ceil', intuition: 'floor 与 ceil 夹住原数（逐位比较）。',
    math: '⌊a⌋ ≤ a ≤ ⌈a⌉', pseudocode: 'return Math.floor(a)', complexity: 'O(1)',
    ref: 'IEEE 754', commonPitfall: '无',
    invariant: 'run(a) ≤ a ≤ run(a)+1（ceil(a) ≤ floor(a)+1，两侧均 run 耦合）',
    inputs: ['a'], domains: { a: 'finite' },
    run: (a) => Math.floor(a),
    predicate: (a) => { const f = runOf('A094')(a); return f <= a && a <= f + 1; },
  });
  reg(95, {
    name: '取整 · round 输出整数', intuition: 'round 的结果必为整数。',
    math: 'isInteger(round(a))', pseudocode: 'return Math.round(a)', complexity: 'O(1)',
    ref: 'IEEE 754', commonPitfall: '半值舍去方向不同于 floor(a+0.5)，故只验证整数性；锚 round(2.5)=3、round(-2.5)=-2 钉住 round 语义。',
    invariant: 'Number.isInteger(run(a))；锚 Object.is(run(2.5),3)、Object.is(run(-2.5),-2)',
    inputs: ['a'], domains: { a: 'finite' },
    run: (a) => Math.round(a),
    predicate: (a) => {
      const run = runOf('A095');
      return Number.isInteger(run(a)) && Object.is(run(2.5), 3) && Object.is(run(-2.5), -2);
    },
  });
  reg(96, {
    name: '取整 · round 整数不动', intuition: '整数经 round 后逐位不变。',
    math: 'round(a) = a (a∈ℤ)', pseudocode: 'return Math.round(a)', complexity: 'O(1)',
    ref: 'IEEE 754', commonPitfall: '无',
    invariant: '对整数 a：Object.is(run(a), a)；锚 round(2.5)=3 与 A095 共同钉住语义',
    inputs: ['a'], domains: { a: 'int' },
    run: (a) => Math.round(a),
    predicate: (a) => Object.is(runOf('A096')(a), a) && Object.is(runOf('A096')(2.5), 3),
  });
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

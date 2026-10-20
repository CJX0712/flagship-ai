/* ============================================================================
 * L10 微指令 · A 域（标量算术与初等函数）· 第 2/3 批（A033–A064）
 * ----------------------------------------------------------------------------
 * 涵盖：绝对值/双曲根（A033–A040）、对数指数（A041–A056）、三角函数（A057–A064）。
 * 七件套 + run + predicate，判据同第 1 批（精确 Object.is / 容差 relClose 1e-9）。
 * ==========================================================================*/
;(function (NS) {
  'use strict';
  NS.L10 = NS.L10 || {};
  function reg(i, o) { NS.L10.register('A', i, o); }
  const rc = (a, b, t) => NS.L10.relClose(a, b, t);

  // ── 绝对值 / 双曲根（A033–A040）───────────────────────────────────────
  reg(33, {
    name: '标量绝对值 · 非负且对称', intuition: '绝对值非负，且对取反不变。',
    math: '|a| ≥ 0 ∧ |-a| = |a|', pseudocode: 'return Math.abs(a)', complexity: 'O(1)',
    ref: 'IEEE 754', commonPitfall: '无', invariant: '|a|≥0 且 Object.is(|-a|,|a|)',
    inputs: ['a'], domains: { a: 'finite' },
    run: (a) => Math.abs(a), predicate: (a) => Math.abs(a) >= 0 && Object.is(Math.abs(-a), Math.abs(a)),
  });
  reg(34, {
    name: '标量绝对值 · 取反对称', intuition: '绝对值对取反逐位对称。',
    math: '|-a| = |a|', pseudocode: 'return Math.abs(-a)', complexity: 'O(1)',
    ref: 'IEEE 754', commonPitfall: '无', invariant: 'Object.is(|-a|, |a|)',
    inputs: ['a'], domains: { a: 'finite' },
    run: (a) => Math.abs(-a), predicate: (a) => Object.is(Math.abs(-a), Math.abs(a)),
  });
  reg(35, {
    name: '标量立方根 · 立方还原', intuition: '∛(a³) 回到 a（允许浮点误差）。',
    math: '∛(a³) ≈ a', pseudocode: 'return Math.cbrt(a*a*a)', complexity: 'O(1)',
    ref: 'IEEE 754 cbrt', commonPitfall: 'cbrt 为奇次根，对负数亦成立。',
    invariant: '对 |a|≤10：relClose(∛(a³), a, 1e-9)',
    inputs: ['a'], domains: { a: 'small' },
    run: (a) => Math.cbrt(a * a * a), predicate: (a) => rc(Math.cbrt(a * a * a), a, 1e-9),
  });
  reg(36, {
    name: '标量 hypot · 非负', intuition: '√(a²+b²) 恒非负。',
    math: 'hypot(a,b) ≥ 0', pseudocode: 'return Math.hypot(a,b)', complexity: 'O(1)',
    ref: 'IEEE 754 hypot', commonPitfall: '避免中间溢出，故用 hypot 而非 sqrt。',
    invariant: 'Math.hypot(a,b) ≥ 0',
    inputs: ['a', 'b'], domains: { a: 'finite', b: 'finite' },
    run: (a, b) => Math.hypot(a, b), predicate: (a, b) => Math.hypot(a, b) >= 0,
  });
  reg(37, {
    name: '标量 hypot · 齐次缩放', intuition: '向量整体缩放 k 倍，模长缩放 |k| 倍。',
    math: 'hypot(ka,kb) = |k|·hypot(a,b)', pseudocode: 'return Math.hypot(2*a,2*b)', complexity: 'O(1)',
    ref: '范数齐次性', commonPitfall: '需容差。',
    invariant: '对 |a|,|b|≤10：relClose(hypot(2a,2b), 2·hypot(a,b), 1e-9)',
    inputs: ['a', 'b'], domains: { a: 'small', b: 'small' },
    run: (a, b) => Math.hypot(2 * a, 2 * b), predicate: (a, b) => rc(Math.hypot(2 * a, 2 * b), 2 * Math.hypot(a, b), 1e-9),
  });
  reg(38, {
    name: '标量 hypot · 与 √(a²+b²) 等价', intuition: 'hypot 是 √(a²+b²) 的稳定实现。',
    math: 'hypot(a,b) ≈ √(a²+b²)', pseudocode: 'return Math.hypot(a,b)', complexity: 'O(1)',
    ref: 'Goldberg 1991', commonPitfall: '需容差。',
    invariant: 'relClose(hypot(a,b), √(a²+b²), 1e-9)',
    inputs: ['a', 'b'], domains: { a: 'small', b: 'small' },
    run: (a, b) => Math.hypot(a, b), predicate: (a, b) => rc(Math.hypot(a, b), Math.sqrt(a * a + b * b), 1e-9),
  });
  reg(39, {
    name: '标量绝对值 · 乘积可分', intuition: '乘积的绝对值等于绝对值的乘积（逐位）。',
    math: '|a·b| = |a|·|b|', pseudocode: 'return Math.abs(a*b)', complexity: 'O(1)',
    ref: 'IEEE 754', commonPitfall: '无', invariant: 'Object.is(|a·b|, |a|·|b|)',
    inputs: ['a', 'b'], domains: { a: 'finite', b: 'finite' },
    run: (a, b) => Math.abs(a * b), predicate: (a, b) => Object.is(Math.abs(a * b), Math.abs(a) * Math.abs(b)),
  });
  reg(40, {
    name: '标量平方 · 与绝对值平方一致', intuition: 'a² = |a|² 逐位成立。',
    math: 'a² = |a|²', pseudocode: 'return a*a', complexity: 'O(1)',
    ref: '代数', commonPitfall: '无', invariant: 'Object.is(a², |a|²)',
    inputs: ['a'], domains: { a: 'finite' },
    run: (a) => a * a, predicate: (a) => Object.is(a * a, Math.abs(a) * Math.abs(a)),
  });

  // ── 对数 / 指数（A041–A056）───────────────────────────────────────────
  reg(41, {
    name: '标量指数 · 对数还原', intuition: 'exp(ln a) 回到 a（允许浮点误差）。',
    math: 'exp(ln a) ≈ a', pseudocode: 'return Math.exp(Math.log(a))', complexity: 'O(1)',
    ref: '初等函数', commonPitfall: 'a 须 >0。',
    invariant: '对 a>0：relClose(exp(ln a), a, 1e-9)',
    inputs: ['a'], domains: { a: 'pos' },
    run: (a) => Math.exp(Math.log(a)), predicate: (a) => rc(Math.exp(Math.log(a)), a, 1e-9),
  });
  reg(42, {
    name: '标量对数 · 指数还原', intuition: 'ln(exp a) 回到 a（允许浮点误差）。',
    math: 'ln(exp a) ≈ a', pseudocode: 'return Math.log(Math.exp(a))', complexity: 'O(1)',
    ref: '初等函数', commonPitfall: 'a 不宜过大，避免 exp 溢出。',
    invariant: '对 |a|≤10：relClose(ln(exp a), a, 1e-9)',
    inputs: ['a'], domains: { a: 'small' },
    run: (a) => Math.log(Math.exp(a)), predicate: (a) => rc(Math.log(Math.exp(a)), a, 1e-9),
  });
  reg(43, {
    name: '标量指数 · 加法转乘法', intuition: 'exp(a+b) = exp(a)·exp(b)（允许浮点误差）。',
    math: 'exp(a+b) ≈ exp(a)·exp(b)', pseudocode: 'return Math.exp(a+b)', complexity: 'O(1)',
    ref: '指数恒等式', commonPitfall: '需容差。',
    invariant: 'relClose(exp(a+b), exp(a)*exp(b), 1e-9)',
    inputs: ['a', 'b'], domains: { a: 'small', b: 'small' },
    run: (a, b) => Math.exp(a + b), predicate: (a, b) => rc(Math.exp(a + b), Math.exp(a) * Math.exp(b), 1e-9),
  });
  reg(44, {
    name: '标量指数 · 零次幂为 1', intuition: 'exp(0) 逐位为 1。',
    math: 'exp(0) = 1', pseudocode: 'return Math.exp(0)', complexity: 'O(1)',
    ref: 'IEEE 754', commonPitfall: '无', invariant: 'Object.is(exp(0), 1)',
    inputs: [], domains: {},
    run: () => Math.exp(0), predicate: () => Object.is(Math.exp(0), 1),
  });
  reg(45, {
    name: '标量指数 · 负指数倒数', intuition: 'exp(-a) = 1/exp(a)（允许浮点误差）。',
    math: 'exp(-a) ≈ 1/exp(a)', pseudocode: 'return Math.exp(-a)', complexity: 'O(1)',
    ref: '指数恒等式', commonPitfall: '需容差。',
    invariant: 'relClose(exp(-a), 1/exp(a), 1e-9)',
    inputs: ['a'], domains: { a: 'small' },
    run: (a) => Math.exp(-a), predicate: (a) => rc(Math.exp(-a), 1 / Math.exp(a), 1e-9),
  });
  reg(46, {
    name: '标量对数 · 乘积转求和', intuition: 'ln(a·b) = ln a + ln b（允许浮点误差）。',
    math: 'ln(ab) ≈ ln a + ln b', pseudocode: 'return Math.log(a*b)', complexity: 'O(1)',
    ref: '对数恒等式', commonPitfall: 'a,b 须 >0；需容差。',
    invariant: '对 a,b>0：relClose(ln(ab), ln a+ln b, 1e-9)',
    inputs: ['a', 'b'], domains: { a: 'pos', b: 'pos' },
    run: (a, b) => Math.log(a * b), predicate: (a, b) => rc(Math.log(a * b), Math.log(a) + Math.log(b), 1e-9),
  });
  reg(47, {
    name: '标量对数 · 商转差', intuition: 'ln(a/b) = ln a − ln b（允许浮点误差）。',
    math: 'ln(a/b) ≈ ln a − ln b', pseudocode: 'return Math.log(a/b)', complexity: 'O(1)',
    ref: '对数恒等式', commonPitfall: '需容差。',
    invariant: '对 a,b>0：relClose(ln(a/b), ln a−ln b, 1e-9)',
    inputs: ['a', 'b'], domains: { a: 'pos', b: 'pos' },
    run: (a, b) => Math.log(a / b), predicate: (a, b) => rc(Math.log(a / b), Math.log(a) - Math.log(b), 1e-9),
  });
  reg(48, {
    name: '标量对数 · log1p 稳定性', intuition: 'log1p(x) 在 |x| 很小时比 log(1+x) 更准（仍近似一致）。',
    math: 'log1p(x) ≈ log(1+x)', pseudocode: 'return Math.log1p(x)', complexity: 'O(1)',
    ref: 'Cody & Waite 1980', commonPitfall: '需容差。',
    invariant: '对 |x|≤10：relClose(log1p(x), log(1+x), 1e-9)',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => Math.log1p(x), predicate: (x) => rc(Math.log1p(x), Math.log(1 + x), 1e-9),
  });
  reg(49, {
    name: '标量指数 · expm1 稳定性', intuition: 'expm1(x) 在 |x| 很小时比 exp(x)−1 更准（仍近似一致）。',
    math: 'expm1(x) ≈ exp(x)−1', pseudocode: 'return Math.expm1(x)', complexity: 'O(1)',
    ref: 'Cody & Waite 1980', commonPitfall: '需容差。',
    invariant: 'relClose(expm1(x), exp(x)−1, 1e-9)',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => Math.expm1(x), predicate: (x) => rc(Math.expm1(x), Math.exp(x) - 1, 1e-9),
  });
  reg(50, {
    name: '标量指数 · 正负乘积为 1', intuition: 'exp(a)·exp(-a) ≈ 1（允许浮点误差）。',
    math: 'exp(a)·exp(-a) ≈ 1', pseudocode: 'return Math.exp(a)*Math.exp(-a)', complexity: 'O(1)',
    ref: '指数恒等式', commonPitfall: '需容差。',
    invariant: 'relClose(exp(a)·exp(-a), 1, 1e-9)',
    inputs: ['a'], domains: { a: 'small' },
    run: (a) => Math.exp(a) * Math.exp(-a), predicate: (a) => rc(Math.exp(a) * Math.exp(-a), 1, 1e-9),
  });
  reg(51, {
    name: '标量指数 · 幂与指数关系', intuition: 'exp(a)^b = exp(a·b)（允许浮点误差）。',
    math: 'exp(a)^b ≈ exp(a·b)', pseudocode: 'return Math.pow(Math.exp(a),b)', complexity: 'O(1)',
    ref: '对数恒等式', commonPitfall: '需容差；a 不宜过大。',
    invariant: 'relClose(pow(exp a, b), exp(a·b), 1e-9)',
    inputs: ['a', 'b'], domains: { a: 'small', b: 'small' },
    run: (a, b) => Math.pow(Math.exp(a), b), predicate: (a, b) => rc(Math.pow(Math.exp(a), b), Math.exp(a * b), 1e-9),
  });
  reg(52, {
    name: '标量对数 · 单位', intuition: 'ln 1 逐位为 0。',
    math: 'ln 1 = 0', pseudocode: 'return Math.log(1)', complexity: 'O(1)',
    ref: '对数定义', commonPitfall: '无', invariant: 'Object.is(log(1), 0)',
    inputs: [], domains: {},
    run: () => Math.log(1), predicate: () => Object.is(Math.log(1), 0),
  });
  reg(53, {
    name: '标量指数 · 2^a 与 exp(a·ln2)', intuition: '2^a = exp(a·ln2)（允许浮点误差）。',
    math: '2^a ≈ exp(a·ln2)', pseudocode: 'return Math.pow(2,a)', complexity: 'O(1)',
    ref: '换底公式', commonPitfall: '需容差。',
    invariant: 'relClose(2^a, exp(a·ln2), 1e-9)',
    inputs: ['a'], domains: { a: 'small' },
    run: (a) => Math.pow(2, a), predicate: (a) => rc(Math.pow(2, a), Math.exp(a * Math.LN2), 1e-9),
  });
  reg(54, {
    name: '标量对数 · log10(exp a) 与 a/ln10', intuition: 'log10(exp a) = a/ln10（允许浮点误差）。',
    math: 'log10(exp a) ≈ a/ln10', pseudocode: 'return Math.log10(Math.exp(a))', complexity: 'O(1)',
    ref: '换底公式', commonPitfall: '需容差。',
    invariant: 'relClose(log10(exp a), a/ln10, 1e-9)',
    inputs: ['a'], domains: { a: 'small' },
    run: (a) => Math.log10(Math.exp(a)), predicate: (a) => rc(Math.log10(Math.exp(a)), a / Math.LN10, 1e-9),
  });
  reg(55, {
    name: '标量指数 · 单调非减', intuition: 'exp 在实数上单调非减（顺序成立时）。',
    math: 'a ≤ b ⇒ exp a ≤ exp b', pseudocode: 'return Math.exp(a)', complexity: 'O(1)',
    ref: '指数函数性质', commonPitfall: '判据用蕴含式，恒真。',
    invariant: '∀a,b：(a≤b) ⇒ exp a ≤ exp b',
    inputs: ['a', 'b'], domains: { a: 'small', b: 'small' },
    run: (a) => Math.exp(a), predicate: (a, b) => (a <= b) ? Math.exp(a) <= Math.exp(b) : true,
  });
  reg(56, {
    name: '标量对数 · 单调非减', intuition: 'ln 在正轴上单调非减（顺序成立时）。',
    math: '0<a≤b ⇒ ln a ≤ ln b', pseudocode: 'return Math.log(a)', complexity: 'O(1)',
    ref: '对数函数性质', commonPitfall: '判据用蕴含式，恒真。',
    invariant: '∀a,b>0：(a≤b) ⇒ ln a ≤ ln b',
    inputs: ['a', 'b'], domains: { a: 'pos', b: 'pos' },
    run: (a) => Math.log(a), predicate: (a, b) => (a <= b) ? Math.log(a) <= Math.log(b) : true,
  });

  // ── 三角函数（A057–A064）──────────────────────────────────────────────
  reg(57, {
    name: '三角 · 勾股恒等式', intuition: 'sin²x + cos²x = 1（允许浮点误差）。',
    math: 'sin²x + cos²x = 1', pseudocode: 'return Math.sin(x)**2+Math.cos(x)**2', complexity: 'O(1)',
    ref: 'Euler 公式', commonPitfall: '需容差。',
    invariant: 'relClose(sin²x+cos²x, 1, 1e-9)',
    inputs: ['x'], domains: { x: 'angle' },
    run: (x) => Math.sin(x) ** 2 + Math.cos(x) ** 2, predicate: (x) => rc(Math.sin(x) ** 2 + Math.cos(x) ** 2, 1, 1e-9),
  });
  reg(58, {
    name: '三角 · sin 奇函数', intuition: 'sin(-x) = -sin(x)（允许浮点误差）。',
    math: 'sin(-x) = -sin x', pseudocode: 'return Math.sin(-x)', complexity: 'O(1)',
    ref: '三角函数奇偶性', commonPitfall: '需容差。',
    invariant: 'relClose(sin(-x), -sin x, 1e-9)',
    inputs: ['x'], domains: { x: 'angle' },
    run: (x) => Math.sin(-x), predicate: (x) => rc(Math.sin(-x), -Math.sin(x), 1e-9),
  });
  reg(59, {
    name: '三角 · cos 偶函数', intuition: 'cos(-x) = cos(x)（允许浮点误差）。',
    math: 'cos(-x) = cos x', pseudocode: 'return Math.cos(-x)', complexity: 'O(1)',
    ref: '三角函数奇偶性', commonPitfall: '需容差。',
    invariant: 'relClose(cos(-x), cos x, 1e-9)',
    inputs: ['x'], domains: { x: 'angle' },
    run: (x) => Math.cos(-x), predicate: (x) => rc(Math.cos(-x), Math.cos(x), 1e-9),
  });
  reg(60, {
    name: '三角 · sin(0)=0', intuition: 'sin 0 逐位为 0。',
    math: 'sin 0 = 0', pseudocode: 'return Math.sin(0)', complexity: 'O(1)',
    ref: '三角函数', commonPitfall: '无', invariant: 'Object.is(sin 0, 0)',
    inputs: [], domains: {},
    run: () => Math.sin(0), predicate: () => Object.is(Math.sin(0), 0),
  });
  reg(61, {
    name: '三角 · cos(0)=1', intuition: 'cos 0 逐位为 1。',
    math: 'cos 0 = 1', pseudocode: 'return Math.cos(0)', complexity: 'O(1)',
    ref: '三角函数', commonPitfall: '无', invariant: 'Object.is(cos 0, 1)',
    inputs: [], domains: {},
    run: () => Math.cos(0), predicate: () => Object.is(Math.cos(0), 1),
  });
  reg(62, {
    name: '三角 · tan 定义式', intuition: 'tan x = sin x / cos x（|x|≤1.5，cos 远离 0）。',
    math: 'tan x = sin x / cos x', pseudocode: 'return Math.tan(x)', complexity: 'O(1)',
    ref: '三角函数定义', commonPitfall: 'x 须远离 π/2，故用 trig 域。',
    invariant: 'relClose(tan x, sin x/cos x, 1e-9)',
    inputs: ['x'], domains: { x: 'trig' },
    run: (x) => Math.tan(x), predicate: (x) => rc(Math.tan(x), Math.sin(x) / Math.cos(x), 1e-9),
  });
  reg(63, {
    name: '三角 · asin(sin x)=x', intuition: '反正弦还原（x∈[-1.5,1.5]⊂[-π/2,π/2]）。',
    math: 'asin(sin x) = x', pseudocode: 'return Math.asin(Math.sin(x))', complexity: 'O(1)',
    ref: '反三角函数', commonPitfall: '需容差。',
    invariant: 'relClose(asin(sin x), x, 1e-9)',
    inputs: ['x'], domains: { x: 'trig' },
    run: (x) => Math.asin(Math.sin(x)), predicate: (x) => rc(Math.asin(Math.sin(x)), x, 1e-9),
  });
  reg(64, {
    name: '三角 · 倍角 sin2x', intuition: 'sin(2x) = 2 sin x cos x（允许浮点误差）。',
    math: 'sin 2x = 2 sin x cos x', pseudocode: 'return Math.sin(2*x)', complexity: 'O(1)',
    ref: '三角倍角公式', commonPitfall: '需容差。',
    invariant: 'relClose(sin 2x, 2 sin x cos x, 1e-9)',
    inputs: ['x'], domains: { x: 'trig' },
    run: (x) => Math.sin(2 * x), predicate: (x) => rc(Math.sin(2 * x), 2 * Math.sin(x) * Math.cos(x), 1e-9),
  });
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

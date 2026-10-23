/* ============================================================================
 * L10 微指令 · A 域（标量算术与初等函数）· 第 1/4 批（A001–A032）
 * ----------------------------------------------------------------------------
 * 所属层：L10-microIsa  域：A（标量）
 * 允许依赖的层：L9（rng/assert/special）、L10（registry/invariant）
 * 主要 @ref：IEEE 754-2019；Goldberg (1991)；Abramowitz & Stegun (1964)
 *
 * 每项含七件套（直觉/数学/伪代码/复杂度/论文/常见坑/不变量）+ run + predicate。
 * 不变量判据：精确类用 Object.is（逐位），数值类用 relClose(tol=1e-9)。
 * 输入域由 L10.TAGS 约束（tests/L10 据此构造 fast-check 任意子）。
 *
 * ── 谓词纪律（本批全部重写，2026-10-23）─────────────────────────────────
 * 门禁只调 op.predicate，从不调 op.run（op-harness.js:43、probe-microisa.js:106）。
 * 原 32 项谓词全部内联复算 run 定义式或直接自比输入，run 若损坏门禁完全失明
 * （假绿）。现全部改为：先经 runOf('Axxx') 调 run 本体，再对 run 的产出断言
 * 不变量；oracle 一律取异构公式（不同结合分组/不同运算路径）、输入导出值
 * 或定点锚（构造确定输入喂 run），绝不复写 run 定义式本身。
 * ==========================================================================*/
;(function (NS) {
  'use strict';
  NS.L10 = NS.L10 || {};
  const special = NS.L9.special;
  function reg(i, o) { NS.L10.register('A', i, o); }
  function runOf(key) { const o = NS.L10.ops[key]; return o && typeof o.run === 'function' ? o.run : null; }

  // ── 加减乘（A001–A016）───────────────────────────────────────────────
  reg(1, {
    name: '标量加法 · 加性交换', intuition: '浮点加法逐位交换，是归约不变量的基础。',
    math: 'a + b = b + a', pseudocode: 'return a + b', complexity: 'O(1)',
    ref: 'IEEE 754-2019 §5.1', commonPitfall: '不要与结合律混淆——浮点加法不可结合。',
    invariant: 'run 对任意有限 a,b 满足 Object.is(run(a,b), run(b,a))，且定点锚确系加法',
    inputs: ['a', 'b'], domains: { a: 'finite', b: 'finite' },
    run: (a, b) => a + b,
    predicate: (a, b) => {
      const run = runOf('A001');
      if (!Object.is(run(a, b), run(b, a))) return false;      // 交换律（run 耦合）
      // 定点锚：排除 max/min/mul 等同样交换的运算
      return Object.is(run(2, 3), 5) && Object.is(run(-2, 3), 1) && Object.is(run(0, 7), 7);
    },
  });
  reg(2, {
    name: '标量加法 · 结合（整数域）', intuition: '整数域下加法精确可结合，用于精确累加证明。',
    math: '(a + b) + c = a + (b + c)', pseudocode: 'return (a+b)+c', complexity: 'O(1)',
    ref: 'Goldberg 1991（整数运算精确）', commonPitfall: '有限双精度域下一般不可结合，故限定整数。',
    invariant: '对整数 a,b,c：Object.is(run(a,b,c), a+(b+c))（异构结合分组）',
    inputs: ['a', 'b', 'c'], domains: { a: 'int', b: 'int', c: 'int' },
    run: (a, b, c) => (a + b) + c,
    predicate: (a, b, c) => Object.is(runOf('A002')(a, b, c), a + (b + c)),
  });
  reg(3, {
    name: '标量加法 · 零元', intuition: '0 是加法的单位元（排除 -0 避免符号歧义）。',
    math: 'a + 0 = a', pseudocode: 'return a + 0', complexity: 'O(1)',
    ref: '域公理', commonPitfall: '-0 + (-0) 仍为 -0；此处限定 a≠0。',
    invariant: '对 a≠0：Object.is(run(a), a)，且定点锚排除非单位变换',
    inputs: ['a'], domains: { a: 'nonZero' },
    run: (a) => a + 0,
    predicate: (a) => {
      const run = runOf('A003');
      if (!Object.is(run(a), a)) return false;
      return Object.is(run(5), 5) && Object.is(run(-3), -3) && Object.is(run(0.25), 0.25);
    },
  });
  reg(4, {
    name: '标量减法 · 自减为零', intuition: '任意数减自身为精确 +0。',
    math: 'a - a = +0', pseudocode: 'return a - a', complexity: 'O(1)',
    ref: 'IEEE 754', commonPitfall: '结果为 +0 而非 -0（符号归正）。',
    invariant: '对所有有限 a：Object.is(run(a), 0)（+0，非 -0）',
    inputs: ['a'], domains: { a: 'finite' },
    run: (a) => a - a,
    predicate: (a) => Object.is(runOf('A004')(a), 0),
  });
  reg(5, {
    name: '标量取反 · 对合', intuition: '两次取反回到原数。',
    math: '-(-a) = a', pseudocode: 'return -a', complexity: 'O(1)',
    ref: 'IEEE 754', commonPitfall: '无', invariant: '对所有有限 a：Object.is(run(-a), a)',
    inputs: ['a'], domains: { a: 'finite' },
    run: (a) => -a,
    predicate: (a) => Object.is(runOf('A005')(-a), a),
  });
  reg(6, {
    name: '标量取反 · 分配的负号', intuition: '整数域下 -(a+b)=(-a)+(-b) 精确成立。',
    math: '-(a + b) = (-a) + (-b)', pseudocode: 'return -(a+b)', complexity: 'O(1)',
    ref: 'Goldberg 1991', commonPitfall: '有限域下一般不成立（舍入），故用整数。',
    invariant: '对整数：Object.is(run(a,b), (-a)+(-b))（异构公式）',
    inputs: ['a', 'b'], domains: { a: 'int', b: 'int' },
    run: (a, b) => -(a + b),
    predicate: (a, b) => Object.is(runOf('A006')(a, b), (-a) + (-b)),
  });
  reg(7, {
    name: '标量减法 · 加负等价', intuition: '减法即加其负，整数域下精确。',
    math: 'a - b = a + (-b)', pseudocode: 'return a - b', complexity: 'O(1)',
    ref: 'Goldberg 1991', commonPitfall: '有限域下舍入可能不同，故用整数。',
    invariant: '对整数：Object.is(run(a,b), a+(-b))（异构公式）',
    inputs: ['a', 'b'], domains: { a: 'int', b: 'int' },
    run: (a, b) => a - b,
    predicate: (a, b) => Object.is(runOf('A007')(a, b), a + (-b)),
  });
  reg(8, {
    name: '标量乘法 · 交换', intuition: '浮点乘法逐位交换。',
    math: 'a · b = b · a', pseudocode: 'return a * b', complexity: 'O(1)',
    ref: 'IEEE 754 §5.1', commonPitfall: 'NaN 与符号零需另行处理，故约束非 NaN。',
    invariant: '对所有有限 a,b：Object.is(run(a,b), run(b,a))，且定点锚确系乘法',
    inputs: ['a', 'b'], domains: { a: 'finite', b: 'finite' },
    run: (a, b) => a * b,
    predicate: (a, b) => {
      const run = runOf('A008');
      if (!Object.is(run(a, b), run(b, a))) return false;      // 交换律（run 耦合）
      // 定点锚：排除 add（run(2,3)=5≠6）/max/min
      return Object.is(run(2, 3), 6) && Object.is(run(0, 5), 0) && Object.is(run(-2, 3), -6);
    },
  });
  reg(9, {
    name: '标量乘法 · 结合（整数域）', intuition: '整数域下乘法精确可结合。',
    math: '(a · b) · c = a · (b · c)', pseudocode: 'return (a*b)*c', complexity: 'O(1)',
    ref: 'Goldberg 1991', commonPitfall: '有限域下不可结合，故限定整数。',
    invariant: '对整数：Object.is(run(a,b,c), a*(b*c))（异构结合分组）',
    inputs: ['a', 'b', 'c'], domains: { a: 'int', b: 'int', c: 'int' },
    run: (a, b, c) => (a * b) * c,
    predicate: (a, b, c) => Object.is(runOf('A009')(a, b, c), a * (b * c)),
  });
  reg(10, {
    name: '标量乘法 · 单位元', intuition: '1 是乘法的单位元（排除 -0）。',
    math: 'a · 1 = a', pseudocode: 'return a * 1', complexity: 'O(1)',
    ref: '域公理', commonPitfall: '-0 · 1 = -0 ≠ +0，故限定 a≠0。',
    invariant: '对 a≠0：Object.is(run(a), a)，且定点锚排除非单位变换',
    inputs: ['a'], domains: { a: 'nonZero' },
    run: (a) => a * 1,
    predicate: (a) => {
      const run = runOf('A010');
      if (!Object.is(run(a), a)) return false;
      return Object.is(run(5), 5) && Object.is(run(-4), -4) && Object.is(run(0.5), 0.5);
    },
  });
  reg(11, {
    name: '标量乘法 · 零元', intuition: '任意数乘 0 为精确 0（符号零：负 a 得 −0，仍等于 0）。',
    math: 'a · 0 = 0', pseudocode: 'return a * 0', complexity: 'O(1)',
    ref: 'IEEE 754', commonPitfall: '符号零：负 a·0 得 −0，须用 == 或 Object.is 容差判等。',
    invariant: '对所有有限 a：run(a) 为 0（−0 亦视为 0，=== 判等）',
    inputs: ['a'], domains: { a: 'finite' },
    run: (a) => a * 0,
    predicate: (a) => runOf('A011')(a) === 0,
  });
  reg(12, {
    name: '标量乘法 · 分配（整数域）', intuition: '乘法对加法可分配，整数域下精确。',
    math: 'a·(b + c) = a·b + a·c', pseudocode: 'return a*(b+c)', complexity: 'O(1)',
    ref: 'Goldberg 1991', commonPitfall: '有限域下舍入破坏等式，故用整数。',
    invariant: '对整数：Object.is(run(a,b,c), a*b+a*c)（异构展开）',
    inputs: ['a', 'b', 'c'], domains: { a: 'int', b: 'int', c: 'int' },
    run: (a, b, c) => a * (b + c),
    predicate: (a, b, c) => Object.is(runOf('A012')(a, b, c), a * b + a * c),
  });
  reg(13, {
    name: '标量除法 · 乘除抵消', intuition: '整数域内 (a·b)/b 精确还原 a（b≠0）。',
    math: '(a·b)/b = a', pseudocode: 'return (a*b)/b', complexity: 'O(1)',
    ref: '整数除法精确性', commonPitfall: 'a·b 须不溢出 2^53，故用 smallInt。',
    invariant: '对 smallInt, b≠0：Object.is(run(a,b), a)（oracle 为输入本身）',
    inputs: ['a', 'b'], domains: { a: 'smallInt', b: 'smallInt' },
    run: (a, b) => (b === 0 ? 0 : (a * b) / b),
    predicate: (a, b) => (b === 0 ? true : Object.is(runOf('A013')(a, b), a)),
  });
  reg(14, {
    name: '标量除法 · 除以 -1 即取反', intuition: '除以 -1 与取反逐位等价。',
    math: 'a / (-1) = -a', pseudocode: 'return a / (-1)', complexity: 'O(1)',
    ref: 'IEEE 754', commonPitfall: '无', invariant: '对所有有限 a：Object.is(run(a), -a)（异构公式）',
    inputs: ['a'], domains: { a: 'finite' },
    run: (a) => a / (-1),
    predicate: (a) => Object.is(runOf('A014')(a), -a),
  });
  reg(15, {
    name: '标量倒数 · 双倒数近似', intuition: '两次取倒数回到原数（允许浮点误差）。',
    math: '1/(1/a) ≈ a', pseudocode: 'return 1/(1/a)', complexity: 'O(1)',
    ref: '数值分析', commonPitfall: '非精确，需容差 1e-9；a 不可为 0。',
    invariant: '对 a≠0：relClose(run(a), a, 1e-9)（oracle 为输入本身）',
    inputs: ['a'], domains: { a: 'smallInt' },
    run: (a) => 1 / (1 / a),
    predicate: (a) => (a === 0 ? true : NS.L10.relClose(runOf('A015')(a), a, 1e-9)),
  });
  reg(16, {
    name: '标量除法 · 倒数乘积近似', intuition: 'a/b 等价于 a·(1/b)（允许浮点误差）。',
    math: 'a/b ≈ a·(1/b)', pseudocode: 'return a/b', complexity: 'O(1)',
    ref: '数值分析', commonPitfall: '两者舍入路径不同，需容差。',
    invariant: '对 b≠0：relClose(run(a,b), a*(1/b), 1e-9)（异构公式）',
    inputs: ['a', 'b'], domains: { a: 'smallInt', b: 'smallInt' },
    run: (a, b) => (b === 0 ? 0 : a / b),
    predicate: (a, b) => (b === 0 ? true : NS.L10.relClose(runOf('A016')(a, b), a * (1 / b), 1e-9)),
  });

  // ── 除与倒数（A017–A024）─────────────────────────────────────────────
  reg(17, {
    name: '标量除法 · 连除结合近似', intuition: '(a/b)/c 与 a/(b·c) 近似相等。',
    math: '(a/b)/c ≈ a/(b·c)', pseudocode: 'return (a/b)/c', complexity: 'O(1)',
    ref: '数值分析', commonPitfall: '需容差；b,c 不可为 0。',
    invariant: '对 b,c≠0：relClose(run(a,b,c), a/(b*c), 1e-9)（异构分组）',
    inputs: ['a', 'b', 'c'], domains: { a: 'smallInt', b: 'smallInt', c: 'smallInt' },
    run: (a, b, c) => ((b === 0 || c === 0) ? 0 : (a / b) / c),
    predicate: (a, b, c) => (b === 0 || c === 0) ? true : NS.L10.relClose(runOf('A017')(a, b, c), a / (b * c), 1e-9),
  });
  reg(18, {
    name: '标量除法 · 商再除近似', intuition: 'a/(b·c) 与 (a/b)/c 一致（舍入等价）。',
    math: 'a/(b·c) ≈ (a/b)/c', pseudocode: 'return a/(b*c)', complexity: 'O(1)',
    ref: '数值分析', commonPitfall: '需容差。',
    invariant: '对 b,c≠0：relClose(run(a,b,c), (a/b)/c, 1e-9)（异构分组）',
    inputs: ['a', 'b', 'c'], domains: { a: 'smallInt', b: 'smallInt', c: 'smallInt' },
    run: (a, b, c) => ((b === 0 || c === 0) ? 0 : a / (b * c)),
    predicate: (a, b, c) => (b === 0 || c === 0) ? true : NS.L10.relClose(runOf('A018')(a, b, c), (a / b) / c, 1e-9),
  });
  reg(19, {
    name: '标量除法 · 自商为 1', intuition: '非零数自除为精确 1。',
    math: 'a/a = 1', pseudocode: 'return a / a', complexity: 'O(1)',
    ref: 'IEEE 754', commonPitfall: 'a 不可为 0（0/0=NaN）。',
    invariant: '对 a≠0：Object.is(run(a), 1)',
    inputs: ['a'], domains: { a: 'nonZero' },
    run: (a) => a / a,
    predicate: (a) => Object.is(runOf('A019')(a), 1),
  });
  reg(20, {
    name: '标量倒数 · 乘积近似 1', intuition: '(1/a)·a 接近 1（允许浮点误差）。',
    math: '(1/a)·a ≈ 1', pseudocode: 'return (1/a)*a', complexity: 'O(1)',
    ref: '数值分析', commonPitfall: '非精确，需容差。',
    invariant: '对 a≠0：relClose(run(a), 1, 1e-9)（oracle 为常数 1）',
    inputs: ['a'], domains: { a: 'smallInt' },
    run: (a) => (a === 0 ? 0 : (1 / a) * a),
    predicate: (a) => (a === 0 ? true : NS.L10.relClose(runOf('A020')(a), 1, 1e-9)),
  });
  reg(21, {
    name: '标量倒数 · 负倒数对称近似', intuition: '1/(-a) 与 -(1/a) 一致（舍入等价）。',
    math: '1/(-a) ≈ -(1/a)', pseudocode: 'return 1/(-a)', complexity: 'O(1)',
    ref: '数值分析', commonPitfall: '需容差。',
    invariant: '对 a≠0：relClose(run(a), -(1/a), 1e-9)（异构公式）',
    inputs: ['a'], domains: { a: 'smallInt' },
    run: (a) => (a === 0 ? 0 : 1 / (-a)),
    predicate: (a) => (a === 0 ? true : NS.L10.relClose(runOf('A021')(a), -(1 / a), 1e-9)),
  });
  reg(22, {
    name: '标量除法 · 同号为正', intuition: '两同号有限数相除为正。',
    math: 'a>0 ∧ b>0 ⇒ a/b>0', pseudocode: 'return a/b', complexity: 'O(1)',
    ref: '符号规则', commonPitfall: '需 a,b>0。',
    invariant: '对 a,b>0：run(a,b) > 0（符号性质，run 耦合）',
    inputs: ['a', 'b'], domains: { a: 'pos', b: 'pos' },
    run: (a, b) => a / b,
    predicate: (a, b) => runOf('A022')(a, b) > 0,
  });
  reg(23, {
    name: '标量除法 · 符号翻转', intuition: 'a/(-b) 与 -(a/b) 逐位等价。',
    math: 'a/(-b) = -(a/b)', pseudocode: 'return a/(-b)', complexity: 'O(1)',
    ref: 'IEEE 754', commonPitfall: 'b 不可为 0。',
    invariant: '对 b≠0：Object.is(run(a,b), -(a/b))（异构公式）',
    inputs: ['a', 'b'], domains: { a: 'smallInt', b: 'smallInt' },
    run: (a, b) => (b === 0 ? 0 : a / (-b)),
    predicate: (a, b) => (b === 0 ? true : Object.is(runOf('A023')(a, b), -(a / b))),
  });
  reg(24, {
    name: '标量倒数 · 乘积倒数近似', intuition: '1/(a·b) 与 (1/a)·(1/b) 近似相等。',
    math: '1/(a·b) ≈ (1/a)·(1/b)', pseudocode: 'return 1/(a*b)', complexity: 'O(1)',
    ref: '数值分析', commonPitfall: '需容差。',
    invariant: '对 a,b≠0：relClose(run(a,b), (1/a)*(1/b), 1e-9)（异构公式）',
    inputs: ['a', 'b'], domains: { a: 'smallInt', b: 'smallInt' },
    run: (a, b) => ((a === 0 || b === 0) ? 0 : 1 / (a * b)),
    predicate: (a, b) => (a === 0 || b === 0) ? true : NS.L10.relClose(runOf('A024')(a, b), (1 / a) * (1 / b), 1e-9),
  });

  // ── 幂与根（A025–A040，本批至 A032）─────────────────────────────────
  reg(25, {
    name: '标量平方 · 与 Math.pow(a,2)', intuition: '小范围内 a*a 与 Math.pow(a,2) 一致。',
    math: 'a² = pow(a,2)', pseudocode: 'return a*a', complexity: 'O(1)',
    ref: 'pow 实现', commonPitfall: '大数下 pow 可能走不同路径，故用 small 域。',
    invariant: '对 |a|≤10：relClose(run(a), Math.pow(a,2), 1e-9)（异构实现路径）',
    inputs: ['a'], domains: { a: 'small' },
    run: (a) => a * a,
    predicate: (a) => NS.L10.relClose(runOf('A025')(a), Math.pow(a, 2), 1e-9),
  });
  reg(26, {
    name: '标量平方根 · 平方还原', intuition: '√a·√a 回到 a（允许浮点误差）。',
    math: '√a·√a ≈ a', pseudocode: 'return Math.sqrt(a)*Math.sqrt(a)', complexity: 'O(1)',
    ref: 'IEEE 754 sqrt', commonPitfall: 'a 须 ≥0。',
    invariant: '对 a≥0：relClose(run(a), a, 1e-9)（oracle 为输入本身）',
    inputs: ['a'], domains: { a: 'nonNegSmall' },
    run: (a) => Math.sqrt(a) * Math.sqrt(a),
    predicate: (a) => NS.L10.relClose(runOf('A026')(a), a, 1e-9),
  });
  reg(27, {
    name: '标量平方根 · √(a²)=|a|', intuition: '√(a²) 等于 |a|（允许浮点误差）。',
    math: '√(a²) = |a|', pseudocode: 'return Math.sqrt(a*a)', complexity: 'O(1)',
    ref: '平方根定义', commonPitfall: '需容差。',
    invariant: 'relClose(run(a), Math.abs(a), 1e-9)（异构公式）',
    inputs: ['a'], domains: { a: 'nonNeg' },
    run: (a) => Math.sqrt(a * a),
    predicate: (a) => NS.L10.relClose(runOf('A027')(a), Math.abs(a), 1e-9),
  });
  reg(28, {
    name: '标量立方 · 与 Math.pow(a,3)', intuition: '小范围 a³ 与 pow(a,3) 一致。',
    math: 'a³ = pow(a,3)', pseudocode: 'return a*a*a', complexity: 'O(1)',
    ref: 'pow 实现', commonPitfall: '用 small 域。',
    invariant: '对 |a|≤10：relClose(run(a), Math.pow(a,3), 1e-9)（异构实现路径）',
    inputs: ['a'], domains: { a: 'small' },
    run: (a) => a * a * a,
    predicate: (a) => NS.L10.relClose(runOf('A028')(a), Math.pow(a, 3), 1e-9),
  });
  reg(29, {
    name: '标量幂 · 零次幂为 1', intuition: '任意非零数 0 次幂为精确 1。',
    math: 'a⁰ = 1', pseudocode: 'return Math.pow(a,0)', complexity: 'O(1)',
    ref: '幂定义', commonPitfall: '0⁰=NaN，故限定 a≠0。',
    invariant: '对 a≠0：Object.is(run(a), 1)',
    inputs: ['a'], domains: { a: 'nonZero' },
    run: (a) => Math.pow(a, 0),
    predicate: (a) => Object.is(runOf('A029')(a), 1),
  });
  reg(30, {
    name: '标量幂 · 一次幂为自身', intuition: 'a¹ 逐位等于 a。',
    math: 'a¹ = a', pseudocode: 'return Math.pow(a,1)', complexity: 'O(1)',
    ref: '幂定义', commonPitfall: '无', invariant: '对所有有限 a：Object.is(run(a), a)',
    inputs: ['a'], domains: { a: 'finite' },
    run: (a) => Math.pow(a, 1),
    predicate: (a) => Object.is(runOf('A030')(a), a),
  });
  reg(31, {
    name: '标量幂 · 负一次幂为倒数', intuition: 'a⁻¹ 与 1/a 一致（允许误差）。',
    math: 'a⁻¹ ≈ 1/a', pseudocode: 'return Math.pow(a,-1)', complexity: 'O(1)',
    ref: '数值分析', commonPitfall: '需容差；a≠0。',
    invariant: '对 a≠0：relClose(run(a), 1/a, 1e-9)（异构公式）',
    inputs: ['a'], domains: { a: 'smallInt' },
    run: (a) => (a === 0 ? 0 : Math.pow(a, -1)),
    predicate: (a) => (a === 0 ? true : NS.L10.relClose(runOf('A031')(a), 1 / a, 1e-9)),
  });
  reg(32, {
    name: '标量绝对值 · 幂等', intuition: '绝对值对自身幂等。',
    math: '| |a| | = |a|', pseudocode: 'return Math.abs(a)', complexity: 'O(1)',
    ref: 'IEEE 754', commonPitfall: '无', invariant: 'Object.is(run(run(a)), run(a)) 且 Object.is(run(a), |a|)',
    inputs: ['a'], domains: { a: 'finite' },
    run: (a) => Math.abs(a),
    predicate: (a) => {
      const run = runOf('A032'), r = run(a);
      return Object.is(r, Math.abs(a)) && Object.is(run(r), r);   // 异构公式 + 幂等（均 run 耦合）
    },
  });
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

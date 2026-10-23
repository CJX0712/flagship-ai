/* ============================================================================
 * L10 微指令 · A 域（标量算术与初等函数）· 第 4/4 批（A097–A128）
 * ----------------------------------------------------------------------------
 * 涵盖：数值安全与特殊函数（A097–A110：符号函数三性 / copysign / fmod 两律 /
 *      clamp 两律 / NaN 不定式两项 / isFinite·isNaN 判定 / ulp·nextafter）、
 *      更多初等函数（A111–A122：exp2×2 / log2×2 / cbrt 奇性 / atan2 象限×2 /
 *      度弧度×2 / 三参 hypot / 幂与对数换底）、比较与选择（A123–A128：
 *      min/max 四律 / clamp 三值 / 对加法次超可加）。
 * 类别相对任务书的微调：ulp/nextafter 纳入本批（数值安全族），比较与选择
 * 收缩为 6 项，总数仍为 32，且与 A001–A096 无重复（cbrt 只补奇性，∛(a³)
 * 还原已由 A035 承担；hypot 只补三参，二参版已由 A036–A038 承担）。
 * 判据：精确恒等式 Object.is（±0 经 nz 归一）；数值恒等式 relClose 1e-9。
 * 激活类条目归 G 域，本批无 L9.special 可复用项（exp2/log2/cbrt/atan2/hypot
 * 均为引擎内建 Math 原语，位级 nextAfter 为本批自建 helper）。
 * K5：本文件仅用 IIFE 注入命名空间，禁用一切模块级语法（详见全局契约 K5）。
 *
 * ── 谓词纪律（本批全部重写，2026-10-23）─────────────────────────────────
 * 门禁只调 op.predicate，从不调 op.run（op-harness.js:43、probe-microisa.js:106）。
 * 原 32 项谓词全部内联复算 run 定义式或直接自比输入，run 若损坏门禁完全失明
 * （假绿）；A105/A106 更用与 run 不同的溢出表达式（10^400 vs 2^1024），
 * run 损坏时谓词照样为真。现全部改为：先经 runOf('Axxx') 调 run 本体，再对
 * run 的产出断言不变量；oracle 取异构公式、输入导出值、常数锚、零参定点或
 * 位级 helper 参考路径。
 * A122 修复真实潜伏 bug：原文档声称「pos 连续域不会精确取到 b=1」不成立——
 * 边界输入（b=1）可达，ln b=0 使 run=±Inf/NaN、roundtrip 无定义，旧谓词必假。
 * 现加 b===1 守卫：断言 run 在该退化基下确产出非有限值（IEEE 诚实行为），
 * 而非放宽容差凑绿。
 * ==========================================================================*/
;(function (NS) {
  'use strict';
  NS.L10 = NS.L10 || {};
  const rc = (a, b, t) => NS.L10.relClose(a, b, t);
  function reg(i, o) { NS.L10.register('A', i, o); }
  function runOf(key) { const o = NS.L10.ops[key]; return o && typeof o.run === 'function' ? o.run : null; }

  // ±0 归一：IEEE 中 -0 与 0 数值等价，逐位比对前统一为 +0。
  const nz = (x) => (x === 0 ? 0 : x);

  // nextAfter(x, up)：沿双精度可表示值格走向 ±∞ 的下一邻接值（位级实现）。
  // 依据 IEEE 754 符号-大小序：非负数的位模式随数值递增，负数反之；±0 邻域
  // 有保护分支（普通分支会把 ±0 的位模式加减 1 落到 NaN 区）。调用方约定
  // 传入有限非零 x（本批两项的域均为 nonZero）。
  const f64one = new Float64Array(1);
  const u32pair = new Uint32Array(f64one.buffer);
  function nextAfter(x, up) {
    f64one[0] = x;
    let lo = u32pair[0], hi = u32pair[1];
    if (hi === 0 && lo === 0) return up ? Number.MIN_VALUE : -Number.MIN_VALUE;
    if (hi === 0x80000000 && lo === 0) return up ? 0 : -Number.MIN_VALUE;
    const neg = (hi & 0x80000000) !== 0;
    if (neg ? up : !up) {                       // 数值减小：位模式递减
      if (lo === 0) hi = (hi - 1) >>> 0;
      lo = (lo - 1) >>> 0;
    } else {                                    // 数值增大：位模式递增
      lo = (lo + 1) >>> 0;
      if (lo === 0) hi = (hi + 1) >>> 0;
    }
    u32pair[0] = lo; u32pair[1] = hi;
    return f64one[0];
  }

  // ── 数值安全与特殊函数（A097–A110）────────────────────────────────────
  reg(97, {
    name: '符号 · 奇函数', intuition: '符号函数是奇函数：反向输入得反向符号（±0 归一后逐位成立，含 a=0）。',
    math: 'sign(−a) = −sign(a)', pseudocode: 'return Math.sign(-a)', complexity: 'O(1)',
    ref: 'IEEE 754-2019（符号函数）', commonPitfall: 'sign(−0)=−0，须归一后逐位比对。',
    invariant: '对所有有限 a：nz(run(a)) = nz(−sign(a))（含 a=0）',
    inputs: ['a'], domains: { a: 'finite' },
    run: (a) => Math.sign(-a),
    predicate: (a) => nz(runOf('A097')(a)) === nz(-Math.sign(a)),
  });
  reg(98, {
    name: '符号 · 乘性', intuition: '乘积的符号等于符号的乘积；任一因子为 0 时两侧同归 0。',
    math: 'sign(a·b) = sign(a)·sign(b)', pseudocode: 'return Math.sign(a*b)', complexity: 'O(1)',
    ref: 'IEEE 754-2019（符号函数）', commonPitfall: 'a·b 不得下溢为 0；small 域保证 |a·b|≥2e-17，含 0 时两侧经 nz 归零。',
    invariant: '对所有 |a|,|b|≤10：nz(run(a,b)) = nz(sign(a)·sign(b))',
    inputs: ['a', 'b'], domains: { a: 'small', b: 'small' },
    run: (a, b) => Math.sign(a * b),
    predicate: (a, b) => nz(runOf('A098')(a, b)) === nz(Math.sign(a) * Math.sign(b)),
  });
  reg(99, {
    name: '符号 · 原点逐位', intuition: 'sign 在原点逐位为 +0（锚定常量，供上层引用）。',
    math: 'sign(0) = 0', pseudocode: 'return Math.sign(0)', complexity: 'O(1)',
    ref: 'IEEE 754-2019（符号函数）', commonPitfall: 'Math.sign(−0)=−0 与 sign(0)=+0 不同；本项只断言 +0 输入。',
    invariant: 'Object.is(run(), 0)（零参定点，run 耦合）',
    inputs: [], domains: {},
    run: () => Math.sign(0),
    predicate: () => Object.is(runOf('A099')(), 0),
  });
  reg(100, {
    name: '符号 · copysign 模乘符号', intuition: 'copysign 取 x 的模与 y 的符号位；±0 视作正号（与 Math.sign 的 ±0 语义不同）。',
    math: 'copysign(x,y) = sgn₁(y)·|x|', pseudocode: 'return y<0 ? -Math.abs(x) : Math.abs(x)', complexity: 'O(1)',
    ref: 'C99 copysign；IEEE 754-2019', commonPitfall: 'Math.sign(±0)=±0，参考路径勿直接用它推符号位；±0 输入下两侧经 nz 归一。',
    invariant: 'nz(run(x,y)) = nz(sgn₁(y)·|x|)（sgn₁ 把 ±0 当作正号，定义式 oracle）',
    inputs: ['x', 'y'], domains: { x: 'finite', y: 'finite' },
    run: (x, y) => (y < 0 || Object.is(y, -0)) ? -Math.abs(x) : Math.abs(x),
    predicate: (x, y) => {
      const s = (y < 0 || Object.is(y, -0)) ? -1 : 1;
      return Object.is(nz(runOf('A100')(x, y)), nz(s * Math.abs(x)));
    },
  });
  reg(101, {
    name: '取模 · fmod 符号随被除数', intuition: 'fmod 余数符号跟随被除数（区别于向偶取整的 remainder），且 |余数|≤|除数|、商为整数。',
    math: 'fmod(a,b) = a − b·trunc(a/b)，符号随 a', pseudocode: 'return a - b*Math.trunc(a/b)', complexity: 'O(1)',
    ref: 'C99 fmod；IEEE 754-2019', commonPitfall: 'b 不可为 0；朴素公式在可生成域内无跨整数舍入（a/b 为有界整数比，已证 fl 不越过整数边界）。',
    invariant: '对 a∈ℝ, b≠0：fmod=0 ∨ sign(run)=sign(a)，且 |run|≤|b| 且商为整数，且 run=a−b·trunc(a/b)',
    inputs: ['a', 'b'], domains: { a: 'finite', b: 'nonZero' },
    run: (a, b) => a - b * Math.trunc(a / b),
    predicate: (a, b) => {
      const r = runOf('A101')(a, b);
      return (r === 0 || Math.sign(r) === Math.sign(a)) && Math.abs(r) <= Math.abs(b)
        && Number.isInteger(Math.trunc(a / b)) && Object.is(r, a - b * Math.trunc(a / b));
    },
  });
  reg(102, {
    name: '取模 · fmod 自模为零', intuition: '任何非零数以自身为模，余数精确为 0（IEEE 下 a/a 精确为 1）。',
    math: 'fmod(a,a) = 0', pseudocode: 'return a - a*Math.trunc(a/a)', complexity: 'O(1)',
    ref: 'C99 fmod；IEEE 754-2019', commonPitfall: 'a 不可为 0（0/0=NaN）；a/a 精确为 1 是关键中间事实。',
    invariant: '对 a≠0：run(a)=0（自模归零）',
    inputs: ['a'], domains: { a: 'nonZero' },
    run: (a) => a - a * Math.trunc(a / a),
    predicate: (a) => runOf('A102')(a) === 0,
  });
  reg(103, {
    name: '饱和 · clamp 夹取/幂等/分段一致', intuition: 'clamp 把输入夹进 [0,1]：结果夹取、对自身幂等，且与分段定义逐位一致。',
    math: 'clamp(a) = min(max(a,0),1) = (a<0 ? 0 : a>1 ? 1 : a)', pseudocode: 'return Math.min(Math.max(a,0),1)', complexity: 'O(1)',
    ref: '饱和运算（DSP）；序理论', commonPitfall: '嵌套 min/max 与分段定义在 0/1 边界须逐位一致；域不产生 −0。',
    invariant: 'run(a)∈[0,1] ∧ run(run(a))=run(a) ∧ run(a)=分段定义(a)',
    inputs: ['a'], domains: { a: 'finite' },
    run: (a) => Math.min(Math.max(a, 0), 1),
    predicate: (a) => {
      const run = runOf('A103'), c = run(a);
      const p = a < 0 ? 0 : (a > 1 ? 1 : a);
      return Object.is(c, p) && Object.is(run(c), c) && c >= 0 && c <= 1;
    },
  });
  reg(104, {
    name: '饱和 · clamp 保序', intuition: 'clamp 保序：输入有序则输出有序（min/max 的单调性逐位保持）。',
    math: 'a ≤ b ⇒ clamp(a) ≤ clamp(b)', pseudocode: 'return Math.min(Math.max(a,0),1)', complexity: 'O(1)',
    ref: '饱和运算（DSP）；单调函数', commonPitfall: '判据用蕴含式；run 只取首参（同 A055 先例）。',
    invariant: '∀a,b：(a≤b) ⇒ run(a) ≤ run(b)（两侧均 run 耦合）',
    inputs: ['a', 'b'], domains: { a: 'finite', b: 'finite' },
    run: (a) => Math.min(Math.max(a, 0), 1),
    predicate: (a, b) => { const run = runOf('A104'); return (a <= b) ? run(a) <= run(b) : true; },
  });
  reg(105, {
    name: 'NaN · 零乘无穷', intuition: '0 乘无穷是 IEEE 不定式，产生 NaN（∞ 由 2^1024 溢出构造，域内无 Inf 输入）。',
    math: '0 · ∞ = NaN', pseudocode: 'return 0 * Math.pow(10,400)', complexity: 'O(1)',
    ref: 'IEEE 754-2019 §6.2（不定式）', commonPitfall: '谓词直接检验 run 的产出（0·10^400）；若 run 被改成有限表达式则不再为 NaN，谓词可抓获。',
    invariant: 'Number.isNaN(run())（run 耦合）',
    inputs: [], domains: {},
    run: () => 0 * Math.pow(10, 400),
    predicate: () => Number.isNaN(runOf('A105')()),
  });
  reg(106, {
    name: 'NaN · 无穷相减', intuition: '无穷减无穷是不定式产生 NaN；对照：无穷加无穷仍是无穷（非 NaN）。',
    math: '∞ − ∞ = NaN', pseudocode: 'return Math.pow(10,400) - Math.pow(10,400)', complexity: 'O(1)',
    ref: 'IEEE 754-2019 §6.2（不定式）', commonPitfall: '对照项 ∞+∞=∞ 用于排除「凡涉 ∞ 皆 NaN」的误判；∞ 经 2^1024 独立构造。',
    invariant: 'Number.isNaN(run()) ∧ ¬Number.isNaN(∞+∞)',
    inputs: [], domains: {},
    run: () => Math.pow(10, 400) - Math.pow(10, 400),
    predicate: () => Number.isNaN(runOf('A106')()) && !Number.isNaN(Math.pow(2, 1024) + Math.pow(2, 1024)),
  });
  reg(107, {
    name: '判定 · isFinite 识别溢出', intuition: '有限数乘大常量必溢出为 ±Inf（a=0 时为 NaN），isFinite 一律判定为假。',
    math: 'a 有限 ⇒ ¬isFinite(a·∞)', pseudocode: 'return a * Math.pow(2,1024)', complexity: 'O(1)',
    ref: 'IEEE 754-2019；C99 isfinite', commonPitfall: 'a=0 时 0·∞=NaN 亦非有限，判定同为假。',
    invariant: '对所有有限 a：run(a) === +∞（a>0）/ −∞（a<0）/ NaN（a=0）',
    inputs: ['a'], domains: { a: 'finite' },
    run: (a) => a * Math.pow(2, 1024),
    predicate: (a) => {
      const r = runOf('A107')(a);
      if (a > 0) return r === Infinity;
      if (a < 0) return r === -Infinity;
      if (a === 0) return Number.isNaN(r);
      return false;
    },
  });
  reg(108, {
    name: '判定 · isNaN 有限运算闭合', intuition: '有限数的加/乘/平方根永不产生 NaN——NaN 只来自不定式，这是调试 NaN 的第一道筛子。',
    math: '¬isNaN(a+a) ∧ ¬isNaN(a·a) ∧ ¬isNaN(√(a²))', pseudocode: 'return Number.isNaN(a+a)', complexity: 'O(1)',
    ref: 'IEEE 754-2019；C99 isnan', commonPitfall: '谓词同时覆盖 +/×/√ 三条闭合路径，避免单项碰巧为真。',
    invariant: '对所有有限 a：run(a)===false ∧ ¬isNaN(a·a) ∧ ¬isNaN(√(a²))',
    inputs: ['a'], domains: { a: 'finite' },
    run: (a) => Number.isNaN(a + a),
    predicate: (a) => runOf('A108')(a) === false && Number.isNaN(a * a) === false && Number.isNaN(Math.sqrt(a * a)) === false,
  });
  reg(109, {
    name: '精度 · ulp 为 2 的幂', intuition: '相邻两双精度的间距即 ulp，必为 2 的幂：ulp(x)=2^(e−52)，其中 2^e≤|x|<2^(e+1)。',
    math: 'ulp(x) = 2^(e−52)', pseudocode: 'return nextAfter(x,+∞) - x', complexity: 'O(1)',
    ref: 'IEEE 754-2019 nextAfter；Muller et al. 2010', commonPitfall: '位级 nextAfter 避免 log2 在 2 的幂边界的不确定舍入；x 不可为 0（ulp(0) 另有约定）。',
    invariant: 'run(x) = 2^(e−52)：g·2^52 ≤ |x| < g·2^53（g=run(x)>0）',
    inputs: ['x'], domains: { x: 'nonZero' },
    run: (x) => nextAfter(x, true) - x,
    predicate: (x) => {
      const g = runOf('A109')(x);
      return g > 0 && g * Math.pow(2, 52) <= Math.abs(x) && Math.abs(x) < g * Math.pow(2, 53);
    },
  });
  reg(110, {
    name: '精度 · nextafter 方向可逆', intuition: 'nextafter 在双精度格上向 ±∞ 各严格走一步，且上下往返逐位还原（格上的双射）。',
    math: 'nextafter(x,+∞) > x > nextafter(x,−∞)', pseudocode: 'return nextAfter(x,+∞)', complexity: 'O(1)',
    ref: 'IEEE 754-2019 nextAfter；Muller et al. 2010', commonPitfall: '域排除 ±0/Inf/NaN 边界（位模式加减 1 在那些点会落到 NaN 区，helper 内有保护分支）。',
    invariant: 'run(x) > x ∧ nextAfter(x,−∞) < x ∧ 往返逐位还原',
    inputs: ['x'], domains: { x: 'nonZero' },
    run: (x) => nextAfter(x, true),
    predicate: (x) => { const u = runOf('A110')(x); return u > x && nextAfter(x, false) < x && nextAfter(u, false) === x; },
  });

  // ── 更多初等函数（A111–A122）──────────────────────────────────────────
  reg(111, {
    name: '指数 · exp2 加法转乘法', intuition: '以 2 为底的指数把加法转乘法（指数律）。',
    math: '2^(a+b) = 2^a · 2^b', pseudocode: 'return Math.pow(2,a+b)', complexity: 'O(1)',
    ref: '指数律；Abramowitz & Stegun 1964', commonPitfall: '两条路径舍入不同，需容差；small 域保证 2^(a+b) 不溢出。',
    invariant: 'relClose(run(a,b), 2^a·2^b, 1e-9)（异构公式）',
    inputs: ['a', 'b'], domains: { a: 'small', b: 'small' },
    run: (a, b) => Math.pow(2, a + b),
    predicate: (a, b) => rc(runOf('A111')(a, b), Math.pow(2, a) * Math.pow(2, b), 1e-9),
  });
  reg(112, {
    name: '指数 · exp2 对数还原', intuition: 'log2 与 2 的幂互为反函数（近似还原）。',
    math: 'log2(2^a) = a', pseudocode: 'return Math.log2(Math.pow(2,a))', complexity: 'O(1)',
    ref: '反函数关系；A&S 1964', commonPitfall: 'pow 与 log2 双重舍入，需容差；small 域避免 exp 端溢出。',
    invariant: 'relClose(run(a), a, 1e-9)（oracle 为输入本身）',
    inputs: ['a'], domains: { a: 'small' },
    run: (a) => Math.log2(Math.pow(2, a)),
    predicate: (a) => rc(runOf('A112')(a), a, 1e-9),
  });
  reg(113, {
    name: '对数 · log2 乘积转求和', intuition: '以 2 为底的对数把乘法转加法。',
    math: 'log2(a·b) = log2 a + log2 b', pseudocode: 'return Math.log2(a*b)', complexity: 'O(1)',
    ref: '对数律；A&S 1964', commonPitfall: 'a,b 须 >0；先乘后取对数与分别取对数再相加舍入不同，需容差。',
    invariant: '对 a,b>0：relClose(run(a,b), log2 a+log2 b, 1e-9)（异构公式）',
    inputs: ['a', 'b'], domains: { a: 'pos', b: 'pos' },
    run: (a, b) => Math.log2(a * b),
    predicate: (a, b) => rc(runOf('A113')(a, b), Math.log2(a) + Math.log2(b), 1e-9),
  });
  reg(114, {
    name: '对数 · log2 的 2 幂精确', intuition: 'log2 对 2 的整数次幂逐位精确（引擎已证实），可作上层锚点。',
    math: 'log2(2^k) = k', pseudocode: 'return Math.log2(1)', complexity: 'O(1)',
    ref: 'A&S 1964；对数定义', commonPitfall: '仅断言 2 的幂输入（k=0,1,10,−10），非幂输入不保证逐位；run 为零参（k=0），其余锚验证 Math.log2 本身。',
    invariant: 'Object.is(run(), 0) ∧ Object.is(log2(2^k), k)（k=1,10,−10）',
    inputs: [], domains: {},
    run: () => Math.log2(1),
    predicate: () => Object.is(runOf('A114')(), 0) && Object.is(Math.log2(2), 1) && Object.is(Math.log2(1024), 10) && Object.is(Math.log2(1 / 1024), -10),
  });
  reg(115, {
    name: '根式 · cbrt 奇函数', intuition: '立方根是奇函数：奇次根对负数同样成立（与 A035 的立方还原互补）。',
    math: 'cbrt(−a) = −cbrt(a)', pseudocode: 'return Math.cbrt(-a)', complexity: 'O(1)',
    ref: 'IEEE 754 cbrt（奇函数性）', commonPitfall: '∛(a³)≈a 已由 A035 承担，本项只补奇性；small 域远离大数相消。',
    invariant: 'relClose(run(a), −cbrt(a), 1e-9)（异构公式）',
    inputs: ['a'], domains: { a: 'small' },
    run: (a) => Math.cbrt(-a),
    predicate: (a) => rc(runOf('A115')(a), -Math.cbrt(a), 1e-9),
  });
  reg(116, {
    name: '三角 · atan2 右半平面', intuition: 'x>0 时四象限反正切退化为普通 atan(y/x)（右半平面无极角跳变）。',
    math: 'atan2(y,x) = atan(y/x)（x>0）', pseudocode: 'return Math.atan2(y,x)', complexity: 'O(1)',
    ref: 'atan2 定义（C99/IEEE 754-2019）', commonPitfall: 'x 须 >0；y/x 不得溢出（域保证 |y/x|≤1e10）；y=0 时两侧同为 0。',
    invariant: '对 x>0：relClose(run(y,x), atan(y/x), 1e-9)（异构公式）',
    inputs: ['y', 'x'], domains: { y: 'finite', x: 'pos' },
    run: (y, x) => Math.atan2(y, x),
    predicate: (y, x) => rc(runOf('A116')(y, x), Math.atan(y / x), 1e-9),
  });
  reg(117, {
    name: '三角 · atan2 对跖点', intuition: '对跖点 (−y,−x) 与 (y,x) 的方向相差 π，正负由 y 的符号决定。',
    math: 'atan2(−y,−x) = atan2(y,x) − sign(y)·π', pseudocode: 'return Math.atan2(-y,-x)', complexity: 'O(1)',
    ref: 'atan2 定义（C99/IEEE 754-2019）', commonPitfall: 'y=0 时关系退化为 ±π 边界（atan2(±0,负)=±π），故 y 取 nonZero。',
    invariant: '对 x>0, y≠0：relClose(run(y,x), atan2(y,x)−sign(y)·π, 1e-9)（异构公式）',
    inputs: ['y', 'x'], domains: { y: 'nonZero', x: 'pos' },
    run: (y, x) => Math.atan2(-y, -x),
    predicate: (y, x) => rc(runOf('A117')(y, x), Math.atan2(y, x) - Math.sign(y) * Math.PI, 1e-9),
  });
  reg(118, {
    name: '角度 · 度弧度往返重建', intuition: '度转弧度后经 sin/cos 重建可还原角度——独立校验 π/180 换算常数。',
    math: 'atan2(sin θ, cos θ) = θ，θ = d·π/180', pseudocode: 'return d * Math.PI / 180', complexity: 'O(1)',
    ref: '弧度定义（弧长/半径）', commonPitfall: 'small 域保证 θ∈(−π/18,π/18]⊂(−π,π]，atan2 才能还原主值；该路径能抓出错的换算常数。',
    invariant: 'relClose(run(d), atan2(sin run(d), cos run(d)), 1e-9)',
    inputs: ['d'], domains: { d: 'small' },
    run: (d) => d * Math.PI / 180,
    predicate: (d) => { const t = runOf('A118')(d); return rc(t, Math.atan2(Math.sin(t), Math.cos(t)), 1e-9); },
  });
  reg(119, {
    name: '角度 · 度弧度特殊角', intuition: '30°/60°/45° 的正、余弦、切是精确已知值，同时校验度表达式与弧度表达式两条路径。',
    math: 'sin 30° = cos 60° = ½；tan 45° = 1', pseudocode: 'return Math.sin(30*Math.PI/180)', complexity: 'O(1)',
    ref: 'A&S 1964（特殊角）', commonPitfall: '30·π/180 与 π/6 的舍入路径不同，两侧都须落入 1e-9。',
    invariant: 'rc(run(), ½, 1e-9) ∧ rc(sin(π/6), ½, 1e-9) ∧ rc(cos60°, ½, 1e-9) ∧ rc(tan45°, 1, 1e-9)',
    inputs: [], domains: {},
    run: () => Math.sin(30 * Math.PI / 180),
    predicate: () => rc(runOf('A119')(), 0.5, 1e-9) && rc(Math.sin(Math.PI / 6), 0.5, 1e-9) && rc(Math.cos(60 * Math.PI / 180), 0.5, 1e-9) && rc(Math.tan(45 * Math.PI / 180), 1, 1e-9),
  });
  reg(120, {
    name: '范数 · 三参 hypot 等价定义', intuition: '三参 hypot 是 √(a²+b²+c²) 的稳定实现（A036–A038 二参版的三参推广）。',
    math: 'hypot(a,b,c) = √(a²+b²+c²)', pseudocode: 'return Math.hypot(a,b,c)', complexity: 'O(1)',
    ref: 'C99 hypot；Goldberg 1991', commonPitfall: '两条路径舍入不同，需容差；small 域保证平方和不过大。',
    invariant: 'relClose(run(a,b,c), √(a²+b²+c²), 1e-9)（异构公式）',
    inputs: ['a', 'b', 'c'], domains: { a: 'small', b: 'small', c: 'small' },
    run: (a, b, c) => Math.hypot(a, b, c),
    predicate: (a, b, c) => rc(runOf('A120')(a, b, c), Math.sqrt(a * a + b * b + c * c), 1e-9),
  });
  reg(121, {
    name: '幂 · 换底为自然指数', intuition: '幂运算可换底为自然指数：a^b = exp(b·ln a)。',
    math: 'a^b = exp(b·ln a)', pseudocode: 'return Math.pow(a,b)', complexity: 'O(1)',
    ref: '换底公式；A&S 1964', commonPitfall: 'a 须 >0；两条路径双重舍入需容差；域保证 |b·ln a|≤92 不溢出。',
    invariant: '对 a>0：relClose(run(a,b), exp(b·ln a), 1e-9)（异构公式）',
    inputs: ['a', 'b'], domains: { a: 'pos', b: 'small' },
    run: (a, b) => Math.pow(a, b),
    predicate: (a, b) => rc(runOf('A121')(a, b), Math.exp(b * Math.log(a)), 1e-9),
  });
  reg(122, {
    name: '对数 · 换底还原', intuition: '以 b 为底取对数后再幂运算必须还原 a：b^(ln a/ln b)=a。',
    math: 'b^(log_b a) = a', pseudocode: 'return Math.log(a)/Math.log(b)', complexity: 'O(1)',
    ref: '换底公式；A&S 1964',
    commonPitfall: 'b=1 时 ln b=0，run=±Inf（a=1 时为 0/0=NaN），roundtrip 无定义——该退化基在边界构造中可达，谓词以守卫断言 IEEE 诚实行为（非有限），不放宽容差。',
    invariant: 'b≠1：relClose(pow(b, run(a,b)), a, 1e-9)；b=1：¬Number.isFinite(run(a,b))',
    inputs: ['a', 'b'], domains: { a: 'pos', b: 'pos' },
    run: (a, b) => Math.log(a) / Math.log(b),
    predicate: (a, b) => {
      const r = runOf('A122')(a, b);
      if (b === 1) return !Number.isFinite(r);               // 退化基：ln b=0 ⇒ ±Inf/NaN（守卫，非放水）
      return rc(Math.pow(b, r), a, 1e-9);                    // roundtrip oracle（run 耦合）
    },
  });

  // ── 比较与选择（A123–A128）────────────────────────────────────────────
  reg(123, {
    name: '比较 · min/max 交换律', intuition: 'min/max 是精确选择运算，交换参数逐位不变。',
    math: 'min(a,b)=min(b,a) ∧ max(a,b)=max(b,a)', pseudocode: 'return Math.min(a,b)', complexity: 'O(1)',
    ref: 'IEEE 754-2019 min/max；格论', commonPitfall: 'min/max 不做算术只做选择，故无舍入；±0 视作相等（域不产生 −0）。',
    invariant: 'Object.is(run(a,b), run(b,a)) ∧ Object.is(max(a,b), max(b,a))；锚 run(2,3)=2',
    inputs: ['a', 'b'], domains: { a: 'finite', b: 'finite' },
    run: (a, b) => Math.min(a, b),
    predicate: (a, b) => {
      const run = runOf('A123');
      return Object.is(run(a, b), run(b, a)) && Object.is(Math.max(a, b), Math.max(b, a)) && Object.is(run(2, 3), 2);
    },
  });
  reg(124, {
    name: '比较 · min/max 结合律', intuition: 'min/max 满足格结合律，嵌套顺序无关。',
    math: 'min(min(a,b),c)=min(a,min(b,c))', pseudocode: 'return Math.min(Math.min(a,b),c)', complexity: 'O(1)',
    ref: 'IEEE 754-2019 min/max；格论（Birkhoff 1948）', commonPitfall: '逐位成立（选择运算无舍入）；与浮点加法的不可结合性对照。',
    invariant: 'run(a,b,c)=min(a,min(b,c)) ∧ max 结合律（oracle 侧格性质）',
    inputs: ['a', 'b', 'c'], domains: { a: 'finite', b: 'finite', c: 'finite' },
    run: (a, b, c) => Math.min(Math.min(a, b), c),
    predicate: (a, b, c) => Object.is(runOf('A124')(a, b, c), Math.min(a, Math.min(b, c)))
      && Object.is(Math.max(Math.max(a, b), c), Math.max(a, Math.max(b, c))),
  });
  reg(125, {
    name: '比较 · min/max 幂等', intuition: 'min/max 对同一输入幂等，逐位还原。',
    math: 'min(a,a)=a ∧ max(a,a)=a', pseudocode: 'return Math.min(a,a)', complexity: 'O(1)',
    ref: 'IEEE 754-2019 min/max；格论', commonPitfall: 'NaN 输入会传播（min(NaN,a)=NaN），但域限定有限。',
    invariant: 'Object.is(run(a), a) ∧ Object.is(max(a,a), a)',
    inputs: ['a'], domains: { a: 'finite' },
    run: (a) => Math.min(a, a),
    predicate: (a) => Object.is(runOf('A125')(a), a) && Object.is(Math.max(a, a), a),
  });
  reg(126, {
    name: '比较 · min/max 对偶', intuition: 'min 与 max 经取反对偶：min(a,b)=−max(−a,−b)。',
    math: 'min(a,b)=−max(−a,−b) ∧ max(a,b)=−min(−a,−b)', pseudocode: 'return Math.min(a,b)', complexity: 'O(1)',
    ref: '格论（序对偶）；IEEE 754-2019', commonPitfall: '取反与选择皆精确，故逐位；±0 经 nz 归一（−0 与 0 等价）。',
    invariant: 'nz(run(a,b))=nz(−max(−a,−b)) ∧ nz(max(a,b))=nz(−min(−a,−b))',
    inputs: ['a', 'b'], domains: { a: 'finite', b: 'finite' },
    run: (a, b) => Math.min(a, b),
    predicate: (a, b) => Object.is(nz(runOf('A126')(a, b)), nz(-Math.max(-a, -b)))
      && Object.is(nz(Math.max(a, b)), nz(-Math.min(-a, -b))),
  });
  reg(127, {
    name: '饱和 · clamp 三值两定义一致', intuition: '三值夹取的两种格定义（先夹上界再夹下界 / 先夹下界再夹上界）必须逐位一致。',
    math: 'min(max(a,lo),hi) = max(lo,min(a,hi))（lo≤hi）', pseudocode: 'return Math.min(Math.max(a,lo),lo+d)', complexity: 'O(1)',
    ref: '序理论（格分配性）；饱和运算', commonPitfall: '上界由 lo+d 构造以保证 lo≤hi（TAGS 无有序二元域，直接给 lo/hi 会有 lo>hi 的反例）。',
    invariant: '两定义 nz 归一逐位一致 ∧ 结果 ∈ [lo, lo+d]',
    inputs: ['a', 'lo', 'd'], domains: { a: 'finite', lo: 'finite', d: 'nonNeg' },
    run: (a, lo, d) => Math.min(Math.max(a, lo), lo + d),
    predicate: (a, lo, d) => {
      const hi = lo + d;
      const c1 = runOf('A127')(a, lo, d);
      const c2 = Math.max(lo, Math.min(a, hi));
      return Object.is(nz(c1), nz(c2)) && c1 >= lo && c1 <= hi;
    },
  });
  reg(128, {
    name: '比较 · min/max 对加法次/超可加', intuition: 'min 对加法次可加、max 超可加；FP 下因正确舍入的单调性仍严格成立，无需容差。',
    math: 'min(a,b)+min(c,d) ≤ min(a+c,b+d)；max 反向', pseudocode: 'return Math.min(a,b)+Math.min(c,d)', complexity: 'O(1)',
    ref: '不等式理论；IEEE 754-2019（舍入单调性）', commonPitfall: '证明关键：m+n≤a+c（实数）⇒ fl(m+n)≤fl(a+c)（舍入单调），故 FP 下不退化。',
    invariant: 'run(a,b,c,d) ≤ min(a+c,b+d) ∧ max(a,b)+max(c,d) ≥ max(a+c,b+d)',
    inputs: ['a', 'b', 'c', 'd'], domains: { a: 'finite', b: 'finite', c: 'finite', d: 'finite' },
    run: (a, b, c, d) => Math.min(a, b) + Math.min(c, d),
    predicate: (a, b, c, d) => runOf('A128')(a, b, c, d) <= Math.min(a + c, b + d)
      && Math.max(a, b) + Math.max(c, d) >= Math.max(a + c, b + d),
  });
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

/* ============================================================================
 * L10 微指令 · B 域（向量运算）· 第 1/3 批（B001–B032）
 * ----------------------------------------------------------------------------
 * 所属层：L10-microIsa  域：B（向量）
 * 允许依赖的层：L9（rng/assert/special）、L10（registry/invariant）
 * 主要 @ref：Lawson et al. (1979) BLAS；Strang (2016) Linear Algebra；
 *           Hadamard (1899) 逐元素积；IEEE 754-2019
 *
 * 涵盖：元素wise 加减与取反（B001–B010）、标量-向量（B011–B019）、点积（B020–B032）。
 * 七件套 + run + predicate，判据同 A 域：整数向量域（vecInt）用 Object.is 逐位
 * 精确判等（-0≡0 归一）；含根号/平方的数值不变量用 relClose(tol=1e-9)。
 * 输入域由 L10.TAGS 约束（tests/L10 据此构造 fast-check 任意子）。
 * ==========================================================================*/
;(function (NS) {
  'use strict';
  NS.L10 = NS.L10 || {};
  function reg(i, o) { NS.L10.register('B', i, o); }
  const rc = (a, b, t) => NS.L10.relClose(a, b, t);

  // ── 向量小工具（纯函数，run/predicate 复用；-0 归一由 exactVec 负责）──────
  function zeros(n) { return new Float64Array(n); }
  function ones(n) { const o = new Float64Array(n); o.fill(1); return o; }
  function e0(n) { const o = new Float64Array(n); o[0] = 1; return o; }
  function vAdd(x, y) { const n = x.length, o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = x[i] + y[i]; return o; }
  function vSub(x, y) { const n = x.length, o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = x[i] - y[i]; return o; }
  function vNeg(x) { const n = x.length, o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = -x[i]; return o; }
  function vScale(c, x) { const n = x.length, o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = c * x[i]; return o; }
  function vMul(x, y) { const n = x.length, o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = x[i] * y[i]; return o; }
  function vRev(x) { const n = x.length, o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = x[n - 1 - i]; return o; }
  function dot(x, y) { let s = 0; for (let i = 0; i < x.length; i++) s += x[i] * y[i]; return s; }
  function norm(x) { return Math.sqrt(dot(x, x)); }
  // 逐位相等（-0 与 +0 归一为相等；IEEE 中 Object.is(-0,0) 为假但数值等价）。
  // 标量版 exactNum 是标量不变量的唯一判据：乘法/取负可产生 -0（如 0·(−792896) === -0），
  // 裸 Object.is 会把数值等价的 ±0 判为不等 → 伪红。向量版复用它。
  function exactNum(p, q) {
    if (p === 0) p = 0;
    if (q === 0) q = 0;
    return Object.is(p, q);
  }
  function exactVec(a, b) {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) if (!exactNum(a[i], b[i])) return false;
    return true;
  }

  // ── 元素wise 加减与取反（B001–B010）────────────────────────────────────
  reg(1, {
    name: 'vecAdd · 交换律', intuition: '逐元素加法交换，是所有归约不变量的基础。',
    math: 'x + y = y + x（逐分量）', pseudocode: 'return vAdd(x,y)', complexity: 'O(n)',
    ref: 'IEEE 754-2019 §5.1', commonPitfall: '浮点逐元素加法可交换（与结合律不同）。',
    invariant: '对 vecInt 向量：exactVec(x+y, y+x)',
    inputs: ['x', 'y'], domains: { x: 'vecInt', y: 'vecInt' },
    run: (x, y) => vAdd(x, y), predicate: (x, y) => exactVec(vAdd(x, y), vAdd(y, x)),
  });
  reg(2, {
    name: 'vecAdd · 结合律（整数域）', intuition: '整数向量域下逐元素加法精确可结合。',
    math: '(x + y) + z = x + (y + z)', pseudocode: 'return vAdd(vAdd(x,y),z)', complexity: 'O(n)',
    ref: 'Goldberg (1991) 整数运算精确', commonPitfall: '一般浮点域不可结合，故限定 vecInt。',
    invariant: '对 vecInt：exactVec((x+y)+z, x+(y+z))',
    inputs: ['x', 'y', 'z'], domains: { x: 'vecInt', y: 'vecInt', z: 'vecInt' },
    run: (x, y, z) => vAdd(vAdd(x, y), z),
    predicate: (x, y, z) => exactVec(vAdd(vAdd(x, y), z), vAdd(x, vAdd(y, z))),
  });
  reg(3, {
    name: 'vecAdd · 零元', intuition: '零向量是逐元素加法的单位元。',
    math: 'x + 0 = x', pseudocode: 'return vAdd(x, zeros(n))', complexity: 'O(n)',
    ref: '向量空间公理', commonPitfall: '-0 分量加 +0 归正为 +0，exactVec 已归一。',
    invariant: '对 vecInt：exactVec(x+0, x)',
    inputs: ['x'], domains: { x: 'vecInt' },
    run: (x) => vAdd(x, zeros(x.length)),
    predicate: (x) => exactVec(vAdd(x, zeros(x.length)), x),
  });
  reg(4, {
    name: 'vecAdd · 自逆（加负为零）', intuition: '任意向量加自身的负向量得精确零向量。',
    math: 'x + (−x) = 0', pseudocode: 'return vAdd(x, vNeg(x))', complexity: 'O(n)',
    ref: 'IEEE 754（a+(−a)=+0）', commonPitfall: '每分量结果为 +0（符号归正）。',
    invariant: '对 vecInt：exactVec(x+(−x), 0)',
    inputs: ['x'], domains: { x: 'vecInt' },
    run: (x) => vAdd(x, vNeg(x)),
    predicate: (x) => exactVec(vAdd(x, vNeg(x)), zeros(x.length)),
  });
  reg(5, {
    name: 'vecSub · 自减为零', intuition: '向量减自身逐分量为精确 +0。',
    math: 'x − x = 0', pseudocode: 'return vSub(x,x)', complexity: 'O(n)',
    ref: 'IEEE 754', commonPitfall: '结果为 +0 而非 -0。',
    invariant: '对 vecInt：exactVec(x−x, 0)',
    inputs: ['x'], domains: { x: 'vecInt' },
    run: (x) => vSub(x, x), predicate: (x) => exactVec(vSub(x, x), zeros(x.length)),
  });
  reg(6, {
    name: 'vecSub · 加负等价', intuition: '逐元素减法即加负向量，整数域精确。',
    math: 'x − y = x + (−y)', pseudocode: 'return vSub(x,y)', complexity: 'O(n)',
    ref: 'Goldberg (1991)', commonPitfall: '一般浮点域舍入路径不同，故用 vecInt。',
    invariant: '对 vecInt：exactVec(x−y, x+(−y))',
    inputs: ['x', 'y'], domains: { x: 'vecInt', y: 'vecInt' },
    run: (x, y) => vSub(x, y), predicate: (x, y) => exactVec(vSub(x, y), vAdd(x, vNeg(y))),
  });
  reg(7, {
    name: 'vecNeg · 对合', intuition: '两次取反回到原向量（逐位）。',
    math: '−(−x) = x', pseudocode: 'return vNeg(vNeg(x))', complexity: 'O(n)',
    ref: 'IEEE 754', commonPitfall: '无', invariant: '对 vecInt：exactVec(−(−x), x)',
    inputs: ['x'], domains: { x: 'vecInt' },
    run: (x) => vNeg(x), predicate: (x) => exactVec(vNeg(vNeg(x)), x),
  });
  reg(8, {
    name: 'vecNeg · 对加法分配', intuition: '和的负等于负的和，整数域精确。',
    math: '−(x + y) = (−x) + (−y)', pseudocode: 'return vNeg(vAdd(x,y))', complexity: 'O(n)',
    ref: 'Goldberg (1991)', commonPitfall: '一般域舍入破坏等式，故用 vecInt。',
    invariant: '对 vecInt：exactVec(−(x+y), (−x)+(−y))',
    inputs: ['x', 'y'], domains: { x: 'vecInt', y: 'vecInt' },
    run: (x, y) => vNeg(vAdd(x, y)),
    predicate: (x, y) => exactVec(vNeg(vAdd(x, y)), vAdd(vNeg(x), vNeg(y))),
  });
  reg(9, {
    name: 'vecAdd · 与反转交换', intuition: '逐元素加法与索引反转可交换（顺序无关性）。',
    math: 'reverse(x + y) = reverse(x) + reverse(y)', pseudocode: 'return vAdd(x,y)', complexity: 'O(n)',
    ref: '索引代数', commonPitfall: '无', invariant: '对 vecInt：exactVec(rev(x+y), rev(x)+rev(y))',
    inputs: ['x', 'y'], domains: { x: 'vecInt', y: 'vecInt' },
    run: (x, y) => vAdd(x, y),
    predicate: (x, y) => exactVec(vRev(vAdd(x, y)), vAdd(vRev(x), vRev(y))),
  });
  reg(10, {
    name: 'vecSub · 交换取负', intuition: '交换减序等于对结果取反。',
    math: 'y − x = −(x − y)', pseudocode: 'return vSub(y,x)', complexity: 'O(n)',
    ref: '向量空间公理', commonPitfall: '无', invariant: '对 vecInt：exactVec(y−x, −(x−y))',
    inputs: ['x', 'y'], domains: { x: 'vecInt', y: 'vecInt' },
    run: (x, y) => vSub(y, x), predicate: (x, y) => exactVec(vSub(y, x), vNeg(vSub(x, y))),
  });

  // ── 标量-向量（B011–B019）──────────────────────────────────────────────
  reg(11, {
    name: 'scaleVec · 对 vecAdd 分配', intuition: '标量乘法对向量加法可分配，整数域精确。',
    math: 'c·(x + y) = c·x + c·y', pseudocode: 'return vScale(c, vAdd(x,y))', complexity: 'O(n)',
    ref: '向量空间公理；Goldberg (1991)', commonPitfall: '一般域舍入破坏等式，故 c 取整数。',
    invariant: '对 int c, vecInt x,y：exactVec(c(x+y), cx+cy)',
    inputs: ['c', 'x', 'y'], domains: { c: 'int', x: 'vecInt', y: 'vecInt' },
    run: (c, x, y) => vScale(c, vAdd(x, y)),
    predicate: (c, x, y) => exactVec(vScale(c, vAdd(x, y)), vAdd(vScale(c, x), vScale(c, y))),
  });
  reg(12, {
    name: 'scaleVec · 对标量加法分配', intuition: '标量乘法对 scalar 加法可分配。',
    math: '(c + d)·x = c·x + d·x', pseudocode: 'return vScale(c+d, x)', complexity: 'O(n)',
    ref: '向量空间公理', commonPitfall: '需整数标量保证精确。',
    invariant: '对 int c,d, vecInt x：exactVec((c+d)x, cx+dx)',
    inputs: ['c', 'd', 'x'], domains: { c: 'int', d: 'int', x: 'vecInt' },
    run: (c, d, x) => vScale(c + d, x),
    predicate: (c, d, x) => exactVec(vScale(c + d, x), vAdd(vScale(c, x), vScale(d, x))),
  });
  reg(13, {
    name: 'scaleVec · 结合律', intuition: '标量乘法可结合，整数域精确。',
    math: '(c·d)·x = c·(d·x)', pseudocode: 'return vScale(c*d, x)', complexity: 'O(n)',
    ref: 'Goldberg (1991)', commonPitfall: 'c·d·x 须不超 2^53，故用 smallInt（≤8e9）。',
    invariant: '对 smallInt c,d, vecInt x：exactVec((cd)x, c(dx))',
    inputs: ['c', 'd', 'x'], domains: { c: 'smallInt', d: 'smallInt', x: 'vecInt' },
    run: (c, d, x) => vScale(c * d, x),
    predicate: (c, d, x) => exactVec(vScale(c * d, x), vScale(c, vScale(d, x))),
  });
  reg(14, {
    name: 'scaleVec · 单位元', intuition: '标量 1 是数乘的单位元。',
    math: '1·x = x', pseudocode: 'return vScale(1, x)', complexity: 'O(n)',
    ref: '向量空间公理', commonPitfall: '无', invariant: '对 vecInt：exactVec(1x, x)',
    inputs: ['x'], domains: { x: 'vecInt' },
    run: (x) => vScale(1, x), predicate: (x) => exactVec(vScale(1, x), x),
  });
  reg(15, {
    name: 'scaleVec · 零元', intuition: '零标量数乘得零向量（符号零归一）。',
    math: '0·x = 0', pseudocode: 'return vScale(0, x)', complexity: 'O(n)',
    ref: 'IEEE 754', commonPitfall: '负分量得 -0，须按数值语义判等。',
    invariant: '对 vecInt：exactVec(0x, 0)（-0≡0）',
    inputs: ['x'], domains: { x: 'vecInt' },
    run: (x) => vScale(0, x), predicate: (x) => exactVec(vScale(0, x), zeros(x.length)),
  });
  reg(16, {
    name: 'scaleVec · 对 vecSub 分配', intuition: '数乘对减法可分配，整数域精确。',
    math: 'c·(x − y) = c·x − c·y', pseudocode: 'return vScale(c, vSub(x,y))', complexity: 'O(n)',
    ref: '向量空间公理', commonPitfall: '需整数标量。',
    invariant: '对 int c, vecInt x,y：exactVec(c(x−y), cx−cy)',
    inputs: ['c', 'x', 'y'], domains: { c: 'int', x: 'vecInt', y: 'vecInt' },
    run: (c, x, y) => vScale(c, vSub(x, y)),
    predicate: (c, x, y) => exactVec(vScale(c, vSub(x, y)), vSub(vScale(c, x), vScale(c, y))),
  });
  reg(17, {
    name: 'scaleVec · 负标量', intuition: '负标量数乘等于先数乘再取反。',
    math: '(−c)·x = −(c·x)', pseudocode: 'return vScale(-c, x)', complexity: 'O(n)',
    ref: 'IEEE 754', commonPitfall: '无', invariant: '对 int c, vecInt x：exactVec((−c)x, −(cx))',
    inputs: ['c', 'x'], domains: { c: 'int', x: 'vecInt' },
    run: (c, x) => vScale(-c, x), predicate: (c, x) => exactVec(vScale(-c, x), vNeg(vScale(c, x))),
  });
  reg(18, {
    name: 'scaleVec · 翻倍即自加', intuition: '2·x 与 x+x 逐位一致。',
    math: '2·x = x + x', pseudocode: 'return vScale(2, x)', complexity: 'O(n)',
    ref: 'IEEE 754（乘 2 精确）', commonPitfall: '无', invariant: '对 vecInt：exactVec(2x, x+x)',
    inputs: ['x'], domains: { x: 'vecInt' },
    run: (x) => vScale(2, x), predicate: (x) => exactVec(vScale(2, x), vAdd(x, x)),
  });
  reg(19, {
    name: 'scaleVec · 三元分配', intuition: '数乘对三向量之和可分配。',
    math: 'c·(x + y + z) = c·x + c·y + c·z', pseudocode: 'return vScale(c, vAdd(vAdd(x,y),z))', complexity: 'O(n)',
    ref: '向量空间公理', commonPitfall: '需整数标量；模长 ≤3e9，远小于 2^53。',
    invariant: '对 int c, vecInt x,y,z：exactVec(c(x+y+z), cx+cy+cz)',
    inputs: ['c', 'x', 'y', 'z'], domains: { c: 'int', x: 'vecInt', y: 'vecInt', z: 'vecInt' },
    run: (c, x, y, z) => vScale(c, vAdd(vAdd(x, y), z)),
    predicate: (c, x, y, z) => exactVec(
      vScale(c, vAdd(vAdd(x, y), z)),
      vAdd(vAdd(vScale(c, x), vScale(c, y)), vScale(c, z))),
  });

  // ── 点积（B020–B032）───────────────────────────────────────────────────
  reg(20, {
    name: 'dot · 交换律', intuition: '点积交换，是内积空间的基本性质。',
    math: 'x·y = y·x', pseudocode: 'return dot(x,y)', complexity: 'O(n)',
    ref: 'Strang (2016) 内积公理', commonPitfall: '整数域下求和顺序不影响精确结果。',
    invariant: '对 vecInt：Object.is(x·y, y·x)',
    inputs: ['x', 'y'], domains: { x: 'vecInt', y: 'vecInt' },
    run: (x, y) => dot(x, y), predicate: (x, y) => Object.is(dot(x, y), dot(y, x)),
  });
  reg(21, {
    name: 'dot · 双线性（对 vecAdd）', intuition: '点积对向量加法线性，整数域精确。',
    math: 'x·(y + z) = x·y + x·z', pseudocode: 'return dot(x, vAdd(y,z))', complexity: 'O(n)',
    ref: 'Strang (2016) 双线性', commonPitfall: '一般域舍入破坏等式，故用 vecInt。',
    invariant: '对 vecInt：Object.is(x·(y+z), x·y + x·z)',
    inputs: ['x', 'y', 'z'], domains: { x: 'vecInt', y: 'vecInt', z: 'vecInt' },
    run: (x, y, z) => dot(x, vAdd(y, z)),
    predicate: (x, y, z) => Object.is(dot(x, vAdd(y, z)), dot(x, y) + dot(x, z)),
  });
  reg(22, {
    name: 'dot · 标量提出', intuition: '点积可把标量因子提到和式外。',
    math: 'x·(c·y) = c·(x·y)', pseudocode: 'return dot(x, vScale(c,y))', complexity: 'O(n)',
    ref: 'Strang (2016) 双线性', commonPitfall: '需整数 c；c·(x·y) ≤8e9 精确。另：c=0 时 c·(x·y) 的符号随 x·y 变（可为 −0），裸 Object.is 判 ±0 不等 → 用 exactNum 归一。',
    invariant: '对 smallInt c, vecInt x,y：exactNum(x·(cy), c(x·y))',
    inputs: ['c', 'x', 'y'], domains: { c: 'smallInt', x: 'vecInt', y: 'vecInt' },
    run: (c, x, y) => dot(x, vScale(c, y)),
    predicate: (c, x, y) => exactNum(dot(x, vScale(c, y)), c * dot(x, y)),
  });
  reg(23, {
    name: 'dot · 零向量湮没', intuition: '任意向量与零向量点积为 0。',
    math: 'x·0 = 0', pseudocode: 'return dot(x, zeros(n))', complexity: 'O(n)',
    ref: '内积正定性', commonPitfall: '负分量产 -0 项，但累加器自 +0 起，结果恒 +0。',
    invariant: '对 vecInt：Object.is(x·0, 0)',
    inputs: ['x'], domains: { x: 'vecInt' },
    run: (x) => dot(x, zeros(x.length)),
    predicate: (x) => Object.is(dot(x, zeros(x.length)), 0),
  });
  reg(24, {
    name: 'dot · 基向量选元', intuition: '与标准基向量点积即选出对应分量。',
    math: 'x·e₀ = x₀', pseudocode: 'return dot(x, e0(n))', complexity: 'O(n)',
    ref: 'Strang (2016) 标准基', commonPitfall: '其余分量乘 0 不影响精确和。',
    invariant: '对 vecInt：Object.is(x·e₀, x[0])',
    inputs: ['x'], domains: { x: 'vecInt' },
    run: (x) => dot(x, e0(x.length)),
    predicate: (x) => Object.is(dot(x, e0(x.length)), x[0]),
  });
  reg(25, {
    name: 'dot · 自点积即范数平方', intuition: 'x·x 与 ‖x‖² 是同一量的两种计算路径（代数 vs 开方）。',
    math: 'x·x = ‖x‖²', pseudocode: 'return dot(x,x)', complexity: 'O(n)',
    ref: 'Strang (2016) 范数定义', commonPitfall: '开方回乘有舍入，需容差 1e-9。',
    invariant: '对 vecSmall：relClose(x·x, ‖x‖², 1e-9)',
    inputs: ['x'], domains: { x: 'vecSmall' },
    run: (x) => dot(x, x),
    predicate: (x) => rc(dot(x, x), norm(x) * norm(x), 1e-9),
  });
  reg(26, {
    name: 'dot · 自点积非负', intuition: '内积正定性：x·x ≥ 0。',
    math: 'x·x ≥ 0', pseudocode: 'return dot(x,x)', complexity: 'O(n)',
    ref: 'Strang (2016) 内积正定性', commonPitfall: '平方和不可能为负（无 NaN/Inf 域）。',
    invariant: '对 vecFinite：dot(x,x) ≥ 0',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => dot(x, x), predicate: (x) => dot(x, x) >= 0,
  });
  reg(27, {
    name: 'dot · 柯西-施瓦茨', intuition: '点积被两个范数之积控制（内积空间核心不等式）。',
    math: '(x·y)² ≤ (x·x)(y·y)', pseudocode: 'return dot(x,y)', complexity: 'O(n)',
    ref: 'Cauchy (1821); Schwarz (1888)', commonPitfall: '浮点下可差数个 ulp，判据加 1e-9 相对 epsilon。',
    invariant: '对 vecSmall：(x·y)² ≤ (x·x)(y·y) + ε，ε=1e-9·max(1,(x·x)(y·y))',
    inputs: ['x', 'y'], domains: { x: 'vecSmall', y: 'vecSmall' },
    run: (x, y) => dot(x, y),
    predicate: (x, y) => {
      const d = dot(x, y), xx = dot(x, x), yy = dot(y, y);
      return d * d <= xx * yy + 1e-9 * Math.max(1, xx * yy);
    },
  });
  reg(28, {
    name: 'dot · 和的平方展开', intuition: '(x+y)·(x+y) 展开为自点积与交叉项。',
    math: '(x+y)·(x+y) = x·x + 2x·y + y·y', pseudocode: 'return dot(vAdd(x,y), vAdd(x,y))', complexity: 'O(n)',
    ref: '双线性展开', commonPitfall: '需容差 1e-9。',
    invariant: '对 vecSmall：relClose((x+y)², x·x+2x·y+y·y, 1e-9)',
    inputs: ['x', 'y'], domains: { x: 'vecSmall', y: 'vecSmall' },
    run: (x, y) => dot(vAdd(x, y), vAdd(x, y)),
    predicate: (x, y) => rc(dot(vAdd(x, y), vAdd(x, y)), dot(x, x) + 2 * dot(x, y) + dot(y, y), 1e-9),
  });
  reg(29, {
    name: 'dot · 差的平方展开', intuition: '(x−y)·(x−y) 展开（平方差结构）。',
    math: '(x−y)·(x−y) = x·x − 2x·y + y·y', pseudocode: 'return dot(vSub(x,y), vSub(x,y))', complexity: 'O(n)',
    ref: '双线性展开', commonPitfall: '需容差 1e-9。',
    invariant: '对 vecSmall：relClose((x−y)², x·x−2x·y+y·y, 1e-9)',
    inputs: ['x', 'y'], domains: { x: 'vecSmall', y: 'vecSmall' },
    run: (x, y) => dot(vSub(x, y), vSub(x, y)),
    predicate: (x, y) => rc(dot(vSub(x, y), vSub(x, y)), dot(x, x) - 2 * dot(x, y) + dot(y, y), 1e-9),
  });
  reg(30, {
    name: 'dot · 与反转交换', intuition: '点积对两个向量同时反转不变（求和顺序无关）。',
    math: 'x·y = reverse(x)·reverse(y)', pseudocode: 'return dot(x,y)', complexity: 'O(n)',
    ref: '索引代数', commonPitfall: '整数域下精确；浮点域一般不成立。',
    invariant: '对 vecInt：Object.is(x·y, rev(x)·rev(y))',
    inputs: ['x', 'y'], domains: { x: 'vecInt', y: 'vecInt' },
    run: (x, y) => dot(x, y),
    predicate: (x, y) => Object.is(dot(x, y), dot(vRev(x), vRev(y))),
  });
  reg(31, {
    name: 'dot · 与 Hadamard 桥接', intuition: 'x·(y⊙z) = (x⊙y)·z：点积吸收逐元素积（结合型）。',
    math: 'x·(y⊙z) = (x⊙y)·z', pseudocode: 'return dot(x, vMul(y,z))', complexity: 'O(n)',
    ref: 'Hadamard (1899); 张量缩并', commonPitfall: '三项积 ≤8e9，vecInt 下精确。',
    invariant: '对 vecInt：Object.is(x·(y⊙z), (x⊙y)·z)',
    inputs: ['x', 'y', 'z'], domains: { x: 'vecInt', y: 'vecInt', z: 'vecInt' },
    run: (x, y, z) => dot(x, vMul(y, z)),
    predicate: (x, y, z) => Object.is(dot(x, vMul(y, z)), dot(vMul(x, y), z)),
  });
  reg(32, {
    name: 'dot · 缩放复合', intuition: '两个向量各自数乘后点积，提出两标量之积。',
    math: '(c·x)·(d·y) = c·d·(x·y)', pseudocode: 'return dot(vScale(c,x), vScale(d,y))', complexity: 'O(n)',
    ref: '双线性', commonPitfall: 'c·d·xᵢyᵢ ≤8e12，smallInt 下精确。另：c 或 d 为 0 时 c·d·(x·y) 的符号随 x·y 变（可为 −0），裸 Object.is 判 ±0 不等 → 用 exactNum 归一（实测捕获种子 314864380 即此情形：d=0、dot(x,y)=−792896 → rhs=−0 而 lhs=+0）。',
    invariant: '对 smallInt c,d, vecInt x,y：exactNum((cx)·(dy), cd(x·y))',
    inputs: ['c', 'd', 'x', 'y'], domains: { c: 'smallInt', d: 'smallInt', x: 'vecInt', y: 'vecInt' },
    run: (c, d, x, y) => dot(vScale(c, x), vScale(d, y)),
    predicate: (c, d, x, y) => exactNum(dot(vScale(c, x), vScale(d, y)), c * d * dot(x, y)),
  });
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

/* ============================================================================
 * L10 微指令 · B 域（向量运算）· 第 2/4 批（B033–B064）
 * ----------------------------------------------------------------------------
 * 所属层：L10-microIsa  域：B（向量）
 * 允许依赖的层：L9（rng/assert/special）、L10（registry/invariant）
 * 主要 @ref：Strang (2016) Linear Algebra；Lawson et al. (1979) BLAS；
 *           Hadamard (1899)；NumPy clip/normalize 语义
 *
 * 涵盖：范数（B033–B040）、min/max/mean/sum 归约（B041–B050）、
 *       Hadamard 积（B051–B056）、axpy（B057–B060）、clamp/relu（B061–B064）。
 * 判据：整数向量域 Object.is 逐位精确；含开方的数值不变量 relClose(1e-9)；
 *       不等式类不变量加 1e-9 相对 epsilon 吸收舍入（仍为真实数学性质）。
 *
 * ── 谓词纪律（本批全部重写，2026-10-23）─────────────────────────────────
 * 门禁只调 op.predicate，从不调 op.run（op-harness.js:43、probe-microisa.js:106）。
 * 原 32 项谓词全部用本地 helper（norm/maxv/clampVec/…）内联复算 run 定义式，
 * run 若损坏门禁完全失明（假绿）。现全部改为：先经 runOf('Bxxx') 调 run 本体，
 * 再对 run 的产出断言不变量；oracle 取异构公式、输入导出值、常数锚或定点锚
 * （构造确定向量喂 run）；幂等类经 run 自复合验证。
 * ==========================================================================*/
;(function (NS) {
  'use strict';
  NS.L10 = NS.L10 || {};
  function reg(i, o) { NS.L10.register('B', i, o); }
  function runOf(key) { const o = NS.L10.ops[key]; return o && typeof o.run === 'function' ? o.run : null; }
  const rc = (a, b, t) => NS.L10.relClose(a, b, t);

  // ── 向量小工具（纯函数）────────────────────────────────────────────────
  function ones(n) { const o = new Float64Array(n); o.fill(1); return o; }
  function vec() { return Float64Array.from(arguments); }
  function vAdd(x, y) { const n = x.length, o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = x[i] + y[i]; return o; }
  function vSub(x, y) { const n = x.length, o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = x[i] - y[i]; return o; }
  function vNeg(x) { const n = x.length, o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = -x[i]; return o; }
  function vScale(c, x) { const n = x.length, o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = c * x[i]; return o; }
  function vMul(x, y) { const n = x.length, o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = x[i] * y[i]; return o; }
  function vRev(x) { const n = x.length, o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = x[n - 1 - i]; return o; }
  function dot(x, y) { let s = 0; for (let i = 0; i < x.length; i++) s += x[i] * y[i]; return s; }
  function norm(x) { return Math.sqrt(dot(x, x)); }
  function sumv(x) { let s = 0; for (let i = 0; i < x.length; i++) s += x[i]; return s; }
  function sumAbs(x) { let s = 0; for (let i = 0; i < x.length; i++) s += Math.abs(x[i]); return s; }
  function maxAbs(x) { let m = Math.abs(x[0]); for (let i = 1; i < x.length; i++) { const a = Math.abs(x[i]); if (a > m) m = a; } return m; }
  function maxv(x) { let m = x[0]; for (let i = 1; i < x.length; i++) if (x[i] > m) m = x[i]; return m; }
  function minv(x) { let m = x[0]; for (let i = 1; i < x.length; i++) if (x[i] < m) m = x[i]; return m; }
  function meanv(x) { return sumv(x) / x.length; }
  function reluVec(x) { const n = x.length, o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = x[i] > 0 ? x[i] : 0; return o; }
  function clampVec(x, lo, hi) { const n = x.length, o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = Math.min(hi, Math.max(lo, x[i])); return o; }
  // 标量逐位相等（-0 与 +0 归一为相等；IEEE 中 Object.is(-0,0) 为假但数值等价）。
  // 标量不变量的唯一判据：乘法/取负可产生 -0（如 0·(−792896) === −0），
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

  // ── 范数（B033–B040）───────────────────────────────────────────────────
  reg(33, {
    name: 'norm · 取反不变', intuition: '范数只问长度，不问方向；取反不改变模长。',
    math: '‖−x‖ = ‖x‖', pseudocode: 'return norm(vNeg(x))', complexity: 'O(n)',
    ref: 'Strang (2016) 范数公理', commonPitfall: '开方有舍入，用 relClose 1e-9。',
    invariant: '对 vecFinite：relClose(run(x), ‖x‖, 1e-9)（异构参数）',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => norm(vNeg(x)),
    predicate: (x) => rc(runOf('B033')(x), norm(x), 1e-9),
  });
  reg(34, {
    name: 'norm · 齐次缩放', intuition: '向量整体缩放 c 倍，模长缩放 |c| 倍。',
    math: '‖c·x‖ = |c|·‖x‖', pseudocode: 'return norm(vScale(c,x))', complexity: 'O(n)',
    ref: 'Strang (2016) 范数齐次性', commonPitfall: '需容差；c 取 smallInt 防溢出。',
    invariant: '对 smallInt c, vecSmall x：relClose(run(c,x), |c|‖x‖, 1e-9)（异构公式）',
    inputs: ['c', 'x'], domains: { c: 'smallInt', x: 'vecSmall' },
    run: (c, x) => norm(vScale(c, x)),
    predicate: (c, x) => rc(runOf('B034')(c, x), Math.abs(c) * norm(x), 1e-9),
  });
  reg(35, {
    name: 'norm · 三角不等式', intuition: '两边之和大于第三边：‖x+y‖ 不超两模长之和。',
    math: '‖x + y‖ ≤ ‖x‖ + ‖y‖', pseudocode: 'return norm(vAdd(x,y))', complexity: 'O(n)',
    ref: 'Strang (2016) 范数三角不等式', commonPitfall: '浮点可差数个 ulp，加 1e-9 相对 epsilon。',
    invariant: '对 vecSmall：run(x,y) ≤ ‖x‖+‖y‖+ε，ε=1e-9·max(1,‖x‖+‖y‖)',
    inputs: ['x', 'y'], domains: { x: 'vecSmall', y: 'vecSmall' },
    run: (x, y) => norm(vAdd(x, y)),
    predicate: (x, y) => runOf('B035')(x, y) <= norm(x) + norm(y) + 1e-9 * Math.max(1, norm(x) + norm(y)),
  });
  reg(36, {
    name: 'norm · 单位化后模为 1', intuition: '向量除以自身模长即得单位向量（方向保持、长度归一）。',
    math: '‖x/‖x‖‖ = 1', pseudocode: 'return vScale(1/norm(x), x)', complexity: 'O(n)',
    ref: 'Strang (2016) 单位向量', commonPitfall: '须 vecPos（严格正分量）保证 ‖x‖>0 不退化。',
    invariant: '对 vecPos：relClose(‖run(x)‖, 1, 1e-9)；锚 run((3,4))=(0.6,0.8)',
    inputs: ['x'], domains: { x: 'vecPos' },
    run: (x) => vScale(1 / norm(x), x),
    predicate: (x) => {
      const u = runOf('B036')(x);
      if (!rc(norm(u), 1, 1e-9)) return false;
      const a = runOf('B036')(vec(3, 4));
      return rc(a[0], 0.6, 1e-12) && rc(a[1], 0.8, 1e-12);
    },
  });
  reg(37, {
    name: 'norm · 严格正（vecPos）', intuition: '全正分量的向量模长严格大于 0（正定性实例）。',
    math: 'x>0 ⇒ ‖x‖ > 0', pseudocode: 'return norm(x)', complexity: 'O(n)',
    ref: 'Strang (2016) 正定性', commonPitfall: 'vecPos 每分量 ≥1e-3，故 ‖x‖≥√8·1e-3>0。',
    invariant: '对 vecPos：run(x) > 0；锚 run((1,1))=√2',
    inputs: ['x'], domains: { x: 'vecPos' },
    run: (x) => norm(x),
    predicate: (x) => runOf('B037')(x) > 0 && rc(runOf('B037')(vec(1, 1)), Math.SQRT2, 1e-12),
  });
  reg(38, {
    name: 'norm · L1 控制', intuition: 'L2 模长不超过各分量绝对值之和（L1）。',
    math: '‖x‖ ≤ Σ|xᵢ|', pseudocode: 'return norm(x)', complexity: 'O(n)',
    ref: '范数单调性 ‖·‖₂ ≤ ‖·‖₁', commonPitfall: '浮点差数个 ulp，加 1e-9 相对 epsilon。',
    invariant: '对 vecFinite：run(x) ≤ Σ|xᵢ| + ε，ε=1e-9·max(1,Σ|xᵢ|)；锚 run((3,4))=5',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => norm(x),
    predicate: (x) => {
      const run = runOf('B038');
      return run(x) <= sumAbs(x) + 1e-9 * Math.max(1, sumAbs(x)) && rc(run(vec(3, 4)), 5, 1e-12);
    },
  });
  reg(39, {
    name: 'norm · 控制最大分量', intuition: '模长不小于任一分量的绝对值（L∞ 下界）。',
    math: 'max|xᵢ| ≤ ‖x‖', pseudocode: 'return maxAbs(x)', complexity: 'O(n)',
    ref: '范数单调性 ‖·‖∞ ≤ ‖·‖₂', commonPitfall: '开方舍入可略低，加 1e-9 相对 epsilon。',
    invariant: '对 vecFinite：run(x) ≤ ‖x‖ + ε，ε=1e-9·max(1,‖x‖)；锚 run((−7,3))=7',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => maxAbs(x),
    predicate: (x) => {
      const run = runOf('B039');
      return run(x) <= norm(x) + 1e-9 * Math.max(1, norm(x)) && Object.is(run(vec(-7, 3)), 7);
    },
  });
  reg(40, {
    name: 'norm · 距离对称', intuition: '欧氏距离对两个端点对称。',
    math: '‖x − y‖ = ‖y − x‖', pseudocode: 'return norm(vSub(x,y))', complexity: 'O(n)',
    ref: 'Strang (2016) 距离函数', commonPitfall: '理论上逐位相等；仍用 relClose 归类数值族。',
    invariant: '对 vecFinite：relClose(run(x,y), ‖y−x‖, 1e-9)（异构参数序）',
    inputs: ['x', 'y'], domains: { x: 'vecFinite', y: 'vecFinite' },
    run: (x, y) => norm(vSub(x, y)),
    predicate: (x, y) => rc(runOf('B040')(x, y), norm(vSub(y, x)), 1e-9),
  });

  // ── min/max/mean/sum 归约（B041–B050）──────────────────────────────────
  reg(41, {
    name: 'sum · 顺序无关（整数域）', intuition: '整数求和与累加顺序无关（精确可结合）。',
    math: 'sum(x) = sum(reverse(x))', pseudocode: 'return sumv(x)', complexity: 'O(n)',
    ref: 'Goldberg (1991)', commonPitfall: '浮点域一般不成立，故用 vecInt。',
    invariant: '对 vecInt：Object.is(run(x), sum(rev(x)))；锚 run((1,2,3))=6',
    inputs: ['x'], domains: { x: 'vecInt' },
    run: (x) => sumv(x),
    predicate: (x) => { const run = runOf('B041'); return Object.is(run(x), sumv(vRev(x))) && Object.is(run(vec(1, 2, 3)), 6); },
  });
  reg(42, {
    name: 'max · 控制每个分量', intuition: '最大值是向量的上界（逐分量成立）。',
    math: 'max(x) ≥ xᵢ ∀i', pseudocode: 'return maxv(x)', complexity: 'O(n)',
    ref: '序统计', commonPitfall: '无',
    invariant: '对 vecFinite：∀i, run(x) ≥ x[i]；锚 run((2,7,3))=7',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => maxv(x),
    predicate: (x) => {
      const m = runOf('B042')(x);
      for (let i = 0; i < x.length; i++) if (!(m >= x[i])) return false;
      return Object.is(runOf('B042')(vec(2, 7, 3)), 7);
    },
  });
  reg(43, {
    name: 'min · 控制每个分量', intuition: '最小值是向量的下界（逐分量成立）。',
    math: 'min(x) ≤ xᵢ ∀i', pseudocode: 'return minv(x)', complexity: 'O(n)',
    ref: '序统计', commonPitfall: '无',
    invariant: '对 vecFinite：∀i, run(x) ≤ x[i]；锚 run((2,7,3))=2',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => minv(x),
    predicate: (x) => {
      const m = runOf('B043')(x);
      for (let i = 0; i < x.length; i++) if (!(m <= x[i])) return false;
      return Object.is(runOf('B043')(vec(2, 7, 3)), 2);
    },
  });
  reg(44, {
    name: 'mean · 夹于 min/max 之间', intuition: '算术平均必落在最小值与最大值之间。',
    math: 'min(x) ≤ mean(x) ≤ max(x)', pseudocode: 'return meanv(x)', complexity: 'O(n)',
    ref: 'Jensen 不等式（线性函数）', commonPitfall: '舍入可越界数个 ulp，加 1e-9 相对 epsilon。',
    invariant: '对 vecFinite：min(x)−ε ≤ run(x) ≤ max(x)+ε，ε=1e-9·max(1,|界|)；锚 run((2,4,6))=4、run((5,5,5))=5',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => meanv(x),
    predicate: (x) => {
      const lo = minv(x), hi = maxv(x), m = runOf('B044')(x);
      if (!(m >= lo - 1e-9 * Math.max(1, Math.abs(lo)) && m <= hi + 1e-9 * Math.max(1, Math.abs(hi)))) return false;
      return rc(runOf('B044')(vec(2, 4, 6)), 4, 1e-12) && Object.is(runOf('B044')(vec(5, 5, 5)), 5);
    },
  });
  reg(45, {
    name: 'max · 顺序无关', intuition: '最大值与遍历顺序无关（同多重集）。',
    math: 'max(x) = max(reverse(x))', pseudocode: 'return maxv(x)', complexity: 'O(n)',
    ref: '序统计', commonPitfall: '无', invariant: '对 vecFinite：Object.is(run(x), max(rev(x)))',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => maxv(x), predicate: (x) => Object.is(runOf('B045')(x), maxv(vRev(x))),
  });
  reg(46, {
    name: 'sum · 与 mean 换算', intuition: '总和等于个数乘平均（两种归约表示一致）。',
    math: 'sum(x) = n · mean(x)', pseudocode: 'return sumv(x)', complexity: 'O(n)',
    ref: '归约恒等式', commonPitfall: '除法再乘有舍入，用 relClose 1e-9。',
    invariant: '对 vecFinite：relClose(run(x), n·mean(x), 1e-9)；锚 run((1,2,3))=6',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => sumv(x),
    predicate: (x) => {
      const run = runOf('B046');
      return rc(run(x), x.length * meanv(x), 1e-9) && rc(run(vec(1, 2, 3)), 6, 1e-12);
    },
  });
  reg(47, {
    name: 'sum · 可加性', intuition: '和的求和等于求和之和（整数域精确）。',
    math: 'sum(x + y) = sum(x) + sum(y)', pseudocode: 'return sumv(vAdd(x,y))', complexity: 'O(n)',
    ref: '线性归约', commonPitfall: '浮点域一般不成立，故用 vecInt。',
    invariant: '对 vecInt：Object.is(run(x,y), sum(x)+sum(y))（异构路径）',
    inputs: ['x', 'y'], domains: { x: 'vecInt', y: 'vecInt' },
    run: (x, y) => sumv(vAdd(x, y)),
    predicate: (x, y) => Object.is(runOf('B047')(x, y), sumv(x) + sumv(y)),
  });
  reg(48, {
    name: 'sum · 齐次性', intuition: '数乘后求和等于标量乘原和。',
    math: 'sum(c·x) = c·sum(x)', pseudocode: 'return sumv(vScale(c,x))', complexity: 'O(n)',
    ref: '线性归约', commonPitfall: 'c ≤1e6、和 ≤8e3，积 ≤8e9 精确。另：c=0 且 sum(x)<0 时 c·sum(x)=−0 而 sum(c·x)=+0 → 用 exactNum 归一 ±0。',
    invariant: '对 int c, vecInt x：exactNum(run(c,x), c·sum(x))',
    inputs: ['c', 'x'], domains: { c: 'int', x: 'vecInt' },
    run: (c, x) => sumv(vScale(c, x)),
    predicate: (c, x) => exactNum(runOf('B048')(c, x), c * sumv(x)),
  });
  reg(49, {
    name: 'max · 取反对偶', intuition: '取反向量的最大值等于原向量最小值的负值。',
    math: 'max(−x) = −min(x)', pseudocode: 'return maxv(vNeg(x))', complexity: 'O(n)',
    ref: '序统计对偶性', commonPitfall: '符号零：两侧同为 ±0，用 === 判等。',
    invariant: '对 vecFinite：run(x) === −min(x)（异构公式）',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => maxv(vNeg(x)), predicate: (x) => runOf('B049')(x) === -minv(x),
  });
  reg(50, {
    name: 'sum · 非负向量和非负', intuition: '逐分量非负的向量，其和必然非负。',
    math: 'x ≥ 0 ⇒ sum(x) ≥ 0', pseudocode: 'return sumv(x)', complexity: 'O(n)',
    ref: '单调归约', commonPitfall: '无', invariant: '对 vecNonNeg：run(x) ≥ 0；锚 run((1,2,3))=6',
    inputs: ['x'], domains: { x: 'vecNonNeg' },
    run: (x) => sumv(x),
    predicate: (x) => { const run = runOf('B050'); return run(x) >= 0 && rc(run(vec(1, 2, 3)), 6, 1e-12); },
  });

  // ── Hadamard 积（B051–B056）────────────────────────────────────────────
  reg(51, {
    name: 'hadamard · 交换律', intuition: '逐元素乘法交换。',
    math: 'x ⊙ y = y ⊙ x', pseudocode: 'return vMul(x,y)', complexity: 'O(n)',
    ref: 'Hadamard (1899)', commonPitfall: '无',
    invariant: '对 vecInt：exactVec(run(x,y), y⊙x)；锚 run((2,3),(4,5))=(8,15)',
    inputs: ['x', 'y'], domains: { x: 'vecInt', y: 'vecInt' },
    run: (x, y) => vMul(x, y),
    predicate: (x, y) => {
      const run = runOf('B051');
      return exactVec(run(x, y), vMul(y, x)) && exactVec(run(vec(2, 3), vec(4, 5)), vec(8, 15));
    },
  });
  reg(52, {
    name: 'hadamard · 结合律', intuition: '逐元素乘法可结合（整数域精确）。',
    math: '(x ⊙ y) ⊙ z = x ⊙ (y ⊙ z)', pseudocode: 'return vMul(vMul(x,y),z)', complexity: 'O(n)',
    ref: 'Hadamard (1899)', commonPitfall: '三项积 ≤1e9，vecInt 下精确。',
    invariant: '对 vecInt：exactVec(run(x,y,z), x⊙(y⊙z))（异构结合分组）',
    inputs: ['x', 'y', 'z'], domains: { x: 'vecInt', y: 'vecInt', z: 'vecInt' },
    run: (x, y, z) => vMul(vMul(x, y), z),
    predicate: (x, y, z) => exactVec(runOf('B052')(x, y, z), vMul(x, vMul(y, z))),
  });
  reg(53, {
    name: 'hadamard · 自积非负', intuition: '向量与自身逐元素相乘得各分量平方，必非负。',
    math: '(x ⊙ x)ᵢ = xᵢ² ≥ 0', pseudocode: 'return vMul(x,x)', complexity: 'O(n)',
    ref: '平方非负性', commonPitfall: '(-0)²=+0 亦非负；域内无 NaN。',
    invariant: '对 vecFinite：∀i, run(x)[i] ≥ 0；锚 run((−3,4))=(9,16)',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => vMul(x, x),
    predicate: (x) => {
      const s = runOf('B053')(x);
      for (let i = 0; i < s.length; i++) if (!(s[i] >= 0)) return false;
      return exactVec(runOf('B053')(vec(-3, 4)), vec(9, 16));
    },
  });
  reg(54, {
    name: 'hadamard · 对 vecAdd 分配', intuition: '逐元素乘法对向量加法可分配（整数域精确）。',
    math: 'x ⊙ (y + z) = x⊙y + x⊙z', pseudocode: 'return vMul(x, vAdd(y,z))', complexity: 'O(n)',
    ref: 'Hadamard 代数', commonPitfall: '一般域舍入破坏等式，故用 vecInt。',
    invariant: '对 vecInt：exactVec(run(x,y,z), x⊙y+x⊙z)（异构展开）',
    inputs: ['x', 'y', 'z'], domains: { x: 'vecInt', y: 'vecInt', z: 'vecInt' },
    run: (x, y, z) => vMul(x, vAdd(y, z)),
    predicate: (x, y, z) => exactVec(runOf('B054')(x, y, z), vAdd(vMul(x, y), vMul(x, z))),
  });
  reg(55, {
    name: 'hadamard · 标量结合', intuition: '数乘可与逐元素积交换结合位置。',
    math: '(c·x) ⊙ y = c·(x ⊙ y)', pseudocode: 'return vMul(vScale(c,x), y)', complexity: 'O(n)',
    ref: 'Hadamard 代数', commonPitfall: 'c·xᵢ·yᵢ ≤1e9，smallInt 下精确。',
    invariant: '对 smallInt c, vecInt x,y：exactVec(run(c,x,y), c(x⊙y))（异构结合）',
    inputs: ['c', 'x', 'y'], domains: { c: 'smallInt', x: 'vecInt', y: 'vecInt' },
    run: (c, x, y) => vMul(vScale(c, x), y),
    predicate: (c, x, y) => exactVec(runOf('B055')(c, x, y), vScale(c, vMul(x, y))),
  });
  reg(56, {
    name: 'hadamard · 单位元', intuition: '全 1 向量是逐元素乘法的单位元。',
    math: 'x ⊙ 1 = x', pseudocode: 'return vMul(x, ones(n))', complexity: 'O(n)',
    ref: 'Hadamard 代数', commonPitfall: '无', invariant: '对 vecInt：exactVec(run(x), x)（oracle 为输入本身）',
    inputs: ['x'], domains: { x: 'vecInt' },
    run: (x) => vMul(x, ones(x.length)),
    predicate: (x) => exactVec(runOf('B056')(x), x),
  });

  // ── axpy（B057–B060）───────────────────────────────────────────────────
  reg(57, {
    name: 'axpy · 增量验证', intuition: 'y ← a·x + y 后，新旧向量之差恰为 a·x。',
    math: '(y + a·x) − y = a·x', pseudocode: 'return vAdd(vScale(a,x), y)', complexity: 'O(n)',
    ref: 'Lawson et al. (1979) BLAS axpy', commonPitfall: '整数域下逐步精确（≤1e6+1e3）。',
    invariant: '对 smallInt a, vecInt x,y：exactVec(run(a,x,y)−y, ax)（异构回减）',
    inputs: ['a', 'x', 'y'], domains: { a: 'smallInt', x: 'vecInt', y: 'vecInt' },
    run: (a, x, y) => vAdd(vScale(a, x), y),
    predicate: (a, x, y) => exactVec(vSub(runOf('B057')(a, x, y), y), vScale(a, x)),
  });
  reg(58, {
    name: 'axpy · 两次复合', intuition: '先后叠加 a·x 与 b·x，等价于一次性叠加 (a+b)·x。',
    math: '(y + a·x) + b·x = y + (a+b)·x', pseudocode: 'return vAdd(vScale(a,x), y)', complexity: 'O(n)',
    ref: 'BLAS axpy 复合', commonPitfall: '整数域精确。',
    invariant: '对 smallInt a,b, vecInt x,y：exactVec(run(a,b,x,y), y+(a+b)x)（异构分组）',
    inputs: ['a', 'b', 'x', 'y'], domains: { a: 'smallInt', b: 'smallInt', x: 'vecInt', y: 'vecInt' },
    run: (a, b, x, y) => vAdd(vAdd(vScale(a, x), vScale(b, x)), y),
    predicate: (a, b, x, y) => exactVec(runOf('B058')(a, b, x, y), vAdd(vScale(a + b, x), y)),
  });
  reg(59, {
    name: 'axpy · 整体缩放交换', intuition: '先 axpy 再整体数乘，等于分别数乘后相加。',
    math: 'c·(y + a·x) = c·y + (c·a)·x', pseudocode: 'return vScale(c, vAdd(vScale(a,x),y))', complexity: 'O(n)',
    ref: 'BLAS 与数乘交换', commonPitfall: 'c·a·x ≤1e9，smallInt 下精确。',
    invariant: '对 smallInt c,a, vecInt x,y：exactVec(run(c,a,x,y), cy+(ca)x)（异构展开）',
    inputs: ['c', 'a', 'x', 'y'], domains: { c: 'smallInt', a: 'smallInt', x: 'vecInt', y: 'vecInt' },
    run: (c, a, x, y) => vScale(c, vAdd(vScale(a, x), y)),
    predicate: (c, a, x, y) => exactVec(
      runOf('B059')(c, a, x, y),
      vAdd(vScale(c, y), vScale(c * a, x))),
  });
  reg(60, {
    name: 'axpy · 点积双线性', intuition: '对更新后的向量做点积，满足双线性拆分。',
    math: '(y + a·x)·x = y·x + a·(x·x)', pseudocode: 'return dot(vAdd(vScale(a,x),y), x)', complexity: 'O(n)',
    ref: 'Strang (2016) 双线性', commonPitfall: 'a·(x·x) ≤8e9，整数域精确。',
    invariant: '对 smallInt a, vecInt x,y：Object.is(run(a,x,y), y·x + a(x·x))；锚 run(2,(1,2,3),0)=28',
    inputs: ['a', 'x', 'y'], domains: { a: 'smallInt', x: 'vecInt', y: 'vecInt' },
    run: (a, x, y) => dot(vAdd(vScale(a, x), y), x),
    predicate: (a, x, y) => {
      const run = runOf('B060');
      return Object.is(run(a, x, y), dot(y, x) + a * dot(x, x))
        && Object.is(run(2, vec(1, 2, 3), vec(0, 0, 0)), 28);
    },
  });

  // ── clamp / relu（B061–B064）──────────────────────────────────────────
  reg(61, {
    name: 'clamp · 值域夹取', intuition: 'clamp 把每个分量压进 [lo,hi]（边界自动排序，容错输入）。',
    math: 'clamp(x,lo,hi)ᵢ ∈ [min(lo,hi), max(lo,hi)]', pseudocode: 'return clampVec(x, lo, hi)', complexity: 'O(n)',
    ref: 'NumPy clip 语义', commonPitfall: 'lo>hi 时先归一化边界，否则夹取结果无定义。',
    invariant: '对 vecFinite x, 标量 lo,hi：∀i, min(lo,hi) ≤ run(x,lo,hi)[i] ≤ max(lo,hi)；锚 run((−5,0,5),0,1)=(0,0,1)',
    inputs: ['x', 'lo', 'hi'], domains: { x: 'vecFinite', lo: 'finite', hi: 'finite' },
    run: (x, lo, hi) => clampVec(x, Math.min(lo, hi), Math.max(lo, hi)),
    predicate: (x, lo, hi) => {
      const a = Math.min(lo, hi), b = Math.max(lo, hi);
      const c = runOf('B061')(x, lo, hi);
      for (let i = 0; i < c.length; i++) if (c[i] < a || c[i] > b) return false;
      const d = runOf('B061')(vec(-5, 0, 5), 0, 1);
      return d[0] === 0 && d[1] === 0 && d[2] === 1;
    },
  });
  reg(62, {
    name: 'clamp · 幂等', intuition: '已夹取到值域内的向量再夹取一次不变。',
    math: 'clamp(clamp(x)) = clamp(x)', pseudocode: 'return clampVec(x, lo, hi)', complexity: 'O(n)',
    ref: 'NumPy clip 幂等性', commonPitfall: 'min/max 对已越界值返回边界表示，逐位不变。',
    invariant: '对 vecFinite x, 标量 lo,hi：exactVec(run(run(x,lo,hi),lo,hi), run(x,lo,hi))（幂等经 run 自复合）',
    inputs: ['x', 'lo', 'hi'], domains: { x: 'vecFinite', lo: 'finite', hi: 'finite' },
    run: (x, lo, hi) => clampVec(x, Math.min(lo, hi), Math.max(lo, hi)),
    predicate: (x, lo, hi) => {
      const run = runOf('B062'), c = run(x, lo, hi);
      return exactVec(run(c, lo, hi), c);
    },
  });
  reg(63, {
    name: 'clamp · 保序', intuition: 'clamp 是单调算子：逐分量小的向量夹取后仍逐分量小。',
    math: 'x ≤ y ⇒ clamp(x) ≤ clamp(y)', pseudocode: 'return clampVec(x, lo, hi)', complexity: 'O(n)',
    ref: '单调函数保持序', commonPitfall: '前提不满足时判据空虚真。',
    invariant: '对 vecFinite x,y, 标量 lo,hi：∀i (x[i]≤y[i]) ⇒ run(x,…)[i]≤run(y,…)[i]（两侧 run 耦合）',
    inputs: ['x', 'y', 'lo', 'hi'], domains: { x: 'vecFinite', y: 'vecFinite', lo: 'finite', hi: 'finite' },
    run: (x, y, lo, hi) => clampVec(x, Math.min(lo, hi), Math.max(lo, hi)),
    predicate: (x, y, lo, hi) => {
      const cx = runOf('B063')(x, y, lo, hi), cy = runOf('B063')(y, x, lo, hi);
      for (let i = 0; i < x.length; i++) if (x[i] <= y[i] && !(cx[i] <= cy[i])) return false;
      return true;
    },
  });
  reg(64, {
    name: 'relu · 非负', intuition: 'ReLU 截断负半轴，输出逐分量非负。',
    math: 'relu(x)ᵢ = max(0, xᵢ) ≥ 0', pseudocode: 'return reluVec(x)', complexity: 'O(n)',
    ref: 'Nair & Hinton (2010) ReLU', commonPitfall: '-0 输入得 +0，仍非负。',
    invariant: '对 vecFinite：∀i, run(x)[i] ≥ 0；锚 run((−2,3))=(0,3)',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => reluVec(x),
    predicate: (x) => {
      const r = runOf('B064')(x);
      for (let i = 0; i < r.length; i++) if (!(r[i] >= 0)) return false;
      return exactVec(runOf('B064')(vec(-2, 3)), vec(0, 3));
    },
  });
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

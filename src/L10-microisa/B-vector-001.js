/* ============================================================================
 * L10 微指令 · B 域（向量运算）· 第 2/3 批（B033–B064）
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
 * ==========================================================================*/
;(function (NS) {
  'use strict';
  NS.L10 = NS.L10 || {};
  function reg(i, o) { NS.L10.register('B', i, o); }
  const rc = (a, b, t) => NS.L10.relClose(a, b, t);

  // ── 向量小工具（纯函数）────────────────────────────────────────────────
  function ones(n) { const o = new Float64Array(n); o.fill(1); return o; }
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
    invariant: '对 vecFinite：relClose(‖−x‖, ‖x‖, 1e-9)',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => norm(vNeg(x)), predicate: (x) => rc(norm(vNeg(x)), norm(x), 1e-9),
  });
  reg(34, {
    name: 'norm · 齐次缩放', intuition: '向量整体缩放 c 倍，模长缩放 |c| 倍。',
    math: '‖c·x‖ = |c|·‖x‖', pseudocode: 'return norm(vScale(c,x))', complexity: 'O(n)',
    ref: 'Strang (2016) 范数齐次性', commonPitfall: '需容差；c 取 smallInt 防溢出。',
    invariant: '对 smallInt c, vecSmall x：relClose(‖cx‖, |c|‖x‖, 1e-9)',
    inputs: ['c', 'x'], domains: { c: 'smallInt', x: 'vecSmall' },
    run: (c, x) => norm(vScale(c, x)),
    predicate: (c, x) => rc(norm(vScale(c, x)), Math.abs(c) * norm(x), 1e-9),
  });
  reg(35, {
    name: 'norm · 三角不等式', intuition: '两边之和大于第三边：‖x+y‖ 不超两模长之和。',
    math: '‖x + y‖ ≤ ‖x‖ + ‖y‖', pseudocode: 'return norm(vAdd(x,y))', complexity: 'O(n)',
    ref: 'Strang (2016) 范数三角不等式', commonPitfall: '浮点可差数个 ulp，加 1e-9 相对 epsilon。',
    invariant: '对 vecSmall：‖x+y‖ ≤ ‖x‖+‖y‖+ε，ε=1e-9·max(1,‖x‖+‖y‖)',
    inputs: ['x', 'y'], domains: { x: 'vecSmall', y: 'vecSmall' },
    run: (x, y) => norm(vAdd(x, y)),
    predicate: (x, y) => norm(vAdd(x, y)) <= norm(x) + norm(y) + 1e-9 * Math.max(1, norm(x) + norm(y)),
  });
  reg(36, {
    name: 'norm · 单位化后模为 1', intuition: '向量除以自身模长即得单位向量（方向保持、长度归一）。',
    math: '‖x/‖x‖‖ = 1', pseudocode: 'return vScale(1/norm(x), x)', complexity: 'O(n)',
    ref: 'Strang (2016) 单位向量', commonPitfall: '须 vecPos（严格正分量）保证 ‖x‖>0 不退化。',
    invariant: '对 vecPos：relClose(‖x/‖x‖‖, 1, 1e-9)',
    inputs: ['x'], domains: { x: 'vecPos' },
    run: (x) => vScale(1 / norm(x), x),
    predicate: (x) => rc(norm(vScale(1 / norm(x), x)), 1, 1e-9),
  });
  reg(37, {
    name: 'norm · 严格正（vecPos）', intuition: '全正分量的向量模长严格大于 0（正定性实例）。',
    math: 'x>0 ⇒ ‖x‖ > 0', pseudocode: 'return norm(x)', complexity: 'O(n)',
    ref: 'Strang (2016) 正定性', commonPitfall: 'vecPos 每分量 ≥1e-3，故 ‖x‖≥√8·1e-3>0。',
    invariant: '对 vecPos：‖x‖ > 0',
    inputs: ['x'], domains: { x: 'vecPos' },
    run: (x) => norm(x), predicate: (x) => norm(x) > 0,
  });
  reg(38, {
    name: 'norm · L1 控制', intuition: 'L2 模长不超过各分量绝对值之和（L1）。',
    math: '‖x‖ ≤ Σ|xᵢ|', pseudocode: 'return norm(x)', complexity: 'O(n)',
    ref: '范数单调性 ‖·‖₂ ≤ ‖·‖₁', commonPitfall: '浮点差数个 ulp，加 1e-9 相对 epsilon。',
    invariant: '对 vecFinite：‖x‖ ≤ Σ|xᵢ| + ε，ε=1e-9·max(1,Σ|xᵢ|)',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => norm(x),
    predicate: (x) => norm(x) <= sumAbs(x) + 1e-9 * Math.max(1, sumAbs(x)),
  });
  reg(39, {
    name: 'norm · 控制最大分量', intuition: '模长不小于任一分量的绝对值（L∞ 下界）。',
    math: 'max|xᵢ| ≤ ‖x‖', pseudocode: 'return maxAbs(x)', complexity: 'O(n)',
    ref: '范数单调性 ‖·‖∞ ≤ ‖·‖₂', commonPitfall: '开方舍入可略低，加 1e-9 相对 epsilon。',
    invariant: '对 vecFinite：max|xᵢ| ≤ ‖x‖ + ε，ε=1e-9·max(1,‖x‖)',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => maxAbs(x),
    predicate: (x) => maxAbs(x) <= norm(x) + 1e-9 * Math.max(1, norm(x)),
  });
  reg(40, {
    name: 'norm · 距离对称', intuition: '欧氏距离对两个端点对称。',
    math: '‖x − y‖ = ‖y − x‖', pseudocode: 'return norm(vSub(x,y))', complexity: 'O(n)',
    ref: 'Strang (2016) 距离函数', commonPitfall: '理论上逐位相等；仍用 relClose 归类数值族。',
    invariant: '对 vecFinite：relClose(‖x−y‖, ‖y−x‖, 1e-9)',
    inputs: ['x', 'y'], domains: { x: 'vecFinite', y: 'vecFinite' },
    run: (x, y) => norm(vSub(x, y)),
    predicate: (x, y) => rc(norm(vSub(x, y)), norm(vSub(y, x)), 1e-9),
  });

  // ── min/max/mean/sum 归约（B041–B050）──────────────────────────────────
  reg(41, {
    name: 'sum · 顺序无关（整数域）', intuition: '整数求和与累加顺序无关（精确可结合）。',
    math: 'sum(x) = sum(reverse(x))', pseudocode: 'return sumv(x)', complexity: 'O(n)',
    ref: 'Goldberg (1991)', commonPitfall: '浮点域一般不成立，故用 vecInt。',
    invariant: '对 vecInt：Object.is(sum(x), sum(rev(x)))',
    inputs: ['x'], domains: { x: 'vecInt' },
    run: (x) => sumv(x), predicate: (x) => Object.is(sumv(x), sumv(vRev(x))),
  });
  reg(42, {
    name: 'max · 控制每个分量', intuition: '最大值是向量的上界（逐分量成立）。',
    math: 'max(x) ≥ xᵢ ∀i', pseudocode: 'return maxv(x)', complexity: 'O(n)',
    ref: '序统计', commonPitfall: '无', invariant: '对 vecFinite：∀i, max(x) ≥ x[i]',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => maxv(x),
    predicate: (x) => { const m = maxv(x); for (let i = 0; i < x.length; i++) if (!(m >= x[i])) return false; return true; },
  });
  reg(43, {
    name: 'min · 控制每个分量', intuition: '最小值是向量的下界（逐分量成立）。',
    math: 'min(x) ≤ xᵢ ∀i', pseudocode: 'return minv(x)', complexity: 'O(n)',
    ref: '序统计', commonPitfall: '无', invariant: '对 vecFinite：∀i, min(x) ≤ x[i]',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => minv(x),
    predicate: (x) => { const m = minv(x); for (let i = 0; i < x.length; i++) if (!(m <= x[i])) return false; return true; },
  });
  reg(44, {
    name: 'mean · 夹于 min/max 之间', intuition: '算术平均必落在最小值与最大值之间。',
    math: 'min(x) ≤ mean(x) ≤ max(x)', pseudocode: 'return meanv(x)', complexity: 'O(n)',
    ref: 'Jensen 不等式（线性函数）', commonPitfall: '舍入可越界数个 ulp，加 1e-9 相对 epsilon。',
    invariant: '对 vecFinite：min(x)−ε ≤ mean(x) ≤ max(x)+ε，ε=1e-9·max(1,|界|)',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => meanv(x),
    predicate: (x) => {
      const lo = minv(x), hi = maxv(x), m = meanv(x);
      return m >= lo - 1e-9 * Math.max(1, Math.abs(lo)) && m <= hi + 1e-9 * Math.max(1, Math.abs(hi));
    },
  });
  reg(45, {
    name: 'max · 顺序无关', intuition: '最大值与遍历顺序无关（同多重集）。',
    math: 'max(x) = max(reverse(x))', pseudocode: 'return maxv(x)', complexity: 'O(n)',
    ref: '序统计', commonPitfall: '无', invariant: '对 vecFinite：Object.is(max(x), max(rev(x)))',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => maxv(x), predicate: (x) => Object.is(maxv(x), maxv(vRev(x))),
  });
  reg(46, {
    name: 'sum · 与 mean 换算', intuition: '总和等于个数乘平均（两种归约表示一致）。',
    math: 'sum(x) = n · mean(x)', pseudocode: 'return sumv(x)', complexity: 'O(n)',
    ref: '归约恒等式', commonPitfall: '除法再乘有舍入，用 relClose 1e-9。',
    invariant: '对 vecFinite：relClose(sum(x), n·mean(x), 1e-9)',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => sumv(x),
    predicate: (x) => rc(sumv(x), x.length * meanv(x), 1e-9),
  });
  reg(47, {
    name: 'sum · 可加性', intuition: '和的求和等于求和之和（整数域精确）。',
    math: 'sum(x + y) = sum(x) + sum(y)', pseudocode: 'return sumv(vAdd(x,y))', complexity: 'O(n)',
    ref: '线性归约', commonPitfall: '浮点域一般不成立，故用 vecInt。',
    invariant: '对 vecInt：Object.is(sum(x+y), sum(x)+sum(y))',
    inputs: ['x', 'y'], domains: { x: 'vecInt', y: 'vecInt' },
    run: (x, y) => sumv(vAdd(x, y)),
    predicate: (x, y) => Object.is(sumv(vAdd(x, y)), sumv(x) + sumv(y)),
  });
  reg(48, {
    name: 'sum · 齐次性', intuition: '数乘后求和等于标量乘原和。',
    math: 'sum(c·x) = c·sum(x)', pseudocode: 'return sumv(vScale(c,x))', complexity: 'O(n)',
    ref: '线性归约', commonPitfall: 'c ≤1e6、和 ≤8e3，积 ≤8e9 精确。另：c=0 且 sum(x)<0 时 c·sum(x)=−0 而 sum(c·x)=+0 → 用 exactNum 归一 ±0。',
    invariant: '对 int c, vecInt x：exactNum(sum(cx), c·sum(x))',
    inputs: ['c', 'x'], domains: { c: 'int', x: 'vecInt' },
    run: (c, x) => sumv(vScale(c, x)),
    predicate: (c, x) => exactNum(sumv(vScale(c, x)), c * sumv(x)),
  });
  reg(49, {
    name: 'max · 取反对偶', intuition: '取反向量的最大值等于原向量最小值的负值。',
    math: 'max(−x) = −min(x)', pseudocode: 'return maxv(vNeg(x))', complexity: 'O(n)',
    ref: '序统计对偶性', commonPitfall: '符号零：两侧同为 ±0，用 === 判等。',
    invariant: '对 vecFinite：max(−x) === −min(x)',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => maxv(vNeg(x)), predicate: (x) => maxv(vNeg(x)) === -minv(x),
  });
  reg(50, {
    name: 'sum · 非负向量和非负', intuition: '逐分量非负的向量，其和必然非负。',
    math: 'x ≥ 0 ⇒ sum(x) ≥ 0', pseudocode: 'return sumv(x)', complexity: 'O(n)',
    ref: '单调归约', commonPitfall: '无', invariant: '对 vecNonNeg：sum(x) ≥ 0',
    inputs: ['x'], domains: { x: 'vecNonNeg' },
    run: (x) => sumv(x), predicate: (x) => sumv(x) >= 0,
  });

  // ── Hadamard 积（B051–B056）────────────────────────────────────────────
  reg(51, {
    name: 'hadamard · 交换律', intuition: '逐元素乘法交换。',
    math: 'x ⊙ y = y ⊙ x', pseudocode: 'return vMul(x,y)', complexity: 'O(n)',
    ref: 'Hadamard (1899)', commonPitfall: '无', invariant: '对 vecInt：exactVec(x⊙y, y⊙x)',
    inputs: ['x', 'y'], domains: { x: 'vecInt', y: 'vecInt' },
    run: (x, y) => vMul(x, y), predicate: (x, y) => exactVec(vMul(x, y), vMul(y, x)),
  });
  reg(52, {
    name: 'hadamard · 结合律', intuition: '逐元素乘法可结合（整数域精确）。',
    math: '(x ⊙ y) ⊙ z = x ⊙ (y ⊙ z)', pseudocode: 'return vMul(vMul(x,y),z)', complexity: 'O(n)',
    ref: 'Hadamard (1899)', commonPitfall: '三项积 ≤1e9，vecInt 下精确。',
    invariant: '对 vecInt：exactVec((x⊙y)⊙z, x⊙(y⊙z))',
    inputs: ['x', 'y', 'z'], domains: { x: 'vecInt', y: 'vecInt', z: 'vecInt' },
    run: (x, y, z) => vMul(vMul(x, y), z),
    predicate: (x, y, z) => exactVec(vMul(vMul(x, y), z), vMul(x, vMul(y, z))),
  });
  reg(53, {
    name: 'hadamard · 自积非负', intuition: '向量与自身逐元素相乘得各分量平方，必非负。',
    math: '(x ⊙ x)ᵢ = xᵢ² ≥ 0', pseudocode: 'return vMul(x,x)', complexity: 'O(n)',
    ref: '平方非负性', commonPitfall: '(-0)²=+0 亦非负；域内无 NaN。',
    invariant: '对 vecFinite：∀i, (x⊙x)[i] ≥ 0',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => vMul(x, x),
    predicate: (x) => { const s = vMul(x, x); for (let i = 0; i < s.length; i++) if (!(s[i] >= 0)) return false; return true; },
  });
  reg(54, {
    name: 'hadamard · 对 vecAdd 分配', intuition: '逐元素乘法对向量加法可分配（整数域精确）。',
    math: 'x ⊙ (y + z) = x⊙y + x⊙z', pseudocode: 'return vMul(x, vAdd(y,z))', complexity: 'O(n)',
    ref: 'Hadamard 代数', commonPitfall: '一般域舍入破坏等式，故用 vecInt。',
    invariant: '对 vecInt：exactVec(x⊙(y+z), x⊙y+x⊙z)',
    inputs: ['x', 'y', 'z'], domains: { x: 'vecInt', y: 'vecInt', z: 'vecInt' },
    run: (x, y, z) => vMul(x, vAdd(y, z)),
    predicate: (x, y, z) => exactVec(vMul(x, vAdd(y, z)), vAdd(vMul(x, y), vMul(x, z))),
  });
  reg(55, {
    name: 'hadamard · 标量结合', intuition: '数乘可与逐元素积交换结合位置。',
    math: '(c·x) ⊙ y = c·(x ⊙ y)', pseudocode: 'return vMul(vScale(c,x), y)', complexity: 'O(n)',
    ref: 'Hadamard 代数', commonPitfall: 'c·xᵢ·yᵢ ≤1e9，smallInt 下精确。',
    invariant: '对 smallInt c, vecInt x,y：exactVec((cx)⊙y, c(x⊙y))',
    inputs: ['c', 'x', 'y'], domains: { c: 'smallInt', x: 'vecInt', y: 'vecInt' },
    run: (c, x, y) => vMul(vScale(c, x), y),
    predicate: (c, x, y) => exactVec(vMul(vScale(c, x), y), vScale(c, vMul(x, y))),
  });
  reg(56, {
    name: 'hadamard · 单位元', intuition: '全 1 向量是逐元素乘法的单位元。',
    math: 'x ⊙ 1 = x', pseudocode: 'return vMul(x, ones(n))', complexity: 'O(n)',
    ref: 'Hadamard 代数', commonPitfall: '无', invariant: '对 vecInt：exactVec(x⊙1, x)',
    inputs: ['x'], domains: { x: 'vecInt' },
    run: (x) => vMul(x, ones(x.length)),
    predicate: (x) => exactVec(vMul(x, ones(x.length)), x),
  });

  // ── axpy（B057–B060）───────────────────────────────────────────────────
  reg(57, {
    name: 'axpy · 增量验证', intuition: 'y ← a·x + y 后，新旧向量之差恰为 a·x。',
    math: '(y + a·x) − y = a·x', pseudocode: 'return vAdd(vScale(a,x), y)', complexity: 'O(n)',
    ref: 'Lawson et al. (1979) BLAS axpy', commonPitfall: '整数域下逐步精确（≤1e6+1e3）。',
    invariant: '对 smallInt a, vecInt x,y：exactVec((y+ax)−y, ax)',
    inputs: ['a', 'x', 'y'], domains: { a: 'smallInt', x: 'vecInt', y: 'vecInt' },
    run: (a, x, y) => vAdd(vScale(a, x), y),
    predicate: (a, x, y) => exactVec(vSub(vAdd(vScale(a, x), y), y), vScale(a, x)),
  });
  reg(58, {
    name: 'axpy · 两次复合', intuition: '先后叠加 a·x 与 b·x，等价于一次性叠加 (a+b)·x。',
    math: '(y + a·x) + b·x = y + (a+b)·x', pseudocode: 'return vAdd(vScale(a,x), y)', complexity: 'O(n)',
    ref: 'BLAS axpy 复合', commonPitfall: '整数域精确。',
    invariant: '对 smallInt a,b, vecInt x,y：exactVec(y+ax+bx, y+(a+b)x)',
    inputs: ['a', 'b', 'x', 'y'], domains: { a: 'smallInt', b: 'smallInt', x: 'vecInt', y: 'vecInt' },
    run: (a, b, x, y) => vAdd(vAdd(vScale(a, x), vScale(b, x)), y),
    predicate: (a, b, x, y) => exactVec(
      vAdd(vAdd(vScale(a, x), vScale(b, x)), y),
      vAdd(vScale(a + b, x), y)),
  });
  reg(59, {
    name: 'axpy · 整体缩放交换', intuition: '先 axpy 再整体数乘，等于分别数乘后相加。',
    math: 'c·(y + a·x) = c·y + (c·a)·x', pseudocode: 'return vScale(c, vAdd(vScale(a,x),y))', complexity: 'O(n)',
    ref: 'BLAS 与数乘交换', commonPitfall: 'c·a·x ≤1e9，smallInt 下精确。',
    invariant: '对 smallInt c,a, vecInt x,y：exactVec(c(y+ax), cy+(ca)x)',
    inputs: ['c', 'a', 'x', 'y'], domains: { c: 'smallInt', a: 'smallInt', x: 'vecInt', y: 'vecInt' },
    run: (c, a, x, y) => vScale(c, vAdd(vScale(a, x), y)),
    predicate: (c, a, x, y) => exactVec(
      vScale(c, vAdd(vScale(a, x), y)),
      vAdd(vScale(c, y), vScale(c * a, x))),
  });
  reg(60, {
    name: 'axpy · 点积双线性', intuition: '对更新后的向量做点积，满足双线性拆分。',
    math: '(y + a·x)·x = y·x + a·(x·x)', pseudocode: 'return dot(vAdd(vScale(a,x),y), x)', complexity: 'O(n)',
    ref: 'Strang (2016) 双线性', commonPitfall: 'a·(x·x) ≤8e9，整数域精确。',
    invariant: '对 smallInt a, vecInt x,y：Object.is((y+ax)·x, y·x + a(x·x))',
    inputs: ['a', 'x', 'y'], domains: { a: 'smallInt', x: 'vecInt', y: 'vecInt' },
    run: (a, x, y) => dot(vAdd(vScale(a, x), y), x),
    predicate: (a, x, y) => Object.is(dot(vAdd(vScale(a, x), y), x), dot(y, x) + a * dot(x, x)),
  });

  // ── clamp / relu（B061–B064）───────────────────────────────────────────
  reg(61, {
    name: 'clamp · 值域夹取', intuition: 'clamp 把每个分量压进 [lo,hi]（边界自动排序，容错输入）。',
    math: 'clamp(x,lo,hi)ᵢ ∈ [min(lo,hi), max(lo,hi)]', pseudocode: 'return clampVec(x, lo, hi)', complexity: 'O(n)',
    ref: 'NumPy clip 语义', commonPitfall: 'lo>hi 时先归一化边界，否则夹取结果无定义。',
    invariant: '对 vecFinite x, 标量 lo,hi：∀i, min(lo,hi) ≤ clampᵢ ≤ max(lo,hi)',
    inputs: ['x', 'lo', 'hi'], domains: { x: 'vecFinite', lo: 'finite', hi: 'finite' },
    run: (x, lo, hi) => clampVec(x, Math.min(lo, hi), Math.max(lo, hi)),
    predicate: (x, lo, hi) => {
      const a = Math.min(lo, hi), b = Math.max(lo, hi);
      const c = clampVec(x, a, b);
      for (let i = 0; i < c.length; i++) if (c[i] < a || c[i] > b) return false;
      return true;
    },
  });
  reg(62, {
    name: 'clamp · 幂等', intuition: '已夹取到值域内的向量再夹取一次不变。',
    math: 'clamp(clamp(x)) = clamp(x)', pseudocode: 'return clampVec(x, lo, hi)', complexity: 'O(n)',
    ref: 'NumPy clip 幂等性', commonPitfall: 'min/max 对已越界值返回边界表示，逐位不变。',
    invariant: '对 vecFinite x, 标量 lo,hi：exactVec(clamp(clamp(x)), clamp(x))',
    inputs: ['x', 'lo', 'hi'], domains: { x: 'vecFinite', lo: 'finite', hi: 'finite' },
    run: (x, lo, hi) => clampVec(x, Math.min(lo, hi), Math.max(lo, hi)),
    predicate: (x, lo, hi) => {
      const a = Math.min(lo, hi), b = Math.max(lo, hi);
      const c = clampVec(x, a, b);
      return exactVec(clampVec(c, a, b), c);
    },
  });
  reg(63, {
    name: 'clamp · 保序', intuition: 'clamp 是单调算子：逐分量小的向量夹取后仍逐分量小。',
    math: 'x ≤ y ⇒ clamp(x) ≤ clamp(y)', pseudocode: 'return clampVec(x, lo, hi)', complexity: 'O(n)',
    ref: '单调函数保持序', commonPitfall: '前提不满足时判据空虚真。',
    invariant: '对 vecFinite x,y, 标量 lo,hi：∀i (x[i]≤y[i]) ⇒ clamp(x)[i]≤clamp(y)[i]',
    inputs: ['x', 'y', 'lo', 'hi'], domains: { x: 'vecFinite', y: 'vecFinite', lo: 'finite', hi: 'finite' },
    run: (x, y, lo, hi) => clampVec(x, Math.min(lo, hi), Math.max(lo, hi)),
    predicate: (x, y, lo, hi) => {
      const a = Math.min(lo, hi), b = Math.max(lo, hi);
      const cx = clampVec(x, a, b), cy = clampVec(y, a, b);
      for (let i = 0; i < x.length; i++) if (x[i] <= y[i] && !(cx[i] <= cy[i])) return false;
      return true;
    },
  });
  reg(64, {
    name: 'relu · 非负', intuition: 'ReLU 截断负半轴，输出逐分量非负。',
    math: 'relu(x)ᵢ = max(0, xᵢ) ≥ 0', pseudocode: 'return reluVec(x)', complexity: 'O(n)',
    ref: 'Nair & Hinton (2010) ReLU', commonPitfall: '-0 输入得 +0，仍非负。',
    invariant: '对 vecFinite：∀i, relu(x)[i] ≥ 0',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => reluVec(x),
    predicate: (x) => { const r = reluVec(x); for (let i = 0; i < r.length; i++) if (!(r[i] >= 0)) return false; return true; },
  });
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

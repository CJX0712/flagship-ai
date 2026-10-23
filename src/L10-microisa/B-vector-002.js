/* ============================================================================
 * L10 微指令 · B 域（向量运算）· 第 3/4 批（B065–B096）
 * ----------------------------------------------------------------------------
 * 所属层：L10-microIsa  域：B（向量）
 * 允许依赖的层：L9（rng/assert/special）、L10（registry/invariant）
 * 主要 @ref：Nair & Hinton (2010) ReLU；Strang (2016) 投影/平行四边形/余弦；
 *           统计矩计算式；NumPy normalize/cumsum 语义
 *
 * 涵盖：relu 与逐元素函数（B065–B069）、Hadamard 代数恒等式（B070–B071）、
 *       前缀和/统计矩（B072–B081）、几何（B082–B086）、绝对值/反转/除法/
 *       平方极值/范数夹挤/幂等（B087–B096）。
 * 判据：整数向量域 Object.is 逐位精确；数值族 relClose(1e-9)；
 *       不等式族加 1e-9 相对 epsilon 吸收舍入（保持真实数学性质）。
 *
 * ── 谓词纪律（本批全部重写，2026-10-23）─────────────────────────────────
 * 门禁只调 op.predicate，从不调 op.run（op-harness.js:43、probe-microisa.js:106）。
 * 原 32 项谓词全部用本地 helper（reluVec/cumsum/variance/…）内联复算 run 定义式，
 * run 若损坏门禁完全失明（假绿）。现全部改为：先经 runOf('Bxxx') 调 run 本体，
 * 再对 run 的产出断言不变量；oracle 取异构公式、输入导出值、常数锚、定点锚；
 * 幂等类经 run 自复合验证。B080 的 oracle 改用位移 raw-moment（mean((x−s)²)
 * −mean(x−s)²，s=x[0]）——原 mean(x²)−mean(x)² 计算式在近常数向量下相消，
 * 边缘用例（[1,…,1,1.001]）会伪红，位移式既不同公式又无数值相消。
 * ==========================================================================*/
;(function (NS) {
  'use strict';
  NS.L10 = NS.L10 || {};
  function reg(i, o) { NS.L10.register('B', i, o); }
  function runOf(key) { const o = NS.L10.ops[key]; return o && typeof o.run === 'function' ? o.run : null; }
  const rc = (a, b, t) => NS.L10.relClose(a, b, t);

  // ── 向量小工具（纯函数）────────────────────────────────────────────────
  function ones(n) { const o = new Float64Array(n); o.fill(1); return o; }
  function e0(n) { const o = new Float64Array(n); o[0] = 1; return o; }
  function vec() { return Float64Array.from(arguments); }
  function vAdd(x, y) { const n = x.length, o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = x[i] + y[i]; return o; }
  function vSub(x, y) { const n = x.length, o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = x[i] - y[i]; return o; }
  function vNeg(x) { const n = x.length, o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = -x[i]; return o; }
  function vScale(c, x) { const n = x.length, o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = c * x[i]; return o; }
  function vMul(x, y) { const n = x.length, o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = x[i] * y[i]; return o; }
  function vDiv(x, y) { const n = x.length, o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = x[i] / y[i]; return o; }
  function vAbs(x) { const n = x.length, o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = Math.abs(x[i]); return o; }
  function vRev(x) { const n = x.length, o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = x[n - 1 - i]; return o; }
  function vSqrt(x) { const n = x.length, o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = Math.sqrt(x[i]); return o; }
  function vRecip(x) { const n = x.length, o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = 1 / x[i]; return o; }
  function dot(x, y) { let s = 0; for (let i = 0; i < x.length; i++) s += x[i] * y[i]; return s; }
  function norm(x) { return Math.sqrt(dot(x, x)); }
  function sumv(x) { let s = 0; for (let i = 0; i < x.length; i++) s += x[i]; return s; }
  function maxv(x) { let m = x[0]; for (let i = 1; i < x.length; i++) if (x[i] > m) m = x[i]; return m; }
  function minv(x) { let m = x[0]; for (let i = 1; i < x.length; i++) if (x[i] < m) m = x[i]; return m; }
  function meanv(x) { return sumv(x) / x.length; }
  function maxAbs(x) { let m = Math.abs(x[0]); for (let i = 1; i < x.length; i++) { const a = Math.abs(x[i]); if (a > m) m = a; } return m; }
  function minAbs(x) { let m = Math.abs(x[0]); for (let i = 1; i < x.length; i++) { const a = Math.abs(x[i]); if (a < m) m = a; } return m; }
  function reluVec(x) { const n = x.length, o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = x[i] > 0 ? x[i] : 0; return o; }
  function cumsum(x) { const n = x.length, o = new Float64Array(n); let s = 0; for (let i = 0; i < n; i++) { s += x[i]; o[i] = s; } return o; }
  function variance(x) {
    const m = meanv(x); let s = 0;
    for (let i = 0; i < x.length; i++) { const d = x[i] - m; s += d * d; }
    return s / x.length;
  }
  // 位移 raw-moment 方差：var = mean((x−s)²) − mean(x−s)²，s=x[0]。
  // 与 variance() 的两趟式不同公式、不同累加路径，且 s 贴近数据中心，
  // 无「大均值 + 极小方差」的相消放大（供 B080 做 oracle）。
  function varShift(x) {
    const s = x[0], n = x.length;
    let s1 = 0, s2 = 0;
    for (let i = n - 1; i >= 0; i--) { const d = x[i] - s; s1 += d; s2 += d * d; }   // 反向累加
    const m1 = s1 / n, m2 = s2 / n;
    const v = m2 - m1 * m1;
    return v < 0 ? 0 : v;
  }
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

  // ── relu 与逐元素函数（B065–B069）──────────────────────────────────────
  reg(65, {
    name: 'relu · 恒等（非负域）', intuition: '对非负输入 ReLU 是恒等映射。',
    math: 'x ≥ 0 ⇒ relu(x) = x', pseudocode: 'return reluVec(x)', complexity: 'O(n)',
    ref: 'Nair & Hinton (2010)', commonPitfall: '无',
    invariant: '对 vecNonNeg：exactVec(run(x), x)；锚 run((−2,3))=(0,3)',
    inputs: ['x'], domains: { x: 'vecNonNeg' },
    run: (x) => reluVec(x),
    predicate: (x) => exactVec(runOf('B065')(x), x) && exactVec(runOf('B065')(vec(-2, 3)), vec(0, 3)),
  });
  reg(66, {
    name: 'relu · 互补分解为绝对值', intuition: '正部与负部拼回原值：relu(x)+relu(−x)=|x|。',
    math: 'relu(x) + relu(−x) = |x|', pseudocode: 'return vAdd(reluVec(x), reluVec(vNeg(x)))', complexity: 'O(n)',
    ref: 'ReLU 正负分解', commonPitfall: '逐位精确（两部分只做选取与加法，无舍入路径分歧）。',
    invariant: '对 vecFinite：exactVec(run(x), |x|)；锚 run((−2,3))=(2,3)',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => vAdd(reluVec(x), reluVec(vNeg(x))),
    predicate: (x) => exactVec(runOf('B066')(x), vAbs(x)) && exactVec(runOf('B066')(vec(-2, 3)), vec(2, 3)),
  });
  reg(67, {
    name: 'relu · 取反对偶', intuition: '负部等于正部减原值（x = relu(x) − relu(−x) 的变形）。',
    math: 'relu(−x) = relu(x) − x', pseudocode: 'return reluVec(vNeg(x))', complexity: 'O(n)',
    ref: 'ReLU 分解', commonPitfall: 'x=0 时两侧均 +0；exactVec 归一 -0。',
    invariant: '对 vecFinite：exactVec(run(x), relu(x)−x)（异构公式）',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => reluVec(vNeg(x)),
    predicate: (x) => exactVec(runOf('B067')(x), vSub(reluVec(x), x)),
  });
  reg(68, {
    name: 'sqrt · 平方还原绝对值', intuition: '对分量先平方再开方得到绝对值。',
    math: '√(x⊙x) = |x|', pseudocode: 'return vSqrt(vMul(x,x))', complexity: 'O(n)',
    ref: 'IEEE 754 sqrt', commonPitfall: '需容差 1e-9。',
    invariant: '对 vecSmall：relClose(run(x), |x|, 1e-9)（异构公式）',
    inputs: ['x'], domains: { x: 'vecSmall' },
    run: (x) => vSqrt(vMul(x, x)),
    predicate: (x) => rc(runOf('B068')(x), vAbs(x), 1e-9),
  });
  reg(69, {
    name: 'recip · 逐元素倒数还原', intuition: '正分量先取倒数再乘回原值（≈1）。',
    math: 'x ⊙ (1/x) = 1', pseudocode: 'return vMul(x, vRecip(x))', complexity: 'O(n)',
    ref: '数值分析', commonPitfall: '非精确，需容差；vecPos 保证无零除。',
    invariant: '对 vecPos：relClose(run(x), 1, 1e-9)（oracle 为常数 1）',
    inputs: ['x'], domains: { x: 'vecPos' },
    run: (x) => vMul(x, vRecip(x)),
    predicate: (x) => rc(runOf('B069')(x), ones(x.length), 1e-9),
  });

  // ── Hadamard 代数恒等式（B070–B071）────────────────────────────────────
  reg(70, {
    name: 'hadamard · 和的平方展开', intuition: '逐元素平方展开：两向量和的逐元素平方。',
    math: '(x+y)⊙(x+y) = x⊙x + 2(x⊙y) + y⊙y', pseudocode: 'return vMul(vAdd(x,y), vAdd(x,y))', complexity: 'O(n)',
    ref: 'Hadamard 代数', commonPitfall: '整数域下逐步精确。',
    invariant: '对 vecInt：exactVec(run(x,y), x⊙x+2(x⊙y)+y⊙y)（异构展开）',
    inputs: ['x', 'y'], domains: { x: 'vecInt', y: 'vecInt' },
    run: (x, y) => vMul(vAdd(x, y), vAdd(x, y)),
    predicate: (x, y) => exactVec(
      runOf('B070')(x, y),
      vAdd(vAdd(vMul(x, x), vScale(2, vMul(x, y))), vMul(y, y))),
  });
  reg(71, {
    name: 'hadamard · 平方差', intuition: '和与差的逐元素积等于逐元素平方差。',
    math: '(x+y) ⊙ (x−y) = x⊙x − y⊙y', pseudocode: 'return vMul(vAdd(x,y), vSub(x,y))', complexity: 'O(n)',
    ref: 'Hadamard 代数', commonPitfall: '整数域下逐步精确。',
    invariant: '对 vecInt：exactVec(run(x,y), x⊙x−y⊙y)（异构公式）',
    inputs: ['x', 'y'], domains: { x: 'vecInt', y: 'vecInt' },
    run: (x, y) => vMul(vAdd(x, y), vSub(x, y)),
    predicate: (x, y) => exactVec(runOf('B071')(x, y), vSub(vMul(x, x), vMul(y, y))),
  });

  // ── 前缀和 / 统计矩（B072–B081）────────────────────────────────────────
  reg(72, {
    name: 'cumsum · 末项即总和', intuition: '前缀和最后一个分量就是全量和。',
    math: 'cumsum(x)[n−1] = sum(x)', pseudocode: 'return cumsum(x)', complexity: 'O(n)',
    ref: '前缀和（扫描）', commonPitfall: '同为左到右累加，整数域逐位一致。',
    invariant: '对 vecInt：Object.is(run(x)[n−1], sum(x))；锚 run((1,2,3))=(1,3,6)',
    inputs: ['x'], domains: { x: 'vecInt' },
    run: (x) => cumsum(x),
    predicate: (x) => {
      const c = runOf('B072')(x);
      if (!Object.is(c[c.length - 1], sumv(x))) return false;
      const a = runOf('B072')(vec(1, 2, 3));
      return exactVec(a, vec(1, 3, 6));
    },
  });
  reg(73, {
    name: 'cumsum · 差分还原', intuition: '前缀和逐次作差还原原向量（扫描之逆）。',
    math: 'diff(cumsum(x)) = x', pseudocode: 'return cumsum(x)', complexity: 'O(n)',
    ref: '前缀和逆运算', commonPitfall: '整数域下 c[i]−c[i−1] 精确还原。',
    invariant: '对 vecInt：exactVec(diff(run(x)), x)（扫描之逆，run 耦合）',
    inputs: ['x'], domains: { x: 'vecInt' },
    run: (x) => cumsum(x),
    predicate: (x) => {
      const c = runOf('B073')(x), d = new Float64Array(x.length);
      d[0] = c[0];
      for (let i = 1; i < x.length; i++) d[i] = c[i] - c[i - 1];
      return exactVec(d, x);
    },
  });
  reg(74, {
    name: 'sum · 取反对偶', intuition: '负向量之和等于原和之负（整数域精确）。',
    math: 'sum(−x) = −sum(x)', pseudocode: 'return sumv(vNeg(x))', complexity: 'O(n)',
    ref: '线性归约', commonPitfall: 'IEEE 舍入符号对称，故逐位成立；但 sum(x)=0 时 −sum(x)=−0 而 sum(−x)=+0（累加器自 +0 起）→ 判据须用 exactNum 归一 ±0。',
    invariant: '对 vecInt：exactNum(run(x), −sum(x))（异构公式）',
    inputs: ['x'], domains: { x: 'vecInt' },
    run: (x) => sumv(vNeg(x)), predicate: (x) => exactNum(runOf('B074')(x), -sumv(x)),
  });
  reg(75, {
    name: 'mean · 可加性', intuition: '向量和的平均等于平均之和。',
    math: 'mean(x + y) = mean(x) + mean(y)', pseudocode: 'return meanv(vAdd(x,y))', complexity: 'O(n)',
    ref: '线性归约', commonPitfall: '需容差 1e-9。',
    invariant: '对 vecFinite：relClose(run(x,y), mean(x)+mean(y), 1e-9)（异构路径）',
    inputs: ['x', 'y'], domains: { x: 'vecFinite', y: 'vecFinite' },
    run: (x, y) => meanv(vAdd(x, y)),
    predicate: (x, y) => rc(runOf('B075')(x, y), meanv(x) + meanv(y), 1e-9),
  });
  reg(76, {
    name: 'mean · 取反对偶', intuition: '负向量的平均等于平均之负。',
    math: 'mean(−x) = −mean(x)', pseudocode: 'return meanv(vNeg(x))', complexity: 'O(n)',
    ref: '线性归约', commonPitfall: '理论上精确；仍用 relClose 归类数值族。',
    invariant: '对 vecFinite：relClose(run(x), −mean(x), 1e-9)（异构公式）',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => meanv(vNeg(x)), predicate: (x) => rc(runOf('B076')(x), -meanv(x), 1e-9),
  });
  reg(77, {
    name: 'max · 对加法次可加', intuition: '和向量的最大值不超各自最大值之和。',
    math: 'max(x + y) ≤ max(x) + max(y)', pseudocode: 'return maxv(vAdd(x,y))', complexity: 'O(n)',
    ref: '序统计次可加性', commonPitfall: '浮点可差数个 ulp，加 1e-9 相对 epsilon。',
    invariant: '对 vecFinite：run(x,y) ≤ max(x)+max(y)+ε，ε=1e-9·max(1,|max(x)+max(y)|)',
    inputs: ['x', 'y'], domains: { x: 'vecFinite', y: 'vecFinite' },
    run: (x, y) => maxv(vAdd(x, y)),
    predicate: (x, y) => runOf('B077')(x, y) <= maxv(x) + maxv(y) + 1e-9 * Math.max(1, Math.abs(maxv(x) + maxv(y))),
  });
  reg(78, {
    name: 'min · 对加法超可加', intuition: '和向量的最小值不低于各自最小值之和。',
    math: 'min(x + y) ≥ min(x) + min(y)', pseudocode: 'return minv(vAdd(x,y))', complexity: 'O(n)',
    ref: '序统计超可加性', commonPitfall: '浮点可差数个 ulp，减 1e-9 相对 epsilon。',
    invariant: 'run(x,y) === minᵢ(xᵢ+yᵢ)（独立最小值 oracle 钉死数值）',
    inputs: ['x', 'y'], domains: { x: 'vecFinite', y: 'vecFinite' },
    run: (x, y) => minv(vAdd(x, y)),
    predicate: (x, y) => {
      const r = runOf('B078')(x, y);
      let m = Infinity; for (let i = 0; i < x.length; i++) m = Math.min(m, x[i] + y[i]);
      return Object.is(r, m);
    },
  });
  reg(79, {
    name: 'variance · 非负', intuition: '方差是平方偏差的平均，恒非负。',
    math: 'var(x) = mean((xᵢ−mean(x))²) ≥ 0', pseudocode: 'return variance(x)', complexity: 'O(n)',
    ref: '统计矩定义', commonPitfall: '域内无 NaN/Inf，平方和非负；锚 run((3,3,3))=0、run((1,2,3))=2/3。',
    invariant: '对 vecFinite：run(x) ≥ 0；锚 run((3,3,3))=0、rc(run((1,2,3)), 2/3, 1e-12)',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => variance(x),
    predicate: (x) => {
      const run = runOf('B079');
      return run(x) >= 0 && Object.is(run(vec(3, 3, 3)), 0) && rc(run(vec(1, 2, 3)), 2 / 3, 1e-12);
    },
  });
  reg(80, {
    name: 'variance · 计算式恒等式', intuition: '方差等于平方的均值减均值的平方（一步计算式）。',
    math: 'var(x) = mean(x⊙x) − mean(x)²', pseudocode: 'return variance(x)', complexity: 'O(n)',
    ref: '统计矩计算式', commonPitfall: '原计算式 mean(x²)−mean(x)² 在近常数向量（如 [1,…,1,1.001]）下相消伪红；oracle 改用位移 raw-moment（s=x[0]、反向累加），既异构又无相消。',
    invariant: '对 vecSmall：relClose(run(x), mean((x−s)²)−mean(x−s)², 1e-9)，s=x[0]',
    inputs: ['x'], domains: { x: 'vecSmall' },
    run: (x) => variance(x),
    predicate: (x) => rc(runOf('B080')(x), varShift(x), 1e-9),
  });
  reg(81, {
    name: 'mean · 中心化和为零', intuition: '减去均值后各分量之和为 0（中心化定义）。',
    math: 'Σ(xᵢ − mean(x)) = 0', pseudocode: 'return vSub(x, vScale(meanv(x), ones(n)))', complexity: 'O(n)',
    ref: '数据中心化', commonPitfall: '浮点残差 ~1e-11，远小于 1e-9 绝对容差。',
    invariant: '对 vecFinite：relClose(Σ run(x), 0, 1e-9)；锚 run((5,5,5)) 和精确为 0',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => vSub(x, vScale(meanv(x), ones(x.length))),
    predicate: (x) => {
      const c = runOf('B081')(x);
      return rc(sumv(c), 0, 1e-9) && Object.is(sumv(runOf('B081')(vec(5, 5, 5))), 0);
    },
  });

  // ── 几何：加权平均 / 平行四边形 / 投影 / 余弦（B082–B086）──────────────
  reg(82, {
    name: 'wmean · 正权重夹逼', intuition: '正权重加权平均仍落在原值 min/max 之间。',
    math: 'min(x) ≤ Σw⊙x/Σw ≤ max(x)（w>0）', pseudocode: 'return sumv(vMul(w,x))/sumv(w)', complexity: 'O(n)',
    ref: '凸组合性质', commonPitfall: 'vecPos 保证权重严格正、分母非零；舍入加 epsilon；锚 run((1,1),(2,4))=3。',
    invariant: '对 vecPos w, vecFinite x：min(x)−ε ≤ run(w,x) ≤ max(x)+ε',
    inputs: ['w', 'x'], domains: { w: 'vecPos', x: 'vecFinite' },
    run: (w, x) => sumv(vMul(w, x)) / sumv(w),
    predicate: (w, x) => {
      const lo = minv(x), hi = maxv(x), wm = runOf('B082')(w, x);
      if (!(wm >= lo - 1e-9 * Math.max(1, Math.abs(lo)) && wm <= hi + 1e-9 * Math.max(1, Math.abs(hi)))) return false;
      return rc(runOf('B082')(vec(1, 1), vec(2, 4)), 3, 1e-12);
    },
  });
  reg(83, {
    name: 'norm · 平行四边形恒等式', intuition: '两对角线长度平方和等于四边平方和。',
    math: '‖x+y‖² + ‖x−y‖² = 2‖x‖² + 2‖y‖²', pseudocode: 'return norm(vAdd(x,y))', complexity: 'O(n)',
    ref: 'Strang (2016) 平行四边形定律', commonPitfall: '开方再平方有舍入，tol 1e-9。',
    invariant: '对 vecSmall：relClose(run(x,y)²+‖x−y‖², 2‖x‖²+2‖y‖², 1e-9)（run 耦合对角线项）',
    inputs: ['x', 'y'], domains: { x: 'vecSmall', y: 'vecSmall' },
    run: (x, y) => norm(vAdd(x, y)),
    predicate: (x, y) => rc(
      runOf('B083')(x, y) * runOf('B083')(x, y) + norm(vSub(x, y)) * norm(vSub(x, y)),
      2 * norm(x) * norm(x) + 2 * norm(y) * norm(y), 1e-9),
  });
  reg(84, {
    name: 'proj · 残差与基正交', intuition: '沿 e₀ 投影后，残差与 e₀ 点积为 0。',
    math: '(x − (x·e₀)e₀)·e₀ = 0', pseudocode: 'return dot(vSub(x, vScale(x[0], e0(n))), e0(n))', complexity: 'O(n)',
    ref: 'Strang (2016) 正交分解', commonPitfall: '整数域下残差首分量精确为 0。',
    invariant: '对 vecInt：Object.is(run(x), 0)',
    inputs: ['x'], domains: { x: 'vecInt' },
    run: (x) => dot(vSub(x, vScale(x[0], e0(x.length))), e0(x.length)),
    predicate: (x) => Object.is(runOf('B084')(x), 0),
  });
  reg(85, {
    name: 'proj · 投影分解还原', intuition: '投影分量与残差分量相加还原原向量。',
    math: '(x·e₀)e₀ + (x − (x·e₀)e₀) = x', pseudocode: 'return vAdd(vScale(x[0], e0(n)), vSub(x, vScale(x[0], e0(n))))', complexity: 'O(n)',
    ref: 'Strang (2016) 正交分解', commonPitfall: '整数域逐位精确。',
    invariant: '对 vecInt：exactVec(run(x), x)（oracle 为输入本身）',
    inputs: ['x'], domains: { x: 'vecInt' },
    run: (x) => vAdd(vScale(x[0], e0(x.length)), vSub(x, vScale(x[0], e0(x.length)))),
    predicate: (x) => exactVec(runOf('B085')(x), x),
  });
  reg(86, {
    name: 'cos · 自相似为 1', intuition: '向量与自身的余弦相似度恒为 1。',
    math: 'cos(x,x) = x·x/(‖x‖‖x‖) = 1', pseudocode: 'return dot(x,x)/(norm(x)*norm(x))', complexity: 'O(n)',
    ref: '余弦相似度；Cauchy–Schwarz 取等', commonPitfall: 'vecPos 保证分母非零；tol 1e-9。',
    invariant: '对 vecPos：relClose(run(x), 1, 1e-9)（oracle 为常数 1）',
    inputs: ['x'], domains: { x: 'vecPos' },
    run: (x) => dot(x, x) / (norm(x) * norm(x)),
    predicate: (x) => rc(runOf('B086')(x), 1, 1e-9),
  });

  // ── 绝对值 / 反转 / 除法 / 平方极值 / 夹挤 / 幂等（B087–B096）──────────
  reg(87, {
    name: 'abs · 幂等', intuition: '绝对值对自身幂等。',
    math: '| |x| | = |x|', pseudocode: 'return vAbs(vAbs(x))', complexity: 'O(n)',
    ref: 'IEEE 754', commonPitfall: '无',
    invariant: '对 vecFinite：exactVec(run(run(x)), run(x))（幂等经 run 自复合）',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => vAbs(x),
    predicate: (x) => { const run = runOf('B087'), r = run(x); return exactVec(run(r), r); },
  });
  reg(88, {
    name: 'abs · 乘积可分', intuition: '逐元素积的绝对值等于绝对值的逐元素积。',
    math: '|x ⊙ y| = |x| ⊙ |y|', pseudocode: 'return vAbs(vMul(x,y))', complexity: 'O(n)',
    ref: 'IEEE 754（乘法符号对称）', commonPitfall: '无（IEEE 乘法的符号规则保证逐位成立）。',
    invariant: '对 vecFinite：exactVec(run(x,y), |x|⊙|y|)（异构公式）',
    inputs: ['x', 'y'], domains: { x: 'vecFinite', y: 'vecFinite' },
    run: (x, y) => vAbs(vMul(x, y)),
    predicate: (x, y) => exactVec(runOf('B088')(x, y), vMul(vAbs(x), vAbs(y))),
  });
  reg(89, {
    name: 'norm · 取绝对值不变', intuition: '模长只依赖分量大小，与符号无关。',
    math: '‖|x|‖ = ‖x‖', pseudocode: 'return norm(vAbs(x))', complexity: 'O(n)',
    ref: 'Strang (2016) 范数', commonPitfall: '理论上逐位相等；用 relClose 归类数值族。',
    invariant: '对 vecFinite：relClose(run(x), ‖x‖, 1e-9)（异构参数）',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => norm(vAbs(x)), predicate: (x) => rc(runOf('B089')(x), norm(x), 1e-9),
  });
  reg(90, {
    name: 'reverse · 对合', intuition: '两次反转回到原向量（索引对称换位）。',
    math: 'reverse(reverse(x)) = x', pseudocode: 'return vRev(vRev(x))', complexity: 'O(n)',
    ref: '索引代数', commonPitfall: '无',
    invariant: '对 vecInt：exactVec(run(run(x)), x)（对合经 run 自复合）',
    inputs: ['x'], domains: { x: 'vecInt' },
    run: (x) => vRev(x),
    predicate: (x) => { const run = runOf('B090'), r = run(x); return exactVec(run(r), x); },
  });
  reg(91, {
    name: 'vDiv · 除法还原', intuition: '先除后乘回原值（正域，≈x）。',
    math: '(x / y) ⊙ y ≈ x', pseudocode: 'return vMul(vDiv(x,y), y)', complexity: 'O(n)',
    ref: '数值分析', commonPitfall: '非精确，需容差；vecPos 保证无零除。',
    invariant: '对 vecPos：relClose(run(x,y), x, 1e-9)（oracle 为输入本身）',
    inputs: ['x', 'y'], domains: { x: 'vecPos', y: 'vecPos' },
    run: (x, y) => vMul(vDiv(x, y), y),
    predicate: (x, y) => rc(runOf('B091')(x, y), x, 1e-9),
  });
  reg(92, {
    name: 'max · 自积平方极值', intuition: '逐元素平方的最大值等于最大绝对分量的平方。',
    math: 'max(x⊙x) = (max|xᵢ|)²', pseudocode: 'return maxv(vMul(x,x))', complexity: 'O(n)',
    ref: '平方单调性 + 序统计', commonPitfall: '理论上成立；用 relClose 防 ulp 歧义。',
    invariant: '对 vecFinite：relClose(run(x), max|xᵢ|², 1e-9)（异构公式）',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => maxv(vMul(x, x)),
    predicate: (x) => rc(runOf('B092')(x), maxAbs(x) * maxAbs(x), 1e-9),
  });
  reg(93, {
    name: 'min · 自积平方极值', intuition: '逐元素平方的最小值等于最小绝对分量的平方。',
    math: 'min(x⊙x) = (min|xᵢ|)²', pseudocode: 'return minv(vMul(x,x))', complexity: 'O(n)',
    ref: '平方单调性 + 序统计', commonPitfall: '理论上成立；用 relClose 防 ulp 歧义。',
    invariant: '对 vecFinite：relClose(run(x), min|xᵢ|², 1e-9)（异构公式）',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => minv(vMul(x, x)),
    predicate: (x) => rc(runOf('B093')(x), minAbs(x) * minAbs(x), 1e-9),
  });
  reg(94, {
    name: 'norm · L∞ 夹挤', intuition: '模长被最大分量从上下两侧夹住：m ≤ ‖x‖ ≤ √n·m。',
    math: 'max|xᵢ| ≤ ‖x‖ ≤ √n·max|xᵢ|', pseudocode: 'return norm(x)', complexity: 'O(n)',
    ref: '范数等价性 ‖·‖∞ ≤ ‖·‖₂ ≤ √n‖·‖∞', commonPitfall: '两侧均加 1e-9 相对 epsilon 吸收舍入。',
    invariant: '对 vecFinite：m ≤ run(x)+ε 且 run(x) ≤ √n·m·(1+ε)，m=max|xᵢ|, ε=1e-9',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => norm(x),
    predicate: (x) => {
      const m = maxAbs(x), n2 = runOf('B094')(x);
      return m <= n2 + 1e-9 * Math.max(1, n2) && n2 <= Math.sqrt(x.length) * m * (1 + 1e-9) + 1e-12;
    },
  });
  reg(95, {
    name: 'normalize · 幂等', intuition: '单位向量再归一化不变（模已为 1）。',
    math: 'normalize(normalize(x)) = normalize(x)', pseudocode: 'return vScale(1/norm(x), x)', complexity: 'O(n)',
    ref: 'Strang (2016) 单位向量', commonPitfall: 'vecPos 保证每次归一化分母非零；tol 1e-9。',
    invariant: '对 vecPos：relClose(run(run(x)), run(x), 1e-9)（幂等经 run 自复合）',
    inputs: ['x'], domains: { x: 'vecPos' },
    run: (x) => vScale(1 / norm(x), x),
    predicate: (x) => {
      const run = runOf('B095'), u = run(x);
      return rc(run(u), u, 1e-9);
    },
  });
  reg(96, {
    name: 'relu · 幂等', intuition: 'ReLU 反复作用不变（截断一次即到位）。',
    math: 'relu(relu(x)) = relu(x)', pseudocode: 'return reluVec(reluVec(x))', complexity: 'O(n)',
    ref: 'Nair & Hinton (2010)', commonPitfall: '无（选取操作无舍入）。',
    invariant: '对 vecFinite：exactVec(run(run(x)), run(x))（幂等经 run 自复合）',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => reluVec(reluVec(x)),
    predicate: (x) => { const run = runOf('B096'), r = run(x); return exactVec(run(r), r); },
  });
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

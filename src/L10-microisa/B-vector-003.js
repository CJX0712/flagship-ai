/* ============================================================================
 * L10 微指令 · B 域（向量运算）· 第 4/4 批（B097–B128）
 * ----------------------------------------------------------------------------
 * 所属层：L10-microIsa  域：B（向量）
 * 允许依赖的层：L9（rng/assert/special）、L10（registry/invariant）
 * 主要 @ref：Strang (2016) Lp 范数与距离函数；Hamming (1950)；
 *           NumPy cumprod/cummax/cummin/one-hot/gather/pad/truncate/rotate
 *           语义；Goodfellow 2016 §6.2 softmax 单纯形结构
 *
 * 涵盖：范数与距离家族（B097–B108：L3 范数齐次/非负/三角、加权 L2 齐次、
 *       余弦距离有界与自零、切比雪夫/汉明/L1 距离的对称与三角）、
 *       向量进阶运算（B109–B120：cumprod 总积与递推、cummax/cummin 单调
 *       与末项、argmax 选中与索引结构、one-hot 构造、gather 寻址、
 *       softmax 和为 1）、切片与组合（B121–B128：concat 长度可加与偏移
 *       还原、slice 半开区间、pad 前补零、truncate 前 n 项、rotate 复合
 *       还原、reverse 与 rotate 复合）。
 * 判据：整数向量域 exactVec/exactNum 逐位精确（±0 归一）；数值族
 *       relClose(1e-9)；不等式族加 1e-9 相对 epsilon 吸收舍入
 *       （保持真实数学性质，不放宽容差）。
 * K5：本文件禁用模块级语法（详见全局契约 K5）。
 * ==========================================================================*/
;(function (NS) {
  'use strict';
  NS.L10 = NS.L10 || {};
  function reg(i, o) { NS.L10.register('B', i, o); }
  const rc = (a, b, t) => NS.L10.relClose(a, b, t);

  // ── 向量小工具（纯函数）────────────────────────────────────────────────
  function vAdd(x, y) { const n = x.length, o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = x[i] + y[i]; return o; }
  function vScale(c, x) { const n = x.length, o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = c * x[i]; return o; }
  function vRev(x) { const n = x.length, o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = x[n - 1 - i]; return o; }
  function dot(x, y) { let s = 0; for (let i = 0; i < x.length; i++) s += x[i] * y[i]; return s; }
  function norm(x) { return Math.sqrt(dot(x, x)); }
  function sumv(x) { let s = 0; for (let i = 0; i < x.length; i++) s += x[i]; return s; }
  function maxv(x) { let m = x[0]; for (let i = 1; i < x.length; i++) if (x[i] > m) m = x[i]; return m; }
  function minv(x) { let m = x[0]; for (let i = 1; i < x.length; i++) if (x[i] < m) m = x[i]; return m; }
  // 范数与距离家族
  function lp3(x) { let s = 0; for (let i = 0; i < x.length; i++) { const a = Math.abs(x[i]); s += a * a * a; } return Math.cbrt(s); }
  function wl2(w, x) { let s = 0; for (let i = 0; i < x.length; i++) s += w[i] * x[i] * x[i]; return Math.sqrt(s); }
  function cosdist(x, y) { return 1 - dot(x, y) / (norm(x) * norm(y)); }
  function cheb(x, y) { let m = Math.abs(x[0] - y[0]); for (let i = 1; i < x.length; i++) { const d = Math.abs(x[i] - y[i]); if (d > m) m = d; } return m; }
  function ham(x, y) { let c = 0; for (let i = 0; i < x.length; i++) if (x[i] !== y[i]) c++; return c; }
  function l1d(x, y) { let s = 0; for (let i = 0; i < x.length; i++) s += Math.abs(x[i] - y[i]); return s; }
  // 前缀积 / 极值扫描 / argmax / one-hot / gather / softmax
  function cumprod(x) { const n = x.length, o = new Float64Array(n); o[0] = x[0]; for (let i = 1; i < n; i++) o[i] = o[i - 1] * x[i]; return o; }
  function cummax(x) { const n = x.length, o = new Float64Array(n); o[0] = x[0]; for (let i = 1; i < n; i++) o[i] = o[i - 1] > x[i] ? o[i - 1] : x[i]; return o; }
  function cummin(x) { const n = x.length, o = new Float64Array(n); o[0] = x[0]; for (let i = 1; i < n; i++) o[i] = o[i - 1] < x[i] ? o[i - 1] : x[i]; return o; }
  function argmaxVec(x) { let a = 0; for (let i = 1; i < x.length; i++) if (x[i] > x[a]) a = i; return a; }
  function onehot(n, k) { const o = new Float64Array(n); o[((k % n) + n) % n] = 1; return o; }
  function gather(v, i) { return v[((i % v.length) + v.length) % v.length]; }
  function softmaxVec(x) {
    const n = x.length; let m = x[0];
    for (let i = 1; i < n; i++) if (x[i] > m) m = x[i];
    const e = new Float64Array(n); let s = 0;
    for (let i = 0; i < n; i++) { e[i] = Math.exp(x[i] - m); s += e[i]; }
    for (let i = 0; i < n; i++) e[i] /= s;
    return e;
  }
  // 切片与组合
  function vConcat(x, y) { const o = new Float64Array(x.length + y.length); o.set(x, 0); o.set(y, x.length); return o; }
  function vSlice(x, s, e) { const o = new Float64Array(e - s); for (let i = s; i < e; i++) o[i - s] = x[i]; return o; }
  function padFront(x, m) { const o = new Float64Array(m); o.set(x, m - x.length); return o; }
  function vTruncate(x, m) { return vSlice(x, 0, m); }
  function vRotate(x, k) { const n = x.length, kw = ((k % n) + n) % n, o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = x[(i + kw) % n]; return o; }
  // 二分树形乘积：cumprod 末项的独立参考路径（结合顺序与线性扫描不同）。
  function treeProd(x, lo, hi) { if (hi - lo === 1) return x[lo]; const mid = (lo + hi) >> 1; return treeProd(x, lo, mid) * treeProd(x, mid, hi); }
  // 索引归约：两个 nat 索引压入 [0,n] 并排序（TAGS 无索引域，mod 归约覆盖全边界）。
  function idx2(a, b, n) { const u = a % (n + 1), v = b % (n + 1); return u <= v ? [u, v] : [v, u]; }
  // 逐位相等（-0 与 +0 归一为相等；IEEE 中 Object.is(-0,0) 为假但数值等价）。
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

  // ── 范数与距离家族（B097–B108）────────────────────────────────────────
  reg(97, {
    name: 'lp3 · 齐次缩放', intuition: 'L3 范数整体缩放 c 倍，模长缩放 |c| 倍。',
    math: '‖c·x‖₃ = |c|·‖x‖₃', pseudocode: 'return lp3(vScale(c,x))', complexity: 'O(n)',
    ref: 'Strang (2016) Lp 范数齐次性', commonPitfall: 'vecSmall 控幅：|c·xᵢ|≤1e4，立方和 ≤8e12 不溢出；tol 1e-9。',
    invariant: '对 smallInt c, vecSmall x：relClose(‖cx‖₃, |c|‖x‖₃, 1e-9)',
    inputs: ['c', 'x'], domains: { c: 'smallInt', x: 'vecSmall' },
    run: (c, x) => lp3(vScale(c, x)),
    predicate: (c, x) => rc(lp3(vScale(c, x)), Math.abs(c) * lp3(x), 1e-9),
  });
  reg(98, {
    name: 'lp3 · 非负且定性', intuition: 'L3 范数是立方和的立方根，恒非负；且为零当且仅当零向量。',
    math: '‖x‖₃ ≥ 0；‖x‖₃ = 0 ⇔ x = 0', pseudocode: 'return lp3(x)', complexity: 'O(n)',
    ref: 'Strang (2016) 范数正定性', commonPitfall: '立方和非负且无相消，故 =0 当且仅当逐分量为 0（浮点同样精确）。',
    invariant: '对 vecFinite：‖x‖₃ ≥ 0 且 (‖x‖₃ = 0 ⇒ ∀i xᵢ = 0)',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => lp3(x),
    predicate: (x) => {
      const L = lp3(x);
      if (L < 0) return false;
      if (L === 0) for (let i = 0; i < x.length; i++) if (x[i] !== 0) return false;
      return true;
    },
  });
  reg(99, {
    name: 'lp3 · 三角不等式', intuition: 'L3 是范数（p≥1）：两边之和大于第三边。',
    math: '‖x + y‖₃ ≤ ‖x‖₃ + ‖y‖₃', pseudocode: 'return lp3(vAdd(x,y))', complexity: 'O(n)',
    ref: 'Strang (2016) 范数三角不等式', commonPitfall: '立方与开方有舍入，加 1e-9 相对 epsilon。',
    invariant: '对 vecSmall：‖x+y‖₃ ≤ ‖x‖₃+‖y‖₃+ε，ε=1e-9·max(1,‖x‖₃+‖y‖₃)',
    inputs: ['x', 'y'], domains: { x: 'vecSmall', y: 'vecSmall' },
    run: (x, y) => lp3(vAdd(x, y)),
    predicate: (x, y) => lp3(vAdd(x, y)) <= lp3(x) + lp3(y) + 1e-9 * Math.max(1, lp3(x) + lp3(y)),
  });
  reg(100, {
    name: 'wl2 · 加权齐次缩放', intuition: '正权重 L2 范数同样齐次：整体缩放提出 |c|。',
    math: '‖c·x‖_{w,2} = |c|·‖x‖_{w,2}（w>0）', pseudocode: 'return wl2(w, vScale(c,x))', complexity: 'O(n)',
    ref: 'Strang (2016) 加权范数', commonPitfall: 'vecPos 保证权重严格正；幅值 ≤1e9 不溢出；tol 1e-9。',
    invariant: '对 smallInt c, vecPos w, vecSmall x：relClose(‖cx‖_{w,2}, |c|‖x‖_{w,2}, 1e-9)',
    inputs: ['c', 'w', 'x'], domains: { c: 'smallInt', w: 'vecPos', x: 'vecSmall' },
    run: (c, w, x) => wl2(w, vScale(c, x)),
    predicate: (c, w, x) => rc(wl2(w, vScale(c, x)), Math.abs(c) * wl2(w, x), 1e-9),
  });
  reg(101, {
    name: 'cosdist · 有界 [0,2]', intuition: '余弦距离 1−cos 落在 [0,2]：同向为 0、反向为 2。',
    math: '0 ≤ 1 − x·y/(‖x‖‖y‖) ≤ 2', pseudocode: 'return cosdist(x,y)', complexity: 'O(n)',
    ref: '余弦距离；Cauchy–Schwarz 给出 |cos|≤1', commonPitfall: 'cos 可越界 1 ulp（如 1+1e-16），两端各放 1e-9 相对 epsilon。',
    invariant: '对 vecPos x,y：−ε ≤ cosdist(x,y) ≤ 2+2ε，ε=1e-9',
    inputs: ['x', 'y'], domains: { x: 'vecPos', y: 'vecPos' },
    run: (x, y) => cosdist(x, y),
    predicate: (x, y) => { const d = cosdist(x, y); return d >= -1e-9 && d <= 2 + 2e-9; },
  });
  reg(102, {
    name: 'cosdist · 自为 0', intuition: '向量与自身的余弦距离为 0（距离函数对角线性质）。',
    math: '1 − cos(x,x) = 0', pseudocode: 'return cosdist(x,x)', complexity: 'O(n)',
    ref: '余弦相似度自相似为 1 的距离视角', commonPitfall: 'vecPos 保证分母非零；tol 1e-9。',
    invariant: '对 vecPos：relClose(cosdist(x,x), 0, 1e-9)',
    inputs: ['x'], domains: { x: 'vecPos' },
    run: (x) => cosdist(x, x),
    predicate: (x) => rc(cosdist(x, x), 0, 1e-9),
  });
  reg(103, {
    name: 'cheb · 对称', intuition: '切比雪夫距离（L∞ 距离）对两个端点对称。',
    math: 'd∞(x,y) = d∞(y,x)，d∞(x,y) = maxᵢ|xᵢ−yᵢ|', pseudocode: 'return cheb(x,y)', complexity: 'O(n)',
    ref: 'Strang (2016) L∞ 距离', commonPitfall: 'IEEE 减法符号对称（b−a 与 −(a−b) 逐位相同），取绝对值后逐位相等 → exactNum 精确。',
    invariant: '对 vecFinite：exactNum(d∞(x,y), d∞(y,x))',
    inputs: ['x', 'y'], domains: { x: 'vecFinite', y: 'vecFinite' },
    run: (x, y) => cheb(x, y),
    predicate: (x, y) => exactNum(cheb(x, y), cheb(y, x)),
  });
  reg(104, {
    name: 'cheb · 三角不等式', intuition: 'L∞ 距离满足三角不等式（度量公理）。',
    math: 'd∞(x,z) ≤ d∞(x,y) + d∞(y,z)', pseudocode: 'return cheb(x,z)', complexity: 'O(n)',
    ref: 'Strang (2016) 距离函数', commonPitfall: '逐位三角不等式经 max 传递；浮点差数个 ulp，加 1e-9 相对 epsilon。',
    invariant: '对 vecFinite：d∞(x,z) ≤ d∞(x,y)+d∞(y,z)+ε，ε=1e-9·max(1, 和)',
    inputs: ['x', 'y', 'z'], domains: { x: 'vecFinite', y: 'vecFinite', z: 'vecFinite' },
    run: (x, y, z) => cheb(x, z),
    predicate: (x, y, z) => cheb(x, z) <= cheb(x, y) + cheb(y, z) + 1e-9 * Math.max(1, cheb(x, y) + cheb(y, z)),
  });
  reg(105, {
    name: 'ham · 对称', intuition: '汉明距离统计不同分量个数，与比较方向无关。',
    math: 'h(x,y) = h(y,x)，h(x,y) = #{i : xᵢ ≠ yᵢ}', pseudocode: 'return ham(x,y)', complexity: 'O(n)',
    ref: 'Hamming (1950)', commonPitfall: '计数为整数，无舍入路径，精确判等。',
    invariant: '对 vecInt：exactNum(h(x,y), h(y,x))',
    inputs: ['x', 'y'], domains: { x: 'vecInt', y: 'vecInt' },
    run: (x, y) => ham(x, y),
    predicate: (x, y) => exactNum(ham(x, y), ham(y, x)),
  });
  reg(106, {
    name: 'ham · 自为 0 与三角', intuition: '自距离为 0；经第三点迂回不缩短（计数版三角不等式）。',
    math: 'h(x,x) = 0；h(x,z) ≤ h(x,y) + h(y,z)', pseudocode: 'return ham(x,z)', complexity: 'O(n)',
    ref: 'Hamming (1950) 度量性质', commonPitfall: '若 xᵢ≠zᵢ 则 xᵢ≠yᵢ 或 yᵢ≠zᵢ，逐位计数使三角不等式精确成立（整数计数无 epsilon）。',
    invariant: '对 vecInt：exactNum(h(x,x),0) 且 h(x,z) ≤ h(x,y)+h(y,z)',
    inputs: ['x', 'y', 'z'], domains: { x: 'vecInt', y: 'vecInt', z: 'vecInt' },
    run: (x, y, z) => ham(x, z),
    predicate: (x, y, z) => exactNum(ham(x, x), 0) && ham(x, z) <= ham(x, y) + ham(y, z),
  });
  reg(107, {
    name: 'l1dist · 对称', intuition: 'L1 距离对两个端点对称（绝对值差求和）。',
    math: 'd₁(x,y) = d₁(y,x)，d₁(x,y) = Σ|xᵢ−yᵢ|', pseudocode: 'return l1d(x,y)', complexity: 'O(n)',
    ref: 'Strang (2016) L1 距离', commonPitfall: '逐项 |a−b| 与 |b−a| 逐位相同，同序累加 → exactNum 精确。',
    invariant: '对 vecFinite：exactNum(d₁(x,y), d₁(y,x))',
    inputs: ['x', 'y'], domains: { x: 'vecFinite', y: 'vecFinite' },
    run: (x, y) => l1d(x, y),
    predicate: (x, y) => exactNum(l1d(x, y), l1d(y, x)),
  });
  reg(108, {
    name: 'l1dist · 三角与对角线', intuition: '自距离精确为 0；经第三点不缩短（L1 度量公理）。',
    math: 'd₁(x,x) = 0；d₁(x,z) ≤ d₁(x,y) + d₁(y,z)', pseudocode: 'return l1d(x,z)', complexity: 'O(n)',
    ref: 'Strang (2016) 距离函数', commonPitfall: '自减为 +0 故对角线精确；三角不等式加 1e-9 相对 epsilon。',
    invariant: '对 vecFinite：exactNum(d₁(x,x),0) 且 d₁(x,z) ≤ d₁(x,y)+d₁(y,z)+ε',
    inputs: ['x', 'y', 'z'], domains: { x: 'vecFinite', y: 'vecFinite', z: 'vecFinite' },
    run: (x, y, z) => l1d(x, z),
    predicate: (x, y, z) => exactNum(l1d(x, x), 0)
      && l1d(x, z) <= l1d(x, y) + l1d(y, z) + 1e-9 * Math.max(1, l1d(x, y) + l1d(y, z)),
  });

  // ── 前缀积 / 极值扫描 / argmax / one-hot / gather / softmax（B109–B120）─
  reg(109, {
    name: 'cumprod · 末项即总积', intuition: '前缀积最后一个分量就是全量积。',
    math: 'cumprod(x)[n−1] = Πᵢ xᵢ', pseudocode: 'return cumprod(x)', complexity: 'O(n)',
    ref: '前缀积（扫描）；NumPy cumprod', commonPitfall: '参考路径用二分树形乘积（结合顺序不同），相对误差 ~1e-15；vecSmall 控幅 ≤1e8 不溢出。',
    invariant: '对 vecSmall：relClose(cumprod(x)[n−1], treeProd(x), 1e-9)',
    inputs: ['x'], domains: { x: 'vecSmall' },
    run: (x) => cumprod(x),
    predicate: (x) => rc(cumprod(x)[x.length - 1], treeProd(x, 0, x.length), 1e-9),
  });
  reg(110, {
    name: 'cumprod · 逐项递推（商桥接）', intuition: '相邻前缀积之商恰为原分量（递推关系的除法验证）。',
    math: 'p[i] = p[i−1]·x[i] ⟺ p[i]/p[i−1] = x[i]', pseudocode: 'return cumprod(x)', complexity: 'O(n)',
    ref: '前缀积递推；数值分析除法桥接', commonPitfall: 'vecPos 保证前缀积 > 0（无零除）；除法是独立于乘法的参考路径；tol 1e-9。',
    invariant: '对 vecPos：exactNum(p[0],x[0]) 且 ∀i≥1 relClose(p[i]/p[i−1], x[i], 1e-9)',
    inputs: ['x'], domains: { x: 'vecPos' },
    run: (x) => cumprod(x),
    predicate: (x) => {
      const p = cumprod(x);
      if (!exactNum(p[0], x[0])) return false;
      for (let i = 1; i < p.length; i++) if (!rc(p[i] / p[i - 1], x[i], 1e-9)) return false;
      return true;
    },
  });
  reg(111, {
    name: 'cummax · 单调不减', intuition: '前缀最大值只增不减（滚动上界）。',
    math: 'cm[i] ≥ cm[i−1] ∀i≥1', pseudocode: 'return cummax(x)', complexity: 'O(n)',
    ref: 'NumPy maximum.accumulate', commonPitfall: '纯比较无舍入，逐位成立。',
    invariant: '对 vecFinite：∀i≥1, cummax(x)[i] ≥ cummax(x)[i−1]',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => cummax(x),
    predicate: (x) => { const c = cummax(x); for (let i = 1; i < c.length; i++) if (!(c[i] >= c[i - 1])) return false; return true; },
  });
  reg(112, {
    name: 'cummax · 末项即最大值', intuition: '滚动到末尾即收获全局最大值。',
    math: 'cummax(x)[n−1] = max(x)', pseudocode: 'return cummax(x)', complexity: 'O(n)',
    ref: '序统计；扫描等价性', commonPitfall: '两侧皆为同一最大值的精确选取（并列时取值相同），exactNum 判等。',
    invariant: '对 vecFinite：exactNum(cummax(x)[n−1], max(x))',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => cummax(x),
    predicate: (x) => exactNum(cummax(x)[x.length - 1], maxv(x)),
  });
  reg(113, {
    name: 'cummin · 单调不增', intuition: '前缀最小值只减不增（滚动下界）。',
    math: 'cm[i] ≤ cm[i−1] ∀i≥1', pseudocode: 'return cummin(x)', complexity: 'O(n)',
    ref: 'NumPy minimum.accumulate', commonPitfall: '纯比较无舍入，逐位成立。',
    invariant: '对 vecFinite：∀i≥1, cummin(x)[i] ≤ cummin(x)[i−1]',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => cummin(x),
    predicate: (x) => { const c = cummin(x); for (let i = 1; i < c.length; i++) if (!(c[i] <= c[i - 1])) return false; return true; },
  });
  reg(114, {
    name: 'cummin · 末项即最小值', intuition: '滚动到末尾即收获全局最小值。',
    math: 'cummin(x)[n−1] = min(x)', pseudocode: 'return cummin(x)', complexity: 'O(n)',
    ref: '序统计；扫描等价性', commonPitfall: '两侧皆为同一最小值的精确选取，exactNum 判等。',
    invariant: '对 vecFinite：exactNum(cummin(x)[n−1], min(x))',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => cummin(x),
    predicate: (x) => exactNum(cummin(x)[x.length - 1], minv(x)),
  });
  reg(115, {
    name: 'argmax · 选中最大值', intuition: 'argmax 下标处的分量就是全局最大值（扫描选元）。',
    math: 'x[argmax(x)] = max(x)', pseudocode: 'return argmaxVec(x)', complexity: 'O(n)',
    ref: '序统计；NumPy argmax（首个最大）', commonPitfall: 'argmax 取首个最大下标；maxv 独立扫描，选中值精确一致。',
    invariant: '对 vecFinite：exactNum(x[argmax(x)], max(x))',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => argmaxVec(x),
    predicate: (x) => exactNum(x[argmaxVec(x)], maxv(x)),
  });
  reg(116, {
    name: 'argmax · 单热命中', intuition: '单热向量的 argmax 恰为置 1 的下标（索引结构视角）。',
    math: 'argmax(onehot(k)) = k mod n ∈ [0,n)', pseudocode: 'return argmaxVec(onehot(n,k))', complexity: 'O(n)',
    ref: '索引代数；one-hot 与 argmax 互逆', commonPitfall: 'x 仅提供长度语境（值不参与）；k 经 mod n 归约覆盖满域索引；返回值恒落在 [0,n)。',
    invariant: '对 vecInt x, nat k：exactNum(argmax(onehot(k)), k mod n) 且 argmax ∈ [0,n)',
    inputs: ['x', 'k'], domains: { x: 'vecInt', k: 'nat' },
    run: (x, k) => argmaxVec(onehot(x.length, k)),
    predicate: (x, k) => {
      const a = argmaxVec(onehot(x.length, k));
      return exactNum(a, k % x.length) && a >= 0 && a < x.length;
    },
  });
  reg(117, {
    name: 'onehot · 恰一个 1', intuition: '单热向量恰一个分量为 1、其余为 0。',
    math: 'onehot(k) = e_{k mod n}：|{i : eᵢ = 1}| = 1', pseudocode: 'return onehot(n,k)', complexity: 'O(n)',
    ref: 'NumPy eye/one-hot 语义', commonPitfall: 'x 仅提供长度语境；k 归约后必落域，故恒恰一个 1（整数判等，无浮点歧义）。',
    invariant: '对 vecInt x, nat k：onehot(k) 恰一个 1 且其余分量均为 0',
    inputs: ['x', 'k'], domains: { x: 'vecInt', k: 'nat' },
    run: (x, k) => onehot(x.length, k),
    predicate: (x, k) => {
      const e = onehot(x.length, k);
      let ones = 0;
      for (let i = 0; i < e.length; i++) {
        if (e[i] === 1) ones++;
        else if (e[i] !== 0) return false;
      }
      return ones === 1;
    },
  });
  reg(118, {
    name: 'onehot · 和为 1', intuition: '单热向量各分量之和为 1（单纯形顶点）。',
    math: 'Σᵢ onehot(k)ᵢ = 1', pseudocode: 'return sumv(onehot(n,k))', complexity: 'O(n)',
    ref: '单纯形结构', commonPitfall: '整数计数无舍入，exactNum 精确。',
    invariant: '对 vecInt x, nat k：exactNum(Σ onehot(k)ᵢ, 1)',
    inputs: ['x', 'k'], domains: { x: 'vecInt', k: 'nat' },
    run: (x, k) => sumv(onehot(x.length, k)),
    predicate: (x, k) => exactNum(sumv(onehot(x.length, k)), 1),
  });
  reg(119, {
    name: 'gather · 单热桥接', intuition: '按下标取址等价于与单热向量点积（选元的两种路径）。',
    math: 'gather(v,i) = v[i mod n] = v · onehot(i)', pseudocode: 'return gather(v,i)', complexity: 'O(n)',
    ref: 'Strang (2016) 标准基选元；gather/embedding 查表', commonPitfall: 'one-hot 点积是独立参考路径（线性组合 vs 直接寻址）；vecInt 下精确（其余项乘 0 不影响和）。',
    invariant: '对 vecInt x, nat i：exactNum(gather(x,i), x·onehot(i))',
    inputs: ['x', 'i'], domains: { x: 'vecInt', i: 'nat' },
    run: (x, i) => gather(x, i),
    predicate: (x, i) => exactNum(gather(x, i), dot(x, onehot(x.length, i))),
  });
  reg(120, {
    name: 'softmax · 和为 1', intuition: 'softmax 输出落在概率单纯形上（结构和为 1）。',
    math: 'Σᵢ softmax(x)ᵢ = 1', pseudocode: 'return softmaxVec(x)', complexity: 'O(n)',
    ref: 'Goodfellow 2016 §6.2', commonPitfall: 'max 移位保证 exp 不溢出（vecSmall |x|≤10）；tol 1e-9。F 域已覆盖 0<p<1 与 Σp²≤1，此处补单纯形归一性，不重复同一性质。',
    invariant: '对 vecSmall：relClose(Σ softmax(x)ᵢ, 1, 1e-9)',
    inputs: ['x'], domains: { x: 'vecSmall' },
    run: (x) => softmaxVec(x),
    predicate: (x) => rc(sumv(softmaxVec(x)), 1, 1e-9),
  });

  // ── 切片与组合（B121–B128）────────────────────────────────────────────
  reg(121, {
    name: 'concat · 长度可加', intuition: '拼接后的长度等于两段长度之和。',
    math: '|concat(x,y)| = |x| + |y|', pseudocode: 'return vConcat(x,y)', complexity: 'O(n+m)',
    ref: 'NumPy concatenate 语义', commonPitfall: 'vecDims 取异长（5/3），验证长度可加而非等长巧合。',
    invariant: '对 vecInt x,y：exactNum(|concat(x,y)|, |x|+|y|)',
    inputs: ['x', 'y'], domains: { x: 'vecInt', y: 'vecInt' }, vecDims: { x: 5, y: 3 },
    run: (x, y) => vConcat(x, y),
    predicate: (x, y) => exactNum(vConcat(x, y).length, x.length + y.length),
  });
  reg(122, {
    name: 'concat · 偏移切片还原', intuition: '按偏移量半开切片可完整还原两段原向量。',
    math: 'slice(concat(x,y),0,|x|) = x；slice(concat(x,y),|x|,|x|+|y|) = y', pseudocode: 'return vConcat(x,y)', complexity: 'O(n+m)',
    ref: '索引代数；半开区间边界', commonPitfall: '偏移 |x| 处归属后半（[0,|x|) 与 [|x|,|x|+|y|) 不重不漏）。',
    invariant: '对 vecInt x,y：exactVec(两段还原切片, 原向量)',
    inputs: ['x', 'y'], domains: { x: 'vecInt', y: 'vecInt' }, vecDims: { x: 5, y: 3 },
    run: (x, y) => vConcat(x, y),
    predicate: (x, y) => {
      const c = vConcat(x, y);
      return exactVec(vSlice(c, 0, x.length), x) && exactVec(vSlice(c, x.length, c.length), y);
    },
  });
  reg(123, {
    name: 'slice · 长度即差', intuition: '半开区间切片的长度恰为终点减起点。',
    math: '|slice(x,s,e)| = e − s（0 ≤ s ≤ e ≤ n）', pseudocode: 'return vSlice(x, lo, hi)', complexity: 'O(e−s)',
    ref: 'Python/NumPy 半开区间切片语义', commonPitfall: 'TAGS 无索引域：nat 索引经 mod (n+1) 归约到 [0,n] 并排序，覆盖空切片与满切片边界；若实现成 [s,e] 则长度多 1 即红。',
    invariant: '对 vecInt x, nat s,e：exactNum(|slice|, hi−lo)，lo=min(s,e) 归约, hi=max(s,e) 归约',
    inputs: ['x', 's', 'e'], domains: { x: 'vecInt', s: 'nat', e: 'nat' },
    run: (x, s, e) => { const r = idx2(s, e, x.length); return vSlice(x, r[0], r[1]); },
    predicate: (x, s, e) => { const r = idx2(s, e, x.length); return exactNum(vSlice(x, r[0], r[1]).length, r[1] - r[0]); },
  });
  reg(124, {
    name: 'slice · 内容逐位一致', intuition: '切片第 i 个分量就是原向量第 s+i 个分量（半开区间选元）。',
    math: 'slice(x,s,e)[i] = x[s+i] ∀i∈[0,e−s)', pseudocode: 'return vSlice(x, lo, hi)', complexity: 'O(e−s)',
    ref: 'Python/NumPy 半开区间切片语义', commonPitfall: '内容经 one-hot 点积独立寻址验证（线性组合路径），非简单复抄；归约方式同 B123。',
    invariant: '对 vecInt x, nat s,e：∀i, exactNum(slice[i], x·onehot(lo+i))',
    inputs: ['x', 's', 'e'], domains: { x: 'vecInt', s: 'nat', e: 'nat' },
    run: (x, s, e) => { const r = idx2(s, e, x.length); return vSlice(x, r[0], r[1]); },
    predicate: (x, s, e) => {
      const r = idx2(s, e, x.length);
      const sl = vSlice(x, r[0], r[1]);
      if (sl.length !== r[1] - r[0]) return false;
      for (let i = 0; i < sl.length; i++) if (!exactNum(sl[i], dot(x, onehot(x.length, r[0] + i)))) return false;
      return true;
    },
  });
  reg(125, {
    name: 'pad · 前补零保原值', intuition: '前面补零到目标长度，原值整体后移保持逐位不变。',
    math: 'pad(x,m) = [0^(m−n), x]，m = n + (k mod 5)', pseudocode: 'return padFront(x, m)', complexity: 'O(m)',
    ref: 'NumPy pad（前补零）语义', commonPitfall: 'k mod 5 覆盖 0..4 个补零（含零补零边界）；尾部经 one-hot 点积独立寻址验证。',
    invariant: '对 vecInt x, nat k：|pad|=m，前 m−n 个为 0，尾部逐位还原 x',
    inputs: ['x', 'k'], domains: { x: 'vecInt', k: 'nat' },
    run: (x, k) => padFront(x, x.length + (k % 5)),
    predicate: (x, k) => {
      const n = x.length, m = n + (k % 5), p = padFront(x, m);
      if (p.length !== m) return false;
      for (let i = 0; i < m - n; i++) if (p[i] !== 0) return false;
      for (let i = 0; i < n; i++) if (!exactNum(p[m - n + i], dot(x, onehot(n, i)))) return false;
      return true;
    },
  });
  reg(126, {
    name: 'truncate · 前 n 项', intuition: '截取前 m 个分量，内容逐位保持。',
    math: 'truncate(x,m) = x[0..m)，m = k mod (n+1)', pseudocode: 'return vTruncate(x, m)', complexity: 'O(m)',
    ref: 'NumPy/PyTorch 截断语义', commonPitfall: 'm 经 mod (n+1) 归约覆盖 m=0（空）与 m=n（恒等）边界；内容经 one-hot 点积独立验证。',
    invariant: '对 vecInt x, nat k：exactNum(|truncate|, m) 且 ∀i<m exactNum(t[i], x·onehot(i))',
    inputs: ['x', 'k'], domains: { x: 'vecInt', k: 'nat' },
    run: (x, k) => vTruncate(x, k % (x.length + 1)),
    predicate: (x, k) => {
      const m = k % (x.length + 1), t = vTruncate(x, m);
      if (t.length !== m) return false;
      for (let i = 0; i < m; i++) if (!exactNum(t[i], dot(x, onehot(x.length, i)))) return false;
      return true;
    },
  });
  reg(127, {
    name: 'rotate · 复合还原', intuition: '循环移位 k 位再移位 n−k 位回到原向量（群结构）。',
    math: 'rotate(rotate(v,k), n−k) = v（模 n 意义下）', pseudocode: 'return vRotate(v,k)', complexity: 'O(n)',
    ref: '循环群索引代数', commonPitfall: '纯索引置换无数值舍入，vecInt 下逐位精确；k=0 时复合为恒等。',
    invariant: '对 vecInt x, nat k：exactVec(rotate(rotate(x,k),n−k mod n), x)',
    inputs: ['x', 'k'], domains: { x: 'vecInt', k: 'nat' },
    run: (x, k) => vRotate(x, k),
    predicate: (x, k) => {
      const n = x.length, kw = k % n;
      return exactVec(vRotate(vRotate(x, kw), (n - kw) % n), x);
    },
  });
  reg(128, {
    name: 'reverse · 与 rotate 复合', intuition: '先反转再循环移位，等于先循环移位（反方向）再反转。',
    math: 'reverse(rotate(v,k)) = rotate(reverse(v), n−k)', pseudocode: 'return vRev(vRotate(v,k))', complexity: 'O(n)',
    ref: '索引代数（反转与循环移位的交换关系）', commonPitfall: '两侧均为索引置换，逐位精确；n−k 取模 n。',
    invariant: '对 vecInt x, nat k：exactVec(reverse(rotate(x,k)), rotate(reverse(x),n−k mod n))',
    inputs: ['x', 'k'], domains: { x: 'vecInt', k: 'nat' },
    run: (x, k) => vRev(vRotate(x, k)),
    predicate: (x, k) => {
      const n = x.length, kw = k % n;
      return exactVec(vRev(vRotate(x, kw)), vRotate(vRev(x), (n - kw) % n));
    },
  });
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

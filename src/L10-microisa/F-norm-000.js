/* ============================================================================
 * L10 微指令 · F 域（归一化）· 第 1/3 批（F001–F032）
 * ----------------------------------------------------------------------------
 * 所属层：L10-microIsa  域：F（归一化）
 * 允许依赖的层：L9（rng/assert/special）、L10（registry/invariant）
 * 主要 @ref：Goodfellow et al. 2016（BatchNorm / LayerNorm）；Zhang & Sennrich 2019
 *           （RMSNorm）；Wu & He 2018（GroupNorm / 权重标准化）；Krizhevsky 2012（LRN）
 *
 * 子域划分：
 *   F001–F012 均值（定义 / 线性 / 平移 / 矩阵按行·列 / 边界）
 *   F013–F024 方差与标准差（总体 / 样本 / 与均方关系 / 切比雪夫界）
 *   F025–F032 L2 范数与向量单位化
 *
 * 不变量判据：数值类用 relClose（tol 1e-9，方差类 1e-12）。每条 predicate 先调用
 * NS.L10.ops['Fxxx'].run(...) 检验真实产出，再走与 run 不同的参考路径（反向累加 /
 * Σx²/n−m² / 三角不等式 / 解析界）比对，避免「自证」。
 * 输入域由 L10.TAGS 约束（tests/L10 据此构造 fast-check 任意子）。
 * 约定：向量默认长度 8（vecSmall/vecPos），矩阵默认 4×4（matSmall）。
 * ==========================================================================*/
;(function (NS) {
  'use strict';
  NS.L10 = NS.L10 || {};
  const rc = NS.L10.relClose;
  function reg(i, o) { NS.L10.register('F', i, o); }
  function runOf(key) { return NS.L10.ops[key].run; }

  /* ---------- 内部工具（本文件自包含，不跨层 require） ---------- */
  const sq = (M) => Math.round(Math.sqrt(M.length));
  function add(a, b) { const o = new Float64Array(a.length); for (let i = 0; i < a.length; i++) o[i] = a[i] + b[i]; return o; }
  function scale(a, c) { const o = new Float64Array(a.length); for (let i = 0; i < a.length; i++) o[i] = c * a[i]; return o; }
  function dot(a, b, n) { let s = 0; for (let i = 0; i < n; i++) s += a[i] * b[i]; return s; }

  function mean(a, n) { let s = 0; for (let i = 0; i < n; i++) s += a[i]; return s / n; }
  function meanRev(a, n) { let s = 0; for (let i = n - 1; i >= 0; i--) s += a[i]; return s / n; }
  function sumSq(a, n) { let s = 0; for (let i = 0; i < n; i++) s += a[i] * a[i]; return s; }
  function sumSqRev(a, n) { let s = 0; for (let i = n - 1; i >= 0; i--) s += a[i] * a[i]; return s; }
  // 总体方差：直接两遍；参考路径：Σx²/n − m²（反向均值）
  function varPop(a, n) { const m = mean(a, n); let s = 0; for (let i = 0; i < n; i++) { const d = a[i] - m; s += d * d; } return s / n; }
  function varPopRef(a, n) { const m = meanRev(a, n); const ss = sumSqRev(a, n); return Math.max(0, ss / n - m * m); }
  function stdPop(a, n) { return Math.sqrt(varPop(a, n)); }
  function l2(a, n) { return Math.sqrt(sumSq(a, n)); }
  function l2Rev(a, n) { return Math.sqrt(sumSqRev(a, n)); }

  // ══ 均值（F001–F012）════════════════════════════════════════════════
  reg(1, {
    name: '均值 · 定义', intuition: '均值是累加均分，最朴素的集中趋势。',
    math: 'mean(x) = Σxᵢ/n', pseudocode: 's=0; for i: s+=x[i]; return s/n',
    complexity: 'O(n)', ref: '描述统计', commonPitfall: '浮点累加不可结合，两遍求均与单遍 Kahan 结果差 ~1e-15。',
    invariant: 'relClose(run, 反向累加, 1e-9)',
    inputs: ['x'], domains: { x: 'vecSmall' },
    run: (x) => mean(x, x.length),
    predicate: (x) => { const r = runOf('F001')(x); return rc(r, meanRev(x, x.length), 1e-9); },
  });
  reg(2, {
    name: '均值 · 与顺序无关', intuition: '求和是交换的，重排向量不改变均值。',
    math: 'mean(perm(x)) = mean(x)', pseudocode: 'return mean(reverse(x))',
    complexity: 'O(n)', ref: '交换律', commonPitfall: '浮点下不同顺序舍入不同，逐位断言会挂——用 1e-9。',
    invariant: 'relClose(run, mean(reversed), 1e-9)',
    inputs: ['x'], domains: { x: 'vecSmall' },
    run: (x) => mean(x, x.length),
    predicate: (x) => { const r = runOf('F002')(x); const rv = new Float64Array(x.length); for (let i = 0; i < x.length; i++) rv[i] = x[x.length - 1 - i]; return rc(r, mean(rv, rv.length), 1e-9); },
  });
  reg(3, {
    name: '均值 · 对加法线性', intuition: '两向量逐位相加再求均，等于分别求均再相加。',
    math: 'mean(a+b) = mean(a)+mean(b)', pseudocode: 'return mean(add(a,b))',
    complexity: 'O(n)', ref: '线性', commonPitfall: '浮点路径不同须容差。',
    invariant: 'relClose(run, mean(a)+mean(b), 1e-9)',
    inputs: ['a', 'b'], domains: { a: 'vecSmall', b: 'vecSmall' },
    run: (a, b) => mean(add(a, b), a.length),
    predicate: (a, b) => { const r = runOf('F003')(a, b); return rc(r, mean(a, a.length) + mean(b, b.length), 1e-9); },
  });
  reg(4, {
    name: '均值 · 齐次性', intuition: '整体缩放 a 倍，均值也缩放 a 倍。',
    math: 'mean(c·x) = c·mean(x)', pseudocode: 'return mean(scale(x,c))',
    complexity: 'O(n)', ref: '齐次', commonPitfall: 'c 为负时符号翻转，用容差规避。',
    invariant: 'relClose(run, c·mean(x), 1e-9)',
    inputs: ['c', 'x'], domains: { c: 'small', x: 'vecSmall' },
    run: (c, x) => mean(scale(x, c), x.length),
    predicate: (c, x) => { const r = runOf('F004')(c, x); return rc(r, c * mean(x, x.length), 1e-9); },
  });
  reg(5, {
    name: '均值 · 平移', intuition: '整体加常数 c，均值也加 c。',
    math: 'mean(x+c) = mean(x)+c', pseudocode: 'return mean(shift(x,c))',
    complexity: 'O(n)', ref: '平移', commonPitfall: '无',
    invariant: 'relClose(run, mean(x)+c, 1e-9)',
    inputs: ['x', 'c'], domains: { x: 'vecSmall', c: 'small' },
    run: (x, c) => { const t = new Float64Array(x.length); for (let i = 0; i < x.length; i++) t[i] = x[i] + c; return mean(t, x.length); },
    predicate: (x, c) => { const r = runOf('F005')(x, c); return rc(r, mean(x, x.length) + c, 1e-9); },
  });
  reg(6, {
    name: '均值 · 常数向量', intuition: '分量全相等的向量，其均值恰为该常数。',
    math: 'mean(c,c,…,c) = c', pseudocode: 'return mean(constVec)',
    complexity: 'O(n)', ref: '定义', commonPitfall: '无',
    invariant: 'relClose(run, c, 1e-9)',
    inputs: ['x'], domains: { x: 'vecSmall' },
    run: (x) => { const c = x[0], n = x.length; let s = 0; for (let i = 0; i < n; i++) s += c; return s / n; },
    predicate: (x) => { const r = runOf('F006')(x); return rc(r, x[0], 1e-9); },
  });
  reg(7, {
    name: '均值 · 矩阵按行', intuition: '批量数据中每行是一条样本，逐行求均得各样本均值。',
    math: 'rᵢ = mean(Mᵢ,:)', pseudocode: 'for i: rowMean[i] = mean(M[i*D:(i+1)*D])',
    complexity: 'O(N·D)', ref: '描述统计', commonPitfall: '行优先索引别写反。',
    invariant: '逐元素 relClose(run, 反向累加参考, 1e-9)',
    inputs: ['M'], domains: { M: 'matSmall' }, matrixDims: { M: [4, 4] },
    run: (M) => { const n = sq(M), d = M.length / n, out = new Float64Array(n); for (let i = 0; i < n; i++) out[i] = mean(M.subarray(i * d, (i + 1) * d), d); return out; },
    predicate: (M) => { const r = runOf('F007')(M); const n = sq(M), d = M.length / n; const ref = new Float64Array(n); for (let i = 0; i < n; i++) ref[i] = meanRev(M.subarray(i * d, (i + 1) * d), d); return rc(r, ref, 1e-9); },
  });
  reg(8, {
    name: '均值 · 矩阵按列', intuition: '特征维上逐列求均，得各通道的批量均值（BatchNorm 的 μ）。',
    math: 'cⱼ = mean(M:,ⱼ)', pseudocode: 'for j: colMean[j] = mean(M[:,j])',
    complexity: 'O(N·D)', ref: '描述统计', commonPitfall: 'BatchNorm 的 μ 正是按列（通道）均值。',
    invariant: '逐元素 relClose(run, 反向累加参考, 1e-9)',
    inputs: ['M'], domains: { M: 'matSmall' }, matrixDims: { M: [4, 4] },
    run: (M) => { const n = sq(M), d = M.length / n, out = new Float64Array(d); for (let j = 0; j < d; j++) { let s = 0; for (let i = 0; i < n; i++) s += M[i * d + j]; out[j] = s / n; } return out; },
    predicate: (M) => { const r = runOf('F008')(M); const n = sq(M), d = M.length / n; const ref = new Float64Array(d); for (let j = 0; j < d; j++) { let s = 0; for (let i = n - 1; i >= 0; i--) s += M[i * d + j]; ref[j] = s / n; } return rc(r, ref, 1e-9); },
  });
  reg(9, {
    name: '均值 · 零向量', intuition: '全零向量的均值恰为 0。',
    math: 'mean(0,…,0) = 0', pseudocode: 'return mean(zeros)',
    complexity: 'O(1)', ref: '定义', commonPitfall: '−0 与 0 在 Object.is 下不同，用容差。',
    invariant: 'relClose(run, 0, 1e-12)',
    inputs: ['x'], domains: { x: 'vecSmall' },
    run: (x) => { const n = x.length; let s = 0; for (let i = 0; i < n; i++) s += 0; return s / n; },
    predicate: (x) => { const r = runOf('F009')(x); return rc(r, 0, 1e-12); },
  });
  reg(10, {
    name: '均值 · 落于极值之间', intuition: '均值恒在最小值与最大值之间，是最稳健的界。',
    math: 'min(x) ≤ mean(x) ≤ max(x)', pseudocode: 'return mean(x)',
    complexity: 'O(n)', ref: '均值不等式', commonPitfall: '浮点误差下取 ≤ 并留 1e-12 裕度。',
    invariant: 'min(x)−tol ≤ run ≤ max(x)+tol',
    inputs: ['x'], domains: { x: 'vecSmall' },
    run: (x) => mean(x, x.length),
    predicate: (x) => { const n = x.length; const r = runOf('F010')(x); let lo = Infinity, hi = -Infinity; for (let i = 0; i < n; i++) { if (x[i] < lo) lo = x[i]; if (x[i] > hi) hi = x[i]; } return lo - 1e-12 <= r && r <= hi + 1e-12; },
  });
  reg(11, {
    name: '均值 · 正向平移单调', intuition: '每项叠加正数，均值不降。',
    math: 'c>0 ⇒ mean(x+c) ≥ mean(x)', pseudocode: 'return mean(shift(x,c))',
    complexity: 'O(n)', ref: '单调性', commonPitfall: 'c 须为正；用容差。',
    invariant: 'relClose(run, mean(x)+c, 1e-9) 且 run ≥ mean(x) − 1e-12',
    inputs: ['x', 'c'], domains: { x: 'vecSmall', c: 'smallPos' },
    run: (x, c) => { const t = new Float64Array(x.length); for (let i = 0; i < x.length; i++) t[i] = x[i] + c; return mean(t, x.length); },
    predicate: (x, c) => { const r = runOf('F011')(x, c); return rc(r, mean(x, x.length) + c, 1e-9) && r >= mean(x, x.length) - 1e-12; },
  });
  reg(12, {
    name: '均值 · 与求和关系', intuition: '均值乘以长度即总和，二者等价。',
    math: 'mean(x)·n = Σxᵢ', pseudocode: 'return mean(x)*n',
    complexity: 'O(n)', ref: '定义', commonPitfall: '大数下乘 n 放大误差，容差 1e-9。',
    invariant: 'relClose(run, Σxᵢ, 1e-9)',
    inputs: ['x'], domains: { x: 'vecSmall' },
    run: (x) => mean(x, x.length) * x.length,
    predicate: (x) => { const r = runOf('F012')(x); const n = x.length; let s = 0; for (let i = 0; i < n; i++) s += x[i]; return rc(r, s, 1e-9); },
  });

  // ══ 方差与标准差（F013–F024）════════════════════════════════════════
  reg(13, {
    name: '总体方差 · 定义', intuition: '方差是离均差平方的平均，度量离散程度。',
    math: 'Var = Σ(xᵢ−μ)²/n', pseudocode: 'm=mean(x); s=0; for i: s+=(x[i]-m)^2; return s/n',
    complexity: 'O(n)', ref: '描述统计', commonPitfall: '两遍法（先求均再求差平方）数值最稳。',
    invariant: 'relClose(run, Σx²/n−μ² 参考路径, 1e-9)',
    inputs: ['x'], domains: { x: 'vecSmall' },
    run: (x) => varPop(x, x.length),
    predicate: (x) => { const r = runOf('F013')(x); return rc(r, varPopRef(x, x.length), 1e-9); },
  });
  reg(14, {
    name: '总体方差 · 非负', intuition: '离均差平方和必非负。',
    math: 'Var ≥ 0', pseudocode: 'return varPop(x)',
    complexity: 'O(n)', ref: '定义', commonPitfall: '浮点可能给出 −1e-16，留 1e-12 裕度。',
    invariant: 'run ≥ −1e-12 且 relClose(run, 参考路径, 1e-9)',
    inputs: ['x'], domains: { x: 'vecSmall' },
    run: (x) => varPop(x, x.length),
    predicate: (x) => { const r = runOf('F014')(x); return r >= -1e-12 && rc(r, varPopRef(x, x.length), 1e-9); },
  });
  reg(15, {
    name: '总体方差 · 常数向量为 0', intuition: '无离散度的数据方差为 0。',
    math: 'Var(c,c,…,c) = 0', pseudocode: 'return varPop(constVec)',
    complexity: 'O(n)', ref: '定义', commonPitfall: '无',
    invariant: 'relClose(run, 0, 1e-12)',
    inputs: ['x'], domains: { x: 'vecSmall' },
    run: (x) => { const c = x[0], n = x.length; const v = new Float64Array(n); for (let i = 0; i < n; i++) v[i] = c; return varPop(v, n); },
    predicate: (x) => { const r = runOf('F015')(x); return rc(r, 0, 1e-12); },
  });
  reg(16, {
    name: '总体方差 · 平移不变', intuition: '整体加常数不改变离散程度（位置变了，形状没变）。',
    math: 'Var(x+c) = Var(x)', pseudocode: 'return varPop(shift(x,c))',
    complexity: 'O(n)', ref: '平移不变性', commonPitfall: '无',
    invariant: 'relClose(run, Var(x), 1e-9)',
    inputs: ['x', 'c'], domains: { x: 'vecSmall', c: 'small' },
    run: (x, c) => { const t = new Float64Array(x.length); for (let i = 0; i < x.length; i++) t[i] = x[i] + c; return varPop(t, x.length); },
    predicate: (x, c) => { const r = runOf('F016')(x, c); return rc(r, varPop(x, x.length), 1e-9); },
  });
  reg(17, {
    name: '总体方差 · 齐次（缩放平方）', intuition: '整体缩放 a 倍，方差缩放 a² 倍。',
    math: 'Var(a·x) = a²·Var(x)', pseudocode: 'return varPop(scale(x,a))',
    complexity: 'O(n)', ref: '齐次', commonPitfall: '注意是 a² 而非 a；a 负时 a² 仍正。',
    invariant: 'relClose(run, a²·Var(x), 1e-9)',
    inputs: ['a', 'x'], domains: { a: 'small', x: 'vecSmall' },
    run: (a, x) => varPop(scale(x, a), x.length),
    predicate: (a, x) => { const r = runOf('F017')(a, x); return rc(r, a * a * varPop(x, x.length), 1e-9); },
  });
  reg(18, {
    name: '总体方差 · 与均方关系', intuition: '方差等于「平方的均值」减「均值的平方」（简化公式）。',
    math: 'Var = E[x²] − E[x]²', pseudocode: 'return sumSq(x)/n - mean(x)^2',
    complexity: 'O(n)', ref: '方差恒等式', commonPitfall: '简化公式相减易抵消精度，容差 1e-9。',
    invariant: 'relClose(run, Σx²/n − μ², 1e-9)',
    inputs: ['x'], domains: { x: 'vecSmall' },
    run: (x) => varPop(x, x.length),
    predicate: (x) => { const r = runOf('F018')(x); const n = x.length; return rc(r, sumSq(x, n) / n - Math.pow(mean(x, n), 2), 1e-9); },
  });
  reg(19, {
    name: '样本方差 · 定义（无偏）', intuition: '用 n−1 而非 n 做分母，消除对总体方差的低估。',
    math: 's² = Σ(xᵢ−μ)²/(n−1)', pseudocode: 'm=mean(x); s=0; for i: s+=(x[i]-m)^2; return s/(n-1)',
    complexity: 'O(n)', ref: '统计推断', commonPitfall: 'n=1 时分母为 0——故域长度固定 ≥2（vecSmall defLen=8）。',
    invariant: 'relClose(run, (Σx²−nμ²)/(n−1) 参考, 1e-9)',
    inputs: ['x'], domains: { x: 'vecSmall' },
    run: (x) => { const n = x.length, m = mean(x, n); let s = 0; for (let i = 0; i < n; i++) { const d = x[i] - m; s += d * d; } return s / (n - 1); },
    predicate: (x) => { const r = runOf('F019')(x); const n = x.length, m = meanRev(x, n); const ss = sumSqRev(x, n); const ref = Math.max(0, ss - n * m * m) / (n - 1); return rc(r, ref, 1e-9); },
  });
  reg(20, {
    name: '样本方差 · 与总体关系', intuition: '样本方差恰为总体方差的 n/(n−1) 倍（Bessel 校正）。',
    math: 's² = n/(n−1)·Var', pseudocode: 'return varSample(x)',
    complexity: 'O(n)', ref: 'Bessel 校正', commonPitfall: '无',
    invariant: 'relClose(run, n/(n−1)·Var, 1e-9)',
    inputs: ['x'], domains: { x: 'vecSmall' },
    run: (x) => { const n = x.length, m = mean(x, n); let s = 0; for (let i = 0; i < n; i++) { const d = x[i] - m; s += d * d; } return s / (n - 1); },
    predicate: (x) => { const r = runOf('F020')(x); const n = x.length; return rc(r, (n / (n - 1)) * varPop(x, n), 1e-9); },
  });
  reg(21, {
    name: '标准差 · 定义', intuition: '标准差是方差的平方根，与原始量纲一致。',
    math: 'σ = √Var', pseudocode: 'return sqrt(varPop(x))',
    complexity: 'O(n)', ref: '描述统计', commonPitfall: '方差非负，开方安全。',
    invariant: 'relClose(run, √Var, 1e-12)',
    inputs: ['x'], domains: { x: 'vecSmall' },
    run: (x) => stdPop(x, x.length),
    predicate: (x) => { const r = runOf('F021')(x); return rc(r, Math.sqrt(varPop(x, x.length)), 1e-12); },
  });
  reg(22, {
    name: '标准差 · 齐次（缩放绝对值）', intuition: '缩放 a 倍，标准差缩放 |a| 倍。',
    math: 'σ(a·x) = |a|·σ(x)', pseudocode: 'return sqrt(varPop(scale(x,a)))',
    complexity: 'O(n)', ref: '齐次', commonPitfall: '取绝对值，a 负时同号。',
    invariant: 'relClose(run, |a|·σ(x), 1e-9)',
    inputs: ['a', 'x'], domains: { a: 'small', x: 'vecSmall' },
    run: (a, x) => stdPop(scale(x, a), x.length),
    predicate: (a, x) => { const r = runOf('F022')(a, x); return rc(r, Math.abs(a) * stdPop(x, x.length), 1e-9); },
  });
  reg(23, {
    name: '标准差 · 非负', intuition: '标准差为非负实数。',
    math: 'σ ≥ 0', pseudocode: 'return stdPop(x)',
    complexity: 'O(n)', ref: '定义', commonPitfall: '无',
    invariant: 'run ≥ −1e-12 且 relClose(run, √VarRef, 1e-9)',
    inputs: ['x'], domains: { x: 'vecSmall' },
    run: (x) => stdPop(x, x.length),
    predicate: (x) => { const r = runOf('F023')(x); return r >= -1e-12 && rc(r, Math.sqrt(varPopRef(x, x.length)), 1e-9); },
  });
  reg(24, {
    name: '切比雪夫界（样本版）', intuition: '距均值超过 kσ 的元素占比，绝不超过 1/k²——最宽松也最稳的分布界。',
    math: '|{i: |xᵢ−μ|>kσ}|/n ≤ 1/k²', pseudocode: 'count beyond kσ; return count/n',
    complexity: 'O(n)', ref: 'Chebyshev 不等式（样本确定性版）', commonPitfall: '这是样本上的确定性事实（Σ(x−μ)² > k²σ²·#beyond），非概率近似；须用严格 >，否则 σ=0 的常数向量会把全部元素计入离群（0≥0），界被违反。',
    invariant: 'run ∈ [0,1] 且 run ≤ 1/k² + 1e-12',
    inputs: ['x', 'k'], domains: { x: 'vecSmall', k: 'smallPos' },
    run: (x, k) => { const n = x.length, m = mean(x, n), sd = stdPop(x, n); let c = 0; for (let i = 0; i < n; i++) if (Math.abs(x[i] - m) > k * sd) c++; return c / n; },
    predicate: (x, k) => { const r = runOf('F024')(x, k); return r >= 0 && r <= 1 && r <= 1 / (k * k) + 1e-12; },
  });

  // ══ L2 范数与向量单位化（F025–F032）════════════════════════════════
  reg(25, {
    name: 'L2 范数 · 定义', intuition: '欧氏长度是分量平方和的平方根。',
    math: '‖x‖₂ = √Σxᵢ²', pseudocode: 's=0; for i: s+=x[i]^2; return sqrt(s)',
    complexity: 'O(n)', ref: '线性代数', commonPitfall: '大分量下平方和易溢出，域限 small 规避。',
    invariant: 'relClose(run, 反向累加, 1e-9)',
    inputs: ['x'], domains: { x: 'vecSmall' },
    run: (x) => l2(x, x.length),
    predicate: (x) => { const r = runOf('F025')(x); return rc(r, l2Rev(x, x.length), 1e-9); },
  });
  reg(26, {
    name: 'L2 范数 · 齐次', intuition: '缩放 a 倍，范数缩放 |a| 倍（齐次函数）。',
    math: '‖a·x‖ = |a|·‖x‖', pseudocode: 'return l2(scale(x,a))',
    complexity: 'O(n)', ref: '齐次', commonPitfall: '取绝对值。',
    invariant: 'relClose(run, |a|·‖x‖, 1e-9)',
    inputs: ['a', 'x'], domains: { a: 'small', x: 'vecSmall' },
    run: (a, x) => l2(scale(x, a), x.length),
    predicate: (a, x) => { const r = runOf('F026')(a, x); return rc(r, Math.abs(a) * l2(x, x.length), 1e-9); },
  });
  reg(27, {
    name: 'L2 范数 · 三角不等式', intuition: '两边之和不小于第三边，范数的核心公理。',
    math: '‖x+y‖ ≤ ‖x‖ + ‖y‖', pseudocode: 'return l2(add(x,y))',
    complexity: 'O(n)', ref: 'Minkowski 不等式', commonPitfall: '浮点误差下取 ≤ 并留 1e-9 裕度。',
    invariant: 'run ≤ ‖x‖ + ‖y‖ + 1e-9 且 run ≥ |‖x‖−‖y‖| − 1e-9',
    inputs: ['x', 'y'], domains: { x: 'vecSmall', y: 'vecSmall' },
    run: (x, y) => l2(add(x, y), x.length),
    predicate: (x, y) => { const r = runOf('F027')(x, y); const lx = l2(x, x.length), ly = l2(y, y.length); return r <= lx + ly + 1e-9 && r >= Math.abs(lx - ly) - 1e-9; },
  });
  reg(28, {
    name: 'L2 范数 · 非负', intuition: '长度非负。',
    math: '‖x‖ ≥ 0', pseudocode: 'return l2(x)',
    complexity: 'O(n)', ref: '定义', commonPitfall: '无',
    invariant: 'run ≥ −1e-12 且 relClose(run, 反向累加, 1e-9)',
    inputs: ['x'], domains: { x: 'vecSmall' },
    run: (x) => l2(x, x.length),
    predicate: (x) => { const r = runOf('F028')(x); return r >= -1e-12 && rc(r, l2Rev(x, x.length), 1e-9); },
  });
  reg(29, {
    name: 'L2 范数 · 零向量为 0', intuition: '零向量长度恰为 0。',
    math: '‖0‖ = 0', pseudocode: 'return l2(zeros)',
    complexity: 'O(1)', ref: '定义', commonPitfall: '−0 用容差。',
    invariant: 'relClose(run, 0, 1e-12)',
    inputs: ['x'], domains: { x: 'vecSmall' },
    run: (x) => { let s = 0; for (let i = 0; i < x.length; i++) s += 0; return Math.sqrt(s); },
    predicate: (x) => { const r = runOf('F029')(x); return rc(r, 0, 1e-12); },
  });
  reg(30, {
    name: '单位化 · 定义（方向保留长度为 1）', intuition: '除以自身长度，得到同方向的单位向量（L2 归一化）。',
    math: 'u = x/‖x‖₂，‖u‖ = 1', pseudocode: 'return x / l2(x)',
    complexity: 'O(n)', ref: '向量归一化', commonPitfall: '零向量除零——域用 vecPos 保证 ‖x‖>0。',
    invariant: 'relClose(‖run‖, 1, 1e-9)',
    inputs: ['x'], domains: { x: 'vecPos' },
    run: (x) => { const n = x.length, L = l2(x, n), out = new Float64Array(n); for (let i = 0; i < n; i++) out[i] = x[i] / L; return out; },
    predicate: (x) => { const r = runOf('F030')(x); return rc(l2(r, r.length), 1, 1e-9); },
  });
  reg(31, {
    name: '单位化 · 与缩放无关', intuition: '同向（正）缩放不改变方向，单位化结果不变（c>0）。',
    math: 'unit(c·x) = unit(x) (c>0)', pseudocode: 'return unit(scale(x,c))',
    complexity: 'O(n)', ref: '齐次', commonPitfall: 'c 须为正，否则方向翻转（unit(−x)=−unit(x)）；域用 smallPos。',
    invariant: '逐元素 relClose(run, unit(x), 1e-9)',
    inputs: ['c', 'x'], domains: { c: 'smallPos', x: 'vecPos' },
    run: (c, x) => { const n = x.length, t = scale(x, c), L = l2(t, n), out = new Float64Array(n); for (let i = 0; i < n; i++) out[i] = t[i] / L; return out; },
    predicate: (c, x) => { const r = runOf('F031')(c, x); const n = x.length; const L = l2(x, n); const u2 = new Float64Array(n); for (let i = 0; i < n; i++) u2[i] = x[i] / L; return rc(r, u2, 1e-9); },
  });
  reg(32, {
    name: '单位化 · 保持方向', intuition: '单位向量与原向量的内积，恰等于原向量长度。',
    math: '⟨unit(x), x⟩ = ‖x‖', pseudocode: 'return dot(unit(x), x)',
    complexity: 'O(n)', ref: '投影', commonPitfall: '等价于「单位向量在 x 上投影为 1」。',
    invariant: 'relClose(run, ‖x‖, 1e-9)',
    inputs: ['x'], domains: { x: 'vecPos' },
    run: (x) => { const n = x.length, L = l2(x, n), u = new Float64Array(n); for (let i = 0; i < n; i++) u[i] = x[i] / L; return dot(u, x, n); },
    predicate: (x) => { const r = runOf('F032')(x); return rc(r, l2(x, x.length), 1e-9); },
  });
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

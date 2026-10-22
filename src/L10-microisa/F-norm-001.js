/* ============================================================================
 * L10 微指令 · F 域（归一化）· 第 2/3 批（F033–F064）
 * ----------------------------------------------------------------------------
 * 所属层：L10-microIsa  域：F（归一化）
 * 允许依赖的层：L10（registry/invariant）
 * 主要 @ref：Ba et al. 2016（LayerNorm）；Zhang & Sennrich 2019（RMSNorm）
 *
 * 子域划分：
 *   F033–F048 LayerNorm（无仿射定义 / 零均值 / 方差 / 平移不变 / 仿射 / 矩阵按行）
 *   F049–F064 RMSNorm（LLaMa 风格 x/√mean(x²)⊙γ：定义 / rms 部分 / 矩阵 / 与 LN 关系）
 *
 * 谓词纪律：门禁只调 op.predicate(...)，从不调 run（tests/lib/op-harness.js:43）。
 * 故每条 predicate 必须先调用 NS.L10.ops['Fxxx'].run(...) 取真实产出，再走与 run
 * 不同的参考路径（反向累加 / Σx²/n−μ² / 解析期望）比对，杜绝「内联复算自证」。
 * 纯性质断言（零均值、方差、有限性、符号保持、常数退化）一律补独立参考基线，
 * 避免被全零等退化输出蒙混（原 F055 谓词左右两个表达式字面相同，恒真 —— 已修）。
 * 数学定理只在非退化域成立时（RN≠LN 的零均值退化、平移/缩放差异低于分辨率），
 * 谓词按解析差异加守卫并同步改正文档声称——不放宽容差凑绿。eps = 1e-5。
 * 输入域由 L10.TAGS 约束（矩阵 4×4、向量长度 8）。
 * ==========================================================================*/
;(function (NS) {
  'use strict';
  NS.L10 = NS.L10 || {};
  const rc = NS.L10.relClose;
  function reg(i, o) { NS.L10.register('F', i, o); }
  function runOf(key) { return NS.L10.ops[key].run; }
  const EPS = 1e-5;

  /* ---------- 内部工具 ---------- */
  const sq = (M) => Math.round(Math.sqrt(M.length));
  function mean(a, n) { let s = 0; for (let i = 0; i < n; i++) s += a[i]; return s / n; }
  function meanRev(a, n) { let s = 0; for (let i = n - 1; i >= 0; i--) s += a[i]; return s / n; }
  function sumSq(a, n) { let s = 0; for (let i = 0; i < n; i++) s += a[i] * a[i]; return s; }
  function sumSqRev(a, n) { let s = 0; for (let i = n - 1; i >= 0; i--) s += a[i] * a[i]; return s; }
  function varPop(a, n) { const m = mean(a, n); let s = 0; for (let i = 0; i < n; i++) { const d = a[i] - m; s += d * d; } return s / n; }
  function varPopRef(a, n) { const m = meanRev(a, n); const ss = sumSqRev(a, n); return Math.max(0, ss / n - m * m); }
  // 位移 raw-moment 方差：Var = Σ(x−s)²/n − (Σ(x−s)/n)²，位移 s 取反向首元素。
  // 与 run 的两遍法既不同公式也不同累加顺序；且 s 自适应贴近数据中心，避开
  // 「大均值 + 极小方差」下 Σx²−μ² 的相消放大（供 z-score 类除法敏感算子使用）。
  function varPopShift(a, n) {
    const s = a[n - 1];
    let t = 0, q = 0;
    for (let i = n - 1; i >= 0; i--) { const d = a[i] - s; t += d; q += d * d; }
    const m = t / n;
    return Math.max(0, q / n - m * m);
  }
  function l2(a, n) { return Math.sqrt(sumSq(a, n)); }
  function l2Rev(a, n) { return Math.sqrt(sumSqRev(a, n)); }

  // LayerNorm 无仿射（中心化 + 缩放）
  function lnRow(x, n, eps) { const m = mean(x, n), v = varPop(x, n), inv = 1 / Math.sqrt(v + eps); const o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = (x[i] - m) * inv; return o; }
  function lnRowRef(x, n, eps) { const m = meanRev(x, n), v = varPopRef(x, n), inv = 1 / Math.sqrt(v + eps); const o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = (x[i] - m) * inv; return o; }
  // LayerNorm 仿射
  function lnAffine(x, n, g, b, eps) { const z = lnRow(x, n, eps); const o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = (b ? b[i] : 0) + (g ? g[i] : 1) * z[i]; return o; }
  function lnAffineRef(x, n, g, b, eps) { const z = lnRowRef(x, n, eps); const o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = (b ? b[i] : 0) + (g ? g[i] : 1) * z[i]; return o; }
  // RMSNorm 无仿射（LLaMa：x/√mean(x²)）
  function rnRow(x, n, eps) { const inv = 1 / Math.sqrt(sumSq(x, n) / n + eps); const o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = x[i] * inv; return o; }
  function rnRowRef(x, n, eps) { const inv = 1 / Math.sqrt(sumSqRev(x, n) / n + eps); const o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = x[i] * inv; return o; }
  // 矩阵逐行参考（反向累加路径）
  function lnRowsRef(M, n, d) { const o = new Float64Array(n * d); for (let i = 0; i < n; i++) o.set(lnRowRef(M.subarray(i * d, (i + 1) * d), d, EPS), i * d); return o; }
  function rnRowsRef(M, n, d) { const o = new Float64Array(n * d); for (let i = 0; i < n; i++) o.set(rnRowRef(M.subarray(i * d, (i + 1) * d), d, EPS), i * d); return o; }

  // ══ LayerNorm（F033–F048）══════════════════════════════════════════
  reg(33, {
    name: 'LayerNorm · 定义（无仿射）', intuition: '逐样本归一化：减均值除标准差，使每样本零均值、近似单位方差。',
    math: 'zᵢ = (xᵢ − μ)/√(Var+ε)', pseudocode: 'm=mean(x); v=var(x); return (x-m)/sqrt(v+eps)',
    complexity: 'O(n)', ref: 'Ba et al. 2016 (LayerNorm)', commonPitfall: 'ε 防零方差除零；数值法用两遍求方差。',
    invariant: '逐元素 relClose(run, 反向累加+Σx²/n−μ² 参考, 1e-9)',
    inputs: ['x'], domains: { x: 'vecSmall' },
    run: (x) => lnRow(x, x.length, EPS),
    predicate: (x) => {
      const r = runOf('F033')(x);
      if (!r || r.length !== x.length) return false;
      return rc(r, lnRowRef(x, x.length, EPS), 1e-9);
    },
  });
  reg(34, {
    name: 'LayerNorm · 零均值性', intuition: '归一化后输出均值恰为 0（这是标准化的定义性后果）。',
    math: 'mean(LN(x)) = 0', pseudocode: 'return mean(LN(x))',
    complexity: 'O(n)', ref: '标准化', commonPitfall: '浮点下中心化误差 ~1e-13，容差 1e-9 足够。',
    invariant: 'relClose(mean(run), 0, 1e-9) 且 run 与反向累加参考一致（防全零蒙混）',
    inputs: ['x'], domains: { x: 'vecSmall' },
    run: (x) => lnRow(x, x.length, EPS),
    predicate: (x) => {
      const r = runOf('F034')(x), n = x.length;
      if (!r || r.length !== n) return false;
      if (!rc(r, lnRowRef(x, n, EPS), 1e-9)) return false;
      return rc(meanRev(r, n), 0, 1e-9);
    },
  });
  reg(35, {
    name: 'LayerNorm · 方差（解析期望）', intuition: '归一化后方差恰为 Var/(Var+ε)，而非恒为 1——ε 会略微压低。',
    math: 'Var(out) = Var/(Var+ε)', pseudocode: 'return var(out)',
    complexity: 'O(n)', ref: '标准化', commonPitfall: '误以为输出方差恒为 1；ε 存在时偏离，须比对解析值。',
    invariant: 'relClose(Var(run), Var/(Var+ε), 1e-9)（Var 取自 Σx²/n−μ² 参考路径）且 run 与参考一致',
    inputs: ['x'], domains: { x: 'vecSmall' },
    run: (x) => lnRow(x, x.length, EPS),
    predicate: (x) => {
      const r = runOf('F035')(x), n = x.length;
      if (!r || r.length !== n) return false;
      if (!rc(r, lnRowRef(x, n, EPS), 1e-9)) return false;
      const v = varPopRef(x, n);
      return rc(varPopRef(r, n), v / (v + EPS), 1e-9);
    },
  });
  reg(36, {
    name: 'LayerNorm · 平移不变', intuition: '整体加常数不改变离散结构，归一化结果不变。',
    math: 'LN(x+c) = LN(x)', pseudocode: 'return LN(shift(x,c))',
    complexity: 'O(n)', ref: '平移不变性', commonPitfall: 'ε 与 c 无关，故严格相等。',
    invariant: '逐元素 relClose(run(x+c), 反向累加参考 LN(x), 1e-9)',
    inputs: ['x', 'c'], domains: { x: 'vecSmall', c: 'small' },
    run: (x, c) => { const t = new Float64Array(x.length); for (let i = 0; i < x.length; i++) t[i] = x[i] + c; return lnRow(t, x.length, EPS); },
    predicate: (x, c) => {
      const r = runOf('F036')(x, c);
      if (!r || r.length !== x.length) return false;
      return rc(r, lnRowRef(x, x.length, EPS), 1e-9);
    },
  });
  reg(37, {
    name: 'LayerNorm · 仿射定义', intuition: '标准 LN 带可学习缩放 γ 与偏移 β：y = γ⊙z + β。',
    math: 'yᵢ = γᵢ·zᵢ + βᵢ', pseudocode: 'z=LN(x); return gamma*z + beta',
    complexity: 'O(n)', ref: 'Ba et al. 2016', commonPitfall: 'γ/β 须与 x 等长。',
    invariant: '逐元素 relClose(run, 反向累加参考仿射, 1e-9)',
    inputs: ['x', 'g', 'b'], domains: { x: 'vecSmall', g: 'vecSmall', b: 'vecSmall' },
    run: (x, g, b) => lnAffine(x, x.length, g, b, EPS),
    predicate: (x, g, b) => {
      const r = runOf('F037')(x, g, b);
      if (!r || r.length !== x.length) return false;
      return rc(r, lnAffineRef(x, x.length, g, b, EPS), 1e-9);
    },
  });
  reg(38, {
    name: 'LayerNorm · 仿射 β 偏移', intuition: '偏移 β 只把输出整体平移 β，与归一化部分无关（γ=1）。',
    math: 'y(x,β₂) − y(x,β₁) = β₂ − β₁', pseudocode: 'return LNaffine(b2) - LNaffine(b1)',
    complexity: 'O(n)', ref: '仿射分解', commonPitfall: '无',
    invariant: '逐元素 relClose(run, Δβ, 1e-9)',
    inputs: ['x', 'b1', 'b2'], domains: { x: 'vecSmall', b1: 'vecSmall', b2: 'vecSmall' },
    run: (x, b1, b2) => { const a = lnAffine(x, x.length, null, b1, EPS), c = lnAffine(x, x.length, null, b2, EPS), o = new Float64Array(x.length); for (let i = 0; i < x.length; i++) o[i] = c[i] - a[i]; return o; },
    predicate: (x, b1, b2) => {
      const r = runOf('F038')(x, b1, b2);
      if (!r || r.length !== x.length) return false;
      for (let i = 0; i < r.length; i++) if (!rc(r[i], b2[i] - b1[i], 1e-9)) return false;
      return true;
    },
  });
  reg(39, {
    name: 'LayerNorm · 仿射 γ 作用', intuition: 'γ=1,β=0 的输出与 γ=0,β=0 的输出之差，恰为无仿射归一化 z。',
    math: 'y(x,γ=1,β=0) − y(x,γ=0,β=0) = z = LN(x)', pseudocode: 'return LNaffine(g=1) - LNaffine(g=0)',
    complexity: 'O(n)', ref: '仿射分解', commonPitfall: '验证了 γ 正是作用在 z 上。',
    invariant: '逐元素 relClose(run, 反向累加参考 z, 1e-9)',
    inputs: ['x'], domains: { x: 'vecSmall' },
    run: (x) => { const ones = new Float64Array(x.length).fill(1), zeros = new Float64Array(x.length); const a = lnAffine(x, x.length, ones, zeros, EPS), b = lnAffine(x, x.length, zeros, zeros, EPS), o = new Float64Array(x.length); for (let i = 0; i < x.length; i++) o[i] = a[i] - b[i]; return o; },
    predicate: (x) => {
      const r = runOf('F039')(x);
      if (!r || r.length !== x.length) return false;
      return rc(r, lnRowRef(x, x.length, EPS), 1e-9);
    },
  });
  reg(40, {
    name: 'LayerNorm · 矩阵按行', intuition: '批量样本逐行独立归一化（NLP 中每行是一个 token 的隐藏态）。',
    math: 'out[i] = LN(Mᵢ,:)', pseudocode: 'for i: out[i]=LN(row_i)',
    complexity: 'O(N·D)', ref: 'LayerNorm', commonPitfall: '行间独立，互不污染。',
    invariant: '逐元素 relClose(run, 反向累加参考, 1e-9)',
    inputs: ['M'], domains: { M: 'matSmall' }, matrixDims: { M: [4, 4] },
    run: (M) => { const n = sq(M), d = M.length / n, out = new Float64Array(n * d); for (let i = 0; i < n; i++) out.set(lnRow(M.subarray(i * d, (i + 1) * d), d, EPS), i * d); return out; },
    predicate: (M) => {
      const r = runOf('F040')(M), n = sq(M), d = M.length / n;
      if (!r || r.length !== n * d) return false;
      return rc(r, lnRowsRef(M, n, d), 1e-9);
    },
  });
  reg(41, {
    name: 'LayerNorm · 矩阵按行零均值', intuition: '矩阵逐行归一化后，每行输出均值均≈0。',
    math: '∀i: mean(out[i]) = 0', pseudocode: 'for i: assert mean(out_row_i)≈0',
    complexity: 'O(N·D)', ref: '标准化', commonPitfall: '无',
    invariant: '逐行 relClose(mean(run 行), 0, 1e-9) 且 run 与反向累加参考一致（防全零蒙混）',
    inputs: ['M'], domains: { M: 'matSmall' }, matrixDims: { M: [4, 4] },
    run: (M) => { const n = sq(M), d = M.length / n, out = new Float64Array(n * d); for (let i = 0; i < n; i++) out.set(lnRow(M.subarray(i * d, (i + 1) * d), d, EPS), i * d); return out; },
    predicate: (M) => {
      const r = runOf('F041')(M), n = sq(M), d = M.length / n;
      if (!r || r.length !== n * d) return false;
      if (!rc(r, lnRowsRef(M, n, d), 1e-9)) return false;
      for (let i = 0; i < n; i++) if (!rc(meanRev(r.subarray(i * d, (i + 1) * d), d), 0, 1e-9)) return false;
      return true;
    },
  });
  reg(42, {
    name: 'LayerNorm · 矩阵按行方差', intuition: '每行输出方差恰为 Var_row/(Var_row+ε)。',
    math: '∀i: Var(out[i]) = Var_row/(Var_row+ε)', pseudocode: 'for i: assert var(out_row)≈Var_row/(Var_row+ε)',
    complexity: 'O(N·D)', ref: '标准化', commonPitfall: '同 F035，须比对解析值而非 1。',
    invariant: '逐行 relClose(Var(run 行), Var_row/(Var_row+ε), 1e-9) 且 run 与反向累加参考一致',
    inputs: ['M'], domains: { M: 'matSmall' }, matrixDims: { M: [4, 4] },
    run: (M) => { const n = sq(M), d = M.length / n, out = new Float64Array(n * d); for (let i = 0; i < n; i++) out.set(lnRow(M.subarray(i * d, (i + 1) * d), d, EPS), i * d); return out; },
    predicate: (M) => {
      const r = runOf('F042')(M), n = sq(M), d = M.length / n;
      if (!r || r.length !== n * d) return false;
      if (!rc(r, lnRowsRef(M, n, d), 1e-9)) return false;
      for (let i = 0; i < n; i++) {
        const v = varPopRef(M.subarray(i * d, (i + 1) * d), d);
        if (!rc(varPopRef(r.subarray(i * d, (i + 1) * d), d), v / (v + EPS), 1e-9)) return false;
      }
      return true;
    },
  });
  reg(43, {
    name: 'LayerNorm · 逐行独立', intuition: '行间归一化互不污染——改动一行不影响其他行输出。',
    math: '改动第 r 行 ⇒ 其余行输出不变', pseudocode: 'M2[row_r]=zeros; assert rows≠r identical',
    complexity: 'O(N·D)', ref: '逐样本归一化', commonPitfall: 'BatchNorm 不具备此性质（统计量跨样本）。',
    invariant: 'run 与反向累加参考一致，且逐位 Object.is(其余行输出) 在改写首行后不变',
    inputs: ['M'], domains: { M: 'matSmall' }, matrixDims: { M: [4, 4] },
    run: (M) => { const n = sq(M), d = M.length / n, out = new Float64Array(n * d); for (let i = 0; i < n; i++) out.set(lnRow(M.subarray(i * d, (i + 1) * d), d, EPS), i * d); return out; },
    predicate: (M) => {
      const n = sq(M), d = M.length / n;
      const a = runOf('F043')(M);
      if (!a || a.length !== n * d) return false;
      if (!rc(a, lnRowsRef(M, n, d), 1e-9)) return false;
      if (n < 2) return true;                                   // 单行无「其他行」可言，仅验正确性
      const M2 = Float64Array.from(M);
      for (let j = 0; j < d; j++) M2[j] = 0;                    // 改写首行
      const b = runOf('F043')(M2);
      if (!b || b.length !== n * d) return false;
      for (let k = d; k < n * d; k++) if (!Object.is(a[k], b[k])) return false;
      return true;
    },
  });
  reg(44, {
    name: 'LayerNorm · 输出有限', intuition: '有限输入（含常数向量）经 ε 保护后输出必有限，绝不 NaN/Inf。',
    math: '∀i: isFinite(outᵢ)', pseudocode: 'for i: assert isFinite(out_i)',
    complexity: 'O(n)', ref: '数值稳定性', commonPitfall: '常数向量 Var+ε=ε>0，除零被规避。',
    invariant: 'every(isFinite(run)) 且 run 与反向累加参考一致（仅查有限性会被全零蒙混）',
    inputs: ['x'], domains: { x: 'vecSmall' },
    run: (x) => lnRow(x, x.length, EPS),
    predicate: (x) => {
      const r = runOf('F044')(x);
      if (!r || r.length !== x.length) return false;
      for (let i = 0; i < r.length; i++) if (!isFinite(r[i])) return false;
      return rc(r, lnRowRef(x, x.length, EPS), 1e-9);
    },
  });
  reg(45, {
    name: 'LayerNorm · 退化（常数向量 → 0）', intuition: '无离散度时中心化全 0，无仿射输出恰为 0。',
    math: 'Var(c,c,…,c)=0 ⇒ LN = 0', pseudocode: 'return LN(constVec)',
    complexity: 'O(n)', ref: '退化情形', commonPitfall: '无',
    invariant: '逐元素 relClose(run, 0, 1e-12)',
    inputs: ['x'], domains: { x: 'vecSmall' },
    run: (x) => { const c = x[0], n = x.length, v = new Float64Array(n); for (let i = 0; i < n; i++) v[i] = c; return lnRow(v, n, EPS); },
    predicate: (x) => {
      const r = runOf('F045')(x), n = x.length;
      if (!r || r.length !== n) return false;
      for (let i = 0; i < n; i++) if (!rc(r[i], 0, 1e-12)) return false;
      return true;
    },
  });
  reg(46, {
    name: 'LayerNorm · γ=1,β=0 退化为无仿射', intuition: '仿射取恒等（γ=1,β=0）时，结果必须严格等于无仿射 LN。',
    math: 'LN(x,γ=1,β=0) = LN(x)', pseudocode: 'return LNaffine(g=1,b=0)',
    complexity: 'O(n)', ref: '仿射退化', commonPitfall: '无',
    invariant: '逐元素 relClose(run, 反向累加参考 LN, 1e-9)',
    inputs: ['x'], domains: { x: 'vecSmall' },
    run: (x) => lnAffine(x, x.length, null, null, EPS),
    predicate: (x) => {
      const r = runOf('F046')(x);
      if (!r || r.length !== x.length) return false;
      return rc(r, lnRowRef(x, x.length, EPS), 1e-9);
    },
  });
  reg(47, {
    name: 'LayerNorm · 仿射后符号保持', intuition: '无仿射中心化项 zᵢ 的符号与原偏离 (xᵢ−μ) 同（γ>0,β=0 时）。',
    math: 'sign(zᵢ) = sign(xᵢ − μ)', pseudocode: 'for i: assert sign(z_i)==sign(x_i-mu)',
    complexity: 'O(n)', ref: '中心化', commonPitfall: 'xᵢ≈μ 时符号不定，容差跳过（|xᵢ−μ|<1e-9）。',
    invariant: 'sign(runᵢ) === sign(xᵢ−μ)（|xᵢ−μ|≥1e-9）且 run 与反向累加参考一致',
    inputs: ['x'], domains: { x: 'vecSmall' },
    run: (x) => lnRow(x, x.length, EPS),
    predicate: (x) => {
      const r = runOf('F047')(x), n = x.length;
      if (!r || r.length !== n) return false;
      if (!rc(r, lnRowRef(x, n, EPS), 1e-9)) return false;
      const m = meanRev(x, n);                                  // 反向累加独立路径
      for (let i = 0; i < n; i++) {
        const d = x[i] - m;
        if (Math.abs(d) < 1e-9) continue;
        if (Math.sign(r[i]) !== Math.sign(d)) return false;
      }
      return true;
    },
  });
  reg(48, {
    name: 'LayerNorm · 零均值单位方差输入 → 缩放常数', intuition: '输入已零均值单位方差时，LN 输出 = x/√(1+ε)（既约情形）。',
    math: 'μ=0,Var=1 ⇒ LN(x) = x/√(1+ε)', pseudocode: 'z = (x-mean(x))/std(x); return LN(z)',
    complexity: 'O(n)', ref: '既约情形', commonPitfall: '先标准化到零均值单位方差，再验证 LN 仅为常数缩放。',
    invariant: '逐元素 relClose(run, 反向累加 z 的 LN 参考, 1e-9) 且显著分量满足 run = z/√(1+ε)',
    inputs: ['x'], domains: { x: 'vecSmall' },
    run: (x) => { const n = x.length, m = mean(x, n), sd = Math.sqrt(varPop(x, n)) || 1, z = new Float64Array(n); for (let i = 0; i < n; i++) z[i] = (x[i] - m) / sd; return lnRow(z, n, EPS); },
    predicate: (x) => {
      const r = runOf('F048')(x), n = x.length;
      if (!r || r.length !== n) return false;
      const m = meanRev(x, n), sd = Math.sqrt(varPopShift(x, n));  // 位移 raw-moment 独立路径
      const z = new Float64Array(n);
      for (let i = 0; i < n; i++) z[i] = sd === 0 ? 0 : (x[i] - m) / sd;
      if (!rc(r, lnRowRef(z, n, EPS), 1e-9)) return false;
      const k = 1 / Math.sqrt(1 + EPS);                          // 既约：LN(z) = z/√(1+ε)
      for (let i = 0; i < n; i++) if (Math.abs(z[i]) > 0.1 && !rc(r[i], z[i] * k, 1e-9)) return false;
      return true;
    },
  });

  // ══ RMSNorm（F049–F064）════════════════════════════════════════════
  reg(49, {
    name: 'RMSNorm · 定义（γ=1,β=0 简化）', intuition: 'LLaMa 用 RMSNorm：不减去均值，仅按均方根缩放，更省算力。',
    math: 'yᵢ = xᵢ/√(mean(x²)+ε)', pseudocode: 'inv=1/sqrt(mean(x^2)+eps); return x*inv',
    complexity: 'O(n)', ref: 'Zhang & Sennrich 2019', commonPitfall: '与 LN 不同：不减均值；对常数向量输出为 常数/√(c²+ε)。',
    invariant: '逐元素 relClose(run, 反向累加参考, 1e-9)',
    inputs: ['x'], domains: { x: 'vecSmall' },
    run: (x) => rnRow(x, x.length, EPS),
    predicate: (x) => {
      const r = runOf('F049')(x);
      if (!r || r.length !== x.length) return false;
      return rc(r, rnRowRef(x, x.length, EPS), 1e-9);
    },
  });
  reg(50, {
    name: 'RMSNorm · RMS 部分定义', intuition: '均方根 rms = √(mean(x²))，是分母的核心。',
    math: 'rms = √(Σxᵢ²/n)', pseudocode: 'return sqrt(sumSq(x)/n)',
    complexity: 'O(n)', ref: 'RMSNorm', commonPitfall: '与标准差不同（不减均值）。',
    invariant: 'relClose(run, √(Σx²/n) 反向累加, 1e-9)',
    inputs: ['x'], domains: { x: 'vecSmall' },
    run: (x) => Math.sqrt(sumSq(x, x.length) / x.length),
    predicate: (x) => {
      const r = runOf('F050')(x), n = x.length;
      if (!isFinite(r)) return false;
      return rc(r, Math.sqrt(sumSqRev(x, n) / n), 1e-9);
    },
  });
  reg(51, {
    name: 'RMSNorm · 输出 RMS（解析期望）', intuition: 'RMSNorm 输出自身的 rms 恰为 rms_x/√(rms_x²+ε)，而非 1。',
    math: 'rms(out) = rms_x/√(rms_x²+ε)', pseudocode: 'return rms(out)',
    complexity: 'O(n)', ref: 'RMSNorm', commonPitfall: '误以为输出 rms 恒为 1（那是 L2 单位化）。',
    invariant: 'relClose(rms(run), rms_x/√(rms_x²+ε), 1e-9)（rms_x 取自反向累加路径）',
    inputs: ['x'], domains: { x: 'vecSmall' },
    run: (x) => rnRow(x, x.length, EPS),
    predicate: (x) => {
      const r = runOf('F051')(x), n = x.length;
      if (!r || r.length !== n) return false;
      const rx = Math.sqrt(sumSqRev(x, n) / n);
      return rc(Math.sqrt(sumSqRev(r, n) / n), rx / Math.sqrt(rx * rx + EPS), 1e-9);
    },
  });
  reg(52, {
    name: 'RMSNorm · 矩阵按行', intuition: '批量样本逐行做 RMSNorm（与 LN 矩阵版同构，仅分母不同）。',
    math: 'out[i] = RMSNorm(Mᵢ,:)', pseudocode: 'for i: out[i]=RMSNorm(row_i)',
    complexity: 'O(N·D)', ref: 'RMSNorm', commonPitfall: '行间独立。',
    invariant: '逐元素 relClose(run, 反向累加参考, 1e-9)',
    inputs: ['M'], domains: { M: 'matSmall' }, matrixDims: { M: [4, 4] },
    run: (M) => { const n = sq(M), d = M.length / n, out = new Float64Array(n * d); for (let i = 0; i < n; i++) out.set(rnRow(M.subarray(i * d, (i + 1) * d), d, EPS), i * d); return out; },
    predicate: (M) => {
      const r = runOf('F052')(M), n = sq(M), d = M.length / n;
      if (!r || r.length !== n * d) return false;
      return rc(r, rnRowsRef(M, n, d), 1e-9);
    },
  });
  reg(53, {
    name: 'RMSNorm · 与 LayerNorm 关系（零均值输入相等）', intuition: '当输入零均值时，Var = mean(x²)，故 RMSNorm 退化为 LayerNorm。',
    math: 'μ=0 ⇒ RMSNorm(x) = LN(x)', pseudocode: 'z=x-mean(x); return RN(z)',
    complexity: 'O(n)', ref: 'RMSNorm vs LN', commonPitfall: '仅在零均值时成立；一般输入下二者不同。',
    invariant: 'run 与反向累加 RN(z) 参考一致，且与反向累加 LN(z) 参考一致⇒ 零均值时 RN≡LN',
    inputs: ['x'], domains: { x: 'vecSmall' },
    run: (x) => { const n = x.length, m = mean(x, n), z = new Float64Array(n); for (let i = 0; i < n; i++) z[i] = x[i] - m; return rnRow(z, n, EPS); },
    predicate: (x) => {
      const r = runOf('F053')(x), n = x.length;
      if (!r || r.length !== n) return false;
      const m = meanRev(x, n), z = new Float64Array(n);           // 反向累加独立路径
      for (let i = 0; i < n; i++) z[i] = x[i] - m;
      if (!rc(r, rnRowRef(z, n, EPS), 1e-9)) return false;        // run 确为 RN(z)
      return rc(r, lnRowRef(z, n, EPS), 1e-9);                    // 零均值 ⇒ RN(z)=LN(z)
    },
  });
  reg(54, {
    name: 'RMSNorm · 仿射 γ 缩放', intuition: 'γ 逐元素缩放输出（β=0）：y = γ⊙x/√rms²。',
    math: 'yᵢ = γᵢ·xᵢ/√(rms²+ε)', pseudocode: 'return gamma * RN(x)',
    complexity: 'O(n)', ref: 'RMSNorm', commonPitfall: 'γ 须与 x 等长。',
    invariant: '逐元素 relClose(run, γᵢ·反向累加参考 RN(x)ᵢ, 1e-9)',
    inputs: ['x', 'g'], domains: { x: 'vecSmall', g: 'vecSmall' },
    run: (x, g) => { const n = x.length, z = rnRow(x, n, EPS), o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = g[i] * z[i]; return o; },
    predicate: (x, g) => {
      const r = runOf('F054')(x, g), n = x.length;
      if (!r || r.length !== n) return false;
      const z = rnRowRef(x, n, EPS), ref = new Float64Array(n);   // 反向累加独立路径
      for (let i = 0; i < n; i++) ref[i] = g[i] * z[i];
      return rc(r, ref, 1e-9);
    },
  });
  reg(55, {
    name: 'RMSNorm · 仿射 β 偏移', intuition: '偏移 β 仅平移输出，与 RMSNorm 部分无关（γ=1）。',
    math: 'y(x,β₂) − y(x,β₁) = β₂ − β₁', pseudocode: 'return RNaffine(b2) - RNaffine(b1)',
    complexity: 'O(n)', ref: '仿射分解', commonPitfall: 'run 原返回 β₂−β₁+z（与文档语义不符，已修正为两路仿射输出之差）。',
    invariant: '逐元素 relClose(run, Δβ, 1e-9)',
    inputs: ['x', 'b1', 'b2'], domains: { x: 'vecSmall', b1: 'vecSmall', b2: 'vecSmall' },
    run: (x, b1, b2) => { const n = x.length, z = rnRow(x, n, EPS), o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = (b2[i] + z[i]) - (b1[i] + z[i]); return o; },
    predicate: (x, b1, b2) => {
      const r = runOf('F055')(x, b1, b2), n = x.length;
      if (!r || r.length !== n) return false;
      for (let i = 0; i < n; i++) if (!rc(r[i], b2[i] - b1[i], 1e-9)) return false;
      return true;
    },
  });
  reg(56, {
    name: 'RMSNorm · 符号保持（γ>0,β=0）', intuition: 'RMSNorm 不减均值只缩放，故输出符号与原分量一致。',
    math: 'sign(yᵢ) = sign(xᵢ)', pseudocode: 'for i: assert sign(y_i)==sign(x_i)',
    complexity: 'O(n)', ref: 'RMSNorm', commonPitfall: 'xᵢ≈0 时跳过（|xᵢ|<1e-9）。',
    invariant: 'sign(runᵢ) === sign(xᵢ)（|xᵢ|≥1e-9）且 run 与反向累加参考一致',
    inputs: ['x'], domains: { x: 'vecSmall' },
    run: (x) => rnRow(x, x.length, EPS),
    predicate: (x) => {
      const r = runOf('F056')(x), n = x.length;
      if (!r || r.length !== n) return false;
      if (!rc(r, rnRowRef(x, n, EPS), 1e-9)) return false;
      for (let i = 0; i < n; i++) {
        if (Math.abs(x[i]) < 1e-9) continue;
        if (Math.sign(r[i]) !== Math.sign(x[i])) return false;
      }
      return true;
    },
  });
  reg(57, {
    name: 'RMSNorm · 输出有限', intuition: '有限输入经 ε 保护后输出必有限。',
    math: '∀i: isFinite(yᵢ)', pseudocode: 'for i: assert isFinite(y_i)',
    complexity: 'O(n)', ref: '数值稳定性', commonPitfall: '无',
    invariant: 'every(isFinite(run)) 且 run 与反向累加参考一致（仅查有限性会被全零蒙混）',
    inputs: ['x'], domains: { x: 'vecSmall' },
    run: (x) => rnRow(x, x.length, EPS),
    predicate: (x) => {
      const r = runOf('F057')(x);
      if (!r || r.length !== x.length) return false;
      for (let i = 0; i < r.length; i++) if (!isFinite(r[i])) return false;
      return rc(r, rnRowRef(x, x.length, EPS), 1e-9);
    },
  });
  reg(58, {
    name: 'RMSNorm · 退化（零向量 → 0）', intuition: '零向量 rms=0，ε 保护下输出为 0，不 NaN。',
    math: 'x=0 ⇒ y=0/(√ε)=0', pseudocode: 'return RN(zeros)',
    complexity: 'O(1)', ref: '退化情形', commonPitfall: '无',
    invariant: '逐元素 relClose(run, 0, 1e-12)',
    inputs: ['x'], domains: { x: 'vecSmall' },
    run: (x) => { const o = new Float64Array(x.length); return rnRow(o, x.length, EPS); },
    predicate: (x) => {
      const r = runOf('F058')(x), n = x.length;
      if (!r || r.length !== n) return false;
      for (let i = 0; i < n; i++) if (!rc(r[i], 0, 1e-12)) return false;
      return true;
    },
  });
  reg(59, {
    name: 'RMSNorm · 缩放关系（解析期望）', intuition: '缩放 a 倍后，输出与输出的逐元素比恰为 a·√(rms²+ε)/√(a²rms²+ε)。',
    math: 'RN(a·x)ᵢ = RN(x)ᵢ·a√(rms²+ε)/√(a²rms²+ε)', pseudocode: 'return RN(scale(x,a))',
    complexity: 'O(n)', ref: 'RMSNorm', commonPitfall: '因分母含 rms² 而 a² 缩放，整体非齐次——须比对解析比。',
    invariant: '逐元素 relClose(run(a·x), 反向累加参考 RN(x)ᵢ·k, 1e-9)，k = a√(rms²+ε)/√(a²rms²+ε)',
    inputs: ['a', 'x'], domains: { a: 'small', x: 'vecSmall' },
    run: (a, x) => { const n = x.length, t = new Float64Array(n); for (let i = 0; i < n; i++) t[i] = a * x[i]; return rnRow(t, n, EPS); },
    predicate: (a, x) => {
      const r = runOf('F059')(a, x), n = x.length;
      if (!r || r.length !== n) return false;
      const rx = Math.sqrt(sumSqRev(x, n) / n);                    // 反向累加独立路径
      const k = (a * Math.sqrt(rx * rx + EPS)) / Math.sqrt(a * a * rx * rx + EPS);
      const base = rnRowRef(x, n, EPS), ref = new Float64Array(n);
      for (let i = 0; i < n; i++) ref[i] = base[i] * k;
      return rc(r, ref, 1e-9);
    },
  });
  reg(60, {
    name: 'RMSNorm · 范数保持（γ=1）', intuition: 'γ=1 时输出范数 ‖y‖ = √n·rms_x/√(rms_x²+ε)。',
    math: '‖y‖ = √n·rms_x/√(rms_x²+ε)', pseudocode: 'return l2(out)',
    complexity: 'O(n)', ref: 'RMSNorm', commonPitfall: '无',
    invariant: 'relClose(‖run‖（反向累加）, √n·rms_x/√(rms_x²+ε), 1e-9)',
    inputs: ['x'], domains: { x: 'vecSmall' },
    run: (x) => rnRow(x, x.length, EPS),
    predicate: (x) => {
      const r = runOf('F060')(x), n = x.length;
      if (!r || r.length !== n) return false;
      const rx = Math.sqrt(sumSqRev(x, n) / n);
      return rc(l2Rev(r, n), Math.sqrt(n) * rx / Math.sqrt(rx * rx + EPS), 1e-9);
    },
  });
  reg(61, {
    name: 'RMSNorm · 与 LayerNorm 差异（一般输入）', intuition: '一般（非零均值）输入下，RMSNorm 与 LayerNorm 不相等——验证二者本质区别。',
    math: 'RN(x)ᵢ − LN(x)ᵢ = xᵢ/√(ms+ε) − (xᵢ−μ)/√(Var+ε)（μ=0 时恒为 0）', pseudocode: 'return RN(x)',
    complexity: 'O(n)', ref: 'RMSNorm vs LN', commonPitfall: 'μ=0 时二者相等（F053）；μ 极小时差异低于 1e-6，须按解析差异守卫而非硬断言。',
    invariant: '逐元素 relClose(run−LN 参考, 解析差异, 1e-9)；解析差异 >1e-6 时实测差异亦须 >1e-6',
    inputs: ['x'], domains: { x: 'vecSmall' },
    run: (x) => rnRow(x, x.length, EPS),
    predicate: (x) => {
      const r = runOf('F061')(x), n = x.length;
      if (!r || r.length !== n) return false;
      const m = meanRev(x, n), ms = sumSqRev(x, n) / n;            // 反向累加独立路径
      const v = Math.max(0, ms - m * m);
      const da = Math.sqrt(ms + EPS), db = Math.sqrt(v + EPS);
      const lRef = lnRowRef(x, n, EPS);
      let maxPred = 0;
      for (let i = 0; i < n; i++) {
        const pred = x[i] / da - (x[i] - m) / db;                  // 解析差异 RNᵢ − LNᵢ
        if (!rc(r[i] - lRef[i], pred, 1e-9)) return false;
        if (Math.abs(pred) > maxPred) maxPred = Math.abs(pred);
      }
      if (maxPred <= 1e-6) return true;                            // 退化域（μ≈0）：RN ≡ LN
      let maxAct = 0;                                               // 一般域：差异须真实存在
      for (let i = 0; i < n; i++) maxAct = Math.max(maxAct, Math.abs(r[i] - lRef[i]));
      return maxAct > 1e-6;
    },
  });
  reg(62, {
    name: 'RMSNorm · γ=1,β=0 无仿射', intuition: '仿射取恒等（γ=1,β=0）时，RMSNorm 退化为纯缩放定义。',
    math: 'RN(x,γ=1,β=0) = x/√(mean(x²)+ε)', pseudocode: 'return RN(x)',
    complexity: 'O(n)', ref: '仿射退化', commonPitfall: '无',
    invariant: '逐元素 relClose(run, 反向累加参考, 1e-9)',
    inputs: ['x'], domains: { x: 'vecSmall' },
    run: (x) => rnRow(x, x.length, EPS),
    predicate: (x) => {
      const r = runOf('F062')(x);
      if (!r || r.length !== x.length) return false;
      return rc(r, rnRowRef(x, x.length, EPS), 1e-9);
    },
  });
  reg(63, {
    name: 'RMSNorm · 平移改变输出', intuition: '因不减均值，加常数会改变 rms，从而改变输出（与 LN 的平移不变形成对照）。',
    math: 'RN(x+c)ᵢ = (xᵢ+c)/√(ms+2cμ+c²+ε)', pseudocode: 'return RN(shift(x,c))',
    complexity: 'O(n)', ref: 'RMSNorm vs LN', commonPitfall: '这是 RMSNorm 与 LN 的关键区别；c 极小时差异低于 1e-9，须按分母相对变化守卫。',
    invariant: '两路解析期望一致（1e-9）；分母相对变化 ≥1e-6 时 ‖RN(x+c)−RN(x)‖ > 1e-9',
    inputs: ['x', 'c'], domains: { x: 'vecSmall', c: 'small' },
    run: (x, c) => { const t = new Float64Array(x.length); for (let i = 0; i < x.length; i++) t[i] = x[i] + c; return rnRow(t, x.length, EPS); },
    predicate: (x, c) => {
      const n = x.length;
      const r1 = runOf('F063')(x, c), r0 = runOf('F063')(x, 0);    // RN(x+c) 与 RN(x) 真实产出
      if (!r1 || r1.length !== n || !r0 || r0.length !== n) return false;
      const m = meanRev(x, n), ms = sumSqRev(x, n) / n;             // 反向累加独立路径
      const d1 = Math.sqrt(Math.max(0, ms + 2 * c * m + c * c) + EPS), d0 = Math.sqrt(ms + EPS);
      for (let i = 0; i < n; i++) {
        if (!rc(r1[i], (x[i] + c) / d1, 1e-9)) return false;        // 平移后解析期望
        if (!rc(r0[i], x[i] / d0, 1e-9)) return false;              // 原向量解析期望
      }
      if (Math.abs(2 * c * m + c * c) < 1e-6 * (ms + EPS)) return true;   // c 过小：差异低于分辨率
      let s = 0;
      for (let i = 0; i < n; i++) { const e = r1[i] - r0[i]; s += e * e; }
      return Math.sqrt(s) > 1e-9;
    },
  });
  reg(64, {
    name: 'RMSNorm · 逐元素符号（仿射 γ>0,β=0）', intuition: '仿射仅缩放不改变符号，输出符号仍同输入分量。',
    math: 'sign(yᵢ) = sign(xᵢ)', pseudocode: 'for i: assert sign(y_i)==sign(x_i)',
    complexity: 'O(n)', ref: 'RMSNorm', commonPitfall: 'xᵢ≈0 时跳过。',
    invariant: 'sign(runᵢ) === sign(xᵢ)（|xᵢ|≥1e-9）且 run 与反向累加参考一致',
    inputs: ['x'], domains: { x: 'vecSmall' },
    run: (x) => rnRow(x, x.length, EPS),
    predicate: (x) => {
      const r = runOf('F064')(x), n = x.length;
      if (!r || r.length !== n) return false;
      if (!rc(r, rnRowRef(x, n, EPS), 1e-9)) return false;
      for (let i = 0; i < n; i++) {
        if (Math.abs(x[i]) < 1e-9) continue;
        if (Math.sign(r[i]) !== Math.sign(x[i])) return false;
      }
      return true;
    },
  });
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

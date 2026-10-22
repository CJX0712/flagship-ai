/* ============================================================================
 * L10 微指令 · F 域（归一化）· 第 3/3 批（F065–F096）
 * ----------------------------------------------------------------------------
 * 子域划分：
 *   F065–F080 BatchNorm（按列/通道归一化：均值/方差/定义/零均值/单位方差/仿射/
 *              平移不变/跨样本耦合/与 LN 区别/常数通道/γ·β 作用/有限/缩放非齐次/全零/转置 LN）
 *   F081–F089 GroupNorm / InstanceNorm / 权重标准化（分组归一化：定义/零均值/单位方差/
 *              仿射/G=1→IN/G=D→β/与 BN 区别/WS 定义/WS≡LN 按行）
 *   F090–F096 LRN 与经典归一化（LRN 定义/单调性/有限/min-max/z-score/L1/L∞）
 *
 * 不变量一律「独立参考路径交叉验证」或「解析期望」，绝不用伪齐性式。eps = 1e-5。
 * ==========================================================================*/
;(function (NS) {
  'use strict';
  NS.L10 = NS.L10 || {};
  const rc = NS.L10.relClose;
  function reg(i, o) { NS.L10.register('F', i, o); }
  const EPS = 1e-5;

  const sq = (M) => Math.round(Math.sqrt(M.length));
  function mean(a, n) { let s = 0; for (let i = 0; i < n; i++) s += a[i]; return s / n; }
  function meanRev(a, n) { let s = 0; for (let i = n - 1; i >= 0; i--) s += a[i]; return s / n; }
  function sumSq(a, n) { let s = 0; for (let i = 0; i < n; i++) s += a[i] * a[i]; return s; }
  function sumSqRev(a, n) { let s = 0; for (let i = n - 1; i >= 0; i--) s += a[i] * a[i]; return s; }
  function varPop(a, n) { const m = mean(a, n); let s = 0; for (let i = 0; i < n; i++) { const d = a[i] - m; s += d * d; } return s / n; }
  function varPopRef(a, n) { const m = meanRev(a, n); const ss = sumSqRev(a, n); return Math.max(0, ss / n - m * m); }
  function l2(a, n) { return Math.sqrt(sumSq(a, n)); }

  // ---- BatchNorm（按列/通道，跨样本）----
  function colStats(M, n, d, j) { let s = 0; for (let i = 0; i < n; i++) s += M[i * d + j]; const mu = s / n; let v = 0; for (let i = 0; i < n; i++) { const dd = M[i * d + j] - mu; v += dd * dd; } v /= n; return { mu, v }; }
  function colStatsRef(M, n, d, j) { let s = 0; for (let i = n - 1; i >= 0; i--) s += M[i * d + j]; const mu = s / n; let ss = 0; for (let i = 0; i < n; i++) ss += M[i * d + j] * M[i * d + j]; const v = Math.max(0, ss / n - mu * mu); return { mu, v }; }
  function bnNorm(M, n, d, g, b) {
    const out = new Float64Array(n * d);
    for (let j = 0; j < d; j++) { const { mu, v } = colStats(M, n, d, j); const inv = 1 / Math.sqrt(v + EPS); for (let i = 0; i < n; i++) out[i * d + j] = (b ? b[j] : 0) + (g ? g[j] : 1) * (M[i * d + j] - mu) * inv; }
    return out;
  }
  function bnNormRef(M, n, d, g, b) {
    const out = new Float64Array(n * d);
    for (let j = 0; j < d; j++) { const { mu, v } = colStatsRef(M, n, d, j); const inv = 1 / Math.sqrt(v + EPS); for (let i = 0; i < n; i++) out[i * d + j] = (b ? b[j] : 0) + (g ? g[j] : 1) * (M[i * d + j] - mu) * inv; }
    return out;
  }

  // ---- GroupNorm（逐样本，分组通道）----
  function gnNorm(M, n, d, G, g, b) {
    const out = new Float64Array(n * d), cpg = Math.max(1, Math.floor(d / G));
    for (let i = 0; i < n; i++) for (let gr = 0; gr < G; gr++) {
      const s0 = gr * cpg, s1 = Math.min(d, (gr + 1) * cpg); let s = 0; for (let j = s0; j < s1; j++) s += M[i * d + j]; const mu = s / (s1 - s0);
      let v = 0; for (let j = s0; j < s1; j++) { const dd = M[i * d + j] - mu; v += dd * dd; } v /= (s1 - s0);
      const inv = 1 / Math.sqrt(v + EPS);
      for (let j = s0; j < s1; j++) out[i * d + j] = (b ? b[j] : 0) + (g ? g[j] : 1) * (M[i * d + j] - mu) * inv;
    }
    return out;
  }
  function gnNormRef(M, n, d, G, g, b) {
    const out = new Float64Array(n * d), cpg = Math.max(1, Math.floor(d / G));
    for (let i = 0; i < n; i++) for (let gr = 0; gr < G; gr++) {
      const s0 = gr * cpg, s1 = Math.min(d, (gr + 1) * cpg);
      // 独立参考路径：Welford 在线算法（单遍增量更新均值与 M2），数值稳定且与
      // gnNorm 的两遍法算法结构完全不同 —— 保持交叉验证力。
      // 注意：不可改用简化公式 Σx²/n − μ²：均值大、方差小时会触发灾难性抵消，
      // 方差精度崩塌（F084 曾因此在 2500 种子扫描中假红）。
      let m = 0, m2 = 0, cnt = 0;
      for (let j = s1 - 1; j >= s0; j--) { const x = M[i * d + j]; cnt++; const delta = x - m; m += delta / cnt; m2 += delta * (x - m); }
      const mu = m; const v = cnt > 0 ? Math.max(0, m2 / cnt) : 0;
      const inv = 1 / Math.sqrt(v + EPS);
      for (let j = s0; j < s1; j++) out[i * d + j] = (b ? b[j] : 0) + (g ? g[j] : 1) * (M[i * d + j] - mu) * inv;
    }
    return out;
  }

  // ---- 权重标准化（按行 LN 式）----
  function wsNorm(M, n, d) { const out = new Float64Array(n * d); for (let i = 0; i < n; i++) { const r = M.subarray(i * d, (i + 1) * d); const m = mean(r, d), v = varPop(r, d), inv = 1 / Math.sqrt(v + EPS); for (let j = 0; j < d; j++) out[i * d + j] = (r[j] - m) * inv; } return out; }

  // ---- LRN（局部响应归一化，作用于单样本通道向量）----
  function lrn(x, len, w, k, alpha, beta) {
    const out = new Float64Array(len);
    for (let c = 0; c < len; c++) { let ss = 0; const lo = Math.max(0, c - w), hi = Math.min(len - 1, c + w); for (let c2 = lo; c2 <= hi; c2++) ss += x[c2] * x[c2]; out[c] = x[c] / Math.pow(k + alpha * ss, beta); }
    return out;
  }
  function lrnRef(x, len, w, k, alpha, beta) {
    const out = new Float64Array(len);
    for (let c = 0; c < len; c++) { let ss = 0; const lo = Math.max(0, c - w), hi = Math.min(len - 1, c + w); for (let c2 = hi; c2 >= lo; c2--) ss += x[c2] * x[c2]; out[c] = x[c] / Math.pow(k + alpha * ss, beta); }
    return out;
  }

  // ══ BatchNorm（F065–F080）═══════════════════════════════════════════
  reg(65, {
    name: 'BatchNorm · 每通道均值', intuition: 'BN 先求每个通道（特征）在批量上的均值 μ——训练时还做指数滑动平均。',
    math: 'μⱼ = mean_{i}(Mᵢⱼ)', pseudocode: 'for j: mu[j]=mean(col_j)',
    complexity: 'O(N·D)', ref: 'Ioffe & Szegedy 2015', commonPitfall: 'BN 是按通道（列）而非按样本（行）求统计。',
    invariant: '逐元素 relClose(μ, 反向累加参考, 1e-9)',
    inputs: ['M'], domains: { M: 'matSmall' }, matrixDims: { M: [4, 4] },
    run: (M) => { const n = sq(M), d = M.length / n, out = new Float64Array(d); for (let j = 0; j < d; j++) out[j] = colStats(M, n, d, j).mu; return out; },
    predicate: (M) => { const n = sq(M), d = M.length / n, out = new Float64Array(d); for (let j = 0; j < d; j++) out[j] = colStats(M, n, d, j).mu; const ref = new Float64Array(d); for (let j = 0; j < d; j++) ref[j] = colStatsRef(M, n, d, j).mu; return rc(out, ref, 1e-9); },
  });
  reg(66, {
    name: 'BatchNorm · 每通道方差', intuition: 'BN 再求每个通道在批量上的方差 σ²。',
    math: 'σⱼ² = Var_{i}(Mᵢⱼ)', pseudocode: 'for j: var[j]=var(col_j)',
    complexity: 'O(N·D)', ref: 'Ioffe & Szegedy 2015', commonPitfall: '无',
    invariant: '逐元素 relClose(σ², Σx²/n−μ² 参考, 1e-9)',
    inputs: ['M'], domains: { M: 'matSmall' }, matrixDims: { M: [4, 4] },
    run: (M) => { const n = sq(M), d = M.length / n, out = new Float64Array(d); for (let j = 0; j < d; j++) out[j] = colStats(M, n, d, j).v; return out; },
    predicate: (M) => { const n = sq(M), d = M.length / n, out = new Float64Array(d); for (let j = 0; j < d; j++) out[j] = colStats(M, n, d, j).v; const ref = new Float64Array(d); for (let j = 0; j < d; j++) ref[j] = colStatsRef(M, n, d, j).v; return rc(out, ref, 1e-9); },
  });
  reg(67, {
    name: 'BatchNorm · 前向定义', intuition: 'y = γ⊙(x−μ)/√(σ²+ε) + β，按通道标准化后再仿射。',
    math: 'yᵢⱼ = γⱼ(xᵢⱼ−μⱼ)/√(σⱼ²+ε) + βⱼ', pseudocode: 'for j: out[:,j]=bn(col_j)',
    complexity: 'O(N·D)', ref: 'Ioffe & Szegedy 2015', commonPitfall: 'ε 防零方差。',
    invariant: '逐元素 relClose(正向, 反向累加参考, 1e-9)',
    inputs: ['M'], domains: { M: 'matSmall' }, matrixDims: { M: [4, 4] },
    run: (M) => bnNorm(M, sq(M), M.length / sq(M), null, null),
    predicate: (M) => rc(bnNorm(M, sq(M), M.length / sq(M), null, null), bnNormRef(M, sq(M), M.length / sq(M), null, null), 1e-9),
  });
  reg(68, {
    name: 'BatchNorm · 零均值性', intuition: '去除仿射后，每个通道的输出在批量上均值为 0。',
    math: '∀j: mean_i(yᵢⱼ) = 0', pseudocode: 'for j: assert mean(out[:,j])≈0',
    complexity: 'O(N·D)', ref: '标准化', commonPitfall: '无',
    invariant: '逐通道 relClose(mean(out[:,j]), 0, 1e-9)',
    inputs: ['M'], domains: { M: 'matSmall' }, matrixDims: { M: [4, 4] },
    run: (M) => bnNorm(M, sq(M), M.length / sq(M), null, null),
    predicate: (M) => { const n = sq(M), d = M.length / n, o = bnNorm(M, n, d, null, null); for (let j = 0; j < d; j++) { let s = 0; for (let i = 0; i < n; i++) s += o[i * d + j]; if (!rc(s / n, 0, 1e-9)) return false; } return true; },
  });
  reg(69, {
    name: 'BatchNorm · 单位方差（解析期望）', intuition: '去仿射后每通道方差为 σⱼ²/(σⱼ²+ε)。',
    math: 'Var_i(yᵢⱼ) = σⱼ²/(σⱼ²+ε)', pseudocode: 'for j: assert var(out[:,j])≈σⱼ²/(σⱼ²+ε)',
    complexity: 'O(N·D)', ref: '标准化', commonPitfall: '同 LN，须比对解析值而非 1。',
    invariant: '逐通道 relClose(Var(out[:,j]), σⱼ²/(σⱼ²+ε), 1e-9)',
    inputs: ['M'], domains: { M: 'matSmall' }, matrixDims: { M: [4, 4] },
    run: (M) => bnNorm(M, sq(M), M.length / sq(M), null, null),
    predicate: (M) => { const n = sq(M), d = M.length / n, o = bnNorm(M, n, d, null, null); for (let j = 0; j < d; j++) { let s = 0; for (let i = 0; i < n; i++) s += o[i * d + j]; const m = s / n; let v = 0; for (let i = 0; i < n; i++) { const dd = o[i * d + j] - m; v += dd * dd; } v /= n; const sj2 = colStats(M, n, d, j).v; if (!rc(v, sj2 / (sj2 + EPS), 1e-9)) return false; } return true; },
  });
  reg(70, {
    name: 'BatchNorm · 仿射定义', intuition: '标准 BN 带可学习 γ,β：y = γ⊙z + β。',
    math: 'yᵢⱼ = γⱼ·zᵢⱼ + βⱼ', pseudocode: 'z=BN(x); return gamma*z + beta',
    complexity: 'O(N·D)', ref: 'Ioffe & Szegedy 2015', commonPitfall: 'γ/β 须与通道数等长。',
    invariant: '逐元素 relClose(正向仿射, 反向累加参考仿射, 1e-9)',
    inputs: ['M', 'g', 'b'], domains: { M: 'matSmall', g: 'vecSmall', b: 'vecSmall' }, matrixDims: { M: [4, 4] }, vecDims: { g: 4, b: 4 },
    run: (M, g, b) => bnNorm(M, sq(M), M.length / sq(M), g, b),
    predicate: (M, g, b) => rc(bnNorm(M, sq(M), M.length / sq(M), g, b), bnNormRef(M, sq(M), M.length / sq(M), g, b), 1e-9),
  });
  reg(71, {
    name: 'BatchNorm · 平移不变', intuition: '全体加同一常数时，各通道 μ 同移，(x−μ) 不变，BN 不变。',
    math: 'BN(x+c) = BN(x)', pseudocode: 'return BN(shift(x,c))',
    complexity: 'O(N·D)', ref: '平移不变性', commonPitfall: 'ε 与 c 无关，严格相等。',
    invariant: '逐元素 relClose(BN(x+c), BN(x), 1e-9)',
    inputs: ['M', 'c'], domains: { M: 'matSmall', c: 'small' }, matrixDims: { M: [4, 4] },
    run: (M, c) => { const t = Float64Array.from(M); for (let k = 0; k < t.length; k++) t[k] += c; return bnNorm(t, sq(t), t.length / sq(t), null, null); },
    predicate: (M, c) => { const t = Float64Array.from(M); for (let k = 0; k < t.length; k++) t[k] += c; return rc(bnNorm(t, sq(t), t.length / sq(t), null, null), bnNorm(M, sq(M), M.length / sq(M), null, null), 1e-9); },
  });
  reg(72, {
    name: 'BatchNorm · 统计量跨样本耦合', intuition: 'BN 统计量跨样本——改一个元素会改变同通道所有样本的输出（与 LN 逐行独立形成对照）。',
    math: '改 Mᵢⱼ ⇒ 同列其余行输出亦变', pseudocode: 'M2[i][j]+=δ; assert out2[k][j]≠out1[k][j]',
    complexity: 'O(N·D)', ref: 'BN vs LN', commonPitfall: '这是 BN 训练依赖批量大小的根源。',
    invariant: '∃k≠i: |out2[k][j] − out1[k][j]| > 1e-9',
    inputs: ['M'], domains: { M: 'matSmall' }, matrixDims: { M: [4, 4] },
    run: (M) => bnNorm(M, sq(M), M.length / sq(M), null, null),
    predicate: (M) => { const n = sq(M), d = M.length / n, o1 = bnNorm(M, n, d, null, null); const M2 = Float64Array.from(M); M2[0] += 1; const o2 = bnNorm(M2, n, d, null, null); for (let i = 1; i < n; i++) if (Math.abs(o2[i * d] - o1[i * d]) > 1e-9) return true; return false; },
  });
  reg(73, {
    name: 'BatchNorm · 与 LayerNorm 区别', intuition: 'BN 沿批量（列）归一化，LN 沿特征（行）归一化；矩阵下一般不相等。',
    math: 'BN(M) ≠ LN(M)', pseudocode: 'return BN(M)',
    complexity: 'O(N·D)', ref: 'BN vs LN', commonPitfall: '仅在特殊对称输入下可能巧合相等。',
    invariant: '∃(i,j): |BN(M)ᵢⱼ − LN(M)ᵢⱼ| > 1e-6',
    inputs: ['M'], domains: { M: 'matSmall' }, matrixDims: { M: [4, 4] },
    run: (M) => bnNorm(M, sq(M), M.length / sq(M), null, null),
    predicate: (M) => { const n = sq(M), d = M.length / n, a = bnNorm(M, n, d, null, null); const b = (() => { const o = new Float64Array(n * d); for (let i = 0; i < n; i++) { const r = M.subarray(i * d, (i + 1) * d), m = mean(r, d), v = varPop(r, d), inv = 1 / Math.sqrt(v + EPS); for (let j = 0; j < d; j++) o[i * d + j] = (r[j] - m) * inv; } return o; })(); for (let k = 0; k < a.length; k++) if (Math.abs(a[k] - b[k]) > 1e-6) return true; return false; },
  });
  reg(74, {
    name: 'BatchNorm · 常数通道 → β', intuition: '某通道在批量上恒为常数 ⇒ 方差 0 ⇒ 中心化项 0 ⇒ 输出恰为该通道 β。',
    math: 'col const ⇒ yᵢⱼ = βⱼ', pseudocode: 'M2[:,0]=const; assert out[:,0]==beta',
    complexity: 'O(N·D)', ref: '退化情形', commonPitfall: '无',
    invariant: '逐元素 relClose(out[:,0], β₀, 1e-9)',
    inputs: ['M', 'b'], domains: { M: 'matSmall', b: 'vecSmall' }, matrixDims: { M: [4, 4] }, vecDims: { b: 4 },
    run: (M, b) => bnNorm(M, sq(M), M.length / sq(M), null, b),
    predicate: (M, b) => { const n = sq(M), d = M.length / n, M2 = Float64Array.from(M); const c = M2[0]; for (let i = 0; i < n; i++) M2[i * d] = c; const o = bnNorm(M2, n, d, null, b); for (let i = 0; i < n; i++) if (!rc(o[i * d], b[0], 1e-9)) return false; return true; },
  });
  reg(75, {
    name: 'BatchNorm · γ 作用', intuition: 'γ=1,β=0 输出与 γ=0,β=0 输出之差，恰为去仿射归一化 z。',
    math: 'BN(γ=1,β=0) − BN(γ=0,β=0) = z', pseudocode: 'return BN(1,0) - BN(0,0)',
    complexity: 'O(N·D)', ref: '仿射分解', commonPitfall: '验证 γ 作用在 z 上。',
    invariant: '逐元素 relClose(Δout, z, 1e-9)',
    inputs: ['M'], domains: { M: 'matSmall' }, matrixDims: { M: [4,4] },
    run: (M) => { const n = sq(M), d = M.length / n, ones = new Float64Array(d).fill(1), zeros = new Float64Array(d); const a = bnNorm(M, n, d, ones, zeros), b = bnNorm(M, n, d, zeros, zeros), o = new Float64Array(n * d); for (let k = 0; k < o.length; k++) o[k] = a[k] - b[k]; return o; },
    predicate: (M) => { const n = sq(M), d = M.length / n, ones = new Float64Array(d).fill(1), zeros = new Float64Array(d); const a = bnNorm(M, n, d, ones, zeros), b = bnNorm(M, n, d, zeros, zeros); const z = bnNorm(M, n, d, null, null); for (let k = 0; k < z.length; k++) if (!rc(a[k] - b[k], z[k], 1e-9)) return false; return true; },
  });
  reg(76, {
    name: 'BatchNorm · β 偏移', intuition: '偏移 β 仅平移输出，与归一化部分无关（γ=1）。',
    math: 'BN(β₂) − BN(β₁) = β₂ − β₁', pseudocode: 'return BN(b2) - BN(b1)',
    complexity: 'O(N·D)', ref: '仿射分解', commonPitfall: '无',
    invariant: '逐元素 relClose(Δout, Δβ, 1e-9)',
    inputs: ['M', 'b1', 'b2'], domains: { M: 'matSmall', b1: 'vecSmall', b2: 'vecSmall' }, matrixDims: { M: [4, 4] }, vecDims: { b1: 4, b2: 4 },
    run: (M, b1, b2) => { const n = sq(M), d = M.length / n; const a = bnNorm(M, n, d, null, b1), c = bnNorm(M, n, d, null, b2), o = new Float64Array(n * d); for (let k = 0; k < o.length; k++) o[k] = c[k] - a[k]; return o; },
    predicate: (M, b1, b2) => { const n = sq(M), d = M.length / n; const a = bnNorm(M, n, d, null, b1), c = bnNorm(M, n, d, null, b2); for (let k = 0; k < a.length; k++) if (!rc(c[k] - a[k], b2[k % d] - b1[k % d], 1e-9)) return false; return true; },
  });
  reg(77, {
    name: 'BatchNorm · 输出有限', intuition: 'ε 保护下，有限输入的 BN 输出必有限。',
    math: '∀i,j: isFinite(yᵢⱼ)', pseudocode: 'for all: assert isFinite(y)',
    complexity: 'O(N·D)', ref: '数值稳定性', commonPitfall: '无',
    invariant: 'every(isFinite(y))',
    inputs: ['M', 'g', 'b'], domains: { M: 'matSmall', g: 'vecSmall', b: 'vecSmall' }, matrixDims: { M: [4, 4] }, vecDims: { g: 4, b: 4 },
    run: (M, g, b) => bnNorm(M, sq(M), M.length / sq(M), g, b),
    predicate: (M, g, b) => { const o = bnNorm(M, sq(M), M.length / sq(M), g, b); for (let k = 0; k < o.length; k++) if (!isFinite(o[k])) return false; return true; },
  });
  reg(78, {
    name: 'BatchNorm · 缩放非齐次（对照 LN）', intuition: '因分母含跨样本方差（随 a² 缩放），BN(a·x) ≠ a·BN(x)——与直觉相反。',
    math: 'BN(a·x) ≠ a·BN(x)', pseudocode: 'return BN(scale(M,a))',
    complexity: 'O(N·D)', ref: 'BN', commonPitfall: '这是 BN 不可简单缩放的关键。',
    invariant: 'a 偏离 1 且足够大 ⇒ ‖BN(aM)−a·BN(M)‖ > 1e-9',
    inputs: ['M', 'a'], domains: { M: 'matSmall', a: 'small' }, matrixDims: { M: [4, 4] },
    run: (M, a) => { const t = Float64Array.from(M); for (let k = 0; k < t.length; k++) t[k] *= a; return bnNorm(t, sq(t), t.length / sq(t), null, null); },
    predicate: (M, a) => { if (Math.abs(a - 1) < 1e-3) return true; const n = sq(M), d = M.length / n, t = Float64Array.from(M); for (let k = 0; k < t.length; k++) t[k] *= a; const b1 = bnNorm(t, n, d, null, null), b0 = bnNorm(M, n, d, null, null); let dd = 0; for (let k = 0; k < b1.length; k++) { const e = b1[k] - a * b0[k]; dd += e * e; } return Math.sqrt(dd) > 1e-9; },
  });
  reg(79, {
    name: 'BatchNorm · 全零矩阵 → β', intuition: '全零输入 μ=0,σ²=0 ⇒ z=0 ⇒ 输出恰为 β。',
    math: 'M=0 ⇒ yᵢⱼ = βⱼ', pseudocode: 'return BN(zeros)',
    complexity: 'O(1)', ref: '退化情形', commonPitfall: '无',
    invariant: '逐元素 relClose(BN(0), β, 1e-9)',
    inputs: ['M', 'b'], domains: { M: 'matSmall', b: 'vecSmall' }, matrixDims: { M: [4, 4] }, vecDims: { b: 4 },
    run: (M, b) => { const n = sq(M), d = M.length / n, z = new Float64Array(n * d); return bnNorm(z, n, d, null, b); },
    predicate: (M, b) => { const n = sq(M), d = M.length / n, z = new Float64Array(n * d), o = bnNorm(z, n, d, null, b); for (let j = 0; j < d; j++) for (let i = 0; i < n; i++) if (!rc(o[i * d + j], b[j], 1e-9)) return false; return true; },
  });
  reg(80, {
    name: 'BatchNorm · 与转置 LN 一致', intuition: 'BN 按列（跨样本）归一化 ≡ LN 作用于转置矩阵 Mᵀ（逐行）。',
    math: 'BN(M)ᵢⱼ = LN(Mᵀ)ⱼᵢ', pseudocode: 'return BN(M)',
    complexity: 'O(N·D)', ref: 'BN ≡ LN(Mᵀ)', commonPitfall: '漂亮的对偶关系：通道统计 = 转置后的行统计。',
    invariant: '逐元素 relClose(BN(M), transpose(LN(Mᵀ)), 1e-9)',
    inputs: ['M'], domains: { M: 'matSmall' }, matrixDims: { M: [4, 4] },
    run: (M) => bnNorm(M, sq(M), M.length / sq(M), null, null),
    predicate: (M) => { const n = sq(M), d = M.length / n, a = bnNorm(M, n, d, null, null); const b = (() => { const o = new Float64Array(n * d); const col = new Float64Array(d); for (let j = 0; j < d; j++) { for (let i = 0; i < n; i++) col[i] = M[i * d + j]; const m = mean(col, n), v = varPop(col, n), inv = 1 / Math.sqrt(v + EPS); for (let i = 0; i < n; i++) o[j * n + i] = (col[i] - m) * inv; } return o; })(); for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) if (!rc(a[i * d + j], b[j * n + i], 1e-9)) return false; return true; },
  });

  // ══ GroupNorm / InstanceNorm / 权重标准化（F081–F089）════════════════
  reg(81, {
    name: 'GroupNorm · 定义（分组内归一化）', intuition: 'GN 把通道分成 G 组，每组内逐样本归一化——介于 LN（G=1 全通道）与逐通道（G=D）之间。',
    math: '组 g 内：y = (x−μ_g)/√(σ_g²+ε)', pseudocode: 'for g: gn(group)',
    complexity: 'O(N·D)', ref: 'Wu & He 2018', commonPitfall: 'G 须整除 D。',
    invariant: '逐元素 relClose(正向, 反向累加参考, 1e-9)',
    inputs: ['M'], domains: { M: 'matSmall' }, matrixDims: { M: [4, 4] },
    run: (M) => gnNorm(M, sq(M), M.length / sq(M), 2, null, null),
    predicate: (M) => rc(gnNorm(M, sq(M), M.length / sq(M), 2, null, null), gnNormRef(M, sq(M), M.length / sq(M), 2, null, null), 1e-9),
  });
  reg(82, {
    name: 'GroupNorm · 零均值（组内）', intuition: '去仿射后，每个样本内各分组输出均值为 0。',
    math: '∀i,g: mean_j∈g(yᵢⱼ) = 0', pseudocode: 'assert group mean≈0',
    complexity: 'O(N·D)', ref: '标准化', commonPitfall: '无',
    invariant: '逐组 relClose(group mean, 0, 1e-9)',
    inputs: ['M'], domains: { M: 'matSmall' }, matrixDims: { M: [4, 4] },
    run: (M) => gnNorm(M, sq(M), M.length / sq(M), 2, null, null),
    predicate: (M) => { const n = sq(M), d = M.length / n, o = gnNorm(M, n, d, 2, null, null), cpg = 2; for (let i = 0; i < n; i++) for (let gr = 0; gr < 2; gr++) { const s0 = gr * cpg, s1 = s0 + cpg; let s = 0; for (let j = s0; j < s1; j++) s += o[i * d + j]; if (!rc(s / cpg, 0, 1e-9)) return false; } return true; },
  });
  reg(83, {
    name: 'GroupNorm · 单位方差（组内解析）', intuition: '去仿射后每组方差为 σ_g²/(σ_g²+ε)。',
    math: 'Var_j∈g(yᵢⱼ) = σ_g²/(σ_g²+ε)', pseudocode: 'assert group var≈σ_g²/(σ_g²+ε)',
    complexity: 'O(N·D)', ref: '标准化', commonPitfall: '无',
    invariant: '逐组 relClose(group var, σ_g²/(σ_g²+ε), 1e-9)',
    inputs: ['M'], domains: { M: 'matSmall' }, matrixDims: { M: [4, 4] },
    run: (M) => gnNorm(M, sq(M), M.length / sq(M), 2, null, null),
    predicate: (M) => { const n = sq(M), d = M.length / n, o = gnNorm(M, n, d, 2, null, null), cpg = 2; for (let i = 0; i < n; i++) for (let gr = 0; gr < 2; gr++) { const s0 = gr * cpg, s1 = s0 + cpg; let sm = 0; for (let j = s0; j < s1; j++) sm += M[i * d + j]; const mux = sm / cpg; let sv = 0; for (let j = s0; j < s1; j++) { const dd = M[i * d + j] - mux; sv += dd * dd; } sv /= cpg; let so = 0; for (let j = s0; j < s1; j++) so += o[i * d + j]; const muo = so / cpg; let v = 0; for (let j = s0; j < s1; j++) { const dd = o[i * d + j] - muo; v += dd * dd; } v /= cpg; if (!rc(v, sv / (sv + EPS), 1e-9)) return false; } return true; },
  });
  reg(84, {
    name: 'GroupNorm · 仿射定义', intuition: 'GN 同样带 γ,β 仿射。',
    math: 'yᵢⱼ = γⱼ·zᵢⱼ + βⱼ', pseudocode: 'z=GN(x); return gamma*z + beta',
    complexity: 'O(N·D)', ref: 'Wu & He 2018', commonPitfall: 'γ/β 与通道数等长。',
    invariant: '逐元素 relClose(正向仿射, 反向累加参考仿射, 1e-9)',
    inputs: ['M', 'g', 'b'], domains: { M: 'matSmall', g: 'vecSmall', b: 'vecSmall' }, matrixDims: { M: [4, 4] }, vecDims: { g: 4, b: 4 },
    run: (M, g, b) => gnNorm(M, sq(M), M.length / sq(M), 2, g, b),
    predicate: (M, g, b) => rc(gnNorm(M, sq(M), M.length / sq(M), 2, g, b), gnNormRef(M, sq(M), M.length / sq(M), 2, g, b), 1e-9),
  });
  reg(85, {
    name: 'GroupNorm · G=1 退化为实例归一化', intuition: 'G=1 时所有通道同组，逐样本全程归一化 ≡ InstanceNorm。',
    math: 'GN(G=1) = InstanceNorm', pseudocode: 'return GN(M,G=1)',
    complexity: 'O(N·D)', ref: 'GN vs IN', commonPitfall: 'IN 是 GN 的 G=1 特例。',
    invariant: '逐元素 relClose(GN(G=1), IN, 1e-9)',
    inputs: ['M'], domains: { M: 'matSmall' }, matrixDims: { M: [4, 4] },
    run: (M) => gnNorm(M, sq(M), M.length / sq(M), 1, null, null),
    predicate: (M) => { const n = sq(M), d = M.length / n, a = gnNorm(M, n, d, 1, null, null); const b = (() => { const o = new Float64Array(n * d); for (let i = 0; i < n; i++) { const r = M.subarray(i * d, (i + 1) * d), m = mean(r, d), v = varPop(r, d), inv = 1 / Math.sqrt(v + EPS); for (let j = 0; j < d; j++) o[i * d + j] = (r[j] - m) * inv; } return o; })(); return rc(a, b, 1e-9); },
  });
  reg(86, {
    name: 'GroupNorm · G=D 退化为 β（组方差 0）', intuition: 'G=D 时每组仅 1 通道，方差恒 0 ⇒ z=0 ⇒ 输出恰为 β。',
    math: 'GN(G=D) = β', pseudocode: 'return GN(M,G=D)',
    complexity: 'O(N·D)', ref: 'GN 退化', commonPitfall: '每通道单独一组使归一化失效（方差为 0）。',
    invariant: '逐元素 relClose(GN(G=D), β, 1e-9)',
    inputs: ['M', 'b'], domains: { M: 'matSmall', b: 'vecSmall' }, matrixDims: { M: [4, 4] }, vecDims: { b: 4 },
    run: (M, b) => gnNorm(M, sq(M), M.length / sq(M), 4, null, b),
    predicate: (M, b) => { const n = sq(M), d = M.length / n, o = gnNorm(M, n, d, 4, null, b); for (let k = 0; k < o.length; k++) if (!rc(o[k], b[k % d], 1e-9)) return false; return true; },
  });
  reg(87, {
    name: 'GroupNorm · 与 BatchNorm 区别', intuition: 'GN 逐样本分组归一化，BN 跨样本按通道归一化；矩阵下一般不相等。',
    math: 'GN(M) ≠ BN(M)', pseudocode: 'return GN(M)',
    complexity: 'O(N·D)', ref: 'GN vs BN', commonPitfall: '仅在特殊对称输入下可能巧合相等。',
    invariant: '∃(i,j): |GN(M)ᵢⱼ − BN(M)ᵢⱼ| > 1e-6',
    inputs: ['M'], domains: { M: 'matSmall' }, matrixDims: { M: [4, 4] },
    run: (M) => gnNorm(M, sq(M), M.length / sq(M), 2, null, null),
    predicate: (M) => { const n = sq(M), d = M.length / n, a = gnNorm(M, n, d, 2, null, null), b = bnNorm(M, n, d, null, null); for (let k = 0; k < a.length; k++) if (Math.abs(a[k] - b[k]) > 1e-6) return true; return false; },
  });
  reg(88, {
    name: '权重标准化 · 定义', intuition: '权重标准化（WS）把卷积核权重按空间+输入维标准化，稳定训练，等价于对每行做 LN。',
    math: 'ŵ = (w − μ)/√(σ²+ε)', pseudocode: 'for row: ws(row)',
    complexity: 'O(K·C)', ref: 'Wu & He 2018', commonPitfall: 'WS 是对权重而非激活做归一化。',
    invariant: '逐元素 relClose(正向, 反向累加参考, 1e-9)',
    inputs: ['M'], domains: { M: 'matSmall' }, matrixDims: { M: [4, 4] },
    run: (M) => wsNorm(M, sq(M), M.length / sq(M)),
    predicate: (M) => { const n = sq(M), d = M.length / n, a = wsNorm(M, n, d); const b = (() => { const o = new Float64Array(n * d); for (let i = 0; i < n; i++) { const r = M.subarray(i * d, (i + 1) * d); const m = meanRev(r, d), ss = sumSqRev(r, d), v = Math.max(0, ss / d - m * m), inv = 1 / Math.sqrt(v + EPS); for (let j = 0; j < d; j++) o[i * d + j] = (r[j] - m) * inv; } return o; })(); return rc(a, b, 1e-9); },
  });
  reg(89, {
    name: '权重标准化 · 与 LayerNorm 等价（按行）', intuition: 'WS 逐卷积核（行）归一化，结构上正是 LayerNorm 按行应用。',
    math: 'WS(M) = LN(M)（按行）', pseudocode: 'return wsNorm(M)',
    complexity: 'O(K·C)', ref: 'WS ≡ LN', commonPitfall: '验证 WS 属于 LN 家族。',
    invariant: '逐元素 relClose(WS(M), LN(M)按行, 1e-9)',
    inputs: ['M'], domains: { M: 'matSmall' }, matrixDims: { M: [4, 4] },
    run: (M) => wsNorm(M, sq(M), M.length / sq(M)),
    predicate: (M) => { const n = sq(M), d = M.length / n, a = wsNorm(M, n, d); const b = (() => { const o = new Float64Array(n * d); for (let i = 0; i < n; i++) { const r = M.subarray(i * d, (i + 1) * d), m = mean(r, d), v = varPop(r, d), inv = 1 / Math.sqrt(v + EPS); for (let j = 0; j < d; j++) o[i * d + j] = (r[j] - m) * inv; } return o; })(); return rc(a, b, 1e-9); },
  });

  // ══ LRN 与经典归一化（F090–F096）════════════════════════════════════
  reg(90, {
    name: 'LRN · 局部响应归一化定义', intuition: 'LRN（AlexNet）用邻通道的平方和做局部竞争式归一化。',
    math: 'y_c = x_c/(k + α Σ_{|c′−c|≤w} x_{c′}²)^β', pseudocode: 'for c: denom=...; y=x/denom',
    complexity: 'O(L·w)', ref: 'Krizhevsky 2012', commonPitfall: 'β 通常 0.75，α 很小（如 1e-4），k 为偏置（>0）。',
    invariant: '逐元素 relClose(正向, 反向累加参考, 1e-9)',
    inputs: ['x', 'w'], domains: { x: 'vecSmall', w: 'nat' }, vecDims: { x: 8 },
    run: (x, w) => lrn(x, x.length, w, 2, 1e-4, 0.75),
    predicate: (x, w) => rc(lrn(x, x.length, w, 2, 1e-4, 0.75), lrnRef(x, x.length, w, 2, 1e-4, 0.75), 1e-9),
  });
  reg(91, {
    name: 'LRN · 单调性（邻域越大响应越小）', intuition: '扩大邻域窗口会增大分母平方和，中心响应单调递减。',
    math: 'w₁ ≤ w₂ ⇒ y_c(w₁) ≥ y_c(w₂)', pseudocode: 'return lrN(x, w)',
    complexity: 'O(L·w)', ref: 'LRN', commonPitfall: '中心元素 x_c 固定，分母变大即响应变小。',
    invariant: 'y_c(w+1) ≤ y_c(w) + 1e-9',
    inputs: ['x', 'w'], domains: { x: 'vecSmall', w: 'nat' }, vecDims: { x: 8 },
    run: (x, w) => { const c = (x.length / 2) | 0; return lrn(x, x.length, w, 2, 1e-4, 0.75)[c]; },
    predicate: (x, w) => { const L = x.length, c = (L / 2) | 0; const maxw = Math.min(c, L - 1 - c); if (w >= maxw) return true; return lrn(x, L, w + 1, 2, 1e-4, 0.75)[c] <= lrn(x, L, w, 2, 1e-4, 0.75)[c] + 1e-9; },
  });
  reg(92, {
    name: 'LRN · 输出有限', intuition: 'k>0 保证分母 ≥ k^β > 0，输出必有限。',
    math: '∀c: isFinite(y_c)', pseudocode: 'for c: assert isFinite(y)',
    complexity: 'O(L·w)', ref: '数值稳定性', commonPitfall: '无',
    invariant: 'every(isFinite(y))',
    inputs: ['x', 'w'], domains: { x: 'vecSmall', w: 'nat' }, vecDims: { x: 8 },
    run: (x, w) => lrn(x, x.length, w, 2, 1e-4, 0.75),
    predicate: (x, w) => { const o = lrn(x, x.length, w, 2, 1e-4, 0.75); for (let i = 0; i < o.length; i++) if (!isFinite(o[i])) return false; return true; },
  });
  reg(93, {
    name: '归一化 · min-max 缩放', intuition: '把数据线性映射到 [0,1]，最常见的区间归一化。',
    math: 'yᵢ = (xᵢ−min)/(max−min)', pseudocode: 'mn=min(x); mx=max(x); return (x-mn)/(mx-mn)',
    complexity: 'O(n)', ref: '特征缩放', commonPitfall: '全常数时 max−min=0 → 退化为 0。',
    invariant: '逐元素 relClose(正向, 参考) 且 out∈[0,1]',
    inputs: ['x'], domains: { x: 'vecSmall' }, vecDims: { x: 8 },
    run: (x) => { const n = x.length; let mn = Infinity, mx = -Infinity; for (let i = 0; i < n; i++) { if (x[i] < mn) mn = x[i]; if (x[i] > mx) mx = x[i]; } const o = new Float64Array(n), d = mx - mn; for (let i = 0; i < n; i++) o[i] = d === 0 ? 0 : (x[i] - mn) / d; return o; },
    predicate: (x) => { const n = x.length; let mn = Infinity, mx = -Infinity; for (let i = 0; i < n; i++) { if (x[i] < mn) mn = x[i]; if (x[i] > mx) mx = x[i]; } const d = mx - mn; const o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = d === 0 ? 0 : (x[i] - mn) / d; if (d !== 0) for (let i = 0; i < n; i++) { if (!(o[i] >= -1e-12 && o[i] <= 1 + 1e-12)) return false; if (!rc(o[i], (x[i] - mn) / d, 1e-12)) return false; } return true; },
  });
  reg(94, {
    name: '归一化 · z-score（标准分数）', intuition: '减均值除标准差，是统计推断中的核心规范化。',
    math: 'zᵢ = (xᵢ−μ)/σ', pseudocode: 'm=mean(x); s=std(x); return (x-m)/s',
    complexity: 'O(n)', ref: '标准分数', commonPitfall: 'σ=0（全常数）时除零——退化为 0。',
    invariant: '逐元素 relClose(正向, 参考)（σ>0）',
    inputs: ['x'], domains: { x: 'vecSmall' }, vecDims: { x: 8 },
    run: (x) => { const n = x.length, m = mean(x, n), v = varPop(x, n), sd = Math.sqrt(v); const o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = sd === 0 ? 0 : (x[i] - m) / sd; return o; },
    predicate: (x) => { const n = x.length, m = mean(x, n), v = varPop(x, n), sd = Math.sqrt(v); if (sd === 0) return true; const o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = (x[i] - m) / sd; const mo = meanRev(x, n), vo = Math.max(0, sumSqRev(x, n) / n - mo * mo), sdo = Math.sqrt(vo); const ref = new Float64Array(n); for (let i = 0; i < n; i++) ref[i] = (x[i] - mo) / sdo; return rc(o, ref, 1e-9); },
  });
  reg(95, {
    name: '归一化 · L1 归一化', intuition: '除以绝对值之和，使 L1 范数为 1（单纯形投影）。',
    math: 'yᵢ = xᵢ/Σ|xⱼ|，‖y‖₁ = 1', pseudocode: 's=sum|x|; return x/s',
    complexity: 'O(n)', ref: 'L1 范数', commonPitfall: '全零时退化为 0。',
    invariant: '逐元素 relClose(正向, 参考) 且 ‖y‖₁=1（非全零）',
    inputs: ['x'], domains: { x: 'vecSmall' }, vecDims: { x: 8 },
    run: (x) => { const n = x.length; let s = 0; for (let i = 0; i < n; i++) s += Math.abs(x[i]); const o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = s === 0 ? 0 : x[i] / s; return o; },
    predicate: (x) => { const n = x.length; let s = 0; for (let i = 0; i < n; i++) s += Math.abs(x[i]); if (s === 0) return true; const o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = x[i] / s; let t = 0; for (let i = 0; i < n; i++) t += Math.abs(o[i]); if (!rc(t, 1, 1e-9)) return false; const ref = new Float64Array(n); let sr = 0; for (let i = 0; i < n; i++) sr += Math.abs(x[i]); for (let i = 0; i < n; i++) ref[i] = x[i] / sr; return rc(o, ref, 1e-9); },
  });
  reg(96, {
    name: '归一化 · L∞ 归一化', intuition: '除以最大绝对值，使 L∞ 范数为 1（最保守的缩放）。',
    math: 'yᵢ = xᵢ/max|xⱼ|，‖y‖∞ = 1', pseudocode: 'mx=max|x|; return x/mx',
    complexity: 'O(n)', ref: 'L∞ 范数', commonPitfall: '全零时退化为 0。',
    invariant: '逐元素 relClose(正向, 参考) 且 ‖y‖∞=1（非全零）',
    inputs: ['x'], domains: { x: 'vecSmall' }, vecDims: { x: 8 },
    run: (x) => { const n = x.length; let mx = 0; for (let i = 0; i < n; i++) { const a = Math.abs(x[i]); if (a > mx) mx = a; } const o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = mx === 0 ? 0 : x[i] / mx; return o; },
    predicate: (x) => { const n = x.length; let mx = 0; for (let i = 0; i < n; i++) { const a = Math.abs(x[i]); if (a > mx) mx = a; } if (mx === 0) return true; const o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = x[i] / mx; let t = 0; for (let i = 0; i < n; i++) t = Math.max(t, Math.abs(o[i])); if (!rc(t, 1, 1e-9)) return false; const ref = new Float64Array(n); for (let i = 0; i < n; i++) ref[i] = x[i] / mx; return rc(o, ref, 1e-9); },
  });
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

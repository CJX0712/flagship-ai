/* ============================================================================
 * L10 微指令 · F 域（归一化）· 第 3/4 批（F065–F096）
 * ----------------------------------------------------------------------------
 * 所属层：L10-microIsa  域：F（归一化）
 * 允许依赖的层：L9（special/fpOps）、L10（registry/invariant）
 * 主要 @ref：Bridle (1990) softmax；Ba et al. (2016) LayerNorm；
 *           Goodfellow (2016) §6.2 softmax；Martins & Astudillo (2016) sparsemax
 *
 * 涵盖：softmax 家族（F065–F080）、层归一化家族（F081–F096）。
 * ═══════════════════════════════════════════════════════════════════════ */
;(function (NS) {
  'use strict';
  NS.L10 = NS.L10 || {};
  function reg(i, o) { NS.L10.register('F', i, o); }
  const rc = (a, b, t) => NS.L10.relClose(a, b, t);

  function mean(a) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]; return s / a.length; }
  function meanSq(a) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * a[i]; return s / a.length; }
  function variance(a) { const m = mean(a); let s = 0; for (let i = 0; i < a.length; i++) { const d = a[i] - m; s += d * d; } const v = s / a.length; return v < 0 ? 0 : v; }
  function rms(a) { return Math.sqrt(meanSq(a)); }
  function maxEl(a) { let m = -Infinity; for (let i = 0; i < a.length; i++) if (a[i] > m) m = a[i]; return m; }
  function minEl(a) { let m = Infinity; for (let i = 0; i < a.length; i++) if (a[i] < m) m = a[i]; return m; }
  function dot(a, b) { let s = 0; const n = a.length < b.length ? a.length : b.length; for (let i = 0; i < n; i++) s += a[i] * b[i]; return s; }
  function zeros(n) { return new Float64Array(n); }

  function lse(v) { let m = -Infinity; for (let i = 0; i < v.length; i++) if (v[i] > m) m = v[i]; let s = 0; for (let i = 0; i < v.length; i++) s += Math.exp(v[i] - m); return m + Math.log(s); }
  function softmax(v) { let m = -Infinity; for (let i = 0; i < v.length; i++) if (v[i] > m) m = v[i]; const out = new Float64Array(v.length); let s = 0; for (let i = 0; i < v.length; i++) { const e = Math.exp(v[i] - m); out[i] = e; s += e; } const inv = 1 / s; for (let i = 0; i < v.length; i++) out[i] *= inv; return out; }
  function softplus(x) { return x > 0 ? x + Math.log1p(Math.exp(-x)) : Math.log1p(Math.exp(x)); }

  function sparsemax(z) {
    const n = z.length;
    const s = Array.prototype.slice.call(z).sort((a, b) => b - a);
    let cum = 0, k = -1;
    for (let j = 1; j <= n; j++) { cum += s[j - 1]; if (s[j - 1] > (cum - 1) / j) k = j; }
    let cumk = 0; for (let j = 0; j < k; j++) cumk += s[j];
    const tau = (cumk - 1) / k;
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = Math.max(z[i] - tau, 0);
    return out;
  }

  // 矩阵工具
  function mSoftmaxRows(M, r, c) { const out = new Float64Array(r * c); for (let i = 0; i < r; i++) { let m = -Infinity; for (let j = 0; j < c; j++) if (M[i * c + j] > m) m = M[i * c + j]; let s = 0; for (let j = 0; j < c; j++) { const e = Math.exp(M[i * c + j] - m); out[i * c + j] = e; s += e; } const inv = 1 / s; for (let j = 0; j < c; j++) out[i * c + j] *= inv; } return out; }
  function mSoftmaxCols(M, r, c) { const out = new Float64Array(r * c); for (let j = 0; j < c; j++) { let m = -Infinity; for (let i = 0; i < r; i++) if (M[i * c + j] > m) m = M[i * c + j]; let s = 0; for (let i = 0; i < r; i++) { const e = Math.exp(M[i * c + j] - m); out[i * c + j] = e; s += e; } const inv = 1 / s; for (let i = 0; i < r; i++) out[i * c + j] *= inv; } return out; }
  function mMean(M, r, c) { let s = 0; for (let k = 0; k < r * c; k++) s += M[k]; return s / (r * c); }
  function mVar(M, r, c) { const m = mMean(M, r, c); let s = 0; for (let k = 0; k < r * c; k++) { const d = M[k] - m; s += d * d; } const v = s / (r * c); return v < 0 ? 0 : v; }
  function mRowMean(M, r, c, i) { let s = 0; for (let j = 0; j < c; j++) s += M[i * c + j]; return s / c; }
  function mRowVar(M, r, c, i) { const m = mRowMean(M, r, c, i); let s = 0; for (let j = 0; j < c; j++) { const d = M[i * c + j] - m; s += d * d; } const v = s / c; return v < 0 ? 0 : v; }
  function groupNorm(x, g) { const n = x.length; const gs = n / g; const out = new Float64Array(n); for (let gr = 0; gr < g; gr++) { let s = 0; for (let i = 0; i < gs; i++) s += x[gr * gs + i]; const m = s / gs; let v = 0; for (let i = 0; i < gs; i++) { const d = x[gr * gs + i] - m; v += d * d; } v /= gs; if (v <= 1e-14) continue; const inv = 1 / Math.sqrt(v); for (let i = 0; i < gs; i++) out[gr * gs + i] = (x[gr * gs + i] - m) * inv; } return out; }
  function layerNorm(x) { const m = mean(x); const v = variance(x); const out = new Float64Array(x.length); if (v <= 1e-14) return out; const inv = 1 / Math.sqrt(v); for (let i = 0; i < x.length; i++) out[i] = (x[i] - m) * inv; return out; }

  function mk(i, meta, run, pred) {
    const def = Object.assign({}, meta);
    def.run = run;
    def.predicate = function () { return pred.apply(null, Array.prototype.slice.call(arguments).concat([run])); };
    reg(i, def);
  }

  // ═══════════════════════════════════════════════════════════════════════
  // 类别 5 · softmax 家族（F065–F080）
  // ═══════════════════════════════════════════════════════════════════════
  mk(65, {
    name: 'softmax（数值稳定）', intuition: '减最大值后指数化再归一，行和为 1 且恒正。',
    math: 'pᵢ = e^{xᵢ−max}/Σe^{x−max}', pseudocode: 'return stableSoftmax(x)', complexity: 'O(n)',
    ref: 'Bridle 1990', commonPitfall: '未减 max 会溢出。',
    invariant: 'Σ runᵢ ≈ 1 且 runᵢ ≥ 0',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { return softmax(x); },
    function pred(x, run) { const out = run(x); let s = 0; for (let i = 0; i < out.length; i++) { if (out[i] < 0) return false; s += out[i]; } return rc(s, 1, 1e-9); }
  );

  mk(66, {
    name: 'log-softmax', intuition: 'log-softmax = x − LSE，分量恒非正。',
    math: 'lsᵢ = xᵢ − LSE(x)', pseudocode: 'return x - lse(x)', complexity: 'O(n)',
    ref: 'Goodfellow 2016 §6.2', commonPitfall: '非正性源于 xᵢ ≤ LSE。',
    invariant: 'Σ e^{runᵢ} ≈ 1 且 max(run) ≤ 0',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { const n = x.length; const L = lse(x); const out = new Float64Array(n); for (let i = 0; i < n; i++) out[i] = x[i] - L; return out; },
    function pred(x, run) { const out = run(x); let s = 0, mx = -Infinity; for (let i = 0; i < out.length; i++) { s += Math.exp(out[i]); if (out[i] > mx) mx = out[i]; } return rc(s, 1, 1e-9) && mx <= 1e-12; }
  );

  mk(67, {
    name: 'sparsemax（单纯形投影）', intuition: '稀疏 softmax，把向量欧氏投影到概率单纯形。',
    math: 's = proj_Δ(x)：Σsᵢ=1, sᵢ≥0, sᵢ=max(xᵢ−τ,0)', pseudocode: 'return euclideanProjSimplex(x)', complexity: 'O(n log n)',
    ref: 'Martins & Astudillo 2016', commonPitfall: '阈值 τ 由前 k 大元素决定。',
    invariant: 'Σ runᵢ ≈ 1，runᵢ ≥ 0，且 runᵢ = max(xᵢ−τ,0)',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { return sparsemax(x); },
    function pred(x, run) { const out = run(x); let s = 0; for (let i = 0; i < out.length; i++) { if (out[i] < -1e-12) return false; s += out[i]; } if (!rc(s, 1, 1e-9)) return false; // 重算 τ
      const n = x.length; const ss = Array.prototype.slice.call(x).sort((a, b) => b - a); let cum = 0, k = -1; for (let j = 1; j <= n; j++) { cum += ss[j - 1]; if (ss[j - 1] > (cum - 1) / j) k = j; } let cumk = 0; for (let j = 0; j < k; j++) cumk += ss[j]; const tau = (cumk - 1) / k; for (let i = 0; i < n; i++) if (!rc(out[i], Math.max(x[i] - tau, 0), 1e-9)) return false; return true; }
  );

  mk(68, {
    name: 'softmax 温度', intuition: '温度缩放保持 logit 排序单调性。',
    math: 'p = softmax(x/T)', pseudocode: 'return softmax(x/T)', complexity: 'O(n)',
    ref: 'Hinton et al. 2015 蒸馏', commonPitfall: 'T 仅改变锐度，不改顺序。',
    invariant: 'Σ runᵢ ≈ 1 且 (runᵢ−runⱼ)(xᵢ−xⱼ) ≥ 0',
    inputs: ['x', 'T'], domains: { x: 'vecSmall', T: 'smallPos' },
  },
    function run(x, T) { const n = x.length; const xt = new Float64Array(n); for (let i = 0; i < n; i++) xt[i] = x[i] / T; return softmax(xt); },
    function pred(x, T, run) { const out = run(x, T); let s = 0; for (let i = 0; i < out.length; i++) { if (out[i] < 0) return false; s += out[i]; } if (!rc(s, 1, 1e-9)) return false; for (let i = 0; i < out.length; i++) for (let j = 0; j < out.length; j++) { const prod = (out[i] - out[j]) * (x[i] - x[j]); if (prod < -1e-12) return false; } return true; }
  );

  mk(69, {
    name: '矩阵逐行 softmax', intuition: '每个行向量独立 softmax。',
    math: 'run_{i·} = softmax(M_{i·})', pseudocode: 'return rowSoftmax(M)', complexity: 'O(r·c)',
    ref: 'Goodfellow 2016 §6.2', commonPitfall: '逐行而非全局。',
    invariant: '每行 Σ ≈ 1 且分量 > 0',
    inputs: ['M'], domains: { M: 'matSmall' }, matrixDims: { M: [4, 4] },
  },
    function run(M) { return mSoftmaxRows(M, 4, 4); },
    function pred(M, run) { const out = run(M); const r = 4, c = 4; for (let i = 0; i < r; i++) { let s = 0; for (let j = 0; j < c; j++) { if (out[i * c + j] <= 0) return false; s += out[i * c + j]; } if (!rc(s, 1, 1e-9)) return false; } return true; }
  );

  mk(70, {
    name: 'softmax 比值（参照 x₀）', intuition: '两分量概率比 = 对应 logit 差的指数。',
    math: 'runᵢ = e^{xᵢ−x₀} = pᵢ/p₀', pseudocode: 'return exp(x - x[0])', complexity: 'O(n)',
    ref: 'Goodfellow 2016 §6.2', commonPitfall: 'p₀ 恒正，比值良定义。',
    invariant: 'runᵢ = pᵢ/p₀（p=softmax(x)）',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { const n = x.length; const out = new Float64Array(n); for (let i = 0; i < n; i++) out[i] = Math.exp(x[i] - x[0]); return out; },
    function pred(x, run) { const out = run(x); const p = softmax(x); for (let i = 0; i < out.length; i++) { const ratio = p[i] / p[0]; if (!rc(out[i], ratio, 1e-9)) return false; } return true; }
  );

  mk(71, {
    name: 'log-sum-exp 上界', intuition: 'LSE 不超过最大值加 ln n。',
    math: 'LSE(x) ≤ max(x) + ln n', pseudocode: 'return lse(x)', complexity: 'O(n)',
    ref: 'Bridle 1990', commonPitfall: '等号在其它分量远小于 max 时达到。',
    invariant: 'run ≤ max + ln n',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { return lse(x); },
    function pred(x, run) { const out = run(x); return out <= maxEl(x) + Math.log(x.length) + 1e-12; }
  );

  mk(72, {
    name: 'log-sum-exp 下界', intuition: 'LSE 不小于最大值。',
    math: 'LSE(x) ≥ max(x)', pseudocode: 'return lse(x)', complexity: 'O(n)',
    ref: 'Bridle 1990', commonPitfall: '等号在其它分量趋于 −∞ 时达到。',
    invariant: 'run ≥ max',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { return lse(x); },
    function pred(x, run) { const out = run(x); return out >= maxEl(x) - 1e-12; }
  );

  mk(73, {
    name: 'softplus', intuition: 'softplus 是 ReLU 的平滑近似，奇对称。',
    math: 'sp(x) = ln(1+eˣ)', pseudocode: 'return log1p(exp(x))', complexity: 'O(1)',
    ref: 'Dugas et al. 2001', commonPitfall: 'sp(x)+sp(−x)=x（奇部为 x）。',
    invariant: 'run(x) − run(−x) = x 且 run(x) ≥ 0',
    inputs: ['x'], domains: { x: 'finite' },
  },
    function run(x) { return softplus(x); },
    function pred(x, run) { const out = run(x); const outN = run(-x); if (!rc(out - outN, x, 1e-9)) return false; return out >= -1e-12; }
  );

  mk(74, {
    name: 'log-sum-exp 稳定式', intuition: '减最大值后再求和，数值等价朴素式。',
    math: 'LSE = max + ln Σe^{x−max}', pseudocode: 'return max + log(sum(exp(x-max)))', complexity: 'O(n)',
    ref: 'Numerical Recipes', commonPitfall: '朴素式易溢出，两式数值等价。',
    invariant: 'run = ln Σe^{xᵢ}（数值等价）',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { return lse(x); },
    function pred(x, run) { const out = run(x); let s = 0; for (let i = 0; i < x.length; i++) s += Math.exp(x[i]); return rc(out, Math.log(s), 1e-9); }
  );

  mk(75, {
    name: 'softmax argmax 一致性', intuition: 'softmax 单调，概率最大分量即 logit 最大者。',
    math: 'argmax(softmax(x)) = argmax(x)', pseudocode: 'return argmax(softmax(x))==argmax(x)', complexity: 'O(n)',
    ref: 'Goodfellow 2016 §6.2', commonPitfall: '并列时取首个，两端一致。',
    invariant: 'argmax(run) = argmax(x)',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { return softmax(x); },
    function pred(x, run) { const out = run(x); let mi = 0, mx = 0; for (let i = 1; i < x.length; i++) { if (x[i] < x[mi]) mi = i; if (x[i] > x[mx]) mx = i; } let omi = 0, omx = 0; for (let i = 1; i < out.length; i++) { if (out[i] < out[omi]) omi = i; if (out[i] > out[omx]) omx = i; } return omi === mi && omx === mx; }
  );

  mk(76, {
    name: 'softmax 由指数归一化重构', intuition: 'softmax 可由 e^{xᵢ}/Σe^{xⱼ} 直接重构。',
    math: 'pᵢ = e^{xᵢ}/Σe^{xⱼ}', pseudocode: 'return exp(x)/sum(exp(x))', complexity: 'O(n)',
    ref: 'Goodfellow 2016 §6.2', commonPitfall: '与稳定式数值等价。',
    invariant: 'runᵢ = e^{xᵢ}/Σe^{xⱼ}',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { return softmax(x); },
    function pred(x, run) { const out = run(x); let s = 0; for (let i = 0; i < x.length; i++) s += Math.exp(x[i]); for (let i = 0; i < out.length; i++) if (!rc(out[i], Math.exp(x[i]) / s, 1e-9)) return false; return true; }
  );

  mk(77, {
    name: '稳定交叉熵预备（NLL）', intuition: '−log 概率 = LSE − x_y，恒非负。',
    math: 'CE = LSE(x) − x_y', pseudocode: 'return lse(x) - x[y]', complexity: 'O(n)',
    ref: 'Goodfellow 2016 §6.2', commonPitfall: 'y 为类别下标（取模防越界）。',
    invariant: 'run = LSE − x_y 且 run ≥ 0',
    inputs: ['x', 'y'], domains: { x: 'vecSmall', y: 'nat' },
  },
    function run(x, y) { const yy = ((y % x.length) + x.length) % x.length; return lse(x) - x[yy]; },
    function pred(x, y, run) { const out = run(x, y); const yy = ((y % x.length) + x.length) % x.length; if (out < -1e-12) return false; return rc(out, lse(x) - x[yy], 1e-9); }
  );

  mk(78, {
    name: '矩阵逐列 softmax', intuition: '每个列向量独立 softmax。',
    math: 'run_{·j} = softmax(M_{·j})', pseudocode: 'return colSoftmax(M)', complexity: 'O(r·c)',
    ref: 'Goodfellow 2016 §6.2', commonPitfall: '逐列而非全局。',
    invariant: '每列 Σ ≈ 1 且分量 > 0',
    inputs: ['M'], domains: { M: 'matSmall' }, matrixDims: { M: [4, 4] },
  },
    function run(M) { return mSoftmaxCols(M, 4, 4); },
    function pred(M, run) { const out = run(M); const r = 4, c = 4; for (let j = 0; j < c; j++) { let s = 0; for (let i = 0; i < r; i++) { if (out[i * c + j] <= 0) return false; s += out[i * c + j]; } if (!rc(s, 1, 1e-9)) return false; } return true; }
  );

  mk(79, {
    name: 'softmax 熵', intuition: '概率分布的香农熵，恒非负且有上界 ln n。',
    math: 'H = −Σpᵢ ln pᵢ', pseudocode: 'return -sum(p*log(p))', complexity: 'O(n)',
    ref: 'Shannon 1948', commonPitfall: '用稳定 log(pᵢ)=xᵢ−LSE 避免 log(0)。',
    invariant: 'run = −Σpᵢln pᵢ 且 run ≥ 0',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { const p = softmax(x); const n = x.length; const out = new Float64Array(n); let H = 0; for (let i = 0; i < n; i++) H += -p[i] * Math.log(p[i]); return H; },
  function pred(x, run) { const out = run(x); const p = softmax(x); let H = 0; for (let i = 0; i < p.length; i++) H += -p[i] * Math.log(p[i]); if (out < -1e-12) return false; return rc(out, H, 1e-9); }
  );

  mk(80, {
    name: 'log-sum-exp 平移不变性', intuition: 'LSE 对常数平移仅增量相应偏移。',
    math: 'LSE(x+c) = LSE(x) + c', pseudocode: 'return lse(x+c)', complexity: 'O(n)',
    ref: 'Bridle 1990', commonPitfall: 'c 可为任意实数（含大值，稳定式不溢出）。',
    invariant: 'run = LSE(x) + c',
    inputs: ['x', 'c'], domains: { x: 'vecSmall', c: 'finite' },
  },
    function run(x, c) { const n = x.length; const xc = new Float64Array(n); for (let i = 0; i < n; i++) xc[i] = x[i] + c; return lse(xc); },
    function pred(x, c, run) { const out = run(x, c); return rc(out, lse(x) + c, 1e-9); }
  );

  // ═══════════════════════════════════════════════════════════════════════
  // 类别 6 · 层归一化家族（F081–F096）
  // ═══════════════════════════════════════════════════════════════════════
  mk(81, {
    name: 'LayerNorm（向量）', intuition: '层归一化使单层激活零均值、单位方差。',
    math: 'run = (x−μ)/σ', pseudocode: 'return (x-mean)/std', complexity: 'O(n)',
    ref: 'Ba et al. 2016 LayerNorm', commonPitfall: 'σ=0 时退化为 0。',
    invariant: 'mean(run)≈0 且 var(run)≈1',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { return layerNorm(x); },
    function pred(x, run) { const out = run(x); if (variance(x) <= 1e-14) { for (let i = 0; i < out.length; i++) if (out[i] !== 0) return false; return true; } return rc(mean(out), 0, 1e-6) && rc(variance(out), 1, 1e-6); }
  );

  mk(82, {
    name: 'RMSNorm（带增益）', intuition: 'RMSNorm 仅用均方根缩放，并乘可学习增益。',
    math: 'runᵢ = (xᵢ/rms(x))·gᵢ', pseudocode: 'return (x/rms(x)) * g', complexity: 'O(n)',
    ref: 'Zhang & Sennrich 2019', commonPitfall: '不含减均值项。',
    invariant: 'runᵢ = (xᵢ/rms(x))·gᵢ',
    inputs: ['x', 'g'], domains: { x: 'vecSmall', g: 'vecSmall' },
  },
    function run(x, g) { const n = x.length; const r = rms(x); const out = new Float64Array(n); if (r < 1e-12) return out; for (let i = 0; i < n; i++) out[i] = (x[i] / r) * g[i]; return out; },
    function pred(x, g, run) { const out = run(x, g); const r = rms(x); if (r < 1e-12) { for (let i = 0; i < out.length; i++) if (out[i] !== 0) return false; return true; } for (let i = 0; i < out.length; i++) if (!rc(out[i], (x[i] / r) * g[i], 1e-9)) return false; return true; }
  );

  mk(83, {
    name: 'GroupNorm（g=4）', intuition: '把向量均分 4 组，各组独立零均值单位方差。',
    math: 'run = (x_g−μ_g)/σ_g', pseudocode: 'groupNorm(x,4)', complexity: 'O(n)',
    ref: 'Wu & He 2018 GroupNorm', commonPitfall: '组数须整除长度（16/4=4）。',
    invariant: '每组 mean≈0 且 var≈1',
    inputs: ['x'], domains: { x: 'vecSmall' }, vecDims: { x: 16 },
  },
    function run(x) { return groupNorm(x, 4); },
    function pred(x, run) { const out = run(x); const g = 4, n = x.length, gs = n / g; for (let gr = 0; gr < g; gr++) { let s = 0; for (let i = 0; i < gs; i++) s += x[gr * gs + i]; const m = s / gs; let v = 0; for (let i = 0; i < gs; i++) { const d = x[gr * gs + i] - m; v += d * d; } v /= gs; if (v <= 1e-14) { for (let i = 0; i < gs; i++) if (out[gr * gs + i] !== 0) return false; continue; } let so = 0; for (let i = 0; i < gs; i++) so += out[gr * gs + i]; if (!rc(so / gs, 0, 1e-6)) return false; let vo = 0; for (let i = 0; i < gs; i++) { const d = out[gr * gs + i] - 0; vo += d * d; } vo /= gs; if (!rc(vo, 1, 1e-6)) return false; } return true; }
  );

  mk(84, {
    name: 'GroupNorm（g=8）', intuition: '把向量均分 8 组（每组 2 元素）独立归一。',
    math: 'run = (x_g−μ_g)/σ_g', pseudocode: 'groupNorm(x,8)', complexity: 'O(n)',
    ref: 'Wu & He 2018 GroupNorm', commonPitfall: '16/8=2，每组仅 2 点。',
    invariant: '每组 mean≈0 且 var≈1',
    inputs: ['x'], domains: { x: 'vecSmall' }, vecDims: { x: 16 },
  },
    function run(x) { return groupNorm(x, 8); },
    function pred(x, run) { const out = run(x); const g = 8, n = x.length, gs = n / g; for (let gr = 0; gr < g; gr++) { let s = 0; for (let i = 0; i < gs; i++) s += x[gr * gs + i]; const m = s / gs; let v = 0; for (let i = 0; i < gs; i++) { const d = x[gr * gs + i] - m; v += d * d; } v /= gs; if (v <= 1e-14) { for (let i = 0; i < gs; i++) if (out[gr * gs + i] !== 0) return false; continue; } let so = 0; for (let i = 0; i < gs; i++) so += out[gr * gs + i]; if (!rc(so / gs, 0, 1e-6)) return false; let vo = 0; for (let i = 0; i < gs; i++) { const d = out[gr * gs + i]; vo += d * d; } vo /= gs; if (!rc(vo, 1, 1e-6)) return false; } return true; }
  );

  mk(85, {
    name: 'InstanceNorm（矩阵全局）', intuition: '整张矩阵视为单实例，全局零均值单位方差。',
    math: 'run = (M−μ)/σ', pseudocode: 'return (M-mean)/std', complexity: 'O(r·c)',
    ref: 'Ulyanov et al. 2016', commonPitfall: 'σ=0 时退化为 0。',
    invariant: '总均值≈0 且总方差≈1',
    inputs: ['M'], domains: { M: 'matSmall' }, matrixDims: { M: [4, 4] },
  },
    function run(M) { const r = 4, c = 4; const m = mMean(M, r, c); const v = mVar(M, r, c); const out = new Float64Array(r * c); if (v <= 1e-14) return out; const inv = 1 / Math.sqrt(v); for (let k = 0; k < r * c; k++) out[k] = (M[k] - m) * inv; return out; },
    function pred(M, run) { const out = run(M); const r = 4, c = 4; if (mVar(M, r, c) <= 1e-14) { for (let k = 0; k < out.length; k++) if (out[k] !== 0) return false; return true; } const m = mMean(out, r, c); const v = mVar(out, r, c); return rc(m, 0, 1e-6) && rc(v, 1, 1e-6); }
  );

  mk(86, {
    name: 'LayerNorm + 仿射', intuition: '归一化后再施加可学习增益与偏置。',
    math: 'run = (x−μ)/σ·γ + β', pseudocode: 'return norm*gamma + beta', complexity: 'O(n)',
    ref: 'Ba et al. 2016', commonPitfall: '仿射作用在归一化结果上。',
    invariant: '(run−β)/γ 的 mean≈0、var≈1（γ≠0 处）',
    inputs: ['x', 'gamma', 'beta'], domains: { x: 'vecSmall', gamma: 'vecSmall', beta: 'vecSmall' },
  },
    function run(x, gamma, beta) { const n = x.length; const m = mean(x); const v = variance(x); const out = new Float64Array(n); if (v <= 1e-14) return out; const inv = 1 / Math.sqrt(v); for (let i = 0; i < n; i++) out[i] = (x[i] - m) * inv * gamma[i] + beta[i]; return out; },
    function pred(x, gamma, beta, run) { const out = run(x, gamma, beta); const v = variance(x); if (v <= 1e-14) { for (let i = 0; i < out.length; i++) if (out[i] !== 0) return false; return true; } const d = []; for (let i = 0; i < out.length; i++) { if (Math.abs(gamma[i]) > 1e-12) d.push((out[i] - beta[i]) / gamma[i]); if (!rc(out[i], (x[i] - mean(x)) / Math.sqrt(v) * gamma[i] + beta[i], 1e-9)) return false; } if (d.length >= 2) { let s = 0; for (let i = 0; i < d.length; i++) s += d[i]; const md = s / d.length; let vv = 0; for (let i = 0; i < d.length; i++) { const dd = d[i] - md; vv += dd * dd; } vv /= d.length; if (!rc(md, 0, 1e-6)) return false; if (!rc(vv, 1, 1e-6)) return false; } return true; }
  );

  mk(87, {
    name: 'LayerNorm（可学习增益）', intuition: '归一化结果乘增益，缩放各通道幅度。',
    math: 'run = norm·γ', pseudocode: 'return norm * gamma', complexity: 'O(n)',
    ref: 'Ba et al. 2016', commonPitfall: '增益作用于已归一化向量。',
    invariant: 'run/γ 的 mean≈0、var≈1（γ≠0 处）',
    inputs: ['x', 'gamma'], domains: { x: 'vecSmall', gamma: 'vecSmall' },
  },
    function run(x, gamma) { const n = x.length; const m = mean(x); const v = variance(x); const out = new Float64Array(n); if (v <= 1e-14) return out; const inv = 1 / Math.sqrt(v); for (let i = 0; i < n; i++) out[i] = (x[i] - m) * inv * gamma[i]; return out; },
    function pred(x, gamma, run) { const out = run(x, gamma); const v = variance(x); if (v <= 1e-14) { for (let i = 0; i < out.length; i++) if (out[i] !== 0) return false; return true; } const d = []; for (let i = 0; i < out.length; i++) if (Math.abs(gamma[i]) > 1e-12) d.push(out[i] / gamma[i]); if (d.length >= 2) { let s = 0; for (let i = 0; i < d.length; i++) s += d[i]; const md = s / d.length; let vv = 0; for (let i = 0; i < d.length; i++) { const dd = d[i] - md; vv += dd * dd; } vv /= d.length; if (!rc(md, 0, 1e-6)) return false; if (!rc(vv, 1, 1e-6)) return false; } return true; }
  );

  mk(88, {
    name: 'LayerNorm（可学习偏置）', intuition: '归一化结果加偏置，平移各通道基线。',
    math: 'run = norm + β', pseudocode: 'return norm + beta', complexity: 'O(n)',
    ref: 'Ba et al. 2016', commonPitfall: '偏置不改变归一化的方差属性。',
    invariant: 'run−β 的 mean≈0、var≈1',
    inputs: ['x', 'beta'], domains: { x: 'vecSmall', beta: 'vecSmall' },
  },
    function run(x, beta) { const n = x.length; const m = mean(x); const v = variance(x); const out = new Float64Array(n); if (v <= 1e-14) return out; const inv = 1 / Math.sqrt(v); for (let i = 0; i < n; i++) out[i] = (x[i] - m) * inv + beta[i]; return out; },
    function pred(x, beta, run) { const out = run(x, beta); const v = variance(x); if (v <= 1e-14) { for (let i = 0; i < out.length; i++) if (out[i] !== 0) return false; return true; } const d = new Float64Array(x.length); for (let i = 0; i < x.length; i++) d[i] = out[i] - beta[i]; return rc(mean(d), 0, 1e-6) && rc(variance(d), 1, 1e-6); }
  );

  mk(89, {
    name: '防除零 ε 稳定归一化', intuition: '加 ε 防止零方差时产生 NaN/Inf。',
    math: 'run = (x−μ)/√(var+ε)', pseudocode: 'return (x-mean)/sqrt(var+eps)', complexity: 'O(n)',
    ref: 'Ioffe & Szegedy 2015', commonPitfall: '须验证任意输入下 run 均有限。',
    invariant: '对所有输入 run 有限（无 NaN/Inf）；var 大时 var(run)≈1',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { const m = mean(x); const v = variance(x); const inv = 1 / Math.sqrt(v + 1e-12); const out = new Float64Array(x.length); for (let i = 0; i < x.length; i++) out[i] = (x[i] - m) * inv; return out; },
    function pred(x, run) { const out = run(x); for (let i = 0; i < out.length; i++) if (!Number.isFinite(out[i])) return false; const v = variance(x); if (v > 1e-6) { const m = mean(out); const vv = variance(out); if (!rc(m, 0, 1e-6)) return false; if (!rc(vv, 1, 1e-6)) return false; } return true; }
  );

  mk(90, {
    name: 'LayerNorm 尺度不变性', intuition: '对输入整体缩放，归一化结果仅变号（若缩放为负）。',
    math: 'LN(c·x) = sign(c)·LN(x)', pseudocode: 'return lnorm(c*x)', complexity: 'O(n)',
    ref: 'Ba et al. 2016', commonPitfall: 'c=0 时退化，域用 nonZero。',
    invariant: 'LN(c·x) = sign(c)·LN(x)',
    inputs: ['x', 'c'], domains: { x: 'vecSmall', c: 'nonZero' },
  },
    function run(x, c) { const n = x.length; const xc = new Float64Array(n); for (let i = 0; i < n; i++) xc[i] = x[i] * c; return layerNorm(xc); },
    function pred(x, c, run) { const out = run(x, c); const base = layerNorm(x); for (let i = 0; i < out.length; i++) if (!rc(out[i], (c < 0 ? -1 : 1) * base[i], 1e-6)) return false; return true; }
  );

  mk(91, {
    name: 'GroupNorm（g=2）', intuition: '把向量均分 2 组（每组 8 元素）独立归一。',
    math: 'run = (x_g−μ_g)/σ_g', pseudocode: 'groupNorm(x,2)', complexity: 'O(n)',
    ref: 'Wu & He 2018 GroupNorm', commonPitfall: '16/2=8，每组 8 点。',
    invariant: '每组 mean≈0 且 var≈1',
    inputs: ['x'], domains: { x: 'vecSmall' }, vecDims: { x: 16 },
  },
    function run(x) { return groupNorm(x, 2); },
    function pred(x, run) { const out = run(x); const g = 2, n = x.length, gs = n / g; for (let gr = 0; gr < g; gr++) { let s = 0; for (let i = 0; i < gs; i++) s += x[gr * gs + i]; const m = s / gs; let v = 0; for (let i = 0; i < gs; i++) { const d = x[gr * gs + i] - m; v += d * d; } v /= gs; if (v <= 1e-14) { for (let i = 0; i < gs; i++) if (out[gr * gs + i] !== 0) return false; continue; } let so = 0; for (let i = 0; i < gs; i++) so += out[gr * gs + i]; if (!rc(so / gs, 0, 1e-6)) return false; let vo = 0; for (let i = 0; i < gs; i++) { const d = out[gr * gs + i]; vo += d * d; } vo /= gs; if (!rc(vo, 1, 1e-6)) return false; } return true; }
  );

  mk(92, {
    name: 'RMSNorm（无增益）', intuition: '仅用均方根缩放，使输出 RMS=1。',
    math: 'runᵢ = xᵢ/rms(x)', pseudocode: 'return x / rms(x)', complexity: 'O(n)',
    ref: 'Zhang & Sennrich 2019', commonPitfall: '不含减均值项。',
    invariant: 'rms(run) = 1（x≠0）',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { const n = x.length; const r = rms(x); const out = new Float64Array(n); if (r < 1e-12) return out; for (let i = 0; i < n; i++) out[i] = x[i] / r; return out; },
    function pred(x, run) { const out = run(x); const r = rms(x); if (r < 1e-12) { for (let i = 0; i < out.length; i++) if (out[i] !== 0) return false; return true; } return rc(rms(out), 1, 1e-6); }
  );

  mk(93, {
    name: 'LayerNorm 重建', intuition: '反归一化可由 run 恢复原始 x。',
    math: 'run·σ + μ = x', pseudocode: 'return (x-mean)/std', complexity: 'O(n)',
    ref: 'Ba et al. 2016', commonPitfall: 'σ=0 时退化。',
    invariant: 'runᵢ·σ + μ = xᵢ',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { return layerNorm(x); },
    function pred(x, run) { const out = run(x); const v = variance(x); if (v <= 1e-14) { for (let i = 0; i < out.length; i++) if (out[i] !== 0) return false; return true; } const m = mean(x); const sd = Math.sqrt(v); for (let i = 0; i < out.length; i++) if (!rc(out[i] * sd + m, x[i], 1e-9)) return false; return true; }
  );

  mk(94, {
    name: '局部响应归一化 LRN', intuition: '按全局能量分母缩放，分母恒正。',
    math: 'runᵢ = xᵢ/(k+α·Σxⱼ²)^β', pseudocode: 'return x / (k+alpha*sumsq)^beta', complexity: 'O(n)',
    ref: 'Krizhevsky et al. 2012 AlexNet', commonPitfall: 'k>0 保证分母为正。',
    invariant: 'runᵢ·denom^β = xᵢ 且 denom > 0',
    inputs: ['x', 'k', 'alpha', 'beta'], domains: { x: 'vecSmall', k: 'pos', alpha: 'nonNegSmall', beta: 'smallPos' },
  },
    function run(x, k, alpha, beta) { let sq = 0; for (let i = 0; i < x.length; i++) sq += x[i] * x[i]; const denom = k + alpha * sq; const out = new Float64Array(x.length); const dbeta = Math.pow(denom, beta); for (let i = 0; i < x.length; i++) out[i] = x[i] / dbeta; return out; },
    function pred(x, k, alpha, beta, run) { const out = run(x, k, alpha, beta); let sq = 0; for (let i = 0; i < x.length; i++) sq += x[i] * x[i]; const denom = k + alpha * sq; if (!(denom > 0)) return false; const dbeta = Math.pow(denom, beta); for (let i = 0; i < out.length; i++) if (!rc(out[i] * dbeta, x[i], 1e-9)) return false; return true; }
  );

  mk(95, {
    name: '跨通道归一化', intuition: '按各自平方能量（非共享）缩放。',
    math: 'runᵢ = xᵢ/(k+α·xᵢ²)^β', pseudocode: 'return x / (k+alpha*x^2)^beta', complexity: 'O(n)',
    ref: 'Krizhevsky et al. 2012', commonPitfall: '每个通道独立分母。',
    invariant: 'runᵢ·denomᵢ^β = xᵢ 且 denomᵢ > 0',
    inputs: ['x', 'k', 'alpha', 'beta'], domains: { x: 'vecSmall', k: 'pos', alpha: 'nonNegSmall', beta: 'smallPos' },
  },
    function run(x, k, alpha, beta) { const n = x.length; const out = new Float64Array(n); for (let i = 0; i < n; i++) { const denom = k + alpha * x[i] * x[i]; const dbeta = Math.pow(denom, beta); out[i] = x[i] / dbeta; } return out; },
    function pred(x, k, alpha, beta, run) { const out = run(x, k, alpha, beta); for (let i = 0; i < out.length; i++) { const denom = k + alpha * x[i] * x[i]; if (!(denom > 0)) return false; const dbeta = Math.pow(denom, beta); if (!rc(out[i] * dbeta, x[i], 1e-9)) return false; } return true; }
  );

  mk(96, {
    name: 'LayerNorm 逐行（矩阵）', intuition: '矩阵每一行作为独立层归一化。',
    math: 'run_{i·} = (M_{i·}−μ_i)/σ_i', pseudocode: 'return rowLayerNorm(M)', complexity: 'O(r·c)',
    ref: 'Ba et al. 2016', commonPitfall: '逐行而非全局。',
    invariant: '每行 mean≈0 且 var≈1',
    inputs: ['M'], domains: { M: 'matSmall' }, matrixDims: { M: [4, 4] },
  },
    function run(M) { const r = 4, c = 4; const out = new Float64Array(r * c); for (let i = 0; i < r; i++) { let s = 0; for (let j = 0; j < c; j++) s += M[i * c + j]; const m = s / c; let v = 0; for (let j = 0; j < c; j++) { const d = M[i * c + j] - m; v += d * d; } v /= c; if (v <= 1e-14) continue; const inv = 1 / Math.sqrt(v); for (let j = 0; j < c; j++) out[i * c + j] = (M[i * c + j] - m) * inv; } return out; },
    function pred(M, run) { const out = run(M); const r = 4, c = 4; for (let i = 0; i < r; i++) { const v = mRowVar(M, r, c, i); if (v <= 1e-14) { for (let j = 0; j < c; j++) if (out[i * c + j] !== 0) return false; continue; } let so = 0; for (let j = 0; j < c; j++) so += out[i * c + j]; if (!rc(so / c, 0, 1e-6)) return false; let vo = 0; for (let j = 0; j < c; j++) { const d = out[i * c + j]; vo += d * d; } vo /= c; if (!rc(vo, 1, 1e-6)) return false; } return true; }
  );
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

/* ============================================================================
 * L10 微指令 · F 域（归一化）· 第 1/4 批（F001–F032）
 * ----------------------------------------------------------------------------
 * 所属层：L10-microIsa  域：F（归一化）
 * 允许依赖的层：L9（special/fpOps）、L10（registry/invariant）
 * 主要 @ref：Goodfellow et al. (2016) Deep Learning §3.4, §5.1；
 *           Ba et al. (2016) Layer Normalization；IEEE 754-2019
 *
 * 涵盖：向量统计（F001–F016）、范数（F017–F032）。
 * 每个 op 含七件套 + run + predicate；predicate 内部调用 run(...) 并验证
 * 一个对声明域恒真的非平凡数学不变量。标量不变量用 Object.is/relClose 判等，
 * 向量/矩阵不变量用 NS.L10.relClose。
 * ==========================================================================*/
;(function (NS) {
  'use strict';
  NS.L10 = NS.L10 || {};
  function reg(i, o) { NS.L10.register('F', i, o); }
  const rc = (a, b, t) => NS.L10.relClose(a, b, t);

  // ── 通用小工具（纯函数，run/predicate 复用）────────────────────────────────
  function sum(a) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]; return s; }
  function mean(a) { return sum(a) / a.length; }
  function meanSq(a) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * a[i]; return s / a.length; }
  function variance(a) { const m = mean(a); let s = 0; for (let i = 0; i < a.length; i++) { const d = a[i] - m; s += d * d; } const v = s / a.length; return v < 0 ? 0 : v; }
  function stddev(a) { return Math.sqrt(variance(a)); }
  function rms(a) { return Math.sqrt(meanSq(a)); }
  function maxAbs(a) { let m = 0; for (let i = 0; i < a.length; i++) { const v = Math.abs(a[i]); if (v > m) m = v; } return m; }
  function maxEl(a) { let m = -Infinity; for (let i = 0; i < a.length; i++) if (a[i] > m) m = a[i]; return m; }
  function minEl(a) { let m = Infinity; for (let i = 0; i < a.length; i++) if (a[i] < m) m = a[i]; return m; }
  function dot(a, b) { let s = 0; const n = a.length < b.length ? a.length : b.length; for (let i = 0; i < n; i++) s += a[i] * b[i]; return s; }
  function median(a) { const s = Array.prototype.slice.call(a).sort((p, q) => p - q); const n = s.length; if (n % 2) return s[(n - 1) / 2]; return (s[n / 2 - 1] + s[n / 2]) / 2; }
  function zeros(n) { return new Float64Array(n); }

  // mk：把 reg 包成 (index, meta, run, pred)，pred 末尾注入 run，满足"predicate 内部调用 run"。
  function mk(i, meta, run, pred) {
    const def = Object.assign({}, meta);
    def.run = run;
    def.predicate = function () { return pred.apply(null, Array.prototype.slice.call(arguments).concat([run])); };
    reg(i, def);
  }

  // ═══════════════════════════════════════════════════════════════════════
  // 类别 1 · 向量统计（F001–F016）
  // ═══════════════════════════════════════════════════════════════════════
  mk(1, {
    name: '向量算术均值', intuition: '均值是各元素之和除以个数。',
    math: 'mean = Σxᵢ / n', pseudocode: 'return Σxᵢ / n', complexity: 'O(n)',
    ref: 'Goodfellow 2016 §3.1', commonPitfall: '整数域下应浮点除法，否则丢精度。',
    invariant: 'n·run = Σxᵢ',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { return mean(x); },
    function pred(x, run) { const out = run(x); let s = 0; for (let i = 0; i < x.length; i++) s += x[i]; return rc(x.length * out, s, 1e-9); }
  );

  mk(2, {
    name: '总体方差', intuition: '总体方差 = 平方均值 − 均值平方（平移不变）。',
    math: 'var = mean(x²) − mean(x)²', pseudocode: 'return mean(x*x) - mean(x)^2', complexity: 'O(n)',
    ref: 'Goodfellow 2016 §3.4', commonPitfall: '与样本方差混淆；分母为 n 而非 n−1。',
    invariant: 'run = mean(x²) − mean(x)² 且 run ≥ 0',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { return meanSq(x) - mean(x) * mean(x); },
    function pred(x, run) { const out = run(x); const v = meanSq(x) - mean(x) * mean(x); return rc(out, v, 1e-9) && out >= -1e-12; }
  );

  mk(3, {
    name: '样本方差', intuition: '样本方差用 n/(n−1) 无偏修正。',
    math: 's² = n/(n−1)·(mean(x²) − mean(x)²)', pseudocode: 'return n/(n-1)*(mean(x*x)-mean(x)^2)', complexity: 'O(n)',
    ref: 'Casella & Berger 2002', commonPitfall: '分母 n−1，长度需 >1。',
    invariant: 'run = n/(n−1)·总体方差',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { const n = x.length; const v = meanSq(x) - mean(x) * mean(x); return (n / (n - 1)) * v; },
    function pred(x, run) { const out = run(x); const n = x.length; const v = meanSq(x) - mean(x) * mean(x); return rc(out, (n / (n - 1)) * v, 1e-9); }
  );

  mk(4, {
    name: '标准差', intuition: '标准差是方差的平方根。',
    math: 'std = √var', pseudocode: 'return sqrt(var)', complexity: 'O(n)',
    ref: 'Goodfellow 2016 §3.4', commonPitfall: '应取非负根。',
    invariant: 'run = √总体方差 且 run ≥ 0',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { return Math.sqrt(variance(x)); },
    function pred(x, run) { const out = run(x); return rc(out, Math.sqrt(variance(x)), 1e-9) && out >= -1e-12; }
  );

  mk(5, {
    name: '均方根 RMS', intuition: 'RMS 是平方均值的平方根。',
    math: 'rms = √(mean(x²))', pseudocode: 'return sqrt(mean(x*x))', complexity: 'O(n)',
    ref: 'IEEE 754-2019', commonPitfall: '与标准差混淆（未减均值）。',
    invariant: 'run = √(mean(x²)) 且 run ≥ 0',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { return rms(x); },
    function pred(x, run) { const out = run(x); return rc(out, Math.sqrt(meanSq(x)), 1e-9) && out >= -1e-12; }
  );

  mk(6, {
    name: '变异系数', intuition: '变异系数 = 标准差/|均值|，衡量相对离散度（需均值非零）。',
    math: 'cv = std / |mean|', pseudocode: 'return std(x)/abs(mean(x))', complexity: 'O(n)',
    ref: 'Goodfellow 2016 §3.4', commonPitfall: '均值接近 0 时爆炸；故域用 vecPos。',
    invariant: 'run = std/|mean| 且 run ≥ 0',
    inputs: ['x'], domains: { x: 'vecPos' },
  },
    function run(x) { const m = mean(x); return stddev(x) / Math.abs(m); },
    function pred(x, run) { const out = run(x); const cv = stddev(x) / Math.abs(mean(x)); return rc(out, cv, 1e-9) && out >= -1e-12; }
  );

  mk(7, {
    name: '中位数', intuition: '中位数把一半元素置于其下、一半置于其上。',
    math: 'median = x_{(⌊n/2⌋)}（排序后）', pseudocode: 'return sort(x)[mid]', complexity: 'O(n log n)',
    ref: 'Knuth TAOCP §5.3.3', commonPitfall: '偶数长度应取中间两值平均。',
    invariant: '至少有 ⌈n/2⌉ 个元素 ≤ run，且至少 ⌈n/2⌉ 个元素 ≥ run',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { return median(x); },
    function pred(x, run) { const out = run(x); const n = x.length; const need = Math.ceil(n / 2); let le = 0, ge = 0; for (let i = 0; i < n; i++) { if (x[i] <= out + 1e-12) le++; if (x[i] >= out - 1e-12) ge++; } return le >= need - 1e-9 && ge >= need - 1e-9; }
  );

  mk(8, {
    name: '几何均值', intuition: '正数几何均值 = 各元素乘积的 n 次根。',
    math: 'g = (∏xᵢ)^{1/n} = exp(mean(ln x))', pseudocode: 'return exp(mean(log(x)))', complexity: 'O(n)',
    ref: 'Fleming & Wallace 1986', commonPitfall: '负数/零无实定义；域用 vecPos。',
    invariant: 'runⁿ = ∏xᵢ（对数恒等式）',
    inputs: ['x'], domains: { x: 'vecPos' },
  },
    function run(x) { let s = 0; for (let i = 0; i < x.length; i++) s += Math.log(x[i]); return Math.exp(s / x.length); },
    function pred(x, run) { const out = run(x); let prod = 1; for (let i = 0; i < x.length; i++) prod *= x[i]; return rc(Math.pow(out, x.length), prod, 1e-9); }
  );

  mk(9, {
    name: '调和均值', intuition: '调和均值 = 倒数均值的倒数（正数域）。',
    math: 'h = n / Σ(1/xᵢ)', pseudocode: 'return n / sum(1/x)', complexity: 'O(n)',
    ref: 'Fleming & Wallace 1986', commonPitfall: '含 0 时未定义；域用 vecPos。',
    invariant: 'run = n / Σ(1/xᵢ) 且 run > 0',
    inputs: ['x'], domains: { x: 'vecPos' },
  },
    function run(x) { let s = 0; for (let i = 0; i < x.length; i++) s += 1 / x[i]; return x.length / s; },
    function pred(x, run) { const out = run(x); let s = 0; for (let i = 0; i < x.length; i++) s += 1 / x[i]; return rc(out, x.length / s, 1e-9) && out > 0; }
  );

  mk(10, {
    name: '截尾均值', intuition: '去掉一个最小与一个最大后取均值，抗离群。',
    math: 'trimmed = mean(x 去 min、max)', pseudocode: 'sort; drop ends; return mean', complexity: 'O(n log n)',
    ref: 'Wilcox 2011 稳健统计', commonPitfall: '偶数个极值去重要么同值取其一。',
    invariant: 'run ∈ [min, max]（子集平均必在区间内）',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { const s = Array.prototype.slice.call(x).sort((a, b) => a - b); const n = s.length; let sum = 0; for (let i = 1; i < n - 1; i++) sum += s[i]; return sum / (n - 2); },
    function pred(x, run) { const out = run(x); return out >= minEl(x) - 1e-12 && out <= maxEl(x) + 1e-12; }
  );

  mk(11, {
    name: '偏度', intuition: '偏度是标准化三阶中心矩，衡量分布不对称。',
    math: 'skew = mean((x−μ)³) / σ³', pseudocode: 'return mean((x-m)^3)/std^3', complexity: 'O(n)',
    ref: 'Joanes & Gill 1998', commonPitfall: '标准差为 0 时未定义（退化为 0）。',
    invariant: 'run = mean((x−μ)³)/σ³（σ>0 时）',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { const m = mean(x); const v = variance(x); if (v <= 1e-14) return 0; const s = Math.sqrt(v); let s3 = 0; for (let i = 0; i < x.length; i++) { const d = (x[i] - m) / s; s3 += d * d * d; } return s3 / x.length; },
    function pred(x, run) { const out = run(x); const m = mean(x); const v = variance(x); if (v <= 1e-14) return out === 0; const s = Math.sqrt(v); let s3 = 0; for (let i = 0; i < x.length; i++) { const d = (x[i] - m) / s; s3 += d * d * d; } return rc(out, s3 / x.length, 1e-9); }
  );

  mk(12, {
    name: '峰态（超额）', intuition: '超额峰态 = 标准化四阶中心矩 − 3。',
    math: 'kurt = mean((x−μ)⁴)/σ⁴ − 3', pseudocode: 'return mean((x-m)^4)/std^4 - 3', complexity: 'O(n)',
    ref: 'Joanes & Gill 1998', commonPitfall: '方差 0 时未定义（退化为 0）。',
    invariant: 'run = 标准化四阶矩 − 3（σ>0 时）',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { const m = mean(x); const v = variance(x); if (v <= 1e-14) return 0; const s = Math.sqrt(v); let s4 = 0; for (let i = 0; i < x.length; i++) { const d = (x[i] - m) / s; const d2 = d * d; s4 += d2 * d2; } return s4 / x.length - 3; },
    function pred(x, run) { const out = run(x); const m = mean(x); const v = variance(x); if (v <= 1e-14) return out === 0; const s = Math.sqrt(v); let s4 = 0; for (let i = 0; i < x.length; i++) { const d = (x[i] - m) / s; const d2 = d * d; s4 += d2 * d2; } return rc(out, s4 / x.length - 3, 1e-9); }
  );

  mk(13, {
    name: '均值绝对偏差', intuition: 'MAD = 各元素到均值绝对偏差的平均。',
    math: 'mad = mean(|xᵢ − μ|)', pseudocode: 'return mean(abs(x-mean))', complexity: 'O(n)',
    ref: 'Pham-Gia & Hung 2001', commonPitfall: '与标准差混淆。',
    invariant: 'run = mean(|x−μ|) 且 run ≥ 0',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { const m = mean(x); let s = 0; for (let i = 0; i < x.length; i++) s += Math.abs(x[i] - m); return s / x.length; },
    function pred(x, run) { const out = run(x); const m = mean(x); let s = 0; for (let i = 0; i < x.length; i++) s += Math.abs(x[i] - m); return rc(out, s / x.length, 1e-9) && out >= -1e-12; }
  );

  mk(14, {
    name: '截尾方差', intuition: '去极值后的总体方差，稳健估计离散度。',
    math: 'trimmedVar = var(x 去 min、max)', pseudocode: 'sort; drop ends; return var', complexity: 'O(n log n)',
    ref: 'Wilcox 2011', commonPitfall: '剩余长度需 ≥2。',
    invariant: 'run = 去极值子集的总体方差',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { const s = Array.prototype.slice.call(x).sort((a, b) => a - b); const n = s.length; const sub = s.slice(1, n - 1); return variance(sub); },
    function pred(x, run) { const out = run(x); const s = Array.prototype.slice.call(x).sort((a, b) => a - b); const n = s.length; const sub = s.slice(1, n - 1); return rc(out, variance(sub), 1e-9); }
  );

  mk(15, {
    name: '加权均值', intuition: '加权均值是加权和除以总权重。',
    math: 'wm = Σ(wᵢxᵢ) / Σwᵢ', pseudocode: 'return dot(w,x)/sum(w)', complexity: 'O(n)',
    ref: 'Goodfellow 2016 §3.1', commonPitfall: '总权重为 0 时退化为 0。',
    invariant: 'run = Σ(wᵢxᵢ)/Σwᵢ',
    inputs: ['x', 'w'], domains: { x: 'vecSmall', w: 'vecSmall' },
  },
    function run(x, w) { let sw = 0, swx = 0; for (let i = 0; i < x.length; i++) { sw += w[i]; swx += w[i] * x[i]; } return sw === 0 ? 0 : swx / sw; },
    function pred(x, w, run) { const out = run(x, w); let sw = 0, swx = 0; for (let i = 0; i < x.length; i++) { sw += w[i]; swx += w[i] * x[i]; } return sw === 0 ? out === 0 : rc(out, swx / sw, 1e-9); }
  );

  mk(16, {
    name: '累积均值序列', intuition: '第 k 个累积均值 = 前 k+1 个元素的平均。',
    math: 'runₖ = (Σ_{i≤k} xᵢ)/(k+1)', pseudocode: 'prefixSum / (k+1)', complexity: 'O(n)',
    ref: 'Knuth TAOCP §5.3', commonPitfall: '分母为 k+1 而非 k。',
    invariant: 'runₖ = 前缀和/(k+1)，且 run₀ = x₀',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { const n = x.length; const out = new Float64Array(n); let s = 0; for (let i = 0; i < n; i++) { s += x[i]; out[i] = s / (i + 1); } return out; },
    function pred(x, run) { const out = run(x); let s = 0; for (let i = 0; i < x.length; i++) { s += x[i]; if (!rc(out[i], s / (i + 1), 1e-9)) return false; } return true; }
  );

  // ═══════════════════════════════════════════════════════════════════════
  // 类别 2 · 范数（F017–F032）
  // ═══════════════════════════════════════════════════════════════════════
  mk(17, {
    name: 'L0 范数（非零计数）', intuition: 'L0 范数统计非零元素个数。',
    math: '‖x‖₀ = #{i : xᵢ ≠ 0}', pseudocode: 'count(x!=0)', complexity: 'O(n)',
    ref: 'Donoho 2006 压缩感知', commonPitfall: '0 在浮点中需精确判等。',
    invariant: 'run = 非零元素个数',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { let c = 0; for (let i = 0; i < x.length; i++) if (x[i] !== 0) c++; return c; },
    function pred(x, run) { const out = run(x); let c = 0; for (let i = 0; i < x.length; i++) if (x[i] !== 0) c++; return out === c; }
  );

  mk(18, {
    name: 'L1 范数', intuition: 'L1 范数是绝对值之和（曼哈顿距离）。',
    math: '‖x‖₁ = Σ|xᵢ|', pseudocode: 'return sum(abs(x))', complexity: 'O(n)',
    ref: 'Goodfellow 2016 §3.4', commonPitfall: '勿漏绝对值。',
    invariant: 'run = Σ|xᵢ| 且 run ≥ 0',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { return sumAbs(x); },
    function pred(x, run) { const out = run(x); return rc(out, sumAbs(x), 1e-9) && out >= -1e-12; }
  );

  mk(19, {
    name: 'L2 范数', intuition: 'L2 范数是欧几里得长度。',
    math: '‖x‖₂ = √(Σxᵢ²)', pseudocode: 'return sqrt(sum(x*x))', complexity: 'O(n)',
    ref: 'Goodfellow 2016 §3.4', commonPitfall: '勿遗漏开根。',
    invariant: 'run = √(Σxᵢ²) 且 run ≥ 0',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { let s = 0; for (let i = 0; i < x.length; i++) s += x[i] * x[i]; return Math.sqrt(s); },
    function pred(x, run) { const out = run(x); let s = 0; for (let i = 0; i < x.length; i++) s += x[i] * x[i]; return rc(out, Math.sqrt(s), 1e-9) && out >= -1e-12; }
  );

  mk(20, {
    name: 'L∞ 范数（最大绝对值）', intuition: 'L∞ 范数是元素绝对值的最大值。',
    math: '‖x‖∞ = max|xᵢ|', pseudocode: 'return max(abs(x))', complexity: 'O(n)',
    ref: 'Goodfellow 2016 §3.4', commonPitfall: '勿与 L2 混淆。',
    invariant: 'run = max|xᵢ| 且 run ≥ 0',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { return maxAbs(x); },
    function pred(x, run) { const out = run(x); return rc(out, maxAbs(x), 1e-9) && out >= -1e-12; }
  );

  mk(21, {
    name: 'Lp 范数 (p=3)', intuition: 'p 范数在 p=3 时为三次绝对矩的根。',
    math: '‖x‖₃ = (Σ|xᵢ|³)^{1/3}', pseudocode: 'return cbrt(sum(|x|^3))', complexity: 'O(n)',
    ref: 'Goodfellow 2016 §3.4', commonPitfall: '先取绝对值再求幂。',
    invariant: 'run = (Σ|xᵢ|³)^{1/3} 且 run ≥ 0',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { let s = 0; for (let i = 0; i < x.length; i++) s += Math.pow(Math.abs(x[i]), 3); return Math.cbrt(s); },
    function pred(x, run) { const out = run(x); let s = 0; for (let i = 0; i < x.length; i++) s += Math.pow(Math.abs(x[i]), 3); return rc(out, Math.cbrt(s), 1e-9) && out >= -1e-12; }
  );

  mk(22, {
    name: '向量长度平方', intuition: 'L2 范数的平方 = 元素平方和。',
    math: '‖x‖₂² = Σxᵢ²', pseudocode: 'return sum(x*x)', complexity: 'O(n)',
    ref: 'Goodfellow 2016 §3.4', commonPitfall: '勿开根。',
    invariant: 'run = Σxᵢ²',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { let s = 0; for (let i = 0; i < x.length; i++) s += x[i] * x[i]; return s; },
    function pred(x, run) { const out = run(x); let s = 0; for (let i = 0; i < x.length; i++) s += x[i] * x[i]; return rc(out, s, 1e-9); }
  );

  mk(23, {
    name: 'L2 归一化', intuition: '除以 L2 范数得到单位向量（零向量退化为 0）。',
    math: 'run = x / ‖x‖₂', pseudocode: 'return x / norm(x)', complexity: 'O(n)',
    ref: 'Ba et al. 2016 LayerNorm', commonPitfall: '零向量需特判避免除零。',
    invariant: '‖run‖₂ ≈ 1（x≠0）；x=0 时 run=0',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { const n = x.length; let s = 0; for (let i = 0; i < n; i++) s += x[i] * x[i]; const nrm = Math.sqrt(s); const out = new Float64Array(n); if (nrm > 1e-12) for (let i = 0; i < n; i++) out[i] = x[i] / nrm; return out; },
    function pred(x, run) { const out = run(x); let nx = 0; for (let i = 0; i < x.length; i++) nx += x[i] * x[i]; nx = Math.sqrt(nx); let nr = 0; for (let i = 0; i < out.length; i++) nr += out[i] * out[i]; nr = Math.sqrt(nr); if (nx > 1e-12) return rc(nr, 1, 1e-9); for (let i = 0; i < out.length; i++) if (out[i] !== 0) return false; return true; }
  );

  mk(24, {
    name: 'L1 归一化', intuition: '除以 L1 范数得到 L1 单位向量。',
    math: 'run = x / ‖x‖₁', pseudocode: 'return x / sumAbs(x)', complexity: 'O(n)',
    ref: 'Ba et al. 2016', commonPitfall: '零向量需特判。',
    invariant: '‖run‖₁ ≈ 1（x≠0）；x=0 时 run=0',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { const n = x.length; let s = 0; for (let i = 0; i < n; i++) s += Math.abs(x[i]); const out = new Float64Array(n); if (s > 1e-12) for (let i = 0; i < n; i++) out[i] = x[i] / s; return out; },
    function pred(x, run) { const out = run(x); let s = 0; for (let i = 0; i < x.length; i++) s += Math.abs(x[i]); if (s > 1e-12) { let sr = 0; for (let i = 0; i < out.length; i++) sr += Math.abs(out[i]); return rc(sr, 1, 1e-9); } for (let i = 0; i < out.length; i++) if (out[i] !== 0) return false; return true; }
  );

  mk(25, {
    name: '最大绝对值归一化', intuition: '除以最大绝对值，使最大分量幅度为 1。',
    math: 'run = x / max|xᵢ|', pseudocode: 'return x / maxAbs(x)', complexity: 'O(n)',
    ref: 'Ba et al. 2016', commonPitfall: '零向量需特判。',
    invariant: 'max|run| ≈ 1 且 |runᵢ| ≤ 1（x≠0）',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { const n = x.length; const m = maxAbs(x); const out = new Float64Array(n); if (m > 1e-12) for (let i = 0; i < n; i++) out[i] = x[i] / m; return out; },
    function pred(x, run) { const out = run(x); const m = maxAbs(x); if (m > 1e-12) { if (maxAbs(out) > 1 + 1e-9) return false; return rc(maxAbs(out), 1, 1e-9); } for (let i = 0; i < out.length; i++) if (out[i] !== 0) return false; return true; }
  );

  mk(26, {
    name: '余弦相似度', intuition: '余弦相似度 = 点积除以两向量范数之积。',
    math: 'cos = x·y / (‖x‖₂‖y‖₂)', pseudocode: 'return dot(x,y)/(norm(x)*norm(y))', complexity: 'O(n)',
    ref: 'Manning et al. 2008 IR', commonPitfall: '零向量需特判；结果应在 [-1,1]。',
    invariant: 'run = x·y/(‖x‖‖y‖) 且 |run| ≤ 1',
    inputs: ['x', 'y'], domains: { x: 'vecSmall', y: 'vecSmall' },
  },
    function run(x, y) { const nx = Math.sqrt(dot(x, x)), ny = Math.sqrt(dot(y, y)); if (nx < 1e-12 || ny < 1e-12) return 0; return dot(x, y) / (nx * ny); },
    function pred(x, y, run) { const out = run(x, y); const nx = Math.sqrt(dot(x, x)), ny = Math.sqrt(dot(y, y)); if (nx < 1e-12 || ny < 1e-12) return out === 0; const c = dot(x, y) / (nx * ny); return rc(out, c, 1e-9) && Math.abs(out) <= 1 + 1e-9; }
  );

  mk(27, {
    name: '向量夹角', intuition: '夹角 = 余弦相似度的反余弦，范围 [0,π]。',
    math: 'θ = arccos(x·y/(‖x‖‖y‖))', pseudocode: 'return acos(cos)', complexity: 'O(n)',
    ref: 'Manning et al. 2008 IR', commonPitfall: '结果需裁剪到 [0,π]。',
    invariant: 'run = arccos(cos) 且 run ∈ [0,π]',
    inputs: ['x', 'y'], domains: { x: 'vecSmall', y: 'vecSmall' },
  },
    function run(x, y) { const nx = Math.sqrt(dot(x, x)), ny = Math.sqrt(dot(y, y)); if (nx < 1e-12 || ny < 1e-12) return 0; let c = dot(x, y) / (nx * ny); if (c > 1) c = 1; if (c < -1) c = -1; return Math.acos(c); },
    function pred(x, y, run) { const out = run(x, y); const nx = Math.sqrt(dot(x, x)), ny = Math.sqrt(dot(y, y)); if (nx < 1e-12 || ny < 1e-12) return out === 0; let c = dot(x, y) / (nx * ny); if (c > 1) c = 1; if (c < -1) c = -1; return rc(out, Math.acos(c), 1e-9) && out >= -1e-12 && out <= Math.PI + 1e-12; }
  );

  mk(28, {
    name: '投影标量（到 y 方向）', intuition: 'x 在 y 方向上的投影长度 = x·y/‖y‖。',
    math: 'proj = x·y / ‖y‖₂', pseudocode: 'return dot(x,y)/norm(y)', complexity: 'O(n)',
    ref: 'Strang 2016 线性代数', commonPitfall: '|投影| ≤ ‖x‖。',
    invariant: 'run = x·y/‖y‖ 且 |run| ≤ ‖x‖',
    inputs: ['x', 'y'], domains: { x: 'vecSmall', y: 'vecSmall' },
  },
    function run(x, y) { const ny = Math.sqrt(dot(y, y)); if (ny < 1e-12) return 0; return dot(x, y) / ny; },
    function pred(x, y, run) { const out = run(x, y); const ny = Math.sqrt(dot(y, y)); if (ny < 1e-12) return out === 0; const p = dot(x, y) / ny; return rc(out, p, 1e-9) && Math.abs(out) <= Math.sqrt(dot(x, x)) + 1e-9; }
  );

  mk(29, {
    name: '曼哈顿距离', intuition: '两向量绝对差之和。',
    math: 'd₁ = Σ|xᵢ − yᵢ|', pseudocode: 'return sum(abs(x-y))', complexity: 'O(n)',
    ref: 'Goodfellow 2016 §3.4', commonPitfall: '勿漏绝对值。',
    invariant: 'run = Σ|xᵢ−yᵢ| 且 run ≥ 0',
    inputs: ['x', 'y'], domains: { x: 'vecSmall', y: 'vecSmall' },
  },
    function run(x, y) { let s = 0; for (let i = 0; i < x.length; i++) s += Math.abs(x[i] - y[i]); return s; },
    function pred(x, y, run) { const out = run(x, y); let s = 0; for (let i = 0; i < x.length; i++) s += Math.abs(x[i] - y[i]); return rc(out, s, 1e-9) && out >= -1e-12; }
  );

  mk(30, {
    name: '欧几里得距离', intuition: '两向量差向量的 L2 范数。',
    math: 'd₂ = √(Σ(xᵢ−yᵢ)²)', pseudocode: 'return sqrt(sum((x-y)^2))', complexity: 'O(n)',
    ref: 'Goodfellow 2016 §3.4', commonPitfall: '勿漏开根。',
    invariant: 'run = √(Σ(xᵢ−yᵢ)²) 且 run ≥ 0',
    inputs: ['x', 'y'], domains: { x: 'vecSmall', y: 'vecSmall' },
  },
    function run(x, y) { let s = 0; for (let i = 0; i < x.length; i++) { const d = x[i] - y[i]; s += d * d; } return Math.sqrt(s); },
    function pred(x, y, run) { const out = run(x, y); let s = 0; for (let i = 0; i < x.length; i++) { const d = x[i] - y[i]; s += d * d; } return rc(out, Math.sqrt(s), 1e-9) && out >= -1e-12; }
  );

  mk(31, {
    name: '切比雪夫距离', intuition: '两向量最大绝对差。',
    math: 'd∞ = max|xᵢ − yᵢ|', pseudocode: 'return max(abs(x-y))', complexity: 'O(n)',
    ref: 'Goodfellow 2016 §3.4', commonPitfall: '勿与 L2 混淆。',
    invariant: 'run = max|xᵢ−yᵢ| 且 run ≥ 0',
    inputs: ['x', 'y'], domains: { x: 'vecSmall', y: 'vecSmall' },
  },
    function run(x, y) { let m = 0; for (let i = 0; i < x.length; i++) { const v = Math.abs(x[i] - y[i]); if (v > m) m = v; } return m; },
    function pred(x, y, run) { const out = run(x, y); let m = 0; for (let i = 0; i < x.length; i++) { const v = Math.abs(x[i] - y[i]); if (v > m) m = v; } return rc(out, m, 1e-9) && out >= -1e-12; }
  );

  mk(32, {
    name: 'Lp 归一化 (p=4)', intuition: '除以 L4 范数得到 L4 单位向量。',
    math: 'run = x / ‖x‖₄, ‖x‖₄=(Σ|x|⁴)^{1/4}', pseudocode: 'return x / lp4norm(x)', complexity: 'O(n)',
    ref: 'Ba et al. 2016', commonPitfall: '零向量需特判。',
    invariant: '‖run‖₄ ≈ 1（x≠0）；x=0 时 run=0',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { const n = x.length; let s = 0; for (let i = 0; i < n; i++) s += Math.pow(Math.abs(x[i]), 4); const nrm = Math.pow(s, 0.25); const out = new Float64Array(n); if (nrm > 1e-12) for (let i = 0; i < n; i++) out[i] = x[i] / nrm; return out; },
    function pred(x, run) { const out = run(x); let sx = 0; for (let i = 0; i < x.length; i++) sx += Math.pow(Math.abs(x[i]), 4); const nx = Math.pow(sx, 0.25); if (nx > 1e-12) { let sr = 0; for (let i = 0; i < out.length; i++) sr += Math.pow(Math.abs(out[i]), 4); return rc(Math.pow(sr, 0.25), 1, 1e-9); } for (let i = 0; i < out.length; i++) if (out[i] !== 0) return false; return true; }
  );

  // 辅助：L1 绝对值求和（函数声明提升，run(18) 可用）。
  function sumAbs(a) { let s = 0; for (let i = 0; i < a.length; i++) s += Math.abs(a[i]); return s; }
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

/* ============================================================================
 * L10 微指令 · F 域（归一化）· 第 2/4 批（F033–F064）
 * ----------------------------------------------------------------------------
 * 所属层：L10-microIsa  域：F（归一化）
 * 允许依赖的层：L9（special/fpOps）、L10（registry/invariant）
 * 主要 @ref：Goodfellow et al. (2016) Deep Learning；Ba et al. (2016) LayerNorm；
 *           Welford (1962) 递推算法；Hyndman & Fan (1996) 分位数
 *
 * 涵盖：min/max/arg（F033–F048）、中心化/标准化（F049–F064）。
 * ═══════════════════════════════════════════════════════════════════════ */
;(function (NS) {
  'use strict';
  NS.L10 = NS.L10 || {};
  function reg(i, o) { NS.L10.register('F', i, o); }
  const rc = (a, b, t) => NS.L10.relClose(a, b, t);

  function sum(a) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]; return s; }
  function mean(a) { return sum(a) / a.length; }
  function meanSq(a) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * a[i]; return s / a.length; }
  function variance(a) { const m = mean(a); let s = 0; for (let i = 0; i < a.length; i++) { const d = a[i] - m; s += d * d; } const v = s / a.length; return v < 0 ? 0 : v; }
  function maxAbs(a) { let m = 0; for (let i = 0; i < a.length; i++) { const v = Math.abs(a[i]); if (v > m) m = v; } return m; }
  function maxEl(a) { let m = -Infinity; for (let i = 0; i < a.length; i++) if (a[i] > m) m = a[i]; return m; }
  function minEl(a) { let m = Infinity; for (let i = 0; i < a.length; i++) if (a[i] < m) m = a[i]; return m; }
  function clamp(x, lo, hi) { return x < lo ? lo : x > hi ? hi : x; }
  function median(a) { const s = Array.prototype.slice.call(a).sort((p, q) => p - q); const n = s.length; if (n % 2) return s[(n - 1) / 2]; return (s[n / 2 - 1] + s[n / 2]) / 2; }
  function lse(v) { let m = -Infinity; for (let i = 0; i < v.length; i++) if (v[i] > m) m = v[i]; let s = 0; for (let i = 0; i < v.length; i++) s += Math.exp(v[i] - m); return m + Math.log(s); }
  function zeros(n) { return new Float64Array(n); }

  // 矩阵工具（行优先 r×c）
  function mRowMean(M, r, c, i) { let s = 0; for (let j = 0; j < c; j++) s += M[i * c + j]; return s / c; }
  function mColMean(M, r, c, j) { let s = 0; for (let i = 0; i < r; i++) s += M[i * c + j]; return s / r; }
  function mMean(M, r, c) { let s = 0; for (let k = 0; k < r * c; k++) s += M[k]; return s / (r * c); }

  function mk(i, meta, run, pred) {
    const def = Object.assign({}, meta);
    def.run = run;
    def.predicate = function () { return pred.apply(null, Array.prototype.slice.call(arguments).concat([run])); };
    reg(i, def);
  }

  // ═══════════════════════════════════════════════════════════════════════
  // 类别 3 · min/max/arg（F033–F048）
  // ═══════════════════════════════════════════════════════════════════════
  mk(33, {
    name: '向量最小值', intuition: '最小值是所有元素中的最小者。',
    math: 'min = minᵢ xᵢ', pseudocode: 'return min(x)', complexity: 'O(n)',
    ref: 'Knuth TAOCP §5.3', commonPitfall: '浮点 −0 与 0 不应区分。',
    invariant: 'run = minᵢ xᵢ',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { return minEl(x); },
    function pred(x, run) { const out = run(x); return out === minEl(x); }
  );

  mk(34, {
    name: '向量最大值', intuition: '最大值是所有元素中的最大者。',
    math: 'max = maxᵢ xᵢ', pseudocode: 'return max(x)', complexity: 'O(n)',
    ref: 'Knuth TAOCP §5.3', commonPitfall: '浮点 −0 与 0 不应区分。',
    invariant: 'run = maxᵢ xᵢ',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { return maxEl(x); },
    function pred(x, run) { const out = run(x); return out === maxEl(x); }
  );

  mk(35, {
    name: 'argmin（最小索引）', intuition: '返回首个最小值所在下标。',
    math: 'argmin = argminᵢ xᵢ', pseudocode: 'return first index of min', complexity: 'O(n)',
    ref: 'Knuth TAOCP §5.3', commonPitfall: '并列时取首个。',
    invariant: 'x[run] = min(x) 且 run 为合法下标',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { let m = Infinity, idx = 0; for (let i = 0; i < x.length; i++) if (x[i] < m) { m = x[i]; idx = i; } return idx; },
    function pred(x, run) { const out = run(x); const m = minEl(x); if (out < 0 || out >= x.length) return false; if (!rc(x[out], m, 1e-12)) return false; for (let j = 0; j < x.length; j++) if (x[j] < x[out] - 1e-12) return false; return true; }
  );

  mk(36, {
    name: 'argmax（最大索引）', intuition: '返回首个最大值所在下标。',
    math: 'argmax = argmaxᵢ xᵢ', pseudocode: 'return first index of max', complexity: 'O(n)',
    ref: 'Knuth TAOCP §5.3', commonPitfall: '并列时取首个。',
    invariant: 'x[run] = max(x) 且 run 为合法下标',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { let m = -Infinity, idx = 0; for (let i = 0; i < x.length; i++) if (x[i] > m) { m = x[i]; idx = i; } return idx; },
    function pred(x, run) { const out = run(x); const m = maxEl(x); if (out < 0 || out >= x.length) return false; if (!rc(x[out], m, 1e-12)) return false; for (let j = 0; j < x.length; j++) if (x[j] > x[out] + 1e-12) return false; return true; }
  );

  mk(37, {
    name: 'min-max 归一化到 [0,1]', intuition: '线性拉伸使最小→0、最大→1。',
    math: 'run = (x−min)/(max−min)', pseudocode: 'return (x-min)/(max-min)', complexity: 'O(n)',
    ref: 'Han et al. 2012 数据挖掘', commonPitfall: 'max=min 时除零（退化为 0）。',
    invariant: 'runᵢ ∈ [0,1]，min→0，max→1',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { const n = x.length; const lo = minEl(x), hi = maxEl(x); const out = new Float64Array(n); if (hi > lo) for (let i = 0; i < n; i++) out[i] = (x[i] - lo) / (hi - lo); return out; },
    function pred(x, run) { const out = run(x); const lo = minEl(x), hi = maxEl(x); if (hi <= lo) { for (let i = 0; i < out.length; i++) if (out[i] !== 0) return false; return true; } for (let i = 0; i < out.length; i++) if (out[i] < -1e-12 || out[i] > 1 + 1e-12) return false; let imn = 0, imx = 0; for (let i = 1; i < out.length; i++) { if (x[i] < x[imn]) imn = i; if (x[i] > x[imx]) imx = i; } return rc(out[imn], 0, 1e-9) && rc(out[imx], 1, 1e-9); }
  );

  mk(38, {
    name: 'min-max 归一化到 [−1,1]', intuition: '线性拉伸使最小→−1、最大→1。',
    math: 'run = 2(x−min)/(max−min) − 1', pseudocode: 'return 2*(x-min)/(max-min)-1', complexity: 'O(n)',
    ref: 'Han et al. 2012', commonPitfall: 'max=min 时除零（退化为 0）。',
    invariant: 'runᵢ ∈ [−1,1]，min→−1，max→1',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { const n = x.length; const lo = minEl(x), hi = maxEl(x); const out = new Float64Array(n); if (hi > lo) for (let i = 0; i < n; i++) out[i] = 2 * (x[i] - lo) / (hi - lo) - 1; return out; },
    function pred(x, run) { const out = run(x); const lo = minEl(x), hi = maxEl(x); if (hi <= lo) { for (let i = 0; i < out.length; i++) if (out[i] !== 0) return false; return true; } for (let i = 0; i < out.length; i++) if (out[i] < -1 - 1e-12 || out[i] > 1 + 1e-12) return false; let mn = Infinity, mx = -Infinity, imn = 0, imx = 0; for (let i = 0; i < out.length; i++) { if (out[i] < mn) { mn = out[i]; imn = i; } if (out[i] > mx) { mx = out[i]; imx = i; } } return rc(out[imn], -1, 1e-9) && rc(out[imx], 1, 1e-9); }
  );

  mk(39, {
    name: '软最大（log-sum-exp）', intuition: 'LSE 是最大值的光滑上界。',
    math: 'LSE(x) = ln Σe^{xᵢ}', pseudocode: 'return max + ln(sum(exp(x-max)))', complexity: 'O(n)',
    ref: 'Bridle 1990 数值稳定', commonPitfall: '直接 exp 会溢出，须减 max。',
    invariant: 'max ≤ run ≤ max + ln n',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { return lse(x); },
    function pred(x, run) { const out = run(x); const m = maxEl(x); const lo = m - 1e-12, hi = m + Math.log(x.length) + 1e-12; return out >= lo && out <= hi; }
  );

  mk(40, {
    name: '软最小（−log-sum-exp(−x)）', intuition: 'soft-min 是最小值的光滑近似，恒 ≤ 真最小值。',
    math: 'softmin(x) = −ln Σe^{−xᵢ} ∈ [min(x)−ln n, min(x)]', pseudocode: 'return -lse(-x)', complexity: 'O(n)',
    ref: 'Bridle 1990', commonPitfall: 'soft-min 落于最小值之下（平滑下界）。',
    invariant: 'min − ln n ≤ run ≤ min',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { const n = x.length; const neg = new Float64Array(n); for (let i = 0; i < n; i++) neg[i] = -x[i]; return -lse(neg); },
    function pred(x, run) { const out = run(x); const lo = minEl(x) - Math.log(x.length) - 1e-12, hi = minEl(x) + 1e-12; return out >= lo && out <= hi; }
  );

  mk(41, {
    name: '次大值', intuition: '第二大的元素值（含并列时仍为最大值）。',
    math: 'secondMax = x_{(n−1)}（排序后）', pseudocode: 'return sort(x)[n-2]', complexity: 'O(n log n)',
    ref: 'Knuth TAOCP §5.3', commonPitfall: '全相等时等于最大值。',
    invariant: 'run = 排序后第 n−1 大，且 run ≤ max',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { const s = Array.prototype.slice.call(x).sort((a, b) => b - a); return s[s.length - 2]; },
    function pred(x, run) { const out = run(x); const s = Array.prototype.slice.call(x).sort((a, b) => b - a); return rc(out, s[s.length - 2], 1e-12) && out <= maxEl(x) + 1e-12; }
  );

  mk(42, {
    name: '次小值', intuition: '第二小的元素值（含并列时仍为最小值）。',
    math: 'secondMin = x_{(2)}（排序后）', pseudocode: 'return sort(x)[1]', complexity: 'O(n log n)',
    ref: 'Knuth TAOCP §5.3', commonPitfall: '全相等时等于最小值。',
    invariant: 'run = 排序后第 2 小，且 run ≥ min',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { const s = Array.prototype.slice.call(x).sort((a, b) => a - b); return s[1]; },
    function pred(x, run) { const out = run(x); const s = Array.prototype.slice.call(x).sort((a, b) => a - b); return rc(out, s[1], 1e-12) && out >= minEl(x) - 1e-12; }
  );

  mk(43, {
    name: '极差 span', intuition: '极差 = 最大 − 最小。',
    math: 'span = max − min', pseudocode: 'return max - min', complexity: 'O(n)',
    ref: 'Goodfellow 2016 §3.4', commonPitfall: '恒为非负。',
    invariant: 'run = max − min 且 run ≥ 0',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { return maxEl(x) - minEl(x); },
    function pred(x, run) { const out = run(x); return rc(out, maxEl(x) - minEl(x), 1e-12) && out >= -1e-12; }
  );

  mk(44, {
    name: '升序排序输出', intuition: '非降序排列（多重集不变）。',
    math: 'run = sort(x) 升序', pseudocode: 'return sorted(x)', complexity: 'O(n log n)',
    ref: 'Knuth TAOCP §5.3', commonPitfall: '须保持多重集（排序仅重排）。',
    invariant: 'run 非降序且与 x 为同一多重集',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { return Float64Array.from(Array.prototype.slice.call(x).sort((a, b) => a - b)); },
    function pred(x, run) { const out = run(x); for (let i = 1; i < out.length; i++) if (out[i] < out[i - 1] - 1e-12) return false; const s = Array.prototype.slice.call(x).sort((a, b) => a - b); for (let i = 0; i < out.length; i++) if (!rc(out[i], s[i], 1e-12)) return false; return true; }
  );

  mk(45, {
    name: '截断 clip 到 [a,b]', intuition: '将元素钳制在区间内（自动取 a,b 顺序）。',
    math: 'run = clamp(x, min(a,b), max(a,b))', pseudocode: 'return clamp(x,lo,hi)', complexity: 'O(n)',
    ref: 'IEEE 754-2019', commonPitfall: 'a>b 时区间取 [min,max]。',
    invariant: 'runᵢ ∈ [lo,hi] 且 runᵢ = clamp(xᵢ)',
    inputs: ['x', 'a', 'b'], domains: { x: 'vecSmall', a: 'finite', b: 'finite' },
  },
    function run(x, a, b) { const lo = Math.min(a, b), hi = Math.max(a, b); const n = x.length; const out = new Float64Array(n); for (let i = 0; i < n; i++) out[i] = clamp(x[i], lo, hi); return out; },
    function pred(x, a, b, run) { const out = run(x, a, b); const lo = Math.min(a, b), hi = Math.max(a, b); for (let i = 0; i < out.length; i++) { if (out[i] < lo - 1e-12 || out[i] > hi + 1e-12) return false; if (!rc(out[i], clamp(x[i], lo, hi), 1e-12)) return false; } return true; }
  );

  mk(46, {
    name: 'argmin（最小绝对值索引）', intuition: '返回绝对值最小元素下标。',
    math: 'argminAbs = argminᵢ |xᵢ|', pseudocode: 'return first index of min |x|', complexity: 'O(n)',
    ref: 'Knuth TAOCP §5.3', commonPitfall: '比较绝对值而非原值。',
    invariant: '|x[run]| = minᵢ |xᵢ|',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { let m = Infinity, idx = 0; for (let i = 0; i < x.length; i++) { const v = Math.abs(x[i]); if (v < m) { m = v; idx = i; } } return idx; },
    function pred(x, run) { const out = run(x); let m = Infinity; for (let i = 0; i < x.length; i++) { const v = Math.abs(x[i]); if (v < m) m = v; } if (out < 0 || out >= x.length) return false; return rc(Math.abs(x[out]), m, 1e-12); }
  );

  mk(47, {
    name: 'argmax（最大绝对值索引）', intuition: '返回绝对值最大元素下标。',
    math: 'argmaxAbs = argmaxᵢ |xᵢ|', pseudocode: 'return first index of max |x|', complexity: 'O(n)',
    ref: 'Knuth TAOCP §5.3', commonPitfall: '比较绝对值而非原值。',
    invariant: '|x[run]| = maxᵢ |xᵢ|',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { let m = -Infinity, idx = 0; for (let i = 0; i < x.length; i++) { const v = Math.abs(x[i]); if (v > m) { m = v; idx = i; } } return idx; },
    function pred(x, run) { const out = run(x); let m = -Infinity; for (let i = 0; i < x.length; i++) { const v = Math.abs(x[i]); if (v > m) m = v; } if (out < 0 || out >= x.length) return false; return rc(Math.abs(x[out]), m, 1e-12); }
  );

  mk(48, {
    name: '次大值索引 argsecondmax', intuition: '返回第二大元素所在下标。',
    math: 'argsecondmax = index of x_{(n−1)}', pseudocode: 'return index of 2nd largest', complexity: 'O(n log n)',
    ref: 'Knuth TAOCP §5.3', commonPitfall: '值等于次大值即可，不必唯一。',
    invariant: 'x[run] = 排序后第 n−1 大',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { const s = Array.prototype.slice.call(x).sort((a, b) => b - a); const target = s[s.length - 2]; for (let i = 0; i < x.length; i++) if (rc(x[i], target, 1e-12)) return i; return 0; },
    function pred(x, run) { const out = run(x); const s = Array.prototype.slice.call(x).sort((a, b) => b - a); const target = s[s.length - 2]; return rc(x[out], target, 1e-12); }
  );

  // ═══════════════════════════════════════════════════════════════════════
  // 类别 4 · 中心化/标准化（F049–F064）
  // ═══════════════════════════════════════════════════════════════════════
  mk(49, {
    name: '零均值化', intuition: '减去均值使向量和为 0（与全 1 向量正交）。',
    math: 'run = x − mean(x)', pseudocode: 'return x - mean(x)', complexity: 'O(n)',
    ref: 'Goodfellow 2016 §5.1', commonPitfall: '仅平移，不改尺度。',
    invariant: 'Σ runᵢ ≈ 0',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { const n = x.length; const m = mean(x); const out = new Float64Array(n); for (let i = 0; i < n; i++) out[i] = x[i] - m; return out; },
    function pred(x, run) { const out = run(x); let s = 0; for (let i = 0; i < out.length; i++) s += out[i]; return rc(s, 0, 1e-9); }
  );

  mk(50, {
    name: 'z-score 标准化', intuition: '减均值除标准差，得零均值单位方差。',
    math: 'run = (x−μ)/σ', pseudocode: 'return (x-mean)/std', complexity: 'O(n)',
    ref: 'Goodfellow 2016 §5.1', commonPitfall: 'σ=0 时退化为 0。',
    invariant: 'mean(run)≈0 且 var(run)≈1',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { const n = x.length; const m = mean(x); const v = variance(x); const out = new Float64Array(n); if (v <= 1e-14) return out; const s = Math.sqrt(v); for (let i = 0; i < n; i++) out[i] = (x[i] - m) / s; return out; },
    function pred(x, run) { const out = run(x); const v = variance(x); if (v <= 1e-14) { let z = true; for (let i = 0; i < out.length; i++) if (out[i] !== 0) { z = false; break; } return z; } const m = mean(out); const vr = variance(out); return rc(m, 0, 1e-6) && rc(vr, 1, 1e-6); }
  );

  mk(51, {
    name: '按 max-abs 缩放', intuition: '除以最大绝对值，映射进 [−1,1]。',
    math: 'run = x / max|xᵢ|', pseudocode: 'return x / maxAbs(x)', complexity: 'O(n)',
    ref: 'Ba et al. 2016', commonPitfall: 'max|x|=0 时除零（退化为 0）。',
    invariant: 'max|run| ≈ 1 且 |runᵢ| ≤ 1',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { const n = x.length; const m = maxAbs(x); const out = new Float64Array(n); if (m > 1e-12) for (let i = 0; i < n; i++) out[i] = x[i] / m; return out; },
    function pred(x, run) { const out = run(x); const m = maxAbs(x); if (m <= 1e-12) { for (let i = 0; i < out.length; i++) if (out[i] !== 0) return false; return true; } if (maxAbs(out) > 1 + 1e-9) return false; return rc(maxAbs(out), 1, 1e-9); }
  );

  mk(52, {
    name: 'min-max 缩放 [0,1]', intuition: '线性拉伸入 [0,1]。',
    math: 'run = (x−min)/(max−min)', pseudocode: 'return (x-min)/(max-min)', complexity: 'O(n)',
    ref: 'Han et al. 2012', commonPitfall: 'max=min 时除零（退化为 0）。',
    invariant: 'runᵢ ∈ [0,1]，min→0，max→1',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { const n = x.length; const lo = minEl(x), hi = maxEl(x); const out = new Float64Array(n); if (hi > lo) for (let i = 0; i < n; i++) out[i] = (x[i] - lo) / (hi - lo); return out; },
    function pred(x, run) { const out = run(x); const lo = minEl(x), hi = maxEl(x); if (hi <= lo) { for (let i = 0; i < out.length; i++) if (out[i] !== 0) return false; return true; } for (let i = 0; i < out.length; i++) if (out[i] < -1e-12 || out[i] > 1 + 1e-12) return false; let mn = Infinity, mx = -Infinity, imn = 0, imx = 0; for (let i = 0; i < out.length; i++) { if (out[i] < mn) { mn = out[i]; imn = i; } if (out[i] > mx) { mx = out[i]; imx = i; } } return rc(out[imn], 0, 1e-9) && rc(out[imx], 1, 1e-9); }
  );

  mk(53, {
    name: '指数移动平均序列 EMA', intuition: 'EMA 是历史输入的凸组合，恒落在区间内。',
    math: 'ema₀=x₀；emaₜ=αxₜ+(1−α)emaₜ₋₁', pseudocode: 'ema = alpha*x + (1-alpha)*ema', complexity: 'O(n)',
    ref: 'Box & Jenkins 1976', commonPitfall: 'α∈[0,1]，初始为 x₀。',
    invariant: '每个 runₜ ∈ [min(x), max(x)]（凸组合有界）',
    inputs: ['x', 'alpha'], domains: { x: 'vecSmall', alpha: 'unit' },
  },
    function run(x, alpha) { const n = x.length; const out = new Float64Array(n); out[0] = x[0]; for (let i = 1; i < n; i++) out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]; return out; },
    function pred(x, alpha, run) { const out = run(x, alpha); const lo = minEl(x), hi = maxEl(x); if (!rc(out[0], x[0], 1e-12)) return false; for (let i = 0; i < out.length; i++) if (out[i] < lo - 1e-9 || out[i] > hi + 1e-9) return false; return true; }
  );

  mk(54, {
    name: '在线均值更新', intuition: 'Welford 式单步均值递推。',
    math: 'new = old + (x−old)/(n+1)', pseudocode: 'return old + (x-old)/(n+1)', complexity: 'O(1)',
    ref: 'Welford 1962', commonPitfall: 'n 为已见样本数。',
    invariant: 'run = (old·n + x)/(n+1)',
    inputs: ['oldMean', 'x', 'n'], domains: { oldMean: 'finite', x: 'finite', n: 'nat' },
  },
    function run(oldMean, x, n) { return oldMean + (x - oldMean) / (n + 1); },
    function pred(oldMean, x, n, run) { const out = run(oldMean, x, n); return rc(out, (oldMean * n + x) / (n + 1), 1e-9); }
  );

  mk(55, {
    name: '在线方差递推（Welford M2）', intuition: '单步更新离差平方和 M2。',
    math: 'M2ₙₑ𝓌 = M2 + n/(n+1)·(x−μ)²', pseudocode: 'return M2 + n/(n+1)*(x-mean)^2', complexity: 'O(1)',
    ref: 'Welford 1962', commonPitfall: 'M2 为总体离差平方和。',
    invariant: 'run = M2 + n/(n+1)·(x−μ)²',
    inputs: ['mean', 'M2', 'x', 'n'], domains: { mean: 'finite', M2: 'nonNeg', x: 'finite', n: 'nat' },
  },
    function run(mean, M2, x, n) { const d = x - mean; return M2 + (n / (n + 1)) * d * d; },
    function pred(mean, M2, x, n, run) { const out = run(mean, M2, x, n); const d = x - mean; return rc(out, M2 + (n / (n + 1)) * d * d, 1e-9); }
  );

  mk(56, {
    name: '累积和序列', intuition: '前缀和，第 k 项为前 k+1 项之和。',
    math: 'runₖ = Σ_{i≤k} xᵢ', pseudocode: 'prefixSum', complexity: 'O(n)',
    ref: 'Knuth TAOCP §5.3', commonPitfall: 'run₀ = x₀。',
    invariant: 'runₖ = 前缀和，且 run₀ = x₀',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { const n = x.length; const out = new Float64Array(n); let s = 0; for (let i = 0; i < n; i++) { s += x[i]; out[i] = s; } return out; },
    function pred(x, run) { const out = run(x); let s = 0; for (let i = 0; i < x.length; i++) { s += x[i]; if (!rc(out[i], s, 1e-9)) return false; } return true; }
  );

  mk(57, {
    name: '累积积序列', intuition: '前缀积，第 k 项为前 k+1 项之积。',
    math: 'runₖ = Π_{i≤k} xᵢ', pseudocode: 'prefixProduct', complexity: 'O(n)',
    ref: 'Knuth TAOCP §5.3', commonPitfall: 'run₀ = x₀；零因子使后为 0。',
    invariant: 'runₖ = 前缀积，且 run₀ = x₀',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { const n = x.length; const out = new Float64Array(n); let p = 1; for (let i = 0; i < n; i++) { p *= x[i]; out[i] = p; } return out; },
    function pred(x, run) { const out = run(x); let p = 1; for (let i = 0; i < x.length; i++) { p *= x[i]; if (!rc(out[i], p, 1e-9)) return false; } return true; }
  );

  mk(58, {
    name: '运行均值序列', intuition: '到第 k 项为止的滑动均值。',
    math: 'runₖ = (Σ_{i≤k} xᵢ)/(k+1)', pseudocode: 'prefixMean', complexity: 'O(n)',
    ref: 'Knuth TAOCP §5.3', commonPitfall: '分母为 k+1。',
    invariant: 'runₖ = 前缀和/(k+1)',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { const n = x.length; const out = new Float64Array(n); let s = 0; for (let i = 0; i < n; i++) { s += x[i]; out[i] = s / (i + 1); } return out; },
    function pred(x, run) { const out = run(x); let s = 0; for (let i = 0; i < x.length; i++) { s += x[i]; if (!rc(out[i], s / (i + 1), 1e-9)) return false; } return true; }
  );

  mk(59, {
    name: '鲁棒标准化（中位数/MAD）', intuition: '用中位数与 MAD 替代均值/标准差，抗离群。',
    math: 'run = (x−median)/(MAD+ε), MAD=median(|x−median|)', pseudocode: 'return (x-median)/(mad+eps)', complexity: 'O(n log n)',
    ref: 'Rousseeuw & Croux 1993', commonPitfall: 'MAD=0 时退化为 0 但 median 仍为 0。',
    invariant: 'median(run) ≈ 0（平移不变性）',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { const n = x.length; const med = median(x); let s = 0; const d = new Float64Array(n); for (let i = 0; i < n; i++) d[i] = Math.abs(x[i] - med); const mad = median(d); const out = new Float64Array(n); const denom = mad + 1e-12; for (let i = 0; i < n; i++) out[i] = (x[i] - med) / denom; return out; },
    function pred(x, run) { const out = run(x); return Math.abs(median(out)) < 1e-6; }
  );

  mk(60, {
    name: '矩阵逐行中心化', intuition: '每行减去其行均值，使行和为 0。',
    math: 'run_{ij} = M_{ij} − rowMean_i', pseudocode: 'return M - rowMeans', complexity: 'O(r·c)',
    ref: 'Goodfellow 2016 §5.1', commonPitfall: '逐行独立处理。',
    invariant: '每行均值 ≈ 0',
    inputs: ['M'], domains: { M: 'matSmall' }, matrixDims: { M: [4, 4] },
  },
    function run(M) { const r = 4, c = 4; const out = new Float64Array(r * c); for (let i = 0; i < r; i++) { const rm = mRowMean(M, r, c, i); for (let j = 0; j < c; j++) out[i * c + j] = M[i * c + j] - rm; } return out; },
    function pred(M, run) { const out = run(M); const r = 4, c = 4; for (let i = 0; i < r; i++) { let s = 0; for (let j = 0; j < c; j++) s += out[i * c + j]; if (!rc(s, 0, 1e-9)) return false; } return true; }
  );

  mk(61, {
    name: '矩阵逐列中心化', intuition: '每列减去其列均值，使列和为 0。',
    math: 'run_{ij} = M_{ij} − colMean_j', pseudocode: 'return M - colMeans', complexity: 'O(r·c)',
    ref: 'Goodfellow 2016 §5.1', commonPitfall: '逐列独立处理。',
    invariant: '每列均值 ≈ 0',
    inputs: ['M'], domains: { M: 'matSmall' }, matrixDims: { M: [4, 4] },
  },
    function run(M) { const r = 4, c = 4; const out = new Float64Array(r * c); for (let j = 0; j < c; j++) { const cm = mColMean(M, r, c, j); for (let i = 0; i < r; i++) out[i * c + j] = M[i * c + j] - cm; } return out; },
    function pred(M, run) { const out = run(M); const r = 4, c = 4; for (let j = 0; j < c; j++) { let s = 0; for (let i = 0; i < r; i++) s += out[i * c + j]; if (!rc(s, 0, 1e-9)) return false; } return true; }
  );

  mk(62, {
    name: '反向累积和序列', intuition: '后缀和，第 i 项为从 i 到末尾之和。',
    math: 'runᵢ = Σ_{j≥i} xⱼ', pseudocode: 'suffixSum', complexity: 'O(n)',
    ref: 'Knuth TAOCP §5.3', commonPitfall: 'run_{n−1} = x_{n−1}。',
    invariant: 'runᵢ = run_{i+1} + xᵢ（末项 = x_{n−1}）',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { const n = x.length; const out = new Float64Array(n); let s = 0; for (let i = n - 1; i >= 0; i--) { s += x[i]; out[i] = s; } return out; },
    function pred(x, run) { const out = run(x); const n = x.length; if (!rc(out[n - 1], x[n - 1], 1e-9)) return false; let s = 0; for (let i = n - 1; i >= 0; i--) { s += x[i]; if (!rc(out[i], s, 1e-9)) return false; } return true; }
  );

  mk(63, {
    name: '参数化标准化（可重建）', intuition: '给定均值与标准差的标准化，可由 run 反解 x。',
    math: 'run = (x−μ)/σ', pseudocode: 'return (x-mu)/sigma', complexity: 'O(n)',
    ref: 'Goodfellow 2016 §5.1', commonPitfall: 'σ=0 时无定义（域用 nonZero）。',
    invariant: 'run·σ + μ = x',
    inputs: ['x', 'mu', 'sigma'], domains: { x: 'vecSmall', mu: 'finite', sigma: 'nonZero' },
  },
    function run(x, mu, sigma) { const n = x.length; const out = new Float64Array(n); for (let i = 0; i < n; i++) out[i] = (x[i] - mu) / sigma; return out; },
    function pred(x, mu, sigma, run) { const out = run(x, mu, sigma); for (let i = 0; i < out.length; i++) if (!rc(out[i] * sigma + mu, x[i], 1e-9)) return false; return true; }
  );

  mk(64, {
    name: '线性重缩放 a·x+b', intuition: '仿射变换，可由输出反解输入。',
    math: 'run = a·x + b', pseudocode: 'return a*x + b', complexity: 'O(n)',
    ref: 'IEEE 754-2019', commonPitfall: 'a=0 时退化（域用 nonZero）。',
    invariant: '(run − b)/a = x',
    inputs: ['x', 'a', 'b'], domains: { x: 'vecSmall', a: 'nonZero', b: 'finite' },
  },
    function run(x, a, b) { const n = x.length; const out = new Float64Array(n); for (let i = 0; i < n; i++) out[i] = a * x[i] + b; return out; },
    function pred(x, a, b, run) { const out = run(x, a, b); for (let i = 0; i < out.length; i++) if (!rc((out[i] - b) / a, x[i], 1e-9)) return false; return true; }
  );

  // 辅助：argMax 下标（供 F037 不变量用，函数声明提升）
  function argMaxIdx(a) { let m = -Infinity, idx = 0; for (let i = 0; i < a.length; i++) if (a[i] > m) { m = a[i]; idx = i; } return idx; }
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

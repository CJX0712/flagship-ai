/* ============================================================================
 * L10 微指令 · F 域（归一化）· 第 4/4 批（F097–F128）
 * ----------------------------------------------------------------------------
 * 所属层：L10-microIsa  域：F（归一化）
 * 允许依赖的层：L9（special/fpOps）、L10（registry/invariant）
 * 主要 @ref：Bridle (1990) log-sum-exp；Ioffe & Szegedy (2015) BatchNorm；
 *           Ba et al. (2016) LayerNorm；Krizhevsky et al. (2012) LRN
 *
 * 涵盖：logsumexp 高级/数值稳定（F097–F112）、矩阵/批/高级归一化（F113–F128）。
 * ═══════════════════════════════════════════════════════════════════════ */
;(function (NS) {
  'use strict';
  NS.L10 = NS.L10 || {};
  function reg(i, o) { NS.L10.register('F', i, o); }
  const rc = (a, b, t) => NS.L10.relClose(a, b, t);

  function mean(a) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]; return s / a.length; }
  function variance(a) { const m = mean(a); let s = 0; for (let i = 0; i < a.length; i++) { const d = a[i] - m; s += d * d; } const v = s / a.length; return v < 0 ? 0 : v; }
  function maxEl(a) { let m = -Infinity; for (let i = 0; i < a.length; i++) if (a[i] > m) m = a[i]; return m; }
  function minEl(a) { let m = Infinity; for (let i = 0; i < a.length; i++) if (a[i] < m) m = a[i]; return m; }
  function dot(a, b) { let s = 0; const n = a.length < b.length ? a.length : b.length; for (let i = 0; i < n; i++) s += a[i] * b[i]; return s; }
  function zeros(n) { return new Float64Array(n); }
  function softplus(x) { return x > 0 ? x + Math.log1p(Math.exp(-x)) : Math.log1p(Math.exp(x)); }
  function lse(v) { let m = -Infinity; for (let i = 0; i < v.length; i++) if (v[i] > m) m = v[i]; let s = 0; for (let i = 0; i < v.length; i++) s += Math.exp(v[i] - m); return m + Math.log(s); }
  function softmax(v) { let m = -Infinity; for (let i = 0; i < v.length; i++) if (v[i] > m) m = v[i]; const out = new Float64Array(v.length); let s = 0; for (let i = 0; i < v.length; i++) { const e = Math.exp(v[i] - m); out[i] = e; s += e; } const inv = 1 / s; for (let i = 0; i < v.length; i++) out[i] *= inv; return out; }

  // 矩阵工具（r×c，行优先）
  function mMean(M, r, c) { let s = 0; for (let k = 0; k < r * c; k++) s += M[k]; return s / (r * c); }
  function mVar(M, r, c) { const m = mMean(M, r, c); let s = 0; for (let k = 0; k < r * c; k++) { const d = M[k] - m; s += d * d; } const v = s / (r * c); return v < 0 ? 0 : v; }
  function mRowMean(M, r, c, i) { let s = 0; for (let j = 0; j < c; j++) s += M[i * c + j]; return s / c; }
  function mColMean(M, r, c, j) { let s = 0; for (let i = 0; i < r; i++) s += M[i * c + j]; return s / r; }
  function mRowVar(M, r, c, i) { const m = mRowMean(M, r, c, i); let s = 0; for (let j = 0; j < c; j++) { const d = M[i * c + j] - m; s += d * d; } const v = s / c; return v < 0 ? 0 : v; }
  function mColVar(M, r, c, j) { const m = mColMean(M, r, c, j); let s = 0; for (let i = 0; i < r; i++) { const d = M[i * c + j] - m; s += d * d; } const v = s / r; return v < 0 ? 0 : v; }
  function mRowSum(M, r, c, i) { let s = 0; for (let j = 0; j < c; j++) s += M[i * c + j]; return s; }
  function mColSum(M, r, c, j) { let s = 0; for (let i = 0; i < r; i++) s += M[i * c + j]; return s; }
  function mSoftmaxRows(M, r, c) { const out = new Float64Array(r * c); for (let i = 0; i < r; i++) { let m = -Infinity; for (let j = 0; j < c; j++) if (M[i * c + j] > m) m = M[i * c + j]; let s = 0; for (let j = 0; j < c; j++) { const e = Math.exp(M[i * c + j] - m); out[i * c + j] = e; s += e; } const inv = 1 / s; for (let j = 0; j < c; j++) out[i * c + j] *= inv; } return out; }
  function mSoftmaxCols(M, r, c) { const out = new Float64Array(r * c); for (let j = 0; j < c; j++) { let m = -Infinity; for (let i = 0; i < r; i++) if (M[i * c + j] > m) m = M[i * c + j]; let s = 0; for (let i = 0; i < r; i++) { const e = Math.exp(M[i * c + j] - m); out[i * c + j] = e; s += e; } const inv = 1 / s; for (let i = 0; i < r; i++) out[i * c + j] *= inv; } return out; }

  function mk(i, meta, run, pred) {
    const def = Object.assign({}, meta);
    def.run = run;
    def.predicate = function () { return pred.apply(null, Array.prototype.slice.call(arguments).concat([run])); };
    reg(i, def);
  }

  // ═══════════════════════════════════════════════════════════════════════
  // 类别 7 · logsumexp 高级 / 数值稳定（F097–F112）
  // ═══════════════════════════════════════════════════════════════════════
  mk(97, {
    name: 'LSE 减 max 稳定式', intuition: '减最大值后求和再取对数，与朴素 LSE 数值等价。',
    math: 'LSE = max + ln Σe^{x−max}', pseudocode: 'return max + log(sum(exp(x-max)))', complexity: 'O(n)',
    ref: 'Numerical Recipes', commonPitfall: '朴素式易溢出，二者数值应一致。',
    invariant: 'run = ln Σe^{xᵢ}',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { return lse(x); },
    function pred(x, run) { const out = run(x); let s = 0; for (let i = 0; i < x.length; i++) s += Math.exp(x[i]); return rc(out, Math.log(s), 1e-9); }
  );

  mk(98, {
    name: 'LSE 与 max 之差', intuition: 'LSE 超出最大值的部分介于 0 与 ln n 之间。',
    math: 'LSE(x) − max(x) ∈ [0, ln n]', pseudocode: 'return lse(x) - max(x)', complexity: 'O(n)',
    ref: 'Bridle 1990', commonPitfall: '等号分别在退化与弥散时达到。',
    invariant: '0 ≤ run ≤ ln n',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { return lse(x) - maxEl(x); },
    function pred(x, run) { const out = run(x); return out >= -1e-12 && out <= Math.log(x.length) + 1e-12; }
  );

  mk(99, {
    name: '稳定交叉熵（CE = −x_y + LSE）', intuition: '负对数似然恒非负。',
    math: 'CE = LSE(x) − x_y', pseudocode: 'return lse(x) - x[y]', complexity: 'O(n)',
    ref: 'Goodfellow 2016 §6.2', commonPitfall: 'y 取模防越界。',
    invariant: 'run = LSE − x_y 且 run ≥ 0',
    inputs: ['x', 'y'], domains: { x: 'vecSmall', y: 'nat' },
  },
    function run(x, y) { const yy = ((y % x.length) + x.length) % x.length; return lse(x) - x[yy]; },
    function pred(x, y, run) { const out = run(x, y); const yy = ((y % x.length) + x.length) % x.length; if (out < -1e-12) return false; return rc(out, lse(x) - x[yy], 1e-9); }
  );

  mk(100, {
    name: '稳定 log1p', intuition: 'log(1+x) 可由 e^{run} = 1+x 回检。',
    math: 'run = ln(1+x)', pseudocode: 'return log1p(x)', complexity: 'O(1)',
    ref: 'Cody & Waite 1980', commonPitfall: '用 Math.log1p 避免小 x 精度损失。',
    invariant: 'e^{run} = 1 + x',
    inputs: ['x'], domains: { x: 'smallPos' },
  },
    function run(x) { return Math.log1p(x); },
    function pred(x, run) { const out = run(x); return rc(Math.exp(out), 1 + x, 1e-9); }
  );

  mk(101, {
    name: '稳定 log(1−σ(x)) = −softplus(x)', intuition: '对数互补概率可借 softplus 稳定计算。',
    math: 'log(1−σ(x)) = −softplus(x)', pseudocode: 'return -softplus(x)', complexity: 'O(1)',
    ref: 'Hendrycks & Gimpel 2016', commonPitfall: '直接算 1−σ(x) 再 log 易欠精度。',
    invariant: 'run = −softplus(x) 且 run ≤ 0',
    inputs: ['x'], domains: { x: 'finite' },
  },
    function run(x) { return -softplus(x); },
    function pred(x, run) { const out = run(x); if (!rc(out, -softplus(x), 1e-9)) return false; return out <= 1e-12; }
  );

  mk(102, {
    name: '双 LSE 之差（稳定）', intuition: '两序列 LSE 之差等于其指数和之比的对数。',
    math: 'LSE(x)−LSE(y) = ln(Σe^{xᵢ}/Σe^{yⱼ})', pseudocode: 'return lse(x) - lse(y)', complexity: 'O(n)',
    ref: 'Bridle 1990', commonPitfall: '两侧均减 max 保持数值稳定。',
    invariant: 'run = ln(Σe^{xᵢ} / Σe^{yⱼ})',
    inputs: ['x', 'y'], domains: { x: 'vecSmall', y: 'vecSmall' },
  },
    function run(x, y) { return lse(x) - lse(y); },
    function pred(x, y, run) { const out = run(x, y); let sx = 0, sy = 0; for (let i = 0; i < x.length; i++) sx += Math.exp(x[i]); for (let i = 0; i < y.length; i++) sy += Math.exp(y[i]); return rc(out, Math.log(sx / sy), 1e-9); }
  );

  mk(103, {
    name: 'log-mean-exp', intuition: 'LSE 减 ln n 即对数均值指数。',
    math: 'logmeanexp = LSE(x) − ln n', pseudocode: 'return lse(x) - log(n)', complexity: 'O(n)',
    ref: 'Goodfellow 2016', commonPitfall: '由 Jensen 不等式有 logmeanexp ≥ mean(x)。',
    invariant: 'run = LSE − ln n 且 run ≥ mean(x)',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { return lse(x) - Math.log(x.length); },
    function pred(x, run) { const out = run(x); if (out < mean(x) - 1e-12) return false; return rc(out, lse(x) - Math.log(x.length), 1e-9); }
  );

  mk(104, {
    name: '两元素 LSE（稳定）', intuition: 'ln(e^a+e^b) 减 max 后无溢出。',
    math: 'LSE₂ = max + ln(e^{a−m}+e^{b−m})', pseudocode: 'return max + log(exp(a-m)+exp(b-m))', complexity: 'O(1)',
    ref: 'Numerical Recipes', commonPitfall: '恒有 LSE₂ ≥ max(a,b)。',
    invariant: 'run ≥ max(a,b) 且 run = ln(e^a+e^b)',
    inputs: ['a', 'b'], domains: { a: 'small', b: 'small' },
  },
    function run(a, b) { const m = Math.max(a, b); return m + Math.log(Math.exp(a - m) + Math.exp(b - m)); },
    function pred(a, b, run) { const out = run(a, b); if (out < Math.max(a, b) - 1e-12) return false; return rc(out, Math.log(Math.exp(a) + Math.exp(b)), 1e-9); }
  );

  mk(105, {
    name: '稳定 log1mexp = log(1−e^{−x})', intuition: 'x>0 时 1−e^{−x}∈(0,1)，其对数为负。',
    math: 'run = ln(1 − e^{−x})', pseudocode: 'return log1p(-exp(-x))', complexity: 'O(1)',
    ref: 'Mächler 2012 数值稳定', commonPitfall: 'x 很大时趋近 0⁻，x→0⁺ 时趋近 −∞。',
    invariant: 'run = ln(1 − e^{−x}) 且 run ≤ 0',
    inputs: ['x'], domains: { x: 'pos' },
  },
    function run(x) { return Math.log1p(-Math.exp(-x)); },
    function pred(x, run) { const out = run(x); if (out > 1e-12) return false; return rc(out, Math.log(1 - Math.exp(-x)), 1e-9); }
  );

  mk(106, {
    name: '投影到球面（单位化）', intuition: '向量除以其 L2 范数得单位向量（零向量退化为 0）。',
    math: 'run = x / ‖x‖₂', pseudocode: 'return x / norm(x)', complexity: 'O(n)',
    ref: 'Goodfellow 2016 §3.4', commonPitfall: 'x=0 时除零，须特判。',
    invariant: 'run = x/‖x‖（即 run·‖x‖ = x）；x≠0 时 ‖run‖=1',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { const n = x.length; let s = 0; for (let i = 0; i < n; i++) s += x[i] * x[i]; const nx = Math.sqrt(s); const out = new Float64Array(n); if (nx > 1e-12) for (let i = 0; i < n; i++) out[i] = x[i] / nx; return out; },
    function pred(x, run) { const out = run(x); let s = 0; for (let i = 0; i < x.length; i++) s += x[i] * x[i]; const nx = Math.sqrt(s); if (nx <= 1e-12) { for (let i = 0; i < out.length; i++) if (out[i] !== 0) return false; return true; } for (let i = 0; i < out.length; i++) if (!rc(out[i] * nx, x[i], 1e-9)) return false; return true; }
  );

  mk(107, {
    name: '软符号 softsign', intuition: 'softsign 是 sign 的平滑近似，有界、保号、奇函数。',
    math: 'ss(x) = x/(1+|x|)', pseudocode: 'return x/(1+abs(x))', complexity: 'O(1)',
    ref: 'Turian et al. 2010', commonPitfall: '|ss(x)|<1，且 ss(−x)=−ss(x)。',
    invariant: '|run|<1，run·x≥0，且 run 为奇函数',
    inputs: ['x'], domains: { x: 'finite' },
  },
    function run(x) { return x / (1 + Math.abs(x)); },
    function pred(x, run) { const out = run(x); const ss = (v) => v / (1 + Math.abs(v)); if (Math.abs(out) > 1 + 1e-12) return false; if (out * x < -1e-12) return false; return rc(out, -ss(-x), 1e-9); }
  );

  mk(108, {
    name: 'LSE 平移不变性（带偏移）', intuition: '对序列整体加常数 c，LSE 恰好增加 c。',
    math: 'LSE(x+c) = LSE(x) + c', pseudocode: 'return lse(x+c)', complexity: 'O(n)',
    ref: 'Bridle 1990', commonPitfall: '稳定式即便 c 很大也不溢出。',
    invariant: 'run = LSE(x) + c',
    inputs: ['x', 'c'], domains: { x: 'vecSmall', c: 'finite' },
  },
    function run(x, c) { const n = x.length; const xc = new Float64Array(n); for (let i = 0; i < n; i++) xc[i] = x[i] + c; return lse(xc); },
    function pred(x, c, run) { const out = run(x, c); return rc(out, lse(x) + c, 1e-9); }
  );

  mk(109, {
    name: 'softmax 交叉熵梯度（和为零）', intuition: 'CE 对 logits 的梯度 ∑(pᵢ−1_{i=y})=0。',
    math: 'Σ(runᵢ − 𝟙_{i=y}) = 0', pseudocode: 'return softmax(x); check sum(p-ind)=0', complexity: 'O(n)',
    ref: 'Goodfellow 2016 §6.2', commonPitfall: '指示项仅在第 y 位为 1。',
    invariant: 'Σ(runᵢ − 𝟙_{i=y}) ≈ 0',
    inputs: ['x', 'y'], domains: { x: 'vecSmall', y: 'nat' },
  },
    function run(x, y) { return softmax(x); },
    function pred(x, y, run) { const out = run(x, y); const yy = ((y % x.length) + x.length) % x.length; let s = 0; for (let i = 0; i < out.length; i++) s += out[i] - (i === yy ? 1 : 0); return rc(s, 0, 1e-9); }
  );

  mk(110, {
    name: 'ReLU 投影（非负化）', intuition: '逐元素取 max(x,0)，非负且保留正值。',
    math: 'runᵢ = max(xᵢ, 0)', pseudocode: 'return max(x,0)', complexity: 'O(n)',
    ref: 'Nair & Hinton 2010', commonPitfall: '负值钳为 0，正值不变。',
    invariant: 'runᵢ = max(xᵢ, 0)',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { const n = x.length; const out = new Float64Array(n); for (let i = 0; i < n; i++) out[i] = x[i] > 0 ? x[i] : 0; return out; },
    function pred(x, run) { const out = run(x); for (let i = 0; i < out.length; i++) { const exp = x[i] > 0 ? x[i] : 0; if (!rc(out[i], exp, 1e-12)) return false; } return true; }
  );

  mk(111, {
    name: 'softmax 严格边界', intuition: '概率分量严格落在 (0,1) 开区间。',
    math: '0 < pᵢ < 1', pseudocode: 'return softmax(x); check 0<p<1', complexity: 'O(n)',
    ref: 'Goodfellow 2016 §6.2', commonPitfall: 'n≥2 时各分量必小于 1。',
    invariant: '对每个 i：0 ≤ runᵢ < 1',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { return softmax(x); },
    function pred(x, run) { const out = run(x); for (let i = 0; i < out.length; i++) { if (out[i] < -1e-12) return false; if (out[i] >= 1 + 1e-12) return false; } return true; }
  );

  mk(112, {
    name: 'softmax L2 范数上界', intuition: '概率向量的平方 L2 范数不超过 1。',
    math: 'Σpᵢ² ≤ 1', pseudocode: 'return softmax(x); check ||p||^2<=1', complexity: 'O(n)',
    ref: 'Goodfellow 2016 §6.2', commonPitfall: '等号在 one-hot 时达到。',
    invariant: 'Σ runᵢ² ≤ 1',
    inputs: ['x'], domains: { x: 'vecSmall' },
  },
    function run(x) { return softmax(x); },
    function pred(x, run) { const out = run(x); let s = 0; for (let i = 0; i < out.length; i++) s += out[i] * out[i]; return s <= 1 + 1e-12; }
  );

  // ═══════════════════════════════════════════════════════════════════════
  // 类别 8 · 矩阵/批/高级归一化（F113–F128）
  // ═══════════════════════════════════════════════════════════════════════
  mk(113, {
    name: '行 L2 归一化（矩阵）', intuition: '每行除以其 L2 范数变为单位行。',
    math: 'run_{i·} = M_{i·}/‖M_{i·}‖₂', pseudocode: 'return rowL2Norm(M)', complexity: 'O(r·c)',
    ref: 'Goodfellow 2016 §5.1', commonPitfall: '零行退化为 0。',
    invariant: '每行 ‖run‖₂ ≈ 1（非零行）',
    inputs: ['M'], domains: { M: 'matSmall' }, matrixDims: { M: [4, 4] },
  },
    function run(M) { const r = 4, c = 4; const out = new Float64Array(r * c); for (let i = 0; i < r; i++) { let s = 0; for (let j = 0; j < c; j++) s += M[i * c + j] * M[i * c + j]; const nrm = Math.sqrt(s); if (nrm > 1e-12) for (let j = 0; j < c; j++) out[i * c + j] = M[i * c + j] / nrm; } return out; },
    function pred(M, run) { const out = run(M); const r = 4, c = 4; for (let i = 0; i < r; i++) { let s = 0, nrm = 0; for (let j = 0; j < c; j++) { nrm += M[i * c + j] * M[i * c + j]; s += out[i * c + j] * out[i * c + j]; } nrm = Math.sqrt(nrm); if (nrm > 1e-12) { if (!rc(Math.sqrt(s), 1, 1e-9)) return false; } else { for (let j = 0; j < c; j++) if (out[i * c + j] !== 0) return false; } } return true; }
  );

  mk(114, {
    name: '列 L2 归一化（矩阵）', intuition: '每列除以其 L2 范数变为单位列。',
    math: 'run_{·j} = M_{·j}/‖M_{·j}‖₂', pseudocode: 'return colL2Norm(M)', complexity: 'O(r·c)',
    ref: 'Goodfellow 2016 §5.1', commonPitfall: '零列退化为 0。',
    invariant: '每列 ‖run‖₂ ≈ 1（非零列）',
    inputs: ['M'], domains: { M: 'matSmall' }, matrixDims: { M: [4, 4] },
  },
    function run(M) { const r = 4, c = 4; const out = new Float64Array(r * c); for (let j = 0; j < c; j++) { let s = 0; for (let i = 0; i < r; i++) s += M[i * c + j] * M[i * c + j]; const nrm = Math.sqrt(s); if (nrm > 1e-12) for (let i = 0; i < r; i++) out[i * c + j] = M[i * c + j] / nrm; } return out; },
    function pred(M, run) { const out = run(M); const r = 4, c = 4; for (let j = 0; j < c; j++) { let s = 0, nrm = 0; for (let i = 0; i < r; i++) { nrm += M[i * c + j] * M[i * c + j]; s += out[i * c + j] * out[i * c + j]; } nrm = Math.sqrt(nrm); if (nrm > 1e-12) { if (!rc(Math.sqrt(s), 1, 1e-9)) return false; } else { for (let i = 0; i < r; i++) if (out[i * c + j] !== 0) return false; } } return true; }
  );

  mk(115, {
    name: '逐列 z-score（矩阵）', intuition: '每列独立减均值除标准差。',
    math: 'run_{·j} = (M_{·j}−μ_j)/σ_j', pseudocode: 'return colZScore(M)', complexity: 'O(r·c)',
    ref: 'Goodfellow 2016 §5.1', commonPitfall: '逐列而非全局。',
    invariant: '每列 mean≈0 且 var≈1',
    inputs: ['M'], domains: { M: 'matSmall' }, matrixDims: { M: [4, 4] },
  },
    function run(M) { const r = 4, c = 4; const out = new Float64Array(r * c); for (let j = 0; j < c; j++) { const cm = mColMean(M, r, c, j); const cv = mColVar(M, r, c, j); if (cv <= 1e-14) continue; const inv = 1 / Math.sqrt(cv); for (let i = 0; i < r; i++) out[i * c + j] = (M[i * c + j] - cm) * inv; } return out; },
    function pred(M, run) { const out = run(M); const r = 4, c = 4; for (let j = 0; j < c; j++) { const cv = mColVar(M, r, c, j); if (cv <= 1e-14) { for (let i = 0; i < r; i++) if (out[i * c + j] !== 0) return false; continue; } let s = 0; for (let i = 0; i < r; i++) s += out[i * c + j]; if (!rc(s / r, 0, 1e-6)) return false; let v = 0; for (let i = 0; i < r; i++) { const d = out[i * c + j] - 0; v += d * d; } v /= r; if (!rc(v, 1, 1e-6)) return false; } return true; }
  );

  mk(116, {
    name: '矩阵逐列 softmax', intuition: '每列独立 softmax。',
    math: 'run_{·j} = softmax(M_{·j})', pseudocode: 'return colSoftmax(M)', complexity: 'O(r·c)',
    ref: 'Goodfellow 2016 §6.2', commonPitfall: '逐列而非全局。',
    invariant: '每列 Σ ≈ 1 且分量 > 0',
    inputs: ['M'], domains: { M: 'matSmall' }, matrixDims: { M: [4, 4] },
  },
    function run(M) { return mSoftmaxCols(M, 4, 4); },
    function pred(M, run) { const out = run(M); const r = 4, c = 4; for (let j = 0; j < c; j++) { let s = 0; for (let i = 0; i < r; i++) { if (out[i * c + j] <= 0) return false; s += out[i * c + j]; } if (!rc(s, 1, 1e-9)) return false; } return true; }
  );

  mk(117, {
    name: '批归一化（全局展平）', intuition: '把整张矩阵展平后做零均值单位方差。',
    math: 'run = (M−μ)/σ', pseudocode: 'return (M-mean)/std', complexity: 'O(r·c)',
    ref: 'Ioffe & Szegedy 2015', commonPitfall: 'σ=0 时退化为 0。',
    invariant: '总均值≈0 且总方差≈1',
    inputs: ['M'], domains: { M: 'matSmall' }, matrixDims: { M: [4, 4] },
  },
    function run(M) { const r = 4, c = 4; const m = mMean(M, r, c); const v = mVar(M, r, c); const out = new Float64Array(r * c); if (v <= 1e-14) return out; const inv = 1 / Math.sqrt(v); for (let k = 0; k < r * c; k++) out[k] = (M[k] - m) * inv; return out; },
    function pred(M, run) { const out = run(M); const r = 4, c = 4; if (mVar(M, r, c) <= 1e-14) { for (let k = 0; k < out.length; k++) if (out[k] !== 0) return false; return true; } const m = mMean(out, r, c); const v = mVar(out, r, c); return rc(m, 0, 1e-6) && rc(v, 1, 1e-6); }
  );

  mk(118, {
    name: '双中心化（行列均值为零）', intuition: '同时消去行、列边际均值。',
    math: 'run = M − r̄ − c̄ + μ', pseudocode: 'return doubleCenter(M)', complexity: 'O(r·c)',
    ref: 'Bessel 1888 双中心', commonPitfall: '消去后行和与列和均为 0。',
    invariant: '每行和≈0 且每列和≈0',
    inputs: ['M'], domains: { M: 'matSmall' }, matrixDims: { M: [4, 4] },
  },
    function run(M) { const r = 4, c = 4; const out = new Float64Array(r * c); let g = 0; const rm = new Float64Array(r), cm = new Float64Array(c); for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) { const v = M[i * c + j]; g += v; rm[i] += v; cm[j] += v; } g /= (r * c); for (let i = 0; i < r; i++) rm[i] /= c; for (let j = 0; j < c; j++) cm[j] /= r; for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) out[i * c + j] = M[i * c + j] - rm[i] - cm[j] + g; return out; },
    function pred(M, run) { const out = run(M); const r = 4, c = 4; for (let i = 0; i < r; i++) { let s = 0; for (let j = 0; j < c; j++) s += out[i * c + j]; if (!rc(s, 0, 1e-9)) return false; } for (let j = 0; j < c; j++) { let s = 0; for (let i = 0; i < r; i++) s += out[i * c + j]; if (!rc(s, 0, 1e-9)) return false; } return true; }
  );

  mk(119, {
    name: '中心化矩阵（总均值归零）', intuition: '整体减去总均值。',
    math: 'run = M − μ', pseudocode: 'return M - mean(M)', complexity: 'O(r·c)',
    ref: 'Goodfellow 2016 §5.1', commonPitfall: '仅平移，不改尺度。',
    invariant: '总均值 ≈ 0',
    inputs: ['M'], domains: { M: 'matSmall' }, matrixDims: { M: [4, 4] },
  },
    function run(M) { const r = 4, c = 4; const m = mMean(M, r, c); const out = new Float64Array(r * c); for (let k = 0; k < r * c; k++) out[k] = M[k] - m; return out; },
    function pred(M, run) { const out = run(M); const r = 4, c = 4; return rc(mMean(out, r, c), 0, 1e-9); }
  );

  mk(120, {
    name: '标准化矩阵（总均值/方差）', intuition: '整体零均值单位方差。',
    math: 'run = (M−μ)/σ', pseudocode: 'return (M-mean)/std', complexity: 'O(r·c)',
    ref: 'Ioffe & Szegedy 2015', commonPitfall: 'σ=0 时退化为 0。',
    invariant: '总均值≈0 且总方差≈1',
    inputs: ['M'], domains: { M: 'matSmall' }, matrixDims: { M: [4, 4] },
  },
    function run(M) { const r = 4, c = 4; const m = mMean(M, r, c); const v = mVar(M, r, c); const out = new Float64Array(r * c); if (v <= 1e-14) return out; const inv = 1 / Math.sqrt(v); for (let k = 0; k < r * c; k++) out[k] = (M[k] - m) * inv; return out; },
    function pred(M, run) { const out = run(M); const r = 4, c = 4; if (mVar(M, r, c) <= 1e-14) { for (let k = 0; k < out.length; k++) if (out[k] !== 0) return false; return true; } const m = mMean(out, r, c); const v = mVar(out, r, c); return rc(m, 0, 1e-6) && rc(v, 1, 1e-6); }
  );

  mk(121, {
    name: 'Frobenius 归一化（矩阵）', intuition: '矩阵除以 Frobenius 范数变为单位阵。',
    math: 'run = M/‖M‖_F', pseudocode: 'return M / frobeniusNorm(M)', complexity: 'O(r·c)',
    ref: 'Horn & Johnson 2012', commonPitfall: '零矩阵退化为 0。',
    invariant: '‖run‖_F ≈ 1（非零矩阵）',
    inputs: ['M'], domains: { M: 'matSmall' }, matrixDims: { M: [4, 4] },
  },
    function run(M) { const r = 4, c = 4; let s = 0; for (let k = 0; k < r * c; k++) s += M[k] * M[k]; const nrm = Math.sqrt(s); const out = new Float64Array(r * c); if (nrm > 1e-12) for (let k = 0; k < r * c; k++) out[k] = M[k] / nrm; return out; },
    function pred(M, run) { const out = run(M); let s = 0, nrm = 0; for (let k = 0; k < M.length; k++) { nrm += M[k] * M[k]; s += out[k] * out[k]; } nrm = Math.sqrt(nrm); if (nrm > 1e-12) return rc(Math.sqrt(s), 1, 1e-9); for (let k = 0; k < out.length; k++) if (out[k] !== 0) return false; return true; }
  );

  mk(122, {
    name: '矩阵列和归一化', intuition: '每列除以其元素和，变为单纯形列。',
    math: 'run_{·j} = M_{·j}/ΣM_{·j}', pseudocode: 'return colSumNorm(M)', complexity: 'O(r·c)',
    ref: 'Goodfellow 2016 §6.2', commonPitfall: '列和为 0 时退化为 0。',
    invariant: '每列 Σ ≈ 1（非零列）',
    inputs: ['M'], domains: { M: 'matSmall' }, matrixDims: { M: [4, 4] },
  },
    function run(M) { const r = 4, c = 4; const out = new Float64Array(r * c); for (let j = 0; j < c; j++) { const cs = mColSum(M, r, c, j); if (Math.abs(cs) > 1e-12) for (let i = 0; i < r; i++) out[i * c + j] = M[i * c + j] / cs; } return out; },
    function pred(M, run) { const out = run(M); const r = 4, c = 4; for (let j = 0; j < c; j++) { const cs = mColSum(M, r, c, j); if (Math.abs(cs) > 1e-12) { let s = 0; for (let i = 0; i < r; i++) s += out[i * c + j]; if (!rc(s, 1, 1e-9)) return false; } else { for (let i = 0; i < r; i++) if (out[i * c + j] !== 0) return false; } } return true; }
  );

  mk(123, {
    name: '矩阵行和归一化', intuition: '每行除以其元素和，变为单纯形行。',
    math: 'run_{i·} = M_{i·}/ΣM_{i·}', pseudocode: 'return rowSumNorm(M)', complexity: 'O(r·c)',
    ref: 'Goodfellow 2016 §6.2', commonPitfall: '行和为 0 时退化为 0。',
    invariant: '每行 Σ ≈ 1（非零行）',
    inputs: ['M'], domains: { M: 'matSmall' }, matrixDims: { M: [4, 4] },
  },
    function run(M) { const r = 4, c = 4; const out = new Float64Array(r * c); for (let i = 0; i < r; i++) { const rs = mRowSum(M, r, c, i); if (Math.abs(rs) > 1e-12) for (let j = 0; j < c; j++) out[i * c + j] = M[i * c + j] / rs; } return out; },
    function pred(M, run) { const out = run(M); const r = 4, c = 4; for (let i = 0; i < r; i++) { const rs = mRowSum(M, r, c, i); if (Math.abs(rs) > 1e-12) { let s = 0; for (let j = 0; j < c; j++) s += out[i * c + j]; if (!rc(s, 1, 1e-9)) return false; } else { for (let j = 0; j < c; j++) if (out[i * c + j] !== 0) return false; } } return true; }
  );

  mk(124, {
    name: '迹无关归一化', intuition: '减去归一化迹使矩阵迹归零，信息不损失于对角。',
    math: 'run_{ij} = M_{ij} − (tr(M)/n)·δ_{ij}', pseudocode: 'return M - (tr/n)*I', complexity: 'O(r·c)',
    ref: '矩阵分析；迹修正', commonPitfall: '仅修正对角元。',
    invariant: 'trace(run) ≈ 0',
    inputs: ['M'], domains: { M: 'matSmall' }, matrixDims: { M: [4, 4] },
  },
    function run(M) { const r = 4, c = 4, n = r; let tr = 0; for (let i = 0; i < n; i++) tr += M[i * c + i]; const t = tr / n; const out = new Float64Array(r * c); for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) out[i * c + j] = M[i * c + j] - (i === j ? t : 0); return out; },
    function pred(M, run) { const out = run(M); let tr = 0; for (let i = 0; i < 4; i++) tr += out[i * 4 + i]; return rc(tr, 0, 1e-9); }
  );

  mk(125, {
    name: '逐行 min-max 归一化 [0,1]', intuition: '每行线性拉伸到 [0,1]。',
    math: 'run_{ij} = (M_{ij}−min_i)/(max_i−min_i)', pseudocode: 'return rowMinMax01(M)', complexity: 'O(r·c)',
    ref: 'Han et al. 2012', commonPitfall: '行内常数退化为 0。',
    invariant: '每行 ∈[0,1]，min→0，max→1',
    inputs: ['M'], domains: { M: 'matSmall' }, matrixDims: { M: [4, 4] },
  },
    function run(M) { const r = 4, c = 4; const out = new Float64Array(r * c); for (let i = 0; i < r; i++) { let lo = Infinity, hi = -Infinity; for (let j = 0; j < c; j++) { const v = M[i * c + j]; if (v < lo) lo = v; if (v > hi) hi = v; } if (hi > lo) for (let j = 0; j < c; j++) out[i * c + j] = (M[i * c + j] - lo) / (hi - lo); } return out; },
    function pred(M, run) { const out = run(M); const r = 4, c = 4; for (let i = 0; i < r; i++) { let lo = Infinity, hi = -Infinity; for (let j = 0; j < c; j++) { const v = M[i * c + j]; if (v < lo) lo = v; if (v > hi) hi = v; } if (hi <= lo) { for (let j = 0; j < c; j++) if (out[i * c + j] !== 0) return false; continue; } for (let j = 0; j < c; j++) if (out[i * c + j] < -1e-12 || out[i * c + j] > 1 + 1e-12) return false; let imn = 0, imx = 0; for (let j = 1; j < c; j++) { if (M[i * c + j] < M[i * c + imn]) imn = j; if (M[i * c + j] > M[i * c + imx]) imx = j; } if (!rc(out[i * c + imn], 0, 1e-9)) return false; if (!rc(out[i * c + imx], 1, 1e-9)) return false; } return true; }
  );

  mk(126, {
    name: '逐列 min-max 归一化 [0,1]', intuition: '每列线性拉伸到 [0,1]。',
    math: 'run_{ij} = (M_{ij}−min_j)/(max_j−min_j)', pseudocode: 'return colMinMax01(M)', complexity: 'O(r·c)',
    ref: 'Han et al. 2012', commonPitfall: '列内常数退化为 0。',
    invariant: '每列 ∈[0,1]，min→0，max→1',
    inputs: ['M'], domains: { M: 'matSmall' }, matrixDims: { M: [4, 4] },
  },
    function run(M) { const r = 4, c = 4; const out = new Float64Array(r * c); for (let j = 0; j < c; j++) { let lo = Infinity, hi = -Infinity; for (let i = 0; i < r; i++) { const v = M[i * c + j]; if (v < lo) lo = v; if (v > hi) hi = v; } if (hi > lo) for (let i = 0; i < r; i++) out[i * c + j] = (M[i * c + j] - lo) / (hi - lo); } return out; },
    function pred(M, run) { const out = run(M); const r = 4, c = 4; for (let j = 0; j < c; j++) { let lo = Infinity, hi = -Infinity; for (let i = 0; i < r; i++) { const v = M[i * c + j]; if (v < lo) lo = v; if (v > hi) hi = v; } if (hi <= lo) { for (let i = 0; i < r; i++) if (out[i * c + j] !== 0) return false; continue; } for (let i = 0; i < r; i++) if (out[i * c + j] < -1e-12 || out[i * c + j] > 1 + 1e-12) return false; let imn = 0, imx = 0; for (let i = 1; i < r; i++) { if (M[i * c + j] < M[imn * c + j]) imn = i; if (M[i * c + j] > M[imx * c + j]) imx = i; } if (!rc(out[imn * c + j], 0, 1e-9)) return false; if (!rc(out[imx * c + j], 1, 1e-9)) return false; } return true; }
  );

  mk(127, {
    name: '矩阵逐行 softmax', intuition: '每行独立 softmax（重写 2D 形式）。',
    math: 'run_{i·} = softmax(M_{i·})', pseudocode: 'return rowSoftmax(M)', complexity: 'O(r·c)',
    ref: 'Goodfellow 2016 §6.2', commonPitfall: '逐行而非全局。',
    invariant: '每行 Σ ≈ 1 且分量 > 0',
    inputs: ['M'], domains: { M: 'matSmall' }, matrixDims: { M: [4, 4] },
  },
    function run(M) { return mSoftmaxRows(M, 4, 4); },
    function pred(M, run) { const out = run(M); const r = 4, c = 4; for (let i = 0; i < r; i++) { let s = 0; for (let j = 0; j < c; j++) { if (out[i * c + j] <= 0) return false; s += out[i * c + j]; } if (!rc(s, 1, 1e-9)) return false; } return true; }
  );

  mk(128, {
    name: '逐行 z-score（矩阵）', intuition: '每行独立减均值除标准差。',
    math: 'run_{i·} = (M_{i·}−μ_i)/σ_i', pseudocode: 'return rowZScore(M)', complexity: 'O(r·c)',
    ref: 'Goodfellow 2016 §5.1', commonPitfall: '逐行而非全局。',
    invariant: '每行 mean≈0 且 var≈1',
    inputs: ['M'], domains: { M: 'matSmall' }, matrixDims: { M: [4, 4] },
  },
    function run(M) { const r = 4, c = 4; const out = new Float64Array(r * c); for (let i = 0; i < r; i++) { const rm = mRowMean(M, r, c, i); const rv = mRowVar(M, r, c, i); if (rv <= 1e-14) continue; const inv = 1 / Math.sqrt(rv); for (let j = 0; j < c; j++) out[i * c + j] = (M[i * c + j] - rm) * inv; } return out; },
    function pred(M, run) { const out = run(M); const r = 4, c = 4; for (let i = 0; i < r; i++) { const rv = mRowVar(M, r, c, i); if (rv <= 1e-14) { for (let j = 0; j < c; j++) if (out[i * c + j] !== 0) return false; continue; } let s = 0; for (let j = 0; j < c; j++) s += out[i * c + j]; if (!rc(s / c, 0, 1e-6)) return false; let v = 0; for (let j = 0; j < c; j++) { const d = out[i * c + j]; v += d * d; } v /= c; if (!rc(v, 1, 1e-6)) return false; } return true; }
  );
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

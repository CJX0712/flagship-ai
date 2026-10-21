/* ============================================================================
 * L10 微指令 · G 域（激活与损失）· 第 3/4 批（G065–G096）
 * ----------------------------------------------------------------------------
 * 所属层：L10-microIsa  域：G（激活与损失）
 * 允许依赖的层：L9（special）、L10（registry/invariant）
 * 主要 @ref：Rajaraman & Ullman (2011) 相似度/距离；Vaswani et al. (2017)
 *           Attention；Pereyra et al. (2017) 标签平滑；Hinton (2015) 温度缩放
 *
 * 涵盖：相似度/距离 G065–G080；注意力相关损失/激活 G081–G096。
 * 约定：集合型（汉明/Jaccard）用 vecMask（{0,1}）；分类下标用 smallInt 取模；
 *       概率用 vecProb；温度用 smallPos。
 * ==========================================================================*/
;(function (NS) {
  'use strict';
  NS.L10 = NS.L10 || {};
  function reg(i, o) { NS.L10.register('G', i, o); }
  var S = NS.L9.special;
  var rc = function (a, b, t) { return NS.L10.relClose(a, b, t); };

  // ── 基元 ────────────────────────────────────────────────────────────────
  function clip(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }
  function mean(v) { var s = 0; for (var i = 0; i < v.length; i++) s += v[i]; return s / v.length; }
  function dotv(a, b) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
  function normv(v) { return Math.sqrt(dotv(v, v)); }
  function argmax(v) { var m = -Infinity, idx = 0; for (var i = 0; i < v.length; i++) if (v[i] > m) { m = v[i]; idx = i; } return idx; }
  function safeIdx(t, n) { t = Math.round(t); t = ((t % n) + n) % n; return t; }
  function logsumexp(v) {
    var m = -Infinity; for (var i = 0; i < v.length; i++) if (v[i] > m) m = v[i];
    var s = 0; for (var i = 0; i < v.length; i++) s += Math.exp(v[i] - m);
    return m + Math.log(s);
  }
  function softmax(v) {
    var m = -Infinity; for (var i = 0; i < v.length; i++) if (v[i] > m) m = v[i];
    var s = 0; var o = new Float64Array(v.length);
    for (var i = 0; i < v.length; i++) { o[i] = Math.exp(v[i] - m); s += o[i]; }
    var inv = 1 / s; for (var j = 0; j < v.length; j++) o[j] *= inv;
    return o;
  }
  function entropy(p) { var s = 0; for (var i = 0; i < p.length; i++) { var pi = Math.max(p[i], 1e-12); s += -pi * Math.log(pi); } return s; }
  function kl(p, q) {
    var s = 0; for (var i = 0; i < p.length; i++) { var pi = Math.max(p[i], 1e-12), qi = Math.max(q[i], 1e-12); s += pi * Math.log(pi / qi); } return s;
  }

  // ════════════════════════ 相似度/距离 G065–G080 ════════════════════════
  var f_dot = dotv;
  reg(65, {
    name: '点积', intuition: '向量内积，相似度与投影的基础。',
    math: 'dot(x,y)=Σx_i y_i', pseudocode: 'return Σx_i y_i', complexity: 'O(n)',
    ref: 'Strang 2016', commonPitfall: '与余弦不同，不受模长归一影响。',
    invariant: 'run(x,y) 等于手动 Σx_i y_i',
    inputs: ['x', 'y'], domains: { x: 'vecSmall', y: 'vecSmall' },
    run: f_dot,
    predicate: function (x, y) { var r = f_dot(x, y); var s = 0; for (var i = 0; i < x.length; i++) s += x[i] * y[i]; return rc(r, s, 1e-9, true); },
  });

  var f_cosine = function (x, y) {
    var nx = normv(x), ny = normv(y); return (nx === 0 || ny === 0) ? 0 : dotv(x, y) / (nx * ny);
  };
  reg(66, {
    name: '余弦相似度', intuition: '方向相似度，对模长不敏感，∈[-1,1]。',
    math: 'cos(x,y)=x·y/(‖x‖‖y‖)', pseudocode: 'return dot/(norm*norm)', complexity: 'O(n)',
    ref: 'Salton et al. 1975', commonPitfall: '零向量无定义，此处约定为 0。',
    invariant: 'run∈[-1,1] 且等于手动余弦',
    inputs: ['x', 'y'], domains: { x: 'vecSmall', y: 'vecSmall' },
    run: f_cosine,
    predicate: function (x, y) {
      var nx = normv(x), ny = normv(y); var refv = (nx === 0 || ny === 0) ? 0 : dotv(x, y) / (nx * ny);
      var r = f_cosine(x, y);
      return r >= -1 - 1e-9 && r <= 1 + 1e-9 && rc(r, refv, 1e-9);
    },
  });

  var f_euclid = function (x, y) { var s = 0; for (var i = 0; i < x.length; i++) { var d = x[i] - y[i]; s += d * d; } return Math.sqrt(s); };
  reg(67, {
    name: '欧氏距离', intuition: 'L2 距离，最常用几何距离。',
    math: 'd=‖x-y‖₂', pseudocode: 'return sqrt(Σ(x-y)²)', complexity: 'O(n)',
    ref: 'Strang 2016', commonPitfall: '非负且平移不变。',
    invariant: 'run>=0 且等于手动 L2',
    inputs: ['x', 'y'], domains: { x: 'vecSmall', y: 'vecSmall' },
    run: f_euclid,
    predicate: function (x, y) { var r = f_euclid(x, y); var s = 0; for (var i = 0; i < x.length; i++) { var d = x[i] - y[i]; s += d * d; } return r >= 0 && rc(r, Math.sqrt(s), 1e-9); },
  });

  var f_manhattan = function (x, y) { var s = 0; for (var i = 0; i < x.length; i++) s += Math.abs(x[i] - y[i]); return s; };
  reg(68, {
    name: '曼哈顿距离', intuition: 'L1 距离，出租车距离，对离群点更稳健。',
    math: 'd=Σ|x-y|', pseudocode: 'return Σ|x-y|', complexity: 'O(n)',
    ref: 'Krause 1987', commonPitfall: 'L1>=L2。',
    invariant: 'run>=0 且等于手动 L1',
    inputs: ['x', 'y'], domains: { x: 'vecSmall', y: 'vecSmall' },
    run: f_manhattan,
    predicate: function (x, y) { var r = f_manhattan(x, y); var s = 0; for (var i = 0; i < x.length; i++) s += Math.abs(x[i] - y[i]); return r >= 0 && rc(r, s, 1e-9); },
  });

  var f_cheby = function (x, y) { var m = 0; for (var i = 0; i < x.length; i++) { var d = Math.abs(x[i] - y[i]); if (d > m) m = d; } return m; };
  reg(69, {
    name: '切比雪夫距离', intuition: 'L∞ 距离，最大坐标差。',
    math: 'd=max|x-y|', pseudocode: 'return max|x-y|', complexity: 'O(n)',
    ref: 'Chebyshev 1992', commonPitfall: 'L∞<=L1。',
    invariant: 'run>=0 且等于手动 L∞',
    inputs: ['x', 'y'], domains: { x: 'vecSmall', y: 'vecSmall' },
    run: f_cheby,
    predicate: function (x, y) { var r = f_cheby(x, y); var m = 0; for (var i = 0; i < x.length; i++) { var d = Math.abs(x[i] - y[i]); if (d > m) m = d; } return r >= 0 && rc(r, m, 1e-9, true); },
  });

  var f_pearson = function (x, y) {
    var mx = mean(x), my = mean(y), n = x.length; var cov = 0, vx = 0, vy = 0;
    for (var i = 0; i < n; i++) { var dx = x[i] - mx, dy = y[i] - my; cov += dx * dy; vx += dx * dx; vy += dy * dy; }
    if (vx === 0 || vy === 0) return 0;
    return cov / Math.sqrt(vx * vy);
  };
  reg(70, {
    name: '皮尔逊相关', intuition: '线性相关性，∈[-1,1]，对仿射变换不变。',
    math: 'ρ=cov(x,y)/(σ_x σ_y)', pseudocode: 'return cov/(sx*sy)', complexity: 'O(n)',
    ref: 'Pearson 1895', commonPitfall: '标准差为 0 时无定义（约定 0）。',
    invariant: '若 std>0 则 |run|<=1，且等于手动 ρ',
    inputs: ['x', 'y'], domains: { x: 'vecSmall', y: 'vecSmall' },
    run: f_pearson,
    predicate: function (x, y) {
      var r = f_pearson(x, y); var mx = mean(x), my = mean(y), n = x.length, cov = 0, vx = 0, vy = 0;
      for (var i = 0; i < n; i++) { var dx = x[i] - mx, dy = y[i] - my; cov += dx * dy; vx += dx * dx; vy += dy * dy; }
      var refv = (vx === 0 || vy === 0) ? 0 : cov / Math.sqrt(vx * vy);
      return Math.abs(r) <= 1 + 1e-9 && rc(r, refv, 1e-9);
    },
  });

  var f_hamming = function (x, y) { var c = 0; for (var i = 0; i < x.length; i++) if (x[i] !== y[i]) c++; return c; };
  reg(71, {
    name: '汉明距离', intuition: '不同位计数，离散向量/集合差异度量。',
    math: 'd=#{i: x_i≠y_i}', pseudocode: 'return count(x!=y)', complexity: 'O(n)',
    ref: 'Hamming 1950', commonPitfall: '仅对离散/二值向量有意义。',
    invariant: 'run∈[0,len] 且等于手动计数',
    inputs: ['x', 'y'], domains: { x: 'vecMask', y: 'vecMask' },
    run: f_hamming,
    predicate: function (x, y) { var r = f_hamming(x, y); var c = 0; for (var i = 0; i < x.length; i++) if (x[i] !== y[i]) c++; return r >= 0 && r <= x.length && rc(r, c, 1e-9, true); },
  });

  var f_jaccard = function (x, y) {
    var inter = 0, uni = 0; for (var i = 0; i < x.length; i++) { var a = x[i] === 1, b = y[i] === 1; if (a || b) { uni++; if (a && b) inter++; } }
    return uni === 0 ? 0 : inter / uni;
  };
  reg(72, {
    name: 'Jaccard', intuition: '集合交并比，衡量二值集合相似度。',
    math: 'J=|A∩B|/|A∪B|', pseudocode: 'return inter/union', complexity: 'O(n)',
    ref: 'Jaccard 1901', commonPitfall: '两空集合时约定为 0。',
    invariant: 'run∈[0,1] 且等于手动 Jaccard',
    inputs: ['x', 'y'], domains: { x: 'vecMask', y: 'vecMask' },
    run: f_jaccard,
    predicate: function (x, y) {
      var r = f_jaccard(x, y); var inter = 0, uni = 0;
      for (var i = 0; i < x.length; i++) { var a = x[i] === 1, b = y[i] === 1; if (a || b) { uni++; if (a && b) inter++; } }
      var refv = uni === 0 ? 0 : inter / uni;
      return r >= 0 - 1e-9 && r <= 1 + 1e-9 && rc(r, refv, 1e-9);
    },
  });

  var f_cosdist = function (x, y) { return 1 - f_cosine(x, y); };
  reg(73, {
    name: '余弦距离', intuition: '余弦相似度的补，∈[0,2]。',
    math: 'd=1-cos(x,y)', pseudocode: 'return 1-cos', complexity: 'O(n)',
    ref: 'Salton et al. 1975', commonPitfall: '非严格度量（不满足三角不等式）。',
    invariant: 'run∈[0,2] 且等于 1-cos',
    inputs: ['x', 'y'], domains: { x: 'vecSmall', y: 'vecSmall' },
    run: f_cosdist,
    predicate: function (x, y) { var r = f_cosdist(x, y); var c = f_cosine(x, y); return r >= -1e-9 && r <= 2 + 1e-9 && rc(r, 1 - c, 1e-9); },
  });

  var f_angdist = function (x, y) { var c = f_cosine(x, y); c = clip(c, -1, 1); return Math.acos(c); };
  reg(74, {
    name: '角距离', intuition: '向量夹角，∈[0,π]。',
    math: 'θ=arccos(cos(x,y))', pseudocode: 'return acos(cos)', complexity: 'O(n)',
    ref: 'Strang 2016', commonPitfall: '余弦越接近 1，角越接近 0。',
    invariant: 'run∈[0,π] 且等于 acos(cos)',
    inputs: ['x', 'y'], domains: { x: 'vecSmall', y: 'vecSmall' },
    run: f_angdist,
    predicate: function (x, y) { var r = f_angdist(x, y); var c = f_cosine(x, y); c = clip(c, -1, 1); var refv = Math.acos(c); return r >= -1e-9 && r <= Math.PI + 1e-9 && rc(r, refv, 1e-9); },
  });

  var f_inprod_sign = function (x, y) { return Math.sign(dotv(x, y)) === Math.sign(f_cosine(x, y)); };
  reg(75, {
    name: '内积符号一致', intuition: '内积符号与余弦符号一致（模长为正）。',
    math: 'sign(x·y)=sign(cos(x,y))', pseudocode: 'return sign(dot)==sign(cos)', complexity: 'O(n)',
    ref: '向量代数', commonPitfall: '零向量时两者均为 0。',
    invariant: 'sign(dot(x,y))===sign(cos(x,y))',
    inputs: ['x', 'y'], domains: { x: 'vecSmall', y: 'vecSmall' },
    run: f_inprod_sign,
    predicate: function (x, y) { return Math.sign(dotv(x, y)) === Math.sign(f_cosine(x, y)); },
  });

  var f_cos_sym = function (x, y) { return f_cosine(x, y); };
  reg(76, {
    name: '余弦对称', intuition: '余弦相似度对称：cos(x,y)=cos(y,x)。',
    math: 'cos(x,y)=cos(y,x)', pseudocode: 'return cos(x,y)==cos(y,x)', complexity: 'O(n)',
    ref: '向量代数', commonPitfall: '无。',
    invariant: 'run(x,y) 与 run(y,x) 逐位相等',
    inputs: ['x', 'y'], domains: { x: 'vecSmall', y: 'vecSmall' },
    run: f_cos_sym,
    predicate: function (x, y) { return rc(f_cosine(x, y), f_cosine(y, x), 1e-9, true); },
  });

  var f_l1geL2 = function (x, y) { return f_manhattan(x, y) >= f_euclid(x, y) - 1e-9; };
  reg(77, {
    name: 'L1≥L2', intuition: '范数不等式：曼哈顿距离不小于欧氏距离。',
    math: '‖x-y‖₁ >= ‖x-y‖₂', pseudocode: 'return L1>=L2', complexity: 'O(n)',
    ref: '范数不等式', commonPitfall: '由 Cauchy–Schwarz 推出。',
    invariant: 'manhattan(x,y) >= euclidean(x,y)',
    inputs: ['x', 'y'], domains: { x: 'vecSmall', y: 'vecSmall' },
    run: f_l1geL2,
    predicate: function (x, y) { return f_manhattan(x, y) >= f_euclid(x, y) - 1e-9; },
  });

  var f_linfLeL1 = function (x, y) { return f_cheby(x, y) <= f_manhattan(x, y) + 1e-9; };
  reg(78, {
    name: 'L∞≤L1', intuition: '最大分量差不超过分量差之和。',
    math: '‖x-y‖∞ <= ‖x-y‖₁', pseudocode: 'return Linf<=L1', complexity: 'O(n)',
    ref: '范数不等式', commonPitfall: '显然：max<=sum。',
    invariant: 'chebyshev(x,y) <= manhattan(x,y)',
    inputs: ['x', 'y'], domains: { x: 'vecSmall', y: 'vecSmall' },
    run: f_linfLeL1,
    predicate: function (x, y) { return f_cheby(x, y) <= f_manhattan(x, y) + 1e-9; },
  });

  var f_cos_self = function (x) { var nx = normv(x); return nx === 0 ? 1 : f_cosine(x, x); };
  reg(79, {
    name: '余弦自相似', intuition: '向量与自身的余弦相似度为 1（非零时）。',
    math: 'cos(x,x)=1 (x≠0)', pseudocode: 'return cos(x,x)', complexity: 'O(n)',
    ref: '向量代数', commonPitfall: '零向量约定返回 1 以避免除零。',
    invariant: '若 ‖x‖>0 则 run(x,x)=1',
    inputs: ['x'], domains: { x: 'vecSmall' },
    run: f_cos_self,
    predicate: function (x) { var nx = normv(x); if (nx === 0) return true; return rc(f_cosine(x, x), 1, 1e-9, true); },
  });

  var f_jac_sym = function (x, y) { return f_jaccard(x, y); };
  reg(80, {
    name: 'Jaccard对称', intuition: 'Jaccard 系数对称：J(x,y)=J(y,x)。',
    math: 'J(x,y)=J(y,x)', pseudocode: 'return J(x,y)==J(y,x)', complexity: 'O(n)',
    ref: 'Jaccard 1901', commonPitfall: '无。',
    invariant: 'run(x,y) 与 run(y,x) 逐位相等',
    inputs: ['x', 'y'], domains: { x: 'vecMask', y: 'vecMask' },
    run: f_jac_sym,
    predicate: function (x, y) { return rc(f_jaccard(x, y), f_jaccard(y, x), 1e-9, true); },
  });

  // ════════════════════════ 注意力相关损失/激活 G081–G096 ════════════════════════
  var f_logistic_stable = function (z, y) { var yb = y[0]; return Math.max(0, z) - z * yb + S.softplus(-Math.abs(z)); };
  reg(81, {
    name: '稳定sigmoid+CE==BCE-logits', intuition: '在 logit 空间稳定计算二分类交叉熵，等价于 BCE(σ(z),y)。',
    math: 'L=max(z,0)-z·y+ln(1+e^{-|z|})', pseudocode: 'return stable term', complexity: 'O(1)',
    ref: 'Paszke et al. 2019', commonPitfall: '避免 σ 饱和处 log(0)。',
    invariant: 'run(z,y) relClose BCE(σ(z),y)',
    inputs: ['z', 'y'], domains: { z: 'small', y: 'vecMask' },
    run: f_logistic_stable,
    predicate: function (z, y) {
      var r = f_logistic_stable(z, y); var p = S.sigmoid(z);
      var bce = -(y[0] * Math.log(Math.max(p, 1e-12)) + (1 - y[0]) * Math.log(Math.max(1 - p, 1e-12)));
      return rc(r, bce, 1e-9);
    },
  });

  var f_log1msig = function (z) { return -S.softplus(z); };
  reg(82, {
    name: '稳定log(1-σ)=-softplus', intuition: 'ln(1-σ(z)) 用 -softplus(z) 稳定计算（防 log(0)）。',
    math: 'ln(1-σ(z))=-softplus(z)', pseudocode: 'return -softplus(z)', complexity: 'O(1)',
    ref: '数值稳定技巧', commonPitfall: '1-σ(z) 直接算在饱和处下溢为 0。',
    invariant: 'run(z) relClose ln(1-σ(z))',
    inputs: ['z'], domains: { z: 'small' },
    run: f_log1msig,
    predicate: function (z) { var r = f_log1msig(z); return rc(r, Math.log(1 - S.sigmoid(z)), 1e-5); },
  });

  var f_logsoftmax = function (z, t) { var k = safeIdx(t, z.length); return z[k] - logsumexp(z); };
  reg(83, {
    name: '稳定log-softmax', intuition: 'log-softmax 经 LSE 稳定计算，目标处值 ≤0。',
    math: 'logsm(z)_t=z_t-LSE(z)', pseudocode: 'return z_t-LSE(z)', complexity: 'O(n)',
    ref: 'Goodfellow 2016', commonPitfall: '直接 log(softmax) 在饱和处失稳。',
    invariant: 'run(z,t) relClose ln(softmax(z)_t) 且 ≤0',
    inputs: ['logits', 'target'], domains: { logits: 'vecSmall', target: 'smallInt' },
    run: f_logsoftmax,
    predicate: function (logits, target) {
      var r = f_logsoftmax(logits, target); var k = safeIdx(target, logits.length);
      var refv = Math.log(Math.max(softmax(logits)[k], 1e-12));
      return r <= 0 + 1e-9 && rc(r, refv, 1e-9);
    },
  });

  var f_labelSmoothCE = function (z, t) {
    var eps = 0.1, n = z.length, k = safeIdx(t, n), sm = softmax(z), q = new Float64Array(n);
    for (var i = 0; i < n; i++) q[i] = (1 - eps) * (i === k ? 1 : 0) + eps / n;
    var s = 0; for (var j = 0; j < n; j++) s += -q[j] * Math.log(Math.max(sm[j], 1e-12));
    return s;
  };
  reg(84, {
    name: '标签平滑CE', intuition: '标签平滑(ε=0.1)缓解过自信，目标为平滑分布。',
    math: 'q=(1-ε)y+ε/n; L=-Σq_i ln p_i', pseudocode: 'return -Σq ln p', complexity: 'O(n)',
    ref: 'Pereyra et al. 2017', commonPitfall: 'ε=0 退化为普通 CE。',
    invariant: 'run>=0 且等于手动平滑 CE',
    inputs: ['logits', 'target'], domains: { logits: 'vecSmall', target: 'smallInt' },
    run: f_labelSmoothCE,
    predicate: function (logits, target) {
      var r = f_labelSmoothCE(logits, target); var eps = 0.1, n = logits.length, k = safeIdx(target, n), sm = softmax(logits), q = new Float64Array(n);
      for (var i = 0; i < n; i++) q[i] = (1 - eps) * (i === k ? 1 : 0) + eps / n;
      var s = 0; for (var j = 0; j < n; j++) s += -q[j] * Math.log(Math.max(sm[j], 1e-12));
      return r >= 0 && rc(r, s, 1e-9);
    },
  });

  var f_weightedCE = function (z, t, w) {
    var k = safeIdx(t, z.length), sm = softmax(z); return -w[k] * Math.log(Math.max(sm[k], 1e-12));
  };
  reg(85, {
    name: '加权CE', intuition: '按类别权重缩放目标类负对数似然。',
    math: 'L=-w_y ln p_y', pseudocode: 'return -w_y*ln(p_y)', complexity: 'O(n)',
    ref: 'Cui et al. 2019 (类别均衡)', commonPitfall: '权重须非负。',
    invariant: 'run>=0；且 w 全 1 时等于普通 CE',
    inputs: ['logits', 'target', 'w'], domains: { logits: 'vecSmall', target: 'smallInt', w: 'vecPos' },
    run: f_weightedCE,
    predicate: function (logits, target, w) {
      var r = f_weightedCE(logits, target, w); var k = safeIdx(target, logits.length), sm = softmax(logits);
      var refv = -w[k] * Math.log(Math.max(sm[k], 1e-12));
      if (!(r >= 0 && rc(r, refv, 1e-9))) return false;
      var wu = new Float64Array(w.length); wu.fill(1);
      return rc(f_weightedCE(logits, target, wu), f_logsoftmax(logits, target) * -1, 1e-9);
    },
  });

  var f_softmaxT = function (z, T) {
    var n = z.length, o = new Float64Array(n);
    for (var i = 0; i < n; i++) o[i] = z[i] / T;
    return softmax(o);
  };
  reg(86, {
    name: '温度缩放softmax Σ=1', intuition: '温度缩放 softmax 仍为合法概率分布，和为 1。',
    math: 'p_i=exp(z_i/T)/Σexp(z_j/T)', pseudocode: 'return softmax(z/T)', complexity: 'O(n)',
    ref: 'Hinton et al. 2015', commonPitfall: 'T→0 趋近 one-hot，T→∞ 趋近均匀。',
    invariant: 'Σrun(z,T)=1 且 run>=0',
    inputs: ['logits', 'T'], domains: { logits: 'vecSmall', T: 'smallPos' },
    run: f_softmaxT,
    predicate: function (logits, T) {
      var sm = f_softmaxT(logits, T); var s = 0; for (var i = 0; i < sm.length; i++) { if (!(sm[i] >= 0)) return false; s += sm[i]; }
      return rc(s, 1, 1e-9);
    },
  });

  var f_gumbel = f_softmaxT; // 单调收敛视角：softmax(z/T)
  reg(87, {
    name: 'gumbel-softmax单调收敛', intuition: 'gumbel-softmax 在 T→0 收敛到 one-hot：输出随 logit 严格单调。',
    math: 'y_i=softmax(z_i/T): z_i>z_j⇒y_i>y_j', pseudocode: 'return softmax(z/T)', complexity: 'O(n)',
    ref: 'Jang et al. 2017', commonPitfall: '完整版须加 Gumbel 噪声（此处验证结构单调性）。',
    invariant: 'z_i>z_j ⇒ run_i>run_j 且 Σ=1',
    inputs: ['logits', 'T'], domains: { logits: 'vecSmall', T: 'smallPos' },
    run: f_gumbel,
    predicate: function (logits, T) {
      var sm = f_gumbel(logits, T); var s = 0;
      var m = -Infinity; for (var k = 0; k < logits.length; k++) if (logits[k] > m) m = logits[k];
      for (var i = 0; i < logits.length; i++) {
        if (!(sm[i] >= 0)) return false; s += sm[i];
        for (var j = 0; j < logits.length; j++) {
          if (logits[i] > logits[j]) {
            // 精确序：(z_i-m)/T > (z_j-m)/T ⇔ z_i>z_j（T>0），仅在可解析时才断言严格输出序
            var si = (logits[i] - m) / T, sj = (logits[j] - m) / T;
            if (si - sj > 1e-6 && si > -740) { if (!(sm[i] > sm[j])) return false; }
          }
        }
      }
      return rc(s, 1, 1e-9);
    },
  });

  var f_selfinfo = function (p, k) { var idx = safeIdx(k, p.length); return -Math.log(Math.max(p[idx], 1e-12)); };
  reg(88, {
    name: '自信息≥0', intuition: '事件自信息 -ln p_x 非负。',
    math: 'I(x)=-ln p_x>=0', pseudocode: 'return -ln(p[k])', complexity: 'O(1)',
    ref: 'Shannon 1948', commonPitfall: 'p=0 须裁剪。',
    invariant: 'run(p,k)>=0 且等于 -ln p_k',
    inputs: ['p', 'k'], domains: { p: 'vecProb', k: 'smallInt' },
    run: f_selfinfo,
    predicate: function (p, k) { var r = f_selfinfo(p, k); var idx = safeIdx(k, p.length); return r >= 0 && rc(r, -Math.log(Math.max(p[idx], 1e-12)), 1e-9); },
  });

  var f_entropy = entropy;
  reg(89, {
    name: '香农熵≥0,one-hot→0', intuition: '熵衡量不确定性，one-hot 分布熵为 0。',
    math: 'H(p)=-Σp ln p>=0', pseudocode: 'return entropy(p)', complexity: 'O(n)',
    ref: 'Shannon 1948', commonPitfall: '裁剪避免 0·ln0。',
    invariant: 'run(p)>=0；one-hot 时 relClose 0',
    inputs: ['p'], domains: { p: 'vecProb' },
    run: f_entropy,
    predicate: function (p) {
      var r = f_entropy(p); if (!(r >= 0)) return false;
      var oh = new Float64Array(p.length); oh[safeIdx(0, p.length)] = 1; // 任意 one-hot 均熵 0
      return rc(f_entropy(oh), 0, 1e-9);
    },
  });

  var f_ce_pq = function (p, q) { var s = 0; for (var i = 0; i < p.length; i++) s += -p[i] * Math.log(Math.max(q[i], 1e-12)); return s; };
  reg(90, {
    name: '交叉熵≥熵', intuition: '交叉熵 = 熵 + KL ≥ 熵。',
    math: 'H(p,q)>=H(p)', pseudocode: 'return run>=entropy(p)', complexity: 'O(n)',
    ref: 'Cover & Thomas 2006', commonPitfall: '由 KL>=0 推出。',
    invariant: 'run(p,q) >= entropy(p)',
    inputs: ['p', 'q'], domains: { p: 'vecProb', q: 'vecProb' },
    run: f_ce_pq,
    predicate: function (p, q) { var r = f_ce_pq(p, q); return r >= f_entropy(p) - 1e-9; },
  });

  var f_nmi = function (p, q) { var d = kl(p, q) + kl(q, p); return 1 / (1 + d); };
  reg(91, {
    name: '归一化互信息[0,1]', intuition: '标准化互信息界定在 (0,1]，p=q 时为 1（对称）。',
    math: 'NMI=1/(1+KL(p‖q)+KL(q‖p))', pseudocode: 'return 1/(1+KL+KL)', complexity: 'O(n)',
    ref: 'Strehl & Ghosh 2002', commonPitfall: '此处定义为对称有界量。',
    invariant: '0<run<=1 且 p==q 时为 1',
    inputs: ['p', 'q'], domains: { p: 'vecProb', q: 'vecProb' },
    run: f_nmi,
    predicate: function (p, q) {
      var r = f_nmi(p, q); if (!(r > 0 && r <= 1 + 1e-9)) return false;
      return rc(f_nmi(p, p), 1, 1e-9);
    },
  });

  var f_perplexity = function (p) { return Math.exp(f_entropy(p)); };
  reg(92, {
    name: '困惑度≥1', intuition: '困惑度 = 2^H（自然底数），熵非负故 ≥1。',
    math: 'PPL=exp(H(p))>=1', pseudocode: 'return exp(entropy(p))', complexity: 'O(n)',
    ref: 'Jelinek et al. 1977', commonPitfall: '底数选择影响数值但不影响 ≥1。',
    invariant: 'run(p)>=1 且等于 exp(entropy)',
    inputs: ['p'], domains: { p: 'vecProb' },
    run: f_perplexity,
    predicate: function (p) { var r = f_perplexity(p); return r >= 1 - 1e-9 && rc(r, Math.exp(f_entropy(p)), 1e-9); },
  });

  var f_tempMono = f_softmaxT;
  reg(93, {
    name: '温度缩放概率单调性', intuition: 'softmax(z/T) 的排序与 z 一致，且为分布。',
    math: 'z_i>z_j ⇒ softmax(z/T)_i>softmax(z/T)_j', pseudocode: 'return softmax(z/T)', complexity: 'O(n)',
    ref: 'Hinton et al. 2015', commonPitfall: '仅排序保持，绝对值随 T 变化。',
    invariant: 'z_i>z_j ⇒ run_i>run_j 且 Σ=1',
    inputs: ['logits', 'T'], domains: { logits: 'vecSmall', T: 'smallPos' },
    run: f_tempMono,
    predicate: function (logits, T) {
      var sm = f_tempMono(logits, T); var s = 0;
      var m = -Infinity; for (var k = 0; k < logits.length; k++) if (logits[k] > m) m = logits[k];
      for (var i = 0; i < logits.length; i++) {
        if (!(sm[i] >= 0)) return false; s += sm[i];
        for (var j = 0; j < logits.length; j++) {
          if (logits[i] > logits[j]) {
            var si = (logits[i] - m) / T, sj = (logits[j] - m) / T;
            if (si - sj > 1e-6 && si > -740) { if (!(sm[i] > sm[j])) return false; }
          }
        }
      }
      return rc(s, 1, 1e-9);
    },
  });

  reg(94, {
    name: '自信息=熵期望', intuition: '熵是各事件自信息的期望：H(p)=Σp_i I(p_i)。',
    math: 'H(p)=Σp_i(-ln p_i)', pseudocode: 'return Σp_i I(p_i)', complexity: 'O(n)',
    ref: 'Shannon 1948', commonPitfall: '逐项恒等。',
    invariant: 'run(p) relClose Σp_i(-ln p_i)',
    inputs: ['p'], domains: { p: 'vecProb' },
    run: f_entropy,
    predicate: function (p) {
      var r = f_entropy(p); var s = 0; for (var i = 0; i < p.length; i++) s += p[i] * (-Math.log(Math.max(p[i], 1e-12)));
      return rc(r, s, 1e-9);
    },
  });

  reg(95, {
    name: 'CE=H+KL', intuition: '交叉熵分解为熵加 KL 散度。',
    math: 'H(p,q)=H(p)+KL(p‖q)', pseudocode: 'return entropy(p)+kl(p,q)', complexity: 'O(n)',
    ref: 'Cover & Thomas 2006', commonPitfall: '处处非负的加性分解。',
    invariant: 'run(p,q) relClose H(p)+KL(p,q)',
    inputs: ['p', 'q'], domains: { p: 'vecProb', q: 'vecProb' },
    run: f_ce_pq,
    predicate: function (p, q) { var r = f_ce_pq(p, q); return rc(r, f_entropy(p) + kl(p, q), 1e-9); },
  });

  var f_softmaxShift = function (z) { return softmax(z); };
  reg(96, {
    name: 'softmax常数平移不变', intuition: 'softmax 对输入整体加常数不变（数值稳定核心）。',
    math: 'softmax(z+c)=softmax(z)', pseudocode: 'return softmax(z)==softmax(z+c)', complexity: 'O(n)',
    ref: 'Bridle 1990', commonPitfall: '逐元素平移不改变分布。',
    invariant: 'run(z) relClose run(z+c)（c=3）',
    inputs: ['z'], domains: { z: 'vecSmall' },
    run: f_softmaxShift,
    predicate: function (z) {
      var a = f_softmaxShift(z); var b = new Float64Array(z.length);
      for (var i = 0; i < z.length; i++) b[i] = z[i] + 3;
      return rc(a, softmax(b), 1e-9);
    },
  });
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

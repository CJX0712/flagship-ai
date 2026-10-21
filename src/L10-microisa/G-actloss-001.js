/* ============================================================================
 * L10 微指令 · G 域（激活与损失）· 第 2/4 批（G033–G064）
 * ----------------------------------------------------------------------------
 * 所属层：L10-microIsa  域：G（激活与损失）
 * 允许依赖的层：L9（special）、L10（registry/invariant）
 * 主要 @ref：Paszke et al. (2019) PyTorch Loss；Goodfellow (2016)；
 *           Lin et al. (2017) Focal Loss；Girshick (2015) Hinge/SmoothL1
 *
 * 涵盖：损失函数 G033–G048；损失性质 G049–G064。
 * 约定：二值标签取自 vecMask（值∈{0,1}），读取 [0] 并映射 ±1；分类目标下标取自
 *       smallInt 并经 safeIdx 取模到向量长度；概率向量用 vecProb（单纯形，已归一）。
 * ==========================================================================*/
;(function (NS) {
  'use strict';
  NS.L10 = NS.L10 || {};
  function reg(i, o) { NS.L10.register('G', i, o); }
  var S = NS.L9.special;
  var rc = function (a, b, t) { return NS.L10.relClose(a, b, t); };

  // ── 损失基元 ────────────────────────────────────────────────────────────
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
  function js(p, q) {
    var n = p.length; var m = new Float64Array(n);
    for (var i = 0; i < n; i++) m[i] = (p[i] + q[i]) / 2;
    return 0.5 * kl(p, m) + 0.5 * kl(q, m);
  }
  function signLabel(y) { return y === 1 ? 1 : -1; } // vecMask[0]∈{0,1} → ±1

  // ════════════════════════ 损失函数 G033–G048 ════════════════════════
  var f_mse = function (y, yh) { var s = 0; for (var i = 0; i < y.length; i++) { var e = y[i] - yh[i]; s += e * e; } return s / y.length; };
  reg(33, {
    name: 'MSE', intuition: '均方误差，回归最常用的距离型损失。',
    math: 'MSE=mean((y-ŷ)²)', pseudocode: 'return mean((y-yh)^2)', complexity: 'O(n)',
    ref: 'Goodfellow 2016 §6.2', commonPitfall: '对离群值敏感。',
    invariant: 'run(y,yh)>=0 且等于手动实现',
    inputs: ['y', 'yh'], domains: { y: 'vecSmall', yh: 'vecSmall' },
    run: f_mse,
    predicate: function (y, yh) { var r = f_mse(y, yh); var s = 0; for (var i = 0; i < y.length; i++) { var e = y[i] - yh[i]; s += e * e; } return r >= 0 && rc(r, s / y.length, 1e-9); },
  });

  var f_mae = function (y, yh) { var s = 0; for (var i = 0; i < y.length; i++) s += Math.abs(y[i] - yh[i]); return s / y.length; };
  reg(34, {
    name: 'MAE', intuition: '平均绝对误差，对离群值更稳健。',
    math: 'MAE=mean|y-ŷ|', pseudocode: 'return mean(|y-yh|)', complexity: 'O(n)',
    ref: 'Goodfellow 2016 §6.2', commonPitfall: '在 0 处不可导。',
    invariant: 'run(y,yh)>=0 且等于手动实现',
    inputs: ['y', 'yh'], domains: { y: 'vecSmall', yh: 'vecSmall' },
    run: f_mae,
    predicate: function (y, yh) { var r = f_mae(y, yh); var s = 0; for (var i = 0; i < y.length; i++) s += Math.abs(y[i] - yh[i]); return r >= 0 && rc(r, s / y.length, 1e-9); },
  });

  var f_rmse = function (y, yh) { return Math.sqrt(f_mse(y, yh)); };
  reg(35, {
    name: 'RMSE', intuition: '均方根误差，量纲与目标一致。',
    math: 'RMSE=√MSE', pseudocode: 'return sqrt(MSE)', complexity: 'O(n)',
    ref: 'Goodfellow 2016', commonPitfall: '单位与 MSE 不同。',
    invariant: 'run(y,yh)>=0 且等于 √MSE',
    inputs: ['y', 'yh'], domains: { y: 'vecSmall', yh: 'vecSmall' },
    run: f_rmse,
    predicate: function (y, yh) { var r = f_rmse(y, yh); return r >= 0 && rc(r, Math.sqrt(f_mse(y, yh)), 1e-9); },
  });

  var f_huber = function (y, yh) {
    var s = 0; for (var i = 0; i < y.length; i++) { var e = y[i] - yh[i], ae = Math.abs(e); s += ae <= 1 ? 0.5 * e * e : ae - 0.5; } return s / y.length;
  };
  reg(36, {
    name: 'Huber', intuition: 'δ=1 的 Huber 损失，小误差平方、大误差线性，兼顾稳健与平滑。',
    math: 'L_δ=0.5e² (|e|≤δ) else δ(|e|-0.5δ)', pseudocode: 'return mean(piecewise)', complexity: 'O(n)',
    ref: 'Huber 1964', commonPitfall: 'δ 须与实现一致（此处 δ=1）。',
    invariant: 'run(y,yh)>=0 且等于分段手动实现',
    inputs: ['y', 'yh'], domains: { y: 'vecSmall', yh: 'vecSmall' },
    run: f_huber,
    predicate: function (y, yh) {
      var r = f_huber(y, yh); var s = 0;
      for (var i = 0; i < y.length; i++) { var e = y[i] - yh[i], ae = Math.abs(e); s += ae <= 1 ? 0.5 * e * e : ae - 0.5; }
      return r >= 0 && rc(r, s / y.length, 1e-9);
    },
  });

  var f_smoothl1 = function (y, yh) {
    var s = 0; for (var i = 0; i < y.length; i++) { var e = y[i] - yh[i], ae = Math.abs(e); s += ae < 1 ? 0.5 * e * e : ae - 0.5; } return s / y.length;
  };
  reg(37, {
    name: 'SmoothL1', intuition: 'Smooth L1：|e|<1 用平方、否则线性，避免 Huber 在 0 处导数跳变。',
    math: 'L=0.5e² (|e|<1) else |e|-0.5', pseudocode: 'return mean(piecewise)', complexity: 'O(n)',
    ref: 'Girshick 2015 (Fast R-CNN)', commonPitfall: '与 Huber 差别在边界判定（< vs ≤）。',
    invariant: 'run(y,yh)>=0 且等于分段手动实现',
    inputs: ['y', 'yh'], domains: { y: 'vecSmall', yh: 'vecSmall' },
    run: f_smoothl1,
    predicate: function (y, yh) {
      var r = f_smoothl1(y, yh); var s = 0;
      for (var i = 0; i < y.length; i++) { var e = y[i] - yh[i], ae = Math.abs(e); s += ae < 1 ? 0.5 * e * e : ae - 0.5; }
      return r >= 0 && rc(r, s / y.length, 1e-9);
    },
  });

  var f_bce = function (y, p) {
    var s = 0; for (var i = 0; i < y.length; i++) { var pi = Math.max(p[i], 1e-12), oi = Math.max(1 - p[i], 1e-12); s += -(y[i] * Math.log(pi) + (1 - y[i]) * Math.log(oi)); } return s / y.length;
  };
  reg(38, {
    name: 'BCE', intuition: '二分类交叉熵，需 p∈(0,1)（vecProb 归一化后近似满足）。',
    math: 'BCE=-(y ln p+(1-y)ln(1-p))', pseudocode: 'return mean(-(...))', complexity: 'O(n)',
    ref: 'Goodfellow 2016', commonPitfall: 'p=0 或 1 致 log(0) 发散，须裁剪。',
    invariant: 'run>=0 且等于 y·(-ln p)+(1-y)·(-ln(1-p)) 分解',
    inputs: ['y', 'p'], domains: { y: 'vecMask', p: 'vecProb' },
    run: f_bce,
    predicate: function (y, p) {
      var r = f_bce(y, p); var s = 0;
      for (var i = 0; i < y.length; i++) { var pi = Math.max(p[i], 1e-12), oi = Math.max(1 - p[i], 1e-12); s += y[i] * (-Math.log(pi)) + (1 - y[i]) * (-Math.log(oi)); }
      return r >= 0 && rc(r, s / y.length, 1e-9);
    },
  });

  var f_bce_logits = function (z, y) {
    var s = 0; for (var i = 0; i < z.length; i++) { var zi = z[i], yi = y[i]; s += Math.max(0, zi) - zi * yi + S.softplus(-Math.abs(zi)); } return s / z.length;
  };
  reg(39, {
    name: 'BCE-with-logits', intuition: '数值稳定的二分类损失：在 logit 空间计算，避免 σ 饱和。',
    math: 'L=max(z,0)-z·y+ln(1+e^{-|z|})', pseudocode: 'return mean(stable term)', complexity: 'O(n)',
    ref: 'Paszke et al. 2019 (BCEWithLogitsLoss)', commonPitfall: '等价于 BCE(σ(z),y) 但更稳定。',
    invariant: 'run>=0 且等于 BCE(σ(z),y)',
    inputs: ['z', 'y'], domains: { z: 'vecSmall', y: 'vecMask' },
    run: f_bce_logits,
    predicate: function (z, y) {
      var r = f_bce_logits(z, y);
      var p = new Float64Array(z.length); for (var i = 0; i < z.length; i++) p[i] = S.sigmoid(z[i]);
      var bce = f_bce(y, p);
      return r >= 0 && rc(r, bce, 1e-9);
    },
  });

  var f_ce = function (logits, t) {
    var z = logits, k = safeIdx(t, z.length); var sm = softmax(z); return -Math.log(Math.max(sm[k], 1e-12));
  };
  reg(40, {
    name: 'CrossEntropy', intuition: '多分类交叉熵（softmax + NLL），目标为类别下标。',
    math: 'CE=-ln softmax(z)_y', pseudocode: 'return -log(softmax(z)[y])', complexity: 'O(n)',
    ref: 'Goodfellow 2016 §6.2', commonPitfall: '目标须为整型类别下标。',
    invariant: 'run>=0 且等于 -z_y+LSE(z)（logsumexp 路径）',
    inputs: ['logits', 'target'], domains: { logits: 'vecSmall', target: 'smallInt' },
    run: f_ce,
    predicate: function (logits, target) {
      var r = f_ce(logits, target); var k = safeIdx(target, logits.length);
      var refv = -logits[k] + logsumexp(logits);
      return r >= 0 && rc(r, refv, 1e-9);
    },
  });

  var f_nll = function (p, t) { var k = safeIdx(t, p.length); return -Math.log(Math.max(p[k], 1e-12)); };
  reg(41, {
    name: 'NLLLoss', intuition: '负对数似然：对目标类概率取负对数。',
    math: 'NLL=-ln p_y', pseudocode: 'return -log(p[y])', complexity: 'O(1)',
    ref: 'Goodfellow 2016', commonPitfall: '输入须为概率（已 softmax），否则非概率度量。',
    invariant: 'run>=0 且等于 -ln p_y',
    inputs: ['p', 'target'], domains: { p: 'vecProb', target: 'smallInt' },
    run: f_nll,
    predicate: function (p, target) { var r = f_nll(p, target); var k = safeIdx(target, p.length); return r >= 0 && rc(r, -Math.log(Math.max(p[k], 1e-12)), 1e-9); },
  });

  var f_hinge = function (z, y) { var yb = signLabel(y[0]); var v = 1 - yb * z; return v > 0 ? v : 0; };
  reg(42, {
    name: 'Hinge', intuition: '合页损失，SVM 与二分类 Margin 损失基础。',
    math: 'L=max(0,1-y·s), y∈{-1,1}', pseudocode: 'return max(0,1-y*s)', complexity: 'O(1)',
    ref: 'Crammer & Singer 2001', commonPitfall: 'y 须为 ±1（此处由 vecMask 映射）。',
    invariant: 'run(z,y)>=0 且等于 max(0,1-sign(y)s)',
    inputs: ['z', 'label'], domains: { z: 'small', label: 'vecMask' },
    run: f_hinge,
    predicate: function (z, label) { var yb = signLabel(label[0]); var r = f_hinge(z, label); var refv = (1 - yb * z) > 0 ? (1 - yb * z) : 0; return r >= 0 && rc(r, refv, 1e-9, true); },
  });

  var f_sqhinge = function (z, y) { var yb = signLabel(y[0]); var v = 1 - yb * z; var m = v > 0 ? v : 0; return m * m; };
  reg(43, {
    name: 'SquaredHinge', intuition: '平方合页，对大边际违例施加更强惩罚。',
    math: 'L=max(0,1-y·s)²', pseudocode: 'return max(0,1-y*s)^2', complexity: 'O(1)',
    ref: 'Chang & Lin 2011', commonPitfall: '梯度在边界处连续（对比 Hinge 的 0/1 跳变）。',
    invariant: 'run>=0 且等于 max(0,1-y s)²',
    inputs: ['z', 'label'], domains: { z: 'small', label: 'vecMask' },
    run: f_sqhinge,
    predicate: function (z, label) { var yb = signLabel(label[0]); var r = f_sqhinge(z, label); var v = 1 - yb * z; var m = v > 0 ? v : 0; return r >= 0 && rc(r, m * m, 1e-9, true); },
  });

  var f_focal = function (z, y) {
    var yb = signLabel(y[0]); var pt = yb === 1 ? S.sigmoid(z) : S.sigmoid(-z);
    var ce = -Math.log(Math.max(pt, 1e-12));
    return -(1 - pt) * (1 - pt) * Math.log(Math.max(pt, 1e-12));
  };
  reg(44, {
    name: 'Focal', intuition: 'Focal Loss(γ=2)：对易分样本降权，聚焦难例。',
    math: 'FL=-(1-p_t)² ln p_t, p_t=σ(z) 或 σ(-z)', pseudocode: 'return -(1-pt)^2*ln(pt)', complexity: 'O(1)',
    ref: 'Lin et al. 2017', commonPitfall: 'FL 一般 ≥0 且 ≤ 标准 CE（并非恒 ∈[0,1]）。',
    invariant: '0<=run<=CE(=-ln p_t)',
    inputs: ['z', 'label'], domains: { z: 'small', label: 'vecMask' },
    run: f_focal,
    predicate: function (z, label) {
      var yb = signLabel(label[0]); var pt = yb === 1 ? S.sigmoid(z) : S.sigmoid(-z);
      var r = f_focal(z, label); var ce = -Math.log(Math.max(pt, 1e-12));
      return r >= -1e-12 && r <= ce + 1e-9;
    },
  });

  var f_kl = kl;
  reg(45, {
    name: 'KL离散', intuition: '离散 KL 散度，衡量两分布的额外编码代价。',
    math: 'KL(p‖q)=Σp ln(p/q)', pseudocode: 'return sum(p*ln(p/q))', complexity: 'O(n)',
    ref: 'Cover & Thomas 2006', commonPitfall: '非对称、非负，p=q 时为 0。',
    invariant: 'run>=0 且 p=q 时 relClose 0',
    inputs: ['p', 'q'], domains: { p: 'vecProb', q: 'vecProb' },
    run: f_kl,
    predicate: function (p, q) {
      var r = f_kl(p, q);
      if (!(r >= 0 && Number.isFinite(r))) return false;
      return rc(f_kl(p, p), 0, 1e-9);
    },
  });

  var f_js = js;
  reg(46, {
    name: 'JS离散', intuition: 'Jensen–Shannon 散度，对称且有界，适合度量分布相似度。',
    math: 'JS=0.5KL(p‖m)+0.5KL(q‖m), m=(p+q)/2', pseudocode: 'return 0.5*KL(p,m)+0.5*KL(q,m)', complexity: 'O(n)',
    ref: 'Lin 1991', commonPitfall: '上界为 ln2。',
    invariant: '0<=run<=ln2，且 p=q 时为 0、对称',
    inputs: ['p', 'q'], domains: { p: 'vecProb', q: 'vecProb' },
    run: f_js,
    predicate: function (p, q) {
      var r = f_js(p, q);
      if (!(r >= 0 && r <= Math.log(2) + 1e-9)) return false;
      if (!rc(f_js(p, p), 0, 1e-9)) return false;
      return rc(r, f_js(q, p), 1e-9);
    },
  });

  var f_cosEmb = function (x, y, label) {
    var nx = normv(x), ny = normv(y); var c = (nx === 0 || ny === 0) ? 0 : dotv(x, y) / (nx * ny);
    return label[0] === 1 ? 1 - c : 1 + c;
  };
  reg(47, {
    name: 'CosineEmbedding', intuition: '余弦嵌入损失：同类拉近、异类推远。',
    math: 'L=1-cos (同) else 1+cos', pseudocode: 'return label?1-cos:1+cos', complexity: 'O(n)',
    ref: 'PyTorch CosineEmbeddingLoss', commonPitfall: 'label 只能取 0/1。',
    invariant: 'run∈[0,2] 且等于 1∓cos(x,y)',
    inputs: ['x', 'y', 'label'], domains: { x: 'vecSmall', y: 'vecSmall', label: 'vecMask' },
    run: f_cosEmb,
    predicate: function (x, y, label) {
      var nx = normv(x), ny = normv(y); var c = (nx === 0 || ny === 0) ? 0 : dotv(x, y) / (nx * ny);
      var r = f_cosEmb(x, y, label); var refv = label[0] === 1 ? 1 - c : 1 + c;
      return r >= 0 - 1e-12 && r <= 2 + 1e-9 && rc(r, refv, 1e-9);
    },
  });

  var f_contrastive = function (x, y, label, margin) {
    var d = normv(x) === 0 && normv(y) === 0 ? 0 : Math.sqrt(Math.max(0, dotv(x, x) + dotv(y, y) - 2 * dotv(x, y)));
    return label[0] === 1 ? 0.5 * d * d : 0.5 * Math.max(0, margin - d) * Math.max(0, margin - d);
  };
  reg(48, {
    name: 'Contrastive', intuition: '对比损失：同类收缩距离、异类维持至少 margin 间隔。',
    math: 'L=½d² (同) else ½max(0,m-d)²', pseudocode: 'return piece', complexity: 'O(n)',
    ref: 'Hadsell et al. 2006', commonPitfall: '异类损失随 d 超 margin 而 0。',
    invariant: 'run>=0；异类时 run<=½margin²',
    inputs: ['x', 'y', 'label', 'margin'], domains: { x: 'vecSmall', y: 'vecSmall', label: 'vecMask', margin: 'smallPos' },
    run: f_contrastive,
    predicate: function (x, y, label, margin) {
      var nx = normv(x), ny = normv(y);
      var d = (nx === 0 && ny === 0) ? 0 : Math.sqrt(Math.max(0, dotv(x, x) + dotv(y, y) - 2 * dotv(x, y)));
      var r = f_contrastive(x, y, label, margin);
      if (!(r >= 0)) return false;
      if (label[0] === 0) return r <= 0.5 * margin * margin + 1e-9;
      return true;
    },
  });

  // ════════════════════════ 损失性质 G049–G064 ════════════════════════
  reg(49, {
    name: 'MSE对称', intuition: 'MSE 关于预测/真值对称，体现平方误差无方向性。',
    math: 'MSE(y,ŷ)=MSE(ŷ,y)', pseudocode: 'return run(y,yh)==run(yh,y)', complexity: 'O(n)',
    ref: '基础性质', commonPitfall: '仅平方误差损失具备该对称性。',
    invariant: 'run(y,yh) 与 run(yh,y) 逐位相等',
    inputs: ['y', 'yh'], domains: { y: 'vecSmall', yh: 'vecSmall' },
    run: f_mse,
    predicate: function (y, yh) { return rc(f_mse(y, yh), f_mse(yh, y), 1e-9, true); },
  });

  reg(50, {
    name: 'MAE对称', intuition: 'MAE 同样对称，绝对值无方向性。',
    math: 'MAE(y,ŷ)=MAE(ŷ,y)', pseudocode: 'return run(y,yh)==run(yh,y)', complexity: 'O(n)',
    ref: '基础性质', commonPitfall: '无。',
    invariant: 'run(y,yh) 与 run(yh,y) 逐位相等',
    inputs: ['y', 'yh'], domains: { y: 'vecSmall', yh: 'vecSmall' },
    run: f_mae,
    predicate: function (y, yh) { return rc(f_mae(y, yh), f_mae(yh, y), 1e-9, true); },
  });

  reg(51, {
    name: 'KL≥0', intuition: 'KL 散度非负（吉布斯不等式），是交叉熵减熵。',
    math: 'KL(p‖q)>=0', pseudocode: 'return run(p,q)>=0', complexity: 'O(n)',
    ref: 'Cover & Thomas 2006', commonPitfall: '为 0 当且仅当 p=q。',
    invariant: 'run(p,q)>=0 且有限',
    inputs: ['p', 'q'], domains: { p: 'vecProb', q: 'vecProb' },
    run: f_kl,
    predicate: function (p, q) { var r = f_kl(p, q); return Number.isFinite(r) && r >= 0; },
  });

  reg(52, {
    name: 'KL=0 iff p==q', intuition: 'KL 为 0 当且仅当两分布一致。',
    math: 'KL(p‖p)=0', pseudocode: 'return run(p,p)~0', complexity: 'O(n)',
    ref: 'Cover & Thomas 2006', commonPitfall: '用裁剪避免 log(0)。',
    invariant: 'run(p,p) relClose 0',
    inputs: ['p', 'q'], domains: { p: 'vecProb', q: 'vecProb' },
    run: f_kl,
    predicate: function (p, q) { var r = f_kl(p, q); if (!(r >= 0)) return false; return rc(f_kl(p, p), 0, 1e-9); },
  });

  reg(53, {
    name: 'JS∈[0,1]', intuition: '离散 JS 上界 ln2<1，故恒落 [0,1]。',
    math: '0<=JS<=1', pseudocode: 'return 0<=run<=1', complexity: 'O(n)',
    ref: 'Lin 1991', commonPitfall: '连续情形上界 ln2、离散亦同。',
    invariant: '0<=run(p,q)<=1',
    inputs: ['p', 'q'], domains: { p: 'vecProb', q: 'vecProb' },
    run: f_js,
    predicate: function (p, q) { var r = f_js(p, q); return r >= 0 && r <= 1 + 1e-9; },
  });

  reg(54, {
    name: 'CE≥0', intuition: '交叉熵为非负（概率 ≤1 致对数 ≤0）。',
    math: 'CE>=0', pseudocode: 'return run>=0', complexity: 'O(n)',
    ref: 'Goodfellow 2016', commonPitfall: 'logits 非归一亦不影响非负性（softmax 输出概率）。',
    invariant: 'run(logits,target)>=0',
    inputs: ['logits', 'target'], domains: { logits: 'vecSmall', target: 'smallInt' },
    run: f_ce,
    predicate: function (logits, target) { var r = f_ce(logits, target); return r >= 0; },
  });

  reg(55, {
    name: 'softmax+CE=logsumexp路径', intuition: 'CE 可经 log-sum-exp 稳定计算，等价于 softmax+NLL。',
    math: 'CE=-z_y+LSE(z)', pseudocode: 'return -z_y+LSE(z)', complexity: 'O(n)',
    ref: 'Goodfellow 2016', commonPitfall: 'LSE 用 max 平移防溢出。',
    invariant: 'run(softmax路径) relClose run(logsumexp路径)',
    inputs: ['logits', 'target'], domains: { logits: 'vecSmall', target: 'smallInt' },
    run: f_ce,
    predicate: function (logits, target) {
      var r = f_ce(logits, target); var k = safeIdx(target, logits.length);
      return rc(r, -logits[k] + logsumexp(logits), 1e-9);
    },
  });

  reg(56, {
    name: 'focal∈[0,CE]', intuition: 'Focal(g=2) 非负且不超过同 p_t 的标准 CE。',
    math: '0<=FL<=-ln p_t', pseudocode: 'return run<=CE', complexity: 'O(1)',
    ref: 'Lin et al. 2017', commonPitfall: 'FL 并非恒 ≤1（p_t→0 时可很大），故正确上界为 CE。',
    invariant: '0<=run(z,label)<=CE(z,label)',
    inputs: ['z', 'label'], domains: { z: 'small', label: 'vecMask' },
    run: f_focal,
    predicate: function (z, label) {
      var yb = signLabel(label[0]); var pt = yb === 1 ? S.sigmoid(z) : S.sigmoid(-z);
      var r = f_focal(z, label); var ce = -Math.log(Math.max(pt, 1e-12));
      return r >= -1e-12 && r <= ce + 1e-9;
    },
  });

  reg(57, {
    name: 'MSE(y,y)==0', intuition: '预测等于真值时 MSE 为 0。',
    math: 'MSE(y,y)=0', pseudocode: 'return run(y,y)~0', complexity: 'O(n)',
    ref: '基础性质', commonPitfall: '无。',
    invariant: 'run(y,y) relClose 0',
    inputs: ['y', 'yh'], domains: { y: 'vecSmall', yh: 'vecSmall' },
    run: f_mse,
    predicate: function (y, yh) { return rc(f_mse(y, y), 0, 1e-9); },
  });

  reg(58, {
    name: 'BCE分解', intuition: 'BCE 可逐点分解为 y·(-ln p)+(1-y)·(-ln(1-p))。',
    math: 'BCE=mean(y·(-ln p)+(1-y)·(-ln(1-p)))', pseudocode: 'return decomposition', complexity: 'O(n)',
    ref: 'Goodfellow 2016', commonPitfall: '裁剪保证有限。',
    invariant: 'run(y,p) relClose 分解式',
    inputs: ['y', 'p'], domains: { y: 'vecMask', p: 'vecProb' },
    run: f_bce,
    predicate: function (y, p) {
      var s = 0; for (var i = 0; i < y.length; i++) { var pi = Math.max(p[i], 1e-12), oi = Math.max(1 - p[i], 1e-12); s += y[i] * (-Math.log(pi)) + (1 - y[i]) * (-Math.log(oi)); }
      return rc(f_bce(y, p), s / y.length, 1e-9);
    },
  });

  var f_entropy = entropy;
  reg(59, {
    name: '熵H(p)≥0', intuition: '香农熵非负，衡量分布不确定性。',
    math: 'H(p)=-Σp ln p>=0', pseudocode: 'return entropy(p)>=0', complexity: 'O(n)',
    ref: 'Shannon 1948', commonPitfall: '裁剪避免 0·ln0。',
    invariant: 'run(p)>=0 且有限',
    inputs: ['p'], domains: { p: 'vecProb' },
    run: f_entropy,
    predicate: function (p) { var r = f_entropy(p); return Number.isFinite(r) && r >= 0; },
  });

  reg(60, {
    name: '熵上界', intuition: '熵 ≤ ln n，均匀时取等号。',
    math: 'H(p)<=ln n；均匀时 =ln n', pseudocode: 'return run<=ln n', complexity: 'O(n)',
    ref: 'Cover & Thomas 2006', commonPitfall: '界依赖支撑集大小 n。',
    invariant: 'run(p)<=ln n；且均匀分布时 relClose ln n',
    inputs: ['p'], domains: { p: 'vecProb' },
    run: f_entropy,
    predicate: function (p) {
      var r = f_entropy(p); var n = p.length;
      if (!(r <= Math.log(n) + 1e-9)) return false;
      var uni = new Float64Array(n); for (var i = 0; i < n; i++) uni[i] = 1 / n;
      return rc(f_entropy(uni), Math.log(n), 1e-9);
    },
  });

  var f_selfinfo = function (p, k) { var idx = safeIdx(k, p.length); return -Math.log(Math.max(p[idx], 1e-12)); };
  reg(61, {
    name: '自信息', intuition: '事件 x 的自信息 -ln p_x，衡量惊讶度。',
    math: 'I(x)=-ln p_x>=0', pseudocode: 'return -ln(p[k])', complexity: 'O(1)',
    ref: 'Shannon 1948', commonPitfall: 'k 越界时取模。',
    invariant: 'run(p,k)>=0 且等于 -ln p_k',
    inputs: ['p', 'k'], domains: { p: 'vecProb', k: 'smallInt' },
    run: f_selfinfo,
    predicate: function (p, k) { var r = f_selfinfo(p, k); var idx = safeIdx(k, p.length); return r >= 0 && rc(r, -Math.log(Math.max(p[idx], 1e-12)), 1e-9); },
  });

  reg(62, {
    name: 'NLL≥0', intuition: '负对数似然非负（概率 ≤1）。',
    math: 'NLL=-ln p_y>=0', pseudocode: 'return run>=0', complexity: 'O(1)',
    ref: 'Goodfellow 2016', commonPitfall: '无。',
    invariant: 'run(p,target)>=0',
    inputs: ['p', 'target'], domains: { p: 'vecProb', target: 'smallInt' },
    run: f_nll,
    predicate: function (p, target) { var r = f_nll(p, target); return r >= 0; },
  });

  reg(63, {
    name: 'CE-onehot==NLL', intuition: '目标为 one-hot 时，交叉熵退化为 NLL。',
    math: 'Σ oh_i(-ln p_i)=-ln p_t', pseudocode: 'return CE(onehot)==NLL', complexity: 'O(n)',
    ref: 'Goodfellow 2016', commonPitfall: '仅当目标为独热成立。',
    invariant: 'run(one-hot 目标)==NLL(p,t)',
    inputs: ['p', 'target'], domains: { p: 'vecProb', target: 'smallInt' },
    run: function (p, target) {
      var k = safeIdx(target, p.length); var s = 0;
      for (var i = 0; i < p.length; i++) { var oh = i === k ? 1 : 0; s += oh * (-Math.log(Math.max(p[i], 1e-12))); }
      return s;
    },
    predicate: function (p, target) { var k = safeIdx(target, p.length); var r = 0; for (var i = 0; i < p.length; i++) { var oh = i === k ? 1 : 0; r += oh * (-Math.log(Math.max(p[i], 1e-12))); } return rc(r, f_nll(p, target), 1e-9); },
  });

  reg(64, {
    name: 'Huber≥0', intuition: 'Huber 损失由非负片断组成，整体非负。',
    math: 'Huber>=0', pseudocode: 'return run>=0', complexity: 'O(n)',
    ref: 'Huber 1964', commonPitfall: '无。',
    invariant: 'run(y,yh)>=0',
    inputs: ['y', 'yh'], domains: { y: 'vecSmall', yh: 'vecSmall' },
    run: f_huber,
    predicate: function (y, yh) { return f_huber(y, yh) >= 0; },
  });
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

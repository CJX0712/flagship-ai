/* ============================================================================
 * L10 微指令 · G 域（激活与损失）· 第 4/4 批（G097–G128）
 * ----------------------------------------------------------------------------
 * 所属层：L10-microIsa  域：G（激活与损失）
 * 允许依赖的层：L9（special）、L10（registry/invariant）
 * 主要 @ref：Goodfellow (2016) 数值稳定；Paszke et al. (2019)；He et al. (2016)
 *           ResNet；Schroff et al. (2015) Triplet；Hadsell et al. (2006) Contrastive
 *
 * 涵盖：数值稳定交叉熵 G097–G112；高级/组合 G113–G128。
 * 约定：概率用 vecProb；分类下标用 smallInt 取模；温度/margin/eps 用 smallPos；
 *       二值标签用 vecMask 映射 ±1/-1。
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
  function js(p, q) {
    var n = p.length; var m = new Float64Array(n);
    for (var i = 0; i < n; i++) m[i] = (p[i] + q[i]) / 2;
    return 0.5 * kl(p, m) + 0.5 * kl(q, m);
  }
  function signLabel(y) { return y === 1 ? 1 : -1; }
  function silu(x) { return x * S.sigmoid(x); }

  // ════════════════════════ 数值稳定交叉熵 G097–G112 ════════════════════════
  var f_expclip = function (x) { return x > 700 ? Number.MAX_VALUE : Math.exp(x); };
  reg(97, {
    name: '稳定exp-clip', intuition: '指数运算在大正输入处裁剪以防溢出为 Inf。',
    math: 'exp(x)（x>700 裁剪到 FINITE_MAX）', pseudocode: 'return x>700?MAX:exp(x)', complexity: 'O(1)',
    ref: 'IEEE 754 / 数值稳定', commonPitfall: '裁剪会牺牲大正输入精度（极少触发）。',
    invariant: 'run 有限且 >=0；|x|<=700 时等于 exp(x)',
    inputs: ['x'], domains: { x: 'finite' },
    run: f_expclip,
    predicate: function (x) { var r = f_expclip(x); if (!(Number.isFinite(r) && r >= 0)) return false; return x <= 700 ? rc(r, Math.exp(x), 1e-9, true) : true; },
  });

  var f_log1p = function (x) { return Math.log1p(x); };
  reg(98, {
    name: '稳定log1p', intuition: 'log(1+x) 对小 x 用补偿算法，避免灾难性抵消。',
    math: 'log1p(x)=ln(1+x)', pseudocode: 'return Math.log1p(x)', complexity: 'O(1)',
    ref: 'Cody & Waite 1980', commonPitfall: 'x<-1 致 NaN（此处 x>=0 保证安全）。',
    invariant: 'run(x) relClose ln(1+x)（x>=0）',
    inputs: ['x'], domains: { x: 'nonNegSmall' },
    run: f_log1p,
    predicate: function (x) { var r = f_log1p(x); return rc(r, Math.log(1 + x), 1e-9); },
  });

  var f_log1pexp = function (x) { return Math.log1p(Math.exp(x)); };
  reg(99, {
    name: '稳定log(1+e^x)=softplus', intuition: 'softplus 即 ln(1+e^x)，对正负 x 均稳定。',
    math: 'ln(1+e^x)=softplus(x)', pseudocode: 'return log1p(exp(x))', complexity: 'O(1)',
    ref: 'Dugas et al. 2001', commonPitfall: '直接 exp 在大正 x 溢出（softplus 内部 max 平移规避）。',
    invariant: 'run(x) relClose softplus(x)',
    inputs: ['x'], domains: { x: 'small' },
    run: f_log1pexp,
    predicate: function (x) { var r = f_log1pexp(x); return rc(r, S.softplus(x), 1e-9); },
  });

  var f_lse = logsumexp;
  reg(100, {
    name: '稳定softmax分母LSE', intuition: 'log-sum-exp 用 max 平移防溢出，是 softmax 的分母。',
    math: 'LSE(z)=max+ln Σe^{z_i-max}', pseudocode: 'return max+log(sum exp(z-max))', complexity: 'O(n)',
    ref: '数值方法标准', commonPitfall: 'LSE(z)>=max(z)。',
    invariant: 'run(z)>=max(z) 且 relClose ln(Σe^{z_i})',
    inputs: ['z'], domains: { z: 'vecSmall' },
    run: f_lse,
    predicate: function (z) {
      var r = f_lse(z); var m = -Infinity; for (var i = 0; i < z.length; i++) if (z[i] > m) m = z[i];
      if (!(r >= m - 1e-9)) return false;
      var s = 0; for (var j = 0; j < z.length; j++) s += Math.exp(z[j]);
      return rc(r, Math.log(s), 1e-9);
    },
  });

  var f_ce_lse = function (z, t) { var k = safeIdx(t, z.length); return -z[k] + logsumexp(z); };
  reg(101, {
    name: '稳定CE logsumexp路径', intuition: '交叉熵经 LSE 稳定计算：L=-z_y+LSE(z)。',
    math: 'CE=-z_y+LSE(z)', pseudocode: 'return -z_y+LSE(z)', complexity: 'O(n)',
    ref: 'Goodfellow 2016', commonPitfall: '与 softmax+NLL 数值一致但更稳定。',
    invariant: 'run(z,t) relClose -ln(softmax(z)_t) 且 >=0',
    inputs: ['logits', 'target'], domains: { logits: 'vecSmall', target: 'smallInt' },
    run: f_ce_lse,
    predicate: function (logits, target) {
      var r = f_ce_lse(logits, target); var k = safeIdx(target, logits.length);
      var refv = -Math.log(Math.max(softmax(logits)[k], 1e-12));
      return r >= 0 && rc(r, refv, 1e-9);
    },
  });

  var f_logclip = function (x) { return Math.log(Math.max(x, 1e-12)); };
  reg(102, {
    name: '防log(0)clip', intuition: '对 log 输入下限裁剪，避免 0 处 -Inf。',
    math: 'log(max(x,ε))', pseudocode: 'return log(max(x,1e-12))', complexity: 'O(1)',
    ref: '数值稳定技巧', commonPitfall: 'ε 须远小于典型输入。',
    invariant: 'run(x) 有限且 relClose ln(x)（x>=ε）',
    inputs: ['x'], domains: { x: 'smallPos' },
    run: f_logclip,
    predicate: function (x) { var r = f_logclip(x); return Number.isFinite(r) && rc(r, Math.log(x), 1e-9); },
  });

  var f_tanh2 = function (x) { return Math.tanh(x); };
  reg(103, {
    name: '稳定tanh', intuition: 'tanh 在饱和处稳定趋近 ±1，奇函数。',
    math: 'tanh(-x)=-tanh(x)', pseudocode: 'return tanh(x)', complexity: 'O(1)',
    ref: 'LeCun et al. 1998', commonPitfall: '|tanh(x)|<1。',
    invariant: '|run(x)|<1 且 tanh(-x)=-tanh(x)',
    inputs: ['x'], domains: { x: 'small' },
    run: f_tanh2,
    predicate: function (x) { var r = f_tanh2(x); return Math.abs(r) < 1 && rc(r, -Math.tanh(-x), 1e-9); },
  });

  var f_exp2 = function (x) { return Math.exp(x); };
  reg(104, {
    name: '稳定exp(大负→0)', intuition: 'exp 对大负输入平滑趋近 0，恒正。',
    math: 'x<0 ⇒ 0<exp(x)<=1', pseudocode: 'return exp(x)', complexity: 'O(1)',
    ref: 'IEEE 754', commonPitfall: 'x 很大正时溢出为 Inf（此处仅验证符号区间）。',
    invariant: 'x<0 ⇒ 0<run<=1；x>=0 ⇒ run>=1',
    inputs: ['x'], domains: { x: 'finite' },
    run: f_exp2,
    predicate: function (x) {
      var r = f_exp2(x);
      if (x < 0) { if (r < 0 || r > 1 + 1e-9) return false; if (x > -700 && !(r > 0)) return false; return true; }
      return r >= 1;
    },
  });

  var f_sig2 = S.sigmoid;
  reg(105, {
    name: '稳定sigmoid', intuition: 'sigmoid 在饱和处稳定为 0/1，满足 σ(-x)=1-σ(x)。',
    math: 'σ(-x)=1-σ(x)', pseudocode: 'return 1/(1+exp(-x))', complexity: 'O(1)',
    ref: 'Goodfellow 2016', commonPitfall: '0<σ(x)<1。',
    invariant: 'σ(-x)=1-σ(x)；饱和处趋 0/1',
    inputs: ['x'], domains: { x: 'finite' },
    run: f_sig2,
    predicate: function (x) {
      var r = f_sig2(x);
      var sat = x > 20 ? r > 0.999 : x < -20 ? r < 1e-8 : (r > 0 && r < 1);
      return sat && rc(r, 1 - S.sigmoid(-x), 1e-9);
    },
  });

  var f_logsoftmax2 = function (z, t) { var k = safeIdx(t, z.length); return z[k] - logsumexp(z); };
  reg(106, {
    name: '稳定log-softmax', intuition: 'log-softmax 经 LSE 稳定计算，目标处值 ≤0 且单调递减于 logit。',
    math: 'logsm(z)_t=z_t-LSE(z)', pseudocode: 'return z_t-LSE(z)', complexity: 'O(n)',
    ref: 'Goodfellow 2016', commonPitfall: '逐元素 ≤0。',
    invariant: 'run(z,t) relClose ln(softmax(z)_t) 且 ≤0',
    inputs: ['logits', 'target'], domains: { logits: 'vecSmall', target: 'smallInt' },
    run: f_logsoftmax2,
    predicate: function (logits, target) {
      var r = f_logsoftmax2(logits, target); var k = safeIdx(target, logits.length);
      var refv = Math.log(Math.max(softmax(logits)[k], 1e-12));
      return r <= 0 + 1e-9 && rc(r, refv, 1e-9);
    },
  });

  var f_kl2 = kl;
  reg(107, {
    name: '稳定KL(eps)', intuition: 'KL 散度对 q 裁剪防 log(0)，保持非负与 p=q 时为 0。',
    math: 'KL(p‖q)=Σp ln(p/q)', pseudocode: 'return sum(p*ln(p/q))', complexity: 'O(n)',
    ref: 'Cover & Thomas 2006', commonPitfall: '非对称。',
    invariant: 'run>=0 且 p=q 时 relClose 0',
    inputs: ['p', 'q'], domains: { p: 'vecProb', q: 'vecProb' },
    run: f_kl2,
    predicate: function (p, q) { var r = f_kl2(p, q); if (!(Number.isFinite(r) && r >= 0)) return false; return rc(f_kl2(p, p), 0, 1e-9); },
  });

  var f_js2 = js;
  reg(108, {
    name: '稳定JS(eps)', intuition: 'JS 散度裁剪防 log(0)，对称且有界 [0,ln2]。',
    math: 'JS=0.5KL(p‖m)+0.5KL(q‖m)', pseudocode: 'return 0.5*KL+0.5*KL', complexity: 'O(n)',
    ref: 'Lin 1991', commonPitfall: '上界 ln2。',
    invariant: '0<=run<=ln2，对称，p=q 时为 0',
    inputs: ['p', 'q'], domains: { p: 'vecProb', q: 'vecProb' },
    run: f_js2,
    predicate: function (p, q) {
      var r = f_js2(p, q);
      if (!(r >= 0 && r <= Math.log(2) + 1e-9)) return false;
      if (!rc(f_js2(p, p), 0, 1e-9)) return false;
      return rc(r, f_js2(q, p), 1e-9);
    },
  });

  var f_softmax3 = softmax;
  reg(109, {
    name: '稳定softmax分布', intuition: 'softmax 输出合法概率分布，稳定且 argmax 不变。',
    math: 'p_i=e^{z_i-m}/Σe^{z_j-m}', pseudocode: 'return softmax(z)', complexity: 'O(n)',
    ref: 'Bridle 1990', commonPitfall: 'sum=1，元素 ≥0。',
    invariant: 'Σrun(z)=1、run>=0、argmax 与 z 一致',
    inputs: ['z'], domains: { z: 'vecSmall' },
    run: f_softmax3,
    predicate: function (z) {
      var sm = f_softmax3(z); var s = 0;
      for (var i = 0; i < sm.length; i++) { if (!(sm[i] >= 0)) return false; s += sm[i]; }
      if (!rc(s, 1, 1e-9)) return false;
      return argmax(sm) === argmax(z);
    },
  });

  var f_nll_logsm = function (z, t) { var k = safeIdx(t, z.length); return -(z[k] - logsumexp(z)); };
  reg(110, {
    name: '稳定NLL via log-softmax', intuition: 'NLL 经 log-softmax 稳定计算，避免 log(0)。',
    math: 'NLL=-logsm(z)_t=-(z_t-LSE(z))', pseudocode: 'return -(z_t-LSE(z))', complexity: 'O(n)',
    ref: 'Goodfellow 2016', commonPitfall: '>=0。',
    invariant: 'run(z,t) relClose -ln(softmax(z)_t) 且 >=0',
    inputs: ['logits', 'target'], domains: { logits: 'vecSmall', target: 'smallInt' },
    run: f_nll_logsm,
    predicate: function (logits, target) {
      var r = f_nll_logsm(logits, target); var k = safeIdx(target, logits.length);
      var refv = -Math.log(Math.max(softmax(logits)[k], 1e-12));
      return r >= 0 && rc(r, refv, 1e-9);
    },
  });

  var f_exp_ident = function (x) { return Math.exp(x); };
  reg(111, {
    name: '稳定exp-identity', intuition: 'exp 满足 exp(x)·exp(-x)=1，验证指数对合一致性。',
    math: 'e^x·e^{-x}=1', pseudocode: 'return exp(x)*exp(-x)', complexity: 'O(1)',
    ref: '初等指数性质', commonPitfall: '大 x 处 exp(x)=Inf 使乘积 NaN（此处域受限）。',
    invariant: 'run(x)·exp(-x) relClose 1',
    inputs: ['x'], domains: { x: 'small' },
    run: f_exp_ident,
    predicate: function (x) { var e = f_exp_ident(x); return rc(e * Math.exp(-x), 1, 1e-9); },
  });

  var f_silu3 = silu;
  reg(112, {
    name: '稳定SiLU', intuition: 'SiLU 数值稳定，满足 silu(x)=silu(-x)+x 的对称恒等式。',
    math: 'silu(x)=x·σ(x); silu(x)=silu(-x)+x', pseudocode: 'return x*sig(x)', complexity: 'O(1)',
    ref: 'Ramachandran et al. 2017', commonPitfall: '与 Swish 同义。',
    invariant: 'run(x) relClose silu(-x)+x',
    inputs: ['x'], domains: { x: 'small' },
    run: f_silu3,
    predicate: function (x) { var r = f_silu3(x); return rc(r, silu(-x) + x, 1e-9); },
  });

  // ════════════════════════ 高级/组合 G113–G128 ════════════════════════
  var f_acc = function (z, t) { var k = safeIdx(t, z.length); return argmax(z) === k ? 1 : 0; };
  reg(113, {
    name: '多分类准确率', intuition: '预测类别（argmax）等于目标时记为正确。',
    math: 'acc=1[argmax(z)=t]', pseudocode: 'return argmax(z)==t?1:0', complexity: 'O(n)',
    ref: '评估标准', commonPitfall: '仅考察 argmax 是否命中。',
    invariant: 'run(z,t)∈{0,1} 且等于 argmax 命中指示',
    inputs: ['logits', 'target'], domains: { logits: 'vecSmall', target: 'smallInt' },
    run: f_acc,
    predicate: function (logits, target) { var k = safeIdx(target, logits.length); var r = f_acc(logits, target); return r === (argmax(logits) === k ? 1 : 0); },
  });

  var f_bacc = function (s, y) {
    var c = 0; for (var i = 0; i < s.length; i++) { var pred = s[i] > 0 ? 1 : 0; if (pred === (y[i] === 1 ? 1 : 0)) c++; } return c / s.length;
  };
  reg(114, {
    name: '二分类准确率', intuition: '以 0 为阈值的二分类命中率。',
    math: 'acc=mean(1[pred==label]), pred=score>0', pseudocode: 'return mean(correct)', complexity: 'O(n)',
    ref: '评估标准', commonPitfall: '阈值固定为 0（logit 空间）。',
    invariant: 'run∈[0,1] 且等于手动命中率',
    inputs: ['scores', 'labels'], domains: { scores: 'vecSmall', labels: 'vecMask' },
    run: f_bacc,
    predicate: function (s, y) {
      var r = f_bacc(s, y); var c = 0;
      for (var i = 0; i < s.length; i++) { var pred = s[i] > 0 ? 1 : 0; if (pred === (y[i] === 1 ? 1 : 0)) c++; }
      return r >= 0 && r <= 1 + 1e-9 && rc(r, c / s.length, 1e-9);
    },
  });

  var f_wmse = function (y, yh, w) {
    var num = 0, den = 0; for (var i = 0; i < y.length; i++) { var e = y[i] - yh[i]; num += w[i] * e * e; den += w[i]; }
    return den > 0 ? num / den : 0;
  };
  reg(115, {
    name: '加权MSE', intuition: '按样本权重加权均方误差。',
    math: 'wMSE=Σw_i(y_i-ŷ_i)²/Σw_i', pseudocode: 'return weighted mean sq', complexity: 'O(n)',
    ref: 'Goodfellow 2016', commonPitfall: '权重须为正且归一化前的和 >0。',
    invariant: 'run>=0 且等于手动加权 MSE',
    inputs: ['y', 'yh', 'w'], domains: { y: 'vecSmall', yh: 'vecSmall', w: 'vecPos' },
    run: f_wmse,
    predicate: function (y, yh, w) {
      var r = f_wmse(y, yh, w); var num = 0, den = 0;
      for (var i = 0; i < y.length; i++) { var e = y[i] - yh[i]; num += w[i] * e * e; den += w[i]; }
      return r >= 0 && rc(r, num / den, 1e-9);
    },
  });

  var f_lsce2 = function (z, t, eps) {
    var n = z.length, k = safeIdx(t, n), sm = softmax(z), q = new Float64Array(n);
    for (var i = 0; i < n; i++) q[i] = (1 - eps) * (i === k ? 1 : 0) + eps / n;
    var s = 0; for (var j = 0; j < n; j++) s += -q[j] * Math.log(Math.max(sm[j], 1e-12));
    return s;
  };
  reg(116, {
    name: '标签平滑变体CE', intuition: '通用 ε 标签平滑交叉熵（ε 由调用方给定）。',
    math: 'q=(1-ε)y+ε/n; L=-Σq_i ln p_i', pseudocode: 'return -Σq ln p', complexity: 'O(n)',
    ref: 'Pereyra et al. 2017', commonPitfall: 'ε=0 退化为普通 CE。',
    invariant: 'run>=0 且等于手动平滑 CE',
    inputs: ['logits', 'target', 'eps'], domains: { logits: 'vecSmall', target: 'smallInt', eps: 'unit' },
    run: f_lsce2,
    predicate: function (logits, target, eps) {
      var r = f_lsce2(logits, target, eps); var n = logits.length, k = safeIdx(target, n), sm = softmax(logits), q = new Float64Array(n);
      for (var i = 0; i < n; i++) q[i] = (1 - eps) * (i === k ? 1 : 0) + eps / n;
      var s = 0; for (var j = 0; j < n; j++) s += -q[j] * Math.log(Math.max(sm[j], 1e-12));
      return r >= 0 && rc(r, s, 1e-9);
    },
  });

  var f_tce = function (z, t, T) {
    var n = z.length, zt = new Float64Array(n);
    for (var i = 0; i < n; i++) zt[i] = z[i] / T;
    var k = safeIdx(t, n); return -Math.log(Math.max(softmax(zt)[k], 1e-12));
  };
  reg(117, {
    name: '温度缩放交叉熵', intuition: '先按温度缩放 logits 再算交叉熵，用于知识蒸馏。',
    math: 'L=-ln softmax(z/T)_t', pseudocode: 'return -log(softmax(z/T)[t])', complexity: 'O(n)',
    ref: 'Hinton et al. 2015', commonPitfall: 'T 同时缩放学生与教师。',
    invariant: 'run>=0 且等于手动温度缩放 CE',
    inputs: ['logits', 'target', 'T'], domains: { logits: 'vecSmall', target: 'smallInt', T: 'smallPos' },
    run: f_tce,
    predicate: function (logits, target, T) {
      var r = f_tce(logits, target, T); var n = logits.length, zt = new Float64Array(n);
      for (var i = 0; i < n; i++) zt[i] = logits[i] / T;
      var k = safeIdx(target, n);
      var refv = -Math.log(Math.max(softmax(zt)[k], 1e-12));
      return r >= 0 && rc(r, refv, 1e-9);
    },
  });

  function topkIdx(z, kk) {
    var idx = []; for (var i = 0; i < z.length; i++) idx.push(i);
    idx.sort(function (a, b) { return z[b] - z[a]; });
    return idx.slice(0, kk);
  }
  function logsumexpIdx(z, idx) {
    var m = -Infinity; for (var i = 0; i < idx.length; i++) if (z[idx[i]] > m) m = z[idx[i]];
    var s = 0; for (var i = 0; i < idx.length; i++) s += Math.exp(z[idx[i]] - m);
    return m + Math.log(s);
  }
  var f_topkce = function (z, t, k) {
    var n = z.length, kk = safeIdx(k, n); if (kk < 1) kk = 1; if (kk > n) kk = n;
    var kidx = topkIdx(z, kk); var kk2 = safeIdx(t, n);
    var inTop = kidx.indexOf(kk2) >= 0;
    return -z[kk2] + logsumexpIdx(z, kidx);
  };
  reg(118, {
    name: 'top-k交叉熵', intuition: '仅前 k 大 logit 参与归一化，其余被遮掩。',
    math: 'L=-z_y+logsumexp(topk z)', pseudocode: 'return -z_y+LSE(topk)', complexity: 'O(n log n)',
    ref: 'Szegedy et al. 2016 (top-k 训练)', commonPitfall: 'top-k 包含目标时退化为普通 CE。',
    invariant: 'run(z,t,k) <= 完整 CE（-z_y+LSE(z)）',
    inputs: ['logits', 'target', 'k'], domains: { logits: 'vecSmall', target: 'smallInt', k: 'smallInt' },
    run: f_topkce,
    predicate: function (logits, target, k) {
      var r = f_topkce(logits, target, k);
      var full = -logits[safeIdx(target, logits.length)] + logsumexp(logits);
      return r <= full + 1e-9;
    },
  });

  var f_focal_logits = function (z, y) {
    var yb = signLabel(y[0]); var pt = yb === 1 ? S.sigmoid(z) : S.sigmoid(-z);
    return -(1 - pt) * (1 - pt) * Math.log(Math.max(pt, 1e-12));
  };
  reg(119, {
    name: 'focal-with-logits', intuition: 'Focal Loss(γ=2) 直接作用于 logit，稳定计算。',
    math: 'p_t=σ(z) 或 σ(-z); FL=-(1-p_t)²ln p_t', pseudocode: 'return focal(z,y)', complexity: 'O(1)',
    ref: 'Lin et al. 2017', commonPitfall: '0<=FL<=CE。',
    invariant: '0<=run<=CE(=-ln p_t)',
    inputs: ['z', 'label'], domains: { z: 'small', label: 'vecMask' },
    run: f_focal_logits,
    predicate: function (z, label) {
      var yb = signLabel(label[0]); var pt = yb === 1 ? S.sigmoid(z) : S.sigmoid(-z);
      var r = f_focal_logits(z, label); var ce = -Math.log(Math.max(pt, 1e-12));
      return r >= -1e-12 && r <= ce + 1e-9;
    },
  });

  var f_contrastive2 = function (x, y, label, margin) {
    var nx = normv(x), ny = normv(y);
    var d = (nx === 0 && ny === 0) ? 0 : Math.sqrt(Math.max(0, dotv(x, x) + dotv(y, y) - 2 * dotv(x, y)));
    return label[0] === 1 ? 0.5 * d * d : 0.5 * Math.max(0, margin - d) * Math.max(0, margin - d);
  };
  reg(120, {
    name: '对比损失(margin)', intuition: '同类收缩、异类维持至少 margin 间隔的对比损失。',
    math: 'L=½d² (同) else ½max(0,m-d)²', pseudocode: 'return piece', complexity: 'O(n)',
    ref: 'Hadsell et al. 2006', commonPitfall: '异类超 margin 后损失为 0。',
    invariant: 'run>=0；异类时 <=½margin²',
    inputs: ['x', 'y', 'label', 'margin'], domains: { x: 'vecSmall', y: 'vecSmall', label: 'vecMask', margin: 'smallPos' },
    run: f_contrastive2,
    predicate: function (x, y, label, margin) {
      var nx = normv(x), ny = normv(y);
      var d = (nx === 0 && ny === 0) ? 0 : Math.sqrt(Math.max(0, dotv(x, x) + dotv(y, y) - 2 * dotv(x, y)));
      var r = f_contrastive2(x, y, label, margin);
      if (!(r >= 0)) return false;
      if (label[0] === 0) return r <= 0.5 * margin * margin + 1e-9;
      return true;
    },
  });

  var f_triplet = function (a, p, n, margin) {
    var dap = dotv(a, a) + dotv(p, p) - 2 * dotv(a, p);
    var dan = dotv(a, a) + dotv(n, n) - 2 * dotv(a, n);
    var v = (dap - dan) + margin; return v > 0 ? v : 0;
  };
  reg(121, {
    name: 'triplet(margin)', intuition: '三元组损失：锚-正距离应小于锚-负距离至少 margin。',
    math: 'L=max(0,‖a-p‖²-‖a-n‖²+margin)', pseudocode: 'return max(0,dap-dan+m)', complexity: 'O(n)',
    ref: 'Schroff et al. 2015', commonPitfall: '距离用平方欧氏。',
    invariant: 'run>=0 且等于 max(0,‖a-p‖²-‖a-n‖²+m)',
    inputs: ['a', 'p', 'n', 'margin'], domains: { a: 'vecSmall', p: 'vecSmall', n: 'vecSmall', margin: 'smallPos' },
    run: f_triplet,
    predicate: function (a, p, n, margin) {
      var dap = dotv(a, a) + dotv(p, p) - 2 * dotv(a, p);
      var dan = dotv(a, a) + dotv(n, n) - 2 * dotv(a, n);
      var v = (dap - dan) + margin; var refv = v > 0 ? v : 0; var r = f_triplet(a, p, n, margin);
      return r >= 0 && rc(r, refv, 1e-9, true);
    },
  });

  var f_l1reg = function (w) { var s = 0; for (var i = 0; i < w.length; i++) s += Math.abs(w[i]); return s; };
  reg(122, {
    name: '正则L1', intuition: 'L1 权重正则，诱导稀疏。',
    math: 'R=Σ|w_i|', pseudocode: 'return sum|w|', complexity: 'O(n)',
    ref: 'Tibshirani 1996', commonPitfall: '在 0 处不可导。',
    invariant: 'run>=0 且等于 Σ|w|',
    inputs: ['w'], domains: { w: 'vecSmall' },
    run: f_l1reg,
    predicate: function (w) { var r = f_l1reg(w); var s = 0; for (var i = 0; i < w.length; i++) s += Math.abs(w[i]); return r >= 0 && rc(r, s, 1e-9); },
  });

  var f_l2reg = function (w) { var s = 0; for (var i = 0; i < w.length; i++) s += w[i] * w[i]; return 0.5 * s; };
  reg(123, {
    name: '正则L2', intuition: 'L2 权重正则（权重衰减），平滑。',
    math: 'R=½Σw_i²', pseudocode: 'return 0.5*sum w^2', complexity: 'O(n)',
    ref: 'Hoerl & Kennard 1970', commonPitfall: '½ 因子易遗漏。',
    invariant: 'run>=0 且等于 ½Σw²',
    inputs: ['w'], domains: { w: 'vecSmall' },
    run: f_l2reg,
    predicate: function (w) { var r = f_l2reg(w); var s = 0; for (var i = 0; i < w.length; i++) s += w[i] * w[i]; return r >= 0 && rc(r, 0.5 * s, 1e-9); },
  });

  var f_tvreg = function (w) { var s = 0; for (var i = 1; i < w.length; i++) s += Math.abs(w[i] - w[i - 1]); return s; };
  reg(124, {
    name: 'TV正则', intuition: '全变分正则，惩罚相邻权重跳变，促进平滑。',
    math: 'R=Σ|w_{i+1}-w_i|', pseudocode: 'return sum|Δw|', complexity: 'O(n)',
    ref: 'Rudin et al. 1992', commonPitfall: '沿索引的相邻差分。',
    invariant: 'run>=0 且等于 Σ|Δw|',
    inputs: ['w'], domains: { w: 'vecSmall' },
    run: f_tvreg,
    predicate: function (w) { var r = f_tvreg(w); var s = 0; for (var i = 1; i < w.length; i++) s += Math.abs(w[i] - w[i - 1]); return r >= 0 && rc(r, s, 1e-9); },
  });

  var f_hinge_margin = function (z, y, m) { var yb = signLabel(y[0]); var v = m - yb * z; return v > 0 ? v : 0; };
  reg(125, {
    name: '合页随margin非减', intuition: '带 margin 的合页损失随 margin 增大而单调不减。',
    math: 'L_m(z,y)=max(0,m-y·z) 关于 m 非减', pseudocode: 'return max(0,m-y*z)', complexity: 'O(1)',
    ref: 'Crammer & Singer 2001', commonPitfall: 'm 越大损失越大（非减）。',
    invariant: '若 m1<=m2 则 L(z,y,m1)<=L(z,y,m2)',
    inputs: ['z', 'label', 'm1', 'm2'], domains: { z: 'small', label: 'vecMask', m1: 'smallPos', m2: 'smallPos' },
    run: f_hinge_margin,
    predicate: function (z, label, m1, m2) {
      var yb = signLabel(label[0]);
      var h1 = f_hinge_margin(z, label, m1, m2);
      var v2 = m2 - yb * z; var h2 = v2 > 0 ? v2 : 0;
      return m1 <= m2 ? (h1 <= h2 + 1e-9) : (h1 >= h2 - 1e-9);
    },
  });

  reg(126, {
    name: '加权MSE退化(均匀权重==MSE)', intuition: '权重全相等时加权 MSE 退化为普通 MSE。',
    math: 'w_i≡c ⇒ wMSE=MSE', pseudocode: 'return wMSE(y,yh,1)=MSE', complexity: 'O(n)',
    ref: '基础性质', commonPitfall: '权重常数倍不影响归一化结果。',
    invariant: 'run(y,yh,全1) relClose MSE(y,yh)',
    inputs: ['y', 'yh', 'w'], domains: { y: 'vecSmall', yh: 'vecSmall', w: 'vecPos' },
    run: f_wmse,
    predicate: function (y, yh, w) {
      var cw = new Float64Array(w.length); cw.fill(1);
      var s = 0; for (var i = 0; i < y.length; i++) { var e = y[i] - yh[i]; s += e * e; }
      return rc(f_wmse(y, yh, cw), s / y.length, 1e-9);
    },
  });

  var f_perp = function (p) { return Math.exp(entropy(p)); };
  reg(127, {
    name: '熵与困惑度(perp=exp(H))', intuition: '困惑度等于熵的指数，熵非负故困惑度 ≥1。',
    math: 'PPL=exp(H(p))>=1', pseudocode: 'return exp(entropy(p))', complexity: 'O(n)',
    ref: 'Jelinek et al. 1977', commonPitfall: '与 G092 同源，此处强调与熵的精确关系。',
    invariant: 'run(p)>=1 且 relClose exp(entropy(p))',
    inputs: ['p'], domains: { p: 'vecProb' },
    run: f_perp,
    predicate: function (p) { var r = f_perp(p); return r >= 1 - 1e-9 && rc(r, Math.exp(entropy(p)), 1e-9); },
  });

  var f_mi = kl;
  reg(128, {
    name: '互信息非负', intuition: '互信息（此处以 KL 形式度量）非负，p=q 时为 0。',
    math: 'MI(p,q)=Σp_i ln(p_i/q_i)>=0', pseudocode: 'return kl(p,q)', complexity: 'O(n)',
    ref: 'Cover & Thomas 2006', commonPitfall: '非对称度量。',
    invariant: 'run(p,q)>=0 且 p=q 时 relClose 0',
    inputs: ['p', 'q'], domains: { p: 'vecProb', q: 'vecProb' },
    run: f_mi,
    predicate: function (p, q) { var r = f_mi(p, q); if (!(Number.isFinite(r) && r >= 0)) return false; return rc(f_mi(p, p), 0, 1e-9); },
  });
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

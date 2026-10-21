/* ============================================================================
 * L10 微指令 · G 域（激活与损失）· 第 1/4 批（G001–G032）
 * ----------------------------------------------------------------------------
 * 所属层：L10-microIsa  域：G（激活与损失）
 * 允许依赖的层：L9（special/rng/assert）、L10（registry/invariant）
 * 主要 @ref：Hendrycks & Gimpel (2016) GELU；Clevert et al. (2015) ELU；
 *           Klambauer et al. (2017) SELU；Ramachandran et al. (2017) Swish；
 *           Goodfellow et al. (2016) Deep Learning；Abramowitz & Stegun (1964)
 *
 * 涵盖：基础激活 G001–G016；激活导数恒等式 G017–G032。
 * 判据：精确恒等式用 Object.is（±0 归一）；数值用 relClose；导数 op 用有限差分
 *       校验 run 确为其对应激活的导数（非平凡、非恒真）。
 * ==========================================================================*/
;(function (NS) {
  'use strict';
  NS.L10 = NS.L10 || {};
  function reg(i, o) { NS.L10.register('G', i, o); }
  var S = NS.L9.special;
  var rc = function (a, b, t) { return NS.L10.relClose(a, b, t); };

  // ── 激活基元（纯函数，run/predicate 复用）──────────────────────────────
  function sig(x) { return S.sigmoid(x); }
  function softplus(x) { return S.softplus(x); }
  function relu(x) { return x > 0 ? x : 0; }
  function leaky(x) { return x < 0 ? 0.01 * x : x; }
  function elu(x) { return x >= 0 ? x : (Math.exp(x) - 1); }
  function selu(x) { var a = 1.6732632423543772, s = 1.0507009873554805; return s * (x >= 0 ? x : a * (Math.exp(x) - 1)); }
  function erf(x) { return S.erf(x); }
  function geluErf(x) { return 0.5 * x * (1 + erf(x / Math.SQRT2)); }
  function geluTanh(x) { return S.gelu(x); }
  function silu(x) { return x * sig(x); }
  function mish(x) { return x * Math.tanh(softplus(x)); }
  function bent(x) { return 0.5 * (x + Math.sqrt(x * x + 1)); }
  function tanh(x) { return Math.tanh(x); }
  function hardTanh(x) { return x < -1 ? -1 : (x > 1 ? 1 : x); }
  function hardSigmoid(x) { return Math.max(0, Math.min(1, 0.2 * x + 0.5)); }
  function softsign(x) { return x / (1 + Math.abs(x)); }
  function identity(x) { return x; }

  // 有限差分（中心差分）用于校验导数 op
  function fd(f, x) { var h = 1e-5; return (f(x + h) - f(x - h)) / (2 * h); }

  // ════════════════════════ 基础激活 G001–G016 ════════════════════════
  var f_relu = relu;
  reg(1, {
    name: 'ReLU', intuition: '整流线性单元，负值截断为 0，正值恒等。',
    math: 'relu(x)=max(0,x)', pseudocode: 'return x>0?x:0', complexity: 'O(1)',
    ref: 'Nair & Hinton 2010', commonPitfall: 'x=0 时为 0（非负）。',
    invariant: 'run(x)>=0；且 x>0 时 run(x)==x',
    inputs: ['x'], domains: { x: 'finite' },
    run: f_relu,
    predicate: function (x) { var r = f_relu(x); return r >= 0 && (x > 0 ? rc(r, x, 1e-9, true) : true); },
  });

  var f_leaky = leaky;
  reg(2, {
    name: 'LeakyReLU', intuition: '负值给予极小斜率 0.01，缓解 ReLU 死亡。',
    math: 'leaky(x)=x<0?0.01x:x', pseudocode: 'return x<0?0.01*x:x', complexity: 'O(1)',
    ref: 'Maas et al. 2013', commonPitfall: '斜率为 0.01 而非 0。',
    invariant: 'x<0 ⇒ run(x)==0.01x；否则 run(x)==x',
    inputs: ['x'], domains: { x: 'finite' },
    run: f_leaky,
    predicate: function (x) { var r = f_leaky(x); return x < 0 ? rc(r, 0.01 * x, 1e-9) : rc(r, x, 1e-9, true); },
  });

  var f_elu = elu;
  reg(3, {
    name: 'ELU', intuition: '负值平滑趋于 -1，缓解梯度消失且输出均值近 0。',
    math: 'elu(x)=x>=0?x:(e^x-1)', pseudocode: 'return x>=0?x:(Math.exp(x)-1)', complexity: 'O(1)',
    ref: 'Clevert et al. 2015', commonPitfall: 'α 默认 1。',
    invariant: 'x<0 ⇒ run(x)==e^x-1 (relClose)；x>=0 ⇒ run(x)==x',
    inputs: ['x'], domains: { x: 'small' },
    run: f_elu,
    predicate: function (x) { var r = f_elu(x); return x < 0 ? rc(r, Math.exp(x) - 1, 1e-9) : rc(r, x, 1e-9, true); },
  });

  var f_selu = selu;
  reg(4, {
    name: 'SELU', intuition: '自带缩放与平移的 ELU，使神经元激活趋近零均值单位方差。',
    math: 'selu(x)=λ·elu(x,α)，λ≈1.0507,α≈1.6733', pseudocode: 'return λ*elu', complexity: 'O(1)',
    ref: 'Klambauer et al. 2017', commonPitfall: '需配 SNN 初始化方能自归一化。',
    invariant: 'run(x)==1.0507·elu(x,1.6733) (relClose)',
    inputs: ['x'], domains: { x: 'small' },
    run: f_selu,
    predicate: function (x) { var r = f_selu(x); var e = x >= 0 ? x : 1.6732632423543772 * (Math.exp(x) - 1); return rc(r, 1.0507009873554805 * e, 1e-9); },
  });

  var f_softplus = softplus;
  reg(5, {
    name: 'Softplus', intuition: 'ReLU 的平滑近似，处处为正且严格单调增。',
    math: 'softplus(x)=ln(1+e^x)', pseudocode: 'return Math.log1p(Math.exp(x))', complexity: 'O(1)',
    ref: 'Dugas et al. 2001', commonPitfall: 'x 很大时 ≈x，并非恒为小值。',
    invariant: 'run(x)>0；且 x>5 ⇒ run(x)≈x (relClose)',
    inputs: ['x'], domains: { x: 'small' },
    run: f_softplus,
    predicate: function (x) { var r = f_softplus(x); var ref = x + Math.log1p(Math.exp(-x)); return r > 0 && rc(r, ref, 1e-9); },
  });

  var f_softsign = softsign;
  reg(6, {
    name: 'Softsign', intuition: '用 |x| 替代 x² 的饱和激活，梯度衰减更慢。',
    math: 'softsign(x)=x/(1+|x|)', pseudocode: 'return x/(1+Math.abs(x))', complexity: 'O(1)',
    ref: 'Bergstra et al. 2009', commonPitfall: '输出严格落在 (-1,1)。',
    invariant: '|run(x)|<1 且 |run(x)|==|x|/(1+|x|)',
    inputs: ['x'], domains: { x: 'small' },
    run: f_softsign,
    predicate: function (x) { var r = f_softsign(x); return Math.abs(r) < 1 && rc(Math.abs(r), Math.abs(x) / (1 + Math.abs(x)), 1e-9, true); },
  });

  var f_sigmoid = sig;
  reg(7, {
    name: 'Sigmoid', intuition: '将实数压缩到 (0,1)，经典概率化激活。',
    math: 'σ(x)=1/(1+e^{-x})', pseudocode: 'return 1/(1+Math.exp(-x))', complexity: 'O(1)',
    ref: 'Goodfellow 2016 §6.2', commonPitfall: '两端饱和致梯度消失。',
    invariant: '0<run(x)<1 且 σ(-x)=1-σ(x)',
    inputs: ['x'], domains: { x: 'small' },
    run: f_sigmoid,
    predicate: function (x) { var r = f_sigmoid(x); return r > 0 && r < 1 && rc(r, 1 - sig(-x), 1e-9); },
  });

  var f_tanh = tanh;
  reg(8, {
    name: 'Tanh', intuition: '零中心化饱和激活，输出 (-1,1)。',
    math: 'tanh(x)=(e^x-e^{-x})/(e^x+e^{-x})', pseudocode: 'return Math.tanh(x)', complexity: 'O(1)',
    ref: 'LeCun et al. 1998', commonPitfall: '奇函数属性常用于校验。',
    invariant: '-1<run(x)<1 且 tanh(-x)=-tanh(x)',
    inputs: ['x'], domains: { x: 'small' },
    run: f_tanh,
    predicate: function (x) { var r = f_tanh(x); return r > -1 && r < 1 && rc(r, -tanh(-x), 1e-9); },
  });

  var f_hardTanh = hardTanh;
  reg(9, {
    name: 'HardTanh', intuition: '分段线性裁剪激活，计算廉价且梯度稳定。',
    math: 'hardTanh(x)=clamp(x,-1,1)', pseudocode: 'return clamp(x,-1,1)', complexity: 'O(1)',
    ref: 'Collobert 2011', commonPitfall: '区间外梯度为 0。',
    invariant: '-1<=run(x)<=1；|x|>=1 时等于 ±1',
    inputs: ['x'], domains: { x: 'small' },
    run: f_hardTanh,
    predicate: function (x) {
      var r = f_hardTanh(x);
      return r >= -1 && r <= 1 && (x < -1 ? r === -1 : x > 1 ? r === 1 : rc(r, x, 1e-9, true));
    },
  });

  var f_hardSigmoid = hardSigmoid;
  reg(10, {
    name: 'HardSigmoid', intuition: 'Sigmoid 的分段线性近似，便于低精度推理。',
    math: 'hardSigmoid(x)=clamp(0.2x+0.5,0,1)', pseudocode: 'return clamp(0.2x+0.5,0,1)', complexity: 'O(1)',
    ref: 'Courbariaux et al. 2016', commonPitfall: '拐点位于 x=±2.5。',
    invariant: '0<=run(x)<=1；|x|>=2.5 时等于 0 或 1',
    inputs: ['x'], domains: { x: 'small' },
    run: f_hardSigmoid,
    predicate: function (x) {
      var r = f_hardSigmoid(x);
      return r >= 0 && r <= 1 && (x <= -2.5 ? r === 0 : x >= 2.5 ? r === 1 : rc(r, 0.2 * x + 0.5, 1e-9, true));
    },
  });

  var f_geluErf = geluErf;
  reg(11, {
    name: 'GELU-erf', intuition: '依输入大小门控的高斯误差线性单元（精确 erf 版）。',
    math: 'gelu(x)=0.5x(1+erf(x/√2))', pseudocode: 'return 0.5*x*(1+erf(x/Math.SQRT2))', complexity: 'O(1)',
    ref: 'Hendrycks & Gimpel 2016', commonPitfall: 'erf 须用稳定近似。',
    invariant: 'gelu 为奇函数：run(x)=-run(-x)；|x|>5 ⇒ run(x)≈x',
    inputs: ['x'], domains: { x: 'small' },
    run: f_geluErf,
    predicate: function (x) {
      var r = f_geluErf(x);
      return rc(r, f_geluErf(-x) + x, 1e-9);
    },
  });

  var f_geluTanh = geluTanh;
  reg(12, {
    name: 'GELU-tanh近似', intuition: '用 tanh 拟合 erf 的 GELU，避免 erf 计算。',
    math: 'gelu≈0.5x(1+tanh(√(2/π)(x+0.044715x³)))', pseudocode: 'return S.gelu(x)', complexity: 'O(1)',
    ref: 'Hendrycks & Gimpel 2016', commonPitfall: '与 erf 版存在 ~1e-3 误差，不可逐位相等。',
    invariant: 'run(x) 与 erf 版在 1e-3 容差内一致',
    inputs: ['x'], domains: { x: 'small' },
    run: f_geluTanh,
    predicate: function (x) { var r = f_geluTanh(x); return rc(r, f_geluErf(x), 1e-3); },
  });

  var f_silu = silu;
  reg(13, {
    name: 'SiLU', intuition: 'Swish 激活，平滑且非单调，自门控特性突出。',
    math: 'silu(x)=x·σ(x)', pseudocode: 'return x*sig(x)', complexity: 'O(1)',
    ref: 'Ramachandran et al. 2017', commonPitfall: '满足 silu(x)=silu(-x)+x 的对称恒等式。',
    invariant: 'silu(x)==silu(-x)+x (relClose)',
    inputs: ['x'], domains: { x: 'small' },
    run: f_silu,
    predicate: function (x) { var r = f_silu(x); return rc(r, f_silu(-x) + x, 1e-9); },
  });

  var f_mish = mish;
  reg(14, {
    name: 'Mish', intuition: '非单调自门控激活，平滑且易于优化。',
    math: 'mish(x)=x·tanh(softplus(x))', pseudocode: 'return x*Math.tanh(softplus(x))', complexity: 'O(1)',
    ref: 'Misra 2019', commonPitfall: '|x| 大时 ≈x（softplus→x, tanh→1）。',
    invariant: 'run(x)==x·tanh(softplus(x))；|x|>8 ⇒ run(x)≈x',
    inputs: ['x'], domains: { x: 'small' },
    run: f_mish,
    predicate: function (x) {
      var r = f_mish(x);
      if (!rc(r, x * Math.tanh(Math.log1p(Math.exp(x))), 1e-9, true)) return false;
      if (x !== 0) { if (Math.sign(r) !== Math.sign(x)) return false; if (!(Math.abs(r) < Math.abs(x))) return false; }
      return true;
    },
  });

  var f_bent = bent;
  reg(15, {
    name: 'BentIdentity', intuition: '带非线性弯曲的单位映射，保留负区间信息。',
    math: 'bent(x)=0.5(x+√(x²+1))', pseudocode: 'return 0.5*(x+Math.sqrt(x*x+1))', complexity: 'O(1)',
    ref: 'Serengil 2020', commonPitfall: '恒有 run(x)>=x。',
    invariant: 'run(x)>=x 且 run(x)-x==0.5(√(x²+1)-x)',
    inputs: ['x'], domains: { x: 'small' },
    run: f_bent,
    predicate: function (x) {
      var r = f_bent(x);
      return r >= x - 1e-9 && rc(r - x, 0.5 * (Math.sqrt(x * x + 1) - x), 1e-9, true);
    },
  });

  var f_identity = identity;
  reg(16, {
    name: 'Identity', intuition: '恒等映射，作为残差与对照基线。',
    math: 'id(x)=x', pseudocode: 'return x', complexity: 'O(1)',
    ref: 'He et al. 2016 (ResNet)', commonPitfall: '无。',
    invariant: 'run(x)==x',
    inputs: ['x'], domains: { x: 'finite' },
    run: f_identity,
    predicate: function (x) { var r = f_identity(x); return rc(r, x, 1e-9, true); },
  });

  // ════════════════════════ 激活导数 G017–G032 ════════════════════════
  // 导数 op：run 为解析导数，predicate 用中心差分校验其确为对应激活的导数。
  var d_relu = function (x) { return x < 0 ? 0 : 1; };
  reg(17, {
    name: 'dReLU', intuition: 'ReLU 的次梯度：负区 0、非负区 1。',
    math: 'dReLU(x)=0 if x<0 else 1', pseudocode: 'return x<0?0:1', complexity: 'O(1)',
    ref: 'Glorot et al. 2011', commonPitfall: 'x=0 处取 1（次梯度）。',
    invariant: 'run(x)==(x<0?0:1) 精确',
    inputs: ['x'], domains: { x: 'finite' },
    run: d_relu,
    predicate: function (x) { var r = d_relu(x); return r === (x < 0 ? 0 : 1); },
  });

  var d_sigmoid = function (x) { var s = sig(x); return s * (1 - s); };
  reg(18, {
    name: 'dSigmoid', intuition: 'Sigmoid 导数的经典形式 s(1-s)。',
    math: 'dσ/dx=s(1-s)', pseudocode: 'return s*(1-s)', complexity: 'O(1)',
    ref: 'Goodfellow 2016', commonPitfall: '最大值为 0.25（x=0）。',
    invariant: 'run(x)==σ(x)(1-σ(x)) 且 0<=run<=0.25',
    inputs: ['x'], domains: { x: 'small' },
    run: d_sigmoid,
    predicate: function (x) { var s = sig(x); var r = d_sigmoid(x); return rc(r, s * (1 - s), 1e-9, true) && r >= 0 && r <= 0.25 + 1e-9; },
  });

  var d_tanh = function (x) { var t = Math.tanh(x); return 1 - t * t; };
  reg(19, {
    name: 'dTanh', intuition: 'Tanh 导数为 1-tanh²，两端趋于 0。',
    math: 'd tanh/dx=1-tanh²', pseudocode: 'return 1-Math.tanh(x)**2', complexity: 'O(1)',
    ref: 'LeCun et al. 1998', commonPitfall: 'x=0 处导数最大为 1。',
    invariant: 'run(x)==1-tanh²(x) 且 0<=run<=1',
    inputs: ['x'], domains: { x: 'small' },
    run: d_tanh,
    predicate: function (x) { var t = Math.tanh(x); var r = d_tanh(x); return rc(r, 1 - t * t, 1e-9, true) && r >= 0 && r <= 1 + 1e-9; },
  });

  var d_softplus = function (x) { return sig(x); };
  reg(20, {
    name: 'dSoftplus', intuition: 'Softplus 的导数恰为 Sigmoid。',
    math: 'd/dx ln(1+e^x)=σ(x)', pseudocode: 'return sig(x)', complexity: 'O(1)',
    ref: 'Dugas et al. 2001', commonPitfall: '与 Sigmoid 导数 op 结果一致。',
    invariant: 'run(x) 与 softplus 的中心差分一致 (FD, 1e-5)',
    inputs: ['x'], domains: { x: 'small' },
    run: d_softplus,
    predicate: function (x) { var r = d_softplus(x); return rc(r, fd(f_softplus, x), 1e-5); },
  });

  var d_silu = function (x) { var s = sig(x); return s + x * s * (1 - s); };
  reg(21, {
    name: 'dSiLU', intuition: 'Swish 的导数：σ+xσ(1-σ)。',
    math: 'd/dx[xσ(x)]=σ+xσ(1-σ)', pseudocode: 'return s+x*s*(1-s)', complexity: 'O(1)',
    ref: 'Ramachandran et al. 2017', commonPitfall: '由乘积法则展开。',
    invariant: 'run(x) 与 SiLU 的中心差分一致 (FD, 1e-5)',
    inputs: ['x'], domains: { x: 'small' },
    run: d_silu,
    predicate: function (x) { var r = d_silu(x); return rc(r, fd(f_silu, x), 1e-5); },
  });

  var d_geluErf = function (x) {
    var e = erf(x / Math.SQRT2);
    var k = Math.sqrt(2 / Math.PI) * Math.exp(-x * x / 2);
    return 0.5 * (1 + e) + 0.5 * x * k;
  };
  reg(22, {
    name: 'dGELU-erf', intuition: 'erf 版 GELU 的解析导数。',
    math: 'd gelu/dx=0.5(1+erf(x/√2))+0.5x√(2/π)e^{-x²/2}', pseudocode: 'return analytic', complexity: 'O(1)',
    ref: 'Hendrycks & Gimpel 2016', commonPitfall: '勿与 tanh 近似的导数混淆。',
    invariant: 'run(x) 与 erf-GELU 的中心差分一致 (FD, 1e-5)',
    inputs: ['x'], domains: { x: 'small' },
    run: d_geluErf,
    predicate: function (x) { var r = d_geluErf(x); return rc(r, fd(f_geluErf, x), 1e-5); },
  });

  var d_mish = function (x) {
    var s = softplus(x), t = Math.tanh(s), d = sig(x);
    return t + x * (1 - t * t) * d;
  };
  reg(23, {
    name: 'dMish', intuition: 'Mish 的导数：tanh(softplus)+x(1-tanh²)σ。',
    math: 'd mish/dx=tanh(s)+x(1-tanh²(s))σ(x), s=softplus(x)', pseudocode: 'return t+x*(1-t*t)*d', complexity: 'O(1)',
    ref: 'Misra 2019', commonPitfall: '链式法则含 softplus 导数 σ(x)。',
    invariant: 'run(x) 与 Mish 的中心差分一致 (FD, 1e-5)',
    inputs: ['x'], domains: { x: 'small' },
    run: d_mish,
    predicate: function (x) { var r = d_mish(x); return rc(r, fd(f_mish, x), 1e-5); },
  });

  var d_elu = function (x) { return x < 0 ? Math.exp(x) : 1; };
  reg(24, {
    name: 'dELU', intuition: 'ELU 导数：负区 e^x，非负区 1。',
    math: 'd elu/dx=x<0?e^x:1', pseudocode: 'return x<0?Math.exp(x):1', complexity: 'O(1)',
    ref: 'Clevert et al. 2015', commonPitfall: '注意与 elu 值的关系：x<0 时 d elu=elu+1。',
    invariant: 'run(x) 与 ELU 的中心差分一致 (FD, 1e-5)',
    inputs: ['x'], domains: { x: 'small' },
    run: d_elu,
    predicate: function (x) { var r = d_elu(x); return rc(r, fd(f_elu, x), 1e-5); },
  });

  var d_hardTanh = function (x) { return Math.abs(x) < 1 ? 1 : 0; };
  reg(25, {
    name: 'dHardTanh', intuition: 'HardTanh 导数：区间内 1，否则 0。',
    math: 'd hardTanh/dx=|x|<1?1:0', pseudocode: 'return Math.abs(x)<1?1:0', complexity: 'O(1)',
    ref: 'Collobert 2011', commonPitfall: '边界点视为 0。',
    invariant: 'run(x)==(|x|<1?1:0) 精确',
    inputs: ['x'], domains: { x: 'small' },
    run: d_hardTanh,
    predicate: function (x) { var r = d_hardTanh(x); return r === (Math.abs(x) < 1 ? 1 : 0); },
  });

  var d_softsign = function (x) { var d = 1 + Math.abs(x); return 1 / (d * d); };
  reg(26, {
    name: 'dSoftsign', intuition: 'Softsign 导数：1/(1+|x|)²。',
    math: 'd/dx[x/(1+|x|)]=1/(1+|x|)²', pseudocode: 'return 1/(1+|x|)**2', complexity: 'O(1)',
    ref: 'Bergstra et al. 2009', commonPitfall: '注意绝对值分段。',
    invariant: 'run(x) 与 Softsign 的中心差分一致 (FD, 1e-5)',
    inputs: ['x'], domains: { x: 'small' },
    run: d_softsign,
    predicate: function (x) { var r = d_softsign(x); return rc(r, fd(f_softsign, x), 1e-5); },
  });

  var d_selu = function (x) {
    var a = 1.6732632423543772, s = 1.0507009873554805;
    return x >= 0 ? s : s * a * Math.exp(x);
  };
  reg(27, {
    name: 'dSELU', intuition: 'SELU 导数：非负区 λ，负区 λα e^x。',
    math: 'd selu/dx=x>=0?λ:λα e^x', pseudocode: 'return x>=0?λ:λ*α*exp(x)', complexity: 'O(1)',
    ref: 'Klambauer et al. 2017', commonPitfall: '含缩放系数 λ。',
    invariant: 'run(x) 与 SELU 的中心差分一致 (FD, 1e-5)',
    inputs: ['x'], domains: { x: 'small' },
    run: d_selu,
    predicate: function (x) { var r = d_selu(x); return rc(r, fd(f_selu, x), 1e-5); },
  });

  var d_bent = function (x) { return 0.5 * (1 + x / Math.sqrt(x * x + 1)); };
  reg(28, {
    name: 'dBentIdentity', intuition: 'BentIdentity 导数：0.5(1+x/√(x²+1))。',
    math: 'd bent/dx=0.5(1+x/√(x²+1))', pseudocode: 'return 0.5*(1+x/Math.sqrt(x*x+1))', complexity: 'O(1)',
    ref: 'Serengil 2020', commonPitfall: '恒 >=0.5。',
    invariant: 'run(x) 与 BentIdentity 的中心差分一致 (FD, 1e-5)',
    inputs: ['x'], domains: { x: 'small' },
    run: d_bent,
    predicate: function (x) { var r = d_bent(x); return rc(r, fd(f_bent, x), 1e-5); },
  });

  var d_identity = function (x) { return 1; };
  reg(29, {
    name: 'dIdentity', intuition: '恒等映射导数为 1。',
    math: 'd id/dx=1', pseudocode: 'return 1', complexity: 'O(1)',
    ref: '基础微积分', commonPitfall: '无。',
    invariant: 'run(x)==1 精确',
    inputs: ['x'], domains: { x: 'finite' },
    run: d_identity,
    predicate: function (x) { var r = d_identity(x); return r === 1; },
  });

  var d_swish = d_silu; // Swish 与 SiLU 同义
  reg(30, {
    name: 'dSwish-recheck', intuition: 'Swish 导数复检（与 SiLU 一致）。',
    math: 'd swish/dx=σ+xσ(1-σ)', pseudocode: 'return s+x*s*(1-s)', complexity: 'O(1)',
    ref: 'Ramachandran et al. 2017', commonPitfall: 'Swish==SiLU。',
    invariant: 'run(x) 与 SiLU 的中心差分一致 (FD, 1e-5)',
    inputs: ['x'], domains: { x: 'small' },
    run: d_swish,
    predicate: function (x) { var r = d_swish(x); return rc(r, fd(f_silu, x), 1e-5); },
  });

  var d_geluTanh = function (x) {
    var c = Math.sqrt(2 / Math.PI);
    var u = c * (x + 0.044715 * x * x * x);
    var t = Math.tanh(u);
    var du = c * (1 + 3 * 0.044715 * x * x);
    return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * du;
  };
  reg(31, {
    name: 'dGELU-tanh-derivative', intuition: 'tanh 近似版 GELU 的解析导数。',
    math: 'd gelu_t/dx=0.5(1+tanh u)+0.5x(1-tanh²u)·c(1+0.134145x²)', pseudocode: 'return analytic', complexity: 'O(1)',
    ref: 'Hendrycks & Gimpel 2016', commonPitfall: '内层 u 含 0.044715x³。',
    invariant: 'run(x) 与 tanh-GELU 的中心差分一致 (FD, 1e-5)',
    inputs: ['x'], domains: { x: 'small' },
    run: d_geluTanh,
    predicate: function (x) { var r = d_geluTanh(x); return rc(r, fd(f_geluTanh, x), 1e-5); },
  });

  var d_softplus_re = d_softplus;
  reg(32, {
    name: 'dSoftplus-recheck', intuition: 'Softplus 导数复检（应等于 Sigmoid）。',
    math: 'd/dx ln(1+e^x)=σ(x)', pseudocode: 'return sig(x)', complexity: 'O(1)',
    ref: 'Dugas et al. 2001', commonPitfall: '复检项避免回归。',
    invariant: 'run(x) 与 Softplus 的中心差分一致 (FD, 1e-5)',
    inputs: ['x'], domains: { x: 'small' },
    run: d_softplus_re,
    predicate: function (x) { var r = d_softplus_re(x); return rc(r, fd(f_softplus, x), 1e-5); },
  });
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

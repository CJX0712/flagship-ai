/* ============================================================================
 * L9 数值基础 · 特殊函数（special）
 * ----------------------------------------------------------------------------
 * 所属层：L9-numBase
 * 允许依赖的层：无
 * 主要 @ref：Goodfellow et al. (2016) Deep Learning；Hendrycks & Gimpel (2016)
 *           GELU；Ramachandran et al. (2017) Swish；Clevert et al. (2015) ELU；
 *           Abramowitz & Stegun (1964) §7.1.26 erf 近似。
 *
 * 责任：提供神经网络中常用的标量特殊函数（确定性、纯函数）。每个函数附论文
 *       出处与单调性/凹凸性等不变性质，供 L10 微指令直接引用（零重写）。
 * ==========================================================================*/
;(function (NS) {
  'use strict';
  NS.L9 = NS.L9 || {};
  NS.L9.special = NS.L9.special || {};

  const GELU_C = Math.sqrt(2 / Math.PI); // GPT-2 tanh 近似系数

  // Sigmoid：σ(x)=1/(1+e^{-x})。单调递增，(−∞,0)→0，(+∞,1)→1。
  // @invariant σ(-x)=1-σ(x)；σ(0)=0.5；严格单调增。
  function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

  // softplus：log(1+e^x)。ReLU 的平滑近似，处处正、严格单调增。
  // @invariant softplus(x) >= 0；x→+∞ 趋近 x；x→-∞ 趋近 0。
  function softplus(x) { return Math.log1p(Math.exp(x)); }

  // ReLU：max(0,x)。
  // @invariant relu(0)=0；x>0 时 relu(x)=x。
  function relu(x) { return x > 0 ? x : 0; }

  // ELU：x>=0 为 x，否则 α(e^x-1)。默认 α=1。
  // @ref Clevert et al. 2015。
  function elu(x, alpha) { alpha = alpha == null ? 1 : alpha; return x >= 0 ? x : alpha * (Math.exp(x) - 1); }

  // SELU：scale·ELU(x,alpha)，默认值取自 Klambauer et al. (2017) SNN。
  function selu(x) {
    const a = 1.6732632423543772, s = 1.0507009873554805;
    return s * (x >= 0 ? x : a * (Math.exp(x) - 1));
  }

  // GELU（tanh 近似，GPT-2 标准）。
  // @ref Hendrycks & Gimpel 2016。
  function gelu(x) { return 0.5 * x * (1 + Math.tanh(GELU_C * (x + 0.044715 * x * x * x))); }

  // SiLU / Swish：x·σ(x)。@ref Ramachandran 2017。
  // @invariant x<0 时 ∈(-1/e·s,0)；单调增；x→0⁻ ≈ x/2。
  function swish(x) { return x / (1 + Math.exp(-x)); }

  // 两元素 log-sum-exp：log(e^{a}+e^{b})，用 max 平移防溢出。
  // @ref numerical recipes。@invariant logsumexp2(a,b) >= max(a,b)。
  function logsumexp2(a, b) {
    const m = Math.max(a, b);
    return m + Math.log(Math.exp(a - m) + Math.exp(b - m));
  }

  // Abramowitz & Stegun 7.1.26 误差函数近似（最大相对误差 ~1.5e-7）。
  // @complexity O(1)。@invariant erf(-x)=-erf(x)；erf(0)=0；|erf(x)|<1。
  function erf(x) {
    const s = x < 0 ? -1 : 1;
    const z = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * z);
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-z * z);
    return s * y;
  }

  // 行稳定 softmax（单向量）：先减最大值再 exp，避免溢出；行和 = 1（有限输入下）。
  // @invariant 输出非负且和=1（至浮点容差）；argmax 位置概率最大。
  function softmaxStable(v) {
    let m = -Infinity;
    for (let i = 0; i < v.length; i++) if (v[i] > m) m = v[i];
    let s = 0;
    const out = new Float64Array(v.length);
    for (let i = 0; i < v.length; i++) { const e = Math.exp(v[i] - m); out[i] = e; s += e; }
    const inv = 1 / s;
    for (let i = 0; i < v.length; i++) out[i] *= inv;
    return out;
  }

  NS.L9.special.sigmoid = sigmoid;
  NS.L9.special.softplus = softplus;
  NS.L9.special.relu = relu;
  NS.L9.special.elu = elu;
  NS.L9.special.selu = selu;
  NS.L9.special.gelu = gelu;
  NS.L9.special.swish = swish;
  NS.L9.special.logsumexp2 = logsumexp2;
  NS.L9.special.erf = erf;
  NS.L9.special.softmaxStable = softmaxStable;
  NS.L9.special.GELU_C = GELU_C;
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

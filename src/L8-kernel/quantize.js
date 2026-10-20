/* ============================================================================
 * L8 数值内核 · 权重量化（quantize）
 * ----------------------------------------------------------------------------
 * 所属层：L8-kernel
 * 允许依赖的层：无
 * 主要 @ref：Dettmers et al. (2022) LLM.int8()；Lin et al. (2023) AWQ；
 *           GPTQ (Frantar et al. 2022)
 *
 * 责任：对称逐张量 INT8 量化（weight-only PTQ）的张量级原语。与 v11
 *       GPTModel.quantize 的量化逻辑逐字一致（保证 R8 量化 logits 误差界）。
 *       注意：本层只提供量化/解量化数学，不接触模型结构（模型级量化在 L3/L2）。
 * ==========================================================================*/
;(function (NS) {
  'use strict';
  NS.L8 = NS.L8 || {};
  NS.L8.quantize = NS.L8.quantize || {};

  // 对称逐张量量化：scale = max|w|/127，q = round(w/scale) 钳位到 [-127,127]。
  // @invariant scale>=0；q 元素 ∈ [-127,127]；dequant(q,scale) 为 w 的近似。
  // @complexity O(n)。@ref LLM.int8（对称量化）。
  function quantizeTensor(d) {
    let maxAbs = 0;
    for (let i = 0; i < d.length; i++) { const a = Math.abs(d[i]); if (a > maxAbs) maxAbs = a; }
    const scale = maxAbs / 127 || 1;
    const q = new Int8Array(d.length);
    for (let i = 0; i < d.length; i++) {
      let v = Math.round(d[i] / scale);
      q[i] = v > 127 ? 127 : v < -127 ? -127 : v;
    }
    return { q, scale };
  }

  // 解量化：w ≈ q · scale。返回 Float64Array。
  function dequantizeTensor(t) {
    const out = new Float64Array(t.q.length);
    for (let i = 0; i < out.length; i++) out[i] = t.q[i] * t.scale;
    return out;
  }

  // 量化相对误差上界估计（用于误差预算 K10 登记）。返回 ||w - deq||_max / ||w||_max。
  // @invariant 对稠密权重，相对误差通常 < 0.03（R8 经验界）。
  function quantError(d) {
    const t = quantizeTensor(d);
    const deq = dequantizeTensor(t);
    let num = 0, den = 0;
    for (let i = 0; i < d.length; i++) {
      num = Math.max(num, Math.abs(d[i] - deq[i]));
      den = Math.max(den, Math.abs(d[i]));
    }
    return den === 0 ? 0 : num / den;
  }

  NS.L8.quantize.quantizeTensor = quantizeTensor;
  NS.L8.quantize.dequantizeTensor = dequantizeTensor;
  NS.L8.quantize.quantError = quantError;
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

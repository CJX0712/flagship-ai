/* ============================================================================
 * L8 数值内核 · 归约算子（reduce）
 * ----------------------------------------------------------------------------
 * 所属层：L8-kernel
 * 允许依赖的层：无
 * 主要 @ref：softmax (Bridle 1990 "Probabilistic Interpretation of Neural
 *           Networks")；top-k 路由 (Shazeer et al. 2017 Switch Transformer)
 *
 * 责任：提供按行归约内核。softmaxRows/softmaxBwd/addInto/rowSumInto/topkRows
 *       函数体逐字沿用 v11（保证 R9 Δ=0）。另补充 rowSum（独立归约原语）。
 * 存储序（K1）：行优先。
 * ==========================================================================*/
;(function (NS) {
  'use strict';
  NS.L8 = NS.L8 || {};
  NS.L8.reduce = NS.L8.reduce || {};

  // 按行累加到 dst（就地）：dst += src。
  function addInto(dst, src) {
    for (let i = 0; i < dst.length; i++) dst[i] += src[i];
  }

  // 按行求和并累加到 dst（dst 为 c 长度的行和缓冲）：dst[j] += Σ_i Y[i,j]。
  function rowSumInto(dst, Y, r, c) {
    for (let i = 0; i < r; i++) {
      const o = i * c;
      for (let j = 0; j < c; j++) dst[j] += Y[o + j];
    }
  }

  // 按行求和，返回长度 c 的向量（独立归约原语，不就地）。
  // @invariant rowSum(Y,r,c)[j] === Σ_i Y[i,j]；r=0 时返回全 0。
  function rowSum(Y, r, c) {
    const s = new Float64Array(c);
    rowSumInto(s, Y, r, c);
    return s;
  }

  // 按行 softmax（数值稳定：先减行最大值）。
  function softmaxRows(S, r, c) {
    const P = new Float64Array(r * c);
    for (let i = 0; i < r; i++) {
      const o = i * c;
      let mx = -Infinity;
      for (let j = 0; j < c; j++) if (S[o + j] > mx) mx = S[o + j];
      let s = 0;
      for (let j = 0; j < c; j++) { const e = Math.exp(S[o + j] - mx); P[o + j] = e; s += e; }
      const inv = 1 / s;
      for (let j = 0; j < c; j++) P[o + j] *= inv;
    }
    return P;
  }

  // softmax 反向：dS = P ⊙ (dP − Σ_j dP_j·P_j)。
  function softmaxBwd(dP, P, r, c) {
    const dS = new Float64Array(r * c);
    for (let i = 0; i < r; i++) {
      const o = i * c;
      let s = 0;
      for (let j = 0; j < c; j++) s += dP[o + j] * P[o + j];
      for (let j = 0; j < c; j++) dS[o + j] = P[o + j] * (dP[o + j] - s);
    }
    return dS;
  }

  // 每行取概率最大的 k 个下标（升序稳定排序后取尾部），返回 idx/val 行主序数组
  function topkRows(P, r, c, k) {
    const idx = new Int32Array(r * k);
    const val = new Float64Array(r * k);
    for (let i = 0; i < r; i++) {
      const o = i * c;
      const order = [];
      for (let j = 0; j < c; j++) order.push(j);
      order.sort((a, b) => P[o + a] - P[o + b]); // 升序；并列保持插入序（稳定）
      for (let t = 0; t < k; t++) {
        const e = order[c - 1 - t];
        idx[i * k + t] = e;
        val[i * k + t] = P[o + e];
      }
    }
    return { idx, val };
  }

  NS.L8.reduce.addInto = addInto;
  NS.L8.reduce.rowSumInto = rowSumInto;
  NS.L8.reduce.rowSum = rowSum;
  NS.L8.reduce.softmaxRows = softmaxRows;
  NS.L8.reduce.softmaxBwd = softmaxBwd;
  NS.L8.reduce.topkRows = topkRows;
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

/* L4 神经算子 · 交叉熵损失
 * 所属层：L4（神经算子）｜ 允许依赖：L8、L9
 * 主要 @ref：Softmax + NLL（等价于 cross-entropy）；数值稳定 log-sum-exp 取 max
 * 迁移方式：从 legacy flagship-engine.v11.js 第 502–521 行【原样搬】。
 *   自包含，无跨层调用，逐位等价有 R9 把关。
 *   @param logits  Float64Array(n·V) 行主序
 *   @param targets number[] 长度 n 的目标 id
 *   @param n       有效前 n 行（目标行）
 *   @param V       词表宽度
 *   @param totalRows 补零至完整行数（供反向使用），缺省 = n
 *   @returns {loss, dlogits} dlogits 已乘 1/n
 */
;(function (NS) {
  NS.L4 = NS.L4 || {};
  NS.L4.loss = NS.L4.loss || {};

  function crossEntropy(logits, targets, n, V, totalRows) {
    const rows = totalRows || n;
    let loss = 0;
    const dlogits = new Float64Array(rows * V);
    for (let i = 0; i < n; i++) {
      const o = i * V;
      let mx = -Infinity;
      for (let j = 0; j < V; j++) if (logits[o + j] > mx) mx = logits[o + j];
      let s = 0;
      for (let j = 0; j < V; j++) { const e = Math.exp(logits[o + j] - mx); dlogits[o + j] = e; s += e; }
      const inv = 1 / s;
      for (let j = 0; j < V; j++) dlogits[o + j] *= inv;
      loss -= Math.log(dlogits[o + targets[i]]);
      dlogits[o + targets[i]] -= 1;
    }
    const invN = 1 / n;
    for (let k = 0; k < n * V; k++) dlogits[k] *= invN;
    return { loss: loss * invN, dlogits };
  }

  NS.L4.loss.crossEntropy = crossEntropy;
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

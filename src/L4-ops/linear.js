/* L4 神经算子 · 线性层门面（mm + bias-add 统一封装）
 * 所属层：L4（神经算子）｜ 允许依赖：L8（数值内核 matmul 族）、L9
 * 主要 @ref：LLM.int8() 权重解量化分发；weight-only PTQ
 * 说明：量化分支由 L8 的 mm/mmAtB 承担（param 含 .q 即走 matmulQ/matmulAtBQ）。
 *   本文件提供 L4 语义门面：把 mm/mmAtB 与「逐行 bias 广播累加」组合成 linear，
 *   与 legacy GPTModel.forward/backward 内的 inline 写法逐位等价。
 */
;(function (NS) {
  NS.L4 = NS.L4 || {};
  NS.L4.linear = NS.L4.linear || {};
  const L8 = NS.L8;

  // mm：A(r×k) @ W，W 含 .q 则走解量化路径（含量化分支）
  function mm(A, W, r, k, c) { return L8.matmul.mm(A, W, r, k, c); }
  function mmAtB(A, W, r, k, m) { return L8.matmul.mmAtB(A, W, r, k, m); }

  // 逐行 bias 广播累加：y[i·c+j] += b[j]（与 legacy forward/backward inline 循环逐位一致）
  function addBias(y, b, r, c) {
    for (let i = 0; i < r; i++) {
      const o = i * c;
      for (let j = 0; j < c; j++) y[o + j] += b[j];
    }
    return y;
  }

  // 线性层前向：先 mm，再 bias 广播累加（含量化分支）
  function linear(A, W, b, r, k, c) {
    const y = mm(A, W, r, k, c);
    if (b) addBias(y, b, r, c);
    return y;
  }

  NS.L4.linear.mm = mm;
  NS.L4.linear.mmAtB = mmAtB;
  NS.L4.linear.addBias = addBias;
  NS.L4.linear.linear = linear;
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

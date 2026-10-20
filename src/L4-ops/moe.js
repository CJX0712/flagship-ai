/* L4 神经算子 · MoE 稀疏前馈（路由 + top-k 专家散射 + 负载均衡辅助损失）
 * 所属层：L4（神经算子）｜ 允许依赖：L8（matmul 族 / 归约 / softmax）、L4.activation、L9
 * 主要 @ref：Shazeer 2018 (Sparsely-Gated MoE)；Switch Transformer / GDP-MoE 负载均衡
 * 迁移方式：从 legacy flagship-engine.v11.js 第 415–500 行【原样搬】。
 *   内部跨调用已改写为 NS.L8 / NS.L4.activation（数值逐位一致，有 R9/probe-moe 把关）。
 */
;(function (NS) {
  NS.L4 = NS.L4 || {};
  NS.L4.moe = NS.L4.moe || {};
  const L8 = NS.L8, act = NS.L4.activation;
  const mm = L8.matmul.mm, softmaxRows = L8.reduce.softmaxRows, topkRows = L8.reduce.topkRows;
  const addInto = L8.reduce.addInto, rowSumInto = L8.reduce.rowSumInto;
  const matmulTA = L8.matmul.matmulTA, matmulAtB = L8.matmul.matmulAtB, softmaxBwd = L8.reduce.softmaxBwd;

  /* ---- MoE 前馈：路由网络 + top-k 专家散射 + 负载均衡辅助损失 ----
   * 前向：logits = M·wg + bg → softmax → top-k 选专家 → 各专家 FFN 加权求和。
   * 反向所需的中间量（各专家子集的 M/pre/h1/out、路由概率、计数）一并缓存。 */
  function moeForward(model, P, k, M, T, D, FH) {
    const E = model.nExp, K = model.topK;
    const W1 = model.useSwi ? 2 * FH : FH;
    const rlogits = mm(M, P[k + 'wg'], T, D, E);
    for (let i = 0; i < T; i++) { const o = i * E; for (let j = 0; j < E; j++) rlogits[o + j] += P[k + 'bg'].d[j]; }
    const probs = softmaxRows(rlogits, T, E);
    const tk = topkRows(probs, T, E, K);
    const f = new Int32Array(E);
    const exp = [];
    for (let e = 0; e < E; e++) exp.push({ rows: [], g: [], pre: null, h1: null, out: null, Msub: null });
    for (let i = 0; i < T; i++) {
      for (let j = 0; j < K; j++) { const e = tk.idx[i * K + j]; exp[e].rows.push(i); exp[e].g.push(tk.val[i * K + j]); f[e]++; }
    }
    const ff = new Float64Array(T * D);
    for (let e = 0; e < E; e++) {
      const ex = exp[e];
      const cnt = ex.rows.length;
      if (cnt === 0) continue;
      const Msub = new Float64Array(cnt * D);
      for (let r2 = 0; r2 < cnt; r2++) { const oi = ex.rows[r2] * D, o2 = r2 * D; for (let j = 0; j < D; j++) Msub[o2 + j] = M[oi + j]; }
      const ek = `${k}e${e}.`;
      const pre = mm(Msub, P[ek + 'w1'], cnt, D, W1);
      for (let r2 = 0; r2 < cnt; r2++) { const o = r2 * W1; for (let j = 0; j < W1; j++) pre[o + j] += P[ek + 'b1'].d[j]; }
      const h1 = model.useSwi ? act.swigluFwd(pre, cnt, FH) : act.geluFwd(pre);
      const out = mm(h1, P[ek + 'w2'], cnt, FH, D);
      for (let r2 = 0; r2 < cnt; r2++) { const o = r2 * D; for (let j = 0; j < D; j++) out[o + j] += P[ek + 'b2'].d[j]; }
      ex.pre = pre; ex.h1 = h1; ex.out = out; ex.Msub = Msub;
      for (let r2 = 0; r2 < cnt; r2++) { const oi = ex.rows[r2] * D, o2 = r2 * D, g = ex.g[r2]; for (let j = 0; j < D; j++) ff[oi + j] += g * out[o2 + j]; }
    }
    // 辅助负载均衡损失（Switch/GDP-MoE）：E·Σ_e f_e·P_e，冻结 f_e（离散选择不回传）
    let aux = 0;
    for (let e = 0; e < E; e++) {
      let sp = 0;
      for (let i = 0; i < T; i++) sp += probs[i * E + e];
      aux += E * (f[e] / T) * (sp / T);
    }
    return { ff, moe: { rlogits, probs, idx: tk.idx, gate: tk.val, exp, f, E, K, T, M, k, aux } };
  }

  /* ---- MoE 反向：专家路径（标准 FFN 反向，按 top-k 子集累加）+ 路由路径 ----
   * 路由路径：dgate=dot(dff, out) 注入 dprobs（仅选中位置）；再加辅助损失梯度
   * ∂/∂probs[i,e]=auxWeight·E·count_e/T²（冻结 f_e），经 softmaxBwd 回传至 wg/bg。 */
  function moeBackward(model, P, G, moe, dff, T, D) {
    const E = moe.E, K = moe.K, FH = model.fh, W1 = model.useSwi ? 2 * FH : FH;
    const dM = new Float64Array(T * D);
    const k = moe.k;
    // 1) 专家路径
    for (let e = 0; e < E; e++) {
      const ex = moe.exp[e];
      const cnt = ex.rows.length;
      if (cnt === 0) continue;
      const ek = `${k}e${e}.`;
      const dout = new Float64Array(cnt * D);
      for (let r2 = 0; r2 < cnt; r2++) { const oi = ex.rows[r2] * D, o2 = r2 * D, g = ex.g[r2]; for (let j = 0; j < D; j++) dout[o2 + j] = dff[oi + j] * g; }
      addInto(G[ek + 'w2'], matmulTA(ex.h1, dout, cnt, FH, D));
      rowSumInto(G[ek + 'b2'], dout, cnt, D);
      const dh1 = matmulAtB(dout, P[ek + 'w2'].d, cnt, D, FH);
      const dpre = model.useSwi ? act.swigluBwd(dh1, ex.pre, cnt, FH) : act.geluBwd(dh1, ex.pre);
      addInto(G[ek + 'w1'], matmulTA(ex.Msub, dpre, cnt, D, W1));
      rowSumInto(G[ek + 'b1'], dpre, cnt, W1);
      const dMsub = matmulAtB(dpre, P[ek + 'w1'].d, cnt, W1, D);
      for (let r2 = 0; r2 < cnt; r2++) { const oi = ex.rows[r2] * D, o2 = r2 * D; for (let j = 0; j < D; j++) dM[oi + j] += dMsub[o2 + j]; }
    }
    // 2) 路由路径
    const dprobs = new Float64Array(T * E);
    for (let e = 0; e < E; e++) {
      const ex = moe.exp[e];
      for (let r2 = 0; r2 < ex.rows.length; r2++) {
        const ti = ex.rows[r2];
        let s = 0; for (let j = 0; j < D; j++) s += dff[ti * D + j] * ex.out[r2 * D + j];
        dprobs[ti * E + e] = s;
      }
    }
    // 辅助损失梯度（冻结 f_e）：∂/∂probs[i,e] = auxWeight·E·count_e/T²
    const aw = model.auxWeight;
    for (let e = 0; e < E; e++) {
      const term = aw * E * moe.f[e] / (T * T);
      for (let i = 0; i < T; i++) dprobs[i * E + e] += term;
    }
    const dRlogits = softmaxBwd(dprobs, moe.probs, T, E);
    addInto(G[k + 'wg'], matmulTA(moe.M, dRlogits, T, D, E));
    rowSumInto(G[k + 'bg'], dRlogits, T, E);
    const dMr = matmulAtB(dRlogits, P[k + 'wg'].d, T, E, D);
    for (let i = 0; i < T * D; i++) dM[i] += dMr[i];
    return dM;
  }

  NS.L4.moe.moeForward = moeForward;
  NS.L4.moe.moeBackward = moeBackward;
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

/* L3 架构 · Block 级 forward/backward 组装（替代 legacy 123+134 行单体）
 * 所属层：L3（架构组装）｜ 允许依赖：L4（神经算子）、L8（数值内核）、L9
 * 主要 @ref：GPT-2 (Radford 2019) 残差栈；LLaMA/Mistral (RMSNorm+SwiGLU+RoPE)；
 *            Gemma 2/3 (QK-norm + logit soft-cap + 层交错 SWA)；MoE (Shazeer 2018)
 * 迁移方式：把 legacy GPTModel.forward(911–1033) / backward(1035–1168) 原样重组装。
 *   算子调用全部下沉到 L4（norm/activation/positional/attention/moe），数值内核下沉到
 *   L8（matmul 族 / softmax / 归约），本文件只保留「残差加法 / 组装顺序 / 缓存布局」。
 *   调用顺序与 legacy 逐行一致 → 输出逐位等价（R2 logits Δ=0 / R3 grads Δ=0）。
 *   index.js 以 GPTModel.prototype.forward/backward = blocks.forward/backward 接管，
 *   构造器（RNG 消费顺序）仍冻结在 legacy，R1 不受影响。
 */
;(function (NS) {
  NS.L3 = NS.L3 || {};
  NS.L3.blocks = NS.L3.blocks || {};

  /** 前向传播，返回 { logits(T×V), cache }，cache 供反向传播使用 */
  function forward(model, tokens) {
    const { vocabSize: V, blockSize: B, nLayer: L, nHead: H, dModel: D } = model.cfg;
    const dh = model.dh, KV = model.KV, g = model.g, qkvDim = model.qkvDim;
    const P = model.params;
    const T = Math.min(tokens.length, B);
    const toks = tokens.length > B ? tokens.slice(tokens.length - B) : tokens;

    const L4 = NS.L4, L8 = NS.L8;
    const mm = L8.matmul.mm, mmAtB = L8.matmul.mmAtB;
    const ropeApply = L4.positional.ropeApply, qkNormFwd = L4.norm.qkNormFwd;
    const attnFwd = L4.attention.attnFwd;
    const swigluFwd = L4.activation.swigluFwd, geluFwd = L4.activation.geluFwd;
    const moeForward = L4.moe.moeForward;
    const NF = model.useRms ? L4.norm.rmsnormFwd : L4.norm.layernormFwd;
    const nb = (key) => (P[key] ? P[key].d : null);

    // 1) token 嵌入 + 位置嵌入（RoPE 模式下无位置嵌入）
    let X = new Float64Array(T * D);
    const te = P.tokEmb, pe = P.posEmb;
    const teQ = te.q, peQ = pe ? pe.q : null;
    for (let i = 0; i < T; i++) {
      const t = toks[i], o = i * D;
      for (let j = 0; j < D; j++) {
        const tv = teQ ? te.q[t * D + j] * te.scale : te.d[t * D + j];
        X[o + j] = peQ ? tv + pe.q[o + j] * pe.scale : (pe ? tv + pe.d[o + j] : tv);
      }
    }

    const layers = [];
    for (let l = 0; l < L; l++) {
      const k = `l${l}.`;
      // 2) pre-LN #1 -> 因果多头自注意力
      const ln1 = NF(X, P[k + 'ln1.g'].d, nb(k + 'ln1.b'), T, D);
      const A = ln1.Y;
      const qkv = mm(A, P[k + 'wqkv'], T, D, qkvDim);
      for (let i = 0; i < T; i++) {
        const o = i * qkvDim;
        for (let j = 0; j < qkvDim; j++) qkv[o + j] += P[k + 'bqkv'].d[j];
      }
      // RoPE：就地旋转 Q/K 段（V 段不旋转）。QK-norm 先于 RoPE（Gemma 2 顺序）
      let qk = null;
      if (model.qkNorm) {
        const qn = qkNormFwd(qkv, T, H, dh, qkvDim, 0, P[k + 'qg'].d);  // Q 段（偏移 0，H 个头）
        const kn = qkNormFwd(qkv, T, KV, dh, qkvDim, D, P[k + 'kg'].d); // K 段（偏移 D，KV 个头）
        qk = { qx: qn.xhat, qr: qn.rstd, kx: kn.xhat, kr: kn.rstd };
      }
      if (model.useRope) ropeApply(qkv, T, D, H, dh, 0, 1, KV, qkvDim, model.ropeFreqScale);
      const scale = 1 / Math.sqrt(dh);
      const attnScale = model.attentionScaling; // YaRN 注意力温度（=1 恒等）
      const { Ps, ctx } = attnFwd(qkv, T, D, H, dh, qkvDim, KV, g, model.windows[l], scale, attnScale);
      let attn = mm(ctx, P[k + 'wproj'], T, D, D);
      for (let i = 0; i < T; i++) {
        const o = i * D;
        for (let j = 0; j < D; j++) attn[o + j] += P[k + 'bproj'].d[j];
      }
      const xMid = new Float64Array(T * D);
      for (let i = 0; i < T * D; i++) xMid[i] = X[i] + attn[i]; // 残差连接

      // 3) pre-LN #2 -> 前馈网络（MoE 稀疏 / 稠密 GELU/SwiGLU 双路径）
      const ln2 = NF(xMid, P[k + 'ln2.g'].d, nb(k + 'ln2.b'), T, D);
      const M = ln2.Y;
      const FH = model.fh;
      let ff, pre1 = null, h1 = null, moe = null;
      if (model.useMoE) {
        const r = moeForward(model, P, k, M, T, D, FH);
        ff = r.ff; moe = r.moe;
      } else {
        const W1 = model.useSwi ? 2 * FH : FH;
        pre1 = mm(M, P[k + 'w1'], T, D, W1);
        for (let i = 0; i < T; i++) {
          const o = i * W1;
          for (let j = 0; j < W1; j++) pre1[o + j] += P[k + 'b1'].d[j];
        }
        h1 = model.useSwi ? swigluFwd(pre1, T, FH) : geluFwd(pre1);
        ff = mm(h1, P[k + 'w2'], T, FH, D);
        for (let i = 0; i < T; i++) {
          const o = i * D;
          for (let j = 0; j < D; j++) ff[o + j] += P[k + 'b2'].d[j];
        }
      }
      const xOut = new Float64Array(T * D);
      for (let i = 0; i < T * D; i++) xOut[i] = xMid[i] + ff[i]; // 残差连接

      layers.push({ ln1, A, qkv, Ps, ctx, xMid, ln2, M, pre1, h1, moe, qk });
      X = xOut;
    }

    // 4) 最终归一化 + 权重共享输出头（+ Gemma 2 logit soft-cap：cap·tanh(x/cap)）
    const lnf = NF(X, P['lnf.g'].d, nb('lnf.b'), T, D);
    const Hf = lnf.Y;
    let logits = mmAtB(Hf, P.tokEmb, T, D, V);
    if (model.logitCap > 0) {
      const cap = model.logitCap;
      for (let i = 0; i < logits.length; i++) logits[i] = cap * Math.tanh(logits[i] / cap);
    }
    return { logits, cache: { T, dh, H, D, L, layers, lnf, Hf, toks, logits } };
  }

  /** 反向传播：dlogits(T×V) -> 累加至 model.grads */
  function backward(model, cache, dlogits) {
    const { T, dh, H, D, L } = cache;
    const { vocabSize: V } = model.cfg;
    const P = model.params, G = model.grads;
    const NB = model.useRms ? NS.L4.norm.rmsnormBwd : NS.L4.norm.layernormBwd;

    const L4 = NS.L4, L8 = NS.L8;
    const matmul = L8.matmul.matmul, matmulAtB = L8.matmul.matmulAtB, matmulTA = L8.matmul.matmulTA;
    const addInto = L8.reduce.addInto, rowSumInto = L8.reduce.rowSumInto;
    const ropeApply = L4.positional.ropeApply, qkNormBwd = L4.norm.qkNormBwd;
    const attnBwd = L4.attention.attnBwd;
    const geluBwd = L4.activation.geluBwd, swigluBwd = L4.activation.swigluBwd;
    const moeBackward = L4.moe.moeBackward;

    // 权重共享输出头反向：logits = Hf @ Wte^T
    //   dWte = dlogits^T @ Hf ；dHf = dlogits @ Wte
    // Gemma 2 logit soft-cap 反向：lg = cap·tanh(x/cap) ⇒ dlg/dx = sech²(x/cap) = 1 − (lg/cap)²
    // （lg 为缓存中的 capped logits，lg/cap 即 tanh(x/cap)，不可再套一层 tanh）
    if (model.logitCap > 0) {
      const cap = model.logitCap, lg = cache.logits;
      for (let i = 0; i < dlogits.length; i++) {
        const u = lg[i] / cap;
        dlogits[i] *= 1 - u * u;
      }
    }
    addInto(G.tokEmb, matmulTA(dlogits, cache.Hf, T, V, D));
    const dHf = matmul(dlogits, P.tokEmb.d, T, V, D);
    // 最终归一化反向（dY 形状 T×D）
    const lnfb = NB(dHf, cache.lnf.xhat, cache.lnf.rstd, P['lnf.g'].d, T, D);
    addInto(G['lnf.g'], lnfb.dg);
    if (G['lnf.b']) addInto(G['lnf.b'], lnfb.db);
    let dX = lnfb.dX;

    for (let l = L - 1; l >= 0; l--) {
      const k = `l${l}.`;
      const c = cache.layers[l];
      // xOut = xMid + ff => dff = dX, dxMid = dX
      const dff = dX;
      let dM;
      if (model.useMoE) {
        dM = moeBackward(model, P, G, c.moe, dff, T, D);
      } else {
        const FH = model.fh, W1 = model.useSwi ? 2 * FH : FH;
        addInto(G[k + 'w2'], matmulTA(c.h1, dff, T, FH, D));
        rowSumInto(G[k + 'b2'], dff, T, D);
        const dh1 = matmulAtB(dff, P[k + 'w2'].d, T, D, FH);
        const dpre1 = model.useSwi ? swigluBwd(dh1, c.pre1, T, FH) : geluBwd(dh1, c.pre1);
        addInto(G[k + 'w1'], matmulTA(c.M, dpre1, T, D, W1));
        rowSumInto(G[k + 'b1'], dpre1, T, W1);
        dM = matmulAtB(dpre1, P[k + 'w1'].d, T, W1, D);
      }
      const ln2b = NB(dM, c.ln2.xhat, c.ln2.rstd, P[k + 'ln2.g'].d, T, D);
      addInto(G[k + 'ln2.g'], ln2b.dg);
      if (G[k + 'ln2.b']) addInto(G[k + 'ln2.b'], ln2b.db);
      const dxMid = new Float64Array(T * D);
      for (let i = 0; i < T * D; i++) dxMid[i] = dX[i] + ln2b.dX[i];

      // xMid = x0 + attn => dattn = dxMid, dx0 = dxMid
      const dattn = dxMid;
      addInto(G[k + 'wproj'], matmulTA(c.ctx, dattn, T, D, D));
      rowSumInto(G[k + 'bproj'], dattn, T, D);
      const dctx = matmulAtB(dattn, P[k + 'wproj'].d, T, D, D);

      // 注意力反向（dP/dV/dS/dQ/dK 累加 -> dqkv），下沉到 L4.attention.attnBwd
      const qkvDim = model.qkvDim, KV = model.KV, g = model.g;
      const scale = 1 / Math.sqrt(dh);
      const attnScale = model.attentionScaling; // 与 forward 同因子（线性推送）
      const dqkv = attnBwd(c, dctx, T, D, H, dh, qkvDim, KV, g, scale, attnScale);

      // RoPE 反向：旋转是正交变换，其雅可比的转置 = 反向旋转（频率缩放与正向一致）
      if (model.useRope) ropeApply(dqkv, T, D, H, dh, 0, -1, KV, qkvDim, model.ropeFreqScale);
      // QK-norm 反向（前向顺序：norm → RoPE，反向顺序：RoPE⁻¹ → norm⁻¹）
      if (model.qkNorm) {
        qkNormBwd(dqkv, c.qk.qx, c.qk.qr, T, H, dh, qkvDim, 0, P[k + 'qg'].d, G[k + 'qg']);
        qkNormBwd(dqkv, c.qk.kx, c.qk.kr, T, KV, dh, qkvDim, D, P[k + 'kg'].d, G[k + 'kg']);
      }
      addInto(G[k + 'wqkv'], matmulTA(c.A, dqkv, T, D, qkvDim));
      rowSumInto(G[k + 'bqkv'], dqkv, T, qkvDim);
      const dA = matmulAtB(dqkv, P[k + 'wqkv'].d, T, qkvDim, D);
      const ln1b = NB(dA, c.ln1.xhat, c.ln1.rstd, P[k + 'ln1.g'].d, T, D);
      addInto(G[k + 'ln1.g'], ln1b.dg);
      if (G[k + 'ln1.b']) addInto(G[k + 'ln1.b'], ln1b.db);
      const dx0 = new Float64Array(T * D);
      for (let i = 0; i < T * D; i++) dx0[i] = dxMid[i] + ln1b.dX[i];
      dX = dx0;
    }

    // 嵌入层反向（token 嵌入散列累加 + 位置嵌入）
    for (let i = 0; i < T; i++) {
      const t = cache.toks[i], o = i * D;
      for (let j = 0; j < D; j++) {
        G.tokEmb[t * D + j] += dX[o + j];
        if (G.posEmb) G.posEmb[o + j] += dX[o + j];
      }
    }
  }

  NS.L3.blocks.forward = forward;
  NS.L3.blocks.backward = backward;
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

/* L4 神经算子 · 归一化族
 * 所属层：L4（神经算子）｜ 允许依赖：L8（数值内核）、L9（数值基础）
 * 主要 @ref：Ba et al. 2016 (LayerNorm)；Zhang & Sennrich 2019 (RMSNorm)；
 *            Gemma 2/3 / Chameleon / ViT-22B (QK-Norm)
 * 迁移方式：从 legacy flagship-engine.v11.js 第 145–295 行【原样搬】。
 *   本文件内的四个函数均为自包含（不跨层调用），逐位等价有 R9 把关。
 *   前向分发仍由 L3 blocks.js 以 `useRms ? rmsnormFwd : layernormFwd` 选择。
 */
;(function (NS) {
  NS.L4 = NS.L4 || {};
  NS.L4.norm = NS.L4.norm || {};

  /* ---- LayerNorm（Ba et al., 2016） ---- */
  function layernormFwd(X, g, b, r, c) {
    const Y = new Float64Array(r * c);
    const xhat = new Float64Array(r * c);
    const rstd = new Float64Array(r);
    for (let i = 0; i < r; i++) {
      const o = i * c;
      let mu = 0;
      for (let j = 0; j < c; j++) mu += X[o + j];
      mu /= c;
      let v = 0;
      for (let j = 0; j < c; j++) { const d = X[o + j] - mu; v += d * d; }
      v /= c;
      const rs = 1 / Math.sqrt(v + 1e-5);
      rstd[i] = rs;
      for (let j = 0; j < c; j++) {
        const xh = (X[o + j] - mu) * rs;
        xhat[o + j] = xh;
        Y[o + j] = g[j] * xh + b[j];
      }
    }
    return { Y, xhat, rstd };
  }

  function layernormBwd(dY, xhat, rstd, g, r, c) {
    const dX = new Float64Array(r * c);
    const dg = new Float64Array(c), db = new Float64Array(c);
    for (let i = 0; i < r; i++) {
      const o = i * c;
      let s1 = 0, s2 = 0;
      for (let j = 0; j < c; j++) {
        const dxh = dY[o + j] * g[j];
        dg[j] += dY[o + j] * xhat[o + j];
        db[j] += dY[o + j];
        s1 += dxh;
        s2 += dxh * xhat[o + j];
      }
      const rs = rstd[i];
      for (let j = 0; j < c; j++) {
        dX[o + j] = (rs / c) * (c * (dY[o + j] * g[j]) - s1 - xhat[o + j] * s2);
      }
    }
    return { dX, dg, db };
  }

  /* ---- RMSNorm（Zhang & Sennrich 2019；LLaMA/Mistral 标准） ----
   * y = g ⊙ (x / RMS(x))，RMS(x) = sqrt(mean(x²) + eps)。
   * 与 LayerNorm 的差别：不去均值、无偏置 —— 更省算力，且实测在大模型上等价或更优。
   * 反向用同一 xhat 约定（增益前归一化值），故与 layernormBwd 签名完全兼容。 */
  const RMS_EPS = 1e-5;
  // 签名与 layernormFwd 对齐（b 位置保留但忽略），这样前向分发可以无脑二选一
  function rmsnormFwd(X, g, b, r, c) {
    const Y = new Float64Array(r * c);
    const xhat = new Float64Array(r * c);
    const rstd = new Float64Array(r);
    for (let i = 0; i < r; i++) {
      const o = i * c;
      let ss = 0;
      for (let j = 0; j < c; j++) ss += X[o + j] * X[o + j];
      const rs = 1 / Math.sqrt(ss / c + RMS_EPS);
      rstd[i] = rs;
      for (let j = 0; j < c; j++) {
        const xh = X[o + j] * rs;
        xhat[o + j] = xh;
        Y[o + j] = g[j] * xh;
      }
    }
    return { Y, xhat, rstd };
  }
  function rmsnormBwd(dY, xhat, rstd, g, r, c) {
    const dX = new Float64Array(r * c);
    const dg = new Float64Array(c);
    for (let i = 0; i < r; i++) {
      const o = i * c;
      const rs = rstd[i];
      let s = 0; // Σ_j (dY_j · g_j · xhat_j)
      for (let j = 0; j < c; j++) s += dY[o + j] * g[j] * xhat[o + j];
      const inv = 1 / c;
      for (let j = 0; j < c; j++) {
        const dy = dY[o + j] * g[j];
        dg[j] += dY[o + j] * xhat[o + j];
        dX[o + j] = rs * (dy - xhat[o + j] * inv * s);
      }
    }
    return { dX, dg, db: null };
  }

  /* ---- QK-norm（Gemma 2/3 / Chameleon / ViT-22B）----
   * 对每个注意力头的 Q/K 向量做 RMSNorm（增益按 head_dim 维、跨头共享），
   * 在 RoPE 旋转之前应用：稳定注意力 logits 尺度，抑制训练发散。
   * 行布局：每个头向量长 dh，起始偏移 off，行间跨度 stride（qkvDim）。 */
  function qkNormFwd(qkv, rows, nHeads, dh, stride, off, g) {
    const xhat = new Float64Array(rows * nHeads * dh);
    const rstd = new Float64Array(rows * nHeads);
    for (let i = 0; i < rows; i++) {
      for (let h = 0; h < nHeads; h++) {
        const o = i * stride + off + h * dh;
        const hi = (i * nHeads + h) * dh;
        let ss = 0;
        for (let p = 0; p < dh; p++) ss += qkv[o + p] * qkv[o + p];
        const rs = 1 / Math.sqrt(ss / dh + RMS_EPS);
        rstd[i * nHeads + h] = rs;
        for (let p = 0; p < dh; p++) {
          const xh = qkv[o + p] * rs;
          xhat[hi + p] = xh;
          qkv[o + p] = g[p] * xh; // 就地归一（RoPE 前的 Q/K 段）
        }
      }
    }
    return { xhat, rstd };
  }
  // 反向：dX = rs·(dY·g − xhat·mean(dY·g·xhat))；dg 跨头跨行累加（增益共享）
  function qkNormBwd(dqkv, xhat, rstd, rows, nHeads, dh, stride, off, g, dg) {
    const inv = 1 / dh;
    for (let i = 0; i < rows; i++) {
      for (let h = 0; h < nHeads; h++) {
        const o = i * stride + off + h * dh;
        const hi = (i * nHeads + h) * dh;
        const rs = rstd[i * nHeads + h];
        let s = 0; // Σ_p (dY_p · g_p · xhat_p)
        for (let p = 0; p < dh; p++) s += dqkv[o + p] * g[p] * xhat[hi + p];
        for (let p = 0; p < dh; p++) {
          const dy = dqkv[o + p] * g[p];
          dg[p] += dqkv[o + p] * xhat[hi + p];
          dqkv[o + p] = rs * (dy - xhat[hi + p] * inv * s);
        }
      }
    }
  }

  NS.L4.norm.RMS_EPS = RMS_EPS;
  NS.L4.norm.layernormFwd = layernormFwd;
  NS.L4.norm.layernormBwd = layernormBwd;
  NS.L4.norm.rmsnormFwd = rmsnormFwd;
  NS.L4.norm.rmsnormBwd = rmsnormBwd;
  NS.L4.norm.qkNormFwd = qkNormFwd;
  NS.L4.norm.qkNormBwd = qkNormBwd;
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

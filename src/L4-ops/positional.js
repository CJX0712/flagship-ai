/* L4 神经算子 · 位置编码（RoPE）
 * 所属层：L4（神经算子）｜ 允许依赖：L8、L9
 * 主要 @ref：Su et al. 2021 (RoFormer RoPE)；(P1) NTK-by-parts / YaRN 外推
 * 迁移方式：从 legacy flagship-engine.v11.js 第 324–366 行【原样搬】。
 *   sign=+1 前向（逆时针）；sign=-1 反向（正交矩阵的逆 = 其转置）。
 *   freqScale[p] 为 YaRN per-(dim-pair) 频率缩放，恒等时缺省。
 */
;(function (NS) {
  NS.L4 = NS.L4 || {};
  NS.L4.positional = NS.L4.positional || {};

  /* ---- RoPE 旋转位置编码（Su et al. 2021；LLaMA/Mistral 标准） ----
   * 对 Q/K 的每个相邻维度对施加角度 pos·θ_p 的旋转，θ_p = base^(-2p/dh)。
   * 关键性质：旋转后 q_i·k_j 只依赖相对距离 (i-j) —— 故天然支持长度外推，
   * 且使滑动窗口 KV cache 在任意绝对位置下都与全序列重算逐位一致。
   * sign=+1 前向（逆时针）；sign=-1 反向（正交矩阵的逆 = 其转置）。 */
  const ROPE_BASE = 10000;
  function ropeApply(X, T, D, H, dh, posOff, sign, Hk, stride, freqScale) {
    Hk = Hk || H;                 // K 段头数（GQA 下 < H）；默认 = H（MHA）
    stride = stride || 3 * D;     // qkv 行宽（GQA 下 = D + 2·KV·dh）
    const half = dh >> 1;
    if (half <= 0) return;
    // YaRN/NTK：per-(dim-pair) 频率缩放（freqScale[p] = 1/scale_i，恒等时缺省）。
    // 位置无关，故缓存的已旋转 K 与任意 Q 的相对角度一致。
    const inv = new Float64Array(half);
    for (let p = 0; p < half; p++) {
      let f = Math.pow(ROPE_BASE, -2 * p / dh);
      if (freqScale) f *= freqScale[p];
      inv[p] = f;
    }
    for (let i = 0; i < T; i++) {
      const pos = i + posOff;
      const row = i * stride;
      for (let p = 0; p < half; p++) {
        const th = pos * inv[p];
        const cs = Math.cos(th), sn = Math.sin(th) * sign;
        // Q 段：base 0，H 个头
        for (let h = 0; h < H; h++) {
          const q = row + h * dh + (p << 1);
          const x0 = X[q], x1 = X[q + 1];
          X[q] = x0 * cs - x1 * sn;
          X[q + 1] = x0 * sn + x1 * cs;
        }
        // K 段：base D，Hk 个头
        const base = row + D;
        for (let h = 0; h < Hk; h++) {
          const q = base + h * dh + (p << 1);
          const x0 = X[q], x1 = X[q + 1];
          X[q] = x0 * cs - x1 * sn;
          X[q + 1] = x0 * sn + x1 * cs;
        }
      }
    }
  }

  NS.L4.positional.ROPE_BASE = ROPE_BASE;
  NS.L4.positional.ropeApply = ropeApply;
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

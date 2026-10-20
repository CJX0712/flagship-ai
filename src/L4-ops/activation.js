/* L4 神经算子 · 激活族
 * 所属层：L4（神经算子）｜ 允许依赖：L8、L9
 * 主要 @ref：Hendrycks & Gimpel 2016 (GELU)；Shazeer 2020 (SwiGLU/SiLU)
 * 迁移方式：从 legacy flagship-engine.v11.js 第 190–322 行【原样搬】。
 *   本文件内全部函数自包含（swigluFwd 仅调用同文件 siluF），逐位等价有 R9 把关。
 */
;(function (NS) {
  NS.L4 = NS.L4 || {};
  NS.L4.activation = NS.L4.activation || {};

  /* ---- GELU（tanh 近似，GPT-2 标准） ---- */
  const GELU_C = Math.sqrt(2 / Math.PI);
  function geluFwd(X) {
    const Y = new Float64Array(X.length);
    for (let i = 0; i < X.length; i++) {
      const x = X[i];
      Y[i] = 0.5 * x * (1 + Math.tanh(GELU_C * (x + 0.044715 * x * x * x)));
    }
    return Y;
  }
  function geluBwd(dY, X) {
    const dX = new Float64Array(X.length);
    for (let i = 0; i < X.length; i++) {
      const x = X[i];
      const inner = GELU_C * (x + 0.044715 * x * x * x);
      const t = Math.tanh(inner);
      const dt = (1 - t * t) * GELU_C * (1 + 3 * 0.044715 * x * x);
      dX[i] = dY[i] * (0.5 * (1 + t) + 0.5 * x * dt);
    }
    return dX;
  }

  /* ---- SiLU / SwiGLU（Shazeer 2020；LLaMA/Mistral/PaLM 标准 FFN） ----
   * SwiGLU(x) = SiLU(xW_gate) ⊙ (xW_up)，SiLU(z) = z·σ(z)。
   * 本实现把 gate 与 up 拼成一次矩阵乘（w1: D → 2·FH），少一次 matmul。
   * pre 布局：每行 [gate(FH) | up(FH)]。 */
  function siluF(x) { return x / (1 + Math.exp(-x)); }
  function swigluFwd(pre, r, h) {
    const out = new Float64Array(r * h);
    for (let i = 0; i < r; i++) {
      const po = i * 2 * h, oo = i * h;
      for (let j = 0; j < h; j++) out[oo + j] = siluF(pre[po + j]) * pre[po + h + j];
    }
    return out;
  }
  function swigluBwd(dout, pre, r, h) {
    const dpre = new Float64Array(r * 2 * h);
    for (let i = 0; i < r; i++) {
      const po = i * 2 * h, oo = i * h;
      for (let j = 0; j < h; j++) {
        const g = pre[po + j], u = pre[po + h + j];
        const sg = 1 / (1 + Math.exp(-g));
        dpre[po + j] = dout[oo + j] * u * (sg + g * sg * (1 - sg)); // ∂/∂gate
        dpre[po + h + j] = dout[oo + j] * g * sg;                    // ∂/∂up
      }
    }
    return dpre;
  }

  NS.L4.activation.GELU_C = GELU_C;
  NS.L4.activation.geluFwd = geluFwd;
  NS.L4.activation.geluBwd = geluBwd;
  NS.L4.activation.siluF = siluF;
  NS.L4.activation.swigluFwd = swigluFwd;
  NS.L4.activation.swigluBwd = swigluBwd;
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

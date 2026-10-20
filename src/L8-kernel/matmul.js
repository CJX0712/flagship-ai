/* ============================================================================
 * L8 数值内核 · 矩阵乘法内核（matmul）
 * ----------------------------------------------------------------------------
 * 所属层：L8-kernel
 * 允许依赖的层：无（纯数值；如需 RNG 仅用于构造演示，不在热路径）
 * 主要 @ref：Goto & van de Geijn (2008) "Anatomy of High-Performance Matrix
 *           Multiplication"（朴素实现的标量极限分析；V8 已把朴素循环优化到极限）
 *
 * 责任：提供零依赖 GEMM 家族。函数体与 v11 逐字一致（含 matmul 的
 *       `if (a===0) continue` 跳过），以保证 R9 逐算子 Δ=0。这是 L4 线性
 *       算子与 L2 训练/推理热路径的底层内核。
 * 存储序（K1）：Float64Array 行优先；M(r×c) 元素 (i,j) 位于 i*c+j。
 * ==========================================================================*/
;(function (NS) {
  'use strict';
  NS.L8 = NS.L8 || {};
  NS.L8.matmul = NS.L8.matmul || {};

  // A(r×k) @ B(k×c) -> C(r×c)
  // 关键：保留 `if (a === 0) continue` —— 这是 R9 逐位等价的契约前提。
  function matmul(A, B, r, k, c) {
    const C = new Float64Array(r * c);
    for (let i = 0; i < r; i++) {
      const ci = i * c;
      for (let p = 0; p < k; p++) {
        const a = A[i * k + p];
        if (a === 0) continue;
        const bp = p * c;
        for (let j = 0; j < c; j++) C[ci + j] += a * B[bp + j];
      }
    }
    return C;
  }

  // A(r×k) @ B(m×k)^T -> C(r×m)（用于 tied embedding 输出头）
  function matmulAtB(A, B, r, k, m) {
    const C = new Float64Array(r * m);
    for (let i = 0; i < r; i++) {
      const ai = i * k, ci = i * m;
      for (let j = 0; j < m; j++) {
        const bj = j * k;
        let s = 0;
        for (let p = 0; p < k; p++) s += A[ai + p] * B[bj + p];
        C[ci + j] = s;
      }
    }
    return C;
  }

  // A(k×r)^T @ B(r×c) -> C(k×c)（用于计算权重梯度 dW = X^T @ dY）
  function matmulTA(A, B, r, k, c) {
    const C = new Float64Array(k * c);
    for (let i = 0; i < r; i++) {
      const ai = i * k, bi = i * c;
      for (let l = 0; l < k; l++) {
        const a = A[ai + l];
        if (a === 0) continue;
        const cl = l * c;
        for (let j = 0; j < c; j++) C[cl + j] += a * B[bi + j];
      }
    }
    return C;
  }

  // ── INT8 权重量化（weight-only PTQ，LLM.int8 / AWQ 谱系）──────────────
  // 对称逐张量量化：q = round(w / scale), clamp[-127,127], scale = max|w|/127。
  // 推理时 w = q·scale 在矩阵乘内联解量化，权重常驻 Int8 → 内存/体积 ≈ 1/4。
  function matmulQ(A, q, scale, r, k, c) {
    const C = new Float64Array(r * c);
    for (let i = 0; i < r; i++) {
      const ai = i * k, ci = i * c;
      for (let j = 0; j < c; j++) {
        let s = 0;
        for (let p = 0; p < k; p++) s += A[ai + p] * q[p * c + j];
        C[ci + j] = s * scale;
      }
    }
    return C;
  }
  // A(r×k) @ (scale·q)(m×k)^T -> C(r×m)，q 按 [m][k] 行主序展开
  function matmulAtBQ(A, q, scale, r, k, m) {
    const C = new Float64Array(r * m);
    for (let i = 0; i < r; i++) {
      const ai = i * k, ci = i * m;
      for (let j = 0; j < m; j++) {
        let s = 0;
        const qj = j * k;
        for (let p = 0; p < k; p++) s += A[ai + p] * q[qj + p];
        C[ci + j] = s * scale;
      }
    }
    return C;
  }
  // 权重分发：param 含 .q 即走解量化路径，否则走原始 Float64 路径（零回归）
  function mm(A, W, r, k, c) {
    return W.q ? matmulQ(A, W.q, W.scale, r, k, c) : matmul(A, W.d, r, k, c);
  }
  function mmAtB(A, W, r, k, m) {
    return W.q ? matmulAtBQ(A, W.q, W.scale, r, k, m) : matmulAtB(A, W.d, r, k, m);
  }
  const QUANT_LINEAR_RE = /^(l\d+\.(wqkv|wproj|w1|w2|wg)|l\d+\.e\d+\.(w1|w2)|tokEmb|posEmb)$/;
  function isQuantLinear(key) { return QUANT_LINEAR_RE.test(key); }

  NS.L8.matmul.matmul = matmul;
  NS.L8.matmul.matmulAtB = matmulAtB;
  NS.L8.matmul.matmulTA = matmulTA;
  NS.L8.matmul.matmulQ = matmulQ;
  NS.L8.matmul.matmulAtBQ = matmulAtBQ;
  NS.L8.matmul.mm = mm;
  NS.L8.matmul.mmAtB = mmAtB;
  NS.L8.matmul.isQuantLinear = isQuantLinear;
  NS.L8.matmul.QUANT_LINEAR_RE = QUANT_LINEAR_RE;
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

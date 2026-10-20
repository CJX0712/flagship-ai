/* ============================================================================
 * 旗舰 AI 引擎（flagship-engine.js）
 * ----------------------------------------------------------------------------
 * 零依赖纯 JavaScript 手写 GPT 引擎：字符级分词 + Decoder-only Transformer
 * （因果多头自注意力 + 前馈网络 + pre-LayerNorm + 残差连接 + 权重共享输出头）
 * + 手写反向传播 + AdamW 优化器 + 温度/top-k 采样生成。
 *
 * 架构不是发明创造，而是站在世界顶级成果之上直接复用：
 *   - 模型结构：GPT-2 (Radford et al., 2019) / nanoGPT (Karpathy, 2022)
 *   - 极简参照：picoGPT (Larroy, 2023) —— 单文件讲清 Transformer 全部计算
 *   - 优化器：AdamW (Loshchilov & Hutter, 2019)
 *   - LayerNorm (Ba et al., 2016) / GELU (Hendrycks & Gimpel, 2016)
 *   - 初始化与残差缩放：GPT-2 论文标准做法
 *
 * 运行环境：Node >= 14 与浏览器（无任何外部依赖，无构建步骤）。
 * 作者：晨星  ·  License: MIT
 * ==========================================================================*/
(function (root, factory) {
  if (typeof module !== 'undefined' && module.exports) module.exports = factory();
  else root.FlagshipAI = factory();
})(typeof self !== 'undefined' ? self : globalThis, function () {
  'use strict';

  /* ==========================================================================
   * 一、基础数学：RNG / 矩阵算子 / 激活 / 归一化
   * 全部使用 Float64Array 行优先存储，M(r×c) 元素 (i,j) 位于 i*c+j。
   * ======================================================================== */

  // 确定性伪随机数发生器（mulberry32）：同种子 => 完全可复现
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // Box-Muller 标准正态采样
  function randn(rng) {
    let u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  /* 矩阵乘法内核。实测：V8 已把朴素循环优化到标量极限（~900M MAC/s），
   * 手写 4 路展开反而更慢（32.7µs vs 30.2µs @ 1×96×288），故保持朴素实现。 */
  // A(r×k) @ B(k×c) -> C(r×c)
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

  function addInto(dst, src) {
    for (let i = 0; i < dst.length; i++) dst[i] += src[i];
  }

  function rowSumInto(dst, Y, r, c) {
    for (let i = 0; i < r; i++) {
      const o = i * c;
      for (let j = 0; j < c; j++) dst[j] += Y[o + j];
    }
  }

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

  /* ---- RoPE 旋转位置编码（Su et al. 2021；LLaMA/Mistral 标准） ----
   * 对 Q/K 的每个相邻维度对施加角度 pos·θ_p 的旋转，θ_p = base^(-2p/dh)。
   * 关键性质：旋转后 q_i·k_j 只依赖相对距离 (i-j) —— 故天然支持长度外推，
   * 且使滑动窗口 KV cache 在任意绝对位置下都与全序列重算逐位一致。
   * sign=+1 前向（逆时针）；sign=-1 反向（正交矩阵的逆 = 其转置）。 */
  const ROPE_BASE = 10000;
  function ropeApply(X, T, D, H, dh, posOff, sign, Hk, stride) {
    Hk = Hk || H;                 // K 段头数（GQA 下 < H）；默认 = H（MHA）
    stride = stride || 3 * D;     // qkv 行宽（GQA 下 = D + 2·KV·dh）
    const half = dh >> 1;
    if (half <= 0) return;
    for (let i = 0; i < T; i++) {
      const pos = i + posOff;
      const row = i * stride;
      for (let p = 0; p < half; p++) {
        const th = pos * Math.pow(ROPE_BASE, -2 * p / dh);
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

  /* ---- Softmax（按行） ---- */
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

  /* ---- Top-k 行选择（MoE 路由用） ---- */
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
      const h1 = model.useSwi ? swigluFwd(pre, cnt, FH) : geluFwd(pre);
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
      const dpre = model.useSwi ? swigluBwd(dh1, ex.pre, cnt, FH) : geluBwd(dh1, ex.pre);
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

  /* ---- 交叉熵（logits 前 n 行对 targets；totalRows 补零至完整行数供反向使用） ---- */
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

  /* ==========================================================================
   * 二、字符级分词器
   * ======================================================================== */
  class Tokenizer {
    constructor(text) {
      const chars = [...new Set([...String(text)])].sort();
      this.kind = 'char';
      this.chars = chars;
      this.stoi = new Map(chars.map((ch, i) => [ch, i]));
      this.itos = chars;
    }
    get vocabSize() { return this.chars.length; }
    encode(text) {
      const out = [];
      for (const ch of String(text)) {
        const id = this.stoi.get(ch);
        if (id !== undefined) out.push(id); // 未知字符静默跳过（字符级词表的既定行为）
      }
      return out;
    }
    decode(ids) {
      let s = '';
      for (const id of ids) s += this.itos[id] ?? '';
      return s;
    }
  }

  /* ==========================================================================
   * 二之二、BPE 分词器（GPT-2 字节级标准实现）
   *
   * 三件套，与 GPT-2 / tiktoken 同源：
   *   1. 字节级映射：把 256 个字节映射到 256 个可打印 Unicode 字符，
   *      使任意 UTF-8 文本（含中文、emoji）都能无损表示为字符序列。
   *   2. 预分词 regex（GPT-2 原文）：先把文本切成"词块"，BPE 只在块内合并，
   *      避免跨词污染（这是 BPE 质量的关键，也是朴素实现的常见缺陷）。
   *   3. 频率合并训练：反复统计相邻符号对频次、合并最高频者，直至词表达标。
   * 编码用 merge-rank 贪心（等价 GPT-2 实现），解码一次性拼字节再 UTF-8 解码，
   * 因此跨 token 的多字节汉字也能正确还原。
   * ======================================================================== */
  const BYTE_ENC = (() => {
    const bs = [];
    for (let b = 0x21; b <= 0x7e; b++) bs.push(b);   // '!' ~ '~'
    for (let b = 0xa1; b <= 0xac; b++) bs.push(b);   // '¡' ~ '¬'
    for (let b = 0xae; b <= 0xff; b++) bs.push(b);   // '®' ~ 'ÿ'
    const map = new Map();
    let n = 0;
    for (const b of bs) map.set(b, String.fromCharCode(b));
    for (let b = 0; b < 256; b++) {
      if (!map.has(b)) map.set(b, String.fromCharCode(256 + n++));
    }
    return map;
  })();
  const BYTE_DEC = new Map([...BYTE_ENC].map(([b, c]) => [c, b]));

  // GPT-2 预分词正则（原文照搬）：缩写 | 可选空格+字母 | 可选空格+数字 | 可选空格+其它非空白 | 尾随空白
  const PRE_RE = new RegExp(
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+", 'gu');

  /** UTF-8 文本 -> 字节级字符序列（可逆） */
  function textToByteChars(s) {
    const bytes = new TextEncoder().encode(s);
    let out = '';
    for (let i = 0; i < bytes.length; i++) out += BYTE_ENC.get(bytes[i]);
    return out;
  }
  /** 字节级字符序列 -> UTF-8 文本（可逆） */
  function byteCharsToText(s) {
    const bytes = new Uint8Array(s.length);
    for (let i = 0; i < s.length; i++) bytes[i] = BYTE_DEC.get(s[i]) ?? 63;
    return new TextDecoder().decode(bytes);
  }

  class BPETokenizer {
    /**
     * @param {string} text 训练语料
     * @param {number} vocabSize 目标词表大小（字节级下 ≥256）
     * @param {Object} opts
     *   minPairFreq: 最低合并频次（低于则提前停止，默认 2）
     *   byteLevel:   基元选择 —— true = 256 字节（GPT-2 标准，覆盖任意 UTF-8）；
     *                 false = 语料字符（中文语料下显著更省 token，且不会生成半截 UTF-8）
     */
    constructor(text, vocabSize = 512, opts) {
      const o = Object.assign({ minPairFreq: 2, byteLevel: true }, opts || {});
      this.kind = 'bpe';
      this.byteLevel = !!o.byteLevel;
      const target = Math.max(this.byteLevel ? 256 : 1, vocabSize | 0);
      // 词块 <-> 符号序列（字节级要多一层 UTF-8 编解码）
      const toSym = (w) => (this.byteLevel ? [...textToByteChars(w)] : [...w]);

      // 1) 预分词并统计词频
      const freq = new Map();
      for (const m of String(text).matchAll(PRE_RE)) {
        const w = m[0];
        freq.set(w, (freq.get(w) || 0) + 1);
      }
      // 2) 每个词块转成符号数组（BPE 只在块内合并）
      const seqs = [];
      for (const [w, n] of freq) seqs.push({ s: toSym(w), n });

      // 3) 初始词表：字节级 = 256 个字节基元；字符级 = 语料中的全部字符
      const vocab = [];
      if (this.byteLevel) { for (let b = 0; b < 256; b++) vocab.push(BYTE_ENC.get(b)); }
      else { vocab.push(...[...new Set([...String(text)])].sort()); }
      if (vocab.length > target) vocab.length = target;
      const merges = []; // [[a, b], ...]，下标即 merge rank

      // 4) 频率合并：每轮统计相邻对频次，合并最高频者
      while (vocab.length < target) {
        const cnt = new Map();
        for (const { s, n } of seqs) {
          for (let i = 0; i + 1 < s.length; i++) {
            const key = s[i] + '\u0000' + s[i + 1];
            cnt.set(key, (cnt.get(key) || 0) + n);
          }
        }
        let best = null, bestN = o.minPairFreq - 1;
        for (const [k, n] of cnt) if (n > bestN) { bestN = n; best = k; }
        if (!best) break; // 没有再值得合并的对
        const parts = best.split('\u0000');
        const a = parts[0], b = parts[1], tok = a + b;
        vocab.push(tok);
        merges.push([a, b]);
        for (const seq of seqs) {
          const out = [];
          let i = 0;
          while (i < seq.s.length) {
            if (i + 1 < seq.s.length && seq.s[i] === a && seq.s[i + 1] === b) { out.push(tok); i += 2; }
            else { out.push(seq.s[i]); i++; }
          }
          seq.s = out;
        }
      }

      this.chars = vocab;
      this.merges = merges;
      this.itos = vocab;
      this.stoi = new Map(vocab.map((t, i) => [t, i]));
      this.ranks = new Map(merges.map((m, i) => [m[0] + m[1], i]));
    }

    get vocabSize() { return this.chars.length; }

    /** 文本 -> id 序列。逐词块用 merge-rank 贪心合并（等价 GPT-2/tiktoken） */
    encode(text) {
      const out = [];
      for (const m of String(text).matchAll(PRE_RE)) {
        let s = this.byteLevel ? [...textToByteChars(m[0])] : [...m[0]];
        while (s.length > 1) {
          let bestRank = Infinity, bestI = -1;
          for (let i = 0; i + 1 < s.length; i++) {
            const r = this.ranks.get(s[i] + s[i + 1]);
            if (r !== undefined && r < bestRank) { bestRank = r; bestI = i; }
          }
          if (bestI < 0) break;
          s.splice(bestI, 2, s[bestI] + s[bestI + 1]);
        }
        for (const t of s) {
          const id = this.stoi.get(t);
          if (id !== undefined) out.push(id);
          else for (const ch of t) { // 兜底：拆到字节基元（理论上不会触发）
            const cid = this.stoi.get(ch);
            if (cid !== undefined) out.push(cid);
          }
        }
      }
      return out;
    }

    /** id 序列 -> 文本。字节级先拼全部字节再一次性 UTF-8 解码，跨 token 汉字无损 */
    decode(ids) {
      let s = '';
      for (const id of ids) s += this.itos[id] ?? '';
      return this.byteLevel ? byteCharsToText(s) : s;
    }
  }

  /* ==========================================================================
   * 三、GPT 模型（前向 + 手写反向传播）
   * ======================================================================== */
  class GPTModel {
    /**
     * @param {Object} cfg { vocabSize, blockSize, nLayer, nHead, dModel, seed }
     *   架构开关（默认全部为 GPT-2 原设计，保持向后兼容；'llama' 预设见 ARCH_PRESETS）：
     *   norm: 'ln'（LayerNorm，GPT-2）| 'rmsnorm'（RMSNorm，LLaMA/Mistral）
     *   ffn:  'gelu'（GPT-2）        | 'swiglu'（SiLU 门控，LLaMA/Mistral）
     *   pos:  'learned'（可学习位置嵌入，GPT-2）| 'rope'（旋转位置编码，LLaMA/Mistral）
     *   ffnHidden: FFN 隐藏层宽度，默认 4·dModel（LLaMA 惯例为 ⌊2/3·4D⌋，可显式传入）
     *   MoE 开关（默认稠密，nExp=1；nExp>1 即混合专家，现代顶级 LLM 核心稀疏架构）：
     *   nExp:    专家数量（每层 N 个独立 FFN + 1 个路由网络）
     *   topK:    top-k 门控（每个 token 仅激活 topK 个专家，默认 1）
     *   auxWeight: 负载均衡辅助损失权重（Switch/GDP-MoE，默认 0.01；非 MoE 为 0）
     *   稀疏注意力开关（Mistral/Gemma 谱系）：
     *   window: 滑动窗口。标量=所有层同窗口；数组=逐层窗口（长度须=nLayer，Gemma 2/3 局部/全局交错）；
     *           哨兵 {alt:N}=偶数层全因果(0)、奇数层窗口 N 的 1:1 交错。0 = 全因果，默认
     *   稳定化开关（Gemma 2 配方，默认全关 = 逐位兼容旧模型）：
     *   qkNorm:  true 时对每个头的 Q/K 向量做 RMSNorm（RoPE 前，增益 head_dim 维跨头共享）
     *   logitCap: 最终 logits 软帽 logits = cap·tanh(logits/cap)（Gemma 2 用 30；0 = 不启用）
     *   kvQuant: KV 缓存量化（8 = INT8，per-(token,KV头) 对称 absmax；0 = Float64，默认）
     */
    constructor(cfg) {
      if (cfg.dModel % cfg.nHead !== 0) {
        throw new Error(`dModel(${cfg.dModel}) 必须能被 nHead(${cfg.nHead}) 整除`);
      }
      this.cfg = Object.assign({ seed: 42, norm: 'ln', ffn: 'gelu', pos: 'learned' }, cfg);
      const { vocabSize: V, blockSize: T, nLayer: L, nHead: H, dModel: D } = this.cfg;
      if (this.cfg.pos === 'rope' && (D / H) % 2 !== 0) {
        throw new Error(`RoPE 要求每个 head 维度 dh=${D / H} 为偶数`);
      }
      // GQA（分组查询注意力）：KV 头数可小于 Q 头数（LLaMA-2/3 标准）。
      // 默认 nKvHead = nHead 即 MHA，完全向后兼容。
      const KV = cfg.nKvHead != null ? cfg.nKvHead : H;
      if (KV < 1 || KV > H || H % KV !== 0) {
        throw new Error(`nKvHead(${KV}) 需满足 1 <= nKvHead <= nHead(${H}) 且 nHead 可被整除`);
      }
      const dh = D / H;
      this.KV = KV; this.dh = dh; this.g = H / KV; this.qkvDim = D + 2 * KV * dh;
      this.useRope = this.cfg.pos === 'rope';
      this.useRms = this.cfg.norm === 'rmsnorm';
      this.useSwi = this.cfg.ffn === 'swiglu';
      this.fh = this.cfg.ffnHidden || 4 * D;
      // MoE（混合专家）：nExp>1 时每层 FFN 替换为 N 个专家 + top-k 路由网络；
      // 现代顶级 LLM（Mixtral/GPT-4/DeepSeek）的核心稀疏架构。nExp=1 即稠密（零回归）。
      this.nExp = cfg.nExp || 1;
      this.topK = Math.min(cfg.topK || 1, this.nExp);
      this.useMoE = this.nExp > 1;
      this.auxWeight = cfg.auxWeight != null ? cfg.auxWeight : (this.useMoE ? 0.01 : 0);
      // 滑动窗口注意力（SWA，Mistral/Gemma）：window>0 时查询 i 只 attend 键 [i-window+1, i]；
      // window=0（默认）即标准全因果，逐位兼容旧模型。三种写法归一化为逐层数组 this.windows：
      //   标量 s      -> 所有层同窗口 [s,s,...]
      //   数组 [a,b]  -> 逐层窗口（长度须 = nLayer，Gemma 2/3 局部/全局层交错）
      //   哨兵 {alt:N} -> 偶数层全因果(0)、奇数层窗口 N（Gemma 2 的 1:1 交错模式）
      const wRaw = cfg.window;
      let wArr;
      if (Array.isArray(wRaw)) {
        if (wRaw.length !== L) {
          throw new Error(`window 数组长度(${wRaw.length}) 需等于 nLayer(${L})`);
        }
        wArr = wRaw.map((v) => (v > 0 ? Math.floor(v) : 0));
      } else if (wRaw && typeof wRaw === 'object' && wRaw.alt > 0) {
        const w = Math.floor(wRaw.alt);
        wArr = [];
        for (let l = 0; l < L; l++) wArr.push(l % 2 === 1 ? w : 0); // 偶数层全局，奇数层局部
      } else {
        const w = wRaw > 0 ? Math.floor(wRaw) : 0;
        wArr = new Array(L).fill(w);
      }
      this.windows = wArr;   // 逐层窗口（权威字段，0 = 该层全因果/全局）
      this.window = Array.isArray(wRaw) ? wArr.slice()
        : (wRaw && typeof wRaw === 'object' ? { alt: Math.floor(wRaw.alt) } : wArr[0]); // 兼容读取：原样反映配置
      // Gemma 2 稳定化组件：QK-norm（RoPE 前逐头 RMSNorm）与 logit soft-cap。默认全关。
      this.qkNorm = !!cfg.qkNorm;
      this.logitCap = cfg.logitCap > 0 ? +cfg.logitCap : 0;
      // KV 缓存 INT8 量化：仅影响 forwardStep 增量路径（forward 全序列不走缓存）
      this.kvQuant = cfg.kvQuant === 8 ? 8 : 0;
      // Attention Sink（StreamingLLM）：恒驻缓存首 K 个锚点 token，支持无限上下文流式推理；0 = 关闭（逐位兼容）
      this.attentionSink = cfg.attentionSink > 0 ? Math.floor(cfg.attentionSink) : 0;
      this.quantized = false;
      const rng = mulberry32(this.cfg.seed);
      const P = (this.params = {});
      const add = (name, r, c, scale) => {
        const p = { d: new Float64Array(r * c), r, c };
        const s = 0.02 * (scale || 1);
        for (let i = 0; i < p.d.length; i++) p.d[i] = randn(rng) * s;
        P[name] = p;
      };
      const zeros = (name, n) => { P[name] = { d: new Float64Array(n), r: 1, c: n }; };
      const ones = (name, n) => {
        const p = { d: new Float64Array(n), r: 1, c: n };
        p.d.fill(1);
        P[name] = p;
      };
      add('tokEmb', V, D);
      if (!this.useRope) add('posEmb', T, D); // RoPE 不需要位置嵌入
      const rscale = 1 / Math.sqrt(2 * L); // GPT-2：按残差深度缩放输出投影初始化
      const FH = this.fh;
      for (let l = 0; l < L; l++) {
        const k = `l${l}.`;
        add(k + 'wqkv', D, this.qkvDim); zeros(k + 'bqkv', this.qkvDim);
        add(k + 'wproj', D, D, rscale); zeros(k + 'bproj', D);
        if (this.useMoE) {
          // 路由网络：x → 各专家门控 logits（softmax over experts）
          add(k + 'wg', D, this.nExp); zeros(k + 'bg', this.nExp);
          // N 个专家 FFN（各自独立的 SwiGLU/GELU 两层），复用稠密 FFN 的同一宽度约定
          for (let e = 0; e < this.nExp; e++) {
            const ek = `${k}e${e}.`;
            add(ek + 'w1', D, this.useSwi ? 2 * FH : FH);
            zeros(ek + 'b1', this.useSwi ? 2 * FH : FH);
            add(ek + 'w2', FH, D, rscale); zeros(ek + 'b2', D);
          }
        } else {
          // SwiGLU 的 w1 一次产出 gate 与 up 两半（D → 2·FH），少一次矩阵乘
          add(k + 'w1', D, this.useSwi ? 2 * FH : FH);
          zeros(k + 'b1', this.useSwi ? 2 * FH : FH);
          add(k + 'w2', FH, D, rscale); zeros(k + 'b2', D);
        }
        ones(k + 'ln1.g', D); if (!this.useRms) zeros(k + 'ln1.b', D);
        ones(k + 'ln2.g', D); if (!this.useRms) zeros(k + 'ln2.b', D);
        if (this.qkNorm) { ones(k + 'qg', dh); ones(k + 'kg', dh); } // QK-norm 增益（head_dim 维跨头共享）
      }
      ones('lnf.g', D); if (!this.useRms) zeros('lnf.b', D);
      // 输出头与 tokEmb 权重共享（weight tying，GPT-1/2 标准做法）
      this.grads = {};
      for (const key in P) this.grads[key] = new Float64Array(P[key].d.length);
      this.zeroGrads();
    }

    zeroGrads() {
      for (const key in this.grads) this.grads[key].fill(0);
    }

    paramCount() {
      let n = 0;
      for (const key in this.params) {
        const p = this.params[key];
        n += p.q ? p.q.length : p.d.length;
      }
      return n;
    }

    /** 导出 INT8 权重量化推理副本（weight-only PTQ）。
     * 线性权重（wqkv/wproj/w1/w2/tokEmb/posEmb）转为 {q, scale} 常驻 Int8，
     * 偏置与归一化参数保持 Float64。原模型（训练态）不受影响。 */
    quantize() {
      const m = Object.create(GPTModel.prototype);
      m.cfg = this.cfg;
      m.useRms = this.useRms; m.useSwi = this.useSwi; m.useRope = this.useRope;
      m.KV = this.KV; m.g = this.g; m.fh = this.fh; m.dh = this.dh;
      m.qkvDim = this.qkvDim; m.quantized = true;
      m.nExp = this.nExp; m.topK = this.topK; m.useMoE = this.useMoE; m.auxWeight = this.auxWeight;
      m.window = this.window; m.windows = this.windows.slice();
      m.qkNorm = this.qkNorm; m.logitCap = this.logitCap; m.kvQuant = this.kvQuant; m.attentionSink = this.attentionSink;
      m.params = {};
      for (const key in this.params) {
        const p = this.params[key];
        if (p && p.d && isQuantLinear(key)) {
          const d = p.d;
          let maxAbs = 0;
          for (let i = 0; i < d.length; i++) { const a = Math.abs(d[i]); if (a > maxAbs) maxAbs = a; }
          const scale = maxAbs / 127 || 1;
          const q = new Int8Array(d.length);
          for (let i = 0; i < d.length; i++) {
            let v = Math.round(d[i] / scale);
            q[i] = v > 127 ? 127 : v < -127 ? -127 : v;
          }
          m.params[key] = { q, scale };
        } else {
          m.params[key] = { d: p.d ? p.d.slice() : p.d };
        }
      }
      return m;
    }

    /** 前向传播，返回 { logits(T×V), cache }，cache 供反向传播使用 */
    forward(tokens) {
      const { vocabSize: V, blockSize: B, nLayer: L, nHead: H, dModel: D } = this.cfg;
      const dh = this.dh, KV = this.KV, g = this.g, qkvDim = this.qkvDim;
      const P = this.params;
      const T = Math.min(tokens.length, B);
      const toks = tokens.length > B ? tokens.slice(tokens.length - B) : tokens;

      const NF = this.useRms ? rmsnormFwd : layernormFwd;
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
        if (this.qkNorm) {
          const qn = qkNormFwd(qkv, T, H, dh, qkvDim, 0, P[k + 'qg'].d);  // Q 段（偏移 0，H 个头）
          const kn = qkNormFwd(qkv, T, KV, dh, qkvDim, D, P[k + 'kg'].d); // K 段（偏移 D，KV 个头）
          qk = { qx: qn.xhat, qr: qn.rstd, kx: kn.xhat, kr: kn.rstd };
        }
        if (this.useRope) ropeApply(qkv, T, D, H, dh, 0, 1, KV, qkvDim);
        const ctx = new Float64Array(T * D);
        const Ps = [];
        const scale = 1 / Math.sqrt(dh);
        for (let h = 0; h < H; h++) {
          const kvh = (h / g) | 0;                 // 该 Q 头对应的 KV 头
          const ko = D + kvh * dh;                 // K 段偏移
          const vo = D + KV * dh + kvh * dh;        // V 段偏移
          const S = new Float64Array(T * T);
          const win = this.windows[l]; // 0 = 该层全因果（GPT-2 原行为）
          for (let i = 0; i < T; i++) {
            const qi = i * qkvDim + h * dh;
            // 因果掩码（j <= i）叠加滑动窗口左边界（j >= i-window+1，Mistral/Gemma）
            const jLo = win > 0 && i - win + 1 > 0 ? i - win + 1 : 0;
            for (let j = jLo; j <= i; j++) { // 窗口内：计算注意力分数
              const kj = j * qkvDim + ko;
              let s = 0;
              for (let p = 0; p < dh; p++) s += qkv[qi + p] * qkv[kj + p];
              S[i * T + j] = s * scale;
            }
            for (let j = 0; j < jLo; j++) S[i * T + j] = -1e9;  // 窗口左侧屏蔽（窗外历史）
            for (let j = i + 1; j < T; j++) S[i * T + j] = -1e9; // 未来位置屏蔽
          }
          const Pm = softmaxRows(S, T, T);
          Ps.push(Pm);
          for (let i = 0; i < T; i++) {
            const jLo = win > 0 && i - win + 1 > 0 ? i - win + 1 : 0;
            for (let p = 0; p < dh; p++) {
              let s = 0;
              for (let j = jLo; j <= i; j++) s += Pm[i * T + j] * qkv[j * qkvDim + vo + p];
              ctx[i * D + h * dh + p] = s;
            }
          }
        }
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
        const FH = this.fh;
        let ff, pre1 = null, h1 = null, moe = null;
        if (this.useMoE) {
          const r = moeForward(this, P, k, M, T, D, FH);
          ff = r.ff; moe = r.moe;
        } else {
          const W1 = this.useSwi ? 2 * FH : FH;
          pre1 = mm(M, P[k + 'w1'], T, D, W1);
          for (let i = 0; i < T; i++) {
            const o = i * W1;
            for (let j = 0; j < W1; j++) pre1[o + j] += P[k + 'b1'].d[j];
          }
          h1 = this.useSwi ? swigluFwd(pre1, T, FH) : geluFwd(pre1);
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
      if (this.logitCap > 0) {
        const cap = this.logitCap;
        for (let i = 0; i < logits.length; i++) logits[i] = cap * Math.tanh(logits[i] / cap);
      }
      return { logits, cache: { T, dh, H, D, L, layers, lnf, Hf, toks, logits } };
    }

    /** 反向传播：dlogits(T×V) -> 累加至 this.grads */
    backward(cache, dlogits) {
      const { T, dh, H, D, L } = cache;
      const { vocabSize: V } = this.cfg;
      const P = this.params, G = this.grads;
      const NB = this.useRms ? rmsnormBwd : layernormBwd;

      // 权重共享输出头反向：logits = Hf @ Wte^T
      //   dWte = dlogits^T @ Hf ；dHf = dlogits @ Wte
      // Gemma 2 logit soft-cap 反向：lg = cap·tanh(x/cap) ⇒ dlg/dx = sech²(x/cap) = 1 − (lg/cap)²
      // （lg 为缓存中的 capped logits，lg/cap 即 tanh(x/cap)，不可再套一层 tanh）
      if (this.logitCap > 0) {
        const cap = this.logitCap, lg = cache.logits;
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
        if (this.useMoE) {
          dM = moeBackward(this, P, G, c.moe, dff, T, D);
        } else {
          const FH = this.fh, W1 = this.useSwi ? 2 * FH : FH;
          addInto(G[k + 'w2'], matmulTA(c.h1, dff, T, FH, D));
          rowSumInto(G[k + 'b2'], dff, T, D);
          const dh1 = matmulAtB(dff, P[k + 'w2'].d, T, D, FH);
          const dpre1 = this.useSwi ? swigluBwd(dh1, c.pre1, T, FH) : geluBwd(dh1, c.pre1);
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

        const qkvDim = this.qkvDim, KV = this.KV, g = this.g;
        const dqkv = new Float64Array(T * qkvDim);
        for (let h = 0; h < H; h++) {
          const kvh = (h / g) | 0;
          const ko = D + kvh * dh;
          const vo = D + KV * dh + kvh * dh;
          const Pm = c.Ps[h];
          const dP = new Float64Array(T * T);
          const dVh = new Float64Array(T * dh);
          for (let i = 0; i < T; i++) {
            for (let j = 0; j < T; j++) {
              let s = 0;
              for (let p = 0; p < dh; p++) s += dctx[i * D + h * dh + p] * c.qkv[j * qkvDim + vo + p];
              dP[i * T + j] = s;
            }
          }
          // dV[j] = Σ_{i≥j} P[i][j] · dctx[i]（ctx[i] = Σ_j P[i][j]·V[j]，按列累加）
          for (let j = 0; j < T; j++) {
            for (let p = 0; p < dh; p++) {
              let s = 0;
              for (let i = j; i < T; i++) s += Pm[i * T + j] * dctx[i * D + h * dh + p];
              dVh[j * dh + p] = s;
            }
          }
          const dS = softmaxBwd(dP, Pm, T, T);
          const scale = 1 / Math.sqrt(dh);
          // dQ[i] = Σ_{j≤i} dS[i][j] · K[j]（因果掩码右侧为零）
          for (let i = 0; i < T; i++) {
            for (let p = 0; p < dh; p++) {
              let sq = 0;
              for (let j = 0; j <= i; j++) {
                sq += dS[i * T + j] * c.qkv[j * qkvDim + ko + p];
              }
              dqkv[i * qkvDim + h * dh + p] = sq * scale;
            }
          }
          // dK[i] = Σ_{a≥i} dS[a][i] · Q[a]（列方向：所有 a≥i 的行都向 K[i] 传梯度）
          for (let i = 0; i < T; i++) {
            for (let p = 0; p < dh; p++) {
              let sk = 0;
              for (let a = i; a < T; a++) {
                sk += dS[a * T + i] * c.qkv[a * qkvDim + h * dh + p];
              }
              dqkv[i * qkvDim + ko + p] += sk * scale; // 共享 KV 头：多个 Q 头累加
            }
          }
          for (let i = 0; i < T; i++) {
            for (let p = 0; p < dh; p++) dqkv[i * qkvDim + vo + p] += dVh[i * dh + p];
          }
        }
        // RoPE 反向：旋转是正交变换，其雅可比的转置 = 反向旋转
        if (this.useRope) ropeApply(dqkv, T, D, H, dh, 0, -1, KV, qkvDim);
        // QK-norm 反向（前向顺序：norm → RoPE，反向顺序：RoPE⁻¹ → norm⁻¹）
        if (this.qkNorm) {
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

    /** 评估模式：返回 (loss, 预测准确率) */
    evaluate(tokens) {
      const V = this.cfg.vocabSize;
      const T = Math.min(tokens.length, this.cfg.blockSize);
      if (T < 2) return { loss: Infinity, acc: 0 };
      const toks = tokens.length > this.cfg.blockSize ? tokens.slice(tokens.length - this.cfg.blockSize) : tokens;
      const { logits } = this.forward(toks);
      const targets = [];
      for (let i = 1; i < T; i++) targets.push(toks[i]);
      const { loss } = crossEntropy(logits, targets, T - 1, V);
      let correct = 0;
      for (let i = 0; i < T - 1; i++) {
        let best = 0;
        for (let j = 1; j < V; j++) if (logits[i * V + j] > logits[i * V + best]) best = j;
        if (best === targets[i]) correct++;
      }
      return { loss, acc: correct / (T - 1) };
    }
  }

  /* ==========================================================================
   * 四、训练器（AdamW + 梯度裁剪 + warmup/cosine 学习率）
   * ======================================================================== */
  function train(model, tokens, opts) {
    const o = Object.assign({
      steps: 200, lr: 3e-3, batch: null, warmup: 20, seed: 0,
      logEvery: 20, onStep: null, wd: 0.01, gradClip: 1.0, fixedBatch: null,
      state: null, stepOffset: 0, totalSteps: null,
    }, opts);
    if (tokens.length < 2 && !o.fixedBatch) throw new Error('训练语料至少需要 2 个 token');
    if (o.fixedBatch && o.fixedBatch.length < 2) throw new Error('固定批次至少需要 2 个 token');
    const B = o.batch || model.cfg.blockSize;
    const V = model.cfg.vocabSize;
    const total = o.totalSteps || o.steps;
    const off = o.stepOffset;
    // Adam 状态：传入 state 可断点续训（分块训练不丢动量与数据流）
    const st = o.state || { m: {}, v: {}, t: 0, rngA: o.seed >>> 0 };
    const m = st.m, v = st.v;
    for (const key in model.params) {
      if (!(key in m)) m[key] = new Float64Array(model.params[key].d.length);
      if (!(key in v)) v[key] = new Float64Array(model.params[key].d.length);
    }
    let a = st.rngA === undefined ? (o.seed >>> 0) : st.rngA; // 批次采样 RNG 状态随续训延续
    const rng = function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    const b1 = 0.9, b2 = 0.95, eps = 1e-8;
    let t = st.t;
    const history = [];
    for (let step = 0; step < o.steps; step++) {
      const g = step + off; // 全局步号（用于学习率调度）
      // 学习率：线性 warmup -> cosine 衰减至 10%
      const lrNow = g < o.warmup
        ? o.lr * (g + 1) / o.warmup
        : o.lr * (0.1 + 0.9 * 0.5 * (1 + Math.cos(Math.PI * Math.min(1, (g - o.warmup) / Math.max(1, total - o.warmup)))));
      let batchToks;
      if (o.fixedBatch) {
        batchToks = o.fixedBatch;
      } else {
        const maxStart = Math.max(0, tokens.length - B - 1);
        const start = Math.floor(rng() * (maxStart + 1));
        batchToks = [];
        for (let i = 0; i < Math.min(B, tokens.length - 1); i++) batchToks.push(tokens[start + i]);
      }
      const T = batchToks.length;
      const targets = [];
      for (let i = 1; i < T; i++) targets.push(batchToks[i]);
      model.zeroGrads();
      const { logits, cache } = model.forward(batchToks);
      const { loss, dlogits } = crossEntropy(logits, targets, T - 1, V, T);
      model.backward(cache, dlogits);
      // 负载均衡辅助损失（仅 MoE）：E·Σ_e f_e·P_e，乘以 auxWeight 并入总损失
      let reported = loss;
      if (model.useMoE && model.auxWeight) {
        let aux = 0;
        for (let l = 0; l < model.cfg.nLayer; l++) aux += cache.layers[l].moe.aux;
        reported = loss + model.auxWeight * aux;
      }
      // 全局梯度范数裁剪
      let gsq = 0;
      for (const key in model.grads) {
        const g = model.grads[key];
        for (let i = 0; i < g.length; i++) gsq += g[i] * g[i];
      }
      const norm = Math.sqrt(gsq);
      const clipScale = norm > o.gradClip ? o.gradClip / norm : 1;
      t++;
      const bc1 = 1 - Math.pow(b1, t), bc2 = 1 - Math.pow(b2, t);
      for (const key in model.params) {
        const p = model.params[key].d, g = model.grads[key], mk = m[key], vk = v[key];
        for (let i = 0; i < p.length; i++) {
          const gi = g[i] * clipScale;
          mk[i] = b1 * mk[i] + (1 - b1) * gi;
          vk[i] = b2 * vk[i] + (1 - b2) * gi * gi;
          const mh = mk[i] / bc1, vh = vk[i] / bc2;
          p[i] -= lrNow * (mh / (Math.sqrt(vh) + eps) + o.wd * p[i]); // AdamW：解耦权重衰减
        }
      }
      history.push(reported);
      if (o.onStep) o.onStep(step, reported, lrNow, norm);
    }
    st.t = t;
    st.rngA = a;
    return { history, finalLoss: history[history.length - 1], state: st };
  }

  /* ==========================================================================
   * 五、推理加速：KV Cache（增量解码）
   * ----------------------------------------------------------------------------
   * 复用 GPT-2 标准 KV 缓存：每步只算 1 个新 token，历史 K/V 直接复用。
   * 生成 N 个 token 的复杂度由 O(N·T²·D) 降至 O(T²·D + N·T·D)。
   * 硬不变量：缓存路径输出的 logits 必须与全序列 forward 的最后一行逐位相等。
   * ======================================================================== */

  /** 初始化 KV 缓存：每层 K/V 各预分配 blockSize×D */
  /**
   * KV 缓存量化写入：把一行 K/V（kvDim 维）按「每 token × 每 KV 头」粒度做对称 absmax 量化。
   * 粒度选 per-(token,head) 而非 per-token：RoPE 后同一头内数值量级接近，分头可显著压低动态范围，
   * 且写入时即可确定 scale（无需校准集、无需回填重标量），天然适配增量追加。
   * 误差硬界：|dequant(x) − x| ≤ scale/2 = absmax/254（相对误差 ≤ 1/254 ≈ 0.39%）。
   */
  function kvWriteRow(buf, sc, row, kvDim, KV, dh, src, srcOff) {
    const o = row * kvDim;
    for (let hh = 0; hh < KV; hh++) {
      let ab = 0;
      for (let p = 0; p < dh; p++) { const v = Math.abs(src[srcOff + hh * dh + p]); if (v > ab) ab = v; }
      const s = ab > 0 ? ab / 127 : 1; // 全零行 scale 取 1（解量化仍为 0）
      sc[row * KV + hh] = s;
      const b = o + hh * dh;
      for (let p = 0; p < dh; p++) {
        let q = Math.round(src[srcOff + hh * dh + p] / s);
        if (q > 127) q = 127; else if (q < -127) q = -127;
        buf[b + p] = q;
      }
    }
  }

  /**
   * 初始化 KV 缓存。kvQuant=8 时 K/V 存 Int8Array（+ per-(token,head) scale），
   * 每元素 1 字节 + 每头 8 字节 scale；kvQuant=0（默认）为 Float64Array，逐位兼容旧行为。
   */
  function initKVCache(model) {
    const { nLayer: L, blockSize: B } = model.cfg;
    const kvDim = model.KV * model.dh; // GQA 下 K/V 每位置维度 = KV 头数 · headDim
    const KV = model.KV, q8 = model.kvQuant === 8;
    // Attention Sink / StreamingLLM：缓存容量 = 锚点 K + 滚动窗口 R；无锚点时退化为原滑动窗口（容量 = blockSize）
    const sink = model.attentionSink > 0 ? model.attentionSink : 0;
    const maxWin = model.windows.reduce((a, b) => (b > a ? b : a), 0);
    const R = maxWin > 0 ? maxWin : (model.cfg.rollingWindow > 0 ? Math.floor(model.cfg.rollingWindow) : 256);
    const cap = sink > 0 ? sink + R : B;
    const K = [], V = [], Ks = [], Vs = [];
    for (let l = 0; l < L; l++) {
      K.push(q8 ? new Int8Array(cap * kvDim) : new Float64Array(cap * kvDim));
      V.push(q8 ? new Int8Array(cap * kvDim) : new Float64Array(cap * kvDim));
      if (q8) { Ks.push(new Float64Array(cap * KV)); Vs.push(new Float64Array(cap * KV)); }
    }
    return { K, V, Ks, Vs, n: 0, q8, cap, sink };
  }

  /**
   * 单步增量前向：处理绝对位置 pos 上的单个 token，写入并复用 KV 缓存。
   * 缓存槽位（kv.n）与绝对位置（pos）解耦：
   *  - 槽位是缓存数组下标，写满后靠 kvShift 平移；
   *  - pos 只喂给 RoPE（可无限增长）与位置嵌入（钳位到 blockSize-1）。
   *  RoPE 的注意力分数只依赖相对距离，因此滑动窗口后仍与全序列重算逐位一致。
   */
  function forwardStep(model, tokenId, pos, kv) {
    const { dModel: D, nHead: H, nLayer: L, vocabSize: V0, blockSize: B } = model.cfg;
    const dh = model.dh, scale = 1 / Math.sqrt(dh);
    const KV = model.KV, qkvDim = model.qkvDim, kvDim = KV * dh;
    const P = model.params;
    const NF = model.useRms ? rmsnormFwd : layernormFwd;
    const nb = (key) => (P[key] ? P[key].d : null);
    const FH = model.fh;
    const slot = kv.n;        // 写入槽位
    const nValid = slot + 1;  // 参与注意力的有效位置数（含当前 token）

    // 1) token 嵌入 + 位置嵌入（单行，量化权重走解量化查表）
    let X = new Float64Array(D);
    const posIdx = Math.min(pos, B - 1);
    const te = P.tokEmb, pe = P.posEmb;
    for (let j = 0; j < D; j++) {
      const tv = te.q ? te.q[tokenId * D + j] * te.scale : te.d[tokenId * D + j];
      X[j] = pe
        ? (pe.q ? pe.q[posIdx * D + j] * pe.scale : pe.d[posIdx * D + j]) + tv
        : tv;
    }

    for (let l = 0; l < L; l++) {
      const k = `l${l}.`;
      // 2) pre-norm #1 -> QKV
      const A = NF(X, P[k + 'ln1.g'].d, nb(k + 'ln1.b'), 1, D).Y;
      const qkv = mm(A, P[k + 'wqkv'], 1, D, qkvDim);
      for (let j = 0; j < qkvDim; j++) qkv[j] += P[k + 'bqkv'].d[j];
      // QK-norm（RoPE 前，逐头 RMSNorm，与 forward 同序）
      if (model.qkNorm) {
        qkNormFwd(qkv, 1, H, dh, qkvDim, 0, P[k + 'qg'].d);
        qkNormFwd(qkv, 1, KV, dh, qkvDim, D, P[k + 'kg'].d);
      }
      if (model.useRope) ropeApply(qkv, 1, D, H, dh, pos, 1, KV, qkvDim);

      // 3) 写入缓存（K 取 QKV 中段，V 取尾段；GQA 下 K/V 各 kvDim 维）
      //    kvQuant=8：写入即量化（Int8 + per-(token,head) scale）；否则原样存 Float64
      const Ko = kv.K[l], Vo = kv.V[l], o = slot * kvDim;
      if (kv.q8) {
        kvWriteRow(Ko, kv.Ks[l], slot, kvDim, KV, dh, qkv, D);
        kvWriteRow(Vo, kv.Vs[l], slot, kvDim, KV, dh, qkv, D + KV * dh);
      } else {
        for (let j = 0; j < kvDim; j++) { Ko[o + j] = qkv[D + j]; Vo[o + j] = qkv[D + KV * dh + j]; }
      }

      // 4) 因果 + 滑动窗口 + Attention Sink 注意力：
      //    窗口 [jLo, nValid) 内 attend；若启用锚点(sink>0) 且锚点在窗口左侧，额外补 [0, sinkCount) 恒驻锚点（StreamingLLM）。
      //    sink=0 时退化为原窗口注意力（逐位兼容）。
      const win = model.windows[l] > 0 ? Math.min(model.windows[l], nValid) : nValid;
      const jLo = nValid - win; // 窗口左边界（缓存槽位下标）
      const sink = model.attentionSink > 0 ? model.attentionSink : 0;
      const sinkCount = sink > 0 ? Math.min(sink, jLo) : 0; // 窗口外、需显式补的锚点数
      const attnLen = win + sinkCount;
      const ctx = new Float64Array(D);
      const kSc = kv.q8 ? kv.Ks[l] : null, vSc = kv.q8 ? kv.Vs[l] : null;
      for (let h = 0; h < H; h++) {
        const kvh = (h / model.g) | 0;
        const S = new Float64Array(attnLen);
        let si = 0;
        for (let j = jLo; j < nValid; j++) {
          // kvQuant=8 时缓存里是 int8 码字，读出时乘回该 (token,head) 的 scale 即解量化
          const ks = kSc ? kSc[j * KV + kvh] : 1;
          let s = 0;
          for (let p = 0; p < dh; p++) s += qkv[h * dh + p] * (Ko[j * kvDim + kvh * dh + p] * ks);
          S[si++] = s * scale;
        }
        for (let j = 0; j < sinkCount; j++) {
          const ks = kSc ? kSc[j * KV + kvh] : 1;
          let s = 0;
          for (let p = 0; p < dh; p++) s += qkv[h * dh + p] * (Ko[j * kvDim + kvh * dh + p] * ks);
          S[si++] = s * scale;
        }
        const Pm = softmaxRows(S, 1, attnLen);
        for (let p = 0; p < dh; p++) {
          let s = 0, pi = 0;
          for (let j = jLo; j < nValid; j++) {
            const vs = vSc ? vSc[j * KV + kvh] : 1;
            s += Pm[pi++] * (Vo[j * kvDim + kvh * dh + p] * vs);
          }
          for (let j = 0; j < sinkCount; j++) {
            const vs = vSc ? vSc[j * KV + kvh] : 1;
            s += Pm[pi++] * (Vo[j * kvDim + kvh * dh + p] * vs);
          }
          ctx[h * dh + p] = s;
        }
      }
      let attn = mm(ctx, P[k + 'wproj'], 1, D, D);
      for (let j = 0; j < D; j++) attn[j] += P[k + 'bproj'].d[j];
      const xMid = new Float64Array(D);
      for (let j = 0; j < D; j++) xMid[j] = X[j] + attn[j];

      // 5) pre-norm #2 -> FFN（MoE 稀疏 / 稠密 双路径，单 token 增量）
      const M = NF(xMid, P[k + 'ln2.g'].d, nb(k + 'ln2.b'), 1, D).Y;
      let ff;
      if (model.useMoE) {
        const E = model.nExp, K = model.topK, W1 = model.useSwi ? 2 * FH : FH;
        const rlogits = mm(M, P[k + 'wg'], 1, D, E);
        for (let j = 0; j < E; j++) rlogits[j] += P[k + 'bg'].d[j];
        const probs = softmaxRows(rlogits, 1, E);
        const order = [];
        for (let j = 0; j < E; j++) order.push(j);
        order.sort((a, b) => probs[a] - probs[b]); // 升序，取尾部 K 个
        ff = new Float64Array(D);
        for (let t = 0; t < K; t++) {
          const e = order[E - 1 - t], ek = `${k}e${e}.`;
          const pre = mm(M, P[ek + 'w1'], 1, D, W1);
          for (let j = 0; j < W1; j++) pre[j] += P[ek + 'b1'].d[j];
          const h1 = model.useSwi ? swigluFwd(pre, 1, FH) : geluFwd(pre);
          const out = mm(h1, P[ek + 'w2'], 1, FH, D);
          for (let j = 0; j < D; j++) out[j] += P[ek + 'b2'].d[j];
          const g = probs[e];
          for (let j = 0; j < D; j++) ff[j] += g * out[j];
        }
      } else {
        const W1 = model.useSwi ? 2 * FH : FH;
        const pre1 = mm(M, P[k + 'w1'], 1, D, W1);
        for (let j = 0; j < W1; j++) pre1[j] += P[k + 'b1'].d[j];
        const h1 = model.useSwi ? swigluFwd(pre1, 1, FH) : geluFwd(pre1);
        ff = mm(h1, P[k + 'w2'], 1, FH, D);
        for (let j = 0; j < D; j++) ff[j] += P[k + 'b2'].d[j];
      }
      const xOut = new Float64Array(D);
      for (let j = 0; j < D; j++) xOut[j] = xMid[j] + ff[j];
      X = xOut;
    }

    // 6) 最终归一化 + 权重共享输出头（+ logit soft-cap，与 forward 一致）
    const Hf = NF(X, P['lnf.g'].d, nb('lnf.b'), 1, D).Y;
    kv.n = slot + 1;
    let logits = mmAtB(Hf, P.tokEmb, 1, D, V0);
    if (model.logitCap > 0) {
      const cap = model.logitCap;
      for (let j = 0; j < logits.length; j++) logits[j] = cap * Math.tanh(logits[j] / cap);
    }
    return { logits };
  }

  /** 缓存淘汰（Attention Sink / 滑动窗口）：
   *  - sink>0（StreamingLLM）：恒驻 [0,sink) 锚点，仅平移窗口区 [sink, n) 左移 1，丢弃最旧的非锚点 token；
   *  - sink=0：整体平移（丢弃最旧 token，含首 token）—— 纯滑动窗口，逐位兼容旧行为。
   *  K/V 与（kvQuant=8 时）scale 同步平移，解量化不错位。 */
  function kvShift(model, kv) {
    const { nLayer: L } = model.cfg;
    const kvDim = model.KV * model.dh, KV = model.KV;
    const n = kv.n, sink = kv.sink || 0;
    if (sink > 0) {
      // 仅滚动窗口区 [sink+1, n) -> [sink, n-1)，锚点 [0,sink) 原地不动（copyWithin 处理重叠左移）
      const from = (sink + 1) * kvDim, to = sink * kvDim, cnt = (n - sink) * kvDim;
      for (let l = 0; l < L; l++) {
        kv.K[l].copyWithin(to, from, from + cnt);
        kv.V[l].copyWithin(to, from, from + cnt);
        if (kv.q8) { // scale 与码字必须同步平移，否则解量化错位
          const Ksf = (sink + 1) * KV, Kst = sink * KV, Ksc = (n - sink) * KV;
          kv.Ks[l].copyWithin(Kst, Ksf, Ksf + Ksc);
          kv.Vs[l].copyWithin(Kst, Ksf, Ksf + Ksc);
        }
      }
    } else {
      for (let l = 0; l < L; l++) {
        kv.K[l].copyWithin(0, kvDim, n * kvDim);
        kv.V[l].copyWithin(0, kvDim, n * kvDim);
        if (kv.q8) {
          kv.Ks[l].copyWithin(0, KV, n * KV);
          kv.Vs[l].copyWithin(0, KV, n * KV);
        }
      }
    }
    kv.n = n - 1;
  }

  /** 用给定 ids 预热缓存，返回末位 logits。
   *  - sink>0（StreamingLLM）：处理整段 prompt，缓存满则按「锚点 + 滚动窗口」淘汰中段（保留首 sink 个锚点）。
   *  - sink=0：原行为，超出 blockSize 取最后 B 个。 */
  function warmupKV(model, ids, kv) {
    if (kv.sink > 0) {
      let logits = null;
      for (let i = 0; i < ids.length; i++) {
        if (kv.n >= kv.cap) kvShift(model, kv);
        logits = forwardStep(model, ids[i], i, kv).logits;
      }
      return logits;
    }
    const B = model.cfg.blockSize;
    const toks = ids.length > B ? ids.slice(ids.length - B) : ids;
    let logits = null;
    for (let i = 0; i < toks.length; i++) logits = forwardStep(model, toks[i], i, kv).logits;
    kv.n = toks.length;
    return logits;
  }

  /* ==========================================================================
   * 六、模型持久化（无损 JSON）
   * ======================================================================== */
  /** 序列化模型。一并存词表（字符数组），保证导入后可完整重建分词器。
   * 量化权重存为 {q:[...], s:scale}（Int8），未量化存为普通浮点数组。 */
  function saveModel(model, tok) {
    const params = {};
    let quantized = false;
    for (const key in model.params) {
      const p = model.params[key];
      if (p.q) { params[key] = { q: Array.from(p.q), s: p.scale }; quantized = true; }
      else params[key] = Array.from(p.d);
    }
    return JSON.stringify({
      format: 'flagship-ai', version: 2,
      cfg: model.cfg, params, quantized,
      vocab: tok ? tok.chars : null,
      tokKind: tok ? (tok.kind || 'char') : null,
      merges: tok && tok.kind === 'bpe' ? tok.merges : null,
      bpeByteLevel: tok && tok.kind === 'bpe' ? !!tok.byteLevel : null,
    });
  }

  /** 从存档恢复分词器（存档无词表时返回 null） */
  function loadTokenizer(json) {
    const obj = typeof json === 'string' ? JSON.parse(json) : json;
    if (!obj || !Array.isArray(obj.vocab)) return null;
    if (obj.tokKind === 'bpe') {
      const t = Object.create(BPETokenizer.prototype);
      t.kind = 'bpe';
      t.chars = obj.vocab;
      t.itos = obj.vocab;
      t.stoi = new Map(obj.vocab.map((s, i) => [s, i]));
      t.merges = obj.merges || [];
      t.byteLevel = obj.bpeByteLevel !== false;
      t.ranks = new Map(t.merges.map((m, i) => [m[0] + m[1], i]));
      return t;
    }
    const t = Object.create(Tokenizer.prototype);
    t.kind = 'char';
    t.chars = obj.vocab;
    t.stoi = new Map(obj.vocab.map((ch, i) => [ch, i]));
    t.itos = obj.vocab;
    return t;
  }

  function loadModel(json) {
    const obj = typeof json === 'string' ? JSON.parse(json) : json;
    if (!obj || obj.format !== 'flagship-ai') throw new Error('无法识别的模型格式');
    const m = new GPTModel(obj.cfg);
    for (const key in obj.params) {
      const src = obj.params[key];
      if (Array.isArray(src)) {
        const dst = m.params[key];
        if (!dst) throw new Error(`模型缺少参数 ${key}`);
        if (src.length !== dst.d.length) throw new Error(`参数 ${key} 长度不匹配`);
        dst.d.set(src);
      } else {
        // 量化权重：{q, s}
        m.params[key] = { q: Int8Array.from(src.q), scale: src.s };
      }
    }
    if (obj.quantized) m.quantized = true;
    return m;
  }

  /* ==========================================================================
   * 七、生成器（温度 / top-k / top-p / 贪心 / KV Cache）
   * ======================================================================== */
  function generate(model, tok, prompt, opts) {
    const o = Object.assign({
      maxNew: 100, temperature: 0.8, topK: 0, topP: 0, seed: 1234, useCache: true,
    }, opts);
    // maxTokens 为 maxNew 的别名，两者等价
    const total = o.maxTokens != null ? o.maxTokens : o.maxNew;
    const rng = mulberry32(o.seed);
    const V = model.cfg.vocabSize;
    const B = model.cfg.blockSize;
    const ids = tok.encode(prompt);
    if (ids.length === 0) ids.push(Math.floor(rng() * V));
    const out = [];

    const kv = o.useCache ? (o.kv || initKVCache(model)) : null;
    // 预热（缓存模式）：位置嵌入从 0 开始，与全序列 forward 保持一致
    let logits = kv ? warmupKV(model, ids, kv) : null;
    // 锚点模式：pos 取绝对长度（warmup 已按绝对位置写缓存）；否则沿用末位（与原行为一致）
    let pos = kv ? (kv.sink > 0 ? ids.length : Math.min(ids.length, B)) : 0;

    for (let s = 0; s < total; s++) {
      if (!kv) {
        // 回退路径：每步全序列重算（用于交叉验证与数值对照）
        const ctxIds = ids.slice(-B);
        logits = model.forward(ctxIds).logits;
      }
      const last = kv ? 0 : (Math.min(ids.length, B) - 1) * V;
      let idx;
      if (o.temperature <= 0) { // 贪心解码
        let best = 0;
        for (let j = 1; j < V; j++) if (logits[last + j] > logits[last + best]) best = j;
        idx = best;
      } else {
        const probs = new Float64Array(V);
        let mx = -Infinity;
        for (let j = 0; j < V; j++) {
          const val = logits[last + j] / o.temperature;
          probs[j] = val;
          if (val > mx) mx = val;
        }
        // top-k：仅保留概率最高的 k 个
        if (o.topK > 0 && o.topK < V) {
          const sorted = Array.from(probs).sort((a, b) => b - a);
          const thr = sorted[o.topK - 1];
          for (let j = 0; j < V; j++) if (probs[j] < thr) probs[j] = -Infinity;
          mx = -Infinity;
          for (let j = 0; j < V; j++) if (probs[j] > mx) mx = probs[j];
        }
        let sum = 0;
        for (let j = 0; j < V; j++) { const e = Math.exp(probs[j] - mx); probs[j] = e; sum += e; }
        for (let j = 0; j < V; j++) probs[j] /= sum;
        // top-p（核心采样）：在 top-k 之后的分布上按累积概率截断长尾
        if (o.topP > 0 && o.topP < 1) {
          const order = Array.from(probs).map((p, j) => ({ p, j })).sort((a, b) => b.p - a.p);
          let acc = 0, cut = order.length;
          for (let i = 0; i < order.length; i++) {
            acc += order[i].p;
            if (acc >= o.topP) { cut = i + 1; break; }
          }
          sum = 0;
          for (let i = 0; i < order.length; i++) {
            if (i >= cut) probs[order[i].j] = 0; else sum += probs[order[i].j];
          }
          for (let j = 0; j < V; j++) probs[j] /= sum;
        }
        const r = rng();
        let acc = 0;
        idx = V - 1;
        for (let j = 0; j < V; j++) { acc += probs[j]; if (r <= acc) { idx = j; break; } }
      }
      ids.push(idx);
      out.push(idx);

      // 推进 KV 缓存（pos 为绝对位置，可无限增长；槽位由 kv.n 管理）
      // - 序列长度 < 缓存容量：纯增量，与全序列 forward 严格逐位等价
      // - 超出后：缓存满则淘汰（Attention Sink 保锚点 / 否则滑动窗口丢弃最旧 token）。
      //   RoPE 下注意力只依赖相对距离，故窗口滑动后仍与全序列重算逐位一致。
      if (kv) {
        if (kv.n >= (kv.cap || B)) kvShift(model, kv);
        logits = forwardStep(model, idx, pos, kv).logits;
        pos++;
      }
    }
    return { text: tok.decode(out), ids: out };
  }

  /* 架构预设：一行切换到现代 LLM 的标准配方 */
  const ARCH_PRESETS = {
    gpt2: { norm: 'ln', ffn: 'gelu', pos: 'learned' },                 // GPT-2 / 本项目默认
    llama: { norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope' },            // LLaMA / Mistral（MHA）
    'llama-gqa': { norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope', nKvHead: 2 }, // LLaMA + GQA 演示
    // Mixtral 风格：LLaMA 配方 + GQA + MoE（4 专家 top-2 路由）。nExp/topK 可在 cfg 覆盖。
    mixtral: { norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope', nKvHead: 2, nExp: 4, topK: 2, auxWeight: 0.01 },
    // 滑动窗口注意力（SWA）：LLaMA 配方 + GQA + window=64（Mistral/Gemma 稀疏注意力）。
    swa: { norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope', nKvHead: 2, window: 64 },
    // Gemma 2/3 完整配方：LLaMA 基座 + GQA + 局部/全局层交错 + QK-norm + logit soft-cap 30。
    gemma: { norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope', nKvHead: 2, window: { alt: 64 }, qkNorm: true, logitCap: 30 },
  };
  /** 把预设叠到配置上：applyPreset(cfg, 'llama') */
  function applyPreset(cfg, preset) {
    const p = ARCH_PRESETS[preset];
    if (!p) throw new Error(`未知架构预设：${preset}（可用：${Object.keys(ARCH_PRESETS).join(', ')}）`);
    return Object.assign({}, cfg, p);
  }

  return {
    mulberry32, randn,
    matmul, matmulAtB, matmulTA,
    layernormFwd, layernormBwd,
    rmsnormFwd, rmsnormBwd,
    geluFwd, geluBwd, siluF, swigluFwd, swigluBwd,
    ropeApply, softmaxRows, softmaxBwd,
    crossEntropy,
    Tokenizer, BPETokenizer, GPTModel, train, generate,
    initKVCache, forwardStep, warmupKV, kvShift, saveModel, loadModel, loadTokenizer,
    ARCH_PRESETS, applyPreset, textToByteChars, byteCharsToText,
  };
});

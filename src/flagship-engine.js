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
   * 三、GPT 模型（前向 + 手写反向传播）
   * ======================================================================== */
  class GPTModel {
    /**
     * @param {Object} cfg { vocabSize, blockSize, nLayer, nHead, dModel, seed }
     */
    constructor(cfg) {
      if (cfg.dModel % cfg.nHead !== 0) {
        throw new Error(`dModel(${cfg.dModel}) 必须能被 nHead(${cfg.nHead}) 整除`);
      }
      this.cfg = Object.assign({ seed: 42 }, cfg);
      const { vocabSize: V, blockSize: T, nLayer: L, nHead: H, dModel: D } = this.cfg;
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
      add('posEmb', T, D);
      const rscale = 1 / Math.sqrt(2 * L); // GPT-2：按残差深度缩放输出投影初始化
      for (let l = 0; l < L; l++) {
        const k = `l${l}.`;
        add(k + 'wqkv', D, 3 * D); zeros(k + 'bqkv', 3 * D);
        add(k + 'wproj', D, D, rscale); zeros(k + 'bproj', D);
        add(k + 'w1', D, 4 * D); zeros(k + 'b1', 4 * D);
        add(k + 'w2', 4 * D, D, rscale); zeros(k + 'b2', D);
        ones(k + 'ln1.g', D); zeros(k + 'ln1.b', D);
        ones(k + 'ln2.g', D); zeros(k + 'ln2.b', D);
      }
      ones('lnf.g', D); zeros('lnf.b', D);
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
      for (const key in this.params) n += this.params[key].d.length;
      return n;
    }

    /** 前向传播，返回 { logits(T×V), cache }，cache 供反向传播使用 */
    forward(tokens) {
      const { vocabSize: V, blockSize: B, nLayer: L, nHead: H, dModel: D } = this.cfg;
      const dh = D / H;
      const P = this.params;
      const T = Math.min(tokens.length, B);
      const toks = tokens.length > B ? tokens.slice(tokens.length - B) : tokens;

      // 1) token 嵌入 + 位置嵌入
      let X = new Float64Array(T * D);
      const tokEmb = P.tokEmb.d, posEmb = P.posEmb.d;
      for (let i = 0; i < T; i++) {
        const t = toks[i], o = i * D;
        for (let j = 0; j < D; j++) X[o + j] = tokEmb[t * D + j] + posEmb[o + j];
      }

      const layers = [];
      for (let l = 0; l < L; l++) {
        const k = `l${l}.`;
        // 2) pre-LN #1 -> 因果多头自注意力
        const ln1 = layernormFwd(X, P[k + 'ln1.g'].d, P[k + 'ln1.b'].d, T, D);
        const A = ln1.Y;
        const qkv = matmul(A, P[k + 'wqkv'].d, T, D, 3 * D);
        for (let i = 0; i < T; i++) {
          const o = i * 3 * D;
          for (let j = 0; j < 3 * D; j++) qkv[o + j] += P[k + 'bqkv'].d[j];
        }
        const ctx = new Float64Array(T * D);
        const Ps = [];
        const scale = 1 / Math.sqrt(dh);
        for (let h = 0; h < H; h++) {
          const S = new Float64Array(T * T);
          for (let i = 0; i < T; i++) {
            const qi = i * 3 * D + h * dh;
            for (let j = 0; j <= i; j++) { // 因果掩码：只 attend 到 j <= i
              const kj = j * 3 * D + D + h * dh;
              let s = 0;
              for (let p = 0; p < dh; p++) s += qkv[qi + p] * qkv[kj + p];
              S[i * T + j] = s * scale;
            }
            for (let j = i + 1; j < T; j++) S[i * T + j] = -1e9; // 未来位置屏蔽
          }
          const Pm = softmaxRows(S, T, T);
          Ps.push(Pm);
          for (let i = 0; i < T; i++) {
            for (let p = 0; p < dh; p++) {
              let s = 0;
              for (let j = 0; j <= i; j++) s += Pm[i * T + j] * qkv[j * 3 * D + 2 * D + h * dh + p];
              ctx[i * D + h * dh + p] = s;
            }
          }
        }
        let attn = matmul(ctx, P[k + 'wproj'].d, T, D, D);
        for (let i = 0; i < T; i++) {
          const o = i * D;
          for (let j = 0; j < D; j++) attn[o + j] += P[k + 'bproj'].d[j];
        }
        const xMid = new Float64Array(T * D);
        for (let i = 0; i < T * D; i++) xMid[i] = X[i] + attn[i]; // 残差连接

        // 3) pre-LN #2 -> 前馈网络（GELU）
        const ln2 = layernormFwd(xMid, P[k + 'ln2.g'].d, P[k + 'ln2.b'].d, T, D);
        const M = ln2.Y;
        const pre1 = matmul(M, P[k + 'w1'].d, T, D, 4 * D);
        for (let i = 0; i < T; i++) {
          const o = i * 4 * D;
          for (let j = 0; j < 4 * D; j++) pre1[o + j] += P[k + 'b1'].d[j];
        }
        const h1 = geluFwd(pre1);
        const ff = matmul(h1, P[k + 'w2'].d, T, 4 * D, D);
        for (let i = 0; i < T; i++) {
          const o = i * D;
          for (let j = 0; j < D; j++) ff[o + j] += P[k + 'b2'].d[j];
        }
        const xOut = new Float64Array(T * D);
        for (let i = 0; i < T * D; i++) xOut[i] = xMid[i] + ff[i]; // 残差连接

        layers.push({ ln1, A, qkv, Ps, ctx, xMid, ln2, M, pre1, h1 });
        X = xOut;
      }

      // 4) 最终 LayerNorm + 权重共享输出头
      const lnf = layernormFwd(X, P['lnf.g'].d, P['lnf.b'].d, T, D);
      const Hf = lnf.Y;
      const logits = matmulAtB(Hf, P.tokEmb.d, T, D, V);
      return { logits, cache: { T, dh, H, D, L, layers, lnf, Hf, toks } };
    }

    /** 反向传播：dlogits(T×V) -> 累加至 this.grads */
    backward(cache, dlogits) {
      const { T, dh, H, D, L } = cache;
      const { vocabSize: V } = this.cfg;
      const P = this.params, G = this.grads;

      // 权重共享输出头反向：logits = Hf @ Wte^T
      //   dWte = dlogits^T @ Hf ；dHf = dlogits @ Wte
      addInto(G.tokEmb, matmulTA(dlogits, cache.Hf, T, V, D));
      const dHf = matmul(dlogits, P.tokEmb.d, T, V, D);
      // 最终 LayerNorm 反向（dY 形状 T×D）
      const lnfb = layernormBwd(dHf, cache.lnf.xhat, cache.lnf.rstd, P['lnf.g'].d, T, D);
      addInto(G['lnf.g'], lnfb.dg);
      addInto(G['lnf.b'], lnfb.db);
      let dX = lnfb.dX;

      for (let l = L - 1; l >= 0; l--) {
        const k = `l${l}.`;
        const c = cache.layers[l];
        // xOut = xMid + ff => dff = dX, dxMid = dX
        const dff = dX;
        addInto(G[k + 'w2'], matmulTA(c.h1, dff, T, 4 * D, D));
        rowSumInto(G[k + 'b2'], dff, T, D);
        const dh1 = matmulAtB(dff, P[k + 'w2'].d, T, D, 4 * D);
        const dpre1 = geluBwd(dh1, c.pre1);
        addInto(G[k + 'w1'], matmulTA(c.M, dpre1, T, D, 4 * D));
        rowSumInto(G[k + 'b1'], dpre1, T, 4 * D);
        const dM = matmulAtB(dpre1, P[k + 'w1'].d, T, 4 * D, D);
        const ln2b = layernormBwd(dM, c.ln2.xhat, c.ln2.rstd, P[k + 'ln2.g'].d, T, D);
        addInto(G[k + 'ln2.g'], ln2b.dg);
        addInto(G[k + 'ln2.b'], ln2b.db);
        const dxMid = new Float64Array(T * D);
        for (let i = 0; i < T * D; i++) dxMid[i] = dX[i] + ln2b.dX[i];

        // xMid = x0 + attn => dattn = dxMid, dx0 = dxMid
        const dattn = dxMid;
        addInto(G[k + 'wproj'], matmulTA(c.ctx, dattn, T, D, D));
        rowSumInto(G[k + 'bproj'], dattn, T, D);
        const dctx = matmulAtB(dattn, P[k + 'wproj'].d, T, D, D);

        const dqkv = new Float64Array(T * 3 * D);
        for (let h = 0; h < H; h++) {
          const Pm = c.Ps[h];
          const dP = new Float64Array(T * T);
          const dVh = new Float64Array(T * dh);
          for (let i = 0; i < T; i++) {
            for (let j = 0; j < T; j++) {
              let s = 0;
              for (let p = 0; p < dh; p++) s += dctx[i * D + h * dh + p] * c.qkv[j * 3 * D + 2 * D + h * dh + p];
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
                sq += dS[i * T + j] * c.qkv[j * 3 * D + D + h * dh + p];
              }
              dqkv[i * 3 * D + h * dh + p] = sq * scale;
            }
          }
          // dK[i] = Σ_{a≥i} dS[a][i] · Q[a]（列方向：所有 a≥i 的行都向 K[i] 传梯度）
          for (let i = 0; i < T; i++) {
            for (let p = 0; p < dh; p++) {
              let sk = 0;
              for (let a = i; a < T; a++) {
                sk += dS[a * T + i] * c.qkv[a * 3 * D + h * dh + p];
              }
              dqkv[i * 3 * D + D + h * dh + p] = sk * scale;
            }
          }
          for (let i = 0; i < T; i++) {
            for (let p = 0; p < dh; p++) dqkv[i * 3 * D + 2 * D + h * dh + p] = dVh[i * dh + p];
          }
        }
        addInto(G[k + 'wqkv'], matmulTA(c.A, dqkv, T, D, 3 * D));
        rowSumInto(G[k + 'bqkv'], dqkv, T, 3 * D);
        const dA = matmulAtB(dqkv, P[k + 'wqkv'].d, T, 3 * D, D);
        const ln1b = layernormBwd(dA, c.ln1.xhat, c.ln1.rstd, P[k + 'ln1.g'].d, T, D);
        addInto(G[k + 'ln1.g'], ln1b.dg);
        addInto(G[k + 'ln1.b'], ln1b.db);
        const dx0 = new Float64Array(T * D);
        for (let i = 0; i < T * D; i++) dx0[i] = dxMid[i] + ln1b.dX[i];
        dX = dx0;
      }

      // 嵌入层反向（token 嵌入散列累加 + 位置嵌入）
      for (let i = 0; i < T; i++) {
        const t = cache.toks[i], o = i * D;
        for (let j = 0; j < D; j++) {
          G.tokEmb[t * D + j] += dX[o + j];
          G.posEmb[o + j] += dX[o + j];
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
      history.push(loss);
      if (o.onStep) o.onStep(step, loss, lrNow, norm);
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
  function initKVCache(model) {
    const { nLayer: L, blockSize: B, dModel: D } = model.cfg;
    const K = [], V = [];
    for (let l = 0; l < L; l++) {
      K.push(new Float64Array(B * D));
      V.push(new Float64Array(B * D));
    }
    return { K, V, n: 0 };
  }

  /** 单步增量前向：处理位置 pos 上的单个 token，写入并复用 KV 缓存 */
  function forwardStep(model, tokenId, pos, kv) {
    const { dModel: D, nHead: H, nLayer: L, vocabSize: V0 } = model.cfg;
    const dh = D / H, scale = 1 / Math.sqrt(dh);
    const P = model.params;

    // 1) token 嵌入 + 位置嵌入（单行）
    let X = new Float64Array(D);
    for (let j = 0; j < D; j++) X[j] = P.tokEmb.d[tokenId * D + j] + P.posEmb.d[pos * D + j];

    for (let l = 0; l < L; l++) {
      const k = `l${l}.`;
      // 2) pre-LN #1 -> QKV
      const A = layernormFwd(X, P[k + 'ln1.g'].d, P[k + 'ln1.b'].d, 1, D).Y;
      const qkv = matmul(A, P[k + 'wqkv'].d, 1, D, 3 * D);
      for (let j = 0; j < 3 * D; j++) qkv[j] += P[k + 'bqkv'].d[j];

      // 3) 写入缓存（K 取 QKV 中段，V 取尾段）
      const Ko = kv.K[l], Vo = kv.V[l], o = pos * D;
      for (let j = 0; j < D; j++) { Ko[o + j] = qkv[D + j]; Vo[o + j] = qkv[2 * D + j]; }

      // 4) 因果注意力：仅对已缓存的 pos+1 个位置 attend
      const ctx = new Float64Array(D);
      for (let h = 0; h < H; h++) {
        const S = new Float64Array(pos + 1);
        for (let j = 0; j <= pos; j++) {
          let s = 0;
          for (let p = 0; p < dh; p++) s += qkv[h * dh + p] * Ko[j * D + h * dh + p];
          S[j] = s * scale;
        }
        const Pm = softmaxRows(S, 1, pos + 1);
        for (let p = 0; p < dh; p++) {
          let s = 0;
          for (let j = 0; j <= pos; j++) s += Pm[j] * Vo[j * D + h * dh + p];
          ctx[h * dh + p] = s;
        }
      }
      let attn = matmul(ctx, P[k + 'wproj'].d, 1, D, D);
      for (let j = 0; j < D; j++) attn[j] += P[k + 'bproj'].d[j];
      const xMid = new Float64Array(D);
      for (let j = 0; j < D; j++) xMid[j] = X[j] + attn[j];

      // 5) pre-LN #2 -> FFN
      const M = layernormFwd(xMid, P[k + 'ln2.g'].d, P[k + 'ln2.b'].d, 1, D).Y;
      const pre1 = matmul(M, P[k + 'w1'].d, 1, D, 4 * D);
      for (let j = 0; j < 4 * D; j++) pre1[j] += P[k + 'b1'].d[j];
      const ff = matmul(geluFwd(pre1), P[k + 'w2'].d, 1, 4 * D, D);
      for (let j = 0; j < D; j++) ff[j] += P[k + 'b2'].d[j];
      const xOut = new Float64Array(D);
      for (let j = 0; j < D; j++) xOut[j] = xMid[j] + ff[j];
      X = xOut;
    }

    // 6) 最终 LayerNorm + 权重共享输出头
    const Hf = layernormFwd(X, P['lnf.g'].d, P['lnf.b'].d, 1, D).Y;
    return { logits: matmulAtB(Hf, P.tokEmb.d, 1, D, V0) };
  }

  /** 滑动窗口：丢弃最旧的 1 个 token（copyWithin 原生 memmove，O(B·D) 无重算）。
   * K/V 向量在写入时已携带当时的绝对位置信息，搬移不改变其数值。 */
  function kvShift(model, kv) {
    const { nLayer: L, dModel: D } = model.cfg;
    const n = kv.n;
    for (let l = 0; l < L; l++) {
      kv.K[l].copyWithin(0, D, n * D);
      kv.V[l].copyWithin(0, D, n * D);
    }
    kv.n = n - 1;
  }

  /** 用给定 ids 预热缓存，返回末位 logits（超出 blockSize 时取最后 blockSize 个） */
  function warmupKV(model, ids, kv) {
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
  /** 序列化模型。一并存词表（字符数组），保证导入后可完整重建分词器 */
  function saveModel(model, tok) {
    const params = {};
    for (const key in model.params) params[key] = Array.from(model.params[key].d);
    return JSON.stringify({
      format: 'flagship-ai', version: 1,
      cfg: model.cfg, params,
      vocab: tok ? tok.chars : null,
    });
  }

  /** 从存档恢复分词器（存档无词表时返回 null） */
  function loadTokenizer(json) {
    const obj = typeof json === 'string' ? JSON.parse(json) : json;
    if (!obj || !Array.isArray(obj.vocab)) return null;
    const t = Object.create(Tokenizer.prototype);
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
      const dst = m.params[key];
      if (!dst) throw new Error(`模型缺少参数 ${key}`);
      if (obj.params[key].length !== dst.d.length) throw new Error(`参数 ${key} 长度不匹配`);
      dst.d.set(obj.params[key]);
    }
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
    let pos = kv ? Math.min(ids.length, B) : 0;

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

      // 推进 KV 缓存
      // - 序列长度 < blockSize：纯增量，与全序列 forward 严格逐位等价
      // - 超出后：滑动窗口丢弃最旧 token（新 token 落在饱和位置 B-1）
      if (kv) {
        if (pos < B) {
          logits = forwardStep(model, idx, pos, kv).logits;
          pos++;
          kv.n++;
        } else {
          kvShift(model, kv);
          logits = forwardStep(model, idx, B - 1, kv).logits;
          kv.n = B;
        }
      }
    }
    return { text: tok.decode(out), ids: out };
  }

  return {
    mulberry32, randn,
    matmul, matmulAtB, matmulTA,
    layernormFwd, layernormBwd, geluFwd, geluBwd, softmaxRows, softmaxBwd,
    crossEntropy,
    Tokenizer, GPTModel, train, generate,
    initKVCache, forwardStep, warmupKV, kvShift, saveModel, loadModel, loadTokenizer,
  };
});

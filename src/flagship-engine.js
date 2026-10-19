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
  function ropeApply(X, T, D, H, dh, posOff, sign) {
    const half = dh >> 1;
    if (half <= 0) return;
    const bases = [0, D]; // Q 段与 K 段在 3D 行内的偏移
    for (let i = 0; i < T; i++) {
      const pos = i + posOff;
      const row = i * 3 * D;
      for (let p = 0; p < half; p++) {
        const th = pos * Math.pow(ROPE_BASE, -2 * p / dh);
        const cs = Math.cos(th), sn = Math.sin(th) * sign;
        for (let b = 0; b < 2; b++) {
          const o = row + bases[b] + (p << 1);
          // 每个 head 的该维度对
          for (let h = 0; h < H; h++) {
            const q = o + h * dh;
            const x0 = X[q], x1 = X[q + 1];
            X[q] = x0 * cs - x1 * sn;
            X[q + 1] = x0 * sn + x1 * cs;
          }
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
      this.useRope = this.cfg.pos === 'rope';
      this.useRms = this.cfg.norm === 'rmsnorm';
      this.useSwi = this.cfg.ffn === 'swiglu';
      this.fh = this.cfg.ffnHidden || 4 * D;
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
        add(k + 'wqkv', D, 3 * D); zeros(k + 'bqkv', 3 * D);
        add(k + 'wproj', D, D, rscale); zeros(k + 'bproj', D);
        // SwiGLU 的 w1 一次产出 gate 与 up 两半（D → 2·FH），少一次矩阵乘
        add(k + 'w1', D, this.useSwi ? 2 * FH : FH);
        zeros(k + 'b1', this.useSwi ? 2 * FH : FH);
        add(k + 'w2', FH, D, rscale); zeros(k + 'b2', D);
        ones(k + 'ln1.g', D); if (!this.useRms) zeros(k + 'ln1.b', D);
        ones(k + 'ln2.g', D); if (!this.useRms) zeros(k + 'ln2.b', D);
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

      const NF = this.useRms ? rmsnormFwd : layernormFwd;
      const nb = (key) => (P[key] ? P[key].d : null);

      // 1) token 嵌入 + 位置嵌入（RoPE 模式下无位置嵌入）
      let X = new Float64Array(T * D);
      const tokEmb = P.tokEmb.d, posEmb = P.posEmb ? P.posEmb.d : null;
      for (let i = 0; i < T; i++) {
        const t = toks[i], o = i * D;
        for (let j = 0; j < D; j++) {
          X[o + j] = posEmb ? tokEmb[t * D + j] + posEmb[o + j] : tokEmb[t * D + j];
        }
      }

      const layers = [];
      for (let l = 0; l < L; l++) {
        const k = `l${l}.`;
        // 2) pre-LN #1 -> 因果多头自注意力
        const ln1 = NF(X, P[k + 'ln1.g'].d, nb(k + 'ln1.b'), T, D);
        const A = ln1.Y;
        const qkv = matmul(A, P[k + 'wqkv'].d, T, D, 3 * D);
        for (let i = 0; i < T; i++) {
          const o = i * 3 * D;
          for (let j = 0; j < 3 * D; j++) qkv[o + j] += P[k + 'bqkv'].d[j];
        }
        // RoPE：就地旋转 Q/K 段（V 段不旋转）
        if (this.useRope) ropeApply(qkv, T, D, H, dh, 0, 1);
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
        const ln2 = NF(xMid, P[k + 'ln2.g'].d, nb(k + 'ln2.b'), T, D);
        const M = ln2.Y;
        const FH = this.fh;
        const pre1 = matmul(M, P[k + 'w1'].d, T, D, this.useSwi ? 2 * FH : FH);
        const W1 = this.useSwi ? 2 * FH : FH;
        for (let i = 0; i < T; i++) {
          const o = i * W1;
          for (let j = 0; j < W1; j++) pre1[o + j] += P[k + 'b1'].d[j];
        }
        const h1 = this.useSwi ? swigluFwd(pre1, T, FH) : geluFwd(pre1);
        const ff = matmul(h1, P[k + 'w2'].d, T, FH, D);
        for (let i = 0; i < T; i++) {
          const o = i * D;
          for (let j = 0; j < D; j++) ff[o + j] += P[k + 'b2'].d[j];
        }
        const xOut = new Float64Array(T * D);
        for (let i = 0; i < T * D; i++) xOut[i] = xMid[i] + ff[i]; // 残差连接

        layers.push({ ln1, A, qkv, Ps, ctx, xMid, ln2, M, pre1, h1 });
        X = xOut;
      }

      // 4) 最终归一化 + 权重共享输出头
      const lnf = NF(X, P['lnf.g'].d, nb('lnf.b'), T, D);
      const Hf = lnf.Y;
      const logits = matmulAtB(Hf, P.tokEmb.d, T, D, V);
      return { logits, cache: { T, dh, H, D, L, layers, lnf, Hf, toks } };
    }

    /** 反向传播：dlogits(T×V) -> 累加至 this.grads */
    backward(cache, dlogits) {
      const { T, dh, H, D, L } = cache;
      const { vocabSize: V } = this.cfg;
      const P = this.params, G = this.grads;
      const NB = this.useRms ? rmsnormBwd : layernormBwd;

      // 权重共享输出头反向：logits = Hf @ Wte^T
      //   dWte = dlogits^T @ Hf ；dHf = dlogits @ Wte
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
        const FH = this.fh, W1 = this.useSwi ? 2 * FH : FH;
        addInto(G[k + 'w2'], matmulTA(c.h1, dff, T, FH, D));
        rowSumInto(G[k + 'b2'], dff, T, D);
        const dh1 = matmulAtB(dff, P[k + 'w2'].d, T, D, FH);
        const dpre1 = this.useSwi ? swigluBwd(dh1, c.pre1, T, FH) : geluBwd(dh1, c.pre1);
        addInto(G[k + 'w1'], matmulTA(c.M, dpre1, T, D, W1));
        rowSumInto(G[k + 'b1'], dpre1, T, W1);
        const dM = matmulAtB(dpre1, P[k + 'w1'].d, T, W1, D);
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
        // RoPE 反向：旋转是正交变换，其雅可比的转置 = 反向旋转
        if (this.useRope) ropeApply(dqkv, T, D, H, dh, 0, -1);
        addInto(G[k + 'wqkv'], matmulTA(c.A, dqkv, T, D, 3 * D));
        rowSumInto(G[k + 'bqkv'], dqkv, T, 3 * D);
        const dA = matmulAtB(dqkv, P[k + 'wqkv'].d, T, 3 * D, D);
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

  /**
   * 单步增量前向：处理绝对位置 pos 上的单个 token，写入并复用 KV 缓存。
   * 缓存槽位（kv.n）与绝对位置（pos）解耦：
   *  - 槽位是缓存数组下标，写满后靠 kvShift 平移；
   *  - pos 只喂给 RoPE（可无限增长）与位置嵌入（钳位到 blockSize-1）。
   *  RoPE 的注意力分数只依赖相对距离，因此滑动窗口后仍与全序列重算逐位一致。
   */
  function forwardStep(model, tokenId, pos, kv) {
    const { dModel: D, nHead: H, nLayer: L, vocabSize: V0, blockSize: B } = model.cfg;
    const dh = D / H, scale = 1 / Math.sqrt(dh);
    const P = model.params;
    const NF = model.useRms ? rmsnormFwd : layernormFwd;
    const nb = (key) => (P[key] ? P[key].d : null);
    const FH = model.fh;
    const slot = kv.n;        // 写入槽位
    const nValid = slot + 1;  // 参与注意力的有效位置数（含当前 token）

    // 1) token 嵌入 + 位置嵌入（单行）
    let X = new Float64Array(D);
    const posIdx = Math.min(pos, B - 1);
    for (let j = 0; j < D; j++) {
      X[j] = P.posEmb
        ? P.tokEmb.d[tokenId * D + j] + P.posEmb.d[posIdx * D + j]
        : P.tokEmb.d[tokenId * D + j];
    }

    for (let l = 0; l < L; l++) {
      const k = `l${l}.`;
      // 2) pre-norm #1 -> QKV
      const A = NF(X, P[k + 'ln1.g'].d, nb(k + 'ln1.b'), 1, D).Y;
      const qkv = matmul(A, P[k + 'wqkv'].d, 1, D, 3 * D);
      for (let j = 0; j < 3 * D; j++) qkv[j] += P[k + 'bqkv'].d[j];
      if (model.useRope) ropeApply(qkv, 1, D, H, dh, pos, 1);

      // 3) 写入缓存（K 取 QKV 中段，V 取尾段）
      const Ko = kv.K[l], Vo = kv.V[l], o = slot * D;
      for (let j = 0; j < D; j++) { Ko[o + j] = qkv[D + j]; Vo[o + j] = qkv[2 * D + j]; }

      // 4) 因果注意力：仅对已缓存的 nValid 个位置 attend
      const ctx = new Float64Array(D);
      for (let h = 0; h < H; h++) {
        const S = new Float64Array(nValid);
        for (let j = 0; j < nValid; j++) {
          let s = 0;
          for (let p = 0; p < dh; p++) s += qkv[h * dh + p] * Ko[j * D + h * dh + p];
          S[j] = s * scale;
        }
        const Pm = softmaxRows(S, 1, nValid);
        for (let p = 0; p < dh; p++) {
          let s = 0;
          for (let j = 0; j < nValid; j++) s += Pm[j] * Vo[j * D + h * dh + p];
          ctx[h * dh + p] = s;
        }
      }
      let attn = matmul(ctx, P[k + 'wproj'].d, 1, D, D);
      for (let j = 0; j < D; j++) attn[j] += P[k + 'bproj'].d[j];
      const xMid = new Float64Array(D);
      for (let j = 0; j < D; j++) xMid[j] = X[j] + attn[j];

      // 5) pre-norm #2 -> FFN
      const M = NF(xMid, P[k + 'ln2.g'].d, nb(k + 'ln2.b'), 1, D).Y;
      const W1 = model.useSwi ? 2 * FH : FH;
      const pre1 = matmul(M, P[k + 'w1'].d, 1, D, W1);
      for (let j = 0; j < W1; j++) pre1[j] += P[k + 'b1'].d[j];
      const h1 = model.useSwi ? swigluFwd(pre1, 1, FH) : geluFwd(pre1);
      const ff = matmul(h1, P[k + 'w2'].d, 1, FH, D);
      for (let j = 0; j < D; j++) ff[j] += P[k + 'b2'].d[j];
      const xOut = new Float64Array(D);
      for (let j = 0; j < D; j++) xOut[j] = xMid[j] + ff[j];
      X = xOut;
    }

    // 6) 最终归一化 + 权重共享输出头
    const Hf = NF(X, P['lnf.g'].d, nb('lnf.b'), 1, D).Y;
    kv.n = slot + 1;
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
      format: 'flagship-ai', version: 2,
      cfg: model.cfg, params,
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

      // 推进 KV 缓存（pos 为绝对位置，可无限增长；槽位由 kv.n 管理）
      // - 序列长度 < blockSize：纯增量，与全序列 forward 严格逐位等价
      // - 超出后：滑动窗口丢弃最旧 token。RoPE 下注意力只依赖相对距离，
      //   故窗口滑动后仍与全序列重算逐位一致；可学习位置嵌入下为有界近似。
      if (kv) {
        if (pos >= B) kvShift(model, kv);
        logits = forwardStep(model, idx, pos, kv).logits;
        pos++;
      }
    }
    return { text: tok.decode(out), ids: out };
  }

  /* 架构预设：一行切换到现代 LLM 的标准配方 */
  const ARCH_PRESETS = {
    gpt2: { norm: 'ln', ffn: 'gelu', pos: 'learned' },                 // GPT-2 / 本项目默认
    llama: { norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope' },            // LLaMA / Mistral
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

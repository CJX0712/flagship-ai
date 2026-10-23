/* L7 张量与内存 · KV Cache（连续 KV + kvQuant=8 + Attention Sink）
 * 所属层：L7（张量与内存）｜ 允许依赖：L8（数值内核）、L9、L4（神经算子，惰性引用）
 * 主要 @ref：GPT-2 KV Cache；LLM.int8() kvQuant；StreamingLLM (Attention Sink)
 * 迁移方式：从 legacy flagship-engine.v11.js 第 1287–1528 行【原样搬】。
 *   - initKVCache / kvWriteRow / kvShift 为纯缓冲管理（自包含）。
 *   - forwardStep 内的算子调用改写为 NS.L8 / NS.L4（L4 于本文件之后加载，故惰性引用）；
 *     增量注意力数学下沉到 NS.L4.attention.attnStep（与本文件 forwardStep 逐位一致）。
 *   硬不变量：缓存路径 logits 与全序列 forward 末行逐位相等（R6 把关；generate 属 L2/T04）。
 */
;(function (NS) {
  NS.L7 = NS.L7 || {};
  NS.L7.kv_cache = NS.L7.kv_cache || {};
  const L8 = NS.L8;

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
    // ALiBi：各缓存槽位对应的「绝对位置」。仅在 ALiBi 开启时分配，默认路径零开销、逐位不变。
    //   为什么必需：ALiBi 偏置依赖 |qPos − keyPos|，而槽位与绝对位置在淘汰后会解耦
    //   （RoPE 只依赖相对距离故幸免；Attention Sink 的锚点+滚动窗口还会把序列“截断”，
    //    此时只有显式的 kv.pos 才能还原锚点到当前 token 的真实距离）。
    const pos = model.alibi ? new Float64Array(cap) : null;
    return { K, V, Ks, Vs, n: 0, q8, cap, sink, pos };
  }

  /**
   * 单步增量前向：处理绝对位置 pos 上的单个 token，写入并复用 KV 缓存。
   * 缓存槽位（kv.n）与绝对位置（pos）解耦：槽位是缓存数组下标，写满后靠 kvShift 平移；
   * pos 只喂给 RoPE（可无限增长）与位置嵌入（钳位到 blockSize-1）。
   * RoPE 的注意力分数只依赖相对距离，因此滑动窗口后仍与全序列重算逐位一致。
   */
  function forwardStep(model, tokenId, pos, kv) {
    const L4 = NS.L4; // 惰性引用：L4 于本文件之后加载
    const { dModel: D, nHead: H, nLayer: L, vocabSize: V0, blockSize: B } = model.cfg;
    const dh = model.dh, scale = 1 / Math.sqrt(dh);
    const KV = model.KV, qkvDim = model.qkvDim, kvDim = KV * dh;
    const P = model.params;
    const NF = model.useRms ? L4.norm.rmsnormFwd : L4.norm.layernormFwd;
    const mm = L8.matmul.mm, mmAtB = L8.matmul.mmAtB;
    const qkNormFwd = L4.norm.qkNormFwd, ropeApply = L4.positional.ropeApply;
    const attnStep = L4.attention.attnStep;
    const softmaxRows = L8.reduce.softmaxRows;
    const swigluFwd = L4.activation.swigluFwd, geluFwd = L4.activation.geluFwd;
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
      if (model.useRope) ropeApply(qkv, 1, D, H, dh, pos, 1, KV, qkvDim, model.ropeFreqScale);

      // 3) 写入缓存（K 取 QKV 中段，V 取尾段；GQA 下 K/V 各 kvDim 维）
      //    kvQuant=8：写入即量化（Int8 + per-(token,head) scale）；否则原样存 Float64
      const Ko = kv.K[l], Vo = kv.V[l], o = slot * kvDim;
      if (kv.q8) {
        kvWriteRow(Ko, kv.Ks[l], slot, kvDim, KV, dh, qkv, D);
        kvWriteRow(Vo, kv.Vs[l], slot, kvDim, KV, dh, qkv, D + KV * dh);
      } else {
        for (let j = 0; j < kvDim; j++) { Ko[o + j] = qkv[D + j]; Vo[o + j] = qkv[D + KV * dh + j]; }
      }
      // ALiBi：记录该槽位的绝对位置（attentionStep 用它算 |qPos − keyPos| 惩罚）
      if (kv.pos) kv.pos[slot] = pos;

      // 4) 因果 + 滑动窗口 + Attention Sink 注意力（数学下沉到 L4.attention.attnStep）
      const ctx = attnStep(model, l, qkv, kv, slot, nValid, D, H, dh, KV, kvDim, scale, pos);
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
   *  K/V 与（kvQuant=8 时）scale 同步平移，解量化不错位。
   *  ALiBi 的 kv.pos（绝对位置）同步平移 —— 否则偏置会按槽位距离而非真实距离计算。 */
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
      // 位置数组的每元素是 1 个数（不是 kvDim 个）：窗口区 [sink+1, n) -> [sink, n-1)，锚点不动
      if (kv.pos) kv.pos.copyWithin(sink, sink + 1, n);
    } else {
      for (let l = 0; l < L; l++) {
        kv.K[l].copyWithin(0, kvDim, n * kvDim);
        kv.V[l].copyWithin(0, kvDim, n * kvDim);
        if (kv.q8) {
          kv.Ks[l].copyWithin(0, KV, n * KV);
          kv.Vs[l].copyWithin(0, KV, n * KV);
        }
      }
      if (kv.pos) kv.pos.copyWithin(0, 1, n); // 整体左移 1
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

  NS.L7.kv_cache.kvWriteRow = kvWriteRow;
  NS.L7.kv_cache.initKVCache = initKVCache;
  NS.L7.kv_cache.forwardStep = forwardStep;
  NS.L7.kv_cache.kvShift = kvShift;
  NS.L7.kv_cache.warmupKV = warmupKV;
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

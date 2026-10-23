/* L4 神经算子 · 多头注意力（MHA / GQA / SWA / 层交错 / KV 增量）
 * 所属层：L4（神经算子）｜ 允许依赖：L8（softmaxRows/softmaxBwd 等）、L9
 * 主要 @ref：Vaswani et al. 2017 (Attention)；Gemma 2/3 层交错 SWA；StreamingLLM
 * 迁移方式：从 legacy flagship-engine.v11.js 拆出三类注意力数学：
 *   - attnFwd   全序列因果+滑动窗口注意力前向（forward 第 953–986 行）【原样搬】
 *   - attnBwd   全序列注意力反向 dP/dV/dS/dQ/dK 累加（backward 第 1091–1141 行）【原样搬】
 *   - attnStep  KV 增量单步注意力（forwardStep 第 1393–1425 行）【原样搬】
 *   RoPE 旋转 / QK-norm 属前置/后置步骤，仍由 L3 blocks 通过 L4.positional/norm 调用，
 *   与本文件的打分·softmax·上下文聚合解耦（保持 legacy 的调用顺序不变）。
 */
;(function (NS) {
  NS.L4 = NS.L4 || {};
  NS.L4.attention = NS.L4.attention || {};
  const L8 = NS.L8;
  const softmaxRows = L8.reduce.softmaxRows;
  const softmaxBwd = L8.reduce.softmaxBwd;

  /**
   * 全序列注意力前向（GQA + 因果 + 滑动窗口）。对 Q/K/V 段已备好的 qkv 打分。
   * @param qkv      Float64Array T×qkvDim，已过 QK-norm + RoPE（L3 blocks 前置处理）
   * @param T/D/H/dh/qkvDim/KV/g  形状与头配置
   * @param win      该层滑动窗口宽度（0 = 全因果，GPT-2 原行为）
   * @param scale    注意力缩放（=1/sqrt(dh)）
   * @param attnScale YaRN 注意力温度（=1 恒等）
   * @param slopes   ALiBi 逐头斜率（Float64Array(H)）；关闭时传 null/undefined（逐位兼容）
   * @returns {Ps, ctx} Ps 为各头 softmax 概率（反向复用），ctx 为 T×D 上下文
   */
  function attnFwd(qkv, T, D, H, dh, qkvDim, KV, g, win, scale, attnScale, slopes) {
    const ctx = new Float64Array(T * D);
    const Ps = [];
    for (let h = 0; h < H; h++) {
      const kvh = (h / g) | 0;                 // 该 Q 头对应的 KV 头
      const ko = D + kvh * dh;                 // K 段偏移
      const vo = D + KV * dh + kvh * dh;        // V 段偏移
      const S = new Float64Array(T * T);
      for (let i = 0; i < T; i++) {
        const qi = i * qkvDim + h * dh;
        // 因果掩码（j <= i）叠加滑动窗口左边界（j >= i-window+1，Mistral/Gemma）
        const jLo = win > 0 && i - win + 1 > 0 ? i - win + 1 : 0;
        for (let j = jLo; j <= i; j++) { // 窗口内：计算注意力分数
          const kj = j * qkvDim + ko;
          let s = 0;
          for (let p = 0; p < dh; p++) s += qkv[qi + p] * qkv[kj + p];
          S[i * T + j] = s * scale * attnScale;
          // ALiBi：把「距离惩罚」直接加到 softmax 前的打分上（−slope[h]·|i−j|；此处 j ≤ i）
          if (slopes) S[i * T + j] -= slopes[h] * (i - j);
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
    return { Ps, ctx };
  }

  /**
   * 全序列注意力反向：由 dctx 累加得到 dqkv（Q/K/V 三段一次成型）。
   * 共享 KV 头下多个 Q 头向同一 K/V 段累加；RoPE⁻¹/QK-norm⁻¹ 由 L3 blocks 后续处理。
   * @param c       该层 cache（读 c.Ps / c.qkv）
   * @param dctx    T×D 上游梯度
   * @param T/D/H/dh/qkvDim/KV/g  形状与头配置
   * @param scale / attnScale      与 forward 同因子（线性推送）
   * @returns dqkv  Float64Array T×qkvDim
   */
  function attnBwd(c, dctx, T, D, H, dh, qkvDim, KV, g, scale, attnScale) {
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
      // dQ[i] = Σ_{j≤i} dS[i][j] · K[j]（因果掩码右侧为零）
      for (let i = 0; i < T; i++) {
        for (let p = 0; p < dh; p++) {
          let sq = 0;
          for (let j = 0; j <= i; j++) {
            sq += dS[i * T + j] * c.qkv[j * qkvDim + ko + p];
          }
          dqkv[i * qkvDim + h * dh + p] = sq * scale * attnScale;
        }
      }
      // dK[i] = Σ_{a≥i} dS[a][i] · Q[a]（列方向：所有 a≥i 的行都向 K[i] 传梯度）
      for (let i = 0; i < T; i++) {
        for (let p = 0; p < dh; p++) {
          let sk = 0;
          for (let a = i; a < T; a++) {
            sk += dS[a * T + i] * c.qkv[a * qkvDim + h * dh + p];
          }
          dqkv[i * qkvDim + ko + p] += sk * scale * attnScale; // 共享 KV 头：多个 Q 头累加
        }
      }
      for (let i = 0; i < T; i++) {
        for (let p = 0; p < dh; p++) dqkv[i * qkvDim + vo + p] += dVh[i * dh + p];
      }
    }
    return dqkv;
  }

  /**
   * KV 增量单步注意力（forwardStep 路径）：读缓存 K/V（含 kvQuant=8 解量化），
   * 在窗口 [jLo, nValid) + Attention Sink 锚点 [0, sinkCount) 上聚合上下文。
   * @param model / l   模型与该层
   * @param qkv         当前 token 的 QKV（已过 QK-norm + RoPE，长度 qkvDim）
   * @param kv          KV 缓存对象（含 K/V/Ks/Vs）
   * @param slot        当前写入槽位；nValid 参与注意力的有效位置数（含当前）
   * @param D/H/dh/KV/kvDim/scale  形状配置
   * @param pos     当前 token 的绝对位置（ALiBi 需要；省略时退化为刚写入槽位 kv.pos[slot]）
   * @returns ctx  Float64Array(D)
   */
  function attnStep(model, l, qkv, kv, slot, nValid, D, H, dh, KV, kvDim, scale, pos) {
    const attnScale = model.attentionScaling || 1; // YaRN 注意力温度（与 forward 同因子）
    const Ko = kv.K[l], Vo = kv.V[l];
    const win = model.windows[l] > 0 ? Math.min(model.windows[l], nValid) : nValid;
    const jLo = nValid - win; // 窗口左边界（缓存槽位下标）
    const sink = model.attentionSink > 0 ? model.attentionSink : 0;
    const sinkCount = sink > 0 ? Math.min(sink, jLo) : 0; // 窗口外、需显式补的锚点数
    const attnLen = win + sinkCount;
    const ctx = new Float64Array(D);
    const kSc = kv.q8 ? kv.Ks[l] : null, vSc = kv.q8 ? kv.Vs[l] : null;
    // ALiBi：逐头斜率 + 缓存槽位的绝对位置（kv.pos）。关闭时 slopes=null ⇒ 零开销、零数值影响。
    const slopes = model.alibiSlopes || null;
    const posArr = kv.pos || null;
    const qPos = pos == null ? (posArr ? posArr[slot] : slot) : pos;
    for (let h = 0; h < H; h++) {
      const kvh = (h / model.g) | 0;
      const S = new Float64Array(attnLen);
      let si = 0;
      for (let j = jLo; j < nValid; j++) {
        // kvQuant=8 时缓存里是 int8 码字，读出时乘回该 (token,head) 的 scale 即解量化
        const ks = kSc ? kSc[j * KV + kvh] : 1;
        let s = 0;
        for (let p = 0; p < dh; p++) s += qkv[h * dh + p] * (Ko[j * kvDim + kvh * dh + p] * ks);
        const bias = slopes ? slopes[h] * (qPos - (posArr ? posArr[j] : j)) : 0;
        S[si++] = s * scale * attnScale - bias; // −slope·|qPos − keyPos|（窗口内 j ≤ qPos）
      }
      for (let j = 0; j < sinkCount; j++) {
        const ks = kSc ? kSc[j * KV + kvh] : 1;
        let s = 0;
        for (let p = 0; p < dh; p++) s += qkv[h * dh + p] * (Ko[j * kvDim + kvh * dh + p] * ks);
        // Attention Sink 锚点：它们在缓存里的绝对位置由 kv.pos[j] 记录（恒为 0..sinkCount-1）
        const bias = slopes ? slopes[h] * (qPos - (posArr ? posArr[j] : j)) : 0;
        S[si++] = s * scale * attnScale - bias;
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
    return ctx;
  }

  NS.L4.attention.attnFwd = attnFwd;
  NS.L4.attention.attnBwd = attnBwd;
  NS.L4.attention.attnStep = attnStep;
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

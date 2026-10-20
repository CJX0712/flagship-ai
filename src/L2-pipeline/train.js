/* L2 编排管线 · 训练器（AdamW + 梯度裁剪 + warmup/cosine 学习率）
 * 所属层：L2（编排管线）｜ 允许依赖：L4（神经算子 · crossEntropy）
 * 主要 @ref：AdamW（Loshchilov & Hutter 2019）；GPT-2 训练配方
 * 迁移方式：从 legacy flagship-engine.v11.js 第 1193–1277 行【原样搬】（Δ=0 契约）。
 *   - 唯一改写：模块级 crossEntropy(...) -> NS.L4.loss.crossEntropy(...)
 *     （L4 先于 L2 加载，顶部 const 捕获；loss 算子本身与 legacy 逐位一致，T03 已证）。
 *   - model.forward/backward/zeroGrads、model.params/grads/cfg/useMoE/auxWeight
 *     均为模型壳既有成员；批次采样 RNG（mulberry32 变体内联）原样保留，数据流冻结。
 */
;(function (NS) {
  NS.L2 = NS.L2 || {};
  NS.L2.train = NS.L2.train || {};
  const crossEntropy = NS.L4.loss.crossEntropy;

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

  NS.L2.train.train = train;
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

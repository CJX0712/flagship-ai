/* L2 编排管线 · 生成器（温度 / top-k / top-p / 贪心 / KV Cache）
 * 所属层：L2（编排管线）｜ 允许依赖：L7（张量与内存 · KV Cache）、L9（数值基础 · mulberry32）
 * 主要 @ref：GPT-2 采样；top-k/top-p（Holtzman 2019）；Attention Sink（StreamingLLM）
 * 迁移方式：从 legacy flagship-engine.v11.js 第 1599–1682 行【原样搬】（Δ=0 契约）。
 *   - 唯一改写：模块级 mulberry32 / initKVCache / warmupKV / forwardStep / kvShift
 *     分别改指 NS.L9.rng.mulberry32 与 NS.L7.kv_cache.*（L7/L9 先于 L2 加载，顶部捕获）。
 *   - 【关键】内联采样逻辑（temperature/top-k/top-p/greedy + RNG 抽取顺序）保持逐字一致，
 *     未提取为公共函数 —— 抽走会改变 RNG 消费顺序/数值路径，R5（generate Δ=0）即破。
 *     可复用的干净版本在 NS.L2.sampler.sample（generate 不使用它，仅供后续阶段）。
 *   - model.forward(ctxIds) 原样保留（模型由调用方传入）。
 */
;(function (NS) {
  NS.L2 = NS.L2 || {};
  NS.L2.generate = NS.L2.generate || {};
  const mulberry32 = NS.L9.rng.mulberry32;
  const initKVCache = NS.L7.kv_cache.initKVCache;
  const warmupKV = NS.L7.kv_cache.warmupKV;
  const forwardStep = NS.L7.kv_cache.forwardStep;
  const kvShift = NS.L7.kv_cache.kvShift;

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

  NS.L2.generate.generate = generate;
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

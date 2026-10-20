/* L2 编排管线 · 采样器（可复用的干净版 greedy / temperature / top-k / top-p）
 * 所属层：L2（编排管线）｜ 允许依赖：无（自包含，零跨层调用）
 * 主要 @ref：top-k/top-p 采样（Holtzman 2019）；temperature scaling
 *
 * 【注意】本文件是 NEW 代码（legacy 无对应物），不参与 Δ=0 契约：
 *   generate.js（NS.L2.generate）为保 R5 bit-exact，自带一份【逐字内联】的采样逻辑，
 *   并未调用本模块 —— 一旦抽成公共函数，RNG 消费顺序/数值路径有变动风险。
 *   本模块是面向后续阶段的 API 完备件；数值语义与 generate 内联版逐行同构
 *   （top-k 截断在温度缩放后的 logits 上、top-p 截断在归一化后的分布上）。
 */
;(function (NS) {
  NS.L2 = NS.L2 || {};
  NS.L2.sampler = NS.L2.sampler || {};

  /** 贪心解码：返回 argmax id */
  function greedy(logits) {
    let best = 0;
    for (let j = 1; j < logits.length; j++) if (logits[j] > logits[best]) best = j;
    return best;
  }

  /** 温度缩放 + 数值稳定 softmax：logits/T -> 归一化概率（Float64Array） */
  function softmaxTemp(logits, temperature) {
    const V = logits.length;
    const probs = new Float64Array(V);
    let mx = -Infinity;
    for (let j = 0; j < V; j++) {
      const val = logits[j] / temperature;
      probs[j] = val;
      if (val > mx) mx = val;
    }
    let sum = 0;
    for (let j = 0; j < V; j++) { const e = Math.exp(probs[j] - mx); probs[j] = e; sum += e; }
    for (let j = 0; j < V; j++) probs[j] /= sum;
    return probs;
  }

  /**
   * 采样一个 token id。语义与 generate 内联逻辑同构：
   *   - temperature <= 0：贪心（不消费 rng）
   *   - topK  > 0：保留概率最高的 k 个（在温度缩放后的 logits 上截断）
   *   - topP  > 0：在 top-k 之后的分布上按累积概率截断长尾
   *   - 最后按 rng() 单次抽取，累积概率线性扫描命中
   * @param logits Float64Array(V) 末行 logits
   * @param rng     () => [0,1) 确定性随机源（如 NS.L9.rng.mulberry32(seed)）
   * @param opts    { temperature, topK, topP }
   * @returns token id
   */
  function sample(logits, rng, opts) {
    const o = Object.assign({ temperature: 0.8, topK: 0, topP: 0 }, opts || {});
    const V = logits.length;
    if (o.temperature <= 0) return greedy(logits); // 贪心解码
    const probs = new Float64Array(V);
    let mx = -Infinity;
    for (let j = 0; j < V; j++) {
      const val = logits[j] / o.temperature;
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
    let idx = V - 1;
    for (let j = 0; j < V; j++) { acc += probs[j]; if (r <= acc) { idx = j; break; } }
    return idx;
  }

  NS.L2.sampler.sample = sample;
  NS.L2.sampler.greedy = greedy;
  NS.L2.sampler.softmaxTemp = softmaxTemp;
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

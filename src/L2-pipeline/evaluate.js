/* L2 编排管线 · 评估器（loss + 预测准确率）
 * 所属层：L2（编排管线）｜ 允许依赖：L4（神经算子 · crossEntropy）
 * 主要 @ref：next-token 预测评估（交叉熵 + top-1 准确率）
 * 迁移方式：从 legacy flagship-engine.v11.js 第 1170–1187 行 evaluate(tokens) 方法
 *   【改写为独立函数】（Δ=0 契约）：
 *   - this.* -> model.*（模型壳由参数传入）；this.forward -> model.forward
 *   - 模块级 crossEntropy(...) -> NS.L4.loss.crossEntropy(...)（顶部捕获，L4 先于 L2 加载）
 */
;(function (NS) {
  NS.L2 = NS.L2 || {};
  NS.L2.evaluate = NS.L2.evaluate || {};
  const crossEntropy = NS.L4.loss.crossEntropy;

  /** 评估模式：返回 (loss, 预测准确率) */
  function evaluate(model, tokens) {
    const V = model.cfg.vocabSize;
    const T = Math.min(tokens.length, model.cfg.blockSize);
    if (T < 2) return { loss: Infinity, acc: 0 };
    const toks = tokens.length > model.cfg.blockSize ? tokens.slice(tokens.length - model.cfg.blockSize) : tokens;
    const { logits } = model.forward(toks);
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

  NS.L2.evaluate.evaluate = evaluate;
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

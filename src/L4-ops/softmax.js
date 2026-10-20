/* L4 神经算子 · Softmax / Top-k
 * 所属层：L4（神经算子）｜ 允许依赖：L8、L9
 * 主要 @ref：数值稳定 softmax（max 平移）；MoE 路由 top-k（Shazeer 2018）
 * 迁移方式：从 legacy flagship-engine.v11.js 第 368–410 行【原样搬】。
 *   注意：这些与 L8 reduce.js 中的 softmaxRows/softmaxBwd/topkRows 逐位同源。
 *   模型热路径（forward/backward）统一走 L8 数值内核；此处提供 L4 语义门面，
 *   供 L5 自动微分与独立算子测试引用，数值与 L8 完全一致。
 */
;(function (NS) {
  NS.L4 = NS.L4 || {};
  NS.L4.softmax = NS.L4.softmax || {};
  // L8 数值内核（早于 L4 加载，见 __loadOrder）；提供逐位同源的实现
  const L8 = NS.L8;

  function softmaxRows(S, r, c) { return L8.reduce.softmaxRows(S, r, c); }
  function softmaxBwd(dP, P, r, c) { return L8.reduce.softmaxBwd(dP, P, r, c); }
  function topkRows(P, r, c, k) { return L8.reduce.topkRows(P, r, c, k); }

  NS.L4.softmax.softmaxRows = softmaxRows;
  NS.L4.softmax.softmaxBwd = softmaxBwd;
  NS.L4.softmax.topkRows = topkRows;
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

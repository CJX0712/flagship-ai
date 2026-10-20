/* L4 神经算子 · 算子注册表（v11 的 if-else → 注册式开关）
 * 所属层：L4（神经算子）｜ 允许依赖：L8、L9（本文件仅登记元数据，不参与热路径）
 * 说明：登记 L4 已实现的算子及其 fwd/bwd 引用，供自检 / UI / 文档查询。
 *   热路径（forward/backward）仍由 L3 blocks.js 直接调用具体算子函数，零 dispatch 开销；
 *   本表仅作为「算子目录 + 状态自省」，不改任何数值路径（Δ=0 无影响）。
 */
;(function (NS) {
  NS.L4 = NS.L4 || {};
  NS.L4.registry = NS.L4.registry || {};

  const _ops = Object.create(null);

  /**
   * 登记一个算子。
   * @param name   算子名（如 'rmsnorm'）
   * @param meta   { fwd, bwd, desc } —— fwd/bwd 为 L4 子命名空间下的函数引用
   */
  function register(name, meta) {
    _ops[name] = Object.assign({ name }, meta);
    return _ops[name];
  }

  function get(name) { return _ops[name] || null; }
  function has(name) { return name in _ops; }
  function list() { return Object.keys(_ops).map((k) => _ops[k]); }

  // 初次求值时立即登记当前已加载的 L4 算子（顺序无关，取值惰性安全）
  function bootstrap() {
    const L4 = NS.L4;
    if (!L4.norm) return; // 依赖未就绪则不登记（由 L3 装配时保证已加载）
    register('layernorm', { fwd: L4.norm.layernormFwd, bwd: L4.norm.layernormBwd, desc: 'LayerNorm (Ba et al. 2016)' });
    register('rmsnorm', { fwd: L4.norm.rmsnormFwd, bwd: L4.norm.rmsnormBwd, desc: 'RMSNorm (Zhang & Sennrich 2019)' });
    register('qknorm', { fwd: L4.norm.qkNormFwd, bwd: L4.norm.qkNormBwd, desc: 'QK-Norm (Gemma 2/3)' });
    register('gelu', { fwd: L4.activation.geluFwd, bwd: L4.activation.geluBwd, desc: 'GELU (tanh)' });
    register('swiglu', { fwd: L4.activation.swigluFwd, bwd: L4.activation.swigluBwd, desc: 'SwiGLU (Shazeer 2020)' });
    register('silu', { fwd: L4.activation.siluF, desc: 'SiLU/Swish' });
    register('rope', { fwd: L4.positional.ropeApply, desc: 'RoPE (Su et al. 2021)' });
    register('softmax', { fwd: L4.softmax.softmaxRows, bwd: L4.softmax.softmaxBwd, desc: '数值稳定 softmax' });
    register('topk', { fwd: L4.softmax.topkRows, desc: 'top-k 行选择（MoE 路由）' });
    register('moe', { fwd: L4.moe.moeForward, bwd: L4.moe.moeBackward, desc: 'MoE 稀疏 FFN' });
    register('attention', { fwd: L4.attention.attnFwd, bwd: L4.attention.attnBwd, desc: 'MHA/GQA/SWA 注意力' });
    register('crossentropy', { fwd: L4.loss.crossEntropy, desc: '交叉熵 + softmax 反向' });
  }

  NS.L4.registry.register = register;
  NS.L4.registry.get = get;
  NS.L4.registry.has = has;
  NS.L4.registry.list = list;
  NS.L4.registry.bootstrap = bootstrap;
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

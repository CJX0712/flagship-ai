/* ============================================================================
 * L5 自动微分 · VJP 注册中心（registry）
 * ----------------------------------------------------------------------------
 * 所属层：L5（自动微分）｜ 允许依赖：L4（仅函数体内惰性引用）、L8、L9
 * 主要 @ref：micrograd (Karpathy 2020) 的逐算子反向约定；
 *            Baydin et al. 2018 自动微分综述（反向模式 = VJP 链式复合）
 *
 * 背景（Path A「手写反向」决策）：P0 阶段 L5 不拥有计算路径，而是把 L4 已
 * 手写的前向/反向函数对登记为 VJP（vector-Jacobian product，向量-雅可比积）：
 *     vjpFn(cache, dY, ...rest) -> gradient(s)
 * 返回形状与对应 L4 算子的 backward 完全一致（如 layernormBwd -> {dX,dg,db}，
 * attnBwd -> dqkv）。P1 的 IR/tape 按名字取用同一份反向实现，从而与 v11
 * 逐位一致（R3 grads Δ=0），gradcheck 工厂再逐一证明解析 VJP ≈ 有限差分。
 *
 * 签名说明：登记的是 L4 backward 的原函数引用，调用签名即其原生签名 ——
 * 概念约定 (cache, dY, ...rest) 中 cache 位置由各算子反向需求决定：
 *   layernorm/rmsnorm: (dY, xhat, rstd, g, r, c) -> {dX,dg,db}
 *   qknorm:           (dqkv, xhat, rstd, rows, nHeads, dh, stride, off, g, dg)
 *   gelu/swiglu:      (dY, X) / (dout, pre, r, h)
 *   softmax:          (dP, P, r, c) -> dS
 *   moe:              (model, P, G, moe, dff, T, D) -> dM
 *   attention:        (c, dctx, T,D,H,dh,qkvDim,KV,g,scale,attnScale) -> dqkv
 *
 * 加载顺序注意（K4）：L5 求值期 L4 尚未加载（__loadOrder 中 L5 先于 L4），
 * 故全部 NS.L4 引用必须出现在函数体内（惰性），绝不可在顶层求值期捕获。
 * bootstrap() 因此对 L4 缺席安全：稍后由 get/has/list 惰性重试补齐。
 * ==========================================================================*/
;(function (NS) {
  'use strict';
  NS.L5 = NS.L5 || {};
  NS.L5.registry = NS.L5.registry || {};

  const vjps = new Map(); // name -> vjpFn（插入序；list() 输出前排序保确定性）
  let bootstrapped = false;

  /** 登记一个 VJP。name 必须为非空字符串，vjpFn 必须为函数；重复登记以新覆旧。 */
  function register(name, vjpFn) {
    if (typeof name !== 'string' || name.length === 0) {
      throw new TypeError('[L5] VJP 名称必须为非空字符串');
    }
    if (typeof vjpFn !== 'function') {
      throw new TypeError(`[L5] VJP "${name}" 必须是函数`);
    }
    vjps.set(name, vjpFn);
    return vjpFn;
  }

  /** 取用 VJP；未登记返回 null。内部惰性自举（L4 后到也能补齐）。 */
  function get(name) {
    if (!bootstrapped) bootstrap();
    const fn = vjps.get(name);
    return fn === undefined ? null : fn;
  }

  /** 是否已登记该 VJP。 */
  function has(name) {
    return get(name) !== null;
  }

  /** 已登记名称清单（字典序，确定性输出）。 */
  function list() {
    if (!bootstrapped) bootstrap();
    return Array.from(vjps.keys()).sort();
  }

  /**
   * 自举：把 L4 手写 backward 登记为 VJP（幂等）。
   * @returns true = 已登记就绪；false = L4 尚未加载（稍后再试，不抛）。
   */
  function bootstrap() {
    if (!NS.L4) return false;        // L4 尚未加载：安全退出，交由后续调用重试
    if (bootstrapped) return true;   // 幂等：重复调用不重复登记
    register('layernorm', NS.L4.norm.layernormBwd);
    register('rmsnorm',   NS.L4.norm.rmsnormBwd);
    register('qknorm',    NS.L4.norm.qkNormBwd);
    register('gelu',      NS.L4.activation.geluBwd);
    register('swiglu',    NS.L4.activation.swigluBwd);
    register('softmax',   NS.L4.softmax.softmaxBwd);
    register('moe',       NS.L4.moe.moeBackward);
    register('attention', NS.L4.attention.attnBwd);
    bootstrapped = true;
    return true;
  }

  NS.L5.registry.register = register;
  NS.L5.registry.get = get;
  NS.L5.registry.has = has;
  NS.L5.registry.list = list;
  NS.L5.registry.bootstrap = bootstrap;
  // 占位期旧键名兼容（docs/ARCHITECTURE-v12.md 曾以 vjp-registry 指代本模块）
  NS.L5.vjp_registry = NS.L5.registry;
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

/* ============================================================================
 * L6 后端运行时 · 内核注册表（kernel registry）
 * ----------------------------------------------------------------------------
 * 所属层：L6-backend
 * 允许依赖的层：L8/L9（求值期已加载，见 __loadOrder：L6 加载于 L8 之后）；
 *               L4 仅在函数体内惰性引用（L4 晚于 L6 加载，故 NS.L4 一律在
 *               bootstrap() 内守卫访问，绝不在本文件顶层求值期触碰）。
 * 主要 @ref：零回归契约 R9（逐算子 Δ=0）；共享规约 K3/K5（IIFE 注入 +
 *           跨层仅经 NS.Ln.x + 禁浏览器不友好语法）。
 *
 * 责任：提供 name → kernel fn 的登记/查询/派发设施。P0 定位为「零开销派发
 *       门面」：热路径不经过 dispatch，而是经 get() 一次性捕获函数引用后
 *       直接调用（与 legacy 闭包捕获同构，零间接层）；dispatch 仅供自检、
 *       工具与测试使用。bootstrap() 登记的全部内核均为 NS.L8 / NS.L4 的
 *       【同一函数引用】（Object.is 恒等），故数值路径逐位等价，Δ=0。
 * ==========================================================================*/
;(function (NS) {
  'use strict';
  NS.L6 = NS.L6 || {};
  NS.L6.kernel = NS.L6.kernel || {};

  // name → fn。null 原型：避免与 Object.prototype 键名冲突（如 'toString'）。
  const _kernels = Object.create(null);

  /**
   * 登记一个内核。允许覆盖同名登记 —— 显式再注册是注册表的预期用法。
   * @param name  内核名（如 'matmul'）
   * @param fn    函数引用
   * @returns     登记成功的 fn（便于调用方顺手捕获引用）
   */
  function register(name, fn) {
    if (typeof name !== 'string' || name.length === 0) {
      throw new Error('[L6] kernel.register: name 必须为非空字符串');
    }
    if (typeof fn !== 'function') {
      throw new Error('[L6] kernel.register: fn 必须为函数（收到 ' + typeof fn + '）');
    }
    _kernels[name] = fn;
    return fn;
  }

  // 仅当缺席时登记。bootstrap 的 L4 段使用：保护 L8 段已登记引用的
  // Object.is 恒等契约（L4 若出现同名门面也不得顶替 L8 原引用）。
  function registerIfAbsent(name, fn) {
    if (has(name)) return _kernels[name];
    return register(name, fn);
  }

  /** 取内核 fn；未登记返回 null。热路径契约：调用方捕获本返回值后直接调用。 */
  function get(name) {
    const fn = _kernels[name];
    return typeof fn === 'function' ? fn : null;
  }

  /** 是否已登记。 */
  function has(name) { return typeof _kernels[name] === 'function'; }

  /** 全部已登记内核名（插入序）。 */
  function list() { return Object.keys(_kernels); }

  /**
   * 派发（非热路径助手）：查表并调用。
   * 热路径请勿走本函数 —— 应经 get() 一次性捕获 fn 引用后直接调用，
   * 与 legacy 的闭包捕获同构，零间接层、零回归。
   */
  function dispatch(name, ...args) {
    const fn = get(name);
    if (!fn) throw new Error('[L6] kernel.dispatch: 未登记的内核 "' + name + '"');
    return fn(...args);
  }

  /**
   * 自举登记（幂等，可重复调用）：
   *   1) L8 段：matmul 族 5 项 + reduce 族 5 项（L8 先于 L6 加载，求值期可用）；
   *   2) L4 段：norm / activation / positional / attention / moe / loss /
   *      linear 门面（L4 晚于 L6 加载，故整体守卫 if (NS.L4)；缺席则本段
   *      跳过，待 L4 就绪后任意时机再次调用本函数即可补登 —— 在 L4 加载
   *      之前或之后调用均安全）；
   *   3) 桥接：若后端层已就绪，联动刷新各已注册后端（补登其 L4 内核镜像）。
   * 全部登记项均为 NS.L8 / NS.L4 的同一函数引用（Object.is 恒等 → Δ=0）。
   */
  function bootstrap() {
    // ── L8 段（求值期即可用）────────────────────────────────────────────
    const L8 = NS.L8;
    if (L8) {
      if (L8.matmul) {
        register('matmul', L8.matmul.matmul);
        register('matmulAtB', L8.matmul.matmulAtB);
        register('matmulTA', L8.matmul.matmulTA);
        register('mm', L8.matmul.mm);
        register('mmAtB', L8.matmul.mmAtB);
      }
      if (L8.reduce) {
        register('softmaxRows', L8.reduce.softmaxRows);
        register('softmaxBwd', L8.reduce.softmaxBwd);
        register('addInto', L8.reduce.addInto);
        register('rowSumInto', L8.reduce.rowSumInto);
        register('topkRows', L8.reduce.topkRows);
      }
    }
    // ── L4 段（惰性守卫：L4 缺席则整段跳过）─────────────────────────────
    const L4 = NS.L4;
    if (L4) {
      if (L4.norm) {
        registerIfAbsent('layernormFwd', L4.norm.layernormFwd);
        registerIfAbsent('layernormBwd', L4.norm.layernormBwd);
        registerIfAbsent('rmsnormFwd', L4.norm.rmsnormFwd);
        registerIfAbsent('rmsnormBwd', L4.norm.rmsnormBwd);
        registerIfAbsent('qkNormFwd', L4.norm.qkNormFwd);
        registerIfAbsent('qkNormBwd', L4.norm.qkNormBwd);
      }
      if (L4.activation) {
        registerIfAbsent('geluFwd', L4.activation.geluFwd);
        registerIfAbsent('geluBwd', L4.activation.geluBwd);
        registerIfAbsent('siluF', L4.activation.siluF);
        registerIfAbsent('swigluFwd', L4.activation.swigluFwd);
        registerIfAbsent('swigluBwd', L4.activation.swigluBwd);
      }
      if (L4.positional) registerIfAbsent('ropeApply', L4.positional.ropeApply);
      if (L4.attention) {
        registerIfAbsent('attnFwd', L4.attention.attnFwd);
        registerIfAbsent('attnBwd', L4.attention.attnBwd);
        registerIfAbsent('attnStep', L4.attention.attnStep);
      }
      if (L4.moe) {
        registerIfAbsent('moeForward', L4.moe.moeForward);
        registerIfAbsent('moeBackward', L4.moe.moeBackward);
      }
      if (L4.loss) registerIfAbsent('crossEntropy', L4.loss.crossEntropy);
      if (L4.linear) {
        registerIfAbsent('linear', L4.linear.linear);
        registerIfAbsent('addBias', L4.linear.addBias);
      }
      // 注：L4.softmax 三姐妹与 L4.linear.mm/mmAtB 是 L8 的纯门面（同名同义），
      // 不重复登记 —— 保留 L8 原引用，维持 Object.is 恒等。
    }
    // ── 桥接刷新后端 ────────────────────────────────────────────────────
    // 本文件先于 backend.js 求值，故求值期调用时 NS.L6.backend 尚不存在，
    // 守卫自然跳过；待 L4 就绪后由调用方再次触发 bootstrap 时生效。
    if (NS.L6.backend && typeof NS.L6.backend.refreshBackends === 'function') {
      NS.L6.backend.refreshBackends();
    }
  }

  NS.L6.kernel.register = register;
  NS.L6.kernel.get = get;
  NS.L6.kernel.has = has;
  NS.L6.kernel.list = list;
  NS.L6.kernel.dispatch = dispatch;
  NS.L6.kernel.bootstrap = bootstrap;

  // 求值期自举：L8 已就绪（先于 L6 加载），立即登记 10 项 L8 内核；
  // L4 尚未加载，L4 段按守卫跳过（L4 就绪后再次调用 bootstrap() 补登）。
  bootstrap();
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

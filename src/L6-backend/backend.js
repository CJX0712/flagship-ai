/* ============================================================================
 * L6 后端运行时 · 后端抽象（backend registry / profile / fallback）
 * ----------------------------------------------------------------------------
 * 所属层：L6-backend
 * 允许依赖的层：无跨层求值依赖（本文件只管理后端对象容器与 profile 状态；
 *               对 L8/L9/L4 的引用经各后端对象自身持有，本文件不直接触碰）。
 * 主要 @ref：零回归契约 R9（Δ=0）；P0 范围裁定（默认 CPU(JS) exact 单通道，
 *           fast/blocked 路径 deferred to P2）。
 *
 * 责任：后端对象的注册 / 激活 / 查询，以及数值通道 profile 的状态管理。
 *   - kernel(name) 返回【活跃后端】的内核 fn（或 null）。热路径契约：调用方
 *     一次性捕获该引用后直接调用，此后零派发开销（与 legacy 闭包捕获同构）。
 *   - profile：'exact'（默认，逐位精确通道）/ 'fast'（快速通道意图声明）。
 *     P0 仅实现 exact 一档：'fast' 与 'exact' 解析到同一批逐位等价内核，
 *     数值差恒为 0（≤ 任何 diff 上界）；fast/blocked 路径 deferred to P2，
 *     届时接入新后端对象而非改动本契约。
 *   - caps：各后端自省元数据（如 cpu: { simd:false, threads:false,
 *     deterministic:true }）。P0 不做真实能力协商，fallback 仅为可用性排序。
 * ==========================================================================*/
;(function (NS) {
  'use strict';
  NS.L6 = NS.L6 || {};
  NS.L6.backend = NS.L6.backend || {};

  // name → backendObj。null 原型：避免与 Object.prototype 键名冲突。
  const _backends = Object.create(null);
  let _active = null;
  // 默认精确通道（零回归契约的默认数值路径）。
  let _profile = 'exact';
  const PROFILES = ['exact', 'fast'];

  /**
   * 注册一个后端。
   * @param name        后端名（如 'cpu'）
   * @param backendObj  { name, kernels:{name→fn}, caps:{...}, profile }
   *                    kernels 约定使用 null 原型映射（防原型链键名误命中）。
   * @returns           登记的后端对象
   */
  function registerBackend(name, backendObj) {
    if (typeof name !== 'string' || name.length === 0) {
      throw new Error('[L6] registerBackend: name 必须为非空字符串');
    }
    if (!backendObj || typeof backendObj !== 'object') {
      throw new Error('[L6] registerBackend: backendObj 必须为对象');
    }
    if (!backendObj.kernels || typeof backendObj.kernels !== 'object') {
      throw new Error('[L6] registerBackend: backendObj.kernels 必须为 name→fn 映射对象');
    }
    _backends[name] = backendObj;
    return backendObj;
  }

  /** 切换活跃后端；未知后端抛 '[L6]' 前缀错误（绝不静默回退）。 */
  function setBackend(name) {
    const b = _backends[name];
    if (!b) {
      throw new Error('[L6] setBackend: 未注册的后端 "' + name +
        '"（已注册: ' + Object.keys(_backends).join(', ') + '）');
    }
    _active = b;
    return b;
  }

  /** 当前活跃后端对象（未设置时为 null）。 */
  function getBackend() { return _active; }

  /**
   * 取活跃后端的内核 fn（或 null）。热路径契约：调用方一次性捕获引用后
   * 直接调用，绕过本查表层。
   */
  function kernel(name) {
    if (!_active) return null;
    const fn = _active.kernels[name];
    return typeof fn === 'function' ? fn : null;
  }

  /** 设置数值通道 profile（'exact' | 'fast'）。 */
  function setProfile(p) {
    if (PROFILES.indexOf(p) < 0) {
      throw new Error('[L6] setProfile: 无效 profile "' + p +
        '"（有效: ' + PROFILES.join('|') + '）');
    }
    _profile = p;
    return _profile;
  }

  /** 当前 profile；默认 'exact'。 */
  function getProfile() { return _profile; }

  /**
   * 能力回退：按序取第一个【已注册】的后端并激活、返回之。
   * P0 诚实从简：不做真实能力协商（caps 仅作自省元数据），此处仅为
   * 可用性排序；无一可用时返回 null，由调用方决策（P0 无更多后端可实现）。
   */
  function fallback(order) {
    if (!Array.isArray(order)) {
      throw new Error('[L6] fallback: 参数必须为后端名数组');
    }
    for (let i = 0; i < order.length; i++) {
      const b = _backends[order[i]];
      if (b) { _active = b; return b; }
    }
    return null;
  }

  /**
   * 桥接：刷新所有已注册后端（调用其 refresh()，若提供）。
   * 供 NS.L6.kernel.bootstrap() 在 L4 就绪后联动补登各后端的 L4 内核镜像；
   * 无后端或后端未提供 refresh 时静默跳过。
   */
  function refreshBackends() {
    const names = Object.keys(_backends);
    for (let i = 0; i < names.length; i++) {
      const b = _backends[names[i]];
      if (b && typeof b.refresh === 'function') b.refresh();
    }
  }

  NS.L6.backend.registerBackend = registerBackend;
  NS.L6.backend.setBackend = setBackend;
  NS.L6.backend.getBackend = getBackend;
  NS.L6.backend.kernel = kernel;
  NS.L6.backend.setProfile = setProfile;
  NS.L6.backend.getProfile = getProfile;
  NS.L6.backend.fallback = fallback;
  NS.L6.backend.refreshBackends = refreshBackends;
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

/* ============================================================================
 * L6 后端运行时 · 默认 CPU(JS) 后端
 * ----------------------------------------------------------------------------
 * 所属层：L6-backend
 * 允许依赖的层：L8（求值期已加载，先于 L6）；L4 经 refresh() 惰性镜像
 *               （L4 晚于 L6 加载，故构建期只捕 L8，L4 段留给 refresh）。
 * 主要 @ref：零回归契约 R9（Δ=0）；P0 范围裁定（默认 CPU(JS) exact 单通道）。
 *
 * 责任：定义并注册默认后端 cpu —— P0 的唯一实现，也是默认活跃后端。
 *   - kernels 与 NS.L8 为【同一函数引用】（Object.is 恒等）→ 逐位等价 Δ=0。
 *   - caps 自省：{ simd:false, threads:false, deterministic:true }
 *     （纯 JS 标量循环；无 SIMD/多线程；确定性有 R1/R9 把关）。
 *   - profile:'exact'：P0 唯一实现档。'fast' 档 deferred to P2（届时以新
 *     后端对象接入，而非在本对象内混入数值不同的路径），故当前 fast/exact
 *     两档解析到同一批内核，数值差恒 0 ≤ bound。
 *   - refresh()：L4 就绪后镜像补登 —— 把 NS.L6.kernel 注册表中尚未收录的
 *     内核（L4 段及任何后续登记项）并入本后端；已有条目不覆盖，守护 L8
 *     引用的 Object.is 恒等契约。调用方式：NS.L6.kernel.bootstrap()（内部
 *     经桥接自动刷新各后端）或直接调用本 refresh()。
 * ==========================================================================*/
;(function (NS) {
  'use strict';
  NS.L6 = NS.L6 || {};
  NS.L6.cpu = NS.L6.cpu || {};

  /**
   * 构建内核表：直接捕获 NS.L8 的同一函数引用（Object.is 恒等 → Δ=0）。
   * L8 先于 L6 加载（__loadOrder），求值期可用；仍做缺席守卫以防单独加载。
   * 返回 null 原型映射（防原型链键名误命中）。
   */
  function build() {
    const kernels = Object.create(null);
    const L8 = NS.L8;
    if (L8 && L8.matmul) {
      kernels.matmul = L8.matmul.matmul;
      kernels.matmulAtB = L8.matmul.matmulAtB;
      kernels.matmulTA = L8.matmul.matmulTA;
      kernels.mm = L8.matmul.mm;
      kernels.mmAtB = L8.matmul.mmAtB;
    }
    if (L8 && L8.reduce) {
      kernels.softmaxRows = L8.reduce.softmaxRows;
      kernels.softmaxBwd = L8.reduce.softmaxBwd;
      kernels.addInto = L8.reduce.addInto;
      kernels.rowSumInto = L8.reduce.rowSumInto;
      kernels.topkRows = L8.reduce.topkRows;
    }
    return kernels;
  }

  /**
   * 惰性补登：镜像 NS.L6.kernel 注册表中尚未收录的内核（L4 段 + 后续登记项）。
   * 已有条目不覆盖（守护 L8 引用的 Object.is 恒等契约）。
   * @returns 本次新增补登的内核数量
   */
  function refresh() {
    const K = NS.L6.kernel;
    if (!K || typeof K.list !== 'function' || typeof K.get !== 'function') return 0;
    const names = K.list();
    let added = 0;
    for (let i = 0; i < names.length; i++) {
      const n = names[i];
      if (typeof cpuBackend.kernels[n] !== 'function') {
        const fn = K.get(n);
        if (fn) { cpuBackend.kernels[n] = fn; added++; }
      }
    }
    return added;
  }

  const cpuBackend = {
    name: 'cpu',
    caps: { simd: false, threads: false, deterministic: true },
    kernels: build(),
    profile: 'exact',
    refresh: refresh,
  };

  NS.L6.cpu.backend = cpuBackend;
  NS.L6.cpu.build = build;
  NS.L6.cpu.refresh = refresh;

  // 注册为默认后端并激活；显式落定默认数值通道 'exact'（P0 唯一实现档）。
  NS.L6.backend.registerBackend('cpu', cpuBackend);
  NS.L6.backend.setBackend('cpu');
  NS.L6.backend.setProfile('exact');
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

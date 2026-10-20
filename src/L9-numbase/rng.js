/* ============================================================================
 * L9 数值基础 · 确定性伪随机（rng）
 * ----------------------------------------------------------------------------
 * 所属层：L9-numBase（数值地基）
 * 允许依赖的层：无
 * 主要 @ref：mulberry32 (Christopher & Salmon 2011, "Fast Random Number
 *           Generation and Applications")；Box–Muller (Box & Muller 1958)
 *
 * 责任：提供与 v11 逐位一致的确定性 PRNG。任何需要随机的函数必须以注入的
 *       `rng` 为唯一熵源（共享约定 K6）。本文件的函数体逐字沿用 v11，保证
 *       R1（参数 Δ=0）契约不被破坏。
 * ==========================================================================*/
;(function (NS) {
  'use strict';
  NS.L9 = NS.L9 || {};
  NS.L9.rng = NS.L9.rng || {};

  // 确定性伪随机数发生器（mulberry32）：同种子 => 完全可复现
  // @invariant mulberry32(seed)() 在任意平台/Node 版本下严格逐位一致。
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // Box-Muller 标准正态采样
  // @invariant randn(rng) 的采样序列与 v11 逐位一致（依赖 mulberry32 顺序）。
  function randn(rng) {
    let u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  NS.L9.rng.mulberry32 = mulberry32;
  NS.L9.rng.randn = randn;
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

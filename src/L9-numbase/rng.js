/* L9 · 数值基础层 / 确定性随机源 —— 依赖：无（最底层之一，与 L10 同批加载）
 * 一切随机必须来自显式注入的 rng（K6 红线）；禁止全局随机源。
 * @ref JAX rngSplit · NumPy PCG64 命名 · 确定性重放协议
 */
;(function (NS) {
  'use strict';
  // 所属层：L9 · 允许依赖：L10（μH rng 由本层提供，供上层复用）· @ref JAX rngSplit / NumPy

  // mulberry32：32-bit 确定性 PRNG，同种子逐位可复现（v11 沿用）
  function mulberry32(seed) {
    var s = (seed >>> 0) || 1;
    return function () {
      s = (s + 0x6D2B79F5) | 0;
      var t = Math.imul(s ^ (s >>> 15), 1 | s);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // 标准正态（Box-Muller）；rng 显式注入，确定性
  function randn(rng) {
    var u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  // JAX 式 key 派生：从父流确定性派生 n 个独立子流（用于可重放训练协议）
  function rngSplit(stream, n) {
    var out = [];
    for (var i = 0; i < n; i++) {
      var base = (Math.floor(stream() * 4294967296) ^ ((i + 1) * 0x9E3779B1)) >>> 0;
      out.push(mulberry32(base));
    }
    return out;
  }

  NS.L9 = Object.assign(NS.L9 || {}, {
    mulberry32: mulberry32, randn: randn, rngSplit: rngSplit
  });
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

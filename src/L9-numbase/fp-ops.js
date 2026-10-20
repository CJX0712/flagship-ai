/* ============================================================================
 * L9 数值基础 · 浮点原语（fp-ops）
 * ----------------------------------------------------------------------------
 * 所属层：L9-numBase
 * 允许依赖的层：无
 * 主要 @ref：IEEE 754-2019；Dekker (1971) "A floating-point technique for
 *           extending the available precision"；Goldberg (1991) "What Every
 *           Computer Scientist Should Know About Floating-Point Arithmetic"
 *
 * 责任：提供数值稳定的浮点基元与常量。所有函数均为纯函数、确定性、无副作用，
 *       供 L8 内核与 L10 微指令复用（零重写：直接站在 IEEE 754 与 Dekker 之上）。
 * ==========================================================================*/
;(function (NS) {
  'use strict';
  NS.L9 = NS.L9 || {};
  NS.L9.fpOps = NS.L9.fpOps || {};

  const EPS = Number.EPSILON;            // 2.220446049250313e-16
  const FINITE_MAX = Number.MAX_VALUE;

  // 符号函数：返回 -1（负）/ 0（±0）/ 1（正）。NaN 返回 NaN。
  // @invariant sign(-x) === -sign(x)；sign(0)===0；sign(-0)===0。
  function sign(x) { return x > 0 ? 1 : x < 0 ? -1 : x; }

  // 复制 b 的符号到 |a|（IEEE copysign 语义）。
  // @invariant copysign(a, -0) 末位为负；copysign(a, 1) 末位非负。
  function copysign(a, b) { return b < 0 || (b === 0 && 1 / b < 0) ? -Math.abs(a) : Math.abs(a); }

  // 钳位到 [lo, hi]。
  // @invariant lo<=hi ⇒ lo <= clamp(x,lo,hi) <= hi；clamp(x,x,x)===x。
  function clamp(x, lo, hi) { return x < lo ? lo : x > hi ? hi : x; }

  // 稳定 log(1+x)：|x| 很小时相对误差远小于 log(1+x) 朴素实现。
  // @ref Cody & Waite (1980) 初等函数算法。
  function log1p(x) { return Math.log1p(x); }

  // 稳定 exp(x)-1。
  function expm1(x) { return Math.expm1(x); }

  // 2-范数（hypot）避免中间溢出。
  function hypot(a, b) { return Math.hypot(a, b); }

  // 倒数平方根（Newton 一步可迭代，但单步 Math 已足够精确）。
  function rsqrt(x) { return 1 / Math.sqrt(x); }

  // Dekker 两数和：返回 [hi, lo]，hi+lo 无舍入误差（当 |a|,|b| 不溢出）。
  // @ref Dekker 1971。@complexity O(1)。
  function twoSum(a, b) {
    const s = a + b;
    const a1 = s - b;
    const b1 = s - a1;
    const da = a - a1, db = b - b1;
    return [s, da + db];
  }

  // 有限性 / NaN 判定（朴素封装，统一 API）。
  const isFiniteNum = Number.isFinite;
  const isNaNNum = Number.isNaN;

  NS.L9.fpOps.EPS = EPS;
  NS.L9.fpOps.FINITE_MAX = FINITE_MAX;
  NS.L9.fpOps.sign = sign;
  NS.L9.fpOps.copysign = copysign;
  NS.L9.fpOps.clamp = clamp;
  NS.L9.fpOps.log1p = log1p;
  NS.L9.fpOps.expm1 = expm1;
  NS.L9.fpOps.hypot = hypot;
  NS.L9.fpOps.rsqrt = rsqrt;
  NS.L9.fpOps.twoSum = twoSum;
  NS.L9.fpOps.isFiniteNum = isFiniteNum;
  NS.L9.fpOps.isNaNNum = isNaNNum;
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

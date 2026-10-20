/* L9 · 数值基础层 / 断言与误差预算 —— 依赖：无
 * NaN/Inf 哨兵、逐位/容差断言、误差预算表（K9/K10 红线）。
 * @ref tfjs 断言风格 · Karpathy 梯度检验规范
 */
;(function (NS) {
  'use strict';
  // 所属层：L9 · 允许依赖：无 · 主要 @ref：自动微分断言惯例

  function assert(cond, msg) {
    if (!cond) throw new Error('[L9.assert] ' + (msg || '断言失败'));
  }

  // 容差断言：|a-b| <= tol
  function assertClose(a, b, tol, msg) {
    assert(isFinite(a) && isFinite(b), (msg || '') + ' 非有限');
    assert(Math.abs(a - b) <= tol, (msg || '') + ' |a-b|=' + Math.abs(a - b) + ' > tol=' + tol);
  }

  // 逐位断言：数组或标量 Object.is 全等
  function assertBitExact(a, b, msg) {
    if (typeof a === 'number' && typeof b === 'number') {
      assert(Object.is(a, b), (msg || '') + ' ' + a + ' !== ' + b);
      return;
    }
    assert(a && b && a.length === b.length, (msg || '') + ' 长度不等');
    for (var i = 0; i < a.length; i++) assert(Object.is(a[i], b[i]), (msg || '') + ' @' + i + ' ' + a[i] + ' !== ' + b[i]);
  }

  // NaN/Inf 哨兵：debug 开启时抛错，生产降级为 warn（K12）
  function sentinel(arr, ctx, debug) {
    for (var i = 0; i < arr.length; i++) {
      if (!isFinite(arr[i])) {
        if (debug) throw new Error('[L9.sentinel] ' + (ctx || '') + ' 出现非有限 @' + i);
        if (typeof console !== 'undefined') console.warn('[L9.sentinel] ' + (ctx || '') + ' 非有限 @' + i);
        return false;
      }
    }
    return true;
  }

  // 误差预算表（集中登记理论界，测试引用而非魔法数）
  var BUDGET = {
    softmaxRowSum: 1e-14, ropeRelInv: 1e-12, qkNorm: 1e-12, kvDelta: 0, quantLogits: 0.03
  };
  function budget(name) { return BUDGET[name]; }

  NS.L9 = Object.assign(NS.L9 || {}, {
    assert: assert, assertClose: assertClose, assertBitExact: assertBitExact,
    sentinel: sentinel, budget: budget, BUDGET: BUDGET
  });
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

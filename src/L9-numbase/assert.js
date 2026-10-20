/* ============================================================================
 * L9 数值基础 · 断言与哨兵（assert）
 * ----------------------------------------------------------------------------
 * 所属层：L9-numBase
 * 允许依赖的层：无
 * 主要 @ref：K9 共享约定（逐位比对 / 误差预算 / NaN·Inf 哨兵）
 *
 * 责任：提供全工程统一的断言工具。debug 模式下抛出带层号前缀的精确报错；
 *       生产（demo）路径降级为 console.warn 不崩 UI（K12）。所有比较均为
 *       纯函数，无副作用。
 * ==========================================================================*/
;(function (NS) {
  'use strict';
  NS.L9 = NS.L9 || {};
  NS.L9.assert = NS.L9.assert || {};

  const DEBUG = (typeof process !== 'undefined' && process.env && process.env.FLAGSHIP_DEBUG === '1');

  // 通用断言：cond 为假时抛 [layer] 前缀错误。
  function assert(cond, msg, layer) {
    if (!cond) {
      const m = `[${layer || 'L9'}] 断言失败: ${msg || ''}`;
      if (DEBUG) throw new Error(m);
      else { console.warn(m); }
    }
    return cond;
  }

  // 容差近似：|a-b| <= tol 或相对误差 <= tol（针对 0 安全）。
  // @invariant assertClose(a,b,0) 等价于 Object.is。返回布尔。
  function assertClose(a, b, tol, msg, layer) {
    tol = tol == null ? 1e-9 : tol;
    let ok;
    if (typeof a === 'number' && typeof b === 'number') {
      ok = Object.is(a, b) || Math.abs(a - b) <= tol;
    } else if (a && b && a.length === b.length) {
      ok = true;
      for (let i = 0; i < a.length; i++) {
        if (!(Object.is(a[i], b[i]) || Math.abs(a[i] - b[i]) <= tol)) { ok = false; break; }
      }
    } else ok = false;
    if (!ok) {
      const m = `[${layer || 'L9'}] 近似比对失败 (tol=${tol}): ${msg || ''}`;
      if (DEBUG) throw new Error(m); else console.warn(m);
    }
    return ok;
  }

  // 逐位全等：逐元素 Object.is，任意偏差即返回 false（不抛）。
  // @invariant assertBitExact(a,b) === true ⟺ 所有元素 Object.is 相等。
  function assertBitExact(a, b, msg, layer) {
    let ok;
    if (typeof a === 'number' && typeof b === 'number') ok = Object.is(a, b);
    else if (a && b && a.length === b.length) {
      ok = true;
      for (let i = 0; i < a.length; i++) if (!Object.is(a[i], b[i])) { ok = false; break; }
    } else ok = false;
    if (!ok && DEBUG) {
      // 仅 debug 时定位首个偏差索引（生产降级不抛）
      let idx = -1;
      if (a && b && a.length === b.length) { for (let i = 0; i < a.length; i++) if (!Object.is(a[i], b[i])) { idx = i; break; } }
      throw new Error(`[${layer || 'L9'}] 逐位比对失败 @idx=${idx} a=${a && a[idx]} b=${b && b[idx]} ${msg || ''}`);
    }
    return ok;
  }

  // NaN / Inf 哨兵：debug 模式下发现非有限值即抛；返回布尔。
  function sentinel(arr, ctx, layer) {
    if (!arr) return true;
    let bad = -1;
    for (let i = 0; i < arr.length; i++) {
      if (!Number.isFinite(arr[i])) { bad = i; break; }
    }
    const ok = bad < 0;
    if (!ok) {
      const m = `[${layer || 'L9'}] 非有限哨兵触发 @idx=${bad} (${ctx || ''})`;
      if (DEBUG) throw new Error(m); else console.warn(m);
    }
    return ok;
  }

  NS.L9.assert.assert = assert;
  NS.L9.assert.assertClose = assertClose;
  NS.L9.assert.assertBitExact = assertBitExact;
  NS.L9.assert.sentinel = sentinel;
  NS.L9.assert.DEBUG = DEBUG;
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

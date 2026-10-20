/* ============================================================================
 * L7 张量与内存 · 形状运算（shape）
 * ----------------------------------------------------------------------------
 * 所属层：L7-tensor（张量与内存）
 * 允许依赖的层：无（纯形状代数；跨层一律经 NS.Ln.x 惰性引用）
 *
 * 责任：提供 ndarray 风格张量的形状/步长代数原语：
 *   - contiguousStrides：行主序（row-major）连续步长，末维恒为 1；
 *   - size：维度乘积（元素数）；
 *   - linearize：多维下标 + 步长 → 扁平偏移（offset + Σ idx[i]·stride[i]）；
 *   - broadcastShapes：numpy 风格尾部对齐广播，不匹配即抛；
 *   - permute / invertPerm：轴置换与其逆置换。
 * 存储序（K1）：行主序，与 L8 数值内核的 Float64Array 扁平布局一致。
 * ==========================================================================*/
;(function (NS) {
  'use strict';
  NS.L7 = NS.L7 || {};
  NS.L7.shape = NS.L7.shape || {};

  // 维度校验：必须是非负整数（0 维 = 标量；某维为 0 = 空张量）。
  function checkDim(d, ctx) {
    if (!Number.isInteger(d) || d < 0) {
      throw new Error(`[L7.shape] ${ctx} 必须是非负整数，收到: ${String(d)}`);
    }
    return d;
  }

  // 行主序连续步长：末维 stride 恒为 1，向前逐维累积。
  // @invariant contiguousStrides([r,c]) === [c,1]；contiguousStrides([]) === []。
  function contiguousStrides(shape) {
    const n = shape.length;
    const out = new Array(n);
    let acc = 1;
    for (let i = n - 1; i >= 0; i--) {
      out[i] = acc;
      acc *= checkDim(shape[i], 'shape 维度');
    }
    return out;
  }

  // 元素总数 = 维度乘积；空 shape（标量）为 1；含 0 维为 0。
  function size(shape) {
    let n = 1;
    for (let i = 0; i < shape.length; i++) n *= checkDim(shape[i], 'shape 维度');
    return n;
  }

  // 扁平偏移 = offset + Σ index[i]·stride[i]。
  // stride 省略时按行主序连续步长计算；offset 省略时为 0。
  // 下标个数必须与维数一致，且逐维做边界检查（越界即抛）。
  function linearize(index, shape, stride, offset) {
    if (!Array.isArray(index)) throw new Error('[L7.shape] index 必须是数组');
    if (index.length !== shape.length) {
      throw new Error(`[L7.shape] 下标个数 ${index.length} 与维数 ${shape.length} 不一致`);
    }
    const st = stride || contiguousStrides(shape);
    let off = offset == null ? 0 : offset;
    for (let i = 0; i < index.length; i++) {
      const k = index[i];
      if (!Number.isInteger(k) || k < 0 || k >= shape[i]) {
        throw new Error(`[L7.shape] 下标越界: dim ${i} 下标 ${String(k)} 范围 [0,${shape[i]})`);
      }
      off += k * st[i];
    }
    return off;
  }

  // numpy 风格广播：尾部对齐；两维相等或其一为 1 时取较大者，否则抛。
  // 任一侧缺省的维度按 1 补齐（前置补 1）。
  function broadcastShapes(a, b) {
    const ra = a.length, rb = b.length;
    const n = Math.max(ra, rb);
    const out = new Array(n);
    for (let i = 0; i < n; i++) {
      const da = i < ra ? checkDim(a[ra - 1 - i], 'shape 维度') : 1;
      const db = i < rb ? checkDim(b[rb - 1 - i], 'shape 维度') : 1;
      let d;
      if (da === db) d = da;
      else if (da === 1) d = db;
      else if (db === 1) d = da;
      else throw new Error(`[L7.shape] 广播失败: 维度 ${da} 与 ${db} 不兼容（尾部对齐第 ${n - i} 维）`);
      out[n - 1 - i] = d;
    }
    return out;
  }

  // 轴置换：axes 必须是 0..ndim-1 的一个排列（permutation）。
  // 返回新的 {shape, stride}，原数组不被修改。
  function permute(shape, stride, axes) {
    const n = shape.length;
    if (!Array.isArray(axes) || axes.length !== n) {
      throw new Error(`[L7.shape] axes 必须是 0..${n - 1} 的排列，收到: ${JSON.stringify(axes)}`);
    }
    const seen = new Array(n).fill(false);
    for (let i = 0; i < n; i++) {
      const ax = axes[i];
      if (!Number.isInteger(ax) || ax < 0 || ax >= n || seen[ax]) {
        throw new Error(`[L7.shape] axes 含非法或重复轴: ${String(ax)}`);
      }
      seen[ax] = true;
    }
    const ns = new Array(n), nt = new Array(n);
    for (let i = 0; i < n; i++) { ns[i] = shape[axes[i]]; nt[i] = stride[axes[i]]; }
    return { shape: ns, stride: nt };
  }

  // 逆置换：inv[axes[i]] = i。
  // @invariant invertPerm(invertPerm(axes)) 逐元素等于 axes。
  function invertPerm(axes) {
    const inv = new Array(axes.length);
    for (let i = 0; i < axes.length; i++) {
      if (!Number.isInteger(axes[i]) || axes[i] < 0 || axes[i] >= axes.length) {
        throw new Error(`[L7.shape] axes 含非法轴: ${String(axes[i])}`);
      }
      inv[axes[i]] = i;
    }
    return inv;
  }

  NS.L7.shape.checkDim = checkDim;
  NS.L7.shape.contiguousStrides = contiguousStrides;
  NS.L7.shape.size = size;
  NS.L7.shape.linearize = linearize;
  NS.L7.shape.broadcastShapes = broadcastShapes;
  NS.L7.shape.permute = permute;
  NS.L7.shape.invertPerm = invertPerm;
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

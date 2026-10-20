/* ============================================================================
 * L7 张量与内存 · 竞技场分配器（arena）
 * ----------------------------------------------------------------------------
 * 所属层：L7-tensor（张量与内存）
 * 允许依赖的层：无跨层硬依赖（shape/tensor 经 NS.L7.x 惰性引用，
 *   二者按 __loadOrder 均先于本文件求值）。
 *
 * 责任：bump（指针碰撞）分配器。单一 ArrayBuffer 后备存储，alloc 按 8 字节
 * 对齐切出 Float64Array 子视图（零拷贝，不复制数据）；reset 仅把用量指针
 * 归零（不释放缓冲），便于逐请求/逐批次复用同一块内存。
 * ==========================================================================*/
;(function (NS) {
  'use strict';
  NS.L7 = NS.L7 || {};
  NS.L7.arena = NS.L7.arena || {};

  function align8(x) { return (x + 7) & ~7; }

  // 创建容量为 byteCapacity 字节的竞技场。
  // 返回 { alloc(shape, dtype), reset(), usedBytes(), capacityBytes() }。
  function arena(byteCapacity) {
    if (!Number.isInteger(byteCapacity) || byteCapacity < 0) {
      throw new Error(`[L7.arena] 容量必须是非负整数，收到: ${String(byteCapacity)}`);
    }
    const buf = new ArrayBuffer(byteCapacity);
    const f64 = new Float64Array(buf); // 唯一后备存储；subarray 视图共享它
    let used = 0; // 碰撞指针（字节，已对齐）

    return {
      // 从竞技场切出一块连续 Float64Array 区域（零拷贝子视图，offset 0）。
      // 仅支持 dtype 'float64'；容量不足即抛（不静默回退）。
      alloc(shape, dtype) {
        if (dtype != null && dtype !== 'float64') {
          throw new Error(`[L7.arena] 仅支持 dtype 'float64'，收到: ${String(dtype)}`);
        }
        if (!Array.isArray(shape)) throw new Error('[L7.arena] alloc 需要 shape 数组');
        const n = NS.L7.shape.size(shape);
        const bytes = n * 8;
        const start = align8(used);
        if (start + bytes > byteCapacity) {
          throw new Error(`[L7.arena] 容量不足: 需要 ${start + bytes} 字节，容量 ${byteCapacity} 字节`);
        }
        const data = f64.subarray(start / 8, start / 8 + n);
        used = start + bytes;
        return NS.L7.tensor.make({ data, shape, dtype: 'float64' });
      },
      // 用量指针归零（保留 ArrayBuffer，可立即重新分配）。
      reset() { used = 0; },
      // 已发出的字节数（含对齐填充；float64 分配下无填充）。
      usedBytes() { return used; },
      // 竞技场总容量（字节）。
      capacityBytes() { return byteCapacity; },
    };
  }

  NS.L7.arena.arena = arena;
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

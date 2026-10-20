/* ============================================================================
 * L7 张量与内存 · 张量（tensor）
 * ----------------------------------------------------------------------------
 * 所属层：L7-tensor（张量与内存）
 * 允许依赖的层：无跨层硬依赖（形状运算经同层 NS.L7.shape 惰性引用）。
 * 注意加载顺序（K4）：本文件先于 shape.js 求值，故对 NS.L7.shape 的引用
 * 一律写在函数体内（调用时 shape.js 已加载完成），不在模块顶层捕获。
 *
 * 责任：ndarray 风格张量。张量是普通对象 { data, shape, stride, offset, dtype }
 * （dtype 默认 'float64'，data 为 Float64Array）。视图运算（reshape /
 *  transpose / slice）零拷贝：视图与母张量共享同一 data 缓冲；
 * 仅 alloc / contiguous / fromArray2d 会分配新缓冲。
 * 存储序（K1）：行主序；偏移 = offset + Σ index[i]·stride[i]。
 * ==========================================================================*/
;(function (NS) {
  'use strict';
  NS.L7 = NS.L7 || {};
  NS.L7.tensor = NS.L7.tensor || {};

  const DEFAULT_DTYPE = 'float64';

  // 形状规范化：返回非负整数维度数组的拷贝（不持有调用方数组的别名）。
  function normalizeShape(shape) {
    if (!Array.isArray(shape)) throw new Error('[L7.tensor] shape 必须是数组');
    const out = new Array(shape.length);
    for (let i = 0; i < shape.length; i++) {
      out[i] = NS.L7.shape.checkDim(shape[i], 'shape 维度');
    }
    return out;
  }

  function checkDtype(dtype) {
    if (dtype !== DEFAULT_DTYPE) {
      throw new Error(`[L7.tensor] 仅支持 dtype '${DEFAULT_DTYPE}'，收到: ${String(dtype)}`);
    }
    return dtype;
  }

  // 是否行主序连续（stride 与形状的连续步长逐一相等）。
  // 尺寸为 1 的维不影响视图合法性判定之外的任何语义；这里用严格相等，
  // 因为本层产生的 stride 均为规范连续步长。
  function isContiguous(t) {
    const cs = NS.L7.shape.contiguousStrides(t.shape);
    for (let i = 0; i < cs.length; i++) if (t.stride[i] !== cs[i]) return false;
    return true;
  }

  // 原始构造器：{data, shape, stride, offset, dtype}。
  // - data 省略 → 按 shape 新分配 Float64Array（零填充）；
  // - stride 省略 → 行主序连续步长；offset 省略 → 0；dtype 省略 → 'float64'。
  function make(spec) {
    spec = spec || {};
    const shape = normalizeShape(spec.shape || []);
    const dtype = checkDtype(spec.dtype || DEFAULT_DTYPE);
    let data = spec.data;
    if (data == null) data = new Float64Array(NS.L7.shape.size(shape));
    const stride = spec.stride ? spec.stride.slice() : NS.L7.shape.contiguousStrides(shape);
    const offset = spec.offset == null ? 0 : spec.offset;
    return { data, shape, stride, offset, dtype };
  }

  // 新建连续张量：全新 Float64Array，offset 0，行主序步长。
  function alloc(shape, dtype) {
    return make({ data: new Float64Array(NS.L7.shape.size(normalizeShape(shape))), shape: normalizeShape(shape), dtype: dtype || DEFAULT_DTYPE });
  }

  // 零拷贝包装：result.data === data（身份恒等），连续步长，offset 0。
  function fromFlat(data, shape) {
    const s = normalizeShape(shape);
    if (!data || typeof data.length !== 'number') {
      throw new Error('[L7.tensor] fromFlat 需要类数组 data（Float64Array）');
    }
    if (data.length !== NS.L7.shape.size(s)) {
      throw new Error(`[L7.tensor] fromFlat 长度不匹配: data ${data.length} ≠ shape 乘积 ${NS.L7.shape.size(s)}`);
    }
    return { data, shape: s, stride: NS.L7.shape.contiguousStrides(s), offset: 0, dtype: DEFAULT_DTYPE };
  }

  // 全零张量（Float64Array 天然零填充，等同 alloc）。
  function zeros(shape) { return alloc(shape); }

  // 二维嵌套数组 → 连续张量（拷贝，不与入参共享缓冲）。
  // 非规则矩形（行宽不一）即抛；空数组 → shape [0]。
  function fromArray2d(nested) {
    if (!Array.isArray(nested)) throw new Error('[L7.tensor] fromArray2d 需要二维数组');
    const rows = nested.length;
    if (rows === 0) return alloc([0]);
    const first = nested[0];
    if (!Array.isArray(first)) throw new Error('[L7.tensor] fromArray2d 需要二维数组（行必须是数组）');
    const cols = first.length;
    const data = new Float64Array(rows * cols);
    for (let i = 0; i < rows; i++) {
      const row = nested[i];
      if (!Array.isArray(row) || row.length !== cols) {
        throw new Error(`[L7.tensor] fromArray2d 非规则矩形: 第 ${i} 行宽度 ${row && row.length} ≠ ${cols}`);
      }
      for (let j = 0; j < cols; j++) {
        const v = row[j];
        if (typeof v !== 'number') throw new Error(`[L7.tensor] fromArray2d 元素必须是 number: [${i}][${j}]`);
        data[i * cols + j] = v;
      }
    }
    return { data, shape: [rows, cols], stride: [cols, 1], offset: 0, dtype: DEFAULT_DTYPE };
  }

  // 读取元素：t.get(i0, i1, ...) = data[offset + Σ idx·stride]。
  function get(t, ...idx) {
    return t.data[NS.L7.shape.linearize(idx, t.shape, t.stride, t.offset)];
  }

  // 写入元素：t.set(v, i0, i1, ...)（写穿视图直达共享缓冲）。
  function set(t, v, ...idx) {
    t.data[NS.L7.shape.linearize(idx, t.shape, t.stride, t.offset)] = v;
    return t;
  }

  // 变形（零拷贝）：总元素数一致且当前连续时，返回共享 data 的视图；
  // 支持 -1 自动推断某一维；否则抛。非连续张量（如转置视图）不可零拷贝变形。
  function reshape(t, ...dims) {
    let shape;
    if (dims.length === 1 && Array.isArray(dims[0])) shape = dims[0].slice();
    else shape = dims;
    const total = NS.L7.shape.size(t.shape);
    let known = 1, infer = -1;
    for (let i = 0; i < shape.length; i++) {
      const d = shape[i];
      if (d === -1) {
        if (infer >= 0) throw new Error('[L7.tensor] reshape 只允许一个 -1');
        infer = i;
      } else {
        if (!Number.isInteger(d) || d < 0) throw new Error(`[L7.tensor] reshape 维度必须是非负整数: ${String(d)}`);
        known *= d;
      }
    }
    if (infer >= 0) {
      if (known !== 0 && total % known === 0) shape[infer] = total / known;
      else if (total === 0) shape[infer] = 0;
      else throw new Error(`[L7.tensor] reshape -1 无法推断: total=${total} known=${known}`);
    } else if (known !== total) {
      throw new Error(`[L7.tensor] reshape 总大小不匹配: ${known} ≠ ${total}`);
    }
    if (!isContiguous(t)) throw new Error('[L7.tensor] reshape 仅支持连续张量（零拷贝契约）');
    return make({ data: t.data, shape, offset: t.offset, dtype: t.dtype });
  }

  // 转置（零拷贝）：交换前两维的 shape/stride，offset 不变（ndim ≥ 2）。
  function transpose(t) {
    if (t.shape.length < 2) throw new Error('[L7.tensor] transpose 需要 ndim ≥ 2');
    const shape = t.shape.slice(), stride = t.stride.slice();
    const ts = shape[0]; shape[0] = shape[1]; shape[1] = ts;
    const tt = stride[0]; stride[0] = stride[1]; stride[1] = tt;
    return { data: t.data, shape, stride, offset: t.offset, dtype: t.dtype };
  }

  // 物化连续副本：新 Float64Array（!== 原 data），行主序收集（经步长）。
  function contiguous(t) {
    const n = NS.L7.shape.size(t.shape);
    const out = new Float64Array(n);
    if (isContiguous(t)) {
      for (let i = 0; i < n; i++) out[i] = t.data[t.offset + i];
    } else {
      const idx = new Array(t.shape.length).fill(0);
      for (let i = 0; i < n; i++) {
        let off = t.offset;
        for (let d = 0; d < idx.length; d++) off += idx[d] * t.stride[d];
        out[i] = t.data[off];
        for (let d = idx.length - 1; d >= 0; d--) {
          if (++idx[d] < t.shape[d]) break;
          idx[d] = 0;
        }
      }
    }
    return make({ data: out, shape: t.shape.slice(), dtype: t.dtype });
  }

  // 切片视图（零拷贝）：调整 offset/shape，stride 不变；支持负索引。
  function slice(t, axis, start, end) {
    const nd = t.shape.length;
    if (!Number.isInteger(axis) || axis < 0 || axis >= nd) {
      throw new Error(`[L7.tensor] slice 轴非法: ${String(axis)}（ndim=${nd}）`);
    }
    const len = t.shape[axis];
    let s = start == null ? 0 : start;
    let e = end == null ? len : end;
    if (s < 0) s += len;
    if (e < 0) e += len;
    if (s < 0 || e > len || s > e) {
      throw new Error(`[L7.tensor] slice 越界: [${start},${end}) 不在 [0,${len})（轴 ${axis}）`);
    }
    const shape = t.shape.slice();
    shape[axis] = e - s;
    return { data: t.data, shape, stride: t.stride.slice(), offset: t.offset + s * t.stride[axis], dtype: t.dtype };
  }

  // 元素总数 / 维数（实例无关辅助）。
  function size(t) { return NS.L7.shape.size(t.shape); }
  function ndim(t) { return t.shape.length; }

  NS.L7.tensor.make = make;
  NS.L7.tensor.alloc = alloc;
  NS.L7.tensor.fromFlat = fromFlat;
  NS.L7.tensor.zeros = zeros;
  NS.L7.tensor.fromArray2d = fromArray2d;
  NS.L7.tensor.get = get;
  NS.L7.tensor.set = set;
  NS.L7.tensor.reshape = reshape;
  NS.L7.tensor.transpose = transpose;
  NS.L7.tensor.contiguous = contiguous;
  NS.L7.tensor.slice = slice;
  NS.L7.tensor.size = size;
  NS.L7.tensor.ndim = ndim;
  NS.L7.tensor.isContiguous = isContiguous;
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

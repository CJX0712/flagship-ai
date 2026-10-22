/* ============================================================================
 * L10 微指令 · E 域（形状与索引）· 第 4/4 批（E097–E128，T07 补齐）
 * ----------------------------------------------------------------------------
 * 所属层：L10-microIsa  域：E（形状与索引）
 * 允许依赖的层：L7（shape 代数）、L9（rng）、L10（registry/invariant）
 * 主要 @ref：numpy ndarray strides；numpy.squeeze/expand_dims/broadcast_to/
 *           flatten/reshape/moveaxis；numpy 负下标与切片视图语义；
 *           unravel_index（混合 radix 分解）；置换群共轭类
 *
 * 每项含七件套（直觉/数学/伪代码/复杂度/论文/常见坑/不变量）+ run + predicate。
 * 判据：整数与结构精确，一律 === 与逐元素深比（eqArr）；无数值容差路径。
 * 结构：E097–E108 形状变换（squeeze 结构/幂等、unsqueeze 指定轴、
 *       expand 广播复制与 0 步长视图、flatten 任意轴区间、split 互逆、
 *       moveaxis 归约 permute）；E109–E120 索引高级（ravel/unravel 往返、
 *       sub2ind/ind2sub 2D 往返与越界抛、负索引归一、slice 偏移/长度/分段
 *       公式、翻转轴镜像恒等与对合、contiguous 后缀积判定）；
 *       E121–E128 形状代数进阶（三形状结合律、共轭置换、广播-reshape
 *       复合、空维指定轴往返、形状相等判定、展平-置换可交换、flatten
 *       分步结合、广播同伙伴幂等）。
 * 与已落地项的分界：E001–E096 已覆盖连续步长/size/linearize/广播全套/
 *       置换全套/重塑 size 层（含滤 1 保 size、合并尾维保 size、合并拆分
 *       size 互证）——本批一律升级到「结构结果 / 一般化区间 / 视图步长 /
 *       真正互逆往返」层，逐项在 commonPitfall 注明与最近邻项的差异。
 * 加载顺序注意：本文件先于 L7 求值，NS.L7.shape 一律惰性引用（仅在
 *       run/predicate/助手函数体内求值）。
 * ==========================================================================*/
;(function (NS) {
  'use strict';
  NS.L10 = NS.L10 || {};
  const rc=(a,b,t)=>NS.L10.relClose(a,b,t);
  function reg(i,o){ NS.L10.register('E', i, o); }

  // ── 本地第一性原理助手 ──────────────────────────────────────────────
  function eqArr(a, b) {
    if (!a || !b || a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
    return true;
  }
  function prodRange(a, lo, hi) { let n = 1; for (let i = lo; i <= hi; i++) n *= a[i]; return n; }
  function zerosOf(n) { const a = new Array(n); for (let i = 0; i < n; i++) a[i] = 0; return a; }
  function midOf(shape) { const a = new Array(shape.length); for (let i = 0; i < shape.length; i++) a[i] = shape[i] >> 1; return a; }
  function identAxes(n) { const a = new Array(n); for (let i = 0; i < n; i++) a[i] = i; return a; }
  function reverseAxes(n) { const a = new Array(n); for (let i = 0; i < n; i++) a[i] = n - 1 - i; return a; }
  function rotateAxes(n) { const a = new Array(n); for (let i = 0; i < n; i++) a[i] = (i + 1) % n; return a; }
  // 后缀积：suffix[i] = ∏_{j>i} s[j]（末位为空积 1）——与 L7 连续步长同构（E065）。
  function suffixProds(shape) {
    const n = shape.length; const out = new Array(n); let acc = 1;
    for (let i = n - 1; i >= 0; i--) { out[i] = acc; acc *= shape[i]; }
    return out;
  }
  function throws(fn) { try { fn(); return false; } catch (e) { return true; } }

  // ── 形状变换助手 ────────────────────────────────────────────────────
  // squeeze：去掉所有长度为 1 的维（numpy.squeeze 全轴版）。
  function squeezeAll(shape) { return shape.filter((d) => d > 1); }
  // unsqueeze：在指定轴位置插入长度 1 的维（numpy.expand_dims，a ∈ [0,n]）。
  function unsqueezeAt(shape, axis) {
    if (!Number.isInteger(axis) || axis < 0 || axis > shape.length) {
      throw new Error('[E] unsqueeze 轴非法: ' + String(axis));
    }
    const out = shape.slice(0, axis);
    out.push(1);
    for (let i = axis; i < shape.length; i++) out.push(shape[i]);
    return out;
  }
  // flatten：把闭区间 [a,b] 的轴合并为一维（numpy 轴区间展平的一般化）。
  function flattenRange(shape, a, b) {
    const n = shape.length;
    if (!Number.isInteger(a) || !Number.isInteger(b) || a < 0 || a > b || b >= n) {
      throw new Error('[E] flatten 轴区间非法: [' + String(a) + ',' + String(b) + ']');
    }
    const out = shape.slice(0, a);
    out.push(prodRange(shape, a, b));
    for (let i = b + 1; i < n; i++) out.push(shape[i]);
    return out;
  }
  // split：flatten 的逆——把 axis 轴按 dims 拆开（维度之积必须等于该轴）。
  function splitAxis(shape, axis, dims) {
    const n = shape.length;
    if (!Number.isInteger(axis) || axis < 0 || axis >= n) {
      throw new Error('[E] split 轴非法: ' + String(axis));
    }
    if (prodRange(dims, 0, dims.length - 1) !== shape[axis]) {
      throw new Error('[E] split 维度之积与目标轴大小不等');
    }
    const out = shape.slice(0, axis);
    for (let i = 0; i < dims.length; i++) out.push(dims[i]);
    for (let i = axis + 1; i < n; i++) out.push(shape[i]);
    return out;
  }
  // moveaxis 的置换向量：把 src 轴搬到 dst 位置（numpy.moveaxis 的置换刻画）。
  function moveAxesOf(n, src, dst) {
    if (!Number.isInteger(src) || !Number.isInteger(dst) || src < 0 || src >= n || dst < 0 || dst >= n) {
      throw new Error('[E] moveaxis 轴非法: ' + String(src) + '→' + String(dst));
    }
    const axes = [];
    for (let i = 0; i < n; i++) if (i !== src) axes.push(i);
    axes.splice(dst, 0, src);
    return axes;
  }
  function moveAxis(shape, src, dst) {
    const axes = moveAxesOf(shape.length, src, dst);
    const out = new Array(shape.length);
    for (let i = 0; i < axes.length; i++) out[i] = shape[axes[i]];
    return out;
  }
  // 广播视图步长：源为 1 且目标 >1 的被扩展轴步长为 0（单元素复制语义），
  // 其余轴沿用源连续步长。前提：src 与 dst 同长（尾对齐后）。
  function bcastStrides(src, dst) {
    const st = NS.L7.shape.contiguousStrides(src);
    const out = new Array(src.length);
    for (let i = 0; i < src.length; i++) out[i] = (src[i] === 1 && dst[i] > 1) ? 0 : st[i];
    return out;
  }

  // ── 索引助手（行主序 C order）───────────────────────────────────────
  // 下标归一：负索引 −a → n−a（numpy 语义）；归一后仍越界即抛。
  function normIdx(k, n) {
    if (!Number.isInteger(k)) throw new Error('[E] 下标必须为整数: ' + String(k));
    const t = k < 0 ? n + k : k;
    if (t < 0 || t >= n) throw new Error('[E] 下标越界: ' + String(k) + ' 对维度 ' + String(n));
    return t;
  }
  // 多维下标 → 扁平偏移：Σ idx[i]·strides[i]；逐维边界检查，负下标先归一。
  function sub2ind(idx, shape, strides) {
    const n = shape.length;
    if (!Array.isArray(idx) || idx.length !== n) throw new Error('[E] 下标个数与维数不一致');
    const st = strides || NS.L7.shape.contiguousStrides(shape);
    let off = 0;
    for (let i = 0; i < n; i++) off += normIdx(idx[i], shape[i]) * st[i];
    return off;
  }
  // 扁平偏移 → 多维下标：混合 radix 分解（逐维除后缀积取商、余数续传）。
  function ind2sub(flat, shape) {
    if (!Number.isInteger(flat) || flat < 0 || flat >= NS.L7.shape.size(shape)) {
      throw new Error('[E] 扁平下标越界: ' + String(flat));
    }
    const suf = suffixProds(shape);
    const out = new Array(shape.length);
    let f = flat;
    for (let i = 0; i < shape.length; i++) { out[i] = Math.floor(f / suf[i]); f -= out[i] * suf[i]; }
    return out;
  }
  // 半开切片 [s,e) 步长 step≥1 的长度公式：⌈(e−s)/step⌉ 的整数写法。
  function sliceLen(s, e, step) {
    if (!Number.isInteger(step) || step < 1) throw new Error('[E] 切片步长必须 ≥1: ' + String(step));
    return Math.floor((e - s + step - 1) / step);
  }
  // 循环计数参考路径：数 [s,e) 内 s+i·step 的个数（与公式互为独立验证）。
  function sliceLenCount(s, e, step) {
    let c = 0;
    for (let i = 0; s + i * step < e; i++) c++;
    return c;
  }
  // 轴 j 上的确定性切片参数（起/步/长/半开终点），全部落在维内。
  function axisSlice(shape, k, j) {
    const d = shape[j];
    const step = (k % 4) + 1;
    const start = (k >> 2) % d;
    const lenMax = Math.floor((d - 1 - start) / step) + 1;
    const len = ((k >> 6) % lenMax) + 1;
    return { start: start, step: step, len: len, end: start + (len - 1) * step + 1 };
  }
  // 翻转指定轴的视图：形状不变、该轴步长取负、基偏移平移到该轴末端。
  function flipAxisView(shape, strides, j) {
    const st = strides.slice();
    st[j] = -st[j];
    return { stride: st, offset: (shape[j] - 1) * strides[j] };
  }
  // C 连续判定：维大小 >1 的轴步长必须等于后缀积；大小为 1 的轴步长任意
  // （numpy flags.c_contiguous 语义：size-1 轴的步长不参与判定）。
  function isContiguous(shape, strides) {
    const suf = suffixProds(shape);
    for (let i = 0; i < shape.length; i++) {
      if (shape[i] > 1 && strides[i] !== suf[i]) return false;
    }
    return true;
  }
  // 形状相等判定：逐维相等（结构判定，不与 size 比较混同）。
  function shapeEqual(a, b) {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
    return true;
  }

  // ── 形状变换（E097–E108）────────────────────────────────────────────
  reg(97, {
    name: 'squeeze · 去 1 维后无 1 残留', intuition: '把所有长度 1 的维挤掉，结果里不该再出现 1 维。',
    math: 'squeeze(s) = filter(s, d>1); ∀ d ∈ result: d > 1', pseudocode: 'loop assert all >1 && eqArr(result, independentFilter)', complexity: 'O(n)',
    ref: 'numpy.squeeze（全轴版）', commonPitfall: 'E088 只断言 size 守恒；本项断言结构结果——无 1 残留且是原形状的保序子序列。',
    invariant: '对任意正整型 shape：squeeze 结果无 1 维且逐项等于原形状非 1 维子序列',
    inputs: ['shape'], domains: { shape: 'shape4' },
    run: (shape) => squeezeAll(shape),
    predicate: (shape) => {
      const r = squeezeAll(shape);
      for (let i = 0; i < r.length; i++) if (r[i] <= 1) return false;
      const ref = [];
      for (let i = 0; i < shape.length; i++) if (shape[i] > 1) ref.push(shape[i]);
      return eqArr(r, ref);
    },
  });
  reg(98, {
    name: 'squeeze · 幂等（再榨不变）', intuition: '挤过一次就没有 1 维了，再挤一遍还是它自己。',
    math: 'squeeze(squeeze(s)) = squeeze(s)', pseudocode: 'assert eqArr', complexity: 'O(n)',
    ref: 'numpy.squeeze（幂等性）', commonPitfall: '幂等性依赖「结果无 1 维」（E097）；两项合起来才是完整刻画，只测一半会漏掉递归错误。',
    invariant: '对任意正整型 shape：squeeze 二次作用等于一次作用',
    inputs: ['shape'], domains: { shape: 'shape4' },
    run: (shape) => squeezeAll(squeezeAll(shape)),
    predicate: (shape) => eqArr(squeezeAll(squeezeAll(shape)), squeezeAll(shape)),
  });
  reg(99, {
    name: 'unsqueeze · 指定轴插 1 保 size', intuition: '在任意轴位置插一个长度 1 的维，元素总数不变。',
    math: 'size(unsqueeze(s,a)) = size(s), a ∈ [0,n]', pseudocode: 'a = k%(n+1); t = s[:a]+[1]+s[a:]; assert size & 结构', complexity: 'O(n)',
    ref: 'numpy.expand_dims / unsqueeze', commonPitfall: '插入位 a 可取到 n（追加到末维之后）；漏掉端点会少测一种合法位。',
    invariant: '对任意正整型 shape 与 nat k：unsqueeze 保 size 且插入位结构正确',
    inputs: ['shape', 'k'], domains: { shape: 'shape3', k: 'nat' },
    run: (shape, k) => unsqueezeAt(shape, k % (shape.length + 1)),
    predicate: (shape, k) => {
      const a = k % (shape.length + 1);
      const t = unsqueezeAt(shape, a);
      return t.length === shape.length + 1 && t[a] === 1
        && NS.L7.shape.size(t) === NS.L7.shape.size(shape)
        && eqArr(t.slice(0, a).concat(t.slice(a + 1)), shape);
    },
  });
  reg(100, {
    name: 'unsqueeze · 与 squeeze 互逆（无 1 基形）', intuition: '在没有任何 1 维的形状上插一个 1 再挤掉，回到原形。',
    math: 'squeeze(unsqueeze(squeeze(s), a)) = squeeze(s)', pseudocode: 'base = squeeze(s); a = k%(len+1); assert 往返', complexity: 'O(n)',
    ref: 'numpy.squeeze / expand_dims 互逆', commonPitfall: '基形若含 1 维，squeeze 会连它们一起去掉，往返不成立——故先榨一次取无 1 基形。',
    invariant: '对任意正整型 shape 与 nat k：无 1 基形上 unsqueeze 与 squeeze 互为逆',
    inputs: ['shape', 'k'], domains: { shape: 'shape3', k: 'nat' },
    run: (shape, k) => { const base = squeezeAll(shape); const a = k % (base.length + 1); return squeezeAll(unsqueezeAt(base, a)); },
    predicate: (shape, k) => {
      const base = squeezeAll(shape);
      const a = k % (base.length + 1);
      const t = unsqueezeAt(base, a);
      return t.length === base.length + 1 && t[a] === 1 && eqArr(squeezeAll(t), base);
    },
  });
  reg(101, {
    name: 'expand · 尾形广播复制 size', intuition: '尾维为 1 的形状被广播复制到目标尾维，size 恰好放大对应倍数。',
    math: 'size(bcast(b,s)) = size(b)·s[n−1]，b = s 且 b[n−1]=1', pseudocode: 'b = copy(s); b[last]=1; assert eqArr(bc,s) && size 比', complexity: 'O(n)',
    ref: 'numpy.broadcast_to（复制语义）', commonPitfall: '扩展比是目标尾维 s[n−1]；为 1 时是退化复制（比 1），仍合法，勿判失败。',
    invariant: '对任意正整型 shape：尾维制 1 后广播复制的 size === 原 size × 扩展比',
    inputs: ['shape'], domains: { shape: 'shape3' },
    run: (shape) => { const b = shape.slice(); b[b.length - 1] = 1; return NS.L7.shape.broadcastShapes(b, shape); },
    predicate: (shape) => {
      const n = shape.length;
      const b = shape.slice(); b[n - 1] = 1;
      const bc = NS.L7.shape.broadcastShapes(b, shape);
      return eqArr(bc, shape) && NS.L7.shape.size(bc) === NS.L7.shape.size(b) * shape[n - 1];
    },
  });
  reg(102, {
    name: 'expand · 单元素复制语义（0 步长视图）', intuition: '广播视图不复制存储：被扩展轴的步长为 0，整条轴共享同一个元素。',
    math: 'stV[j]=0 ∧ lin(idx,s,stV) = lin(idx|_{j=0}, b)', pseudocode: 'stV = bcastStrides(b,s); assert 0 步长 & 共享偏移 & 源缓冲一致', complexity: 'O(n)',
    ref: 'numpy.broadcast_to（0 步长视图）', commonPitfall: '目标尾维为 1 时没有发生扩展，该轴步长保持 1 而非 0——判据须含此分支，否则退化输入假阳性。',
    invariant: '对任意正整型 shape：广播视图在被扩展尾轴上步长为 0、复制方向偏移共享且与源缓冲偏移一致',
    inputs: ['shape'], domains: { shape: 'shape3' },
    run: (shape) => { const n = shape.length; const b = shape.slice(); b[n - 1] = 1; return bcastStrides(b, shape); },
    predicate: (shape) => {
      const n = shape.length;
      const b = shape.slice(); b[n - 1] = 1;
      const stV = bcastStrides(b, shape);
      const stB = NS.L7.shape.contiguousStrides(b);
      if ((stV[n - 1] === 0) !== (shape[n - 1] > 1)) return false;   // 扩展 ⇔ 0 步长
      const i1 = midOf(shape);
      const i2 = i1.slice(); i2[n - 1] = shape[n - 1] - 1;           // 仅尾坐标不同
      const i0 = i1.slice(); i0[n - 1] = 0;                          // 源侧坐标
      const o1 = NS.L7.shape.linearize(i1, shape, stV);
      return o1 === NS.L7.shape.linearize(i2, shape, stV)            // 复制语义：同偏移
        && o1 === NS.L7.shape.linearize(i0, b, stB)                  // 视图偏移即源偏移
        && o1 >= 0 && o1 < NS.L7.shape.size(b);                      // 落在源缓冲内
    },
  });
  reg(103, {
    name: 'flatten · 指定轴区间保 size', intuition: '把任意闭区间 [a,b] 的轴合并成一维，元素总数不变。',
    math: 'size(flatten(s,a,b)) = size(s), 0≤a≤b<n', pseudocode: 'a=k%n, b=(k>>5)%n, 逆序则交换; assert size & 维数', complexity: 'O(n)',
    ref: 'numpy.flatten / reshape（轴区间一般化）', commonPitfall: 'E056 只合并尾段；本项覆盖任意起止轴（含中段），维数恰减少 (b−a)。',
    invariant: '对任意正整型 shape 与 nat k：指定轴区间合并后 size 不变、维数减少 (b−a)',
    inputs: ['shape', 'k'], domains: { shape: 'shape4', k: 'nat' },
    run: (shape, k) => { const n = shape.length; let a = k % n, b = (k >> 5) % n; if (a > b) { const t = a; a = b; b = t; } return flattenRange(shape, a, b); },
    predicate: (shape, k) => {
      const n = shape.length;
      let a = k % n, b = (k >> 5) % n;
      if (a > b) { const t = a; a = b; b = t; }
      const flat = flattenRange(shape, a, b);
      return NS.L7.shape.size(flat) === NS.L7.shape.size(shape) && flat.length === n - (b - a);
    },
  });
  reg(104, {
    name: 'flatten · 区间外维保序', intuition: '合并只吃掉 [a,b] 之间的轴，区间外的维按原相对次序留在原地。',
    math: 't = s[:a] + [∏_{i=a..b} s[i]] + s[b+1:]', pseudocode: 'assert 头段 eq & 尾段 eq & t[a] = 区间积', complexity: 'O(n)',
    ref: 'numpy.reshape（保序块合并）', commonPitfall: '尾段在结果中的下标要整体左移 (b−a) 位；逐项比对时错位是最常见的错。',
    invariant: '对任意正整型 shape 与 nat k：flatten 结果头段、尾段与原形状对应段逐项相等',
    inputs: ['shape', 'k'], domains: { shape: 'shape4', k: 'nat' },
    run: (shape, k) => { const n = shape.length; let a = k % n, b = (k >> 5) % n; if (a > b) { const t = a; a = b; b = t; } return flattenRange(shape, a, b); },
    predicate: (shape, k) => {
      const n = shape.length;
      let a = k % n, b = (k >> 5) % n;
      if (a > b) { const t = a; a = b; b = t; }
      const flat = flattenRange(shape, a, b);
      for (let i = 0; i < a; i++) if (flat[i] !== shape[i]) return false;
      if (flat[a] !== prodRange(shape, a, b)) return false;
      for (let i = 0; i < n - 1 - b; i++) if (flat[a + 1 + i] !== shape[b + 1 + i]) return false;
      return flat.length === a + 1 + (n - 1 - b);
    },
  });
  reg(105, {
    name: 'flatten · 端点即全展平', intuition: '区间取遍全部轴时，合并结果就是单维 [size]。',
    math: 'flatten(s,0,n−1) = [size(s)]', pseudocode: 'assert 长度 1 且唯一维 === L7 size', complexity: 'O(n)',
    ref: 'numpy.ravel（C 序）', commonPitfall: '与 E055 的 size 层断言不同：本项是形状结构逐项相等（长度 1 且唯一维为 size）。',
    invariant: '对任意正整型 shape：全轴区间 flatten 结果逐项等于 [size]',
    inputs: ['shape'], domains: { shape: 'shape4' },
    run: (shape) => flattenRange(shape, 0, shape.length - 1),
    predicate: (shape) => {
      const t = flattenRange(shape, 0, shape.length - 1);
      return t.length === 1 && t[0] === NS.L7.shape.size(shape);
    },
  });
  reg(106, {
    name: 'split · flatten 互逆', intuition: '按原维度序列拆回合并轴，得到原形状——flatten 与 split 互为逆操作。',
    math: 'split(flatten(s,a,b), a, s[a:b+1]) = s', pseudocode: 't = flatten; back = split(t,a,原维度); assert eqArr', complexity: 'O(n)',
    ref: 'numpy.reshape（互逆拆分）', commonPitfall: 'E062 只比对两种拆法的 size；本项是真正的结构往返，且 split 内部校验维度之积相等。',
    invariant: '对任意正整型 shape 与 nat k：split 是 flatten 在指定轴区间上的逆',
    inputs: ['shape', 'k'], domains: { shape: 'shape4', k: 'nat' },
    run: (shape, k) => { const n = shape.length; let a = k % n, b = (k >> 5) % n; if (a > b) { const t = a; a = b; b = t; } return splitAxis(flattenRange(shape, a, b), a, shape.slice(a, b + 1)); },
    predicate: (shape, k) => {
      const n = shape.length;
      let a = k % n, b = (k >> 5) % n;
      if (a > b) { const t = a; a = b; b = t; }
      const flat = flattenRange(shape, a, b);
      return eqArr(splitAxis(flat, a, shape.slice(a, b + 1)), shape);
    },
  });
  reg(107, {
    name: 'moveaxis · 指定轴迁移保 size', intuition: '把一个轴搬到另一个位置，只是重新解释维度次序，元素总数不变。',
    math: 'size(moveaxis(s,src,dst)) = size(s)', pseudocode: 'src=k%n, dst=(k>>5)%n; assert size', complexity: 'O(n)',
    ref: 'numpy.moveaxis / transpose', commonPitfall: 'src=dst 时是恒等迁移，仍须成立——勿在谓词里特判掉这种退化。',
    invariant: '对任意正整型 shape 与 nat k：moveaxis 保 size',
    inputs: ['shape', 'k'], domains: { shape: 'shape3', k: 'nat' },
    run: (shape, k) => moveAxis(shape, k % shape.length, (k >> 5) % shape.length),
    predicate: (shape, k) => {
      const n = shape.length;
      const src = k % n, dst = (k >> 5) % n;
      return NS.L7.shape.size(moveAxis(shape, src, dst)) === NS.L7.shape.size(shape);
    },
  });
  reg(108, {
    name: 'moveaxis · 与 permute 复合', intuition: 'moveaxis 就是一次置换：与 L7 permute 结果一致，且搬过去再搬回来是原形。',
    math: 'moveaxis(s,src,dst) = permute(s,σ) ∧ moveaxis(moveaxis(s,src,dst),dst,src) = s', pseudocode: 'σ = moveAxesOf(n,src,dst); assert eqArr & 往返', complexity: 'O(n)',
    ref: 'numpy.moveaxis ≡ transpose（零重写：复用 L7 permute）', commonPitfall: '往返的第二跳要把「当前在 dst 的轴」搬回 src——src/dst 参数写反会得到另一个置换。',
    invariant: '对任意正整型 shape 与 nat k：moveaxis 归约为 L7 permute 且往返还原',
    inputs: ['shape', 'k'], domains: { shape: 'shape3', k: 'nat' },
    run: (shape, k) => {
      const n = shape.length;
      const src = k % n, dst = (k >> 5) % n;
      return NS.L7.shape.permute(shape, NS.L7.shape.contiguousStrides(shape), moveAxesOf(n, src, dst)).shape;
    },
    predicate: (shape, k) => {
      const n = shape.length;
      const src = k % n, dst = (k >> 5) % n;
      const axes = moveAxesOf(n, src, dst);
      const viaPermute = NS.L7.shape.permute(shape, NS.L7.shape.contiguousStrides(shape), axes).shape;
      const moved = moveAxis(shape, src, dst);
      return eqArr(moved, viaPermute) && eqArr(moveAxis(moved, dst, src), shape);
    },
  });

  // ── 索引高级（E109–E120）────────────────────────────────────────────
  reg(109, {
    name: 'ravel/unravel · 多维往返', intuition: '多维下标压成扁平偏移再解回来，还是原来那个下标。',
    math: 'ind2sub(sub2ind(idx,s,st),s) = idx（混合 radix 分解互逆）', pseudocode: 'idx 由 k 派生; f = sub2ind; assert eqArr(back, idx)', complexity: 'O(n)',
    ref: 'numpy.ravel_multi_index / unravel_index', commonPitfall: 'unravel 必须用后缀积逐维取商；改用前缀积会得到列主序分解，往返必挂。',
    invariant: '对任意正整型 shape 与 nat k：构造下标经 ravel/unravel 往返后逐项还原',
    inputs: ['shape', 'k'], domains: { shape: 'shape4', k: 'nat' },
    run: (shape, k) => {
      const n = shape.length;
      const idx = new Array(n);
      for (let i = 0; i < n; i++) idx[i] = (k + i * 7919) % shape[i];
      return ind2sub(sub2ind(idx, shape), shape);
    },
    predicate: (shape, k) => {
      const n = shape.length;
      const idx = new Array(n);
      for (let i = 0; i < n; i++) idx[i] = (k + i * 7919) % shape[i];
      const st = NS.L7.shape.contiguousStrides(shape);
      const f = sub2ind(idx, shape, st);
      const back = ind2sub(f, shape);
      return eqArr(back, idx) && sub2ind(back, shape, st) === f
        && f >= 0 && f < NS.L7.shape.size(shape);
    },
  });
  reg(110, {
    name: 'sub2ind · 负下标归一后与 L7 一致', intuition: '负下标先按 numpy 规则归一成正下标，再走同一条点积，与 L7 linearize 逐位相同。',
    math: 'sub2ind(idx−s, s, st) = lin(idx, s, st)（逐维 −s[i] 归一回 +idx[i]）', pseudocode: 'neg[i] = idx[i]−s[i]; assert sub2ind(neg) === linearize(idx)', complexity: 'O(n)',
    ref: 'numpy 负下标语义 + ndarray strides', commonPitfall: '归一必须在边界检查之前；先查后归会把合法负下标误判越界而误抛。',
    invariant: '对任意正整型 shape 与 nat k：负下标归一后的 sub2ind === L7 linearize',
    inputs: ['shape', 'k'], domains: { shape: 'shape4', k: 'nat' },
    run: (shape, k) => {
      const n = shape.length;
      const idx = new Array(n); const neg = new Array(n);
      for (let i = 0; i < n; i++) { idx[i] = (k + i * 104729) % shape[i]; neg[i] = idx[i] - shape[i]; }
      return sub2ind(neg, shape);
    },
    predicate: (shape, k) => {
      const n = shape.length;
      const idx = new Array(n); const neg = new Array(n);
      for (let i = 0; i < n; i++) { idx[i] = (k + i * 104729) % shape[i]; neg[i] = idx[i] - shape[i]; }
      const st = NS.L7.shape.contiguousStrides(shape);
      return sub2ind(neg, shape, st) === NS.L7.shape.linearize(idx, shape, st);
    },
  });
  reg(111, {
    name: 'sub2ind/ind2sub · 2D 行主序往返', intuition: '二维上扁平下标 = 行·列数 + 列，解回来再编回去还是它。',
    math: 'f ∈ [0,r·c); sub = [⌊f/c⌋, f%c]; sub2ind(sub) = f', pseudocode: 'f = k%(r*c); assert 行列公式 & 往返 & 界', complexity: 'O(1)',
    ref: 'numpy.unravel_index（C 序）', commonPitfall: '行主序是 ⌊f/c⌋ 在前；写成 ⌊f/r⌋ 是列主序，往返立即挂。',
    invariant: '对任意二阶正整型 shape 与 nat k：ind2sub/sub2ind 在行主序下往返且行列公式正确',
    inputs: ['shape', 'k'], domains: { shape: 'shape2', k: 'nat' },
    run: (shape, k) => ind2sub(k % (shape[0] * shape[1]), shape),
    predicate: (shape, k) => {
      const r = shape[0], c = shape[1];
      const f = k % (r * c);
      const sub = ind2sub(f, shape);
      return sub.length === 2 && sub[0] === Math.floor(f / c) && sub[1] === f % c
        && sub2ind(sub, shape) === f
        && sub[0] >= 0 && sub[0] < r && sub[1] >= 0 && sub[1] < c;
    },
  });
  reg(112, {
    name: 'sub2ind/ind2sub · 越界即抛', intuition: '扁平下标到顶、行越界、列越界，都必须抛错而不是静默绕回。',
    math: 'f ≥ r·c ∨ f < 0 ∨ row ≥ r ∨ col ≥ c ⇒ throw', pseudocode: '四类越界构造; 全部必须抛', complexity: 'O(1)',
    ref: 'numpy IndexError 语义', commonPitfall: '只测上界不够；负扁平下标同样要抛（归一只对「分量为负但归一后仍在域内」生效）。',
    invariant: '对任意二阶正整型 shape：sub2ind/ind2sub 对四类越界构造均抛错',
    inputs: ['shape'], domains: { shape: 'shape2' },
    run: (shape) => {
      const r = shape[0], c = shape[1];
      const st = NS.L7.shape.contiguousStrides(shape);
      const all = throws(() => ind2sub(r * c, shape)) && throws(() => ind2sub(-1, shape))
        && throws(() => sub2ind([r, 0], shape, st)) && throws(() => sub2ind([0, c], shape, st));
      return all ? 'threw' : 'no-throw';
    },
    predicate: (shape) => {
      const r = shape[0], c = shape[1];
      const st = NS.L7.shape.contiguousStrides(shape);
      return throws(() => ind2sub(r * c, shape))
        && throws(() => ind2sub(-1, shape))
        && throws(() => sub2ind([r, 0], shape, st))
        && throws(() => sub2ind([0, c], shape, st));
    },
  });
  reg(113, {
    name: '负索引 · 归一合法域', intuition: '−1 指最后一维、−n 指第 0 维，负索引整体平移到正域。',
    math: 'norm(−1,n)=n−1 ∧ norm(−n,n)=0 ∧ norm(−a,n)=n−a', pseudocode: 'a = k%n+1; assert 三个等式', complexity: 'O(1)',
    ref: 'numpy 负下标归一', commonPitfall: '−a 的合法域是 a ∈ [1,n]；写成 [0,n) 会把 −0（即 0）误当负索引，多算一种情形。',
    invariant: '对任意正整型 shape 与 nat k：负索引归一满足端点与一般式',
    inputs: ['shape', 'k'], domains: { shape: 'shape3', k: 'nat' },
    run: (shape, k) => { const n = shape.length; const a = (k % n) + 1; return [normIdx(-1, n), normIdx(-n, n), normIdx(-a, n)]; },
    predicate: (shape, k) => {
      const n = shape.length;
      const a = (k % n) + 1;
      return normIdx(-1, n) === n - 1 && normIdx(-n, n) === 0 && normIdx(-a, n) === n - a;
    },
  });
  reg(114, {
    name: '负索引 · 越界仍抛', intuition: '负过头（−n−1）或正到顶（n），归一后依然越界，必须抛。',
    math: 'norm(−n−1,n) throw ∧ norm(n,n) throw', pseudocode: '两类构造; 均须抛', complexity: 'O(1)',
    ref: 'numpy IndexError 语义', commonPitfall: '归一式 n+k 对 k=−n−1 得 −1，仍 <0——不能因为「减过头」就放行。',
    invariant: '对任意正整型 shape：负索引归一对双向越界均抛错',
    inputs: ['shape'], domains: { shape: 'shape2' },
    run: (shape) => (throws(() => normIdx(-shape.length - 1, shape.length)) && throws(() => normIdx(shape.length, shape.length))) ? 'threw' : 'no-throw',
    predicate: (shape) => {
      const n = shape.length;
      return throws(() => normIdx(-n - 1, n)) && throws(() => normIdx(n, n));
    },
  });
  reg(115, {
    name: 'slice · 偏移公式', intuition: '半开切片视图的基偏移就是起点坐标与步长的点积。',
    math: 'offset(s) = Σ s[i]·st[i] = lin(s) ∧ off(last) = off + (len−1)·step·st[j]', pseudocode: 'j = k%n; sl = axisSlice; assert 点积 === L7 lin & 末元素可加', complexity: 'O(n)',
    ref: 'numpy 切片视图（base offset）', commonPitfall: '末元素偏移 = 基偏移 + (len−1)·step·st[j]；只查基偏移不查末元素，等于没验证切片长度。',
    invariant: '对任意正整型 shape 与 nat k：切片视图基偏移等于起点点积且末元素偏移可加',
    inputs: ['shape', 'k'], domains: { shape: 'shape4', k: 'nat' },
    run: (shape, k) => {
      const j = k % shape.length;
      const sl = axisSlice(shape, k, j);
      const idx = midOf(shape); idx[j] = sl.start;
      return NS.L7.shape.linearize(idx, shape);
    },
    predicate: (shape, k) => {
      const n = shape.length;
      const j = k % n;
      const sl = axisSlice(shape, k, j);
      const st = NS.L7.shape.contiguousStrides(shape);
      const idx = midOf(shape); idx[j] = sl.start;
      const last = sl.start + (sl.len - 1) * sl.step;
      const idxLast = midOf(shape); idxLast[j] = last;
      let dot = 0;
      for (let i = 0; i < n; i++) dot += idx[i] * st[i];
      const base = NS.L7.shape.linearize(idx, shape, st);
      return base === dot && base === sub2ind(idx, shape, st)
        && NS.L7.shape.linearize(idxLast, shape, st) === base + (sl.len - 1) * sl.step * st[j]
        && last < shape[j];
    },
  });
  reg(116, {
    name: 'slice · 半开长度公式', intuition: '[s,e) 里按 step 跳跃能取几个数，ceil 除法一次算对。',
    math: 'len = ⌈(e−s)/step⌉ = ⌊(e−s+step−1)/step⌋', pseudocode: 'assert 公式 === 循环计数 && 末元素在界 && 再一步出界', complexity: 'O(1)',
    ref: 'numpy 切片长度语义', commonPitfall: '整数除法向下取整，(e−s+step−1) 的 +step−1 补偿不能省；省略会少算一个元素。',
    invariant: '对任意正整型 shape 与 nat k：切片长度公式与循环计数参考路径一致且末元素刻画正确',
    inputs: ['shape', 'k'], domains: { shape: 'shape4', k: 'nat' },
    run: (shape, k) => { const j = k % shape.length; const sl = axisSlice(shape, k, j); return sliceLen(sl.start, sl.end, sl.step); },
    predicate: (shape, k) => {
      const j = k % shape.length;
      const sl = axisSlice(shape, k, j);
      return sliceLen(sl.start, sl.end, sl.step) === sl.len
        && sliceLenCount(sl.start, sl.end, sl.step) === sl.len
        && sl.start + (sl.len - 1) * sl.step < sl.end
        && sl.start + sl.len * sl.step >= sl.end;
    },
  });
  reg(117, {
    name: 'slice · 分段偏移可加', intuition: '把一段切片从中间切开，两段的偏移差之和等于整段的偏移差。',
    math: '(off(m)−off(s)) + (off(e)−off(m)) = off(e)−off(s) = (e−s)·st[j]', pseudocode: 'm ∈ [s,e); assert 长度可加 & Δoff = Δcoord·st[j]', complexity: 'O(n)',
    ref: '仿射视图复合（切片拼接）', commonPitfall: '偏移可加只在「仅同一轴变化」时成立；跨轴拼接还要叠加其它轴的固定贡献。',
    invariant: '对任意正整型 shape 与 nat k：切片分段后长度与偏移均可加',
    inputs: ['shape', 'k'], domains: { shape: 'shape4', k: 'nat' },
    run: (shape, k) => {
      const j = k % shape.length;
      const sl = axisSlice(shape, k, j);
      const m = sl.start + (k % (sl.end - sl.start));
      const st = NS.L7.shape.contiguousStrides(shape);
      const iS = midOf(shape); iS[j] = sl.start;
      const iM = midOf(shape); iM[j] = m;
      return NS.L7.shape.linearize(iM, shape, st) - NS.L7.shape.linearize(iS, shape, st);
    },
    predicate: (shape, k) => {
      const j = k % shape.length;
      const sl = axisSlice(shape, k, j);
      const m = sl.start + (k % (sl.end - sl.start));
      const st = NS.L7.shape.contiguousStrides(shape);
      const iS = midOf(shape); iS[j] = sl.start;
      const iM = midOf(shape); iM[j] = m;
      return (m - sl.start) + (sl.end - m) === sl.end - sl.start
        && NS.L7.shape.linearize(iM, shape, st) - NS.L7.shape.linearize(iS, shape, st) === (m - sl.start) * st[j];
    },
  });
  reg(118, {
    name: '翻转轴 · 步长取负镜像恒等式', intuition: '把一根轴翻过来看：该轴步长取负、基偏移平移到末端，读到的正是镜像位置的元素。',
    math: 'lin(idx,s,−st,off) = lin(mirror(idx),s,st), off = (s[j]−1)·st[j]', pseudocode: 'v = flipAxisView; assert 取负 & 偏移式 & 镜像 & 值域', complexity: 'O(n)',
    ref: 'numpy 负步长视图（a[::-1]）', commonPitfall: '只把步长取负而不平移基偏移，视图会整体读出缓冲区左侧的垃圾。',
    invariant: '对任意正整型 shape 与 nat k：翻转视图满足镜像恒等式且所有探测偏移保持在 [0,size)',
    inputs: ['shape', 'k'], domains: { shape: 'shape4', k: 'nat' },
    run: (shape, k) => {
      const j = k % shape.length;
      const st = NS.L7.shape.contiguousStrides(shape);
      const v = flipAxisView(shape, st, j);
      return NS.L7.shape.linearize(midOf(shape), shape, v.stride, v.offset);
    },
    predicate: (shape, k) => {
      const n = shape.length;
      const j = k % n;
      const st = NS.L7.shape.contiguousStrides(shape);
      const v = flipAxisView(shape, st, j);
      const mid = midOf(shape);
      const mir = mid.slice(); mir[j] = shape[j] - 1 - mid[j];
      if (v.stride[j] !== -st[j] || v.offset !== (shape[j] - 1) * st[j]) return false;
      if (NS.L7.shape.linearize(mid, shape, v.stride, v.offset) !== NS.L7.shape.linearize(mir, shape, st)) return false;
      const sz = NS.L7.shape.size(shape);
      const probes = [zerosOf(n), mid, shape.map((d) => d - 1)];
      for (let p = 0; p < probes.length; p++) {
        const o = NS.L7.shape.linearize(probes[p], shape, v.stride, v.offset);
        if (o < 0 || o >= sz) return false;
      }
      return true;
    },
  });
  reg(119, {
    name: '翻转 · 二次对合', intuition: '同一根轴连翻两次，步长负负得正、偏移抵消，视图回到原状。',
    math: 'flip(flip(v)) = v（stride 还原、offset 归零）', pseudocode: 'v2 = flip(v1); assert eqArr(stride) && offset 和 === 0', complexity: 'O(n)',
    ref: '对合（involution）', commonPitfall: '两次翻转必须沿同一轴；翻完 j 再翻 k≠j 得到的是双负步长，不是还原。',
    invariant: '对任意正整型 shape 与 nat k：同轴二次翻转步长与偏移均还原',
    inputs: ['shape', 'k'], domains: { shape: 'shape4', k: 'nat' },
    run: (shape, k) => {
      const j = k % shape.length;
      const st = NS.L7.shape.contiguousStrides(shape);
      const v1 = flipAxisView(shape, st, j);
      return flipAxisView(shape, v1.stride, j).stride;
    },
    predicate: (shape, k) => {
      const j = k % shape.length;
      const st = NS.L7.shape.contiguousStrides(shape);
      const v1 = flipAxisView(shape, st, j);
      const v2 = flipAxisView(shape, v1.stride, j);
      const mid = midOf(shape);
      return eqArr(v2.stride, st) && v1.offset + v2.offset === 0
        && NS.L7.shape.linearize(mid, shape, v2.stride, v1.offset + v2.offset) === NS.L7.shape.linearize(mid, shape, st);
    },
  });
  reg(120, {
    name: 'contiguous 判定 · 后缀积准则', intuition: 'C 连续的充要条件是非 1 维步长都等于后缀积；1 维轴的步长无所谓。',
    math: 'contig(s,st) ⟺ ∀i: s[i]>1 ⇒ st[i]=∏_{j>i}s[j]', pseudocode: 'assert L7 步长判连续; 可探测扰动脉冲判不连续; 全 1 形状任意步长皆连续', complexity: 'O(n)',
    ref: 'numpy flags.c_contiguous（size-1 轴步长不计）', commonPitfall: '「首不确定」是误读：不确定的只是大小为 1 的轴（含首维为 1 的情形），非 1 维一步长都不能差。',
    invariant: '对任意正整型 shape：连续步长判定为连续、可探测扰动脉冲为不连续、全 1 形状任意步长皆连续',
    inputs: ['shape'], domains: { shape: 'shape4' },
    run: (shape) => isContiguous(shape, NS.L7.shape.contiguousStrides(shape)),
    predicate: (shape) => {
      const st = NS.L7.shape.contiguousStrides(shape);
      if (!isContiguous(shape, st)) return false;
      let j = -1;
      for (let i = 0; i < shape.length; i++) if (shape[i] > 1) { j = i; break; }
      if (j < 0) return isContiguous(shape, zerosOf(shape.length));   // 全 1 形状：任意步长
      const bad = st.slice(); bad[j] += 1;                            // 可探测扰动
      return !isContiguous(shape, bad);
    },
  });

  // ── 形状代数进阶（E121–E128）────────────────────────────────────────
  reg(121, {
    name: '广播 · 三形状结合律（奇偶 1 型）', intuition: '三种兼容形状按不同顺序两两广播再复合，结果相同。',
    math: '(s⊕e)⊕o = s⊕(e⊕o)，e = 偶维置 1，o = 奇维置 1', pseudocode: 'assert eqArr(lhs, rhs) && 长度不变', complexity: 'O(n)',
    ref: 'numpy broadcasting 结合律', commonPitfall: '结合律前提是每一对中间结果都兼容；E081 用切片构造覆盖缺长链，本项用等长奇偶 1 型覆盖全维兼容面，两者互补。',
    invariant: '对任意正整型 shape：L7 广播对等长奇偶 1 型三形状链满足结合律',
    inputs: ['shape'], domains: { shape: 'shape3' },
    run: (shape) => {
      const e = shape.map((d, i) => (i % 2 === 0 ? 1 : d));
      const o = shape.map((d, i) => (i % 2 === 1 ? 1 : d));
      return NS.L7.shape.broadcastShapes(NS.L7.shape.broadcastShapes(shape, e), o);
    },
    predicate: (shape) => {
      const e = shape.map((d, i) => (i % 2 === 0 ? 1 : d));
      const o = shape.map((d, i) => (i % 2 === 1 ? 1 : d));
      const lhs = NS.L7.shape.broadcastShapes(NS.L7.shape.broadcastShapes(shape, e), o);
      const rhs = NS.L7.shape.broadcastShapes(shape, NS.L7.shape.broadcastShapes(e, o));
      return eqArr(lhs, rhs) && lhs.length === shape.length;
    },
  });
  reg(122, {
    name: '共轭置换仍是置换', intuition: '置换群里共轭一个置换，得到的还是合法排列，且保 size、可逆。',
    math: 'σ∘τ∘σ⁻¹ ∈ S_n ∧ size 守恒 ∧ 逆置换还原', pseudocode: 'σ=reverse, τ=rotate; assert L7 接受 & size & 往返', complexity: 'O(n)',
    ref: '置换群共轭类（Cayley 定理视角）', commonPitfall: '共轭复合顺序容易写反（σ∘τ∘σ⁻¹ ≠ σ⁻¹∘τ∘σ 一般情形）；按 comp(p,q)[i]=p[q[i]] 从右往左应用。',
    invariant: '对任意正整型 shape：构造共轭置换被 L7 接受、保 size 且可逆',
    inputs: ['shape'], domains: { shape: 'shape3' },
    run: (shape) => {
      const n = shape.length;
      const sig = reverseAxes(n); const tau = rotateAxes(n); const invSig = NS.L7.shape.invertPerm(sig);
      const conj = new Array(n);
      for (let i = 0; i < n; i++) conj[i] = sig[tau[invSig[i]]];
      return conj;
    },
    predicate: (shape) => {
      const n = shape.length;
      const st = NS.L7.shape.contiguousStrides(shape);
      const sig = reverseAxes(n); const tau = rotateAxes(n); const invSig = NS.L7.shape.invertPerm(sig);
      const conj = new Array(n);
      for (let i = 0; i < n; i++) conj[i] = sig[tau[invSig[i]]];
      let p1;
      try { p1 = NS.L7.shape.permute(shape, st, conj); } catch (e) { return false; }  // 合法性硬门禁
      if (NS.L7.shape.size(p1.shape) !== NS.L7.shape.size(shape)) return false;
      const p2 = NS.L7.shape.permute(p1.shape, p1.stride, NS.L7.shape.invertPerm(conj));
      return eqArr(p2.shape, shape);
    },
  });
  reg(123, {
    name: '广播后 reshape · size 不减', intuition: '广播把 1 放大之后整体展平，元素数不比任何输入少。',
    math: 'size([size(bcast(s,b))]) = size(bcast(s,b)) ≥ size(s), size(b)', pseudocode: 'b = 偶维 1; bc = bcast(s,b); t = [size(bc)]; assert 链', complexity: 'O(n)',
    ref: 'numpy broadcasting + reshape', commonPitfall: '「不减」含取等（对手全是 1 的维时）；断言成严格大于会在退化输入上假阳性。',
    invariant: '对任意正整型 shape：广播结果展平后 size 与广播 size 一致且不小于双方输入 size',
    inputs: ['shape'], domains: { shape: 'shape4' },
    run: (shape) => {
      const b = shape.map((d, i) => (i % 2 === 0 ? 1 : d));
      return [NS.L7.shape.size(NS.L7.shape.broadcastShapes(shape, b))];
    },
    predicate: (shape) => {
      const b = shape.map((d, i) => (i % 2 === 0 ? 1 : d));
      const bc = NS.L7.shape.broadcastShapes(shape, b);
      const t = [NS.L7.shape.size(bc)];
      const sz = NS.L7.shape.size(shape);
      return NS.L7.shape.size(t) === NS.L7.shape.size(bc)
        && NS.L7.shape.size(bc) >= sz && NS.L7.shape.size(bc) >= NS.L7.shape.size(b)
        && NS.L7.shape.linearize([t[0] - 1], t) === sz - 1;
    },
  });
  reg(124, {
    name: '空维 · 指定轴移除与插回互逆', intuition: '把一个大小为 1 的轴拿掉再插回原位，形状还原、size 不变。',
    math: 'unsqueeze(remove(s,j), j) = s|_{s[j]=1} ∧ size 守恒', pseudocode: 'j = k%n; s1 = s 且 s1[j]=1; assert 互逆 & size', complexity: 'O(n)',
    ref: 'numpy.squeeze(axis) / expand_dims 互逆', commonPitfall: 'E088 是「滤全部 1 维保 size」；本项是「指定单轴」的结构往返，二者互为补充。',
    invariant: '对任意正整型 shape 与 nat k：指定轴置 1 后移除与插回互为逆且 size 不变',
    inputs: ['shape', 'k'], domains: { shape: 'shape3', k: 'nat' },
    run: (shape, k) => {
      const j = k % shape.length;
      const s1 = shape.slice(); s1[j] = 1;
      return s1.slice(0, j).concat(s1.slice(j + 1));
    },
    predicate: (shape, k) => {
      const n = shape.length;
      const j = k % n;
      const s1 = shape.slice(); s1[j] = 1;
      const removed = s1.slice(0, j).concat(s1.slice(j + 1));
      return removed.length === n - 1
        && NS.L7.shape.size(removed) === NS.L7.shape.size(s1)
        && eqArr(unsqueezeAt(removed, j), s1);
    },
  });
  reg(125, {
    name: '形状相等 · 判定与 size 一致', intuition: '形状相等就是逐维相等；判定必须看结构，不能只看 size。',
    math: 'shapeEqual(a,b) ⟺ len(a)=len(b) ∧ ∀i a[i]=b[i]', pseudocode: 'assert 自反 & 拷贝等 & 逐维差即不等 & 与深比较一致', complexity: 'O(n)',
    ref: '结构相等（structural equality）', commonPitfall: 'size 相同不蕴涵形状相同（逆序对即反例）；判定退化成长为 size 比较会放过真差异。',
    invariant: '对任意正整型 shape：shapeEqual 自反、拷贝相等、逐维差即不等、且与逐维深比较完全一致',
    inputs: ['shape'], domains: { shape: 'shape2' },
    run: (shape) => [shapeEqual(shape, shape.slice()), shapeEqual(shape, shape.map((d) => d + 1))],
    predicate: (shape) => {
      const rev = shape.slice().reverse();
      const plus1 = shape.map((d) => d + 1);
      return shapeEqual(shape, shape) && shapeEqual(shape, shape.slice())
        && NS.L7.shape.size(shape.slice()) === NS.L7.shape.size(shape)
        && !shapeEqual(shape, plus1) && !shapeEqual(shape, shape.concat([1]))
        && shapeEqual(shape, rev) === eqArr(shape, rev);
    },
  });
  reg(126, {
    name: 'reshape 与 permute · 先展平可交换', intuition: '先展平再置换，一维上置换是恒等；两条路线都落到同一个 [size]。',
    math: 'perm([size]) = [size] = [size(perm(s))]', pseudocode: 'assert eqArr(路线一, 路线二) && 最大偏移守恒', complexity: 'O(n)',
    ref: 'numpy.ravel + transpose（一维情形）', commonPitfall: '多维情形 reshape 与 permute 一般不可交换；可交换性只在展平后的一维上成立。',
    invariant: '对任意正整型 shape：展平路线与置换展平路线得到同一形状且最大偏移守恒',
    inputs: ['shape'], domains: { shape: 'shape3' },
    run: (shape) => [NS.L7.shape.size(shape)],
    predicate: (shape) => {
      const n = shape.length;
      const st = NS.L7.shape.contiguousStrides(shape);
      const sz = NS.L7.shape.size(shape);
      const flat = [sz];
      const pFlat = NS.L7.shape.permute(flat, [1], [0]).shape;      // 路线一：先展平再置换
      const perm = NS.L7.shape.permute(shape, st, reverseAxes(n));
      const flat2 = [NS.L7.shape.size(perm.shape)];                 // 路线二：先置换再展平
      return eqArr(pFlat, flat2) && flat2[0] === sz
        && NS.L7.shape.linearize([sz - 1], flat) === sz - 1
        && NS.L7.shape.linearize(shape.map((d) => d - 1), shape) === sz - 1;
    },
  });
  reg(127, {
    name: 'flatten · 分步合并等价一次合并', intuition: '先合并尾段再合并前两轴，与一口气合并全部轴，结果相同。',
    math: 'flatten(flatten(s,1,n−1),0,1) = flatten(s,0,n−1) = [size(s)]', pseudocode: 'assert eqArr(两步, 一步) && size 链', complexity: 'O(n)',
    ref: 'reshape 结合律（块合并）', commonPitfall: '分步合并的第二次区间要按第一步的结果重新取轴——直接照抄原区间会错位。',
    invariant: '对任意正整型 shape：分步 flatten 与一次全轴 flatten 结果逐项相同',
    inputs: ['shape'], domains: { shape: 'shape4' },
    run: (shape) => {
      const tail = flattenRange(shape, 1, shape.length - 1);
      return flattenRange(tail, 0, 1);
    },
    predicate: (shape) => {
      const n = shape.length;
      const oneShot = flattenRange(shape, 0, n - 1);
      const tail = flattenRange(shape, 1, n - 1);
      const twoShot = flattenRange(tail, 0, 1);
      return eqArr(twoShot, oneShot) && oneShot.length === 1
        && oneShot[0] === NS.L7.shape.size(shape)
        && NS.L7.shape.size(twoShot) === NS.L7.shape.size(shape);
    },
  });
  reg(128, {
    name: '广播 · 同伙伴复合幂等', intuition: '和同一个兼容伙伴广播两次，第二次什么也不改变。',
    math: 'bcast(bcast(s,b),b) = bcast(s,b)', pseudocode: 'b = 偶维 1; assert eqArr(bc2, bc1) && eqArr(bc1, s)', complexity: 'O(n)',
    ref: 'numpy broadcasting（吸收律）', commonPitfall: '与 E034 自身幂等不同：本项的伙伴是异形兼容对；幂等性对「已被吸收过的结果」成立。',
    invariant: '对任意正整型 shape：L7 广播对同一兼容伙伴二次复合结果不变',
    inputs: ['shape'], domains: { shape: 'shape4' },
    run: (shape) => {
      const b = shape.map((d, i) => (i % 2 === 0 ? 1 : d));
      return NS.L7.shape.broadcastShapes(NS.L7.shape.broadcastShapes(shape, b), b);
    },
    predicate: (shape) => {
      const b = shape.map((d, i) => (i % 2 === 0 ? 1 : d));
      const bc1 = NS.L7.shape.broadcastShapes(shape, b);
      const bc2 = NS.L7.shape.broadcastShapes(bc1, b);
      return eqArr(bc2, bc1) && eqArr(bc1, shape);
    },
  });
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

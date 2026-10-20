/* ============================================================================
 * L10 微指令 · E 域（形状与索引）· 第 1/3 批（E001–E032）
 * ----------------------------------------------------------------------------
 * 所属层：L10-microIsa  域：E（形状与索引）
 * 允许依赖的层：L7（shape 代数）、L9（rng）、L10（registry/invariant）
 * 主要 @ref：numpy ndarray strides（行主序 C order）；
 *            Oliphant, "Guide to NumPy"（strides/size/linearize 语义）
 *
 * 每项含七件套（直觉/数学/伪代码/复杂度/论文/常见坑/不变量）+ run + predicate。
 * 判据：形状代数为整数精确，一律用 === 与逐元素深比（eqArr），不用容差。
 * 输入：'shape' 经 L10.genShape 生成长度 2–4、每维 ∈ [1,8] 的正整数元组；
 *       需要第二个形状/下标时，在 predicate 内由第一个形状确定性地构造
 *       （广播兼容性、下标合法性均由构造保证，不依赖外部任意子碰巧合法）。
 * 加载顺序注意：本文件先于 L7 求值，故对 NS.L7.shape 一律惰性引用
 *       （predicate/run 调用期解析），模块求值期不做顶层捕获。
 * ==========================================================================*/
;(function (NS) {
  'use strict';
  NS.L10 = NS.L10 || {};
  function reg(i, o) { NS.L10.register('E', i, o); }

  // ── 本地第一性原理助手（与 L7 实现互为交叉验证）──────────────────────
  function prod(a, from) { let n = 1; for (let i = from || 0; i < a.length; i++) n *= a[i]; return n; }
  function stridesOf(shape) {
    const n = shape.length; const out = new Array(n); let acc = 1;
    for (let i = n - 1; i >= 0; i--) { out[i] = acc; acc *= shape[i]; }
    return out;
  }
  function eqArr(a, b) {
    if (!a || !b || a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
    return true;
  }
  function zerosOf(n) { const a = new Array(n); for (let i = 0; i < n; i++) a[i] = 0; return a; }
  function midOf(shape) { const a = new Array(shape.length); for (let i = 0; i < shape.length; i++) a[i] = shape[i] >> 1; return a; }

  // ── 连续步长（E001–E006）────────────────────────────────────────────
  reg(1, {
    name: '连续步长 · 末维恒为 1', intuition: '行主序下末维在内存中相邻，步长必为 1，这是 C 序的指纹。',
    math: 'st = contiguousStrides(s) ⇒ st[n-1] = 1', pseudocode: 'return st[st.length-1] === 1', complexity: 'O(n)',
    ref: 'numpy ndarray strides（C order）', commonPitfall: '勿与列主序（Fortran）混淆——那是首维为 1。',
    invariant: '对任意正整型 shape：L7 连续步长末项恒为 1',
    inputs: ['shape'], domains: { shape: 'shape3' },
    run: (shape) => NS.L7.shape.contiguousStrides(shape),
    predicate: (shape) => { const st = NS.L7.shape.contiguousStrides(shape); return st[st.length - 1] === 1; },
  });
  reg(2, {
    name: '连续步长 · 首维即后缀元素数', intuition: '沿首维走一步，跨过的是其余全部元素。',
    math: 'st[0] = size(s[1:])', pseudocode: 'return st[0] === size(shape.slice(1))', complexity: 'O(n)',
    ref: 'numpy ndarray strides', commonPitfall: '空后缀（一维形状）时 st[0] 为 1，勿按 0 处理。',
    invariant: '对任意正整型 shape：L7 连续步长首项等于去掉首维后的 size',
    inputs: ['shape'], domains: { shape: 'shape2' },
    run: (shape) => NS.L7.shape.contiguousStrides(shape)[0],
    predicate: (shape) => {
      const st = NS.L7.shape.contiguousStrides(shape);
      return st[0] === NS.L7.shape.size(shape.slice(1));
    },
  });
  reg(3, {
    name: '连续步长 · 逐维后缀积公式', intuition: '第 i 维步长 = 其后所有维的乘积，逐维成立。',
    math: '∀i: st[i] = ∏_{j>i} s[j]', pseudocode: 'for i: assert st[i] === prod(shape, i+1)', complexity: 'O(n²)',
    ref: 'numpy ndarray strides', commonPitfall: '末维后缀积为空积，约定为 1，勿漏判。',
    invariant: '对任意正整型 shape：L7 连续步长逐维等于后缀维度积',
    inputs: ['shape'], domains: { shape: 'shape4' },
    run: (shape) => NS.L7.shape.contiguousStrides(shape),
    predicate: (shape) => {
      const st = NS.L7.shape.contiguousStrides(shape);
      for (let i = 0; i < shape.length; i++) if (st[i] !== prod(shape, i + 1)) return false;
      return true;
    },
  });
  reg(4, {
    name: '连续步长 · 第一性原理交叉验证', intuition: '用自写的行主序累积算法与 L7 逐项比对，互为基准。',
    math: 'stridesOf(s) ≡ contiguousStrides(s)', pseudocode: 'return eqArr(stridesOf(shape), L7strides(shape))', complexity: 'O(n)',
    ref: 'numpy ndarray strides', commonPitfall: '深比须逐元素，勿用 ===（数组引用不同）。',
    invariant: '本地第一性原理步长与 L7 连续步长逐元素一致',
    inputs: ['shape'], domains: { shape: 'shape3' },
    run: (shape) => stridesOf(shape),
    predicate: (shape) => eqArr(stridesOf(shape), NS.L7.shape.contiguousStrides(shape)),
  });
  reg(5, {
    name: '连续步长 · 纯函数（确定且不改输入）', intuition: '同一形状两次调用结果一致，且原数组不被写坏。',
    math: 'f(s)=f(s) ∧ s 不变', pseudocode: 'return eqArr(f(s), f(s)) && eqArr(s, copy)', complexity: 'O(n)',
    ref: '函数式纯度（QuickCheck 性质）', commonPitfall: '累积变量若复用入参数组会造成隐式修改。',
    invariant: 'L7 连续步长对同输入确定可复现，且不修改输入形状',
    inputs: ['shape'], domains: { shape: 'shape2' },
    run: (shape) => NS.L7.shape.contiguousStrides(shape),
    predicate: (shape) => {
      const copy = shape.slice();
      const s1 = NS.L7.shape.contiguousStrides(shape);
      const s2 = NS.L7.shape.contiguousStrides(shape);
      return eqArr(s1, s2) && eqArr(shape, copy);
    },
  });
  reg(6, {
    name: '连续步长 · 追加 1 维不前移', intuition: '尾部补一个 1 维，前面各维步长原样保留，新增步长为 1。',
    math: 'strides(s∥[1])[i] = strides(s)[i] (i<n)', pseudocode: 'compare prefixes', complexity: 'O(n)',
    ref: 'numpy ndarray strides（newaxis 语义）', commonPitfall: '把 1 维当成 0 维会错位；1 是乘性单位。',
    invariant: '对任意正整型 shape：追加 1 维后前 n 项步长不变、新末项为 1',
    inputs: ['shape'], domains: { shape: 'shape4' },
    run: (shape) => NS.L7.shape.contiguousStrides(shape.concat([1])),
    predicate: (shape) => {
      const st = NS.L7.shape.contiguousStrides(shape);
      const st1 = NS.L7.shape.contiguousStrides(shape.concat([1]));
      if (st1.length !== shape.length + 1 || st1[shape.length] !== 1) return false;
      for (let i = 0; i < shape.length; i++) if (st1[i] !== st[i]) return false;
      return true;
    },
  });

  // ── size（E007–E012）────────────────────────────────────────────────
  reg(7, {
    name: 'size · 维度乘积（第一性原理）', intuition: '元素数就是各维相乘；空形状（标量）约定为 1。',
    math: 'size(s) = ∏ s[i]', pseudocode: 'return L7size(shape) === prod(shape, 0)', complexity: 'O(n)',
    ref: 'numpy ndarray.size', commonPitfall: '含 0 维时 size 为 0（空张量），本域输入恒 ≥1 故不触发。',
    invariant: 'L7 size 与本地维度乘积逐值一致',
    inputs: ['shape'], domains: { shape: 'shape3' },
    run: (shape) => NS.L7.shape.size(shape),
    predicate: (shape) => NS.L7.shape.size(shape) === prod(shape, 0),
  });
  reg(8, {
    name: 'size · 空形状约定为 1', intuition: '标量是 0 维张量，元素数为 1（空积），步长表为空。',
    math: 'size([]) = 1 ∧ strides([]) = []', pseudocode: 'return size([]) === 1', complexity: 'O(1)',
    ref: 'numpy ndarray（0-d array 语义）', commonPitfall: '把空形状当成 0 元素是经典错；空积定义为 1。',
    invariant: 'L7 size([]) 精确为 1 且其连续步长为空表',
    inputs: ['shape'], domains: { shape: 'shape2' },
    run: () => NS.L7.shape.size([]),
    predicate: () => NS.L7.shape.size([]) === 1 && NS.L7.shape.contiguousStrides([]).length === 0,
  });
  reg(9, {
    name: 'size · 任意切分乘性', intuition: '沿任意轴切开，两半元素数之积等于整块元素数。',
    math: 'size(s[:k])·size(s[k:]) = size(s)', pseudocode: 'for k in 1..n-1: assert product', complexity: 'O(n²)',
    ref: 'numpy ndarray.size', commonPitfall: '切分点 k 须在 [1, n-1]；k=0 或 n 退化无意义。',
    invariant: '对任意正整型 shape 与每个内部切分点：前缀 size × 后缀 size === 总 size',
    inputs: ['shape'], domains: { shape: 'shape4' },
    run: (shape) => shape.map((d, i) => NS.L7.shape.size(shape.slice(0, i + 1)) * NS.L7.shape.size(shape.slice(i + 1))),
    predicate: (shape) => {
      for (let k = 1; k < shape.length; k++) {
        if (NS.L7.shape.size(shape.slice(0, k)) * NS.L7.shape.size(shape.slice(k)) !== NS.L7.shape.size(shape)) return false;
      }
      return true;
    },
  });
  reg(10, {
    name: 'size · 追加 1 维不变', intuition: '补一个长度为 1 的维不改变元素总数。',
    math: 'size(s∥[1]) = size(s)', pseudocode: 'return size(concat(shape,[1])) === size(shape)', complexity: 'O(n)',
    ref: 'numpy ndarray（newaxis 不改变元素数）', commonPitfall: '与追加非 1 维混淆——那会真正放大。',
    invariant: '对任意正整型 shape：追加 1 维后 L7 size 不变',
    inputs: ['shape'], domains: { shape: 'shape3' },
    run: (shape) => NS.L7.shape.size(shape.concat([1])),
    predicate: (shape) => NS.L7.shape.size(shape.concat([1])) === NS.L7.shape.size(shape),
  });
  reg(11, {
    name: 'size · 全 1 形状为单元素', intuition: '各维都是 1 时只有一个元素，步长也全为 1。',
    math: 'size(1ⁿ) = 1 ∧ strides(1ⁿ) = 1ⁿ', pseudocode: 'return size(ones) === 1', complexity: 'O(n)',
    ref: 'numpy ndarray', commonPitfall: '全 1 形状仍是有形状的张量，勿与 0 维标量混同。',
    invariant: '对任意正整型 shape：其全 1 化的 size 为 1 且步长全为 1',
    inputs: ['shape'], domains: { shape: 'shape2' },
    run: (shape) => NS.L7.shape.size(shape.map(() => 1)),
    predicate: (shape) => {
      const ones = shape.map(() => 1);
      return NS.L7.shape.size(ones) === 1 && eqArr(NS.L7.shape.contiguousStrides(ones), ones);
    },
  });
  reg(12, {
    name: 'size · 首维跨度', intuition: '首维大小乘其步长恰好覆盖全部元素。',
    math: 's[0]·st[0] = size(s)', pseudocode: 'return shape[0]*st[0] === size(shape)', complexity: 'O(n)',
    ref: 'numpy ndarray strides', commonPitfall: '该式对非连续步长不成立，故限定连续步长。',
    invariant: '对任意正整型 shape：首维 × 首维步长 === L7 size',
    inputs: ['shape'], domains: { shape: 'shape4' },
    run: (shape) => NS.L7.shape.contiguousStrides(shape)[0] * shape[0],
    predicate: (shape) => {
      const st = NS.L7.shape.contiguousStrides(shape);
      return shape[0] * st[0] === NS.L7.shape.size(shape);
    },
  });

  // ── linearize 基础（E013–E024）──────────────────────────────────────
  reg(13, {
    name: '连续步长 · 相邻维度跨度', intuition: '第 i-1 维一步跨过的元素数 = 第 i 维大小 × 第 i 维步长。',
    math: 'st[i-1] = s[i]·st[i]', pseudocode: 'for i in 1..n-1: assert', complexity: 'O(n)',
    ref: 'numpy ndarray strides', commonPitfall: '末维无后继，循环上界为 n-1。',
    invariant: '对任意正整型 shape：L7 连续步长满足逐维递推 st[i-1] === s[i]·st[i]',
    inputs: ['shape'], domains: { shape: 'shape3' },
    run: (shape) => NS.L7.shape.contiguousStrides(shape),
    predicate: (shape) => {
      const st = NS.L7.shape.contiguousStrides(shape);
      for (let i = 1; i < shape.length; i++) if (st[i - 1] !== shape[i] * st[i]) return false;
      return true;
    },
  });
  reg(14, {
    name: 'linearize · 零下标等于偏移', intuition: '全零下标不贡献任何位移，扁平偏移就是传入的 offset。',
    math: 'linearize(0ⁿ, s, st, off) = off', pseudocode: 'return linearize(zeros, shape, st, 7) === 7', complexity: 'O(n)',
    ref: 'numpy ndarray（offset 语义）', commonPitfall: 'offset 省略时默认为 0，勿与 stride 省略混淆。',
    invariant: '对任意正整型 shape：零下标的线性化结果精确等于给定 offset',
    inputs: ['shape'], domains: { shape: 'shape2' },
    run: (shape) => NS.L7.shape.linearize(zerosOf(shape.length), shape, null, 7),
    predicate: (shape) => NS.L7.shape.linearize(zerosOf(shape.length), shape, null, 7) === 7,
  });
  reg(15, {
    name: 'linearize · 省略步长即行主序', intuition: '不传 stride 时按行主序连续步长计算，与显式传入一致。',
    math: 'linearize(idx, s) = linearize(idx, s, strides(s), 0)', pseudocode: 'compare default vs explicit', complexity: 'O(n)',
    ref: 'numpy ndarray strides', commonPitfall: '只在你确信布局连续时才可省略 stride。',
    invariant: '对任意正整型 shape：L7 linearize 默认步长与显式连续步长结果一致',
    inputs: ['shape'], domains: { shape: 'shape4' },
    run: (shape) => NS.L7.shape.linearize(midOf(shape), shape),
    predicate: (shape) => {
      const st = NS.L7.shape.contiguousStrides(shape);
      const mid = midOf(shape);
      return NS.L7.shape.linearize(mid, shape) === NS.L7.shape.linearize(mid, shape, st, 0);
    },
  });
  reg(16, {
    name: 'linearize · 单维 +1 增量', intuition: '某维下标 +1（其余为 0），扁平偏移恰好增加该维步长。',
    math: 'lin(e_j) = st[j]（s[j]>1）', pseudocode: 'for each j with s[j]>1: assert', complexity: 'O(n²)',
    ref: 'numpy ndarray strides', commonPitfall: 's[j]=1 时该维唯一合法下标是 0，无 +1 可言。',
    invariant: '对任意正整型 shape 与每个 >1 的维 j：零下标 + e_j 的线性化 === st[j]',
    inputs: ['shape'], domains: { shape: 'shape3' },
    run: (shape) => {
      const st = NS.L7.shape.contiguousStrides(shape);
      for (let j = 0; j < shape.length; j++) {
        if (shape[j] > 1) { const idx = zerosOf(shape.length); idx[j] = 1; return NS.L7.shape.linearize(idx, shape, st); }
      }
      return 0;
    },
    predicate: (shape) => {
      const st = NS.L7.shape.contiguousStrides(shape);
      for (let j = 0; j < shape.length; j++) {
        if (shape[j] > 1) {
          const idx = zerosOf(shape.length); idx[j] = 1;
          if (NS.L7.shape.linearize(idx, shape, st) !== st[j]) return false;
        }
      }
      return true;
    },
  });
  reg(17, {
    name: 'linearize · 最大下标 ⇒ size−1', intuition: '每维取最大合法下标，得到最大扁平偏移 size−1。',
    math: 'lin(s-1, s, st) = size(s)−1', pseudocode: 'return lin(maxIdx) === size-1', complexity: 'O(n)',
    ref: 'numpy ndarray（C 序最大偏移）', commonPitfall: '越界下标（=s[i]）会抛错，须用 s[i]−1。',
    invariant: '对任意正整型 shape：全最大下标的线性化精确等于 size−1',
    inputs: ['shape'], domains: { shape: 'shape2' },
    run: (shape) => NS.L7.shape.linearize(shape.map((d) => d - 1), shape),
    predicate: (shape) => NS.L7.shape.linearize(shape.map((d) => d - 1), shape) === NS.L7.shape.size(shape) - 1,
  });
  reg(18, {
    name: 'linearize · 单热点线性可加', intuition: '两个不同维各放一个 1，扁平偏移等于各自增量之和。',
    math: 'lin(e_j+e_m) = lin(e_j)+lin(e_m)', pseudocode: 'assert additivity on two hot dims', complexity: 'O(n)',
    ref: '线性代数（点积的线性性）', commonPitfall: 'j=m 时不是可加性而是翻倍，须取两个不同维。',
    invariant: '对任意正整型 shape 与两个不同且 >1 的维 j,m：双热点线性化 === 两单热点之和',
    inputs: ['shape'], domains: { shape: 'shape4' },
    run: (shape) => {
      const st = NS.L7.shape.contiguousStrides(shape);
      let j = -1; let m = -1;
      for (let i = 0; i < shape.length; i++) {
        if (shape[i] > 1) { if (j < 0) j = i; else { m = i; break; } }
      }
      if (m < 0) return 0;
      const c = zerosOf(shape.length); c[j] = 1; c[m] = 1;
      return NS.L7.shape.linearize(c, shape, st);
    },
    predicate: (shape) => {
      const st = NS.L7.shape.contiguousStrides(shape);
      let j = -1; let m = -1;
      for (let i = 0; i < shape.length; i++) {
        if (shape[i] > 1) { if (j < 0) j = i; else { m = i; break; } }
      }
      if (j < 0 || m < 0) return true; // 少于两个 >1 维：前提不成立，命题空真
      const a = zerosOf(shape.length); a[j] = 1;
      const b = zerosOf(shape.length); b[m] = 1;
      const c = zerosOf(shape.length); c[j] = 1; c[m] = 1;
      const fa = NS.L7.shape.linearize(a, shape, st);
      const fb = NS.L7.shape.linearize(b, shape, st);
      const fc = NS.L7.shape.linearize(c, shape, st);
      return fc === fa + fb;
    },
  });
  reg(19, {
    name: 'linearize · 偏移平移', intuition: '偏移量整体平移，不与下标点积耦合。',
    math: 'lin(idx,s,st,o₁+o₂) = lin(idx,s,st,o₁)+o₂', pseudocode: 'compare shifted offsets', complexity: 'O(n)',
    ref: 'numpy ndarray（offset 语义）', commonPitfall: 'offset 为负表示视图回退，本域恒用正偏移。',
    invariant: '对任意正整型 shape：L7 linearize 对 offset 满足平移可加',
    inputs: ['shape'], domains: { shape: 'shape3' },
    run: (shape) => NS.L7.shape.linearize(midOf(shape), shape, null, 5),
    predicate: (shape) => {
      const st = NS.L7.shape.contiguousStrides(shape);
      const mid = midOf(shape);
      return NS.L7.shape.linearize(mid, shape, st, 5) === 5 + NS.L7.shape.linearize(mid, shape, st, 0);
    },
  });
  reg(20, {
    name: 'linearize · 中点偏移有界', intuition: '构造的中间下标线性化后必在 [0, size) 内。',
    math: '0 ≤ lin(mid) < size(s)', pseudocode: 'return f >= 0 && f < size', complexity: 'O(n)',
    ref: 'numpy ndarray', commonPitfall: '上界是严格小于；等于 size 即越界。',
    invariant: '对任意正整型 shape：构造中点的扁平偏移落在 [0, size)',
    inputs: ['shape'], domains: { shape: 'shape2' },
    run: (shape) => NS.L7.shape.linearize(midOf(shape), shape),
    predicate: (shape) => {
      const f = NS.L7.shape.linearize(midOf(shape), shape);
      return f >= 0 && f < NS.L7.shape.size(shape);
    },
  });
  reg(21, {
    name: 'linearize · 扁平↔下标往返', intuition: '最大扁平值按混合 radix 解开成下标，再线性化必还原。',
    math: 'lin(unravel(size−1)) = size−1', pseudocode: 'idx[i] = (f / st[i]) % s[i]; assert lin(idx) === f', complexity: 'O(n)',
    ref: 'numpy ndarray（unravel_index 语义）', commonPitfall: 'radix 权重必须是连续步长；乱序步长会错。',
    invariant: '对任意正整型 shape：size−1 经自实现 unravel 再 linearize 精确还原',
    inputs: ['shape'], domains: { shape: 'shape4' },
    run: (shape) => {
      const st = NS.L7.shape.contiguousStrides(shape);
      const f = NS.L7.shape.size(shape) - 1;
      const idx = new Array(shape.length);
      for (let i = 0; i < shape.length; i++) idx[i] = Math.floor(f / st[i]) % shape[i];
      return idx;
    },
    predicate: (shape) => {
      const st = NS.L7.shape.contiguousStrides(shape);
      const f = NS.L7.shape.size(shape) - 1;
      const idx = new Array(shape.length);
      for (let i = 0; i < shape.length; i++) idx[i] = Math.floor(f / st[i]) % shape[i];
      return NS.L7.shape.linearize(idx, shape, st) === f;
    },
  });
  reg(22, {
    name: 'linearize · 末维连续', intuition: '只动末维下标时，扁平偏移就等于该下标本身（步长 1）。',
    math: 'lin(0…0,k) = k（k < s[n-1]）', pseudocode: 'k = s[n-1]-1; assert lin === k', complexity: 'O(n)',
    ref: 'numpy ndarray strides（C 序末维连续）', commonPitfall: '该性质是行主序独有；列主序下首维才连续。',
    invariant: '对任意正整型 shape：末维取最大下标时线性化 === s[n-1]−1',
    inputs: ['shape'], domains: { shape: 'shape3' },
    run: (shape) => {
      const idx = zerosOf(shape.length); idx[shape.length - 1] = shape[shape.length - 1] - 1;
      return NS.L7.shape.linearize(idx, shape);
    },
    predicate: (shape) => {
      const n = shape.length;
      const k = shape[n - 1] - 1;
      const idx = zerosOf(n); idx[n - 1] = k;
      return NS.L7.shape.linearize(idx, shape) === k;
    },
  });
  reg(23, {
    name: 'linearize · 首维步长即后缀 size', intuition: '首维下标 1 的扁平偏移就是首维步长，也即后缀元素数。',
    math: 'lin(e₀) = st[0] = size(s[1:])', pseudocode: 'assert both sides', complexity: 'O(n)',
    ref: 'numpy ndarray strides', commonPitfall: '一维形状时后缀为空、size 为 1，结论仍成立；但首维为 1 时「下标 1」越界会抛 —— 故本项域声明为首维 ≥2（shape2f2）。',
    invariant: '对首维 ≥2 的正整型 shape：lin(e₀) === st[0] === size(s[1:])',
    inputs: ['shape'], domains: { shape: 'shape2f2' },
    run: (shape) => { const idx = zerosOf(shape.length); idx[0] = 1; return NS.L7.shape.linearize(idx, shape); },
    predicate: (shape) => {
      const st = NS.L7.shape.contiguousStrides(shape);
      const idx = zerosOf(shape.length); idx[0] = 1;
      return NS.L7.shape.linearize(idx, shape, st) === st[0] && st[0] === NS.L7.shape.size(shape.slice(1));
    },
  });
  reg(24, {
    name: 'linearize · 分块恒等式', intuition: '拼接形状 a∥b 的下标偏移 = a 内偏移 × size(b) + b 内偏移。',
    math: 'lin([i,j], a∥b) = lin(i,a)·size(b) + lin(j,b)', pseudocode: 'construct b = copy(a); assert', complexity: 'O(n)',
    ref: 'numpy ndarray（行主序分块布局）', commonPitfall: '步长须用拼接后的连续步长，不能沿用 a 的。',
    invariant: '对任意正整型 shape：拼接分块线性化恒等式精确成立',
    inputs: ['shape'], domains: { shape: 'shape4' },
    run: (shape) => NS.L7.shape.linearize(midOf(shape).concat(midOf(shape)), shape.concat(shape.slice())),
    predicate: (shape) => {
      const a = shape; const b = shape.slice();
      const ia = midOf(a); const ib = midOf(b);
      const s = a.concat(b);
      const lhs = NS.L7.shape.linearize(ia.concat(ib), s, NS.L7.shape.contiguousStrides(s));
      const rhs = NS.L7.shape.linearize(ia, a) * NS.L7.shape.size(b) + NS.L7.shape.linearize(ib, b);
      return lhs === rhs;
    },
  });

  // ── size 与步长的组合恒等式（E025–E027）─────────────────────────────
  reg(25, {
    name: 'size · 拼接乘性', intuition: '两块接起来的元素数 = 两块各自元素数之积。',
    math: 'size(a∥a) = size(a)²', pseudocode: 'return size(concat) === size(shape)**2', complexity: 'O(n)',
    ref: 'numpy ndarray.size', commonPitfall: '勿把拼接当相加；形状拼接是乘法性的。',
    invariant: '对任意正整型 shape：L7 size(shape∥shape) === size(shape)²',
    inputs: ['shape'], domains: { shape: 'shape3' },
    run: (shape) => NS.L7.shape.size(shape.concat(shape.slice())),
    predicate: (shape) => NS.L7.shape.size(shape.concat(shape.slice())) === NS.L7.shape.size(shape) * NS.L7.shape.size(shape),
  });
  reg(26, {
    name: '连续步长 · 拼接分块', intuition: '拼接后前段的步长整体放大 size(b) 倍，后段步长原样。',
    math: 'strides(a∥b)[i] = strides(a)[i]·size(b)（i<n）', pseudocode: 'for i<n: assert scaled', complexity: 'O(n)',
    ref: 'numpy ndarray（行主序分块步长）', commonPitfall: '后段（i≥n）步长不受前缀影响，勿一起缩放。',
    invariant: '对任意正整型 shape：拼接形状前 n 项步长 === 原步长 × size(shape)',
    inputs: ['shape'], domains: { shape: 'shape2' },
    run: (shape) => NS.L7.shape.contiguousStrides(shape.concat(shape.slice())),
    predicate: (shape) => {
      const b = shape.slice();
      const stS = NS.L7.shape.contiguousStrides(shape.concat(b));
      const stA = NS.L7.shape.contiguousStrides(shape);
      const szB = NS.L7.shape.size(b);
      for (let i = 0; i < shape.length; i++) if (stS[i] !== stA[i] * szB) return false;
      return true;
    },
  });
  reg(27, {
    name: 'size · 不小于最大维', intuition: '各维 ≥1，故总数不小于任何单维，也至少为 1。',
    math: 'size(s) ≥ max(s) ∧ size(s) ≥ 1', pseudocode: 'return size >= max(shape)', complexity: 'O(n)',
    ref: 'numpy ndarray.size', commonPitfall: '含 0 维时 size 为 0，该下界破裂；本域输入恒 ≥1。',
    invariant: '对任意正整型 shape：L7 size ≥ 最大维且 ≥ 1',
    inputs: ['shape'], domains: { shape: 'shape4' },
    run: (shape) => NS.L7.shape.size(shape),
    predicate: (shape) => {
      const sz = NS.L7.shape.size(shape);
      let mx = 1;
      for (let i = 0; i < shape.length; i++) if (shape[i] > mx) mx = shape[i];
      return sz >= mx && sz >= 1;
    },
  });

  // ── 参数化轴不变量（E028–E032，第二输入 nat 选轴）────────────────────
  reg(28, {
    name: '参数化 · 任意轴后缀积', intuition: '外部任意选的轴（取模落到合法范围）仍满足后缀积公式。',
    math: 'st[k%n] = ∏_{j>k%n} s[j]', pseudocode: 'j = k % n; assert st[j] === prod(shape, j+1)', complexity: 'O(n)',
    ref: 'numpy ndarray strides', commonPitfall: 'k 可能远超轴数，必须先取模；负轴本域不产生。',
    invariant: '对任意正整型 shape 与 nat k：L7 连续步长在轴 k%n 上等于后缀积',
    inputs: ['shape', 'k'], domains: { shape: 'shape3', k: 'nat' },
    run: (shape, k) => NS.L7.shape.contiguousStrides(shape)[k % shape.length],
    predicate: (shape, k) => {
      const j = k % shape.length;
      const st = NS.L7.shape.contiguousStrides(shape);
      return st[j] === prod(shape, j + 1);
    },
  });
  reg(29, {
    name: '参数化 · 任意轴切分乘性', intuition: '任意轴处切开，前后缀 size 之积仍是总 size。',
    math: 'size(s[:j+1])·size(s[j+1:]) = size(s)', pseudocode: 'j = k % n; assert product', complexity: 'O(n)',
    ref: 'numpy ndarray.size', commonPitfall: 'j = n-1 时后缀为空、size 为 1，勿当 0。',
    invariant: '对任意正整型 shape 与 nat k：轴 k%n 处切分满足 size 乘性',
    inputs: ['shape', 'k'], domains: { shape: 'shape2', k: 'nat' },
    run: (shape, k) => NS.L7.shape.size(shape.slice(0, (k % shape.length) + 1)),
    predicate: (shape, k) => {
      const j = k % shape.length;
      const pre = NS.L7.shape.size(shape.slice(0, j + 1));
      const suf = NS.L7.shape.size(shape.slice(j + 1));
      return pre * suf === NS.L7.shape.size(shape);
    },
  });
  reg(30, {
    name: '参数化 · 任意轴 +1 增量', intuition: '指定轴（取模）下标 +1，扁平偏移增加该轴步长。',
    math: 's[j]>1 ⇒ lin(e_j) = st[j]', pseudocode: 'j = k % n; if s[j]>1 assert', complexity: 'O(n)',
    ref: 'numpy ndarray strides', commonPitfall: 's[j]=1 时该轴无 +1 合法下标，命题空真。',
    invariant: '对任意正整型 shape 与 nat k：若轴 k%n 大小 >1，则 e_j 线性化 === st[j]',
    inputs: ['shape', 'k'], domains: { shape: 'shape3', k: 'nat' },
    run: (shape, k) => {
      const j = k % shape.length;
      if (shape[j] <= 1) return 0;
      const st = NS.L7.shape.contiguousStrides(shape);
      const idx = zerosOf(shape.length); idx[j] = 1;
      return NS.L7.shape.linearize(idx, shape, st);
    },
    predicate: (shape, k) => {
      const j = k % shape.length;
      if (shape[j] <= 1) return true;
      const st = NS.L7.shape.contiguousStrides(shape);
      const idx = zerosOf(shape.length); idx[j] = 1;
      return NS.L7.shape.linearize(idx, shape, st) === st[j];
    },
  });
  reg(31, {
    name: '参数化 · 任意轴上界偏移', intuition: '指定轴取最大下标，偏移 = (s[j]−1)·st[j] 且严格小于 size。',
    math: 'lin(e_j·(s[j]−1)) = (s[j]−1)·st[j] < size', pseudocode: 'j = k % n; assert both', complexity: 'O(n)',
    ref: 'numpy ndarray strides', commonPitfall: '上界严格小于 size；取等即越界。',
    invariant: '对任意正整型 shape 与 nat k：轴 k%n 的最大下标偏移 === (s[j]−1)·st[j] 且 < size',
    inputs: ['shape', 'k'], domains: { shape: 'shape4', k: 'nat' },
    run: (shape, k) => {
      const j = k % shape.length;
      const idx = zerosOf(shape.length); idx[j] = shape[j] - 1;
      return NS.L7.shape.linearize(idx, shape);
    },
    predicate: (shape, k) => {
      const j = k % shape.length;
      const st = NS.L7.shape.contiguousStrides(shape);
      const idx = zerosOf(shape.length); idx[j] = shape[j] - 1;
      const f = NS.L7.shape.linearize(idx, shape, st);
      return f === (shape[j] - 1) * st[j] && f < NS.L7.shape.size(shape);
    },
  });
  reg(32, {
    name: 'size · 首维整除刻画', intuition: '首维步长恰为 size 除以首维，无余数——可整除性判据。',
    math: 'size % s[0] = 0 ∧ st[0] = size / s[0]', pseudocode: 'return st[0]*s[0]===size && st[0]===size/s[0]', complexity: 'O(n)',
    ref: 'numpy ndarray strides', commonPitfall: '浮点除法有误差时勿用 ==；此处整数精确。',
    invariant: '对任意正整型 shape：L7 size 被首维整除且商等于首维步长',
    inputs: ['shape'], domains: { shape: 'shape2' },
    run: (shape) => NS.L7.shape.size(shape) / shape[0],
    predicate: (shape) => {
      const st = NS.L7.shape.contiguousStrides(shape);
      const sz = NS.L7.shape.size(shape);
      return sz % shape[0] === 0 && st[0] === sz / shape[0] && st[0] * shape[0] === sz;
    },
  });
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

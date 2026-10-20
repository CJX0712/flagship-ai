/* ============================================================================
 * L10 微指令 · E 域（形状与索引）· 第 3/3 批（E065–E096）
 * ----------------------------------------------------------------------------
 * 所属层：L10-microIsa  域：E（形状与索引）
 * 允许依赖的层：L7（shape 代数）、L9（rng）、L10（registry/invariant）
 * 主要 @ref：numpy ndarray strides；broadcasting 复合律；置换群；行主序 C order
 *
 * 每项含七件套（直觉/数学/伪代码/复杂度/论文/常见坑/不变量）+ run + predicate。
 * 判据：整数精确，一律 === 与逐元素深比（eqArr）。
 * 结构：E065–E070 前/后缀积；E071–E076 边界与单调性；E077–E084 多形状
 *       组合律（第二形状一律在 predicate 内构造）；E085–E090 形状代数；
 *       E091–E096 linearize 深化（齐次性/可加性/分解）。
 * 加载顺序注意：本文件先于 L7 求值，NS.L7.shape 一律惰性引用。
 * ==========================================================================*/
;(function (NS) {
  'use strict';
  NS.L10 = NS.L10 || {};
  function reg(i, o) { NS.L10.register('E', i, o); }

  // ── 本地第一性原理助手 ──────────────────────────────────────────────
  function eqArr(a, b) {
    if (!a || !b || a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
    return true;
  }
  function zerosOf(n) { const a = new Array(n); for (let i = 0; i < n; i++) a[i] = 0; return a; }
  function midOf(shape) { const a = new Array(shape.length); for (let i = 0; i < shape.length; i++) a[i] = shape[i] >> 1; return a; }
  function identAxes(n) { const a = new Array(n); for (let i = 0; i < n; i++) a[i] = i; return a; }
  function reverseAxes(n) { const a = new Array(n); for (let i = 0; i < n; i++) a[i] = n - 1 - i; return a; }
  // 后缀积：suffix[i] = ∏_{j>i} s[j]（末位为空积 1）；前缀积：prefix[i] = ∏_{j≤i} s[j]。
  function suffixProds(shape) {
    const n = shape.length; const out = new Array(n); let acc = 1;
    for (let i = n - 1; i >= 0; i--) { out[i] = acc; acc *= shape[i]; }
    return out;
  }
  function prefixProds(shape) {
    const n = shape.length; const out = new Array(n); let acc = 1;
    for (let i = 0; i < n; i++) { acc *= shape[i]; out[i] = acc; }
    return out;
  }

  // ── 前/后缀积（E065–E070）──────────────────────────────────────────
  reg(65, {
    name: '后缀积 · 与 L7 步长一致', intuition: '后缀积就是行主序连续步长，二者应逐项相同。',
    math: 'suffix[i] ≡ st[i]', pseudocode: 'loop assert', complexity: 'O(n)',
    ref: 'numpy ndarray strides', commonPitfall: '末位后缀积为空积 1，勿算成 0。',
    invariant: '对任意正整型 shape：本地后缀积与 L7 连续步长逐项一致',
    inputs: ['shape'], domains: { shape: 'shape3' },
    run: (shape) => suffixProds(shape),
    predicate: (shape) => eqArr(suffixProds(shape), NS.L7.shape.contiguousStrides(shape)),
  });
  reg(66, {
    name: '后缀积 · 末位恒 1', intuition: '最后一维之后没有维度，空积定义为 1。',
    math: 'suffix[n-1] = 1', pseudocode: 'return suffix[last] === 1', complexity: 'O(n)',
    ref: '空积约定（乘法单位元）', commonPitfall: '把空积当 0 会让整条步长链塌掉。',
    invariant: '对任意正整型 shape：本地后缀积末位为 1',
    inputs: ['shape'], domains: { shape: 'shape2' },
    run: (shape) => suffixProds(shape)[shape.length - 1],
    predicate: (shape) => suffixProds(shape)[shape.length - 1] === 1,
  });
  reg(67, {
    name: '后缀积 · 递推关系', intuition: '每个后缀积 = 下一维大小 × 该维的后缀积。',
    math: 'suffix[i] = s[i+1]·suffix[i+1]', pseudocode: 'for i<n-1: assert', complexity: 'O(n)',
    ref: 'numpy ndarray strides', commonPitfall: '是「下一维」s[i+1] 而非本维 s[i] —— 由 suffix[i]=∏_{j>i}s[j] 直接展开即得；写成 s[i] 会恒不成立（除非恰有 s[i]=s[i+1]·suffix[i+1]）。递推止于 n-2，末位无后继。',
    invariant: '对任意正整型 shape：本地后缀积满足逐维递推 suffix[i] = s[i+1]·suffix[i+1]',
    inputs: ['shape'], domains: { shape: 'shape4' },
    run: (shape) => suffixProds(shape),
    predicate: (shape) => {
      const suf = suffixProds(shape);
      for (let i = 0; i < shape.length - 1; i++) if (suf[i] !== shape[i + 1] * suf[i + 1]) return false;
      return true;
    },
  });
  reg(68, {
    name: '前缀积 · 末位为 size', intuition: '前缀积累到最后一维就是全部维度之积，即元素总数。',
    math: 'prefix[n-1] = size(s)', pseudocode: 'assert prefix[last] === L7size', complexity: 'O(n)',
    ref: 'numpy ndarray.size', commonPitfall: '前缀积与步长不同物，勿与 st 混用。',
    invariant: '对任意正整型 shape：本地前缀积末位 === L7 size',
    inputs: ['shape'], domains: { shape: 'shape3' },
    run: (shape) => prefixProds(shape)[shape.length - 1],
    predicate: (shape) => prefixProds(shape)[shape.length - 1] === NS.L7.shape.size(shape),
  });
  reg(69, {
    name: '前缀积 · 首维为 shape[0]', intuition: '第一个前缀积只含首维，就是首维大小。',
    math: 'prefix[0] = s[0]', pseudocode: 'return prefix[0] === shape[0]', complexity: 'O(n)',
    ref: '前缀积定义', commonPitfall: '空形状无前缀积；本域输入长度 ≥2。',
    invariant: '对任意正整型 shape：本地前缀积首位 === shape[0]',
    inputs: ['shape'], domains: { shape: 'shape2' },
    run: (shape) => prefixProds(shape)[0],
    predicate: (shape) => prefixProds(shape)[0] === shape[0],
  });
  reg(70, {
    name: '前后缀积 · 乘积为 size', intuition: '任一轴处：prefix[i] × suffix[i] 恰好覆盖全部维度（不含重叠、不漏维）。',
    math: 'prefix[i]·suffix[i] = size(s)', pseudocode: 'for i<n: assert', complexity: 'O(n)',
    ref: 'numpy ndarray.size', commonPitfall: '两个因子必须用同一轴 i：prefix[i]=∏_{j≤i}、suffix[i]=∏_{j>i}，合起来才是全体；写成 prefix[i]·suffix[i+1] 会漏掉第 i+1 维（差一个因子 s[i+1]）。末轴 i=n−1 时 suffix[n−1]=1（空积），仍成立。',
    invariant: '对任意正整型 shape：每个轴处 prefix[i]·suffix[i] === L7 size',
    inputs: ['shape'], domains: { shape: 'shape4' },
    run: (shape) => prefixProds(shape),
    predicate: (shape) => {
      const pre = prefixProds(shape); const suf = suffixProds(shape);
      const sz = NS.L7.shape.size(shape);
      for (let i = 0; i < shape.length; i++) if (pre[i] * suf[i] !== sz) return false;
      return true;
    },
  });

  // ── 边界与单调性（E071–E076）───────────────────────────────────────
  reg(71, {
    name: '边界 · 中点下标逐维合法', intuition: '每维取半（floor），必落在 [0, s[i]) 内。',
    math: '0 ≤ s[i]>>1 < s[i]', pseudocode: 'loop assert', complexity: 'O(n)',
    ref: 'numpy ndarray（下标域）', commonPitfall: 's[i]=1 时半值为 0，仍合法，勿判越界。',
    invariant: '对任意正整型 shape：构造中点下标逐维满足 0 ≤ mid[i] < s[i]',
    inputs: ['shape'], domains: { shape: 'shape3' },
    run: (shape) => midOf(shape),
    predicate: (shape) => {
      const mid = midOf(shape);
      for (let i = 0; i < shape.length; i++) if (!(mid[i] >= 0 && mid[i] < shape[i])) return false;
      return true;
    },
  });
  reg(72, {
    name: '边界 · L7 越界即抛', intuition: '下标等于维度大小或为负，L7 linearize 必须抛错。',
    math: 'k ≥ s[i] ∨ k < 0 ⇒ throw', pseudocode: 'two OOB indices; both must throw', complexity: 'O(n)',
    ref: 'numpy ndarray（IndexError 语义）', commonPitfall: '只测一侧（上界）不够；负下标同样要抛。',
    invariant: '对任意正整型 shape：L7 linearize 对上界与负下标构造均抛错',
    inputs: ['shape'], domains: { shape: 'shape2' },
    run: (shape) => {
      const oob = midOf(shape); oob[0] = shape[0];
      try { NS.L7.shape.linearize(oob, shape); return 'no-throw'; } catch (e) { return 'threw'; }
    },
    predicate: (shape) => {
      const hi = midOf(shape); hi[0] = shape[0]; // = s[0]，上界外
      const lo = midOf(shape); lo[0] = -1;        // 负下标
      let threwHi = false; let threwLo = false;
      try { NS.L7.shape.linearize(hi, shape); } catch (e) { threwHi = true; }
      try { NS.L7.shape.linearize(lo, shape); } catch (e) { threwLo = true; }
      return threwHi && threwLo;
    },
  });
  reg(73, {
    name: '边界 · 中点偏移有界', intuition: '中点下标的扁平偏移必须落在 [0, size) 半开区间。',
    math: '0 ≤ lin(mid) < size(s)', pseudocode: 'assert range', complexity: 'O(n)',
    ref: 'numpy ndarray', commonPitfall: '上界严格；等于 size 即越界。',
    invariant: '对任意正整型 shape：L7 linearize(中点) ∈ [0, size)',
    inputs: ['shape'], domains: { shape: 'shape4' },
    run: (shape) => NS.L7.shape.linearize(midOf(shape), shape),
    predicate: (shape) => {
      const f = NS.L7.shape.linearize(midOf(shape), shape);
      return f >= 0 && f < NS.L7.shape.size(shape);
    },
  });
  reg(74, {
    name: '边界 · 最小扁平偏移为 0', intuition: '全零下标对应存储起点，偏移为 0（无 offset 时）。',
    math: 'lin(0ⁿ) = 0', pseudocode: 'return lin(zeros) === 0', complexity: 'O(n)',
    ref: 'numpy ndarray', commonPitfall: '传入非零 offset 时结果为 offset，勿忘平移。',
    invariant: '对任意正整型 shape：L7 linearize(全零) === 0',
    inputs: ['shape'], domains: { shape: 'shape3' },
    run: (shape) => NS.L7.shape.linearize(zerosOf(shape.length), shape),
    predicate: (shape) => NS.L7.shape.linearize(zerosOf(shape.length), shape) === 0,
  });
  reg(75, {
    name: '边界 · 首维跨度', intuition: '首维从 0 到最大，跨过的偏移恰为 size 减去后缀 size。',
    math: '(s[0]−1)·st[0] = size(s) − size(s[1:])', pseudocode: 'assert both sides', complexity: 'O(n)',
    ref: 'numpy ndarray strides', commonPitfall: 'st[0] 就是 size(s[1:])，别重复累乘。',
    invariant: '对任意正整型 shape：首维最大增量 === size − 后缀 size',
    inputs: ['shape'], domains: { shape: 'shape2' },
    run: (shape) => (shape[0] - 1) * NS.L7.shape.contiguousStrides(shape)[0],
    predicate: (shape) => {
      const st = NS.L7.shape.contiguousStrides(shape);
      const lhs = (shape[0] - 1) * st[0];
      const rhs = NS.L7.shape.size(shape) - NS.L7.shape.size(shape.slice(1));
      return lhs === rhs;
    },
  });
  reg(76, {
    name: '边界 · 中点偏移单调不越顶', intuition: '中点逐维不超过最大下标，步长为正，故偏移不超最大偏移。',
    math: 'lin(mid) ≤ lin(s−1)', pseudocode: 'assert <=', complexity: 'O(n)',
    ref: 'numpy ndarray（正步长单调性）', commonPitfall: '负步长（翻转视图）下单调性反向。',
    invariant: '对任意正整型 shape：L7 linearize(中点) ≤ linearize(最大下标)',
    inputs: ['shape'], domains: { shape: 'shape4' },
    run: (shape) => NS.L7.shape.linearize(midOf(shape), shape),
    predicate: (shape) => {
      const fMid = NS.L7.shape.linearize(midOf(shape), shape);
      const fMax = NS.L7.shape.linearize(shape.map((d) => d - 1), shape);
      return fMid <= fMax;
    },
  });

  // ── 多形状组合律（E077–E084，第二形状一律由第一形状构造）────────────
  reg(77, {
    name: '多形状 · size 结合律', intuition: '三块拼接的 size 既满足连乘也满足拼接结合律。',
    math: 'size(a∥b∥c) = size(a)·size(b)·size(c)', pseudocode: 'b = a+1, c = [2,3]; assert both', complexity: 'O(n)',
    ref: 'numpy ndarray.size', commonPitfall: '勿把结合律与交换律混同；拼接不交换。',
    invariant: '对任意正整型 shape：L7 size 对构造三块满足连乘与拼接结合律',
    inputs: ['shape'], domains: { shape: 'shape3' },
    run: (shape) => NS.L7.shape.size(shape.concat(shape.map((d) => d + 1)).concat([2, 3])),
    predicate: (shape) => {
      const a = shape; const b = shape.map((d) => d + 1); const c = [2, 3];
      const left = NS.L7.shape.size(a.concat(b).concat(c));
      const assoc = NS.L7.shape.size(a.concat(b.concat(c)));
      const prod3 = NS.L7.shape.size(a) * NS.L7.shape.size(b) * NS.L7.shape.size(c);
      return left === assoc && left === prod3;
    },
  });
  reg(78, {
    name: '多形状 · 拼接步长尾块', intuition: '拼接后段（原 b 的部分）步长不受前缀影响，原样保留。',
    math: 'strides(a∥b)[i] = strides(b)[i−n]（i≥n）', pseudocode: 'b = copy(a); loop assert tail', complexity: 'O(n)',
    ref: 'numpy ndarray（行主序分块步长）', commonPitfall: '前段步长要乘 size(b)，尾段不乘——两段规则不同。',
    invariant: '对任意正整型 shape：L7 拼接步长尾块与 b 自身步长逐项一致',
    inputs: ['shape'], domains: { shape: 'shape2' },
    run: (shape) => NS.L7.shape.contiguousStrides(shape.concat(shape.slice())),
    predicate: (shape) => {
      const b = shape.slice();
      const stS = NS.L7.shape.contiguousStrides(shape.concat(b));
      const stB = NS.L7.shape.contiguousStrides(b);
      for (let i = 0; i < b.length; i++) if (stS[shape.length + i] !== stB[i]) return false;
      return true;
    },
  });
  reg(79, {
    name: '多形状 · 广播与拼接 size', intuition: '拼接形状与同形全 1 广播，size 等于两块 size 之积。',
    math: 'size(broadcast(a∥b, 1)) = size(a)·size(b)', pseudocode: 'b = copy(a); assert', complexity: 'O(n)',
    ref: 'numpy broadcasting', commonPitfall: '全 1 形状长度必须与拼接后一致，否则是另一回事。',
    invariant: '对任意正整型 shape：L7 broadcast(shape∥shape, 全1) 的 size === size(shape)²',
    inputs: ['shape'], domains: { shape: 'shape4' },
    run: (shape) => NS.L7.shape.broadcastShapes(shape.concat(shape.slice()), shape.concat(shape.slice()).map(() => 1)),
    predicate: (shape) => {
      const b = shape.slice();
      const s = shape.concat(b);
      const bc = NS.L7.shape.broadcastShapes(s, s.map(() => 1));
      return NS.L7.shape.size(bc) === NS.L7.shape.size(shape) * NS.L7.shape.size(b);
    },
  });
  reg(80, {
    name: '多形状 · 广播交换律（不等长）', intuition: '不等长的兼容对（短的对齐尾部）广播结果与顺序无关。',
    math: 'broadcast(a,b) = broadcast(b,a)（b = a[1:]）', pseudocode: 'assert eqArr', complexity: 'O(n)',
    ref: 'numpy broadcasting', commonPitfall: '交换律前提是兼容；不兼容两侧都抛，谈不上相等。',
    invariant: '对任意正整型 shape：L7 广播对 (shape, 去首维) 满足交换律',
    inputs: ['shape'], domains: { shape: 'shape3' },
    run: (shape) => NS.L7.shape.broadcastShapes(shape, shape.slice(1)),
    predicate: (shape) => {
      const b = shape.slice(1);
      return eqArr(NS.L7.shape.broadcastShapes(shape, b), NS.L7.shape.broadcastShapes(b, shape));
    },
  });
  reg(81, {
    name: '多形状 · 广播结合律', intuition: '三级广播按不同顺序结合，结果相同（构造尾段与前置 1 段）。',
    math: 'bcast(bcast(a,b),c) = bcast(a,bcast(b,c))', pseudocode: 'b = a[1:], c = [1]∥a[1:]; assert', complexity: 'O(n)',
    ref: 'numpy broadcasting', commonPitfall: '结合律要求每对中间结果都兼容；构造已保证。',
    invariant: '对任意正整型 shape：L7 广播对构造三级链满足结合律',
    inputs: ['shape'], domains: { shape: 'shape2' },
    run: (shape) => NS.L7.shape.broadcastShapes(NS.L7.shape.broadcastShapes(shape, shape.slice(1)), [1].concat(shape.slice(1))),
    predicate: (shape) => {
      const b = shape.slice(1);
      const c = [1].concat(shape.slice(1));
      const lhs = NS.L7.shape.broadcastShapes(NS.L7.shape.broadcastShapes(shape, b), c);
      const rhs = NS.L7.shape.broadcastShapes(shape, NS.L7.shape.broadcastShapes(b, c));
      return eqArr(lhs, rhs);
    },
  });
  reg(82, {
    name: '多形状 · 1 维吸收链', intuition: '连续与全 1 形状广播两次，结果始终是原形状。',
    math: 'bcast(bcast(s,1),1) = s', pseudocode: 'assert both equal shape', complexity: 'O(n)',
    ref: 'numpy broadcasting', commonPitfall: '第二次广播的 1 形状要由第一次结果构造，长度才对。',
    invariant: '对任意正整型 shape：L7 广播链两次吸收全 1 后仍为原形状',
    inputs: ['shape'], domains: { shape: 'shape4' },
    run: (shape) => {
      const bc1 = NS.L7.shape.broadcastShapes(shape, shape.map(() => 1));
      return NS.L7.shape.broadcastShapes(bc1, bc1.map(() => 1));
    },
    predicate: (shape) => {
      const bc1 = NS.L7.shape.broadcastShapes(shape, shape.map(() => 1));
      const bc2 = NS.L7.shape.broadcastShapes(bc1, bc1.map(() => 1));
      return eqArr(bc1, shape) && eqArr(bc2, shape);
    },
  });
  reg(83, {
    name: '多形状 · 置换与广播可交换', intuition: '先广播再置换 ≡ 先置换再广播（全 1 形状置换后仍全 1）。',
    math: 'perm(bcast(s,1)) = bcast(perm(s), perm(1))', pseudocode: 'axes = reverse; assert shapes', complexity: 'O(n)',
    ref: 'numpy transpose + broadcasting', commonPitfall: '只对齐形状层；L7 广播不产步长，无法比对步长层。另注意 broadcastShapes 直接返回形状数组（不是 {shape} 对象），不可写 p2.shape。',
    invariant: '对任意正整型 shape：L7 置换与广播（全 1）在形状层可交换',
    inputs: ['shape'], domains: { shape: 'shape3' },
    run: (shape) => NS.L7.shape.permute(NS.L7.shape.broadcastShapes(shape, shape.map(() => 1)), NS.L7.shape.contiguousStrides(shape), reverseAxes(shape.length)).shape,
    predicate: (shape) => {
      const st = NS.L7.shape.contiguousStrides(shape);
      const ones = shape.map(() => 1);
      const axes = reverseAxes(shape.length);
      const p1 = NS.L7.shape.permute(NS.L7.shape.broadcastShapes(shape, ones), st, axes);
      const pa = NS.L7.shape.permute(shape, st, axes);
      const po = NS.L7.shape.permute(ones, NS.L7.shape.contiguousStrides(ones), axes);
      const p2 = NS.L7.shape.broadcastShapes(pa.shape, po.shape);
      // broadcastShapes 返回形状数组本身（permute 才返回 {shape,stride} 对象）
      return eqArr(p1.shape, p2);
    },
  });
  reg(84, {
    name: '多形状 · 广播 size 不减', intuition: '广播只把 1 放大成更大维，size 不会缩水。',
    math: 'size(bcast(a,b)) ≥ size(a), size(b)', pseudocode: 'b = 偶数维 1; assert both', complexity: 'O(n)',
    ref: 'numpy broadcasting', commonPitfall: '不减不等于严格增；全 1 对手下取等。',
    invariant: '对任意正整型 shape：L7 广播结果 size 不小于双方 size',
    inputs: ['shape'], domains: { shape: 'shape2' },
    run: (shape) => NS.L7.shape.broadcastShapes(shape, shape.map((d, i) => (i % 2 === 0 ? 1 : d))),
    predicate: (shape) => {
      const b = shape.map((d, i) => (i % 2 === 0 ? 1 : d));
      const sz = NS.L7.shape.size(NS.L7.shape.broadcastShapes(shape, b));
      return sz >= NS.L7.shape.size(shape) && sz >= NS.L7.shape.size(b);
    },
  });

  // ── 形状代数（E085–E090）───────────────────────────────────────────
  reg(85, {
    name: '形状代数 · 拼接单位元', intuition: '空形状是拼接的单位元：接上去什么都不变。',
    math: 'a∥[] = a', pseudocode: 'assert size & length', complexity: 'O(n)',
    ref: '自由幺半群（形状拼接）', commonPitfall: '单位元在 size 层是 1，在长度层是 0，两层面都要查。',
    invariant: '对任意正整型 shape：L7 size 与长度在拼接空形状后均不变',
    inputs: ['shape'], domains: { shape: 'shape4' },
    run: (shape) => shape.concat([]),
    predicate: (shape) => {
      const s2 = shape.concat([]);
      return s2.length === shape.length && NS.L7.shape.size(s2) === NS.L7.shape.size(shape);
    },
  });
  reg(86, {
    name: '形状代数 · 逆序保 size', intuition: '维度次序颠倒不改变元素总数（乘法交换）。',
    math: 'size(reverse(s)) = size(s)', pseudocode: 'assert', complexity: 'O(n)',
    ref: 'numpy ndarray.size', commonPitfall: 'size 同但步长表通常不同，勿混谈。',
    invariant: '对任意正整型 shape：L7 size 在逆序后不变',
    inputs: ['shape'], domains: { shape: 'shape3' },
    run: (shape) => NS.L7.shape.size(shape.slice().reverse()),
    predicate: (shape) => NS.L7.shape.size(shape.slice().reverse()) === NS.L7.shape.size(shape),
  });
  reg(87, {
    name: '形状代数 · 排序保 size', intuition: '把维度按大小排序，乘积不变。',
    math: 'size(sort(s)) = size(s)', pseudocode: 'assert', complexity: 'O(n log n)',
    ref: '乘法交换律', commonPitfall: '排序会改变轴语义（哪维是哪维），size 层无所谓。',
    invariant: '对任意正整型 shape：L7 size 在排序后不变',
    inputs: ['shape'], domains: { shape: 'shape2' },
    run: (shape) => NS.L7.shape.size(shape.slice().sort((x, y) => x - y)),
    predicate: (shape) => NS.L7.shape.size(shape.slice().sort((x, y) => x - y)) === NS.L7.shape.size(shape),
  });
  reg(88, {
    name: '形状代数 · 滤 1 保 size', intuition: '长度 1 的维是乘性单位元，滤掉不改变 size。',
    math: 'size(filter(s, d>1)) = size(s)', pseudocode: 'assert', complexity: 'O(n)',
    ref: 'numpy.squeeze（size 不变）', commonPitfall: 'squeeze 改变形状语义，size 层面才等价。',
    invariant: '对任意正整型 shape：L7 size 在滤除 1 维后不变',
    inputs: ['shape'], domains: { shape: 'shape4' },
    run: (shape) => NS.L7.shape.size(shape.filter((d) => d > 1)),
    predicate: (shape) => NS.L7.shape.size(shape.filter((d) => d > 1)) === NS.L7.shape.size(shape),
  });
  reg(89, {
    name: '形状代数 · 交换两维保 size', intuition: '任意交换两个维度（乘法交换律），size 不变。',
    math: 'size(swap(s,i,j)) = size(s)', pseudocode: 'i = k%n, j = (k>>5)%n; assert', complexity: 'O(n)',
    ref: '乘法交换律', commonPitfall: 'i=j 时是恒等交换，结论仍成立。',
    invariant: '对任意正整型 shape 与 nat k：交换轴 k%n 与 (k>>5)%n 后 L7 size 不变',
    inputs: ['shape', 'k'], domains: { shape: 'shape3', k: 'nat' },
    run: (shape, k) => {
      const s = shape.slice();
      const i = k % s.length; const j = (k >> 5) % s.length;
      const t = s[i]; s[i] = s[j]; s[j] = t;
      return NS.L7.shape.size(s);
    },
    predicate: (shape, k) => {
      const s = shape.slice();
      const i = k % s.length; const j = (k >> 5) % s.length;
      const t = s[i]; s[i] = s[j]; s[j] = t;
      return NS.L7.shape.size(s) === NS.L7.shape.size(shape);
    },
  });
  reg(90, {
    name: '形状代数 · size 为正整数', intuition: '正整数维的乘积仍是正整数，至少为 1。',
    math: 'size(s) ∈ ℤ⁺', pseudocode: 'return isInteger && >= 1', complexity: 'O(n)',
    ref: 'numpy ndarray.size', commonPitfall: '含 0 维时为 0，非正；本域输入恒 ≥1。',
    invariant: '对任意正整型 shape：L7 size 是 ≥1 的整数',
    inputs: ['shape'], domains: { shape: 'shape2' },
    run: (shape) => NS.L7.shape.size(shape),
    predicate: (shape) => {
      const sz = NS.L7.shape.size(shape);
      return Number.isInteger(sz) && sz >= 1;
    },
  });

  // ── linearize 深化（E091–E096）─────────────────────────────────────
  reg(91, {
    name: '线性化 · 步长缩放齐次性', intuition: '所有步长同乘 m，偏移整体放大 m 倍。',
    math: 'lin(idx,s,m·st) = m·lin(idx,s,st)', pseudocode: 'm = k%7+2; assert', complexity: 'O(n)',
    ref: '点积的齐次性', commonPitfall: 'm 必须为正；缩放步长不缩放 offset。',
    invariant: '对任意正整型 shape 与 nat k：L7 linearize 对步长缩放满足齐次性',
    inputs: ['shape', 'k'], domains: { shape: 'shape4', k: 'nat' },
    run: (shape, k) => {
      const m = (k % 7) + 2;
      const st = NS.L7.shape.contiguousStrides(shape).map((s) => s * m);
      return NS.L7.shape.linearize(midOf(shape), shape, st);
    },
    predicate: (shape, k) => {
      const m = (k % 7) + 2;
      const st = NS.L7.shape.contiguousStrides(shape).map((s) => s * m);
      const mid = midOf(shape);
      return NS.L7.shape.linearize(mid, shape, st) === m * NS.L7.shape.linearize(mid, shape);
    },
  });
  reg(92, {
    name: '线性化 · 偏移合并可加', intuition: '两个偏移先后传入等于一次性传入其和。',
    math: 'lin(idx,s,st,o₁+o₂) = lin(idx,s,st,o₁)+o₂', pseudocode: 'o1 = k%5+1, o2 = k%7+1; assert', complexity: 'O(n)',
    ref: 'numpy ndarray（offset 语义）', commonPitfall: '可加性对 offset 成立，对 stride 不成立。',
    invariant: '对任意正整型 shape 与 nat k：L7 linearize 对构造 offset 满足合并可加',
    inputs: ['shape', 'k'], domains: { shape: 'shape3', k: 'nat' },
    run: (shape, k) => NS.L7.shape.linearize(midOf(shape), shape, null, (k % 5) + 1 + (k % 7) + 1),
    predicate: (shape, k) => {
      const st = NS.L7.shape.contiguousStrides(shape);
      const mid = midOf(shape);
      const o1 = (k % 5) + 1; const o2 = (k % 7) + 1;
      return NS.L7.shape.linearize(mid, shape, st, o1 + o2) === NS.L7.shape.linearize(mid, shape, st, o1) + o2;
    },
  });
  reg(93, {
    name: '线性化 · 分量点积公式', intuition: '线性化的定义就是下标与步长的点积，逐项求和验证。',
    math: 'lin(idx,s,st) = Σ idx[i]·st[i]', pseudocode: 'local dot; assert', complexity: 'O(n)',
    ref: 'numpy ndarray strides', commonPitfall: '点积不含 offset；offset 是外加项。',
    invariant: '对任意正整型 shape：L7 linearize(中点) === 本地点积',
    inputs: ['shape'], domains: { shape: 'shape2' },
    run: (shape) => NS.L7.shape.linearize(midOf(shape), shape),
    predicate: (shape) => {
      const mid = midOf(shape);
      const st = NS.L7.shape.contiguousStrides(shape);
      let dot = 0;
      for (let i = 0; i < shape.length; i++) dot += mid[i] * st[i];
      return NS.L7.shape.linearize(mid, shape, st) === dot;
    },
  });
  reg(94, {
    name: '线性化 · 分量独立性', intuition: '把某一维下标置零，偏移恰好减去该维的贡献。',
    math: 'lin(idx−e_j·idx[j]) = lin(idx) − idx[j]·st[j]', pseudocode: 'j = k%n; assert', complexity: 'O(n)',
    ref: 'numpy ndarray strides', commonPitfall: 'idx[j] 为 0 时等式退化为 0=0，仍成立但无信息量。',
    invariant: '对任意正整型 shape 与 nat k：L7 linearize 对置零单维满足贡献扣除',
    inputs: ['shape', 'k'], domains: { shape: 'shape4', k: 'nat' },
    run: (shape, k) => {
      const j = k % shape.length;
      const idx = midOf(shape); idx[j] = 0;
      return NS.L7.shape.linearize(idx, shape);
    },
    predicate: (shape, k) => {
      const j = k % shape.length;
      const mid = midOf(shape);
      const idx = mid.slice(); idx[j] = 0;
      const st = NS.L7.shape.contiguousStrides(shape);
      return NS.L7.shape.linearize(idx, shape, st) === NS.L7.shape.linearize(mid, shape, st) - mid[j] * st[j];
    },
  });
  reg(95, {
    name: '线性化 · 全 1 形状唯一元素', intuition: '全 1 形状只有一个元素，任何合法下标（只有全零）偏移为 0。',
    math: 'size(1ⁿ)=1 ∧ lin(0ⁿ,1ⁿ)=0', pseudocode: 'ones = map(shape,()=>1); assert', complexity: 'O(n)',
    ref: 'numpy ndarray（0-d 类比物）', commonPitfall: '全 1 形状不是标量（0 维），但行为等价。',
    invariant: '对任意正整型 shape：其全 1 化的 L7 size 为 1 且零下标偏移为 0',
    inputs: ['shape'], domains: { shape: 'shape3' },
    run: (shape) => NS.L7.shape.linearize(zerosOf(shape.length), shape.map(() => 1)),
    predicate: (shape) => {
      const ones = shape.map(() => 1);
      return NS.L7.shape.size(ones) === 1 && NS.L7.shape.linearize(zerosOf(shape.length), ones) === 0;
    },
  });
  reg(96, {
    name: '线性化 · 末维连续分解', intuition: '总偏移 = 前 n-1 维贡献之和 + 末维下标（末维步长 1）。',
    math: 'lin(idx) = Σ_{i<n-1} idx[i]·st[i] + idx[n-1]', pseudocode: 'split sum; assert', complexity: 'O(n)',
    ref: 'numpy ndarray strides（C 序末维连续）', commonPitfall: '分解依赖 st[n-1]===1；非连续布局不成立。',
    invariant: '对任意正整型 shape：L7 linearize(中点) === 前缀贡献 + 末维下标',
    inputs: ['shape'], domains: { shape: 'shape2' },
    run: (shape) => NS.L7.shape.linearize(midOf(shape), shape),
    predicate: (shape) => {
      const n = shape.length;
      const mid = midOf(shape);
      const st = NS.L7.shape.contiguousStrides(shape);
      let pre = 0;
      for (let i = 0; i < n - 1; i++) pre += mid[i] * st[i];
      return NS.L7.shape.linearize(mid, shape, st) === pre + mid[n - 1];
    },
  });
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

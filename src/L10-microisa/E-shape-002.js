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
 *
 * ── 谓词纪律（本批全部重写，2026-10-24）─────────────────────────────────
 * 门禁只调 op.predicate，从不调 op.run（op-harness.js:43、probe-microisa.js:106）。
 * 原 32 项谓词**全部直接调 NS.L7.shape.* 或本地 helper，从不调本 op 的 run**
 * ——run 若损坏门禁完全失明（假绿）。现全部改为：先经 runOf('Exxx') 调 run
 * 本体，再对 run 的产出断言不变量。
 * oracle 全部取本文件内第一性原理实现，刻意与 run 异构：
 *   · prod(a, from)   —— 维度乘积
 *   · 前向递推刻画    —— 前缀积 pre[i]=pre[i−1]·s[i]、后缀积 suf[i]=suf[i−1]/s[i]
 *                        （与 run 的反向递推方向相反，构成独立基准）
 *   · 本地点积        —— Σ idx[i]·st[i]（独立于 L7.linearize）
 *   · bcastLocal(a,b) —— 自写尾部对齐广播
 *   · Math.floor(d/2) —— 与 run 的 d>>1（位运算）写法不同
 * ==========================================================================*/
;(function (NS) {
  'use strict';
  NS.L10 = NS.L10 || {};
  function reg(i, o) { NS.L10.register('E', i, o); }
  function runOf(key) { const o = NS.L10.ops[key]; return o && typeof o.run === 'function' ? o.run : null; }

  // ── 本地第一性原理助手 ──────────────────────────────────────────────
  function prod(a, from) { let n = 1; for (let i = from || 0; i < a.length; i++) n *= a[i]; return n; }
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
  // 本地点积（独立于 L7.linearize）：Σ idx[i]·st[i]。
  function dotIdx(idx, st) { let s = 0; for (let i = 0; i < idx.length; i++) s += idx[i] * st[i]; return s; }
  // 本地尾部对齐广播（不相容返回 null）。
  function bcastLocal(a, b) {
    const n = Math.max(a.length, b.length);
    const out = new Array(n);
    for (let i = 0; i < n; i++) {
      const da = i < a.length ? a[a.length - 1 - i] : 1;
      const db = i < b.length ? b[b.length - 1 - i] : 1;
      if (da !== db && da !== 1 && db !== 1) return null;
      out[n - 1 - i] = da === 1 ? db : da;
    }
    return out;
  }

  // ── 前/后缀积（E065–E070）──────────────────────────────────────────
  reg(65, {
    name: '后缀积 · 与 L7 步长一致', intuition: '后缀积就是行主序连续步长，二者应逐项相同。',
    math: 'suffix[i] ≡ st[i]', pseudocode: 'loop assert', complexity: 'O(n)',
    ref: 'numpy ndarray strides', commonPitfall: '末位后缀积为空积 1，勿算成 0。',
    invariant: 'run 满足前向递推 suf[i] = suf[i−1]/s[i] 且 suf[0]·s[0] = ∏s（独立于 run 的反向递推）',
    inputs: ['shape'], domains: { shape: 'shape3' },
    run: (shape) => suffixProds(shape),
    predicate: (shape) => {
      const suf = runOf('E065')(shape);
      if (suf.length !== shape.length) return false;
      if (suf[0] * shape[0] !== prod(shape, 0)) return false;
      for (let i = 1; i < shape.length; i++) if (suf[i] !== suf[i - 1] / shape[i]) return false;
      return true;
    },
  });
  reg(66, {
    name: '后缀积 · 末位恒 1', intuition: '最后一维之后没有维度，空积定义为 1。',
    math: 'suffix[n-1] = 1', pseudocode: 'return suffix[last] === 1', complexity: 'O(n)',
    ref: '空积约定（乘法单位元）', commonPitfall: '把空积当 0 会让整条步长链塌掉。',
    invariant: 'run === 1（末位后缀为空积）',
    inputs: ['shape'], domains: { shape: 'shape2' },
    run: (shape) => suffixProds(shape)[shape.length - 1],
    predicate: (shape) => runOf('E066')(shape) === 1,
  });
  reg(67, {
    name: '后缀积 · 递推关系', intuition: '每个后缀积 = 下一维大小 × 该维的后缀积。',
    math: 'suffix[i] = s[i+1]·suffix[i+1]', pseudocode: 'for i<n-1: assert', complexity: 'O(n)',
    ref: 'numpy ndarray strides', commonPitfall: '是「下一维」s[i+1] 而非本维 s[i] —— 由 suffix[i]=∏_{j>i}s[j] 直接展开即得；写成 s[i] 会恒不成立（除非恰有 s[i]=s[i+1]·suffix[i+1]）。递推止于 n-2，末位无后继。',
    invariant: 'run 满足后向递推 suf[i] = s[i+1]·suf[i+1] 且末位为 1（独立于 run 的乘积实现）',
    inputs: ['shape'], domains: { shape: 'shape4' },
    run: (shape) => suffixProds(shape),
    predicate: (shape) => {
      const suf = runOf('E067')(shape);
      if (suf.length !== shape.length) return false;
      if (suf[shape.length - 1] !== 1) return false;
      for (let i = 0; i < shape.length - 1; i++) if (suf[i] !== shape[i + 1] * suf[i + 1]) return false;
      return true;
    },
  });
  reg(68, {
    name: '前缀积 · 末位为 size', intuition: '前缀积累到最后一维就是全部维度之积，即元素总数。',
    math: 'prefix[n-1] = size(s)', pseudocode: 'assert prefix[last] === L7size', complexity: 'O(n)',
    ref: 'numpy ndarray.size', commonPitfall: '前缀积与步长不同物，勿与 st 混用。',
    invariant: 'run === ∏s（前缀积累到末位即总元素数）',
    inputs: ['shape'], domains: { shape: 'shape3' },
    run: (shape) => prefixProds(shape)[shape.length - 1],
    predicate: (shape) => runOf('E068')(shape) === prod(shape, 0),
  });
  reg(69, {
    name: '前缀积 · 首维为 shape[0]', intuition: '第一个前缀积只含首维，就是首维大小。',
    math: 'prefix[0] = s[0]', pseudocode: 'return prefix[0] === shape[0]', complexity: 'O(n)',
    ref: '前缀积定义', commonPitfall: '空形状无前缀积；本域输入长度 ≥2。',
    invariant: 'run === shape[0]（首个前缀积只含首维）',
    inputs: ['shape'], domains: { shape: 'shape2' },
    run: (shape) => prefixProds(shape)[0],
    predicate: (shape) => runOf('E069')(shape) === shape[0],
  });
  reg(70, {
    name: '前后缀积 · 乘积为 size', intuition: '任一轴处：prefix[i] × suffix[i] 恰好覆盖全部维度（不含重叠、不漏维）。',
    math: 'prefix[i]·suffix[i] = size(s)', pseudocode: 'for i<n: assert', complexity: 'O(n)',
    ref: 'numpy ndarray.size', commonPitfall: '两个因子必须用同一轴 i：prefix[i]=∏_{j≤i}、suffix[i]=∏_{j>i}，合起来才是全体；写成 prefix[i]·suffix[i+1] 会漏掉第 i+1 维（差一个因子 s[i+1]）。末轴 i=n−1 时 suffix[n−1]=1（空积），仍成立。',
    invariant: 'run 满足前向递推 pre[i] = pre[i−1]·s[i] 且 pre[0] = s[0]（独立于 run 的连乘实现）',
    inputs: ['shape'], domains: { shape: 'shape4' },
    run: (shape) => prefixProds(shape),
    predicate: (shape) => {
      const pre = runOf('E070')(shape);
      if (pre.length !== shape.length) return false;
      if (pre[0] !== shape[0]) return false;
      for (let i = 1; i < shape.length; i++) if (pre[i] !== pre[i - 1] * shape[i]) return false;
      return true;
    },
  });

  // ── 边界与单调性（E071–E076）───────────────────────────────────────
  reg(71, {
    name: '边界 · 中点下标逐维合法', intuition: '每维取半（floor），必落在 [0, s[i]) 内。',
    math: '0 ≤ s[i]>>1 < s[i]', pseudocode: 'loop assert', complexity: 'O(n)',
    ref: 'numpy ndarray（下标域）', commonPitfall: 's[i]=1 时半值为 0，仍合法，勿判越界。',
    invariant: 'run 逐维 === Math.floor(s[i]/2)（浮点除法，独立于 run 的位运算写法）',
    inputs: ['shape'], domains: { shape: 'shape3' },
    run: (shape) => midOf(shape),
    predicate: (shape) => {
      const mid = runOf('E071')(shape);
      if (mid.length !== shape.length) return false;
      for (let i = 0; i < shape.length; i++) {
        if (mid[i] !== Math.floor(shape[i] / 2)) return false;
        if (!(mid[i] >= 0 && mid[i] < shape[i])) return false;
      }
      return true;
    },
  });
  reg(72, {
    name: '边界 · L7 越界即抛', intuition: '下标等于维度大小或为负，L7 linearize 必须抛错。',
    math: 'k ≥ s[i] ∨ k < 0 ⇒ throw', pseudocode: 'two OOB indices; both must throw', complexity: 'O(n)',
    ref: 'numpy ndarray（IndexError 语义）', commonPitfall: '只测一侧（上界）不够；负下标同样要抛。',
    invariant: "run === 'threw'（构造越界下标必抛），且 L7 对上界与负下标均实测抛错",
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
      return runOf('E072')(shape) === 'threw' && threwHi && threwLo;
    },
  });
  reg(73, {
    name: '边界 · 中点偏移有界', intuition: '中点下标的扁平偏移必须落在 [0, size) 半开区间。',
    math: '0 ≤ lin(mid) < size(s)', pseudocode: 'assert range', complexity: 'O(n)',
    ref: 'numpy ndarray', commonPitfall: '上界严格；等于 size 即越界。',
    invariant: 'run === 本地点积 Σ mid[i]·st[i] 且落在 [0, ∏s)',
    inputs: ['shape'], domains: { shape: 'shape4' },
    run: (shape) => NS.L7.shape.linearize(midOf(shape), shape),
    predicate: (shape) => {
      const f = runOf('E073')(shape);
      const dot = dotIdx(midOf(shape), suffixProds(shape));
      if (f !== dot) return false;
      return f >= 0 && f < prod(shape, 0);
    },
  });
  reg(74, {
    name: '边界 · 最小扁平偏移为 0', intuition: '全零下标对应存储起点，偏移为 0（无 offset 时）。',
    math: 'lin(0ⁿ) = 0', pseudocode: 'return lin(zeros) === 0', complexity: 'O(n)',
    ref: 'numpy ndarray', commonPitfall: '传入非零 offset 时结果为 offset，勿忘平移。',
    invariant: 'run === 0（全零下标无位移）',
    inputs: ['shape'], domains: { shape: 'shape3' },
    run: (shape) => NS.L7.shape.linearize(zerosOf(shape.length), shape),
    predicate: (shape) => runOf('E074')(shape) === 0,
  });
  reg(75, {
    name: '边界 · 首维跨度', intuition: '首维从 0 到最大，跨过的偏移恰为 size 减去后缀 size。',
    math: '(s[0]−1)·st[0] = size(s) − size(s[1:])', pseudocode: 'assert both sides', complexity: 'O(n)',
    ref: 'numpy ndarray strides', commonPitfall: 'st[0] 就是 size(s[1:])，别重复累乘。',
    invariant: 'run === (s[0]−1)·∏s[1:] 且 === ∏s − ∏s[1:]',
    inputs: ['shape'], domains: { shape: 'shape2' },
    run: (shape) => (shape[0] - 1) * NS.L7.shape.contiguousStrides(shape)[0],
    predicate: (shape) => {
      const r = runOf('E075')(shape);
      const suf = prod(shape, 1);
      return r === (shape[0] - 1) * suf && r === prod(shape, 0) - suf;
    },
  });
  reg(76, {
    name: '边界 · 中点偏移单调不越顶', intuition: '中点逐维不超过最大下标，步长为正，故偏移不超最大偏移。',
    math: 'lin(mid) ≤ lin(s−1)', pseudocode: 'assert <=', complexity: 'O(n)',
    ref: 'numpy ndarray（正步长单调性）', commonPitfall: '负步长（翻转视图）下单调性反向。',
    invariant: 'run === 本地点积 Σ mid[i]·st[i] 且 ≤ ∏s − 1',
    inputs: ['shape'], domains: { shape: 'shape4' },
    run: (shape) => NS.L7.shape.linearize(midOf(shape), shape),
    predicate: (shape) => {
      const f = runOf('E076')(shape);
      const dot = dotIdx(midOf(shape), suffixProds(shape));
      if (f !== dot) return false;
      return f <= prod(shape, 0) - 1;
    },
  });

  // ── 多形状组合律（E077–E084，第二形状一律由第一形状构造）────────────
  reg(77, {
    name: '多形状 · size 结合律', intuition: '三块拼接的 size 既满足连乘也满足拼接结合律。',
    math: 'size(a∥b∥c) = size(a)·size(b)·size(c)', pseudocode: 'b = a+1, c = [2,3]; assert both', complexity: 'O(n)',
    ref: 'numpy ndarray.size', commonPitfall: '勿把结合律与交换律混同；拼接不交换。',
    invariant: 'run === ∏(a∥b∥c) 且 === ∏a·∏b·∏c',
    inputs: ['shape'], domains: { shape: 'shape3' },
    run: (shape) => NS.L7.shape.size(shape.concat(shape.map((d) => d + 1)).concat([2, 3])),
    predicate: (shape) => {
      const a = shape; const b = shape.map((d) => d + 1); const c = [2, 3];
      const r = runOf('E077')(shape);
      return r === prod(a.concat(b).concat(c), 0) && r === prod(a, 0) * prod(b, 0) * prod(c, 0);
    },
  });
  reg(78, {
    name: '多形状 · 拼接步长尾块', intuition: '拼接后段（原 b 的部分）步长不受前缀影响，原样保留。',
    math: 'strides(a∥b)[i] = strides(b)[i−n]（i≥n）', pseudocode: 'b = copy(a); loop assert tail', complexity: 'O(n)',
    ref: 'numpy ndarray（行主序分块步长）', commonPitfall: '前段步长要乘 size(b)，尾段不乘——两段规则不同。',
    invariant: 'run 逐维 === ∏s[1:]（拼接形状后缀积）；尾块逐项 === b 自身后缀积',
    inputs: ['shape'], domains: { shape: 'shape2' },
    run: (shape) => NS.L7.shape.contiguousStrides(shape.concat(shape.slice())),
    predicate: (shape) => {
      const b = shape.slice();
      const s = shape.concat(b);
      const st = runOf('E078')(shape);
      if (st.length !== s.length) return false;
      for (let i = 0; i < s.length; i++) if (st[i] !== prod(s, i + 1)) return false;
      for (let i = 0; i < b.length; i++) if (st[shape.length + i] !== prod(b, i + 1)) return false;
      return true;
    },
  });
  reg(79, {
    name: '多形状 · 广播与拼接 size', intuition: '拼接形状与同形全 1 广播，size 等于两块 size 之积。',
    math: 'size(broadcast(a∥b, 1)) = size(a)·size(b)', pseudocode: 'b = copy(a); assert', complexity: 'O(n)',
    ref: 'numpy broadcasting', commonPitfall: '全 1 形状长度必须与拼接后一致，否则是另一回事。',
    invariant: 'run 逐元素 === 拼接形状本身（本地广播 bcastLocal 吸收全 1），且 ∏ = ∏a·∏b',
    inputs: ['shape'], domains: { shape: 'shape4' },
    run: (shape) => NS.L7.shape.broadcastShapes(shape.concat(shape.slice()), shape.concat(shape.slice()).map(() => 1)),
    predicate: (shape) => {
      const b = shape.slice();
      const s = shape.concat(b);
      const r = runOf('E079')(shape);
      if (!eqArr(r, bcastLocal(s, s.map(() => 1)))) return false;
      return prod(r, 0) === prod(shape, 0) * prod(b, 0);
    },
  });
  reg(80, {
    name: '多形状 · 广播交换律（不等长）', intuition: '不等长的兼容对（短的对齐尾部）广播结果与顺序无关。',
    math: 'broadcast(a,b) = broadcast(b,a)（b = a[1:]）', pseudocode: 'assert eqArr', complexity: 'O(n)',
    ref: 'numpy broadcasting', commonPitfall: '交换律前提是兼容；不兼容两侧都抛，谈不上相等。',
    invariant: 'run 逐元素 === 本地广播 bcastLocal(shape, 去首维)',
    inputs: ['shape'], domains: { shape: 'shape3' },
    run: (shape) => NS.L7.shape.broadcastShapes(shape, shape.slice(1)),
    predicate: (shape) => {
      const b = shape.slice(1);
      const r = runOf('E080')(shape);
      return eqArr(r, bcastLocal(shape, b)) && eqArr(r, bcastLocal(b, shape));
    },
  });
  reg(81, {
    name: '多形状 · 广播结合律', intuition: '三级广播按不同顺序结合，结果相同（构造尾段与前置 1 段）。',
    math: 'bcast(bcast(a,b),c) = bcast(a,bcast(b,c))', pseudocode: 'b = a[1:], c = [1]∥a[1:]; assert', complexity: 'O(n)',
    ref: 'numpy broadcasting', commonPitfall: '结合律要求每对中间结果都兼容；构造已保证。',
    invariant: 'run 逐元素 === 本地两级广播链 bcastLocal(bcastLocal(a,b), c)',
    inputs: ['shape'], domains: { shape: 'shape2' },
    run: (shape) => NS.L7.shape.broadcastShapes(NS.L7.shape.broadcastShapes(shape, shape.slice(1)), [1].concat(shape.slice(1))),
    predicate: (shape) => {
      const b = shape.slice(1);
      const c = [1].concat(shape.slice(1));
      const m1 = bcastLocal(shape, b);
      if (m1 === null) return false;
      const m2 = bcastLocal(m1, c);
      if (m2 === null) return false;
      const r = runOf('E081')(shape);
      return eqArr(r, m2) && eqArr(r, bcastLocal(shape, bcastLocal(b, c)));
    },
  });
  reg(82, {
    name: '多形状 · 1 维吸收链', intuition: '连续与全 1 形状广播两次，结果始终是原形状。',
    math: 'bcast(bcast(s,1),1) = s', pseudocode: 'assert both equal shape', complexity: 'O(n)',
    ref: 'numpy broadcasting', commonPitfall: '第二次广播的 1 形状要由第一次结果构造，长度才对。',
    invariant: 'run 逐元素 === 本地两级广播链 bcastLocal(bcastLocal(s,1),1)，且 === 原形状',
    inputs: ['shape'], domains: { shape: 'shape4' },
    run: (shape) => {
      const bc1 = NS.L7.shape.broadcastShapes(shape, shape.map(() => 1));
      return NS.L7.shape.broadcastShapes(bc1, bc1.map(() => 1));
    },
    predicate: (shape) => {
      const m1 = bcastLocal(shape, shape.map(() => 1));
      if (m1 === null) return false;
      const m2 = bcastLocal(m1, m1.map(() => 1));
      if (m2 === null) return false;
      const r = runOf('E082')(shape);
      return eqArr(r, m2) && eqArr(r, shape);
    },
  });
  reg(83, {
    name: '多形状 · 置换与广播可交换', intuition: '先广播再置换 ≡ 先置换再广播（全 1 形状置换后仍全 1）。',
    math: 'perm(bcast(s,1)) = bcast(perm(s), perm(1))', pseudocode: 'axes = reverse; assert shapes', complexity: 'O(n)',
    ref: 'numpy transpose + broadcasting', commonPitfall: '只对齐形状层；L7 广播不产步长，无法比对步长层。另注意 broadcastShapes 直接返回形状数组（不是 {shape} 对象），不可写 p2.shape。',
    invariant: 'run 逐项 === shape[axes[i]]（反转轴序），全 1 广播不改变各维取值',
    inputs: ['shape'], domains: { shape: 'shape3' },
    run: (shape) => NS.L7.shape.permute(NS.L7.shape.broadcastShapes(shape, shape.map(() => 1)), NS.L7.shape.contiguousStrides(shape), reverseAxes(shape.length)).shape,
    predicate: (shape) => {
      const axes = reverseAxes(shape.length);
      const r = runOf('E083')(shape);
      if (r.length !== shape.length) return false;
      for (let i = 0; i < shape.length; i++) if (r[i] !== shape[axes[i]]) return false;
      return true;
    },
  });
  reg(84, {
    name: '多形状 · 广播 size 不减', intuition: '广播只把 1 放大成更大维，size 不会缩水。',
    math: 'size(bcast(a,b)) ≥ size(a), size(b)', pseudocode: 'b = 偶数维 1; assert both', complexity: 'O(n)',
    ref: 'numpy broadcasting', commonPitfall: '不减不等于严格增；全 1 对手下取等。',
    invariant: 'run 逐元素 === 本地广播 bcastLocal(shape, b)，且 ∏run 不小于双方 ∏',
    inputs: ['shape'], domains: { shape: 'shape2' },
    run: (shape) => NS.L7.shape.broadcastShapes(shape, shape.map((d, i) => (i % 2 === 0 ? 1 : d))),
    predicate: (shape) => {
      const b = shape.map((d, i) => (i % 2 === 0 ? 1 : d));
      const r = runOf('E084')(shape);
      const mine = bcastLocal(shape, b);
      if (mine === null) return false;
      if (!eqArr(r, mine)) return false;
      const sz = prod(r, 0);
      return sz >= prod(shape, 0) && sz >= prod(b, 0);
    },
  });

  // ── 形状代数（E085–E090）───────────────────────────────────────────
  reg(85, {
    name: '形状代数 · 拼接单位元', intuition: '空形状是拼接的单位元：接上去什么都不变。',
    math: 'a∥[] = a', pseudocode: 'assert size & length', complexity: 'O(n)',
    ref: '自由幺半群（形状拼接）', commonPitfall: '单位元在 size 层是 1，在长度层是 0，两层面都要查。',
    invariant: 'run 逐元素 === 原形状（拼接空形状是单位元）',
    inputs: ['shape'], domains: { shape: 'shape4' },
    run: (shape) => shape.concat([]),
    predicate: (shape) => {
      const r = runOf('E085')(shape);
      return eqArr(r, shape) && r.length === shape.length;
    },
  });
  reg(86, {
    name: '形状代数 · 逆序保 size', intuition: '维度次序颠倒不改变元素总数（乘法交换）。',
    math: 'size(reverse(s)) = size(s)', pseudocode: 'assert', complexity: 'O(n)',
    ref: 'numpy ndarray.size', commonPitfall: 'size 同但步长表通常不同，勿混谈。',
    invariant: 'run === ∏s（逆序不改变乘积）',
    inputs: ['shape'], domains: { shape: 'shape3' },
    run: (shape) => NS.L7.shape.size(shape.slice().reverse()),
    predicate: (shape) => runOf('E086')(shape) === prod(shape, 0),
  });
  reg(87, {
    name: '形状代数 · 排序保 size', intuition: '把维度按大小排序，乘积不变。',
    math: 'size(sort(s)) = size(s)', pseudocode: 'assert', complexity: 'O(n log n)',
    ref: '乘法交换律', commonPitfall: '排序会改变轴语义（哪维是哪维），size 层无所谓。',
    invariant: 'run === ∏s（排序不改变乘积）',
    inputs: ['shape'], domains: { shape: 'shape2' },
    run: (shape) => NS.L7.shape.size(shape.slice().sort((x, y) => x - y)),
    predicate: (shape) => runOf('E087')(shape) === prod(shape, 0),
  });
  reg(88, {
    name: '形状代数 · 滤 1 保 size', intuition: '长度 1 的维是乘性单位元，滤掉不改变 size。',
    math: 'size(filter(s, d>1)) = size(s)', pseudocode: 'assert', complexity: 'O(n)',
    ref: 'numpy.squeeze（size 不变）', commonPitfall: 'squeeze 改变形状语义，size 层面才等价。',
    invariant: 'run === ∏(滤除 1 维后的形状) 且 === ∏s',
    inputs: ['shape'], domains: { shape: 'shape4' },
    run: (shape) => NS.L7.shape.size(shape.filter((d) => d > 1)),
    predicate: (shape) => {
      const r = runOf('E088')(shape);
      const f = shape.filter((d) => d > 1);
      return r === prod(f, 0) && r === prod(shape, 0);
    },
  });
  reg(89, {
    name: '形状代数 · 交换两维保 size', intuition: '任意交换两个维度（乘法交换律），size 不变。',
    math: 'size(swap(s,i,j)) = size(s)', pseudocode: 'i = k%n, j = (k>>5)%n; assert', complexity: 'O(n)',
    ref: '乘法交换律', commonPitfall: 'i=j 时是恒等交换，结论仍成立。',
    invariant: 'run === ∏(交换两轴后的形状) 且 === ∏s',
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
      const r = runOf('E089')(shape, k);
      return r === prod(s, 0) && r === prod(shape, 0);
    },
  });
  reg(90, {
    name: '形状代数 · size 为正整数', intuition: '正整数维的乘积仍是正整数，至少为 1。',
    math: 'size(s) ∈ ℤ⁺', pseudocode: 'return isInteger && >= 1', complexity: 'O(n)',
    ref: 'numpy ndarray.size', commonPitfall: '含 0 维时为 0，非正；本域输入恒 ≥1。',
    invariant: 'run === ∏s 且为 ≥1 的整数',
    inputs: ['shape'], domains: { shape: 'shape2' },
    run: (shape) => NS.L7.shape.size(shape),
    predicate: (shape) => {
      const sz = runOf('E090')(shape);
      return sz === prod(shape, 0) && Number.isInteger(sz) && sz >= 1;
    },
  });

  // ── linearize 深化（E091–E096）─────────────────────────────────────
  reg(91, {
    name: '线性化 · 步长缩放齐次性', intuition: '所有步长同乘 m，偏移整体放大 m 倍。',
    math: 'lin(idx,s,m·st) = m·lin(idx,s,st)', pseudocode: 'm = k%7+2; assert', complexity: 'O(n)',
    ref: '点积的齐次性', commonPitfall: 'm 必须为正；缩放步长不缩放 offset。',
    invariant: 'run === m·Σ mid[i]·st[i]（本地点积，独立于 L7.linearize）',
    inputs: ['shape', 'k'], domains: { shape: 'shape4', k: 'nat' },
    run: (shape, k) => {
      const m = (k % 7) + 2;
      const st = NS.L7.shape.contiguousStrides(shape).map((s) => s * m);
      return NS.L7.shape.linearize(midOf(shape), shape, st);
    },
    predicate: (shape, k) => {
      const m = (k % 7) + 2;
      const r = runOf('E091')(shape, k);
      return r === m * dotIdx(midOf(shape), suffixProds(shape));
    },
  });
  reg(92, {
    name: '线性化 · 偏移合并可加', intuition: '两个偏移先后传入等于一次性传入其和。',
    math: 'lin(idx,s,st,o₁+o₂) = lin(idx,s,st,o₁)+o₂', pseudocode: 'o1 = k%5+1, o2 = k%7+1; assert', complexity: 'O(n)',
    ref: 'numpy ndarray（offset 语义）', commonPitfall: '可加性对 offset 成立，对 stride 不成立。',
    invariant: 'run === Σ mid[i]·st[i] + o₁ + o₂（本地点积 + 偏移）',
    inputs: ['shape', 'k'], domains: { shape: 'shape3', k: 'nat' },
    run: (shape, k) => NS.L7.shape.linearize(midOf(shape), shape, null, (k % 5) + 1 + (k % 7) + 1),
    predicate: (shape, k) => {
      const o1 = (k % 5) + 1; const o2 = (k % 7) + 1;
      const r = runOf('E092')(shape, k);
      return r === dotIdx(midOf(shape), suffixProds(shape)) + o1 + o2;
    },
  });
  reg(93, {
    name: '线性化 · 分量点积公式', intuition: '线性化的定义就是下标与步长的点积，逐项求和验证。',
    math: 'lin(idx,s,st) = Σ idx[i]·st[i]', pseudocode: 'local dot; assert', complexity: 'O(n)',
    ref: 'numpy ndarray strides', commonPitfall: '点积不含 offset；offset 是外加项。',
    invariant: 'run === Σ mid[i]·st[i]（本地点积，独立于 L7.linearize）',
    inputs: ['shape'], domains: { shape: 'shape2' },
    run: (shape) => NS.L7.shape.linearize(midOf(shape), shape),
    predicate: (shape) => {
      const r = runOf('E093')(shape);
      return r === dotIdx(midOf(shape), suffixProds(shape));
    },
  });
  reg(94, {
    name: '线性化 · 分量独立性', intuition: '把某一维下标置零，偏移恰好减去该维的贡献。',
    math: 'lin(idx−e_j·idx[j]) = lin(idx) − idx[j]·st[j]', pseudocode: 'j = k%n; assert', complexity: 'O(n)',
    ref: 'numpy ndarray strides', commonPitfall: 'idx[j] 为 0 时等式退化为 0=0，仍成立但无信息量。',
    invariant: 'run === 置零轴后的本地点积，且 === 完整点积 − mid[j]·st[j]',
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
      const st = suffixProds(shape);
      const r = runOf('E094')(shape, k);
      if (r !== dotIdx(idx, st)) return false;
      return r === dotIdx(mid, st) - mid[j] * st[j];
    },
  });
  reg(95, {
    name: '线性化 · 全 1 形状唯一元素', intuition: '全 1 形状只有一个元素，任何合法下标（只有全零）偏移为 0。',
    math: 'size(1ⁿ)=1 ∧ lin(0ⁿ,1ⁿ)=0', pseudocode: 'ones = map(shape,()=>1); assert', complexity: 'O(n)',
    ref: 'numpy ndarray（0-d 类比物）', commonPitfall: '全 1 形状不是标量（0 维），但行为等价。',
    invariant: 'run === 0（全 1 形状的零下标偏移恒为 0）',
    inputs: ['shape'], domains: { shape: 'shape3' },
    run: (shape) => NS.L7.shape.linearize(zerosOf(shape.length), shape.map(() => 1)),
    predicate: (shape) => runOf('E095')(shape) === 0,
  });
  reg(96, {
    name: '线性化 · 末维连续分解', intuition: '总偏移 = 前 n-1 维贡献之和 + 末维下标（末维步长 1）。',
    math: 'lin(idx) = Σ_{i<n-1} idx[i]·st[i] + idx[n-1]', pseudocode: 'split sum; assert', complexity: 'O(n)',
    ref: 'numpy ndarray strides（C 序末维连续）', commonPitfall: '分解依赖 st[n-1]===1；非连续布局不成立。',
    invariant: 'run === Σ_{i<n-1} mid[i]·st[i] + mid[n-1]（末维步长为 1）',
    inputs: ['shape'], domains: { shape: 'shape2' },
    run: (shape) => NS.L7.shape.linearize(midOf(shape), shape),
    predicate: (shape) => {
      const n = shape.length;
      const mid = midOf(shape);
      const st = suffixProds(shape);
      let pre = 0;
      for (let i = 0; i < n - 1; i++) pre += mid[i] * st[i];
      return runOf('E096')(shape) === pre + mid[n - 1];
    },
  });
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

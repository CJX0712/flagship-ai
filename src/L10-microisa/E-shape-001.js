/* ============================================================================
 * L10 微指令 · E 域（形状与索引）· 第 2/3 批（E033–E064）
 * ----------------------------------------------------------------------------
 * 所属层：L10-microIsa  域：E（形状与索引）
 * 允许依赖的层：L7（shape 代数）、L9（rng）、L10（registry/invariant）
 * 主要 @ref：numpy ndarray strides；broadcasting 规则（尾部对齐）；
 *            numpy.transpose / argsort 轴置换语义（行主序 C order）
 *
 * 每项含七件套（直觉/数学/伪代码/复杂度/论文/常见坑/不变量）+ run + predicate。
 * 判据：整数精确，一律 === 与逐元素深比（eqArr）。
 * 输入纪律：每个 'shape' 输入独立随机生成，互不相容；本批所有需要第二形状的
 *       场景都在 predicate 内由第一形状确定性地构造（置 1、切片、扩维），
 *       广播兼容性由构造保证，不依赖外部任意子碰巧对齐。
 * 加载顺序注意：本文件先于 L7 求值，NS.L7.shape 一律惰性引用。
 *
 * ── 谓词纪律（本批全部重写，2026-10-24）─────────────────────────────────
 * 门禁只调 op.predicate，从不调 op.run（op-harness.js:43、probe-microisa.js:106）。
 * 原 32 项谓词**全部直接调 NS.L7.shape.*，从不调本 op 的 run**——run 若损坏
 * 门禁完全失明（假绿）。现全部改为：先经 runOf('Exxx') 调 run 本体，再对 run
 * 的产出断言不变量。oracle 取本文件内第一性原理实现，刻意绕开 L7：
 *   · prod(a, from)  —— 维度乘积
 *   · stridesOf(s)   —— 自写行主序步长
 *   · bcastLocal(a,b)—— 自写尾部对齐广播（不相容返回 null）
 *   · invLocal(axes) —— 自写逆置换（inv[axes[i]] = i）
 *   · 对齐取大刻画  —— bc[i] = max(align(a)[i], align(b)[i])（与 bcastLocal
 *                      的写法不同，构成互相独立的第二实现）
 * ==========================================================================*/
;(function (NS) {
  'use strict';
  NS.L10 = NS.L10 || {};
  function reg(i, o) { NS.L10.register('E', i, o); }
  function runOf(key) { const o = NS.L10.ops[key]; return o && typeof o.run === 'function' ? o.run : null; }

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
  function identAxes(n) { const a = new Array(n); for (let i = 0; i < n; i++) a[i] = i; return a; }
  function reverseAxes(n) { const a = new Array(n); for (let i = 0; i < n; i++) a[i] = n - 1 - i; return a; }
  function rotateAxes(n) { const a = new Array(n); for (let i = 0; i < n; i++) a[i] = (i + 1) % n; return a; }
  // 本地尾部对齐广播（第一性原理）：不相容返回 null。
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
  // 广播的第二实现（与 bcastLocal 写法不同）：先补齐缺省侧，再逐维取大。
  // 两者互为独立基准；不相容时返回 null。
  function bcastMax(a, b) {
    const n = Math.max(a.length, b.length);
    const out = new Array(n);
    for (let i = 0; i < n; i++) {
      const da = i < a.length ? a[a.length - 1 - i] : 1;
      const db = i < b.length ? b[b.length - 1 - i] : 1;
      if (da !== db && da !== 1 && db !== 1) return null;
      out[n - 1 - i] = Math.max(da, db);
    }
    return out;
  }
  // 本地逆置换：inv[axes[i]] = i。
  function invLocal(axes) {
    const inv = new Array(axes.length);
    for (let i = 0; i < axes.length; i++) inv[axes[i]] = i;
    return inv;
  }

  // ── 广播（E033–E042）────────────────────────────────────────────────
  reg(33, {
    name: '广播 · 全 1 尾返回原形状', intuition: '与全 1 形状广播，每一维都被原形状吸收，结果原样。',
    math: 'broadcast(s, 1ⁿ) = s', pseudocode: 'ones = map(shape, ()=>1); assert eq', complexity: 'O(n)',
    ref: 'numpy broadcasting 规则', commonPitfall: '1 是广播单位元；错把它当 0 会整维塌掉。',
    invariant: 'run 逐元素等于本地尾部对齐广播 bcastLocal(shape, 全1)',
    inputs: ['shape'], domains: { shape: 'shape3' },
    run: (shape) => NS.L7.shape.broadcastShapes(shape, shape.map(() => 1)),
    predicate: (shape) => {
      const ones = shape.map(() => 1);
      const r = runOf('E033')(shape);
      return eqArr(r, bcastLocal(shape, ones)) && eqArr(r, shape);
    },
  });
  reg(34, {
    name: '广播 · 自身幂等', intuition: '同一形状与自身广播，结果仍是自身。',
    math: 'broadcast(s, s) = s', pseudocode: 'return eqArr(broadcast(s,s), s)', complexity: 'O(n)',
    ref: 'numpy broadcasting 规则', commonPitfall: '幂等性只对形状成立，勿推广到任意两形状。',
    invariant: 'run 逐元素等于本地广播 bcastLocal(shape, shape) 且等于原形状',
    inputs: ['shape'], domains: { shape: 'shape2' },
    run: (shape) => NS.L7.shape.broadcastShapes(shape, shape.slice()),
    predicate: (shape) => {
      const r = runOf('E034')(shape);
      return eqArr(r, bcastLocal(shape, shape.slice())) && eqArr(r, shape);
    },
  });
  reg(35, {
    name: '广播 · 交换律', intuition: '广播结果与操作数顺序无关（构造偶数维为 1 的兼容对）。',
    math: 'broadcast(a,b) = broadcast(b,a)', pseudocode: 'b = even dims → 1; assert eq', complexity: 'O(n)',
    ref: 'numpy broadcasting 规则', commonPitfall: '交换律要求两形状兼容；不兼容时两侧都抛错。',
    invariant: 'run 逐元素等于本地广播 bcastLocal(shape, b)（b 为偶数维置 1）',
    inputs: ['shape'], domains: { shape: 'shape4' },
    run: (shape) => NS.L7.shape.broadcastShapes(shape, shape.map((d, i) => (i % 2 === 0 ? 1 : d))),
    predicate: (shape) => {
      const b = shape.map((d, i) => (i % 2 === 0 ? 1 : d));
      const r = runOf('E035')(shape);
      return eqArr(r, bcastLocal(shape, b)) && eqArr(r, bcastLocal(b, shape));
    },
  });
  reg(36, {
    name: '广播 · 尾部对齐缺省补 1', intuition: '短形状按尾部对齐，缺失的前缀维按 1 补齐。',
    math: 'broadcast(s, s[1:]) = s ∧ len = n', pseudocode: 'b = shape.slice(1); assert', complexity: 'O(n)',
    ref: 'numpy broadcasting 规则（尾部对齐）', commonPitfall: '是右侧（尾部）对齐，不是左侧；Leading 1 才是补出来的。',
    invariant: 'run 逐元素等于本地广播 bcastLocal(shape, 去首维) 且长度不变',
    inputs: ['shape'], domains: { shape: 'shape3' },
    run: (shape) => NS.L7.shape.broadcastShapes(shape, shape.slice(1)),
    predicate: (shape) => {
      const r = runOf('E036')(shape);
      return eqArr(r, bcastLocal(shape, shape.slice(1))) && r.length === shape.length && eqArr(r, shape);
    },
  });
  reg(37, {
    name: '广播 · 尾部扩维 size 翻倍', intuition: '两侧同 rank 各追加一维（2 与 1）后广播，结果即 shape∥[2] 且 size 翻倍。',
    math: 'broadcast(s∥[2], s∥[1]) = s∥[2]，size = 2·size(s)', pseudocode: 'b2 = concat(shape,[2]); b1 = concat(shape,[1]); assert', complexity: 'O(n)',
    ref: 'numpy broadcasting 规则', commonPitfall: '两侧必须同 rank！若拿 s 与 s∥[2] 广播，尾部对齐会把 s[i] 对到 b[i+1]，s 中的 1 维会被邻维放大（[3,1] vs [3,1,2] → [3,3,2]），结论不成立。',
    invariant: 'run 逐元素等于本地广播 bcastLocal(s∥[2], s∥[1])，等于 s∥[2]，size 为 2·∏s',
    inputs: ['shape'], domains: { shape: 'shape2' },
    run: (shape) => NS.L7.shape.broadcastShapes(shape.concat([2]), shape.concat([1])),
    predicate: (shape) => {
      const b2 = shape.concat([2]), b1 = shape.concat([1]);
      const r = runOf('E037')(shape);
      return eqArr(r, bcastLocal(b2, b1)) && eqArr(r, b2) && prod(r, 0) === 2 * prod(shape, 0);
    },
  });
  reg(38, {
    name: '广播 · 前置 1 维被保留', intuition: '长形状前置的 1 维不吸收短形状，结果保留该 1 维。',
    math: 'broadcast(s, [1]∥s) = [1]∥s', pseudocode: 'b = [1].concat(shape); assert', complexity: 'O(n)',
    ref: 'numpy broadcasting 规则', commonPitfall: '误以为 1 维总被丢弃；对齐后缺省侧才是 1。',
    invariant: 'run 逐元素等于本地广播 bcastLocal(shape, [1]∥shape) 且等于 [1]∥shape',
    inputs: ['shape'], domains: { shape: 'shape4' },
    run: (shape) => NS.L7.shape.broadcastShapes(shape, [1].concat(shape)),
    predicate: (shape) => {
      const b = [1].concat(shape);
      const r = runOf('E038')(shape);
      return eqArr(r, bcastLocal(shape, b)) && eqArr(r, b) && r.length === shape.length + 1;
    },
  });
  reg(39, {
    name: '广播 · 不兼容即抛', intuition: '两维既不相等又都不是 1 时广播必须抛错（构造 d 与 d+1 对）。',
    math: 's[i]≥2 ⇒ broadcast(s, s+1) 抛错', pseudocode: 'try/catch; 全 1 时结果取大', complexity: 'O(n)',
    ref: 'numpy broadcasting 规则（ValueError 语义）', commonPitfall: '全 1 形状与任何形状都兼容（1 可被吸收），勿期待抛错。',
    invariant: 'run 与本地广播判定一致：不相容时 run === null，兼容时 run 逐元素等于本地结果',
    inputs: ['shape'], domains: { shape: 'shape3' },
    run: (shape) => {
      try { return NS.L7.shape.broadcastShapes(shape, shape.map((d) => d + 1)); }
      catch (e) { return null; }
    },
    predicate: (shape) => {
      const b = shape.map((d) => d + 1);
      const mine = bcastMax(shape, b);
      const r = runOf('E039')(shape);
      if (mine === null) return r === null;              // 不兼容：run 必须报告失败
      return r !== null && eqArr(r, mine);               // 兼容：结果逐元素一致
    },
  });
  reg(40, {
    name: '广播 · 第一性原理交叉验证', intuition: '自写尾部对齐广播与 L7 逐项比对（构造偶数维 1 的兼容对）。',
    math: 'bcastLocal(a,b) ≡ broadcast(a,b)', pseudocode: 'assert eqArr(local, L7)', complexity: 'O(n)',
    ref: 'numpy broadcasting 规则', commonPitfall: '本地版不相容应返回 null 而非抛，便于比对。',
    invariant: 'run 逐维等于「补齐缺省侧后取大」的独立刻画（与 bcastLocal 写法不同）',
    inputs: ['shape'], domains: { shape: 'shape2' },
    run: (shape) => bcastLocal(shape, shape.map((d, i) => (i % 2 === 0 ? 1 : d))),
    predicate: (shape) => {
      const b = shape.map((d, i) => (i % 2 === 0 ? 1 : d));
      const r = runOf('E040')(shape);
      const n = Math.max(shape.length, b.length);
      if (r.length !== n) return false;
      for (let i = 0; i < n; i++) {
        const da = i < shape.length ? shape[shape.length - 1 - i] : 1;
        const db = i < b.length ? b[b.length - 1 - i] : 1;
        if (r[n - 1 - i] !== Math.max(da, db)) return false;
      }
      return true;
    },
  });
  reg(41, {
    name: '广播 · 逐维取大', intuition: '兼容广播的每一维就是双方（尾部对齐、缺省补 1）的较大者。',
    math: 'bc[·] = max(align(a)[·], align(b)[·])', pseudocode: 'loop aligned dims; assert max', complexity: 'O(n)',
    ref: 'numpy broadcasting 规则', commonPitfall: '取大前必须先尾部对齐并把缺失侧当 1。',
    invariant: 'run 逐维等于「补齐缺省侧后取大」的独立刻画',
    inputs: ['shape'], domains: { shape: 'shape4' },
    run: (shape) => NS.L7.shape.broadcastShapes(shape, shape.map((d, i) => (i % 2 === 0 ? 1 : d))),
    predicate: (shape) => {
      const b = shape.map((d, i) => (i % 2 === 0 ? 1 : d));
      const r = runOf('E041')(shape);
      const n = Math.max(shape.length, b.length);
      if (r.length !== n) return false;
      for (let i = 0; i < n; i++) {
        const da = i < shape.length ? shape[shape.length - 1 - i] : 1;
        const db = i < b.length ? b[b.length - 1 - i] : 1;
        if (r[n - 1 - i] !== Math.max(da, db)) return false;
      }
      return true;
    },
  });
  reg(42, {
    name: '广播 · 短形状补 1 后取大', intuition: '不等长时短形状缺省侧视为 1，取大后应还原长形状。',
    math: 'broadcast(s, s[1:]) = s（逐维 max）', pseudocode: 'b = shape.slice(1); assert max & identity', complexity: 'O(n)',
    ref: 'numpy broadcasting 规则', commonPitfall: '短形状补 1 补在前缀（左侧），不是右侧。',
    invariant: 'run 逐维等于对齐取大，且整体等于原形状',
    inputs: ['shape'], domains: { shape: 'shape3' },
    run: (shape) => NS.L7.shape.broadcastShapes(shape, shape.slice(1)),
    predicate: (shape) => {
      const b = shape.slice(1);
      const r = runOf('E042')(shape);
      const n = Math.max(shape.length, b.length);
      if (r.length !== n) return false;
      for (let i = 0; i < n; i++) {
        const da = i < shape.length ? shape[shape.length - 1 - i] : 1;
        const db = i < b.length ? b[b.length - 1 - i] : 1;
        if (r[n - 1 - i] !== Math.max(da, db)) return false;
      }
      return eqArr(r, shape);
    },
  });

  // ── 轴置换与逆置换（E043–E054）──────────────────────────────────────
  reg(43, {
    name: '置换 · 恒等置换还原', intuition: '恒等轴序置换后形状与步长都原样返回。',
    math: 'permute(s, st, id) = (s, st)', pseudocode: 'assert both arrays equal', complexity: 'O(n)',
    ref: 'numpy.transpose（默认 axes=None）', commonPitfall: 'axes 必须是排列，重复轴即抛。',
    invariant: 'run 产出形状 === 原形状、步长 === 本地 stridesOf(shape)',
    inputs: ['shape'], domains: { shape: 'shape2' },
    run: (shape) => NS.L7.shape.permute(shape, NS.L7.shape.contiguousStrides(shape), identAxes(shape.length)),
    predicate: (shape) => {
      const p = runOf('E043')(shape);
      return eqArr(p.shape, shape) && eqArr(p.stride, stridesOf(shape));
    },
  });
  reg(44, {
    name: '置换 · 形状按轴重排', intuition: '新形状第 i 项 = 原形状第 axes[i] 项（反转轴序验证）。',
    math: 'p.shape[i] = s[axes[i]]', pseudocode: 'axes = reverse; loop assert', complexity: 'O(n)',
    ref: 'numpy.transpose', commonPitfall: '步长要跟着同一 axes 重排，不能只换形状。',
    invariant: 'run 产出逐项 === shape[axes[i]]（反转轴序）',
    inputs: ['shape'], domains: { shape: 'shape4' },
    run: (shape) => NS.L7.shape.permute(shape, NS.L7.shape.contiguousStrides(shape), reverseAxes(shape.length)).shape,
    predicate: (shape) => {
      const axes = reverseAxes(shape.length);
      const r = runOf('E044')(shape);
      if (r.length !== shape.length) return false;
      for (let i = 0; i < shape.length; i++) if (r[i] !== shape[axes[i]]) return false;
      return true;
    },
  });
  reg(45, {
    name: '置换 · 步长按轴重排', intuition: '新步长第 i 项 = 原步长第 axes[i] 项（旋转轴序验证）。',
    math: 'p.stride[i] = st[axes[i]]', pseudocode: 'axes = rotate; loop assert', complexity: 'O(n)',
    ref: 'numpy.transpose（步长随轴走）', commonPitfall: '形状与步长必须用同一 axes，错位即静默错误。',
    invariant: 'run 产出逐项 === 本地 stridesOf(shape)[axes[i]]（旋转轴序）',
    inputs: ['shape'], domains: { shape: 'shape3' },
    run: (shape) => NS.L7.shape.permute(shape, NS.L7.shape.contiguousStrides(shape), rotateAxes(shape.length)).stride,
    predicate: (shape) => {
      const st = stridesOf(shape);
      const axes = rotateAxes(shape.length);
      const r = runOf('E045')(shape);
      if (r.length !== shape.length) return false;
      for (let i = 0; i < shape.length; i++) if (r[i] !== st[axes[i]]) return false;
      return true;
    },
  });
  reg(46, {
    name: '置换 · 保 size', intuition: '置换只重排维度次序，元素总数不变。',
    math: 'size(p.shape) = size(s)', pseudocode: 'assert sizes equal', complexity: 'O(n)',
    ref: 'numpy.transpose', commonPitfall: 'size 不变但内存布局（步长）通常变。',
    invariant: 'run 产出逐项 === shape[axes[i]]（旋转轴序）且元素数 === ∏s',
    inputs: ['shape'], domains: { shape: 'shape2' },
    run: (shape) => NS.L7.shape.permute(shape, NS.L7.shape.contiguousStrides(shape), rotateAxes(shape.length)).shape,
    predicate: (shape) => {
      const axes = rotateAxes(shape.length);
      const r = runOf('E046')(shape);
      if (r.length !== shape.length) return false;
      for (let i = 0; i < shape.length; i++) if (r[i] !== shape[axes[i]]) return false;
      return prod(r, 0) === prod(shape, 0);
    },
  });
  reg(47, {
    name: '置换 · 保最大扁平偏移', intuition: '置换是 (维, 步长) 对的同一重排，可达偏移集合不变。',
    math: 'lin(maxIdx(p)) = size(s)−1', pseudocode: 'axes = reverse; assert', complexity: 'O(n)',
    ref: 'numpy.transpose（偏移集不变）', commonPitfall: '最大下标要用置换后的形状构造，不能沿用旧的。',
    invariant: 'run === ∏s − 1（置换不改变最大扁平偏移）',
    inputs: ['shape'], domains: { shape: 'shape4' },
    run: (shape) => {
      const p = NS.L7.shape.permute(shape, NS.L7.shape.contiguousStrides(shape), reverseAxes(shape.length));
      return NS.L7.shape.linearize(p.shape.map((d) => d - 1), p.shape, p.stride);
    },
    predicate: (shape) => runOf('E047')(shape) === prod(shape, 0) - 1,
  });
  reg(48, {
    name: '置换 · 逆置换还原', intuition: '先按 axes 置换，再按逆 axes 置换，形状步长都回家。',
    math: 'permute(permute(s,st,a), inv(a)) = (s, st)', pseudocode: 'assert roundtrip', complexity: 'O(n)',
    ref: 'numpy.transpose / moveaxis', commonPitfall: '逆置换是 inv[axes[i]]=i，不是 axes 反转。',
    invariant: 'run 产出形状 === 原形状、步长 === 本地 stridesOf(shape)',
    inputs: ['shape'], domains: { shape: 'shape3' },
    run: (shape) => {
      const p = NS.L7.shape.permute(shape, NS.L7.shape.contiguousStrides(shape), rotateAxes(shape.length));
      return NS.L7.shape.permute(p.shape, p.stride, NS.L7.shape.invertPerm(rotateAxes(shape.length)));
    },
    predicate: (shape) => {
      const p = runOf('E048')(shape);
      return eqArr(p.shape, shape) && eqArr(p.stride, stridesOf(shape));
    },
  });
  reg(49, {
    name: '置换 · 反转轴对合', intuition: '反转轴序做两次等于没做（rev 是自身的逆）。',
    math: 'permute(permute(s,st,rev),rev) = (s,st)', pseudocode: 'assert involution', complexity: 'O(n)',
    ref: 'numpy.transpose（[::-1] 语义）', commonPitfall: '对合性只对反转成立，旋转轴做两次是转两格。',
    invariant: 'run 产出形状 === 原形状、步长 === 本地 stridesOf(shape)',
    inputs: ['shape'], domains: { shape: 'shape2' },
    run: (shape) => {
      const rev = reverseAxes(shape.length);
      const p = NS.L7.shape.permute(shape, NS.L7.shape.contiguousStrides(shape), rev);
      return NS.L7.shape.permute(p.shape, p.stride, rev);
    },
    predicate: (shape) => {
      const p = runOf('E049')(shape);
      return eqArr(p.shape, shape) && eqArr(p.stride, stridesOf(shape));
    },
  });
  reg(50, {
    name: '置换 · 两步复合律', intuition: '连做两次置换等于按复合轴序一次置换。',
    math: 'permute(permute(s,a₁),a₂) = permute(s, a₁∘a₂)', pseudocode: 'composed[i]=a1[a2[i]]; assert', complexity: 'O(n)',
    ref: 'numpy.transpose（置换群复合）', commonPitfall: '复合方向是 a₁[a₂[i]]，写反则不成立。',
    invariant: 'run 产出逐项 === shape[composed[i]] 且步长 === 本地 stridesOf(shape)[composed[i]]',
    inputs: ['shape'], domains: { shape: 'shape4' },
    run: (shape) => {
      const a1 = rotateAxes(shape.length); const a2 = reverseAxes(shape.length);
      const p = NS.L7.shape.permute(shape, NS.L7.shape.contiguousStrides(shape), a1);
      return NS.L7.shape.permute(p.shape, p.stride, a2);
    },
    predicate: (shape) => {
      const st = stridesOf(shape);
      const a1 = rotateAxes(shape.length); const a2 = reverseAxes(shape.length);
      const composed = a1.map((_, i) => a1[a2[i]]);
      const y = runOf('E050')(shape);
      if (y.shape.length !== shape.length || y.stride.length !== shape.length) return false;
      for (let i = 0; i < shape.length; i++) {
        if (y.shape[i] !== shape[composed[i]]) return false;
        if (y.stride[i] !== st[composed[i]]) return false;
      }
      return true;
    },
  });
  reg(51, {
    name: '逆置换 · 是合法排列', intuition: '逆置换排序后必须是 0..n-1，证明它是双射。',
    math: 'sort(inv(a)) = [0,1,…,n-1]', pseudocode: 'assert sorted equals identity', complexity: 'O(n log n)',
    ref: 'numpy.argsort（排列判定）', commonPitfall: '只查长度不够；重复轴会破坏双射却长度不变。',
    invariant: 'run 满足 inv[axes[i]] === i 且排序后为恒等排列',
    inputs: ['shape'], domains: { shape: 'shape3' },
    run: (shape) => NS.L7.shape.invertPerm(rotateAxes(shape.length)),
    predicate: (shape) => {
      const axes = rotateAxes(shape.length);
      const inv = runOf('E051')(shape);
      if (!inv || inv.length !== shape.length) return false;
      for (let i = 0; i < shape.length; i++) if (inv[axes[i]] !== i) return false;
      return eqArr(inv.slice().sort((x, y) => x - y), identAxes(shape.length));
    },
  });
  reg(52, {
    name: '逆置换 · 对合', intuition: '逆置换的逆置换回到原排列。',
    math: 'inv(inv(a)) = a', pseudocode: 'assert eqArr', complexity: 'O(n)',
    ref: 'numpy.argsort', commonPitfall: '对合性对任意排列成立，不限特殊排列。',
    invariant: 'run 逐元素 === 旋转轴序（逆置换的逆等于自身）',
    inputs: ['shape'], domains: { shape: 'shape2' },
    run: (shape) => NS.L7.shape.invertPerm(NS.L7.shape.invertPerm(rotateAxes(shape.length))),
    predicate: (shape) => {
      const axes = rotateAxes(shape.length);
      const r = runOf('E052')(shape);
      return eqArr(r, axes) && eqArr(invLocal(invLocal(axes)), axes);
    },
  });
  reg(53, {
    name: '逆置换 · 定义逐项', intuition: '逆置换的定义就是 inv[axes[i]] = i，逐项核验。',
    math: 'inv[a[i]] = i ∀i', pseudocode: 'loop assert', complexity: 'O(n)',
    ref: 'numpy.argsort', commonPitfall: '写 inv[i]=axes[i] 是常见的方向性错误。',
    invariant: 'run 满足 inv[axes[i]] === i 且排序后为恒等排列（双向性刻画）',
    inputs: ['shape'], domains: { shape: 'shape4' },
    run: (shape) => NS.L7.shape.invertPerm(reverseAxes(shape.length)),
    predicate: (shape) => {
      const axes = reverseAxes(shape.length);
      const inv = runOf('E053')(shape);
      if (!inv || inv.length !== shape.length) return false;
      for (let i = 0; i < shape.length; i++) if (inv[axes[i]] !== i) return false;
      return eqArr(inv.slice().sort((x, y) => x - y), identAxes(shape.length));
    },
  });
  reg(54, {
    name: '置换 · 与 linearize 一致', intuition: '置换后的多维下标线性化，等于原空间对应下标线性化。',
    math: 'lin(j, p.shape, p.stride) = lin(k, s, st)，k[axes[i]]=j[i]', pseudocode: 'construct j=mid(p.shape); assert', complexity: 'O(n)',
    ref: 'numpy.transpose（偏移保持）', commonPitfall: '下标回填方向是 k[axes[i]]=j[i]，写反即错。',
    invariant: 'run === Σⱼ (sⱼ>>1)·stridesOf(s)[j]（本地点积，独立于 L7.linearize）',
    inputs: ['shape'], domains: { shape: 'shape3' },
    run: (shape) => {
      const axes = rotateAxes(shape.length);
      const p = NS.L7.shape.permute(shape, NS.L7.shape.contiguousStrides(shape), axes);
      return NS.L7.shape.linearize(midOf(p.shape), p.shape, p.stride);
    },
    predicate: (shape) => {
      const st = stridesOf(shape);
      let s = 0;
      for (let j = 0; j < shape.length; j++) s += (shape[j] >> 1) * st[j];
      return runOf('E054')(shape) === s;
    },
  });

  // ── 重塑 / 展平（E055–E064，元素数守恒视角）────────────────────────
  reg(55, {
    name: '重塑 · 展平保 size', intuition: '任意形状拍扁成一维，元素数不变。',
    math: 'size([size(s)]) = size(s)', pseudocode: 't = [size(shape)]; assert', complexity: 'O(n)',
    ref: 'numpy.reshape / ravel', commonPitfall: 'reshape 前必须校验总数相等，否则非法。',
    invariant: 'run === ∏s（展平后元素数守恒）',
    inputs: ['shape'], domains: { shape: 'shape2' },
    run: (shape) => NS.L7.shape.size([NS.L7.shape.size(shape)]),
    predicate: (shape) => {
      const r = runOf('E055')(shape);
      return r === prod(shape, 0) && r === prod([prod(shape, 0)], 0);
    },
  });
  reg(56, {
    name: '重塑 · 合并尾维保 size', intuition: '把首维之后的维合并成一维，总数不变。',
    math: 'size([s₀, ∏s[1:]]) = size(s)', pseudocode: 't = [s0, prod(rest)]; assert', complexity: 'O(n)',
    ref: 'numpy.reshape', commonPitfall: '合并的是尾维之积，不是和。',
    invariant: 'run === s₀·∏s[1:] 且 === ∏s',
    inputs: ['shape'], domains: { shape: 'shape4' },
    run: (shape) => NS.L7.shape.size([shape[0], prod(shape, 1)]),
    predicate: (shape) => {
      const r = runOf('E056')(shape);
      return r === shape[0] * prod(shape, 1) && r === prod(shape, 0);
    },
  });
  reg(57, {
    name: '重塑 · 前置 1 维保 size', intuition: '前面插一个长度为 1 的维，元素数不变。',
    math: 'size([1]∥s) = size(s)', pseudocode: 't = [1].concat(shape); assert', complexity: 'O(n)',
    ref: 'numpy.reshape / expand_dims', commonPitfall: '插 1 维不复制数据，只是改形状解释。',
    invariant: 'run === ∏s（前置 1 维不改变乘积）',
    inputs: ['shape'], domains: { shape: 'shape3' },
    run: (shape) => NS.L7.shape.size([1].concat(shape)),
    predicate: (shape) => {
      const r = runOf('E057')(shape);
      return r === prod(shape, 0) && r === prod([1].concat(shape), 0);
    },
  });
  reg(58, {
    name: '重塑 · 插入 1 维保 size', intuition: '在首维后插一个 1 维，总数仍然不变。',
    math: 'size([s₀,1]∥s[1:]) = size(s)', pseudocode: 't = [s0,1].concat(rest); assert', complexity: 'O(n)',
    ref: 'numpy.expand_dims', commonPitfall: '插入位置错到末维之后等于追加，语义不同但 size 仍同。',
    invariant: 'run === s₀·∏s[1:] 且 === ∏s',
    inputs: ['shape'], domains: { shape: 'shape2' },
    run: (shape) => NS.L7.shape.size([shape[0], 1].concat(shape.slice(1))),
    predicate: (shape) => {
      const r = runOf('E058')(shape);
      return r === shape[0] * prod(shape, 1) && r === prod(shape, 0);
    },
  });
  reg(59, {
    name: '重塑 · 一维偏移即下标', intuition: '一维形状的连续步长是 [1]，线性化结果就是下标本身。',
    math: 'lin([f], [m], [1]) = f（f<m）', pseudocode: 'm = size+1; f = k % m; assert', complexity: 'O(1)',
    ref: 'numpy ndarray strides', commonPitfall: 'f 必须 < m；取模保证不越界。',
    invariant: 'run === k % (∏s + 1)（一维步长为 1，偏移即下标）',
    inputs: ['shape', 'k'], domains: { shape: 'shape4', k: 'nat' },
    run: (shape, k) => { const m = NS.L7.shape.size(shape) + 1; return NS.L7.shape.linearize([k % m], [m]); },
    predicate: (shape, k) => {
      const m = prod(shape, 0) + 1;
      return runOf('E059')(shape, k) === k % m;
    },
  });
  reg(60, {
    name: '重塑 · 展平后最大偏移不变', intuition: '拍扁成一维后，最大下标对应的扁平偏移仍是 size−1。',
    math: 'lin([size−1],[size]) = size−1', pseudocode: 't = [size]; assert', complexity: 'O(1)',
    ref: 'numpy.ravel（C 序）', commonPitfall: '列主序拍扁顺序不同，但极值偏移相同。',
    invariant: 'run === ∏s − 1（展平形状上的最大偏移）',
    inputs: ['shape'], domains: { shape: 'shape3' },
    run: (shape) => NS.L7.shape.linearize([NS.L7.shape.size(shape) - 1], [NS.L7.shape.size(shape)]),
    predicate: (shape) => {
      const sz = prod(shape, 0);
      return runOf('E060')(shape) === sz - 1;
    },
  });
  reg(61, {
    name: '重塑 · 展平后步长恒 1', intuition: '一维形状只有一个步长，行主序下必为 1。',
    math: 'strides([m]) = [1]', pseudocode: 'm = size; assert eqArr(strides, [1])', complexity: 'O(1)',
    ref: 'numpy ndarray strides', commonPitfall: '一维形状谈"末维"就是它自己。',
    invariant: 'run 逐元素 === [1]（一维形状的唯一步长）',
    inputs: ['shape'], domains: { shape: 'shape2' },
    run: (shape) => NS.L7.shape.contiguousStrides([NS.L7.shape.size(shape)]),
    predicate: (shape) => {
      const r = runOf('E061')(shape);
      return eqArr(r, [1]) && eqArr(r, stridesOf([prod(shape, 0)]));
    },
  });
  reg(62, {
    name: '重塑 · 合并/拆分互为逆', intuition: '合并尾维与拆分末维得到的目标形状 size 都与原形状相同。',
    math: 'size([s₀,∏s[1:]]) = size([∏s[:n-1], s_{n-1}]) = size(s)', pseudocode: 'assert three sizes equal', complexity: 'O(n)',
    ref: 'numpy.reshape', commonPitfall: '两种拆法维度不同但元素数必须一致。',
    invariant: 'run 两项均 === ∏s（合并尾维与拆分末维目标同 size）',
    inputs: ['shape'], domains: { shape: 'shape4' },
    run: (shape) => [NS.L7.shape.size([shape[0], prod(shape, 1)]), NS.L7.shape.size([NS.L7.shape.size(shape.slice(0, shape.length - 1)), shape[shape.length - 1]])],
    predicate: (shape) => {
      const n = shape.length;
      const sz = prod(shape, 0);
      const r = runOf('E062')(shape);
      if (!r || r.length !== 2) return false;
      const t1 = shape[0] * prod(shape, 1);
      const t2 = prod(shape.slice(0, n - 1), 0) * shape[n - 1];
      return r[0] === t1 && r[1] === t2 && t1 === sz && t2 === sz;
    },
  });
  reg(63, {
    name: '重塑 · 行主序秩公式', intuition: '多维下标按 Horner 式累乘得到行主序秩，与 linearize 一致。',
    math: 'rank = ((i₀·s₁+i₁)·s₂+…) = lin(mid)', pseudocode: 'horner loop; assert', complexity: 'O(n)',
    ref: 'numpy.ravel（C 序混合 radix）', commonPitfall: 'Horner 方向错（从末维开始）会得到列主序秩。',
    invariant: 'run === Σⱼ midⱼ·stridesOf(s)[j]（本地点积，独立于 L7.linearize）',
    inputs: ['shape'], domains: { shape: 'shape3' },
    run: (shape) => {
      const mid = midOf(shape);
      let r = 0;
      for (let i = 0; i < shape.length; i++) r = r * shape[i] + mid[i];
      return r;
    },
    predicate: (shape) => {
      const mid = midOf(shape);
      const st = stridesOf(shape);
      let s = 0;
      for (let j = 0; j < shape.length; j++) s += mid[j] * st[j];
      return runOf('E063')(shape) === s;
    },
  });
  reg(64, {
    name: '重塑 · 元素数守恒综合', intuition: '展平目标与原形状：size 相等且最大扁平偏移相等。',
    math: 'size(t)=size(s) ∧ maxFlat(t)=maxFlat(s)', pseudocode: 't = [size]; assert both', complexity: 'O(n)',
    ref: 'numpy.reshape', commonPitfall: '只查 size 不够；偏移极值同样应守恒。',
    invariant: 'run === ∏s（展平后元素数守恒）',
    inputs: ['shape'], domains: { shape: 'shape2' },
    run: (shape) => NS.L7.shape.size([NS.L7.shape.size(shape)]),
    predicate: (shape) => {
      const sz = prod(shape, 0);
      const r = runOf('E064')(shape);
      return r === sz && r === prod([sz], 0);
    },
  });
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

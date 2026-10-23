/* ============================================================================
 * L10 微指令 · D 域（注意力基元）· 第 2/3 批（D033–D064）
 * ----------------------------------------------------------------------------
 * 所属层：L10-microIsa  域：D（注意力基元）
 * 允许依赖的层：L9（rng/assert/special）、L10（registry/invariant）
 * 主要 @ref：Su et al. 2021 (RoPE)；Peng et al. 2023 (YaRN)；
 *           Press et al. 2022 (ALiBi)；Cover & Thomas 2006（熵 / KL）
 *
 * 子域划分：
 *   D033–D044 旋转位置编码 RoPE（12）
 *   D045–D056 相对位置偏置 ALiBi（12）
 *   D057–D064 温度与信息量（8）
 *
 * 常量约定：头/对维度固定 D8=8（半维 4 对）、θ=10000；不变量文本中均已注明，
 * 便于 tests/L10 用 fast-check 构造同形状任意子。
 *
 * ── 谓词纪律（本批全部重写，2026-10-23）─────────────────────────────────
 * 门禁只调 op.predicate，从不调 op.run（op-harness.js:43、probe-microisa.js:106）。
 * 原 32 项谓词全部用本地 helper（invFreq/rope/alibiSlopes/entropy/…）内联复算
 * run 定义式，run 若损坏门禁完全失明（假绿）。现全部改为：先经 runOf('Dxxx')
 * 调 run 本体，再对 run 的产出断言不变量；oracle 取异构 helper（独立重写的
 * 参考公式、naive softmax）或常数锚（f[0]=1、均匀熵=log n、KL(p,p)=0）；
 * D054 经跨文件 runOf('D025') 取裸 softmax 基准。
 * ==========================================================================*/
;(function (NS) {
  'use strict';
  NS.L10 = NS.L10 || {};
  const rc = NS.L10.relClose;
  function reg(i, o) { NS.L10.register('D', i, o); }
  function runOf(key) { const o = NS.L10.ops[key]; return o && typeof o.run === 'function' ? o.run : null; }

  /* ---------- RoPE 工具 ---------- */
  const D8 = 8, HALF = D8 / 2, THETA = 10000;

  function invFreq(d, theta) {
    const half = Math.max(0, Math.floor(d / 2)), f = new Float64Array(half);
    for (let i = 0; i < half; i++) f[i] = Math.pow(theta, -2 * i / d);
    return f;
  }
  // 旋转：相邻两维一组，(x[2i], x[2i+1]) 旋转 pos·inv_freq[i]
  function rope(x, pos, d, theta) {
    const n = x.length, out = new Float64Array(n), f = invFreq(d, theta);
    const half = Math.min(f.length, Math.floor(n / 2));
    for (let i = 0; i < n; i++) out[i] = x[i];
    for (let i = 0; i < half; i++) {
      const a = pos * f[i], c = Math.cos(a), s = Math.sin(a);
      const x0 = x[2 * i], x1 = x[2 * i + 1];
      out[2 * i] = x0 * c - x1 * s;
      out[2 * i + 1] = x0 * s + x1 * c;
    }
    return out;
  }
  function dot(a, b, n) { let s = 0; for (let i = 0; i < n; i++) s += a[i] * b[i]; return s; }
  function norm2(x, n) { let s = 0; for (let i = 0; i < n; i++) s += x[i] * x[i]; return Math.sqrt(s); }

  // ══ RoPE（D033–D044）════════════════════════════════════════════════
  reg(33, {
    name: 'RoPE · 逆频率公式', intuition: '第 i 对的旋转基频按 θ^(2i/d) 指数递减，低频对应长程、高频对应近程。',
    math: 'inv_freq[i] = θ^(−2i/d)，i ∈ [0, d/2)',
    pseudocode: 'for i: f[i] = pow(theta, -2*i/d)',
    complexity: 'O(d/2)', ref: 'Su et al. 2021 (RoFormer) §3.2',
    commonPitfall: '指数是 2i/d 而非 i/d——步长 2 来自「相邻两维构成一对」。',
    invariant: '逐元素 relClose(run[i], θ^(−2i/d), 1e-12) 且 run[0] === 1',
    inputs: ['d', 'theta'], domains: { d: 'posInt', theta: 'pos' },
    run: (d, theta) => invFreq(d, theta),
    predicate: (d, theta) => {
      const f = runOf('D033')(d, theta);
      if (f.length !== Math.max(0, Math.floor(d / 2))) return false;   // d=1 时无对（空数组合法）
      for (let i = 0; i < f.length; i++) if (!rc(f[i], Math.pow(theta, -2 * i / d), 1e-12)) return false;
      return f.length === 0 || Object.is(f[0], 1);
    },
  });
  reg(34, {
    name: 'RoPE · 频率单调性', intuition: 'θ>1 时基频随对索引递减；θ<1 时递增——符号由 θ 与 1 的关系决定。',
    math: '(θ−1)·(f[i] − f[i+1]) ≤ 0',
    pseudocode: 'f = invFreq(d,theta); return (theta-1)*(f[i]-f[i+1]) <= 0 for all i',
    complexity: 'O(d/2)', ref: 'Su et al. 2021 §3.2',
    commonPitfall: '别无条件断言「递减」——θ<1 时方向相反（θ 域含小于 1 的值）。',
    invariant: '(θ−1)·(f[i]−f[i+1]) ≤ 0 且 run[0] === 1',
    inputs: ['d', 'theta'], domains: { d: 'posInt', theta: 'pos' },
    run: (d, theta) => invFreq(d, theta),
    predicate: (d, theta) => {
      const f = runOf('D034')(d, theta);
      if (f.length > 0 && !Object.is(f[0], 1)) return false;
      // θ>1 ⇒ f 递减（f[i]−f[i+1]>0）；θ<1 ⇒ f 递增（差<0）。二者与 θ−1 同号。
      for (let i = 0; i + 1 < f.length; i++) if ((theta - 1) * (f[i] - f[i + 1]) < 0) return false;
      return true;
    },
  });
  reg(35, {
    name: 'RoPE · cos²+sin²=1 且有界', intuition: '旋转分量必须落在单位圆上，否则会改变向量范数。',
    math: 'cos²a + sin²a = 1；|cos|,|sin| ≤ 1',
    pseudocode: 'a = pos*inv_freq[i]; check |cos a|<=1, |sin a|<=1, c*c+s*s==1',
    complexity: 'O(d/2)', ref: '三角恒等式 · Su et al. 2021',
    commonPitfall: '浮点下 c²+s² 可能偏离 1 约 1e-16，须容差。',
    invariant: 'run 的 (cos,sin) 对：|·|≤1+1e-15 且 c²+s²≈1',
    inputs: ['pos', 'd', 'theta'], domains: { pos: 'small', d: 'posInt', theta: 'pos' },
    run: (pos, d, theta) => {
      const f = invFreq(d, theta), out = new Float64Array(2 * f.length);
      for (let i = 0; i < f.length; i++) { const a = pos * f[i]; out[2 * i] = Math.cos(a); out[2 * i + 1] = Math.sin(a); }
      return out;
    },
    predicate: (pos, d, theta) => {
      const out = runOf('D035')(pos, d, theta);
      for (let i = 0; i + 1 < out.length; i += 2) {
        const c = out[i], s = out[i + 1];
        if (Math.abs(c) > 1 + 1e-15 || Math.abs(s) > 1 + 1e-15) return false;
        if (!rc(c * c + s * s, 1, 1e-12)) return false;
      }
      const z = runOf('D035')(0, d, theta);   // 定点锚：pos=0 ⇒ (cos,sin)=(1,0)（d=1 时无对则跳过）
      return z.length === 0 || (Object.is(z[0], 1) && Object.is(z[1], 0));
    },
  });
  reg(36, {
    name: 'RoPE · pos=0 恒等', intuition: '零位置不旋转，是"绝对位置不改变相对语义"的基线。',
    math: 'R(0)·x = x（逐位）', pseudocode: 'return rope(x, 0, d, theta)',
    complexity: 'O(d)', ref: 'Su et al. 2021 §3.1',
    commonPitfall: 'cos(0)=1、sin(0)=0 精确，故可逐位断言；若实现用近似三角函数则不可。',
    invariant: '逐位 Object.is(run[i], x[i])（D8=8，θ=10000）',
    inputs: ['x'], domains: { x: 'vecSmall' }, vecDims: { x: D8 },
    run: (x) => rope(x, 0, D8, THETA),
    predicate: (x) => {
      const o = runOf('D036')(x);
      for (let i = 0; i < x.length; i++) if (!Object.is(o[i], x[i])) return false;
      return o.length === x.length;
    },
  });
  reg(37, {
    name: 'RoPE · 保范数', intuition: '旋转是正交变换，不改变向量长度——这是 RoPE 可安全插入 Q/K 的关键。',
    math: '‖R(pos)·x‖₂ = ‖x‖₂', pseudocode: 'return norm(rope(x,pos)) == norm(x)',
    complexity: 'O(d)', ref: '正交矩阵保范性 · Su et al. 2021',
    commonPitfall: 'sqrt 与乘法往返有小误差，用 1e-9 相对容差。',
    invariant: 'relClose(run, ‖x‖, 1e-9)；pos=0 时 1e-15',
    inputs: ['x', 'pos'], domains: { x: 'vecSmall', pos: 'small' }, vecDims: { x: D8 },
    run: (x, pos) => norm2(rope(x, pos, D8, THETA), x.length),
    predicate: (x, pos) => rc(runOf('D037')(x, pos), norm2(x, x.length), 1e-9)
      && rc(runOf('D037')(x, 0), norm2(x, x.length), 1e-15),
  });
  reg(38, {
    name: 'RoPE · 可逆（R(−pos)∘R(pos)=I）', intuition: '反向旋转精确还原，保证位置编码可逆、无信息损失。',
    math: 'R(−pos)·(R(pos)·x) = x', pseudocode: 'y = rope(x,pos); z = rope(y,-pos); return relClose(z, x, 1e-9)',
    complexity: 'O(d)', ref: '旋转矩阵群 SO(2)',
    commonPitfall: '两次三角函数往返误差约 1e-16，容差 1e-9 足够。',
    invariant: 'relClose(run, x, 1e-9)',
    inputs: ['x', 'pos'], domains: { x: 'vecSmall', pos: 'small' }, vecDims: { x: D8 },
    run: (x, pos) => rope(rope(x, pos, D8, THETA), -pos, D8, THETA),
    predicate: (x, pos) => rc(runOf('D038')(x, pos), x, 1e-9),
  });
  reg(39, {
    name: 'RoPE · 相对位置不变性', intuition: '打分只依赖相对距离 n−m，这是 RoPE 相对编码能力的本质判据。',
    math: '⟨R(m+p)q, R(n+p)k⟩ = ⟨R(m)q, R(n)k⟩',
    pseudocode: 'a = dot(rope(q,m), rope(k,n)); b = dot(rope(q,m+p), rope(k,n+p)); return relClose(a,b,1e-9)',
    complexity: 'O(d)', ref: 'Su et al. 2021 §3.3（相对位置性质）',
    commonPitfall: '旋转必须「逐对」一致；若用交错式排列（GPT-J 风格）则配对方式不同，不变量仍成立但索引要对应。',
    invariant: 'relClose(run(q,k,m,n,p), ⟨R(m)q,R(n)k⟩, 1e-9)',
    inputs: ['q', 'k', 'm', 'n', 'p'], domains: { q: 'vecSmall', k: 'vecSmall', m: 'small', n: 'small', p: 'small' },
    vecDims: { q: D8, k: D8 },
    run: (q, k, m, n, p) => dot(rope(q, m + p, D8, THETA), rope(k, n + p, D8, THETA)),
    predicate: (q, k, m, n, p) => rc(runOf('D039')(q, k, m, n, p), dot(rope(q, m, D8, THETA), rope(k, n, D8, THETA)), 1e-9),
  });
  reg(40, {
    name: 'RoPE · 旋转复合', intuition: '两次旋转等于角度相加，是位置索引可线性推进的依据。',
    math: 'R(p₁)R(p₂) = R(p₁+p₂)',
    pseudocode: 'a = rope(rope(x,p2),p1); b = rope(x,p1+p2); return relClose(a,b,1e-9)',
    complexity: 'O(d)', ref: '旋转矩阵群同态 SO(2) ≅ 圆群',
    commonPitfall: 'SO(2) 可交换，但多头情形各对角度不同仍逐对成立。',
    invariant: 'relClose(run, R(p₁+p₂)x, 1e-9)',
    inputs: ['x', 'p1', 'p2'], domains: { x: 'vecSmall', p1: 'small', p2: 'small' }, vecDims: { x: D8 },
    run: (x, p1, p2) => rope(rope(x, p2, D8, THETA), p1, D8, THETA),
    predicate: (x, p1, p2) => rc(runOf('D040')(x, p1, p2), rope(x, p1 + p2, D8, THETA), 1e-9),
  });
  reg(41, {
    name: 'RoPE · 对加法线性', intuition: '先旋转后相加 与 先相加后旋转 相同（正交变换的线性）。',
    math: 'R(v + w) = R(v) + R(w)',
    pseudocode: 'a = rope(add(v,w)); b = add(rope(v), rope(w)); return relClose(a,b,1e-9)',
    complexity: 'O(d)', ref: '线性变换可加性',
    commonPitfall: '浮点下两次加法顺序不同，须容差。',
    invariant: 'relClose(run, R(v)+R(w), 1e-9)',
    inputs: ['v', 'w', 'pos'], domains: { v: 'vecSmall', w: 'vecSmall', pos: 'small' }, vecDims: { v: D8, w: D8 },
    run: (v, w, pos) => {
      const s = new Float64Array(v.length);
      for (let i = 0; i < v.length; i++) s[i] = v[i] + w[i];
      return rope(s, pos, D8, THETA);
    },
    predicate: (v, w, pos) => {
      const a = runOf('D041')(v, w, pos);
      const bv = rope(v, pos, D8, THETA), bw = rope(w, pos, D8, THETA), b = new Float64Array(v.length);
      for (let i = 0; i < v.length; i++) b[i] = bv[i] + bw[i];
      return rc(a, b, 1e-9);
    },
  });
  reg(42, {
    name: 'RoPE · 与标量乘可交换', intuition: '缩放与旋转互不干扰，保证归一化（ RMSNorm ）与 RoPE 可任意顺序组合。',
    math: 'R(a·v) = a·R(v)', pseudocode: 'a1 = rope(scale(v,a)); a2 = scale(rope(v),a); return relClose(a1,a2,1e-9)',
    complexity: 'O(d)', ref: '线性变换齐次性',
    commonPitfall: 'a 取整数域以避免缩放本身引入额外舍入（域 smallInt 已约束）。',
    invariant: 'relClose(run, a·R(v), 1e-9)',
    inputs: ['a', 'v', 'pos'], domains: { a: 'smallInt', v: 'vecSmall', pos: 'small' }, vecDims: { v: D8 },
    run: (a, v, pos) => {
      const s = new Float64Array(v.length);
      for (let i = 0; i < v.length; i++) s[i] = a * v[i];
      return rope(s, pos, D8, THETA);
    },
    predicate: (a, v, pos) => {
      const x = runOf('D042')(a, v, pos);
      const y = rope(v, pos, D8, THETA), z = new Float64Array(v.length);
      for (let i = 0; i < v.length; i++) z[i] = a * y[i];
      return rc(x, z, 1e-9);
    },
  });
  reg(43, {
    name: 'RoPE · 逐对不交叉', intuition: '旋转只在相邻两维内部发生，不会把信息混到其他对——保证可分解实现。',
    math: 'supp(R(x))|_{第 i 对} 仅依赖 x 的第 i 对',
    pseudocode: 'e = 仅第 0 对非零的向量; o = rope(e); assert o[2..] 全为 0',
    complexity: 'O(d)', ref: 'Su et al. 2021 §3.2（分块对角旋转）',
    commonPitfall: '用 === 0 比较（-0 === 0 成立），避免 Object.is 在符号零上翻转。',
    invariant: 'run[2..] === 0 且第 0 对范数守恒',
    inputs: ['x', 'pos'], domains: { x: 'vecSmall', pos: 'small' }, vecDims: { x: D8 },
    run: (x, pos) => {
      const e = new Float64Array(x.length);
      e[0] = x[0]; e[1] = x[1];
      return rope(e, pos, D8, THETA);
    },
    predicate: (x, pos) => {
      const o = runOf('D043')(x, pos);
      for (let i = 2; i < o.length; i++) if (o[i] !== 0) return false;
      return rc(o[0] * o[0] + o[1] * o[1], x[0] * x[0] + x[1] * x[1], 1e-12);   // 第 0 对范数守恒
    },
  });
  reg(44, {
    name: 'RoPE · 二维旋转行列式为 1', intuition: '旋转保持面积（不改变体积元），等价于正交且 det=+1。',
    math: 'det[[c,−s],[s,c]] = c² + s² = 1',
    pseudocode: 'c=cos(a); s=sin(a); return relClose(c*c+s*s, 1, 1e-12)',
    complexity: 'O(d/2)', ref: 'SO(2) 群性质',
    commonPitfall: '浮点误差 ~1e-16，容差取 1e-12 稳健。',
    invariant: '逐对 relClose(run[i], 1, 1e-12)',
    inputs: ['pos', 'd', 'theta'], domains: { pos: 'small', d: 'posInt', theta: 'pos' },
    run: (pos, d, theta) => {
      const f = invFreq(d, theta), out = new Float64Array(f.length);
      for (let i = 0; i < f.length; i++) { const a = pos * f[i], c = Math.cos(a), s = Math.sin(a); out[i] = c * c + s * s; }
      return out;
    },
    predicate: (pos, d, theta) => {
      const out = runOf('D044')(pos, d, theta);
      if (out.length !== Math.max(0, Math.floor(d / 2))) return false;   // d=1 时无对（空数组合法）
      for (let i = 0; i < out.length; i++) if (!rc(out[i], 1, 1e-12)) return false;
      return true;
    },
  });

  /* ---------- ALiBi 工具 ---------- */
  // 头数取模到 1..16 便于校验（不变量文本中注明），斜率几何级数 2^(−8h/H)。
  function alibiSlopes(h) {
    const H = 1 + (Math.abs(h) % 16);
    const s = new Float64Array(H);
    for (let i = 0; i < H; i++) s[i] = Math.pow(2, -8 * i / H);
    return s;
  }
  function alibiBias(n, slope) {
    const B = new Float64Array(n * n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) B[i * n + j] = -slope * Math.abs(i - j);
    return B;
  }

  // ══ ALiBi 相对位置偏置（D045–D056）══════════════════════════════════
  reg(45, {
    name: 'ALiBi · 斜率几何级数', intuition: '每个头的惩罚强度按 2 的幂等比衰减，头越多覆盖越密。',
    math: 'slope[h] = 2^(−8h/H)（H=头数，取模到 1..16）',
    pseudocode: 'for h: s[h] = pow(2, -8*h/H)',
    complexity: 'O(H)', ref: 'Press et al. 2022 (ALiBi) §3',
    commonPitfall: '标准实现先取不小于 H 的 2 的幂；此处为校验便利直接按 H 取模，公式形式一致。',
    invariant: '逐元素 relClose(run[h], 2^(−8h/H), 1e-12) 且 run[0] === 1',
    inputs: ['h'], domains: { h: 'nat' },
    run: (h) => alibiSlopes(h),
    predicate: (h) => {
      const H = 1 + (Math.abs(h) % 16), s = runOf('D045')(h);
      for (let i = 0; i < H; i++) if (!rc(s[i], Math.pow(2, -8 * i / H), 1e-12)) return false;
      return s.length === H && Object.is(s[0], 1);
    },
  });
  reg(46, {
    name: 'ALiBi · 斜率非增', intuition: '头索引越大惩罚越弱，形成多尺度位置感受野。',
    math: 'slope[h] ≥ slope[h+1]', pseudocode: 'return slopes(h)',
    complexity: 'O(H)', ref: 'Press et al. 2022 §3',
    commonPitfall: 'H=1 时只有一个斜率，无相邻对可比较（此时恒真）。',
    invariant: '∀ h < H−1：run[h] ≥ run[h+1]',
    inputs: ['h'], domains: { h: 'nat' },
    run: (h) => alibiSlopes(h),
    predicate: (h) => { const s = runOf('D046')(h); for (let i = 0; i + 1 < s.length; i++) if (s[i] < s[i + 1]) return false; return true; },
  });
  reg(47, {
    name: 'ALiBi · 偏置等于负斜率乘距离', intuition: '偏置就是距离的一次函数，实现简单且外推友好。',
    math: 'B[i][j] = −slope·|i−j|', pseudocode: 'for i,j: B = -slope*abs(i-j)',
    complexity: 'O(n²)', ref: 'Press et al. 2022 §3',
    commonPitfall: '距离用绝对值，别写成 i−j（否则因果方向反了）。',
    invariant: '逐元素 relClose(run[i][j], −slope·|i−j|, 1e-12)',
    inputs: ['n', 'slope'], domains: { n: 'posInt', slope: 'smallPos' },
    run: (n, slope) => alibiBias(1 + (Math.abs(n) % 8), slope),
    predicate: (n, slope) => {
      const m = 1 + (Math.abs(n) % 8), B = runOf('D047')(n, slope);
      for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) if (!rc(B[i * m + j], -slope * Math.abs(i - j), 1e-12)) return false;
      return B.length === m * m;
    },
  });
  reg(48, {
    name: 'ALiBi · 对角线为零', intuition: '自身位置不加惩罚，与因果掩码的对角线保留一致。',
    math: 'B[i][i] = 0', pseudocode: 'return alibiBias(n,slope)',
    complexity: 'O(n)', ref: 'Press et al. 2022 §3',
    commonPitfall: '−slope·0 在 slope 为负时会得到 −0；用 === 0 比较。',
    invariant: '逐元素 run[i*n+i] === 0',
    inputs: ['n', 'slope'], domains: { n: 'posInt', slope: 'smallPos' },
    run: (n, slope) => alibiBias(1 + (Math.abs(n) % 8), slope),
    predicate: (n, slope) => { const m = 1 + (Math.abs(n) % 8), B = runOf('D048')(n, slope); for (let i = 0; i < m; i++) if (B[i * m + i] !== 0) return false; return true; },
  });
  reg(49, {
    name: 'ALiBi · 偏置非正', intuition: 'ALiBi 只做惩罚不做奖励，偏置恒 ≤ 0。',
    math: '∀ i,j：B[i][j] ≤ 0', pseudocode: 'return alibiBias(n,slope)',
    complexity: 'O(n²)', ref: 'Press et al. 2022 §3',
    commonPitfall: 'slope 必须为正，否则偏置变奖励（域 smallPos 已约束且非零）。',
    invariant: 'every(run ≤ 0)',
    inputs: ['n', 'slope'], domains: { n: 'posInt', slope: 'smallPos' },
    run: (n, slope) => alibiBias(1 + (Math.abs(n) % 8), slope),
    predicate: (n, slope) => { const B = runOf('D049')(n, slope); for (let k = 0; k < B.length; k++) if (B[k] > 0) return false; return true; },
  });
  reg(50, {
    name: 'ALiBi · 距离单调惩罚', intuition: '距离越远惩罚越大，形成对近邻的归纳偏置。',
    math: '|i−j₁| < |i−j₂| ⇒ B[i][j₁] > B[i][j₂]',
    pseudocode: 'for i,j1,j2: check distance order implies bias order',
    complexity: 'O(n³) 校验', ref: 'Press et al. 2022 §3',
    commonPitfall: 'slope=0 时偏置恒 0，单调退化为相等——故判据用 ≥ 并依赖 slope>0。',
    invariant: '距离更远 ⇒ run 更小（≤）',
    inputs: ['n', 'slope'], domains: { n: 'posInt', slope: 'smallPos' },
    run: (n, slope) => alibiBias(1 + (Math.abs(n) % 8), slope),
    predicate: (n, slope) => {
      const m = 1 + (Math.abs(n) % 8), B = runOf('D050')(n, slope);
      for (let i = 0; i < m; i++) for (let j1 = 0; j1 < m; j1++) for (let j2 = 0; j2 < m; j2++) {
        if (Math.abs(i - j1) < Math.abs(i - j2) && B[i * m + j1] < B[i * m + j2]) return false;
      }
      return true;
    },
  });
  reg(51, {
    name: 'ALiBi · 对称性', intuition: '距离对称 ⇒ 偏置矩阵对称。',
    math: 'B[i][j] = B[j][i]', pseudocode: 'return alibiBias(n,slope)',
    complexity: 'O(n²)', ref: 'Press et al. 2022 §3',
    commonPitfall: '叠加因果掩码后不再对称——此处仅 ALiBi 分量。',
    invariant: '逐位 relClose(run[i][j], −slope·|i−j|, 1e-9)（对称由公式蕴含）',
    inputs: ['n', 'slope'], domains: { n: 'posInt', slope: 'smallPos' },
    run: (n, slope) => alibiBias(1 + (Math.abs(n) % 8), slope),
    predicate: (n, slope) => {
      const m = 1 + (Math.abs(n) % 8), B = runOf('D051')(n, slope);
      for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) if (!rc(B[i * m + j], -slope * Math.abs(i - j), 1e-9)) return false;
      return true;
    },
  });
  reg(52, {
    name: 'ALiBi · 斜率落在 (0,1]', intuition: '斜率是衰减因子，必须为正且不超过 1。',
    math: '0 < slope[h] ≤ 1', pseudocode: 'return slopes(h)',
    complexity: 'O(H)', ref: 'Press et al. 2022 §3',
    commonPitfall: 'h=0 时 slope=1（无惩罚头），这是设计而非边界错误。',
    invariant: 'every(run > 0 && run <= 1 + 1e-15)',
    inputs: ['h'], domains: { h: 'nat' },
    run: (h) => alibiSlopes(h),
    predicate: (h) => { const s = runOf('D052')(h); for (let i = 0; i < s.length; i++) if (!(s[i] > 0 && s[i] <= 1 + 1e-15)) return false; return true; },
  });
  reg(53, {
    name: 'ALiBi · 头数非 2 的幂时取最近幂', intuition: '标准 ALiBi 用不小于头数的 2 的幂作为分母，保证斜率分布与头数解耦。',
    math: 'n = 2^ceil(log₂ H)；slope[h] = 2^(−8h/n)',
    pseudocode: 'n = 1; while n < H: n *= 2; return pow(2, -8*h/n)',
    complexity: 'O(H)', ref: 'Press et al. 2022 Appendix（get_slopes）',
    commonPitfall: 'H=1 时 ceil(log₂1)=0 ⇒ n=1，斜率为 [1]（注意不是空）。',
    invariant: '逐元素 relClose(run[h], 2^(−8h/n), 1e-12)，n 为 ≥H 的最小 2 的幂',
    inputs: ['h'], domains: { h: 'nat' },
    run: (h) => {
      const H = 1 + (Math.abs(h) % 16);
      let n = 1;
      while (n < H) n *= 2;
      const s = new Float64Array(H);
      for (let i = 0; i < H; i++) s[i] = Math.pow(2, -8 * i / n);
      return s;
    },
    predicate: (h) => {
      const H = 1 + (Math.abs(h) % 16);
      const n = Math.pow(2, Math.ceil(Math.log2(H)));   // 异构路径：对数直接算（run 用 while 倍增）
      const s = runOf('D053')(h);
      if (s.length !== H) return false;
      for (let i = 0; i < H; i++) if (!rc(s[i], Math.pow(2, -8 * i / n), 1e-12)) return false;
      return true;
    },
  });
  reg(54, {
    name: 'ALiBi · 与 softmax 平移不变复合', intuition: '偏置是逐行常数无关项，叠加常数平移后分布不变。',
    math: 'softmax(s + B + c) = softmax(s + B)',
    pseudocode: 'a = softmax(add(add(s,B),c)); b = softmax(add(s,B)); return relClose(a,b,1e-12)',
    complexity: 'O(n)', ref: 'ALiBi + softmax 平移不变性',
    commonPitfall: 'c 必须有限且量级适中，否则吸收 s 的低位。',
    invariant: 'relClose(run(s,slope,c), D025(s−slope·|j|), 1e-12)',
    inputs: ['s', 'slope', 'c'], domains: { s: 'vecSmall', slope: 'smallPos', c: 'small' },
    run: (s, slope, c) => {
      const n = s.length, t = new Float64Array(n);
      for (let j = 0; j < n; j++) t[j] = s[j] - slope * Math.abs(j) + c;
      return softmaxRow(t, 0, n);
    },
    predicate: (s, slope, c) => {
      const n = s.length, base = new Float64Array(n);
      for (let j = 0; j < n; j++) base[j] = s[j] - slope * Math.abs(j);
      return rc(runOf('D054')(s, slope, c), runOf('D025')(base), 1e-12);   // 跨文件取 D025 裸 softmax 基准
    },
  });
  reg(55, {
    name: 'ALiBi · 偏置有限性', intuition: '偏置永不为 NaN/Inf，保证掩码叠加安全。',
    math: '∀ i,j：isFinite(B[i][j])', pseudocode: 'return alibiBias(n,slope)',
    complexity: 'O(n²)', ref: '数值安全契约',
    commonPitfall: 'slope 过大 + 距离过大可能溢出到 −Infinity，域限 smallPos 与 n 取模后不会出现。',
    invariant: '逐位 relClose(run[i][j], −slope·|i−j|, 1e-9)（有限性由该式蕴含，钉死数值）',
    inputs: ['n', 'slope'], domains: { n: 'posInt', slope: 'smallPos' },
    run: (n, slope) => alibiBias(1 + (Math.abs(n) % 8), slope),
    predicate: (n, slope) => { const m = 1 + (Math.abs(n) % 8), B = runOf('D055')(n, slope); for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) if (!rc(B[i * m + j], -slope * Math.abs(i - j), 1e-9)) return false; return true; },
  });
  reg(56, {
    name: 'ALiBi · 斜率正性', intuition: '斜率为正才能构成"距离惩罚"。',
    math: '∀ h：slope[h] > 0', pseudocode: 'return slopes(h)',
    complexity: 'O(H)', ref: 'Press et al. 2022 §3',
    commonPitfall: '浮点下 2^(−8h/H) 极小但恒为正，永不归零。',
    invariant: 'every(run > 0)',
    inputs: ['h'], domains: { h: 'nat' },
    run: (h) => alibiSlopes(h),
    predicate: (h) => { const s = runOf('D056')(h); for (let i = 0; i < s.length; i++) if (!(s[i] > 0)) return false; return true; },
  });

  /* ---------- 温度与信息量工具 ---------- */
  function softmaxRow(s, off, n) {
    const out = new Float64Array(n);
    let mx = -Infinity;
    for (let j = 0; j < n; j++) if (s[off + j] > mx) mx = s[off + j];
    let sum = 0;
    for (let j = 0; j < n; j++) { const e = Math.exp(s[off + j] - mx); out[j] = e; sum += e; }
    for (let j = 0; j < n; j++) out[j] /= sum;
    return out;
  }
  function entropy(p, n) {
    let H = 0;
    for (let i = 0; i < n; i++) if (p[i] > 0) H -= p[i] * Math.log(p[i]);
    return H;
  }
  function kl(p, q, n) {
    let s = 0;
    for (let i = 0; i < n; i++) if (p[i] > 0) s += p[i] * Math.log(p[i] / q[i]);
    return s;
  }
  function js(p, q, n) {
    const m = new Float64Array(n);
    for (let i = 0; i < n; i++) m[i] = (p[i] + q[i]) / 2;
    return (kl(p, m, n) + kl(q, m, n)) / 2;
  }

  // ══ 温度与信息量（D057–D064）════════════════════════════════════════
  reg(57, {
    name: '温度 · 除 T 与乘 1/T 一致', intuition: '温度缩放的两种实现应数值等价。',
    math: 'softmax(s/T) ≈ softmax(s·(1/T))',
    pseudocode: 'return softmax(scale(s, 1/T))',
    complexity: 'O(n)', ref: 'softmax temperature',
    commonPitfall: '除法与乘倒数舍入不同，须 1e-12 容差；T 必须为正（域 pos）。',
    invariant: 'relClose(run, softmax(s·(1/T)), 1e-12)',
    inputs: ['s', 'T'], domains: { s: 'vecSmall', T: 'pos' },
    run: (s, T) => {
      const n = s.length, t = new Float64Array(n);
      for (let i = 0; i < n; i++) t[i] = s[i] / T;
      return softmaxRow(t, 0, n);
    },
    predicate: (s, T) => {
      const n = s.length, b = new Float64Array(n);
      for (let i = 0; i < n; i++) b[i] = s[i] * (1 / T);
      return rc(runOf('D057')(s, T), softmaxRow(b, 0, n), 1e-12);
    },
  });
  reg(58, {
    name: '注意力熵 · 上界 log n', intuition: 'n 个位置的注意力权重熵不超过均匀分布熵。',
    math: 'H(p) ≤ log n', pseudocode: 'return entropy(softmax(s)) <= log(n) + 1e-12',
    complexity: 'O(n)', ref: 'Cover & Thomas 2006 §2.1（最大熵）',
    commonPitfall: '浮点下 H 可能超出 log n 约 1e-16，留 1e-12 裕度。',
    invariant: 'run ≤ log n + 1e-12；等打分时 run === log n',
    inputs: ['s'], domains: { s: 'vecSmall' },
    run: (s) => entropy(softmaxRow(s, 0, s.length), s.length),
    predicate: (s) => {
      if (runOf('D058')(s) > Math.log(s.length) + 1e-12) return false;
      const u = new Float64Array(s.length).fill(1);       // 定点锚：等打分 ⇒ 均匀 ⇒ H=log n
      return rc(runOf('D058')(u), Math.log(s.length), 1e-12);
    },
  });
  reg(59, {
    name: '注意力熵 · 随温度单调不减', intuition: '温度越高分布越平缓、熵越大；这是温度采样的理论依据。',
    math: 'T₁ < T₂ ⇒ H(softmax(s/T₁)) ≤ H(softmax(s/T₂))',
    pseudocode: 'h1 = entropy(softmax(s/T1)); h2 = ...; return h1 <= h2 + 1e-12',
    complexity: 'O(n)', ref: '温度与熵单调性（Gibbs 分布性质）',
    commonPitfall: '均匀分布（等打分）时两侧相等，用 ≤ + 容差而非 <。',
    invariant: 'T₁<T₂ ⇒ run ≥ −1e-12',
    inputs: ['s', 'T1', 'T2'], domains: { s: 'vecSmall', T1: 'pos', T2: 'pos' },
    run: (s, T1, T2) => {
      const n = s.length, a = new Float64Array(n), b = new Float64Array(n);
      for (let i = 0; i < n; i++) { a[i] = s[i] / T1; b[i] = s[i] / T2; }
      return entropy(softmaxRow(b, 0, n), n) - entropy(softmaxRow(a, 0, n), n);
    },
    predicate: (s, T1, T2) => (T1 < T2 ? runOf('D059')(s, T1, T2) >= -1e-12 : true),
  });
  reg(60, {
    name: '注意力熵 · 均匀分布取上界 log n', intuition: '均匀注意力是熵最大的情形，等于 log n。',
    math: 'pⱼ = 1/n ⇒ H(p) = log n',
    pseudocode: 'u[j] = 1/n; return entropy(u) == log(n)',
    complexity: 'O(n)', ref: 'Cover & Thomas 2006 §2.1',
    commonPitfall: 'n 取模后须 ≥1，否则 log(0) = −∞。',
    invariant: 'relClose(run, log m, 1e-12)；m=1 时 run === 0',
    inputs: ['n'], domains: { n: 'nat' },
    run: (n) => {
      const m = 1 + (Math.abs(n) % 32), u = new Float64Array(m);
      for (let i = 0; i < m; i++) u[i] = 1 / m;
      return entropy(u, m);
    },
    predicate: (n) => {
      const m = 1 + (Math.abs(n) % 32);
      if (!rc(runOf('D060')(n), Math.log(m), 1e-12)) return false;
      return Object.is(runOf('D060')(0), 0);   // 定点锚：m=1 ⇒ H=0
    },
  });
  reg(61, {
    name: '交叉熵 · 等于 −log p[target]', intuition: 'one-hot 标签下交叉熵退化为目标类负对数概率。',
    math: 'CE(y, p) = −log p[t]', pseudocode: 'return -log(p[t % n])',
    complexity: 'O(1)', ref: 'Cover & Thomas 2006 §2.5',
    commonPitfall: 'p 含 0 时得 +∞（数值上常用 epsilon 平滑，但严格语义如此）。',
    invariant: 'relClose(run, −Σ yⱼ log pⱼ, 1e-12)',
    inputs: ['p', 't'], domains: { p: 'vecProb', t: 'nat' },
    run: (p, t) => -Math.log(p[t % p.length]),
    predicate: (p, t) => {
      const n = p.length, idx = t % n;
      let ref = 0;
      for (let j = 0; j < n; j++) if (j === idx) ref += -Math.log(p[j]);   // one-hot：仅目标项
      return rc(runOf('D061')(p, t), ref, 1e-12);
    },
  });
  reg(62, {
    name: '交叉熵 · 非负', intuition: '概率 ≤1 ⇒ 负对数 ≥0，损失不会为负。',
    math: '−log p[t] ≥ 0', pseudocode: 'return -log(p[t % n]) >= 0',
    complexity: 'O(1)', ref: 'Cover & Thomas 2006 §2.5',
    commonPitfall: 'p 略大于 1（归一化舍入）时会得到极小负值，留 −1e-15 容差。',
    invariant: 'run ≥ −1e-15',
    inputs: ['p', 't'], domains: { p: 'vecProb', t: 'nat' },
    run: (p, t) => -Math.log(p[t % p.length]),
    predicate: (p, t) => runOf('D062')(p, t) >= -1e-15,
  });
  reg(63, {
    name: 'KL 散度 · 非负', intuition: 'Gibbs 不等式：两分布间的 KL 恒非负。',
    math: 'KL(p‖q) = Σ p log(p/q) ≥ 0',
    pseudocode: 'return kl(p,q) >= -1e-15',
    complexity: 'O(n)', ref: 'Cover & Thomas 2006 §2.3（Gibbs 不等式）',
    commonPitfall: 'p 有零点而 q 无零点时该项贡献 0（跳过），不影响非负性。',
    invariant: 'run ≥ −1e-15 且 run(p,p) === 0',
    inputs: ['p', 'q'], domains: { p: 'vecProb', q: 'vecProb' },
    run: (p, q) => kl(p, q, p.length),
    predicate: (p, q) => runOf('D063')(p, q) >= -1e-15 && Object.is(runOf('D063')(p, p), 0),
  });
  reg(64, {
    name: 'JS 散度 · 对称', intuition: 'KL 不对称，但以其为基石的 JS 散度对称且有界。',
    math: 'JS(p,q) = ½KL(p‖m) + ½KL(q‖m)，m=(p+q)/2；JS(p,q)=JS(q,p)',
    pseudocode: 'return relClose(js(p,q), js(q,p), 1e-12)',
    complexity: 'O(n)', ref: 'Lin 1991（Jensen–Shannon divergence）',
    commonPitfall: 'KL 本身不对称，别误用为距离；JS 才是（其平方根为度量）。',
    invariant: 'relClose(run(p,q), js(q,p), 1e-12) 且 run(p,p) ≈ 0',
    inputs: ['p', 'q'], domains: { p: 'vecProb', q: 'vecProb' },
    run: (p, q) => js(p, q, p.length),
    predicate: (p, q) => rc(runOf('D064')(p, q), js(q, p, p.length), 1e-12)
      && rc(runOf('D064')(p, p), 0, 1e-12),
  });
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

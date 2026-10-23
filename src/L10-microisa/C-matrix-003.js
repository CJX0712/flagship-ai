/* ============================================================================
 * L10 微指令 · C 域（矩阵/张量运算）· 第 4/4 批（C097–C128）
 * ----------------------------------------------------------------------------
 * 所属层：L10-microIsa  域：C（矩阵/张量）
 * 允许依赖的层：L8（matmul/reduce）、L9（special）、L10（registry/invariant/mh）；
 *              C124 行切片在调用期惰性复用 NS.L7.tensor.slice/contiguous
 *              （本文件先于 L7 求值，故不在模块顶层捕获，同 E 域先例）。
 * 主要 @ref：Horn & Johnson (1985)（行列式/特征值/三角结构）；Vieta（根和
 *           定理）；Bridle (1990)（softmax）；NumPy（拼接/切片/pad 语义）。
 *
 * 涵盖：矩阵结构与行列式（C097–C108）、归约进阶（C109–C120）、
 *       张量操作（C121–C128）。至此 C 域 128 槽位满额（T07）。
 * 矩阵以 Float64Array 行优先（K1），输入维度由 op.matrixDims 精确声明。
 * 整数域（matInt）下的恒等逐位精确（±0 归一 Object.is，r=4 时除以 4 为
 * 幂二故方差/均值路径亦精确）；含 exp/log/开方的数值断言用 relClose(1e-9)。
 * 复用清单（零重写）：matmul → NS.L8.matmul；softmaxRows → NS.L8.reduce；
 *   logsumexp2 → NS.L9.special；转置/缩放/全1等 → NS.L10.mh；
 *   行切片 → NS.L7.tensor.slice+contiguous。det/拼接/pad/对角提取在 L7/L8
 *   无裸矩阵内核，于本文件本地实现（谓词走独立参考路径）。
 *
 * ── 谓词纪律（本批全部重写，2026-10-23）─────────────────────────────────
 * 门禁只调 op.predicate，从不调 op.run（op-harness.js:43、probe-microisa.js:106）。
 * 原 32 项谓词虽多为"独立校验"写法，但校验对象是本地 helper（det2/isSym/
 * colVar/vcat/…）而非 run 本体——helper 与 run 同源，run 若损坏门禁失明。
 * 现全部改为：先经 runOf('Cxxx') 调 run 本体，再对 run 的产出断言；oracle
 * 取内联索引路径（行列式展开、列和、方差、前缀和、拼接定位）或常数锚。
 * ==========================================================================*/
;(function (NS) {
  'use strict';
  NS.L10 = NS.L10 || {};
  const M = NS.L8.matmul, R = NS.L8.reduce, SP = NS.L9.special, L10 = NS.L10;
  const mh = L10.mh;
  function reg(i, o) { NS.L10.register('C', i, o); }
  function runOf(key) { const o = NS.L10.ops[key]; return o && typeof o.run === 'function' ? o.run : null; }
  const { Z, madd, msub, mscale, mneg, mabs, mrelu, mclamp, emp, msq, trans, id, ones, allIn, sumAll, fro, colSum, meq } = mh;
  const rc = (a, b, t) => NS.L10.relClose(a, b, t);
  const z0 = (v) => v === 0 || v === -0; // ±0 归一判零（IEEE 中 −0 与 0 数值等价）

  // ── 本地助手（L7/L8 无对应裸矩阵内核）─────────────────────────────────
  // 行列式：2×2 公式；3×3 沿首行余子式展开（固定规模 O(1)）。
  function det2(A) { return A[0] * A[3] - A[1] * A[2]; }
  function det3(A) {
    return A[0] * (A[4] * A[8] - A[5] * A[7]) - A[1] * (A[3] * A[8] - A[5] * A[6]) + A[2] * (A[3] * A[7] - A[4] * A[6]);
  }
  // 结构投影：取对角/上三角/下三角部分（另一侧写 +0）。
  function diagPart(A, n) { const o = Z(n, n); for (let i = 0; i < n; i++) o[i * n + i] = A[i * n + i]; return o; }
  function triuPart(A, n) { const o = Float64Array.from(A); for (let i = 0; i < n; i++) for (let j = 0; j < i; j++) o[i * n + j] = 0; return o; }
  function trilPart(A, n) { const o = Float64Array.from(A); for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) o[i * n + j] = 0; return o; }
  // 结构判定（±0 归一后与定义逐位比对）。
  function isSym(A, n) {
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      let x = A[i * n + j], y = A[j * n + i];
      if (x === 0) x = 0; if (y === 0) y = 0;
      if (x !== y) return false;
    }
    return true;
  }
  function isAnti(A, n) {
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      let x = A[i * n + j], y = -A[j * n + i];
      if (x === 0) x = 0; if (y === 0) y = 0;
      if (x !== y) return false;
    }
    return true;
  }
  function isUpperTri(A, n) { for (let i = 0; i < n; i++) for (let j = 0; j < i; j++) if (!z0(A[i * n + j])) return false; return true; }
  function isLowerTri(A, n) { for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) if (!z0(A[i * n + j])) return false; return true; }
  // 张量结构操作。
  function vcat(A, B, r, c) { const o = new Float64Array(2 * r * c); o.set(A, 0); o.set(B, r * c); return o; }
  function hcat(A, B, r, c1, c2) {
    const o = new Float64Array(r * (c1 + c2));
    for (let i = 0; i < r; i++) {
      const s = i * (c1 + c2);
      o.set(A.subarray(i * c1, i * c1 + c1), s);
      o.set(B.subarray(i * c2, i * c2 + c2), s + c1);
    }
    return o;
  }
  function rowSlice(A, r, c, s, e) {
    const t = NS.L7.tensor.fromFlat(A, [r, c]);          // 惰性引用：调用期 L7 已加载
    return NS.L7.tensor.contiguous(NS.L7.tensor.slice(t, 0, s, e)).data;
  }
  function diagOf(A, n) { const o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = A[i * n + i]; return o; }
  function padTo(A, r, c, R2, C2) {
    const o = new Float64Array(R2 * C2);
    for (let i = 0; i < r; i++) o.set(A.subarray(i * c, i * c + c), i * C2);
    return o;
  }
  // 归约助手：perColSum 跨行累加（各列和，复用 L8 rowSum）；perRowSum 跨列累加（各行和，复用 mh.colSum）。
  const perColSum = (A, r, c) => R.rowSum(A, r, c);
  const perRowSum = (A, r, c) => mh.colSum(A, r, c);
  function cumsumRows(A, r, c) {
    const o = new Float64Array(r * c);
    for (let i = 0; i < r; i++) { const b = i * c; let s = 0; for (let j = 0; j < c; j++) { s += A[b + j]; o[b + j] = s; } }
    return o;
  }
  // 列 softmax = 转置的行 softmax 再转置（复用 L8 softmaxRows）。
  function colSoftmax(A, r, c) { return trans(R.softmaxRows(trans(A, r, c), c, r), c, r); }
  // argmax：并列取第一个最大位置。
  function argmaxRow(A, r, c) {
    const idx = new Int32Array(r);
    for (let i = 0; i < r; i++) { const o = i * c; let m = 0; for (let j = 1; j < c; j++) if (A[o + j] > A[o + m]) m = j; idx[i] = m; }
    return idx;
  }
  function argmaxCol(A, r, c) {
    const idx = new Int32Array(c);
    for (let j = 0; j < c; j++) { let m = 0; for (let i = 1; i < r; i++) if (A[i * c + j] > A[m * c + j]) m = i; idx[j] = m; }
    return idx;
  }
  // 行 log-sum-exp：折叠 L9 logsumexp2（数值稳定）。
  function rowLSE(A, r, c) {
    const o = new Float64Array(r);
    for (let i = 0; i < r; i++) { const b = i * c; let acc = A[b]; for (let j = 1; j < c; j++) acc = SP.logsumexp2(acc, A[b + j]); o[i] = acc; }
    return o;
  }
  function colMinMax(A, r, c) {
    const mn = new Float64Array(c), mx = new Float64Array(c);
    for (let j = 0; j < c; j++) {
      let lo = A[j], hi = A[j];
      for (let i = 1; i < r; i++) { const v = A[i * c + j]; if (v < lo) lo = v; if (v > hi) hi = v; }
      mn[j] = lo; mx[j] = hi;
    }
    return { min: mn, max: mx };
  }
  // 总体方差（除以个数）。本批一律 r=4（幂二），整数域下均值/偏差/平方/求和全程精确。
  function colVar(A, r, c) {
    const mu = perColSum(A, r, c);
    const o = new Float64Array(c);
    for (let j = 0; j < c; j++) {
      const m = mu[j] / r;
      let s = 0;
      for (let i = 0; i < r; i++) { const d = A[i * c + j] - m; s += d * d; }
      o[j] = s / r;
    }
    return o;
  }
  function rowVar(X, r, c) {
    const o = new Float64Array(r);
    for (let i = 0; i < r; i++) {
      const b = i * c;
      let s = 0;
      for (let j = 0; j < c; j++) s += X[b + j];
      const m = s / c;
      let v = 0;
      for (let j = 0; j < c; j++) { const d = X[b + j] - m; v += d * d; }
      o[i] = v / c;
    }
    return o;
  }
  // 内联列方差 oracle（4×5，总体方差除以 4）：与 colVar 完全独立的参考路径。
  function inlineColVar(A) {
    const o = new Float64Array(5);
    for (let j = 0; j < 5; j++) {
      let s = 0;
      for (let i = 0; i < 4; i++) s += A[i * 5 + j];
      const m = s / 4;
      let v = 0;
      for (let i = 0; i < 4; i++) { const d = A[i * 5 + j] - m; v += d * d; }
      o[j] = v / 4;
    }
    return o;
  }
  const THREES = mscale(3, ones(4, 5), 20); // 全 3 矩阵（4×5），供平移不变性对照
  const DIAG235 = (() => { const o = Z(3, 3); o[0] = 2; o[4] = 3; o[8] = 5; return o; })(); // diag(2,3,5)

  // ── 矩阵结构与行列式（C097–C108）──────────────────────────────────────
  reg(97, {
    name: 'det 2×2 · 公式定义', intuition: '2×2 行列式等于主对角线积减副对角线积（ad−bc）。',
    math: 'det([[a,b],[c,d]]) = ad − bc', pseudocode: 'return det2(A)', complexity: 'O(1)',
    ref: 'Horn & Johnson 1985', commonPitfall: '副对角线项符号为负。', invariant: '逐位 Object.is（整数域）',
    inputs: ['A'], domains: { A: 'matInt' }, matrixDims: { A: [2, 2] },
    run: (A) => det2(A),
    predicate: (A) => meq(runOf('C097')(A), A[0] * A[3] - A[1] * A[2])
      && Object.is(runOf('C097')(id(2)), 1),   // 定点锚：det(I₂)=1
  });
  reg(98, {
    name: 'det · 转置不变', intuition: '行列式对转置不变：det(Aᵀ) = det(A)（逐位）。',
    math: 'det(Aᵀ) = det(A)', pseudocode: 'return det2(Aᵀ)', complexity: 'O(1)',
    ref: 'Horn & Johnson 1985', commonPitfall: '无。', invariant: '逐位 Object.is',
    inputs: ['A'], domains: { A: 'matInt' }, matrixDims: { A: [2, 2] },
    run: (A) => det2(trans(A, 2, 2)),
    predicate: (A) => meq(runOf('C098')(A), det2(A)) && Object.is(runOf('C098')(id(2)), 1),
  });
  reg(99, {
    name: 'det · 单位阵等于 1', intuition: '单位阵行列式为 1（矩阵乘的单位元）。',
    math: 'det(I₂) = 1', pseudocode: 'return det2(I)', complexity: 'O(1)',
    ref: '线性代数', commonPitfall: '无。', invariant: 'Object.is(det(I₂), 1)',
    inputs: ['A'], domains: { A: 'matInt' }, matrixDims: { A: [2, 2] },
    run: () => det2(id(2)), predicate: () => Object.is(runOf('C099')(), 1),
  });
  reg(100, {
    name: 'det · 对角阵等于对角线积', intuition: '对角阵行列式等于对角线元素之积（3×3，从输入提取对角块验证）。',
    math: 'det(diag(d₀,d₁,d₂)) = d₀d₁d₂', pseudocode: 'return det3(diagPart(A))', complexity: 'O(1)',
    ref: 'Horn & Johnson 1985', commonPitfall: '非对角项必须精确为 0。', invariant: '逐位 Object.is（整数域）',
    inputs: ['A'], domains: { A: 'matInt' }, matrixDims: { A: [3, 3] },
    run: (A) => det3(diagPart(A, 3)),
    predicate: (A) => meq(runOf('C100')(A), A[0] * A[4] * A[8])   // 内联对角线积（不经 diagPart/det3）
      && Object.is(runOf('C100')(id(3)), 1),                     // 定点锚：det(I₃)=1
  });
  reg(101, {
    name: 'det · 乘积性（2×2 精确）', intuition: 'det(AB) = det(A)·det(B)；整数域两侧皆为精确整数（≤8e12 < 2⁵³），逐位相等。',
    math: 'det(AB) = det(A)det(B)', pseudocode: 'return det2(A·B)', complexity: 'O(1)',
    ref: 'Horn & Johnson 1985', commonPitfall: '更大规模会溢出精确整数区，故限 2×2。', invariant: '逐位 Object.is',
    inputs: ['A', 'B'], domains: { A: 'matInt', B: 'matInt' }, matrixDims: { A: [2, 2], B: [2, 2] },
    run: (A, B) => det2(M.matmul(A, B, 2, 2, 2)),
    predicate: (A, B) => meq(runOf('C101')(A, B), det2(A) * det2(B))
      && Object.is(runOf('C101')(id(2), id(2)), 1),   // 定点锚：det(I·I)=1
  });
  reg(102, {
    name: 'det · 乘积性（3×3 对角因子）', intuition: 'B 取对角阵 diag(2,3,5) 时 det(AB) = det(A)·30，即列缩放定理；3×3 全程整数精确（≤4e11 < 2⁵³）。',
    math: 'det(A·D) = det(A)·det(D), D = diag(2,3,5)', pseudocode: 'return det3(A·D)', complexity: 'O(1)',
    ref: 'Horn & Johnson 1985', commonPitfall: '一般 3×3 双矩阵乘积有大数抵消，故选固定对角因子保精确。', invariant: '逐位 Object.is',
    inputs: ['A'], domains: { A: 'matInt' }, matrixDims: { A: [3, 3] },
    run: (A) => det3(M.matmul(A, DIAG235, 3, 3, 3)),
    predicate: (A) => meq(runOf('C102')(A), det3(A) * 30)
      && Object.is(runOf('C102')(id(3)), 30),   // 定点锚：det(I·D)=30
  });
  reg(103, {
    name: '2×2 · 迹等于特征值之和', intuition: '特征方程 λ²−tr·λ+det=0 的两根之和等于迹（Vieta）；实根还验证积等于 det 且根满足方程，复共轭根验证多项式在根处消失。',
    math: 'λ₁+λ₂ = tr(A), λ₁λ₂ = det(A)', pseudocode: 'return 根之和', complexity: 'O(1)',
    ref: 'Cayley–Hamilton / Vieta', commonPitfall: '判别式为负时根为共轭复数，须按复根验证（实部和为迹）。', invariant: 'relClose(根和, tr, 1e-9) 且各根满足特征方程',
    inputs: ['A'], domains: { A: 'matSmall' }, matrixDims: { A: [2, 2] },
    run: (A) => {
      const t = A[0] + A[3], d = A[0] * A[3] - A[1] * A[2], disc = t * t - 4 * d;
      if (disc >= 0) { const s = Math.sqrt(disc); return (t + s) / 2 + (t - s) / 2; }
      return t; // 共轭复根 a±ih 之和（实部和）为 t
    },
    predicate: (A) => {
      const t = A[0] + A[3], d = A[0] * A[3] - A[1] * A[2], disc = t * t - 4 * d;
      if (!rc(runOf('C103')(A), t, 1e-9)) return false;          // run 根和 vs 内联迹
      if (disc >= 0) {
        const s = Math.sqrt(disc), l1 = (t + s) / 2, l2 = (t - s) / 2;
        return rc(l1 * l2, d, 1e-9) &&
          rc(l1 * l1 - t * l1 + d, 0, 1e-9) && rc(l2 * l2 - t * l2 + d, 0, 1e-9);
      }
      const h = Math.sqrt(-disc) / 2, a = t / 2; // 根为 a ± i·h
      return rc(a * a - h * h - t * a + d, 0, 1e-9) &&   // Re p(a+ih) = 0
        rc(2 * a * h - t * h, 0, 1e-9);                 // Im p(a+ih) = 0
    },
  });
  reg(104, {
    name: '对称判定 · 与定义一致', intuition: 'isSym(A) 当且仅当 A = Aᵀ；且对称化 A+Aᵀ 必为对称阵（正向构造恒真）。',
    math: 'isSym(A) ⟺ A = Aᵀ', pseudocode: 'return isSym(A)', complexity: 'O(n²)',
    ref: 'Horn & Johnson 1985', commonPitfall: '±0 须视为相等（IEEE 中 Object.is(−0,0) 为假）。', invariant: 'isSym(A) === (A=Aᵀ) 且 isSym(A+Aᵀ) 恒真',
    inputs: ['A'], domains: { A: 'matInt' }, matrixDims: { A: [4, 4] },
    run: (A) => isSym(A, 4),
    predicate: (A) => {
      if (runOf('C104')(A) !== meq(A, trans(A, 4, 4))) return false;   // run 判定 vs 异构转置比对
      return runOf('C104')(id(4)) === true && runOf('C104')(Z(4, 4)) === true
        && isSym(madd(A, trans(A, 4, 4), 16), 4) === true;             // 正向构造恒真
    },
  });
  reg(105, {
    name: '反对称判定 · 与定义一致', intuition: 'isAnti(A) 当且仅当 A = −Aᵀ（对角线必为 0）；且反对称化 A−Aᵀ 必为反对称阵。',
    math: 'isAnti(A) ⟺ A = −Aᵀ', pseudocode: 'return isAnti(A)', complexity: 'O(n²)',
    ref: 'Horn & Johnson 1985', commonPitfall: '对角线元素必须为 0（x = −x ⇒ x = 0）。', invariant: 'isAnti(A) === (A=−Aᵀ)、isAnti(A) ⇒ 对角线为 0、isAnti(A−Aᵀ) 恒真',
    inputs: ['A'], domains: { A: 'matInt' }, matrixDims: { A: [4, 4] },
    run: (A) => isAnti(A, 4),
    predicate: (A) => {
      const anti = runOf('C105')(A);
      if (anti !== meq(A, mneg(trans(A, 4, 4), 16))) return false;
      if (anti) { for (let i = 0; i < 4; i++) if (!z0(A[i * 4 + i])) return false; }
      return runOf('C105')(Z(4, 4)) === true && runOf('C105')(id(4)) === false
        && isAnti(msub(A, trans(A, 4, 4), 16), 4) === true;
    },
  });
  reg(106, {
    name: '三角阵判定 · 上与下', intuition: 'isUpper(A) 当且仅当 A 等于其上三角投影，isLower 同理；投影结果必为三角阵（正向构造恒真）。',
    math: 'isUpper(A) ⟺ A = triu(A)；isLower(A) ⟺ A = tril(A)', pseudocode: 'return {up, lo}', complexity: 'O(n²)',
    ref: '线性代数', commonPitfall: '判定与投影互为充要，整数零用严格比对而非容差。', invariant: '双向等价 + triu/tril 投影恒为三角阵',
    inputs: ['A'], domains: { A: 'matInt' }, matrixDims: { A: [4, 4] },
    run: (A) => ({ up: isUpperTri(A, 4), lo: isLowerTri(A, 4) }),
    predicate: (A) => {
      const r = runOf('C106')(A);
      const U = triuPart(A, 4), L = trilPart(A, 4);
      if (r.up !== meq(A, U) || r.lo !== meq(A, L)) return false;   // run 判定 vs 异构投影比对
      if (!isUpperTri(U, 4) || !isLowerTri(L, 4)) return false;     // 正向构造：投影恒为三角阵
      const ri = runOf('C106')(id(4));                               // 定点锚：I 上下皆三角
      const strict = Z(4, 4); strict[4] = 1;                         // (1,0) 非零 → 下三角而非上三角
      const rl = runOf('C106')(strict);
      return ri.up === true && ri.lo === true && rl.up === false && rl.lo === true;
    },
  });
  reg(107, {
    name: 'det · 三角阵等于对角线积', intuition: '上三角阵（3×3）行列式等于对角线之积；一般矩阵先取上三角投影再验证。',
    math: 'det(triu(A)) = ∏ᵢ Uᵢᵢ', pseudocode: 'return det3(triu(A))', complexity: 'O(1)',
    ref: 'Horn & Johnson 1985', commonPitfall: '仅对三角阵成立，一般矩阵须先投影。', invariant: '逐位 Object.is（整数域）',
    inputs: ['A'], domains: { A: 'matInt' }, matrixDims: { A: [3, 3] },
    run: (A) => det3(triuPart(A, 3)),
    predicate: (A) => meq(runOf('C107')(A), A[0] * A[4] * A[8])   // 内联对角线积（投影保留对角）
      && Object.is(runOf('C107')(id(3)), 1),                     // 定点锚：det(triu(I))=1
  });
  reg(108, {
    name: 'det · 数乘齐次性', intuition: 'n 阶矩阵数乘 s 后行列式乘 sⁿ；2×2 时 det(2A) = 4·det(A)（整数域逐位精确）。',
    math: 'det(sA) = sⁿ·det(A)', pseudocode: 'return det2(2A)', complexity: 'O(1)',
    ref: 'Horn & Johnson 1985', commonPitfall: '指数是阶数 n 而非 1。', invariant: '逐位 Object.is',
    inputs: ['A'], domains: { A: 'matInt' }, matrixDims: { A: [2, 2] },
    run: (A) => det2(mscale(2, A, 4)),
    predicate: (A) => meq(runOf('C108')(A), 4 * det2(A))
      && Object.is(runOf('C108')(id(2)), 4),   // 定点锚：det(2I₂)=4
  });

  // ── 归约进阶（C109–C120）──────────────────────────────────────────────
  reg(109, {
    name: '列均值 · 等于列和除以行数', intuition: '每列均值 = 该列元素和 / 行数；r=4 时除以 4 为幂二，整数域逐位精确。',
    math: 'μⱼ = (Σᵢ Aᵢⱼ) / r', pseudocode: 'return colSum(A)/r', complexity: 'O(rc)',
    ref: '统计量', commonPitfall: '除以行数而非列数。', invariant: '逐位 Object.is',
    inputs: ['A'], domains: { A: 'matInt' }, matrixDims: { A: [4, 5] },
    run: (A) => mscale(0.25, perColSum(A, 4, 5), 5),
    predicate: (A) => {
      const m = runOf('C109')(A);
      for (let j = 0; j < 5; j++) { let acc = 0; for (let i = 0; i < 4; i++) acc += A[i * 5 + j]; if (!meq(m[j], acc * 0.25)) return false; }
      return meq(runOf('C109')(ones(4, 5)), ones(1, 5));   // 定点锚：ones 列均值=1
    },
  });
  reg(110, {
    name: '列 softmax · 每列和为 1', intuition: '对每列做 softmax（复用 L8 softmaxRows 作用于转置），各列概率和趋近 1。',
    math: 'Σᵢ colSoftmax(A)ᵢⱼ = 1（每列）', pseudocode: 'return colSoftmax(A)', complexity: 'O(rc)',
    ref: 'Bridle 1990', commonPitfall: '需容差；沿列而非行归一；极端输入下 exp 可下溢为 0（合法，概率和仍为 1）。', invariant: 'relClose(列和, 1, 1e-9)',
    inputs: ['A'], domains: { A: 'matFinite' }, matrixDims: { A: [4, 5] },
    run: (A) => colSoftmax(A, 4, 5),
    predicate: (A) => {
      const P = runOf('C110')(A);
      for (let j = 0; j < 5; j++) {
        let s = 0;
        for (let i = 0; i < 4; i++) { if (!(P[i * 5 + j] >= 0 && P[i * 5 + j] <= 1 + 1e-12)) return false; s += P[i * 5 + j]; }
        if (!rc(s, 1, 1e-9)) return false;
      }
      return true;
    },
  });
  reg(111, {
    name: '列 softmax · 输出非负', intuition: 'softmax 输出为概率，列 softmax 的每个元素恒非负。',
    math: 'colSoftmax(A)ᵢⱼ ≥ 0', pseudocode: 'return colSoftmax(A)', complexity: 'O(rc)',
    ref: 'Bridle 1990', commonPitfall: '无。', invariant: '∀i,j: Pᵢⱼ ≥ 0',
    inputs: ['A'], domains: { A: 'matFinite' }, matrixDims: { A: [4, 5] },
    run: (A) => colSoftmax(A, 4, 5), predicate: (A) => allIn(runOf('C111')(A), (x) => x >= 0 && x <= 1 + 1e-12),
  });
  reg(112, {
    name: '行 argmax · 取值等于行最大', intuition: '每行 argmax 下标处的取值等于该行最大值（并列取首处），且下标合法。',
    math: 'A[i, argmaxᵢ] = maxⱼ Aᵢⱼ', pseudocode: 'return argmaxRow(A)', complexity: 'O(rc)',
    ref: '归约', commonPitfall: '并列时取第一个最大位置。', invariant: '取值=行最大 且为首个最大位置',
    inputs: ['A'], domains: { A: 'matInt' }, matrixDims: { A: [4, 5] },
    run: (A) => argmaxRow(A, 4, 5),
    predicate: (A) => {
      const idx = runOf('C112')(A);
      for (let i = 0; i < 4; i++) {
        const o = i * 5;
        let mx = A[o];
        for (let j = 1; j < 5; j++) if (A[o + j] > mx) mx = A[o + j];
        if (idx[i] < 0 || idx[i] >= 5) return false;
        if (!Object.is(A[o + idx[i]], mx)) return false;
        for (let j = 0; j < idx[i]; j++) if (!(A[o + j] < mx)) return false;
      }
      return true;
    },
  });
  reg(113, {
    name: '列 argmax · 取值等于列最大', intuition: '每列 argmax 行下标处的取值等于该列最大值（并列取首处），且下标合法。',
    math: 'A[argmaxⱼ, j] = maxᵢ Aᵢⱼ', pseudocode: 'return argmaxCol(A)', complexity: 'O(rc)',
    ref: '归约', commonPitfall: '并列时取第一个最大位置。', invariant: '取值=列最大 且为首个最大位置',
    inputs: ['A'], domains: { A: 'matInt' }, matrixDims: { A: [4, 5] },
    run: (A) => argmaxCol(A, 4, 5),
    predicate: (A) => {
      const idx = runOf('C113')(A);
      for (let j = 0; j < 5; j++) {
        let mx = A[j];
        for (let i = 1; i < 4; i++) if (A[i * 5 + j] > mx) mx = A[i * 5 + j];
        if (idx[j] < 0 || idx[j] >= 4) return false;
        if (!Object.is(A[idx[j] * 5 + j], mx)) return false;
        for (let i = 0; i < idx[j]; i++) if (!(A[i * 5 + j] < mx)) return false;
      }
      return true;
    },
  });
  reg(114, {
    name: '行 LSE · 上下界', intuition: '每行 log-sum-exp 满足 max ≤ LSE ≤ max + ln c（折叠 L9 logsumexp2，数值稳定）。',
    math: 'maxⱼAᵢⱼ ≤ log Σⱼe^{Aᵢⱼ} ≤ maxⱼAᵢⱼ + ln c', pseudocode: 'return rowLSE(A)', complexity: 'O(rc)',
    ref: '数值稳定 / logsumexp', commonPitfall: '直接 exp 会溢出，须先减行最大再折叠。', invariant: 'max ≤ LSE ≤ max + ln c（±1e-9 松弛）',
    inputs: ['A'], domains: { A: 'matFinite' }, matrixDims: { A: [4, 5] },
    run: (A) => rowLSE(A, 4, 5),
    predicate: (A) => {
      const lse = runOf('C114')(A);
      for (let i = 0; i < 4; i++) {
        const o = i * 5;
        let mx = A[o];
        for (let j = 1; j < 5; j++) if (A[o + j] > mx) mx = A[o + j];
        if (!(lse[i] >= mx - 1e-9 && lse[i] <= mx + Math.log(5) + 1e-9)) return false;
      }
      const S7 = mscale(7, ones(4, 5), 20);                 // 定点锚：全 7 行 LSE = 7+ln5
      return rc(runOf('C114')(S7)[0], 7 + Math.log(5), 1e-12);
    },
  });
  reg(115, {
    name: '列 min/max · 逐列正确', intuition: '逐列最小值与最大值等于独立扫描结果（整数域逐位精确）。',
    math: 'colMinⱼ = minᵢ Aᵢⱼ；colMaxⱼ = maxᵢ Aᵢⱼ', pseudocode: 'return colMinMax(A)', complexity: 'O(rc)',
    ref: '归约', commonPitfall: '无。', invariant: '逐位 Object.is',
    inputs: ['A'], domains: { A: 'matInt' }, matrixDims: { A: [4, 5] },
    run: (A) => colMinMax(A, 4, 5),
    predicate: (A) => {
      const mm = runOf('C115')(A);
      for (let j = 0; j < 5; j++) {
        let lo = A[j], hi = A[j];
        for (let i = 1; i < 4; i++) { const v = A[i * 5 + j]; if (v < lo) lo = v; if (v > hi) hi = v; }
        if (!Object.is(mm.min[j], lo) || !Object.is(mm.max[j], hi)) return false;
      }
      return true;
    },
  });
  reg(116, {
    name: '全元素和 · 等于各列和之和', intuition: '矩阵全部元素之和等于各列和向量之和（整数域精确，与累加顺序无关）。',
    math: 'ΣᵢⱼAᵢⱼ = Σⱼ(ΣᵢAᵢⱼ)', pseudocode: 'return sumAll(colSums(A))', complexity: 'O(rc)',
    ref: '归约', commonPitfall: '无。', invariant: '逐位 Object.is',
    inputs: ['A'], domains: { A: 'matInt' }, matrixDims: { A: [4, 5] },
    run: (A) => sumAll(perColSum(A, 4, 5)),
    predicate: (A) => {
      let t = 0;
      for (let i = 0; i < 20; i++) t += A[i];               // 内联直合计（异构归约路径）
      return Object.is(runOf('C116')(A), t);
    },
  });
  reg(117, {
    name: '行 cumsum · 末列等于行和', intuition: '按行前缀和到最后一列即该行元素和（整数域逐位精确）。',
    math: 'cumsum(A)ᵢ,c₋₁ = Σⱼ Aᵢⱼ', pseudocode: 'return cumsumRows(A)', complexity: 'O(rc)',
    ref: '归约', commonPitfall: '前缀和沿行内列方向累加。', invariant: '逐位 Object.is',
    inputs: ['A'], domains: { A: 'matInt' }, matrixDims: { A: [4, 5] },
    run: (A) => cumsumRows(A, 4, 5),
    predicate: (A) => {
      const cs = runOf('C117')(A);
      for (let i = 0; i < 4; i++) {
        let s = 0;
        for (let j = 0; j < 5; j++) { s += A[i * 5 + j]; if (!meq(cs[i * 5 + j], s)) return false; }   // 全前缀逐位内联 oracle
      }
      return meq(runOf('C117')(Z(4, 5)), Z(4, 5));   // 定点锚：零阵前缀和恒零
    },
  });
  reg(118, {
    name: '列方差 · 非负', intuition: '每列总体方差为偏差平方的均值，恒非负（r=4 幂二除法，整数域精确）。',
    math: 'Varⱼ = (Σᵢ(Aᵢⱼ−μⱼ)²)/r ≥ 0', pseudocode: 'return colVar(A)', complexity: 'O(rc)',
    ref: '统计量', commonPitfall: '无。', invariant: '∀j: Varⱼ ≥ 0',
    inputs: ['A'], domains: { A: 'matInt' }, matrixDims: { A: [4, 5] },
    run: (A) => colVar(A, 4, 5),
    predicate: (A) => allIn(runOf('C118')(A), (x) => x >= 0)
      && allIn(runOf('C118')(THREES), (x) => x === 0),   // 定点锚：常数列方差=0
  });
  reg(119, {
    name: '列方差 · 等于转置的行方差', intuition: 'A 的列方差等于 Aᵀ 的行方差（r=4 时两种路径累加顺序一致，逐位相等）。',
    math: 'Var_col(A)ⱼ = Var_row(Aᵀ)ⱼ', pseudocode: 'return colVar(A)', complexity: 'O(rc)',
    ref: '统计量', commonPitfall: '总体方差（除以个数）而非样本方差（除以 n−1）。', invariant: '逐位 Object.is',
    inputs: ['A'], domains: { A: 'matInt' }, matrixDims: { A: [4, 5] },
    run: (A) => colVar(A, 4, 5),
    predicate: (A) => meq(runOf('C119')(A), rowVar(trans(A, 4, 5), 5, 4))   // 异构方差路径（colVar vs rowVar）
      && meq(runOf('C119')(THREES), Z(1, 5)),                               // 定点锚：常数列方差=0
  });
  reg(120, {
    name: '列方差 · 平移不变', intuition: '每列加同一常数不改变方差（整数域下偏移在均值中精确抵消，逐位相等）。',
    math: 'Var(A + c·J) = Var(A)', pseudocode: 'return colVar(A+3J)', complexity: 'O(rc)',
    ref: '统计量', commonPitfall: '常数须加到每一个元素（全矩阵同常量）。', invariant: '逐位 Object.is',
    inputs: ['A'], domains: { A: 'matInt' }, matrixDims: { A: [4, 5] },
    run: (A) => colVar(madd(A, THREES, 20), 4, 5),
    predicate: (A) => meq(runOf('C120')(A), inlineColVar(A)),   // run(A+3J) vs 内联 Var(A)
  });

  // ── 张量操作（C121–C128）──────────────────────────────────────────────
  reg(121, {
    name: '垂直拼接 · 形状守恒', intuition: '两块同行宽矩阵垂直拼接，行数相加、元素数 = (r₁+r₂)·c。',
    math: 'vcat(A(r₁×c), B(r₂×c)) ∈ ℝ^{(r₁+r₂)×c}', pseudocode: 'return vcat(A,B)', complexity: 'O((r₁+r₂)c)',
    ref: 'NumPy vstack', commonPitfall: '两块列宽须相等。', invariant: '逐位：run(A,B)[i] === A[i]（i<6）、run(A,B)[6+i] === B[i]（钉死拼接内容，非仅长度）',
    inputs: ['A', 'B'], domains: { A: 'matInt', B: 'matInt' }, matrixDims: { A: [2, 3], B: [2, 3] },
    run: (A, B) => vcat(A, B, 2, 3),
    predicate: (A, B) => {
      const r = runOf('C121')(A, B);
      if (r.length !== 12) return false;
      for (let i = 0; i < A.length; i++) if (!Object.is(r[i], A[i])) return false;
      for (let i = 0; i < B.length; i++) if (!Object.is(r[A.length + i], B[i])) return false;
      return true;
    },
  });
  reg(122, {
    name: '垂直拼接 · 内容定位', intuition: '拼接结果上块逐位等于 A、下块逐位等于 B。',
    math: 'vcat(A,B)[0:r₁·c] = A；[r₁·c:] = B', pseudocode: 'return vcat(A,B)', complexity: 'O((r₁+r₂)c)',
    ref: 'NumPy vstack', commonPitfall: '行主序下块偏移为 r₁·c。', invariant: '逐位 Object.is',
    inputs: ['A', 'B'], domains: { A: 'matInt', B: 'matInt' }, matrixDims: { A: [2, 3], B: [2, 3] },
    run: (A, B) => vcat(A, B, 2, 3),
    predicate: (A, B) => {
      const o = runOf('C122')(A, B);
      return meq(o.subarray(0, 6), A) && meq(o.subarray(6, 12), B);
    },
  });
  reg(123, {
    name: '水平拼接 · 形状与内容', intuition: '两块同高矩阵水平拼接，列数相加；左块为 A、右块为 B（逐位）。',
    math: 'hcat(A(r×c₁), B(r×c₂))ᵢⱼ = Aᵢⱼ (j<c₁)；= Bᵢ,ⱼ₋c₁ (j≥c₁)', pseudocode: 'return hcat(A,B)', complexity: 'O(r(c₁+c₂))',
    ref: 'NumPy hstack', commonPitfall: '行数须相等；块内偏移按行计算。', invariant: 'length = r·(c₁+c₂) 且逐位 Object.is',
    inputs: ['A', 'B'], domains: { A: 'matInt', B: 'matInt' }, matrixDims: { A: [2, 3], B: [2, 4] },
    run: (A, B) => hcat(A, B, 2, 3, 4),
    predicate: (A, B) => {
      const o = runOf('C123')(A, B);
      if (o.length !== 14) return false;
      for (let i = 0; i < 2; i++) for (let j = 0; j < 3; j++) if (!meq(o[i * 7 + j], A[i * 3 + j])) return false;
      for (let i = 0; i < 2; i++) for (let j = 0; j < 4; j++) if (!meq(o[i * 7 + 3 + j], B[i * 4 + j])) return false;
      return true;
    },
  });
  reg(124, {
    name: '行切片 · 区间形状与内容', intuition: '取行区间 [1,3) 得 2×5 子矩阵（复用 L7 slice+contiguous），内容逐位等于原矩阵对应两行。',
    math: 'A[1:3, :] ∈ ℝ^{2×5}', pseudocode: 'return rowSlice(A,1,3)', complexity: 'O((e−s)c)',
    ref: 'L7.tensor.slice / NumPy', commonPitfall: '区间左闭右开；L7 后于本文件加载故调用期惰性引用。', invariant: 'length=10 且逐位 Object.is',
    inputs: ['A'], domains: { A: 'matInt' }, matrixDims: { A: [4, 5] },
    run: (A) => rowSlice(A, 4, 5, 1, 3),
    predicate: (A) => {
      const o = runOf('C124')(A);
      if (o.length !== 10) return false;
      for (let i = 0; i < 2; i++) for (let j = 0; j < 5; j++) if (!meq(o[i * 5 + j], A[(1 + i) * 5 + j])) return false;
      return true;
    },
  });
  reg(125, {
    name: '矩形双转置 · 还原与元素定位', intuition: '2×6 矩形矩阵双转置还原；并全位置验证元素定位律 (Aᵀ)ᵢⱼ = Aⱼᵢ。',
    math: '(Aᵀ)ᵀ = A；(Aᵀ)ᵢⱼ = Aⱼᵢ', pseudocode: 'return Aᵀᵀ', complexity: 'O(rc)',
    ref: '线性代数', commonPitfall: '矩形转置交换维度（2×6 ↔ 6×2）。', invariant: '双转置逐位 Object.is + 元素定位逐位',
    inputs: ['A'], domains: { A: 'matInt' }, matrixDims: { A: [2, 6] },
    run: (A) => trans(trans(A, 2, 6), 6, 2),
    predicate: (A) => {
      if (!meq(runOf('C125')(A), A)) return false;
      const T = trans(A, 2, 6);
      for (let i = 0; i < 6; i++) for (let j = 0; j < 2; j++) if (!meq(T[i * 2 + j], A[j * 6 + i])) return false;
      return meq(runOf('C125')(ones(2, 6)), ones(2, 6));   // 定点锚：全1双转置守恒
    },
  });
  reg(126, {
    name: '对角提取 · 方阵对角线', intuition: '提取 4×4 方阵对角线得 4 个元素，逐位等于 Aᵢᵢ。',
    math: 'diag(A)ᵢ = Aᵢᵢ', pseudocode: 'return diagOf(A)', complexity: 'O(n)',
    ref: '线性代数', commonPitfall: '仅方阵有对角线；结果长度为阶数 n。', invariant: 'length=n 且逐位 Object.is',
    inputs: ['A'], domains: { A: 'matInt' }, matrixDims: { A: [4, 4] },
    run: (A) => diagOf(A, 4),
    predicate: (A) => {
      const d = runOf('C126')(A);
      if (d.length !== 4) return false;
      for (let i = 0; i < 4; i++) if (!meq(d[i], A[i * 4 + i])) return false;
      return meq(runOf('C126')(id(4)), Float64Array.from([1, 1, 1, 1]));   // 定点锚：diag(I)=(1,1,1,1)
    },
  });
  reg(127, {
    name: '矩阵 pad · 补零行列', intuition: '2×3 矩阵补零到 4×5：原内容居左上，新增行列精确为 0。',
    math: 'pad(A)ᵢⱼ = Aᵢⱼ (i<r, j<c)；否则 0', pseudocode: 'return padTo(A,4,5)', complexity: 'O(RC)',
    ref: 'NumPy pad', commonPitfall: '补零区域必须精确 0（非未初始化内存）。', invariant: 'length=20、原内容逐位、新区域为 0',
    inputs: ['A'], domains: { A: 'matInt' }, matrixDims: { A: [2, 3] },
    run: (A) => padTo(A, 2, 3, 4, 5),
    predicate: (A) => {
      const o = runOf('C127')(A);
      if (o.length !== 20) return false;
      for (let i = 0; i < 4; i++) for (let j = 0; j < 5; j++) {
        const want = (i < 2 && j < 3) ? A[i * 3 + j] : 0;
        if (!meq(o[i * 5 + j], want)) return false;
      }
      return true;
    },
  });
  reg(128, {
    name: '拼接复合 · 形状公式', intuition: '先垂直拼再接水平拼接：最终形状 (r₁+r₂)×(c₁+c₂)，四块内容各归其位。',
    math: 'hcat(vcat(A,B), vcat(C,D)) ∈ ℝ^{(r₁+r₂)×(c₁+c₂)}', pseudocode: 'return hcat(vcat(A,B), vcat(C,D))', complexity: 'O((r₁+r₂)(c₁+c₂))',
    ref: 'NumPy vstack/hstack', commonPitfall: '两对的行数须分别相等；块定位按象限。', invariant: 'length=28 且四块逐位 Object.is',
    inputs: ['A', 'B', 'C', 'D'], domains: { A: 'matInt', B: 'matInt', C: 'matInt', D: 'matInt' },
    matrixDims: { A: [2, 3], B: [2, 3], C: [2, 4], D: [2, 4] },
    run: (A, B, C, D) => hcat(vcat(A, B, 2, 3), vcat(C, D, 2, 4), 4, 3, 4),
    predicate: (A, B, C, D) => {
      const o = runOf('C128')(A, B, C, D);
      if (o.length !== 28) return false;
      for (let i = 0; i < 2; i++) for (let j = 0; j < 3; j++) if (!meq(o[i * 7 + j], A[i * 3 + j])) return false;
      for (let i = 0; i < 2; i++) for (let j = 0; j < 4; j++) if (!meq(o[i * 7 + 3 + j], C[i * 4 + j])) return false;
      for (let i = 0; i < 2; i++) for (let j = 0; j < 3; j++) if (!meq(o[(2 + i) * 7 + j], B[i * 3 + j])) return false;
      for (let i = 0; i < 2; i++) for (let j = 0; j < 4; j++) if (!meq(o[(2 + i) * 7 + 3 + j], D[i * 4 + j])) return false;
      return true;
    },
  });
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

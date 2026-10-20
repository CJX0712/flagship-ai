/* ============================================================================
 * L10 微指令 · C 域（矩阵/张量运算）· 第 3/3 批（C065–C096）
 * ----------------------------------------------------------------------------
 * 涵盖：矩阵乘法/恒等式（C065–C084）、逐元素函数（C085–C096）。
 * 复用 NS.L10.mh 与 NS.L8.matmul。乘法恒等式以朴素三重循环独立校验，确保
 * 新内核与数学定义逐位一致（R9 同类安全网）。
 * ==========================================================================*/
;(function (NS) {
  'use strict';
  NS.L10 = NS.L10 || {};
  const M = NS.L8.matmul, SP = NS.L9.special, L10 = NS.L10;
  const mh = L10.mh;
  function reg(i, o) { NS.L10.register('C', i, o); }
  const { Z, madd, msub, mscale, mneg, mabs, mrelu, mclamp, emp, msq, trans, id, ones, allIn, sumAll, fro, colSum, meq, rc } = mh;
  function rowSum(A, r, c) { return NS.L8.reduce.rowSum(A, r, c); }

  function naive(A, B, r, k, c) {
    const o = new Float64Array(r * c);
    for (let i = 0; i < r; i++) for (let p = 0; p < k; p++) { const a = A[i * k + p]; for (let j = 0; j < c; j++) o[i * c + j] += a * B[p * c + j]; }
    return o;
  }
  function naiveTA(A, B, r, k, c) {
    const o = new Float64Array(k * c);
    for (let i = 0; i < r; i++) for (let l = 0; l < k; l++) { const a = A[i * k + l]; for (let j = 0; j < c; j++) o[l * c + j] += a * B[i * c + j]; }
    return o;
  }
  function naiveAtB(A, B, r, k, m) {
    const o = new Float64Array(r * m);
    for (let i = 0; i < r; i++) for (let j = 0; j < m; j++) { let s = 0; for (let p = 0; p < k; p++) s += A[i * k + p] * B[j * k + p]; o[i * m + j] = s; }
    return o;
  }

  // ── 矩阵乘法 / 恒等式（C065–C084）─────────────────────────────────────
  reg(65, {
    name: '矩阵乘法 · 输出维度', intuition: 'A(r×k)·B(k×c) 输出为 r×c（元素数校验）。',
    math: 'A∈ℝ^{r×k}, B∈ℝ^{k×c} ⇒ C∈ℝ^{r×c}', pseudocode: 'return A·B', complexity: 'O(rkc)',
    ref: '线性代数', commonPitfall: '无', invariant: 'C.length = r·c',
    inputs: ['A', 'B'], domains: { A: 'matInt', B: 'matInt' }, matrixDims: { A: [4,3], B: [3,5] },
    run: (A, B) => M.matmul(A, B, 4, 3, 5), predicate: (A, B) => M.matmul(A, B, 4, 3, 5).length === 20,
  });
  reg(66, {
    name: '矩阵·向量 · 等于线性组合', intuition: 'A(r×k)·x(k×1) 第 i 元素 = Σₖ Aᵢₖ xₖ（逐位）。',
    math: '(A x)ᵢ = Σₖ Aᵢₖ xₖ', pseudocode: 'return A·x', complexity: 'O(rk)',
    ref: '线性代数', commonPitfall: '无', invariant: '逐位 Object.is',
    inputs: ['A', 'x'], domains: { A: 'matInt', x: 'matInt' }, matrixDims: { A: [4,3], x: [3,1] },
    run: (A, x) => M.matmul(A, x, 4, 3, 1), predicate: (A, x) => { const y = M.matmul(A, x, 4, 3, 1); for (let i=0;i<4;i++){ let s=0; for (let k=0;k<3;k++) s+=A[i*3+k]*x[k]; if (!Object.is(y[i], s)) return false; } return true; },
  });
  reg(67, {
    name: '矩阵·向量 · 对加法线性', intuition: 'A(x+y) = Ax + Ay（逐位）。',
    math: 'A(x+y) = Ax + Ay', pseudocode: 'return A(x+y)', complexity: 'O(rk)',
    ref: '线性代数', commonPitfall: '无', invariant: '逐位 Object.is',
    inputs: ['A', 'x', 'y'], domains: { A: 'matInt', x: 'matInt', y: 'matInt' }, matrixDims: { A: [4,3], x: [3,1], y: [3,1] },
    run: (A, x, y) => M.matmul(A, madd(x, y, 3), 4, 3, 1), predicate: (A, x, y) => meq(M.matmul(A, madd(x, y, 3), 4, 3, 1), madd(M.matmul(A, x, 4, 3, 1), M.matmul(A, y, 4, 3, 1), 4)),
  });
  reg(68, {
    name: '矩阵·向量 · 齐次性', intuition: 'A(s x) = s (Ax)（逐位）。',
    math: 'A(sx) = s(Ax)', pseudocode: 'return A(sx)', complexity: 'O(rk)',
    ref: '线性代数', commonPitfall: '无', invariant: '逐位 Object.is',
    inputs: ['A', 'x'], domains: { A: 'matInt', x: 'matInt' }, matrixDims: { A: [4,3], x: [3,1] },
    run: (A, x) => M.matmul(A, mscale(2, x, 3), 4, 3, 1), predicate: (A, x) => meq(M.matmul(A, mscale(2, x, 3), 4, 3, 1), mscale(2, M.matmul(A, x, 4, 3, 1), 4)),
  });
  reg(69, {
    name: '向量·矩阵 · 等于线性组合', intuition: 'u(1×r)·B(r×c) 第 j 元素 = Σᵣ uᵣ Bᵣⱼ（逐位）。',
    math: '(u B)ⱼ = Σᵣ uᵣ Bᵣⱼ', pseudocode: 'return u·B', complexity: 'O(rc)',
    ref: '线性代数', commonPitfall: '无', invariant: '逐位 Object.is',
    inputs: ['u', 'B'], domains: { u: 'matInt', B: 'matInt' }, matrixDims: { u: [1,4], B: [4,5] },
    run: (u, B) => M.matmul(u, B, 1, 4, 5), predicate: (u, B) => { const y = M.matmul(u, B, 1, 4, 5); for (let j=0;j<5;j++){ let s=0; for (let r=0;r<4;r++) s+=u[r]*B[r*5+j]; if (!Object.is(y[j], s)) return false; } return true; },
  });
  reg(70, {
    name: '外积 · u⊗v = u·vᵀ', intuition: 'u(r×1)·v(1×c) 第 ij 元素 = uᵢ vⱼ（逐位）。',
    math: '(u⊗v)ᵢⱼ = uᵢ vⱼ', pseudocode: 'return u·vᵀ', complexity: 'O(rc)',
    ref: '张量积', commonPitfall: '无', invariant: '逐位 Object.is',
    inputs: ['u', 'v'], domains: { u: 'matInt', v: 'matInt' }, matrixDims: { u: [4,1], v: [1,5] },
    run: (u, v) => M.matmul(u, v, 4, 1, 5), predicate: (u, v) => { const O = M.matmul(u, v, 4, 1, 5); for (let i=0;i<4;i++) for (let j=0;j<5;j++) if (!Object.is(O[i*5+j], u[i]*v[j])) return false; return true; },
  });
  reg(71, {
    name: '外积 · (u⊗v)·w = ⟨v,w⟩·u', intuition: '秩 1 矩阵乘向量满足结合恒等式（逐位）。',
    math: '(u vᵀ) w = (vᵀ w) u', pseudocode: 'return (u⊗v)·w', complexity: 'O(rc)',
    ref: '张量积', commonPitfall: '无', invariant: '逐位 Object.is',
    inputs: ['u', 'v', 'w'], domains: { u: 'matInt', v: 'matInt', w: 'matInt' }, matrixDims: { u: [4,1], v: [1,4], w: [4,1] },
    run: (u, v, w) => M.matmul(M.matmul(u, v, 4, 1, 4), w, 4, 4, 1),
    predicate: (u, v, w) => { const y = M.matmul(M.matmul(u, v, 4, 1, 4), w, 4, 4, 1); let dot=0; for (let i=0;i<4;i++) dot+=v[i]*w[i]; const exp = mscale(dot, u, 4); return meq(y, exp); },
  });
  reg(72, {
    name: '矩阵·向量 · 单位阵作用不变', intuition: 'I·x = x（逐位）。',
    math: 'I x = x', pseudocode: 'return I·x', complexity: 'O(r)',
    ref: '线性代数', commonPitfall: '无', invariant: '逐位 Object.is',
    inputs: ['x'], domains: { x: 'matInt' }, matrixDims: { x: [4,1] },
    run: (x) => M.matmul(id(4), x, 4, 4, 1), predicate: (x) => meq(M.matmul(id(4), x, 4, 4, 1), x),
  });
  reg(73, {
    name: '矩阵·向量 · 零矩阵作用为零', intuition: '0·x = 0（逐位）。',
    math: '0 x = 0', pseudocode: 'return 0·x', complexity: 'O(rk)',
    ref: '线性代数', commonPitfall: '无', invariant: '所有元素 = 0',
    inputs: ['x'], domains: { x: 'matInt' }, matrixDims: { x: [3,1] },
    run: (x) => M.matmul(Z(4,3), x, 4, 3, 1), predicate: (x) => allIn(M.matmul(Z(4,3), x, 4, 3, 1), (v) => v === 0),
  });
  reg(74, {
    name: '矩阵乘法 · Frobenius 次乘性', intuition: '‖AB‖_F ≤ ‖A‖_F·‖B‖_F（允许浮点误差）。',
    math: '‖AB‖_F ≤ ‖A‖_F‖B‖_F', pseudocode: 'return ‖A·B‖_F', complexity: 'O(n³)',
    ref: 'Horn & Johnson 1985', commonPitfall: '需容差；为 ≤ 界。', invariant: 'relClose(‖AB‖_F, ‖A‖_F‖B‖_F, 1e-6) 且 ≤',
    inputs: ['A', 'B'], domains: { A: 'matInt', B: 'matInt' }, matrixDims: { A: [4,4], B: [4,4] },
    run: (A, B) => M.matmul(A, B, 4, 4, 4),
    predicate: (A, B) => { const P = M.matmul(A, B, 4, 4, 4); const lhs = fro(P); const rhs = fro(A) * fro(B); return lhs <= rhs * (1 + 1e-9) + 1e-9; },
  });
  reg(75, {
    name: '矩阵乘法 · Frobenius 范数定义一致', intuition: '‖AB‖_F = √(Σᵢⱼ (AB)ᵢⱼ²)（允许浮点误差）。',
    math: '‖AB‖_F = √Σ(AB)²', pseudocode: 'return fro(A·B)', complexity: 'O(rkc)',
    ref: 'Frobenius 范数', commonPitfall: '需容差。', invariant: 'relClose(‖AB‖_F, √Σ(AB)², 1e-9)',
    inputs: ['A', 'B'], domains: { A: 'matInt', B: 'matInt' }, matrixDims: { A: [4,3], B: [3,5] },
    run: (A, B) => M.matmul(A, B, 4, 3, 5), predicate: (A, B) => { const P = M.matmul(A, B, 4, 3, 5); let s=0; for (let i=0;i<P.length;i++) s+=P[i]*P[i]; return rc(fro(P), Math.sqrt(s), 1e-9); },
  });
  reg(76, {
    name: '矩阵迹 · 加法线性', intuition: 'tr(A+B) = tr(A) + tr(B)（逐位）。',
    math: 'tr(A+B) = tr(A)+tr(B)', pseudocode: 'return tr(A+B)', complexity: 'O(min(r,c))',
    ref: '线性代数', commonPitfall: '无', invariant: '逐位 Object.is',
    inputs: ['A', 'B'], domains: { A: 'matInt', B: 'matInt' }, matrixDims: { A: [4,4], B: [4,4] },
    run: (A, B) => { let s=0; for (let i=0;i<4;i++) s+=A[i*4+i]+B[i*4+i]; return s; },
    predicate: (A, B) => { let l=0, ra=0, rb=0; for (let i=0;i<4;i++){ l+=A[i*4+i]+B[i*4+i]; ra+=A[i*4+i]; rb+=B[i*4+i]; } return Object.is(l, ra+rb); },
  });
  reg(77, {
    name: '矩阵迹 · 齐次缩放', intuition: 'tr(sA) = s·tr(A)（逐位）。',
    math: 'tr(sA) = s·tr(A)', pseudocode: 'return tr(sA)', complexity: 'O(min(r,c))',
    ref: '线性代数', commonPitfall: '无', invariant: '逐位 Object.is',
    inputs: ['A'], domains: { A: 'matInt' }, matrixDims: { A: [4,4] },
    run: (A) => { let s=0; for (let i=0;i<4;i++) s+=2*A[i*4+i]; return s; },
    predicate: (A) => { let l=0, r=0; for (let i=0;i<4;i++){ l+=2*A[i*4+i]; r+=A[i*4+i]; } return Object.is(l, 2*r); },
  });
  reg(78, {
    name: '矩阵迹 · 转置不变', intuition: 'tr(Aᵀ) = tr(A)（逐位）。',
    math: 'tr(Aᵀ) = tr(A)', pseudocode: 'return tr(Aᵀ)', complexity: 'O(min(r,c))',
    ref: '线性代数', commonPitfall: '无', invariant: '逐位 Object.is',
    inputs: ['A'], domains: { A: 'matInt' }, matrixDims: { A: [4,4] },
    run: (A) => { let s=0; for (let i=0;i<4;i++) s+=trans(A,4,4)[i*4+i]; return s; },
    predicate: (A) => { let l=0, r=0; for (let i=0;i<4;i++){ l+=trans(A,4,4)[i*4+i]; r+=A[i*4+i]; } return Object.is(l, r); },
  });
  reg(79, {
    name: '矩阵乘法 · 全 1 矩阵复制行和', intuition: 'A·J（J 全 1）的每一列都等于 A 的行和（逐位）。',
    math: '(A·J)ᵢⱼ = Σₖ Aᵢₖ', pseudocode: 'return A·J', complexity: 'O(rkc)',
    ref: '线性代数', commonPitfall: '无', invariant: '逐位 Object.is',
    inputs: ['A'], domains: { A: 'matInt' }, matrixDims: { A: [4,5] },
    run: (A) => M.matmul(A, ones(5, 5), 4, 5, 5), predicate: (A) => { const P = M.matmul(A, ones(5, 5), 4, 5, 5); for (let i=0;i<4;i++){ let rs=0; for (let k=0;k<5;k++) rs+=A[i*5+k]; for (let j=0;j<5;j++) if (!Object.is(P[i*5+j], rs)) return false; } return true; },
  });
  reg(80, {
    name: '矩阵乘法 · 一般输出维度', intuition: 'A(2×3)·B(3×4) 输出元素数为 8。',
    math: 'dim(A·B) = 2×4', pseudocode: 'return A·B', complexity: 'O(rkc)',
    ref: '线性代数', commonPitfall: '无', invariant: 'C.length = 8',
    inputs: ['A', 'B'], domains: { A: 'matInt', B: 'matInt' }, matrixDims: { A: [2,3], B: [3,4] },
    run: (A, B) => M.matmul(A, B, 2, 3, 4), predicate: (A, B) => M.matmul(A, B, 2, 3, 4).length === 8,
  });
  reg(81, {
    name: '秩 1 更新 · A + u⊗v', intuition: '(A + u vᵀ)ᵢⱼ = Aᵢⱼ + uᵢ vⱼ（逐位）。',
    math: '(A+u⊗v)ᵢⱼ = Aᵢⱼ + uᵢ vⱼ', pseudocode: 'return A+u·vᵀ', complexity: 'O(rc)',
    ref: 'Sherman–Morrison 基础', commonPitfall: '无', invariant: '逐位 Object.is',
    inputs: ['A', 'u', 'v'], domains: { A: 'matInt', u: 'matInt', v: 'matInt' }, matrixDims: { A: [4,4], u: [4,1], v: [1,4] },
    run: (A, u, v) => madd(A, M.matmul(u, v, 4, 1, 4), 16), predicate: (A, u, v) => { const O = madd(A, M.matmul(u, v, 4, 1, 4), 16); for (let i=0;i<4;i++) for (let j=0;j<4;j++) if (!Object.is(O[i*4+j], A[i*4+j] + u[i]*v[j])) return false; return true; },
  });
  reg(82, {
    name: '矩阵乘法 · 等于数学定义（朴素三重循环）', intuition: '新内核 matmul 与朴素定义逐位一致（R9 同源安全网）。',
    math: 'Cᵢⱼ = Σₚ Aᵢₚ Bₚⱼ', pseudocode: 'return matmul(A,B)', complexity: 'O(rkc)',
    ref: 'Goto & van de Geijn 2008', commonPitfall: '浮点求和顺序须一致。', invariant: '逐位 Object.is（含 a=0 跳过）',
    inputs: ['A', 'B'], domains: { A: 'matInt', B: 'matInt' }, matrixDims: { A: [4,3], B: [3,5] },
    run: (A, B) => M.matmul(A, B, 4, 3, 5), predicate: (A, B) => meq(M.matmul(A, B, 4, 3, 5), naive(A, B, 4, 3, 5)),
  });
  reg(83, {
    name: '矩阵乘法 · matmulTA = Aᵀ·B（定义一致）', intuition: 'matmulTA 与 Aᵀ·B 的朴素定义逐位一致。',
    math: 'matmulTA = Aᵀ·B', pseudocode: 'return matmulTA(A,B)', complexity: 'O(rkc)',
    ref: 'Goto & van de Geijn 2008', commonPitfall: '无', invariant: '逐位 Object.is',
    inputs: ['A', 'B'], domains: { A: 'matInt', B: 'matInt' }, matrixDims: { A: [3,4], B: [4,5] },
    run: (A, B) => M.matmulTA(A, B, 4, 3, 5), predicate: (A, B) => meq(M.matmulTA(A, B, 4, 3, 5), naiveTA(A, B, 4, 3, 5)),
  });
  reg(84, {
    name: '矩阵乘法 · matmulAtB = A·Bᵀ（定义一致）', intuition: 'matmulAtB 与 A·Bᵀ 的朴素定义逐位一致。',
    math: 'matmulAtB = A·Bᵀ', pseudocode: 'return matmulAtB(A,B)', complexity: 'O(rkm)',
    ref: 'Goto & van de Geijn 2008', commonPitfall: '无', invariant: '逐位 Object.is',
    inputs: ['A', 'B'], domains: { A: 'matInt', B: 'matInt' }, matrixDims: { A: [4,3], B: [2,3] },
    run: (A, B) => M.matmulAtB(A, B, 4, 3, 2), predicate: (A, B) => meq(M.matmulAtB(A, B, 4, 3, 2), naiveAtB(A, B, 4, 3, 2)),
  });

  // ── 逐元素函数（C085–C096）────────────────────────────────────────────
  reg(85, {
    name: '逐元素三角 · sin²+cos²=1', intuition: '矩阵逐元素 sin²+cos² = 1（允许浮点误差）。',
    math: 'sin²A + cos²A = 1', pseudocode: 'return sin²A+cos²A', complexity: 'O(rc)',
    ref: 'Euler 公式', commonPitfall: '需容差。', invariant: 'relClose(Σ, 1, 1e-9)',
    inputs: ['A'], domains: { A: 'matFinite' }, matrixDims: { A: [4,4] },
    run: (A) => { const o = Float64Array.from(A); for (let i=0;i<o.length;i++) o[i] = Math.sin(A[i])**2 + Math.cos(A[i])**2; return o; },
    predicate: (A) => { const o = Float64Array.from(A); for (let i=0;i<o.length;i++) o[i] = Math.sin(A[i])**2 + Math.cos(A[i])**2; return allIn(o, (x) => rc(x, 1, 1e-9)); },
  });
  reg(86, {
    name: '逐元素绝对值 · 非负', intuition: '|A| 的每个元素恒非负。',
    math: '|Aᵢ| ≥ 0', pseudocode: 'return |A|', complexity: 'O(rc)',
    ref: 'IEEE 754', commonPitfall: '无', invariant: '∀i: |Aᵢ| ≥ 0',
    inputs: ['A'], domains: { A: 'matFinite' }, matrixDims: { A: [4,4] },
    run: (A) => mabs(A, 16), predicate: (A) => allIn(mabs(A, 16), (x) => x >= 0),
  });
  reg(87, {
    name: '逐元素 Sigmoid · 区间 (0,1)', intuition: 'σ(A) 的每个元素严格落在 (0,1)。',
    math: '0 < σ(Aᵢ) < 1', pseudocode: 'return σ(A)', complexity: 'O(rc)',
    ref: 'Goodfellow 2016', commonPitfall: '无', invariant: '∀i: 0 < σ(Aᵢ) < 1',
    inputs: ['A'], domains: { A: 'matSmall' }, matrixDims: { A: [4,4] },
    run: (A) => { const o = Float64Array.from(A); for (let i=0;i<o.length;i++) o[i] = SP.sigmoid(A[i]); return o; },
    predicate: (A) => { const o = Float64Array.from(A); for (let i=0;i<o.length;i++) { const s = SP.sigmoid(A[i]); if (!(s > 0 && s < 1)) return false; } return true; },
  });
  reg(88, {
    name: '逐元素 Sigmoid · 对称', intuition: 'σ(−A) = 1 − σ(A)（允许浮点误差）。',
    math: 'σ(−A) = 1 − σ(A)', pseudocode: 'return σ(−A)', complexity: 'O(rc)',
    ref: 'Goodfellow 2016', commonPitfall: '需容差。', invariant: 'relClose(σ(−A), 1−σ(A), 1e-9)',
    inputs: ['A'], domains: { A: 'matFinite' }, matrixDims: { A: [4,4] },
    run: (A) => { const o = Float64Array.from(A); for (let i=0;i<o.length;i++) o[i] = SP.sigmoid(-A[i]); return o; },
    predicate: (A) => { const o = Float64Array.from(A); for (let i=0;i<o.length;i++) if (!rc(SP.sigmoid(-A[i]), 1 - SP.sigmoid(A[i]), 1e-9)) return false; return true; },
  });
  reg(89, {
    name: '逐元素 tanh · 有界', intuition: '|tanh(A)| < 1（有限输入）。',
    math: '|tanh(Aᵢ)| < 1', pseudocode: 'return tanh(A)', complexity: 'O(rc)',
    ref: '双曲函数', commonPitfall: '无', invariant: '∀i: |tanh(Aᵢ)| < 1',
    inputs: ['A'], domains: { A: 'matSmall' }, matrixDims: { A: [4,4] },
    run: (A) => { const o = Float64Array.from(A); for (let i=0;i<o.length;i++) o[i] = Math.tanh(A[i]); return o; },
    predicate: (A) => { const o = Float64Array.from(A); for (let i=0;i<o.length;i++) if (!(Math.abs(Math.tanh(A[i])) < 1)) return false; return true; },
  });
  reg(90, {
    name: '逐元素指数 · 正值', intuition: 'exp(A) 的每个元素恒为正。',
    math: 'exp(Aᵢ) > 0', pseudocode: 'return exp(A)', complexity: 'O(rc)',
    ref: '初等函数', commonPitfall: '无', invariant: '∀i: exp(Aᵢ) > 0',
    inputs: ['A'], domains: { A: 'matSmall' }, matrixDims: { A: [4,4] },
    run: (A) => { const o = Float64Array.from(A); for (let i=0;i<o.length;i++) o[i] = Math.exp(A[i]); return o; },
    predicate: (A) => { const o = Float64Array.from(A); for (let i=0;i<o.length;i++) if (!(Math.exp(A[i]) > 0)) return false; return true; },
  });
  reg(91, {
    name: '逐元素 · exp 后 log 还原', intuition: 'log(exp(A)) 回到 A（允许浮点误差）。',
    math: 'log(exp A) ≈ A', pseudocode: 'return log(exp A)', complexity: 'O(rc)',
    ref: '初等函数', commonPitfall: '需容差；A 不宜过大。', invariant: 'relClose(log(exp A), A, 1e-9)',
    inputs: ['A'], domains: { A: 'matSmall' }, matrixDims: { A: [4,4] },
    run: (A) => { const o = Float64Array.from(A); for (let i=0;i<o.length;i++) o[i] = Math.log(Math.exp(A[i])); return o; },
    predicate: (A) => { const o = Float64Array.from(A); for (let i=0;i<o.length;i++) if (!rc(Math.log(Math.exp(A[i])), A[i], 1e-9)) return false; return true; },
  });
  reg(92, {
    name: '逐元素绝对值 · 取反对称', intuition: '|−A| = |A|（逐位）。',
    math: '|−A| = |A|', pseudocode: 'return |−A|', complexity: 'O(rc)',
    ref: 'IEEE 754', commonPitfall: '无', invariant: '逐位 Object.is',
    inputs: ['A'], domains: { A: 'matFinite' }, matrixDims: { A: [4,4] },
    run: (A) => mabs(mneg(A, 16), 16), predicate: (A) => meq(mabs(mneg(A, 16), 16), mabs(A, 16)),
  });
  reg(93, {
    name: '逐元素 · √(a²) = |a|', intuition: '√(A²) = |A|（允许浮点误差）。',
    math: '√(A²) = |A|', pseudocode: 'return √(A²)', complexity: 'O(rc)',
    ref: '平方根定义', commonPitfall: '需容差。', invariant: 'relClose(√(A²), |A|, 1e-9)',
    inputs: ['A'], domains: { A: 'matFinite' }, matrixDims: { A: [4,4] },
    run: (A) => { const o = Float64Array.from(A); for (let i=0;i<o.length;i++) o[i] = Math.sqrt(A[i]*A[i]); return o; },
    predicate: (A) => { const o = Float64Array.from(A); for (let i=0;i<o.length;i++) if (!rc(Math.sqrt(A[i]*A[i]), Math.abs(A[i]), 1e-9)) return false; return true; },
  });
  reg(94, {
    name: '逐元素平方 · 符号恒非负', intuition: 'A² 的每个元素恒非负。',
    math: 'A²ᵢ ≥ 0', pseudocode: 'return A²', complexity: 'O(rc)',
    ref: '代数', commonPitfall: '无', invariant: '∀i: A²ᵢ ≥ 0',
    inputs: ['A'], domains: { A: 'matFinite' }, matrixDims: { A: [4,4] },
    run: (A) => msq(A, 16), predicate: (A) => allIn(msq(A, 16), (x) => x >= 0),
  });
  reg(95, {
    name: '逐元素取整 · 整数不动', intuition: '整数矩阵经 round 后逐位不变。',
    math: 'round(A) = A (A∈ℤ)', pseudocode: 'return round(A)', complexity: 'O(rc)',
    ref: 'IEEE 754', commonPitfall: '无', invariant: '逐位 Object.is',
    inputs: ['A'], domains: { A: 'matInt' }, matrixDims: { A: [4,4] },
    run: (A) => { const o = Float64Array.from(A); for (let i=0;i<o.length;i++) o[i] = Math.round(A[i]); return o; },
    predicate: (A) => { const o = Float64Array.from(A); for (let i=0;i<o.length;i++) if (!Object.is(Math.round(A[i]), A[i])) return false; return true; },
  });
  reg(96, {
    name: '逐元素 ReLU · 作用于绝对值幂等', intuition: 'abs(A)≥0，故 ReLU(|A|) = |A|（逐位）。',
    math: 'ReLU(|A|) = |A|', pseudocode: 'return ReLU(|A|)', complexity: 'O(rc)',
    ref: 'Nair & Hinton 2010', commonPitfall: '无', invariant: '逐位 Object.is',
    inputs: ['A'], domains: { A: 'matFinite' }, matrixDims: { A: [4,4] },
    run: (A) => mrelu(mabs(A, 16), 16), predicate: (A) => meq(mrelu(mabs(A, 16), 16), mabs(A, 16)),
  });
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

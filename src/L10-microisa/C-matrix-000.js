/* ============================================================================
 * L10 微指令 · C 域（矩阵/张量运算）· 第 1/3 批（C001–C032）
 * ----------------------------------------------------------------------------
 * 所属层：L10-microIsa  域：C（矩阵/张量）
 * 允许依赖的层：L8（matmul/reduce）、L9（special）、L10（registry/invariant）
 * 主要 @ref：Strassen (1969)；Goto & van de Geijn (2008)；Horn & Johnson (1985)
 *
 * 每项含七件套 + run + predicate。矩阵以 Float64Array 行优先（K1）。
 * 输入维度的兼容性由 op.matrixDims 精确声明，tests/L10 据此构造矩阵。
 * 整数域（matInt）下多数等式逐位精确；浮点域用 relClose(tol=1e-9)。
 *
 * ── 谓词纪律（本批全部重写，2026-10-23）─────────────────────────────────
 * 门禁只调 op.predicate，从不调 op.run（op-harness.js:43、probe-microisa.js:106）。
 * 原 32 项谓词全部用本地 helper（madd/msub/trans/matmul/…）内联复算 run
 * 定义式，run 若损坏门禁完全失明（假绿）。现全部改为：先经 runOf('Cxxx')
 * 调 run 本体，再对 run 的产出断言不变量；oracle 取异构分组/输入导出值/
 * 常数锚；幂等与对合类经 run 双重调用；交换律经 run 双向调用互比。
 * ==========================================================================*/
;(function (NS) {
  'use strict';
  NS.L10 = NS.L10 || {};
  const M = NS.L8.matmul, R = NS.L8.reduce, L10 = NS.L10;
  function reg(i, o) { NS.L10.register('C', i, o); }
  function runOf(key) { const o = NS.L10.ops[key]; return o && typeof o.run === 'function' ? o.run : null; }
  const rc = (a, b, t) => L10.relClose(a, b, t);
  const meq = (a, b) => L10.relClose(a, b, 0, true);

  // ── 矩阵工具（本批定义并挂到 NS.L10.mh，供 C 域其余批次复用）────────────
  function Z(r, c) { return new Float64Array(r * c); }
  function madd(A, B, n) { const o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = A[i] + B[i]; return o; }
  function msub(A, B, n) { const o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = A[i] - B[i]; return o; }
  function mscale(s, A, n) { const o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = s * A[i]; return o; }
  function mneg(A, n) { const o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = -A[i]; return o; }
  function mabs(A, n) { const o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = Math.abs(A[i]); return o; }
  function mrelu(A, n) { const o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = A[i] > 0 ? A[i] : 0; return o; }
  function mclamp(A, n, lo, hi) { const o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = A[i] < lo ? lo : A[i] > hi ? hi : A[i]; return o; }
  function emp(A, B, n) { const o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = A[i] * B[i]; return o; }
  function msq(A, n) { const o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = A[i] * A[i]; return o; }
  function trans(A, r, c) { const o = new Float64Array(r * c); for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) o[j * r + i] = A[i * c + j]; return o; }
  function id(r) { const o = new Float64Array(r * r); for (let i = 0; i < r; i++) o[i * r + i] = 1; return o; }
  function ones(r, c) { const o = new Float64Array(r * c); o.fill(1); return o; }
  function allIn(A, p) { for (let i = 0; i < A.length; i++) if (!p(A[i])) return false; return true; }
  function sumAll(A) { let s = 0; for (let i = 0; i < A.length; i++) s += A[i]; return s; }
  function fro(A) { let s = 0; for (let i = 0; i < A.length; i++) s += A[i] * A[i]; return Math.sqrt(s); }
  function colSum(A, r, c) { return R.rowSum(trans(A, r, c), c, r); }
  L10.mh = { Z, madd, msub, mscale, mneg, mabs, mrelu, mclamp, emp, msq, trans, id, ones, allIn, sumAll, fro, colSum, meq, rc };

  // ── 矩阵算术（C001–C020）──────────────────────────────────────────────
  reg(1, {
    name: '矩阵加法 · 交换', intuition: '矩阵加法逐位交换。',
    math: 'A + B = B + A', pseudocode: 'return A + B', complexity: 'O(rc)',
    ref: '线性代数', commonPitfall: '维度须一致。', invariant: '逐位 Object.is',
    inputs: ['A', 'B'], domains: { A: 'matInt', B: 'matInt' }, matrixDims: { A: [4, 4], B: [4, 4] },
    run: (A, B) => madd(A, B, 16),
    predicate: (A, B) => meq(runOf('C001')(A, B), runOf('C001')(B, A)),
  });
  reg(2, {
    name: '矩阵加法 · 结合', intuition: '整数域下矩阵加法精确可结合。',
    math: '(A+B)+C = A+(B+C)', pseudocode: 'return (A+B)+C', complexity: 'O(rc)',
    ref: '线性代数', commonPitfall: '浮点域下一般不可结合，故用整数。', invariant: '逐位 Object.is',
    inputs: ['A', 'B', 'C'], domains: { A: 'matInt', B: 'matInt', C: 'matInt' }, matrixDims: { A: [4, 4], B: [4, 4], C: [4, 4] },
    run: (A, B, C) => madd(madd(A, B, 16), C, 16),
    predicate: (A, B, C) => {
      const run = runOf('C002');
      if (!meq(run(A, B, C), madd(A, madd(B, C, 16), 16))) return false;
      return meq(run(id(4), id(4), id(4)), mscale(3, id(4), 16));   // 定点锚：I+I+I=3I
    },
  });
  reg(3, {
    name: '矩阵加法 · 零元', intuition: '零矩阵是加法的单位元。',
    math: 'A + 0 = A', pseudocode: 'return A + 0', complexity: 'O(rc)',
    ref: '线性代数', commonPitfall: '无', invariant: '逐位 Object.is',
    inputs: ['A'], domains: { A: 'matInt' }, matrixDims: { A: [4, 4] },
    run: (A) => madd(A, Z(4, 4), 16), predicate: (A) => meq(runOf('C003')(A), A),
  });
  reg(4, {
    name: '矩阵减法 · 自减为零', intuition: '矩阵减自身为精确零矩阵。',
    math: 'A - A = 0', pseudocode: 'return A - A', complexity: 'O(rc)',
    ref: '线性代数', commonPitfall: '无', invariant: '逐位 Object.is',
    inputs: ['A'], domains: { A: 'matInt' }, matrixDims: { A: [4, 4] },
    run: (A) => msub(A, A, 16), predicate: (A) => meq(runOf('C004')(A), Z(4, 4)),
  });
  reg(5, {
    name: '矩阵取反 · 对合', intuition: '两次取反回到原矩阵。',
    math: '-(-A) = A', pseudocode: 'return -A', complexity: 'O(rc)',
    ref: '线性代数', commonPitfall: '无', invariant: '逐位 Object.is',
    inputs: ['A'], domains: { A: 'matInt' }, matrixDims: { A: [4, 4] },
    run: (A) => mneg(A, 16),
    predicate: (A) => {
      const run = runOf('C005');
      return meq(run(run(A)), A) && meq(run(Z(4, 4)), Z(4, 4));   // run 双重调用 + 零锚
    },
  });
  reg(6, {
    name: '矩阵缩放 · 单位元', intuition: '标量 1 缩放不改变矩阵。',
    math: '1·A = A', pseudocode: 'return s*A', complexity: 'O(rc)',
    ref: '线性代数', commonPitfall: '无', invariant: '逐位 Object.is',
    inputs: ['A'], domains: { A: 'matInt' }, matrixDims: { A: [4, 4] },
    run: (A) => mscale(1, A, 16), predicate: (A) => meq(runOf('C006')(A), A),
  });
  reg(7, {
    name: '矩阵乘法 · 乘以零矩阵', intuition: '任意矩阵乘零矩阵为精确零矩阵。',
    math: 'A·0 = 0', pseudocode: 'return A·0', complexity: 'O(rkc)',
    ref: '线性代数', commonPitfall: '无', invariant: '所有元素 = 0',
    inputs: ['A'], domains: { A: 'matInt' }, matrixDims: { A: [4, 3] },
    run: (A) => M.matmul(A, Z(3, 5), 4, 3, 5),
    predicate: (A) => allIn(runOf('C007')(A), (x) => x === 0),
  });
  reg(8, {
    name: '矩阵乘法 · 左乘单位阵', intuition: 'A·I = A（逐位）。',
    math: 'A·I = A', pseudocode: 'return A·I', complexity: 'O(rc)',
    ref: '线性代数', commonPitfall: '无', invariant: '逐位 Object.is',
    inputs: ['A'], domains: { A: 'matInt' }, matrixDims: { A: [4, 5] },
    run: (A) => M.matmul(A, id(5), 4, 5, 5), predicate: (A) => meq(runOf('C008')(A), A),
  });
  reg(9, {
    name: '矩阵乘法 · 右乘单位阵', intuition: 'I·A = A（逐位）。',
    math: 'I·A = A', pseudocode: 'return I·A', complexity: 'O(rc)',
    ref: '线性代数', commonPitfall: '无', invariant: '逐位 Object.is',
    inputs: ['A'], domains: { A: 'matInt' }, matrixDims: { A: [4, 5] },
    run: (A) => M.matmul(id(4), A, 4, 4, 5), predicate: (A) => meq(runOf('C009')(A), A),
  });
  reg(10, {
    name: '矩阵乘法 · 结合', intuition: '整数域下矩阵乘法精确可结合。',
    math: '(A·B)·C = A·(B·C)', pseudocode: 'return (A·B)·C', complexity: 'O(n³)',
    ref: '线性代数', commonPitfall: '浮点域下舍入破坏等式，故用整数。', invariant: '逐位 Object.is',
    inputs: ['A', 'B', 'C'], domains: { A: 'matInt', B: 'matInt', C: 'matInt' }, matrixDims: { A: [2, 3], B: [3, 4], C: [4, 5] },
    run: (A, B, C) => M.matmul(M.matmul(A, B, 2, 3, 4), C, 2, 4, 5),
    predicate: (A, B, C) => {
      const run = runOf('C010');
      if (!meq(run(A, B, C), M.matmul(A, M.matmul(B, C, 3, 4, 5), 2, 3, 5))) return false;
      return meq(run(ones(2, 3), ones(3, 4), ones(4, 5)), mscale(12, ones(2, 5), 10));   // 定点锚：全1链乘每项=12
    },
  });
  reg(11, {
    name: '矩阵乘法 · 对加法分配', intuition: 'A·(B+C) = A·B + A·C（整数域精确）。',
    math: 'A·(B+C) = A·B + A·C', pseudocode: 'return A·(B+C)', complexity: 'O(n³)',
    ref: '线性代数', commonPitfall: '浮点舍入，故用整数。', invariant: '逐位 Object.is',
    inputs: ['A', 'B', 'C'], domains: { A: 'matInt', B: 'matInt', C: 'matInt' }, matrixDims: { A: [2, 3], B: [3, 4], C: [3, 4] },
    run: (A, B, C) => M.matmul(A, madd(B, C, 12), 2, 3, 4),
    predicate: (A, B, C) => {
      const run = runOf('C011');
      if (!meq(run(A, B, C), madd(M.matmul(A, B, 2, 3, 4), M.matmul(A, C, 2, 3, 4), 8))) return false;
      return meq(run(ones(2, 3), ones(3, 4), Z(3, 4)), mscale(3, ones(2, 4), 8));   // 定点锚：ones·(ones+0) 每项=3
    },
  });
  reg(12, {
    name: '矩阵缩放 · 关联标量乘法', intuition: 's·(t·A) = (s·t)·A（整数域精确）。',
    math: 's(tA) = (st)A', pseudocode: 'return s*(t*A)', complexity: 'O(rc)',
    ref: '线性代数', commonPitfall: '无', invariant: '逐位 Object.is',
    inputs: ['A'], domains: { A: 'matInt' }, matrixDims: { A: [4, 4] },
    run: (A) => mscale(2, mscale(3, A, 16), 16),
    predicate: (A) => meq(runOf('C012')(A), mscale(6, A, 16)) && meq(runOf('C012')(Z(4, 4)), Z(4, 4)),
  });
  reg(13, {
    name: '矩阵缩放 · 对加法分配', intuition: 's·(A+B) = sA + sB（整数域精确）。',
    math: 's(A+B) = sA + sB', pseudocode: 'return s*(A+B)', complexity: 'O(rc)',
    ref: '线性代数', commonPitfall: '无', invariant: '逐位 Object.is',
    inputs: ['A', 'B'], domains: { A: 'matInt', B: 'matInt' }, matrixDims: { A: [4, 4], B: [4, 4] },
    run: (A, B) => mscale(2, madd(A, B, 16), 16),
    predicate: (A, B) => {
      const run = runOf('C013');
      return meq(run(A, B), madd(mscale(2, A, 16), mscale(2, B, 16), 16))
        && meq(run(Z(4, 4), Z(4, 4)), Z(4, 4));
    },
  });
  reg(14, {
    name: '矩阵转置 · 双转置还原', intuition: '转置的转置回到原矩阵（逐位）。',
    math: '(Aᵀ)ᵀ = A', pseudocode: 'return Aᵀᵀ', complexity: 'O(rc)',
    ref: '线性代数', commonPitfall: 'run 固定 4×5→5×4，双重调用会维度错位，故用内联定义 + 边际交叉验证。', invariant: '逐位 Object.is',
    inputs: ['A'], domains: { A: 'matInt' }, matrixDims: { A: [4, 5] },
    run: (A) => trans(A, 4, 5),
    predicate: (A) => {
      const T = runOf('C014')(A);                       // Aᵀ（5×4）
      if (T.length !== 20) return false;
      for (let i = 0; i < 4; i++) for (let j = 0; j < 5; j++) if (!meq(T[j * 4 + i], A[i * 5 + j])) return false;
      const rsA = new Float64Array(4);                  // A 行和（内联索引路径）
      for (let i = 0; i < 4; i++) { let s = 0; for (let j = 0; j < 5; j++) s += A[i * 5 + j]; rsA[i] = s; }
      return meq(R.rowSum(T, 5, 4), rsA);               // Aᵀ 列和（L8 归约）== A 行和
    },
  });
  reg(15, {
    name: '矩阵转置 · 形状互换', intuition: '转置后元素个数与原矩阵相同。',
    math: '|Aᵀ| = |A|', pseudocode: 'return Aᵀ', complexity: 'O(rc)',
    ref: '线性代数', commonPitfall: '无', invariant: '逐位：run(A)[j·4+i] === A[i·5+j]（钉死转置内容，非仅长度）',
    inputs: ['A'], domains: { A: 'matInt' }, matrixDims: { A: [4, 5] },
    run: (A) => trans(A, 4, 5),
    predicate: (A) => {
      const T = runOf('C015')(A);
      if (T.length !== 20) return false;
      for (let i = 0; i < 4; i++) for (let j = 0; j < 5; j++) if (!Object.is(T[j * 4 + i], A[i * 5 + j])) return false;
      return true;
    },
  });
  reg(16, {
    name: '逐元素绝对值 · 幂等', intuition: '矩阵逐元素绝对值对自身幂等。',
    math: '| |A| | = |A|', pseudocode: 'return |A|', complexity: 'O(rc)',
    ref: 'IEEE 754', commonPitfall: '无', invariant: '逐位 Object.is',
    inputs: ['A'], domains: { A: 'matFinite' }, matrixDims: { A: [4, 4] },
    run: (A) => mabs(A, 16),
    predicate: (A) => {
      const run = runOf('C016');
      return meq(run(run(A)), run(A)) && meq(run(Z(4, 4)), Z(4, 4));   // run 双重调用 + 零锚
    },
  });
  reg(17, {
    name: '逐元素钳位 · 有界', intuition: '钳位到 [−2,2] 后所有元素落界内。',
    math: '−2 ≤ clamp(A) ≤ 2', pseudocode: 'return clamp(A)', complexity: 'O(rc)',
    ref: '数值稳定', commonPitfall: '无', invariant: '∀i: −2 ≤ Aᵢ ≤ 2',
    inputs: ['A'], domains: { A: 'matFinite' }, matrixDims: { A: [4, 4] },
    run: (A) => mclamp(A, 16, -2, 2),
    predicate: (A) => allIn(runOf('C017')(A), (x) => x >= -2 && x <= 2),
  });
  reg(18, {
    name: '逐元素 ReLU · 非负', intuition: '矩阵逐元素 ReLU 输出恒非负。',
    math: 'ReLU(A) ≥ 0', pseudocode: 'return ReLU(A)', complexity: 'O(rc)',
    ref: 'Nair & Hinton 2010', commonPitfall: '无', invariant: '∀i: ReLU(Aᵢ) ≥ 0',
    inputs: ['A'], domains: { A: 'matFinite' }, matrixDims: { A: [4, 4] },
    run: (A) => mrelu(A, 16), predicate: (A) => allIn(runOf('C018')(A), (x) => x >= 0),
  });
  reg(19, {
    name: '矩阵取反 · 与标量 −1 缩放等价', intuition: '逐元素取反等于乘以 −1（逐位）。',
    math: '-A = (-1)·A', pseudocode: 'return -A', complexity: 'O(rc)',
    ref: '线性代数', commonPitfall: '无', invariant: '逐位 Object.is',
    inputs: ['A'], domains: { A: 'matInt' }, matrixDims: { A: [4, 4] },
    run: (A) => mneg(A, 16),
    predicate: (A) => meq(runOf('C019')(A), mscale(-1, A, 16)) && meq(runOf('C019')(Z(4, 4)), Z(4, 4)),
  });
  reg(20, {
    name: '矩阵加法后转置 · 等于转置后相加', intuition: '(A+B)ᵀ = Aᵀ + Bᵀ（逐位）。',
    math: '(A+B)ᵀ = Aᵀ + Bᵀ', pseudocode: 'return (A+B)ᵀ', complexity: 'O(rc)',
    ref: '线性代数', commonPitfall: '无', invariant: '逐位 Object.is',
    inputs: ['A', 'B'], domains: { A: 'matInt', B: 'matInt' }, matrixDims: { A: [4, 4], B: [4, 4] },
    run: (A, B) => trans(madd(A, B, 16), 4, 4),
    predicate: (A, B) => meq(runOf('C020')(A, B), madd(trans(A, 4, 4), trans(B, 4, 4), 16)),
  });

  // ── 转置 / 对称（C021–C032）──────────────────────────────────────────
  reg(21, {
    name: '矩阵缩放后转置 · 等于转置后缩放', intuition: '(sA)ᵀ = s(Aᵀ)（逐位）。',
    math: '(sA)ᵀ = s(Aᵀ)', pseudocode: 'return (sA)ᵀ', complexity: 'O(rc)',
    ref: '线性代数', commonPitfall: '无', invariant: '逐位 Object.is',
    inputs: ['A'], domains: { A: 'matInt' }, matrixDims: { A: [4, 4] },
    run: (A) => trans(mscale(2, A, 16), 4, 4),
    predicate: (A) => meq(runOf('C021')(A), mscale(2, trans(A, 4, 4), 16)),
  });
  reg(22, {
    name: '对称化 · A+Aᵀ 为对称阵', intuition: 'S=(A+Aᵀ)/2 满足 Sᵀ=S（逐位）。',
    math: '((A+Aᵀ)/2)ᵀ = (A+Aᵀ)/2', pseudocode: 'return sym(A)', complexity: 'O(rc)',
    ref: 'Horn & Johnson 1985', commonPitfall: '无', invariant: '逐位 Object.is',
    inputs: ['A'], domains: { A: 'matInt' }, matrixDims: { A: [4, 4] },
    run: (A) => mscale(0.5, madd(A, trans(A, 4, 4), 16), 16),
    predicate: (A) => {
      const S = runOf('C022')(A);
      return meq(trans(S, 4, 4), S) && meq(runOf('C022')(Z(4, 4)), Z(4, 4));   // run 产出自检验 + 零锚
    },
  });
  reg(23, {
    name: '零矩阵转置 · 仍为零', intuition: '零矩阵转置后仍为精确零矩阵。',
    math: '0ᵀ = 0', pseudocode: 'return 0ᵀ', complexity: 'O(rc)',
    ref: '线性代数', commonPitfall: '无', invariant: '所有元素 = 0',
    inputs: ['A'], domains: { A: 'matInt' }, matrixDims: { A: [4, 5] },
    run: (A) => trans(Z(4, 5), 4, 5), predicate: (A) => allIn(runOf('C023')(A), (x) => x === 0),
  });
  reg(24, {
    name: '单位阵转置 · 仍为自身', intuition: '单位阵是对称阵（逐位）。',
    math: 'Iᵀ = I', pseudocode: 'return Iᵀ', complexity: 'O(r²)',
    ref: '线性代数', commonPitfall: '无', invariant: '逐位 Object.is',
    inputs: ['A'], domains: { A: 'matInt' }, matrixDims: { A: [4, 4] },
    run: (A) => trans(id(4), 4, 4), predicate: (A) => meq(runOf('C024')(A), id(4)),
  });
  reg(25, {
    name: '矩阵转置 · Frobenius 范数不变', intuition: '转置不改变 Frobenius 范数（允许浮点误差）。',
    math: '‖Aᵀ‖_F = ‖A‖_F', pseudocode: 'return ‖Aᵀ‖_F', complexity: 'O(rc)',
    ref: 'Frobenius 范数', commonPitfall: '需容差。', invariant: 'relClose(‖Aᵀ‖_F, ‖A‖_F, 1e-9)',
    inputs: ['A'], domains: { A: 'matInt' }, matrixDims: { A: [4, 4] },
    run: (A) => trans(A, 4, 4),
    predicate: (A) => {
      const AT = runOf('C025')(A);
      return rc(fro(AT), fro(A), 1e-9) && rc(fro(AT), Math.sqrt(sumAll(msq(A, 16))), 1e-9);
    },
  });
  reg(26, {
    name: '矩阵减后转置 · 等于转置后相减', intuition: '(A−B)ᵀ = Aᵀ − Bᵀ（逐位）。',
    math: '(A−B)ᵀ = Aᵀ − Bᵀ', pseudocode: 'return (A−B)ᵀ', complexity: 'O(rc)',
    ref: '线性代数', commonPitfall: '无', invariant: '逐位 Object.is',
    inputs: ['A', 'B'], domains: { A: 'matInt', B: 'matInt' }, matrixDims: { A: [4, 4], B: [4, 4] },
    run: (A, B) => trans(msub(A, B, 16), 4, 4),
    predicate: (A, B) => meq(runOf('C026')(A, B), msub(trans(A, 4, 4), trans(B, 4, 4), 16)),
  });
  reg(27, {
    name: '逐元素绝对值转置 · 等于转置后取绝对值', intuition: '|Aᵀ| = |A|ᵀ（逐位）。',
    math: '|Aᵀ| = |A|ᵀ', pseudocode: 'return |Aᵀ|', complexity: 'O(rc)',
    ref: 'IEEE 754', commonPitfall: '无', invariant: '逐位 Object.is',
    inputs: ['A'], domains: { A: 'matFinite' }, matrixDims: { A: [4, 4] },
    run: (A) => mabs(trans(A, 4, 4), 16),
    predicate: (A) => meq(runOf('C027')(A), trans(mabs(A, 16), 4, 4)),
  });
  reg(28, {
    name: '矩阵乘法 · matmulAtB 右乘单位阵', intuition: 'A·Bᵀ 在 B=I 时退化为 A·I=A（逐位）。',
    math: 'A·Iᵀ = A', pseudocode: 'return A·Iᵀ', complexity: 'O(rc)',
    ref: '线性代数', commonPitfall: '无', invariant: '逐位 Object.is',
    inputs: ['A'], domains: { A: 'matInt' }, matrixDims: { A: [4, 5] },
    run: (A) => M.matmulAtB(A, id(5), 4, 5, 5),
    predicate: (A) => meq(runOf('C028')(A), A) && meq(runOf('C028')(Z(4, 5)), Z(4, 5)),
  });
  reg(29, {
    name: '矩阵乘法 · matmulTA 左乘单位阵', intuition: 'Iᵀ·B = B（逐位）。',
    math: 'Iᵀ·B = B', pseudocode: 'return Iᵀ·B', complexity: 'O(rc)',
    ref: '线性代数', commonPitfall: '无', invariant: '逐位 Object.is',
    inputs: ['A'], domains: { A: 'matInt' }, matrixDims: { A: [4, 5] },
    run: (A) => M.matmulTA(id(4), A, 4, 4, 5),
    predicate: (A) => meq(runOf('C029')(A), A) && meq(runOf('C029')(Z(4, 5)), Z(4, 5)),
  });
  reg(30, {
    name: '矩阵乘法 · 标量可提出（左）', intuition: 's(A·B) = (sA)·B（逐位）。',
    math: 's(A·B) = (sA)·B', pseudocode: 'return s*(A·B)', complexity: 'O(rkc)',
    ref: '线性代数', commonPitfall: '无', invariant: '逐位 Object.is',
    inputs: ['A', 'B'], domains: { A: 'matInt', B: 'matInt' }, matrixDims: { A: [4, 3], B: [3, 5] },
    run: (A, B) => mscale(2, M.matmul(A, B, 4, 3, 5), 20),
    predicate: (A, B) => {
      const run = runOf('C030');
      return meq(run(A, B), M.matmul(mscale(2, A, 12), B, 4, 3, 5))
        && meq(run(Z(4, 3), Z(3, 5)), Z(4, 5));   // 定点锚：2·(0·0)=0
    },
  });
  reg(31, {
    name: '矩阵乘法 · 对右侧加法分配', intuition: 'C·(A+B) = C·A + C·B（整数域精确）。',
    math: 'C·(A+B) = C·A + C·B', pseudocode: 'return C·(A+B)', complexity: 'O(n³)',
    ref: '线性代数', commonPitfall: '浮点舍入，故用整数。', invariant: '逐位 Object.is',
    inputs: ['A', 'B', 'C'], domains: { A: 'matInt', B: 'matInt', C: 'matInt' }, matrixDims: { C: [4, 5], A: [5, 3], B: [5, 3] },
    run: (A, B, C) => M.matmul(C, madd(A, B, 15), 4, 5, 3),
    predicate: (A, B, C) => {
      const run = runOf('C031');
      return meq(run(A, B, C), madd(M.matmul(C, A, 4, 5, 3), M.matmul(C, B, 4, 5, 3), 12))
        && meq(run(ones(5, 3), Z(5, 3), ones(4, 5)), mscale(5, ones(4, 3), 12));   // 定点锚：ones·(ones+0) 每项=5
    },
  });
  reg(32, {
    name: '矩阵乘法 · 左右乘单位阵皆还原', intuition: 'I·A = A 且 A·I = A（逐位）。',
    math: 'I·A = A = A·I', pseudocode: 'return I·A, A·I', complexity: 'O(rc)',
    ref: '线性代数', commonPitfall: 'A·I 侧由 C008（4×5）覆盖；本项 4×4 域验 I·A + I·I 锚。', invariant: '逐位 Object.is',
    inputs: ['A'], domains: { A: 'matInt' }, matrixDims: { A: [4, 4] },
    run: (A) => M.matmul(id(4), A, 4, 4, 4),
    predicate: (A) => meq(runOf('C032')(A), A) && meq(runOf('C032')(id(4)), id(4)),   // 两侧均经 run 本体
  });
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

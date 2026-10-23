/* ============================================================================
 * L10 微指令 · C 域（矩阵/张量运算）· 第 2/3 批（C033–C064）
 * ----------------------------------------------------------------------------
 * 涵盖：广播/逐元素（C033–C048）、归约（C049–C064）。复用 NS.L10.mh 工具块。
 *
 * ── 谓词纪律（本批全部重写，2026-10-23）─────────────────────────────────
 * 门禁只调 op.predicate，从不调 op.run（op-harness.js:43、probe-microisa.js:106）。
 * 原 32 项谓词全部内联复算 run 定义式——其中 C035/C062 甚至是 meq(o,o) 自比较
 * （恒真，100% 假绿）。现全部改为：先经 runOf('Cxxx') 调 run 本体，再对 run
 * 的产出断言不变量；oracle 取内联索引路径（广播逐位定义、列和、行均值）、
 * 异构归约顺序或常数锚；幂等类经 run 双重调用。
 * 注：colSum(A,r,c) 实为 A 的行和（length r），命名易误读，oracle 按定义直算。
 * ==========================================================================*/
;(function (NS) {
  'use strict';
  NS.L10 = NS.L10 || {};
  const M = NS.L8.matmul, R = NS.L8.reduce, SP = NS.L9.special, L10 = NS.L10;
  const mh = L10.mh;
  function reg(i, o) { NS.L10.register('C', i, o); }
  function runOf(key) { const o = NS.L10.ops[key]; return o && typeof o.run === 'function' ? o.run : null; }
  const { Z, madd, msub, mscale, mneg, mabs, mrelu, mclamp, emp, msq, trans, id, ones, allIn, sumAll, fro, colSum, meq, rc } = mh;
  const TWOS4 = mscale(2, ones(4, 4), 16); // 全 2 矩阵（4×4），供 C047 广播缩放对照
  function rowSum(A, r, c) { return R.rowSum(A, r, c); }
  function matmul(A, B, r, k, c) { return M.matmul(A, B, r, k, c); }
  // 内联列和（4×5 → length 5）：C049/C050/C062 的独立 oracle 路径。
  function inlineColSum(A) {
    const s = new Float64Array(5);
    for (let j = 0; j < 5; j++) { let acc = 0; for (let i = 0; i < 4; i++) acc += A[i * 5 + j]; s[j] = acc; }
    return s;
  }
  // 内联行和（4×5 → length 4）。
  function inlineRowSum(A) {
    const s = new Float64Array(4);
    for (let i = 0; i < 4; i++) { let acc = 0; for (let j = 0; j < 5; j++) acc += A[i * 5 + j]; s[i] = acc; }
    return s;
  }

  // ── 广播 / 逐元素（C033–C048）─────────────────────────────────────────
  reg(33, {
    name: '矩阵缩放 · 乘以 0 为零矩阵', intuition: '任意矩阵乘以标量 0 为精确零矩阵。',
    math: '0·A = 0', pseudocode: 'return 0*A', complexity: 'O(rc)',
    ref: '线性代数', commonPitfall: '0·负数为 −0，=== 0 判定兼容。', invariant: '所有元素 = 0',
    inputs: ['A'], domains: { A: 'matFinite' }, matrixDims: { A: [4,4] },
    run: (A) => mscale(0, A, 16), predicate: (A) => allIn(runOf('C033')(A), (x) => x === 0),
  });
  reg(34, {
    name: '广播 · 行向量加到每行', intuition: '将 1×c 行向量广播加到 A 的每一行（逐位）。',
    math: '(A + r⃗)ᵢⱼ = Aᵢⱼ + rⱼ', pseudocode: 'broadcast row add', complexity: 'O(rc)',
    ref: 'NumPy broadcasting', commonPitfall: '维度须匹配。', invariant: '逐位 Object.is',
    inputs: ['A', 'r'], domains: { A: 'matInt', r: 'matInt' }, matrixDims: { A: [4,4], r: [1,4] },
    run: (A, r) => { const o = Float64Array.from(A); for (let i=0;i<4;i++) for (let j=0;j<4;j++) o[i*4+j] += r[j]; return o; },
    predicate: (A, r) => {
      const o = runOf('C034')(A, r);
      for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) if (!meq(o[i * 4 + j], A[i * 4 + j] + r[j])) return false;
      return meq(runOf('C034')(Z(4, 4), ones(1, 4)), ones(4, 4));   // 定点锚：0+广播1=全1
    },
  });
  reg(35, {
    name: '广播 · 列向量加到每列', intuition: '将 r×1 列向量广播加到 A 的每一列（逐位）。',
    math: '(A + c⃗)ᵢⱼ = Aᵢⱼ + cᵢ', pseudocode: 'broadcast col add', complexity: 'O(rc)',
    ref: 'NumPy broadcasting', commonPitfall: '维度须匹配。', invariant: '逐位 Object.is',
    inputs: ['A', 'c'], domains: { A: 'matInt', c: 'matInt' }, matrixDims: { A: [4,4], c: [4,1] },
    run: (A, c) => { const o = Float64Array.from(A); for (let i=0;i<4;i++) for (let j=0;j<4;j++) o[i*4+j] += c[i]; return o; },
    predicate: (A, c) => {
      const o = runOf('C035')(A, c);
      for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) if (!meq(o[i * 4 + j], A[i * 4 + j] + c[i])) return false;
      return meq(runOf('C035')(Z(4, 4), ones(4, 1)), ones(4, 4));   // 定点锚：0+广播1=全1
    },
  });
  reg(36, {
    name: '逐元素乘法 · 单位元（全 1）', intuition: '矩阵逐元素乘以全 1 矩阵不变（逐位）。Hadamard 积的单位元是全 1 矩阵（非单位阵）。',
    math: 'A ⊙ 1 = A', pseudocode: 'return A⊙1', complexity: 'O(rc)',
    ref: 'Hadamard 积', commonPitfall: '单位元为全 1 矩阵；单位阵用于矩阵乘而非逐元素乘。', invariant: '逐位 Object.is',
    inputs: ['A'], domains: { A: 'matInt' }, matrixDims: { A: [4,4] },
    run: (A) => emp(A, ones(4, 4), 16), predicate: (A) => meq(runOf('C036')(A), A),
  });
  reg(37, {
    name: '逐元素乘法 · 交换', intuition: 'Hadamard 积逐位交换。',
    math: 'A ⊙ B = B ⊙ A', pseudocode: 'return A⊙B', complexity: 'O(rc)',
    ref: 'Hadamard 积', commonPitfall: '维度须一致。', invariant: '逐位 Object.is',
    inputs: ['A', 'B'], domains: { A: 'matInt', B: 'matInt' }, matrixDims: { A: [4,4], B: [4,4] },
    run: (A, B) => emp(A, B, 16), predicate: (A, B) => meq(runOf('C037')(A, B), runOf('C037')(B, A)),
  });
  reg(38, {
    name: '逐元素乘法 · 乘以零矩阵', intuition: '任意矩阵逐元素乘以零矩阵为精确零。',
    math: 'A ⊙ 0 = 0', pseudocode: 'return A⊙0', complexity: 'O(rc)',
    ref: 'Hadamard 积', commonPitfall: '无', invariant: '所有元素 = 0',
    inputs: ['A'], domains: { A: 'matInt' }, matrixDims: { A: [4,4] },
    run: (A) => emp(A, Z(4,4), 16), predicate: (A) => allIn(runOf('C038')(A), (x) => x === 0),
  });
  reg(39, {
    name: '逐元素乘法 · 结合', intuition: 'Hadamard 积逐位可结合。',
    math: '(A⊙B)⊙C = A⊙(B⊙C)', pseudocode: 'return (A⊙B)⊙C', complexity: 'O(rc)',
    ref: 'Hadamard 积', commonPitfall: '无', invariant: '逐位 Object.is',
    inputs: ['A', 'B', 'C'], domains: { A: 'matInt', B: 'matInt', C: 'matInt' }, matrixDims: { A: [4,4], B: [4,4], C: [4,4] },
    run: (A, B, C) => emp(emp(A, B, 16), C, 16),
    predicate: (A, B, C) => {
      const run = runOf('C039');
      return meq(run(A, B, C), emp(A, emp(B, C, 16), 16))
        && meq(run(id(4), id(4), id(4)), id(4));   // 定点锚：I⊙I⊙I=I
    },
  });
  reg(40, {
    name: '逐元素乘法 · 对加法分配', intuition: 'A⊙(B+C) = A⊙B + A⊙C（逐位）。',
    math: 'A⊙(B+C) = A⊙B + A⊙C', pseudocode: 'return A⊙(B+C)', complexity: 'O(rc)',
    ref: 'Hadamard 积', commonPitfall: '无', invariant: '逐位 Object.is',
    inputs: ['A', 'B', 'C'], domains: { A: 'matInt', B: 'matInt', C: 'matInt' }, matrixDims: { A: [4,4], B: [4,4], C: [4,4] },
    run: (A, B, C) => emp(A, madd(B, C, 16), 16),
    predicate: (A, B, C) => {
      const run = runOf('C040');
      return meq(run(A, B, C), madd(emp(A, B, 16), emp(A, C, 16), 16))
        && meq(run(ones(4, 4), ones(4, 4), Z(4, 4)), ones(4, 4));   // 锚：ones⊙(ones+0)=ones
    },
  });
  reg(41, {
    name: '逐元素绝对值 · 取反对称', intuition: '|−A| = |A|（逐位）。',
    math: '|−A| = |A|', pseudocode: 'return |−A|', complexity: 'O(rc)',
    ref: 'IEEE 754', commonPitfall: '无', invariant: '逐位 Object.is',
    inputs: ['A'], domains: { A: 'matFinite' }, matrixDims: { A: [4,4] },
    run: (A) => mabs(mneg(A, 16), 16), predicate: (A) => meq(runOf('C041')(A), mabs(A, 16)),
  });
  reg(42, {
    name: '逐元素绝对值 · 缩放齐次', intuition: '|s·A| = |s|·|A|（逐位）。',
    math: '|sA| = |s|·|A|', pseudocode: 'return |sA|', complexity: 'O(rc)',
    ref: 'IEEE 754', commonPitfall: '无', invariant: '逐位 Object.is',
    inputs: ['A'], domains: { A: 'matInt' }, matrixDims: { A: [4,4] },
    run: (A) => mabs(mscale(2, A, 16), 16),
    predicate: (A) => meq(runOf('C042')(A), mscale(2, mabs(A, 16), 16)) && meq(runOf('C042')(Z(4, 4)), Z(4, 4)),
  });
  reg(43, {
    name: '逐元素乘法 · 与转置可交换', intuition: 'Aᵀ⊙Bᵀ = (A⊙B)ᵀ（逐位）。',
    math: 'Aᵀ⊙Bᵀ = (A⊙B)ᵀ', pseudocode: 'return Aᵀ⊙Bᵀ', complexity: 'O(rc)',
    ref: 'Hadamard 积', commonPitfall: '无', invariant: '逐位 Object.is',
    inputs: ['A', 'B'], domains: { A: 'matInt', B: 'matInt' }, matrixDims: { A: [4,4], B: [4,4] },
    run: (A, B) => emp(trans(A, 4, 4), trans(B, 4, 4), 16),
    predicate: (A, B) => meq(runOf('C043')(A, B), trans(emp(A, B, 16), 4, 4)),
  });
  reg(44, {
    name: '矩阵缩放 · 保留零元', intuition: 'A 中为零的元素经缩放后仍为零（逐位）。',
    math: 'Aᵢ=0 ⇒ (sA)ᵢ=0', pseudocode: 'return s*A', complexity: 'O(rc)',
    ref: '线性代数', commonPitfall: '无', invariant: 'Aᵢ=0 ⇒ (sA)ᵢ=0',
    inputs: ['A'], domains: { A: 'matInt' }, matrixDims: { A: [4,4] },
    run: (A) => mscale(3, A, 16),
    predicate: (A) => {
      const s = runOf('C044')(A);
      for (let i = 0; i < 16; i++) if (A[i] === 0 && s[i] !== 0) return false;
      return meq(s, mscale(3, A, 16)) && meq(runOf('C044')(Z(4, 4)), Z(4, 4));
    },
  });
  reg(45, {
    name: '逐元素平方 · 非负', intuition: '逐元素平方恒非负。',
    math: 'A²ᵢ ≥ 0', pseudocode: 'return A²', complexity: 'O(rc)',
    ref: '代数', commonPitfall: '无', invariant: '∀i: A²ᵢ ≥ 0',
    inputs: ['A'], domains: { A: 'matFinite' }, matrixDims: { A: [4,4] },
    run: (A) => msq(A, 16),
    predicate: (A) => allIn(runOf('C045')(A), (x) => x >= 0),
  });
  reg(46, {
    name: '逐元素 ReLU · 负值归零', intuition: 'A 中为负的位置经 ReLU 后为精确 0。',
    math: 'Aᵢ<0 ⇒ ReLU(A)ᵢ=0', pseudocode: 'return ReLU(A)', complexity: 'O(rc)',
    ref: 'Nair & Hinton 2010', commonPitfall: '无', invariant: 'Aᵢ<0 ⇒ 输出 0',
    inputs: ['A'], domains: { A: 'matFinite' }, matrixDims: { A: [4,4] },
    run: (A) => mrelu(A, 16),
    predicate: (A) => {
      const r = runOf('C046')(A);
      for (let i = 0; i < 16; i++) {
        if (A[i] < 0 && r[i] !== 0) return false;
        if (r[i] < A[i]) return false;            // ReLU(A) ≥ A（逐元素保序）
      }
      return meq(runOf('C046')(Z(4, 4)), Z(4, 4));
    },
  });
  reg(47, {
    name: '广播缩放 · 等于常矩阵逐元素乘', intuition: '标量 s 缩放等价于与全 s 矩阵逐元素乘（逐位）。',
    math: 's·A = A⊙s', pseudocode: 'return s*A', complexity: 'O(rc)',
    ref: '线性代数', commonPitfall: '标量广播应与逐元素乘全 s 矩阵一致。', invariant: '逐位 Object.is',
    inputs: ['A'], domains: { A: 'matInt' }, matrixDims: { A: [4,4] },
    run: (A) => mscale(2, A, 16),
    predicate: (A) => meq(runOf('C047')(A), emp(A, TWOS4, 16)) && meq(runOf('C047')(Z(4, 4)), Z(4, 4)),
  });
  reg(48, {
    name: '逐元素钳位 · 幂等', intuition: '对同一区间重复钳位结果不变（逐位）。',
    math: 'clamp(clamp(A)) = clamp(A)', pseudocode: 'return clamp(A)', complexity: 'O(rc)',
    ref: '数值稳定', commonPitfall: '无', invariant: '逐位 Object.is',
    inputs: ['A'], domains: { A: 'matFinite' }, matrixDims: { A: [4,4] },
    run: (A) => mclamp(A, 16, -2, 2),
    predicate: (A) => {
      const run = runOf('C048');
      const r1 = run(A);
      return meq(run(r1), r1) && allIn(r1, (x) => x >= -2 && x <= 2);   // run 双重调用 + 有界性
    },
  });

  // ── 归约（C049–C064）──────────────────────────────────────────────────
  reg(49, {
    name: '归约 · 按行求和的列正确性', intuition: 'rowSum 第 j 列等于 A 各行列 j 之和（逐位）。',
    math: 'rowSum(A)ⱼ = Σᵢ Aᵢⱼ', pseudocode: 'return rowSum(A)', complexity: 'O(rc)',
    ref: '归约', commonPitfall: '无', invariant: '逐位 Object.is',
    inputs: ['A'], domains: { A: 'matInt' }, matrixDims: { A: [4,5] },
    run: (A) => rowSum(A, 4, 5), predicate: (A) => meq(runOf('C049')(A), inlineColSum(A)),
  });
  reg(50, {
    name: '归约 · 列和等于转置的行和', intuition: '列和 = 转置后的行和（逐位）。',
    math: 'colSum(A) = rowSum(Aᵀ)', pseudocode: 'return colSum(A)', complexity: 'O(rc)',
    ref: '归约', commonPitfall: 'colSum(A,r,c) 实为 A 的行和（length r），命名易误读。', invariant: '逐位 Object.is',
    inputs: ['A'], domains: { A: 'matInt' }, matrixDims: { A: [4,5] },
    run: (A) => colSum(A, 4, 5), predicate: (A) => meq(runOf('C050')(A), inlineRowSum(A)),
  });
  reg(51, {
    name: '归约 · 零矩阵行和为零', intuition: '零矩阵按行求和得精确零向量。',
    math: 'rowSum(0) = 0', pseudocode: 'return rowSum(0)', complexity: 'O(rc)',
    ref: '归约', commonPitfall: '无', invariant: '所有元素 = 0',
    inputs: ['A'], domains: { A: 'matInt' }, matrixDims: { A: [4,5] },
    run: (A) => rowSum(Z(4,5), 4, 5), predicate: (A) => allIn(runOf('C051')(A), (x) => x === 0),
  });
  reg(52, {
    name: '归约 · 行和齐次缩放', intuition: 'rowSum(s·A) = s·rowSum(A)（逐位）。',
    math: 'rowSum(sA) = s·rowSum(A)', pseudocode: 'return rowSum(sA)', complexity: 'O(rc)',
    ref: '线性', commonPitfall: '无', invariant: '逐位 Object.is',
    inputs: ['A'], domains: { A: 'matInt' }, matrixDims: { A: [4,5] },
    run: (A) => rowSum(mscale(2, A, 20), 4, 5),
    predicate: (A) => meq(runOf('C052')(A), mscale(2, rowSum(A, 4, 5), 5)) && meq(runOf('C052')(Z(4, 5)), Z(1, 5)),
  });
  reg(53, {
    name: '归约 · 行和对加法分配', intuition: 'rowSum(A+B) = rowSum(A) + rowSum(B)（逐位）。',
    math: 'rowSum(A+B) = rowSum(A)+rowSum(B)', pseudocode: 'return rowSum(A+B)', complexity: 'O(rc)',
    ref: '线性', commonPitfall: '无', invariant: '逐位 Object.is',
    inputs: ['A', 'B'], domains: { A: 'matInt', B: 'matInt' }, matrixDims: { A: [4,5], B: [4,5] },
    run: (A, B) => rowSum(madd(A, B, 20), 4, 5),
    predicate: (A, B) => {
      const run = runOf('C053');
      return meq(run(A, B), madd(rowSum(A, 4, 5), rowSum(B, 4, 5), 5))
        && meq(run(Z(4, 5), Z(4, 5)), Z(1, 5));
    },
  });
  reg(54, {
    name: '归约 · 全元素和 = 行和之和', intuition: '矩阵全部元素之和 = 行和向量之和（逐位）。',
    math: 'ΣAᵢⱼ = Σⱼ rowSum(A)ⱼ', pseudocode: 'return sumAll(rowSum(A))', complexity: 'O(rc)',
    ref: '归约', commonPitfall: '无', invariant: '逐位 Object.is',
    inputs: ['A'], domains: { A: 'matInt' }, matrixDims: { A: [4,5] },
    run: (A) => sumAll(rowSum(A, 4, 5)),
    predicate: (A) => {
      let t = 0;
      for (let i = 0; i < 20; i++) t += A[i];               // 内联直合计（异构归约路径）
      return Object.is(runOf('C054')(A), t);
    },
  });
  reg(55, {
    name: '归约 · 行 softmax 行和为 1', intuition: '每行 softmax 后概率和趋近 1（允许浮点误差）。',
    math: 'Σⱼ softmax(A)ᵢⱼ = 1', pseudocode: 'return softmaxRows(A)', complexity: 'O(rc)',
    ref: 'Bridle 1990', commonPitfall: '需容差。', invariant: 'relClose(行和, 1, 1e-9)',
    inputs: ['A'], domains: { A: 'matFinite' }, matrixDims: { A: [4,5] },
    run: (A) => R.softmaxRows(A, 4, 5),
    predicate: (A) => {
      const P = runOf('C055')(A);
      for (let i = 0; i < 4; i++) {
        let s = 0;
        for (let j = 0; j < 5; j++) { const p = P[i * 5 + j]; if (!(p >= 0 && p <= 1 + 1e-12)) return false; s += p; }
        if (!rc(s, 1, 1e-9)) return false;
      }
      return true;
    },
  });
  reg(56, {
    name: '归约 · 行 softmax 非负', intuition: 'softmax 输出概率恒非负。',
    math: 'softmax(A)ᵢⱼ ≥ 0', pseudocode: 'return softmaxRows(A)', complexity: 'O(rc)',
    ref: 'Bridle 1990', commonPitfall: '无', invariant: '∀: Pᵢⱼ ≥ 0',
    inputs: ['A'], domains: { A: 'matFinite' }, matrixDims: { A: [4,5] },
    run: (A) => R.softmaxRows(A, 4, 5),
    predicate: (A) => allIn(runOf('C056')(A), (x) => x >= 0 && x <= 1 + 1e-12),   // 概率 ∈ [0,1]
  });
  reg(57, {
    name: '归约 · softmax 保序（argmax）', intuition: 'softmax 单调，输入行最大值位置概率最大（允许浮点误差）。',
    math: 'argmax_i A = argmax_i softmax(A)', pseudocode: 'return softmaxRows(A)', complexity: 'O(rc)',
    ref: 'Bridle 1990', commonPitfall: '需容差。', invariant: 'softmax 行最大概率位于输入 argmax',
    inputs: ['A'], domains: { A: 'matSmall' }, matrixDims: { A: [4,5] },
    run: (A) => R.softmaxRows(A, 4, 5),
    predicate: (A) => {
      const P = runOf('C057')(A);
      for (let i = 0; i < 4; i++) {
        let m = 0;
        for (let j = 1; j < 5; j++) if (A[i * 5 + j] > A[i * 5 + m]) m = j;
        for (let j = 0; j < 5; j++) {
          const pm = P[i * 5 + m], pj = P[i * 5 + j];
          if (!rc(pm, pj, 1e-9) && pm < pj) return false;   // argmax 处概率须为行内最大（或并列）
        }
      }
      return true;
    },
  });
  reg(58, {
    name: '归约 · top-k 行选最大 k 个', intuition: 'topkRows 返回的 k 个值确为该行最大的 k 个（允许浮点误差）。',
    math: 'topkRows = k 个最大元素', pseudocode: 'return topkRows(A,k)', complexity: 'O(rc log c)',
    ref: 'Shazeer 2017', commonPitfall: '需容差。', invariant: '返回值为行内前 k 大',
    inputs: ['A'], domains: { A: 'matFinite' }, matrixDims: { A: [4,5] },
    run: (A) => R.topkRows(A, 4, 5, 2),
    predicate: (A) => {
      const { idx, val } = runOf('C058')(A);
      for (let i = 0; i < 4; i++) {
        const row = [];
        for (let j = 0; j < 5; j++) row.push(A[i * 5 + j]);
        row.sort((a, b) => b - a);
        for (let t = 0; t < 2; t++) {
          if (!rc(val[i * 2 + t], row[t], 1e-9)) return false;
          const ix = idx[i * 2 + t];
          if (!(ix >= 0 && ix < 5)) return false;                     // 索引须落界内
          if (!rc(A[i * 5 + ix], val[i * 2 + t], 1e-9)) return false; // idx 与 val 自洽
        }
      }
      return true;
    },
  });
  reg(59, {
    name: '归约 · top-k 形状', intuition: 'topkRows 返回的 idx/val 行数为 r、列数为 k。',
    math: 'idx.length = r·k', pseudocode: 'return topkRows(A,k)', complexity: 'O(rc log c)',
    ref: 'Shazeer 2017', commonPitfall: '无', invariant: 'idx.length = val.length = r·k = 8；每行取值为行内前 2 大（独立排序 oracle 钉死，非仅形状）',
    inputs: ['A'], domains: { A: 'matFinite' }, matrixDims: { A: [4,5] },
    run: (A) => R.topkRows(A, 4, 5, 2),
    predicate: (A) => {
      const t = runOf('C059')(A);
      if (t.idx.length !== 8 || t.val.length !== 8) return false;
      for (let i = 0; i < 4; i++) {
        const row = [];
        for (let j = 0; j < 5; j++) row.push(A[i * 5 + j]);
        row.sort((a, b) => b - a);                                     // 独立降序排序
        for (let m = 0; m < 2; m++) {
          if (!rc(t.val[i * 2 + m], row[m], 1e-9)) return false;        // 次序统计值
          const ix = t.idx[i * 2 + m];
          if (!(ix >= 0 && ix < 5)) return false;                      // 索引落界内
          if (!rc(A[i * 5 + ix], t.val[i * 2 + m], 1e-9)) return false; // idx 与 val 自洽
        }
      }
      return true;
    },
  });
  reg(60, {
    name: '归约 · log-sum-exp 上下界', intuition: '对行向量 v：max(v) ≤ logsumexp(v) ≤ max(v) + log(n)。',
    math: 'max ≤ LSE ≤ max + ln n', pseudocode: 'return LSE(v)', complexity: 'O(rc)',
    ref: '数值稳定', commonPitfall: '需容差。', invariant: 'max ≤ LSE ≤ max + ln n',
    inputs: ['A'], domains: { A: 'matInt' }, matrixDims: { A: [4,5] },
    run: (A) => { let acc = A[0]; for (let i=1;i<A.length;i++) acc = SP.logsumexp2(acc, A[i]); return acc; },
    predicate: (A) => {
      const acc = runOf('C060')(A);
      let mx = A[0];
      for (let i = 1; i < A.length; i++) if (A[i] > mx) mx = A[i];
      if (!(acc >= mx - 1e-9 && acc <= mx + Math.log(A.length) + 1e-9)) return false;
      const S7 = mscale(7, ones(4, 5), 20);                 // 定点锚：20 个 7 的 LSE = 7+ln20
      return rc(runOf('C060')(S7), 7 + Math.log(20), 1e-12);
    },
  });
  reg(61, {
    name: '归约 · addInto 累加正确', intuition: 'rowSumInto 将行和累加到目标向量（逐位）。',
    math: 'dst += rowSum(A)', pseudocode: 'rowSumInto(dst,A)', complexity: 'O(rc)',
    ref: '归约', commonPitfall: '无', invariant: '逐位 Object.is',
    inputs: ['A'], domains: { A: 'matInt' }, matrixDims: { A: [4,5] },
    run: (A) => { const dst = Z(1, 5); R.rowSumInto(dst, A, 4, 5); return dst; },
    predicate: (A) => meq(runOf('C061')(A), rowSum(A, 4, 5)) && meq(runOf('C061')(Z(4, 5)), Z(1, 5)),
  });
  reg(62, {
    name: '归约 · 行均值 = 行和 / 列数', intuition: '逐元素均值 = 行和 / c（允许 1 ulp 舍入差）。',
    math: 'meanᵢ = rowSumᵢ / c', pseudocode: 'return rowSum(A)/c', complexity: 'O(rc)',
    ref: '统计量', commonPitfall: 'run 以乘 0.2 实现、oracle 以 /5 实现：0.2 非精确表示，末位可差 1 ulp，故容差 1e-12（数学性质精确满足）。', invariant: 'relClose(mean, rowSum/c, 1e-12)',
    inputs: ['A'], domains: { A: 'matInt' }, matrixDims: { A: [4,5] },
    run: (A) => mscale(1/5, rowSum(A, 4, 5), 5),
    predicate: (A) => {
      const ref = new Float64Array(5);
      for (let j = 0; j < 5; j++) { let s = 0; for (let i = 0; i < 4; i++) s += A[i * 5 + j]; ref[j] = s / 5; }
      return rc(runOf('C062')(A), ref, 1e-12) && meq(runOf('C062')(Z(4, 5)), Z(1, 5));
    },
  });
  reg(63, {
    name: '归约 · 平方和 = Frobenius 范数平方', intuition: 'ΣAᵢⱼ² = ‖A‖_F²（允许浮点误差）。',
    math: 'ΣA² = ‖A‖_F²', pseudocode: 'return sumAll(A²)', complexity: 'O(rc)',
    ref: 'Frobenius 范数', commonPitfall: '需容差。', invariant: 'relClose(ΣA², ‖A‖_F², 1e-9)',
    inputs: ['A'], domains: { A: 'matFinite' }, matrixDims: { A: [4,4] },
    run: (A) => sumAll(msq(A, 16)),
    predicate: (A) => {
      const s = runOf('C063')(A);
      if (!rc(s, fro(A) * fro(A), 1e-9)) return false;
      return rc(runOf('C063')(id(4)), 4, 1e-12);   // 定点锚：ΣI² = 4
    },
  });
  reg(64, {
    name: '归约 · 绝对值和的非负', intuition: '矩阵绝对值之和是 IEEE 范数的下界，恒非负。',
    math: 'Σ|Aᵢⱼ| ≥ 0', pseudocode: 'return sumAll(|A|)', complexity: 'O(rc)',
    ref: 'ℓ1 范数', commonPitfall: '无', invariant: 'Σ|A| ≥ 0',
    inputs: ['A'], domains: { A: 'matFinite' }, matrixDims: { A: [4,4] },
    run: (A) => sumAll(mabs(A, 16)),
    predicate: (A) => {
      const s = runOf('C064')(A);
      return s >= 0 && s >= fro(A) - 1e-9 && Object.is(runOf('C064')(Z(4, 4)), 0);   // ℓ1 ≥ ℓ2 ≥ 0
    },
  });
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

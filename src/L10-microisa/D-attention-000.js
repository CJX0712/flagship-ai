/* ============================================================================
 * L10 微指令 · D 域（注意力基元）· 第 1/3 批（D001–D032）
 * ----------------------------------------------------------------------------
 * 所属层：L10-microIsa  域：D（注意力基元）
 * 允许依赖的层：L9（rng/assert/special）、L10（registry/invariant）
 * 主要 @ref：Vaswani et al. 2017（注意力）；Su et al. 2021（RoPE）；
 *           Press et al. 2022（ALiBi）；Milakov & Gimelshein 2018（在线 softmax）；
 *           Xiao et al. 2024（StreamingLLM / Attention Sink）
 *
 * 子域划分：
 *   D001–D012 打分与缩放（点积 / 缩放 / Cauchy–Schwarz / mscale）
 *   D013–D024 掩码（因果 / 滑动窗口 / padding / 安全 softmax 回退）
 *   D025–D032 softmax 与 log-sum-exp 归约
 *
 * 不变量判据：精确类（0 / 位级对称 / 幂等）用 Object.is；数值类用 relClose
 * （tol 1e-9，缩放类 1e-12）。每条 predicate 都走与 run 不同的参考路径，
 * 例如 run 正向累加、predicate 反向累加，避免「自证」。
 * 输入域由 L10.TAGS 约束（tests/L10 据此构造 fast-check 任意子）。
 * 约定：矩阵参数为方阵（默认 4×4），行数取 round(sqrt(M.length))。
 *
 * ── 谓词纪律（本批全部重写，2026-10-23）─────────────────────────────────
 * 门禁只调 op.predicate，从不调 op.run（op-harness.js:43、probe-microisa.js:106）。
 * 原 32 项谓词的"参考路径"虽异构（反向累加/朴素 softmax），但比对对象是本地
 * helper 而非 run 本体——run 若被损坏（如点积写错、掩码方向反），门禁完全
 * 失明（假绿）。现全部改为：先经 runOf('Dxxx') 调 run 本体，再对 run 的产出
 * 断言；oracle 取异构 helper（dotRev/naive）、跨 op 互引（D001 裸点积、
 * D013 因果掩码、D025 softmax、D030 LSE）或常数锚。
 * ==========================================================================*/
;(function (NS) {
  'use strict';
  NS.L10 = NS.L10 || {};
  const rc = NS.L10.relClose;
  function reg(i, o) { NS.L10.register('D', i, o); }
  function runOf(key) { const o = NS.L10.ops[key]; return o && typeof o.run === 'function' ? o.run : null; }

  /* ---------- 内部工具（本文件自包含，不跨层 require） ---------- */
  const sq = (M) => Math.round(Math.sqrt(M.length));

  function dot(a, b, n) {                 // 正向累加
    let s = 0;
    for (let i = 0; i < n; i++) s += a[i] * b[i];
    return s;
  }
  function dotRev(a, b, n) {              // 参考路径：反向累加
    let s = 0;
    for (let i = n - 1; i >= 0; i--) s += a[i] * b[i];
    return s;
  }
  function norm2(x, n) {
    let s = 0;
    for (let i = 0; i < n; i++) s += x[i] * x[i];
    return Math.sqrt(s);
  }
  function negv(x, n) {
    const o = new Float64Array(n);
    for (let i = 0; i < n; i++) o[i] = -x[i];
    return o;
  }
  // 安全行 softmax：减行最大值；整行 −∞（全掩）回退为均匀分布（sink 保护）
  function softmaxRow(s, off, n) {
    const out = new Float64Array(n);
    let mx = -Infinity;
    for (let j = 0; j < n; j++) if (s[off + j] > mx) mx = s[off + j];
    if (mx === -Infinity) {
      const u = 1 / n;
      for (let j = 0; j < n; j++) out[j] = u;
      return out;
    }
    let sum = 0;
    for (let j = 0; j < n; j++) { const e = Math.exp(s[off + j] - mx); out[j] = e; sum += e; }
    for (let j = 0; j < n; j++) out[j] /= sum;
    return out;
  }
  function softmaxRows(M, n) {
    const out = new Float64Array(n * n);
    for (let i = 0; i < n; i++) out.set(softmaxRow(M, i * n, n), i * n);
    return out;
  }
  function lse(s, off, n) {
    let mx = -Infinity;
    for (let j = 0; j < n; j++) if (s[off + j] > mx) mx = s[off + j];
    if (mx === -Infinity) return -Infinity;
    let sum = 0;
    for (let j = 0; j < n; j++) sum += Math.exp(s[off + j] - mx);
    return mx + Math.log(sum);
  }
  function causal(M, n) {
    const out = new Float64Array(n * n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) out[i * n + j] = (j <= i ? M[i * n + j] : -Infinity);
    return out;
  }
  function windowMask(M, n, w) {
    const out = new Float64Array(n * n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) out[i * n + j] = (Math.abs(i - j) <= w ? M[i * n + j] : -Infinity);
    return out;
  }
  function padMask(M, n, keep) {
    const out = new Float64Array(n * n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) out[i * n + j] = (keep[j] !== 0 ? M[i * n + j] : -Infinity);
    return out;
  }
  const mscale = (f) => Math.sqrt(1 + 0.1 * Math.log(f));

  // ══ 打分与缩放（D001–D012）══════════════════════════════════════════
  reg(1, {
    name: '打分 · 点积（正向累加）', intuition: 'q·k 是注意力的原子打分：一次乘加流。',
    math: 's(q,k) = Σᵢ₌₀^{d−1} qᵢ·kᵢ', pseudocode: 's=0; for i: s += q[i]*k[i]; return s',
    complexity: 'O(d) 时间 · O(1) 额外空间', ref: 'Vaswani et al. 2017 §3.2.1',
    commonPitfall: '浮点加法不可结合：分块/并行归约的结果与串行不同，跨实现比对须用容差。',
    invariant: 'relClose(run(q,k), 反向累加, 1e-9)',
    inputs: ['q', 'k'], domains: { q: 'vecSmall', k: 'vecSmall' },
    run: (q, k) => dot(q, k, q.length),
    predicate: (q, k) => rc(runOf('D001')(q, k), dotRev(q, k, q.length), 1e-9)
      && Object.is(runOf('D001')(Float64Array.from([2, 3]), Float64Array.from([5, 7])), 31),   // 定点锚：2·5+3·7=31
  });
  reg(2, {
    name: '打分 · 交换律', intuition: '两向量点积逐位可交换，是打分对称性与缓存复用（一份 K 服务多 Q）的前提。',
    math: 's(q,k) = s(k,q)', pseudocode: 'return dot(q,k) - dot(k,q)',
    complexity: 'O(d)', ref: 'IEEE 754-2019 §5.1',
    commonPitfall: '交换律成立，结合律不成立——不要据此推断分块求和可重排。',
    invariant: 'Object.is(run(q,k), run(k,q))（run 双向调用）',
    inputs: ['q', 'k'], domains: { q: 'vecSmall', k: 'vecSmall' },
    run: (q, k) => dot(q, k, q.length),
    predicate: (q, k) => Object.is(runOf('D002')(q, k), runOf('D002')(k, q)),
  });
  reg(3, {
    name: '打分 · 齐次性', intuition: 'q 整体放大 a 倍，打分也放大 a 倍（缩放点积可被温度吸收）。',
    math: 's(a·q, k) = a·s(q,k)', pseudocode: 'return dot(scale(q,a), k)',
    complexity: 'O(d)', ref: '线性代数（双线性型）',
    commonPitfall: '浮点下两条路径舍入不同，逐位断言会偶发失败——用 1e-9 相对容差。',
    invariant: 'relClose(run(a,q,k), a·run(1,q,k), 1e-9)',
    inputs: ['a', 'q', 'k'], domains: { a: 'smallInt', q: 'vecSmall', k: 'vecSmall' },
    run: (a, q, k) => {
      const s = new Float64Array(q.length);
      for (let i = 0; i < q.length; i++) s[i] = a * q[i];
      return dot(s, k, q.length);
    },
    predicate: (a, q, k) => rc(runOf('D003')(a, q, k), a * runOf('D003')(1, q, k), 1e-9),   // run 双重调用（a=1 基准）
  });
  reg(4, {
    name: '打分 · 对加法分配', intuition: '两个 query 合并成一条再打分，等于分别打分再相加。',
    math: 's(q₁+q₂, k) = s(q₁,k) + s(q₂,k)', pseudocode: 'return dot(add(q1,q2), k)',
    complexity: 'O(d)', ref: '双线性型（对第一变元线性）',
    commonPitfall: '浮点下两侧舍入路径不同，须容差比对。',
    invariant: 'relClose(run(q1,q2,k), run(q1,0,k)+run(q2,0,k), 1e-9)',
    inputs: ['q1', 'q2', 'k'], domains: { q1: 'vecSmall', q2: 'vecSmall', k: 'vecSmall' },
    run: (q1, q2, k) => {
      const s = new Float64Array(q1.length);
      for (let i = 0; i < q1.length; i++) s[i] = q1[i] + q2[i];
      return dot(s, k, q1.length);
    },
    predicate: (q1, q2, k) => {
      const zv = new Float64Array(q1.length);   // 零向量：run(q1,0,k) 即裸 dot(q1,k)
      return rc(runOf('D004')(q1, q2, k), runOf('D004')(q1, zv, k) + runOf('D004')(q2, zv, k), 1e-9);
    },
  });
  reg(5, {
    name: '缩放点积 · 除与乘倒数一致', intuition: '1/√d 缩放的两种写法（除法 / 乘倒数）数值等价。',
    math: 's/√d ≈ s·(1/√d)', pseudocode: 'return dot(q,k)/sqrt(d)',
    complexity: 'O(d)', ref: 'Vaswani et al. 2017 §3.2.1',
    commonPitfall: '除法和乘倒数舍入不同，逐位断言必挂——用 1e-12。',
    invariant: 'relClose(run(q,k,d), D001(q,k)·(1/√d), 1e-12)',
    inputs: ['q', 'k', 'd'], domains: { q: 'vecSmall', k: 'vecSmall', d: 'posInt' },
    run: (q, k, d) => dot(q, k, q.length) / Math.sqrt(d),
    predicate: (q, k, d) => rc(runOf('D005')(q, k, d), runOf('D001')(q, k) * (1 / Math.sqrt(d)), 1e-12),
  });
  reg(6, {
    name: '打分 · Cauchy–Schwarz 上界', intuition: '点积的绝对值不超过两向量范数之积，是打分有界性的理论依据。',
    math: '|⟨q,k⟩| ≤ ‖q‖₂·‖k‖₂', pseudocode: 'return abs(dot(q,k)) <= norm(q)*norm(k)',
    complexity: 'O(d)', ref: 'Cauchy–Schwarz 不等式',
    commonPitfall: '浮点误差下需留 1e-12 相对裕度，否则等号情形会假失败。',
    invariant: 'run(q,k) === |⟨q,k⟩|（Cauchy–Schwarz 上界由其导出，此处钉死数值）',
    inputs: ['q', 'k'], domains: { q: 'vecSmall', k: 'vecSmall' },
    run: (q, k) => Math.abs(dot(q, k, q.length)),
    predicate: (q, k) => Object.is(runOf('D006')(q, k), Math.abs(dot(q, k, q.length))),
  });
  reg(7, {
    name: '打分 · 自点积非负', intuition: '向量与自身的点积是范数平方，必非负。',
    math: '⟨q,q⟩ = ‖q‖² ≥ 0', pseudocode: 'return dot(q,q)',
    complexity: 'O(d)', ref: '内积正定性',
    commonPitfall: 'NaN 输入需另行排除（域标签已约束为有限）。',
    invariant: 'run(q) ≥ 0 且 run(0) === 0',
    inputs: ['q'], domains: { q: 'vecSmall' },
    run: (q) => dot(q, q, q.length),
    predicate: (q) => runOf('D007')(q) >= 0 && Object.is(runOf('D007')(new Float64Array(q.length)), 0),
  });
  reg(8, {
    name: '打分 · 自点积等于范数平方', intuition: '两条独立公式（累加式与 sqrt 后平方）应给出同一值。',
    math: 'Σ qᵢ² = (√(Σ qᵢ²))²', pseudocode: 'return dot(q,q)',
    complexity: 'O(d)', ref: 'L2 范数定义',
    commonPitfall: 'sqrt 再平方不是恒等操作，须容差。',
    invariant: 'relClose(run(q), ‖q‖², 1e-9)',
    inputs: ['q'], domains: { q: 'vecSmall' },
    run: (q) => dot(q, q, q.length),
    predicate: (q) => rc(runOf('D008')(q), Math.pow(norm2(q, q.length), 2), 1e-9),
  });
  reg(9, {
    name: '打分 · 符号翻转', intuition: 'k 取反使打分变号，是掩码 −∞ 之外做「抑制」的理论基础。',
    math: '⟨q, −k⟩ = −⟨q,k⟩', pseudocode: 'return dot(q, neg(k))',
    complexity: 'O(d)', ref: '双线性型（对第二变元线性）',
    commonPitfall: '累加后取负 与 逐项取负再累加 在符号零上可能差一个位（−0 vs +0），用容差规避。',
    invariant: 'relClose(run(q,k), −D001(q,k), 1e-9)',
    inputs: ['q', 'k'], domains: { q: 'vecSmall', k: 'vecSmall' },
    run: (q, k) => dot(q, negv(k, k.length), q.length),
    predicate: (q, k) => rc(runOf('D009')(q, k), -runOf('D001')(q, k), 1e-9),
  });
  reg(10, {
    name: '打分矩阵 · QKᵀ 与逐行点积一致', intuition: '批量打分矩阵的每个元素就是对应行向量的点积。',
    math: 'S[i][j] = ⟨Qᵢ, Kⱼ⟩', pseudocode: 'for i,j: S[i*n+j] = dot(Q_i, K_j)',
    complexity: 'O(n²·d) 时间 · O(n²) 空间', ref: 'Vaswani et al. 2017 §3.2.1',
    commonPitfall: '行优先索引别写反；此处所有矩阵按行优先展平。',
    invariant: '逐元素 relClose(run(Q,K)[i][j], dot(Qᵢ,Kⱼ), 1e-9)',
    inputs: ['Q', 'K'], domains: { Q: 'matSmall', K: 'matSmall' },
    matrixDims: { Q: [4, 4], K: [4, 4] },
    run: (Q, K) => {
      const n = sq(Q), d = Q.length / n, S = new Float64Array(n * n);
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        let s = 0;
        for (let t = 0; t < d; t++) s += Q[i * d + t] * K[j * d + t];
        S[i * n + j] = s;
      }
      return S;
    },
    predicate: (Q, K) => {
      const n = sq(Q), d = Q.length / n, S = runOf('D010')(Q, K);
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        let s = 0;
        for (let t = d - 1; t >= 0; t--) s += Q[i * d + t] * K[j * d + t];  // 参考：反向累加
        if (!rc(S[i * n + j], s, 1e-9)) return false;
      }
      return S.length === n * n;
    },
  });
  reg(11, {
    name: '缩放 · 温度单调性', intuition: '缩放分母越大，打分绝对值越小（分布越平缓）。',
    math: 'scale₁ < scale₂ ⇒ |s/scale₁| ≥ |s/scale₂|', pseudocode: 'return s/scale1',
    complexity: 'O(1)', ref: '注意力温度（softmax temperature）',
    commonPitfall: '分母须为正，否则符号翻转破坏单调性（域标签 pos 已约束）。',
    invariant: 'scale₁<scale₂ ⇒ |run(s,scale₁)| ≥ |run(s,scale₂)|',
    inputs: ['s', 'scale1', 'scale2'], domains: { s: 'small', scale1: 'pos', scale2: 'pos' },
    run: (s, scale1) => s / scale1,
    predicate: (s, scale1, scale2) => (scale1 < scale2 ? Math.abs(runOf('D011')(s, scale1)) >= Math.abs(runOf('D011')(s, scale2)) : true),
  });
  reg(12, {
    name: 'YaRN · mscale 公式', intuition: '长上下文外推时按 ln(factor) 放大 logits，补偿注意力熵增。',
    math: 'mscale(f) = √(1 + 0.1·ln f)；f=1 ⇒ 1', pseudocode: 'return sqrt(1 + 0.1*log(f))',
    complexity: 'O(1)', ref: 'Peng et al. 2023 (YaRN) §3.4 · Gemma 2 (2024)',
    commonPitfall: 'f<1 时 1+0.1·ln f 可能开负根得 NaN——故输入域限定 f≥1。',
    invariant: 'relClose(run(f), √(1+0.1·ln f), 1e-15) 且 run(1)===1',
    inputs: ['factor'], domains: { factor: 'factor1' },
    run: (f) => mscale(f),
    predicate: (f) => rc(runOf('D012')(f), Math.sqrt(1 + 0.1 * Math.log(f)), 1e-15) && Object.is(runOf('D012')(1), 1),
  });

  // ══ 掩码（D013–D024）════════════════════════════════════════════════
  reg(13, {
    name: '因果掩码 · 构造', intuition: '位置 i 只能看见 j≤i：上三角置 −∞，其余保留原打分。',
    math: 'M[i][j] = s[i][j] (j≤i)；−∞ (j>i)',
    pseudocode: 'for i,j: out[i*n+j] = j<=i ? s[i*n+j] : -Infinity',
    complexity: 'O(n²)', ref: 'Vaswani et al. 2017 §3.2.3',
    commonPitfall: '方向别搞反——生成时 query 在后、key 在前，掩的是「未来 key」。',
    invariant: '逐元素：j≤i 时 Object.is(run, s)；j>i 时 run === −Infinity',
    inputs: ['s'], domains: { s: 'matSmall' }, matrixDims: { s: [4, 4] },
    run: (s) => causal(s, sq(s)),
    predicate: (s) => {
      const n = sq(s), o = runOf('D013')(s);
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        if (j <= i) { if (!Object.is(o[i * n + j], s[i * n + j])) return false; }
        else if (o[i * n + j] !== -Infinity) return false;
      }
      return true;
    },
  });
  reg(14, {
    name: '因果掩码 · 对角线保留', intuition: '自关注权重必须可见，对角线不能被掩。',
    math: 'M[i][i] = s[i][i]', pseudocode: 'return causal(s)[i*n+i]',
    complexity: 'O(n)', ref: 'Vaswani et al. 2017 §3.2.3',
    commonPitfall: '把对角线也掩掉会让每个位置「看不见自己」，训练直接退化。',
    invariant: '逐位：Object.is(run[i*n+i], s[i*n+i])',
    inputs: ['s'], domains: { s: 'matSmall' }, matrixDims: { s: [4, 4] },
    run: (s) => { const n = sq(s), o = causal(s, n), d = new Float64Array(n); for (let i = 0; i < n; i++) d[i] = o[i * n + i]; return d; },
    predicate: (s) => {
      const n = sq(s), d = runOf('D014')(s);
      for (let i = 0; i < n; i++) if (!Object.is(d[i], s[i * n + i])) return false;
      return d.length === n;
    },
  });
  reg(15, {
    name: '因果掩码 · 上三角全 −∞', intuition: '所有未来位置严格不可见，softmax 后权重恰为 0。',
    math: '∀ j>i：M[i][j] = −∞', pseudocode: 'for i,j: assert j>i implies out === −Infinity',
    complexity: 'O(n²)', ref: 'Vaswani et al. 2017 §3.2.3',
    commonPitfall: '用 −1e9 代替 −∞ 是常见工程近似，但须保证 softmax 后真为 0（此处用 −∞ 取严格语义）。',
    invariant: '∀ j>i：run[i*n+j] === −Infinity',
    inputs: ['s'], domains: { s: 'matSmall' }, matrixDims: { s: [4, 4] },
    run: (s) => causal(s, sq(s)),
    predicate: (s) => {
      const n = sq(s), o = runOf('D015')(s);
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) if (o[i * n + j] !== -Infinity) return false;
      return true;
    },
  });
  reg(16, {
    name: '因果掩码 · 幂等', intuition: '掩一次与掩两次结果完全相同（−∞+0=−∞，0+0=0）。',
    math: 'causal(causal(s)) = causal(s)', pseudocode: 'return causal(causal(s,n),n)',
    complexity: 'O(n²)', ref: '掩码代数（加法 0 单位元 + 吸收元 −∞）',
    commonPitfall: '若用有限大负数做掩码，重复叠加会溢出——幂等性正是 −∞ 语义的价值。',
    invariant: '逐位 Object.is(run(s), D013(s))（run 双重掩码 vs 单次）',
    inputs: ['s'], domains: { s: 'matSmall' }, matrixDims: { s: [4, 4] },
    run: (s) => causal(causal(s, sq(s)), sq(s)),
    predicate: (s) => {
      const a = runOf('D016')(s), b = runOf('D013')(s);   // 双重掩码（run） vs 单次掩码（D013 run）
      for (let k = 0; k < a.length; k++) if (!Object.is(a[k], b[k])) return false;
      return a.length === b.length;
    },
  });
  reg(17, {
    name: '滑动窗口掩码 · 构造', intuition: '只保留距离 ≤w 的邻居，把全局注意力压成带状（SWA 核心）。',
    math: 'M[i][j] = s[i][j] (|i−j|≤w)；−∞ 否则',
    pseudocode: 'for i,j: out = abs(i-j)<=w ? s : -Infinity',
    complexity: 'O(n²)', ref: 'Beltagy et al. 2020 (Longformer) · Jiang et al. 2023 (Mistral SWA)',
    commonPitfall: '窗口是「距离」而非「行区间」；w 为 0 时只剩对角线。',
    invariant: '逐元素：|i−j|≤w 时 Object.is(run, s)；否则 run === −Infinity',
    inputs: ['s', 'w'], domains: { s: 'matSmall', w: 'nat' }, matrixDims: { s: [4, 4] },
    run: (s, w) => windowMask(s, sq(s), w),
    predicate: (s, w) => {
      const n = sq(s), o = runOf('D017')(s, w);
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        if (Math.abs(i - j) <= w) { if (!Object.is(o[i * n + j], s[i * n + j])) return false; }
        else if (o[i * n + j] !== -Infinity) return false;
      }
      return true;
    },
  });
  reg(18, {
    name: '滑动窗口掩码 · 对称性', intuition: '距离度量对称，故窗口掩码矩阵对称。',
    math: 'M[i][j] = M[j][i]', pseudocode: 'return mask(s,w)',
    complexity: 'O(n²)', ref: 'Longformer (2020)（对称窗口）',
    commonPitfall: '若再叠加因果掩码，对称性会被打破——此处仅窗口分量。',
    invariant: '逐位：(run[i*n+j]===−∞) === (|i−j|>w)（独立可见性 oracle，钉死掩码）',
    inputs: ['s', 'w'], domains: { s: 'matSmall', w: 'nat' }, matrixDims: { s: [4, 4] },
    run: (s, w) => windowMask(s, sq(s), w),
    predicate: (s, w) => {
      const n = sq(s), o = runOf('D018')(s, w);
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        const masked = o[i * n + j] === -Infinity;
        if (masked !== (Math.abs(i - j) > w)) return false;
      }
      return true;
    },
  });
  reg(19, {
    name: '滑动窗口 · 窗口单调性', intuition: '窗口越宽，可见位置只增不减。',
    math: 'w₁ ≤ w₂ ⇒ visible(w₁) ⊆ visible(w₂)',
    pseudocode: 'count1 = countVisible(mask(s,w1)); count2 = ...; return count1 <= count2',
    complexity: 'O(n²)', ref: 'SWA 复杂度随窗口单调（O(n·w)）',
    commonPitfall: '计数须对两个窗口用同一份 s，否则随机数干扰结论。',
    invariant: 'run(s,w₁,w₂) === Σᵢ[min(n,i+w₂+1)−max(0,i−w₂)] − Σᵢ[min(n,i+w₁+1)−max(0,i−w₁)]（可见格点差，闭式钉死）',
    inputs: ['s', 'w1', 'w2'], domains: { s: 'matSmall', w1: 'nat', w2: 'nat' }, matrixDims: { s: [4, 4] },
    run: (s, w1, w2) => {
      const n = sq(s);
      const c = (w) => { const o = windowMask(s, n, w); let k = 0; for (let i = 0; i < o.length; i++) if (o[i] !== -Infinity) k++; return k; };
      return c(w2) - c(w1);
    },
    predicate: (s, w1, w2) => {
      const n = sq(s);
      const cnt = (w) => { let k = 0; for (let i = 0; i < n; i++) k += Math.min(n, i + w + 1) - Math.max(0, i - w); return k; };
      return Object.is(runOf('D019')(s, w1, w2), cnt(w2) - cnt(w1));
    },
  });
  reg(20, {
    name: 'padding 掩码 · 无效列全 −∞', intuition: '把补齐位置彻底屏蔽，且不改变有效列的原打分。',
    math: 'M[i][j] = s[i][j] (keep[j]=1)；−∞ (keep[j]=0)',
    pseudocode: 'for i,j: out = keep[j] ? s : -Infinity',
    complexity: 'O(n²)', ref: 'Vaswani et al. 2017 §3.2.3（key padding mask）',
    commonPitfall: 'keep 长度须等于 key 数（列数），与 query 数（行数）无关。',
    inputs: ['s', 'keep'], domains: { s: 'matSmall', keep: 'vecMask' },
    matrixDims: { s: [4, 4] }, vecDims: { keep: 4 },
    run: (s, keep) => padMask(s, sq(s), keep),
    predicate: (s, keep) => {
      const n = sq(s), o = runOf('D020')(s, keep);
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        if (keep[j] !== 0) { if (!Object.is(o[i * n + j], s[i * n + j])) return false; }
        else if (o[i * n + j] !== -Infinity) return false;
      }
      return true;
    },
  });
  reg(21, {
    name: '掩码 · 与有限偏置可交换', intuition: '先掩后加偏置 与 先加偏置后掩 结果一致（−∞ 是吸收元）。',
    math: 'causal(s + b) = causal(s) + b',
    pseudocode: 'left = causal(add(s,b)); right = add(causal(s), b); return left == right',
    complexity: 'O(n²)', ref: '掩码代数（−∞ 吸收律）',
    commonPitfall: '−∞ + (+∞) 会得 NaN——偏置 b 必须有限（域 small 已约束）。',
    invariant: '逐元素：run(s,b) === D013(s)+b（含 −∞ 吸收）',
    inputs: ['s', 'b'], domains: { s: 'matSmall', b: 'small' }, matrixDims: { s: [4, 4] },
    run: (s, b) => {
      const n = sq(s), t = new Float64Array(s.length);
      for (let k = 0; k < s.length; k++) t[k] = s[k] + b;
      return causal(t, n);
    },
    predicate: (s, b) => {
      const n = sq(s);
      const left = runOf('D021')(s, b), right = runOf('D013')(s);   // run（先加后掩） vs D013 run（先掩）
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        const l = left[i * n + j], r = right[i * n + j] + b;
        if (l === -Infinity || !isFinite(r)) { if (!(l === -Infinity && !isFinite(r))) return false; }
        else if (!Object.is(l, r)) return false;
      }
      return true;
    },
  });
  reg(22, {
    name: '安全 softmax · 行和为一（含全掩回退）', intuition: '减行最大值防溢出；整行被掩时回退均匀分布，避免 NaN（sink 保护）。',
    math: 'pⱼ = exp(sⱼ − m)/Σ exp(sₖ − m)，m = max s；全掩时 pⱼ = 1/n',
    pseudocode: 'm = max(s); if m == -inf: return uniform; e = exp(s-m); return e/sum(e)',
    complexity: 'O(n) 时间 · O(n) 空间',
    ref: 'Milakov & Gimelshein 2018（在线 softmax）· Xiao et al. 2024（sink）',
    commonPitfall: '不做全掩回退会得到 0/0=NaN，长上下文预填充时极易踩到。',
    invariant: 'run 每行 Σp = 1、p∈[0,1] 有限；全 −∞ 行（helper 直测）恰为均匀 1/n',
    inputs: ['s'], domains: { s: 'matSmall' }, matrixDims: { s: [4, 4] },
    run: (s) => softmaxRows(s, sq(s)),
    predicate: (s) => {
      const n = sq(s), P = runOf('D022')(s);
      for (let i = 0; i < n; i++) {
        let sum = 0;
        for (let j = 0; j < n; j++) { const v = P[i * n + j]; if (!(v >= 0 && v <= 1) || !isFinite(v)) return false; sum += v; }
        if (!rc(sum, 1, 1e-12)) return false;
      }
      const all = new Float64Array(n);
      for (let j = 0; j < n; j++) all[j] = -Infinity;
      const pu = softmaxRow(all, 0, n);
      let s2 = 0;
      for (let j = 0; j < n; j++) { s2 += pu[j]; if (!rc(pu[j], 1 / n, 1e-15)) return false; }
      return rc(s2, 1, 1e-12);
    },
  });
  reg(23, {
    name: '掩码 · 布尔掩码与加性掩码等价', intuition: '两种掩码表示（0/1 布尔 → −∞/0 加性）在 softmax 后给出同一分布。',
    math: 'softmax(s + (1−keep)·(−∞)) = softmax(mask(s, keep))',
    pseudocode: 'p1 = softmax(addMask(s, keep)); p2 = softmax(masked(s, keep)); return relClose(p1, p2)',
    complexity: 'O(n)', ref: '掩码表示等价性',
    commonPitfall: '别对 −∞ 做减法运算（−∞ − (−∞) = NaN），应在 softmax 内先取 max 再跳过 −∞。',
    invariant: 'run 产出掩码后：掩位列 softmax 权重为 0，有效列与仅有效列归一化一致',
    inputs: ['s', 'keep'], domains: { s: 'matSmall', keep: 'vecMask' },
    matrixDims: { s: [4, 4] }, vecDims: { keep: 4 },
    run: (s, keep) => padMask(s, sq(s), keep),
    predicate: (s, keep) => {
      const n = sq(s);
      const o = runOf('D023')(s, keep);          // run 产出的加性掩码矩阵
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        if (keep[j] !== 0) { if (!Object.is(o[i * n + j], s[i * n + j])) return false; }
        else if (o[i * n + j] !== -Infinity) return false;
      }
      const P = softmaxRows(o, n);               // 对 run 产出做 softmax
      for (let i = 0; i < n; i++) {
        let cnt = 0;
        for (let j = 0; j < n; j++) if (keep[j] !== 0) cnt++;
        if (cnt === 0) {                              // 全掩行：回退为均匀分布
          for (let j = 0; j < n; j++) if (!rc(P[i * n + j], 1 / n, 1e-15)) return false;
          continue;
        }
        for (let j = 0; j < n; j++) if (keep[j] === 0 && P[i * n + j] !== 0) return false;
        // 参考路径：仅对有效列做 softmax（归一化后再填回 0）
        let mx = -Infinity;
        for (let j = 0; j < n; j++) if (keep[j] !== 0 && s[i * n + j] > mx) mx = s[i * n + j];
        let z = 0;
        for (let j = 0; j < n; j++) if (keep[j] !== 0) z += Math.exp(s[i * n + j] - mx);
        for (let j = 0; j < n; j++) if (keep[j] !== 0 && !rc(P[i * n + j], Math.exp(s[i * n + j] - mx) / z, 1e-12)) return false;
      }
      return true;
    },
  });
  reg(24, {
    name: '掩码 · 可见位置逐位不变', intuition: '掩码只该"屏蔽"，绝不能"篡改"可见位置的打分。',
    math: '∀ j≤i：causal(s)[i][j] = s[i][j]', pseudocode: 'out = causal(s); assert out[visible] === s[visible]',
    complexity: 'O(n²)', ref: '掩码语义契约',
    commonPitfall: '用浮点大负数当掩码时，"加 0" 仍可能因舍入改变原值——−∞ 语义可避免。',
    invariant: '下三角（含对角）Object.is(run, s)',
    inputs: ['s'], domains: { s: 'matSmall' }, matrixDims: { s: [4, 4] },
    run: (s) => causal(s, sq(s)),
    predicate: (s) => {
      const n = sq(s), o = runOf('D024')(s);
      for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) if (!Object.is(o[i * n + j], s[i * n + j])) return false;
      return true;
    },
  });

  // ══ softmax 与 log-sum-exp 归约（D025–D032）═════════════════════════
  reg(25, {
    name: 'softmax · 行和为一', intuition: '归一化后每行恰为概率分布。',
    math: 'Σⱼ pⱼ = 1', pseudocode: 'return softmax(s)', complexity: 'O(n)',
    ref: 'Bridle 1990 (softmax)',
    commonPitfall: '累加顺序影响末位；用 1e-12 相对容差。',
    invariant: 'relClose(Σrun, 1, 1e-12)；全 1 输入恰为均匀 1/n',
    inputs: ['s'], domains: { s: 'vecSmall' },
    run: (s) => softmaxRow(s, 0, s.length),
    predicate: (s) => {
      const p = runOf('D025')(s);
      let sum = 0;
      for (let i = 0; i < p.length; i++) sum += p[i];
      if (!rc(sum, 1, 1e-12)) return false;
      const u = new Float64Array(s.length).fill(1);    // 定点锚：全 1 → 均匀分布
      const pu = runOf('D025')(u);
      return rc(pu[0], 1 / s.length, 1e-15);
    },
  });
  reg(26, {
    name: 'softmax · 平移不变性', intuition: '给整行加常数不改变分布——这是减最大值防溢出的理论依据。',
    math: 'softmax(s + c·1) = softmax(s)', pseudocode: 'return softmax(shift(s,c))',
    complexity: 'O(n)', ref: 'softmax 平移不变性',
    commonPitfall: 'c 过大会让 s+c 丢失 s 的低位信息（吸收），故 c 限 small 域。',
    invariant: 'relClose(run(s,c), D025(s), 1e-12)',
    inputs: ['s', 'c'], domains: { s: 'vecSmall', c: 'small' },
    run: (s, c) => { const t = new Float64Array(s.length); for (let i = 0; i < s.length; i++) t[i] = s[i] + c; return softmaxRow(t, 0, s.length); },
    predicate: (s, c) => rc(runOf('D026')(s, c), runOf('D025')(s), 1e-12),
  });
  reg(27, {
    name: 'softmax · 正值性', intuition: '指数函数恒正，softmax 输出无零无负。',
    math: '∀ j：pⱼ > 0', pseudocode: 'return softmax(s)', complexity: 'O(n)',
    ref: 'Bridle 1990', commonPitfall: '下溢到 0 是可能的（|s| 巨大时），故域限 small。',
    invariant: 'every(run > 0)',
    inputs: ['s'], domains: { s: 'vecSmall' },
    run: (s) => softmaxRow(s, 0, s.length),
    predicate: (s) => { const p = runOf('D027')(s); for (let i = 0; i < p.length; i++) if (!(p[i] > 0)) return false; return true; },
  });
  reg(28, {
    name: 'softmax · 安全版与朴素版一致', intuition: '减最大值的写法必须与直接 exp/sum 给出同一结果。',
    math: 'exp(s−m)/Σexp(s−m) = exp(s)/Σexp(s)',
    pseudocode: 'a = safeSoftmax(s); b = naiveSoftmax(s); return relClose(a, b, 1e-9)',
    complexity: 'O(n)', ref: '数值稳定性（log-sum-exp trick）',
    commonPitfall: '朴素版在大打分下会溢出为 Inf/NaN——域限 small 保证朴素版可用作参考。',
    invariant: 'relClose(run, naive, 1e-9)',
    inputs: ['s'], domains: { s: 'vecSmall' },
    run: (s) => softmaxRow(s, 0, s.length),
    predicate: (s) => {
      const n = s.length, p = runOf('D028')(s), naive = new Float64Array(n);
      let sum = 0;
      for (let i = 0; i < n; i++) { naive[i] = Math.exp(s[i]); sum += naive[i]; }
      for (let i = 0; i < n; i++) naive[i] /= sum;
      return rc(p, naive, 1e-9);
    },
  });
  reg(29, {
    name: 'softmax · 保序性', intuition: '打分大小顺序在概率上保持（严格单调）。',
    math: 'sᵢ < sⱼ ⇒ pᵢ < pⱼ', pseudocode: 'return softmax(s)',
    complexity: 'O(n²) 校验', ref: 'softmax 严格单调',
    commonPitfall: '极接近的打分在浮点下可能并列，故用 ≤ 判据更稳（不变量表述为不逆序）。',
    invariant: 'sᵢ < sⱼ ⇒ runᵢ ≤ runⱼ',
    inputs: ['s'], domains: { s: 'vecSmall' },
    run: (s) => softmaxRow(s, 0, s.length),
    predicate: (s) => {
      const n = s.length, p = runOf('D029')(s);
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (s[i] < s[j] && p[i] > p[j]) return false;
      return true;
    },
  });
  reg(30, {
    name: 'log-sum-exp · 与 log Σexp 一致', intuition: 'lse 的减最大值写法必须与朴素 log(Σexp) 同值。',
    math: 'lse(s) = m + log Σ exp(sₖ − m) = log Σ exp(sₖ)',
    pseudocode: 'return lse(s)', complexity: 'O(n)',
    ref: 'log-sum-exp trick', commonPitfall: '朴素版会溢出，域限 small 使其可用作参考。',
    invariant: 'relClose(run, log Σexp, 1e-12)',
    inputs: ['s'], domains: { s: 'vecSmall' },
    run: (s) => lse(s, 0, s.length),
    predicate: (s) => { let sum = 0; for (let i = 0; i < s.length; i++) sum += Math.exp(s[i]); return rc(runOf('D030')(s), Math.log(sum), 1e-12); },
  });
  reg(31, {
    name: 'log-sum-exp · 平移公式', intuition: '整行加常数 c，lse 也恰好加 c（softmax 平移不变的对数形式）。',
    math: 'lse(s + c·1) = lse(s) + c', pseudocode: 'return lse(shift(s,c))',
    complexity: 'O(n)', ref: 'log-sum-exp 平移性质',
    commonPitfall: 'c 极大时低位被吸收，容差需 1e-9 量级。',
    invariant: 'relClose(run(s,c), D030(s)+c, 1e-9)',
    inputs: ['s', 'c'], domains: { s: 'vecSmall', c: 'small' },
    run: (s, c) => { const t = new Float64Array(s.length); for (let i = 0; i < s.length; i++) t[i] = s[i] + c; return lse(t, 0, s.length); },
    predicate: (s, c) => rc(runOf('D031')(s, c), runOf('D030')(s) + c, 1e-9),
  });
  reg(32, {
    name: 'softmax · 等于 exp(s − lse(s))', intuition: 'softmax 可完全由 log-sum-exp 表出，是 flash attention 合并两块的代数基础。',
    math: 'pⱼ = exp(sⱼ − lse(s))', pseudocode: 'L = lse(s); for j: p[j] = exp(s[j]-L)',
    complexity: 'O(n)', ref: 'Milakov & Gimelshein 2018 · Dao et al. 2022 (FlashAttention)',
    commonPitfall: '两条路径都要减最大值，否则 exp 溢出；误差量级 ~1e-16 相对。',
    invariant: '逐元素 relClose(run[j], exp(s[j]−D030(s)), 1e-12)',
    inputs: ['s'], domains: { s: 'vecSmall' },
    run: (s) => softmaxRow(s, 0, s.length),
    predicate: (s) => {
      const n = s.length, L = runOf('D030')(s), p = runOf('D032')(s), ref = new Float64Array(n);
      for (let j = 0; j < n; j++) ref[j] = Math.exp(s[j] - L);
      return rc(p, ref, 1e-12);
    },
  });
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

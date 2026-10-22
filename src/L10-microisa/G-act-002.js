/* ============================================================================
 * L10 微指令 · G 域（激活与损失）· 第 3/3 批（G065–G096）
 * ----------------------------------------------------------------------------
 * 所属层：L10-microIsa  域：G（激活与损失）
 * 允许依赖的层：L9（special.sigmoid / logsumexp2 / softmaxStable）
 * 主要 @ref：Goodfellow 2016（损失函数）；Lin et al. 2017（Focal）；
 *           Girshick 2015（SmoothL1）；Huber 1964；Kullback & Leibler 1951
 *
 * 子域划分：
 *   G065–G072 回归损失：MSE / MAE
 *   G073–G078 鲁棒损失：Huber / SmoothL1
 *   G079–G085 二分类：BCE / BCE-with-logits（数值稳定）
 *   G086–G092 多分类：CE / CE-with-logits（与 softmax 一致）
 *   G093–G095 分布散度：Kullback-Leibler
 *   G096 聚焦损失：Focal（BCE 的难例加权泛化）
 *
 * 谓词纪律：门禁只调 op.predicate(...)，从不调 run（tests/lib/op-harness.js:43）。
 * 故每条 predicate 必须先调用 NS.L10.ops['Gxxx'].run(...) 取真实产出，先与「反向
 * 累加 + 各自独立公式」的参考路径比对（基线），再验证该算子声明的性质。性质本身
 * 需要特定结构的输入（one-hot、零误差、固定 γ）时，一律把构造好的输入喂给 run，
 * 禁止脱离 run 直接调本文件的 f_* helper。
 * 原 32 项谓词几乎全部在硬编码定点上自比或直接调 helper（恒真假绿），已全部重写。
 * 文档纠错：G088「p=y ⇒ CE=0」仅对 one-hot 成立（一般分布为 H(y)）；
 * G089「CE ≤ lnK」对任意 p 不成立（p_t→0 时无界），已改为 p=y 时的熵界 H(y) ≤ ln n；
 * G096 的 γ 提升为输入参数，γ=0 退化才可通过 run 验证。
 * 概率类 run 内部对 p/q 做 1e-300 下界投影以防 -ln0；参考路径用 max(p,1e-300) 夹取。
 * 不变量判据：数值类用 relClose（精确 1e-12，分布类 1e-9）。
 * ==========================================================================*/
;(function (NS) {
  'use strict';
  NS.L10 = NS.L10 || {};
  const rc = NS.L10.relClose;
  const SP = NS.L9.special;
  const log = Math.log, exp = Math.exp, abs = Math.abs;
  const log1p = Math.log1p;
  function reg(i, o) { NS.L10.register('G', i, o); }
  function runOf(key) { return NS.L10.ops[key].run; }

  const PFLOOR = 1e-300;               // 概率下界（防 -ln0）
  const clip1 = (v) => Math.max(v, PFLOOR);

  // 内部工具（本文件自包含）
  function clipProb(v) { const o = new Float64Array(v.length); for (let i = 0; i < v.length; i++) o[i] = clip1(v[i]); return o; }
  function mse(y, p) { let s = 0; for (let i = 0; i < y.length; i++) { const e = y[i] - p[i]; s += e * e; } return s / y.length; }
  function mae(y, p) { let s = 0; for (let i = 0; i < y.length; i++) s += abs(y[i] - p[i]); return s / y.length; }
  function huber(y, p, d) { let s = 0; for (let i = 0; i < y.length; i++) { const e = abs(y[i] - p[i]); s += e <= d ? 0.5 * e * e : d * (e - 0.5 * d); } return s / y.length; }
  function smoothL1(y, p) { let s = 0; for (let i = 0; i < y.length; i++) { const e = abs(y[i] - p[i]); s += e < 1 ? 0.5 * e * e : e - 0.5; } return s / y.length; }
  function bce(y, p) { const pc = clipProb(p); let s = 0; for (let i = 0; i < y.length; i++) { const yi = y[i], pi = pc[i]; if (yi > 1e-15) s += -yi * log(pi); if (yi < 1 - 1e-15) s += -(1 - yi) * log(1 - pi); } return s / y.length; }
  function bceLogits(z, y) { let s = 0; for (let i = 0; i < z.length; i++) { const zi = z[i]; s += Math.max(zi, 0) - zi * y[i] + log(1 + exp(-abs(zi))); } return s / y.length; }
  function ce(y, p) { const pc = clipProb(p); let s = 0; for (let i = 0; i < y.length; i++) { const yi = y[i]; if (yi > 1e-15) s += -yi * log(pc[i]); } return s / y.length; }
  function ceLogits(z, y) {
    let m = -Infinity; for (let i = 0; i < z.length; i++) m = Math.max(m, z[i]);
    let acc = 0; for (let i = 0; i < z.length; i++) acc += exp(z[i] - m);
    const lse = m + log(acc);
    let s = 0; for (let i = 0; i < z.length; i++) s += -y[i] * (z[i] - lse); return s / y.length;
  }
  function kl(p, q) { const qc = clipProb(q); let s = 0; for (let i = 0; i < p.length; i++) { const pi = p[i]; if (pi > 0) s += pi * (log(pi) - log(qc[i])); } return s; }
  function focal(y, p, g) { const pc = clipProb(p); let s = 0; for (let i = 0; i < y.length; i++) { const pt = y[i] * pc[i] + (1 - y[i]) * (1 - pc[i]); s += -Math.pow(1 - pt, g) * log(clip1(pt)); } return s / y.length; }

  // ── 独立参考路径：一律反向累加，且用与 run 不同的表达形式 ──
  function mseRef(y, p) { let s = 0; for (let i = y.length - 1; i >= 0; i--) { const e = y[i] - p[i]; s += e * e; } return s / y.length; }
  function maeRef(y, p) { let s = 0; for (let i = y.length - 1; i >= 0; i--) s += abs(y[i] - p[i]); return s / y.length; }
  function huberRef(y, p, d) { let s = 0; for (let i = y.length - 1; i >= 0; i--) { const e = abs(y[i] - p[i]); s += e <= d ? 0.5 * e * e : d * (e - 0.5 * d); } return s / y.length; }
  function smoothL1Ref(y, p) { let s = 0; for (let i = y.length - 1; i >= 0; i--) { const e = abs(y[i] - p[i]); s += e < 1 ? 0.5 * e * e : e - 0.5; } return s / y.length; }
  // BCE 参考：夹取式通项（对照 run 的逐项守卫）
  function bceRef(y, p) { let s = 0; for (let i = y.length - 1; i >= 0; i--) s += -(y[i] * log(clip1(p[i])) + (1 - y[i]) * log(clip1(1 - p[i]))); return s / y.length; }
  // BCE-logits 参考：log1p 式（对照 run 的 log(1+e^{-|z|})）
  function bceLogitsRef(z, y) { let s = 0; for (let i = z.length - 1; i >= 0; i--) { const zi = z[i]; s += Math.max(zi, 0) - zi * y[i] + log1p(exp(-abs(zi))); } return s / y.length; }
  function ceRef(y, p) { let s = 0; for (let i = y.length - 1; i >= 0; i--) s += -y[i] * log(clip1(p[i])); return s / y.length; }
  // CE-logits 参考：直接式 lse=lnΣe^z（对照 run 的 max 平移式），并按 lse−z 排列
  function ceLogitsRef(z, y) {
    let acc = 0; for (let i = z.length - 1; i >= 0; i--) acc += exp(z[i]);
    const lse = log(acc);
    let s = 0; for (let i = z.length - 1; i >= 0; i--) s += y[i] * (lse - z[i]);
    return s / y.length;
  }
  function klRef(p, q) { let s = 0; for (let i = p.length - 1; i >= 0; i--) if (p[i] > 0) s += p[i] * (log(p[i]) - log(clip1(q[i]))); return s; }
  function focalRef(y, p, g) { let s = 0; for (let i = y.length - 1; i >= 0; i--) { const pt = y[i] * p[i] + (1 - y[i]) * (1 - p[i]); s += -Math.pow(1 - pt, g) * log(clip1(pt)); } return s / y.length; }

  // ══ 回归损失（G065–G072）═════════════════════════════════════════════
  reg(65, {
    name: 'MSE · 定义', intuition: '均方误差，回归最常用，对大误差平方惩罚。',
    math: 'MSE(y,p) = (1/n)Σ(y_i-p_i)²', pseudocode: 's=0;for:e=y-p;s+=e*e;return s/n',
    complexity: 'O(n)', ref: 'Goodfellow 2016 §6.2.1', commonPitfall: '漏除 n 是典型 bug，反向累加交叉可检出。',
    invariant: 'relClose(run(y,p), 反向累加 Σ(y-p)²/n, 1e-12)',
    inputs: ['y', 'p'], domains: { y: 'vecSmall', p: 'vecSmall' },
    run: (y, p) => mse(y, p),
    predicate: (y, p) => rc(runOf('G065')(y, p), mseRef(y, p), 1e-12),
  });
  reg(66, {
    name: 'MSE · 非负', intuition: '平方和非负，MSE ≥ 0。',
    math: 'MSE ≥ 0', pseudocode: 'return mse(y,p) >= -1e-12',
    complexity: 'O(n)', ref: '非负', commonPitfall: '浮点下溢近 0。',
    invariant: 'run(y,p) ≥ -1e-12 且 run 与反向累加参考一致（仅查非负会被恒 0 蒙混）',
    inputs: ['y', 'p'], domains: { y: 'vecSmall', p: 'vecSmall' },
    run: (y, p) => mse(y, p),
    predicate: (y, p) => {
      const r = runOf('G066')(y, p);
      return r >= -1e-12 && rc(r, mseRef(y, p), 1e-12);
    },
  });
  reg(67, {
    name: 'MSE · 对称', intuition: '误差对称，(y-p)²=(p-y)²。',
    math: 'MSE(y,p) = MSE(p,y)', pseudocode: 'return rc(mse(y,p), mse(p,y))',
    complexity: 'O(n)', ref: '对称', commonPitfall: '无。',
    invariant: 'relClose(run(y,p), run(p,y), 1e-12)（两路均经 run，参数交换）',
    inputs: ['y', 'p'], domains: { y: 'vecSmall', p: 'vecSmall' },
    run: (y, p) => mse(y, p),
    predicate: (y, p) => rc(runOf('G067')(y, p), runOf('G067')(p, y), 1e-12),
  });
  reg(68, {
    name: 'MSE · 零损', intuition: '预测与真值相等时损失为 0。',
    math: 'y = p ⇒ MSE = 0', pseudocode: 'return rc(mse(y,y), 0)',
    complexity: 'O(n)', ref: '定点', commonPitfall: '无。',
    invariant: 'relClose(run(y,y), 0, 1e-12)（同一向量经 run 求值）',
    inputs: ['y', 'p'], domains: { y: 'vecSmall', p: 'vecSmall' },
    run: (y, p) => mse(y, p),
    predicate: (y, p) => {
      const r = runOf('G068')(y, y);
      if (!rc(r, 0, 1e-12)) return false;
      return rc(runOf('G068')(y, p), mseRef(y, p), 1e-12);
    },
  });
  reg(69, {
    name: 'MAE · 定义', intuition: '平均绝对误差，对离群更鲁棒。',
    math: 'MAE(y,p) = (1/n)Σ|y_i-p_i|', pseudocode: 's=0;for:e=abs(y-p);s+=e;return s/n',
    complexity: 'O(n)', ref: '稳健回归', commonPitfall: '反向累加交叉验证。',
    invariant: 'relClose(run(y,p), 反向累加 Σ|y-p|/n, 1e-12)',
    inputs: ['y', 'p'], domains: { y: 'vecSmall', p: 'vecSmall' },
    run: (y, p) => mae(y, p),
    predicate: (y, p) => rc(runOf('G069')(y, p), maeRef(y, p), 1e-12),
  });
  reg(70, {
    name: 'MAE · 非负', intuition: '绝对值和非负。',
    math: 'MAE ≥ 0', pseudocode: 'return mae(y,p) >= -1e-12',
    complexity: 'O(n)', ref: '非负', commonPitfall: '无。',
    invariant: 'run(y,p) ≥ -1e-12 且 run 与反向累加参考一致（仅查非负会被恒 0 蒙混）',
    inputs: ['y', 'p'], domains: { y: 'vecSmall', p: 'vecSmall' },
    run: (y, p) => mae(y, p),
    predicate: (y, p) => {
      const r = runOf('G070')(y, p);
      return r >= -1e-12 && rc(r, maeRef(y, p), 1e-12);
    },
  });
  reg(71, {
    name: 'MAE · 零损', intuition: '预测等于真值时损失为 0。',
    math: 'y = p ⇒ MAE = 0', pseudocode: 'return rc(mae(y,y),0)',
    complexity: 'O(n)', ref: '定点', commonPitfall: '无。',
    invariant: 'relClose(run(y,y), 0, 1e-12)（同一向量经 run 求值）',
    inputs: ['y', 'p'], domains: { y: 'vecSmall', p: 'vecSmall' },
    run: (y, p) => mae(y, p),
    predicate: (y, p) => {
      const r = runOf('G071')(y, y);
      if (!rc(r, 0, 1e-12)) return false;
      return rc(runOf('G071')(y, p), maeRef(y, p), 1e-12);
    },
  });
  reg(72, {
    name: 'MAE · 上界', intuition: 'MAE ≤ 最大绝对误差（平均不超过最大项）。',
    math: 'MAE(y,p) ≤ max_i |y_i-p_i|', pseudocode: 'return mae(y,p) <= maxErr+1e-12',
    complexity: 'O(n)', ref: '上界', commonPitfall: '无。',
    invariant: 'run(y,p) ≤ max_i|y_i-p_i| + 1e-12 且 run 与反向累加参考一致',
    inputs: ['y', 'p'], domains: { y: 'vecSmall', p: 'vecSmall' },
    run: (y, p) => mae(y, p),
    predicate: (y, p) => {
      const r = runOf('G072')(y, p);
      if (!rc(r, maeRef(y, p), 1e-12)) return false;
      let mx = 0; for (let i = y.length - 1; i >= 0; i--) mx = Math.max(mx, abs(y[i] - p[i]));
      return r <= mx + 1e-12;
    },
  });

  // ══ 鲁棒损失（G073–G078）═════════════════════════════════════════════
  reg(73, {
    name: 'Huber · 定义', intuition: '误差小用平方、大用线性，结合 MSE/MAE 优点（δ=1）。',
    math: 'L_δ(e)= e≤δ?½e² : δ(e-½δ)', pseudocode: '分段累积 /n',
    complexity: 'O(n)', ref: 'Huber 1964', commonPitfall: '断点 e=δ 连续（½δ²=δ(δ-½δ)）。',
    invariant: 'relClose(run(y,p), 反向累加分段式/n, 1e-12)（原为硬编码定点自比，已修）',
    inputs: ['y', 'p'], domains: { y: 'vecSmall', p: 'vecSmall' },
    run: (y, p) => huber(y, p, 1),
    predicate: (y, p) => rc(runOf('G073')(y, p), huberRef(y, p, 1), 1e-12),
  });
  reg(74, {
    name: 'Huber · 小误差二次', intuition: '|e|≤δ 时退化为 ½e²（同 MSE）。',
    math: '|e|≤1 ⇒ L_δ(e)=½e²', pseudocode: 'return rc(huber([e],[0]), 0.5*e*e)',
    complexity: 'O(1)', ref: '平滑区', commonPitfall: '须构造 |e|≤1 的输入喂给 run，而非直接调 helper。',
    invariant: 'run 与反向累加参考一致；且构造 |e|=0.4 输入时 run = ½e²',
    inputs: ['y', 'p'], domains: { y: 'vecSmall', p: 'vecSmall' },
    run: (y, p) => huber(y, p, 1),
    predicate: (y, p) => {
      if (!rc(runOf('G074')(y, p), huberRef(y, p, 1), 1e-12)) return false;
      return rc(runOf('G074')([0.4], [0]), 0.5 * 0.4 * 0.4, 1e-12);
    },
  });
  reg(75, {
    name: 'Huber · 大误差线性', intuition: '|e|>δ 时线性增长，抑制离群。',
    math: '|e|>1 ⇒ L_δ(e)=|e|-½', pseudocode: 'return rc(huber([e],[0]), |e|-0.5)',
    complexity: 'O(1)', ref: '线性区', commonPitfall: '须构造 |e|>1 的输入喂给 run。',
    invariant: 'run 与反向累加参考一致；且构造 |e|=2.5 输入时 run = |e|−½',
    inputs: ['y', 'p'], domains: { y: 'vecSmall', p: 'vecSmall' },
    run: (y, p) => huber(y, p, 1),
    predicate: (y, p) => {
      if (!rc(runOf('G075')(y, p), huberRef(y, p, 1), 1e-12)) return false;
      return rc(runOf('G075')([2.5], [0]), 2.5 - 0.5, 1e-12);
    },
  });
  reg(76, {
    name: 'SmoothL1 · 定义', intuition: 'Faster R-CNN 用：小误差 ½e²、大误差 |e|-½。',
    math: 'L(e)= |e|<1?½e² : |e|-½', pseudocode: '分段累积 /n',
    complexity: 'O(n)', ref: 'Girshick 2015', commonPitfall: '断点 e=1 连续（½·1=1-½）。',
    invariant: 'relClose(run(y,p), 反向累加分段式/n, 1e-12)（原为硬编码定点自比，已修）',
    inputs: ['y', 'p'], domains: { y: 'vecSmall', p: 'vecSmall' },
    run: (y, p) => smoothL1(y, p),
    predicate: (y, p) => rc(runOf('G076')(y, p), smoothL1Ref(y, p), 1e-12),
  });
  reg(77, {
    name: 'SmoothL1 · 小误差二次', intuition: '|e|<1 时与 ½e² 一致。',
    math: '|e|<1 ⇒ L(e)=½e²', pseudocode: 'return rc(smoothL1([e],[0]), 0.5*e*e)',
    complexity: 'O(1)', ref: '平滑区', commonPitfall: '须构造 |e|<1 的输入喂给 run。',
    invariant: 'run 与反向累加参考一致；且构造 |e|=0.4 输入时 run = ½e²',
    inputs: ['y', 'p'], domains: { y: 'vecSmall', p: 'vecSmall' },
    run: (y, p) => smoothL1(y, p),
    predicate: (y, p) => {
      if (!rc(runOf('G077')(y, p), smoothL1Ref(y, p), 1e-12)) return false;
      return rc(runOf('G077')([0.4], [0]), 0.5 * 0.4 * 0.4, 1e-12);
    },
  });
  reg(78, {
    name: 'SmoothL1 · 大误差线性', intuition: '|e|≥1 时线性，无平方爆炸。',
    math: '|e|≥1 ⇒ L(e)=|e|-½', pseudocode: 'return rc(smoothL1([e],[0]), |e|-0.5)',
    complexity: 'O(1)', ref: '线性区', commonPitfall: '须构造 |e|≥1 的输入喂给 run。',
    invariant: 'run 与反向累加参考一致；且构造 |e|=2.5 输入时 run = |e|−½',
    inputs: ['y', 'p'], domains: { y: 'vecSmall', p: 'vecSmall' },
    run: (y, p) => smoothL1(y, p),
    predicate: (y, p) => {
      if (!rc(runOf('G078')(y, p), smoothL1Ref(y, p), 1e-12)) return false;
      return rc(runOf('G078')([2.5], [0]), 2.5 - 0.5, 1e-12);
    },
  });

  // ══ 二分类（G079–G085）══════════════════════════════════════════════
  reg(79, {
    name: 'BCE · 定义', intuition: '二元交叉熵，y∈{0,1} 时 = -[y ln p+(1-y)ln(1-p)]。',
    math: 'BCE = (1/n)Σ-[y ln p+(1-y)ln(1-p)]', pseudocode: '逐元素对数交叉熵 /n',
    complexity: 'O(n)', ref: 'Goodfellow 2016', commonPitfall: 'p∈(0,1) 投影防 -ln0；参考路径用夹取通项（对照 run 的逐项守卫）。',
    invariant: 'relClose(run(y,p), 反向累加夹取式/n, 1e-12)（原为硬编码定点自比，已修）',
    inputs: ['y', 'p'], domains: { y: 'vecMask', p: 'vecProb' },
    run: (y, p) => bce(y, p),
    predicate: (y, p) => rc(runOf('G079')(y, p), bceRef(y, p), 1e-12),
  });
  reg(80, {
    name: 'BCE · 非负', intuition: '负对数损失恒非负。',
    math: 'BCE ≥ 0', pseudocode: 'return bce(y,p) >= -1e-12',
    complexity: 'O(n)', ref: '非负', commonPitfall: 'p 投影后有限。',
    invariant: 'run(y,p) ≥ -1e-12 且 run 与反向累加参考一致（仅查非负会被恒 0 蒙混）',
    inputs: ['y', 'p'], domains: { y: 'vecMask', p: 'vecProb' },
    run: (y, p) => bce(y, p),
    predicate: (y, p) => {
      const r = runOf('G080')(y, p);
      return r >= -1e-12 && rc(r, bceRef(y, p), 1e-12);
    },
  });
  reg(81, {
    name: 'BCE · 零损', intuition: 'y=1 且 p=1（或 y=0 且 p=0）时损失为 0。',
    math: 'y=p∈{0,1} ⇒ BCE=0', pseudocode: 'return rc(bce([1],[1]),0) ∧ rc(bce([0],[0]),0)',
    complexity: 'O(1)', ref: '定点', commonPitfall: 'p=1 → ln1=0。',
    invariant: 'relClose(run([1],[1]), 0) ∧ relClose(run([0],[0]), 0)（定点经 run 求值）',
    inputs: ['y', 'p'], domains: { y: 'vecMask', p: 'vecProb' },
    run: (y, p) => bce(y, p),
    predicate: (y, p) => {
      if (!rc(runOf('G081')([1], [1]), 0, 1e-12)) return false;
      if (!rc(runOf('G081')([0], [0]), 0, 1e-12)) return false;
      return rc(runOf('G081')(y, p), bceRef(y, p), 1e-12);
    },
  });
  reg(82, {
    name: 'BCE · 单调性', intuition: 'y=1 时 p 越大损失越小。',
    math: 'y=1, p₁<p₂ ⇒ BCE(p₁)>BCE(p₂)', pseudocode: 'return bce([1],[.3]) > bce([1],[.9])',
    complexity: 'O(1)', ref: '单调', commonPitfall: '不取等号。',
    invariant: 'run([1],[0.3]) > run([1],[0.9]) + 1e-12（定点经 run 求值）',
    inputs: ['y', 'p'], domains: { y: 'vecMask', p: 'vecProb' },
    run: (y, p) => bce(y, p),
    predicate: (y, p) => {
      if (!rc(runOf('G082')(y, p), bceRef(y, p), 1e-12)) return false;
      return runOf('G082')([1], [0.3]) > runOf('G082')([1], [0.9]) + 1e-12;
    },
  });
  reg(83, {
    name: 'BCE-with-logits · 定义', intuition: '用 logit z 直接算，数值稳定免 sigmoid 溢出。',
    math: 'L = max(z,0)-z·y+ln(1+e^{-|z|})', pseudocode: '逐元素稳定项 /n',
    complexity: 'O(n)', ref: '数值稳定', commonPitfall: '参考路径用 log1p(e^{-|z|})（对照 run 的 log(1+…)）。',
    invariant: 'relClose(run(z,y), 反向累加 log1p 式/n, 1e-12)（原为硬编码定点自比，已修）',
    inputs: ['z', 'y'], domains: { z: 'vecSmall', y: 'vecMask' },
    run: (z, y) => bceLogits(z, y),
    predicate: (z, y) => rc(runOf('G083')(z, y), bceLogitsRef(z, y), 1e-12),
  });
  reg(84, {
    name: 'BCE-logits · 等价 BCE(σ(z))', intuition: '稳定式与先 sigmoid 再 BCE 数学相等。',
    math: 'BCE-logits(z) = BCE(y, σ(z))', pseudocode: 'return rc(bceLogits(z,y), bce(y,sig(z)))',
    complexity: 'O(n)', ref: '等价恒等式', commonPitfall: '两路径交叉（logits 稳定式 vs σ 后 BCE）。',
    invariant: 'relClose(run(z,y), G079 的 run(y,σ(z)), 1e-9)（跨算子经 run 交叉）',
    inputs: ['z', 'y'], domains: { z: 'vecSmall', y: 'vecMask' },
    run: (z, y) => bceLogits(z, y),
    predicate: (z, y) => {
      const r = runOf('G084')(z, y);
      if (!rc(r, bceLogitsRef(z, y), 1e-12)) return false;
      const P = new Float64Array(z.length);
      for (let i = 0; i < z.length; i++) P[i] = SP.sigmoid(z[i]);
      return rc(r, runOf('G079')(y, P), 1e-9);
    },
  });
  reg(85, {
    name: 'BCE-logits · 非负', intuition: '负对数损失恒非负。',
    math: 'BCE-logits ≥ 0', pseudocode: 'return bceLogits(z,y) >= -1e-12',
    complexity: 'O(n)', ref: '非负', commonPitfall: '无。',
    invariant: 'run(z,y) ≥ -1e-12 且 run 与 log1p 式参考一致（仅查非负会被恒 0 蒙混）',
    inputs: ['z', 'y'], domains: { z: 'vecSmall', y: 'vecMask' },
    run: (z, y) => bceLogits(z, y),
    predicate: (z, y) => {
      const r = runOf('G085')(z, y);
      return r >= -1e-12 && rc(r, bceLogitsRef(z, y), 1e-12);
    },
  });

  // ══ 多分类（G086–G092）══════════════════════════════════════════════
  reg(86, {
    name: 'CE · 多类定义', intuition: '交叉熵，y 为类别分布、p 为预测分布。',
    math: 'CE = -(1/n)Σ y_i ln p_i', pseudocode: '逐元素 -y*ln(p_clip) 累和 /n',
    complexity: 'O(n)', ref: 'Goodfellow 2016', commonPitfall: 'p 投影防 -ln0；参考路径用夹取通项（对照 run 的守卫）。',
    invariant: 'relClose(run(y,p), 反向累加 -(y ln p̂)/n, 1e-12)（原为硬编码定点自比，已修）',
    inputs: ['y', 'p'], domains: { y: 'vecProb', p: 'vecProb' },
    run: (y, p) => ce(y, p),
    predicate: (y, p) => rc(runOf('G086')(y, p), ceRef(y, p), 1e-12),
  });
  reg(87, {
    name: 'CE · 非负', intuition: '负对数期望恒非负。',
    math: 'CE ≥ 0', pseudocode: 'return ce(y,p) >= -1e-12',
    complexity: 'O(n)', ref: '非负', commonPitfall: 'p 投影后有限。',
    invariant: 'run(y,p) ≥ -1e-12 且 run 与反向累加参考一致（仅查非负会被恒 0 蒙混）',
    inputs: ['y', 'p'], domains: { y: 'vecProb', p: 'vecProb' },
    run: (y, p) => ce(y, p),
    predicate: (y, p) => {
      const r = runOf('G087')(y, p);
      return r >= -1e-12 && rc(r, ceRef(y, p), 1e-12);
    },
  });
  reg(88, {
    name: 'CE · 零损（one-hot）', intuition: 'one-hot 目标分布下预测完全一致时损失为 0（一般分布的 CE(y,y)=H(y)≠0）。',
    math: 'y one-hot ⇒ CE(y,y)=0；一般 y: CE(y,y)=H(y)/n', pseudocode: 'return rc(ce([1,0],[1,0]), 0)',
    complexity: 'O(n)', ref: '定点', commonPitfall: '原文档写「p=y ⇒ CE=0」仅对 one-hot 成立，一般分布得熵 H(y)，已改正。',
    invariant: 'relClose(run([1,0],[1,0]), 0, 1e-12) ∧ relClose(run([0,1],[0,1]), 0, 1e-12)（定点经 run 求值）',
    inputs: ['y', 'p'], domains: { y: 'vecProb', p: 'vecProb' },
    run: (y, p) => ce(y, p),
    predicate: (y, p) => {
      if (!rc(runOf('G088')([1, 0], [1, 0]), 0, 1e-12)) return false;
      if (!rc(runOf('G088')([0, 1], [0, 1]), 0, 1e-12)) return false;
      return rc(runOf('G088')(y, p), ceRef(y, p), 1e-12);
    },
  });
  reg(89, {
    name: 'CE · 熵上界（p=y）', intuition: 'p=y 时 CE = H(y)/n，而离散熵 H(y) ≤ ln n。',
    math: 'CE(y,y) = H(y)/n ≤ ln n / n', pseudocode: 'return rc(ce(y,y), H/n) ∧ ce(y,y) <= ln(n)/n',
    complexity: 'O(n)', ref: '熵上界', commonPitfall: '原文档写「CE ≤ lnK」对任意 p 不成立（p_t→0 时 CE 无上界），已改为 p=y 的熵界。',
    invariant: 'relClose(run(y,y), H(y)/n, 1e-12) ∧ run(y,y) ≤ ln(n)/n + 1e-12',
    inputs: ['y', 'p'], domains: { y: 'vecProb', p: 'vecProb' },
    run: (y, p) => ce(y, p),
    predicate: (y, p) => {
      const n = y.length;
      const r = runOf('G089')(y, y);
      if (!rc(r, ceRef(y, y), 1e-12)) return false;
      return r <= log(n) / n + 1e-12;
    },
  });
  reg(90, {
    name: 'CE-logits · 定义', intuition: 'logits 直接算 CE，免去 softmax 显式归一。',
    math: 'CE = (1/n)Σ y_i(lse - z_i), lse = lnΣe^{z_j}', pseudocode: 'lse=log(Σexp(z)); return mean(y*(lse-z))',
    complexity: 'O(n)', ref: '数值稳定', commonPitfall: '参考路径用直接式 lse（对照 run 的 max 平移式）。',
    invariant: 'relClose(run(z,y), 反向累加直接式/n, 1e-12)（原为硬编码定点自比，已修）',
    inputs: ['z', 'y'], domains: { z: 'vecSmall', y: 'vecProb' },
    run: (z, y) => ceLogits(z, y),
    predicate: (z, y) => rc(runOf('G090')(z, y), ceLogitsRef(z, y), 1e-12),
  });
  reg(91, {
    name: 'CE-logits · 平移不变', intuition: 'logits 整体加常数 c：lse 增 c 与 -z 项抵消，CE 不变。',
    math: 'CE-logits(z+c,y) = CE-logits(z,y)', pseudocode: 'return rc(ceLogits(z+c,y), ceLogits(z,y))',
    complexity: 'O(n)', ref: '平移协变', commonPitfall: '避免走 softmax 路径（分量下溢 log(0) 致 NaN）；用平移不变解析验证。',
    invariant: 'relClose(run(z+c,y), run(z,y), 1e-9)（两路均经 run，取 c=3）',
    inputs: ['z', 'y'], domains: { z: 'vecSmall', y: 'vecProb' },
    run: (z, y) => ceLogits(z, y),
    predicate: (z, y) => {
      const r = runOf('G091')(z, y);
      if (!rc(r, ceLogitsRef(z, y), 1e-12)) return false;
      const zc = new Float64Array(z.length);
      for (let i = 0; i < z.length; i++) zc[i] = z[i] + 3;
      return rc(runOf('G091')(zc, y), r, 1e-9);
    },
  });
  reg(92, {
    name: 'CE-logits · 非负', intuition: 'lse ≥ max z ≥ Σy·z（凸组合不超最大值），故 CE ≥ 0。',
    math: 'CE-logits ≥ 0', pseudocode: 'return ceLogits(z,y) >= -1e-12',
    complexity: 'O(n)', ref: '非负', commonPitfall: '无。',
    invariant: 'run(z,y) ≥ -1e-12 且 run 与直接式参考一致（仅查非负会被恒 0 蒙混）',
    inputs: ['z', 'y'], domains: { z: 'vecSmall', y: 'vecProb' },
    run: (z, y) => ceLogits(z, y),
    predicate: (z, y) => {
      const r = runOf('G092')(z, y);
      return r >= -1e-9 && rc(r, ceLogitsRef(z, y), 1e-12);
    },
  });

  // ══ KL 散度与 Focal（G093–G096）═════════════════════════════════════
  reg(93, {
    name: 'KL 散度 · 定义', intuition: 'Kullback-Leibler 散度，前向 p‖q。',
    math: 'KL(p‖q) = Σ p_i ln(p_i/q_i)', pseudocode: '逐元素 p*(ln p - ln q_clip) 反向累和',
    complexity: 'O(n)', ref: 'Kullback & Leibler 1951', commonPitfall: 'q 投影防 -ln0；参考路径反向累加。',
    invariant: 'relClose(run(p,q), 反向累加 Σ p_i ln(p_i/q̂_i), 1e-12)（原为硬编码定点自比，已修）',
    inputs: ['p', 'q'], domains: { p: 'vecProb', q: 'vecProb' },
    run: (p, q) => kl(p, q),
    predicate: (p, q) => rc(runOf('G093')(p, q), klRef(p, q), 1e-12),
  });
  reg(94, {
    name: 'KL 散度 · 非负', intuition: 'Gibbs 不等式：KL ≥ 0（p,q 均为归一分布）。',
    math: 'KL(p‖q) ≥ 0', pseudocode: 'return kl(p,q) >= -1e-12',
    complexity: 'O(n)', ref: 'Gibbs 不等式', commonPitfall: 'p=q 时恰为 0。',
    invariant: 'run(p,q) ≥ -1e-12 且 run 与反向累加参考一致（仅查非负会被恒 0 蒙混）',
    inputs: ['p', 'q'], domains: { p: 'vecProb', q: 'vecProb' },
    run: (p, q) => kl(p, q),
    predicate: (p, q) => {
      const r = runOf('G094')(p, q);
      return r >= -1e-12 && rc(r, klRef(p, q), 1e-12);
    },
  });
  reg(95, {
    name: 'KL 散度 · 零（同分布）', intuition: 'p=q 时 KL=0（每项 ln1=0）。',
    math: 'p = q ⇒ KL = 0', pseudocode: 'return rc(kl(p,p),0)',
    complexity: 'O(n)', ref: '定点', commonPitfall: 'p=q 时每项 0。',
    invariant: 'relClose(run(p,p), 0, 1e-12)（同一分布经 run 求值）',
    inputs: ['p', 'q'], domains: { p: 'vecProb', q: 'vecProb' },
    run: (p, q) => kl(p, q),
    predicate: (p, q) => {
      const r = runOf('G095')(p, p);
      if (!rc(r, 0, 1e-12)) return false;
      return rc(runOf('G095')(p, q), klRef(p, q), 1e-12);
    },
  });
  reg(96, {
    name: 'Focal · 定义与 γ=0 退化', intuition: 'FL=-(1-p_t)^γ ln p_t，难例加权；γ=0 时退化为 BCE。',
    math: 'FL_γ = -(1-p_t)^γ ln p_t；p_t = y·p+(1-y)(1-p)；FL_0 = BCE', pseudocode: 'pt=...; return mean(-(1-pt)^g ln pt)',
    complexity: 'O(n)', ref: 'Lin et al. 2017', commonPitfall: 'γ 提升为输入参数，否则 γ=0 退化无法通过 run 验证（原 γ 硬编码为 2）。',
    invariant: 'relClose(run(y,p,g), 反向累加 -(1-p_t)^g ln p_t, 1e-12) ∧ relClose(run(y,p,0), G079 的 run(y,p), 1e-12)',
    inputs: ['y', 'p', 'g'], domains: { y: 'vecMask', p: 'vecProb', g: 'nonNegSmall' },
    run: (y, p, g) => focal(y, p, g),
    predicate: (y, p, g) => {
      const r = runOf('G096')(y, p, g);
      if (!rc(r, focalRef(y, p, g), 1e-12)) return false;
      return rc(runOf('G096')(y, p, 0), runOf('G079')(y, p), 1e-12);
    },
  });
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

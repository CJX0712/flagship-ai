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
 * 不变量判据：数值类用 relClose（精确 1e-12，分布类 1e-9）。概率类 run 内部对
 * p/q 做 1e-300 下界投影以防 -ln0 数值爆炸；predicate 用硬编码定点（非零向量）
 * 做精确数学交叉 + 解析性质（非负 / 零损 / 平移 / 上界），避免随机边界与自证。
 * ==========================================================================*/
;(function (NS) {
  'use strict';
  NS.L10 = NS.L10 || {};
  const rc = NS.L10.relClose;
  const SP = NS.L9.special;
  const log = Math.log, exp = Math.exp, abs = Math.abs;
  function reg(i, o) { NS.L10.register('G', i, o); }

  // 内部工具（本文件自包含）
  function clipProb(v) { const o = new Float64Array(v.length); for (let i = 0; i < v.length; i++) o[i] = Math.max(v[i], 1e-300); return o; }
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
  function focal(y, p, g) { const pc = clipProb(p); let s = 0; for (let i = 0; i < y.length; i++) { const pt = y[i] * pc[i] + (1 - y[i]) * (1 - pc[i]); s += -Math.pow(1 - pt, g) * log(Math.max(pt, 1e-300)); } return s / y.length; }

  // ══ 回归损失（G065–G072）═════════════════════════════════════════════
  reg(65, {
    name: 'MSE · 定义', intuition: '均方误差，回归最常用，对大误差平方惩罚。',
    math: 'MSE(y,p) = (1/n)Σ(y_i-p_i)²', pseudocode: 's=0;for:e=y-p;s+=e*e;return s/n',
    complexity: 'O(n)', ref: 'Goodfellow 2016 §6.2.1', commonPitfall: 'rand 顺序不同路径交叉验证（防漏除 n）。',
    invariant: 'relClose(MSE(y,p), 反向累加 Σ(y-p)²/n, 1e-12)',
    inputs: ['y', 'p'], domains: { y: 'vecSmall', p: 'vecSmall' },
    run: (y, p) => mse(y, p),
    predicate: (y, p) => { let s = 0; for (let i = y.length - 1; i >= 0; i--) { const e = y[i] - p[i]; s += e * e; } return rc(mse(y, p), s / y.length, 1e-12); },
  });
  reg(66, {
    name: 'MSE · 非负', intuition: '平方和非负，MSE ≥ 0。',
    math: 'MSE ≥ 0', pseudocode: 'return mse(y,p) >= -1e-12',
    complexity: 'O(n)', ref: '非负', commonPitfall: '浮点下溢近 0。',
    invariant: '布尔：MSE ≥ 0',
    inputs: ['y', 'p'], domains: { y: 'vecSmall', p: 'vecSmall' },
    run: (y, p) => mse(y, p),
    predicate: (y, p) => mse(y, p) >= -1e-12,
  });
  reg(67, {
    name: 'MSE · 对称', intuition: '误差对称，(y-p)²=(p-y)²。',
    math: 'MSE(y,p) = MSE(p,y)', pseudocode: 'return rc(mse(y,p), mse(p,y))',
    complexity: 'O(n)', ref: '对称', commonPitfall: '无。',
    invariant: 'relClose(MSE(y,p), MSE(p,y), 1e-12)',
    inputs: ['y', 'p'], domains: { y: 'vecSmall', p: 'vecSmall' },
    run: (y, p) => mse(y, p),
    predicate: (y, p) => rc(mse(y, p), mse(p, y), 1e-12),
  });
  reg(68, {
    name: 'MSE · 零损', intuition: '预测与真值相等时损失为 0。',
    math: 'y = p ⇒ MSE = 0', pseudocode: 'return rc(mse(y,y), 0)',
    complexity: 'O(n)', ref: '定点', commonPitfall: '无。',
    invariant: 'relClose(MSE(y,y), 0, 1e-12)',
    inputs: ['y', 'p'], domains: { y: 'vecSmall', p: 'vecSmall' },
    run: (y, p) => mse(y, p),
    predicate: (y, p) => rc(mse(y, y), 0, 1e-12),
  });
  reg(69, {
    name: 'MAE · 定义', intuition: '平均绝对误差，对离群更鲁棒。',
    math: 'MAE(y,p) = (1/n)Σ|y_i-p_i|', pseudocode: 's=0;for:e=abs(y-p);s+=e;return s/n',
    complexity: 'O(n)', ref: '稳健回归', commonPitfall: '反向累加交叉验证。',
    invariant: 'relClose(MAE(y,p), 反向 Σ|y-p|/n, 1e-12)',
    inputs: ['y', 'p'], domains: { y: 'vecSmall', p: 'vecSmall' },
    run: (y, p) => mae(y, p),
    predicate: (y, p) => { let s = 0; for (let i = p.length - 1; i >= 0; i--) s += abs(y[i] - p[i]); return rc(mae(y, p), s / y.length, 1e-12); },
  });
  reg(70, {
    name: 'MAE · 非负', intuition: '绝对值和非负。',
    math: 'MAE ≥ 0', pseudocode: 'return mae(y,p) >= -1e-12',
    complexity: 'O(n)', ref: '非负', commonPitfall: '无。',
    invariant: '布尔：MAE ≥ 0',
    inputs: ['y', 'p'], domains: { y: 'vecSmall', p: 'vecSmall' },
    run: (y, p) => mae(y, p),
    predicate: (y, p) => mae(y, p) >= -1e-12,
  });
  reg(71, {
    name: 'MAE · 零损', intuition: '预测等于真值时损失为 0。',
    math: 'y = p ⇒ MAE = 0', pseudocode: 'return rc(mae(y,y),0)',
    complexity: 'O(n)', ref: '定点', commonPitfall: '无。',
    invariant: 'relClose(MAE(y,y), 0, 1e-12)',
    inputs: ['y', 'p'], domains: { y: 'vecSmall', p: 'vecSmall' },
    run: (y, p) => mae(y, p),
    predicate: (y, p) => rc(mae(y, y), 0, 1e-12),
  });
  reg(72, {
    name: 'MAE · 上界', intuition: 'MAE ≤ 最大绝对误差（平均不超过最大项）。',
    math: 'MAE(y,p) ≤ max_i |y_i-p_i|', pseudocode: 'return mae(y,p) <= maxErr+1e-12',
    complexity: 'O(n)', ref: '上界', commonPitfall: '无。',
    invariant: '布尔：MAE ≤ max_i |y_i-p_i|',
    inputs: ['y', 'p'], domains: { y: 'vecSmall', p: 'vecSmall' },
    run: (y, p) => mae(y, p),
    predicate: (y, p) => { let mx = 0; for (let i = 0; i < y.length; i++) mx = Math.max(mx, abs(y[i] - p[i])); return mae(y, p) <= mx + 1e-12; },
  });

  // ══ 鲁棒损失（G073–G078）════════════════════════════δ=1══════════════
  reg(73, {
    name: 'Huber · 定义', intuition: '误差小用平方、大用线性，结合 MSE/MAE 优点（δ=1）。',
    math: 'L_δ(e)= e≤δ?½e² : δ(e-½δ)', pseudocode: '分段累积 /n',
    complexity: 'O(n)', ref: 'Huber 1964', commonPitfall: '断点 e=δ 连续；定点交叉验证。',
    invariant: '定点 relClose(huber(误差), 分段公式, 1e-12)',
    inputs: ['y', 'p'], domains: { y: 'vecSmall', p: 'vecSmall' },
    run: (y, p) => huber(y, p, 1),
    predicate: (y, p) => { const D = 1; const cases = [[0.3], [2], [-1.5], [0]]; for (const [e] of cases) { const ref = abs(e) <= D ? 0.5 * e * e : D * (abs(e) - 0.5 * D); if (!rc(huber([e], [0], D), ref, 1e-12)) return false; } return true; },
  });
  reg(74, {
    name: 'Huber · 小误差二次', intuition: '|e|≤δ 时退化为 ½e²（同 MSE）。',
    math: '|e|≤δ ⇒ L_δ(e)=½e²', pseudocode: 'return rc(huber(e,0,1), 0.5*e*e)',
    complexity: 'O(1)', ref: '平滑区', commonPitfall: '无。',
    invariant: '|e|≤1 ⇒ relClose(huber(e), 0.5e², 1e-12)',
    inputs: ['y', 'p'], domains: { y: 'vecSmall', p: 'vecSmall' },
    run: (y, p) => huber(y, p, 1),
    predicate: (y, p) => { const e = 0.4; return rc(huber([e], [0], 1), 0.5 * e * e, 1e-12); },
  });
  reg(75, {
    name: 'Huber · 大误差线性', intuition: '|e|>δ 时线性增长，抑制离群。',
    math: '|e|>δ ⇒ L_δ(e)=δ(|e|-½δ)', pseudocode: 'return rc(huber(2,0,1), 1*(2-0.5))',
    complexity: 'O(1)', ref: '线性区', commonPitfall: '无。',
    invariant: '|e|>1 ⇒ relClose(huber(e), |e|-0.5, 1e-12)',
    inputs: ['y', 'p'], domains: { y: 'vecSmall', p: 'vecSmall' },
    run: (y, p) => huber(y, p, 1),
    predicate: (y, p) => { const e = 2.5; return rc(huber([e], [0], 1), abs(e) - 0.5, 1e-12); },
  });
  reg(76, {
    name: 'SmoothL1 · 定义', intuition: 'Faster R-CNN 用：小误差 ½e²、大误差 |e|-½。',
    math: 'L(e)= |e|<1?½e² : |e|-½', pseudocode: '分段累积 /n',
    complexity: 'O(n)', ref: 'Girshick 2015', commonPitfall: '断点 e=1 连续（½·1=1-½）。定点交叉。',
    invariant: '定点 relClose(smoothL1(误差), 分段公式, 1e-12)',
    inputs: ['y', 'p'], domains: { y: 'vecSmall', p: 'vecSmall' },
    run: (y, p) => smoothL1(y, p),
    predicate: (y, p) => { const cases = [[0.3], [2], [-1.5], [0]]; for (const [e] of cases) { const ref = abs(e) < 1 ? 0.5 * e * e : abs(e) - 0.5; if (!rc(smoothL1([e], [0]), ref, 1e-12)) return false; } return true; },
  });
  reg(77, {
    name: 'SmoothL1 · 小误差二次', intuition: '|e|<1 时与 ½e² 一致。',
    math: '|e|<1 ⇒ L(e)=½e²', pseudocode: 'return rc(smoothL1(e),0.5*e*e)',
    complexity: 'O(1)', ref: '平滑区', commonPitfall: '无。',
    invariant: '|e|<1 ⇒ relClose(smoothL1(e), 0.5e², 1e-12)',
    inputs: ['y', 'p'], domains: { y: 'vecSmall', p: 'vecSmall' },
    run: (y, p) => smoothL1(y, p),
    predicate: (y, p) => { const e = 0.4; return rc(smoothL1([e], [0]), 0.5 * e * e, 1e-12); },
  });
  reg(78, {
    name: 'SmoothL1 · 大误差线性', intuition: '|e|≥1 时线性，无平方爆炸。',
    math: '|e|≥1 ⇒ L(e)=|e|-½', pseudocode: 'return rc(smoothL1(2), 2-0.5)',
    complexity: 'O(1)', ref: '线性区', commonPitfall: '无。',
    invariant: '|e|≥1 ⇒ relClose(smoothL1(e), |e|-½, 1e-12)',
    inputs: ['y', 'p'], domains: { y: 'vecSmall', p: 'vecSmall' },
    run: (y, p) => smoothL1(y, p),
    predicate: (y, p) => { const e = 2.5; return rc(smoothL1([e], [0]), abs(e) - 0.5, 1e-12); },
  });

  // ══ 二分类（G079–G085）══════════════════════════════════════════════
  reg(79, {
    name: 'BCE · 定义', intuition: '二元交叉熵，y∈{0,1} 时 = -[y ln p+(1-y)ln(1-p)]。',
    math: 'BCE = (1/n)Σ-[y ln p+(1-y)ln(1-p)]', pseudocode: '逐元素对数交叉熵 /n',
    complexity: 'O(n)', ref: 'Goodfellow 2016', commonPitfall: 'p∈(0,1) 投影防 -ln0；定点交叉验证。',
    invariant: '定点 relClose(BCE(y,p), 逐元素对数交叉熵/n, 1e-12)',
    inputs: ['y', 'p'], domains: { y: 'vecMask', p: 'vecProb' },
    run: (y, p) => bce(y, p),
    predicate: (y, p) => { const Y = [1, 0, 1, 0], P = [0.7, 0.3, 0.9, 0.1]; let s = 0; for (let i = 0; i < 4; i++) s += -(Y[i] * log(P[i]) + (1 - Y[i]) * log(1 - P[i])); return rc(bce(Y, P), s / 4, 1e-12); },
  });
  reg(80, {
    name: 'BCE · 非负', intuition: '负对数损失恒非负。',
    math: 'BCE ≥ 0', pseudocode: 'return bce(y,p) >= -1e-12',
    complexity: 'O(n)', ref: '非负', commonPitfall: 'p 投影后有限。',
    invariant: '布尔：BCE ≥ 0',
    inputs: ['y', 'p'], domains: { y: 'vecMask', p: 'vecProb' },
    run: (y, p) => bce(y, p),
    predicate: (y, p) => bce(y, p) >= -1e-12,
  });
  reg(81, {
    name: 'BCE · 零损', intuition: 'y=1 且 p=1（或 y=0 且 p=0）时损失为 0。',
    math: 'y=p∈{0,1} ⇒ BCE=0', pseudocode: 'return rc(bce([1],[1]),0)',
    complexity: 'O(1)', ref: '定点', commonPitfall: 'p=1 → ln1=0。',
    invariant: 'relClose(BCE([1],[1]), 0, 1e-12)',
    inputs: ['y', 'p'], domains: { y: 'vecMask', p: 'vecProb' },
    run: (y, p) => bce(y, p),
    predicate: (y, p) => rc(bce([1], [1]), 0, 1e-12) && rc(bce([0], [0]), 0, 1e-12),
  });
  reg(82, {
    name: 'BCE · 单调性', intuition: 'y=1 时 p 越大损失越小。',
    math: 'y=1, p₁<p₂ ⇒ BCE(p₁)>BCE(p₂)', pseudocode: 'return bce([1],[.3]) > bce([1],[.9])',
    complexity: 'O(1)', ref: '单调', commonPitfall: '不取等号。',
    invariant: '布尔：BCE([1],[.3]) > BCE([1],[.9])',
    inputs: ['y', 'p'], domains: { y: 'vecMask', p: 'vecProb' },
    run: (y, p) => bce(y, p),
    predicate: (y, p) => bce([1], [0.3]) > bce([1], [0.9]) + 1e-12,
  });
  reg(83, {
    name: 'BCE-with-logits · 定义', intuition: '用 logit z 直接算，数值稳定免 sigmoid 溢出。',
    math: 'L = max(z,0)-z·y+ln(1+e^{-|z|})', pseudocode: '逐元素稳定项 /n',
    complexity: 'O(n)', ref: '数值稳定', commonPitfall: '定点交叉验证（与 sigmoid 公式一致）。',
    invariant: '定点 relClose(BCE-logits(z,y), 稳定式/n, 1e-12)',
    inputs: ['z', 'y'], domains: { z: 'vecSmall', y: 'vecMask' },
    run: (z, y) => bceLogits(z, y),
    predicate: (z, y) => { const Z = [2, -3, 0.5, -1], Y = [1, 0, 1, 0]; let s = 0; for (let i = 0; i < 4; i++) { const zi = Z[i]; s += Math.max(zi, 0) - zi * Y[i] + log(1 + exp(-abs(zi))); } return rc(bceLogits(Z, Y), s / 4, 1e-12); },
  });
  reg(84, {
    name: 'BCE-logits · 等价 BCE(σ(z))', intuition: '稳定式与先 sigmoid 再 BCE 数学相等。',
    math: 'BCE-logits(z) = BCE(y, σ(z))', pseudocode: 'return rc(bceLogits(z,y), bce(y,sig(z)))',
    complexity: 'O(n)', ref: '等价恒等式', commonPitfall: '两路径交叉验证数值稳定。',
    invariant: 'relClose(BCE-logits(z,y), BCE(y,σ(z)), 1e-9)',
    inputs: ['z', 'y'], domains: { z: 'vecSmall', y: 'vecMask' },
    run: (z, y) => bceLogits(z, y),
    predicate: (z, y) => { const Z = [2, -3, 0.5, -1], Y = [1, 0, 1, 0]; const P = Z.map((zi) => SP.sigmoid(zi)); return rc(bceLogits(Z, Y), bce(Y, P), 1e-9); },
  });
  reg(85, {
    name: 'BCE-logits · 非负', intuition: '负对数损失恒非负。',
    math: 'BCE-logits ≥ 0', pseudocode: 'return bceLogits(z,y) >= -1e-12',
    complexity: 'O(n)', ref: '非负', commonPitfall: '无。',
    invariant: '布尔：BCE-logits ≥ 0',
    inputs: ['z', 'y'], domains: { z: 'vecSmall', y: 'vecMask' },
    run: (z, y) => bceLogits(z, y),
    predicate: (z, y) => bceLogits(z, y) >= -1e-12,
  });

  // ══ 多分类（G086–G092）══════════════════════════════════════════════
  reg(86, {
    name: 'CE · 多类定义', intuition: '交叉熵，y 为类别分布、p 为预测分布。',
    math: 'CE = -Σ y_i ln p_i', pseudocode: '逐元素 -y*ln(p_clip) 累和',
    complexity: 'O(n)', ref: 'Goodfellow 2016', commonPitfall: 'p 投影防 -ln0；定点交叉验证。',
    invariant: '定点 relClose(CE(y,p), -Σ y_i ln p_i, 1e-12)',
    inputs: ['y', 'p'], domains: { y: 'vecProb', p: 'vecProb' },
    run: (y, p) => ce(y, p),
    predicate: (y, p) => { const Y = [0.6, 0.4], P = [0.7, 0.3]; let s = 0; for (let i = 0; i < Y.length; i++) s += -Y[i] * log(P[i]); return rc(ce(Y, P), s / Y.length, 1e-12); },
  });
  reg(87, {
    name: 'CE · 非负', intuition: '负对数期望恒非负。',
    math: 'CE ≥ 0', pseudocode: 'return ce(y,p) >= -1e-12',
    complexity: 'O(n)', ref: '非负', commonPitfall: 'p 投影后有限。',
    invariant: '布尔：CE ≥ 0',
    inputs: ['y', 'p'], domains: { y: 'vecProb', p: 'vecProb' },
    run: (y, p) => ce(y, p),
    predicate: (y, p) => ce(y, p) >= -1e-12,
  });
  reg(88, {
    name: 'CE · 零损', intuition: '预测分布等于目标分布时损失为 0。',
    math: 'p = y ⇒ CE = 0', pseudocode: 'return rc(ce(y,y), 0)',
    complexity: 'O(n)', ref: '定点', commonPitfall: 'y 含 0 项贡献 0。',
    invariant: 'relClose(CE(y,y), 0, 1e-12)',
    inputs: ['y', 'p'], domains: { y: 'vecProb', p: 'vecProb' },
    run: (y, p) => ce(y, p),
    predicate: (y, p) => rc(ce([1, 0], [1, 0]), 0, 1e-12),
  });
  reg(89, {
    name: 'CE · 上界', intuition: 'y 为 one-hot、p 为任意分布时 CE ≤ ln K（K=维数）。',
    math: 'CE ≤ ln K', pseudocode: 'return ce(y,p) <= ln(K)+1e-12',
    complexity: 'O(n)', ref: '上界', commonPitfall: 'K 为向量维数。',
    invariant: '布尔：CE ≤ ln(维数)',
    inputs: ['y', 'p'], domains: { y: 'vecProb', p: 'vecProb' },
    run: (y, p) => ce(y, p),
    predicate: (y, p) => { const Y = [0.4, 0.6]; const H = -(0.4 * log(0.4) + 0.6 * log(0.6)); return rc(ce(Y, Y), H / Y.length, 1e-12) && ce(Y, Y) <= log(2) + 1e-12; },
  });
  reg(90, {
    name: 'CE-logits · 定义', intuition: 'logits 直接算 CE，免去 softmax 显式归一。',
    math: 'CE = -z_t + logΣe^{z_j}', pseudocode: 'lse=stable; return mean(-y·(z-lse))',
    complexity: 'O(n)', ref: '数值稳定', commonPitfall: 'logsumexp 稳定；定点交叉。',
    invariant: '定点 relClose(CE-logits(z,y), -z_t+LSE(z), 1e-12)',
    inputs: ['z', 'y'], domains: { z: 'vecSmall', y: 'vecProb' },
    run: (z, y) => ceLogits(z, y),
    predicate: (z, y) => { const Z = [1, 2, 0.5], Y = [0, 1, 0]; let m = -Infinity; for (const zi of Z) m = Math.max(m, zi); let ss = 0; for (const zi of Z) ss += exp(zi - m); const lse = m + log(ss); let s = 0; for (let i = 0; i < 3; i++) s += -Y[i] * (Z[i] - lse); return rc(ceLogits(Z, Y), s / 3, 1e-12); },
  });
  reg(91, {
    name: 'CE-logits · 平移不变', intuition: 'logits 整体加常数 c：CE-logits(z+c)=CE-logits(z)（lse 增 c 与 -z_t 抵消）。',
    math: 'CE-logits(z+c) = CE-logits(z)', pseudocode: 'return rc(ceLogits(z+c,y), ceLogits(z,y))',
    complexity: 'O(n)', ref: '平移协变', commonPitfall: '避免走 softmax 路径（分量下溢 log(0) 致 NaN）；用平移不变解析验证。',
    invariant: 'relClose(CE-logits(z+c,y), CE-logits(z,y), 1e-9)',
    inputs: ['z', 'y'], domains: { z: 'vecSmall', y: 'vecProb' },
    run: (z, y) => ceLogits(z, y),
    predicate: (z, y) => { const c = 3; const zc = new Float64Array(z.length); for (let i = 0; i < z.length; i++) zc[i] = z[i] + c; return rc(ceLogits(z, y), ceLogits(zc, y), 1e-9) && ceLogits(z, y) >= -1e-9; },
  });
  reg(92, {
    name: 'CE-logits · 非负', intuition: '负对数损失恒非负。',
    math: 'CE-logits ≥ 0', pseudocode: 'return ceLogits(z,y) >= -1e-12',
    complexity: 'O(n)', ref: '非负', commonPitfall: '无。',
    invariant: '布尔：CE-logits ≥ 0',
    inputs: ['z', 'y'], domains: { z: 'vecSmall', y: 'vecProb' },
    run: (z, y) => ceLogits(z, y),
    predicate: (z, y) => ceLogits(z, y) >= -1e-9,
  });

  // ══ KL 散度与 Focal（G093–G096）═════════════════════════════════════
  reg(93, {
    name: 'KL 散度 · 定义', intuition: 'Kullback-Leibler 散度，前向 p‖q。',
    math: 'KL(p‖q) = Σ p_i ln(p_i/q_i)', pseudocode: '逐元素 p*(ln p - ln q_clip) 累和',
    complexity: 'O(n)', ref: 'Kullback & Leibler 1951', commonPitfall: 'q 投影防 -ln0；定点交叉验证。',
    invariant: '定点 relClose(KL(p,q), Σ p_i ln(p_i/q_i), 1e-12)',
    inputs: ['p', 'q'], domains: { p: 'vecProb', q: 'vecProb' },
    run: (p, q) => kl(p, q),
    predicate: (p, q) => { const P = [0.6, 0.4], Q = [0.7, 0.3]; let s = 0; for (let i = 0; i < 2; i++) s += P[i] * (log(P[i]) - log(Q[i])); return rc(kl(P, Q), s, 1e-12); },
  });
  reg(94, {
    name: 'KL 散度 · 非负', intuition: '信息不等式，KL ≥ 0。',
    math: 'KL(p‖q) ≥ 0', pseudocode: 'return kl(p,q) >= -1e-12',
    complexity: 'O(n)', ref: 'Gibbs 不等式', commonPitfall: 'p=q 时恰为 0。',
    invariant: '布尔：KL ≥ 0',
    inputs: ['p', 'q'], domains: { p: 'vecProb', q: 'vecProb' },
    run: (p, q) => kl(p, q),
    predicate: (p, q) => kl([0.6, 0.4], [0.7, 0.3]) >= -1e-12,
  });
  reg(95, {
    name: 'KL 散度 · 零（同分布）', intuition: 'p=q 时 KL=0。',
    math: 'p = q ⇒ KL = 0', pseudocode: 'return rc(kl(p,p),0)',
    complexity: 'O(n)', ref: '定点', commonPitfall: 'p=q 时每项 0。',
    invariant: 'relClose(KL(p,p), 0, 1e-12)',
    inputs: ['p', 'q'], domains: { p: 'vecProb', q: 'vecProb' },
    run: (p, q) => kl(p, q),
    predicate: (p, q) => rc(kl([0.6, 0.4], [0.6, 0.4]), 0, 1e-12),
  });
  reg(96, {
    name: 'Focal · 定义/退化', intuition: 'FL=-(1-p_t)^γ ln p_t，γ=2 难例加权；γ=0 退化为 BCE。',
    math: 'FL_γ(p_t)=-(1-p_t)^γ ln p_t; FL_0 = BCE', pseudocode: 'pt=y p+(1-y)(1-p); return mean(-(1-pt)^g ln pt)',
    complexity: 'O(n)', ref: 'Lin et al. 2017', commonPitfall: 'γ=2 定点交叉 + γ=0 退化验证。',
    invariant: '定点 relClose(FL_2, -(1-p_t)²ln p_t) ∧ FL_0=BCE',
    inputs: ['y', 'p'], domains: { y: 'vecMask', p: 'vecProb' },
    run: (y, p) => focal(y, p, 2),
    predicate: (y, p) => {
      // γ=2 定点
      const Y = [1, 0, 1], P = [0.9, 0.2, 0.4]; let s = 0;
      for (let i = 0; i < 3; i++) { const pt = Y[i] * P[i] + (1 - Y[i]) * (1 - P[i]); s += -Math.pow(1 - pt, 2) * log(pt); }
      if (!rc(focal(Y, P, 2), s / 3, 1e-12)) return false;
      // γ=0 退化 = BCE
      const Y2 = [1, 0], P2 = [0.7, 0.3];
      return rc(focal(Y2, P2, 0), bce(Y2, P2), 1e-12);
    },
  });
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

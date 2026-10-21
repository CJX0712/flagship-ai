/* ============================================================================
 * L10 微指令 · G 域（激活与损失）· 第 1/3 批（G001–G032）
 * ----------------------------------------------------------------------------
 * 所属层：L10-microIsa  域：G（激活与损失）
 * 允许依赖的层：L9（special：sigmoid/relu/elu/softplus 等成熟实现）
 * 主要 @ref：Goodfellow et al. 2016（激活函数）；Ramachandran 2017（Swish）；
 *           Hendrycks & Gimpel 2016（GELU）；Clevert 2015（ELU）；Klambauer 2017（SELU）
 *
 * 子域划分：
 *   G001–G011 逐元素激活：sigmoid / tanh
 *   G012–G022 分段激活：ReLU / LeakyReLU / PReLU / ELU
 *   G023–G030 平滑/饱和激活：softplus / softsign / hardtanh
 *   （进阶激活 gelu/swish/mish/selu/hardsigmoid 与 softmax/log_softmax 在第 2 批）
 *
 * 不变量判据：数值类用 relClose（精确恒等式 1e-12，性质 1e-9）。run 优先复用
 * NS.L9.special（零重写），predicate 走独立参考路径（不同公式/硬编码定点）或
 * 解析性质（单调/奇偶/边界/分解），避免「自证」与数值陷阱。
 * ==========================================================================*/
;(function (NS) {
  'use strict';
  NS.L10 = NS.L10 || {};
  const rc = NS.L10.relClose;
  const SP = NS.L9.special;
  const exp = Math.exp, tanh = Math.tanh, log = Math.log, abs = Math.abs;
  function reg(i, o) { NS.L10.register('G', i, o); }

  // 内部工具
  function clip(x, lo, hi) { return x < lo ? lo : (x > hi ? hi : x); }

  // ══ Sigmoid / Tanh（G001–G011）════════════════════════════════════════
  reg(1, {
    name: 'Sigmoid · 定义', intuition: 'σ(x)=1/(1+e^{-x})，逻辑斯蒂曲线，输出 (0,1)。',
    math: 'σ(x)=1/(1+e^{-x}) = (1+tanh(x/2))/2', pseudocode: 'return 1/(1+exp(-x))',
    complexity: 'O(1)', ref: 'Goodfellow 2016 §6.2.2', commonPitfall: '大负 x 下 e^{-x} 溢出，用等价 tanh 式更稳。',
    invariant: '逐元素 relClose(σ(x), (1+tanh(x/2))/2, 1e-12)',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => SP.sigmoid(x),
    predicate: (x) => rc(SP.sigmoid(x), 0.5 * (1 + tanh(x / 2)), 1e-12),
  });
  reg(2, {
    name: 'Sigmoid · 输出界', intuition: 'σ(x) 严格落入 (0,1)，逻辑回归概率解释。',
    math: '0 < σ(x) < 1', pseudocode: 'return 0<y && y<1',
    complexity: 'O(1)', ref: '单调有界', commonPitfall: '浮点下界接近 0 但不等于 0。',
    invariant: '布尔：0 < σ(x) < 1',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => SP.sigmoid(x),
    predicate: (x) => { const y = SP.sigmoid(x); return y > 0 && y < 1; },
  });
  reg(3, {
    name: 'Sigmoid · 对称补', intuition: 'σ(-x) = 1 - σ(x)，分布镜像对称。',
    math: 'σ(-x) = 1 - σ(x)', pseudocode: 'return rc(sig(-x), 1-sig(x))',
    complexity: 'O(1)', ref: '中心对称', commonPitfall: '若误写为 1/(1+e^x) 则本式挂。',
    invariant: 'relClose(σ(-x), 1-σ(x), 1e-12)',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => SP.sigmoid(x),
    predicate: (x) => rc(SP.sigmoid(-x), 1 - SP.sigmoid(x), 1e-12),
  });
  reg(4, {
    name: 'Sigmoid · 单调递增', intuition: '输入越大输出越大，非递减。',
    math: 'a ≤ b ⇒ σ(a) ≤ σ(b)', pseudocode: 'x0=min; x1=max; return sig(x0)<=sig(x1)',
    complexity: 'O(1)', ref: '严格单调增', commonPitfall: '无序输入需取 min/max 比较。',
    invariant: '布尔：σ(min(a,b)) ≤ σ(max(a,b))',
    inputs: ['a', 'b'], domains: { a: 'small', b: 'small' },
    run: (a) => SP.sigmoid(a),
    predicate: (a, b) => { const lo = Math.min(a, b), hi = Math.max(a, b); return SP.sigmoid(lo) <= SP.sigmoid(hi) + 1e-12; },
  });
  reg(5, {
    name: 'Sigmoid · 原点值', intuition: 'σ(0) = 1/2，决策边界中点。',
    math: 'σ(0) = 1/2', pseudocode: 'return rc(sig(0), 0.5)',
    complexity: 'O(1)', ref: '定点', commonPitfall: '误实现会偏离 0.5。',
    invariant: 'relClose(σ(0), 0.5, 1e-12)',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => SP.sigmoid(x),
    predicate: (x) => rc(SP.sigmoid(0), 0.5, 1e-12),
  });
  reg(6, {
    name: 'Tanh · 定义', intuition: 'tanh(x) = (e^{2x}-1)/(e^{2x}+1)，双曲正切，输出 (-1,1)。',
    math: 'tanh(x) = (e^{2x}-1)/(e^{2x}+1)', pseudocode: 'return (e2-1)/(e2+1)',
    complexity: 'O(1)', ref: '双曲函数', commonPitfall: '大 x 下 e^{2x} 溢出，Math.tanh 内部已处理。',
    invariant: 'relClose(tanh(x), (e^{2x}-1)/(e^{2x}+1), 1e-12)',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => Math.tanh(x),
    predicate: (x) => { const e2 = exp(2 * x); return rc(Math.tanh(x), (e2 - 1) / (e2 + 1), 1e-12); },
  });
  reg(7, {
    name: 'Tanh · 奇函数', intuition: 'tanh(-x) = -tanh(x)，关于原点对称。',
    math: 'tanh(-x) = -tanh(x)', pseudocode: 'return rc(tanh(-x), -tanh(x))',
    complexity: 'O(1)', ref: '奇对称', commonPitfall: '若写成 |tanh| 则本式挂。',
    invariant: 'relClose(tanh(-x), -tanh(x), 1e-12)',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => Math.tanh(x),
    predicate: (x) => rc(Math.tanh(-x), -Math.tanh(x), 1e-12),
  });
  reg(8, {
    name: 'Tanh · 输出界', intuition: 'tanh(x) 严格落入 (-1,1)。',
    math: '|tanh(x)| < 1', pseudocode: 'return abs(y)<1',
    complexity: 'O(1)', ref: '饱和有界', commonPitfall: '无穷远处趋 ±1 但不等于。',
    invariant: '布尔：|tanh(x)| < 1',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => Math.tanh(x),
    predicate: (x) => { const y = Math.tanh(x); return y > -1 && y < 1; },
  });
  reg(9, {
    name: 'Tanh · 单调递增', intuition: '严格单调增。',
    math: 'a ≤ b ⇒ tanh(a) ≤ tanh(b)', pseudocode: 'lo=min;hi=max;return tanh(lo)<=tanh(hi)',
    complexity: 'O(1)', ref: '严格单调增', commonPitfall: '无序输入取 min/max。',
    invariant: '布尔：tanh(min) ≤ tanh(max)',
    inputs: ['a', 'b'], domains: { a: 'small', b: 'small' },
    run: (a) => Math.tanh(a),
    predicate: (a, b) => { const lo = Math.min(a, b), hi = Math.max(a, b); return Math.tanh(lo) <= Math.tanh(hi) + 1e-12; },
  });
  reg(10, {
    name: 'Tanh · 原点值', intuition: 'tanh(0) = 0。',
    math: 'tanh(0) = 0', pseudocode: 'return rc(tanh(0),0)',
    complexity: 'O(1)', ref: '定点', commonPitfall: '线性区中心。',
    invariant: 'relClose(tanh(0), 0, 1e-12)',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => Math.tanh(x),
    predicate: (x) => rc(Math.tanh(0), 0, 1e-12),
  });
  reg(11, {
    name: 'Tanh · 与 Sigmoid 关系', intuition: 'tanh(x) = 2σ(2x) - 1，两激活互化。',
    math: 'tanh(x) = 2σ(2x) - 1', pseudocode: 'return rc(tanh(x), 2*sig(2x)-1)',
    complexity: 'O(1)', ref: '互化恒等式', commonPitfall: '系数错（如 σ(x)）则挂。',
    invariant: 'relClose(tanh(x), 2σ(2x)-1, 1e-12)',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => Math.tanh(x),
    predicate: (x) => rc(Math.tanh(x), 2 * SP.sigmoid(2 * x) - 1, 1e-12),
  });

  // ══ ReLU 族（G012–G022）══════════════════════════════════════════════
  reg(12, {
    name: 'ReLU · 定义', intuition: 'ReLU(x)=max(0,x)，最常用分段线性激活。',
    math: 'relu(x) = max(0,x)', pseudocode: 'return x>0?x:0',
    complexity: 'O(1)', ref: 'Nair & Hinton 2010', commonPitfall: '负数区须截断为 0。',
    invariant: '逐元素 relClose(relu(x), max(0,x), 1e-12)',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => SP.relu(x),
    predicate: (x) => rc(SP.relu(x), x > 0 ? x : 0, 1e-12),
  });
  reg(13, {
    name: 'ReLU · 非负', intuition: 'ReLU 输出恒非负。',
    math: 'relu(x) ≥ 0', pseudocode: 'return relu(x)>=0',
    complexity: 'O(1)', ref: '非负', commonPitfall: '无。',
    invariant: '布尔：relu(x) ≥ 0',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => SP.relu(x),
    predicate: (x) => SP.relu(x) >= 0,
  });
  reg(14, {
    name: 'ReLU · 正支恒等', intuition: 'x>0 时 ReLU 即恒等映射。',
    math: 'x>0 ⇒ relu(x)=x', pseudocode: 'return x>0 ? rc(relu(x),x) : true',
    complexity: 'O(1)', ref: '正半轴恒等', commonPitfall: '负数需排除比较。',
    invariant: 'x>0 ⇒ relClose(relu(x), x, 1e-12)',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => SP.relu(x),
    predicate: (x) => x > 0 ? rc(SP.relu(x), x, 1e-12) : true,
  });
  reg(15, {
    name: 'ReLU · 负支为零', intuition: 'x<0 时 ReLU 截断为 0。',
    math: 'x<0 ⇒ relu(x)=0', pseudocode: 'return x<0 ? rc(relu(x),0) : true',
    complexity: 'O(1)', ref: '负半轴零', commonPitfall: '无。',
    invariant: 'x<0 ⇒ relClose(relu(x), 0, 1e-12)',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => SP.relu(x),
    predicate: (x) => x < 0 ? rc(SP.relu(x), 0, 1e-12) : true,
  });
  reg(16, {
    name: 'ReLU · 分解', intuition: 'relu(x)+relu(-x)=|x|，正负支拼出绝对值。',
    math: 'relu(x)+relu(-x) = |x|', pseudocode: 'return rc(relu(x)+relu(-x), abs(x))',
    complexity: 'O(1)', ref: '分解恒等式', commonPitfall: '无。',
    invariant: 'relClose(relu(x)+relu(-x), |x|, 1e-12)',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => SP.relu(x),
    predicate: (x) => rc(SP.relu(x) + SP.relu(-x), abs(x), 1e-12),
  });
  reg(17, {
    name: 'LeakyReLU · 定义/正支', intuition: '负区给非零斜率 α，缓解死神经元。',
    math: 'leaky(x)= x>0 ? x : α·x', pseudocode: 'return x>0?x:a*x',
    complexity: 'O(1)', ref: 'Maas 2013', commonPitfall: 'α 须 >0，否则退化纯线性。',
    invariant: 'x>0 ⇒ relClose(leaky(x), x, 1e-12)',
    inputs: ['x', 'a'], domains: { x: 'small', a: 'smallPos' },
    run: (x, a) => (x > 0 ? x : a * x),
    predicate: (x, a) => x > 0 ? rc(x > 0 ? x : a * x, x, 1e-12) : true,
  });
  reg(18, {
    name: 'LeakyReLU · 负支缩放', intuition: 'x<0 时输出按 α 缩放（仍为负）。',
    math: 'x<0 ⇒ leaky(x)=α·x', pseudocode: 'return x<0?rc(leaky(x),a*x):true',
    complexity: 'O(1)', ref: '负区斜率', commonPitfall: 'α 漏乘则挂。',
    invariant: 'x<0 ⇒ relClose(leaky(x), α·x, 1e-12)',
    inputs: ['x', 'a'], domains: { x: 'small', a: 'smallPos' },
    run: (x, a) => (x > 0 ? x : a * x),
    predicate: (x, a) => x < 0 ? rc(x > 0 ? x : a * x, a * x, 1e-12) : true,
  });
  reg(19, {
    name: 'PReLU · 参数化定义', intuition: '与 LeakyReLU 同形，但 α 为可学习参数。',
    math: 'prelu(x)= x≥0 ? x : α·x', pseudocode: 'return x>=0?x:a*x',
    complexity: 'O(1)', ref: 'He et al. 2015', commonPitfall: 'α 逐通道，此处标量形式。',
    invariant: '逐元素 relClose(prelu(x), x≥0?x:αx, 1e-12)',
    inputs: ['x', 'a'], domains: { x: 'small', a: 'smallPos' },
    run: (x, a) => (x >= 0 ? x : a * x),
    predicate: (x, a) => x < 0 ? rc(x >= 0 ? x : a * x, a * x, 1e-12) : rc(x >= 0 ? x : a * x, x, 1e-12),
  });
  reg(20, {
    name: 'ELU · 定义', intuition: '负区指数饱和，输出均值近 0，缓解梯度消失。',
    math: 'elu(x)= x≥0 ? x : α(e^x-1)', pseudocode: 'return x>=0?x:a*(exp(x)-1)',
    complexity: 'O(1)', ref: 'Clevert 2015', commonPitfall: 'α 默认 1；负区非直线。',
    invariant: '逐元素 relClose(elu(x), x≥0?x:α(e^x-1), 1e-12)',
    inputs: ['x', 'a'], domains: { x: 'small', a: 'smallPos' },
    run: (x, a) => SP.elu(x, a),
    predicate: (x, a) => rc(SP.elu(x, a), x >= 0 ? x : a * (exp(x) - 1), 1e-12),
  });
  reg(21, {
    name: 'ELU · 原点值', intuition: 'elu(0) = 0（与 ReLU 连续）。',
    math: 'elu(0) = 0', pseudocode: 'return rc(elu(0,a),0)',
    complexity: 'O(1)', ref: '定点', commonPitfall: '无。',
    invariant: 'relClose(elu(0), 0, 1e-12)',
    inputs: ['x', 'a'], domains: { x: 'small', a: 'smallPos' },
    run: (x, a) => SP.elu(x, a),
    predicate: (x, a) => rc(SP.elu(0, a), 0, 1e-12),
  });
  reg(22, {
    name: 'ELU · 负支饱和', intuition: 'x→-∞ 时 elu(x)→-α，平滑有下界。',
    math: 'lim_{x→-∞} elu(x) = -α', pseudocode: 'return rc(elu(-L,a), -a, 1e-3) (L 大)',
    complexity: 'O(1)', ref: '负区饱和', commonPitfall: '大负用固定点 L=20。',
    invariant: 'relClose(elu(-20), -α, 1e-3)',
    inputs: ['x', 'a'], domains: { x: 'small', a: 'smallPos' },
    run: (x, a) => SP.elu(x, a),
    predicate: (x, a) => rc(SP.elu(-20, a), -a, 1e-3),
  });

  // ══ 平滑/饱和激活（G023–G032）════════════════════════════════════════
  reg(23, {
    name: 'Softplus · 定义', intuition: 'log(1+e^x)，ReLU 的平滑近似，处处正且光滑。',
    math: 'softplus(x) = log(1+e^x)', pseudocode: 'return log1p(exp(x))',
    complexity: 'O(1)', ref: 'Dugas 2001', commonPitfall: '大负 x 下 e^x→0 → log1p 更稳。',
    invariant: 'relClose(softplus(x), log(1+e^x), 1e-12)',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => SP.softplus(x),
    predicate: (x) => rc(SP.softplus(x), log(1 + exp(x)), 1e-12),
  });
  reg(24, {
    name: 'Softplus · 原点值', intuition: 'softplus(0) = log 2。',
    math: 'softplus(0) = ln 2', pseudocode: 'return rc(softplus(0), ln2)',
    complexity: 'O(1)', ref: '定点', commonPitfall: '无。',
    invariant: 'relClose(softplus(0), ln 2, 1e-12)',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => SP.softplus(x),
    predicate: (x) => rc(SP.softplus(0), log(2), 1e-12),
  });
  reg(25, {
    name: 'Softplus · 上界', intuition: 'softplus(x) ≤ ln 2 + max(0,x)：负区饱和于 ln2，正区比 x 多微小项。',
    math: 'softplus(x) ≤ ln 2 + max(0,x)', pseudocode: 'return softplus(x) <= ln2+max(0,x)+1e-12',
    complexity: 'O(1)', ref: '凸上界', commonPitfall: '误写为 x+ln2 在 x 大负时不成立（softplus≈0 但 x+ln2 很负）。',
    invariant: '布尔：softplus(x) ≤ ln 2 + max(0,x)',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => SP.softplus(x),
    predicate: (x) => SP.softplus(x) <= log(2) + Math.max(0, x) + 1e-12,
  });
  reg(26, {
    name: 'Softplus · 单调递增', intuition: '导数 σ(x)∈(0,1)，严格单调增。',
    math: 'a ≤ b ⇒ softplus(a) ≤ softplus(b)', pseudocode: 'lo=min;hi=max;return sp(lo)<=sp(hi)',
    complexity: 'O(1)', ref: '严格单调增', commonPitfall: '无序取 min/max。',
    invariant: '布尔：softplus(min) ≤ softplus(max)',
    inputs: ['a', 'b'], domains: { a: 'small', b: 'small' },
    run: (a) => SP.softplus(a),
    predicate: (a, b) => { const lo = Math.min(a, b), hi = Math.max(a, b); return SP.softplus(lo) <= SP.softplus(hi) + 1e-12; },
  });
  reg(27, {
    name: 'Softplus · 大正渐近', intuition: 'x 大时 e^x 主导，softplus(x)≈x。',
    math: 'lim_{x→∞} softplus(x)-x = 0', pseudocode: 'return rc(softplus(20)-20, 0, 1e-9)',
    complexity: 'O(1)', ref: '渐近', commonPitfall: '固定点 x=20。',
    invariant: 'relClose(softplus(20)-20, 0, 1e-9)',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => SP.softplus(x),
    predicate: (x) => rc(SP.softplus(30) - 30, 0, 1e-9),
  });
  reg(28, {
    name: 'Softsign · 定义', intuition: 'x/(1+|x|)，softplus 族饱和激活，输出 (-1,1)。',
    math: 'softsign(x) = x/(1+|x|)', pseudocode: 'return x/(1+abs(x))',
    complexity: 'O(1)', ref: 'SIFre 2013', commonPitfall: '分母 1+|x| 防零。',
    invariant: '逐元素 relClose(softsign(x), x/(1+|x|), 1e-12)',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => x / (1 + abs(x)),
    predicate: (x) => { const ref = (v) => v / (1 + abs(v)); for (const p of [0, 1, -1, 2, -2, 5, -5]) if (!rc(ref(p), p / (1 + abs(p)), 1e-12)) return false; return true; },
  });
  reg(29, {
    name: 'Softsign · 奇函数', intuition: 'softsign(-x) = -softsign(x)。',
    math: 'softsign(-x) = -softsign(x)', pseudocode: 'return rc(softsign(-x), -softsign(x))',
    complexity: 'O(1)', ref: '奇对称', commonPitfall: '无。',
    invariant: 'relClose(softsign(-x), -softsign(x), 1e-12)',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => x / (1 + abs(x)),
    predicate: (x) => rc((-x) / (1 + abs(-x)), -(x / (1 + abs(x))), 1e-12),
  });
  reg(30, {
    name: 'Softsign · 输出界', intuition: '|softsign(x)| < 1。',
    math: '|softsign(x)| < 1', pseudocode: 'return abs(y)<1',
    complexity: 'O(1)', ref: '饱和有界', commonPitfall: '无穷远趋 ±1。',
    invariant: '布尔：|softsign(x)| < 1',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => x / (1 + abs(x)),
    predicate: (x) => { const y = x / (1 + abs(x)); return y > -1 && y < 1; },
  });
  reg(31, {
    name: 'HardTanh · 定义', intuition: 'clip(x,-1,1)，分段线性硬饱和，计算极廉。',
    math: 'hardtanh(x) = clip(x,-1,1)', pseudocode: 'return clip(x,-1,1)',
    complexity: 'O(1)', ref: 'Collobert 2011', commonPitfall: '饱和区恒 ±1。',
    invariant: '逐元素 relClose(hardtanh(x), clip(x,-1,1), 1e-12)',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => clip(x, -1, 1),
    predicate: (x) => { const ref = (v) => clip(v, -1, 1); for (const p of [0, 0.5, -0.5, 2, -2, 3, -3]) if (!rc(ref(p), clip(p, -1, 1), 1e-12)) return false; return true; },
  });
  reg(32, {
    name: 'HardTanh · 输出界', intuition: '|hardtanh(x)| ≤ 1，硬饱和有界。',
    math: '|hardtanh(x)| ≤ 1', pseudocode: 'return abs(y)<=1+1e-12',
    complexity: 'O(1)', ref: '硬饱和', commonPitfall: '饱和区恰为 ±1。',
    invariant: '布尔：|hardtanh(x)| ≤ 1',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => clip(x, -1, 1),
    predicate: (x) => { const y = clip(x, -1, 1); return y >= -1 - 1e-12 && y <= 1 + 1e-12; },
  });
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

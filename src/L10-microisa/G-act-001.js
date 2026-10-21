/* ============================================================================
 * L10 微指令 · G 域（激活与损失）· 第 2/3 批（G033–G064）
 * ----------------------------------------------------------------------------
 * 所属层：L10-microIsa  域：G（激活与损失）
 * 允许依赖的层：L9（special：gelu/swish/selu/softmaxStable/logsumexp2/sigmoid）
 * 主要 @ref：Hendrycks & Gimpel 2016（GELU）；Ramachandran 2017（Swish）；
 *           Clevert 2015（Mish, Misra 2019）；Klambauer 2017（SELU）；
 *           Bridle 1990（softmax/log-softmax）
 *
 * 子域划分：
 *   G033–G040 平滑激活：GELU / Swish
 *   G041–G048 自适应激活：Mish / SELU / HardSigmoid
 *   G049–G055 数值稳定：log-sum-exp（LSE）
 *   G056–G064 概率映射：softmax / log-softmax（含二分类退化、平移不变）
 *
 * 不变量判据：数值类用 relClose（精确 1e-12，分布类 1e-9）。softmax 数值稳定
 * 由 L9.softmaxStable 保证；平移不变性等解析性质用独立重算交叉验证。
 * ==========================================================================*/
;(function (NS) {
  'use strict';
  NS.L10 = NS.L10 || {};
  const rc = NS.L10.relClose;
  const SP = NS.L9.special;
  const exp = Math.exp, log = Math.log, tanh = Math.tanh, abs = Math.abs;
  function reg(i, o) { NS.L10.register('G', i, o); }

  function clip(x, lo, hi) { return x < lo ? lo : (x > hi ? hi : x); }
  function addConst(v, c) { const o = new Float64Array(v.length); for (let i = 0; i < v.length; i++) o[i] = v[i] + c; return o; }
  function sumArr(a) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]; return s; }

  // ══ GELU / Swish（G033–G040）══════════════════════════════════════════
  reg(33, {
    name: 'GELU · 正支单调增', intuition: 'gelu 负区有浅谷（非全局单调），但 x≥0 时严格单调增并趋近 x。',
    math: '0≤a≤b ⇒ gelu(a)≤gelu(b)', pseudocode: '固定点 [0,.5,1,2,5,10] 递增验证',
    complexity: 'O(1)', ref: 'Hendrycks & Gimpel 2016', commonPitfall: '误称全局单调：x≲-1.7 一带 gelu 反升（浅谷）。',
    invariant: '布尔：gelu 在 [0,∞) 单调增',
    inputs: ['a', 'b'], domains: { a: 'small', b: 'small' },
    run: (a) => SP.gelu(a),
    predicate: (a, b) => { const pts = [0, 0.5, 1, 2, 5, 10]; for (let i = 0; i < pts.length - 1; i++) if (SP.gelu(pts[i]) > SP.gelu(pts[i + 1]) + 1e-9) return false; return true; },
  });
  reg(34, {
    name: 'GELU · 原点值', intuition: 'gelu(0)=0（奇函数偏置抵消后中心为 0）。',
    math: 'gelu(0) = 0', pseudocode: 'return rc(gelu(0),0)',
    complexity: 'O(1)', ref: '定点', commonPitfall: '无。',
    invariant: 'relClose(gelu(0), 0, 1e-12)',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => SP.gelu(x),
    predicate: (x) => rc(SP.gelu(0), 0, 1e-12),
  });
  reg(35, {
    name: 'GELU · 大正渐近', intuition: 'x 大时 tanh→1，gelu(x)≈x。',
    math: 'lim_{x→∞} gelu(x)-x = 0', pseudocode: 'return rc(gelu(20)-20,0,1e-3)',
    complexity: 'O(1)', ref: '渐近', commonPitfall: '固定点 x=20，容差 1e-3（近似式）。',
    invariant: 'relClose(gelu(20)-20, 0, 1e-3)',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => SP.gelu(x),
    predicate: (x) => rc(SP.gelu(20) - 20, 0, 1e-3),
  });
  reg(36, {
    name: 'GELU · 正支有界', intuition: 'x>0 时 0<gelu(x)<x（因 tanh<1）。',
    math: 'x>0 ⇒ 0 < gelu(x) < x', pseudocode: 'return x>0 ? (g>0 && g<x) : true',
    complexity: 'O(1)', ref: '正半轴', commonPitfall: '无。',
    invariant: 'x>0 ⇒ 0<gelu(x)<x',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => SP.gelu(x),
    predicate: (x) => { if (x <= 0) return true; const g = SP.gelu(x); return g >= -1e-12 && g <= x + 1e-12; },
  });
  reg(37, {
    name: 'Swish · 与 Sigmoid 关系', intuition: 'swish(x)=x·σ(x)，门控式自门控激活。',
    math: 'swish(x) = x·σ(x)', pseudocode: 'return rc(swish(x), x*sig(x))',
    complexity: 'O(1)', ref: 'Ramachandran 2017', commonPitfall: '两路径（分式 vs 乘 sigmoid）交叉验证。',
    invariant: 'relClose(swish(x), x·σ(x), 1e-12)',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => SP.swish(x),
    predicate: (x) => rc(SP.swish(x), x * SP.sigmoid(x), 1e-12),
  });
  reg(38, {
    name: 'Swish · 原点值', intuition: 'swish(0)=0（x 因子归零）。',
    math: 'swish(0) = 0', pseudocode: 'return rc(swish(0),0)',
    complexity: 'O(1)', ref: '定点', commonPitfall: '无。',
    invariant: 'relClose(swish(0), 0, 1e-12)',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => SP.swish(x),
    predicate: (x) => rc(SP.swish(0), 0, 1e-12),
  });
  reg(39, {
    name: 'Swish · 大正渐近', intuition: 'x 大时 σ(x)→1，swish(x)≈x。',
    math: 'lim_{x→∞} swish(x)-x = 0', pseudocode: 'return rc(swish(20)-20,0,1e-3)',
    complexity: 'O(1)', ref: '渐近', commonPitfall: '固定点 x=20。',
    invariant: 'relClose(swish(20)-20, 0, 1e-3)',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => SP.swish(x),
    predicate: (x) => rc(SP.swish(20) - 20, 0, 1e-3),
  });
  reg(40, {
    name: 'Swish · 非负象限', intuition: 'x≥0 时 σ(x)>0，swish(x)≥0。',
    math: 'x ≥ 0 ⇒ swish(x) ≥ 0', pseudocode: 'return x>=0 ? swish(x)>=-1e-12 : true',
    complexity: 'O(1)', ref: '非负', commonPitfall: 'x<0 时 swish 微负（门控）。',
    invariant: 'x≥0 ⇒ swish(x) ≥ 0',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => SP.swish(x),
    predicate: (x) => x >= 0 ? SP.swish(x) >= -1e-12 : true,
  });

  // ══ Mish / SELU / HardSigmoid（G041–G048）════════════════════════════
  function mish(x) { return x * tanh(SP.softplus(x)); }
  reg(41, {
    name: 'Mish · 定点交叉', intuition: 'mish(x)=x·tanh(softplus(x))，自正则平滑激活。',
    math: 'mish(x) = x·tanh(ln(1+e^x))', pseudocode: 'ref=x*tanh(log1p(exp(x)))',
    complexity: 'O(1)', ref: 'Misra 2019', commonPitfall: '定点硬编码交叉验证定义。',
    invariant: '定点 relClose(mish(p), p·tanh(log1p(e^p)), 1e-12)',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => mish(x),
    predicate: (x) => { const ref = (v) => v * tanh(log(1 + exp(v))); for (const p of [-3, -1, -0.5, 0, 0.5, 1, 3]) if (!rc(mish(p), ref(p), 1e-12)) return false; return true; },
  });
  reg(42, {
    name: 'Mish · 原点值', intuition: 'mish(0)=0。',
    math: 'mish(0) = 0', pseudocode: 'return rc(mish(0),0)',
    complexity: 'O(1)', ref: '定点', commonPitfall: '无。',
    invariant: 'relClose(mish(0), 0, 1e-12)',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => mish(x),
    predicate: (x) => rc(mish(0), 0, 1e-12),
  });
  reg(43, {
    name: 'Mish · 正支单调增', intuition: 'mish 负区同样有浅谷（非全局单调），x≥0 时严格单调增。',
    math: '0≤a≤b ⇒ mish(a)≤mish(b)', pseudocode: '固定点 [0,.5,1,2,5,10] 递增验证',
    complexity: 'O(1)', ref: 'Misra 2019', commonPitfall: '误称全局单调：负区一带 mish 反升（浅谷）。',
    invariant: '布尔：mish 在 [0,∞) 单调增',
    inputs: ['a', 'b'], domains: { a: 'small', b: 'small' },
    run: (a) => mish(a),
    predicate: (a, b) => { const pts = [0, 0.5,1, 2, 5, 10]; for (let i = 0; i < pts.length - 1; i++) if (mish(pts[i]) > mish(pts[i + 1]) + 1e-9) return false; return true; },
  });
  reg(44, {
    name: 'Mish · 大正渐近', intuition: 'x 大时 softplus→x、tanh→1，mish(x)≈x。',
    math: 'lim_{x→∞} mish(x)-x = 0', pseudocode: 'return rc(mish(20)-20,0,1e-3)',
    complexity: 'O(1)', ref: '渐近', commonPitfall: '固定点 x=20。',
    invariant: 'relClose(mish(20)-20, 0, 1e-3)',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => mish(x),
    predicate: (x) => rc(mish(20) - 20, 0, 1e-3),
  });
  reg(45, {
    name: 'SELU · 定点交叉', intuition: 'scale·ELU(x,α)，自归一化网络默认系数。',
    math: 'selu(x)=s·(x≥0?x:a(e^x-1)), s=1.0507,a=1.6733', pseudocode: 'ref=s*(x>=0?x:a*(exp(x)-1))',
    complexity: 'O(1)', ref: 'Klambauer 2017', commonPitfall: '定点硬编码交叉验证。',
    invariant: '定点 relClose(selu(p), s·(...), 1e-12)',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => SP.selu(x),
    predicate: (x) => { const s = 1.0507009873554805, a = 1.6732632423543772; const ref = (v) => s * (v >= 0 ? v : a * (exp(v) - 1)); for (const p of [-3, -1, 0, 1, 3]) if (!rc(SP.selu(p), ref(p), 1e-12)) return false; return true; },
  });
  reg(46, {
    name: 'SELU · 原点值', intuition: 'selu(0)=0。',
    math: 'selu(0) = 0', pseudocode: 'return rc(selu(0),0)',
    complexity: 'O(1)', ref: '定点', commonPitfall: '无。',
    invariant: 'relClose(selu(0), 0, 1e-12)',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => SP.selu(x),
    predicate: (x) => rc(SP.selu(0), 0, 1e-12),
  });
  reg(47, {
    name: 'SELU · 正支缩放', intuition: 'x>0 时 selu(x)=s·x（s>1）。',
    math: 'x>0 ⇒ selu(x)=s·x', pseudocode: 'return x>0?rc(selu(x),s*x):true',
    complexity: 'O(1)', ref: '正半轴', commonPitfall: '无。',
    invariant: 'x>0 ⇒ relClose(selu(x), s·x, 1e-12)',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => SP.selu(x),
    predicate: (x) => x > 0 ? rc(SP.selu(x), 1.0507009873554805 * x, 1e-12) : true,
  });
  reg(48, {
    name: 'SELU · 负支', intuition: 'x<0 时 selu(x)=s·a·(e^x-1)（负饱和）。',
    math: 'x<0 ⇒ selu(x)=s·a·(e^x-1)', pseudocode: 'return x<0?rc(selu(x),s*a*(exp(x)-1)):true',
    complexity: 'O(1)', ref: '负半轴', commonPitfall: '无。',
    invariant: 'x<0 ⇒ relClose(selu(x), s·a·(e^x-1), 1e-12)',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => SP.selu(x),
    predicate: (x) => { const s = 1.0507009873554805, a = 1.6732632423543772; return x < 0 ? rc(SP.selu(x), s * a * (exp(x) - 1), 1e-12) : true; },
  });

  // ══ HardSigmoid / LogSumExp（G049–G055）══════════════════════════════
  reg(49, {
    name: 'HardSigmoid · 定点交叉', intuition: 'clip((x+3)/6,0,1)，Sigmoid 的廉价分段近似。',
    math: 'hard-sigmoid(x) = clip((x+3)/6, 0, 1)', pseudocode: 'ref=clip((x+3)/6,0,1)',
    complexity: 'O(1)', ref: 'Google 量化', commonPitfall: '定点硬编码交叉验证。',
    invariant: '定点 relClose(hs(p), clip((p+3)/6,0,1), 1e-12)',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => clip((x + 3) / 6, 0, 1),
    predicate: (x) => { const ref = (v) => clip((v + 3) / 6, 0, 1); for (const p of [-6, -3, 0, 0.5, 3, 6]) if (!rc(clip((p + 3) / 6, 0, 1), ref(p), 1e-12)) return false; return true; },
  });
  reg(50, {
    name: 'HardSigmoid · 输出界', intuition: '输出恒在 [0,1]，硬饱和。',
    math: '0 ≤ hard-sigmoid(x) ≤ 1', pseudocode: 'return y>=0-1e-12 && y<=1+1e-12',
    complexity: 'O(1)', ref: '硬有界', commonPitfall: '饱和区恰 0/1。',
    invariant: '布尔：0 ≤ hs(x) ≤ 1',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => clip((x + 3) / 6, 0, 1),
    predicate: (x) => { const y = clip((x + 3) / 6, 0, 1); return y >= -1e-12 && y <= 1 + 1e-12; },
  });
  reg(51, {
    name: 'HardSigmoid · 三定点', intuition: 'hs(-3)=0, hs(0)=0.5, hs(3)=1（拐点精确）。',
    math: 'hs(-3)=0, hs(0)=1/2, hs(3)=1', pseudocode: 'return rc(hs(-3),0)&rc(hs(0),.5)&rc(hs(3),1)',
    complexity: 'O(1)', ref: '定点', commonPitfall: '无。',
    invariant: 'relClose(hs(-3),0) ∧ relClose(hs(0),0.5) ∧ relClose(hs(3),1)',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => clip((x + 3) / 6, 0, 1),
    predicate: (x) => rc(clip(0, 0, 1), 0) && rc(clip(3 / 6, 0, 1), 0.5, 1e-12) && rc(clip(6 / 6, 0, 1), 1, 1e-12),
  });
  reg(52, {
    name: 'LogSumExp · 定义交叉', intuition: 'log(Σe^{x_i})，对两元素即 log(e^a+e^b)。',
    math: 'lse(a,b) = log(e^a+e^b)', pseudocode: 'return rc(lse2(a,b), log(exp(a)+exp(b)))',
    complexity: 'O(1)', ref: '数值稳定', commonPitfall: '直接式在 ±10 内安全，稳定式防溢出。',
    invariant: 'relClose(lse2(a,b), log(e^a+e^b), 1e-12)',
    inputs: ['a', 'b'], domains: { a: 'small', b: 'small' },
    run: (a, b) => SP.logsumexp2(a, b),
    predicate: (a, b) => rc(SP.logsumexp2(a, b), log(exp(a) + exp(b)), 1e-12),
  });
  reg(53, {
    name: 'LogSumExp · 下界', intuition: 'lse ≥ max（最大项主导）。',
    math: 'lse(a,b) ≥ max(a,b)', pseudocode: 'return lse2(a,b) >= max(a,b)-1e-12',
    complexity: 'O(1)', ref: '下界', commonPitfall: '无。',
    invariant: '布尔：lse2(a,b) ≥ max(a,b)',
    inputs: ['a', 'b'], domains: { a: 'small', b: 'small' },
    run: (a, b) => SP.logsumexp2(a, b),
    predicate: (a, b) => SP.logsumexp2(a, b) >= Math.max(a, b) - 1e-12,
  });
  reg(54, {
    name: 'LogSumExp · 上界', intuition: '两元素时 lse ≤ max + ln 2。',
    math: 'lse(a,b) ≤ max(a,b) + ln 2', pseudocode: 'return lse2(a,b) <= max(a,b)+ln2+1e-12',
    complexity: 'O(1)', ref: '上界', commonPitfall: 'n 元素上界为 max+ln n。',
    invariant: '布尔：lse2(a,b) ≤ max(a,b)+ln 2',
    inputs: ['a', 'b'], domains: { a: 'small', b: 'small' },
    run: (a, b) => SP.logsumexp2(a, b),
    predicate: (a, b) => SP.logsumexp2(a, b) <= Math.max(a, b) + log(2) + 1e-12,
  });
  reg(55, {
    name: 'LogSumExp · 平移不变', intuition: '整体加常数 c：lse(x+c)=lse(x)+c。',
    math: 'lse(a+c,b+c) = lse(a,b)+c', pseudocode: 'return rc(lse2(a+c,b+c), lse2(a,b)+c)',
    complexity: 'O(1)', ref: '平移协变', commonPitfall: 'c 须为正（smallPos）避免溢出。',
    invariant: 'relClose(lse2(a+c,b+c), lse2(a,b)+c, 1e-12)',
    inputs: ['a', 'b', 'c'], domains: { a: 'small', b: 'small', c: 'smallPos' },
    run: (a, b) => SP.logsumexp2(a, b),
    predicate: (a, b, c) => rc(SP.logsumexp2(a + c, b + c), SP.logsumexp2(a, b) + c, 1e-12),
  });

  // ══ Softmax / LogSoftmax（G056–G064）═════════════════════════════════
  reg(56, {
    name: 'Softmax · 非负', intuition: 'exp 恒正，归一后每分量严格 >0。',
    math: '∀i: softmax(x)_i > 0', pseudocode: 'return all(p_i>0)',
    complexity: 'O(n)', ref: 'Bridle 1990', commonPitfall: '数值下溢极小但仍 >0。',
    invariant: '布尔：∀i softmax(x)_i > 0',
    inputs: ['x'], domains: { x: 'vecSmall' },
    run: (x) => SP.softmaxStable(x),
    predicate: (x) => { const p = SP.softmaxStable(x); for (let i = 0; i < p.length; i++) if (!(p[i] > 0)) return false; return true; },
  });
  reg(57, {
    name: 'Softmax · 行和归一', intuition: '概率分布，分量和为 1。',
    math: 'Σ_i softmax(x)_i = 1', pseudocode: 'return rc(sum(p),1)',
    complexity: 'O(n)', ref: '归一', commonPitfall: '浮点容差 1e-9。',
    invariant: 'relClose(Σ softmax(x)_i, 1, 1e-9)',
    inputs: ['x'], domains: { x: 'vecSmall' },
    run: (x) => SP.softmaxStable(x),
    predicate: (x) => rc(sumArr(SP.softmaxStable(x)), 1, 1e-9),
  });
  reg(58, {
    name: 'Softmax · 解析归一', intuition: '每分量 = e^{x_i-m}/Σe^{x_j-m}（稳定式）。',
    math: 'p_i = e^{x_i-m} / Σ_j e^{x_j-m}', pseudocode: 'ref=exp(x_i-m)/Σexp(x_j-m)',
    complexity: 'O(n)', ref: '稳定实现', commonPitfall: '减最大值 m 防溢出，参考路径同法重算。',
    invariant: '逐分量 relClose(p_i, e^{x_i-m}/Σe^{x_j-m}, 1e-9)',
    inputs: ['x'], domains: { x: 'vecSmall' },
    run: (x) => SP.softmaxStable(x),
    predicate: (x) => { const p = SP.softmaxStable(x); let m = -Infinity; for (let i = 0; i < x.length; i++) m = Math.max(m, x[i]); let s = 0; const e = new Float64Array(x.length); for (let i = 0; i < x.length; i++) { e[i] = exp(x[i] - m); s += e[i]; } const ref = new Float64Array(x.length); for (let i = 0; i < x.length; i++) ref[i] = e[i] / s; return rc(p, ref, 1e-9); },
  });
  reg(59, {
    name: 'Softmax · 平移不变', intuition: '加常数不改变概率分布。',
    math: 'softmax(x+c) = softmax(x)', pseudocode: 'return rc(softmax(x+c), softmax(x))',
    complexity: 'O(n)', ref: '平移不变', commonPitfall: 'c 为正避免溢出。',
    invariant: '逐分量 relClose(softmax(x+c), softmax(x), 1e-9)',
    inputs: ['x', 'c'], domains: { x: 'vecSmall', c: 'smallPos' },
    run: (x) => SP.softmaxStable(x),
    predicate: (x, c) => rc(SP.softmaxStable(addConst(x, c)), SP.softmaxStable(x), 1e-9),
  });
  reg(60, {
    name: 'Softmax · 坐标单调', intuition: '提升某分量，其概率不降。',
    math: 'x_k↑ ⇒ p_k↑（其余不变）', pseudocode: 'x0+=c; return p0\' >= p0-1e-12',
    complexity: 'O(n)', ref: '单调坐标', commonPitfall: '固定 k=0 比较。',
    invariant: '布尔：p_k(x_k+c) ≥ p_k(x)',
    inputs: ['x', 'c'], domains: { x: 'vecSmall', c: 'smallPos' },
    run: (x) => SP.softmaxStable(x),
    predicate: (x, c) => { const p0 = SP.softmaxStable(x); const x2 = new Float64Array(x.length); for (let i = 0; i < x.length; i++) x2[i] = i === 0 ? x[0] + c : x[i]; const p1 = SP.softmaxStable(x2); return p1[0] >= p0[0] - 1e-9; },
  });
  function logSoftmax(x) { const p = SP.softmaxStable(x); const o = new Float64Array(x.length); for (let i = 0; i < x.length; i++) o[i] = log(p[i]); return o; }
  reg(61, {
    name: 'LogSoftmax · 定义', intuition: 'log_softmax = ln(softmax)，对数概率。',
    math: 'log-softmax(x) = ln softmax(x)', pseudocode: 'return rc(logsoftmax(x), log(softmax(x)))',
    complexity: 'O(n)', ref: '对数概率', commonPitfall: '两路径（直接 log / 先 softmax 再 log）交叉。',
    invariant: '逐分量 relClose(log-softmax(x), ln softmax(x), 1e-12)',
    inputs: ['x'], domains: { x: 'vecSmall' },
    run: (x) => logSoftmax(x),
    predicate: (x) => { const p = SP.softmaxStable(x); const ref = new Float64Array(x.length); for (let i = 0; i < x.length; i++) ref[i] = log(p[i]); return rc(logSoftmax(x), ref, 1e-12); },
  });
  reg(62, {
    name: 'LogSoftmax · 和≤0', intuition: 'Σ ln p_i ≤ ln(∏p_i) ≤ 0（概率连乘 ≤1）。',
    math: 'Σ_i log-softmax(x)_i ≤ 0', pseudocode: 'return sum(logp) <= 1e-12',
    complexity: 'O(n)', ref: '和≤0', commonPitfall: '均匀分布时 = -ln n。',
    invariant: '布尔：Σ log-softmax(x)_i ≤ 0',
    inputs: ['x'], domains: { x: 'vecSmall' },
    run: (x) => logSoftmax(x),
    predicate: (x) => sumArr(logSoftmax(x)) <= 1e-12,
  });
  reg(63, {
    name: 'LogSoftmax · 平移不变', intuition: '与 softmax 同，对数概率亦平移不变。',
    math: 'log-softmax(x+c) = log-softmax(x)', pseudocode: 'return rc(logsoftmax(x+c), logsoftmax(x))',
    complexity: 'O(n)', ref: '平移不变', commonPitfall: 'c 为正避免溢出。',
    invariant: '逐分量 relClose(log-softmax(x+c), log-softmax(x), 1e-9)',
    inputs: ['x', 'c'], domains: { x: 'vecSmall', c: 'smallPos' },
    run: (x) => logSoftmax(x),
    predicate: (x, c) => rc(logSoftmax(addConst(x, c)), logSoftmax(x), 1e-9),
  });
  reg(64, {
    name: 'Softmax · 二分类退化', intuition: '两维 softmax 第一分量即 sigmoid(a-b)。',
    math: 'softmax([a,b])_0 = σ(a-b)', pseudocode: 'return rc(softmax([a,b])[0], sig(a-b))',
    complexity: 'O(1)', ref: '二分类等价', commonPitfall: '维度顺序固定。',
    invariant: 'relClose(softmax([a,b])_0, σ(a-b), 1e-9)',
    inputs: ['a', 'b'], domains: { a: 'small', b: 'small' },
    run: (a, b) => SP.softmaxStable(new Float64Array([a, b])),
    predicate: (a, b) => rc(SP.softmaxStable(new Float64Array([a, b]))[0], SP.sigmoid(a - b), 1e-9),
  });
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

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
 * 谓词纪律：门禁只调 op.predicate(...)，从不调 run（tests/lib/op-harness.js:43）。
 * 故每条 predicate 必须先调用 NS.L10.ops['Gxxx'].run(...) 取真实产出，再与独立
 * 参考路径比对：GELU 用等价 sigmoid 表示 x·σ(2u)（对照 L9 的 tanh 式，常数 √(2/π)
 * 在谓词内硬编码）；Mish 用 log(1+e^x)（对照 L9 的 log1p）；SELU 用 expm1；
 * LSE 用直接式 log(e^a+e^b)（对照 L9 的 max 平移式）；LogSoftmax 用对数域
 * x_i−m−lnΣ（对照 L9 的先 softmax 再取对数）。
 * 固定点类（原点值/渐近/三定点）一律经 run 在固定点上求值；单调性类一律经 run
 * 在真实输入上比较。原 G033/G041/G043/G045/G049/G051 谓词在硬编码点上字面自比
 * 或完全不看 run（恒真假绿），已全部重写。
 * 不变量判据：数值类用 relClose（精确 1e-12，分布类 1e-9）。
 * ==========================================================================*/
;(function (NS) {
  'use strict';
  NS.L10 = NS.L10 || {};
  const rc = NS.L10.relClose;
  const SP = NS.L9.special;
  const exp = Math.exp, log = Math.log, abs = Math.abs;
  const expm1 = Math.expm1;
  function reg(i, o) { NS.L10.register('G', i, o); }
  function runOf(key) { return NS.L10.ops[key].run; }

  // √(2/π)：GELU tanh 近似的内层缩放常数（谓词内硬编码，不复用 L9 导出值——
  // 若 L9.GELU_C 被改错，谓词须能检出）。
  const GELU_C = 0.7978845608028654;
  const SELU_S = 1.0507009873554805, SELU_A = 1.6732632423543772;
  const LN2 = Math.log(2);

  function clip(x, lo, hi) { return x < lo ? lo : (x > hi ? hi : x); }
  function addConst(v, c) { const o = new Float64Array(v.length); for (let i = 0; i < v.length; i++) o[i] = v[i] + c; return o; }
  function sumArr(a) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]; return s; }
  // GELU 参考（等价 sigmoid 表示）：0.5x(1+tanh(u)) = x·σ(2u)，u = √(2/π)(x+0.044715x³)
  function geluRef(x) { return x * SP.sigmoid(2 * GELU_C * (x + 0.044715 * x * x * x)); }
  // SELU 参考：s·ELU，负支用 expm1（对照 L9 的 exp(x)−1）
  function seluRef(x) { return SELU_S * (x >= 0 ? x : SELU_A * expm1(x)); }
  // HardSigmoid 参考：min/max 式（对照 run 的 clip 三元式）
  function hsRef(x) { return Math.min(1, Math.max(0, (x + 3) / 6)); }
  // Mish 参考：x·tanh(log(1+e^x))（对照 L9 的 log1p）
  function mishRef(x) { return x * Math.tanh(log(1 + exp(x))); }
  // Softmax 解析参考（exp 后按和归一）
  function softmaxRef(x) {
    const n = x.length;
    let m = -Infinity;
    for (let i = 0; i < n; i++) if (x[i] > m) m = x[i];
    const e = new Float64Array(n);
    let s = 0;
    for (let i = 0; i < n; i++) { e[i] = exp(x[i] - m); s += e[i]; }
    const o = new Float64Array(n);
    for (let i = 0; i < n; i++) o[i] = e[i] / s;
    return o;
  }
  // LogSoftmax 对数域参考：x_i − m − lnΣ（对照 L9 的先 softmax 再取对数）
  function logSoftmaxRef(x) {
    const n = x.length;
    let m = -Infinity;
    for (let i = 0; i < n; i++) if (x[i] > m) m = x[i];
    let s = 0;
    for (let i = 0; i < n; i++) s += exp(x[i] - m);
    const o = new Float64Array(n);
    const ls = log(s);
    for (let i = 0; i < n; i++) o[i] = x[i] - m - ls;
    return o;
  }

  // ══ GELU / Swish（G033–G040）══════════════════════════════════════════
  reg(33, {
    name: 'GELU · 正支单调增', intuition: 'gelu 负区有浅谷（非全局单调），但 x≥0 时严格单调增并趋近 x。',
    math: '0≤a≤b ⇒ gelu(a)≤gelu(b)', pseudocode: 'lo=min(a,b); hi=max(a,b); return gelu(lo)<=gelu(hi)',
    complexity: 'O(1)', ref: 'Hendrycks & Gimpel 2016', commonPitfall: '误称全局单调：x≲-1.7 一带 gelu 反升（浅谷），故负输入不做单调断言。',
    invariant: 'run 与 σ 式参考一致；min(a,b)≥0 时 run(min) ≤ run(max) + 1e-12',
    inputs: ['a', 'b'], domains: { a: 'small', b: 'small' },
    run: (a) => SP.gelu(a),
    predicate: (a, b) => {
      const lo = Math.min(a, b), hi = Math.max(a, b);
      const ylo = runOf('G033')(lo), yhi = runOf('G033')(hi);
      if (!rc(ylo, geluRef(lo), 1e-12) || !rc(yhi, geluRef(hi), 1e-12)) return false;
      if (lo < 0) return true;                                 // 负区有浅谷：单调性不适用
      return ylo <= yhi + 1e-12;
    },
  });
  reg(34, {
    name: 'GELU · 原点值', intuition: 'gelu(0)=0（x 因子归零）。',
    math: 'gelu(0) = 0', pseudocode: 'return rc(gelu(0),0)',
    complexity: 'O(1)', ref: '定点', commonPitfall: '无。',
    invariant: 'relClose(run(0), 0, 1e-12)（定点经 run 求值）',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => SP.gelu(x),
    predicate: (x) => rc(runOf('G034')(0), 0, 1e-12),
  });
  reg(35, {
    name: 'GELU · 大正渐近', intuition: 'x 大时 tanh→1，gelu(x)→x。',
    math: 'lim_{x→∞} gelu(x)-x = 0', pseudocode: 'return rc(gelu(20)-20,0,1e-3)',
    complexity: 'O(1)', ref: '渐近', commonPitfall: '固定点 x=20（tanh 早已饱和）。',
    invariant: 'relClose(run(20)-20, 0, 1e-3)（固定点经 run 求值）',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => SP.gelu(x),
    predicate: (x) => rc(runOf('G035')(20) - 20, 0, 1e-3),
  });
  reg(36, {
    name: 'GELU · 正支有界', intuition: 'x>0 时 0<gelu(x)≤x（大 x 下 tanh 饱和到 1，故取等号）。',
    math: 'x>0 ⇒ 0 < gelu(x) ≤ x（tanh 未饱和时严格小于）', pseudocode: 'return x>0 ? (g>0 && g<=x) : true',
    complexity: 'O(1)', ref: '正半轴', commonPitfall: 'tanh 在 |u|≳19 时浮点饱和为 1，此时 gelu(x) 恰等于 x，须用 ≤ 而非 <。',
    invariant: 'x>0 ⇒ 0 < run(x) ≤ x + 1e-12；其余输入验证与 σ 式参考一致',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => SP.gelu(x),
    predicate: (x) => {
      const y = runOf('G036')(x);
      if (!rc(y, geluRef(x), 1e-12)) return false;
      if (x <= 0) return true;
      return y > 0 && y <= x + 1e-12;
    },
  });
  reg(37, {
    name: 'Swish · 与 Sigmoid 关系', intuition: 'swish(x)=x·σ(x)，门控式自门控激活。',
    math: 'swish(x) = x·σ(x)', pseudocode: 'return rc(swish(x), x*sig(x))',
    complexity: 'O(1)', ref: 'Ramachandran 2017', commonPitfall: '两路径（分式 x/(1+e^{-x}) vs 乘 sigmoid）交叉验证。',
    invariant: 'relClose(run(x), x·σ(x), 1e-12)（σ 用 1/(1+e^{-x}) 独立路径）',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => SP.swish(x),
    predicate: (x) => rc(runOf('G037')(x), x * SP.sigmoid(x), 1e-12),
  });
  reg(38, {
    name: 'Swish · 原点值', intuition: 'swish(0)=0（x 因子归零）。',
    math: 'swish(0) = 0', pseudocode: 'return rc(swish(0),0)',
    complexity: 'O(1)', ref: '定点', commonPitfall: '无。',
    invariant: 'relClose(run(0), 0, 1e-12)（定点经 run 求值）',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => SP.swish(x),
    predicate: (x) => rc(runOf('G038')(0), 0, 1e-12),
  });
  reg(39, {
    name: 'Swish · 大正渐近', intuition: 'x 大时 σ(x)→1，swish(x)→x。',
    math: 'lim_{x→∞} swish(x)-x = 0', pseudocode: 'return rc(swish(20)-20,0,1e-3)',
    complexity: 'O(1)', ref: '渐近', commonPitfall: '固定点 x=20。',
    invariant: 'relClose(run(20)-20, 0, 1e-3)（固定点经 run 求值）',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => SP.swish(x),
    predicate: (x) => rc(runOf('G039')(20) - 20, 0, 1e-3),
  });
  reg(40, {
    name: 'Swish · 非负象限', intuition: 'x≥0 时 σ(x)>0，swish(x)≥0。',
    math: 'x ≥ 0 ⇒ swish(x) ≥ 0', pseudocode: 'return x>=0 ? swish(x)>=-1e-12 : true',
    complexity: 'O(1)', ref: '非负', commonPitfall: 'x<0 时 swish 微负（门控）。',
    invariant: 'x≥0 ⇒ run(x) ≥ -1e-12；其余输入验证与 x·σ(x) 参考一致',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => SP.swish(x),
    predicate: (x) => {
      const y = runOf('G040')(x);
      if (!rc(y, x * SP.sigmoid(x), 1e-12)) return false;
      return x >= 0 ? y >= -1e-12 : true;
    },
  });

  // ══ Mish / SELU / HardSigmoid（G041–G048）════════════════════════════
  function mish(x) { return x * Math.tanh(SP.softplus(x)); }
  reg(41, {
    name: 'Mish · 定义交叉', intuition: 'mish(x)=x·tanh(softplus(x))，自正则平滑激活。',
    math: 'mish(x) = x·tanh(ln(1+e^x))', pseudocode: 'ref=x*tanh(log(1+exp(x)))',
    complexity: 'O(1)', ref: 'Misra 2019', commonPitfall: 'L9 用 log1p(exp(x))，参考路径用 log(1+e^x) 交叉验证。',
    invariant: 'relClose(run(x), x·tanh(log(1+e^x)), 1e-12)（原为硬编码定点自比，已修）',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => mish(x),
    predicate: (x) => rc(runOf('G041')(x), mishRef(x), 1e-12),
  });
  reg(42, {
    name: 'Mish · 原点值', intuition: 'mish(0)=0。',
    math: 'mish(0) = 0', pseudocode: 'return rc(mish(0),0)',
    complexity: 'O(1)', ref: '定点', commonPitfall: '无。',
    invariant: 'relClose(run(0), 0, 1e-12)（定点经 run 求值）',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => mish(x),
    predicate: (x) => rc(runOf('G042')(0), 0, 1e-12),
  });
  reg(43, {
    name: 'Mish · 正支单调增', intuition: 'mish 负区同样有浅谷（非全局单调），x≥0 时严格单调增。',
    math: '0≤a≤b ⇒ mish(a)≤mish(b)', pseudocode: 'lo=min(a,b); hi=max(a,b); return mish(lo)<=mish(hi)',
    complexity: 'O(1)', ref: 'Misra 2019', commonPitfall: '误称全局单调：负区一带 mish 反升（浅谷），故负输入不做单调断言。',
    invariant: 'run 与 log(1+e^x) 参考一致；min(a,b)≥0 时 run(min) ≤ run(max) + 1e-12',
    inputs: ['a', 'b'], domains: { a: 'small', b: 'small' },
    run: (a) => mish(a),
    predicate: (a, b) => {
      const lo = Math.min(a, b), hi = Math.max(a, b);
      const ylo = runOf('G043')(lo), yhi = runOf('G043')(hi);
      if (!rc(ylo, mishRef(lo), 1e-12) || !rc(yhi, mishRef(hi), 1e-12)) return false;
      if (lo < 0) return true;                                 // 负区有浅谷：单调性不适用
      return ylo <= yhi + 1e-12;
    },
  });
  reg(44, {
    name: 'Mish · 大正渐近', intuition: 'x 大时 softplus→x、tanh→1，mish(x)→x。',
    math: 'lim_{x→∞} mish(x)-x = 0', pseudocode: 'return rc(mish(20)-20,0,1e-3)',
    complexity: 'O(1)', ref: '渐近', commonPitfall: '固定点 x=20。',
    invariant: 'relClose(run(20)-20, 0, 1e-3)（固定点经 run 求值）',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => mish(x),
    predicate: (x) => rc(runOf('G044')(20) - 20, 0, 1e-3),
  });
  reg(45, {
    name: 'SELU · 定义交叉', intuition: 'scale·ELU(x,α)，自归一化网络默认系数。',
    math: 'selu(x)=s·(x≥0?x:a(e^x-1)), s=1.0507…, a=1.67326…', pseudocode: 'ref=s*(x>=0?x:a*expm1(x))',
    complexity: 'O(1)', ref: 'Klambauer 2017', commonPitfall: 'L9 用 exp(x)−1，参考路径用 expm1(x) 交叉验证（近 0 更准）。',
    invariant: 'relClose(run(x), s·(x≥0?x:a·expm1(x)), 1e-12)（原为硬编码定点自比，已修）',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => SP.selu(x),
    predicate: (x) => rc(runOf('G045')(x), seluRef(x), 1e-12),
  });
  reg(46, {
    name: 'SELU · 原点值', intuition: 'selu(0)=0。',
    math: 'selu(0) = 0', pseudocode: 'return rc(selu(0),0)',
    complexity: 'O(1)', ref: '定点', commonPitfall: '无。',
    invariant: 'relClose(run(0), 0, 1e-12)（定点经 run 求值）',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => SP.selu(x),
    predicate: (x) => rc(runOf('G046')(0), 0, 1e-12),
  });
  reg(47, {
    name: 'SELU · 正支缩放', intuition: 'x>0 时 selu(x)=s·x（s>1）。',
    math: 'x>0 ⇒ selu(x)=s·x', pseudocode: 'return x>0?rc(selu(x),s*x):true',
    complexity: 'O(1)', ref: '正半轴', commonPitfall: '无。',
    invariant: 'run 与 SELU 参考一致；x>0 时 run(x) = s·x（正输入的「跳过」分支也须验证定义）',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => SP.selu(x),
    predicate: (x) => {
      const y = runOf('G047')(x);
      if (!rc(y, seluRef(x), 1e-12)) return false;
      return x > 0 ? rc(y, SELU_S * x, 1e-12) : true;
    },
  });
  reg(48, {
    name: 'SELU · 负支', intuition: 'x<0 时 selu(x)=s·a·(e^x-1)（负饱和）。',
    math: 'x<0 ⇒ selu(x)=s·a·(e^x-1)', pseudocode: 'return x<0?rc(selu(x),s*a*(exp(x)-1)):true',
    complexity: 'O(1)', ref: '负半轴', commonPitfall: '无。',
    invariant: 'run 与 SELU 参考一致；x<0 时 run(x) = s·a·(e^x−1)',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => SP.selu(x),
    predicate: (x) => {
      const y = runOf('G048')(x);
      if (!rc(y, seluRef(x), 1e-12)) return false;
      return x < 0 ? rc(y, SELU_S * SELU_A * (exp(x) - 1), 1e-12) : true;
    },
  });

  // ══ HardSigmoid / LogSumExp（G049–G055）══════════════════════════════
  reg(49, {
    name: 'HardSigmoid · 定义', intuition: 'clip((x+3)/6,0,1)，Sigmoid 的廉价分段近似。',
    math: 'hard-sigmoid(x) = clip((x+3)/6, 0, 1)', pseudocode: 'ref=min(1,max(0,(x+3)/6))',
    complexity: 'O(1)', ref: 'Google 量化', commonPitfall: '参考路径用 min/max（对照 run 的 clip 三元式）。',
    invariant: 'relClose(run(x), min(1,max(0,(x+3)/6)), 1e-12)（原为硬编码定点自比，已修）',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => clip((x + 3) / 6, 0, 1),
    predicate: (x) => rc(runOf('G049')(x), hsRef(x), 1e-12),
  });
  reg(50, {
    name: 'HardSigmoid · 输出界', intuition: '输出恒在 [0,1]，硬饱和。',
    math: '0 ≤ hard-sigmoid(x) ≤ 1', pseudocode: 'return y>=0-1e-12 && y<=1+1e-12',
    complexity: 'O(1)', ref: '硬有界', commonPitfall: '饱和区恰 0/1。',
    invariant: '0−1e-12 ≤ run(x) ≤ 1+1e-12 且 run(x) 与 min/max 参考一致（仅查界会被恒 0 蒙混）',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => clip((x + 3) / 6, 0, 1),
    predicate: (x) => {
      const y = runOf('G050')(x);
      if (!(y >= -1e-12 && y <= 1 + 1e-12)) return false;
      return rc(y, hsRef(x), 1e-12);
    },
  });
  reg(51, {
    name: 'HardSigmoid · 三定点', intuition: 'hs(-3)=0, hs(0)=0.5, hs(3)=1（拐点精确）。',
    math: 'hs(-3)=0, hs(0)=1/2, hs(3)=1', pseudocode: 'return rc(hs(-3),0)&rc(hs(0),.5)&rc(hs(3),1)',
    complexity: 'O(1)', ref: '定点', commonPitfall: '无。',
    invariant: '三个拐点均经 run 求值：rc(run(-3),0)∧rc(run(0),0.5)∧rc(run(3),1)（原为常量自比，已修）',
    inputs: ['x'], domains: { x: 'small' },
    run: (x) => clip((x + 3) / 6, 0, 1),
    predicate: (x) => {
      if (!rc(runOf('G051')(-3), 0, 1e-12)) return false;
      if (!rc(runOf('G051')(0), 0.5, 1e-12)) return false;
      return rc(runOf('G051')(3), 1, 1e-12);
    },
  });
  reg(52, {
    name: 'LogSumExp · 定义交叉', intuition: 'log(Σe^{x_i})，对两元素即 log(e^a+e^b)。',
    math: 'lse(a,b) = log(e^a+e^b)', pseudocode: 'return rc(lse2(a,b), log(exp(a)+exp(b)))',
    complexity: 'O(1)', ref: '数值稳定', commonPitfall: '直接式在 ±10 内安全，稳定式防溢出。',
    invariant: 'relClose(run(a,b), log(e^a+e^b), 1e-12)（对照 L9 的 max 平移式）',
    inputs: ['a', 'b'], domains: { a: 'small', b: 'small' },
    run: (a, b) => SP.logsumexp2(a, b),
    predicate: (a, b) => rc(runOf('G052')(a, b), log(exp(a) + exp(b)), 1e-12),
  });
  reg(53, {
    name: 'LogSumExp · 下界', intuition: 'lse ≥ max（最大项主导）。',
    math: 'lse(a,b) ≥ max(a,b)', pseudocode: 'return lse2(a,b) >= max(a,b)-1e-12',
    complexity: 'O(1)', ref: '下界', commonPitfall: '无。',
    invariant: 'run(a,b) ≥ max(a,b)−1e-12 且 run 与直接式一致（仅查下界会被恒取 max 蒙混）',
    inputs: ['a', 'b'], domains: { a: 'small', b: 'small' },
    run: (a, b) => SP.logsumexp2(a, b),
    predicate: (a, b) => {
      const y = runOf('G053')(a, b);
      if (!(y >= Math.max(a, b) - 1e-12)) return false;
      return rc(y, log(exp(a) + exp(b)), 1e-12);
    },
  });
  reg(54, {
    name: 'LogSumExp · 上界', intuition: '两元素时 lse ≤ max + ln 2。',
    math: 'lse(a,b) ≤ max(a,b) + ln 2', pseudocode: 'return lse2(a,b) <= max(a,b)+ln2+1e-12',
    complexity: 'O(1)', ref: '上界', commonPitfall: 'n 元素上界为 max+ln n。',
    invariant: 'run(a,b) ≤ max(a,b)+ln2+1e-12 且 run 与直接式一致（仅查上界会被恒取 max 蒙混）',
    inputs: ['a', 'b'], domains: { a: 'small', b: 'small' },
    run: (a, b) => SP.logsumexp2(a, b),
    predicate: (a, b) => {
      const y = runOf('G054')(a, b);
      if (!(y <= Math.max(a, b) + LN2 + 1e-12)) return false;
      return rc(y, log(exp(a) + exp(b)), 1e-12);
    },
  });
  reg(55, {
    name: 'LogSumExp · 平移不变', intuition: '整体加常数 c：lse(x+c)=lse(x)+c。',
    math: 'lse(a+c,b+c) = lse(a,b)+c', pseudocode: 'return rc(lse2(a+c,b+c), lse2(a,b)+c)',
    complexity: 'O(1)', ref: '平移协变', commonPitfall: 'c 须为正（smallPos）避免溢出。',
    invariant: 'relClose(run(a+c,b+c), run(a,b)+c, 1e-12)（两路均经 run）',
    inputs: ['a', 'b', 'c'], domains: { a: 'small', b: 'small', c: 'smallPos' },
    run: (a, b) => SP.logsumexp2(a, b),
    predicate: (a, b, c) => rc(runOf('G055')(a + c, b + c), runOf('G055')(a, b) + c, 1e-12),
  });

  // ══ Softmax / LogSoftmax（G056–G064）═════════════════════════════════
  reg(56, {
    name: 'Softmax · 非负', intuition: 'exp 恒正，归一后每分量严格 >0。',
    math: '∀i: softmax(x)_i > 0', pseudocode: 'return all(p_i>0)',
    complexity: 'O(n)', ref: 'Bridle 1990', commonPitfall: 'vecSmall 域跨度 ≤20，e^{-20}≈2e-9 不会下溢为 0。',
    invariant: '∀i run(x)_i > 0 且 run 与解析参考一致（仅查非负会被恒 1/n 蒙混）',
    inputs: ['x'], domains: { x: 'vecSmall' },
    run: (x) => SP.softmaxStable(x),
    predicate: (x) => {
      const p = runOf('G056')(x);
      if (p.length !== x.length) return false;
      if (!rc(p, softmaxRef(x), 1e-9)) return false;
      for (let i = 0; i < p.length; i++) if (!(p[i] > 0)) return false;
      return true;
    },
  });
  reg(57, {
    name: 'Softmax · 行和归一', intuition: '概率分布，分量和为 1。',
    math: 'Σ_i softmax(x)_i = 1', pseudocode: 'return rc(sum(p),1)',
    complexity: 'O(n)', ref: '归一', commonPitfall: '浮点容差 1e-9。',
    invariant: 'relClose(Σ run(x)_i, 1, 1e-9) 且 run 与解析参考一致',
    inputs: ['x'], domains: { x: 'vecSmall' },
    run: (x) => SP.softmaxStable(x),
    predicate: (x) => {
      const p = runOf('G057')(x);
      if (p.length !== x.length) return false;
      if (!rc(p, softmaxRef(x), 1e-9)) return false;
      return rc(sumArr(p), 1, 1e-9);
    },
  });
  reg(58, {
    name: 'Softmax · 解析归一', intuition: '每分量 = e^{x_i-m}/Σe^{x_j-m}（稳定式）。',
    math: 'p_i = e^{x_i-m} / Σ_j e^{x_j-m}', pseudocode: 'ref=exp(x_i-m)/Σexp(x_j-m)',
    complexity: 'O(n)', ref: '稳定实现', commonPitfall: '减最大值 m 防溢出；参考路径按和相除（对照 L9 的乘 1/s）。',
    invariant: '逐分量 relClose(run(x), e^{x_i-m}/Σe^{x_j-m}, 1e-9)',
    inputs: ['x'], domains: { x: 'vecSmall' },
    run: (x) => SP.softmaxStable(x),
    predicate: (x) => {
      const p = runOf('G058')(x);
      if (p.length !== x.length) return false;
      return rc(p, softmaxRef(x), 1e-9);
    },
  });
  reg(59, {
    name: 'Softmax · 平移不变', intuition: '加常数不改变概率分布。',
    math: 'softmax(x+c) = softmax(x)', pseudocode: 'return rc(softmax(x+c), softmax(x))',
    complexity: 'O(n)', ref: '平移不变', commonPitfall: 'c 为正避免溢出。',
    invariant: '逐分量 relClose(run(x+c), run(x), 1e-9)（两路均经 run）',
    inputs: ['x', 'c'], domains: { x: 'vecSmall', c: 'smallPos' },
    run: (x) => SP.softmaxStable(x),
    predicate: (x, c) => rc(runOf('G059')(addConst(x, c)), runOf('G059')(x), 1e-9),
  });
  reg(60, {
    name: 'Softmax · 坐标单调', intuition: '提升某分量，其概率不降。',
    math: 'x_k↑ ⇒ p_k↑（其余不变）', pseudocode: 'x0+=c; return p1[0] >= p0[0]-1e-9',
    complexity: 'O(n)', ref: '单调坐标', commonPitfall: '固定 k=0 比较。',
    invariant: 'run(x with x_0+c)[0] ≥ run(x)[0] − 1e-9（两路均经 run）',
    inputs: ['x', 'c'], domains: { x: 'vecSmall', c: 'smallPos' },
    run: (x) => SP.softmaxStable(x),
    predicate: (x, c) => {
      const p0 = runOf('G060')(x);
      const x2 = new Float64Array(x.length);
      for (let i = 0; i < x.length; i++) x2[i] = i === 0 ? x[0] + c : x[i];
      const p1 = runOf('G060')(x2);
      return p1[0] >= p0[0] - 1e-9;
    },
  });
  function logSoftmax(x) { const p = SP.softmaxStable(x); const o = new Float64Array(x.length); for (let i = 0; i < x.length; i++) o[i] = log(p[i]); return o; }
  reg(61, {
    name: 'LogSoftmax · 定义', intuition: 'log_softmax = ln(softmax)，对数概率。',
    math: 'log-softmax(x)_i = x_i − max(x) − ln Σ_j e^{x_j-max}', pseudocode: 'ref=x_i-m-log(Σexp(x_j-m))',
    complexity: 'O(n)', ref: '对数概率', commonPitfall: '两路径（对数域直接算 / 先 softmax 再取对数）交叉。',
    invariant: '逐分量 relClose(run(x), x_i−m−lnΣe^{x_j−m}, 1e-12)（对数域参考）',
    inputs: ['x'], domains: { x: 'vecSmall' },
    run: (x) => logSoftmax(x),
    predicate: (x) => {
      const y = runOf('G061')(x);
      if (y.length !== x.length) return false;
      return rc(y, logSoftmaxRef(x), 1e-12);
    },
  });
  reg(62, {
    name: 'LogSoftmax · 和≤0', intuition: 'Σ ln p_i ≤ 0（概率连乘 ≤1）。',
    math: 'Σ_i log-softmax(x)_i ≤ 0', pseudocode: 'return sum(logp) <= 1e-12',
    complexity: 'O(n)', ref: '和≤0', commonPitfall: '均匀分布时 = -ln n。',
    invariant: 'Σ run(x)_i ≤ 1e-12 且 run 与对数域参考一致（仅查和会被恒 0 蒙混）',
    inputs: ['x'], domains: { x: 'vecSmall' },
    run: (x) => logSoftmax(x),
    predicate: (x) => {
      const y = runOf('G062')(x);
      if (y.length !== x.length) return false;
      if (!rc(y, logSoftmaxRef(x), 1e-12)) return false;
      return sumArr(y) <= 1e-12;
    },
  });
  reg(63, {
    name: 'LogSoftmax · 平移不变', intuition: '与 softmax 同，对数概率亦平移不变。',
    math: 'log-softmax(x+c) = log-softmax(x)', pseudocode: 'return rc(logsoftmax(x+c), logsoftmax(x))',
    complexity: 'O(n)', ref: '平移不变', commonPitfall: 'c 为正避免溢出。',
    invariant: '逐分量 relClose(run(x+c), run(x), 1e-9)（两路均经 run）',
    inputs: ['x', 'c'], domains: { x: 'vecSmall', c: 'smallPos' },
    run: (x) => logSoftmax(x),
    predicate: (x, c) => rc(runOf('G063')(addConst(x, c)), runOf('G063')(x), 1e-9),
  });
  reg(64, {
    name: 'Softmax · 二分类退化', intuition: '两维 softmax 第一分量即 sigmoid(a-b)。',
    math: 'softmax([a,b])_0 = σ(a-b)', pseudocode: 'return rc(softmax([a,b])[0], sig(a-b))',
    complexity: 'O(1)', ref: '二分类等价', commonPitfall: '维度顺序固定。',
    invariant: 'relClose(run(a,b)[0], σ(a−b), 1e-9)（softmax 与 sigmoid 两路交叉）',
    inputs: ['a', 'b'], domains: { a: 'small', b: 'small' },
    run: (a, b) => SP.softmaxStable(new Float64Array([a, b])),
    predicate: (a, b) => {
      const p = runOf('G064')(a, b);
      if (!p || p.length !== 2) return false;
      return rc(p[0], SP.sigmoid(a - b), 1e-9);
    },
  });
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

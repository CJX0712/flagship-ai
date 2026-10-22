/* ============================================================================
 * L10 微指令 · G 域（激活与损失）· 第 4/4 批（G097–G128）
 * ----------------------------------------------------------------------------
 * 所属层：L10-microIsa  域：G（激活与损失）
 * 允许依赖的层：L9（special）、L10（registry/invariant）
 * 主要 @ref：Goodfellow (2016) 数值稳定；Paszke et al. (2019)；He et al. (2016)
 *           ResNet；Schroff et al. (2015) Triplet；Hadsell et al. (2006) Contrastive
 *
 * 涵盖：数值稳定交叉熵 G097–G112；高级/组合 G113–G128。
 * 约定：概率用 vecProb；分类下标用 smallInt 取模；温度/margin/eps 用 smallPos；
 *       二值标签用 vecMask 映射 ±1/-1。
 *
 * 谓词纪律（本批全部重写）：门禁只调 op.predicate(...)，从不调 run
 * （tests/lib/op-harness.js:43、scripts/probe-microisa.js:106）。故每条 predicate
 * 必须先经 NS.L10.ops['Gxxx'].run(...) 取真实产出，再与「反向累加 + 不同表达形式」
 * 的独立参考路径比对（基线），最后才验证该算子声明的性质。禁止把 run 的定义式
 * 抄进 predicate 充当参考（那只能证明 run 等于它自己）。
 * 原 32 项中：G097 的 x>700 分支、G104/G111/G119/G120/G127 为纯区间/界断言（无
 * oracle）；G099 参考与 run 同式；G103 用 rc(r, -tanh(-x))（sin 亦满足，恒真）；
 * G105 用 1-σ(-x) 恒真式；G113 参考与 run 定义式字面相同；G114/G115/G116/G117/
 * G121/G122/G123/G124 的参考循环逐字复制 run；G118 只断言 r<=fullCE。
 * 文档纠错：G103 的 |tanh(x)|<1 只在 |x|<=18 成立（|x|>=19 时 double 饱和为 ±1）；
 * G104 的 exp 在 x>709 溢出为 +Inf、x<-745 下溢为 0（原文档只提「符号区间」）。
 *
 * run 修复（三处，均为真实数值缺陷而非谓词放宽）：
 *   ① G117 温度缩放 CE：原实现 softmax 后 log(max(p,1e-12))，温度极小
 *      （T∈[1e-6,10]，z/T 可达 1e7）时 softmax 分量下溢为 0，损失被截断在
 *      27.63。改为 log 域 -(z_t/T - LSE(z/T))，与 G101/G110 路径统一。
 *   ② G120 对比损失：原用 ‖x‖²+‖y‖²-2x·y 展开式求 ‖x-y‖，x≈y 时相消放大。
 *      改为直接 Σ(x_i-y_i)²。
 *   ③ G121 triplet loss：同理，d_ap/d_an 改为直接 Σ(a_i-p_i)² / Σ(a_i-n_i)²。
 *
 * 数值安全边界：vecSmall∈[-10,10]、vecProb∈[0,1] 且和为 1，故 Σexp(z) <= 8·e^10
 * ≈ 1.76e5，直接式 ln Σexp(z) 永不溢出 —— 参考路径据此可用「无 max 平移的直接式」
 * 与 run 的 max 平移式形成真正的异构比对。
 * ==========================================================================*/
;(function (NS) {
  'use strict';
  NS.L10 = NS.L10 || {};
  function reg(i, o) { NS.L10.register('G', i, o); }
  var S = NS.L9.special;
  var rc = function (a, b, t) { return NS.L10.relClose(a, b, t); };
  function runOf(key) { return NS.L10.ops[key].run; }
  var LOG2 = Math.log(2);
  var PC = 1e-12;                        // 概率 clip 下界（run 内部约定）
  var LOGPC = Math.log(1e-12);

  // ── 基元 ────────────────────────────────────────────────────────────────
  function clip(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }
  function dotv(a, b) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
  function argmaxFirst(v) { var m = -Infinity, idx = 0; for (var i = 0; i < v.length; i++) if (v[i] > m) { m = v[i]; idx = i; } return idx; }
  function safeIdx(t, n) { t = Math.round(t); t = ((t % n) + n) % n; return t; }
  function signLabel(y) { return y === 1 ? 1 : -1; }

  // ── 独立参考路径（一律反向累加，且表达形式与 run 不同）─────────────────
  // 直接式 LSE：ln Σexp(z_i)（vecSmall 域内 Σ<=1.76e5，安全）
  function lseDirect(z) {
    var s = 0;
    for (var i = z.length - 1; i >= 0; i--) s += Math.exp(z[i]);
    return Math.log(s);
  }
  // KL 参考：分离对数 log p - log q（对照 run 的 log(p/q) 比值式），反向累加
  function klRef(p, q) {
    var s = 0;
    for (var i = p.length - 1; i >= 0; i--) {
      var pi = Math.max(p[i], PC), qi = Math.max(q[i], PC);
      s += pi * (Math.log(pi) - Math.log(qi));
    }
    return s;
  }
  // 熵参考：反向累加，p 侧同样 clip（与 run 的 entropy 约定一致）
  function entRef(p) {
    var s = 0;
    for (var i = p.length - 1; i >= 0; i--) { var pi = Math.max(p[i], PC); s += -pi * Math.log(pi); }
    return s;
  }
  // softmax 参考：log 域 exp(z_i - LSE直接式)（对照 run 的 exp(z_i-m)/Σ）
  function softmaxRef(z) {
    var lse = lseDirect(z), o = new Float64Array(z.length);
    for (var i = 0; i < z.length; i++) o[i] = Math.exp(z[i] - lse);
    return o;
  }
  // argmax 参考：两遍扫描（先求 max，再定位首个等于 max 的下标）
  function argmaxRef(v) {
    var mx = -Infinity;
    for (var i = 0; i < v.length; i++) if (v[i] > mx) mx = v[i];
    for (var j = 0; j < v.length; j++) if (v[j] === mx) return j;
    return 0;
  }
  // sigmoid 参考：tanh 形式 (1+tanh(x/2))/2（对照 run 的 1/(1+e^{-x})）
  function sigmoidTanh(x) { return 0.5 * (1 + Math.tanh(x / 2)); }
  // top-k 选取参考：O(n²) 稳定排名 rank_i = #{z_j>z_i} + #{j<i: z_j===z_i}，
  // 取 rank < kk 者。复现 JS 稳定排序的并列语义，但算法完全不同。
  function topkSet(z, kk) {
    var n = z.length, out = [];
    for (var i = 0; i < n; i++) {
      var rank = 0;
      for (var j = 0; j < n; j++) {
        if (z[j] > z[i]) rank++;
        else if (z[j] === z[i] && j < i) rank++;
      }
      if (rank < kk) out.push(z[i]);
    }
    return out;
  }
  function lseValsDirect(vals) {   // 直接式 ln Σexp（对 top-k 子集，元素 ∈[-10,10]）
    var s = 0;
    for (var i = vals.length - 1; i >= 0; i--) s += Math.exp(vals[i]);
    return Math.log(s);
  }
  function fullCE(z, kt) { return lseDirect(z) - z[kt]; }   // 完整 CE（构造断言用）
  function sqDist(a, b) {          // 直接式 Σ(a_i-b_i)²，反向累加
    var s = 0;
    for (var i = a.length - 1; i >= 0; i--) { var d = a[i] - b[i]; s += d * d; }
    return s;
  }
  // floor(log2(v))，对次正规数也正确（G104 次正规区的唯一可用 oracle）
  function ilog2(v) {
    var e = 0;
    while (v < 1) { v *= 2; e--; }
    while (v >= 2) { v /= 2; e++; }
    return e;
  }

  // ════════════════════════ 数值稳定交叉熵 G097–G112 ════════════════════════
  var f_expclip = function (x) { return x > 700 ? Number.MAX_VALUE : Math.exp(x); };
  reg(97, {
    name: '稳定exp-clip', intuition: '指数运算在大正输入处裁剪以防溢出为 Inf。',
    math: 'exp(x)（x>700 裁剪到 FINITE_MAX）', pseudocode: 'return x>700?MAX:exp(x)', complexity: 'O(1)',
    ref: 'IEEE 754 / 数值稳定', commonPitfall: '裁剪会牺牲大正输入精度（极少触发）；x<-745 时 exp 下溢为 0。',
    invariant: 'run 有限且 >=0；|x|<=700 时 relClose exp(x)；x>700 恒等于 FINITE_MAX（构造输入验证，不放行）',
    inputs: ['x'], domains: { x: 'finite' },
    run: f_expclip,
    predicate: function (x) {
      var r = runOf('G097')(x);
      if (!(Number.isFinite(r) && r >= 0)) return false;
      if (x > 700) return r === Number.MAX_VALUE;                 // 裁剪分支：断言真值，而非放行
      if (!rc(r, Math.exp(x), 1e-12)) return false;
      // 反向 oracle：对数还原。exp(x) 与 x 在 [-700,700] 内均为正规数，log 可精确还原
      if (x >= -700 && !rc(Math.log(r), x, 1e-12)) return false;
      // 构造输入喂 run：验证裁剪阈值两侧（700 不裁、701 裁）
      if (!rc(runOf('G097')(700), Math.exp(700), 1e-15)) return false;
      if (runOf('G097')(701) !== Number.MAX_VALUE) return false;
      return true;
    },
  });

  var f_log1p = function (x) { return Math.log1p(x); };
  reg(98, {
    name: '稳定log1p', intuition: 'log(1+x) 对小 x 用补偿算法，避免灾难性抵消。',
    math: 'log1p(x)=ln(1+x)', pseudocode: 'return Math.log1p(x)', complexity: 'O(1)',
    ref: 'Cody & Waite 1980', commonPitfall: 'x<-1 致 NaN（此处域 x>=0 保证安全）。',
    invariant: 'run 与 log(1+x) 一致；run 反向经 expm1 还原 x（含 x=1e-15 的相消区）',
    inputs: ['x'], domains: { x: 'nonNegSmall' },
    run: f_log1p,
    predicate: function (x) {
      var r = runOf('G098')(x);
      if (!rc(r, Math.log(1 + x), 1e-12)) return false;
      // 独立 oracle：expm1(log1p(x)) = x。log(1+x) 在 x<1e-8 会被 1+x 的舍入吃掉精度，
      // expm1 还原路径不经过加法，故能真正区分 log1p 与朴素 log(1+x) 实现。
      if (!rc(Math.expm1(r), x, 1e-12)) return false;
      // 构造输入喂 run：极小 x 下 log1p 仍须保精度（朴素实现此处会返回 0）
      if (!rc(runOf('G098')(1e-15), 1e-15, 1e-15)) return false;
      return rc(runOf('G098')(0), 0, 1e-15);
    },
  });

  var f_log1pexp = function (x) { return Math.log1p(Math.exp(x)); };
  reg(99, {
    name: '稳定log(1+e^x)=softplus', intuition: 'softplus 即 ln(1+e^x)，对正负 x 均稳定。',
    math: 'ln(1+e^x)=softplus(x)', pseudocode: 'return log1p(exp(x))', complexity: 'O(1)',
    ref: 'Dugas et al. 2001', commonPitfall: '直接 exp 在大正 x 溢出（softplus 内部 max 平移规避）。',
    invariant: 'run 与「x>=0 用 x+log1p(e^{-x})、x<0 用 ln(1+e^x)」异构式一致，且与 L9 softplus 逐位同值',
    inputs: ['x'], domains: { x: 'small' },
    run: f_log1pexp,
    predicate: function (x) {
      var r = runOf('G099')(x);
      var ref = x >= 0 ? x + Math.log1p(Math.exp(-x)) : Math.log(1 + Math.exp(x));
      if (!rc(r, ref, 1e-12)) return false;
      // 跨层一致性：与 L9.special.softplus 逐位同值（两个独立实现互证）
      if (!rc(r, S.softplus(x), 1e-15)) return false;
      // 性质：处处非负；单点经 run 求值
      if (r < 0) return false;
      // 构造输入喂 run：验证 x+log1p(e^{-x}) 分支在大 x 的稳定性
      if (!rc(runOf('G099')(10), 10 + Math.log1p(Math.exp(-10)), 1e-12)) return false;
      // 单调性（两路经 run）
      return runOf('G099')(x + 1) >= r - 1e-15;
    },
  });

  var f_lse = function (z) {
    var m = -Infinity; for (var i = 0; i < z.length; i++) if (z[i] > m) m = z[i];
    var s = 0; for (var j = 0; j < z.length; j++) s += Math.exp(z[j] - m);
    return m + Math.log(s);
  };
  reg(100, {
    name: '稳定softmax分母LSE', intuition: 'log-sum-exp 用 max 平移防溢出，是 softmax 的分母。',
    math: 'LSE(z)=max+ln Σe^{z_i-max}', pseudocode: 'return max+log(sum exp(z-max))', complexity: 'O(n)',
    ref: '数值方法标准', commonPitfall: 'LSE(z)>=max(z)。',
    invariant: 'run 与直接式 ln Σexp(z_i) 一致（异构：run 平移 / 参考不平移）；run>=max(z)',
    inputs: ['z'], domains: { z: 'vecSmall' },
    run: f_lse,
    predicate: function (z) {
      var r = runOf('G100')(z);
      var m = -Infinity; for (var i = 0; i < z.length; i++) if (z[i] > m) m = z[i];
      if (!(r >= m)) return false;                                    // 相对判据（非 m-1e-9 绝对）
      if (!rc(r, lseDirect(z), 1e-12)) return false;                 // 独立参考：无平移直接式
      // 性质：上界 max + ln n
      if (!(r <= m + Math.log(z.length) + 1e-12)) return false;
      // 构造输入喂 run：全零向量 ⇒ LSE = ln n
      var z0 = new Float64Array(z.length);
      return rc(runOf('G100')(z0), Math.log(z.length), 1e-12);
    },
  });

  var f_ce_lse = function (z, t) { var k = safeIdx(t, z.length); return -z[k] + f_lse(z); };
  reg(101, {
    name: '稳定CE logsumexp路径', intuition: '交叉熵经 LSE 稳定计算：L=-z_y+LSE(z)。',
    math: 'CE=-z_y+LSE(z)', pseudocode: 'return -z_y+LSE(z)', complexity: 'O(n)',
    ref: 'Goodfellow 2016', commonPitfall: '与 softmax+NLL 数值一致但更稳定。',
    invariant: 'run 与「ln Σexp(z) - z_t」直接式一致；run>=0；两路经 run 的 NLL(G110) 交叉一致',
    inputs: ['logits', 'target'], domains: { logits: 'vecSmall', target: 'smallInt' },
    run: f_ce_lse,
    predicate: function (logits, target) {
      var r = runOf('G101')(logits, target);
      var k = safeIdx(target, logits.length);
      if (!rc(r, lseDirect(logits) - logits[k], 1e-12)) return false;  // 直接式参考
      if (!(r >= 0)) return false;
      // 交叉：G110（NLL via log-softmax）走 -(z_t - LSE) 的另一写法，两路均经 run
      if (!rc(r, runOf('G110')(logits, target), 1e-12)) return false;
      // softmax 路径等价性：-ln softmax_t（softmax 经 L10 G109 run）
      var sm = runOf('G109')(logits);
      return rc(r, -Math.log(Math.max(sm[k], PC)), 1e-12);
    },
  });

  var f_logclip = function (x) { return Math.log(Math.max(x, 1e-12)); };
  reg(102, {
    name: '防log(0)clip', intuition: '对 log 输入下限裁剪，避免 0 处 -Inf。',
    math: 'log(max(x,ε))', pseudocode: 'return log(max(x,1e-12))', complexity: 'O(1)',
    ref: '数值稳定技巧', commonPitfall: 'ε 须远小于典型输入；本 op 域 x>=1e-6 恒不触发 clip，故 clip 路径只能由构造输入覆盖。',
    invariant: 'run 与 ln(x) 一致；构造输入 x=0 / x=1e-30 经 run 须等于 ln(1e-12)（验证 clip 真的生效）',
    inputs: ['x'], domains: { x: 'smallPos' },
    run: f_logclip,
    predicate: function (x) {
      var r = runOf('G102')(x);
      if (!Number.isFinite(r)) return false;
      if (!rc(r, Math.log(x), 1e-12)) return false;
      // 构造输入喂 run：clip 下界（域 x>=1e-6 永不触发，只能这样覆盖）
      if (!rc(runOf('G102')(0), LOGPC, 1e-15)) return false;
      if (!rc(runOf('G102')(1e-30), LOGPC, 1e-15)) return false;
      return true;
    },
  });

  var f_tanh2 = function (x) { return Math.tanh(x); };
  reg(103, {
    name: '稳定tanh', intuition: 'tanh 在饱和处稳定趋近 ±1，奇函数。',
    math: 'tanh(x)=(e^{2x}-1)/(e^{2x}+1)', pseudocode: 'return tanh(x)', complexity: 'O(1)',
    ref: 'LeCun et al. 1998', commonPitfall: '|tanh(x)|<1 只在 |x|<=18 成立；|x|>=19 时 double 饱和为 ±1。',
    invariant: 'run 与 exp 式 (e^{2x}-1)/(e^{2x}+1) 一致；|x|<=18 时 |run|<1；奇函数两路经 run',
    inputs: ['x'], domains: { x: 'small' },
    run: f_tanh2,
    predicate: function (x) {
      var r = runOf('G103')(x);
      // 独立 oracle：exp 形式（|x|<=10 ⇒ e^{2x}<=4.85e8，安全）
      var e2 = Math.exp(2 * x);
      if (!rc(r, (e2 - 1) / (e2 + 1), 1e-12)) return false;
      // 性质（有域条件）：仅 |x|<=18 时严格 |r|<1；饱和区用构造输入验证 == ±1
      if (Math.abs(x) <= 18) { if (!(Math.abs(r) < 1)) return false; }
      else { if (r !== 1 && r !== -1) return false; }
      // 奇函数：两路均经 run
      if (!rc(r, -runOf('G103')(-x), 1e-15)) return false;
      // 构造输入喂 run：饱和阈值两侧
      return rc(runOf('G103')(0), 0, 1e-15) && rc(runOf('G103')(25), 1, 1e-15);
    },
  });

  var f_exp2 = function (x) { return Math.exp(x); };
  reg(104, {
    name: '稳定exp(大负→0)', intuition: 'exp 对大负输入平滑趋近 0，恒非负。',
    math: 'exp(x)；x>709 溢出为 +Inf，x<-745 下溢为 0', pseudocode: 'return exp(x)', complexity: 'O(1)',
    ref: 'IEEE 754', commonPitfall: '纯区间断言（0<r<=1 / r>=1）无 oracle：任何返回 0.5 的实现都能过。须用 log 反向还原。',
    invariant: 'run 与 exp(x) 一致（log 反向 oracle）；x>709 恒为 +Inf、x<-745 恒为 0；x<0 ⇒ 0<run<=1，x>=0 ⇒ run>=1',
    inputs: ['x'], domains: { x: 'finite' },
    run: f_exp2,
    predicate: function (x) {
      var r = runOf('G104')(x);
      // 溢出区：exp 在 x>709.78 溢出为 +Inf
      if (r === Infinity) return x > 709.7;
      // 下溢区：exp 在 x<-745.2 下溢为 0
      if (r === 0) return x < -745;
      // 次正规区（-745<x<-708.4）：float64 只有约 3 位有效数字，log 不还原到 1e-12，
      // 故此处改用「二进制指数」oracle —— ilog2(r) 必须等于 round(x/ln2)（±1）。
      if (x < -700) return Math.abs(ilog2(r) - x / Math.LN2) <= 1;
      // 正规区主 oracle：log 反向还原（r 为正正规数时 log 精确）
      if (!(r > 0) || !rc(Math.log(r), x, 1e-12)) return false;
      // 性质：符号区间
      if (x < 0) { if (!(r > 0 && r <= 1 + 1e-9)) return false; }
      else { if (!(r >= 1)) return false; }
      // 构造输入喂 run：溢出/下溢阈值
      return runOf('G104')(710) === Infinity && runOf('G104')(-746) === 0;
    },
  });

  var f_sig2 = S.sigmoid;
  reg(105, {
    name: '稳定sigmoid', intuition: 'sigmoid 在饱和处稳定为 0/1。',
    math: 'σ(x)=1/(1+e^{-x})', pseudocode: 'return 1/(1+exp(-x))', complexity: 'O(1)',
    ref: 'Goodfellow 2016', commonPitfall: '0<σ(x)<1 仅在有限 x 成立；rc(r,1-σ(-x)) 是恒真式（σ 定义即如此），不能当 oracle。',
    invariant: 'run 与 tanh 形式 (1+tanh(x/2))/2 一致；σ(0)=0.5；|x|>=37 时 |run-饱和值|<1e-15',
    inputs: ['x'], domains: { x: 'finite' },
    run: f_sig2,
    predicate: function (x) {
      var r = runOf('G105')(x);
      // 独立 oracle：tanh 形式（不同表达，非恒真式）
      if (!rc(r, sigmoidTanh(x), 1e-12)) return false;
      // 性质：值域
      if (!(r >= 0 && r <= 1)) return false;
      if (Math.abs(x) <= 18) { if (!(r > 0 && r < 1)) return false; }
      // 单调性（两路经 run）
      if (runOf('G105')(x + 1) < r - 1e-15) return false;
      // 构造输入喂 run：定点与饱和
      if (runOf('G105')(0) !== 0.5) return false;
      return rc(runOf('G105')(50), 1, 1e-15) && runOf('G105')(-50) < 1e-20;
    },
  });

  var f_logsoftmax2 = function (z, t) { var k = safeIdx(t, z.length); return z[k] - f_lse(z); };
  reg(106, {
    name: '稳定log-softmax', intuition: 'log-softmax 经 LSE 稳定计算，目标处值 ≤0 且单调递减于 logit。',
    math: 'logsm(z)_t=z_t-LSE(z)', pseudocode: 'return z_t-LSE(z)', complexity: 'O(n)',
    ref: 'Goodfellow 2016', commonPitfall: '逐元素 <=0。',
    invariant: 'run 与直接式「ln Σexp(z) - z_t」一致；run<=0；两路经 run 的 CE(G101) 交叉一致',
    inputs: ['logits', 'target'], domains: { logits: 'vecSmall', target: 'smallInt' },
    run: f_logsoftmax2,
    predicate: function (logits, target) {
      var r = runOf('G106')(logits, target);
      var k = safeIdx(target, logits.length);
      if (!rc(r, logits[k] - lseDirect(logits), 1e-12)) return false;
      if (!(r <= 0)) return false;
      // 交叉：CE(G101) = -logsm，两路均经 run
      return rc(-r, runOf('G101')(logits, target), 1e-12);
    },
  });

  var f_kl2 = function (p, q) {
    var s = 0; for (var i = 0; i < p.length; i++) { var pi = Math.max(p[i], PC), qi = Math.max(q[i], PC); s += pi * Math.log(pi / qi); } return s;
  };
  reg(107, {
    name: '稳定KL(eps)', intuition: 'KL 散度对 q 裁剪防 log(0)，保持非负与 p=q 时为 0。',
    math: 'KL(p‖q)=Σp ln(p/q)', pseudocode: 'return sum(p*ln(p/q))', complexity: 'O(n)',
    ref: 'Cover & Thomas 2006', commonPitfall: '非对称。',
    invariant: 'run 与分离对数式 Σp(ln p-ln q) 一致（异构：比值 vs 分离）；非负；KL(p,p)=0 两路经 run',
    inputs: ['p', 'q'], domains: { p: 'vecProb', q: 'vecProb' },
    run: f_kl2,
    predicate: function (p, q) {
      var r = runOf('G107')(p, q);
      if (!Number.isFinite(r)) return false;
      if (!rc(r, klRef(p, q), 1e-9)) return false;
      if (!(r >= 0)) return false;
      // 自反性：两路均经 run
      if (!rc(runOf('G107')(p, p), 0, 1e-12)) return false;
      // 构造输入喂 run：one-hot p 对均匀 q ⇒ KL = ln n。
      // 注：不能用 one-hot 对 one-hot —— q 的 0 位会被 clip 到 1e-12，结果是 ln(1e12)=27.63 而非 ln n。
      var n = p.length, e0 = new Float64Array(n), u = new Float64Array(n).fill(1 / n);
      e0[0] = 1;
      return rc(runOf('G107')(e0, u), Math.log(n), 1e-9);
    },
  });

  var f_js2 = function (p, q) {
    var n = p.length, m = new Float64Array(n);
    for (var i = 0; i < n; i++) m[i] = (p[i] + q[i]) / 2;
    var a = f_kl2(p, m), b = f_kl2(q, m);
    return 0.5 * a + 0.5 * b;
  };
  reg(108, {
    name: '稳定JS(eps)', intuition: 'JS 散度裁剪防 log(0)，对称且有界 [0,ln2]。',
    math: 'JS=0.5KL(p‖m)+0.5KL(q‖m)=H(m)-½(H(p)+H(q))', pseudocode: 'return H(m)-0.5*(H(p)+H(q))', complexity: 'O(n)',
    ref: 'Lin 1991', commonPitfall: '上界 ln2；熵形式与「两次 KL 调用」形式在 clip 项上有 ~1e-11 系统差（容差内）。',
    invariant: 'run 与熵形式参考一致；0<=run<=ln2；对称性两路经 run；JS(p,p)=0 经 run',
    inputs: ['p', 'q'], domains: { p: 'vecProb', q: 'vecProb' },
    run: f_js2,
    predicate: function (p, q) {
      var r = runOf('G108')(p, q);
      // 独立参考：熵形式（与 run 的两次 KL 调用结构完全不同）
      var n = p.length, m = new Float64Array(n);
      for (var i = 0; i < n; i++) m[i] = (p[i] + q[i]) / 2;
      var ref = entRef(m) - 0.5 * (entRef(p) + entRef(q));
      if (!rc(r, ref, 1e-9)) return false;
      if (!(r >= 0 && r <= LOG2 + 1e-9)) return false;
      // 对称性：两路均经 run
      if (!rc(r, runOf('G108')(q, p), 1e-12)) return false;
      // 自反性
      if (!rc(runOf('G108')(p, p), 0, 1e-12)) return false;
      // 构造输入喂 run：正交 one-hot ⇒ JS = ln2（上界取等）
      var e0 = new Float64Array(n), e1 = new Float64Array(n);
      e0[0] = 1; e1[n - 1] = 1;
      return rc(runOf('G108')(e0, e1), LOG2, 1e-9);
    },
  });

  var f_softmax3 = function (z) {
    var m = -Infinity; for (var i = 0; i < z.length; i++) if (z[i] > m) m = z[i];
    var s = 0, o = new Float64Array(z.length);
    for (var j = 0; j < z.length; j++) { o[j] = Math.exp(z[j] - m); s += o[j]; }
    var inv = 1 / s; for (var k = 0; k < z.length; k++) o[k] *= inv;
    return o;
  };
  reg(109, {
    name: '稳定softmax分布', intuition: 'softmax 输出合法概率分布，稳定且 argmax 不变。',
    math: 'p_i=e^{z_i-m}/Σe^{z_j-m}', pseudocode: 'return softmax(z)', complexity: 'O(n)',
    ref: 'Bridle 1990', commonPitfall: 'sum=1，元素 ≥0。',
    invariant: 'run 逐元素与 log 域参考 exp(z_i-lse) 一致；Σrun=1；run>=0；argmax(run)=argmax(z)',
    inputs: ['z'], domains: { z: 'vecSmall' },
    run: f_softmax3,
    predicate: function (z) {
      var sm = runOf('G109')(z);
      if (!sm || sm.length !== z.length) return false;
      var ref = softmaxRef(z), s = 0;
      for (var i = 0; i < z.length; i++) {
        if (!(sm[i] >= 0)) return false;
        if (!rc(sm[i], ref[i], 1e-12)) return false;
        s += sm[i];
      }
      if (!rc(s, 1, 1e-9)) return false;
      if (argmaxFirst(sm) !== argmaxRef(z)) return false;
      // 构造输入喂 run：全零向量 ⇒ 均匀分布
      var z0 = new Float64Array(z.length), u = runOf('G109')(z0);
      return rc(u[0], 1 / z.length, 1e-12);
    },
  });

  var f_nll_logsm = function (z, t) { var k = safeIdx(t, z.length); return -(z[k] - f_lse(z)); };
  reg(110, {
    name: '稳定NLL via log-softmax', intuition: 'NLL 经 log-softmax 稳定计算，避免 log(0)。',
    math: 'NLL=-logsm(z)_t=-(z_t-LSE(z))', pseudocode: 'return -(z_t-LSE(z))', complexity: 'O(n)',
    ref: 'Goodfellow 2016', commonPitfall: '>=0。',
    invariant: 'run 与直接式「ln Σexp(z) - z_t」一致；run>=0；与 CE(G101) 两路经 run 交叉一致',
    inputs: ['logits', 'target'], domains: { logits: 'vecSmall', target: 'smallInt' },
    run: f_nll_logsm,
    predicate: function (logits, target) {
      var r = runOf('G110')(logits, target);
      var k = safeIdx(target, logits.length);
      if (!rc(r, lseDirect(logits) - logits[k], 1e-12)) return false;
      if (!(r >= 0)) return false;
      return rc(r, runOf('G101')(logits, target), 1e-12);
    },
  });

  var f_exp_ident = function (x) { return Math.exp(x); };
  reg(111, {
    name: '稳定exp-identity', intuition: 'exp 满足 exp(x)·exp(-x)=1，验证指数对合一致性。',
    math: 'e^x·e^{-x}=1', pseudocode: 'return exp(x)*exp(-x)', complexity: 'O(1)',
    ref: '初等指数性质', commonPitfall: '大 x 处 exp(x)=Inf 使乘积 NaN（域 |x|<=10 安全）。',
    invariant: 'run 经 log 反向还原 x；两路经 run 的乘积恒为 1；run(0)=1、run(1)=e',
    inputs: ['x'], domains: { x: 'small' },
    run: f_exp_ident,
    predicate: function (x) {
      var r = runOf('G111')(x);
      if (!(r > 0)) return false;
      if (!rc(Math.log(r), x, 1e-12)) return false;      // 主 oracle
      if (!rc(r * runOf('G111')(-x), 1, 1e-12)) return false;  // 对合恒等式（两路经 run）
      return rc(runOf('G111')(0), 1, 1e-15) && rc(runOf('G111')(1), Math.E, 1e-15);
    },
  });

  var f_silu3 = function (x) { return x * S.sigmoid(x); };
  reg(112, {
    name: '稳定SiLU', intuition: 'SiLU 数值稳定，满足 silu(x)=silu(-x)+x 的对称恒等式。',
    math: 'silu(x)=x·σ(x)=x/(1+e^{-x}); silu(x)=silu(-x)+x', pseudocode: 'return x*sig(x)', complexity: 'O(1)',
    ref: 'Ramachandran et al. 2017', commonPitfall: '与 Swish 同义。',
    invariant: 'run 与除法形式 x/(1+e^{-x}) 一致（对照 run 的 x·σ(x)）；恒等式两路经 run；silu(0)=0',
    inputs: ['x'], domains: { x: 'small' },
    run: f_silu3,
    predicate: function (x) {
      var r = runOf('G112')(x);
      // 独立 oracle：除法形式（run 用乘法 x·σ(x)）
      if (!rc(r, x / (1 + Math.exp(-x)), 1e-12)) return false;
      // 恒等式：两路均经 run
      if (!rc(r, runOf('G112')(-x) + x, 1e-12)) return false;
      // 性质：x>0 时非负；定点
      if (x > 0 && !(r > 0)) return false;
      if (x < 0 && !(r < 0)) return false;
      return rc(runOf('G112')(0), 0, 1e-15) && rc(runOf('G112')(1), 1 / (1 + Math.exp(-1)), 1e-15);
    },
  });

  // ════════════════════════ 高级/组合 G113–G128 ════════════════════════
  var f_acc = function (z, t) { var k = safeIdx(t, z.length); return argmaxFirst(z) === k ? 1 : 0; };
  reg(113, {
    name: '多分类准确率', intuition: '预测类别（argmax）等于目标时记为正确。',
    math: 'acc=1[argmax(z)=t]', pseudocode: 'return argmax(z)==t?1:0', complexity: 'O(n)',
    ref: '评估标准', commonPitfall: '参考若写成 run 的定义式则恒真（run 返回 2 也过不了，但 run 返回错误 argmax 时无法发现）。须用独立两遍扫描 argmax。',
    invariant: 'run ∈{0,1} 且等于两遍扫描 argmax 的命中指示；构造输入覆盖命中/未命中两路',
    inputs: ['logits', 'target'], domains: { logits: 'vecSmall', target: 'smallInt' },
    run: f_acc,
    predicate: function (logits, target) {
      var r = runOf('G113')(logits, target);
      var k = safeIdx(target, logits.length);
      if (r !== 0 && r !== 1) return false;
      if (r !== (argmaxRef(logits) === k ? 1 : 0)) return false;
      // 构造输入喂 run：命中与未命中
      var z = new Float64Array(logits.length); z[0] = 5;
      return runOf('G113')(z, 0) === 1 && runOf('G113')(z, 1) === 0;
    },
  });

  var f_bacc = function (s, y) {
    var c = 0; for (var i = 0; i < s.length; i++) { var pred = s[i] > 0 ? 1 : 0; if (pred === (y[i] === 1 ? 1 : 0)) c++; } return c / s.length;
  };
  reg(114, {
    name: '二分类准确率', intuition: '以 0 为阈值的二分类命中率。',
    math: 'acc=mean(1[pred==label]), pred=score>0', pseudocode: 'return mean(correct)', complexity: 'O(n)',
    ref: '评估标准', commonPitfall: '阈值固定为 0（logit 空间）；参考若复制 run 的循环则无鉴别力。',
    invariant: 'run 与反向累加命中计数/n 一致；run∈[0,1]；构造输入覆盖全对/全错',
    inputs: ['scores', 'labels'], domains: { scores: 'vecSmall', labels: 'vecMask' },
    run: f_bacc,
    predicate: function (s, y) {
      var r = runOf('G114')(s, y);
      var c = 0;
      // 独立参考：布尔等价比较（(s>0)===(y>0)）+ 反向累加
      for (var i = s.length - 1; i >= 0; i--) if ((s[i] > 0) === (y[i] > 0)) c++;
      if (!rc(r, c / s.length, 1e-12)) return false;
      if (!(r >= 0 && r <= 1)) return false;
      // 构造输入喂 run：全对 ⇒ 1；全错 ⇒ 0；阈值严格性（score=0 判为负类）
      var n = s.length, pos = new Float64Array(n).fill(1), neg = new Float64Array(n).fill(-1),
        zero = new Float64Array(n), one = new Float64Array(n).fill(1), none = new Float64Array(n);
      return runOf('G114')(pos, one) === 1 && runOf('G114')(pos, none) === 0
        && runOf('G114')(neg, none) === 1 && runOf('G114')(zero, one) === 0;
    },
  });

  var f_wmse = function (y, yh, w) {
    var num = 0, den = 0; for (var i = 0; i < y.length; i++) { var e = y[i] - yh[i]; num += w[i] * e * e; den += w[i]; }
    return den > 0 ? num / den : 0;
  };
  reg(115, {
    name: '加权MSE', intuition: '按样本权重加权均方误差。',
    math: 'wMSE=Σw_i(y_i-ŷ_i)²/Σw_i', pseudocode: 'return weighted mean sq', complexity: 'O(n)',
    ref: 'Goodfellow 2016', commonPitfall: '权重须为正且归一化前的和 >0。',
    invariant: 'run 与「先归一化权重 Σ(w_i/Σw)·e_i²」一致（对照 run 的 Σwe²/Σw）；run>=0',
    inputs: ['y', 'yh', 'w'], domains: { y: 'vecSmall', yh: 'vecSmall', w: 'vecPos' },
    run: f_wmse,
    predicate: function (y, yh, w) {
      var r = runOf('G115')(y, yh, w);
      // 独立参考：先归一化权重（不同结合方式）
      var den = 0; for (var i = 0; i < w.length; i++) den += w[i];
      var s = 0; for (var j = w.length - 1; j >= 0; j--) { var e = y[j] - yh[j]; s += (w[j] / den) * e * e; }
      if (!rc(r, s, 1e-12)) return false;
      if (!(r >= 0)) return false;
      // 构造输入喂 run：权重全 1 ⇒ 退化为 MSE
      var ones = new Float64Array(w.length).fill(1), t = 0;
      for (var k = 0; k < y.length; k++) { var d = y[k] - yh[k]; t += d * d; }
      return rc(runOf('G115')(y, yh, ones), t / y.length, 1e-12);
    },
  });

  var f_lsce2 = function (z, t, eps) {
    var n = z.length, k = safeIdx(t, n), sm = f_softmax3(z), q = new Float64Array(n);
    for (var i = 0; i < n; i++) q[i] = (1 - eps) * (i === k ? 1 : 0) + eps / n;
    var s = 0; for (var j = 0; j < n; j++) s += -q[j] * Math.log(Math.max(sm[j], PC));
    return s;
  };
  reg(116, {
    name: '标签平滑变体CE', intuition: '通用 ε 标签平滑交叉熵（ε 由调用方给定）。',
    math: 'q=(1-ε)y+ε/n; L=-Σq_i ln p_i = LSE(z) - [(1-ε)z_t + (ε/n)Σz_i]', pseudocode: 'return LSE(z)-((1-eps)*z_t+(eps/n)*sum z)', complexity: 'O(n)',
    ref: 'Pereyra et al. 2017', commonPitfall: 'ε=0 退化为普通 CE；概率域加权与 log 域展开是两种表达，参考须用后者才非自比。',
    invariant: 'run 与 log 域展开式一致；run>=0；ε=0 经 run 等于 CE(G101)',
    inputs: ['logits', 'target', 'eps'], domains: { logits: 'vecSmall', target: 'smallInt', eps: 'unit' },
    run: f_lsce2,
    predicate: function (logits, target, eps) {
      var r = runOf('G116')(logits, target, eps);
      var n = logits.length, k = safeIdx(target, n);
      // 独立参考：log 域展开（run 在概率域加权）
      var lse = lseDirect(logits), sz = 0;
      for (var i = 0; i < n; i++) sz += logits[i];
      if (!rc(r, lse - ((1 - eps) * logits[k] + (eps / n) * sz), 1e-12)) return false;
      if (!(r >= 0)) return false;
      // ε=0 退化：两路均经 run
      return rc(runOf('G116')(logits, target, 0), runOf('G101')(logits, target), 1e-12);
    },
  });

  var f_tce = function (z, t, T) {
    // log 域实现（修复：原 softmax+log(max(p,1e-12)) 在 T→0 时下溢被截断在 27.63）
    var n = z.length, k = safeIdx(t, n), zt = new Float64Array(n);
    for (var i = 0; i < n; i++) zt[i] = z[i] / T;
    return -(zt[k] - f_lse(zt));
  };
  reg(117, {
    name: '温度缩放交叉熵', intuition: '先按温度缩放 logits 再算交叉熵，用于知识蒸馏。',
    math: 'L = -(z_t/T - LSE(z/T))', pseudocode: 'return LSE(z/T)-z_t/T', complexity: 'O(n)',
    ref: 'Hinton et al. 2015', commonPitfall: 'T 同时缩放学生与教师；T 极小时 softmax 分量下溢为 0，故必须走 log 域（见文件头 run 修复①）。',
    invariant: 'run 与直接式「ln Σexp(z/T) - z_t/T」一致；run>=0；T=1 经 run 等于 CE(G101)；小 T 不被截断',
    inputs: ['logits', 'target', 'T'], domains: { logits: 'vecSmall', target: 'smallInt', T: 'smallPos' },
    run: f_tce,
    predicate: function (logits, target, T) {
      var r = runOf('G117')(logits, target, T);
      var n = logits.length, k = safeIdx(target, n), zt = new Float64Array(n);
      for (var i = 0; i < n; i++) zt[i] = logits[i] / T;
      // 独立参考：直接式（z/T ∈[-1e7,1e7]，run 用 max 平移故参考须直接式亦安全于 lnΣ）
      if (!rc(r, lseDirect(zt) - zt[k], 1e-9)) return false;
      if (!(r >= 0)) return false;
      // T=1 退化：两路均经 run
      if (!rc(runOf('G117')(logits, target, 1), runOf('G101')(logits, target), 1e-12)) return false;
      // 构造输入喂 run：极小温度下损失应爆到 ~2|z|/T，而非被截断在 27.63
      var z = new Float64Array(n); z[0] = 1; z[n - 1] = -1;
      return runOf('G117')(z, n - 1, 1e-6) > 1e5;
    },
  });

  var f_topkce = function (z, t, k) {
    var n = z.length, kk = safeIdx(k, n); if (kk < 1) kk = 1; if (kk > n) kk = n;
    var idx = []; for (var i = 0; i < n; i++) idx.push(i);
    idx.sort(function (a, b) { return z[b] - z[a]; });
    var sel = idx.slice(0, kk), kt = safeIdx(t, n);
    var m = -Infinity; for (var q = 0; q < sel.length; q++) if (z[sel[q]] > m) m = z[sel[q]];
    var s = 0; for (var j = 0; j < sel.length; j++) s += Math.exp(z[sel[j]] - m);
    return -z[kt] + (m + Math.log(s));
  };
  reg(118, {
    name: 'top-k交叉熵', intuition: '仅前 k 大 logit 参与归一化，其余被遮掩。',
    math: 'L=-z_y+ln Σ_{i∈topk} e^{z_i}（kk=safeIdx(k,n) 经 [1,n-1] 夹取，故 kk∈[1,n-1]）',
    pseudocode: 'return -z_y+LSE(topk)', complexity: 'O(n²)（参考为 O(n²) 排名法）',
    ref: 'Szegedy et al. 2016 (top-k 训练)',
    commonPitfall: '只断言 r<=fullCE 会让返回 -1e9 的实现通过；必须与 top-k 集合的直接式 LSE 比对。kk 因 safeIdx 取模后夹取恒 ∈[1,n-1]（不是 [1,n]）。',
    invariant: 'run 与「O(n²) 稳定排名选出的 top-k 集合之直接式 LSE - z_t」一致；0<=run<=完整CE；kk=1 ⇒ run=max(z)-z_t（构造输入验证）',
    inputs: ['logits', 'target', 'k'], domains: { logits: 'vecSmall', target: 'smallInt', k: 'smallInt' },
    run: f_topkce,
    predicate: function (logits, target, k) {
      var r = runOf('G118')(logits, target, k);
      var n = logits.length, kk = safeIdx(k, n); if (kk < 1) kk = 1; if (kk > n) kk = n;
      var kt = safeIdx(target, n);
      // 独立参考：O(n²) 稳定排名取 top-k，再直接式 LSE
      var sel = topkSet(logits, kk);
      if (sel.length !== kk) return false;
      if (!rc(r, lseValsDirect(sel) - logits[kt], 1e-12)) return false;
      // 性质：top-k ⊆ 全集 ⇒ LSE(topk) <= LSE(all)；且 LSE(topk) >= z_t（目标必不小于被选最小值）⇒ run>=0
      var full = lseDirect(logits) - logits[kt];
      if (!(r >= -1e-12 && r <= full + 1e-12)) return false;
      // 构造输入喂 run：kk=1 ⇒ 退化为 max(z)-z_t（z 构造为单峰 3，其余全 0）
      var z = new Float64Array(n); z[0] = 3;
      if (!rc(runOf('G118')(z, n - 1, 0), 3 - 0, 1e-15)) return false;   // k=0 ⇒ kk=1，目标非峰 ⇒ 3
      if (!rc(runOf('G118')(z, 0, 0), 0, 1e-15)) return false;           // 目标即峰 ⇒ 0
      // 构造输入喂 run：k=n-1 ⇒ kk=n-1，top-k 排除全局最小 ⇒ run 比完整 CE 严格小
      var z2 = new Float64Array(n); z2[0] = 3; z2[1] = -5;
      var rBig = runOf('G118')(z2, 0, n - 1);                          // kk=n-1，排除 -5
      var rFull = runOf('G118')(z2, 0, 0);                             // kk=1，排除一切
      if (!(rBig > rFull && rBig <= fullCE(z2, 0) + 1e-12)) return false;
      return true;
    },
  });

  var f_focal_logits = function (z, y) {
    var yb = signLabel(y[0]); var pt = yb === 1 ? S.sigmoid(z) : S.sigmoid(-z);
    return -(1 - pt) * (1 - pt) * Math.log(Math.max(pt, PC));
  };
  reg(119, {
    name: 'focal-with-logits', intuition: 'Focal Loss(γ=2) 直接作用于 logit，稳定计算。',
    math: 'p_t=σ(±z); FL=-(1-p_t)²ln p_t = (1-p_t)²·softplus(∓z)（ln p_t<=0 故 FL>=0）', pseudocode: 'return -(1-pt)^2*log(pt)', complexity: 'O(1)',
    ref: 'Lin et al. 2017', commonPitfall: '纯界断言（0<=FL<=CE）无 oracle：返回 0 的实现也能过。-ln p_t 须用 softplus 形式而非 log(σ)。',
    invariant: 'run 与「tanh 形式 p_t + softplus 形式 -ln p_t」一致；0<=run<=CE；FL<=CE 经 run 验；构造输入验定点',
    inputs: ['z', 'label'], domains: { z: 'small', label: 'vecMask' },
    run: f_focal_logits,
    predicate: function (z, label) {
      var r = runOf('G119')(z, label);
      var yb = signLabel(label[0]);
      // 独立参考：p_t 用 tanh 形式；-ln p_t 用 softplus 形式（run 用 log(max(p_t,1e-12))）
      var pt = sigmoidTanh(yb * z);
      var nlogpt = yb === 1 ? Math.log1p(Math.exp(-z)) : Math.log1p(Math.exp(z));
      if (!rc(r, (1 - pt) * (1 - pt) * nlogpt, 1e-12)) return false;   // FL=(1-p_t)²·(-ln p_t)，恒 >=0
      var ce = yb === 1 ? Math.log1p(Math.exp(-z)) : Math.log1p(Math.exp(z));
      if (!(r >= 0 && r <= ce + 1e-12)) return false;
      // 构造输入喂 run：z=0 ⇒ p_t=0.5 ⇒ FL = ¼ln2
      var lab1 = new Float64Array(label.length).fill(1);
      if (!rc(runOf('G119')(0, lab1), 0.25 * LOG2, 1e-15)) return false;
      // p_t→1（正确分类）时 FL 远小于 CE
      return runOf('G119')(10, lab1) < 1e-7;
    },
  });

  var f_contrastive2 = function (x, y, label, margin) {
    // 直接式 ‖x-y‖（修复：原展开式 ‖x‖²+‖y‖²-2x·y 在 x≈y 时相消放大）
    var d = Math.sqrt(sqDist(x, y));
    return label[0] === 1 ? 0.5 * d * d : 0.5 * Math.max(0, margin - d) * Math.max(0, margin - d);
  };
  reg(120, {
    name: '对比损失(margin)', intuition: '同类收缩、异类维持至少 margin 间隔的对比损失。',
    math: 'L=½d² (同) else ½max(0,m-d)²，d=√Σ(x_i-y_i)²', pseudocode: 'return piece', complexity: 'O(n)',
    ref: 'Hadsell et al. 2006', commonPitfall: '异类超 margin 后损失为 0；d 用展开式会在 x≈y 时严重失真（见文件头 run 修复②）。',
    invariant: 'run 与直接式 d=√Σ(x_i-y_i)² 一致；run>=0；异类 run<=½margin²；构造输入验 d=0 与超 margin 两路',
    inputs: ['x', 'y', 'label', 'margin'], domains: { x: 'vecSmall', y: 'vecSmall', label: 'vecMask', margin: 'smallPos' },
    run: f_contrastive2,
    predicate: function (x, y, label, margin) {
      var r = runOf('G120')(x, y, label, margin);
      // 独立参考：反向累加直接式平方距离
      var d = Math.sqrt(sqDist(x, y));
      var ref = label[0] === 1 ? 0.5 * d * d : 0.5 * Math.max(0, margin - d) * Math.max(0, margin - d);
      if (!rc(r, ref, 1e-12)) return false;
      if (!(r >= 0)) return false;
      if (label[0] === 0 && !(r <= 0.5 * margin * margin + 1e-12)) return false;
      // 构造输入喂 run：d=0（同类⇒0）；d=1（同类⇒½）；d 超 margin（异类⇒0）
      var n = x.length, a = new Float64Array(n), b = new Float64Array(n), lab1 = new Float64Array(label.length).fill(1);
      var lab0 = new Float64Array(label.length);
      b[0] = 1;
      if (runOf('G120')(a, a, lab1, 1) !== 0) return false;
      if (!rc(runOf('G120')(a, b, lab1, 1), 0.5, 1e-15)) return false;
      var far = new Float64Array(n); far[0] = 100;
      if (runOf('G120')(a, far, lab0, 1) !== 0) return false;            // 异类 d=100 > m=1 ⇒ 0
      return rc(runOf('G120')(a, far, lab1, 1), 0.5 * 100 * 100, 1e-12); // 同类 d=100 ⇒ ½d²
    },
  });

  var f_triplet = function (a, p, n, margin) {
    // 直接式平方距离（修复：原展开式相消放大，见文件头 run 修复③）
    var v = (sqDist(a, p) - sqDist(a, n)) + margin;
    return v > 0 ? v : 0;
  };
  reg(121, {
    name: 'triplet(margin)', intuition: '三元组损失：锚-正距离应小于锚-负距离至少 margin。',
    math: 'L=max(0,‖a-p‖²-‖a-n‖²+m)', pseudocode: 'return max(0,dap-dan+m)', complexity: 'O(n)',
    ref: 'Schroff et al. 2015', commonPitfall: '距离用平方欧氏；展开式 ‖a‖²+‖p‖²-2a·p 在 a≈p 时相消放大。',
    invariant: 'run 与直接式 Σ(a-p)²-Σ(a-n)² 一致；run>=0；构造输入验「违反⇒损失>margin」与「满足⇒0」两路',
    inputs: ['a', 'p', 'n', 'margin'], domains: { a: 'vecSmall', p: 'vecSmall', n: 'vecSmall', margin: 'smallPos' },
    run: f_triplet,
    predicate: function (a, p, n, margin) {
      var r = runOf('G121')(a, p, n, margin);
      // 独立参考：反向累加直接式
      var ref = (sqDist(a, p) - sqDist(a, n)) + margin;
      if (!rc(r, ref > 0 ? ref : 0, 1e-12)) return false;
      if (!(r >= 0)) return false;
      // 构造输入喂 run：a=p=n ⇒ d_ap-d_an=0 ⇒ 损失=margin（违反最严重）；
      // a=p 且 n 恰在 margin 处 ⇒ d_ap-d_an=-m² ⇒ 0（恰好满足）
      var m = a.length, z = new Float64Array(m), f = new Float64Array(m); f[0] = 1;
      if (!rc(runOf('G121')(z, z, z, 1), 1, 1e-15)) return false;
      // 构造输入喂 run：a=p=z、n=f（d_an=1）⇒ 0-1+1=0
      if (runOf('G121')(z, z, f, 1) !== 0) return false;
      // 构造输入喂 run：a=p=z、n=2f（d_an=4）⇒ 0-4+1<0 ⇒ 0（间隔已足够）
      var f2 = new Float64Array(m); f2[0] = 2;
      return runOf('G121')(z, z, f2, 1) === 0;
    },
  });

  var f_l1reg = function (w) { var s = 0; for (var i = 0; i < w.length; i++) s += Math.abs(w[i]); return s; };
  reg(122, {
    name: '正则L1', intuition: 'L1 权重正则，诱导稀疏。',
    math: 'R=Σ|w_i|', pseudocode: 'return sum|w|', complexity: 'O(n)',
    ref: 'Tibshirani 1996', commonPitfall: '在 0 处不可导；参考若复制 run 的正向累加则只能检出乱序，无法检出公式错误。',
    invariant: 'run 与反向累加 Σ|w| 一致；run>=0；‖w‖_1 >= ‖w‖_2（跨算子 G123 交叉）；构造输入验定点',
    inputs: ['w'], domains: { w: 'vecSmall' },
    run: f_l1reg,
    predicate: function (w) {
      var r = runOf('G122')(w), s = 0;
      for (var i = w.length - 1; i >= 0; i--) s += Math.abs(w[i]);
      if (!rc(r, s, 1e-12)) return false;
      if (!(r >= 0)) return false;
      // 跨算子不等式：L1 >= L2 = √(2·G123)（两路均经 run）
      if (!(r >= Math.sqrt(2 * runOf('G123')(w)) - 1e-9)) return false;
      var t = new Float64Array(w.length); t[0] = 3; t[1] = -4;
      return rc(runOf('G122')(t), 7, 1e-15);
    },
  });

  var f_l2reg = function (w) { var s = 0; for (var i = 0; i < w.length; i++) s += w[i] * w[i]; return 0.5 * s; };
  reg(123, {
    name: '正则L2', intuition: 'L2 权重正则（权重衰减），平滑。',
    math: 'R=½Σw_i²', pseudocode: 'return 0.5*sum w^2', complexity: 'O(n)',
    ref: 'Hoerl & Kennard 1970', commonPitfall: '½ 因子易遗漏；参考复制 run 则漏 ½ 也发现不了。',
    invariant: 'run 与反向累加 ½Σw² 一致；run>=0；缩放齐次 R(cw)=c²R(w) 两路经 run；构造输入验 ½ 因子',
    inputs: ['w'], domains: { w: 'vecSmall' },
    run: f_l2reg,
    predicate: function (w) {
      var r = runOf('G123')(w), s = 0;
      for (var i = w.length - 1; i >= 0; i--) s += w[i] * w[i];
      if (!rc(r, 0.5 * s, 1e-12)) return false;
      if (!(r >= 0)) return false;
      // 齐次性：c=3 ⇒ R(3w) = 9·R(w)（两路均经 run）
      var w3 = new Float64Array(w.length);
      for (var j = 0; j < w.length; j++) w3[j] = 3 * w[j];
      if (!rc(runOf('G123')(w3), 9 * r, 1e-9)) return false;
      var t = new Float64Array(w.length); t[0] = 2;
      return rc(runOf('G123')(t), 2, 1e-15);
    },
  });

  var f_tvreg = function (w) { var s = 0; for (var i = 1; i < w.length; i++) s += Math.abs(w[i] - w[i - 1]); return s; };
  reg(124, {
    name: 'TV正则', intuition: '全变分正则，惩罚相邻权重跳变，促进平滑。',
    math: 'R=Σ|w_{i+1}-w_i|', pseudocode: 'return sum|Δw|', complexity: 'O(n)',
    ref: 'Rudin et al. 1992', commonPitfall: '沿索引的相邻差分；参考复制 run 则漏项/错索引发现不了。',
    invariant: 'run 与反向累加 Σ|Δw| 一致（n-1 项）；run>=0；常数向量 ⇒ 0；构造输入验跳变累计',
    inputs: ['w'], domains: { w: 'vecSmall' },
    run: f_tvreg,
    predicate: function (w) {
      var r = runOf('G124')(w), s = 0;
      for (var i = w.length - 1; i >= 1; i--) s += Math.abs(w[i] - w[i - 1]);
      if (!rc(r, s, 1e-12)) return false;
      if (!(r >= 0)) return false;
      // 常数向量 ⇒ 0（构造输入喂 run）
      var c = new Float64Array(w.length).fill(2.5);
      if (runOf('G124')(c) !== 0) return false;
      // 构造输入：[1,3,0,...] ⇒ 2+3 = 5
      var t = new Float64Array(w.length); t[0] = 1; t[1] = 3; t[2] = 0;
      return rc(runOf('G124')(t), 5, 1e-15);
    },
  });

  var f_hinge_margin = function (z, y, m) { var yb = signLabel(y[0]); var v = m - yb * z; return v > 0 ? v : 0; };
  reg(125, {
    name: '合页随margin非减', intuition: '带 margin 的合页损失随 margin 增大而单调不减。',
    math: 'L_m(z,y)=max(0,m-y·z) 关于 m 非减', pseudocode: 'return max(0,m-y*z)', complexity: 'O(1)',
    ref: 'Crammer & Singer 2001', commonPitfall: '原谓词把 4 个参数传给 3 参 run（m2 被吞掉），单侧只走 run、另一侧走内联表达式。须两路均经 run。',
    invariant: 'run 与 max(0,m-y·z) 一致；run>=0；m1<=m2 ⇒ 两路经 run 的损失非减；构造输入验「未违反⇒0」',
    inputs: ['z', 'label', 'm1', 'm2'], domains: { z: 'small', label: 'vecMask', m1: 'smallPos', m2: 'smallPos' },
    run: f_hinge_margin,
    predicate: function (z, label, m1, m2) {
      var h1 = runOf('G125')(z, label, m1);
      var h2 = runOf('G125')(z, label, m2);       // 两路均经 run（原实现 m2 走内联）
      var yb = signLabel(label[0]);
      if (!rc(h1, Math.max(0, m1 - yb * z), 1e-12)) return false;
      if (!rc(h2, Math.max(0, m2 - yb * z), 1e-12)) return false;
      if (!(h1 >= 0 && h2 >= 0)) return false;
      if (m1 <= m2) { if (!(h1 <= h2 + 1e-12)) return false; }
      else { if (!(h1 >= h2 - 1e-12)) return false; }
      // 构造输入喂 run：z=10,y=+1,m=1 ⇒ 1-10<0 ⇒ 0（间隔已满足）；
      // z=-10,y=+1,m=1 ⇒ 1+10=11（违反最严重）
      var lab1 = new Float64Array(label.length).fill(1);
      if (runOf('G125')(10, lab1, 1) !== 0) return false;
      return rc(runOf('G125')(-10, lab1, 1), 11, 1e-15);
    },
  });

  reg(126, {
    name: '加权MSE退化(均匀权重==MSE)', intuition: '权重全相等时加权 MSE 退化为普通 MSE。',
    math: 'w_i≡c ⇒ wMSE=MSE', pseudocode: 'return wMSE(y,yh,1)=MSE', complexity: 'O(n)',
    ref: '基础性质', commonPitfall: '原谓词完全忽略传入的 w 参数，只测全 1 权重；应同时验证任意 w 下 run 本身正确。',
    invariant: 'run(全1权重) 与反向累加 MSE 一致；run(传入 w) 与先归一化权重参考一致；权重常数倍不变（两路经 run）',
    inputs: ['y', 'yh', 'w'], domains: { y: 'vecSmall', yh: 'vecSmall', w: 'vecPos' },
    run: f_wmse,
    predicate: function (y, yh, w) {
      var n = y.length, ones = new Float64Array(n).fill(1);
      var r1 = runOf('G126')(y, yh, ones);
      var s = 0;
      for (var i = n - 1; i >= 0; i--) { var e = y[i] - yh[i]; s += e * e; }
      if (!rc(r1, s / n, 1e-12)) return false;
      // 任意 w 下 run 本身须正确（否则本 op 只测退化而漏掉主体）
      var r2 = runOf('G126')(y, yh, w), den = 0, s2 = 0;
      for (var j = 0; j < n; j++) den += w[j];
      for (var k = n - 1; k >= 0; k--) { var d = y[k] - yh[k]; s2 += (w[k] / den) * d * d; }
      if (!rc(r2, s2, 1e-12)) return false;
      // 权重常数倍（c=4）不变性：两路均经 run
      var w4 = new Float64Array(n);
      for (var m = 0; m < n; m++) w4[m] = 4 * w[m];
      return rc(runOf('G126')(y, yh, w4), r2, 1e-12);
    },
  });

  var f_perp = function (p) {
    // run 侧：正向累加 + log(clip(p))（与参考 entRef 的反向累加 + 同一 clip 约定不同序）
    var s = 0; for (var i = 0; i < p.length; i++) { var pi = Math.max(p[i], PC); s += -pi * Math.log(pi); }
    return Math.exp(s);
  };
  reg(127, {
    name: '熵与困惑度(perp=exp(H))', intuition: '困惑度等于熵的指数，熵非负故困惑度 ≥1。',
    math: 'PPL=exp(H(p))=exp(-Σp ln p)', pseudocode: 'return exp(-sum p ln p)', complexity: 'O(n)',
    ref: 'Jelinek et al. 1977', commonPitfall: '参考若直接调本文件 entropy 则与 run 逐字相同（恒真）；须反向累加。',
    invariant: 'run 与反向累加熵之指数一致；run>=1；构造输入验「均匀⇒n」「one-hot⇒1」两条解析极值',
    inputs: ['p'], domains: { p: 'vecProb' },
    run: f_perp,
    predicate: function (p) {
      var r = runOf('G127')(p);
      if (!rc(r, Math.exp(entRef(p)), 1e-12)) return false;
      if (!(r >= 1 - 1e-9)) return false;
      var n = p.length;
      // 构造输入喂 run：均匀分布 ⇒ PPL = n（上界，H=ln n）
      var u = new Float64Array(n).fill(1 / n);
      if (!rc(runOf('G127')(u), n, 1e-9)) return false;
      // 构造输入喂 run：one-hot ⇒ PPL = 1（下界，H=0）
      var e0 = new Float64Array(n); e0[0] = 1;
      return rc(runOf('G127')(e0), 1, 1e-9);
    },
  });

  var f_mi = function (p, q) { return f_kl2(p, q); };
  reg(128, {
    name: '互信息非负', intuition: '互信息（此处以 KL 形式度量）非负，p=q 时为 0。',
    math: 'MI(p,q)=Σp_i ln(p_i/q_i)>=0', pseudocode: 'return kl(p,q)', complexity: 'O(n)',
    ref: 'Cover & Thomas 2006', commonPitfall: '非对称度量；只查 r>=0 无 oracle，须与分离对数式参考比对并与 G107 交叉。',
    invariant: 'run 与分离对数式 Σp(ln p-ln q) 一致；run>=0；与 KL(G107) 两路经 run 交叉一致；MI(p,p)=0 经 run',
    inputs: ['p', 'q'], domains: { p: 'vecProb', q: 'vecProb' },
    run: f_mi,
    predicate: function (p, q) {
      var r = runOf('G128')(p, q);
      if (!Number.isFinite(r)) return false;
      if (!rc(r, klRef(p, q), 1e-9)) return false;
      if (!(r >= 0)) return false;
      if (!rc(r, runOf('G107')(p, q), 1e-15)) return false;      // 与 KL 同实现交叉
      if (!rc(runOf('G128')(p, p), 0, 1e-12)) return false;      // 自反性
      // 构造输入喂 run：one-hot 对均匀 ⇒ ln n（one-hot 对 one-hot 会被 clip 到 27.63）
      var n = p.length, e0 = new Float64Array(n), u = new Float64Array(n).fill(1 / n);
      e0[0] = 1;
      return rc(runOf('G128')(e0, u), Math.log(n), 1e-9);
    },
  });
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});
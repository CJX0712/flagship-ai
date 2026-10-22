/* ============================================================================
 * L10 微指令 · D 域（注意力基元）· 第 4/4 批（D097–D128）
 * ----------------------------------------------------------------------------
 * 所属层：L10-microIsa  域：D（注意力基元）
 * 允许依赖的层：L9（rng/assert/special）、L10（registry/invariant）
 * 主要 @ref：Vaswani et al. 2017（缩放点积注意力组装）；
 *           Su et al. 2021 (RoPE)；Press et al. 2022 (ALiBi)；
 *           Ainslie et al. 2023 (GQA)；Xiao et al. 2024 (StreamingLLM)；
 *           Milakov & Gimelshein 2018（在线 softmax）
 *
 * 子域划分（32 项；D 域至此补满 128 槽）：
 *   D097–D108 缩放点积注意力组装（12）：CS 复合界 / 权重值域 / 凸组合夹取 /
 *              因果支撑 / 等分退化 / 低温 one-hot / V 平移·V 缩放·query 局部性 /
 *              常列保持 / 端到端 lse 一致 / 首行逐位
 *   D109–D120 位置编码进阶（12）：RoPE 单侧相对性 / 步进角严格序 / 基向量旋转 /
 *              ALiBi 几何衰减 / RoPE×ALiBi 联合平移 / 斜率累积积 / 尾维恒等 /
 *              频率公比 / 行距离和 / 相邻位置 Lipschitz 界 / 行极小偏置 /
 *              行分布平移不变
 *   D121–D128 GQA×Cache 复合（8）：头间写入隔离 / 打分前缀逐位 / 组内共享一致 /
 *              滚动长度守恒 / 在线 softmax 等比缩放 / 输出递推式 /
 *              锚点打分保持 / 增量=全量逐位一致
 *
 * 与前三批的分工：D001–D096 覆盖「原子」（打分/掩码/softmax/RoPE/ALiBi/
 * GQA 映射/cache 机制）；本批全部是「组装与复合」层不变量——每条断言都
 * 组合已验证的原子，并在 ref / commonPitfall 中注明与哪些原子复合、与哪些
 * 相邻项分工（避免重复断言同一事实）。
 *
 * 组装语义说明：缩放打分 / 因果 / softmax / 加权和的组装数学与
 * src/L4-ops/attention.js 的 attnFwd 同源；因 __loadOrder 中 L10 文件早于
 * L4-ops 加载（模块求值期捕获不到 NS.L4），按 D001–D003 的既定约定在本文件
 * 内自包含复刻同一组 helper，数学逐字一致、不跨层捕获。
 *
 * 数值判据约定（沿用前三批，并按历史教训收紧）：
 *   - 数值恒等用 rc(a,b,1e-9)（缩放/三角/闭式类 1e-12）；
 *   - 精确恒等用 ±0 归一后的 Object.is（bitEq）：凸组合、掩位归零、
 *     增量=全量等位级判据一律逐位，不放宽容差；
 *   - 涉及 softmax 输出序/集中性的断言一律带守卫：仅当缩放间距 ≥40
 *     （全部 exp 操作数落在正规浮点区）才断言严格序或 one-hot；守卫相对
 *     历史教训的「si>−740」进一步收紧——e^−708 以下进入次正规带、相对粒度
 *     退化，−740 附近的 exp 已不可靠；本批不断言任何符号极限，只用有限
 *     参数下的紧界（见 D102 的论证）；
 *   - 前置条件式不变量：输入尺寸不满足数学前提时直接放行（返回 true），
 *     并在 invariant 文本中显式写出前置条件——不靠「放宽容差」制造假绿。
 * ==========================================================================*/
;(function (NS) {
  'use strict';
  NS.L10 = NS.L10 || {};
  const rc = NS.L10.relClose;
  function reg(i, o) { NS.L10.register('D', i, o); }

  /* ---------- 共享常量与基础工具（本文件自包含，不跨文件/跨层捕获） ---------- */
  const D8 = 8, THETA = 10000;
  const sq = (M) => Math.round(Math.sqrt(M.length));

  function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
  function dotRev(a, b) { let s = 0; for (let i = a.length - 1; i >= 0; i--) s += a[i] * b[i]; return s; }
  // 独立范数参考路径：Math.hypot 逐元素累积（与 sqrt(Σx²) 是不同算法）
  function hypotRef(x) { let s = 0; for (let i = 0; i < x.length; i++) s = Math.hypot(s, x[i]); return s; }
  // 逐位相等：±0 归一后 Object.is（与 relClose 的 exact 模式语义一致）
  function bitEq(a, b) {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) {
      let x = a[i], y = b[i];
      if (x === 0) x = 0; if (y === 0) y = 0;
      if (!Object.is(x, y)) return false;
    }
    return true;
  }

  /* ---------- RoPE（同 D001/D002 的自包含实现） ---------- */
  function invFreq(d, theta) {
    const half = Math.max(0, Math.floor(d / 2)), f = new Float64Array(half);
    for (let i = 0; i < half; i++) f[i] = Math.pow(theta, -2 * i / d);
    return f;
  }
  function rope(x, pos, d, theta) {
    const n = x.length, out = new Float64Array(n), f = invFreq(d, theta);
    const half = Math.min(f.length, Math.floor(n / 2));
    for (let i = 0; i < n; i++) out[i] = x[i];
    for (let i = 0; i < half; i++) {
      const a = pos * f[i], c = Math.cos(a), s = Math.sin(a);
      const x0 = x[2 * i], x1 = x[2 * i + 1];
      out[2 * i] = x0 * c - x1 * s;
      out[2 * i + 1] = x0 * s + x1 * c;
    }
    return out;
  }

  /* ---------- softmax 家族（安全版：减行最大值；全掩回退均匀） ---------- */
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

  /* ---------- ALiBi（同 D001 的自包含实现） ---------- */
  function alibiSlopes(h) {
    const H = 1 + (Math.abs(h) % 16), s = new Float64Array(H);
    for (let i = 0; i < H; i++) s[i] = Math.pow(2, -8 * i / H);
    return s;
  }
  function alibiBias(n, slope) {
    const B = new Float64Array(n * n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) B[i * n + j] = -slope * Math.abs(i - j);
    return B;
  }

  /* ---------- GQA 头映射 + KV Cache（同 D002 的自包含实现） ---------- */
  function kvIndexOf(h, nH, nKV) { return Math.floor(h * nKV / nH); }
  function makeCache(cap, d) { return { data: new Float64Array(cap * d), len: 0, cap, d }; }
  function cacheAppend(c, row) {
    const out = { data: c.data.slice(), len: c.len, cap: c.cap, d: c.d };
    if (c.len < c.cap) { out.data.set(row.subarray(0, c.d), c.len * c.d); out.len = c.len + 1; }
    return out;
  }
  function cacheRow(c, p) {
    if (p < 0 || p >= c.len) return null;
    return c.data.slice(p * c.d, (p + 1) * c.d);
  }
  // 滚动窗口：满时淘汰最旧的非 sink 行（sink 索引集合永不淘汰）
  function cacheAppendWindowed(c, row, sinks) {
    const out = { data: c.data.slice(), len: c.len, cap: c.cap, d: c.d };
    if (c.len < c.cap) { out.data.set(row.subarray(0, c.d), c.len * c.d); out.len = c.len + 1; return out; }
    let victim = -1;
    for (let i = 0; i < c.len; i++) { if (!sinks[i]) { victim = i; break; } }
    if (victim < 0) return out;                    // 全是 sink：拒绝淘汰（保护锚点）
    const nd = new Float64Array(c.cap * c.d);
    let w = 0;
    for (let i = 0; i < c.len; i++) { if (i === victim) continue; nd.set(out.data.subarray(i * c.d, (i + 1) * c.d), w * c.d); w++; }
    nd.set(row.subarray(0, c.d), w * c.d);
    out.data = nd; out.len = w + 1;
    return out;
  }

  /* ---------- 组装工具（与 L4-ops/attention 的 attnFwd 语义同源，自包含复刻） ----------
   * S[i][j] = ⟨Qᵢ,Kⱼ⟩/√d；out[i] = Σⱼ P[i][j]·Vⱼ；端到端 = softmax(causal(S))·V */
  function scoresFromQ(Q, K, m, n, d) {
    const out = new Float64Array(m * n), sc = 1 / Math.sqrt(d);
    for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) {
      out[i * n + j] = dot(Q.subarray(i * d, (i + 1) * d), K.subarray(j * d, (j + 1) * d)) * sc;
    }
    return out;
  }
  function attnFromScores(P, V, m, n, d) {           // 正向累加
    const out = new Float64Array(m * d);
    for (let i = 0; i < m; i++) for (let t = 0; t < d; t++) {
      let s = 0;
      for (let j = 0; j < n; j++) s += P[i * n + j] * V[j * d + t];
      out[i * d + t] = s;
    }
    return out;
  }
  function attnFromScoresRev(P, V, m, n, d) {        // 参考路径：反向累加
    const out = new Float64Array(m * d);
    for (let i = 0; i < m; i++) for (let t = 0; t < d; t++) {
      let s = 0;
      for (let j = n - 1; j >= 0; j--) s += P[i * n + j] * V[j * d + t];
      out[i * d + t] = s;
    }
    return out;
  }
  // 端到端（方阵 n×n、头维 d）：out = softmax(causal(QKᵀ/√d))·V
  function attnFromQKV(Q, K, V, n, d) {
    const P = softmaxRows(causal(scoresFromQ(Q, K, n, n, d), n), n);
    return attnFromScores(P, V, n, n, d);
  }
  // 增量上下文：对 cache 前 len 行做缩放点积（cache 只含可见行，无需掩码）
  function cacheCtx(kc, vc, q) {
    const len = kc.len, d = kc.d, sc = 1 / Math.sqrt(d);
    const s = new Float64Array(len);
    for (let j = 0; j < len; j++) s[j] = dot(q, cacheRow(kc, j)) * sc;
    const P = softmaxRow(s, 0, len);
    const out = new Float64Array(d);
    for (let t = 0; t < d; t++) { let acc = 0; for (let j = 0; j < len; j++) acc += P[j] * vc.data[j * d + t]; out[t] = acc; }
    return out;
  }

  // ══ 缩放点积注意力组装（D097–D108）═══════════════════════════════════
  reg(97, {
    name: '组装 · 缩放打分的有界性（Cauchy–Schwarz 复合）',
    intuition: '组装层先缩放再进 softmax：缩放打分的绝对值被 ‖q‖‖k‖/√d 扼住，是注意力数值稳定的第一道闸门。',
    math: '|q·k|/√d ≤ ‖q‖₂·‖k‖₂/√d',
    pseudocode: 'return abs(dot(q,k)/sqrt(d)) <= hypot(q)*hypot(k)/sqrt(d)*(1+eps)',
    complexity: 'O(d)', ref: 'Cauchy–Schwarz 不等式 · 复合 D006（原始 CS 上界）与 D005（1/√d 缩放）',
    commonPitfall: '范数参考路径用 Math.hypot 逐元素累积，与 sqrt(Σx²) 是不同算法，避免自证；这是组装粒度的复合界，不是 D006 的重复。',
    invariant: '|q·k|/√d ≤ ‖q‖·‖k‖/√d·(1+1e-12)（d=4）',
    inputs: ['q', 'k'], domains: { q: 'vecSmall', k: 'vecSmall' }, vecDims: { q: 4, k: 4 },
    run: (q, k) => Math.abs(dot(q, k) / Math.sqrt(q.length)),
    predicate: (q, k) => Math.abs(dot(q, k) / Math.sqrt(q.length)) <= (hypotRef(q) * hypotRef(k)) / Math.sqrt(q.length) * (1 + 1e-12),
  });
  reg(98, {
    name: '组装 · softmax 权重值域与掩位归零',
    intuition: '组装后的权重矩阵必须同时满足：每项 ∈[0,1]、被掩未来位严格为 0、每行和为一——三条一起才是合法注意力分布。',
    math: 'P = softmax(causal(QKᵀ/√d))：0≤Pᵢⱼ≤1；j>i ⇒ Pᵢⱼ=0；ΣⱼPᵢⱼ=1',
    pseudocode: 'S = causal(scoresFromQ(Q,K)); P = softmaxRows(S); assert range, mask-zero, row-sum',
    complexity: 'O(n²·d + n²)', ref: 'Vaswani et al. 2017 §3.2.2 · 复合 D015（上三角 −∞）与 D025（安全 softmax）',
    commonPitfall: '掩位权重为 0 是精确断言（exp(−∞)=0），不是近似；行和用 lse 参考路径校验（Σⱼexp(sᵢⱼ−lseᵢ)），不与 softmaxRow 自证。',
    invariant: '0≤Pᵢⱼ≤1；j>i ⇒ Pᵢⱼ===0；rc(ΣⱼPᵢⱼ,1,1e-9)；rc(Σⱼexp(sᵢⱼ−lseᵢ),1,1e-9)',
    inputs: ['Q', 'K'], domains: { Q: 'matSmall', K: 'matSmall' }, matrixDims: { Q: [4, 4], K: [4, 4] },
    run: (Q, K) => {
      const n = sq(Q), d = Q.length / n;
      return softmaxRows(causal(scoresFromQ(Q, K, n, n, d), n), n);
    },
    predicate: (Q, K) => {
      const n = sq(Q), d = Q.length / n;
      const S = causal(scoresFromQ(Q, K, n, n, d), n);
      const P = softmaxRows(S, n);
      for (let i = 0; i < n; i++) {
        let rowSum = 0, refSum = 0;
        const li = lse(S, i * n, n);
        for (let j = 0; j < n; j++) {
          const p = P[i * n + j];
          if (!(p >= 0 && p <= 1)) return false;
          if (j > i && p !== 0) return false;
          rowSum += p;
          refSum += Math.exp(S[i * n + j] - li);      // 参考路径：lse 表出
        }
        if (!rc(rowSum, 1, 1e-9)) return false;
        if (!rc(refSum, 1, 1e-9)) return false;
      }
      return true;
    },
  });
  reg(99, {
    name: '组装 · 输出是 V 的凸组合',
    intuition: '注意力输出是 V 各行的凸组合：既不会跳出 V 的值域，也解释了「注意力 = 软查表」。',
    math: 'outᵢ = ΣⱼPᵢⱼ·Vⱼ，Pᵢ∈Δ^{n−1} ⇒ outᵢₜ ∈ [minⱼVⱼₜ, maxⱼVⱼₜ]',
    pseudocode: 'out = attnFromQKV(Q,K,V); assert minV[t]-eps <= out[i][t] <= maxV[t]+eps',
    complexity: 'O(n²·d + n·d)', ref: '凸组合夹取定理 · 注意力加权和语义',
    commonPitfall: '夹取界逐坐标取；加权和本身另用反向累加参考路径（attnFromScoresRev）验证，双重判据互补。',
    invariant: '逐坐标 outᵢₜ ∈ [minⱼVⱼₜ−1e-12, maxⱼVⱼₜ+1e-12]；且 rc(正向加权和, 反向加权和, 1e-9)',
    inputs: ['Q', 'K', 'V'], domains: { Q: 'matSmall', K: 'matSmall', V: 'matSmall' },
    matrixDims: { Q: [4, 4], K: [4, 4], V: [4, 4] },
    run: (Q, K, V) => attnFromQKV(Q, K, V, sq(Q), Q.length / sq(Q)),
    predicate: (Q, K, V) => {
      const n = sq(Q), d = Q.length / n;
      const out = attnFromQKV(Q, K, V, n, d);
      const P = softmaxRows(causal(scoresFromQ(Q, K, n, n, d), n), n);
      if (!rc(out, attnFromScoresRev(P, V, n, n, d), 1e-9)) return false;   // 参考：反向累加
      for (let i = 0; i < n; i++) for (let t = 0; t < d; t++) {
        let lo = Infinity, hi = -Infinity;
        for (let j = 0; j < n; j++) { const v = V[j * d + t]; if (v < lo) lo = v; if (v > hi) hi = v; }
        const o = out[i * d + t];
        if (!(o >= lo - 1e-12 && o <= hi + 1e-12)) return false;
      }
      return true;
    },
  });
  reg(100, {
    name: '组装 · 因果行支撑大小',
    intuition: '第 i 行的注意力只能落在前 i+1 个位置上：未来位权重严格为 0，正权重个数恰为 i+1。',
    math: '|{j : Pᵢⱼ > 0}| = i+1；总数 = n(n+1)/2',
    pseudocode: 'P = softmaxRows(causal(S)); for i: assert count(P[i]>0) === i+1',
    complexity: 'O(n²)', ref: '因果注意力支撑集 · Vaswani et al. 2017 §3.2.3',
    commonPitfall: '正权重要严格 >0：掩位 exp(−∞)=0 精确为 0，可见位 exp(s−mx) 在域内不下溢（打分跨度 ≤400 ≪ 708），故可数。',
    invariant: '第 i 行正权重个数 === i+1；Σᵢ(i+1) === n(n+1)/2',
    inputs: ['Q', 'K'], domains: { Q: 'matSmall', K: 'matSmall' }, matrixDims: { Q: [4, 4], K: [4, 4] },
    run: (Q, K) => {
      const n = sq(Q), d = Q.length / n;
      const P = softmaxRows(causal(scoresFromQ(Q, K, n, n, d), n), n);
      const cnt = new Float64Array(n);
      for (let i = 0; i < n; i++) { let c = 0; for (let j = 0; j < n; j++) if (P[i * n + j] > 0) c++; cnt[i] = c; }
      return cnt;
    },
    predicate: (Q, K) => {
      const n = sq(Q), d = Q.length / n;
      const P = softmaxRows(causal(scoresFromQ(Q, K, n, n, d), n), n);
      let total = 0;
      for (let i = 0; i < n; i++) {
        let c = 0;
        for (let j = 0; j < n; j++) if (P[i * n + j] > 0) c++;
        if (c !== i + 1) return false;
        total += c;
      }
      return total === n * (n + 1) / 2;
    },
  });
  reg(101, {
    name: '组装 · 等分打分权重恰为 1/n',
    intuition: '所有打分相等时 softmax 退化为均匀分布——exp(0)=1、整数和精确，可逐位断言而非容差。',
    math: 'sᵢⱼ 全相等 ⇒ Pᵢⱼ = 1/(i+1)（j≤i，逐位）',
    pseudocode: 'Q=K=行全为 v; P = softmaxRows(causal(S)); assert Object.is(P[i][j], 1/(i+1))',
    complexity: 'O(n²·d + n²)', ref: 'softmax 均匀退化情形（D027 平移不变性的极值推论）',
    commonPitfall: '逐位成立的关键：各行打分由同一次点积算出（位级相等）⇒ s−mx=0 精确 ⇒ exp(0)=1 ⇒ 和为整数 i+1 ⇒ 除法结果位级一致。',
    invariant: 'j≤i 时 Object.is(Pᵢⱼ, 1/(i+1))；j>i 时 Pᵢⱼ===0',
    inputs: ['v'], domains: { v: 'vecSmall' }, vecDims: { v: 4 },
    run: (v) => {
      const n = 4, d = v.length, Q = new Float64Array(n * d), K = new Float64Array(n * d);
      for (let i = 0; i < n; i++) { Q.set(v, i * d); K.set(v, i * d); }
      return softmaxRows(causal(scoresFromQ(Q, K, n, n, d), n), n);
    },
    predicate: (v) => {
      const n = 4, d = v.length, Q = new Float64Array(n * d), K = new Float64Array(n * d);
      for (let i = 0; i < n; i++) { Q.set(v, i * d); K.set(v, i * d); }
      const P = softmaxRows(causal(scoresFromQ(Q, K, n, n, d), n), n);
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        if (j <= i) { if (!Object.is(P[i * n + j], 1 / (i + 1))) return false; }
        else if (P[i * n + j] !== 0) return false;
      }
      return true;
    },
  });
  reg(102, {
    name: '组装 · 低温逼近 one-hot',
    intuition: '温度趋零时 softmax 坍缩到最大打分位；有限低温下用「top-2 间距 ≥ 40·T」守卫给出可断言的数值版极限。',
    math: 'g = sᵢ*−max_{j≠i*}sⱼ；恒有 pᵢ* ≥ 1−(n−1)e^{−g/T}−1e-12；g/T ≥ 40 ⇒ pᵢ* > 1−1e-12',
    pseudocode: 'u = s/T; i* = argmax(s); g = gap; assert tight bound; guarded: strict order && one-hot',
    complexity: 'O(n)', ref: 'softmax 温度极限 · Gibbs 分布集中性',
    commonPitfall: '历史教训的守卫是 si−sj>1e-6 && si>−740；本项进一步收紧为缩放间距 g/T ≥ 40——e^−708 以下进入次正规带（相对粒度退化到 1e−4~1e−2），−740 附近的 exp 已不可靠，而 g/T ≥ 40 保证全部 exp 操作数 ≤ −40，位于正规区（相对精度 ~1e−16 ≪ 1e−6），严格序可证明保持。成对严格序守卫同时要求较大者的 exp 操作数 > −705（扣除 log Σe ≤ log 4 的裕量后仍 > −708 正规下界）。T = 0.05·(1+|t|%20) ∈ [0.05,1] 为低温段，不断言 T→0 的符号极限，只用有限 T 的紧界。',
    invariant: 'pᵢ* ≥ 1−(n−1)·e^{−g/T}−1e-12；(uᵢ−uⱼ>1e-6 且 uᵢ−max(u)>−705) ⇒ pᵢ>pⱼ；g/T ≥ 40 ⇒ pᵢ* > 1−1e-12 且严格最大',
    inputs: ['s', 't'], domains: { s: 'vecSmall', t: 'nat' }, vecDims: { s: 4 },
    run: (s, t) => {
      const T = 0.05 * (1 + (Math.abs(t) % 20)), n = s.length, u = new Float64Array(n);
      for (let j = 0; j < n; j++) u[j] = s[j] / T;
      return softmaxRow(u, 0, n);
    },
    predicate: (s, t) => {
      const T = 0.05 * (1 + (Math.abs(t) % 20)), n = s.length;
      let iStar = 0;
      for (let j = 1; j < n; j++) if (s[j] > s[iStar]) iStar = j;
      let second = -Infinity;
      for (let j = 0; j < n; j++) if (j !== iStar && s[j] > second) second = s[j];
      const g = s[iStar] - second;
      const u = new Float64Array(n);
      let mx = -Infinity;
      for (let j = 0; j < n; j++) { u[j] = s[j] / T; if (u[j] > mx) mx = u[j]; }
      const li = lse(u, 0, n);                          // 参考路径：lse 表出
      const p = new Float64Array(n);
      for (let j = 0; j < n; j++) p[j] = Math.exp(u[j] - li);
      if (!(p[iStar] >= 1 - (n - 1) * Math.exp(-g / T) - 1e-12)) return false;   // 紧界（恒断言）
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {                 // 严格序（守卫）
        if (i === j) continue;
        if (u[i] - u[j] > 1e-6 && u[i] - mx > -705 && !(p[i] > p[j])) return false;
      }
      if (g / T >= 40) {                                // one-hot（守卫：top-2 缩放间距 ≥40）
        if (!(p[iStar] > 1 - 1e-12)) return false;
        for (let j = 0; j < n; j++) if (j !== iStar && !(p[iStar] > p[j])) return false;
      }
      return true;
    },
  });
  reg(103, {
    name: '组装 · V 平移 ⇒ 输出平移',
    intuition: 'V 每个坐标加同一个常数 c，输出也加 c——权重和为 1，平移被无损传递。',
    math: 'out(Q,K,V+c·1) = out(Q,K,V) + c',
    pseudocode: 'a = attnFromQKV(Q,K,add(V,c)); b = attnFromQKV(Q,K,V); return relClose(a-c, b, 1e-9)',
    complexity: 'O(n²·d)', ref: '凸组合平移不变性（ΣⱼPᵢⱼ=1 的推论，D027 的组装形式）',
    commonPitfall: '不是逐位断言：c 先加进 V 再加权 与 加权后再加 c，两条舍入路径不同，用 1e-9 相对容差。',
    invariant: 'relClose(out(V+c·1) − c, out(V), 1e-9)',
    inputs: ['Q', 'K', 'V', 'c'], domains: { Q: 'matSmall', K: 'matSmall', V: 'matSmall', c: 'small' },
    matrixDims: { Q: [4, 4], K: [4, 4], V: [4, 4] },
    run: (Q, K, V, c) => {
      const n = sq(V), d = V.length / n, V2 = new Float64Array(V.length);
      for (let k = 0; k < V.length; k++) V2[k] = V[k] + c;
      return attnFromQKV(Q, K, V2, n, d);
    },
    predicate: (Q, K, V, c) => {
      const n = sq(V), d = V.length / n, V2 = new Float64Array(V.length);
      for (let k = 0; k < V.length; k++) V2[k] = V[k] + c;
      const a = attnFromQKV(Q, K, V2, n, d), b = attnFromQKV(Q, K, V, n, d);
      for (let i = 0; i < a.length; i++) if (!rc(a[i] - c, b[i], 1e-9)) return false;
      return true;
    },
  });
  reg(104, {
    name: '组装 · 端到端与 lse 参考一致',
    intuition: '把缩放、掩码、softmax、加权和四步串起来后，必须与「lse 表出概率 + 反向累加」的独立参考路径吻合。',
    math: 'out = softmax(causal(QKᵀ/√d))·V = Σⱼexp(sᵢⱼ−lseᵢ)·Vⱼ',
    pseudocode: 'out = attnFromQKV(Q,K,V); ref = reverse-accumulate(exp(s-lse)*V); return relClose(out, ref, 1e-9)',
    complexity: 'O(n²·d)', ref: 'Milakov & Gimelshein 2018（在线 softmax 的 lse 表出） · Vaswani et al. 2017 §3.2.2',
    commonPitfall: '正向累加（run）与 lse 表出 + 反向累加（predicate）是两条独立路径；只用同一 helper 家族内的位级断言会自证。',
    invariant: 'relClose(out, Σⱼexp(sᵢⱼ−lseᵢ)·Vⱼ（反向累加）, 1e-9)',
    inputs: ['Q', 'K', 'V'], domains: { Q: 'matSmall', K: 'matSmall', V: 'matSmall' },
    matrixDims: { Q: [4, 4], K: [4, 4], V: [4, 4] },
    run: (Q, K, V) => attnFromQKV(Q, K, V, sq(Q), Q.length / sq(Q)),
    predicate: (Q, K, V) => {
      const n = sq(Q), d = Q.length / n;
      const S = causal(scoresFromQ(Q, K, n, n, d), n);
      const out = attnFromQKV(Q, K, V, n, d);
      for (let i = 0; i < n; i++) {
        const li = lse(S, i * n, n);
        for (let t = 0; t < d; t++) {
          let acc = 0;
          for (let j = n - 1; j >= 0; j--) acc += Math.exp(S[i * n + j] - li) * V[j * d + t];   // 参考：反向累加
          if (!rc(out[i * d + t], acc, 1e-9)) return false;
        }
      }
      return true;
    },
  });
  reg(105, {
    name: '组装 · 因果首行逐位等于 V 首行',
    intuition: '第 0 个位置只能看自己：权重向量 [1,0,0,0] 是精确值，输出必须与 V 第 0 行逐位相同。',
    math: 'P₀ = [1,0,…,0]（精确） ⇒ out₀ = V₀（逐位）',
    pseudocode: 'out = attnFromQKV(Q,K,V); assert bitEq(out[0], V[0])',
    complexity: 'O(d)', ref: '因果注意力第一行 · 逐位一致性（零回归判据）',
    commonPitfall: '0·Vⱼ 可能产出 ±0，与首项相加时符号零会「吃掉」符号——用 ±0 归一的逐位比较（bitEq），不放宽容差。',
    invariant: 'bitEq(out₀, V₀)（±0 归一后 Object.is）',
    inputs: ['Q', 'K', 'V'], domains: { Q: 'matSmall', K: 'matSmall', V: 'matSmall' },
    matrixDims: { Q: [4, 4], K: [4, 4], V: [4, 4] },
    run: (Q, K, V) => attnFromQKV(Q, K, V, sq(Q), Q.length / sq(Q)).subarray(0, Q.length / sq(Q)),
    predicate: (Q, K, V) => {
      const n = sq(Q), d = Q.length / n;
      const out = attnFromQKV(Q, K, V, n, d);
      return bitEq(out.subarray(0, d), V.subarray(0, d));
    },
  });
  reg(106, {
    name: '组装 · V 缩放 ⇒ 输出缩放',
    intuition: '加权和对 V 线性：V 整体放大 a 倍，输出也放大 a 倍（放 Q 则不然——那会改变 softmax 形状）。',
    math: 'out(Q,K,a·V) = a·out(Q,K,V)',
    pseudocode: 'x = attnFromQKV(Q,K,scale(V,a)); y = attnFromQKV(Q,K,V); return relClose(x, a*y, 1e-9)',
    complexity: 'O(n²·d)', ref: '加权和线性性（对值向量齐次） · 与 D003（打分齐次性）对偶',
    commonPitfall: '缩放 Q 会重塑 softmax（非线性），没有此恒等式；a 取整数域（smallInt）避免缩放本身引入额外舍入。',
    invariant: 'relClose(out(Q,K,a·V), a·out(Q,K,V), 1e-9)',
    inputs: ['Q', 'K', 'V', 'a'], domains: { Q: 'matSmall', K: 'matSmall', V: 'matSmall', a: 'smallInt' },
    matrixDims: { Q: [4, 4], K: [4, 4], V: [4, 4] },
    run: (Q, K, V, a) => {
      const n = sq(V), d = V.length / n, V2 = new Float64Array(V.length);
      for (let k = 0; k < V.length; k++) V2[k] = a * V[k];
      return attnFromQKV(Q, K, V2, n, d);
    },
    predicate: (Q, K, V, a) => {
      const n = sq(V), d = V.length / n, V2 = new Float64Array(V.length);
      for (let k = 0; k < V.length; k++) V2[k] = a * V[k];
      const x = attnFromQKV(Q, K, V2, n, d), y = attnFromQKV(Q, K, V, n, d);
      for (let i = 0; i < x.length; i++) if (!rc(x[i], a * y[i], 1e-9)) return false;
      return true;
    },
  });
  reg(107, {
    name: '组装 · query 行局部性',
    intuition: '因果性使输出第 i 行只依赖 Q 第 i 行：扰动 Q 的末段行，前 m+1 行输出逐位不变。',
    math: 'Q′ 仅第 m+1..n−1 行不同 ⇒ ∀i≤m：out(Q′)[i] = out(Q)[i]（逐位）',
    pseudocode: 'Q2 = Q with rows > m shifted by +777; assert bitEq(out(Q2)[0..m], out(Q)[0..m])',
    complexity: 'O(n²·d)', ref: '因果注意力行独立性 · 增量解码正确性基础（与 D128 呼应）',
    commonPitfall: '逐位成立的关键：被比较行的全部输入（Q 第 i 行、K、V）位级相同 ⇒ 同一计算图 ⇒ 位级相同输出。',
    invariant: 'bitEq(out(Q₂) 第 0..m 行, out(Q) 第 0..m 行)；m = |m|%3 恒 ≤2（至少扰动第 3 行）',
    inputs: ['Q', 'K', 'V', 'm'], domains: { Q: 'matSmall', K: 'matSmall', V: 'matSmall', m: 'nat' },
    matrixDims: { Q: [4, 4], K: [4, 4], V: [4, 4] },
    run: (Q, K, V, m) => {
      const n = sq(Q), d = Q.length / n, mm = Math.abs(m) % 3;
      const Q2 = Q.slice();
      for (let i = mm + 1; i < n; i++) for (let t = 0; t < d; t++) Q2[i * d + t] = Q[i * d + t] + 777;
      return attnFromQKV(Q2, K, V, n, d);
    },
    predicate: (Q, K, V, m) => {
      const n = sq(Q), d = Q.length / n, mm = Math.abs(m) % 3;
      const Q2 = Q.slice();
      for (let i = mm + 1; i < n; i++) for (let t = 0; t < d; t++) Q2[i * d + t] = Q[i * d + t] + 777;
      const a = attnFromQKV(Q2, K, V, n, d), b = attnFromQKV(Q, K, V, n, d);
      for (let i = 0; i <= mm; i++) if (!bitEq(a.subarray(i * d, (i + 1) * d), b.subarray(i * d, (i + 1) * d))) return false;
      return true;
    },
  });
  reg(108, {
    name: '组装 · V 常列 ⇒ 输出常列',
    intuition: 'V 某一列在所有行取同一常数 γ 时，输出该列 = γ·ΣⱼPᵢⱼ ≈ γ；其余列因输入未动而逐位不变。',
    math: 'V′ⱼₜ = γ（∀j） ⇒ out′ᵢₜ = γ·ΣⱼPᵢⱼ ≈ γ；t≠t₀ 列逐位不变',
    pseudocode: 'V2 = V with column t0 set to c; assert rc(out2[i][t0], c, 1e-9) && bitEq(other columns)',
    complexity: 'O(n²·d)', ref: '凸组合的列结构保持（ΣⱼPᵢⱼ=1 的列形式推论，与 D103 互为行列对偶）',
    commonPitfall: '目标列是 ≈γ 而非逐位 γ（行和有 1e−16 级舍入）；非目标列才是逐位判据——两类判据别混用。',
    invariant: 'rc(out′ᵢₜ₀, γ, 1e-9)；t≠t₀ 时 bitEq(out′ᵢₜ, outᵢₜ)；t₀ = |t|%4',
    inputs: ['Q', 'K', 'V', 't', 'c'], domains: { Q: 'matSmall', K: 'matSmall', V: 'matSmall', t: 'nat', c: 'small' },
    matrixDims: { Q: [4, 4], K: [4, 4], V: [4, 4] },
    run: (Q, K, V, t, c) => {
      const n = sq(V), d = V.length / n, t0 = Math.abs(t) % d, V2 = V.slice();
      for (let j = 0; j < n; j++) V2[j * d + t0] = c;
      return attnFromQKV(Q, K, V2, n, d);
    },
    predicate: (Q, K, V, t, c) => {
      const n = sq(V), d = V.length / n, t0 = Math.abs(t) % d, V2 = V.slice();
      for (let j = 0; j < n; j++) V2[j * d + t0] = c;
      const a = attnFromQKV(Q, K, V2, n, d), b = attnFromQKV(Q, K, V, n, d);
      for (let i = 0; i < n; i++) {
        if (!rc(a[i * d + t0], c, 1e-9)) return false;
        for (let tt = 0; tt < d; tt++) if (tt !== t0 && !Object.is(a[i * d + tt], b[i * d + tt])) return false;
      }
      return true;
    },
  });

  // ══ 位置编码进阶（D109–D120）════════════════════════════════════════
  reg(109, {
    name: 'RoPE · 打分等于 q 与相对旋转 k 的内积',
    intuition: '把两侧旋转坍缩成「只旋转 k、角度取相对距离 n−m」：相对性的单侧形式，工程上可缓存旋转后的 k。',
    math: '⟨R_m q, R_n k⟩ = ⟨q, R_{n−m} k⟩',
    pseudocode: 'a = dot(rope(q,m), rope(k,n)); b = dot(q, rope(k,n-m)); return relClose(a,b,1e-9)',
    complexity: 'O(d)', ref: 'Su et al. 2021 §3.3（相对位置性质） · 与 D039（双侧平移）互为对偶',
    commonPitfall: 'D039 断言「双侧同时平移打分不变」；此处断言「单侧坍缩为相对旋转」——数学等价但计算路径不同，两条都是必要的回归判据。',
    invariant: 'relClose(⟨R_m q, R_n k⟩, ⟨q, R_{n−m} k⟩, 1e-9)（d=4，θ=10000）',
    inputs: ['q', 'k', 'm', 'n'], domains: { q: 'vecSmall', k: 'vecSmall', m: 'small', n: 'small' },
    vecDims: { q: 4, k: 4 },
    run: (q, k, m, n) => dot(rope(q, m, 4, THETA), rope(k, n, 4, THETA)),
    predicate: (q, k, m, n) => rc(dot(rope(q, m, 4, THETA), rope(k, n, 4, THETA)), dot(q, rope(k, n - m, 4, THETA)), 1e-9),
  });
  reg(110, {
    name: 'RoPE · 步进角严格有序',
    intuition: '同一位置增量下，高频对的角速度更大：步进角随对索引严格递减（pos>0）——多尺度位置分辨力的来源。',
    math: 'aᵢ = pos·θ^(−2i/d)；pos≠0 ⇒ (aᵢ−aᵢ₊₁)·pos > 0；a₀ ≡ pos（逐位）',
    pseudocode: 'f = invFreq(d,theta); a[i] = pos*f[i]; assert (a[i]-a[i+1])*pos > 0; assert Object.is(a[0], pos)',
    complexity: 'O(d/2)', ref: 'Su et al. 2021 §3.2（D034 频率单调性的「步进角」量化形式）',
    commonPitfall: '严格不等式的鲁棒性：最紧相邻间隙 ≥ 9e−3·|pos|（θ=10000、d=8 时 f₂−f₃=9e−3），比舍入误差 ~1e−16·|pos| 大 13 个数量级，故可严格断言；f[0]=1 精确 ⇒ pos·1=pos 逐位。',
    invariant: 'pos≠0 ⇒ ∀i：(aᵢ−aᵢ₊₁)·pos > 0；Object.is(a₀, pos)；全部有限',
    inputs: ['pos'], domains: { pos: 'small' },
    run: (pos) => {
      const f = invFreq(D8, THETA), out = new Float64Array(f.length);
      for (let i = 0; i < f.length; i++) out[i] = pos * f[i];
      return out;
    },
    predicate: (pos) => {
      const f = invFreq(D8, THETA), a = new Float64Array(f.length);
      for (let i = 0; i < f.length; i++) {
        a[i] = pos * f[i];
        if (!Number.isFinite(a[i])) return false;
      }
      if (!Object.is(a[0], pos)) return false;
      if (pos === 0) return true;                       // 前置：pos≠0 才断序
      for (let i = 0; i + 1 < a.length; i++) if (!((a[i] - a[i + 1]) * pos > 0)) return false;
      return true;
    },
  });
  reg(111, {
    name: 'RoPE · 基向量旋转形式',
    intuition: '只有第 i 对非零时，旋转把质量搬到 (cos a, sin a) 方向：验证 rope 确实实现了二维旋转而非近似。',
    math: 'R(a)·(x₂ᵢ, x₂ᵢ₊₁) = r·(cos(φ+a), sin(φ+a))，r=hypot(x₂ᵢ,x₂ᵢ₊₁)，φ=atan2(x₂ᵢ₊₁,x₂ᵢ)；其余维 === 0',
    pseudocode: 'e = 仅第 i 对非零; o = rope(e); assert rc(o[2i], polar form, 1e-12) && o[其他]===0',
    complexity: 'O(d)', ref: 'Su et al. 2021 §3.2（分块对角旋转的显式形式，D043 的量化强化）',
    commonPitfall: '参考路径用极式分解（hypot + atan2）独立重算，不调用 rope 也不复用它的笛卡尔闭式；i 由 nat 取模到 [0,4)，d=8 半维 4 对。',
    invariant: 'rc(o[2i], r·cos(φ+a), 1e-12)；rc(o[2i+1], r·sin(φ+a), 1e-12)；其余维 === 0',
    inputs: ['x', 'pos', 'i'], domains: { x: 'vecSmall', pos: 'small', i: 'nat' }, vecDims: { x: D8 },
    run: (x, pos, i) => {
      const e = new Float64Array(x.length), ii = Math.abs(i) % 4;
      e[2 * ii] = x[2 * ii]; e[2 * ii + 1] = x[2 * ii + 1];
      return rope(e, pos, D8, THETA);
    },
    predicate: (x, pos, i) => {
      const e = new Float64Array(x.length), ii = Math.abs(i) % 4;
      e[2 * ii] = x[2 * ii]; e[2 * ii + 1] = x[2 * ii + 1];
      const o = rope(e, pos, D8, THETA);
      const a = pos * invFreq(D8, THETA)[ii];
      const x0 = x[2 * ii], x1 = x[2 * ii + 1];
      const r = Math.hypot(x0, x1), ph = Math.atan2(x1, x0);       // 参考路径：极式分解
      if (!rc(o[2 * ii], r * Math.cos(ph + a), 1e-12)) return false;
      if (!rc(o[2 * ii + 1], r * Math.sin(ph + a), 1e-12)) return false;
      for (let t = 0; t < o.length; t++) if (t !== 2 * ii && t !== 2 * ii + 1 && o[t] !== 0) return false;
      return true;
    },
  });
  reg(112, {
    name: 'ALiBi · 等分打分的几何衰减',
    intuition: '基分全相等时，ALiBi 把软最大质量压向最近邻：权重随距离几何衰减，且对角线严格最大。',
    math: 'sⱼ = −slope·|i−j| ⇒ log(pⱼ/pᵢ) = −slope·|i−j|；pᵢ > pⱼ（j≠i）',
    pseudocode: 's[j] = -slope*abs(i-j); p = softmaxRow(s); assert p[i] > p[j] && rc(log(p[j]/p[i]), -slope*|i-j|, 1e-9)',
    complexity: 'O(m)', ref: 'Press et al. 2022 (ALiBi) §3 · 等基分情形的闭式（D047/D054 的分布层结论）',
    commonPitfall: '严格最大性依赖 slope>0（域 smallPos 保证 ≥1e−6）；比值比较取对数——pⱼ 可小到 e^{−70}，直接比绝对差会淹没精度。',
    invariant: 'm≥2 且 slope>0 ⇒ pᵢ 严格最大；∀j≠i：rc(log(pⱼ/pᵢ), −slope·|i−j|, 1e-9)',
    inputs: ['n', 'slope', 'i'], domains: { n: 'nat', slope: 'smallPos', i: 'nat' },
    run: (n, slope, i) => {
      const m = 1 + (Math.abs(n) % 8), iq = Math.abs(i) % m;
      const s = new Float64Array(m);
      for (let j = 0; j < m; j++) s[j] = -slope * Math.abs(iq - j);
      return softmaxRow(s, 0, m);
    },
    predicate: (n, slope, i) => {
      const m = 1 + (Math.abs(n) % 8), iq = Math.abs(i) % m;
      if (m < 2) return true;                            // 前置：至少两个位置才可比较
      const s = new Float64Array(m);
      for (let j = 0; j < m; j++) s[j] = -slope * Math.abs(iq - j);
      const p = softmaxRow(s, 0, m);
      for (let j = 0; j < m; j++) {
        if (j === iq) continue;
        if (!(p[iq] > p[j])) return false;
        if (!rc(Math.log(p[j] / p[iq]), -slope * Math.abs(iq - j), 1e-9)) return false;
      }
      return true;
    },
  });
  reg(113, {
    name: 'RoPE×ALiBi · 联合平移不变',
    intuition: '行打分 = RoPE 相对项 + ALiBi 距离项；query 与全部 key 同时平移 p，两项都不变——行分布冻结。',
    math: 'softmaxⱼ(⟨R_{m+p}q, R_{j+p}kⱼ⟩/√d − slope·|m−j|) = softmaxⱼ(⟨R_m q, R_j kⱼ⟩/√d − slope·|m−j|)',
    pseudocode: 'base = rowLogits(m,0); shifted = rowLogits(m+p,p); return rc(lse) && rc(dist)',
    complexity: 'O(n·d)', ref: 'Su et al. 2021 §3.3 + Press et al. 2022 §3（D039/D054 的复合层推论）',
    commonPitfall: 'RoPE 项与 ALiBi 项各自平移不变，复合后仍不变；分布用 lse 参考路径表出，不与 softmaxRow 自证；与 D120（纯 RoPE 基座）分工明确。',
    invariant: 'rc(平移后行分布, 原行分布, 1e-9)；rc(平移后 lse, 原 lse, 1e-9)',
    inputs: ['q', 'K', 'm', 'p', 'slope'], domains: { q: 'vecSmall', K: 'matSmall', m: 'small', p: 'small', slope: 'smallPos' },
    matrixDims: { K: [4, 4] }, vecDims: { q: 4 },
    run: (q, K, m, p, slope) => {
      const n = sq(K), d = q.length, sc = 1 / Math.sqrt(d);
      const s = new Float64Array(n);
      for (let j = 0; j < n; j++) {
        s[j] = dot(rope(q, m + p, d, THETA), rope(K.subarray(j * d, (j + 1) * d), j + p, d, THETA)) * sc - slope * Math.abs(m - j);
      }
      return softmaxRow(s, 0, n);
    },
    predicate: (q, K, m, p, slope) => {
      const n = sq(K), d = q.length, sc = 1 / Math.sqrt(d);
      const rowLogits = (mq, koff) => {
        const s = new Float64Array(n);
        for (let j = 0; j < n; j++) {
          s[j] = dot(rope(q, mq, d, THETA), rope(K.subarray(j * d, (j + 1) * d), koff + j, d, THETA)) * sc - slope * Math.abs(mq - (koff + j));
        }
        return s;
      };
      const base = rowLogits(m, 0), shifted = rowLogits(m + p, p);
      const lb = lse(base, 0, n), ls = lse(shifted, 0, n);
      if (!rc(ls, lb, 1e-9)) return false;
      const pb = new Float64Array(n), ps = new Float64Array(n);
      for (let j = 0; j < n; j++) { pb[j] = Math.exp(base[j] - lb); ps[j] = Math.exp(shifted[j] - ls); }   // 参考：lse 表出
      return rc(ps, pb, 1e-9);
    },
  });
  reg(114, {
    name: 'ALiBi · 斜率累积积闭合式',
    intuition: '前 h 个斜率的乘积有闭式（指数等差求和）：它决定「级联 ALiBi 头」的等效衰减强度。',
    math: 'Π_{i<h}2^(−8i/H) = 2^(−4h(h−1)/H)',
    pseudocode: 'P[h] = prod(slopes[0..h-1]); assert rc(P[h], pow(2,-4*h*(h-1)/H), 1e-12) && P[h+1] <= P[h]',
    complexity: 'O(H)', ref: 'Press et al. 2022 §3 · 等差指数求和（D045 斜率级数的积形式）',
    commonPitfall: 'H=1 时只有 P[0]=1（空积）；单调性判据用 ≤——slope[0]=1 使前两项相等，不是严格递减。',
    invariant: '∀h≤H：rc(P[h], 2^(−4h(h−1)/H), 1e-12)；P 非增',
    inputs: ['h'], domains: { h: 'nat' },
    run: (h) => {
      const s = alibiSlopes(h), out = new Float64Array(s.length + 1);
      out[0] = 1;
      for (let i = 0; i < s.length; i++) out[i + 1] = out[i] * s[i];
      return out;
    },
    predicate: (h) => {
      const H = 1 + (Math.abs(h) % 16), s = alibiSlopes(h), P = new Float64Array(H + 1);
      P[0] = 1;
      for (let i = 0; i < H; i++) P[i + 1] = P[i] * s[i];
      for (let k = 0; k <= H; k++) {
        if (!rc(P[k], Math.pow(2, -4 * k * (k - 1) / H), 1e-12)) return false;   // 参考：闭式
        if (k + 1 <= H && P[k + 1] > P[k]) return false;
      }
      return true;
    },
  });
  reg(115, {
    name: 'RoPE · 旋转只触碰前 2⌊d/2⌋ 维',
    intuition: 'd 为奇数时第 d−1 维（尾维）没有旋转伴侣，必须恒等；d 为偶数时第 d 维及以后同理不被触碰。',
    math: 'supp(R) ⊆ [0, 2⌊d/2⌋)；t ≥ 2⌊d/2⌋ ⇒ Object.is(out[t], x[t])',
    pseudocode: 'o = rope(x, pos, dEff, theta); for t >= 2*floor(dEff/2): assert Object.is(o[t], x[t])',
    complexity: 'O(d)', ref: 'Su et al. 2021 §3.2（floor 配对的边界情形，D043 的尾维强化）',
    commonPitfall: '奇数头维是最常见的形状 bug：若按 (d+1)/2 配对会越界写或读错位；此处用逐位断言锁死 floor 语义。',
    invariant: '∀t ≥ 2⌊d/2⌋：Object.is(o[t], x[t])（±0 归一）；dEff = 2+(|d|%7) ∈ [2,8]',
    inputs: ['x', 'pos', 'd'], domains: { x: 'vecSmall', pos: 'small', d: 'nat' }, vecDims: { x: D8 },
    run: (x, pos, d) => rope(x, pos, 2 + (Math.abs(d) % 7), THETA),
    predicate: (x, pos, d) => {
      const dEff = 2 + (Math.abs(d) % 7), o = rope(x, pos, dEff, THETA);
      const bound = 2 * Math.floor(dEff / 2);
      for (let t = bound; t < x.length; t++) {
        let u = o[t], v = x[t];
        if (u === 0) u = 0; if (v === 0) v = 0;
        if (!Object.is(u, v)) return false;
      }
      return true;
    },
  });
  reg(116, {
    name: 'RoPE · 频率公比恒定',
    intuition: '逆频率是几何级数：相邻频率之比恒为 θ^(−2/d)，与对索引无关——「多尺度」的代数根源。',
    math: 'fᵢ₊₁/fᵢ = θ^(−2/d)（i = 0..⌊d/2⌋−2）；f₀ ≡ 1（逐位）',
    pseudocode: 'f = invFreq(d,theta); assert Object.is(f[0],1) && rc(f[i+1]/f[i], pow(theta,-2/d), 1e-12)',
    complexity: 'O(d/2)', ref: 'Su et al. 2021 §3.2（D034 单调性的量化形式）',
    commonPitfall: 'θ 可小于 1（域 pos 下限 1e−6）：此时公比 >1（频率递增），断言的是「比值恒定」而非「递减」；f[0]=θ^0=1 逐位精确。',
    invariant: 'f₀ === 1；∀i：rc(fᵢ₊₁/fᵢ, θ^(−2/d), 1e-12)；dEff = 2+(|d|%7)',
    inputs: ['d', 'theta'], domains: { d: 'nat', theta: 'pos' },
    run: (d, theta) => invFreq(2 + (Math.abs(d) % 7), theta),
    predicate: (d, theta) => {
      const dEff = 2 + (Math.abs(d) % 7), f = invFreq(dEff, theta), r = Math.pow(theta, -2 / dEff);
      if (!Object.is(f[0], 1)) return false;
      for (let i = 0; i + 1 < f.length; i++) if (!rc(f[i + 1] / f[i], r, 1e-12)) return false;
      return true;
    },
  });
  reg(117, {
    name: 'ALiBi · 行距离和闭合式',
    intuition: '一行偏置的绝对距离和有闭式（左右两段等差数列）：O(1) 核算整行的 ALiBi 总惩罚量。',
    math: 'Σⱼ|i−j| = i(i+1)/2 + (m−1−i)(m−i)/2',
    pseudocode: 's = sum(|i-j| for j); return rc(s, i*(i+1)/2 + (m-1-i)*(m-i)/2, 1e-12)',
    complexity: 'O(m)', ref: '等差数列求和 · Press et al. 2022 §3（行级聚合量）',
    commonPitfall: '两段分别是 j≤i 与 j>i 的距离和；闭式对 i=0 与 i=m−1 退化正确（一段为 0）；反向累加作参考路径。',
    invariant: 'rc(Σⱼ|i−j|, i(i+1)/2 + (m−1−i)(m−i)/2, 1e-12)；m = 1+(|n|%8)',
    inputs: ['n', 'i'], domains: { n: 'nat', i: 'nat' },
    run: (n, i) => {
      const m = 1 + (Math.abs(n) % 8), iq = Math.abs(i) % m;
      let s = 0;
      for (let j = 0; j < m; j++) s += Math.abs(iq - j);
      return s;
    },
    predicate: (n, i) => {
      const m = 1 + (Math.abs(n) % 8), iq = Math.abs(i) % m;
      let s = 0;
      for (let j = m - 1; j >= 0; j--) s += Math.abs(iq - j);      // 参考：反向累加
      return rc(s, iq * (iq + 1) / 2 + (m - 1 - iq) * (m - iq) / 2, 1e-12);
    },
  });
  reg(118, {
    name: 'RoPE · 相邻位置打分 Lipschitz 界',
    intuition: '位置每推进一步，打分变化被各频率对的弦长扼住：Σᵢ2sin(fᵢ/2)·‖qᵢ‖·‖kᵢ‖——RoPE 局部平滑性的定量形式。',
    math: '|g(m+1) − g(m)| ≤ Σᵢ 2·sin(fᵢ/2)·‖qᵢ‖₂·‖kᵢ‖₂，fᵢ = θ^(−2i/8)',
    pseudocode: 'g1 = dot(rope(q,m+1), k); g0 = dot(rope(q,m), k); assert |g1-g0| <= Σᵢ 2*sin(f[i]/2)*hypot(qᵢ)*hypot(kᵢ)',
    complexity: 'O(d)', ref: '旋转矩阵 Lipschitz 性（逐对弦长界）· Su et al. 2021',
    commonPitfall: '取逐对紧式而非粗式 2sin(f₀/2)·‖q‖·‖k‖——随机输入下 ‖R_{m+1}q−R_m q‖ ≈ 0.96·‖q₀‖（仅最高频对有满弦长）而 ‖q‖ 含全部 4 对，粗式松约 2 倍，连「频率翻倍」级 bug 都漏判；每对弦长 2sin(fᵢ/2)·‖qᵢ‖（fᵢ ≤ f₀=1），三角不等式求和；范数参考用 Math.hypot 累积路径，裕度用 (1+1e−12) 相对形式。',
    invariant: '|g(m+1)−g(m)| ≤ Σᵢ 2sin(fᵢ/2)·‖qᵢ‖·‖kᵢ‖·(1+1e-12)',
    inputs: ['q', 'k', 'm'], domains: { q: 'vecSmall', k: 'vecSmall', m: 'small' }, vecDims: { q: D8, k: D8 },
    run: (q, k, m) => Math.abs(dot(rope(q, m + 1, D8, THETA), k) - dot(rope(q, m, D8, THETA), k)),
    predicate: (q, k, m) => {
      const lhs = Math.abs(dot(rope(q, m + 1, D8, THETA), k) - dot(rope(q, m, D8, THETA), k));
      const f = invFreq(D8, THETA);
      let rhs = 0;
      for (let i = 0; i < f.length; i++) {
        rhs += 2 * Math.sin(f[i] / 2) * hypotRef(q.subarray(2 * i, 2 * i + 2)) * hypotRef(k.subarray(2 * i, 2 * i + 2));
      }
      return lhs <= rhs * (1 + 1e-12) + 1e-12;
    },
  });
  reg(119, {
    name: 'ALiBi · 行最小偏置闭合式',
    intuition: '一行里惩罚最重的位置是最远端点：闭式 −slope·max(i, m−1−i)，O(1) 给出行内下界。',
    math: 'minⱼ B[i][j] = −slope·max(i, m−1−i)',
    pseudocode: 'row = alibiBias(m,slope) 第 i 行; return rc(min(row), -slope*max(i, m-1-i), 1e-12)',
    complexity: 'O(m)', ref: 'Press et al. 2022 §3（行级极值，与 D050 距离单调性互补）',
    commonPitfall: '最远点是「左端 0 与右端 m−1 中更远的那个」，不是固定的某一端；反向扫描作参考路径。',
    invariant: 'rc(minⱼBᵢⱼ, −slope·max(i, m−1−i), 1e-12)；m = 1+(|n|%8)',
    inputs: ['n', 'slope', 'i'], domains: { n: 'nat', slope: 'smallPos', i: 'nat' },
    run: (n, slope, i) => {
      const m = 1 + (Math.abs(n) % 8), iq = Math.abs(i) % m, B = alibiBias(m, slope);
      let mn = Infinity;
      for (let j = 0; j < m; j++) { const v = B[iq * m + j]; if (v < mn) mn = v; }
      return mn;
    },
    predicate: (n, slope, i) => {
      const m = 1 + (Math.abs(n) % 8), iq = Math.abs(i) % m, B = alibiBias(m, slope);
      let mn = Infinity;
      for (let j = m - 1; j >= 0; j--) { const v = B[iq * m + j]; if (v < mn) mn = v; }   // 参考：反向扫描
      return rc(mn, -slope * Math.max(iq, m - 1 - iq), 1e-12);
    },
  });
  reg(120, {
    name: 'RoPE · 行分布整体平移不变',
    intuition: 'query 与全部 key 同时平移 p，RoPE 行打分逐项不变 ⇒ softmax 行分布冻结。这是 D113 去掉 ALiBi 后的基座。',
    math: 'softmaxⱼ(⟨R_{m+p}q, R_{j+p}kⱼ⟩/√d) = softmaxⱼ(⟨R_m q, R_j kⱼ⟩/√d)',
    pseudocode: 'base = rowLogits(m,0); shifted = rowLogits(m+p,p); return rc(lse) && rc(dist)',
    complexity: 'O(n·d)', ref: 'Su et al. 2021 §3.3（D039 的行分布形式；无偏置基座）',
    commonPitfall: '与 D113 的分工：此处纯 RoPE（无 ALiBi 项）断言行分布，D113 复合距离偏置后仍不变；两者共同覆盖「位置编码平移族」，不断言同一事实。',
    invariant: 'rc(平移后行分布, 原行分布, 1e-9)；rc(平移后 lse, 原 lse, 1e-9)',
    inputs: ['q', 'K', 'm', 'p'], domains: { q: 'vecSmall', K: 'matSmall', m: 'small', p: 'small' },
    matrixDims: { K: [4, 4] }, vecDims: { q: 4 },
    run: (q, K, m, p) => {
      const n = sq(K), d = q.length, sc = 1 / Math.sqrt(d);
      const s = new Float64Array(n);
      for (let j = 0; j < n; j++) {
        s[j] = dot(rope(q, m + p, d, THETA), rope(K.subarray(j * d, (j + 1) * d), j + p, d, THETA)) * sc;
      }
      return softmaxRow(s, 0, n);
    },
    predicate: (q, K, m, p) => {
      const n = sq(K), d = q.length, sc = 1 / Math.sqrt(d);
      const rowLogits = (mq, koff) => {
        const s = new Float64Array(n);
        for (let j = 0; j < n; j++) {
          s[j] = dot(rope(q, mq, d, THETA), rope(K.subarray(j * d, (j + 1) * d), koff + j, d, THETA)) * sc;
        }
        return s;
      };
      const base = rowLogits(m, 0), shifted = rowLogits(m + p, p);
      const lb = lse(base, 0, n), ls = lse(shifted, 0, n);
      if (!rc(ls, lb, 1e-9)) return false;
      const pb = new Float64Array(n), ps = new Float64Array(n);
      for (let j = 0; j < n; j++) { pb[j] = Math.exp(base[j] - lb); ps[j] = Math.exp(shifted[j] - ls); }   // 参考：lse 表出
      return rc(ps, pb, 1e-9);
    },
  });

  // ══ GQA×Cache 复合（D121–D128）══════════════════════════════════════
  reg(121, {
    name: '多头 cache · 头间写入独立性',
    intuition: '往某个头的缓存写行，绝不能碰到其他头的任何一行——多头 cache 的正确性底线。',
    math: 'append(cacheₕ, row) 后：∀k≠h, ∀p<len：bitEq(cache′ₖ.row(p), cacheₖ.row(p))',
    pseudocode: 'cs = [c0, c1]; cs[h] = append(cs[h], row); assert bitEq(cs[1-h].data, before)',
    complexity: 'O(H·cap·d)', ref: '多头 KV Cache 隔离契约 · 复合 D079–D086（追加语义）',
    commonPitfall: '共享底层缓冲区（视图别名）会让「写头 h」静默污染头 1−h；此处构造两个独立 cache 对象并逐位校验另一个的数据区；H=2、cap=2、d=4。',
    invariant: '写头 h 后，头 1−h 的 data 逐位不变；且头 h 的 len === L+1（L = |len|%2 < cap）',
    inputs: ['seed', 'h', 'len', 'row'], domains: { seed: 'matSmall', h: 'nat', len: 'nat', row: 'vecSmall' },
    matrixDims: { seed: [4, 4] }, vecDims: { row: 4 },
    run: (seed, h, len, row) => {
      const cap = 2, d = 4, H = 2, he = Math.abs(h) % H, L = Math.abs(len) % cap;
      const cs = [];
      for (let k = 0; k < H; k++) cs.push({ data: seed.slice(k * cap * d, (k + 1) * cap * d), len: L, cap, d });
      cs[he] = cacheAppend(cs[he], row);
      return cs[1 - he].data;
    },
    predicate: (seed, h, len, row) => {
      const cap = 2, d = 4, H = 2, he = Math.abs(h) % H, L = Math.abs(len) % cap;
      const cs = [];
      for (let k = 0; k < H; k++) cs.push({ data: seed.slice(k * cap * d, (k + 1) * cap * d), len: L, cap, d });
      const before = cs[1 - he].data.slice();
      cs[he] = cacheAppend(cs[he], row);
      if (!bitEq(cs[1 - he].data, before)) return false;
      return cs[he].len === L + 1;
    },
  });
  reg(122, {
    name: 'cache · 追加后 QK 打分前缀逐位不变',
    intuition: '追加只写新行：缓存里旧行内容位级不变 ⇒ 与同一 query 的打分也位级不变，新打分有限。',
    math: '∀j<L：Object.is(score′(j), score(j))；L<cap ⇒ isFinite(score(L))',
    pseudocode: 'c2 = cacheAppend(c, row); assert prefix scores bitwise equal && new score finite',
    complexity: 'O(cap·d)', ref: 'KV Cache 追加语义 + 打分的列独立性（D086/D011 的复合）',
    commonPitfall: '逐位成立的关键：cacheAppend 复制数据区且只写末尾 ⇒ 旧行位级不变 ⇒ 点积操作数位级不变 ⇒ 结果位级不变。',
    invariant: '前缀打分逐位 Object.is；未满时新打分有限；L = min(|len|%(cap+1), cap)',
    inputs: ['K', 'len', 'q', 'row'], domains: { K: 'matSmall', len: 'nat', q: 'vecSmall', row: 'vecSmall' },
    matrixDims: { K: [4, 4] }, vecDims: { q: 4, row: 4 },
    run: (K, len, q, row) => {
      const cap = sq(K), d = K.length / cap, L = Math.min(Math.abs(len) % (cap + 1), cap);
      const c2 = cacheAppend({ data: K, len: L, cap, d }, row);
      const sc = 1 / Math.sqrt(d), out = new Float64Array(c2.len);
      for (let j = 0; j < c2.len; j++) out[j] = dot(q, cacheRow(c2, j)) * sc;
      return out;
    },
    predicate: (K, len, q, row) => {
      const cap = sq(K), d = K.length / cap, L = Math.min(Math.abs(len) % (cap + 1), cap);
      const c = { data: K, len: L, cap, d };
      const c2 = cacheAppend(c, row);
      const sc = 1 / Math.sqrt(d);
      for (let j = 0; j < L; j++) {
        const before = dot(q, cacheRow(c, j)) * sc, after = dot(q, cacheRow(c2, j)) * sc;
        if (!Object.is(before, after)) return false;
      }
      if (L < cap && !Number.isFinite(dot(q, cacheRow(c2, L)) * sc)) return false;
      return true;
    },
  });
  reg(123, {
    name: 'GQA · 组内共享 KV 一致',
    intuition: '同一 GQA 组的 query 头吃同一份 KV：喂相同的 query 行，经过完整组装后输出必须逐位相同。',
    math: 'kv(h₁) = kv(h₂) ⇒ attnFromQKV(q, KV[kv(h₁)]) === attnFromQKV(q, KV[kv(h₂)])（逐位）',
    pseudocode: 'H=4, nKV=1+|g|%2; for 同组 (h1,h2): assert kvIdx equal && bitEq(out(h1), out(h2))',
    complexity: 'O(H·m·d²)', ref: 'Ainslie et al. 2023 §2.2（组共享的组装层判据，复合 D065 映射 + D097–D108 组装）',
    commonPitfall: '前置 kvIndexOf(h₁)===kvIndexOf(h₂)（H=4、nKV∈{1,2} 时头 0/1 与 2/3 同组）；「相同 query 内容 + 相同 kv 行内容 ⇒ 相同计算图」是逐位判据的依据，与 D073（repeat_kv 逐位）分工：D073 查复制，本项查组装输出。',
    invariant: '前置同组 ⇒ bitEq(out(h₁), out(h₂))（±0 归一）',
    inputs: ['q', 'KV', 'g'], domains: { q: 'vecSmall', KV: 'matSmall', g: 'nat' },
    matrixDims: { KV: [4, 4] }, vecDims: { q: 4 },
    run: (q, KV, g) => {
      const H = 4, nKV = 1 + (Math.abs(g) % 2), n = sq(KV), d = KV.length / n;
      const rows = Math.max(1, Math.floor(n / nKV));
      const out = new Float64Array(H * rows * d);
      for (let h = 0; h < H; h++) {
        const k = kvIndexOf(h, H, nKV);
        const sub = KV.subarray(k * rows * d, Math.min(n, (k + 1) * rows) * d);
        const m = sub.length / d, Q = new Float64Array(m * d);
        for (let i = 0; i < m; i++) Q.set(q, i * d);
        out.set(attnFromQKV(Q, sub, sub, m, d), h * rows * d);
      }
      return out;
    },
    predicate: (q, KV, g) => {
      const H = 4, nKV = 1 + (Math.abs(g) % 2), n = sq(KV), d = KV.length / n;
      const rows = Math.max(1, Math.floor(n / nKV));
      const headOut = (h) => {
        const k = kvIndexOf(h, H, nKV);
        const sub = KV.subarray(k * rows * d, Math.min(n, (k + 1) * rows) * d);
        const m = sub.length / d, Q = new Float64Array(m * d);
        for (let i = 0; i < m; i++) Q.set(q, i * d);
        return attnFromQKV(Q, sub, sub, m, d);
      };
      for (let h1 = 0; h1 < H; h1++) for (let h2 = h1 + 1; h2 < H; h2++) {
        if (kvIndexOf(h1, H, nKV) !== kvIndexOf(h2, H, nKV)) continue;   // 不同组：不适用
        if (!bitEq(headOut(h1), headOut(h2))) return false;
      }
      return true;
    },
  });
  reg(124, {
    name: '滚动窗口 · 可见长度守恒',
    intuition: '窗口满之后每次滚动追加都「进一出一」：可见长度恒为 cap，打分永远有限——流式解码不丢长度的判据。',
    math: '满窗后 N 次 cacheAppendWindowed：len === cap 恒成立；∀j<cap：isFinite(scoreⱼ)',
    pseudocode: 'fill to cap; for N: c = appendWindowed(c, freshRow, noSinks); assert c.len === cap',
    complexity: 'O(N·cap·d)', ref: 'Xiao et al. 2024 (StreamingLLM) §3.3 · 长度守恒（D082 滚动的打分层验证）',
    commonPitfall: '「无 sink」时每步淘汰最旧行；长度判据是 ===cap（精确），不是约等于；N = |n|%9 ∈ [0,8]。',
    invariant: '填满后每次追加 len === cap；最终全部可见打分有限',
    inputs: ['cache', 'n', 'q'], domains: { cache: 'matSmall', n: 'nat', q: 'vecSmall' },
    matrixDims: { cache: [4, 4] }, vecDims: { q: 4 },
    run: (cache, n, q) => {
      const cap = sq(cache), d = cache.length / cap, N = Math.abs(n) % 9;
      let c = { data: cache.slice(), len: cap, cap, d };
      const noSinks = new Float64Array(cap);
      for (let i = 0; i < N; i++) {
        const row = new Float64Array(d);
        for (let t = 0; t < d; t++) row[t] = 777 + i * d + t;
        c = cacheAppendWindowed(c, row, noSinks);
      }
      const sc = 1 / Math.sqrt(d), out = new Float64Array(c.len);
      for (let j = 0; j < c.len; j++) out[j] = dot(q, cacheRow(c, j)) * sc;
      return out;
    },
    predicate: (cache, n, q) => {
      const cap = sq(cache), d = cache.length / cap, N = Math.abs(n) % 9;
      let c = { data: cache.slice(), len: cap, cap, d };
      const noSinks = new Float64Array(cap);
      for (let i = 0; i < N; i++) {
        if (c.len !== cap) return false;
        const row = new Float64Array(d);
        for (let t = 0; t < d; t++) row[t] = 777 + i * d + t;
        c = cacheAppendWindowed(c, row, noSinks);
      }
      if (c.len !== cap) return false;
      const sc = 1 / Math.sqrt(d);
      for (let j = 0; j < c.len; j++) if (!Number.isFinite(dot(q, cacheRow(c, j)) * sc)) return false;
      return true;
    },
  });
  reg(125, {
    name: '增量 softmax · 旧权重等比缩放',
    intuition: '新块并入时，旧权重整体乘同一个常数 c（在线 softmax 的核心）：比例不变，只有尺度重标定。',
    math: 'M = max(m_old, s_new)；c = l_old·e^{m_old−M}/(l_old·e^{m_old−M} + e^{s_new−M})；peⱼ = c·pⱼ（j<L）',
    pseudocode: 'p = softmax(old); pe = softmax(old+[s_new]); assert pairwise ratios equal && rc(pe[0]/p[0], c_closed, 1e-9)',
    complexity: 'O(L)', ref: 'Milakov & Gimelshein 2018（在线 softmax 合并法则）',
    commonPitfall: '两两比值相等用 rc(peⱼ/pⱼ, peₖ/pₖ, 1e-9)——pⱼ 在域内恒 >0（打分跨度 ≤400，exp 不下溢）；闭式 c 由 lse 分解独立算出，与 pe₀/p₀ 互为交叉验证。',
    invariant: '∀j,k<L：rc(peⱼ/pⱼ, peₖ/pₖ, 1e-9)；rc(pe₀/p₀, c闭式, 1e-9)；rc(pe_L, 1−Σ旧, 1e-9)；L=0 时 pe₀===1',
    inputs: ['K', 'V', 'len', 'q', 'row'], domains: { K: 'matSmall', V: 'matSmall', len: 'nat', q: 'vecSmall', row: 'vecSmall' },
    matrixDims: { K: [4, 4], V: [4, 4] }, vecDims: { q: 4, row: 4 },
    run: (K, V, len, q, row) => {
      const cap = sq(K), d = K.length / cap, L = Math.min(Math.abs(len) % (cap + 1), cap), sc = 1 / Math.sqrt(d);
      const all = new Float64Array(L + 1);
      for (let j = 0; j < L; j++) all[j] = dot(q, K.subarray(j * d, (j + 1) * d)) * sc;
      all[L] = dot(q, row) * sc;
      return softmaxRow(all, 0, L + 1);
    },
    predicate: (K, V, len, q, row) => {
      const cap = sq(K), d = K.length / cap, L = Math.min(Math.abs(len) % (cap + 1), cap), sc = 1 / Math.sqrt(d);
      const sOld = new Float64Array(L);
      for (let j = 0; j < L; j++) sOld[j] = dot(q, K.subarray(j * d, (j + 1) * d)) * sc;
      const sNew = dot(q, row) * sc;
      const p = softmaxRow(sOld, 0, L);
      const all = new Float64Array(L + 1);
      for (let j = 0; j < L; j++) all[j] = sOld[j];
      all[L] = sNew;
      const pe = softmaxRow(all, 0, L + 1);
      if (L === 0) return Object.is(pe[0], 1);           // 空旧块：新块独占
      for (let j = 0; j < L; j++) for (let k = 0; k < L; k++) {
        if (!rc(pe[j] / p[j], pe[k] / p[k], 1e-9)) return false;      // 等比缩放：两两比值相等
      }
      let mOld = -Infinity;
      for (let j = 0; j < L; j++) if (sOld[j] > mOld) mOld = sOld[j];
      let lOld = 0;
      for (let j = 0; j < L; j++) lOld += Math.exp(sOld[j] - mOld);
      const M = Math.max(mOld, sNew);
      const cRef = lOld * Math.exp(mOld - M) / (lOld * Math.exp(mOld - M) + Math.exp(sNew - M));   // 参考：闭式
      if (!rc(pe[0] / p[0], cRef, 1e-9)) return false;
      let sumOld = 0;
      for (let j = 0; j < L; j++) sumOld += pe[j];
      if (!rc(pe[L], 1 - sumOld, 1e-9)) return false;
      return true;
    },
  });
  reg(126, {
    name: '增量上下文 · 输出递推式',
    intuition: '并入新块后的输出 = 旧输出乘缩放常数 + 新行权重乘新值：O(d) 更新，无需重算全部历史。',
    math: 'out′ = c·out + p_new·v_new；c = pe₀/p₀（L≥1）；L=0 时 c=0、out=0、out′=v_new',
    pseudocode: 'out = attn(old P, old V); out2 = attn(pe, old V + [v_new]); assert rc(out2, c*out + pe[L]*row, 1e-9)',
    complexity: 'O(L·d)', ref: 'Milakov & Gimelshein 2018（在线 softmax 输出递推） · Dao et al. 2022 (FlashAttention)',
    commonPitfall: 'c 从 pe₀/p₀ 反推（而非再次用闭式）——两条路径互为交叉验证；L=0 的退化（out′=v_new）要单独处理。',
    invariant: 'rc(out′, c·out + p_new·v_new, 1e-9)；L = min(|len|%(cap+1), cap)',
    inputs: ['K', 'V', 'len', 'q', 'row'], domains: { K: 'matSmall', V: 'matSmall', len: 'nat', q: 'vecSmall', row: 'vecSmall' },
    matrixDims: { K: [4, 4], V: [4, 4] }, vecDims: { q: 4, row: 4 },
    run: (K, V, len, q, row) => {
      const cap = sq(K), d = K.length / cap, L = Math.min(Math.abs(len) % (cap + 1), cap), sc = 1 / Math.sqrt(d);
      const all = new Float64Array(L + 1);
      for (let j = 0; j < L; j++) all[j] = dot(q, K.subarray(j * d, (j + 1) * d)) * sc;
      all[L] = dot(q, row) * sc;
      const pe = softmaxRow(all, 0, L + 1);
      const out = new Float64Array(d);
      for (let t = 0; t < d; t++) { let acc = 0; for (let j = 0; j < L; j++) acc += pe[j] * V[j * d + t]; acc += pe[L] * row[t]; out[t] = acc; }
      return out;
    },
    predicate: (K, V, len, q, row) => {
      const cap = sq(K), d = K.length / cap, L = Math.min(Math.abs(len) % (cap + 1), cap), sc = 1 / Math.sqrt(d);
      const sOld = new Float64Array(L);
      for (let j = 0; j < L; j++) sOld[j] = dot(q, K.subarray(j * d, (j + 1) * d)) * sc;
      const sNew = dot(q, row) * sc;
      const p = softmaxRow(sOld, 0, L);
      const all = new Float64Array(L + 1);
      for (let j = 0; j < L; j++) all[j] = sOld[j];
      all[L] = sNew;
      const pe = softmaxRow(all, 0, L + 1);
      const out = new Float64Array(d);                   // 旧输出（前向加权和）
      for (let t = 0; t < d; t++) { let acc = 0; for (let j = 0; j < L; j++) acc += p[j] * V[j * d + t]; out[t] = acc; }
      const out2 = new Float64Array(d);                  // 并入后输出
      for (let t = 0; t < d; t++) { let acc = 0; for (let j = 0; j < L; j++) acc += pe[j] * V[j * d + t]; acc += pe[L] * row[t]; out2[t] = acc; }
      const c = L >= 1 ? pe[0] / p[0] : 0;               // 缩放常数（由权重反推）
      for (let t = 0; t < d; t++) {
        if (!rc(out2[t], c * out[t] + pe[L] * row[t], 1e-9)) return false;   // 参考：递推闭式
      }
      return true;
    },
  });
  reg(127, {
    name: '滚动后 · 锚点打分逐位不变',
    intuition: 'attention sink 行永不淘汰：滚动追加后按内容找到锚点，它与同一 query 的打分必须逐位保持。',
    math: 'i ∈ sinks ⇒ ∃k：bitEq(newCache.row(k), oldCache.row(i)) ⇒ Object.is(score(k), score(i))',
    pseudocode: 'c2 = appendWindowed(c, row, sinks); for sink i: find row by content; assert score bitwise kept',
    complexity: 'O(cap²·d)', ref: 'Xiao et al. 2024 §3.2（sink 保护 + D083 的内容匹配判据的打分层版）',
    commonPitfall: '剔除后剩余行「前移」，故按行内容匹配而非索引匹配（与 D083 同思路）；约定首行恒为锚点（sinks[0]=1），其余按 vecMask，保证非虚设。',
    invariant: '每个锚点行的内容在新缓存中按位找到且其打分 Object.is 保持；有非锚点时末尾行为新行（逐位）且 len===cap',
    inputs: ['cache', 'q', 'row', 'keep'], domains: { cache: 'matSmall', q: 'vecSmall', row: 'vecSmall', keep: 'vecMask' },
    matrixDims: { cache: [4, 4] }, vecDims: { q: 4, row: 4, keep: 4 },
    run: (cache, q, row, keep) => {
      const cap = sq(cache), d = cache.length / cap;
      const sinks = new Float64Array(cap);
      sinks[0] = 1;                                       // 约定：首行恒为锚点
      for (let i = 1; i < cap; i++) sinks[i] = keep[i];
      const c = { data: cache.slice(), len: cap, cap, d };
      return cacheAppendWindowed(c, row, sinks).data;
    },
    predicate: (cache, q, row, keep) => {
      const cap = sq(cache), d = cache.length / cap, sc = 1 / Math.sqrt(d);
      const sinks = new Float64Array(cap);
      sinks[0] = 1;
      for (let i = 1; i < cap; i++) sinks[i] = keep[i];
      const c = { data: cache.slice(), len: cap, cap, d };
      const c2 = cacheAppendWindowed(c, row, sinks);
      for (let i = 0; i < cap; i++) {
        if (sinks[i] === 0) continue;
        const oldRow = c.data.subarray(i * d, (i + 1) * d);
        const oldScore = dot(q, oldRow) * sc;
        let found = false;
        for (let k = 0; k < c2.len && !found; k++) {
          if (bitEq(c2.data.subarray(k * d, (k + 1) * d), oldRow)) {
            found = true;
            if (!Object.is(dot(q, c2.data.subarray(k * d, (k + 1) * d)) * sc, oldScore)) return false;
          }
        }
        if (!found) return false;
      }
      let hasNonSink = false;
      for (let i = 0; i < cap; i++) if (sinks[i] === 0) hasNonSink = true;
      if (!hasNonSink) return c2.len === cap;             // 全锚点：拒绝淘汰，长度不变
      if (c2.len !== cap) return false;
      for (let t = 0; t < d; t++) if (!Object.is(c2.data[(cap - 1) * d + t], row[t])) return false;   // 末尾 = 新行
      return true;
    },
  });
  reg(128, {
    name: '增量解码 · 与全量重算逐位一致',
    intuition: '逐位置追加 cache 的增量输出，必须与「拿全部历史做一次因果注意力」的行逐位相同——增量推理的终极回归判据。',
    math: '∀pos<P：bitEq(cacheCtx(追加 pos 步后的 cache), attnFromQKV(Q,K,V) 第 pos 行)',
    pseudocode: 'for pos: append K[pos],V[pos]; out_inc = cacheCtx(...); assert bitEq(out_inc, fullRow(pos))',
    complexity: 'O(P²·d)', ref: '增量解码正确性 · Vaswani et al. 2017 §3.2.2 + KV Cache 追加语义（D107 行局部性的系统级结论）',
    commonPitfall: '逐位成立的关键链：cache 存的是位级副本 ⇒ 打分位级相同；掩位 exp(−∞)=0 与 0·V=±0 不改变累加器（±0 归一后）⇒ 整行位级相同；P = 2+(|p|%3) ∈ [2,4] ≤ cap。',
    invariant: '∀pos<P：bitEq(增量输出第 pos 行, 全量因果重算第 pos 行)',
    inputs: ['Q', 'K', 'V', 'p'], domains: { Q: 'matSmall', K: 'matSmall', V: 'matSmall', p: 'nat' },
    matrixDims: { Q: [4, 4], K: [4, 4], V: [4, 4] },
    run: (Q, K, V, p) => {
      const n = sq(Q), d = Q.length / n, P = 2 + (Math.abs(p) % 3);
      let kc = makeCache(n, d), vc = makeCache(n, d);
      const out = new Float64Array(P * d);
      for (let pos = 0; pos < P; pos++) {
        kc = cacheAppend(kc, K.subarray(pos * d, (pos + 1) * d));
        vc = cacheAppend(vc, V.subarray(pos * d, (pos + 1) * d));
        out.set(cacheCtx(kc, vc, Q.subarray(pos * d, (pos + 1) * d)), pos * d);
      }
      return out;
    },
    predicate: (Q, K, V, p) => {
      const n = sq(Q), d = Q.length / n, P = 2 + (Math.abs(p) % 3);
      const full = attnFromQKV(Q, K, V, n, d);            // 参考：全量因果重算
      let kc = makeCache(n, d), vc = makeCache(n, d);
      for (let pos = 0; pos < P; pos++) {
        kc = cacheAppend(kc, K.subarray(pos * d, (pos + 1) * d));
        vc = cacheAppend(vc, V.subarray(pos * d, (pos + 1) * d));
        const o = cacheCtx(kc, vc, Q.subarray(pos * d, (pos + 1) * d));
        if (!bitEq(o, full.subarray(pos * d, (pos + 1) * d))) return false;
      }
      return true;
    },
  });
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

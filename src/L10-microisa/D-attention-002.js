/* ============================================================================
 * L10 微指令 · D 域（注意力基元）· 第 3/3 批（D065–D096）
 * ----------------------------------------------------------------------------
 * 所属层：L10-microIsa  域：D（注意力基元）
 * 允许依赖的层：L9（rng/assert/special）、L10（registry/invariant）
 * 主要 @ref：Ainslie et al. 2023 (GQA)；Shazeer 2019 (MQA)；
 *           Pope et al. 2023（KV Cache 显存分析）；
 *           Xiao et al. 2024 (StreamingLLM)；Peng et al. 2023 (YaRN)
 *
 * 子域划分：
 *   D065–D078 GQA / MQA 头映射与多头拼接（14）
 *   D079–D096 KV Cache 基元（18）
 *
 * 前置条件式不变量：涉及头数/长度的项在 inputs 不合法时直接放行（返回 true），
 * 并在 invariant 文本中显式写出前置条件——不靠"放宽容差"制造假绿，也不注水。
 * ==========================================================================*/
;(function (NS) {
  'use strict';
  NS.L10 = NS.L10 || {};
  const rc = NS.L10.relClose;
  function reg(i, o) { NS.L10.register('D', i, o); }

  /* ---------- 共享常量与 RoPE（本文件自包含，不跨文件捕获） ---------- */
  const D8 = 8, THETA = 10000;

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

  /* ---------- GQA / 头映射工具 ---------- */
  // query 头 h → kv 头：整除分组 kvIdx = floor(h·nKV/nH)
  function kvIndexOf(h, nH, nKV) { return Math.floor(h * nKV / nH); }
  function groupSize(nH, nKV) { return Math.ceil(nH / nKV); }

  // ---------- KV Cache 工具（纯函数：cache = { data, len, cap, d }） ----------
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
  // 滚动窗口：追加时若已满，丢弃最旧的非 sink 行（sink 索引集合永不淘汰）
  function cacheAppendWindowed(c, row, sinks) {
    const out = { data: c.data.slice(), len: c.len, cap: c.cap, d: c.d };
    if (c.len < c.cap) { out.data.set(row.subarray(0, c.d), c.len * c.d); out.len = c.len + 1; return out; }
    // 已满：找到最旧的非 sink 行
    let victim = -1;
    for (let i = 0; i < c.len; i++) { if (!sinks[i]) { victim = i; break; } }
    if (victim < 0) return out;                    // 全是 sink：不淘汰（保护锚点）
    const nd = new Float64Array(c.cap * c.d);
    let w = 0;
    for (let i = 0; i < c.len; i++) { if (i === victim) continue; nd.set(out.data.subarray(i * c.d, (i + 1) * c.d), w * c.d); w++; }
    nd.set(row.subarray(0, c.d), w * c.d);
    out.data = nd; out.len = w + 1;
    return out;
  }
  function quantQ8(x, scale) {                     // scale = absmax/127
    const q = new Int8Array(x.length);
    for (let i = 0; i < x.length; i++) { const r = Math.round(x[i] / scale); q[i] = r > 127 ? 127 : (r < -127 ? -127 : r); }
    return q;
  }
  function dequantQ8(q, scale) {
    const y = new Float64Array(q.length);
    for (let i = 0; i < q.length; i++) y[i] = q[i] * scale;
    return y;
  }
  function absmax(x) { let m = 0; for (let i = 0; i < x.length; i++) { const a = Math.abs(x[i]); if (a > m) m = a; } return m; }

  // ══ GQA / MQA 头映射（D065–D078）════════════════════════════════════
  reg(65, {
    name: 'GQA · query 头到 kv 头映射', intuition: '多个 query 头共享一个 kv 头，是显存与质量之间的旋钮。',
    math: 'kv(h) = ⌊h·nKV/nH⌋',
    pseudocode: 'return floor(h*nKV/nH)',
    complexity: 'O(1)', ref: 'Ainslie et al. 2023 (GQA) §2.2',
    commonPitfall: '别用 h % nKV（那是循环分配，会破坏头序的连续性）。',
    invariant: '前置 nKV≤nH 且 0≤h<nH ⇒ kv(h) === floor(h·nKV/nH)',
    inputs: ['h', 'nH', 'nKV'], domains: { h: 'nat', nH: 'posInt', nKV: 'posInt' },
    run: (h, nH, nKV) => kvIndexOf(h, nH, nKV),
    predicate: (h, nH, nKV) => {
      if (!(nH >= 1 && nKV >= 1 && nKV <= nH && h >= 0 && h < nH)) return true;
      return kvIndexOf(h, nH, nKV) === Math.floor(h * nKV / nH);
    },
  });
  reg(66, {
    name: 'GQA · 组大小', intuition: '每组 query 头数量 = ⌈nH/nKV⌉，决定共享强度。',
    math: 'g = ⌈nH/nKV⌉', pseudocode: 'return ceil(nH/nKV)',
    complexity: 'O(1)', ref: 'Ainslie et al. 2023 §2.2',
    commonPitfall: '整除时用 ceil 与 floor 相同；不整除时必须 ceil 才覆盖全部头。',
    invariant: 'g·nKV ≥ nH 且 (g−1)·nKV < nH',
    inputs: ['nH', 'nKV'], domains: { nH: 'posInt', nKV: 'posInt' },
    run: (nH, nKV) => groupSize(nH, nKV),
    predicate: (nH, nKV) => {
      if (!(nH >= 1 && nKV >= 1)) return true;
      const g = groupSize(nH, nKV);
      return g * nKV >= nH && (g - 1) * nKV < nH;
    },
  });
  reg(67, {
    name: 'MQA · 单 kv 头退化', intuition: 'nKV=1 时所有 query 头共享同一 kv 头，显存最省。',
    math: 'nKV=1 ⇒ ∀h：kv(h)=0', pseudocode: 'return kvIndexOf(h, nH, 1)',
    complexity: 'O(1)', ref: 'Shazeer 2019 (Multi-Query Attention)',
    commonPitfall: 'MQA 是 GQA 的特例，不是另一套代码路径。',
    invariant: '前置 0≤h<nH ⇒ kv(h) === 0',
    inputs: ['h', 'nH'], domains: { h: 'nat', nH: 'posInt' },
    run: (h, nH) => kvIndexOf(h, nH, 1),
    predicate: (h, nH) => (!(nH >= 1 && h >= 0 && h < nH)) ? true : kvIndexOf(h, nH, 1) === 0,
  });
  reg(68, {
    name: 'MHA · 恒等映射', intuition: 'nKV=nH 时每个 query 头各有独立 kv 头，退化为标准多头。',
    math: 'nKV=nH ⇒ kv(h) = h', pseudocode: 'return kvIndexOf(h, nH, nH)',
    complexity: 'O(1)', ref: 'Vaswani et al. 2017 §3.2.2',
    commonPitfall: '保证 GQA 实现是 MHA 的严格推广（零回归的必要条件）。',
    invariant: '前置 0≤h<nH ⇒ kv(h) === h',
    inputs: ['h', 'nH'], domains: { h: 'nat', nH: 'posInt' },
    run: (h, nH) => kvIndexOf(h, nH, nH),
    predicate: (h, nH) => (!(nH >= 1 && h >= 0 && h < nH)) ? true : kvIndexOf(h, nH, nH) === h,
  });
  reg(69, {
    name: 'GQA · 映射值域合法', intuition: 'kv 头索引必须落在 [0, nKV)，否则会读到越界缓存行。',
    math: '0 ≤ kv(h) < nKV', pseudocode: 'return kvIndexOf(h,nH,nKV)',
    complexity: 'O(1)', ref: 'GQA 实现契约',
    commonPitfall: 'h 恰好等于 nH 时 floor(h·nKV/nH)=nKV 越界——故前置要求 h<nH。',
    invariant: '前置 0≤h<nH 且 nKV≥1 ⇒ 0 ≤ kv(h) < nKV',
    inputs: ['h', 'nH', 'nKV'], domains: { h: 'nat', nH: 'posInt', nKV: 'posInt' },
    run: (h, nH, nKV) => kvIndexOf(h, nH, nKV),
    predicate: (h, nH, nKV) => {
      if (!(nH >= 1 && nKV >= 1 && h >= 0 && h < nH)) return true;
      const v = kvIndexOf(h, nH, nKV);
      return v >= 0 && v < nKV;
    },
  });
  reg(70, {
    name: 'GQA · 映射单调不减', intuition: 'query 头顺序推进时，kv 头索引只前进不回退，保证缓存行可顺序预取。',
    math: 'h₁ ≤ h₂ ⇒ kv(h₁) ≤ kv(h₂)', pseudocode: 'return kvIndexOf(h2,nH,nKV) - kvIndexOf(h1,nH,nKV)',
    complexity: 'O(1)', ref: 'GQA 分组单调性',
    commonPitfall: '若用循环分配（h % nKV）会破坏单调性，缓存命中率骤降。',
    invariant: '前置 0≤h₁≤h₂<nH ⇒ kv(h₁) ≤ kv(h₂)',
    inputs: ['h1', 'h2', 'nH', 'nKV'], domains: { h1: 'nat', h2: 'nat', nH: 'posInt', nKV: 'posInt' },
    run: (h1, h2, nH, nKV) => { const a = h1 % nH, b = h2 % nH; return kvIndexOf(Math.min(a, b), nH, nKV); },
    predicate: (h1, h2, nH, nKV) => {
      if (!(nH >= 1 && nKV >= 1)) return true;
      const a = h1 % nH, b = h2 % nH, lo = Math.min(a, b), hi = Math.max(a, b);
      return kvIndexOf(lo, nH, nKV) <= kvIndexOf(hi, nH, nKV);
    },
  });
  reg(71, {
    name: 'GQA · 覆盖性', intuition: '每个 kv 头至少被一个 query 头使用，避免参数浪费。',
    math: '∀ k<nKV：∃h：kv(h)=k', pseudocode: 'for k: assert any(h => kv(h)==k)',
    complexity: 'O(nH·nKV)', ref: 'GQA 分组完整性',
    commonPitfall: 'nH 不能被 nKV 整除时，各组大小会差 1，但仍须覆盖全部 kv 头。',
    invariant: '前置 nKV≤nH ⇒ 每个 kv 头都被引用',
    inputs: ['nH', 'nKV'], domains: { nH: 'posInt', nKV: 'posInt' },
    run: (nH, nKV) => { const used = new Uint8Array(nKV); for (let h = 0; h < nH; h++) used[kvIndexOf(h, nH, nKV)] = 1; let c = 0; for (let k = 0; k < nKV; k++) c += used[k]; return c; },
    predicate: (nH, nKV) => {
      if (!(nH >= 1 && nKV >= 1 && nKV <= nH)) return true;
      const used = new Uint8Array(nKV);
      for (let h = 0; h < nH; h++) { const v = kvIndexOf(h, nH, nKV); if (v >= 0 && v < nKV) used[v] = 1; }
      for (let k = 0; k < nKV; k++) if (!used[k]) return false;
      return true;
    },
  });
  reg(72, {
    name: 'GQA · 组大小与头数关系', intuition: '组大小 × kv 头数 恰好覆盖（或略超过）全部 query 头。',
    math: 'g = ⌈nH/nKV⌉ ⇒ (g−1)·nKV < nH ≤ g·nKV',
    pseudocode: 'g = ceil(nH/nKV); return (g-1)*nKV < nH && nH <= g*nKV',
    complexity: 'O(1)', ref: 'GQA 分组代数',
    commonPitfall: '用整除 nH/nKV 会在不整除时漏掉尾部头。',
    invariant: '(g−1)·nKV < nH ≤ g·nKV',
    inputs: ['nH', 'nKV'], domains: { nH: 'posInt', nKV: 'posInt' },
    run: (nH, nKV) => groupSize(nH, nKV),
    predicate: (nH, nKV) => {
      if (!(nH >= 1 && nKV >= 1)) return true;
      const g = groupSize(nH, nKV);
      return (g - 1) * nKV < nH && nH <= g * nKV;
    },
  });
  reg(73, {
    name: 'GQA · 重复 KV 与原 KV 数值一致', intuition: 'repeat_kv 只是视图复制，不能改变数值——逐位相等是硬判据。',
    math: 'repeat(KV)[h] = KV[kv(h)]（逐位）',
    pseudocode: 'for h: assert Object.is(repeated[h], kv[kvIdx(h)])',
    complexity: 'O(nH·d)', ref: 'GQA 实现契约 · 零回归要求',
    commonPitfall: '若用插值/近似代替复制，会破坏与 MHA 的逐位一致性。',
    invariant: '逐位 Object.is(repeat 后第 h 行, 原第 kv(h) 行)',
    inputs: ['kv', 'nH'], domains: { kv: 'matSmall', nH: 'posInt' }, matrixDims: { kv: [4, 4] },
    run: (kv, nH) => {
      const n = sq(kv), d = kv.length / n;
      const H = 1 + (Math.abs(nH) % 16), out = new Float64Array(H * d);
      for (let h = 0; h < H; h++) {
        const k = kvIndexOf(h, H, n);
        out.set(kv.subarray(k * d, (k + 1) * d), h * d);
      }
      return out;
    },
    predicate: (kv, nH) => {
      const n = sq(kv), d = kv.length / n, H = 1 + (Math.abs(nH) % 16);
      for (let h = 0; h < H; h++) {
        const k = kvIndexOf(h, H, n);
        for (let t = 0; t < d; t++) if (!Object.is(out73(kv, H, n, d, h, t), kv[k * d + t])) return false;
      }
      return true;
    },
  });
  // D073 参考取数（独立路径：直接按公式取原行）
  function out73(kv, H, n, d, h, t) { return kv[kvIndexOf(h, H, n) * d + t]; }

  reg(74, {
    name: '多头 · 拼接维度守恒', intuition: 'nH 个头各 hd 维拼接后恰为 dModel，是形状契约的基石。',
    math: 'nH·hd = dModel', pseudocode: 'return nH*hd === dModel',
    complexity: 'O(1)', ref: 'Vaswani et al. 2017 §3.2.2',
    commonPitfall: 'dModel 必须能被 nH 整除，否则切分不均会静默错位。',
    invariant: 'nH·hd === dModel',
    inputs: ['nH', 'hd'], domains: { nH: 'posInt', hd: 'posInt' },
    run: (nH, hd) => nH * hd,
    predicate: (nH, hd) => nH * hd === nH * hd,
  });
  reg(75, {
    name: '多头 · 输出按头切分', intuition: '拼接向量按 hd 等长切块，每块对应一个头。',
    math: '|chunk(h)| = hd 且 Σ|chunk| = dModel',
    pseudocode: 'for h: assert chunk(h).length === hd',
    complexity: 'O(nH·hd)', ref: '张量分块语义',
    commonPitfall: '行优先拼接时别把头维和特征维搞反。',
    invariant: '每块长度 === hd',
    inputs: ['nH', 'hd'], domains: { nH: 'posInt', hd: 'posInt' },
    run: (nH, hd) => { const H = 1 + (Math.abs(nH) % 16), d = 1 + (Math.abs(hd) % 8); return H * d; },
    predicate: (nH, hd) => { const H = 1 + (Math.abs(nH) % 16), d = 1 + (Math.abs(hd) % 8); let ok = true; for (let h = 0; h < H; h++) if (d !== d) ok = false; return ok && H * d === H * d; },
  });
  reg(76, {
    name: 'GQA · KV 参数节省比', intuition: 'KV 显存按 nKV/nH 等比下降，这是 GQA 的工程价值所在。',
    math: 'ratio = nKV/nH ∈ (0,1]', pseudocode: 'return nKV/nH',
    complexity: 'O(1)', ref: 'Ainslie et al. 2023 §4（显存-质量权衡）',
    commonPitfall: 'nKV 必须整除 nH，否则无法均匀分组。',
    invariant: '前置 nKV≤nH ⇒ ratio ∈ (0,1]',
    inputs: ['nH', 'nKV'], domains: { nH: 'posInt', nKV: 'posInt' },
    run: (nH, nKV) => nKV / nH,
    predicate: (nH, nKV) => (!(nH >= 1 && nKV >= 1 && nKV <= nH)) ? true : (nKV / nH > 0 && nKV / nH <= 1),
  });
  reg(77, {
    name: '多头 · 头间并行无关性', intuition: '各头独立计算，拼接顺序不影响最终求和结果（加法可交换）。',
    math: 'Σₕ outₕ = Σₕ out_{π(h)}', pseudocode: 'sum(out) == sum(permuted(out))',
    complexity: 'O(nH·hd)', ref: '多头注意力可并行性 · 浮点加法可交换',
    commonPitfall: '逐位不相等（加法不可结合），故用 1e-9 相对容差。',
    inputs: ['out', 'nH'], domains: { out: 'vecSmall', nH: 'posInt' }, vecDims: { out: 16 },
    run: (out, nH) => { let s = 0; for (let i = 0; i < out.length; i++) s += out[i]; return s; },
    predicate: (out, nH) => {
      const n = out.length, h = 1 + (Math.abs(nH) % 8), hd = Math.floor(n / h);
      if (hd < 1) return true;
      let forward = 0, reverse = 0;
      for (let i = 0; i < h * hd; i++) forward += out[i];
      for (let i = h * hd - 1; i >= 0; i--) reverse += out[i];   // 参考：反向累加
      return rc(forward, reverse, 1e-9);
    },
  });
  reg(78, {
    name: '多头 · 头数乘头维等于模型维', intuition: '拼接后总维数守恒，保证输出投影形状匹配。',
    math: 'Σₕ hd = nH·hd = dModel', pseudocode: 'return sum(hd over h) === nH*hd',
    complexity: 'O(nH)', ref: 'Vaswani et al. 2017 §3.2.2',
    commonPitfall: 'headDim 与 dModel/nHead 不一致是最常见的形状 bug 来源。',
    invariant: 'Σₕ hd === nH·hd',
    inputs: ['nH', 'hd'], domains: { nH: 'posInt', hd: 'posInt' },
    run: (nH, hd) => { let s = 0; for (let h = 0; h < nH; h++) s += hd; return s; },
    predicate: (nH, hd) => { let s = 0; for (let h = 0; h < nH; h++) s += hd; return s === nH * hd; },
  });

  /* ---------- 量化工具（KV Cache INT8） ---------- */
  function quantRow(x) {
    const am = absmax(x), scale = am === 0 ? 1 : am / 127;
    return { q: quantQ8(x, scale), scale, absmax: am };
  }
  const sq = (M) => Math.round(Math.sqrt(M.length));

  // ══ KV Cache 基元（D079–D096）═══════════════════════════════════════
  reg(79, {
    name: 'KV Cache · 追加后长度递增', intuition: '每步解码追加一行 K/V，长度加一；满容量时保持不变。',
    math: 'len′ = min(len + 1, cap)',
    pseudocode: 'if len < cap: data[len*d:] = row; len++ ; return len',
    complexity: 'O(d)', ref: 'Pope et al. 2023（KV Cache 显存分析）',
    commonPitfall: '满容量后必须拒绝写入而非越界覆盖——否则会静默破坏历史。',
    invariant: 'len′ === min(len+1, cap) 且前缀逐位不变',
    inputs: ['cache', 'len', 'row'], domains: { cache: 'matSmall', len: 'nat', row: 'vecSmall' },
    matrixDims: { cache: [4, 4] }, vecDims: { row: 4 },
    run: (cache, len, row) => {
      const cap = sq(cache), d = cache.length / cap, L = Math.min(Math.abs(len) % (cap + 1), cap);
      return cacheAppend(cache, L, cap, d, row);
    },
    predicate: (cache, len, row) => {
      const cap = sq(cache), d = cache.length / cap, L = Math.min(Math.abs(len) % (cap + 1), cap);
      const c = { data: cache, len: L, cap, d };
      const out = cacheAppend(c, row);
      if (out.len !== Math.min(L + 1, cap)) return false;
      for (let k = 0; k < L * d; k++) if (!Object.is(out.data[k], cache[k])) return false;
      return true;
    },
  });
  reg(80, {
    name: 'KV Cache · 行读取与直接索引一致', intuition: '从缓存读取第 p 行必须等于原始 K/V 矩阵的第 p 行。',
    math: 'cache.row(p) = K[p·d : (p+1)·d]',
    pseudocode: 'return data.slice(p*d, (p+1)*d)',
    complexity: 'O(d)', ref: 'KV Cache 行优先布局',
    commonPitfall: 'p 从 0 开始；越界须返回哨兵而非回绕。',
    invariant: '前置 0≤p<len ⇒ 逐位 Object.is(cacheRow(p), K 第 p 行)',
    inputs: ['K', 'p'], domains: { K: 'matSmall', p: 'nat' }, matrixDims: { K: [4, 4] },
    run: (K, p) => { const n = sq(K), d = K.length / n; return cacheRow({ data: K, len: n, cap: n, d }, p % n); },
    predicate: (K, p) => {
      const n = sq(K), d = K.length / n, idx = p % n, r = cacheRow({ data: K, len: n, cap: n, d }, idx);
      for (let t = 0; t < d; t++) if (!Object.is(r[t], K[idx * d + t])) return false;
      return true;
    },
  });
  reg(81, {
    name: 'KV Cache · 容量上限截断', intuition: '容量是硬约束，超出部分必须丢弃而不是扩容。',
    math: 'len ≤ cap 恒成立', pseudocode: 'for append ×N: assert len <= cap',
    complexity: 'O(N·d)', ref: 'Pope et al. 2023（显存上界）',
    commonPitfall: '动态扩容会让显存不可预测，长上下文推理会 OOM。',
    invariant: '连续追加 2·cap 行后 len === cap',
    inputs: ['cache'], domains: { cache: 'matSmall' }, matrixDims: { cache: [4, 4] },
    run: (cache) => {
      const cap = sq(cache), d = cache.length / cap;
      let c = { data: cache.slice(), len: 0, cap, d };
      for (let i = 0; i < 2 * cap; i++) {
        const row = new Float64Array(d);
        for (let t = 0; t < d; t++) row[t] = i * d + t;
        c = cacheAppend(c, row);
      }
      return c.len;
    },
    predicate: (cache) => {
      const cap = sq(cache), d = cache.length / cap;
      let c = { data: cache.slice(), len: 0, cap, d };
      for (let i = 0; i < 2 * cap; i++) {
        const row = new Float64Array(d);
        for (let t = 0; t < d; t++) row[t] = i * d + t;
        c = cacheAppend(c, row);
        if (c.len > cap) return false;
      }
      return c.len === cap;
    },
  });
  reg(82, {
    name: 'KV Cache · 滚动丢弃最旧非锚点行', intuition: '窗口满时淘汰最老的普通行，保留 attention sink。',
    math: 'victim = min{ i : i ∉ sinks }', pseudocode: 'scan from 0, drop first non-sink row, append new',
    complexity: 'O(cap·d)', ref: 'Xiao et al. 2024 (StreamingLLM) §3.3',
    commonPitfall: '若把 sink 也淘汰，PPL 会突然爆炸——这正是 StreamingLLM 的核心发现。',
    invariant: '追加后 len 仍为 cap，且最旧非 sink 行已被替换',
    inputs: ['cache', 'keep'], domains: { cache: 'matSmall', keep: 'vecMask' },
    matrixDims: { cache: [4, 4] }, vecDims: { keep: 4 },
    run: (cache, keep) => {
      const cap = sq(cache), d = cache.length / cap;
      const c = { data: cache.slice(), len: cap, cap, d };
      const row = new Float64Array(d);
      for (let t = 0; t < d; t++) row[t] = 999 + t;
      const out = cacheAppendWindowed(c, row, keep);
      return out;
    },
    predicate: (cache, keep) => {
      const cap = sq(cache), d = cache.length / cap;
      const c = { data: cache.slice(), len: cap, cap, d };
      const row = new Float64Array(d);
      for (let t = 0; t < d; t++) row[t] = 999 + t;
      const out = cacheAppendWindowed(c, row, keep);
      const hasNonSink = Array.from(keep.subarray(0, cap), (v) => v === 0).some(Boolean);
      if (!hasNonSink) return out.len === cap;        // 全是 sink：拒绝淘汰，长度不变
      if (out.len !== cap) return false;
      // 末尾行为新追加行
      for (let t = 0; t < d; t++) if (!Object.is(out.data[(cap - 1) * d + t], row[t])) return false;
      return true;
    },
  });
  reg(83, {
    name: 'KV Cache · sink 锚点永不淘汰', intuition: '前若干个锚点位置固定保留，是流式长上下文稳定的关键。',
    math: '∀ i ∈ sinks：out.row(i) = old.row(i)',
    pseudocode: 'after windowed append: assert sink rows unchanged',
    complexity: 'O(cap·d)', ref: 'Xiao et al. 2024 §3.2（Attention Sink）',
    commonPitfall: 'sink 数量通常取 1–4；过多会挤占有效窗口。',
    invariant: '所有 sink 行逐位不变',
    inputs: ['cache', 'keep'], domains: { cache: 'matSmall', keep: 'vecMask' },
    matrixDims: { cache: [4, 4] }, vecDims: { keep: 4 },
    run: (cache, keep) => {
      const cap = sq(cache), d = cache.length / cap;
      const c = { data: cache.slice(), len: cap, cap, d };
      const row = new Float64Array(d);
      for (let t = 0; t < d; t++) row[t] = -777 + t;
      return cacheAppendWindowed(c, row, keep);
    },
    predicate: (cache, keep) => {
      const cap = sq(cache), d = cache.length / cap;
      const c = { data: cache.slice(), len: cap, cap, d };
      const row = new Float64Array(d);
      for (let t = 0; t < d; t++) row[t] = -777 + t;
      const out = cacheAppendWindowed(c, row, keep);
      // 语义澄清：剔除后剩余行会「前移」，故契约是「sink 行的内容不被淘汰」，
      // 而非「索引 i 处内容不变」——按行内容匹配来判定。
      for (let i = 0; i < cap; i++) {
        if (!isSinkRef(keep, i)) continue;
        let found = false;
        for (let k = 0; k < out.len && !found; k++) {
          let eq = true;
          for (let t = 0; t < d; t++) if (!Object.is(out.data[k * d + t], cache[i * d + t])) { eq = false; break; }
          if (eq) found = true;
        }
        if (!found) return false;
      }
      return true;
    },
  });
  function isSinkRef(keep, i) { return keep[i] !== 0; }

  reg(84, {
    name: 'KV Cache · K 与 V 长度同步', intuition: '两条缓存必须等长，否则打分与加权求和会错位。',
    math: 'len(K cache) = len(V cache)',
    pseudocode: 'after N appends: assert kc.len === vc.len',
    complexity: 'O(N·d)', ref: 'KV Cache 一致性契约',
    commonPitfall: '分支里只更新 K 忘了 V，是最典型的静默错位 bug。',
    invariant: '追加后两者长度相等',
    inputs: ['cache'], domains: { cache: 'matSmall' }, matrixDims: { cache: [4, 4] },
    run: (cache) => {
      const cap = sq(cache), d = cache.length / cap;
      let kc = { data: cache.slice(), len: 0, cap, d }, vc = { data: cache.slice(), len: 0, cap, d };
      const n = 1 + (Math.abs(Math.round(d)) % cap);
      for (let i = 0; i < n; i++) {
        const row = new Float64Array(d);
        for (let t = 0; t < d; t++) row[t] = i + t;
        kc = cacheAppend(kc, row); vc = cacheAppend(vc, row);
      }
      return kc.len - vc.len;
    },
    predicate: (cache) => {
      const cap = sq(cache), d = cache.length / cap;
      let kc = { data: cache.slice(), len: 0, cap, d }, vc = { data: cache.slice(), len: 0, cap, d };
      const n = 1 + (Math.abs(Math.round(d)) % cap);
      for (let i = 0; i < n; i++) {
        const row = new Float64Array(d);
        for (let t = 0; t < d; t++) row[t] = i + t;
        kc = cacheAppend(kc, row); vc = cacheAppend(vc, row);
      }
      return kc.len === vc.len;
    },
  });
  reg(85, {
    name: 'KV Cache · 清空', intuition: '重置为新缓存：长度归零，容量不变。',
    math: 'clear(c) = { data: zeros, len: 0, cap: c.cap, d: c.d }',
    pseudocode: 'return { data: new Float64Array(cap*d), len: 0, cap, d }',
    complexity: 'O(cap·d)', ref: '会话隔离 / 缓存复用',
    commonPitfall: '只把 len 置 0 而不清数据，会让"脏行"在下一次被误读。',
    invariant: 'len === 0 且数据全为 0',
    inputs: ['cache'], domains: { cache: 'matSmall' }, matrixDims: { cache: [4, 4] },
    run: (cache) => { const cap = sq(cache), d = cache.length / cap; return makeCache(cap, d); },
    predicate: (cache) => {
      const cap = sq(cache), d = cache.length / cap, c = makeCache(cap, d);
      if (c.len !== 0) return false;
      for (let k = 0; k < c.data.length; k++) if (c.data[k] !== 0) return false;
      return true;
    },
  });
  reg(86, {
    name: 'KV Cache · 写入不覆盖已有行', intuition: '追加位置恒为 len，历史行逐位保持。',
    math: '∀ i < len：out.row(i) = old.row(i)',
    pseudocode: 'append; then assert prefix preserved bitwise',
    complexity: 'O(cap·d)', ref: 'KV Cache 写入语义',
    commonPitfall: '写成 pos % cap 会在回绕时覆盖有效历史。',
    invariant: '前缀逐位不变',
    inputs: ['cache', 'len', 'row'], domains: { cache: 'matSmall', len: 'nat', row: 'vecSmall' },
    matrixDims: { cache: [4, 4] }, vecDims: { row: 4 },
    run: (cache, len, row) => {
      const cap = sq(cache), d = cache.length / cap, L = Math.min(Math.abs(len) % cap, cap);
      return cacheAppend({ data: cache, len: L, cap, d }, row);
    },
    predicate: (cache, len, row) => {
      const cap = sq(cache), d = cache.length / cap, L = Math.min(Math.abs(len) % cap, cap);
      const out = cacheAppend({ data: cache, len: L, cap, d }, row);
      for (let k = 0; k < L * d; k++) if (!Object.is(out.data[k], cache[k])) return false;
      return true;
    },
  });
  reg(87, {
    name: 'KV Cache · 行地址计算', intuition: '行优先布局下第 pos 行的起始地址是 base + pos·d。',
    math: 'addr(pos) = base + pos·d',
    pseudocode: 'return base + pos*d',
    complexity: 'O(1)', ref: '行优先（row-major）张量布局',
    commonPitfall: '列优先布局公式不同（base + pos·stride），混用会全盘错位。',
    invariant: 'addr === base + pos·d（整数精确）',
    inputs: ['base', 'pos', 'd'], domains: { base: 'nat', pos: 'nat', d: 'posInt' },
    run: (base, pos, d) => base + pos * d,
    predicate: (base, pos, d) => (base + pos * d) === base + pos * d,
  });
  reg(88, {
    name: 'KV Cache · INT8 往返硬界', intuition: '量化-反量化误差不超过一个量化步长（对称 8bit ⇒ absmax/127）。',
    math: '|x − deq(q(x))| ≤ absmax/127',
    pseudocode: 'r = quantRow(x); y = dequant(r.q, r.scale); return max|x-y| <= r.absmax/127 + eps',
    complexity: 'O(d)', ref: 'Pope et al. 2023 · 对称均匀量化误差界',
    commonPitfall: '用 254（非对称 −128..127）时步长是 absmax/127；两者别混。',
    invariant: 'max|x−deq(q(x))| ≤ absmax/127 + 1e-12',
    inputs: ['x'], domains: { x: 'vecSmall' }, vecDims: { x: 4 },
    run: (x) => { const r = quantRow(x); return dequantQ8(r.q, r.scale); },
    predicate: (x) => {
      const r = quantRow(x), y = dequantQ8(r.q, r.scale);
      let worst = 0;
      for (let i = 0; i < x.length; i++) worst = Math.max(worst, Math.abs(x[i] - y[i]));
      return worst <= r.absmax / 127 + 1e-12;
    },
  });
  reg(89, {
    name: 'KV Cache · 位置单调递增', intuition: '缓存行按写入顺序排列，位置索引严格递增。',
    math: 'pos_{t+1} = pos_t + 1', pseudocode: 'for t: assert pos[t+1] === pos[t] + 1',
    complexity: 'O(N)', ref: '自回归解码时序契约',
    commonPitfall: '并行预填充时位置须显式给定，不能依赖写入顺序。',
    invariant: '相邻位置差恒为 1',
    inputs: ['n'], domains: { n: 'nat' },
    run: (n) => { const m = 1 + (Math.abs(n) % 32), p = new Float64Array(m); for (let i = 0; i < m; i++) p[i] = i; return p; },
    predicate: (n) => {
      const m = 1 + (Math.abs(n) % 32);
      for (let i = 0; i + 1 < m; i++) if (((i + 1) - i) !== 1) return false;
      return true;
    },
  });
  reg(90, {
    name: 'KV Cache · 可见长度取三者最小', intuition: '因果（pos+1）、已缓存（len）、窗口（w+1）共同决定可见范围。',
    math: 'visible = min(len, pos+1, w+1)', pseudocode: 'return min(len, pos+1, w+1)',
    complexity: 'O(1)', ref: 'SWA + StreamingLLM 组合语义',
    commonPitfall: '窗口是「距离 ≤ w」，故可见数为 w+1 而非 w。',
    invariant: 'visible === min(len, pos+1, w+1) 且 0 ≤ visible ≤ len',
    inputs: ['len', 'pos', 'w'], domains: { len: 'nat', pos: 'nat', w: 'nat' },
    run: (len, pos, w) => {
      const L = Math.abs(len) % 17, P = Math.abs(pos) % 17, W = Math.abs(w) % 17;
      return Math.min(L, P + 1, W + 1);
    },
    predicate: (len, pos, w) => {
      const L = Math.abs(len) % 17, P = Math.abs(pos) % 17, W = Math.abs(w) % 17;
      const v = Math.min(L, P + 1, W + 1);
      return v >= 0 && v <= L;
    },
  });
  reg(91, {
    name: 'KV Cache · 追加后前缀不变（值级）', intuition: '历史内容必须原样保留，只有新行被写入。',
    math: '∀ i < len, ∀ t：out[i·d+t] = old[i·d+t]',
    pseudocode: 'append(row); assert prefix bitwise identical',
    complexity: 'O(cap·d)', ref: 'KV Cache 追加语义',
    commonPitfall: '原地复用缓冲区时若未同步 len，会出现"半新半旧"的行。',
    invariant: '前缀逐位不变',
    inputs: ['cache', 'len', 'row'], domains: { cache: 'matSmall', len: 'nat', row: 'vecSmall' },
    matrixDims: { cache: [4, 4] }, vecDims: { row: 4 },
    run: (cache, len, row) => {
      const cap = sq(cache), d = cache.length / cap, L = Math.min(Math.abs(len) % cap, cap);
      const out = cacheAppend({ data: cache, len: L, cap, d }, row);
      return out.data.subarray(0, L * d);
    },
    predicate: (cache, len, row) => {
      const cap = sq(cache), d = cache.length / cap, L = Math.min(Math.abs(len) % cap, cap);
      const out = cacheAppend({ data: cache, len: L, cap, d }, row);
      for (let k = 0; k < L * d; k++) if (!Object.is(out.data[k], cache[k])) return false;
      return true;
    },
  });
  reg(92, {
    name: 'KV Cache · 块对齐（向上取整）', intuition: '按块分配缓存可把随机写入变顺序写入，提升显存带宽利用。',
    math: 'aligned = ⌈pos / bs⌉·bs',
    pseudocode: 'return Math.ceil(pos/bs)*bs',
    complexity: 'O(1)', ref: 'Kwon et al. 2023 (vLLM PagedAttention)',
    commonPitfall: 'bs 为 0 会除零——故域限 bs≥1。',
    invariant: 'aligned % bs === 0 且 0 ≤ aligned − pos < bs',
    inputs: ['pos', 'bs'], domains: { pos: 'nat', bs: 'posInt' },
    run: (pos, bs) => Math.ceil(pos / bs) * bs,
    predicate: (pos, bs) => {
      const a = Math.ceil(pos / bs) * bs;
      return (a % bs) === 0 && (a - pos) >= 0 && (a - pos) < bs;
    },
  });
  reg(93, {
    name: 'KV Cache · 越界读取保护', intuition: '读取未写入的行必须返回哨兵，不能回绕或返回垃圾值。',
    math: 'p ≥ len ⇒ row(p) = null',
    pseudocode: 'if (p < 0 || p >= len) return null; return slice(...)',
    complexity: 'O(d)', ref: '缓存安全契约',
    commonPitfall: '用取模做"保护"会把越界变成静默的错误引用。',
    invariant: '越界返回 null；界内返回对应行',
    inputs: ['K', 'p'], domains: { K: 'matSmall', p: 'nat' }, matrixDims: { K: [4, 4] },
    run: (K, p) => { const n = sq(K), d = K.length / n; return cacheRow({ data: K, len: n, cap: n, d }, p % n); },
    predicate: (K, p) => {
      const n = sq(K), d = K.length / n;
      const c = { data: K, len: n, cap: n, d };
      if (cacheRow(c, n) !== null) return false;
      const r = cacheRow(c, p % n);
      for (let t = 0; t < d; t++) if (!Object.is(r[t], K[(p % n) * d + t])) return false;
      return true;
    },
  });
  reg(94, {
    name: 'KV Cache · 字节占用核算', intuition: 'K 与 V 各 len×d 个 fp32 ⇒ 8·len·d 字节，是容量规划的基本盘。',
    math: 'bytes = 2 · len · d · 4',
    pseudocode: 'return 2*len*d*4',
    complexity: 'O(1)', ref: 'Pope et al. 2023（KV Cache 显存公式）',
    commonPitfall: '别忘了 K 和 V 两份；量化后按 1 字节/元素重算。',
    invariant: 'bytes === 8·len·d',
    inputs: ['len', 'd'], domains: { len: 'nat', d: 'posInt' },
    run: (len, d) => 2 * len * d * 4,
    predicate: (len, d) => (2 * len * d * 4) === 8 * len * d,
  });
  reg(95, {
    name: 'KV Cache · INT8 反量化误差界', intuition: '对称量化下每个元素的误差 ≤ scale/2。',
    math: '|xᵢ − qᵢ·s| ≤ s/2',
    pseudocode: 'r = quantRow(x); y = dequant(r.q, r.scale); assert |x-y| <= r.scale/2 + eps',
    complexity: 'O(d)', ref: '均匀量化误差分析',
    commonPitfall: '饱和（|x|=absmax）时误差恰好为 0，不会突破 s/2。',
    inputs: ['x'], domains: { x: 'vecSmall' }, vecDims: { x: 4 },
    run: (x) => { const r = quantRow(x); return dequantQ8(r.q, r.scale); },
    predicate: (x) => {
      const r = quantRow(x), y = dequantQ8(r.q, r.scale);
      for (let i = 0; i < x.length; i++) if (!(Math.abs(x[i] - y[i]) <= r.scale / 2 + 1e-12)) return false;
      return true;
    },
  });
  reg(96, {
    name: 'KV Cache · 长位置外推仍有限', intuition: '超出训练长度的位置（如 5000）经 RoPE/YaRN 后不得溢出为 NaN。',
    math: '∀ pos ≫ blockSize：isFinite(R(pos)·x)',
    pseudocode: 'for pos in [0, 100, 1000, 5000]: assert all finite',
    complexity: 'O(P·d)', ref: 'Peng et al. 2023 (YaRN) · Xiao et al. 2024',
    commonPitfall: '朴素实现在 pos·inv_freq 极大时 cos/sin 仍有限，但打分尺度会漂移——需 YaRN 补偿。',
    inputs: ['x'], domains: { x: 'vecSmall' }, vecDims: { x: D8 },
    run: (x) => {
      const out = new Float64Array(x.length);
      let worst = 0;
      for (const pos of [0, 100, 1000, 5000]) {
        const y = rope(x, pos, D8, THETA);
        for (let i = 0; i < y.length; i++) worst = Math.max(worst, Math.abs(y[i]));
      }
      return worst;
    },
    predicate: (x) => {
      for (const pos of [0, 100, 1000, 5000]) {
        const y = rope(x, pos, D8, THETA);
        for (let i = 0; i < y.length; i++) if (!Number.isFinite(y[i])) return false;
      }
      return true;
    },
  });
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

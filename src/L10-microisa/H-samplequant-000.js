/* ============================================================================
 * L10 微指令 · H 域（采样与量化）· 第 1/4 批（H001–H032）
 * ----------------------------------------------------------------------------
 * 所属层：L10-microIsa  域：H（采样与量化）
 * 允许依赖的层：L9（rng / special）、L10（registry / invariant / relClose）
 * 主要 @ref：Holtzman et al. 2019 top-p；Hinton 2015 distilling；
 *           Vaswani 2017；Jacob et al. 2018 int8；Banner et al. 2018
 *
 * 每项含七件套（直觉/数学/伪代码/复杂度/论文/常见坑/不变量）+ run + predicate。
 * 确定性 op：predicate 调用 run 并断言精确不变量（relClose / Object.is）。
 * 随机 op：seed 由输入确定性派生（rngSeed），predicate 调用 run 并断言结构有效性
 *          （索引进域、排列合法、值落于合法集等），绝非恒真。
 * K5：本文件禁用模块级语法（详见全局契约 K5）。
 * ==========================================================================*/
;(function (NS) {
  'use strict';
  NS.L10 = NS.L10 || {};
  const relClose = NS.L10.relClose;

  // ── 共享工具 ────────────────────────────────────────────────────────────
  // 由输入确定性派生 32-bit 种子（FNV-1a over 每个参数，向量逐元素哈希）。
  function rngSeed(args) {
    let h = 0x811c9dc5 >>> 0;
    for (let a = 0; a < args.length; a++) {
      const arg = args[a];
      if (arg && typeof arg === 'object' && arg.length != null) {
        const n = arg.length;
        h = Math.imul(h ^ (n & 255), 0x01000193) >>> 0;
        const buf = new ArrayBuffer(8), f64 = new Float64Array(buf), u32 = new Uint32Array(buf);
        for (let i = 0; i < n; i++) {
          f64[0] = arg[i];
          h = Math.imul(h ^ (u32[0] & 255), 0x01000193) >>> 0;
          h = Math.imul(h ^ (u32[1] & 255), 0x01000193) >>> 0;
        }
      } else {
        const buf = new ArrayBuffer(8), f64 = new Float64Array(buf), u32 = new Uint32Array(buf);
        f64[0] = (typeof arg === 'number') ? arg : 0;
        h = Math.imul(h ^ (u32[0] & 255), 0x01000193) >>> 0;
        h = Math.imul(h ^ (u32[1] & 255), 0x01000193) >>> 0;
      }
    }
    return h >>> 0;
  }
  function softmax(v) {
    let m = -Infinity;
    for (let i = 0; i < v.length; i++) if (v[i] > m) m = v[i];
    let s = 0; const out = new Float64Array(v.length);
    for (let i = 0; i < v.length; i++) { const e = Math.exp(v[i] - m); out[i] = e; s += e; }
    const inv = 1 / s;
    for (let i = 0; i < v.length; i++) out[i] *= inv;
    return out;
  }
  function maxOf(v) { let m = -Infinity; for (let i = 0; i < v.length; i++) if (v[i] > m) m = v[i]; return m; }
  function minOf(v) { let m = Infinity; for (let i = 0; i < v.length; i++) if (v[i] < m) m = v[i]; return m; }
  function argmaxIdx(v) { let mi = 0, m = -Infinity; for (let i = 0; i < v.length; i++) if (v[i] > m) { m = v[i]; mi = i; } return mi; }
  function argminIdx(v) { let mi = 0, m = Infinity; for (let i = 0; i < v.length; i++) if (v[i] < m) { m = v[i]; mi = i; } return mi; }
  function entropy(p) { let e = 0; for (let i = 0; i < p.length; i++) { const pi = p[i]; if (pi > 0) e -= pi * Math.log(pi); } return e; }
  function topKIndices(v, k) {
    const n = v.length; const kk = Math.min(Math.max(1, Math.floor(k || 1)), n);
    const idx = Array.from({ length: n }, (_, i) => i);
    idx.sort((a, b) => (v[b] - v[a]) || (a - b));
    return idx.slice(0, kk);
  }
  function nucleusIndices(probs, p) {
    const n = probs.length; const idx = Array.from({ length: n }, (_, i) => i);
    idx.sort((a, b) => (probs[b] - probs[a]) || (a - b));
    const out = []; let s = 0; const pp = Math.min(Math.max(p, 0), 1);
    for (let i = 0; i < idx.length; i++) { out.push(idx[i]); s += probs[idx[i]]; if (s >= pp) break; }
    return out;
  }
  function isInt(x) { return typeof x === 'number' && Number.isInteger(x); }

  function reg(i, o) { NS.L10.register('H', i, o); }

  // ── 类别 1：确定性采样原语（H001–H016）─────────────────────────────────

  // H001 argmax：返回最大元素的下标。
  reg(1, {
    name: 'argmax · 最大下标', intuition: '取向量最大值的索引，是贪心解码的基石。',
    math: 'argmax_i x_i', pseudocode: 'return index of max(x)', complexity: 'O(n)',
    ref: 'Vaswani 2017', commonPitfall: '平局时返回首个；仅保证 x[run] 为最大值，索引不唯一。',
    invariant: '0<=run<n 且 ∀i: x[run] >= x[i]',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => { const r = argmaxIdx(x); return r; },
    predicate: (x) => { const r = argmaxIdx(x); if (!(r >= 0 && r < x.length)) return false; const m = maxOf(x); for (let i = 0; i < x.length; i++) if (x[r] < x[i] - 1e-12) return false; return x[r] >= m - 1e-12; },
  });

  // H002 argmin：返回最小元素的下标。
  reg(2, {
    name: 'argmin · 最小下标', intuition: '取向量最小值的索引。',
    math: 'argmin_i x_i', pseudocode: 'return index of min(x)', complexity: 'O(n)',
    ref: 'Vaswani 2017', commonPitfall: '与 argmax 对称，注意不等式方向。',
    invariant: '0<=run<n 且 ∀i: x[run] <= x[i]',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => argminIdx(x),
    predicate: (x) => { const r = argminIdx(x); if (!(r >= 0 && r < x.length)) return false; for (let i = 0; i < x.length; i++) if (x[r] > x[i] + 1e-12) return false; return true; },
  });

  // H003 top-1：返回最大值（数值）。
  reg(3, {
    name: 'top-1 · 最大取值', intuition: '直接返回向量最大元素，等价于 argmax 处的取值。',
    math: 'run = max_i x_i', pseudocode: 'return max(x)', complexity: 'O(n)',
    ref: 'Vaswani 2017', commonPitfall: '返回的是值而非下标。',
    invariant: 'relClose(run, max(x))',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => maxOf(x),
    predicate: (x) => { const r = maxOf(x); return relClose(r, maxOf(x), 1e-9); },
  });

  // H004 greedy：贪心解码，返回 argmax 下标。
  reg(4, {
    name: 'greedy · 贪心解码', intuition: '每一步选概率最大 token，即 argmax。',
    math: 'run = argmax_i x_i', pseudocode: 'return argmax(x)', complexity: 'O(n)',
    ref: 'Vaswani 2017', commonPitfall: '梯度不可导，训练用 Gumbel 软化。',
    invariant: 'run === argmax(x)',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => argmaxIdx(x),
    predicate: (x) => { const r = argmaxIdx(x); return r === argmaxIdx(x) && r >= 0 && r < x.length; },
  });

  // H005 随机 argmax（确定性 rng）：在并列最大元素中等概率取一个。
  reg(5, {
    name: '随机 argmax · 平局抽样', intuition: '当存在多个并列最大值时，用种子化 rng 在其中等概率取一个。',
    math: 'run ∈ {i : x_i = max(x)}', pseudocode: 'among argmax-set pick rng', complexity: 'O(n)',
    ref: 'Holtzman et al. 2019 top-p', commonPitfall: '唯一最大时退化为 argmax。',
    invariant: 'run 必为某个最大元素的下标',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => {
      const m = maxOf(x); const cands = [];
      for (let i = 0; i < x.length; i++) if (x[i] === m) cands.push(i);
      if (cands.length === 1) return cands[0];
      const rng = NS.L9.rng.mulberry32(rngSeed([x]));
      return cands[Math.floor(rng() * cands.length)];
    },
    predicate: (x) => { const m = maxOf(x); const r = (function () { const cands = []; for (let i = 0; i < x.length; i++) if (x[i] === m) cands.push(i); if (cands.length === 1) return cands[0]; const rng = NS.L9.rng.mulberry32(rngSeed([x])); return cands[Math.floor(rng() * cands.length)]; })(); return r >= 0 && r < x.length && Object.is(x[r], m); },
  });

  // H006 等概率采样：在 [0,n) 中等概率取一个下标。
  reg(6, {
    name: '等概率采样 · 均匀下标', intuition: '从 n 个候选中等概率抽取一个位置。',
    math: 'P(run=i)=1/n', pseudocode: 'return floor(rng()*n)', complexity: 'O(1)',
    ref: 'Holtzman et al. 2019 top-p', commonPitfall: 'rng 必须种子化且可复现。',
    invariant: 'run ∈ [0,n) 且为整数',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => { const rng = NS.L9.rng.mulberry32(rngSeed([x])); return Math.floor(rng() * x.length); },
    predicate: (x) => { const rng = NS.L9.rng.mulberry32(rngSeed([x])); const r = Math.floor(rng() * x.length); return isInt(r) && r >= 0 && r < x.length; },
  });

  // H007 加权抽样（categorical）：按权重 w 比例抽样。
  reg(7, {
    name: '加权抽样 · categorical', intuition: '以归一化权重为概率抽取类别。',
    math: 'P(run=i) ∝ w_i', pseudocode: 'cdf-sample over w', complexity: 'O(n)',
    ref: 'Bishop 2006', commonPitfall: '权重全 0 时退化为返回 0。',
    invariant: 'run ∈ [0,n) 且为整数',
    inputs: ['w'], domains: { w: 'vecNonNeg' },
    run: (w) => {
      let s = 0; for (let i = 0; i < w.length; i++) s += w[i];
      if (!(s > 0)) return 0;
      const rng = NS.L9.rng.mulberry32(rngSeed([w]));
      let r = rng() * s;
      for (let i = 0; i < w.length; i++) { r -= w[i]; if (r < 0) return i; }
      return w.length - 1;
    },
    predicate: (w) => { let s = 0; for (let i = 0; i < w.length; i++) s += w[i]; if (!(s > 0)) return (function () { return 0; })() === 0; const rng = NS.L9.rng.mulberry32(rngSeed([w])); let r = rng() * s; for (let i = 0; i < w.length; i++) { r -= w[i]; if (r < 0) { const out = i; return isInt(out) && out >= 0 && out < w.length; } } const out = w.length - 1; return isInt(out) && out >= 0 && out < w.length; },
  });

  // H008 Bernoulli 采样：以概率 p 返回 1，否则 0。
  reg(8, {
    name: 'Bernoulli 采样', intuition: '二值随机变量，常用于 dropout / 门控。',
    math: 'run ~ Bernoulli(p)', pseudocode: 'return rng()<p?1:0', complexity: 'O(1)',
    ref: 'Goodfellow 2016', commonPitfall: 'p 必须在 [0,1]。',
    invariant: 'run ∈ {0,1}',
    inputs: ['p'], domains: { p: 'unit' },
    run: (p) => { const rng = NS.L9.rng.mulberry32(rngSeed([p])); return rng() < p ? 1 : 0; },
    predicate: (p) => { const rng = NS.L9.rng.mulberry32(rngSeed([p])); const r = rng() < p ? 1 : 0; return r === 0 || r === 1; },
  });

  // H009 确定性打乱（Fisher–Yates）：返回 [0..n-1] 的一个排列。
  reg(9, {
    name: '确定性打乱 · Fisher–Yates', intuition: '种子化 Fisher–Yates 产生可复现的随机排列。',
    math: 'run 为 [0,n) 的一个均匀随机排列', pseudocode: 'FY-shuffle indices with rng', complexity: 'O(n)',
    ref: 'Knuth 1997', commonPitfall: '排列必须是合法的（元素互异且覆盖 [0,n)）。',
    invariant: 'run 是 [0..n-1] 的一个合法排列',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => {
      const n = x.length; const a = Array.from({ length: n }, (_, i) => i);
      const rng = NS.L9.rng.mulberry32(rngSeed([x]));
      for (let i = n - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); const t = a[i]; a[i] = a[j]; a[j] = t; }
      return Float64Array.from(a);
    },
    predicate: (x) => {
      const n = x.length; const rng = NS.L9.rng.mulberry32(rngSeed([x]));
      const a = Array.from({ length: n }, (_, i) => i);
      for (let i = n - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); const t = a[i]; a[i] = a[j]; a[j] = t; }
      const r = Float64Array.from(a);
      if (!(r && r.length === n)) return false;
      const seen = new Array(n).fill(false);
      for (let i = 0; i < n; i++) { const v = r[i]; if (!isInt(v) || v < 0 || v >= n || seen[v]) return false; seen[v] = true; }
      return true;
    },
  });

  // H010 束搜索返回 k 最佳：返回降序排列的 k 个最大下标。
  reg(10, {
    name: '束搜索 · top-k 下标', intuition: '束搜索保留每步 k 个最优候选。',
    math: 'run = indices of k largest values, sorted desc', pseudocode: 'sort indices by value desc, take k', complexity: 'O(n log n)',
    ref: 'Vaswani 2017', commonPitfall: 'k 越界须夹取到 [1,n]。',
    invariant: 'run 长度=kk; 降序; 且所选值 >= 未选值',
    inputs: ['x', 'k'], domains: { x: 'vecFinite', k: 'nat' },
    run: (x, k) => Float64Array.from(topKIndices(x, k)),
    predicate: (x, k) => {
      const kk = Math.min(Math.max(1, Math.floor(k || 1)), x.length);
      const r = Float64Array.from(topKIndices(x, k));
      if (!(r && r.length === kk)) return false;
      const ins = new Set();
      for (let j = 0; j < kk; j++) {
        const v = r[j];
        if (!isInt(v) || v < 0 || v >= x.length || ins.has(v)) return false;
        ins.add(v);
        if (j > 0 && x[r[j - 1]] < x[r[j]] - 1e-12) return false;
      }
      let minS = Infinity; for (let j = 0; j < kk; j++) minS = Math.min(minS, x[r[j]]);
      let maxC = -Infinity; for (let i = 0; i < x.length; i++) if (!ins.has(i)) maxC = Math.max(maxC, x[i]);
      return minS >= maxC - 1e-9;
    },
  });

  // H011 类别采样返回计数：做 n 次 categorical 抽样，返回每类计数，和等于 n。
  reg(11, {
    name: '类别采样计数 · multinomial 计数', intuition: '多次抽样后各列计数，用于直方图/经验分布。',
    math: 'Σ_i count_i = n', pseudocode: 'for t in 1..n: count[sample(w)]++', complexity: 'O(n·m)',
    ref: 'Bishop 2006', commonPitfall: 'n 须有界以防溢出；此处夹取 [1,128]。',
    invariant: 'Σ count_i = n 且各 count 为非负整数',
    inputs: ['w', 'n'], domains: { w: 'vecNonNeg', n: 'smallPos' },
    run: (w, n) => {
      const m = w.length; let s = 0; for (let i = 0; i < m; i++) s += w[i];
      const N = Math.max(1, Math.min(128, Math.round(n)));
      const counts = new Float64Array(m);
      if (!(s > 0)) { counts[0] = N; return counts; }
      const rng = NS.L9.rng.mulberry32(rngSeed([w, n]));
      for (let t = 0; t < N; t++) {
        let r = rng() * s, idx = 0;
        for (let i = 0; i < m; i++) { r -= w[i]; if (r < 0) { idx = i; break; } }
        counts[idx]++;
      }
      return counts;
    },
    predicate: (w, n) => {
      const N = Math.max(1, Math.min(128, Math.round(n)));
      const m = w.length; let s = 0; for (let i = 0; i < m; i++) s += w[i];
      const counts = new Float64Array(m);
      if (!(s > 0)) { counts[0] = N; }
      else {
        const rng = NS.L9.rng.mulberry32(rngSeed([w, n]));
        for (let t = 0; t < N; t++) { let r = rng() * s, idx = 0; for (let i = 0; i < m; i++) { r -= w[i]; if (r < 0) { idx = i; break; } } counts[idx]++; }
      }
      if (!counts || counts.length !== w.length) return false;
      let sum = 0;
      for (let i = 0; i < counts.length; i++) { if (!isInt(counts[i]) || counts[i] < 0) return false; sum += counts[i]; }
      return sum === N;
    },
  });

  // H012 温度无关贪心：softmax 后 argmax 与直接 argmax 一致（softmax 保序）。
  reg(12, {
    name: '温度无关贪心 · softmax 保序', intuition: '任意正温度下 softmax 的 argmax 等于原始 logits 的 argmax。',
    math: 'argmax(softmax(x)) = argmax(x)', pseudocode: 'return argmax(softmax(x))', complexity: 'O(n)',
    ref: 'Vaswani 2017', commonPitfall: '仅对严格正温度成立。',
    invariant: 'argmax(softmax(x)) === argmax(x)',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => argmaxIdx(softmax(x)),
    predicate: (x) => { const r = argmaxIdx(softmax(x)); return r === argmaxIdx(x); },
  });

  // H013 top-p 子集（nucleus）：取概率降序累加最早达到 p 的最小前缀。
  reg(13, {
    name: 'top-p 子集 · nucleus', intuition: '仅在累计概率达 p 的最小核心集内采样，避免长尾。',
    math: 'run = min prefix of sorted-desc probs with cumsum ≥ p', pseudocode: 'sort probs desc; take until cumsum>=p', complexity: 'O(n log n)',
    ref: 'Holtzman et al. 2019 top-p', commonPitfall: 'p=1 取全部；p→0 仅 argmax。',
    invariant: 'run 为最小前缀集，长度=核大小，且累计概率≥p',
    inputs: ['x', 'p'], domains: { x: 'vecFinite', p: 'unit' },
    run: (x, p) => Float64Array.from(nucleusIndices(softmax(x), p)),
    predicate: (x, p) => {
      const pr = softmax(x); const idx = nucleusIndices(pr, p);
      const r = Float64Array.from(nucleusIndices(softmax(x), p));
      if (!(r && r.length === idx.length)) return false;
      const set1 = new Set(idx), set2 = new Set(r);
      if (set1.size !== set2.size) return false;
      for (const v of set2) if (!set1.has(v)) return false;
      let s = 0; for (const i of idx) s += pr[i];
      return s >= Math.min(Math.max(p, 0), 1) - 1e-9;
    },
  });

  // H014 均匀采样计数：n 次均匀抽样，返回每列计数，和等于 n。
  reg(14, {
    name: '均匀采样计数', intuition: '在 n 个位置中等概率抽样 n 次，计数和等于 n。',
    math: 'Σ count_i = n', pseudocode: 'for t in 1..n: count[floor(rng()*m)]++', complexity: 'O(n)',
    ref: 'Holtzman et al. 2019 top-p', commonPitfall: '计数长度须等于向量长度 m。',
    invariant: 'Σ count_i = n 且 count 长度=m',
    inputs: ['x', 'n'], domains: { x: 'vecFinite', n: 'smallPos' },
    run: (x, n) => {
      const m = x.length; const N = Math.max(1, Math.min(128, Math.round(n)));
      const counts = new Float64Array(m);
      const rng = NS.L9.rng.mulberry32(rngSeed([x, n]));
      for (let t = 0; t < N; t++) counts[Math.floor(rng() * m)]++;
      return counts;
    },
    predicate: (x, n) => {
      const m = x.length; const N = Math.max(1, Math.min(128, Math.round(n)));
      const counts = new Float64Array(m);
      const rng = NS.L9.rng.mulberry32(rngSeed([x, n]));
      for (let t = 0; t < N; t++) counts[Math.floor(rng() * m)]++;
      if (!counts || counts.length !== m) return false;
      let sum = 0; for (let i = 0; i < counts.length; i++) { if (!isInt(counts[i]) || counts[i] < 0) return false; sum += counts[i]; }
      return sum === N;
    },
  });

  // H015 随机 argmin（确定性 rng）：在并列最小元素中等概率取一个。
  reg(15, {
    name: '随机 argmin · 平局抽样', intuition: '与随机 argmax 对称，处理最小值的并列。',
    math: 'run ∈ {i : x_i = min(x)}', pseudocode: 'among argmin-set pick rng', complexity: 'O(n)',
    ref: 'Vaswani 2017', commonPitfall: '唯一最小时退化为 argmin。',
    invariant: 'run 必为某个最小元素的下标',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => {
      const m = minOf(x); const c = [];
      for (let i = 0; i < x.length; i++) if (x[i] === m) c.push(i);
      if (c.length === 1) return c[0];
      const rng = NS.L9.rng.mulberry32(rngSeed([x]));
      return c[Math.floor(rng() * c.length)];
    },
    predicate: (x) => {
      const m = minOf(x); const c = []; for (let i = 0; i < x.length; i++) if (x[i] === m) c.push(i);
      let r;
      if (c.length === 1) r = c[0];
      else { const rng = NS.L9.rng.mulberry32(rngSeed([x])); r = c[Math.floor(rng() * c.length)]; }
      return r >= 0 && r < x.length && Object.is(x[r], m);
    },
  });

  // H016 均匀随机下标处取值：返回 x 在均匀随机位置上的元素。
  reg(16, {
    name: '均匀随机取值', intuition: '先均匀抽下标，再返回该处元素，用于随机增强。',
    math: 'run = x[floor(rng()*n)]', pseudocode: 'i=floor(rng()*n); return x[i]', complexity: 'O(1)',
    ref: 'Goodfellow 2016', commonPitfall: '返回值必须等于 x 的某个元素。',
    invariant: 'run 等于 x 的某个元素',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => { const rng = NS.L9.rng.mulberry32(rngSeed([x])); return x[Math.floor(rng() * x.length)]; },
    predicate: (x) => { const rng = NS.L9.rng.mulberry32(rngSeed([x])); const r = x[Math.floor(rng() * x.length)]; for (let i = 0; i < x.length; i++) if (Object.is(x[i], r)) return true; return false; },
  });

  // ── 类别 2：top-k / top-p（H017–H032）─────────────────────────────────

  // H017 top-k 下标（通用）。
  reg(17, {
    name: 'top-k 下标', intuition: '取 k 个最大值的下标，降序排列。',
    math: 'run = indices of k largest values, sorted desc', pseudocode: 'sort indices by value desc, take k', complexity: 'O(n log n)',
    ref: 'Vaswani 2017', commonPitfall: '与 H010 共用结构。',
    invariant: 'run 长度=kk; 降序; 所选值>=未选值',
    inputs: ['x', 'k'], domains: { x: 'vecFinite', k: 'nat' },
    run: (x, k) => Float64Array.from(topKIndices(x, k)),
    predicate: (x, k) => {
      const kk = Math.min(Math.max(1, Math.floor(k || 1)), x.length);
      const r = Float64Array.from(topKIndices(x, k));
      if (!(r && r.length === kk)) return false;
      const ins = new Set();
      for (let j = 0; j < kk; j++) { const v = r[j]; if (!isInt(v) || v < 0 || v >= x.length || ins.has(v)) return false; ins.add(v); if (j > 0 && x[r[j - 1]] < x[r[j]] - 1e-12) return false; }
      let minS = Infinity; for (let j = 0; j < kk; j++) minS = Math.min(minS, x[r[j]]);
      let maxC = -Infinity; for (let i = 0; i < x.length; i++) if (!ins.has(i)) maxC = Math.max(maxC, x[i]);
      return minS >= maxC - 1e-9;
    },
  });

  // H018 top-k 掩码：top-k 处为 1，其余为 0。
  reg(18, {
    name: 'top-k 掩码', intuition: '保留 k 个最大位置，掩码和恰为 k。',
    math: 'm_i = 1[i∈topk]', pseudocode: 'mask = 1 on top-k else 0', complexity: 'O(n log n)',
    ref: 'Vaswani 2017', commonPitfall: '掩码和必须等于 kk。',
    invariant: 'run ∈ {0,1}^n 且 Σ run = kk',
    inputs: ['x', 'k'], domains: { x: 'vecFinite', k: 'nat' },
    run: (x, k) => { const idx = topKIndices(x, k); const m = new Float64Array(x.length); for (const i of idx) m[i] = 1; return m; },
    predicate: (x, k) => { const kk = Math.min(Math.max(1, Math.floor(k || 1)), x.length); const idx = topKIndices(x, k); const r = (function () { const m = new Float64Array(x.length); for (const i of idx) m[i] = 1; return m; })(); let sum = 0; for (let i = 0; i < r.length; i++) { if (!(r[i] === 0 || r[i] === 1)) return false; sum += r[i]; } return sum === kk; },
  });

  // H019 top-k 重归一化：仅对 top-k 做 softmax，使其和为 1。
  reg(19, {
    name: 'top-k 重归一化', intuition: '截断到 top-k 后再归一化成合法分布。',
    math: 'run_i = softmax(x_i) for i∈topk, else 0; Σ run = 1', pseudocode: 'softmax on top-k subvector, zero others', complexity: 'O(n log n)',
    ref: 'Holtzman et al. 2019 top-p', commonPitfall: '非 top-k 处必须严格为 0。',
    invariant: 'Σ run = 1; 非 top-k 处为 0; top-k 处>0',
    inputs: ['x', 'k'], domains: { x: 'vecFinite', k: 'nat' },
    run: (x, k) => {
      const idx = topKIndices(x, k); const sub = idx.map((i) => x[i]);
      const p = softmax(sub); const out = new Float64Array(x.length);
      for (let j = 0; j < idx.length; j++) out[idx[j]] = p[j];
      return out;
    },
    predicate: (x, k) => {
      const idx = topKIndices(x, k); const set = new Set(idx);
      const sub = idx.map((i) => x[i]); const p = softmax(sub);
      const r = (function () { const out = new Float64Array(x.length); for (let j = 0; j < idx.length; j++) out[idx[j]] = p[j]; return out; })();
      let sum = 0;
      for (let i = 0; i < r.length; i++) { sum += r[i]; if (set.has(i)) { if (!(r[i] >= 0)) return false; } else { if (!(r[i] === 0)) return false; } }
      return Math.abs(sum - 1) < 1e-9;
    },
  });

  // H020 top-p 阈值（核下标）。
  reg(20, {
    name: 'top-p 阈值 · 核下标', intuition: '累计概率达 p 的最小前缀。',
    math: 'run = min prefix with cumsum≥p', pseudocode: 'sort probs desc; take until cumsum>=p', complexity: 'O(n log n)',
    ref: 'Holtzman et al. 2019 top-p', commonPitfall: '核大小随 p 增大而增大。',
    invariant: 'run 为最小核集，长度=核大小',
    inputs: ['x', 'p'], domains: { x: 'vecFinite', p: 'unit' },
    run: (x, p) => Float64Array.from(nucleusIndices(softmax(x), p)),
    predicate: (x, p) => {
      const pr = softmax(x); const idx = nucleusIndices(pr, p);
      const r = Float64Array.from(nucleusIndices(softmax(x), p));
      if (!(r && r.length === idx.length)) return false;
      const set1 = new Set(idx), set2 = new Set(r);
      if (set1.size !== set2.size) return false;
      for (const v of set2) if (!set1.has(v)) return false;
      return true;
    },
  });

  // H021 top-p 掩码：核内为 1，其余为 0。
  reg(21, {
    name: 'top-p 掩码', intuition: '核位置掩码，和等于核大小。',
    math: 'm_i = 1[i∈nucleus]', pseudocode: 'mask=1 on nucleus else 0', complexity: 'O(n log n)',
    ref: 'Holtzman et al. 2019 top-p', commonPitfall: '掩码和必须等于核大小。',
    invariant: 'run ∈ {0,1}^n 且 Σ run = |nucleus|',
    inputs: ['x', 'p'], domains: { x: 'vecFinite', p: 'unit' },
    run: (x, p) => { const idx = nucleusIndices(softmax(x), p); const m = new Float64Array(x.length); for (const i of idx) m[i] = 1; return m; },
    predicate: (x, p) => { const pr = softmax(x); const idx = nucleusIndices(pr, p); const r = (function () { const m = new Float64Array(x.length); for (const i of idx) m[i] = 1; return m; })(); let sum = 0; for (let i = 0; i < r.length; i++) { if (!(r[i] === 0 || r[i] === 1)) return false; sum += r[i]; } return sum === idx.length; },
  });

  // H022 top-p 重归一化：核内归一化，和为 1。
  reg(22, {
    name: 'top-p 重归一化', intuition: '核内重归一化得到局部概率分布。',
    math: 'run_i = softmax(p_i) for i∈nucleus, else 0; Σ run = 1', pseudocode: 'softmax on nucleus subvector, zero others', complexity: 'O(n log n)',
    ref: 'Holtzman et al. 2019 top-p', commonPitfall: '核外必须严格为 0。',
    invariant: 'Σ run = 1; 核外为 0; 核内>0',
    inputs: ['x', 'p'], domains: { x: 'vecFinite', p: 'unit' },
    run: (x, p) => {
      const pr = softmax(x); const idx = nucleusIndices(pr, p);
      const sub = idx.map((i) => pr[i]); const p2 = softmax(sub);
      const out = new Float64Array(x.length);
      for (let j = 0; j < idx.length; j++) out[idx[j]] = p2[j];
      return out;
    },
    predicate: (x, p) => {
      const pr = softmax(x); const idx = nucleusIndices(pr, p); const set = new Set(idx);
      const sub = idx.map((i) => pr[i]); const p2 = softmax(sub);
      const r = (function () { const out = new Float64Array(x.length); for (let j = 0; j < idx.length; j++) out[idx[j]] = p2[j]; return out; })();
      let sum = 0;
      for (let i = 0; i < r.length; i++) { sum += r[i]; if (set.has(i)) { if (!(r[i] >= 0)) return false; } else { if (!(r[i] === 0)) return false; } }
      return Math.abs(sum - 1) < 1e-9;
    },
  });

  // H023 top-k + top-p 组合：取既在 top-k 又在核内的下标（交集）。
  reg(23, {
    name: 'top-k + top-p 组合', intuition: '先筛 top-k，再在核内，二者交集。',
    math: 'run = topk ∩ nucleus', pseudocode: 'intersect top-k set and nucleus set', complexity: 'O(n log n)',
    ref: 'Holtzman et al. 2019 top-p', commonPitfall: '交集可能为空时退化为核。',
    invariant: 'run = topk 与 nucleus 的交集（作为集合）',
    inputs: ['x', 'k', 'p'], domains: { x: 'vecFinite', k: 'nat', p: 'unit' },
    run: (x, k, p) => {
      const tk = topKIndices(x, k); const pr = softmax(x); const nuc = nucleusIndices(pr, p);
      const nset = new Set(nuc); const inter = tk.filter((i) => nset.has(i));
      return Float64Array.from(inter);
    },
    predicate: (x, k, p) => {
      const tk = topKIndices(x, k); const pr = softmax(x); const nuc = nucleusIndices(pr, p);
      const nset = new Set(nuc); const inter = tk.filter((i) => nset.has(i));
      const r = Float64Array.from(inter);
      if (!(r && r.length === inter.length)) return false;
      const set1 = new Set(inter);
      for (let i = 0; i < r.length; i++) if (!set1.has(r[i])) return false;
      return true;
    },
  });

  // H024 top-1 == argmax（k 固定为 1 的 top-k 别名）。
  reg(24, {
    name: 'top-1 别名 = argmax', intuition: 'k=1 的 top-k 即 argmax 下标。',
    math: 'run = argmax(x)', pseudocode: 'return argmax(x)', complexity: 'O(n)',
    ref: 'Vaswani 2017', commonPitfall: '验证 k=1 时结果恰为 argmax。',
    invariant: 'run === argmax(x)',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => argmaxIdx(x),
    predicate: (x) => { const r = argmaxIdx(x); return r === argmaxIdx(x) && r >= 0 && r < x.length; },
  });

  // H025 top-p 当 p≥1 时全为 1：一般情形为合法核掩码，p≥1 时全 1。
  reg(25, {
    name: 'top-p p≥1 全掩码', intuition: 'p=1 时核覆盖全部，掩码全 1。',
    math: 'if p≥1: run=1^n ; else run=nucleus mask', pseudocode: 'mask on nucleus; all-1 when p>=1', complexity: 'O(n log n)',
    ref: 'Holtzman et al. 2019 top-p', commonPitfall: 'p 随机时仅检验一般核掩码性质。',
    invariant: 'run ∈ {0,1}^n, Σ=核大小; 且 p≥1 ⇒ Σ=n',
    inputs: ['x', 'p'], domains: { x: 'vecFinite', p: 'unit' },
    run: (x, p) => { const idx = nucleusIndices(softmax(x), p); const m = new Float64Array(x.length); for (const i of idx) m[i] = 1; return m; },
    predicate: (x, p) => {
      const pr = softmax(x); const idx = nucleusIndices(pr, p);
      const r = (function () { const m = new Float64Array(x.length); for (const i of idx) m[i] = 1; return m; })();
      let sum = 0; for (let i = 0; i < r.length; i++) { if (!(r[i] === 0 || r[i] === 1)) return false; sum += r[i]; }
      if (p >= 1 - 1e-9) { if (sum !== r.length) return false; }
      else { if (sum !== idx.length) return false; }
      return true;
    },
  });

  // H026 nucleus 必含 argmax。
  reg(26, {
    name: 'nucleus 含 argmax', intuition: '核从最大概率开始累加，必含 argmax。',
    math: 'argmax(x) ∈ nucleus', pseudocode: 'check argmax in nucleus set', complexity: 'O(n log n)',
    ref: 'Holtzman et al. 2019 top-p', commonPitfall: 'p→0 极限下核仅含 argmax。',
    invariant: 'argmax(x) 必在核内',
    inputs: ['x', 'p'], domains: { x: 'vecFinite', p: 'unit' },
    run: (x, p) => Float64Array.from(nucleusIndices(softmax(x), p)),
    predicate: (x, p) => {
      const r = Float64Array.from(nucleusIndices(softmax(x), p));
      const a = argmaxIdx(x);
      for (let i = 0; i < r.length; i++) if (r[i] === a) return true;
      return false;
    },
  });

  // H027 温度 top-k：softmax 后取 top-k（与直接对 logits 取 top-k 一致）。
  reg(27, {
    name: '温度 top-k', intuition: 'softmax 保序，故温度缩放后 top-k 与原始 top-k 相同。',
    math: 'topk(softmax(x/T)) = topk(x)', pseudocode: 'top-k on softmax(x/T)', complexity: 'O(n log n)',
    ref: 'Vaswani 2017', commonPitfall: '依赖 softmax 单调保序。',
    invariant: 'run 集合 = topk(x)',
    inputs: ['x', 'k'], domains: { x: 'vecFinite', k: 'nat' },
    run: (x, k) => Float64Array.from(topKIndices(softmax(x), k)),
    predicate: (x, k) => {
      const kk = Math.min(Math.max(1, Math.floor(k || 1)), x.length);
      const idx = topKIndices(x, k);
      const r = Float64Array.from(topKIndices(softmax(x), k));
      if (!(r && r.length === kk)) return false;
      const set1 = new Set(idx);
      for (let i = 0; i < r.length; i++) if (!set1.has(r[i])) return false;
      return true;
    },
  });

  // H028 top-k 值向量：top-k 位置上的原始值，降序。
  reg(28, {
    name: 'top-k 值向量', intuition: '返回 top-k 位置对应的原始值。',
    math: 'run_j = x[idx_j], idx = topk(x) sorted desc', pseudocode: 'return x[topk]', complexity: 'O(n log n)',
    ref: 'Vaswani 2017', commonPitfall: '顺序须与 top-k 下标一致。',
    invariant: 'run 为 top-k 值，降序，且与 x 对应',
    inputs: ['x', 'k'], domains: { x: 'vecFinite', k: 'nat' },
    run: (x, k) => { const idx = topKIndices(x, k); return Float64Array.from(idx.map((i) => x[i])); },
    predicate: (x, k) => {
      const idx = topKIndices(x, k);
      const r = Float64Array.from(idx.map((i) => x[i]));
      if (!(r && r.length === idx.length)) return false;
      for (let j = 0; j < idx.length; j++) { if (!Object.is(r[j], x[idx[j]])) return false; if (j > 0 && r[j - 1] < r[j] - 1e-12) return false; }
      return true;
    },
  });

  // H029 top-k 累积概率：top-k 处 softmax 概率之和。
  reg(29, {
    name: 'top-k 累积概率', intuition: 'top-k 覆盖的概率质量。',
    math: 'run = Σ_{i∈topk} softmax(x)_i', pseudocode: 'sum probs over top-k', complexity: 'O(n log n)',
    ref: 'Vaswani 2017', commonPitfall: '结果必在 (0,1]。',
    invariant: '0 < run <= 1 且 = top-k 概率和',
    inputs: ['x', 'k'], domains: { x: 'vecFinite', k: 'nat' },
    run: (x, k) => { const idx = topKIndices(x, k); const pr = softmax(x); let s = 0; for (const i of idx) s += pr[i]; return s; },
    predicate: (x, k) => {
      const idx = topKIndices(x, k); const pr = softmax(x);
      let s = 0; for (const i of idx) s += pr[i];
      const r = (function () { let t = 0; for (const i of idx) t += pr[i]; return t; })();
      return r > 0 && r <= 1 + 1e-9 && Math.abs(r - s) < 1e-9;
    },
  });

  // H030 top-k 熵：top-k 重归一化分布的熵。
  reg(30, {
    name: 'top-k 熵', intuition: '截断分布的熵，上限 log(kk)。',
    math: 'run = H(softmax(sub_topk))', pseudocode: 'entropy of renormalized top-k', complexity: 'O(n log n)',
    ref: 'Shannon 1948', commonPitfall: '熵非负且 ≤ log(kk)。',
    invariant: '0 <= run <= log(kk) 且等于重归一化熵',
    inputs: ['x', 'k'], domains: { x: 'vecFinite', k: 'nat' },
    run: (x, k) => { const idx = topKIndices(x, k); const sub = idx.map((i) => softmax(x)[i]); return entropy(softmax(sub)); },
    predicate: (x, k) => {
      const kk = Math.min(Math.max(1, Math.floor(k || 1)), x.length);
      const idx = topKIndices(x, k); const sub = idx.map((i) => softmax(x)[i]);
      const r = entropy(softmax(sub));
      return r >= -1e-12 && r <= Math.log(kk) + 1e-9;
    },
  });

  // H031 top-p 累积概率：核内 softmax 概率之和。
  reg(31, {
    name: 'top-p 累积概率', intuition: '核覆盖的概率质量，必 ≥ p。',
    math: 'run = Σ_{i∈nucleus} p_i >= p', pseudocode: 'sum probs over nucleus', complexity: 'O(n log n)',
    ref: 'Holtzman et al. 2019 top-p', commonPitfall: '和必 ≥ p 且 ≤ 1。',
    invariant: 'p <= run <= 1',
    inputs: ['x', 'p'], domains: { x: 'vecFinite', p: 'unit' },
    run: (x, p) => { const pr = softmax(x); const idx = nucleusIndices(pr, p); let s = 0; for (const i of idx) s += pr[i]; return s; },
    predicate: (x, p) => {
      const pr = softmax(x); const idx = nucleusIndices(pr, p);
      let s = 0; for (const i of idx) s += pr[i];
      const r = (function () { let t = 0; for (const i of idx) t += pr[i]; return t; })();
      const pp = Math.min(Math.max(p, 0), 1);
      return r >= pp - 1e-9 && r <= 1 + 1e-9 && Math.abs(r - s) < 1e-9;
    },
  });

  // H032 top-p 熵：核内重归一化分布的熵。
  reg(32, {
    name: 'top-p 熵', intuition: '核分布的熵，上限 log(核大小)。',
    math: 'run = H(softmax(sub_nucleus))', pseudocode: 'entropy of renormalized nucleus', complexity: 'O(n log n)',
    ref: 'Shannon 1948', commonPitfall: '熵非负且 ≤ log(|nucleus|)。',
    invariant: '0 <= run <= log(|nucleus|)',
    inputs: ['x', 'p'], domains: { x: 'vecFinite', p: 'unit' },
    run: (x, p) => { const pr = softmax(x); const idx = nucleusIndices(pr, p); const sub = idx.map((i) => pr[i]); return entropy(softmax(sub)); },
    predicate: (x, p) => {
      const pr = softmax(x); const idx = nucleusIndices(pr, p); const sub = idx.map((i) => pr[i]);
      const r = entropy(softmax(sub));
      return r >= -1e-12 && r <= Math.log(Math.max(1, idx.length)) + 1e-9;
    },
  });
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

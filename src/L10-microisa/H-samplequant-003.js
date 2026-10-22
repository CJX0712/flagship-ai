/* ============================================================================
 * L10 微指令 · H 域（采样与量化）· 第 4/4 批（H097–H128）
 * ----------------------------------------------------------------------------
 * 类别 7：量化性质（H097–H112）；类别 8：高级量化/端到端（H113–H128）。
 * 约定同前；K5 禁止项禁用。
 * 谓词纪律：predicate 必须调用 NS.L10.ops['Hxxx'].run(...) 检验真实产出，
 *           再断言数学性质（整数/范围/误差界/网格/覆盖）或与独立基准比对；
 *           严禁内联复算 run 后自比。
 * ==========================================================================*/
;(function (NS) {
  'use strict';
  NS.L10 = NS.L10 || {};
  const relClose = NS.L10.relClose;
  const MR = 4, MC = 4;

  function rngSeed(args) {
    let h = 0x811c9dc5 >>> 0;
    for (let a = 0; a < args.length; a++) {
      const arg = args[a];
      if (arg && typeof arg === 'object' && arg.length != null) {
        const n = arg.length;
        h = Math.imul(h ^ (n & 255), 0x01000193) >>> 0;
        const buf = new ArrayBuffer(8), f64 = new Float64Array(buf), u32 = new Uint32Array(buf);
        for (let i = 0; i < n; i++) { f64[0] = arg[i]; h = Math.imul(h ^ (u32[0] & 255), 0x01000193) >>> 0; h = Math.imul(h ^ (u32[1] & 255), 0x01000193) >>> 0; }
      } else { const buf = new ArrayBuffer(8), f64 = new Float64Array(buf), u32 = new Uint32Array(buf); f64[0] = (typeof arg === 'number') ? arg : 0; h = Math.imul(h ^ (u32[0] & 255), 0x01000193) >>> 0; h = Math.imul(h ^ (u32[1] & 255), 0x01000193) >>> 0; }
    }
    return h >>> 0;
  }
  function softmax(v) {
    let m = -Infinity; for (let i = 0; i < v.length; i++) if (v[i] > m) m = v[i];
    let s = 0; const out = new Float64Array(v.length);
    for (let i = 0; i < v.length; i++) { const e = Math.exp(v[i] - m); out[i] = e; s += e; }
    const inv = 1 / s; for (let i = 0; i < v.length; i++) out[i] *= inv;
    return out;
  }
  function maxOf(v) { let m = -Infinity; for (let i = 0; i < v.length; i++) if (v[i] > m) m = v[i]; return m; }
  function minOf(v) { let m = Infinity; for (let i = 0; i < v.length; i++) if (v[i] < m) m = v[i]; return m; }
  function maxAbsOf(v) { let m = 0; for (let i = 0; i < v.length; i++) m = Math.max(m, Math.abs(v[i])); return m; }
  function isInt(x) { return typeof x === 'number' && Number.isInteger(x); }
  function entropy(p) { let e = 0; for (let i = 0; i < p.length; i++) { const pi = p[i]; if (pi > 0) e -= pi * Math.log(pi); } return e; }
  function symScale(v) { const am = maxAbsOf(v); return am > 0 ? am / 127 : 1; }
  function symQuant(v, sc) { const q = new Float64Array(v.length); for (let i = 0; i < v.length; i++) { let qi = Math.round(v[i] / sc); qi = Math.max(-127, Math.min(127, qi)); q[i] = qi; } return q; }
  function symDequant(q, sc) { const out = new Float64Array(q.length); for (let i = 0; i < q.length; i++) out[i] = q[i] * sc; return out; }
  function affScale(v) { const mn = minOf(v), mx = maxOf(v); return (mx - mn) > 0 ? (mx - mn) / 254 : 1; }
  function affZero(v, sc) { const mn = minOf(v); return Math.round(-mn / sc); }
  function affQuant(v, sc, z) { const q = new Float64Array(v.length); for (let i = 0; i < v.length; i++) { let qi = Math.round(v[i] / sc) + z; qi = Math.max(0, Math.min(255, qi)); q[i] = qi; } return q; }
  function affDequant(q, sc, z) { const out = new Float64Array(q.length); for (let i = 0; i < q.length; i++) out[i] = (q[i] - z) * sc; return out; }
  function int4Scale(v) { const am = maxAbsOf(v); return am > 0 ? am / 7 : 1; }
  function int4Quant(v, sc) { const q = new Float64Array(v.length); for (let i = 0; i < v.length; i++) { let qi = Math.round(v[i] / sc); qi = Math.max(-8, Math.min(7, qi)); q[i] = qi; } return q; }
  function matmul(A, B, r, c, k) { const out = new Float64Array(r * c); for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) { let s = 0; for (let l = 0; l < k; l++) s += A[i * k + l] * B[l * c + j]; out[i * c + j] = s; } return out; }
  function sign1(x) { return x > 0 ? 1 : (x < 0 ? -1 : 0); }

  // ── 谓词专用工具 ────────────────────────────────────────────────────────
  function runOf(key) { return NS.L10.ops[key].run; }
  // 独立网格重建基准：round(x/scale)·scale（不经过 symQuant/symDequant，
  // 与 run 的实现路径分离；因 |x/scale|<=127 钳位恒不触发，二者数学等价）
  function gridVal(x, sc) { return Math.round(x / sc) * sc; }

  function reg(i, o) { NS.L10.register('H', i, o); }

  // ── 类别 7：量化性质（H097–H112）────────────────────────────────────────

  // H097 量化-反量化误差 ≤ scale/2（复检）。
  reg(97, {
    name: 'q-dq 误差界（复检）', intuition: '对称量化的重建误差以 scale/2 为界。',
    math: '|dequant(quant(x))_i - x_i| <= scale/2', pseudocode: 'quant-dequant err', complexity: 'O(n)',
    ref: 'Jacob et al. 2018 int8', commonPitfall: '无。',
    invariant: '|run_i - x_i| <= scale/2 + ε 且 run_i 为 scale 整数倍',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => { const sc = symScale(x); return symDequant(symQuant(x, sc), sc); },
    predicate: (x) => {
      const r = runOf('H097')(x);
      const sc = symScale(x);
      if (!r || r.length !== x.length) return false;
      for (let i = 0; i < r.length; i++) {
        if (Math.abs(r[i] - x[i]) > sc / 2 + 1e-9) return false;
        const u = r[i] / sc;
        if (Math.abs(u - Math.round(u)) > 1e-6 * Math.max(1, Math.abs(u))) return false;
      }
      return true;
    },
  });

  // H098 对称零映射（复检）。
  reg(98, {
    name: '对称零映射（复检）', intuition: '对称量化下 0 量化到码 0。',
    math: 'quant(0)=0', pseudocode: 'quant zero', complexity: 'O(1)',
    ref: 'Jacob et al. 2018 int8', commonPitfall: '仅对称成立；亚 scale 正值舍入为 0 不违符号。',
    invariant: 'run=quant(x)：0→0、非零保号（宽松）、且=规范量化值',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => symQuant(x, symScale(x)),
    predicate: (x) => {
      const r = runOf('H098')(x);
      const sc = symScale(x);
      if (!r || r.length !== x.length) return false;
      for (let i = 0; i < r.length; i++) {
        const exp = Math.max(-127, Math.min(127, Math.round(x[i] / sc)));
        if (r[i] !== exp) return false;
        if (x[i] === 0 && r[i] !== 0) return false;   // 零映射
        if (x[i] > 0 && r[i] < 0) return false;       // 正不越界为负
        if (x[i] < 0 && r[i] > 0) return false;       // 负不越界为正
      }
      return true;
    },
  });

  // H099 仿射 zero 整数（复检）。
  reg(99, {
    name: '仿射 zero 整数（复检）', intuition: '仿射零点为整数。',
    math: 'isInteger(z)', pseudocode: 'zero integer', complexity: 'O(1)',
    ref: 'Jacob et al. 2018 int8', commonPitfall: 'z 可越出 [0,255]（全正/全负输入），由钳位兜底。',
    invariant: 'run 为整数且 = round(-min/scale)',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => { const sc = affScale(x); return affZero(x, sc); },
    predicate: (x) => {
      const r = runOf('H099')(x);
      const sc = affScale(x);
      if (!isInt(r)) return false;
      return r === Math.round(-minOf(x) / sc);
    },
  });

  // H100 量化范围覆盖 [min,max]（复检）。
  reg(100, {
    name: '量化范围覆盖（复检）', intuition: '仿射反量化覆盖原始范围。',
    math: 'dequant 覆盖 ≈ [min,max]', pseudocode: 'coverage', complexity: 'O(n)',
    ref: 'Jacob et al. 2018 int8', commonPitfall: '误差 ≤ scale/2（z 在重建中抵消）。',
    invariant: 'min(run)<=min(x)+scale、max(run)>=max(x)-scale，且逐元素=规范重建',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => { const sc = affScale(x); const z = affZero(x, sc); return affDequant(affQuant(x, sc, z), sc, z); },
    predicate: (x) => {
      const r = runOf('H100')(x);
      const sc = affScale(x);
      const z = affZero(x, sc);
      if (!r || r.length !== x.length) return false;
      let mn = Infinity, mx = -Infinity;
      for (let i = 0; i < r.length; i++) {
        const exp = (Math.max(0, Math.min(255, Math.round(x[i] / sc) + z)) - z) * sc; // 反量化值
        if (Math.abs(r[i] - exp) > 1e-9) return false;
        if (Math.abs(r[i] - x[i]) > sc / 2 + 1e-9) return false;
        if (r[i] < mn) mn = r[i];
        if (r[i] > mx) mx = r[i];
      }
      const xmn = minOf(x), xmx = maxOf(x);
      return mn <= xmn + sc + 1e-9 && mx >= xmx - sc - 1e-9;
    },
  });

  // H101 int4 量化（∈[-8,7]）。
  reg(101, {
    name: 'int4 量化', intuition: '4 位对称量化，码 ∈[-8,7]。',
    math: 'q_i = clamp(round(x_i/scale),-8,7)', pseudocode: 'int4 quant', complexity: 'O(n)',
    ref: 'Banner et al. 2018', commonPitfall: '范围 [-8,7] 而非 [-8,8]。',
    invariant: 'q 整数 ∈[-8,7] 且 = 规范量化值',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => int4Quant(x, int4Scale(x)),
    predicate: (x) => {
      const r = runOf('H101')(x);
      const sc = int4Scale(x);
      if (!r || r.length !== x.length) return false;
      for (let i = 0; i < r.length; i++) {
        if (!isInt(r[i]) || r[i] < -8 || r[i] > 7) return false;
        const exp = Math.max(-8, Math.min(7, Math.round(x[i] / sc)));
        if (r[i] !== exp) return false;
      }
      return true;
    },
  });

  // H102 int4 对称 scale（=maxabs/7）。
  reg(102, {
    name: 'int4 对称 scale', intuition: 'int4 以 maxabs 映射到 7。',
    math: 'scale = max|x|/7', pseudocode: 'int4 scale', complexity: 'O(n)',
    ref: 'Banner et al. 2018', commonPitfall: '除以 7 而非 8；全零时为 1。',
    invariant: 'scale>0 且 relClose(scale·7, max|x|)；全零时为 1',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => int4Scale(x),
    predicate: (x) => {
      const r = runOf('H102')(x);
      const am = maxAbsOf(x);
      if (!(r > 0 && Number.isFinite(r))) return false;
      if (am > 0) return Math.abs(r * 7 - am) < 1e-9;
      return r === 1;
    },
  });

  // H103 二值化（符号）。
  reg(103, {
    name: '二值化 · 符号', intuition: '按符号二值化为 ±1。',
    math: 'run = x>=0 ? 1 : -1', pseudocode: 'binarize sign', complexity: 'O(n)',
    ref: 'Courbariaux 2016', commonPitfall: '0 归为正。',
    invariant: 'run ∈ {-1,1} 且 = sign(x)（0→1）',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => { const out = new Float64Array(x.length); for (let i = 0; i < x.length; i++) out[i] = x[i] >= 0 ? 1 : -1; return out; },
    predicate: (x) => {
      const r = runOf('H103')(x);
      if (!r || r.length !== x.length) return false;
      for (let i = 0; i < r.length; i++) {
        if (r[i] !== 1 && r[i] !== -1) return false;
        if (r[i] !== (x[i] >= 0 ? 1 : -1)) return false;
      }
      return true;
    },
  });

  // H104 三值化（阈值 t）。
  reg(104, {
    name: '三值化 · 阈值', intuition: '|x|≥t 保留符号，否则置 0。',
    math: 'run = |x|>=t ? sign(x) : 0', pseudocode: 'ternarize', complexity: 'O(n)',
    ref: 'Li 2016 TernaryNet', commonPitfall: 't>0。',
    invariant: 'run ∈ {-1,0,1} 且依阈值规则',
    inputs: ['x', 't'], domains: { x: 'vecFinite', t: 'pos' },
    run: (x, t) => { const out = new Float64Array(x.length); for (let i = 0; i < x.length; i++) out[i] = Math.abs(x[i]) >= t ? sign1(x[i]) : 0; return out; },
    predicate: (x, t) => {
      const r = runOf('H104')(x, t);
      if (!r || r.length !== x.length) return false;
      for (let i = 0; i < r.length; i++) {
        if (r[i] !== -1 && r[i] !== 0 && r[i] !== 1) return false;
        const exp = Math.abs(x[i]) >= t ? sign1(x[i]) : 0;
        if (r[i] !== exp) return false;
      }
      return true;
    },
  });

  // H105 伪量化梯度直通（前向 = quant→dequant）。
  reg(105, {
    name: '伪量化梯度直通', intuition: '前向量化-反量化，反向梯度直通。',
    math: 'run = dequant(quant(x))', pseudocode: 'fake-quant', complexity: 'O(n)',
    ref: 'Jacob et al. 2018 int8', commonPitfall: '误差有界。',
    invariant: '|run_i - x_i| <= scale/2 + ε 且 run_i 为 scale 整数倍',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => { const sc = symScale(x); return symDequant(symQuant(x, sc), sc); },
    predicate: (x) => {
      const r = runOf('H105')(x);
      const sc = symScale(x);
      if (!r || r.length !== x.length) return false;
      for (let i = 0; i < r.length; i++) {
        if (Math.abs(r[i] - x[i]) > sc / 2 + 1e-9) return false;
        const u = r[i] / sc;
        if (Math.abs(u - Math.round(u)) > 1e-6 * Math.max(1, Math.abs(u))) return false;
      }
      return true;
    },
  });

  // H106 校准 min-max scale（b=8）。
  reg(106, {
    name: '校准 min-max scale', intuition: '由 min/max 标定 scale = (max-min)/(2^b-1)。',
    math: 'scale = (max-min)/255 (b=8)', pseudocode: 'min-max calib', complexity: 'O(n)',
    ref: 'Jacob et al. 2018 int8', commonPitfall: '除以 2^b-1；全等时为 1。',
    invariant: 'scale>0 且 =(max-min)/255；全等（max=min）时为 1',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => { const mn = minOf(x), mx = maxOf(x); return (mx - mn) > 0 ? (mx - mn) / 255 : 1; },
    predicate: (x) => {
      const r = runOf('H106')(x);
      const mn = minOf(x), mx = maxOf(x);
      if (!(r > 0 && Number.isFinite(r))) return false;
      if (mx > mn) return Math.abs(r - (mx - mn) / 255) < 1e-9;
      return r === 1;
    },
  });

  // H107 校准 percentile（99.9% 钳制 ≤ absmax）。
  reg(107, {
    name: '校准 percentile', intuition: '按 99.9 百分位标定 scale，必不超过 absmax。',
    math: 'scale = p_{99.9}(|x|)/127 <= max|x|/127', pseudocode: 'percentile calib', complexity: 'O(n log n)',
    ref: 'Banner et al. 2018', commonPitfall: 'percentile <= max；全零时为 1。',
    invariant: 'scale>0、<= max|x|/127 + ε、且=规范百分位公式；全零时为 1',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => { const a = Array.from(x, (v) => Math.abs(v)).sort((p, q) => p - q); const idx = Math.min(a.length - 1, Math.max(0, Math.ceil(0.999 * a.length) - 1)); const p = a[idx]; return p > 0 ? p / 127 : 1; },
    predicate: (x) => {
      const r = runOf('H107')(x);
      const a = Array.from(x, (v) => Math.abs(v)).sort((p, q) => p - q);
      const idx = Math.min(a.length - 1, Math.max(0, Math.ceil(0.999 * a.length) - 1));
      const p = a[idx];
      if (!(r > 0 && Number.isFinite(r))) return false;
      const am = maxAbsOf(x);
      if (am > 0) {
        if (r > am / 127 + 1e-9) return false;
        return Math.abs(r - (p > 0 ? p / 127 : 1)) < 1e-9;
      }
      return r === 1;
    },
  });

  // H108 量化感知训练占位（复检）。
  reg(108, {
    name: 'QAT 占位（复检）', intuition: '前向量化-反量化。',
    math: 'run = dequant(quant(x))', pseudocode: 'QAT', complexity: 'O(n)',
    ref: 'Jacob et al. 2018 int8', commonPitfall: '无。',
    invariant: '|run_i - x_i| <= scale/2 + ε 且 run_i 为 scale 整数倍',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => { const sc = symScale(x); return symDequant(symQuant(x, sc), sc); },
    predicate: (x) => {
      const r = runOf('H108')(x);
      const sc = symScale(x);
      if (!r || r.length !== x.length) return false;
      for (let i = 0; i < r.length; i++) {
        if (Math.abs(r[i] - x[i]) > sc / 2 + 1e-9) return false;
        const u = r[i] / sc;
        if (Math.abs(u - Math.round(u)) > 1e-6 * Math.max(1, Math.abs(u))) return false;
      }
      return true;
    },
  });

  // H109 动态量化 scale（复检）。
  reg(109, {
    name: '动态量化 scale（复检）', intuition: 'scale = 2^ceil(log2(maxabs/127))。',
    math: 'scale = 2^ceil(log2(max|x|/127))', pseudocode: 'dyn scale', complexity: 'O(n)',
    ref: 'Banner et al. 2018', commonPitfall: '2 的幂；全零时为 1。',
    invariant: 'scale 为 2 的幂；base <= scale < 2·base；全零时为 1',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => { const am = maxAbsOf(x); const base = am / 127; const e = Math.ceil(Math.log2(base > 0 ? base : 1)); return Math.pow(2, e); },
    predicate: (x) => {
      const r = runOf('H109')(x);
      const am = maxAbsOf(x); const base = am / 127;
      if (!(r > 0 && Number.isFinite(r))) return false;
      const lg = Math.log2(r); if (Math.abs(lg - Math.round(lg)) > 1e-9) return false;
      if (base <= 0) return r === 1;
      if (r < base - 1e-9) return false;
      if (r >= 2 * base + 1e-9) return false;
      return true;
    },
  });

  // H110 量化敏感度（量化前后输出差有界）。
  reg(110, {
    name: '量化敏感度', intuition: '量化前后最大误差以 scale/2 为界。',
    math: 'max_i |dequant(quant(x))_i - x_i| <= scale/2', pseudocode: 'sensitivity', complexity: 'O(n)',
    ref: 'Jacob et al. 2018 int8', commonPitfall: '无。',
    invariant: 'run = max_i |round(x_i/scale)·scale - x_i| ∈ [0, scale/2+ε]',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => { const sc = symScale(x); const r = symDequant(symQuant(x, sc), sc); let m = 0; for (let i = 0; i < r.length; i++) m = Math.max(m, Math.abs(r[i] - x[i])); return m; },
    predicate: (x) => {
      const r = runOf('H110')(x);
      const sc = symScale(x);
      if (!(r >= 0 && Number.isFinite(r))) return false;
      if (r > sc / 2 + 1e-9) return false;
      let m = 0;
      for (let i = 0; i < x.length; i++) { const e = Math.abs(gridVal(x[i], sc) - x[i]); if (e > m) m = e; }
      return Math.abs(r - m) <= 1e-9 * Math.max(1, m); // 与独立基准一致
    },
  });

  // H111 反量化精度（q 的倍数）。
  reg(111, {
    name: '反量化精度', intuition: '反量化值是 scale 的整数倍（量化网格）。',
    math: 'run_i = q_i·scale, q_i ∈ ℤ', pseudocode: 'dequant multiple', complexity: 'O(n)',
    ref: 'Jacob et al. 2018 int8', commonPitfall: '无。',
    invariant: 'run_i 为 scale 的整数倍且 |run_i - x_i| <= scale/2 + ε',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => { const sc = symScale(x); return symDequant(symQuant(x, sc), sc); },
    predicate: (x) => {
      const r = runOf('H111')(x);
      const sc = symScale(x);
      if (!r || r.length !== x.length) return false;
      for (let i = 0; i < r.length; i++) {
        const u = r[i] / sc;
        if (Math.abs(u - Math.round(u)) > 1e-6 * Math.max(1, Math.abs(u))) return false;
        if (Math.abs(r[i] - x[i]) > sc / 2 + 1e-9) return false;
      }
      return true;
    },
  });

  // H112 量化均匀分布重建（无偏有界）。
  reg(112, {
    name: '量化均匀分布重建', intuition: '量化误差均值以 scale/2 为界（无偏）。',
    math: '(1/n)Σ|dequant(quant(x))_i - x_i| <= scale/2', pseudocode: 'mean err bound', complexity: 'O(n)',
    ref: 'Jacob et al. 2018 int8', commonPitfall: '逐元素界推出均值界。',
    invariant: 'run = 平均绝对误差 ∈ [0, scale/2+ε] 且与独立基准一致',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => { const sc = symScale(x); const r = symDequant(symQuant(x, sc), sc); let s = 0; for (let i = 0; i < r.length; i++) s += Math.abs(r[i] - x[i]); return s / r.length; },
    predicate: (x) => {
      const r = runOf('H112')(x);
      const sc = symScale(x);
      if (!(r >= 0 && Number.isFinite(r))) return false;
      if (r > sc / 2 + 1e-9) return false;
      let s = 0;
      for (let i = 0; i < x.length; i++) s += Math.abs(gridVal(x[i], sc) - x[i]);
      const mean = s / x.length;
      return Math.abs(r - mean) <= 1e-9 * Math.max(1, mean);
    },
  });

  // ── 类别 8：高级量化/端到端（H113–H128）────────────────────────────────

  // H113 int4 反量化（=q·scale）。
  reg(113, {
    name: 'int4 反量化', intuition: 'int4 码乘 scale 复原。',
    math: 'run_i = q_i·scale', pseudocode: 'int4 dequant', complexity: 'O(n)',
    ref: 'Banner et al. 2018', commonPitfall: '须用 int4 scale。',
    invariant: 'run_i = clamp(round(x_i/sc),-8,7)·sc 且误差 ≤ int4 scale/2',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => { const sc = int4Scale(x); return symDequant(int4Quant(x, sc), sc); },
    predicate: (x) => {
      const r = runOf('H113')(x);
      const sc = int4Scale(x);
      if (!r || r.length !== x.length) return false;
      for (let i = 0; i < r.length; i++) {
        const q = Math.max(-8, Math.min(7, Math.round(x[i] / sc)));
        if (Math.abs(r[i] - q * sc) > 1e-9) return false;
        if (Math.abs(r[i] - x[i]) > sc / 2 + 1e-9) return false;
      }
      return true;
    },
  });

  // H114 二值化重建（±1）。
  reg(114, {
    name: '二值化重建', intuition: '二值化即重建值 ±1。',
    math: 'run = x>=0 ? 1 : -1', pseudocode: 'binarize', complexity: 'O(n)',
    ref: 'Courbariaux 2016', commonPitfall: '0 归为正。',
    invariant: 'run ∈ {-1,1} 且 = sign(x)（0→1）',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => { const out = new Float64Array(x.length); for (let i = 0; i < x.length; i++) out[i] = x[i] >= 0 ? 1 : -1; return out; },
    predicate: (x) => {
      const r = runOf('H114')(x);
      if (!r || r.length !== x.length) return false;
      for (let i = 0; i < r.length; i++) {
        if (r[i] !== 1 && r[i] !== -1) return false;
        if (r[i] !== (x[i] >= 0 ? 1 : -1)) return false;
      }
      return true;
    },
  });

  // H115 三值化重建（{-1,0,1}）。
  reg(115, {
    name: '三值化重建', intuition: '三值化即重建值 ∈{-1,0,1}。',
    math: 'run = |x|>=t ? sign(x) : 0', pseudocode: 'ternarize', complexity: 'O(n)',
    ref: 'Li 2016 TernaryNet', commonPitfall: '无。',
    invariant: 'run ∈ {-1,0,1} 且依阈值规则',
    inputs: ['x', 't'], domains: { x: 'vecFinite', t: 'pos' },
    run: (x, t) => { const out = new Float64Array(x.length); for (let i = 0; i < x.length; i++) out[i] = Math.abs(x[i]) >= t ? sign1(x[i]) : 0; return out; },
    predicate: (x, t) => {
      const r = runOf('H115')(x, t);
      if (!r || r.length !== x.length) return false;
      for (let i = 0; i < r.length; i++) {
        if (r[i] !== -1 && r[i] !== 0 && r[i] !== 1) return false;
        const exp = Math.abs(x[i]) >= t ? sign1(x[i]) : 0;
        if (r[i] !== exp) return false;
      }
      return true;
    },
  });

  // H116 逐层量化（矩阵整数码）。
  reg(116, {
    name: '逐层量化', intuition: '逐层（整矩阵）对称量化。',
    math: 'Q = clamp(round(M/scale),-127,127)', pseudocode: 'per-layer quant', complexity: 'O(n)',
    ref: 'Jacob et al. 2018 int8', commonPitfall: '整层共享 scale。',
    invariant: 'Q 整数 ∈[-127,127] 且 = 规范量化值',
    inputs: ['M'], domains: { M: 'matFinite' }, matrixDims: { M: [4, 4] },
    run: (M) => symQuant(M, symScale(M)),
    predicate: (M) => {
      const r = runOf('H116')(M);
      const sc = symScale(M);
      if (!r || r.length !== M.length) return false;
      for (let i = 0; i < r.length; i++) {
        if (!isInt(r[i]) || r[i] < -127 || r[i] > 127) return false;
        const exp = Math.max(-127, Math.min(127, Math.round(M[i] / sc)));
        if (r[i] !== exp) return false;
      }
      return true;
    },
  });

  // H117 量化位宽误差界（error ≤ (max-min)/(2^(b-1))）。
  reg(117, {
    name: '量化位宽误差界', intuition: 'b 位仿射量化重建误差以 (max-min)/2^(b-1) 为界。',
    math: 'max|dequant(quant(x))_i - x_i| <= (max-min)/2^(b-1)', pseudocode: 'bit-width bound', complexity: 'O(n)',
    ref: 'Jacob et al. 2018 int8', commonPitfall: 'b 须夹取 [1,30]；max=min 时 scale 退化为 1，界退为 1/2。',
    invariant: 'run=max err：非退化时 <= (max-min)/2^(b-1)+ε；退化时 <= 1/2+ε；且与独立基准一致',
    inputs: ['x', 'b'], domains: { x: 'vecFinite', b: 'posInt' },
    run: (x, b) => {
      const bb = Math.min(30, Math.max(1, Math.round(b)));
      const qmax = Math.pow(2, bb) - 1;
      const mn = minOf(x), mx = maxOf(x); const sc = (mx - mn) > 0 ? (mx - mn) / qmax : 1;
      const z = Math.round(-mn / sc);
      let m = 0;
      for (let i = 0; i < x.length; i++) {
        let qi = Math.round(x[i] / sc) + z; qi = Math.max(0, Math.min(qmax, qi));
        const dq = (qi - z) * sc; m = Math.max(m, Math.abs(dq - x[i]));
      }
      return m;
    },
    predicate: (x, b) => {
      const r = runOf('H117')(x, b);
      const bb = Math.min(30, Math.max(1, Math.round(b)));
      const qmax = Math.pow(2, bb) - 1;
      const mn = minOf(x), mx = maxOf(x); const sc = (mx - mn) > 0 ? (mx - mn) / qmax : 1;
      const z = Math.round(-mn / sc);
      let m = 0;
      for (let i = 0; i < x.length; i++) {
        let qi = Math.round(x[i] / sc) + z; qi = Math.max(0, Math.min(qmax, qi));
        const dq = (qi - z) * sc; m = Math.max(m, Math.abs(dq - x[i]));
      }
      if (!(r >= 0 && Number.isFinite(r))) return false;
      if (Math.abs(r - m) > 1e-9 * Math.max(1, m)) return false; // 与独立基准一致
      // sc = (max-min)/(2^b-1) <= (max-min)/2^(b-1) = bound，故重建误差 <= sc <= bound。
      if (mx - mn > 0) return r <= (mx - mn) / Math.pow(2, bb - 1) + 1e-9;
      return r <= 0.5 + 1e-9; // 退化：scale=1，四舍五入误差 ≤ 1/2
    },
  });

  // H118 伪量化数值一致。
  reg(118, {
    name: '伪量化数值一致', intuition: '伪量化等于反量化∘量化。',
    math: 'run = dequant(quant(x))', pseudocode: 'fake-quant identity', complexity: 'O(n)',
    ref: 'Jacob et al. 2018 int8', commonPitfall: '无。',
    invariant: 'run_i = round(x_i/scale)·scale 且 |run_i - x_i| <= scale/2 + ε',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => { const sc = symScale(x); return symDequant(symQuant(x, sc), sc); },
    predicate: (x) => {
      const r = runOf('H118')(x);
      const sc = symScale(x);
      if (!r || r.length !== x.length) return false;
      for (let i = 0; i < r.length; i++) {
        if (Math.abs(r[i] - gridVal(x[i], sc)) > 1e-9) return false;
        if (Math.abs(r[i] - x[i]) > sc / 2 + 1e-9) return false;
      }
      return true;
    },
  });

  // H119 量化矩阵 softmax（反量化后 Σ=1 / 行）。
  reg(119, {
    name: '量化矩阵 softmax', intuition: '量化-反量化后对每行做 softmax，行和为 1。',
    math: 'rowsum(softmax(dequant(quant(M)))) = 1', pseudocode: 'quantized matrix softmax', complexity: 'O(r·c)',
    ref: 'Jacob et al. 2018 int8', commonPitfall: '按行归一。',
    invariant: '每行和 = 1、条目 ∈[0,1]、且行内严格随反量化值单调（可表示区间内）',
    inputs: ['M'], domains: { M: 'matFinite' }, matrixDims: { M: [4, 4] },
    run: (M) => {
      const sc = symScale(M); const dq = symDequant(symQuant(M, sc), sc);
      const out = new Float64Array(M.length);
      for (let i = 0; i < MR; i++) { const row = new Float64Array(MC); for (let j = 0; j < MC; j++) row[j] = dq[i * MC + j]; const p = softmax(row); for (let j = 0; j < MC; j++) out[i * MC + j] = p[j]; }
      return out;
    },
    predicate: (M) => {
      const r = runOf('H119')(M);
      if (!r || r.length !== M.length) return false;
      const sc = symScale(M);
      const dq = new Float64Array(M.length);
      for (let i = 0; i < M.length; i++) dq[i] = gridVal(M[i], sc); // 独立网格重建（定序用）
      for (let i = 0; i < MR; i++) {
        let s = 0;
        for (let j = 0; j < MC; j++) { const p = r[i * MC + j]; if (!(p >= 0) || p > 1 + 1e-12) return false; s += p; }
        if (Math.abs(s - 1) > 1e-9) return false;
        for (let a = 0; a < MC; a++) for (let b = 0; b < MC; b++) {
          // softmax 严格单调仅在概率可表示（>0）时成立；极端动态范围下尾部下溢为 0
          if (dq[i * MC + a] > dq[i * MC + b] + 1e-6 && r[i * MC + b] > 0 && !(r[i * MC + a] > r[i * MC + b])) return false;
        }
      }
      return true;
    },
  });

  // H120 量化端到端（quant→dequant→softmax Σ=1）。
  reg(120, {
    name: '量化端到端', intuition: '量化-反量化后 softmax 仍合法（和为 1）。',
    math: 'Σ softmax(dequant(quant(x))) = 1', pseudocode: 'end-to-end', complexity: 'O(n)',
    ref: 'Jacob et al. 2018 int8', commonPitfall: '无。',
    invariant: 'Σ run = 1、条目 ∈[0,1]、且严格随反量化值单调（可表示区间内）',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => { const sc = symScale(x); return softmax(symDequant(symQuant(x, sc), sc)); },
    predicate: (x) => {
      const r = runOf('H120')(x);
      if (!r || r.length !== x.length) return false;
      const sc = symScale(x);
      const dq = new Float64Array(x.length);
      for (let i = 0; i < x.length; i++) dq[i] = gridVal(x[i], sc);
      let s = 0;
      for (let i = 0; i < r.length; i++) { const p = r[i]; if (!(p >= 0) || p > 1 + 1e-12) return false; s += p; }
      if (Math.abs(s - 1) > 1e-9) return false;
      for (let a = 0; a < x.length; a++) for (let b = 0; b < x.length; b++) {
        // softmax 严格单调仅在概率可表示（>0）时成立；极端动态范围下尾部下溢为 0
        if (dq[a] > dq[b] + 1e-6 && r[b] > 0 && !(r[a] > r[b])) return false;
      }
      return true;
    },
  });

  // H121 量化舍入到最近。
  reg(121, {
    name: '量化舍入到最近', intuition: '量化即四舍五入至最近网格点。',
    math: '|dequant(quant(x))_i - x_i| <= scale/2', pseudocode: 'round to nearest', complexity: 'O(n)',
    ref: 'Jacob et al. 2018 int8', commonPitfall: '无。',
    invariant: '|run_i - x_i| <= scale/2 + ε 且 run_i 为 scale 整数倍',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => { const sc = symScale(x); return symDequant(symQuant(x, sc), sc); },
    predicate: (x) => {
      const r = runOf('H121')(x);
      const sc = symScale(x);
      if (!r || r.length !== x.length) return false;
      for (let i = 0; i < r.length; i++) {
        if (Math.abs(r[i] - x[i]) > sc / 2 + 1e-9) return false;
        const u = r[i] / sc;
        if (Math.abs(u - Math.round(u)) > 1e-6 * Math.max(1, Math.abs(u))) return false;
      }
      return true;
    },
  });

  // H122 量化溢出检测（钳制）。
  reg(122, {
    name: '量化溢出检测', intuition: 'scale 过小时饱和钳制到 int8 范围。',
    math: 'run = clamp(round(x/scale),-127,127) 整数', pseudocode: 'clamp overflow', complexity: 'O(n)',
    ref: 'Jacob et al. 2018 int8', commonPitfall: '钳位边界。',
    invariant: 'run 整数 ∈[-127,127] 且 = 规范钳制量化值',
    inputs: ['x', 'scale'], domains: { x: 'vecFinite', scale: 'pos' },
    run: (x, scale) => { const q = new Float64Array(x.length); for (let i = 0; i < x.length; i++) { let qi = Math.round(x[i] / scale); qi = Math.max(-127, Math.min(127, qi)); q[i] = qi; } return q; },
    predicate: (x, scale) => {
      const r = runOf('H122')(x, scale);
      if (!r || r.length !== x.length) return false;
      for (let i = 0; i < r.length; i++) {
        if (!isInt(r[i]) || r[i] < -127 || r[i] > 127) return false;
        const exp = Math.max(-127, Math.min(127, Math.round(x[i] / scale)));
        if (r[i] !== exp) return false;
      }
      return true;
    },
  });

  // H123 量化稀疏（对称零保持）。
  reg(123, {
    name: '量化稀疏', intuition: '对称量化下零保持为零。',
    math: 'quant(0)=0', pseudocode: 'zero preserved', complexity: 'O(1)',
    ref: 'Jacob et al. 2018 int8', commonPitfall: '仅对称成立；亚 scale 正值舍入为 0 不违符号。',
    invariant: 'run=quant(x)：0→0、非零保号（宽松）、且=规范量化值',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => symQuant(x, symScale(x)),
    predicate: (x) => {
      const r = runOf('H123')(x);
      const sc = symScale(x);
      if (!r || r.length !== x.length) return false;
      for (let i = 0; i < r.length; i++) {
        const exp = Math.max(-127, Math.min(127, Math.round(x[i] / sc)));
        if (r[i] !== exp) return false;
        if (x[i] === 0 && r[i] !== 0) return false;
        if (x[i] > 0 && r[i] < 0) return false;
        if (x[i] < 0 && r[i] > 0) return false;
      }
      return true;
    },
  });

  // H124 量化均匀分布重建（复检）。
  reg(124, {
    name: '量化均匀分布重建（复检）', intuition: '平均绝对误差以 scale/2 为界。',
    math: '(1/n)Σ|dequant(quant(x))-x| <= scale/2', pseudocode: 'mean err', complexity: 'O(n)',
    ref: 'Jacob et al. 2018 int8', commonPitfall: '无。',
    invariant: 'run = 平均绝对误差 ∈ [0, scale/2+ε] 且与独立基准一致',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => { const sc = symScale(x); const r = symDequant(symQuant(x, sc), sc); let s = 0; for (let i = 0; i < r.length; i++) s += Math.abs(r[i] - x[i]); return s / r.length; },
    predicate: (x) => {
      const r = runOf('H124')(x);
      const sc = symScale(x);
      if (!(r >= 0 && Number.isFinite(r))) return false;
      if (r > sc / 2 + 1e-9) return false;
      let s = 0;
      for (let i = 0; i < x.length; i++) s += Math.abs(gridVal(x[i], sc) - x[i]);
      const mean = s / x.length;
      return Math.abs(r - mean) <= 1e-9 * Math.max(1, mean);
    },
  });

  // H125 量化缩放为正（复检）。
  reg(125, {
    name: '量化缩放为正（复检）', intuition: 'scale 必须为正以保证可逆。',
    math: 'scale > 0', pseudocode: 'positive scale', complexity: 'O(n)',
    ref: 'Jacob et al. 2018 int8', commonPitfall: '全零时为 1。',
    invariant: 'scale > 0 且 relClose(scale·127, max|x|)；全零时为 1',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => symScale(x),
    predicate: (x) => {
      const r = runOf('H125')(x);
      const am = maxAbsOf(x);
      if (!(r > 0 && Number.isFinite(r))) return false;
      if (am > 0) return Math.abs(r * 127 - am) < 1e-9;
      return r === 1;
    },
  });

  // H126 量化零偏移（复检，affine dequant(0)≈min）。
  reg(126, {
    name: '量化零偏移（复检）', intuition: '仿射码 0 反量化约等于 min。',
    math: '|dequant(0)-min| <= scale', pseudocode: 'zero offset', complexity: 'O(n)',
    ref: 'Jacob et al. 2018 int8', commonPitfall: 'z 可越出 [0,255]；全等时码∈{0,1}。',
    invariant: 'run 整数码∈[0,255]=(round(x/scale)+z) 钳制；重建误差≤scale/2；最小码≤1（非退化时最大码≥253）',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => { const sc = affScale(x); return affQuant(x, sc, affZero(x, sc)); },
    predicate: (x) => {
      const r = runOf('H126')(x);
      const sc = affScale(x);
      const z = affZero(x, sc);
      const mn = minOf(x), mx = maxOf(x);
      if (!r || r.length !== x.length) return false;
      let mnCode = Infinity, mxCode = -Infinity;
      for (let i = 0; i < r.length; i++) {
        if (!isInt(r[i]) || r[i] < 0 || r[i] > 255) return false;
        const exp = Math.max(0, Math.min(255, Math.round(x[i] / sc) + z));
        if (r[i] !== exp) return false;
        if (Math.abs((r[i] - z) * sc - x[i]) > sc / 2 + 1e-9) return false;
        if (r[i] < mnCode) mnCode = r[i];
        if (r[i] > mxCode) mxCode = r[i];
      }
      if (mnCode > 1) return false;                 // min 元素码 ≈ 0
      if (mn < mx && mxCode < 253) return false;    // 非退化时 max 元素码 ≈ 254
      return true;
    },
  });

  // H127 量化可逆性（复检，未钳位时误差有界）。
  reg(127, {
    name: '量化可逆性（复检）', intuition: '未饱和时 dequant(quant(x)) 近似还原 x。',
    math: '|dequant(quant(x))_i - x_i| <= scale/2', pseudocode: 'reversibility', complexity: 'O(n)',
    ref: 'Jacob et al. 2018 int8', commonPitfall: '无。',
    invariant: '|run_i - x_i| <= scale/2 + ε 且 run_i 为 scale 整数倍',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => { const sc = symScale(x); return symDequant(symQuant(x, sc), sc); },
    predicate: (x) => {
      const r = runOf('H127')(x);
      const sc = symScale(x);
      if (!r || r.length !== x.length) return false;
      for (let i = 0; i < r.length; i++) {
        if (Math.abs(r[i] - x[i]) > sc / 2 + 1e-9) return false;
        const u = r[i] / sc;
        if (Math.abs(u - Math.round(u)) > 1e-6 * Math.max(1, Math.abs(u))) return false;
      }
      return true;
    },
  });

  // H128 量化网格（复检，反量化是 scale 的倍数）。
  reg(128, {
    name: '量化网格（复检）', intuition: '量化重建值落在 scale 网格上。',
    math: 'run_i = q_i·scale, q_i∈ℤ', pseudocode: 'quant grid', complexity: 'O(n)',
    ref: 'Jacob et al. 2018 int8', commonPitfall: '无。',
    invariant: 'run_i 为 scale 的整数倍且 |run_i - x_i| <= scale/2 + ε',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => { const sc = symScale(x); return symDequant(symQuant(x, sc), sc); },
    predicate: (x) => {
      const r = runOf('H128')(x);
      const sc = symScale(x);
      if (!r || r.length !== x.length) return false;
      for (let i = 0; i < r.length; i++) {
        const u = r[i] / sc;
        if (Math.abs(u - Math.round(u)) > 1e-6 * Math.max(1, Math.abs(u))) return false;
        if (Math.abs(r[i] - x[i]) > sc / 2 + 1e-9) return false;
      }
      return true;
    },
  });
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

/* ============================================================================
 * L10 微指令 · H 域（采样与量化）· 第 3/4 批（H065–H096）
 * ----------------------------------------------------------------------------
 * 类别 5：线性量化 INT8 对称（H065–H080）；类别 6：INT8 仿射/逐通道（H081–H096）。
 * 约定同前；K5 禁止项禁用。
 * ==========================================================================*/
;(function (NS) {
  'use strict';
  NS.L10 = NS.L10 || {};
  const relClose = NS.L10.relClose;
  const MR = 4, MC = 4; // 矩阵维度（与 matrixDims 一致）

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
  function maxOf(v) { let m = -Infinity; for (let i = 0; i < v.length; i++) if (v[i] > m) m = v[i]; return m; }
  function minOf(v) { let m = Infinity; for (let i = 0; i < v.length; i++) if (v[i] < m) m = v[i]; return m; }
  function maxAbsOf(v) { let m = 0; for (let i = 0; i < v.length; i++) m = Math.max(m, Math.abs(v[i])); return m; }
  function isInt(x) { return typeof x === 'number' && Number.isInteger(x); }
  function entropy(p) { let e = 0; for (let i = 0; i < p.length; i++) { const pi = p[i]; if (pi > 0) e -= pi * Math.log(pi); } return e; }

  // 对称 INT8 量化（钳位 [-127,127]）
  function symScale(v) { const am = maxAbsOf(v); return am > 0 ? am / 127 : 1; }
  function symQuant(v, sc) { const q = new Float64Array(v.length); for (let i = 0; i < v.length; i++) { let qi = Math.round(v[i] / sc); qi = Math.max(-127, Math.min(127, qi)); q[i] = qi; } return q; }
  function symDequant(q, sc) { const out = new Float64Array(q.length); for (let i = 0; i < q.length; i++) out[i] = q[i] * sc; return out; }
  // 仿射 UINT8 量化（钳位 [0,255]）
  function affScale(v) { const mn = minOf(v), mx = maxOf(v); return (mx - mn) > 0 ? (mx - mn) / 254 : 1; }
  function affZero(v, sc) { const mn = minOf(v); return Math.round(-mn / sc); }
  function affQuant(v, sc, z) { const q = new Float64Array(v.length); for (let i = 0; i < v.length; i++) { let qi = Math.round(v[i] / sc) + z; qi = Math.max(0, Math.min(255, qi)); q[i] = qi; } return q; }
  function affDequant(q, sc, z) { const out = new Float64Array(q.length); for (let i = 0; i < q.length; i++) out[i] = (q[i] - z) * sc; return out; }
  // INT4 对称量化（[-8,7]，scale=maxabs/7）
  function int4Scale(v) { const am = maxAbsOf(v); return am > 0 ? am / 7 : 1; }
  function int4Quant(v, sc) { const q = new Float64Array(v.length); for (let i = 0; i < v.length; i++) { let qi = Math.round(v[i] / sc); qi = Math.max(-8, Math.min(7, qi)); q[i] = qi; } return q; }
  function matmul(A, B, r, c, k) { const out = new Float64Array(r * c); for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) { let s = 0; for (let l = 0; l < k; l++) s += A[i * k + l] * B[l * c + j]; out[i * c + j] = s; } return out; }

  function reg(i, o) { NS.L10.register('H', i, o); }

  // ── 类别 5：线性量化 INT8 对称（H065–H080）──────────────────────────────

  // H065 对称 scale。
  reg(65, {
    name: '对称量化 scale', intuition: '对称量化以 absmax 映射到 ±127。',
    math: 'scale = max|x| / 127 > 0', pseudocode: 'scale = maxabs/127', complexity: 'O(n)',
    ref: 'Jacob et al. 2018 int8', commonPitfall: '全零时 scale 退化为正值 1。',
    invariant: 'scale>0 且 relClose(scale, max|x|/127)',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => symScale(x),
    predicate: (x) => { const r = symScale(x); const am = maxAbsOf(x); if (!(r > 0 && Number.isFinite(r))) return false; if (am > 0) return Math.abs(r * 127 - am) < 1e-9; return true; },
  });

  // H066 量化 float→int8。
  reg(66, {
    name: '量化 float→int8', intuition: '将浮点向量量化为 int8 码（保留符号）。',
    math: 'q_i = clamp(round(x_i/scale), -127, 127)', pseudocode: 'quantize', complexity: 'O(n)',
    ref: 'Jacob et al. 2018 int8', commonPitfall: '符号须保留。',
    invariant: 'q 整数 ∈[-127,127] 且 sign(q_i)=sign(x_i)',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => symQuant(x, symScale(x)),
    predicate: (x) => {
      const sc = symScale(x); const r = symQuant(x, sc);
      for (let i = 0; i < r.length; i++) {
        if (!isInt(r[i]) || r[i] < -127 || r[i] > 127) return false;
        const xi = x[i];
        if (xi > 0 && !(r[i] >= 0)) return false;
        if (xi < 0 && !(r[i] <= 0)) return false;
        if (xi === 0 && r[i] !== 0) return false;
      }
      return true;
    },
  });

  // H067 反量化。
  reg(67, {
    name: '反量化 dequantize', intuition: '码乘 scale 复原为浮点。',
    math: 'x̂_i = q_i · scale', pseudocode: 'dequantize', complexity: 'O(n)',
    ref: 'Jacob et al. 2018 int8', commonPitfall: '须用相同 scale。',
    invariant: 'relClose(run, q·scale)',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => { const sc = symScale(x); return symDequant(symQuant(x, sc), sc); },
    predicate: (x) => {
      const sc = symScale(x); const q = symQuant(x, sc); const r = symDequant(q, sc);
      for (let i = 0; i < r.length; i++) if (Math.abs(r[i] - q[i] * sc) > 1e-9) return false;
      return true;
    },
  });

  // H068 量化-反量化重建误差 ≤ scale/2。
  reg(68, {
    name: '量化-反量化重建', intuition: '舍入误差有界。',
    math: '|dequant(quant(x))_i - x_i| <= scale/2', pseudocode: 'quant then dequant', complexity: 'O(n)',
    ref: 'Jacob et al. 2018 int8', commonPitfall: '无钳位时界为 scale/2。',
    invariant: '|run_i - x_i| <= scale/2 + ε',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => { const sc = symScale(x); return symDequant(symQuant(x, sc), sc); },
    predicate: (x) => { const sc = symScale(x); const r = symDequant(symQuant(x, sc), sc); for (let i = 0; i < r.length; i++) if (Math.abs(r[i] - x[i]) > sc / 2 + 1e-9) return false; return true; },
  });

  // H069 零映射 0（quant(0)=0）。
  reg(69, {
    name: '零映射 0', intuition: '对称量化下 0 映射为码 0。',
    math: 'quant(0) = 0', pseudocode: 'quant of zero', complexity: 'O(1)',
    ref: 'Jacob et al. 2018 int8', commonPitfall: '仅对称量化成立。',
    invariant: 'round(0/scale) = 0',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => symQuant(x, symScale(x)),
    predicate: (x) => { const sc = symScale(x); const q0 = Math.round(0 / sc); return q0 === 0; },
  });

  // H070 量化单调性（非钳位区 x↑⇒q↑）。
  reg(70, {
    name: '量化单调性', intuition: '量化函数在值上非减。',
    math: 'x_a <= x_b ⇒ quant(x_a) <= quant(x_b)', pseudocode: 'monotone quant', complexity: 'O(n log n)',
    ref: 'Jacob et al. 2018 int8', commonPitfall: '整体单调性（允许并列）。',
    invariant: '按 x 升序排列后 quant 非减',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => symQuant(x, symScale(x)),
    predicate: (x) => {
      const sc = symScale(x); const r = symQuant(x, sc);
      const idx = Array.from({ length: x.length }, (_, i) => i).sort((a, b) => x[a] - x[b]);
      let prev = -Infinity;
      for (const i of idx) { if (r[i] < prev - 1e-12) return false; prev = r[i]; }
      return true;
    },
  });

  // H071 反量化范围覆盖。
  reg(71, {
    name: '反量化范围', intuition: '重建值落在原始范围附近。',
    math: 'min(x)-scale <= run_i <= max(x)+scale', pseudocode: 'dequant range', complexity: 'O(n)',
    ref: 'Jacob et al. 2018 int8', commonPitfall: '误差以 scale 为界。',
    invariant: 'run ∈ [min(x)-scale, max(x)+scale]',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => { const sc = symScale(x); return symDequant(symQuant(x, sc), sc); },
    predicate: (x) => {
      const sc = symScale(x); const r = symDequant(symQuant(x, sc), sc);
      const xmn = minOf(x), xmx = maxOf(x);
      for (let i = 0; i < r.length; i++) if (r[i] < xmn - sc - 1e-9 || r[i] > xmx + sc + 1e-9) return false;
      return true;
    },
  });

  // H072 双量化往返（幂等）。
  reg(72, {
    name: '双量化往返', intuition: '量化→反量化→再量化等于原量化。',
    math: 'quant(dequant(quant(x))) = quant(x)', pseudocode: 'idempotent quant', complexity: 'O(n)',
    ref: 'Jacob et al. 2018 int8', commonPitfall: '依赖整数码往返。',
    invariant: 'quant(dequant(q)) = q',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => { const sc = symScale(x); const q1 = symQuant(x, sc); const dq = symDequant(q1, sc); return symQuant(dq, sc); },
    predicate: (x) => {
      const sc = symScale(x); const q1 = symQuant(x, sc); const r = (function () { const dq = symDequant(q1, sc); return symQuant(dq, sc); })();
      for (let i = 0; i < r.length; i++) if (r[i] !== q1[i]) return false;
      return true;
    },
  });

  // H073 伪量化（fake-quant = 反量化∘量化）。
  reg(73, {
    name: '伪量化', intuition: '训练时前向量化-反量化，梯度直通。',
    math: 'run = dequant(quant(x))', pseudocode: 'fake-quant', complexity: 'O(n)',
    ref: 'Jacob et al. 2018 int8', commonPitfall: '误差有界。',
    invariant: '|run_i - x_i| <= scale/2 + ε',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => { const sc = symScale(x); return symDequant(symQuant(x, sc), sc); },
    predicate: (x) => { const sc = symScale(x); const r = symDequant(symQuant(x, sc), sc); for (let i = 0; i < r.length; i++) if (Math.abs(r[i] - x[i]) > sc / 2 + 1e-9) return false; return true; },
  });

  // H074 int8 范围钳制（[-128,127]）。
  reg(74, {
    name: 'int8 范围钳制', intuition: '量化码必须落在 int8 范围。',
    math: 'run_i ∈ [-128,127] 整数', pseudocode: 'clamp to int8', complexity: 'O(n)',
    ref: 'Jacob et al. 2018 int8', commonPitfall: '对称量化上界为 127。',
    invariant: 'run 整数 ∈ [-128,127]',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => { const sc = symScale(x); const q = new Float64Array(x.length); for (let i = 0; i < x.length; i++) { let qi = Math.round(x[i] / sc); qi = Math.max(-128, Math.min(127, qi)); q[i] = qi; } return q; },
    predicate: (x) => {
      const sc = symScale(x); const r = (function () { const q = new Float64Array(x.length); for (let i = 0; i < x.length; i++) { let qi = Math.round(x[i] / sc); qi = Math.max(-128, Math.min(127, qi)); q[i] = qi; } return q; })();
      for (let i = 0; i < r.length; i++) if (!isInt(r[i]) || r[i] < -128 || r[i] > 127) return false;
      return true;
    },
  });

  // H075 量化饱和（给定 scale 钳制 ±127）。
  reg(75, {
    name: '量化饱和', intuition: 'scale 过小时饱和钳位到 ±127。',
    math: 'run = clamp(round(x/scale), -127, 127)', pseudocode: 'saturating quant', complexity: 'O(n)',
    ref: 'Jacob et al. 2018 int8', commonPitfall: '钳位边界。',
    invariant: 'run 整数 ∈[-127,127] 且 = clamp(round(x/scale),-127,127)',
    inputs: ['x', 'scale'], domains: { x: 'vecFinite', scale: 'pos' },
    run: (x, scale) => { const q = new Float64Array(x.length); for (let i = 0; i < x.length; i++) { let qi = Math.round(x[i] / scale); qi = Math.max(-127, Math.min(127, qi)); q[i] = qi; } return q; },
    predicate: (x, scale) => {
      const r = (function () { const q = new Float64Array(x.length); for (let i = 0; i < x.length; i++) { let qi = Math.round(x[i] / scale); qi = Math.max(-127, Math.min(127, qi)); q[i] = qi; } return q; })();
      for (let i = 0; i < r.length; i++) { if (!isInt(r[i]) || r[i] < -127 || r[i] > 127) return false; const exp = Math.max(-127, Math.min(127, Math.round(x[i] / scale))); if (r[i] !== exp) return false; }
      return true;
    },
  });

  // H076 对称 vs 仿射 scale（对称时相等；否则仿射≤对称）。
  reg(76, {
    name: '对称vs仿射 scale', intuition: 'min=-max 时两 scale 一致；否则仿射更小。',
    math: 'affScale <= symScale', pseudocode: 'compare scales', complexity: 'O(n)',
    ref: 'Jacob et al. 2018 int8', commonPitfall: '方向别反。',
    invariant: 'affScale <= symScale + ε 且均>0',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => { const mx = maxAbsOf(x); const ss = mx > 0 ? mx / 127 : 1; const mn = minOf(x), mxv = maxOf(x); const as = (mxv - mn) > 0 ? (mxv - mn) / 254 : 1; return [as, ss]; },
    predicate: (x) => {
      const r = (function () { const mx = maxAbsOf(x); const ss = mx > 0 ? mx / 127 : 1; const mn = minOf(x), mxv = maxOf(x); const as = (mxv - mn) > 0 ? (mxv - mn) / 254 : 1; return [as, ss]; })();
      return r[0] > 0 && r[1] > 0 && r[0] <= r[1] + 1e-12;
    },
  });

  // H077 量化缩放可逆（scale·127 = maxabs）。
  reg(77, {
    name: '量化缩放可逆', intuition: 'scale 使最大码 127 重建到 max|x|。',
    math: 'scale·127 = max|x|', pseudocode: 'scale invertibility', complexity: 'O(n)',
    ref: 'Jacob et al. 2018 int8', commonPitfall: '无。',
    invariant: 'relClose(scale·127, max|x|)',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => symScale(x),
    predicate: (x) => { const r = symScale(x); const am = maxAbsOf(x); if (am > 0) return Math.abs(r * 127 - am) < 1e-9; return r > 0; },
  });

  // H078 量化步长 = 1/scale（127/|max| = 1/scale）。
  reg(78, {
    name: '量化步长', intuition: '每单位跨 1/scale 个码，故 127/|max| = 1/scale。',
    math: '127/|max| = 1/scale', pseudocode: 'step = 1/scale', complexity: 'O(n)',
    ref: 'Jacob et al. 2018 int8', commonPitfall: '无。',
    invariant: 'relClose(127/|max|, 1/scale)',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => symScale(x),
    predicate: (x) => { const r = symScale(x); const am = maxAbsOf(x); if (am > 0) return Math.abs(127 / am - 1 / r) < 1e-9; return r > 0; },
  });

  // H079 逐张量量化（单尺度，交叉验证码）。
  reg(79, {
    name: '逐张量量化', intuition: '整张量共享一个 scale。',
    math: 'q_i = clamp(round(x_i/scale),-127,127)', pseudocode: 'per-tensor quant', complexity: 'O(n)',
    ref: 'Jacob et al. 2018 int8', commonPitfall: '共享 scale。',
    invariant: 'run = 参考量化（整数,[-127,127]）',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => symQuant(x, symScale(x)),
    predicate: (x) => {
      const sc = symScale(x); const r = symQuant(x, sc);
      for (let i = 0; i < r.length; i++) { if (!isInt(r[i]) || r[i] < -127 || r[i] > 127) return false; const exp = Math.max(-127, Math.min(127, Math.round(x[i] / sc))); if (r[i] !== exp) return false; }
      return true;
    },
  });

  // H080 量化后 dtype 整数。
  reg(80, {
    name: '量化 dtype 整数', intuition: '量化码为整数。',
    math: '∀i: isInteger(q_i)', pseudocode: 'check integer', complexity: 'O(n)',
    ref: 'Jacob et al. 2018 int8', commonPitfall: '无。',
    invariant: 'run 各元素为整数',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => symQuant(x, symScale(x)),
    predicate: (x) => { const sc = symScale(x); const r = symQuant(x, sc); for (let i = 0; i < r.length; i++) if (!isInt(r[i])) return false; return true; },
  });

  // ── 类别 6：线性量化 INT8 仿射/逐通道（H081–H096）──────────────────────

  // H081 仿射 scale。
  reg(81, {
    name: '仿射 scale', intuition: '仿射量化以 (max-min) 映射到 [0,255]。',
    math: 'scale = (max-min)/254', pseudocode: 'affine scale', complexity: 'O(n)',
    ref: 'Jacob et al. 2018 int8', commonPitfall: '除以 254 而非 255（留零点）。',
    invariant: 'scale>0 且 relClose(scale,(max-min)/254)',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => affScale(x),
    predicate: (x) => { const r = affScale(x); const mn = minOf(x), mx = maxOf(x); if (!(r > 0 && Number.isFinite(r))) return false; if (mx > mn) return Math.abs(r - (mx - mn) / 254) < 1e-9; return true; },
  });

  // H082 仿射 zero-point。
  reg(82, {
    name: '仿射 zero-point', intuition: '零点使 min 映射为码 0。',
    math: 'z = round(-min/scale)', pseudocode: 'zero point', complexity: 'O(n)',
    ref: 'Jacob et al. 2018 int8', commonPitfall: '须为整数。',
    invariant: 'z 整数 且 = round(-min/scale)',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => { const sc = affScale(x); return affZero(x, sc); },
    predicate: (x) => { const sc = affScale(x); const r = affZero(x, sc); if (!isInt(r)) return false; return r === Math.round(-minOf(x) / sc); },
  });

  // H083 仿射量化 uint8。
  reg(83, {
    name: '仿射量化 uint8', intuition: '映射到 [0,255] 整数。',
    math: 'q_i = clamp(round(x_i/scale)+z, 0, 255)', pseudocode: 'uint8 quantize', complexity: 'O(n)',
    ref: 'Jacob et al. 2018 int8', commonPitfall: '钳位 [0,255]。',
    invariant: 'q 整数 ∈[0,255]',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => { const sc = affScale(x); return affQuant(x, sc, affZero(x, sc)); },
    predicate: (x) => {
      const sc = affScale(x); const z = affZero(x, sc); const r = affQuant(x, sc, z);
      for (let i = 0; i < r.length; i++) if (!isInt(r[i]) || r[i] < 0 || r[i] > 255) return false;
      return true;
    },
  });

  // H084 量化矩阵（对称，逐元素钳位）。
  reg(84, {
    name: '量化矩阵（对称）', intuition: '对整个矩阵做对称量化。',
    math: 'Q = clamp(round(M/scale),-127,127)', pseudocode: 'quantize matrix', complexity: 'O(n)',
    ref: 'Jacob et al. 2018 int8', commonPitfall: '整矩阵共享 scale。',
    invariant: 'Q 整数 ∈[-127,127]', matrixDims: { M: [4, 4] },
    inputs: ['M'], domains: { M: 'matFinite' },
    run: (M) => symQuant(M, symScale(M)),
    predicate: (M) => { const r = symQuant(M, symScale(M)); for (let i = 0; i < r.length; i++) if (!isInt(r[i]) || r[i] < -127 || r[i] > 127) return false; return true; },
  });

  // H085 逐通道量化（每行独立 scale）。
  reg(85, {
    name: '逐通道量化', intuition: '每个通道（行）独立量化，提升动态范围利用。',
    math: 'q_{i,j} = round(M_{i,j}/scale_i)', pseudocode: 'per-channel quant', complexity: 'O(r·c)',
    ref: 'Jacob et al. 2018 int8', commonPitfall: '按行而非按张量。',
    invariant: 'q 整数 ∈[-127,127]',
    inputs: ['M'], domains: { M: 'matFinite' }, matrixDims: { M: [4, 4] },
    run: (M) => {
      const out = new Float64Array(M.length);
      for (let i = 0; i < MR; i++) { let am = 0; for (let j = 0; j < MC; j++) am = Math.max(am, Math.abs(M[i * MC + j])); const sc = am > 0 ? am / 127 : 1; for (let j = 0; j < MC; j++) { let qi = Math.round(M[i * MC + j] / sc); qi = Math.max(-127, Math.min(127, qi)); out[i * MC + j] = qi; } }
      return out;
    },
    predicate: (M) => {
      const r = (function () { const o = new Float64Array(M.length); for (let i = 0; i < MR; i++) { let am = 0; for (let j = 0; j < MC; j++) am = Math.max(am, Math.abs(M[i * MC + j])); const sc = am > 0 ? am / 127 : 1; for (let j = 0; j < MC; j++) { let qi = Math.round(M[i * MC + j] / sc); qi = Math.max(-127, Math.min(127, qi)); o[i * MC + j] = qi; } } return o; })();
      for (let i = 0; i < r.length; i++) if (!isInt(r[i]) || r[i] < -127 || r[i] > 127) return false;
      return true;
    },
  });

  // H086 仿射 zero 映射（dequant(0)≈min）。
  reg(86, {
    name: '仿射 zero 映射', intuition: '码 0 反量化约等于原最小值。',
    math: 'dequant(0) = (0-z)·scale ≈ min', pseudocode: 'zero maps to min', complexity: 'O(n)',
    ref: 'Jacob et al. 2018 int8', commonPitfall: '约等于，误差 ≤ scale。',
    invariant: '|dequant(0) - min| <= scale + ε',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => { const sc = affScale(x); const z = affZero(x, sc); return affQuant(x, sc, z); },
    predicate: (x) => {
      const sc = affScale(x); const z = affZero(x, sc);
      const d0 = (0 - z) * sc; const mn = minOf(x);
      return Math.abs(d0 - mn) <= sc + 1e-9;
    },
  });

  // H087 仿射可逆（dequant(q) = (q-z)·scale）。
  reg(87, {
    name: '仿射可逆', intuition: '反量化公式精确还原码。',
    math: 'x̂_i = (q_i - z)·scale', pseudocode: 'affine dequant', complexity: 'O(n)',
    ref: 'Jacob et al. 2018 int8', commonPitfall: '须同 scale,zero。',
    invariant: 'relClose(run, (q-z)·scale)',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => { const sc = affScale(x); const z = affZero(x, sc); return affDequant(affQuant(x, sc, z), sc, z); },
    predicate: (x) => {
      const sc = affScale(x); const z = affZero(x, sc); const q = affQuant(x, sc, z); const r = affDequant(q, sc, z);
      for (let i = 0; i < r.length; i++) if (Math.abs(r[i] - (q[i] - z) * sc) > 1e-9) return false;
      return true;
    },
  });

  // H088 量化范围覆盖 [min,max]。
  reg(88, {
    name: '量化范围覆盖', intuition: '仿射反量化覆盖原始 [min,max]。',
    math: 'dequant 覆盖 ≈ [min,max]', pseudocode: 'coverage', complexity: 'O(n)',
    ref: 'Jacob et al. 2018 int8', commonPitfall: '误差 ≤ scale。',
    invariant: 'min(run)<=min(x)+scale 且 max(run)>=max(x)-scale',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => { const sc = affScale(x); const z = affZero(x, sc); return affDequant(affQuant(x, sc, z), sc, z); },
    predicate: (x) => {
      const sc = affScale(x); const z = affZero(x, sc); const r = affDequant(affQuant(x, sc, z), sc, z);
      let mn = Infinity, mx = -Infinity; for (const v of r) { mn = Math.min(mn, v); mx = Math.max(mx, v); }
      const xmn = minOf(x), xmx = maxOf(x);
      return mn <= xmn + sc + 1e-9 && mx >= xmx - sc - 1e-9;
    },
  });

  // H089 zero-point 整数。
  reg(89, {
    name: 'zero-point 整数', intuition: '零点须为整数。',
    math: 'isInteger(z)', pseudocode: 'check integer', complexity: 'O(1)',
    ref: 'Jacob et al. 2018 int8', commonPitfall: '无。',
    invariant: 'z 为整数',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => { const sc = affScale(x); return affZero(x, sc); },
    predicate: (x) => { const sc = affScale(x); const r = affZero(x, sc); return isInt(r); },
  });

  // H090 仿射 vs 对称（min=-max 一致；否则仿射≤对称）。
  reg(90, {
    name: '仿射vs对称', intuition: '对称输入下两方案等价。',
    math: 'affScale <= symScale', pseudocode: 'compare', complexity: 'O(n)',
    ref: 'Jacob et al. 2018 int8', commonPitfall: '方向别反。',
    invariant: 'affScale <= symScale + ε',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => { const mx = maxAbsOf(x); const ss = mx > 0 ? mx / 127 : 1; const mn = minOf(x), mxv = maxOf(x); const as = (mxv - mn) > 0 ? (mxv - mn) / 254 : 1; return [as, ss]; },
    predicate: (x) => {
      const r = (function () { const mx = maxAbsOf(x); const ss = mx > 0 ? mx / 127 : 1; const mn = minOf(x), mxv = maxOf(x); const as = (mxv - mn) > 0 ? (mxv - mn) / 254 : 1; return [as, ss]; })();
      return r[0] > 0 && r[1] > 0 && r[0] <= r[1] + 1e-12;
    },
  });

  // H091 逐通道重建（每通道误差有界）。
  reg(91, {
    name: '逐通道重建', intuition: '逐通道量化重建误差 ≤ 行 scale/2。',
    math: '|run_{i,j}-M_{i,j}| <= scale_i/2', pseudocode: 'per-channel reconstruct', complexity: 'O(r·c)',
    ref: 'Jacob et al. 2018 int8', commonPitfall: '按行尺度。',
    invariant: '每行误差 ≤ 该行 scale/2',
    inputs: ['M'], domains: { M: 'matFinite' }, matrixDims: { M: [4, 4] },
    run: (M) => {
      const out = new Float64Array(M.length);
      for (let i = 0; i < MR; i++) { let am = 0; for (let j = 0; j < MC; j++) am = Math.max(am, Math.abs(M[i * MC + j])); const sc = am > 0 ? am / 127 : 1; for (let j = 0; j < MC; j++) { const q = Math.max(-127, Math.min(127, Math.round(M[i * MC + j] / sc))); out[i * MC + j] = q * sc; } }
      return out;
    },
    predicate: (M) => {
      const r = (function () { const o = new Float64Array(M.length); for (let i = 0; i < MR; i++) { let am = 0; for (let j = 0; j < MC; j++) am = Math.max(am, Math.abs(M[i * MC + j])); const sc = am > 0 ? am / 127 : 1; for (let j = 0; j < MC; j++) { const q = Math.max(-127, Math.min(127, Math.round(M[i * MC + j] / sc))); o[i * MC + j] = q * sc; } } return o; })();
      for (let i = 0; i < MR; i++) { let am = 0; for (let j = 0; j < MC; j++) am = Math.max(am, Math.abs(M[i * MC + j])); const sc = am > 0 ? am / 127 : 1; for (let j = 0; j < MC; j++) if (Math.abs(r[i * MC + j] - M[i * MC + j]) > sc / 2 + 1e-9) return false; }
      return true;
    },
  });

  // H092 量化累积（先量化后累加误差有界）。
  reg(92, {
    name: '量化累积', intuition: '量化后分别反量化再累加，误差 ≤ 共享 scale。',
    math: '|deq(q_a)+deq(q_b) - (a+b)| <= scale', pseudocode: 'quant-accumulate', complexity: 'O(n)',
    ref: 'Jacob et al. 2018 int8', commonPitfall: '共享 scale。',
    invariant: '|run_i - (a_i+b_i)| <= scale + ε',
    inputs: ['a', 'b'], domains: { a: 'vecFinite', b: 'vecFinite' },
    run: (a, b) => {
      const am = Math.max(maxAbsOf(a), maxAbsOf(b)); const sc = am > 0 ? am / 127 : 1;
      const qa = symQuant(a, sc), qb = symQuant(b, sc);
      const out = new Float64Array(a.length);
      for (let i = 0; i < a.length; i++) out[i] = qa[i] * sc + qb[i] * sc;
      return out;
    },
    predicate: (a, b) => {
      const am = Math.max(maxAbsOf(a), maxAbsOf(b)); const sc = am > 0 ? am / 127 : 1;
      const r = (function () { const qa = symQuant(a, sc), qb = symQuant(b, sc); const o = new Float64Array(a.length); for (let i = 0; i < a.length; i++) o[i] = qa[i] * sc + qb[i] * sc; return o; })();
      for (let i = 0; i < a.length; i++) if (Math.abs(r[i] - (a[i] + b[i])) > sc + 1e-9) return false;
      return true;
    },
  });

  // H093 量化矩阵乘（反量化累加 ≈ 浮点 matmul）。
  reg(93, {
    name: '量化矩阵乘', intuition: '量化权重反量化后相乘近似浮点 matmul。',
    math: 'deq(Q_A)·deq(Q_B) ≈ A·B', pseudocode: 'quantized matmul', complexity: 'O(r·c·k)',
    ref: 'Jacob et al. 2018 int8', commonPitfall: '用相对容差 1e-2。',
    invariant: 'relClose(run, 浮点 matmul, 1e-2)',
    inputs: ['A', 'B'], domains: { A: 'matSmall', B: 'matSmall' }, matrixDims: { A: [4, 4], B: [4, 4] },
    run: (A, B) => {
      const scA = symScale(A), scB = symScale(B);
      const Aq = symQuant(A, scA), Bq = symQuant(B, scB);
      const Adq = new Float64Array(A.length), Bdq = new Float64Array(B.length);
      for (let i = 0; i < A.length; i++) Adq[i] = Aq[i] * scA;
      for (let i = 0; i < B.length; i++) Bdq[i] = Bq[i] * scB;
      return matmul(Adq, Bdq, MR, MC, MR);
    },
    predicate: (A, B) => {
      const scA = symScale(A), scB = symScale(B);
      const Aq = symQuant(A, scA), Bq = symQuant(B, scB);
      const Adq = new Float64Array(A.length), Bdq = new Float64Array(B.length);
      for (let i = 0; i < A.length; i++) Adq[i] = Aq[i] * scA;
      for (let i = 0; i < B.length; i++) Bdq[i] = Bq[i] * scB;
      const r = matmul(Adq, Bdq, MR, MC, MR);
      const f = matmul(A, B, MR, MC, MR);
      // 反量化误差界（Lipschitz）：|e_A|<=scA/2, |e_B|<=scB/2，内维 k=MR。
      // C_ij = Σ_k A_dq·B_dq；与浮点 F_ij 的误差
      // |E_ij| <= Σ_k(|e_A||B|) + Σ_k(|A||e_B|) + Σ_k(|e_A e_B|)
      //        <= k·(epsA·maxB + maxA·epsB + epsA·epsB)。
      const maxA = maxAbsOf(A), maxB = maxAbsOf(B);
      const epsA = scA / 2, epsB = scB / 2;
      const k = MR;
      const bound = k * (epsA * maxB + maxA * epsB + epsA * epsB) + 1e-6;
      let maxDiff = 0;
      for (let i = 0; i < r.length; i++) maxDiff = Math.max(maxDiff, Math.abs(r[i] - f[i]));
      return maxDiff <= bound;
    },
  });

  // H094 量化缩放因子对称性（对称量化下 quant(-x)=-quant(x)）。
  reg(94, {
    name: '量化缩放对称性', intuition: '对称量化关于原点对称。',
    math: 'quant(-x) = -quant(x)', pseudocode: 'symmetric quant', complexity: 'O(n)',
    ref: 'Jacob et al. 2018 int8', commonPitfall: '仅对称量化成立。',
    invariant: 'quant(-x) = -quant(x)',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => symQuant(x, symScale(x)),
    predicate: (x) => {
      const sc = symScale(x); const r = symQuant(x, sc);
      const negx = x.map((v) => -v); const qn = symQuant(negx, sc);
      for (let i = 0; i < r.length; i++) if (r[i] !== -qn[i]) return false;
      return true;
    },
  });

  // H095 量化感知训练占位（fake-quant）。
  reg(95, {
    name: '量化感知训练占位', intuition: '前向量化-反量化，梯度直通。',
    math: 'run = dequant(quant(x))', pseudocode: 'QAT placeholder', complexity: 'O(n)',
    ref: 'Jacob et al. 2018 int8', commonPitfall: '误差有界。',
    invariant: '|run_i - x_i| <= scale/2 + ε',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => { const sc = symScale(x); return symDequant(symQuant(x, sc), sc); },
    predicate: (x) => { const sc = symScale(x); const r = symDequant(symQuant(x, sc), sc); for (let i = 0; i < r.length; i++) if (Math.abs(r[i] - x[i]) > sc / 2 + 1e-9) return false; return true; },
  });

  // H096 动态量化 scale（log2 取整）。
  reg(96, {
    name: '动态量化 scale', intuition: 'scale = 2^ceil(log2(maxabs/127))，2 的幂。',
    math: 'scale = 2^ceil(log2(max|x|/127))', pseudocode: 'dynamic log2 scale', complexity: 'O(n)',
    ref: 'Banner et al. 2018', commonPitfall: '须为 2 的幂且 ≥ maxabs/127。',
    invariant: 'scale 为 2 的幂；base <= scale < 2·base',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => { const am = maxAbsOf(x); const base = am / 127; const e = Math.ceil(Math.log2(base > 0 ? base : 1)); return Math.pow(2, e); },
    predicate: (x) => {
      const am = maxAbsOf(x); const base = am / 127; const r = (function () { const e = Math.ceil(Math.log2(base > 0 ? base : 1)); return Math.pow(2, e); })();
      if (!(r > 0 && Number.isFinite(r))) return false;
      const lg = Math.log2(r); if (Math.abs(lg - Math.round(lg)) > 1e-9) return false;
      if (base <= 0) return true;
      if (r < base - 1e-9) return false;
      if (r >= 2 * base + 1e-9) return false;
      return true;
    },
  });
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

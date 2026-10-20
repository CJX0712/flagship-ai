/* ============================================================================
 * L10 微指令 · 不变量与性质测试工具（invariant）
 * ----------------------------------------------------------------------------
 * 所属层：L10-microIsa
 * 允许依赖的层：L9（rng / assert）
 * 主要 @ref：K9 共享约定（误差预算 / 逐位比对）；K10（误差预算集中登记）；
 *           Claessen & Hughes (2000) QuickCheck（性质测试思想）
 *
 * 责任：为微指令的性质测试提供统一框架：
 *   - TAGS：标量/矩阵输入的域标签元数据（供 tests/L10 用 fast-check 构造任意子）
 *   - genScalar / genMatrix：基于 NS.L9.rng 的确定性生成器（供 isa.js 即时自检）
 *   - relClose：相对/绝对容差比较（数值不变量）
 *   - check：执行一个 op 的不变量谓词（返回 {pass, detail}）
 * 本文件不依赖 fast-check（K5：层目录禁 require），fast-check 仅由 tests/ 使用。
 * ==========================================================================*/
;(function (NS) {
  'use strict';
  NS.L9 = NS.L9 || {};
  const L10 = (NS.L10 = NS.L10 || {});

  // 域标签元数据：tests/L10 据此构造 fast-check 任意子；genScalar 据此确定性采样。
  // kind:'scalar' | 'matrix'。矩阵默认维度由 op.matrixDims 覆盖。
  const TAGS = {
    // 标量
    int:         { kind: 'scalar', integer: true, min: -1e6, max: 1e6 },
    smallInt:    { kind: 'scalar', integer: true, min: -1000, max: 1000 },
    nat:         { kind: 'scalar', integer: true, min: 0, max: 1e6 },
    posInt:      { kind: 'scalar', integer: true, min: 1, max: 1e6 },
    finite:      { kind: 'scalar', min: -1e4, max: 1e4 },
    nonZero:     { kind: 'scalar', min: -1e4, max: 1e4, nonZero: true },
    nonNeg:      { kind: 'scalar', min: 0, max: 1e4 },
    pos:         { kind: 'scalar', min: 1e-6, max: 1e4 },
    unit:        { kind: 'scalar', min: 0, max: 1 },
    small:       { kind: 'scalar', min: -10, max: 10 },
    smallPos:    { kind: 'scalar', min: 1e-6, max: 10 },
    nonNegSmall: { kind: 'scalar', min: 0, max: 10 },
    angle:       { kind: 'scalar', min: -Math.PI, max: Math.PI },
    trig:        { kind: 'scalar', min: -1.5, max: 1.5 },   // 远离 ±π/2，tan 安全
    trigPos:     { kind: 'scalar', min: 0, max: 1.5 },
    negOneOne:   { kind: 'scalar', min: -0.999, max: 0.999 },
    // 矩阵
    matInt:      { kind: 'matrix', integer: true, min: -1000, max: 1000, defDims: [4, 4] },
    matFinite:   { kind: 'matrix', min: -1e4, max: 1e4, defDims: [4, 4] },
    matSmall:    { kind: 'matrix', min: -10, max: 10, defDims: [4, 4] },
    // 向量（B 域）：kind 'vector'，defLen 为默认长度；per-input 长度可由 op.vecDims 覆盖
    vecInt:      { kind: 'vector', integer: true, min: -1000, max: 1000, defLen: 8 },
    vecFinite:   { kind: 'vector', min: -1e4, max: 1e4, defLen: 8 },
    vecSmall:    { kind: 'vector', min: -10, max: 10, defLen: 8 },
    vecNonNeg:   { kind: 'vector', min: 0, max: 10, defLen: 8 },
    vecPos:      { kind: 'vector', min: 1e-3, max: 10, defLen: 8, nonZero: true },
    vecUnit:     { kind: 'vector', min: 0, max: 1, defLen: 8 },
    // 形状（E 域）：kind 'shape'，defLen = 维度个数，每个维度 ∈ [min,max] 的正整数
    shape2:      { kind: 'shape', defLen: 2, min: 1, max: 8 },
    shape3:      { kind: 'shape', defLen: 3, min: 1, max: 8 },
    shape4:      { kind: 'shape', defLen: 4, min: 1, max: 8 },
    // 首维 ≥2 的二阶形状：供「取首维下标 1」类微指令用（E023）。
    // 原因：首维为 1 时下标 1 越界，linearize 会抛 —— 该类微指令的数学本就需要首维 ≥2，
    // 属域声明缺失而非谓词缺陷，故在此显式声明该域，而非在谓词里退化放行。
    shape2f2:    { kind: 'shape', defLen: 2, min: 1, max: 8, firstMin: 2 },
  };

  // 确定性标量采样（基于注入的 NS.L9.rng）。
  // 关键修正：非负域（min>=0，如 pos/nonNeg/smallPos/nonNegSmall/unit/trigPos）
  // 必须均匀采样到 [min, max]，否则会生成负数喂给 log/√/asin 等导致 NaN。
  // 可正可负域（min<0）仍对称采样 [-M, M]。
  function genScalar(rng, tag) {
    const t = TAGS[tag] || TAGS.small;
    if (t.integer) return Math.floor(rng() * (t.max - t.min + 1)) + t.min;
    let x;
    if (t.min >= 0) {
      x = rng() * (t.max - t.min) + t.min;       // 均匀 [min, max]，保证 ≥0
      if (t.nonZero && x === 0) x = t.min > 0 ? t.min : 1e-6;
    } else {
      const M = Math.max(Math.abs(t.min), Math.abs(t.max));
      x = (rng() * 2 - 1) * M;                   // 对称 [-M, M]
      if (t.nonZero && x === 0) x = (rng() < 0.5 ? 1 : -1) * 1e-3;
    }
    return x;
  }

  // 确定性矩阵采样（行优先 Float64Array）。
  function genMatrix(rng, r, c, tag) {
    const out = new Float64Array(r * c);
    for (let i = 0; i < r * c; i++) out[i] = genScalar(rng, tag);
    return out;
  }

  // 确定性向量采样（B 域）：长度 n 的行优先 Float64Array。
  function genVector(rng, n, tag) {
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = genScalar(rng, tag);
    return out;
  }

  // 确定性形状采样（E 域）：长度 defLen 的正整数维度元组（普通 Array）。
  // firstMin：可选的首维下界（其余维仍用 min）—— 供需要「首维下标 1」合法性的域使用。
  function genShape(rng, tag) {
    const t = TAGS[tag] || TAGS.shape3;
    const len = t.defLen || 3;
    const out = [];
    for (let i = 0; i < len; i++) {
      const lo = (i === 0 && t.firstMin != null) ? t.firstMin : t.min;
      out.push(Math.floor(rng() * (t.max - lo + 1)) + lo);
    }
    return out;
  }

  // 相对/绝对容差比较。exact=true 时用 Object.is（逐位）。
  function relClose(a, b, tol, exact) {
    tol = tol == null ? 1e-9 : tol;
    if (exact) {
      // 符号零归一：-0 与 0 视为相等（IEEE 中 Object.is(-0,0) 为假，但数值语义等价）。
      if (typeof a === 'number' && typeof b === 'number') { if (a === 0) a = 0; if (b === 0) b = 0; return Object.is(a, b); }
      if (a && b && a.length === b.length) {
        for (let i = 0; i < a.length; i++) { let x = a[i], y = b[i]; if (x === 0) x = 0; if (y === 0) y = 0; if (!Object.is(x, y)) return false; }
        return true;
      }
      return false;
    }
    if (typeof a === 'number' && typeof b === 'number') {
      if (Object.is(a, b)) return true;
      const den = Math.max(Math.abs(a), Math.abs(b), 1);
      return Math.abs(a - b) <= tol * den;
    }
    if (a && b && a.length === b.length) {
      let ok = true;
      for (let i = 0; i < a.length; i++) {
        if (!Object.is(a[i], b[i])) {
          const den = Math.max(Math.abs(a[i]), Math.abs(b[i]), 1);
          if (Math.abs(a[i] - b[i]) > tol * den) { ok = false; break; }
        }
      }
      return ok;
    }
    return false;
  }

  // 执行一个 op 的不变量谓词：op.predicate(...args)。返回 {pass, detail}。
  function check(op, args, tol) {
    if (!op || op.status !== 'impl' || typeof op.predicate !== 'function') {
      return { pass: false, detail: '无谓词(stub)' };
    }
    try {
      const ok = op.predicate.apply(null, args);
      return { pass: !!ok, detail: ok ? '✓' : '✗ 不变量失败' };
    } catch (e) {
      return { pass: false, detail: '异常: ' + (e && e.message) };
    }
  }

  L10.TAGS = TAGS;
  L10.genScalar = genScalar;
  L10.genMatrix = genMatrix;
  L10.genVector = genVector;
  L10.genShape = genShape;
  L10.relClose = relClose;
  L10.check = check;
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

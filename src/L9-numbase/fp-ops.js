/* L9 · 数值基础层 / 浮点原语与稳定性 —— 依赖：无
 * 误差补偿、稳定化、特殊函数；上层（L8/L7/L4）确定性调用源。
 * @ref https://github.com/stdlib-js/stdlib（误差界对齐）· NumPy 浮点语义
 */
;(function (NS) {
  'use strict';
  // 所属层：L9 · 允许依赖：无 · 主要 @ref：stdlib 误差界 / Kahan 1972

  // 两数和的精确分解：s 为舍入和，e 为舍入误差（Dekker/two-sum）
  function twoSum(a, b) {
    var s = a + b;
    var a1 = s - b, b1 = s - a1;
    var e = (a - a1) + (b - b1);
    return { s: s, e: e };
  }

  // 快速两数和（要求 |a| >= |b|）：一次减法得误差
  function fastTwoSum(a, b) {
    var s = a + b;
    var e = b - (s - a);
    return { s: s, e: e };
  }

  // Kahan 累加单步：返回 { sum, comp }（补偿部分存 comp）
  function kahanStep(sum, comp, x) {
    var y = x - comp;
    var t = sum + y;
    var c = (t - sum) - y;
    return { sum: t, comp: c };
  }

  // Neumaier 累加单步（比 Kahan 更稳，处理大数吃小数；comp 为累积量）
  function neumaierStep(sum, comp, x) {
    var t = sum + x;
    var c = comp + (Math.abs(sum) >= Math.abs(x) ? (sum - t) + x : (x - t) + sum);
    return { sum: t, comp: c };
  }

  // 两浮点数的 ULP 距离（双字运算避免 64 位 pattern 的 double 精度损失）
  function ulpDiff(a, b) {
    if (a === b) return 0;
    if (!isFinite(a) || !isFinite(b)) return Infinity;
    var dv = new DataView(new ArrayBuffer(8));
    dv.setFloat64(0, a); var ha = dv.getUint32(0), la = dv.getUint32(4);
    dv.setFloat64(0, b); var hb = dv.getUint32(0), lb = dv.getUint32(4);
    var sa = (ha & 0x80000000) !== 0, sb = (hb & 0x80000000) !== 0;
    if (sa === sb) {
      var dh = ha - hb, dl = la - lb;
      if (dh < 0 || (dh === 0 && dl < 0)) { dh = -dh; dl = -dl; }
      if (dl < 0) { dh -= 1; dl += 4294967296; }
      return dh * 4294967296 + dl;
    }
    var d1h = 0x80000000 - hb, d1l = -lb;
    if (d1l < 0) { d1h -= 1; d1l += 4294967296; }
    var rh = d1h + ha, rl = d1l + la;
    if (rl >= 4294967296) { rh += 1; rl -= 4294967296; }
    return rh * 4294967296 + rl;
  }

  // frexp：x = m * 2^e，|m| in [0.5,1)（用位操作精确；DataView 默认大端，高字在偏移 0）
  function frexp(x) {
    if (x === 0) return { m: 0, e: 0 };
    if (!isFinite(x)) return { m: x, e: 0 };
    var dv = new DataView(new ArrayBuffer(8));
    dv.setFloat64(0, x);
    var hi = dv.getUint32(0), lo = dv.getUint32(4);
    var sign = hi & 0x80000000;
    var exp = (hi >>> 20) & 0x7FF;
    var mant = (hi & 0xFFFFF) * 4294967296 + lo; // 52 位尾数：必须用数值运算（32 位位运算会截断）
    var m;
    if (exp === 0) { // 非规格化：数值左移（禁 <<= ，其按 int32 截断会死循环）
      exp = 1;
      while (mant < 4503599627370496) { mant *= 2; exp--; }
      m = mant / 9007199254740992;
    } else {
      m = (mant + 4503599627370496) / 9007199254740992; // 补隐含位 → [0.5,1)
    }
    var eout = exp - 1022;
    if (sign) m = -m;
    return { m: m, e: eout };
  }

  // ldexp：m * 2^e（frexp 逆）
  function ldexp(m, e) {
    return m * Math.pow(2, e);
  }

  // 是否规约浮点（非 0/Inf/NaN/非规格）
  function isNormal(x) {
    return x !== 0 && isFinite(x) && Math.abs(x) >= 2.2250738585072014e-308;
  }

  // 四舍五入到 float32（经 Float32Array），用于量化/精度对照
  function float32Round(x) {
    var f = new Float32Array(1); f[0] = x; return f[0];
  }

  // 安全指数：避免大正指数溢出为 Inf
  function safeExp(x) { return x > 40 ? Math.exp(40) : Math.exp(x); }

  // 指数减一（精确小值）
  function expm1(x) { return Math.expm1 ? Math.expm1(x) : (Math.exp(x) - 1); }

  // 对数加一（精确小值）
  function log1p(x) { return Math.log1p ? Math.log1p(x) : Math.log(1 + x); }

  // 两数和的对数（数值稳定）：log(exp a + exp b)
  function logsumexpPair(a, b) {
    var m = a > b ? a : b;
    return m + Math.log(Math.exp(a - m) + Math.exp(b - m));
  }

  // 稳定 sigmoid：大正/负输入不溢出
  function sgSigmoid(x) {
    if (x >= 0) { var e = Math.exp(-x); return 1 / (1 + e); }
    var e2 = Math.exp(x); return e2 / (1 + e2);
  }

  // Softplus：log(1 + exp x)，负输入稳定（避免 1+0 下溢）
  function softplus(x) { return x > 30 ? x : Math.log1p(Math.exp(x)); }

  // 稳定 tanh（避免大输入 exp 溢出）
  function tanhStable(x) {
    if (x > 20) return 1; if (x < -20) return -1;
    var e = Math.exp(2 * x); return (e - 1) / (e + 1);
  }

  // 欧氏范数 sqrt(a^2+b^2)，避免中间溢出
  function hypot(a, b) { return Math.hypot(a, b); }

  // 立方根
  function cbrt(x) { return Math.cbrt(x); }

  // 钳位
  function clip(x, lo, hi) { return x < lo ? lo : (x > hi ? hi : x); }

  // 符号
  function sign(x) { return x > 0 ? 1 : (x < 0 ? -1 : 0); }

  // 安全倒数（近 0 返回大数而非 Inf，确定性）
  function safeInv(x) { return x === 0 ? 1e308 : 1 / x; }

  // ---- 自检（供 probe 调用）----
  function runSelfTests(seed) {
    var rng = NS.L10 ? NS.L10.rng32((seed || 1) >>> 0) : (function () { var s = 12345; return function () { s = (s + 0x6D2B79F5) | 0; var t = Math.imul(s ^ (s >>> 15), 1 | s); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; })();
    var out = [];
    function approx(a, b, tol) { return Math.abs(a - b) <= tol; }
    var ts = twoSum(1, 1e-16), ok = (ts.s + ts.e) === 1 + 1e-16;
    out.push({ name: 'twoSum', pass: ok, detail: 's+e===a+b' });
    var fts = fastTwoSum(1, 1e-16); out.push({ name: 'fastTwoSum', pass: approx(fts.s + fts.e, 1 + 1e-16, 0), detail: '|a|>=|b| 下精确' });
    var ks = kahanStep(1e16, 0, 1), ok2 = ks.sum === 1e16 + 1; out.push({ name: 'kahanStep', pass: ok2, detail: '大数+小数不丢' });
    var fe = frexp(3.5), ok3 = approx(ldexp(fe.m, fe.e), 3.5, 1e-12); out.push({ name: 'frexp/ldexp', pass: ok3, detail: '往返一致' });
    var ok4 = ulpDiff(1, 1 + 2.220446049250313e-16) === 1; out.push({ name: 'ulpDiff', pass: ok4, detail: '相邻浮点 ulp=1' });
    var ok5 = true; for (var i = 0; i < 50; i++) { var x = (rng() - 0.5) * 100; if (!approx(float32Round(x), x, 1e-3)) { ok5 = false; break; } } out.push({ name: 'float32Round', pass: ok5, detail: 'f32 精度界' });
    var ok6 = safeExp(100) === Math.exp(40) && safeExp(-100) === Math.exp(-100); out.push({ name: 'safeExp', pass: ok6, detail: '不溢出' });
    var ok7 = approx(expm1(1e-8), 1e-8, 1e-14) && approx(log1p(1e-8), 1e-8, 1e-14); out.push({ name: 'expm1/log1p', pass: ok7, detail: '小值精确' });
    var ok8 = approx(logsumexpPair(0, -100), Math.log(1 + Math.exp(-100)), 1e-12); out.push({ name: 'logsumexpPair', pass: ok8, detail: '数值稳定' });
    var ok9 = sgSigmoid(50) > 0.999 && sgSigmoid(-50) < 0.001 && approx(sgSigmoid(0), 0.5, 1e-15); out.push({ name: 'sgSigmoid', pass: ok9, detail: '稳定且过 0.5' });
    var ok10 = softplus(40) > 39.9 && approx(softplus(-40), Math.exp(-40), 1e-40); out.push({ name: 'softplus', pass: ok10, detail: '稳定' });
    var ok11 = tanhStable(40) === 1 && approx(tanhStable(0), 0, 1e-15); out.push({ name: 'tanhStable', pass: ok11, detail: '不溢出' });
    var ok12 = approx(hypot(3, 4), 5, 1e-12); out.push({ name: 'hypot', pass: ok12, detail: '3-4-5' });
    var ok13 = approx(cbrt(8), 2, 1e-12); out.push({ name: 'cbrt', pass: ok13, detail: 'cbrt(8)=2' });
    var ok14 = isNormal(1) && !isNormal(0) && !isNormal(Infinity); out.push({ name: 'isNormal', pass: ok14, detail: '边界正确' });
    return out;
  }

  NS.L9 = Object.assign(NS.L9 || {}, {
    twoSum: twoSum, fastTwoSum: fastTwoSum, kahanStep: kahanStep, neumaierStep: neumaierStep,
    ulpDiff: ulpDiff, frexp: frexp, ldexp: ldexp, isNormal: isNormal, float32Round: float32Round,
    rsqrt: function (x) { return 1 / Math.sqrt(x); }, safeExp: safeExp, expm1: expm1, log1p: log1p,
    logsumexpPair: logsumexpPair, sgSigmoid: sgSigmoid, softplus: softplus, tanhStable: tanhStable,
    hypot: hypot, cbrt: cbrt, clip: clip, sign: sign, safeInv: safeInv, runSelfTests: runSelfTests
  });
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

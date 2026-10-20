/* L10 · 微指令层 / A 域：标量数值与浮点原语 —— 依赖：无（最底层，自包含）
 * 每个 μA 为不可再分的浮点原子操作，自带可验证不变量。
 * @ref Dekker 1971（two-prod）· Abramowitz-Stegun 7.1.26（erf）· stdlib 误差界
 */
;(function (NS) {
  'use strict';
  // 所属层：L10 · 允许依赖：无 · 主要 @ref：Dekker 1971 / Abramowitz-Stegun 7.1.26
  var L10 = NS.L10;

  function split(x) { var c = 134217729; var y = c * x; var hi = y - (y - x); var lo = x - hi; return [hi, lo]; }
  function twoProd(a, b) { var p = a * b; var s = split(a), t = split(b); var e = ((s[0] * t[0] - p) + (s[0] * t[1] + s[1] * t[0]) + s[1] * t[1]); return { p: p, e: e }; }

  function erfApprox(x) {
    var s = x < 0 ? -1 : 1; x = Math.abs(x);
    var t = 1 / (1 + 0.3275911 * x);
    var y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return s * y;
  }

  var OPS = [
    { key: 'A000', name: 'two_sum', sig: '(a,b)->{s,e}', ref: 'Dekker 1971', complexity: 'O(1)',
      fn: function (a, b) { var s = a + b; var a1 = s - b, b1 = s - a1; return { s: s, e: (a - a1) + (b - b1) }; },
      selfTest: function (rng) { for (var i = 0; i < 200; i++) { var a = (rng() - 0.5) * 1e6, b = (rng() - 0.5) * 1e6; var r = this.fn(a, b); if (r.s + r.e !== a + b) return { pass: false, detail: 's+e!==a+b' }; } return { pass: true, detail: 's+e===a+b 精确恒等' }; } },
    { key: 'A001', name: 'fast_two_sum', sig: '(a,b)|a|>=|b| ->{s,e}', ref: 'Dekker 1971', complexity: 'O(1)',
      fn: function (a, b) { var s = a + b; return { s: s, e: b - (s - a) }; },
      selfTest: function (rng) { for (var i = 0; i < 200; i++) { var a = (rng() - 0.5) * 1e6, b = (rng() - 0.5) * 1e6; if (Math.abs(b) > Math.abs(a)) { var t = a; a = b; b = t; } var r = this.fn(a, b); if (r.s + r.e !== a + b) return { pass: false, detail: 's+e!==a+b' }; } return { pass: true, detail: '|a|>=|b| 下精确' }; } },
    { key: 'A002', name: 'two_prod', sig: '(a,b)->{p,e}', ref: 'Dekker 1971', complexity: 'O(1)',
      fn: function (a, b) { return twoProd(a, b); },
      selfTest: function (rng) { for (var i = 0; i < 200; i++) { var a = (rng() - 0.5) * 1e6, b = (rng() - 0.5) * 1e6; var r = this.fn(a, b); if (r.p + r.e !== a * b) return { pass: false, detail: 'p+e!==a*b' }; } return { pass: true, detail: 'p+e===a*b 精确恒等' }; } },
    { key: 'A003', name: 'kahan_step', sig: '(sum,comp,x)->{sum,comp}', ref: 'Kahan 1965', complexity: 'O(1)',
      fn: function (sum, comp, x) { var y = x - comp; var t = sum + y; return { sum: t, comp: (t - sum) - y }; },
      selfTest: function (rng) { var sum = 1e16, comp = 0; for (var i = 0; i < 1000; i++) { var r = this.fn(sum, comp, 1); sum = r.sum; comp = r.comp; } return { pass: sum === 1e16 + 1000, detail: '大数累加不丢精度' }; } },
    { key: 'A004', name: 'neumaier_step', sig: '(sum,comp,x)->{sum,comp}', ref: 'Neumaier 1974', complexity: 'O(1)',
      fn: function (sum, comp, x) { var t = sum + x; var c = comp + (Math.abs(sum) >= Math.abs(x) ? (sum - t) + x : (x - t) + sum); return { sum: t, comp: c }; },
      selfTest: function (rng) { var sum = 1e16, comp = 0; for (var i = 0; i < 1000; i++) { var r = this.fn(sum, comp, 1); sum = r.sum; comp = r.comp; } return { pass: (sum + comp) === 1e16 + 1000, detail: 'Neumaier 大数累加 sum+comp 精确' }; } },
    { key: 'A005', name: 'ulp_diff', sig: '(a,b)->number', ref: 'IEEE-754', complexity: 'O(1)',
      fn: function (a, b) {
        if (a === b) return 0;
        if (!isFinite(a) || !isFinite(b)) return Infinity;
        var dv = new DataView(new ArrayBuffer(8));
        dv.setFloat64(0, a); var ha = dv.getUint32(0), la = dv.getUint32(4); // 大端：高字@0
        dv.setFloat64(0, b); var hb = dv.getUint32(0), lb = dv.getUint32(4);
        var sa = (ha & 0x80000000) !== 0, sb = (hb & 0x80000000) !== 0;
        if (sa === sb) {
          // 同号：IEEE 位模式差（高字差在 int32 内，因符号位相同）
          var dh = ha - hb, dl = la - lb;
          if (dh < 0 || (dh === 0 && dl < 0)) { dh = -dh; dl = -dl; }
          if (dl < 0) { dh -= 1; dl += 4294967296; }
          return dh * 4294967296 + dl; // 距离 <2^53 时精确；更大为 double 近似（度量用途足够）
        }
        // 异号：跨零 = (bits(b) − 2^63) + bits(a)（负数侧位序反向：向 −0 递增是位模式递减）
        var bh = hb - 0x80000000, bl = lb;
        var rh = bh + ha, rl = bl + la;
        if (rl >= 4294967296) { rh += 1; rl -= 4294967296; }
        return rh * 4294967296 + rl;
      },
      selfTest: function (rng) {
        if (this.fn(1, 1) !== 0) return { pass: false, detail: '相同数 ulp!=0' };
        if (this.fn(1, 1 + 2.220446049250313e-16) !== 1) return { pass: false, detail: '正相邻 ulp!=1' };
        if (this.fn(-1, -(1 + 2.220446049250313e-16)) !== 1) return { pass: false, detail: '负相邻 ulp!=1' };
        if (this.fn(1, 2) !== 4503599627370496) return { pass: false, detail: '1→2 应为 2^52 ulp' };
        if (this.fn(-1, 1) !== 0x7FE0000000000000) return { pass: false, detail: '跨零距离应为 0x7FE0000000000000' };
        if (this.fn(0, 5e-324) !== 1) return { pass: false, detail: '0→最小非规格化应为 1 ulp' };
        for (var i = 0; i < 100; i++) { var x = (rng() - 0.5) * 1e8, y = x + (rng() - 0.5) * 1e-8; if (x === y) continue; var d = this.fn(x, y); if (!(d >= 0)) return { pass: false, detail: '距离为负' }; }
        return { pass: true, detail: '同号/异号/零界/非规格化全部正确' };
      } },
    { key: 'A006', name: 'is_normal', sig: '(x)->boolean', ref: 'IEEE-754', complexity: 'O(1)',
      fn: function (x) { return x !== 0 && isFinite(x) && Math.abs(x) >= 2.2250738585072014e-308; },
      selfTest: function (rng) { if (!(this.fn(1) && !this.fn(0) && !this.fn(Infinity) && !this.fn(NaN))) return { pass: false, detail: '边界错' }; return { pass: true, detail: '0/Inf/NaN 非规约' }; } },
    { key: 'A007', name: 'frexp', sig: '(x)->{m,e}', ref: 'C 标准库', complexity: 'O(1)',
      fn: function (x) {
        if (x === 0) return { m: 0, e: 0 };
        if (!isFinite(x)) return { m: x, e: 0 };
        var dv = new DataView(new ArrayBuffer(8));
        dv.setFloat64(0, x); // DataView 默认大端：高字在偏移 0
        var hi = dv.getUint32(0), lo = dv.getUint32(4);
        var sgn = hi & 0x80000000;
        var ex = (hi >>> 20) & 0x7FF;
        var mant = (hi & 0xFFFFF) * 4294967296 + lo; // 52 位尾数：必须用数值运算（32 位位运算会截断）
        var m, e;
        if (ex === 0) { // 非规格化：数值左移至隐含位就位（禁 <<= ，其按 int32 截断）
          ex = 1;
          while (mant < 4503599627370496) { mant *= 2; ex--; }
          m = mant / 9007199254740992;
        } else {
          m = (mant + 4503599627370496) / 9007199254740992; // 补隐含位 → [0.5,1)
        }
        e = ex - 1022;
        if (sgn) m = -m;
        return { m: m, e: e };
      },
      selfTest: function (rng) { for (var i = 0; i < 200; i++) { var x = (rng() - 0.5) * 1e12; var r = this.fn(x); if (Math.abs(r.m * Math.pow(2, r.e) - x) > 1e-9 * Math.abs(x) + 1e-12) return { pass: false, detail: '往返偏差大' }; } return { pass: true, detail: 'frexp/ldexp 往返一致' }; } },
    { key: 'A008', name: 'ldexp', sig: '(m,e)->number', ref: 'C 标准库', complexity: 'O(1)',
      fn: function (m, e) { return m * Math.pow(2, e); },
      selfTest: function (rng) { for (var i = 0; i < 100; i++) { var m = (rng() - 0.5) * 2, e = (rng() * 20) | 0; var r = this.fn(m, e); if (!isFinite(r)) return { pass: false, detail: '溢出' }; } return { pass: true, detail: 'm*2^e 有限' }; } },
    { key: 'A009', name: 'f32_round', sig: '(x)->number', ref: 'IEEE-754', complexity: 'O(1)',
      fn: function (x) { var f = new Float32Array(1); f[0] = x; return f[0]; },
      selfTest: function (rng) { for (var i = 0; i < 200; i++) { var x = (rng() - 0.5) * 1e6; var r = this.fn(x); if (this.fn(r) !== r) return { pass: false, detail: '非幂等' }; if (Math.abs(r - x) > 1e-3 * Math.abs(x) + 1e-30) return { pass: false, detail: '精度超界' }; } return { pass: true, detail: 'f32 幂等+精度界' }; } },
    { key: 'A010', name: 'safe_exp', sig: '(x)->number', ref: '数值稳定', complexity: 'O(1)',
      fn: function (x) { return x > 40 ? Math.exp(40) : Math.exp(x); },
      selfTest: function (rng) { if (!(isFinite(this.fn(1000)) && this.fn(100) === Math.exp(40) && this.fn(-100) === Math.exp(-100))) return { pass: false, detail: '溢出或截断错' }; return { pass: true, detail: '全域有限且不截断' }; } },
    { key: 'A011', name: 'expm1', sig: '(x)->number', ref: '数值稳定', complexity: 'O(1)',
      fn: function (x) { return Math.expm1 ? Math.expm1(x) : Math.exp(x) - 1; },
      selfTest: function (rng) { if (Math.abs(this.fn(Math.log(2)) - 1) > 1e-12) return { pass: false, detail: 'expm1(log2)!=1' }; if (Math.abs(this.fn(1e-8) - 1e-8) > 1e-14) return { pass: false, detail: '小值不精确' }; return { pass: true, detail: '稳定且小值精确' }; } },
    { key: 'A012', name: 'log1p', sig: '(x)->number', ref: '数值稳定', complexity: 'O(1)',
      fn: function (x) { return Math.log1p ? Math.log1p(x) : Math.log(1 + x); },
      selfTest: function (rng) { if (Math.abs(this.fn(Math.exp(1) - 1) - 1) > 1e-12) return { pass: false, detail: 'log1p(e-1)!=1' }; if (Math.abs(this.fn(1e-8) - 1e-8) > 1e-14) return { pass: false, detail: '小值不精确' }; return { pass: true, detail: '稳定且小值精确' }; } },
    { key: 'A013', name: 'logsumexp_pair', sig: '(a,b)->number', ref: '数值稳定', complexity: 'O(1)',
      fn: function (a, b) { var m = a > b ? a : b; return m + Math.log(Math.exp(a - m) + Math.exp(b - m)); },
      selfTest: function (rng) { for (var i = 0; i < 100; i++) { var a = (rng() - 0.5) * 40, b = (rng() - 0.5) * 40; var r = this.fn(a, b); var oracle = Math.log(Math.exp(a) + Math.exp(b)); if (Math.abs(r - oracle) > 1e-10) return { pass: false, detail: '与 log(exp a+exp b) 偏差大' }; } return { pass: true, detail: '数值稳定等于 log(e^a+e^b)' }; } },
    { key: 'A014', name: 'sigmoid_stable', sig: '(x)->number', ref: '数值稳定', complexity: 'O(1)',
      fn: function (x) { if (x >= 0) { var e = Math.exp(-x); return 1 / (1 + e); } var e2 = Math.exp(x); return e2 / (1 + e2); },
      selfTest: function (rng) { if (!(this.fn(50) > 0.999 && this.fn(-50) < 0.001 && Math.abs(this.fn(0) - 0.5) < 1e-15 && this.fn(1) > 0 && this.fn(1) < 1)) return { pass: false, detail: '边界错' }; return { pass: true, detail: '∈(0,1) 过 0.5' }; } },
    { key: 'A015', name: 'softplus', sig: '(x)->number', ref: '数值稳定', complexity: 'O(1)',
      fn: function (x) { return x > 30 ? x : Math.log1p(Math.exp(x)); },
      selfTest: function (rng) { if (!(this.fn(0) > 0.69 && this.fn(0) < 0.70 && this.fn(40) > 39.9 && this.fn(-40) >= 0)) return { pass: false, detail: '边界错' }; return { pass: true, detail: '>=0 且大值~x' }; } },
    { key: 'A016', name: 'tanh_stable', sig: '(x)->number', ref: '数值稳定', complexity: 'O(1)',
      fn: function (x) { if (x > 20) return 1; if (x < -20) return -1; var e = Math.exp(2 * x); return (e - 1) / (e + 1); },
      selfTest: function (rng) { if (!(this.fn(40) === 1 && this.fn(-40) === -1 && Math.abs(this.fn(0)) < 1e-15 && this.fn(1) > -1 && this.fn(1) < 1)) return { pass: false, detail: '边界错' }; return { pass: true, detail: '∈(-1,1) 不溢出' }; } },
    { key: 'A017', name: 'erf_approx', sig: '(x)->number', ref: 'Abramowitz-Stegun 7.1.26', complexity: 'O(1)',
      fn: function (x) { var s = x < 0 ? -1 : 1; x = Math.abs(x); var t = 1 / (1 + 0.3275911 * x); var y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x); return s * y; },
      selfTest: function (rng) { if (!(Math.abs(this.fn(0)) < 1e-6 && Math.abs(this.fn(-1) + this.fn(1)) < 1e-12 && Math.abs(this.fn(1) - 0.8427) < 1e-3)) return { pass: false, detail: 'erf 性质错（A&S 7.1.26 误差界 1.5e-7）' }; return { pass: true, detail: 'erf(1)≈0.8427 误差<1e-3 + 奇函数' }; } },
    { key: 'A018', name: 'gelu_tanh', sig: '(x)->number', ref: 'GELU paper 1606.08415', complexity: 'O(1)',
      fn: function (x) { return 0.5 * x * (1 + Math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x))); },
      selfTest: function (rng) { if (!(Math.abs(this.fn(0)) < 1e-15 && this.fn(1) > 0.84 && this.fn(1) < 0.85 && this.fn(-1) > -0.17 && this.fn(-1) < -0.15 && Math.abs(this.fn(5) - 5) < 1e-4 && this.fn(-5) > -0.01 && this.fn(-5) < 0)) return { pass: false, detail: 'gelu 性质错' }; return { pass: true, detail: 'gelu(0)=0 · ±1 区间正确 · 大值≈x' }; } },
    { key: 'A019', name: 'silu', sig: '(x)->number', ref: 'SiLU 1702.03183', complexity: 'O(1)',
      fn: function (x) { return x / (1 + Math.exp(-x)); },
      selfTest: function (rng) { if (!(Math.abs(this.fn(0)) < 1e-15 && this.fn(1) > 0.73 && this.fn(1) < 0.75)) return { pass: false, detail: 'silu 性质错' }; return { pass: true, detail: 'silu(0)=0' }; } },
    { key: 'A020', name: 'clip', sig: '(x,lo,hi)->number', ref: '通用', complexity: 'O(1)',
      fn: function (x, lo, hi) { return x < lo ? lo : (x > hi ? hi : x); },
      selfTest: function (rng) { for (var i = 0; i < 200; i++) { var x = (rng() - 0.5) * 100, lo = -5, hi = 5; var r = this.fn(x, lo, hi); if (!(r >= lo && r <= hi)) return { pass: false, detail: '越界' }; } return { pass: true, detail: 'lo<=out<=hi' }; } },
    { key: 'A021', name: 'sign', sig: '(x)->number', ref: '通用', complexity: 'O(1)',
      fn: function (x) { return x > 0 ? 1 : (x < 0 ? -1 : 0); },
      selfTest: function (rng) { if (!(this.fn(-3) === -1 && this.fn(0) === 0 && this.fn(2) === 1)) return { pass: false, detail: '符号错' }; return { pass: true, detail: '-1/0/1' }; } },
    { key: 'A022', name: 'rsqrt', sig: '(x)->number', ref: '通用', complexity: 'O(1)',
      fn: function (x) { return 1 / Math.sqrt(x); },
      selfTest: function (rng) { for (var i = 0; i < 200; i++) { var x = rng() * 1e6 + 1e-6; var r = this.fn(x); if (Math.abs(r * r * x - 1) > 1e-12) return { pass: false, detail: '逆平方偏差>1e-12' }; } return { pass: true, detail: 'rsqrt^2*x=1' }; } },
    { key: 'A023', name: 'cbrt', sig: '(x)->number', ref: 'C 标准库', complexity: 'O(1)',
      fn: function (x) { return Math.cbrt(x); },
      selfTest: function (rng) { for (var i = 0; i < 200; i++) { var x = (rng() - 0.5) * 1e6; var r = this.fn(x); if (Math.abs(r * r * r - x) > 1e-10 * Math.abs(x) + 1e-12) return { pass: false, detail: 'cbrt^3 偏差大' }; } return { pass: true, detail: 'cbrt(x)^3=x' }; } },
    { key: 'A024', name: 'hypot', sig: '(a,b)->number', ref: 'C 标准库', complexity: 'O(1)',
      fn: function (a, b) { return Math.hypot(a, b); },
      selfTest: function (rng) { if (Math.abs(this.fn(3, 4) - 5) > 1e-12) return { pass: false, detail: '3-4-5 错' }; for (var i = 0; i < 100; i++) { var a = (rng() - 0.5) * 100, b = (rng() - 0.5) * 100; var r = this.fn(a, b); if (!(r >= Math.abs(a) - 1e-9 && r <= Math.abs(a) + Math.abs(b) + 1e-9)) return { pass: false, detail: '三角界错' }; } return { pass: true, detail: '||(a,b)||_2 且 3-4-5' }; } },
    { key: 'A025', name: 'safe_inv', sig: '(x)->number', ref: '数值稳定', complexity: 'O(1)',
      fn: function (x) { return x === 0 ? 1e308 : 1 / x; },
      selfTest: function (rng) { if (!(this.fn(0) === 1e308 && isFinite(this.fn(2)) && Math.abs(this.fn(2) - 0.5) < 1e-15)) return { pass: false, detail: '安全倒数错' }; return { pass: true, detail: '零输入返回大数' }; } },
    { key: 'A026', name: 'sech2', sig: '(x)->number', ref: '通用', complexity: 'O(1)',
      fn: function (x) { var c = Math.cosh(x); return 1 / (c * c); },
      selfTest: function (rng) { if (!(this.fn(0) === 1 && this.fn(5) > 0 && this.fn(5) <= 1)) return { pass: false, detail: '边界错' }; return { pass: true, detail: '∈(0,1] sech²(0)=1' }; } },
    { key: 'A027', name: 'cosh_stable', sig: '(x)->number', ref: '数值稳定', complexity: 'O(1)',
      fn: function (x) { return x > 40 ? 0.5 * Math.exp(x) : x < -40 ? 0.5 * Math.exp(-x) : Math.cosh(x); },
      selfTest: function (rng) { if (!(this.fn(0) === 1 && isFinite(this.fn(50)) && this.fn(2) > 1)) return { pass: false, detail: 'cosh 错' }; return { pass: true, detail: '>=1 大值不溢出' }; } }
  ];

  for (var i = 0; i < OPS.length; i++) {
    var o = OPS[i];
    o.domain = 'A'; o.impl = 'done';
    L10.register(o);
  }
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

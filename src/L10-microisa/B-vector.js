/* L10 · 微指令层 / B 域：向量归约与内存原语 —— 依赖：无（最底层，自包含）
 * 不变量范式：顺序无关性（补偿/分治求和 vs 朴素求和偏差有硬界）、交换律
 * （dot(a,b) === dot(b,a) 逐位）、结合律（cumsum 前缀关系逐位）。
 * @ref Kahan 1965 · Neumaier 1974 · BLAS 语义（axpy/scal/dot/nrm2）
 */
;(function (NS) {
  'use strict';
  // 所属层：L10 · 允许依赖：无 · 主要 @ref：Kahan 1965 / BLAS
  var L10 = NS.L10;

  function naiveSum(x) { var s = 0; for (var i = 0; i < x.length; i++) s += x[i]; return s; }
  function randArr(rng, n, scale) {
    var a = new Float64Array(n), sc = scale || 2;
    for (var i = 0; i < n; i++) a[i] = (rng() - 0.5) * sc;
    return a;
  }
  function approx(a, b, tol) { return Math.abs(a - b) <= tol; }

  var OPS = [
    { key: 'B000', name: 'vadd', sig: '(x,y)->z', ref: 'BLAS', complexity: 'O(n)',
      fn: function (x, y) { var n = Math.min(x.length, y.length), z = new Float64Array(n); for (var i = 0; i < n; i++) z[i] = x[i] + y[i]; return z; },
      selfTest: function (rng) { for (var t = 0; t < 30; t++) { var n = 1 + ((rng() * 40) | 0), x = randArr(rng, n), y = randArr(rng, n), z = this.fn(x, y); for (var i = 0; i < n; i++) if (!Object.is(z[i], x[i] + y[i])) return { pass: false, detail: '逐位不等 @' + i }; } return { pass: true, detail: '逐位等于逐项相加' }; } },
    { key: 'B001', name: 'vsub', sig: '(x,y)->z', ref: 'BLAS', complexity: 'O(n)',
      fn: function (x, y) { var n = Math.min(x.length, y.length), z = new Float64Array(n); for (var i = 0; i < n; i++) z[i] = x[i] - y[i]; return z; },
      selfTest: function (rng) { for (var t = 0; t < 30; t++) { var n = 1 + ((rng() * 40) | 0), x = randArr(rng, n), y = randArr(rng, n), z = this.fn(x, y); for (var i = 0; i < n; i++) if (!Object.is(z[i], x[i] - y[i])) return { pass: false, detail: '逐位不等 @' + i }; } return { pass: true, detail: '逐位等于逐项相减' }; } },
    { key: 'B002', name: 'vmul', sig: '(x,y)->z', ref: 'BLAS', complexity: 'O(n)',
      fn: function (x, y) { var n = Math.min(x.length, y.length), z = new Float64Array(n); for (var i = 0; i < n; i++) z[i] = x[i] * y[i]; return z; },
      selfTest: function (rng) { for (var t = 0; t < 30; t++) { var n = 1 + ((rng() * 40) | 0), x = randArr(rng, n), y = randArr(rng, n), z = this.fn(x, y); for (var i = 0; i < n; i++) if (!Object.is(z[i], x[i] * y[i])) return { pass: false, detail: '逐位不等 @' + i }; } return { pass: true, detail: '逐位等于逐项相乘' }; } },
    { key: 'B003', name: 'vdiv', sig: '(x,y)->z', ref: 'BLAS', complexity: 'O(n)',
      fn: function (x, y) { var n = Math.min(x.length, y.length), z = new Float64Array(n); for (var i = 0; i < n; i++) z[i] = x[i] / y[i]; return z; },
      selfTest: function (rng) { for (var t = 0; t < 30; t++) { var n = 1 + ((rng() * 40) | 0), x = randArr(rng, n), y = randArr(rng, n), z = this.fn(x, y); for (var i = 0; i < n; i++) { if (y[i] === 0) continue; if (!Object.is(z[i], x[i] / y[i])) return { pass: false, detail: '逐位不等 @' + i }; } } return { pass: true, detail: '逐位等于逐项相除（y=0 跳过）' }; } },
    { key: 'B004', name: 'axpy', sig: '(x,y,a)->y', ref: 'BLAS daxpy', complexity: 'O(n)',
      fn: function (x, y, a) { for (var i = 0; i < x.length && i < y.length; i++) y[i] += a * x[i]; return y; },
      selfTest: function (rng) { for (var t = 0; t < 30; t++) { var n = 1 + ((rng() * 40) | 0), x = randArr(rng, n), y0 = randArr(rng, n), a = (rng() - 0.5) * 10, y = Float64Array.from(y0), r = this.fn(x, y, a); for (var i = 0; i < n; i++) if (!Object.is(r[i], y0[i] + a * x[i])) return { pass: false, detail: '逐位不等 @' + i }; } return { pass: true, detail: 'y += a*x 逐位一致' }; } },
    { key: 'B005', name: 'scal', sig: '(x,s)->x', ref: 'BLAS dscal', complexity: 'O(n)',
      fn: function (x, s) { for (var i = 0; i < x.length; i++) x[i] *= s; return x; },
      selfTest: function (rng) { for (var t = 0; t < 30; t++) { var n = 1 + ((rng() * 40) | 0), x0 = randArr(rng, n), s = (rng() - 0.5) * 10, x = Float64Array.from(x0), r = this.fn(x, s); for (var i = 0; i < n; i++) if (!Object.is(r[i], x0[i] * s)) return { pass: false, detail: '逐位不等 @' + i }; } return { pass: true, detail: 'x *= s 逐位一致' }; } },
    { key: 'B006', name: 'vsum_naive', sig: '(x)->number', ref: '教科书顺序求和', complexity: 'O(n)',
      fn: function (x) { return naiveSum(x); },
      selfTest: function (rng) { var x = randArr(rng, 100); var r = this.fn(x), s = 0; for (var i = 0; i < x.length; i++) s += x[i]; return { pass: Object.is(r, s), detail: '与同序参考逐位一致' }; } },
    { key: 'B007', name: 'vsum_kahan', sig: '(x)->number', ref: 'Kahan 1965', complexity: 'O(n)',
      fn: function (x) { var sum = 0, comp = 0; for (var i = 0; i < x.length; i++) { var y = x[i] - comp, t = sum + y; comp = (t - sum) - y; sum = t; } return sum + comp; },
      selfTest: function (rng) {
        // 大数吃小数陷阱：1e16 之后累加 1000 个 1 再减 1e16，朴素求和得 0，Kahan 必得 1000
        var trap = new Float64Array(1002); trap[0] = 1e16; for (var i = 1; i <= 1000; i++) trap[i] = 1; trap[1001] = -1e16;
        var kn = this.fn(trap), nv = naiveSum(trap);
        if (kn !== 1000) return { pass: false, detail: 'Kahan 未得精确值 1000（得 ' + kn + '）' };
        if (nv === 1000) return { pass: false, detail: '朴素求和竟也精确（陷阱构造失败）' };
        return { pass: true, detail: '朴素=' + nv + ' 丢失全部低位，Kahan=1000 精确' };
      } },
    { key: 'B008', name: 'vsum_pairwise', sig: '(x)->number', ref: '分治求和（两两合并）', complexity: 'O(n)',
      fn: function (x) {
        function rec(lo, hi) { if (hi - lo <= 2) { var s = 0; for (var i = lo; i < hi; i++) s += x[i]; return s; } var mid = (lo + hi) >> 1; return rec(lo, mid) + rec(mid, hi); }
        return x.length ? rec(0, x.length) : 0;
      },
      selfTest: function (rng) { for (var t = 0; t < 30; t++) { var n = 1 + ((rng() * 200) | 0), x = randArr(rng, n), p = this.fn(x), ref = 0, c = 0; for (var i = 0; i < n; i++) { var y = x[i] - c, tt = ref + y; c = (tt - ref) - y; ref = tt; } ref += c; if (Math.abs(p - ref) > 1e-12 * (1 + Math.abs(ref))) return { pass: false, detail: '与 Kahan 参考偏差超界' }; } return { pass: true, detail: '分治 ≡ 补偿参考（≤1e-12 相对）' }; } },
    { key: 'B009', name: 'dot', sig: '(a,b)->number', ref: 'BLAS ddot', complexity: 'O(n)',
      fn: function (a, b) { var s = 0, n = Math.min(a.length, b.length); for (var i = 0; i < n; i++) s += a[i] * b[i]; return s; },
      selfTest: function (rng) { for (var t = 0; t < 30; t++) { var n = 1 + ((rng() * 40) | 0), a = randArr(rng, n), b = randArr(rng, n), d1 = this.fn(a, b), d2 = this.fn(b, a); if (!Object.is(d1, d2)) return { pass: false, detail: 'dot(a,b) !== dot(b,a)' }; } var u = randArr(rng, 20, 10), e = new Float64Array(20); e[7] = 1; if (!Object.is(this.fn(u, e), u[7])) return { pass: false, detail: 'one-hot 点积不等于该元素' }; return { pass: true, detail: '交换律逐位 + one-hot 精确' }; } },
    { key: 'B010', name: 'norm_l2', sig: '(x)->number', ref: 'BLAS dnrm2', complexity: 'O(n)',
      fn: function (x) { var s = 0; for (var i = 0; i < x.length; i++) s += x[i] * x[i]; return Math.sqrt(s); },
      selfTest: function (rng) { var t = new Float64Array([3, 4]); if (!approx(this.fn(t), 5, 1e-12)) return { pass: false, detail: '3-4-5 失败' }; if (this.fn(new Float64Array(5)) !== 0) return { pass: false, detail: '零向量范数非 0' }; for (var k = 0; k < 20; k++) { var x = randArr(rng, 50, 100), r = this.fn(x), s = 0; for (var i = 0; i < x.length; i++) s += x[i] * x[i]; if (!approx(r, Math.sqrt(s), 1e-12 * (1 + r))) return { pass: false, detail: '与参考偏差超界' }; } return { pass: true, detail: '‖x‖₂ 正确且非负' }; } },
    { key: 'B011', name: 'norm_l1', sig: '(x)->number', ref: 'L1 范数', complexity: 'O(n)',
      fn: function (x) { var s = 0; for (var i = 0; i < x.length; i++) s += Math.abs(x[i]); return s; },
      selfTest: function (rng) { var t = new Float64Array([3, -4]); if (this.fn(t) !== 7) return { pass: false, detail: '[3,-4] L1 != 7' }; for (var k = 0; k < 20; k++) { var x = randArr(rng, 50), r = this.fn(x), s = 0; for (var i = 0; i < x.length; i++) s += Math.abs(x[i]); if (!Object.is(r, s)) return { pass: false, detail: '与同序参考逐位不一致' }; } return { pass: true, detail: 'Σ|xᵢ| 逐位一致' }; } },
    { key: 'B012', name: 'norm_inf', sig: '(x)->number', ref: 'L∞ 范数', complexity: 'O(n)',
      fn: function (x) { var m = 0; for (var i = 0; i < x.length; i++) { var a = Math.abs(x[i]); if (a > m) m = a; } return m; },
      selfTest: function (rng) { var t = new Float64Array([3, -7, 2]); if (this.fn(t) !== 7) return { pass: false, detail: 'max|x| 错' }; if (this.fn(new Float64Array(4)) !== 0) return { pass: false, detail: '空/零向量非 0' }; for (var k = 0; k < 20; k++) { var x = randArr(rng, 50), r = this.fn(x), m = 0; for (var i = 0; i < x.length; i++) m = Math.max(m, Math.abs(x[i])); if (r !== m) return { pass: false, detail: '与参考不一致' }; } return { pass: true, detail: 'max|xᵢ| 一致' }; } },
    { key: 'B013', name: 'argmax', sig: '(x)->number', ref: '归约', complexity: 'O(n)',
      fn: function (x) { var bi = 0; for (var i = 1; i < x.length; i++) if (x[i] > x[bi]) bi = i; return bi; },
      selfTest: function (rng) { var t = new Float64Array([1, 3, 3, 2]); if (this.fn(t) !== 1) return { pass: false, detail: '并列未取首下标' }; for (var k = 0; k < 30; k++) { var n = 1 + ((rng() * 50) | 0), x = randArr(rng, n), r = this.fn(x), bi = 0; for (var i = 1; i < n; i++) if (x[i] > x[bi]) bi = i; if (r !== bi) return { pass: false, detail: '与参考不一致' }; for (var j = 0; j < n; j++) if (x[j] > x[r]) return { pass: false, detail: '返回值非最大' }; } return { pass: true, detail: '首下标最大值，并列稳定' }; } },
    { key: 'B014', name: 'cumsum', sig: '(x)->y', ref: '前缀扫描', complexity: 'O(n)',
      fn: function (x) { var y = new Float64Array(x.length), s = 0; for (var i = 0; i < x.length; i++) { s += x[i]; y[i] = s; } return y; },
      selfTest: function (rng) { var t = this.fn(new Float64Array([1, 2, 3])); if (!(t[0] === 1 && t[1] === 3 && t[2] === 6)) return { pass: false, detail: '[1,2,3] 前缀和错' }; for (var k = 0; k < 20; k++) { var n = 1 + ((rng() * 60) | 0), x = randArr(rng, n), y = this.fn(x), s = 0; for (var i = 0; i < n; i++) s += x[i]; if (!Object.is(y[n - 1], s)) return { pass: false, detail: '末项 ≠ 全量和（逐位）' }; for (var i = 1; i < n; i++) if (!Object.is(y[i] - y[i - 1], x[i])) return { pass: false, detail: '前缀关系破坏 @' + i }; } return { pass: true, detail: '前缀关系逐位 + 末项=总和' }; } },
    { key: 'B015', name: 'vclip', sig: '(x,lo,hi)->z', ref: '逐元素钳位', complexity: 'O(n)',
      fn: function (x, lo, hi) { var z = new Float64Array(x.length); for (var i = 0; i < x.length; i++) z[i] = x[i] < lo ? lo : (x[i] > hi ? hi : x[i]); return z; },
      selfTest: function (rng) { for (var k = 0; k < 20; k++) { var n = 1 + ((rng() * 40) | 0), x = randArr(rng, n, 100), lo = -5, hi = 5, z = this.fn(x, lo, hi); for (var i = 0; i < n; i++) if (!(z[i] >= lo && z[i] <= hi)) return { pass: false, detail: '越界 @' + i }; } var u = randArr(rng, 20), id = this.fn(u, -Infinity, Infinity); for (var j = 0; j < 20; j++) if (!Object.is(id[j], u[j])) return { pass: false, detail: '±Inf 边界非恒等' }; return { pass: true, detail: 'lo≤z≤hi 且 ±Inf 恒等' }; } }
  ];

  for (var i = 0; i < OPS.length; i++) {
    var o = OPS[i];
    o.domain = 'B'; o.impl = 'done';
    L10.register(o);
  }
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

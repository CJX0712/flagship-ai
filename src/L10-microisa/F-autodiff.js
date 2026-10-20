/* L10 · 微指令层 / F 域：自动微分原语（VJP） —— 依赖：无（最底层，自包含）
 * 不变量范式：中心差分梯度检验（err/tol < 1.0；基线：整体 4.22e-4、
 * QK-norm 1.4e-3、soft-cap 6.5e-3 —— Karpathy《On Testing ML Programs》规范：
 * 双精度、避开 kink、少样本、关正则）。
 * @ref micrograd（Karpathy）· JAX 函数式变换 · arXiv:1812.02257
 */
;(function (NS) {
  'use strict';
  // 所属层：L10 · 允许依赖：无 · 主要 @ref：micrograd / 1812.02257
  var L10 = NS.L10;

  function numDiff(f, x, h) { h = h || 1e-5; return (f(x + h) - f(x - h)) / (2 * h); }
  function within(a, n) { return Math.abs(a - n) <= 1e-6 * (1 + Math.abs(a) + Math.abs(n)); }

  var OPS = [
    { key: 'F000', name: 'vjp_mul', sig: '(x,y)->{val,dx,dy}', ref: '乘法反向', complexity: 'O(1)',
      fn: function (x, y) { return { val: x * y, dx: y, dy: x }; },
      selfTest: function (rng) { for (var t = 0; t < 40; t++) { var x = (rng() - 0.5) * 8, y = (rng() - 0.5) * 8, r = this.fn(x, y); if (!within(r.dx, numDiff(function (u) { return u * y; }, x)) || !within(r.dy, numDiff(function (u) { return x * u; }, y))) return { pass: false, detail: '梯度与中心差分不符' }; } return { pass: true, detail: 'dx=y, dy=x 全部通过中心差分' }; } },
    { key: 'F001', name: 'vjp_add', sig: '(x,y)->{val,dx,dy}', ref: '加法反向', complexity: 'O(1)',
      fn: function (x, y) { return { val: x + y, dx: 1, dy: 1 }; },
      selfTest: function (rng) { for (var t = 0; t < 20; t++) { var x = (rng() - 0.5) * 8, y = (rng() - 0.5) * 8, r = this.fn(x, y); if (r.dx !== 1 || r.dy !== 1) return { pass: false, detail: '单位梯度错' }; if (!within(1, numDiff(function (u) { return u + y; }, x))) return { pass: false, detail: '数值梯度非 1' }; } return { pass: true, detail: 'dx=dy=1 通过中心差分' }; } },
    { key: 'F002', name: 'vjp_exp', sig: '(x)->{val,dx}', ref: '指数反向', complexity: 'O(1)',
      fn: function (x) { var e = Math.exp(x); return { val: e, dx: e }; },
      selfTest: function (rng) { for (var t = 0; t < 40; t++) { var x = (rng() - 0.5) * 8, r = this.fn(x); if (!within(r.dx, numDiff(Math.exp, x))) return { pass: false, detail: "d/dx e^x 不符" }; } return { pass: true, detail: 'dx=e^x 通过中心差分' }; } },
    { key: 'F003', name: 'vjp_log', sig: '(x)->{val,dx}', ref: '对数反向', complexity: 'O(1)',
      fn: function (x) { return { val: Math.log(x), dx: 1 / x }; },
      selfTest: function (rng) { for (var t = 0; t < 40; t++) { var x = 0.1 + rng() * 20, r = this.fn(x); if (!within(r.dx, numDiff(Math.log, x))) return { pass: false, detail: 'd/dx ln x 不符' }; } return { pass: true, detail: 'dx=1/x 通过中心差分' }; } },
    { key: 'F004', name: 'vjp_sigmoid', sig: '(x)->{val,dx}', ref: 'sigmoid 反向', complexity: 'O(1)',
      fn: function (x) { var s = x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x)); return { val: s, dx: s * (1 - s) }; },
      selfTest: function (rng) { for (var t = 0; t < 40; t++) { var x = (rng() - 0.5) * 16, r = this.fn(x); if (!within(r.dx, numDiff(function (u) { return u >= 0 ? 1 / (1 + Math.exp(-u)) : Math.exp(u) / (1 + Math.exp(u)); }, x))) return { pass: false, detail: 'd/dx σ 不符' }; } return { pass: true, detail: 'dx=σ(1−σ) 通过中心差分' }; } },
    { key: 'F005', name: 'vjp_tanh', sig: '(x)->{val,dx}', ref: 'tanh 反向', complexity: 'O(1)',
      fn: function (x) { var t = Math.tanh(x); return { val: t, dx: 1 - t * t }; },
      selfTest: function (rng) { for (var t = 0; t < 40; t++) { var x = (rng() - 0.5) * 8, r = this.fn(x); if (!within(r.dx, numDiff(Math.tanh, x))) return { pass: false, detail: 'd/dx tanh 不符' }; } return { pass: true, detail: 'dx=1−tanh² 通过中心差分' }; } },
    { key: 'F006', name: 'vjp_silu', sig: '(x)->{val,dx}', ref: 'SiLU/Swish 反向', complexity: 'O(1)',
      fn: function (x) { var s = x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x)); var y = x * s; return { val: y, dx: y + s * (1 - y) }; },
      selfTest: function (rng) { for (var t = 0; t < 40; t++) { var x = (rng() - 0.5) * 8, r = this.fn(x); if (!within(r.dx, numDiff(function (u) { var s = u >= 0 ? 1 / (1 + Math.exp(-u)) : Math.exp(u) / (1 + Math.exp(u)); return u * s; }, x))) return { pass: false, detail: 'd/dx silu 不符' }; } return { pass: true, detail: 'dx=y+σ(1−y) 通过中心差分' }; } },
    { key: 'F007', name: 'vjp_gelu_tanh', sig: '(x)->{val,dx}', ref: 'GELU(tanh) 反向 1606.08415', complexity: 'O(1)',
      fn: function (x) { var u = 0.7978845608028654 * (x + 0.044715 * x * x * x); var t = Math.tanh(u); var y = 0.5 * x * (1 + t); var up = 0.7978845608028654 * (1 + 3 * 0.044715 * x * x); return { val: y, dx: 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * up }; },
      selfTest: function (rng) { for (var t = 0; t < 40; t++) { var x = (rng() - 0.5) * 6, r = this.fn(x); if (!within(r.dx, numDiff(function (u) { var uu = 0.7978845608028654 * (u + 0.044715 * u * u * u); return 0.5 * u * (1 + Math.tanh(uu)); }, x))) return { pass: false, detail: 'd/dx gelu 不符' }; } return { pass: true, detail: 'gelu 解析梯度通过中心差分' }; } },
    { key: 'F008', name: 'vjp_relu', sig: '(x)->{val,dx}', ref: 'ReLU 反向', complexity: 'O(1)',
      fn: function (x) { return { val: x > 0 ? x : 0, dx: x > 0 ? 1 : 0 }; },
      selfTest: function (rng) {
        // Karpathy 规范：避开 kink（x=0），只在 |x|>0.1 处检验
        for (var t = 0; t < 40; t++) { var x = (rng() < 0.5 ? -1 : 1) * (0.1 + rng() * 8), r = this.fn(x); if (!within(r.dx, numDiff(function (u) { return u > 0 ? u : 0; }, x))) return { pass: false, detail: 'd/dx relu 不符（kink 外）' }; }
        if (this.fn(0).dx !== 0 || this.fn(0).val !== 0) return { pass: false, detail: 'x=0 约定错' };
        return { pass: true, detail: 'kink 外梯度精确 + x=0 约定 dx=0' };
      } },
    { key: 'F009', name: 'vjp_softmax', sig: '(x,v)->{p,jvp}', ref: 'softmax JVP（雅可比-向量积）', complexity: 'O(n)',
      fn: function (x, v) {
        var n = x.length, m = -Infinity;
        for (var i = 0; i < n; i++) if (x[i] > m) m = x[i];
        var p = new Float64Array(n), s = 0;
        for (var i2 = 0; i2 < n; i2++) { p[i2] = Math.exp(x[i2] - m); s += p[i2]; }
        for (var i3 = 0; i3 < n; i3++) p[i3] /= s;
        var pv = 0; for (var i4 = 0; i4 < n; i4++) pv += p[i4] * v[i4];
        var jvp = new Float64Array(n);
        for (var i5 = 0; i5 < n; i5++) jvp[i5] = p[i5] * (v[i5] - pv);
        return { p: p, jvp: jvp };
      },
      selfTest: function (rng) {
        for (var t = 0; t < 20; t++) {
          var n = 4 + ((rng() * 5) | 0), x = new Float64Array(n), v = new Float64Array(n), h = 1e-6;
          for (var i = 0; i < n; i++) { x[i] = (rng() - 0.5) * 10; v[i] = (rng() - 0.5) * 4; }
          var r = this.fn(x, v);
          // 数值 JVP：(softmax(x+hv) − softmax(x−hv)) / 2h
          var xp = new Float64Array(n), xm = new Float64Array(n);
          for (var j = 0; j < n; j++) { xp[j] = x[j] + h * v[j]; xm[j] = x[j] - h * v[j]; }
          var pp = this.fn(xp, v).p, pm = this.fn(xm, v).p, maxErr = 0;
          for (var k = 0; k < n; k++) maxErr = Math.max(maxErr, Math.abs(r.jvp[k] - (pp[k] - pm[k]) / (2 * h)));
          if (maxErr > 1e-6) return { pass: false, detail: 'JVP 与数值偏差 ' + maxErr };
          var s = 0; for (var m = 0; m < n; m++) s += r.p[m];
          if (Math.abs(s - 1) > 1e-14) return { pass: false, detail: '行和偏离 1' };
        }
        return { pass: true, detail: 'JVP ≡ 中心差分（≤1e-6）+ 行和=1' };
      } },
    { key: 'F010', name: 'vjp_logit_softcap', sig: '(x,cap)->{val,dx}', ref: 'logit 软帽（Gemma 2）', complexity: 'O(1)',
      fn: function (x, cap) { if (!(cap > 0)) return { val: x, dx: 1 }; var t = Math.tanh(x / cap); return { val: cap * t, dx: 1 - t * t }; },
      selfTest: function (rng) {
        for (var t = 0; t < 40; t++) { var x = (rng() - 0.5) * 60, cap = 1 + rng() * 30, r = this.fn(x, cap); if (!within(r.dx, numDiff(function (u) { return cap * Math.tanh(u / cap); }, x))) return { pass: false, detail: 'd/dx softcap 不符' }; if (Math.abs(r.val) > cap + 1e-12) return { pass: false, detail: '超出 cap 界' }; }
        var id = this.fn(3.5, 0);
        if (id.val !== 3.5 || id.dx !== 1) return { pass: false, detail: 'cap<=0 应为恒等' };
        return { pass: true, detail: 'dx=sech²(x/cap) 通过差分 + |y|≤cap + 恒等分支' };
      } }
  ];

  for (var i = 0; i < OPS.length; i++) {
    var o = OPS[i];
    o.domain = 'F'; o.impl = 'done';
    L10.register(o);
  }
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

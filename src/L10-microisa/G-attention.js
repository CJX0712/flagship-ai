/* L10 · 微指令层 / G 域：注意力与位置编码原语 —— 依赖：无（最底层，自包含）
 * 不变量范式：RoPE 相对位置不变性（v11 实测 1.33e-15）、掩码零泄露
 * （未来位置 logits 变化 = 0）、SWA 窗外逐位 0、softcap |y|≤cap。
 * @ref RoPE 2104.09864 · ALiBi 2108.12409 · YaRN 2309.00071 · Gemma 2 QK-norm
 */
;(function (NS) {
  'use strict';
  // 所属层：L10 · 允许依赖：无 · 主要 @ref：RoPE / ALiBi / YaRN / Gemma 2
  var L10 = NS.L10;

  function rv(rng, n, scale) { var a = new Float64Array(n), s = scale || 2; for (var i = 0; i < n; i++) a[i] = (rng() - 0.5) * s; return a; }
  function numDiff(f, x, h) { h = h || 1e-6; return (f(x + h) - f(x - h)) / (2 * h); }
  function within(a, b, tol) { return Math.abs(a - b) <= (tol || 1e-12); }

  var OPS = [
    { key: 'G000', name: 'rope_inv_freq', sig: '(dh,base)->freqs', ref: 'RoPE 2104.09864', complexity: 'O(dh/2)',
      fn: function (dh, base) { var half = dh >> 1, f = new Float64Array(half); for (var i = 0; i < half; i++) f[i] = Math.pow(base, -2 * i / dh); return f; },
      selfTest: function (rng) {
        var f = this.fn(8, 10000);
        if (f.length !== 4 || f[0] !== 1) return { pass: false, detail: 'freq[0] 应为 1' };
        for (var i = 1; i < f.length; i++) if (!(f[i] < f[i - 1])) return { pass: false, detail: '频率应严格递减' };
        if (!within(f[3], Math.pow(10000, -6 / 8), 1e-18)) return { pass: false, detail: '末频与公式不符' };
        var ratio = f[0] / f[1];
        for (var j = 1; j < f.length - 1; j++) if (!within(f[j] / f[j + 1], ratio, 1e-12)) return { pass: false, detail: '非几何数列' };
        return { pass: true, detail: '几何递减 + 首项 1 + 公比 base^(2/dh)' };
      } },
    { key: 'G001', name: 'rope_cossin', sig: '(pos,freqs)->{cos,sin}', ref: 'RoPE 旋转核', complexity: 'O(dh/2)',
      fn: function (pos, freqs) { var n = freqs.length, cs = new Float64Array(n), sn = new Float64Array(n); for (var i = 0; i < n; i++) { var th = pos * freqs[i]; cs[i] = Math.cos(th); sn[i] = Math.sin(th); } return { cos: cs, sin: sn }; },
      selfTest: function (rng) {
        var z = this.fn(0, NS.L10.run('G000', [8, 10000]));
        for (var i = 0; i < 4; i++) if (z.cos[i] !== 1 || z.sin[i] !== 0) return { pass: false, detail: 'pos=0 应为恒等旋转' };
        for (var t = 0; t < 20; t++) { var pos = (rng() - 0.5) * 4000, f = NS.L10.run('G000', [8, 10000]), r = this.fn(pos, f); for (var j = 0; j < 4; j++) if (Math.abs(r.cos[j] * r.cos[j] + r.sin[j] * r.sin[j] - 1) > 1e-15) return { pass: false, detail: 'cos²+sin²≠1' }; }
        return { pass: true, detail: '单位圆不变量（≤1e-15）+ pos=0 恒等' };
      } },
    { key: 'G002', name: 'rope_rotate_pair', sig: '(x,y,th)->{x,y}', ref: '2D 旋转', complexity: 'O(1)',
      fn: function (x, y, th) { var c = Math.cos(th), s = Math.sin(th); return { x: x * c - y * s, y: x * s + y * c }; },
      selfTest: function (rng) {
        var id = this.fn(3, 4, 0);
        if (id.x !== 3 || id.y !== 4) return { pass: false, detail: 'θ=0 应恒等' };
        var q = this.fn(3, 4, Math.PI / 2);
        if (!within(q.x, -4, 1e-15) || !within(q.y, 3, 1e-15)) return { pass: false, detail: '90° 旋转错' };
        for (var t = 0; t < 30; t++) { var x = (rng() - 0.5) * 100, y = (rng() - 0.5) * 100, th = (rng() - 0.5) * 20, r = this.fn(x, y, th); if (Math.abs(Math.sqrt(r.x * r.x + r.y * r.y) - Math.sqrt(x * x + y * y)) > 1e-13 * (1 + Math.abs(x) + Math.abs(y))) return { pass: false, detail: '旋转不保范数' }; }
        return { pass: true, detail: '保范数 + θ=0/π/2 特例精确' };
      } },
    { key: 'G003', name: 'rope_apply_vec', sig: '(x,pos,freqs)->y', ref: 'RoPE 向量施加', complexity: 'O(dh)',
      fn: function (x, pos, freqs) { var n = x.length, y = new Float64Array(n); for (var i = 0; i < n >> 1; i++) { var th = pos * freqs[i], c = Math.cos(th), s = Math.sin(th); y[2 * i] = x[2 * i] * c - x[2 * i + 1] * s; y[2 * i + 1] = x[2 * i] * s + x[2 * i + 1] * c; } return y; },
      selfTest: function (rng) {
        var f = NS.L10.run('G000', [8, 10000]);
        // 保范数
        for (var t = 0; t < 10; t++) { var x = rv(rng, 8), pos = (rng() - 0.5) * 1000, y = this.fn(x, pos, f), n0 = 0, n1 = 0; for (var i = 0; i < 8; i++) { n0 += x[i] * x[i]; n1 += y[i] * y[i]; } if (Math.abs(Math.sqrt(n1) - Math.sqrt(n0)) > 1e-13 * (1 + Math.sqrt(n0))) return { pass: false, detail: '旋转不保范数' }; }
        // 相对位置不变性：<R(m)q, R(n)k> 只依赖 (n−m)
        var q = rv(rng, 8), k = rv(rng, 8), op = this;
        function score(m, n) { var rq = op.fn(q, m, f), rk = op.fn(k, n, f), s = 0; for (var i = 0; i < 8; i++) s += rq[i] * rk[i]; return s; }
        var m = (rng() * 500) | 0, n = m + 1 + ((rng() * 500) | 0), d = 1 + ((rng() * 300) | 0);
        var s1 = score(m, n), s2 = score(m + d, n + d);
        if (!within(s1, s2, 1e-9 * (1 + Math.abs(s1)))) return { pass: false, detail: '相对位置不变性破坏: ' + Math.abs(s1 - s2) };
        return { pass: true, detail: '保范数 + 相对不变性（≤1e-9）' };
      } },
    { key: 'G004', name: 'yarn_correction_mask', sig: '(i,half,bFast,bSlow)->mask', ref: 'YaRN 2309.00071', complexity: 'O(1)',
      fn: function (i, half, betaFast, betaSlow) { var t = half > 1 ? i / (half - 1) : 0; if (betaSlow <= betaFast) return t >= betaFast ? 1 : 0; var raw = (t - betaFast) / (betaSlow - betaFast); raw = raw < 0 ? 0 : (raw > 1 ? 1 : raw); return raw * raw * (3 - 2 * raw); },
      selfTest: function (rng) {
        var half = 8, lo = 0.25, hi = 0.75;
        if (this.fn(0, half, lo, hi) !== 0) return { pass: false, detail: 'mask(0) 应为 0' };
        if (this.fn(half - 1, half, lo, hi) !== 1) return { pass: false, detail: 'mask(half-1) 应为 1' };
        var prev = -1;
        for (var i = 0; i < half; i++) { var m = this.fn(i, half, lo, hi); if (!(m >= 0 && m <= 1)) return { pass: false, detail: 'mask 越界 [0,1]' }; if (m < prev) return { pass: false, detail: 'mask 应单调不减' }; prev = m; }
        if (this.fn(3, half, 0.5, 0.2) !== 0 || this.fn(4, half, 0.5, 0.2) !== 1) return { pass: false, detail: '退化区间分支错' };
        return { pass: true, detail: '端点 0/1 + 单调 + smoothstep ∈[0,1]' };
      } },
    { key: 'G005', name: 'ntk_base_scale', sig: '(base,factor,dh)->base\'', ref: 'NTK-aware 缩放', complexity: 'O(1)',
      fn: function (base, factor, dh) { return base * Math.pow(factor, dh / (dh - 2)); },
      selfTest: function (rng) {
        if (this.fn(10000, 1, 64) !== 10000) return { pass: false, detail: 'factor=1 应不变' };
        for (var t = 0; t < 10; t++) { var base = 1000 + rng() * 40000, factor = 1 + rng() * 16, dh = 16 + ((rng() * 112) | 0), r = this.fn(base, factor, dh), ref = base * Math.pow(factor, dh / (dh - 2)); if (r !== ref) return { pass: false, detail: '与公式不符' }; if (factor > 1 && !(r > base)) return { pass: false, detail: 'factor>1 应放大 base' }; }
        return { pass: true, detail: 'base·factor^(dh/(dh−2)) 精确' };
      } },
    { key: 'G006', name: 'alibi_slopes', sig: '(nHead)->slopes', ref: 'ALiBi 2108.12409', complexity: 'O(H)',
      fn: function (nHead) { var s = new Float64Array(nHead); for (var i = 0; i < nHead; i++) s[i] = Math.pow(2, -8 * (i + 1) / nHead); return s; },
      selfTest: function (rng) {
        var s = this.fn(8);
        if (!within(s[0], 0.5, 1e-15) || !within(s[7], Math.pow(2, -8), 1e-15)) return { pass: false, detail: '8 头首尾斜率错' };
        for (var i = 1; i < s.length; i++) if (!(s[i] < s[i - 1])) return { pass: false, detail: '斜率应递减' };
        for (var j = 0; j < s.length; j++) if (!(s[j] > 0 && s[j] <= 1)) return { pass: false, detail: '斜率越界 (0,1]' };
        for (var t = 0; t < 10; t++) { var h = 1 + ((rng() * 16) | 0), w = this.fn(h); if (w.length !== h) return { pass: false, detail: '长度错' }; for (var k = 1; k < h; k++) if (!(w[k] < w[k - 1])) return { pass: false, detail: '随机头数非递减' }; }
        return { pass: true, detail: '2^(−8(i+1)/H) 递减且 ∈(0,1]' };
      } },
    { key: 'G007', name: 'alibi_bias', sig: '(i,j,slope)->bias', ref: 'ALiBi 线性偏置', complexity: 'O(1)',
      fn: function (i, j, slope) { return j > i ? -Infinity : -slope * (i - j); },
      selfTest: function (rng) {
        if (this.fn(0, 0, 0.5) !== 0) return { pass: false, detail: '对角线应为 0' };
        if (this.fn(3, 1, 0.5) !== -1) return { pass: false, detail: '距离 2 斜率 0.5 应为 -1' };
        if (this.fn(1, 3, 0.5) !== -Infinity) return { pass: false, detail: '未来位置应为 -Inf' };
        if (this.fn(5, 2, 0) !== 0) return { pass: false, detail: '斜率 0 应全 0' };
        for (var t = 0; t < 20; t++) { var i = 1 + ((rng() * 50) | 0), j = 1 + ((rng() * i) | 0), slope = 1e-3 + rng(); var near = this.fn(i, j, slope), far = this.fn(i, j - 1, slope); if (!(far < near)) return { pass: false, detail: '距离增加偏置应更负' }; }
        return { pass: true, detail: '对角线 0 + 随距离递减 + 未来 -Inf' };
      } },
    { key: 'G008', name: 'score_qk', sig: '(q,k,scale)->s', ref: '注意力打分', complexity: 'O(dh)',
      fn: function (q, k, scale) { var s = 0; for (var i = 0; i < q.length; i++) s += q[i] * k[i]; return s * scale; },
      selfTest: function (rng) {
        var q = new Float64Array([1, 2, 2, 1]); // ‖q‖²=10
        if (!within(this.fn(q, q, 1 / Math.sqrt(4)), 10 / 2, 1e-15)) return { pass: false, detail: 'score(q,q) != ‖q‖²/√dh' };
        if (this.fn(q, q, 0) !== 0) return { pass: false, detail: 'scale=0 应得 0' };
        for (var t = 0; t < 20; t++) { var n = 2 + ((rng() * 14) | 0), a = rv(rng, n), b = rv(rng, n), scale = 1 / Math.sqrt(n), r = this.fn(a, b, scale), ref = 0; for (var i = 0; i < n; i++) ref += a[i] * b[i]; if (!within(r, ref * scale, 1e-12 * (1 + Math.abs(ref * scale)))) return { pass: false, detail: '与参考不符' }; }
        return { pass: true, detail: 'scale·<q,k> 一致' };
      } },
    { key: 'G009', name: 'softmax_row', sig: '(x)->p', ref: '稳定 softmax', complexity: 'O(n)',
      fn: function (x) { var n = x.length, m = -Infinity; for (var i = 0; i < n; i++) if (x[i] > m) m = x[i]; var p = new Float64Array(n), s = 0; for (var j = 0; j < n; j++) { p[j] = Math.exp(x[j] - m); s += p[j]; } for (var k = 0; k < n; k++) p[k] /= s; return p; },
      selfTest: function (rng) {
        for (var t = 0; t < 20; t++) {
          var n = 2 + ((rng() * 14) | 0), x = rv(rng, n, 40), p = this.fn(x), s = 0, am = 0;
          for (var i = 0; i < n; i++) { if (!(p[i] > 0)) return { pass: false, detail: '概率非正' }; s += p[i]; if (p[i] > p[am]) am = i; }
          if (Math.abs(s - 1) > 1e-14) return { pass: false, detail: '行和偏离 1: ' + Math.abs(s - 1) };
          // 平移不变性
          var c = (rng() - 0.5) * 100, xp = new Float64Array(n);
          for (var j = 0; j < n; j++) xp[j] = x[j] + c;
          var p2 = this.fn(xp), md = 0;
          for (var k = 0; k < n; k++) md = Math.max(md, Math.abs(p[k] - p2[k]));
          if (md > 1e-15) return { pass: false, detail: '平移不变性破坏 ' + md };
          if (x[am] !== Math.max.apply(null, Array.prototype.slice.call(x))) return { pass: false, detail: 'argmax 不一致' };
        }
        var big = this.fn(new Float64Array([1000, 1001, 999]));
        var refBig = Math.exp(0) / (Math.exp(-1) + Math.exp(0) + Math.exp(-2));
        if (!within(big[1], refBig, 1e-15) || !big.every(function (v) { return isFinite(v); })) return { pass: false, detail: '大输入不稳定' };
        return { pass: true, detail: '行和=1（≤1e-14）+ 平移不变 + 大值稳定' };
      } },
    { key: 'G010', name: 'causal_mask', sig: '(T)->mask', ref: '因果掩码', complexity: 'O(T²)',
      fn: function (T) { var m = new Float64Array(T * T); for (var i = 0; i < T; i++) for (var j = 0; j < T; j++) m[i * T + j] = j <= i ? 0 : -Infinity; return m; },
      selfTest: function (rng) {
        var m = this.fn(4);
        for (var i = 0; i < 4; i++) { if (m[i * 4 + i] !== 0) return { pass: false, detail: '对角线应为 0' }; for (var j = 0; j < i; j++) if (m[i * 4 + j] !== 0) return { pass: false, detail: '过去位置应为 0' }; for (var j2 = i + 1; j2 < 4; j2++) if (m[i * 4 + j2] !== -Infinity) return { pass: false, detail: '未来位置应为 -Inf' }; }
        if (this.fn(1)[0] !== 0) return { pass: false, detail: 'T=1 应为 [0]' };
        if (this.fn(0).length !== 0) return { pass: false, detail: 'T=0 应为空' };
        return { pass: true, detail: '下三角+对角线 0，上三角 -Inf（零泄露）' };
      } },
    { key: 'G011', name: 'sliding_mask', sig: '(T,w)->mask', ref: 'SWA 因果滑窗掩码', complexity: 'O(T²)',
      fn: function (T, w) { var m = new Float64Array(T * T); for (var i = 0; i < T; i++) for (var j = 0; j < T; j++) m[i * T + j] = (j <= i && i - j <= w) ? 0 : -Infinity; return m; },
      selfTest: function (rng) {
        var m0 = this.fn(4, 0);
        for (var i = 0; i < 4; i++) { for (var j = 0; j < 4; j++) { var expect = j === i ? 0 : -Infinity; if (m0[i * 4 + j] !== expect) return { pass: false, detail: 'w=0 应仅对角线开' }; } }
        var mf = this.fn(4, 10), mc = NS.L10.run('G010', [4]);
        for (var k = 0; k < 16; k++) if (mf[k] !== mc[k]) return { pass: false, detail: 'w>=T-1 应等价因果掩码' };
        var m2 = this.fn(5, 2);
        if (m2[4 * 5 + 4] !== 0 || m2[4 * 5 + 2] !== 0 || m2[4 * 5 + 1] !== -Infinity || m2[4 * 5 + 3] !== 0) return { pass: false, detail: 'w=2 窗口错' };
        return { pass: true, detail: '窗口局部性 + w=0/w≥T-1 边界正确' };
      } },
    { key: 'G012', name: 'logit_softcap', sig: '(x,cap)->y', ref: 'Gemma 2 logit soft-cap', complexity: 'O(1)',
      fn: function (x, cap) { if (!(cap > 0)) return x; return cap * Math.tanh(x / cap); },
      selfTest: function (rng) {
        for (var t = 0; t < 30; t++) { var x = (rng() - 0.5) * 200, cap = 1 + rng() * 50, y = this.fn(x, cap); if (Math.abs(y) > cap + 1e-12) return { pass: false, detail: '超出 cap 界' }; if (Math.sign(y) !== Math.sign(x) && x !== 0) return { pass: false, detail: '符号应保持' }; }
        if (this.fn(0, 30) !== 0) return { pass: false, detail: 'softcap(0) 应为 0' };
        if (!within(this.fn(1000, 30), 30, 1e-12)) return { pass: false, detail: '大值应收敛到 cap' };
        if (this.fn(5, 0) !== 5) return { pass: false, detail: 'cap<=0 应恒等' };
        var a = this.fn(1, 30), b = this.fn(2, 30);
        if (!(b > a)) return { pass: false, detail: '应单调递增' };
        return { pass: true, detail: '|y|≤cap + 单调 + 削峰 + 恒等分支' };
      } },
    { key: 'G013', name: 'logit_softcap_d', sig: '(x,cap)->dy', ref: 'softcap 导数', complexity: 'O(1)',
      fn: function (x, cap) { if (!(cap > 0)) return 1; var t = Math.tanh(x / cap); return 1 - t * t; },
      selfTest: function (rng) {
        for (var t = 0; t < 30; t++) { var x = (rng() - 0.5) * 60, cap = 1 + rng() * 30, d = this.fn(x, cap), nd = numDiff(function (u) { return cap * Math.tanh(u / cap); }, x); if (Math.abs(d - nd) > 1e-6 * (1 + d + nd)) return { pass: false, detail: '与中心差分不符' }; if (!(d > 0 && d <= 1 + 1e-15)) return { pass: false, detail: '导数应 ∈(0,1]' }; }
        if (this.fn(3, 0) !== 1) return { pass: false, detail: 'cap<=0 导数应为 1' };
        return { pass: true, detail: 'dx=sech²(x/cap) 通过中心差分' };
      } },
    { key: 'G014', name: 'qk_norm_head', sig: '(x,eps)->{y,rstd}', ref: 'Gemma 2 QK-norm', complexity: 'O(dh)',
      fn: function (x, eps) { var dh = x.length, ss = 0; for (var i = 0; i < dh; i++) ss += x[i] * x[i]; var rstd = 1 / Math.sqrt(ss / dh + eps); var y = new Float64Array(dh); for (var j = 0; j < dh; j++) y[j] = x[j] * rstd; return { y: y, rstd: rstd }; },
      selfTest: function (rng) {
        for (var t = 0; t < 20; t++) {
          var n = 2 + ((rng() * 14) | 0), x = rv(rng, n, 10), eps = 1e-6, r = this.fn(x, eps), ss = 0, sy = 0;
          for (var i = 0; i < n; i++) { ss += x[i] * x[i]; sy += r.y[i] * r.y[i]; }
          if (Math.abs(r.rstd * r.rstd * (ss / n + eps) - 1) > 1e-14) return { pass: false, detail: 'rstd 恒等式破坏' };
          if (!(sy / n <= 1 / (1 + eps * n / ss) + 1e-15)) return { pass: false, detail: '归一化后能量超界' };
          for (var j = 0; j < n; j++) if (!Object.is(r.y[j], x[j] * r.rstd)) return { pass: false, detail: 'y != x·rstd' };
        }
        var z = this.fn(new Float64Array(4), 1e-6);
        if (z.rstd !== 1 / Math.sqrt(1e-6)) return { pass: false, detail: '零向量 rstd 错' };
        return { pass: true, detail: 'rstd²(ss/dh+eps)=1 + y=x·rstd 逐位' };
      } }
  ];

  for (var i = 0; i < OPS.length; i++) {
    var o = OPS[i];
    o.domain = 'G'; o.impl = 'done';
    L10.register(o);
  }
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

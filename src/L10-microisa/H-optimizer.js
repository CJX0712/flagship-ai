/* L10 · 微指令层 / H 域：优化器·量化·随机原语 —— 依赖：无（最底层，自包含）
 * 不变量范式：RNG 逐位可复现（mulberry32 金值）、量化往返误差硬界
 * （|x−deq(q)| ≤ absmax/254，v5/v12 同源）、优化器单步与参考公式逐位一致。
 * @ref AdamW 1711.05101 · LLM.int8() 2208.07339 · llama.cpp 采样器语义
 */
;(function (NS) {
  'use strict';
  // 所属层：L10 · 允许依赖：无 · 主要 @ref：AdamW / LLM.int8 / llama.cpp
  var L10 = NS.L10;

  function rv(rng, n, scale) { var a = new Float64Array(n), s = scale || 2; for (var i = 0; i < n; i++) a[i] = (rng() - 0.5) * s; return a; }
  function within(a, b, tol) { return Math.abs(a - b) <= tol; }

  // 与 v11 引擎逐位一致的 mulberry32（金值回归锚点）
  function mulberry32(seed) {
    var a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  var OPS = [
    { key: 'H000', name: 'rng_mulberry32', sig: '(seed)->stream', ref: 'mulberry32（v11 同款）', complexity: 'O(1)/步',
      fn: function (seed) { return mulberry32(seed); },
      selfTest: function (rng) {
        var s1 = this.fn(1), gold = [0.6270739405881613, 0.002735721180215478, 0.5274470399599522, 0.9810509674716741, 0.9683778982143849];
        for (var i = 0; i < gold.length; i++) if (!Object.is(s1(), gold[i])) return { pass: false, detail: '金值 #' + i + ' 不符（逐位回归失败）' };
        var a = this.fn(42), b = this.fn(42);
        for (var t = 0; t < 100; t++) if (!Object.is(a(), b())) return { pass: false, detail: '同种子流不一致' };
        var c = this.fn(7), ok = true;
        for (var k = 0; k < 1000; k++) { var v = c(); if (!(v >= 0 && v < 1)) { ok = false; break; } }
        return { pass: ok, detail: '5 个金值逐位一致 + 同种子复现 + 值域 [0,1)' };
      } },
    { key: 'H001', name: 'rng_split', sig: '(seed,index)->childSeed', ref: 'JAX 式 key 派生', complexity: 'O(1)',
      fn: function (seed, index) {
        var h = (seed >>> 0) ^ Math.imul((index + 1) >>> 0, 0x9E3779B1);
        h = Math.imul(h ^ (h >>> 16), 0x85EBCA6B) >>> 0;
        h = Math.imul(h ^ (h >>> 13), 0xC2B2AE35) >>> 0;
        return (h ^ (h >>> 16)) >>> 0;
      },
      selfTest: function (rng) {
        var a = this.fn(123, 0), b = this.fn(123, 0), c = this.fn(123, 1);
        if (a !== b) return { pass: false, detail: '同 (seed,index) 应逐位一致' };
        if (a === c) return { pass: false, detail: '不同 index 应得不同子种子' };
        var seen = Object.create(null), dup = false;
        for (var i = 0; i < 200; i++) { var s = this.fn(9, i); if (seen[s]) { dup = true; break; } seen[s] = 1; }
        if (dup) return { pass: false, detail: '200 个子种子出现碰撞' };
        // 子流与父流不同
        var p = mulberry32(123)(), child = mulberry32(a)();
        if (p === child) return { pass: false, detail: '子流与父流首值相同' };
        return { pass: true, detail: '确定性 + 无碰撞 + 子流独立' };
      } },
    { key: 'H002', name: 'adamw_step', sig: '(p,g,m,v,t,lr,b1,b2,eps,wd)->{p,m,v}', ref: 'AdamW 1711.05101', complexity: 'O(1)',
      fn: function (p, g, m, v, t, lr, b1, b2, eps, wd) {
        m = b1 * m + (1 - b1) * g;
        v = b2 * v + (1 - b2) * g * g;
        var mh = m / (1 - Math.pow(b1, t)), vh = v / (1 - Math.pow(b2, t));
        p = p * (1 - lr * wd) - lr * mh / (Math.sqrt(vh) + eps);
        return { p: p, m: m, v: v };
      },
      selfTest: function (rng) {
        // 参考公式逐位一致（单步）
        for (var t = 0; t < 20; t++) {
          var p0 = (rng() - 0.5) * 4, g = (rng() - 0.5) * 4, m0 = (rng() - 0.5) * 2, v0 = rng() * 2, lr = 0.001 + rng() * 0.01, wd = rng() * 0.1;
          var r = this.fn(p0, g, m0, v0, 3, lr, 0.9, 0.95, 1e-8, wd);
          var m1 = 0.9 * m0 + (1 - 0.9) * g, v1 = 0.95 * v0 + (1 - 0.95) * g * g;
          var mh = m1 / (1 - Math.pow(0.9, 3)), vh = v1 / (1 - Math.pow(0.95, 3));
          var p1 = p0 * (1 - lr * wd) - lr * mh / (Math.sqrt(vh) + 1e-8);
          if (!Object.is(r.p, p1) || !Object.is(r.m, m1) || !Object.is(r.v, v1)) return { pass: false, detail: '与 AdamW 参考公式逐位不一致' };
        }
        // 常梯度收敛行为：步长 ≈ lr·sign(g)
        var p = 0, m = 0, v = 0;
        for (var s = 1; s <= 200; s++) { var st = this.fn(p, 0.1, m, v, s, 0.01, 0.9, 0.95, 1e-8, 0); p = st.p; m = st.m; v = st.v; }
        if (!within(p, -2, 2e-3)) return { pass: false, detail: '200 步常梯度未走 ≈200·lr（得 ' + p + '）' };
        return { pass: true, detail: '单步逐位一致 + 常梯度 200 步 ≈ −2' };
      } },
    { key: 'H003', name: 'sgd_momentum', sig: '(p,g,v,lr,mu)->{p,v}', ref: 'SGD + 动量', complexity: 'O(1)',
      fn: function (p, g, v, lr, mu) { v = mu * v + g; return { p: p - lr * v, v: v }; },
      selfTest: function (rng) {
        var r1 = this.fn(1, 0.5, 0, 0.1, 0.9);
        if (r1.v !== 0.5 || r1.p !== 1 - 0.05) return { pass: false, detail: '首步错' };
        var r2 = this.fn(r1.p, 0.5, r1.v, 0.1, 0.9);
        if (r2.v !== 0.95 || r2.p !== r1.p - 0.095) return { pass: false, detail: '动量累积错' };
        for (var t = 0; t < 20; t++) { var p = (rng() - 0.5) * 4, g = (rng() - 0.5) * 4, v = (rng() - 0.5) * 2, lr = rng() * 0.1, mu = rng(); var r = this.fn(p, g, v, lr, mu); if (!Object.is(r.v, mu * v + g) || !Object.is(r.p, p - lr * (mu * v + g))) return { pass: false, detail: '与参考逐位不一致' }; }
        return { pass: true, detail: 'v=μv+g 逐位 + 动量累积正确' };
      } },
    { key: 'H004', name: 'clip_grad_norm', sig: '(g,maxNorm)->scale', ref: '全局梯度裁剪', complexity: 'O(n)',
      fn: function (g, maxNorm) { var s = 0; for (var i = 0; i < g.length; i++) s += g[i] * g[i]; var n = Math.sqrt(s); return n > maxNorm ? maxNorm / (n + 1e-6) : 1; },
      selfTest: function (rng) {
        var big = new Float64Array([3, 4]); // ‖g‖=5
        var sc = this.fn(big, 1);
        var nn = 0; for (var i = 0; i < 2; i++) nn += (big[i] * sc) * (big[i] * sc);
        if (Math.sqrt(nn) > 1 + 1e-5) return { pass: false, detail: '裁剪后范数超界' };
        var small = new Float64Array([0.3, 0.4]);
        if (this.fn(small, 1) !== 1) return { pass: false, detail: '未超阈应返回 1' };
        for (var t = 0; t < 20; t++) { var n = 2 + ((rng() * 30) | 0), x = rv(rng, n, 20), mx = 0.01 + rng() * 5, s2 = this.fn(x, mx), nrm = 0; for (var j = 0; j < n; j++) nrm += x[j] * x[j]; nrm = Math.sqrt(nrm); var clipped = 0; for (var k = 0; k < n; k++) clipped += (x[k] * s2) * (x[k] * s2); clipped = Math.sqrt(clipped); if (nrm > mx && clipped > mx + 1e-5 * (1 + mx)) return { pass: false, detail: '超阈未裁到界内' }; if (nrm <= mx && s2 !== 1) return { pass: false, detail: '未超阈被缩放' }; }
        return { pass: true, detail: '超阈裁至 maxNorm、未超阈恒 1' };
      } },
    { key: 'H005', name: 'lr_warmup_cosine', sig: '(t,T,warmup,baseLr,minLr)->lr', ref: 'warmup + cosine 衰减', complexity: 'O(1)',
      fn: function (t, T, warmup, baseLr, minLr) {
        if (t < warmup) return baseLr * (t + 1) / warmup;
        var pr = (t - warmup) / Math.max(1, T - warmup);
        if (pr > 1) pr = 1;
        return minLr + 0.5 * (baseLr - minLr) * (1 + Math.cos(Math.PI * pr));
      },
      selfTest: function (rng) {
        var T = 100, w = 10, base = 0.001, mn = 0.0001;
        if (this.fn(w - 1, T, w, base, mn) !== base) return { pass: false, detail: 'warmup 末尾未达 baseLr' };
        if (this.fn(T, T, w, base, mn) !== mn) return { pass: false, detail: '终点未达 minLr' };
        if (!within(this.fn(0, T, w, base, mn), base / w, 1e-18)) return { pass: false, detail: 'lr(0) 应为 base/warmup' };
        var prev = Infinity;
        for (var t = w; t <= T; t++) { var lr = this.fn(t, T, w, base, mn); if (lr > prev + 1e-18) return { pass: false, detail: 'cosine 段应单调不增' }; prev = lr; }
        if (this.fn(5, T, 0, base, mn) !== this.fn(5, T, 0, base, mn)) return { pass: false, detail: '确定性破坏' };
        var w0 = this.fn(0, T, 0, base, mn);
        if (!within(w0, base, 1e-18)) return { pass: false, detail: 'warmup=0 应直接 cosine' };
        return { pass: true, detail: '峰值/终点/起点精确 + cosine 单调不增' };
      } },
    { key: 'H006', name: 'weight_decay_decoupled', sig: '(p,lr,wd)->p\'', ref: 'AdamW 解耦权重衰减', complexity: 'O(1)',
      fn: function (p, lr, wd) { return p * (1 - lr * wd); },
      selfTest: function (rng) {
        for (var t = 0; t < 30; t++) { var p = (rng() - 0.5) * 10, lr = rng() * 0.01, wd = rng() * 0.2; if (!Object.is(this.fn(p, lr, wd), p * (1 - lr * wd))) return { pass: false, detail: '与参考逐位不一致' }; }
        if (this.fn(5, 0.01, 0) !== 5) return { pass: false, detail: 'wd=0 应恒等' };
        if (this.fn(1, 0.1, 1) !== 0.9) return { pass: false, detail: '衰减系数错' };
        return { pass: true, detail: 'p·(1−lr·wd) 逐位一致' };
      } },
    { key: 'H007', name: 'quant_absmax_scale', sig: '(x)->scale', ref: '对称 absmax（v5/v12 同源）', complexity: 'O(n)',
      fn: function (x) { var m = 0; for (var i = 0; i < x.length; i++) { var a = Math.abs(x[i]); if (a > m) m = a; } return m === 0 ? 1 : m / 127; },
      selfTest: function (rng) {
        var z = new Float64Array(5);
        if (this.fn(z) !== 1) return { pass: false, detail: '全零应返回 1（避免除零）' };
        for (var t = 0; t < 20; t++) { var n = 2 + ((rng() * 40) | 0), x = rv(rng, n, 50), s = this.fn(x), m = 0; for (var i = 0; i < n; i++) m = Math.max(m, Math.abs(x[i])); if (!Object.is(s, m / 127)) return { pass: false, detail: 'scale != absmax/127' }; if (s <= 0) return { pass: false, detail: 'scale 应为正' }; }
        return { pass: true, detail: 'absmax/127 + 全零保护' };
      } },
    { key: 'H008', name: 'quant_q8_row', sig: '(x,scale)->Int8', ref: 'Q8_0 行量化', complexity: 'O(n)',
      fn: function (x, scale) { var q = new Int8Array(x.length); for (var i = 0; i < x.length; i++) { var r = x[i] / scale; r = r >= 0 ? Math.round(r) : -Math.round(-r); q[i] = r > 127 ? 127 : (r < -127 ? -127 : r); } return q; },
      selfTest: function (rng) {
        var x = new Float64Array([1, -1, 0.5, -0.5, 0]), s = 1 / 127;
        var q = this.fn(x, s);
        if (!(q[0] === 127 && q[1] === -127 && q[2] === 64 && q[3] === -64 && q[4] === 0)) return { pass: false, detail: '量化值错' };
        for (var t = 0; t < 20; t++) { var n = 2 + ((rng() * 40) | 0), y = rv(rng, n, 100), sc = this.fn(y) ? this.fn(y) : 1; var s2 = (NS.L10.run('H007', [y])); var r = this.fn(y, s2); for (var i = 0; i < n; i++) if (r[i] < -127 || r[i] > 127) return { pass: false, detail: '码字越界' }; }
        var over = this.fn(new Float64Array([1000]), 1);
        if (over[0] !== 127) return { pass: false, detail: '大值应钳位 127' };
        return { pass: true, detail: '码字 ∈[−127,127] + 钳位正确' };
      } },
    { key: 'H009', name: 'dequant_q8_row', sig: '(q,scale)->x', ref: 'Q8_0 反量化', complexity: 'O(n)',
      fn: function (q, scale) { var x = new Float64Array(q.length); for (var i = 0; i < q.length; i++) x[i] = q[i] * scale; return x; },
      selfTest: function (rng) {
        // 往返误差硬界：|x − deq(quant(x))| ≤ absmax/254（v12 KV INT8 同款硬界）
        for (var t = 0; t < 20; t++) {
          var n = 2 + ((rng() * 60) | 0), x = rv(rng, n, 1000), s = NS.L10.run('H007', [x]), q = NS.L10.run('H008', [x, s]), d = this.fn(q, s), m = 0, maxErr = 0;
          for (var i = 0; i < n; i++) { m = Math.max(m, Math.abs(x[i])); maxErr = Math.max(maxErr, Math.abs(x[i] - d[i])); }
          if (maxErr > m / 254 + 1e-12) return { pass: false, detail: '往返误差超硬界: ' + maxErr + ' > ' + (m / 254) };
        }
        var q = new Int8Array([127, -127, 0]);
        var d = this.fn(q, 0.5);
        if (!(d[0] === 63.5 && d[1] === -63.5 && d[2] === 0)) return { pass: false, detail: '反量化值错' };
        return { pass: true, detail: '|x−deq(q)| ≤ absmax/254 硬界成立' };
      } },
    { key: 'H010', name: 'stochastic_round', sig: '(x,rng)->int', ref: '随机舍入（量化训练）', complexity: 'O(1)',
      fn: function (x, rng) { var f = Math.floor(x), frac = x - f; return f + (rng() < frac ? 1 : 0); },
      selfTest: function (rng) {
        // 期望无偏：E[round(x)] = x
        var r1 = mulberry32(2024), sum = 0, N = 4000;
        for (var i = 0; i < N; i++) sum += this.fn(10.3, r1);
        if (Math.abs(sum / N - 10.3) > 0.02) return { pass: false, detail: '期望偏离: ' + (sum / N) };
        // 确定性：同种子流逐位一致
        var ra = mulberry32(7), rb = mulberry32(7), same = true;
        for (var t = 0; t < 50; t++) if (this.fn(3.7, ra) !== this.fn(3.7, rb)) { same = false; break; }
        if (!same) return { pass: false, detail: '同种子不一致' };
        var r2 = mulberry32(9);
        if (this.fn(5, r2) !== 5) return { pass: false, detail: '整数输入应恒等' };
        return { pass: true, detail: 'E[round]=x（±0.02）+ 确定性 + 整数恒等' };
      } },
    { key: 'H011', name: 'topk_filter', sig: '(x,k)->filtered', ref: 'top-k 采样（llama.cpp 语义）', complexity: 'O(n log n)',
      fn: function (x, k) {
        var n = x.length, idx = [];
        for (var i = 0; i < n; i++) idx.push(i);
        idx.sort(function (a, b) { return x[b] - x[a] || a - b; }); // 稳定：值降序，并列索引升序
        var keep = Object.create(null);
        for (var j = 0; j < k && j < n; j++) keep[idx[j]] = true;
        var y = new Float64Array(n);
        for (var m = 0; m < n; m++) y[m] = keep[m] ? x[m] : -Infinity;
        return y;
      },
      selfTest: function (rng) {
        var x = new Float64Array([1, 5, 3, 5, 2]), y = this.fn(x, 2);
        var kept = []; for (var i = 0; i < 5; i++) if (y[i] !== -Infinity) kept.push(i);
        if (kept.length !== 2 || kept[0] !== 1 || kept[1] !== 3) return { pass: false, detail: 'top-2 应取索引 1,3（并列取前）' };
        var all = this.fn(x, 10), cnt = 0; for (var j = 0; j < 5; j++) if (all[j] !== -Infinity) cnt++;
        if (cnt !== 5) return { pass: false, detail: 'k>=n 应全保留' };
        var none = this.fn(x, 0), c2 = 0; for (var k = 0; k < 5; k++) if (none[k] !== -Infinity) c2++;
        if (c2 !== 0) return { pass: false, detail: 'k=0 应全滤除' };
        for (var t = 0; t < 20; t++) { var n = 2 + ((rng() * 30) | 0), v = rv(rng, n, 50), kk = (rng() * (n + 2)) | 0, r = this.fn(v, kk), ref = v.slice().sort(function (a, b) { return b - a; }), expect = Math.min(kk, n), got = 0; for (var m = 0; m < n; m++) if (r[m] !== -Infinity) got++; if (got !== expect) return { pass: false, detail: '保留数错' }; for (var p = 0; p < n; p++) if (r[p] !== -Infinity && r[p] < ref[expect - 1] - 1e-12) return { pass: false, detail: '保留的不是 top-k' }; }
        return { pass: true, detail: 'top-k 稳定选取 + k=0/k≥n 边界正确' };
      } },
    { key: 'H012', name: 'topp_filter', sig: '(x,p)->filtered', ref: 'nucleus 采样', complexity: 'O(n log n)',
      fn: function (x, p) {
        var n = x.length, m = -Infinity;
        for (var i = 0; i < n; i++) if (x[i] > m) m = x[i];
        var pr = new Float64Array(n), s = 0;
        for (var j = 0; j < n; j++) { pr[j] = Math.exp(x[j] - m); s += pr[j]; }
        for (var k = 0; k < n; k++) pr[k] /= s;
        var idx = []; for (var a = 0; a < n; a++) idx.push(a);
        idx.sort(function (a2, b2) { return pr[b2] - pr[a2] || a2 - b2; });
        var cum = 0, cut = 1;
        for (var c = 0; c < n; c++) { cum += pr[idx[c]]; if (cum >= p) { cut = c + 1; break; } }
        if (p >= 1) cut = n;
        if (p <= 0) cut = 1;
        var keep = Object.create(null);
        for (var d = 0; d < cut; d++) keep[idx[d]] = true;
        var y = new Float64Array(n);
        for (var e = 0; e < n; e++) y[e] = keep[e] ? x[e] : -Infinity;
        return y;
      },
      selfTest: function (rng) {
        var x = new Float64Array([2, 1, 0.5, 0.1]);
        var y = this.fn(x, 0.9);
        var keptSum = 0, kept = 0;
        // 用同一 softmax 参考计算保留概率和
        var m = 2, s = 0, pr = [Math.exp(2 - m), Math.exp(1 - m), Math.exp(0.5 - m), Math.exp(0.1 - m)];
        for (var i = 0; i < 4; i++) s += pr[i];
        for (var j = 0; j < 4; j++) pr[j] /= s;
        for (var k = 0; k < 4; k++) if (y[k] !== -Infinity) { kept++; keptSum += pr[k]; }
        if (keptSum < 0.9 - 1e-12) return { pass: false, detail: '保留概率和未达 p' };
        if (kept > 4) return { pass: false, detail: '保留数越界' };
        var full = this.fn(x, 1), c = 0; for (var a = 0; a < 4; a++) if (full[a] !== -Infinity) c++;
        if (c !== 4) return { pass: false, detail: 'p>=1 应全保留' };
        var one = this.fn(x, 0), c2 = 0; for (var b = 0; b < 4; b++) if (one[b] !== -Infinity) c2++;
        if (c2 !== 1) return { pass: false, detail: 'p<=0 应只保留 top1' };
        var uni = new Float64Array(10).fill(0);
        var uf = this.fn(uni, 0.5), cnt = 0; for (var d = 0; d < 10; d++) if (uf[d] !== -Infinity) cnt++;
        if (cnt !== 5) return { pass: false, detail: '均匀分布 p=0.5 应保留 5 个（得 ' + cnt + '）' };
        return { pass: true, detail: '核概率和 ≥p + 边界正确 + 均匀 p=0.5 → 5 个' };
      } },
    { key: 'H013', name: 'gumbel_max', sig: '(logits,rng)->argmax', ref: 'Gumbel-max 采样', complexity: 'O(n)',
      fn: function (logits, rng) {
        var best = 0, bv = -Infinity;
        for (var i = 0; i < logits.length; i++) { var u = rng(); while (u === 0) u = rng(); var g = logits[i] - Math.log(-Math.log(u)); if (g > bv) { bv = g; best = i; } }
        return best;
      },
      selfTest: function (rng) {
        // 确定性：同种子流结果一致
        var r1 = mulberry32(55), r2 = mulberry32(55), lg = [0.1, 0.5, 0.3, 0.9, 0.2];
        if (this.fn(lg, r1) !== this.fn(lg, r2)) return { pass: false, detail: '同种子不一致' };
        // 均匀 logits → 各 index 都应被采到
        var counts = [0, 0, 0, 0, 0], r3 = mulberry32(99), uni = [0, 0, 0, 0, 0];
        for (var t = 0; t < 4000; t++) counts[this.fn(uni, r3)]++;
        for (var i = 0; i < 5; i++) if (counts[i] < 200) return { pass: false, detail: 'index ' + i + ' 出现过少: ' + counts[i] };
        // 支配 logits → 恒定命中
        var r4 = mulberry32(3), dom = [100, 0, 0, 0, 0], ok = true;
        for (var k = 0; k < 200; k++) if (this.fn(dom, r4) !== 0) { ok = false; break; }
        if (!ok) return { pass: false, detail: '支配 logits 未恒定命中' };
        return { pass: true, detail: '确定性 + 均匀分布覆盖全部 index + 支配命中' };
      } },
    { key: 'H014', name: 'ema_update', sig: '(ema,x,decay)->ema\'', ref: '指数移动平均', complexity: 'O(1)',
      fn: function (ema, x, decay) { return decay * ema + (1 - decay) * x; },
      selfTest: function (rng) {
        for (var t = 0; t < 30; t++) { var e = (rng() - 0.5) * 10, x = (rng() - 0.5) * 10, d = rng(); if (!Object.is(this.fn(e, x, d), d * e + (1 - d) * x)) return { pass: false, detail: '与参考逐位不一致' }; }
        if (this.fn(0, 5, 0.5) !== 2.5) return { pass: false, detail: '初值错' };
        var e = 0; for (var i = 0; i < 3000; i++) e = this.fn(e, 1, 0.99);
        if (Math.abs(e - 1) > 1e-9) return { pass: false, detail: '未收敛到 1' };
        return { pass: true, detail: '逐位一致 + 收敛到目标值' };
      } }
  ];

  for (var i = 0; i < OPS.length; i++) {
    var o = OPS[i];
    o.domain = 'H'; o.impl = 'done';
    L10.register(o);
  }
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

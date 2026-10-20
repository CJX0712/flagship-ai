/* L10 · 微指令层 / D 域：线性代数分解与求解 —— 依赖：无（最底层，自包含）
 * 不变量范式：残差界（‖Av−λv‖、‖A−LU‖、‖QᵀQ−I‖、QR=A），
 * 与 forge 系列（pca-forge Jacobi 1.78e-15）同精度文化。
 * @ref Golub & Van Loan《Matrix Computations》· LAPACK dlarfg/dlarf ·
 * forge 系列既有实现（Jacobi EVD / Cholesky）
 */
;(function (NS) {
  'use strict';
  // 所属层：L10 · 允许依赖：无 · 主要 @ref：Golub & Van Loan / LAPACK
  var L10 = NS.L10;

  function rv(rng, n, scale) { var a = new Float64Array(n), s = scale || 2; for (var i = 0; i < n; i++) a[i] = (rng() - 0.5) * s; return a; }
  function matGet(A, r, c, i, j) { return A[i * c + j]; }
  function matVec(A, r, c, x) { var y = new Float64Array(r); for (var i = 0; i < r; i++) { var s = 0; for (var j = 0; j < c; j++) s += A[i * c + j] * x[j]; y[i] = s; } return y; }
  function matTVec(A, r, c, x) { var y = new Float64Array(c); for (var j = 0; j < c; j++) { var s = 0; for (var i = 0; i < r; i++) s += A[i * c + j] * x[i]; y[j] = s; } return y; }
  function vecNorm(x) { var s = 0; for (var i = 0; i < x.length; i++) s += x[i] * x[i]; return Math.sqrt(s); }
  function vecSub(a, b) { var n = a.length, z = new Float64Array(n); for (var i = 0; i < n; i++) z[i] = a[i] - b[i]; return z; }
  function approx(a, b, tol) { return Math.abs(a - b) <= tol; }

  var OPS = [
    { key: 'D000', name: 'det_2x2', sig: '(a,b,c,d)->number', ref: '行列式', complexity: 'O(1)',
      fn: function (a, b, c, d) { return a * d - b * c; },
      selfTest: function (rng) {
        if (this.fn(1, 2, 3, 4) !== -2) return { pass: false, detail: 'det[[1,2],[3,4]] != -2' };
        if (this.fn(2, 4, 1, 2) !== 0) return { pass: false, detail: '奇异矩阵 det != 0' };
        for (var t = 0; t < 30; t++) { var a = (rng() - 0.5) * 20, b = (rng() - 0.5) * 20, c2 = (rng() - 0.5) * 20, d = (rng() - 0.5) * 20; if (!approx(this.fn(a, b, c2, d), a * d - b * c2, 1e-12 * (1 + Math.abs(a * d) + Math.abs(b * c2)))) return { pass: false, detail: '与参考偏差超界' }; }
        return { pass: true, detail: 'ad−bc 精确' };
      } },
    { key: 'D001', name: 'inv_2x2', sig: '(A4)->inv|null', ref: '伴随矩阵求逆', complexity: 'O(1)',
      fn: function (A) { var det = A[0] * A[3] - A[1] * A[2]; if (det === 0) return null; var id = 1 / det; return [A[3] * id, -A[1] * id, -A[2] * id, A[0] * id]; },
      selfTest: function (rng) {
        var Ai = this.fn([4, 7, 2, 6]);
        if (!Ai) return { pass: false, detail: '可逆矩阵返回 null' };
        var P = [4 * Ai[0] + 7 * Ai[2], 4 * Ai[1] + 7 * Ai[3], 2 * Ai[0] + 6 * Ai[2], 2 * Ai[1] + 6 * Ai[3]];
        if (!(approx(P[0], 1, 1e-12) && approx(P[1], 0, 1e-12) && approx(P[2], 0, 1e-12) && approx(P[3], 1, 1e-12))) return { pass: false, detail: 'A·A⁻¹ != I' };
        if (this.fn([1, 2, 2, 4]) !== null) return { pass: false, detail: '奇异矩阵未返回 null' };
        return { pass: true, detail: 'A·A⁻¹=I（≤1e-12）+ 奇异返回 null' };
      } },
    { key: 'D002', name: 'solve_2x2', sig: '(A4,b2)->x|null', ref: 'Cramer 法则', complexity: 'O(1)',
      fn: function (A, b) { var det = A[0] * A[3] - A[1] * A[2]; if (det === 0) return null; var id = 1 / det; return [(b[0] * A[3] - b[1] * A[1]) * id, (A[0] * b[1] - A[2] * b[0]) * id]; },
      selfTest: function (rng) {
        var x = this.fn([2, 1, 1, 3], [3, 5]); // 2x+y=3, x+3y=5 → x=(4/5, 7/5)
        if (!x) return { pass: false, detail: '返回 null' };
        if (!(approx(x[0], 0.8, 1e-12) && approx(x[1], 1.4, 1e-12))) return { pass: false, detail: '解错误: ' + x };
        if (this.fn([1, 2, 2, 4], [1, 2]) !== null) return { pass: false, detail: '奇异未返回 null' };
        for (var t = 0; t < 20; t++) { var A = rv(rng, 4, 10), b = rv(rng, 2, 10), r = this.fn(A, b); if (r === null) continue; var e1 = A[0] * r[0] + A[1] * r[1] - b[0], e2 = A[2] * r[0] + A[3] * r[1] - b[1]; if (Math.abs(e1) > 1e-10 * (1 + Math.abs(b[0])) || Math.abs(e2) > 1e-10 * (1 + Math.abs(b[1]))) return { pass: false, detail: '残差超界' }; }
        return { pass: true, detail: 'Ax=b 残差 ≤1e-10 相对' };
      } },
    { key: 'D003', name: 'householder_vec', sig: '(x)->{v,alpha}', ref: 'LAPACK dlarfg', complexity: 'O(n)',
      fn: function (x) {
        var n = x.length, nx = vecNorm(x);
        if (nx === 0) return { v: new Float64Array(n), alpha: 0 };
        var alpha = x[0] >= 0 ? -nx : nx;
        var v = Float64Array.from(x); v[0] -= alpha;
        var vv = 0; for (var i = 0; i < n; i++) vv += v[i] * v[i];
        if (vv === 0) return { v: Float64Array.from(x), alpha: alpha };
        var beta = 2 / vv;
        return { v: v, alpha: alpha, beta: beta };
      },
      selfTest: function (rng) {
        for (var t = 0; t < 20; t++) {
          var n = 2 + ((rng() * 6) | 0), x = rv(rng, n, 10), h = this.fn(x);
          // H·x = x − 2v(vᵀx)/(vᵀv) = alpha·e1
          var vx = 0, vv = 0; for (var i = 0; i < n; i++) { vx += h.v[i] * x[i]; vv += h.v[i] * h.v[i]; }
          var Hx = vecSub(x, h.v.map(function (e) { return 2 * vx / vv * e; }));
          if (Math.abs(Hx[0] - h.alpha) > 1e-12 * (1 + vecNorm(x))) return { pass: false, detail: 'Hx[0] != alpha' };
          for (var j = 1; j < n; j++) if (Math.abs(Hx[j]) > 1e-12 * (1 + vecNorm(x))) return { pass: false, detail: 'Hx 尾部分量非 0' };
          if (Math.abs(vecNorm(Hx) - vecNorm(x)) > 1e-12 * (1 + vecNorm(x))) return { pass: false, detail: 'H 不保范数' };
        }
        var z = this.fn(new Float64Array(4));
        if (z.alpha !== 0 || vecNorm(z.v) !== 0) return { pass: false, detail: '零向量分支错' };
        return { pass: true, detail: 'Hx=alpha·e1、保范数、零向量安全' };
      } },
    { key: 'D004', name: 'givens_rot', sig: '(a,b)->{c,s,r}', ref: 'Givens 旋转', complexity: 'O(1)',
      fn: function (a, b) { var r = Math.hypot(a, b); if (r === 0) return { c: 1, s: 0, r: 0 }; return { c: a / r, s: b / r, r: r }; },
      selfTest: function (rng) {
        var g = this.fn(3, 4);
        if (!approx(g.r, 5, 1e-12) || !approx(g.c * g.c + g.s * g.s, 1, 1e-15)) return { pass: false, detail: '3-4-5 或归一化错' };
        for (var t = 0; t < 30; t++) { var a = (rng() - 0.5) * 100, b = (rng() - 0.5) * 100, R = this.fn(a, b); var y1 = R.c * a + R.s * b, y2 = -R.s * a + R.c * b; if (Math.abs(y1 - R.r) > 1e-12 * (1 + R.r) || Math.abs(y2) > 1e-12 * (1 + R.r)) return { pass: false, detail: '旋转未清零第二分量' }; }
        return { pass: true, detail: 'c²+s²=1 且旋转后 =[r,0]' };
      } },
    { key: 'D005', name: 'cholesky_2x2', sig: '(A4)->L|null', ref: 'Cholesky 分解', complexity: 'O(1)',
      fn: function (A) {
        var l00 = Math.sqrt(A[0]);
        if (!isFinite(l00) || A[0] <= 0) return null;
        var l10 = A[2] / l00;
        var rem = A[3] - l10 * l10;
        if (rem <= 0) return null;
        return { L: [l00, l10, Math.sqrt(rem)] }; // [[l00,0],[l10,l11]]
      },
      selfTest: function (rng) {
        var r = this.fn([4, 2, 2, 3]); // A=[[4,2],[2,3]] → L=[[2,0],[1,√2]]
        if (!r) return { pass: false, detail: '正定矩阵返回 null' };
        if (!approx(r.L[0], 2, 1e-12) || !approx(r.L[1], 1, 1e-12) || !approx(r.L[2], Math.SQRT2, 1e-12)) return { pass: false, detail: 'L 值错误' };
        var P = [r.L[0] * r.L[0], r.L[0] * r.L[1], r.L[1] * r.L[0], r.L[1] * r.L[1] + r.L[2] * r.L[2]];
        if (!approx(P[0], 4, 1e-12) || !approx(P[1], 2, 1e-12) || !approx(P[3], 3, 1e-12)) return { pass: false, detail: 'L·Lᵀ != A' };
        if (this.fn([1, 2, 2, 1]) !== null) return { pass: false, detail: '半定矩阵未返回 null' };
        return { pass: true, detail: 'L·Lᵀ=A（≤1e-12）+ 非正定返回 null' };
      } },
    { key: 'D006', name: 'qr_2x2', sig: '(A4)->{Q,R}', ref: 'Householder QR', complexity: 'O(1)',
      fn: function (A) {
        var col0 = [A[0], A[2]], nx = vecNorm(col0);
        if (nx === 0) return { Q: [1, 0, 0, 1], R: [0, A[1], 0, A[3]] };
        var alpha = col0[0] >= 0 ? -nx : nx;
        var v = [col0[0] - alpha, col0[1]], vv = v[0] * v[0] + v[1] * v[1];
        var beta = vv === 0 ? 0 : 2 / vv;
        // R = H·A（H 对称），Q = H；H·x = x − (vᵀx)·β·v（β=2/vᵀv）
        function hvec(x) { var d = (v[0] * x[0] + v[1] * x[1]) * beta; return [x[0] - d * v[0], x[1] - d * v[1]]; }
        var c0 = hvec([A[0], A[2]]), c1 = hvec([A[1], A[3]]);
        return { Q: [1 - beta * v[0] * v[0], -beta * v[0] * v[1], -beta * v[1] * v[0], 1 - beta * v[1] * v[1]], R: [c0[0], c1[0], c0[1], c1[1]] };
      },
      selfTest: function (rng) {
        var A = [1, 2, 3, 4], qr = this.fn(A);
        // R 上三角
        if (Math.abs(qr.R[2]) > 1e-12) return { pass: false, detail: 'R[1][0] 非 0' };
        // QᵀQ = I
        var q00 = qr.Q[0] * qr.Q[0] + qr.Q[2] * qr.Q[2], q01 = qr.Q[0] * qr.Q[1] + qr.Q[2] * qr.Q[3], q11 = qr.Q[1] * qr.Q[1] + qr.Q[3] * qr.Q[3];
        if (!approx(q00, 1, 1e-12) || !approx(q01, 0, 1e-12) || !approx(q11, 1, 1e-12)) return { pass: false, detail: 'Q 非正交' };
        // QR = A
        var p00 = qr.Q[0] * qr.R[0] + qr.Q[1] * qr.R[2], p01 = qr.Q[0] * qr.R[1] + qr.Q[1] * qr.R[3], p10 = qr.Q[2] * qr.R[0] + qr.Q[3] * qr.R[2], p11 = qr.Q[2] * qr.R[1] + qr.Q[3] * qr.R[3];
        if (!approx(p00, A[0], 1e-12) || !approx(p01, A[1], 1e-12) || !approx(p10, A[2], 1e-12) || !approx(p11, A[3], 1e-12)) return { pass: false, detail: 'QR != A' };
        return { pass: true, detail: 'Q 正交 + R 上三角 + QR=A' };
      } },
    { key: 'D007', name: 'jacobi_rot_step', sig: '(a,b,d)->{c,s,a2,d2}', ref: 'Jacobi 旋转（forge 同款）', complexity: 'O(1)',
      fn: function (a, b, d) { var th = 0.5 * Math.atan2(2 * b, a - d); var c = Math.cos(th), s = Math.sin(th); return { c: c, s: s, a2: c * c * a + 2 * s * c * b + s * s * d, d2: s * s * a - 2 * s * c * b + c * c * d }; },
      selfTest: function (rng) {
        for (var t = 0; t < 30; t++) {
          var a = (rng() - 0.5) * 10, b = (rng() - 0.5) * 10, d = (rng() - 0.5) * 10;
          var r = this.fn(a, b, d);
          var scale = 1 + Math.abs(a) + Math.abs(d);
          // 旋转后 off-diag = cs(d−a) + b(c²−s²) 必须归零
          if (Math.abs(r.c * r.s * (d - a) + b * (r.c * r.c - r.s * r.s)) > 1e-13 * scale) return { pass: false, detail: '旋转后 off-diag 非 0' };
          if (Math.abs(r.a2 + r.d2 - (a + d)) > 1e-13 * scale) return { pass: false, detail: 'trace 不守恒' };
          if (Math.abs(r.a2 * r.d2 - (a * d - b * b)) > 1e-12 * scale * scale) return { pass: false, detail: 'det 不守恒' };
        }
        return { pass: true, detail: 'off-diag→0 + trace/det 守恒' };
      } },
    { key: 'D008', name: 'eig_sym_2x2', sig: '(a,b,d)->{l1,l2,v1}', ref: '对称 2x2 闭式特征分解', complexity: 'O(1)',
      fn: function (a, b, d) {
        var tr = a + d, disc = Math.sqrt((a - d) * (a - d) + 4 * b * b);
        var l1 = (tr + disc) / 2, l2 = (tr - disc) / 2;
        var v1;
        if (Math.abs(b) > 1e-300) { v1 = [l1 - d, b]; var nv = vecNorm(v1); v1 = [v1[0] / nv, v1[1] / nv]; }
        else v1 = a >= d ? [1, 0] : [0, 1];
        return { l1: l1, l2: l2, v1: v1 };
      },
      selfTest: function (rng) {
        for (var t = 0; t < 30; t++) {
          var a = (rng() - 0.5) * 10, b = (rng() - 0.5) * 10, d = (rng() - 0.5) * 10;
          var e = this.fn(a, b, d);
          var scale = 1 + Math.abs(a) + Math.abs(b) + Math.abs(d);
          // 残差 ‖Av − λv‖
          var r0 = a * e.v1[0] + b * e.v1[1] - e.l1 * e.v1[0], r1 = b * e.v1[0] + d * e.v1[1] - e.l1 * e.v1[1];
          if (Math.sqrt(r0 * r0 + r1 * r1) > 1e-13 * scale) return { pass: false, detail: '特征向量残差超界' };
          if (!(e.l1 >= e.l2 - 1e-15)) return { pass: false, detail: 'l1 < l2' };
          if (!approx(e.l1 + e.l2, a + d, 1e-12 * scale)) return { pass: false, detail: '特征和 != trace' };
          if (!approx(e.l1 * e.l2, a * d - b * b, 1e-12 * scale * scale)) return { pass: false, detail: '特征积 != det' };
          if (Math.abs(vecNorm(e.v1) - 1) > 1e-15) return { pass: false, detail: '特征向量未归一化' };
        }
        return { pass: true, detail: '残差≤1e-13 + trace/det 守恒 + v 归一化' };
      } },
    { key: 'D009', name: 'power_norm_spec', sig: '(A,r,c,iters)->σ̂', ref: '幂迭代谱范数', complexity: 'O(iters·rc)',
      fn: function (A, r, c, iters) {
        var y = new Float64Array(c); for (var j = 0; j < c; j++) y[j] = 1 / Math.sqrt(c);
        for (var it = 0; it < (iters || 50); it++) {
          var t = matVec(A, r, c, y), ny = vecNorm(t);
          if (ny === 0) return 0;
          y = matTVec(A, r, c, t); var nn = vecNorm(y);
          if (nn === 0) return 0;
          for (var i = 0; i < c; i++) y[i] /= nn;
        }
        return vecNorm(matVec(A, r, c, y));
      },
      selfTest: function (rng) {
        var D = [3, 0, 0, -2]; // diag(3,-2) → σ=3
        var s = this.fn(D, 2, 2, 60);
        if (!approx(s, 3, 1e-10)) return { pass: false, detail: 'diag 谱范数估计错: ' + s };
        for (var t = 0; t < 6; t++) {
          var a = (rng() - 0.5) * 6, b = (rng() - 0.5) * 6, d = (rng() - 0.5) * 6;
          var A = [a, b, b, d], e = (NS.L10 && NS.L10.run('D008', [a, b, d]));
          var est = this.fn(A, 2, 2, 200), exact = Math.max(Math.abs(e.l1), Math.abs(e.l2));
          if (!approx(est, exact, 1e-7 * (1 + exact))) return { pass: false, detail: '与特征值偏差 ' + Math.abs(est - exact) };
        }
        var z = new Float64Array(6);
        if (this.fn(z, 2, 3, 50) !== 0) return { pass: false, detail: '零矩阵谱范数非 0' };
        return { pass: true, detail: 'diag 精确 + 对称随机 ≈ max|λ|（≤1e-7）' };
      } }
  ];

  for (var i = 0; i < OPS.length; i++) {
    var o = OPS[i];
    o.domain = 'D'; o.impl = 'done';
    L10.register(o);
  }
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

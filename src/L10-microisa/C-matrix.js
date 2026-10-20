/* L10 · 微指令层 / C 域：矩阵与 BLAS 微内核 —— 依赖：无（最底层，自包含）
 * 不变量范式：GEMM 等价性（分块/转置/量化版 vs 参考版 maxΔ ≤ 1e-12）、
 * 量化误差硬界 |deq(q)x − Wx| ≤ (absmax/254)·Σ|B|（v5/v12 PTQ 同源）。
 * @ref BLIS 微内核思想 · ggml Q8_0 · LLM.int8()（arXiv:2208.07339）
 */
;(function (NS) {
  'use strict';
  // 所属层：L10 · 允许依赖：无 · 主要 @ref：BLIS / ggml / LLM.int8()
  var L10 = NS.L10;

  function rmat(rng, r, c, scale) {
    var a = new Float64Array(r * c), s = scale || 2;
    for (var i = 0; i < a.length; i++) a[i] = (rng() - 0.5) * s;
    return a;
  }
  function refGemm(A, B, m, k, n) { // i-k-j 独立参考（与 i-j-k 舍入序不同）
    var C = new Float64Array(m * n);
    for (var i = 0; i < m; i++) for (var l = 0; l < k; l++) { var a = A[i * k + l]; if (a === 0) continue; for (var j = 0; j < n; j++) C[i * n + j] += a * B[l * n + j]; }
    return C;
  }
  function maxDiff(A, B) { var d = 0; for (var i = 0; i < A.length; i++) d = Math.max(d, Math.abs(A[i] - B[i])); return d; }

  var OPS = [
    { key: 'C000', name: 'gemm_ref', sig: '(A,B,m,k,n)->C', ref: 'BLAS dgemm', complexity: 'O(mkn)',
      fn: function (A, B, m, k, n) { var C = new Float64Array(m * n); for (var i = 0; i < m; i++) for (var j = 0; j < n; j++) { var s = 0; for (var l = 0; l < k; l++) s += A[i * k + l] * B[l * n + j]; C[i * n + j] = s; } return C; },
      selfTest: function (rng) {
        var A = new Float64Array([1, 2, 3, 4]), B = new Float64Array([5, 6, 7, 8]); // 2x2
        var C = this.fn(A, B, 2, 2, 2);
        if (!(C[0] === 19 && C[1] === 22 && C[2] === 43 && C[3] === 50)) return { pass: false, detail: '2x2 手算错' };
        for (var t = 0; t < 10; t++) { var m = 1 + ((rng() * 6) | 0), k = 1 + ((rng() * 6) | 0), n = 1 + ((rng() * 6) | 0), X = rmat(rng, m, k), Y = rmat(rng, k, n); var d = maxDiff(this.fn(X, Y, m, k, n), refGemm(X, Y, m, k, n)); if (d > 1e-12) return { pass: false, detail: '与 i-k-j 参考偏差 ' + d }; }
        return { pass: true, detail: '2x2 手算精确 + 随机 ≡ 参考（≤1e-12）' };
      } },
    { key: 'C001', name: 'gemv', sig: '(A,x,m,k)->y', ref: 'BLAS dgemv', complexity: 'O(mk)',
      fn: function (A, x, m, k) { var y = new Float64Array(m); for (var i = 0; i < m; i++) { var s = 0; for (var l = 0; l < k; l++) s += A[i * k + l] * x[l]; y[i] = s; } return y; },
      selfTest: function (rng) {
        var A = new Float64Array([1, 2, 3, 4, 5, 6]), x = new Float64Array([1, 1, 1]);
        var y = this.fn(A, x, 2, 3);
        if (!(y[0] === 6 && y[1] === 15)) return { pass: false, detail: '行和错' };
        for (var t = 0; t < 20; t++) { var m = 1 + ((rng() * 6) | 0), k = 1 + ((rng() * 6) | 0), X = rmat(rng, m, k), v = rmat(rng, k, 1), r = this.fn(X, v, m, k), ok = true; for (var i = 0; i < m; i++) { var s = 0; for (var l = 0; l < k; l++) s += X[i * k + l] * v[l]; if (Math.abs(r[i] - s) > 1e-12 * (1 + Math.abs(s))) { ok = false; break; } } if (!ok) return { pass: false, detail: '与参考偏差超界' }; }
        return { pass: true, detail: '行和精确 + 随机 ≡ 参考' };
      } },
    { key: 'C002', name: 'ger', sig: '(A,x,y,m,n)->A', ref: 'BLAS dger（秩1更新）', complexity: 'O(mn)',
      fn: function (A, x, y, m, n) { for (var i = 0; i < m; i++) for (var j = 0; j < n; j++) A[i * n + j] += x[i] * y[j]; return A; },
      selfTest: function (rng) {
        var A = new Float64Array([0, 0, 0, 0]), x = new Float64Array([1, 2]), y = new Float64Array([3, 4]);
        var r = this.fn(Float64Array.from(A), x, y, 2, 2);
        if (!(r[0] === 3 && r[1] === 4 && r[2] === 6 && r[3] === 8)) return { pass: false, detail: '外积错' };
        for (var t = 0; t < 10; t++) { var m = 1 + ((rng() * 5) | 0), n = 1 + ((rng() * 5) | 0), X = rmat(rng, m, n), u = rmat(rng, m, 1), v = rmat(rng, n, 1), R = this.fn(Float64Array.from(X), u, v, m, n); for (var i = 0; i < m; i++) for (var j = 0; j < n; j++) if (!Object.is(R[i * n + j], X[i * n + j] + u[i] * v[j])) return { pass: false, detail: '逐位不等 @' + i + ',' + j }; }
        return { pass: true, detail: 'A += xyᵀ 逐位一致' };
      } },
    { key: 'C003', name: 'gemm_bt', sig: '(A,B,m,k,n)->C=A·Bᵀ', ref: 'BLAS dgemm(transb)', complexity: 'O(mkn)',
      fn: function (A, B, m, k, n) { var C = new Float64Array(m * n); for (var i = 0; i < m; i++) for (var j = 0; j < n; j++) { var s = 0; for (var l = 0; l < k; l++) s += A[i * k + l] * B[j * k + l]; C[i * n + j] = s; } return C; },
      selfTest: function (rng) {
        var A = new Float64Array([1, 2, 3, 4]), B = new Float64Array([1, 0, 0, 1]); // Bᵀ=I → C=A
        var C = this.fn(A, B, 2, 2, 2);
        if (!(C[0] === 1 && C[1] === 2 && C[2] === 3 && C[3] === 4)) return { pass: false, detail: 'Bᵀ=I 时应等于 A' };
        for (var t = 0; t < 10; t++) { var m = 1 + ((rng() * 5) | 0), k = 1 + ((rng() * 5) | 0), n = 1 + ((rng() * 5) | 0), X = rmat(rng, m, k), Y = rmat(rng, n, k), BT = new Float64Array(k * n); for (var i2 = 0; i2 < n; i2++) for (var j2 = 0; j2 < k; j2++) BT[j2 * n + i2] = Y[i2 * k + j2]; var d = maxDiff(this.fn(X, Y, m, k, n), refGemm(X, BT, m, k, n)); if (d > 1e-12) return { pass: false, detail: '与 A·Bᵀ 参考偏差 ' + d }; }
        return { pass: true, detail: 'Bᵀ=I 恒等 + 随机 ≡ 参考（≤1e-12）' };
      } },
    { key: 'C004', name: 'transpose', sig: '(A,r,c)->Aᵀ', ref: '转置', complexity: 'O(rc)',
      fn: function (A, r, c) { var T = new Float64Array(r * c); for (var i = 0; i < r; i++) for (var j = 0; j < c; j++) T[j * r + i] = A[i * c + j]; return T; },
      selfTest: function (rng) {
        var A = new Float64Array([1, 2, 3, 4, 5, 6]);
        var T = this.fn(A, 2, 3);
        if (!(T[0] === 1 && T[1] === 4 && T[2] === 2 && T[3] === 5 && T[4] === 3 && T[5] === 6)) return { pass: false, detail: '2x3 转置错' };
        for (var t = 0; t < 10; t++) { var r = 1 + ((rng() * 5) | 0), c = 1 + ((rng() * 5) | 0), X = rmat(rng, r, c), R = this.fn(X, r, c); for (var i = 0; i < r; i++) for (var j = 0; j < c; j++) if (!Object.is(R[j * r + i], X[i * c + j])) return { pass: false, detail: '逐位不等' }; if (!Object.is(this.fn(R, c, r)[0], X[0]) || maxDiff(this.fn(R, c, r), X) !== 0) return { pass: false, detail: '转置两次不还原' }; }
        return { pass: true, detail: '逐位转置 + (Aᵀ)ᵀ=A' };
      } },
    { key: 'C005', name: 'absmax', sig: '(x)->number', ref: 'PTQ 缩放因子', complexity: 'O(n)',
      fn: function (x) { var m = 0; for (var i = 0; i < x.length; i++) { var a = Math.abs(x[i]); if (a > m) m = a; } return m; },
      selfTest: function (rng) { var t = new Float64Array([1, -9, 3]); if (this.fn(t) !== 9) return { pass: false, detail: 'max|x| 错' }; if (this.fn(new Float64Array(3)) !== 0) return { pass: false, detail: '全零应得 0' }; for (var k = 0; k < 20; k++) { var x = rmat(rng, 1, 60, 100), r = this.fn(x), m = 0; for (var i = 0; i < x.length; i++) m = Math.max(m, Math.abs(x[i])); if (r !== m) return { pass: false, detail: '与参考不一致' }; } return { pass: true, detail: 'max|xᵢ| 一致' }; } },
    { key: 'C006', name: 'absmax_rowwise', sig: '(X,r,c)->s', ref: 'per-row PTQ', complexity: 'O(rc)',
      fn: function (X, r, c) { var s = new Float64Array(r); for (var i = 0; i < r; i++) { var m = 0; for (var j = 0; j < c; j++) { var a = Math.abs(X[i * c + j]); if (a > m) m = a; } s[i] = m; } return s; },
      selfTest: function (rng) { for (var t = 0; t < 10; t++) { var r = 1 + ((rng() * 5) | 0), c = 1 + ((rng() * 6) | 0), X = rmat(rng, r, c, 50), s = this.fn(X, r, c); for (var i = 0; i < r; i++) { var m = 0; for (var j = 0; j < c; j++) m = Math.max(m, Math.abs(X[i * c + j])); if (s[i] !== m) return { pass: false, detail: '行 ' + i + ' 不一致' }; } } return { pass: true, detail: '逐行 max|x| 一致' }; } },
    { key: 'C007', name: 'imatmul_int8', sig: '(Ai,Bi,m,k,n)->Ci32', ref: 'INT8 整数 GEMM', complexity: 'O(mkn)',
      fn: function (Ai, Bi, m, k, n) { var C = new Int32Array(m * n); for (var i = 0; i < m; i++) for (var l = 0; l < k; l++) { var a = Ai[i * k + l]; if (a === 0) continue; for (var j = 0; j < n; j++) C[i * n + j] += a * Bi[l * n + j]; } return C; },
      selfTest: function (rng) {
        var Ai = new Int8Array([1, 2, 3, 4]), Bi = new Int8Array([5, 6, 7, 8]);
        var C = this.fn(Ai, Bi, 2, 2, 2);
        if (!(C[0] === 19 && C[1] === 22 && C[2] === 43 && C[3] === 50)) return { pass: false, detail: '2x2 整数错' };
        for (var t = 0; t < 10; t++) { var m = 1 + ((rng() * 5) | 0), k = 1 + ((rng() * 5) | 0), n = 1 + ((rng() * 5) | 0); var X = new Int8Array(m * k), Y = new Int8Array(k * n); for (var i = 0; i < X.length; i++) X[i] = ((rng() * 255) | 0) - 127; for (var j = 0; j < Y.length; j++) Y[j] = ((rng() * 255) | 0) - 127; var R = this.fn(X, Y, m, k, n); for (var a = 0; a < m; a++) for (var b = 0; b < n; b++) { var s = 0; for (var l = 0; l < k; l++) s += X[a * k + l] * Y[l * n + b]; if (R[a * n + b] !== s) return { pass: false, detail: '整数参考不等' }; } }
        return { pass: true, detail: '整数精确（无舍入）' };
      } },
    { key: 'C008', name: 'quant_gemm_int8', sig: '(A,B,m,k,n)->C≈Â·B', ref: 'LLM.int8() 2208.07339', complexity: 'O(mkn)',
      fn: function (A, B, m, k, n) {
        var C = new Float64Array(m * n);
        for (var i = 0; i < m; i++) {
          var am = 0; for (var l = 0; l < k; l++) { var a = Math.abs(A[i * k + l]); if (a > am) am = a; }
          var sc = am === 0 ? 1 : am / 127; // per-row 对称 absmax
          for (var j = 0; j < n; j++) { var s = 0; for (var l2 = 0; l2 < k; l2++) { var q = Math.max(-127, Math.min(127, Math.round(A[i * k + l2] / sc))); s += q * sc * B[l2 * n + j]; } C[i * n + j] = s; }
        }
        return C;
      },
      selfTest: function (rng) {
        for (var t = 0; t < 8; t++) {
          var m = 1 + ((rng() * 4) | 0), k = 1 + ((rng() * 6) | 0), n = 1 + ((rng() * 4) | 0);
          var A = rmat(rng, m, k, 10), B = rmat(rng, k, n, 3), C = this.fn(A, B, m, k, n);
          for (var i = 0; i < m; i++) {
            var am = 0, absB = 0; for (var l = 0; l < k; l++) { am = Math.max(am, Math.abs(A[i * k + l])); absB += Math.abs(B[l * n + 0]); }
            for (var j = 0; j < n; j++) { var s = 0, sb = 0; for (var l2 = 0; l2 < k; l2++) { s += A[i * k + l2] * B[l2 * n + j]; sb += Math.abs(B[l2 * n + j]); } if (Math.abs(C[i * n + j] - s) > (am / 254) * sb + 1e-12) return { pass: false, detail: '量化误差超硬界' }; }
          }
        }
        return { pass: true, detail: '|deq(q)A·B − A·B| ≤ (absmax/254)·Σ|B|' };
      } },
    { key: 'C009', name: 'mat_norm_fro', sig: '(X)->number', ref: 'Frobenius 范数', complexity: 'O(n)',
      fn: function (X) { var s = 0; for (var i = 0; i < X.length; i++) s += X[i] * X[i]; return Math.sqrt(s); },
      selfTest: function (rng) { var I = new Float64Array([1, 0, 0, 1]); if (Math.abs(this.fn(I) - Math.SQRT2) > 1e-15) return { pass: false, detail: 'I₂ Frobenius != √2' }; for (var t = 0; t < 20; t++) { var X = rmat(rng, 1, 80, 5), r = this.fn(X), s = 0; for (var i = 0; i < X.length; i++) s += X[i] * X[i]; if (Math.abs(r - Math.sqrt(s)) > 1e-12 * (1 + r)) return { pass: false, detail: '与参考偏差超界' }; } return { pass: true, detail: '√Σxᵢ² 一致' }; } },
    { key: 'C010', name: 'add_mat', sig: '(C,A)->C', ref: '矩阵加', complexity: 'O(n)',
      fn: function (C, A) { for (var i = 0; i < C.length; i++) C[i] += A[i]; return C; },
      selfTest: function (rng) { for (var t = 0; t < 20; t++) { var n = 1 + ((rng() * 60) | 0), C0 = rmat(rng, 1, n), A = rmat(rng, 1, n), C = Float64Array.from(C0), r = this.fn(C, A); for (var i = 0; i < n; i++) if (!Object.is(r[i], C0[i] + A[i])) return { pass: false, detail: '逐位不等 @' + i }; } return { pass: true, detail: 'C += A 逐位一致' }; } },
    { key: 'C011', name: 'scale_rows', sig: '(X,r,c,s)->X', ref: 'per-row 缩放', complexity: 'O(rc)',
      fn: function (X, r, c, s) { for (var i = 0; i < r; i++) for (var j = 0; j < c; j++) X[i * c + j] *= s[i]; return X; },
      selfTest: function (rng) { for (var t = 0; t < 10; t++) { var r = 1 + ((rng() * 5) | 0), c = 1 + ((rng() * 6) | 0), X0 = rmat(rng, r, c), s = rmat(rng, r, 1, 4), X = Float64Array.from(X0), R = this.fn(X, r, c, s); for (var i = 0; i < r; i++) for (var j = 0; j < c; j++) if (!Object.is(R[i * c + j], X0[i * c + j] * s[i])) return { pass: false, detail: '逐位不等' }; } return { pass: true, detail: '逐行缩放逐位一致' }; } }
  ];

  for (var i = 0; i < OPS.length; i++) {
    var o = OPS[i];
    o.domain = 'C'; o.impl = 'done';
    L10.register(o);
  }
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

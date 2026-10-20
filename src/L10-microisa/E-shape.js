/* L10 · 微指令层 / E 域：张量形状与索引原语 —— 依赖：无（最底层，自包含）
 * 不变量范式：零拷贝恒等（view 不复制数据，元素序列逐位不变）、
 * 排列可逆（permute∘permute⁻¹ ≡ id）、线性化/反线性化往返。
 * @ref ndarray（scijs，MIT）· ggml ne/nb 布局 · NumPy stride 语义
 */
;(function (NS) {
  'use strict';
  // 所属层：L10 · 允许依赖：无 · 主要 @ref：ndarray / ggml / NumPy
  var L10 = NS.L10;

  function numel(shape) { var n = 1; for (var i = 0; i < shape.length; i++) n *= shape[i]; return n; }
  function randShape(rng, maxDim, maxSize) {
    var d = (rng() * (maxDim + 1)) | 0, s = [];
    for (var i = 0; i < d; i++) s.push(1 + ((rng() * maxSize) | 0));
    return s;
  }
  function randIdx(rng, shape) { var idx = []; for (var i = 0; i < shape.length; i++) idx.push((rng() * shape[i]) | 0); return idx; }

  var OPS = [
    { key: 'E000', name: 'stride_from_shape', sig: '(shape)->strides', ref: 'row-major 布局', complexity: 'O(d)',
      fn: function (shape) { var d = shape.length, st = new Array(d); var s = 1; for (var i = d - 1; i >= 0; i--) { st[i] = s; s *= shape[i]; } return st; },
      selfTest: function (rng) {
        var s1 = this.fn([2, 3]); if (!(s1[0] === 3 && s1[1] === 1)) return { pass: false, detail: '[2,3] strides 错' };
        var s2 = this.fn([2, 3, 4]); if (!(s2[0] === 12 && s2[1] === 4 && s2[2] === 1)) return { pass: false, detail: '[2,3,4] strides 错' };
        if (this.fn([]).length !== 0) return { pass: false, detail: '标量应有空 strides' };
        if (this.fn([5])[0] !== 1) return { pass: false, detail: '向量 stride 应为 1' };
        for (var t = 0; t < 20; t++) { var sh = randShape(rng, 4, 5), st = this.fn(sh); if (st.length !== sh.length) return { pass: false, detail: '维度不符' }; for (var i = 0; i < sh.length; i++) { var expect = 1; for (var j = i + 1; j < sh.length; j++) expect *= sh[j]; if (st[i] !== expect) return { pass: false, detail: 'stride 与 shape 不符' }; } }
        return { pass: true, detail: 'row-major strides 正确' };
      } },
    { key: 'E001', name: 'offset_linearize', sig: '(idx,strides)->offset', ref: '多维索引→偏移', complexity: 'O(d)',
      fn: function (idx, strides) { var o = 0; for (var i = 0; i < idx.length; i++) o += idx[i] * strides[i]; return o; },
      selfTest: function (rng) { for (var t = 0; t < 30; t++) { var sh = randShape(rng, 4, 6), st = NS.L10.run('E000', [sh]), idx = randIdx(rng, sh), o = this.fn(idx, st), ref = 0; for (var i = 0; i < sh.length; i++) ref += idx[i] * st[i]; if (o !== ref) return { pass: false, detail: '与参考不一致' }; if (o < 0 || o >= numel(sh)) return { pass: false, detail: '偏移越界' }; } return { pass: true, detail: '线性化一致且不越界' }; } },
    { key: 'E002', name: 'unravel_index', sig: '(flat,shape)->idx', ref: '偏移→多维索引', complexity: 'O(d)',
      fn: function (flat, shape) { var idx = new Array(shape.length), rem = flat; for (var i = 0; i < shape.length; i++) { var st = 1; for (var j = i + 1; j < shape.length; j++) st *= shape[j]; idx[i] = (rem / st) | 0; rem -= idx[i] * st; } return idx; },
      selfTest: function (rng) { for (var t = 0; t < 30; t++) { var sh = randShape(rng, 4, 6), n = numel(sh), p = (rng() * n) | 0, idx = this.fn(p, sh), st = NS.L10.run('E000', [sh]); var o = 0; for (var i = 0; i < sh.length; i++) { if (idx[i] < 0 || idx[i] >= sh[i]) return { pass: false, detail: '索引越界' }; o += idx[i] * st[i]; } if (o !== p) return { pass: false, detail: '往返不等于原 flat' }; } return { pass: true, detail: 'unravel∘linearize ≡ id' }; } },
    { key: 'E003', name: 'reshape_view', sig: '(shape,newShape)->strides|null', ref: '零拷贝视图', complexity: 'O(d)',
      fn: function (shape, newShape) { if (numel(shape) !== numel(newShape)) return null; var st = new Array(newShape.length), s = 1; for (var i = newShape.length - 1; i >= 0; i--) { st[i] = s; s *= newShape[i]; } return st; },
      selfTest: function (rng) {
        if (this.fn([2, 3], [3, 2]) === null) return { pass: false, detail: '等元素数 reshape 不应失败' };
        if (this.fn([2, 3], [2, 2]) !== null) return { pass: false, detail: '元素数不等应返回 null' };
        for (var t = 0; t < 20; t++) {
          var sh = randShape(rng, 3, 4), n = numel(sh), d2 = 1 + ((rng() * 3) | 0);
          // 随机因数分解：d2 个因子乘积 = n
          var ns = [], rem = n;
          for (var k = 0; k < d2 - 1; k++) { var divs = []; for (var d = 1; d <= rem; d++) if (rem % d === 0) divs.push(d); var pick = divs[(rng() * divs.length) | 0]; ns.push(pick); rem /= pick; }
          ns.push(rem);
          var st = this.fn(sh, ns);
          if (st === null) return { pass: false, detail: '合法 reshape 返回 null' };
          // 元素序列逐位不变：新视图线性化第 p 个元素 === p
          for (var p = 0; p < n; p++) { var idx = NS.L10.run('E002', [p, ns]), o = 0; for (var i2 = 0; i2 < ns.length; i2++) o += idx[i2] * st[i2]; if (o !== p) return { pass: false, detail: 'reshape 改变元素顺序 @' + p }; }
        }
        return { pass: true, detail: 'reshape 保持元素顺序（零拷贝恒等）' };
      } },
    { key: 'E004', name: 'permute_axes', sig: '(shape,strides,perm)->{shape,strides}', ref: '轴排列', complexity: 'O(d)',
      fn: function (shape, strides, perm) { var ns = [], nst = []; for (var i = 0; i < perm.length; i++) { ns.push(shape[perm[i]]); nst.push(strides[perm[i]]); } return { shape: ns, strides: nst }; },
      selfTest: function (rng) {
        for (var t = 0; t < 20; t++) {
          var sh = randShape(rng, 4, 5), st = NS.L10.run('E000', [sh]), perm = [];
          for (var i = 0; i < sh.length; i++) perm.push(i);
          // 随机排列（Fisher-Yates，用确定性步骤）
          for (var k = perm.length - 1; k > 0; k--) { var j = (rng() * (k + 1)) | 0; var tmp = perm[k]; perm[k] = perm[j]; perm[j] = tmp; }
          var p = this.fn(sh, st, perm), idx = randIdx(rng, sh);
          var o1 = 0; for (var i2 = 0; i2 < sh.length; i2++) o1 += idx[i2] * st[i2];
          var idxp = []; for (var i3 = 0; i3 < perm.length; i3++) idxp.push(idx[perm[i3]]);
          var o2 = 0; for (var i4 = 0; i4 < idxp.length; i4++) o2 += idxp[i4] * p.strides[i4];
          if (o1 !== o2) return { pass: false, detail: 'permute 改变元素地址' };
          // permute∘permute⁻¹ ≡ id
          var inv = new Array(perm.length); for (var i5 = 0; i5 < perm.length; i5++) inv[perm[i5]] = i5;
          var back = this.fn(p.shape, p.strides, inv);
          for (var i6 = 0; i6 < sh.length; i6++) if (back.shape[i6] !== sh[i6] || back.strides[i6] !== st[i6]) return { pass: false, detail: '逆排列不还原' };
        }
        return { pass: true, detail: '地址不变 + 逆排列还原（逐位）' };
      } },
    { key: 'E005', name: 'slice_dim', sig: '(shape,strides,dim,start,stop)->view', ref: '零拷贝切片', complexity: 'O(d)',
      fn: function (shape, strides, dim, start, stop) { if (start < 0 || stop > shape[dim] || start > stop) return null; var ns = shape.slice(), nst = strides.slice(); ns[dim] = stop - start; return { shape: ns, strides: nst, offset: start * strides[dim] }; },
      selfTest: function (rng) {
        var sh = [4, 5], st = NS.L10.run('E000', [sh]);
        var v = this.fn(sh, st, 0, 1, 3);
        if (!v || v.shape[0] !== 2 || v.shape[1] !== 5 || v.offset !== st[0]) return { pass: false, detail: '切片视图错' };
        if (this.fn(sh, st, 0, 3, 2) !== null) return { pass: false, detail: 'start>stop 应返回 null' };
        if (this.fn(sh, st, 1, 0, 9) !== null) return { pass: false, detail: '越界应返回 null' };
        // 切片内元素地址 = 原地址 + offset
        for (var t = 0; t < 20; t++) { var s2 = randShape(rng, 3, 5); if (!s2.length) continue; var st2 = NS.L10.run('E000', [s2]), dim = (rng() * s2.length) | 0, a = (rng() * s2[dim]) | 0, b2 = a + ((rng() * (s2[dim] - a + 1)) | 0), vv = this.fn(s2, st2, dim, a, b2); if (!vv) continue; var idx = randIdx(rng, vv.shape), o = vv.offset; for (var i = 0; i < idx.length; i++) o += idx[i] * vv.strides[i]; var orig = idx.slice(); orig[dim] += a; var oref = 0; for (var j = 0; j < s2.length; j++) oref += orig[j] * st2[j]; if (o !== oref) return { pass: false, detail: '切片地址错位' }; }
        return { pass: true, detail: '视图地址 = 原地址+offset' };
      } },
    { key: 'E006', name: 'concat_rows', sig: '(A,ra,ca,B,rb,cb)->C|null', ref: '行拼接', complexity: 'O((ra+rb)ca)',
      fn: function (A, ra, ca, B, rb, cb) { if (ca !== cb) return null; var C = new Float64Array((ra + rb) * ca); for (var i = 0; i < ra * ca; i++) C[i] = A[i]; for (var j = 0; j < rb * cb; j++) C[ra * ca + j] = B[j]; return C; },
      selfTest: function (rng) {
        var A = new Float64Array([1, 2, 3, 4]), B = new Float64Array([5, 6]);
        var C = this.fn(A, 2, 2, B, 1, 2);
        if (!C || C.length !== 6 || !(C[0] === 1 && C[2] === 3 && C[4] === 5 && C[5] === 6)) return { pass: false, detail: '拼接错' };
        if (this.fn(A, 2, 2, B, 1, 3) !== null) return { pass: false, detail: '列数不等应返回 null' };
        for (var t = 0; t < 10; t++) { var ra = 1 + ((rng() * 4) | 0), rb = 1 + ((rng() * 4) | 0), ca = 1 + ((rng() * 4) | 0), X = new Float64Array(ra * ca), Y = new Float64Array(rb * ca); for (var i = 0; i < X.length; i++) X[i] = (rng() - 0.5) * 10; for (var j = 0; j < Y.length; j++) Y[j] = (rng() - 0.5) * 10; var R = this.fn(X, ra, ca, Y, rb, ca); for (var k = 0; k < ra * ca; k++) if (!Object.is(R[k], X[k])) return { pass: false, detail: 'A 区逐位不等' }; for (var m = 0; m < rb * ca; m++) if (!Object.is(R[ra * ca + m], Y[m])) return { pass: false, detail: 'B 区逐位不等' }; }
        return { pass: true, detail: 'A|B 逐位拼接 + 列数校验' };
      } },
    { key: 'E007', name: 'pad', sig: '(X,r,c,pt,pb,pl,pr,v)->P', ref: '常量填充', complexity: 'O((r+pt+pb)(c+pl+pr))',
      fn: function (X, r, c, pt, pb, pl, pr, v) { var R = r + pt + pb, C = c + pl + pr, P = new Float64Array(R * C); for (var i = 0; i < R; i++) for (var j = 0; j < C; j++) { var si = i - pt, sj = j - pl; P[i * C + j] = (si >= 0 && si < r && sj >= 0 && sj < c) ? X[si * c + sj] : v; } return P; },
      selfTest: function (rng) {
        var X = new Float64Array([1, 2, 3, 4]);
        var P = this.fn(X, 2, 2, 1, 1, 1, 1, 0);
        if (P.length !== 16 || P[1 * 4 + 1] !== 1 || P[1 * 4 + 2] !== 2 || P[2 * 4 + 1] !== 3 || P[2 * 4 + 2] !== 4) return { pass: false, detail: '内部错位' };
        if (P[0] !== 0 || P[15] !== 0 || P[1 * 4] !== 0) return { pass: false, detail: '边框未填充' };
        for (var t = 0; t < 10; t++) { var r = 1 + ((rng() * 4) | 0), c = 1 + ((rng() * 4) | 0), Y = new Float64Array(r * c); for (var i = 0; i < Y.length; i++) Y[i] = (rng() - 0.5) * 10; var pt = (rng() * 3) | 0, pl = (rng() * 3) | 0, v = (rng() - 0.5) * 5, R = this.fn(Y, r, c, pt, pt, pl, pl, v), R2 = r + 2 * pt, C2 = c + 2 * pl; for (var a = 0; a < R2; a++) for (var b2 = 0; b2 < C2; b2++) { var si = a - pt, sj = b2 - pl; var expect = (si >= 0 && si < r && sj >= 0 && sj < c) ? Y[si * c + sj] : v; if (!Object.is(R[a * C2 + b2], expect)) return { pass: false, detail: '逐位不等 @' + a + ',' + b2 }; } }
        return { pass: true, detail: '内部逐位保留 + 边框=v' };
      } },
    { key: 'E008', name: 'one_hot', sig: '(i,n)->vector', ref: '独热编码', complexity: 'O(n)',
      fn: function (i, n) { var v = new Float64Array(n); if (i >= 0 && i < n) v[i] = 1; return v; },
      selfTest: function (rng) {
        var v = this.fn(2, 4);
        if (!(v[0] === 0 && v[1] === 0 && v[2] === 1 && v[3] === 0)) return { pass: false, detail: 'one_hot(2,4) 错' };
        for (var t = 0; t < 20; t++) { var n = 1 + ((rng() * 10) | 0), i = (rng() * n) | 0, w = this.fn(i, n), s = 0; for (var k = 0; k < n; k++) s += w[k]; if (s !== 1 || w[i] !== 1) return { pass: false, detail: '和不为 1 或位置错' }; }
        var bad = this.fn(9, 4); var s2 = 0; for (var j = 0; j < 4; j++) s2 += bad[j]; if (s2 !== 0) return { pass: false, detail: '越界 i 应得全零' };
        return { pass: true, detail: 'Σ=1 + 越界安全' };
      } },
    { key: 'E009', name: 'arange', sig: '(start,stop,step)->vector', ref: '等差序列', complexity: 'O(n)',
      fn: function (start, stop, step) { if (step === 0 || (step > 0 && start >= stop) || (step < 0 && start <= stop)) return new Float64Array(0); var n = Math.ceil((stop - start) / step), a = new Float64Array(n); for (var i = 0; i < n; i++) a[i] = start + i * step; return a; },
      selfTest: function (rng) {
        var a = this.fn(0, 5, 1);
        if (a.length !== 5 || a[0] !== 0 || a[4] !== 4) return { pass: false, detail: 'arange(0,5,1) 错' };
        var b = this.fn(5, 0, -2);
        if (b.length !== 3 || b[0] !== 5 || b[2] !== 1) return { pass: false, detail: '负步长错' };
        var c = this.fn(0, 1, 0.25);
        if (c.length !== 4 || c[3] !== 0.75) return { pass: false, detail: '小数步长错' };
        if (this.fn(1, 5, 0).length !== 0) return { pass: false, detail: '零步长应返回空' };
        if (this.fn(3, 3, 1).length !== 0) return { pass: false, detail: '空区间应返回空' };
        return { pass: true, detail: '端点/步长/边界正确' };
      } },
    { key: 'E010', name: 'linspace', sig: '(a,b,n)->vector', ref: '等距序列（含端点）', complexity: 'O(n)',
      fn: function (a, b, n) { if (n <= 0) return new Float64Array(0); if (n === 1) return new Float64Array([a]); var v = new Float64Array(n), d = (b - a) / (n - 1); for (var i = 0; i < n; i++) v[i] = a + i * d; v[n - 1] = b; return v; },
      selfTest: function (rng) {
        var v = this.fn(0, 1, 5);
        if (v.length !== 5 || v[0] !== 0 || v[4] !== 1 || v[2] !== 0.5) return { pass: false, detail: 'linspace(0,1,5) 错' };
        if (this.fn(3, 7, 1)[0] !== 3) return { pass: false, detail: 'n=1 应返回 [a]' };
        if (this.fn(0, 1, 0).length !== 0) return { pass: false, detail: 'n=0 应返回空' };
        for (var t = 0; t < 20; t++) { var a = (rng() - 0.5) * 20, b2 = (rng() - 0.5) * 20, n = 2 + ((rng() * 8) | 0), w = this.fn(a, b2, n); if (w[0] !== a || w[n - 1] !== b2) return { pass: false, detail: '端点不精确' }; var step = w[1] - w[0]; for (var i = 2; i < n; i++) if (Math.abs((w[i] - w[i - 1]) - step) > 1e-12 * (1 + Math.abs(step))) return { pass: false, detail: '非等距' }; }
        return { pass: true, detail: '端点逐位精确 + 等距' };
      } },
    { key: 'E011', name: 'repeat_kv_index', sig: '(h,nHead,nKvHead)->kvh', ref: 'GQA 头映射', complexity: 'O(1)',
      fn: function (h, nHead, nKvHead) { if (nKvHead <= 0 || nHead % nKvHead !== 0 || h < 0 || h >= nHead) return null; var g = nHead / nKvHead; return (h / g) | 0; },
      selfTest: function (rng) {
        if (this.fn(0, 8, 2) !== 0 || this.fn(3, 8, 2) !== 0 || this.fn(4, 8, 2) !== 1 || this.fn(7, 8, 2) !== 1) return { pass: false, detail: '8头→2KV头 映射错' };
        if (this.fn(2, 4, 4) !== 2) return { pass: false, detail: 'MHA 应恒等映射' };
        if (this.fn(0, 8, 3) !== null) return { pass: false, detail: '不整除应返回 null' };
        if (this.fn(8, 8, 2) !== null) return { pass: false, detail: '越界应返回 null' };
        for (var t = 0; t < 20; t++) { var kv = 1 + ((rng() * 4) | 0), nh = kv * (1 + ((rng() * 4) | 0)), h = (rng() * nh) | 0, r = this.fn(h, nh, kv), ref = Math.floor(h / (nh / kv)); if (r !== ref) return { pass: false, detail: '与参考不一致' }; if (r < 0 || r >= kv) return { pass: false, detail: 'kvh 越界' }; }
        return { pass: true, detail: 'GQA 分组映射正确 + 边界安全' };
      } },
    { key: 'E012', name: 'broadcast_ok', sig: '(sa,sb)->boolean', ref: '广播兼容判定', complexity: 'O(d)',
      fn: function (sa, sb) { var n = Math.max(sa.length, sb.length); for (var i = 0; i < n; i++) { var da = sa.length - 1 - i, db = sb.length - 1 - i; var x = da >= 0 ? sa[da] : 1, y = db >= 0 ? sb[db] : 1; if (x !== y && x !== 1 && y !== 1) return false; } return true; },
      selfTest: function (rng) {
        if (!this.fn([1, 4], [3, 4])) return { pass: false, detail: '[1,4]vs[3,4] 应兼容' };
        if (!this.fn([3, 1], [3, 5])) return { pass: false, detail: '[3,1]vs[3,5] 应兼容' };
        if (this.fn([2], [3])) return { pass: false, detail: '[2]vs[3] 不应兼容' };
        if (!this.fn([], [3, 4])) return { pass: false, detail: '标量应广播兼容' };
        if (!this.fn([2, 3], [2, 3])) return { pass: false, detail: '同形应兼容' };
        for (var t = 0; t < 30; t++) { var sa = randShape(rng, 3, 4), sb = randShape(rng, 3, 4), r = this.fn(sa, sb), n = Math.max(sa.length, sb.length), ref = true; for (var i = 0; i < n; i++) { var da = sa.length - 1 - i, db = sb.length - 1 - i, x = da >= 0 ? sa[da] : 1, y = db >= 0 ? sb[db] : 1; if (x !== y && x !== 1 && y !== 1) { ref = false; break; } } if (r !== ref) return { pass: false, detail: '与参考不一致' }; }
        return { pass: true, detail: '广播规则逐项一致' };
      } }
  ];

  for (var i = 0; i < OPS.length; i++) {
    var o = OPS[i];
    o.domain = 'E'; o.impl = 'done';
    L10.register(o);
  }
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

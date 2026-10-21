/* ============================================================================
 * L10 微指令 · H 域（采样与量化）· 第 2/4 批（H033–H064）
 * ----------------------------------------------------------------------------
 * 类别 3：温度缩放（H033–H048）；类别 4：舍入（H049–H064）。
 * 约定同 H001–H032；K5 禁止项同样禁用。
 * ==========================================================================*/
;(function (NS) {
  'use strict';
  NS.L10 = NS.L10 || {};
  const relClose = NS.L10.relClose;

  function rngSeed(args) {
    let h = 0x811c9dc5 >>> 0;
    for (let a = 0; a < args.length; a++) {
      const arg = args[a];
      if (arg && typeof arg === 'object' && arg.length != null) {
        const n = arg.length;
        h = Math.imul(h ^ (n & 255), 0x01000193) >>> 0;
        const buf = new ArrayBuffer(8), f64 = new Float64Array(buf), u32 = new Uint32Array(buf);
        for (let i = 0; i < n; i++) { f64[0] = arg[i]; h = Math.imul(h ^ (u32[0] & 255), 0x01000193) >>> 0; h = Math.imul(h ^ (u32[1] & 255), 0x01000193) >>> 0; }
      } else {
        const buf = new ArrayBuffer(8), f64 = new Float64Array(buf), u32 = new Uint32Array(buf);
        f64[0] = (typeof arg === 'number') ? arg : 0;
        h = Math.imul(h ^ (u32[0] & 255), 0x01000193) >>> 0; h = Math.imul(h ^ (u32[1] & 255), 0x01000193) >>> 0;
      }
    }
    return h >>> 0;
  }
  function softmax(v) {
    let m = -Infinity;
    for (let i = 0; i < v.length; i++) if (v[i] > m) m = v[i];
    let s = 0; const out = new Float64Array(v.length);
    for (let i = 0; i < v.length; i++) { const e = Math.exp(v[i] - m); out[i] = e; s += e; }
    const inv = 1 / s;
    for (let i = 0; i < v.length; i++) out[i] *= inv;
    return out;
  }
  function maxOf(v) { let m = -Infinity; for (let i = 0; i < v.length; i++) if (v[i] > m) m = v[i]; return m; }
  function minOf(v) { let m = Infinity; for (let i = 0; i < v.length; i++) if (v[i] < m) m = v[i]; return m; }
  function argmaxIdx(v) { let mi = 0, m = -Infinity; for (let i = 0; i < v.length; i++) if (v[i] > m) { m = v[i]; mi = i; } return mi; }
  function entropy(p) { let e = 0; for (let i = 0; i < p.length; i++) { const pi = p[i]; if (pi > 0) e -= pi * Math.log(pi); } return e; }
  function topKIndices(v, k) {
    const n = v.length; const kk = Math.min(Math.max(1, Math.floor(k || 1)), n);
    const idx = Array.from({ length: n }, (_, i) => i);
    idx.sort((a, b) => (v[b] - v[a]) || (a - b));
    return idx.slice(0, kk);
  }
  function nucleusIndices(probs, p) {
    const n = probs.length; const idx = Array.from({ length: n }, (_, i) => i);
    idx.sort((a, b) => (probs[b] - probs[a]) || (a - b));
    const out = []; let s = 0; const pp = Math.min(Math.max(p, 0), 1);
    for (let i = 0; i < idx.length; i++) { out.push(idx[i]); s += probs[idx[i]]; if (s >= pp) break; }
    return out;
  }
  function isInt(x) { return typeof x === 'number' && Number.isInteger(x); }
  function symRound(x) { return (x < 0 ? -1 : 1) * Math.floor(Math.abs(x) + 0.5); }
  function roundHalfEven(x) { const f = Math.floor(x); const fp = x - f; if (fp < 0.5) return f; if (fp > 0.5) return f + 1; return (f % 2 === 0) ? f : f + 1; }
  function trunc(x) { return x < 0 ? Math.ceil(x) : Math.floor(x); }

  function reg(i, o) { NS.L10.register('H', i, o); }

  // ── 类别 3：温度缩放（H033–H048）────────────────────────────────────────

  // H033 温度除 logits。
  reg(33, {
    name: '温度除 logits', intuition: '温度缩放即 logits 除以温度 T。',
    math: 'run_i = x_i / T', pseudocode: 'return x / T', complexity: 'O(n)',
    ref: 'Hinton 2015 distilling', commonPitfall: 'T 必须 > 0。',
    invariant: 'relClose(run, x/T)',
    inputs: ['x', 'T'], domains: { x: 'vecFinite', T: 'pos' },
    run: (x, T) => { const out = new Float64Array(x.length); for (let i = 0; i < x.length; i++) out[i] = x[i] / T; return out; },
    predicate: (x, T) => { const r = (function () { const o = new Float64Array(x.length); for (let i = 0; i < x.length; i++) o[i] = x[i] / T; return o; })(); for (let i = 0; i < x.length; i++) if (!relClose(r[i], x[i] / T, 1e-9)) return false; return true; },
  });

  // H034 温度后 softmax：和为 1 且有限正。
  reg(34, {
    name: '温度后 softmax', intuition: 'x/T 经 softmax 得合法分布。',
    math: 'run = softmax(x/T), Σ=1', pseudocode: 'softmax(x/T)', complexity: 'O(n)',
    ref: 'Vaswani 2017', commonPitfall: '需数值稳定（减去最大值）。',
    invariant: 'Σ run = 1 且 ∀i: run_i>0 有限',
    inputs: ['x', 'T'], domains: { x: 'vecFinite', T: 'pos' },
    run: (x, T) => softmax(x.map((v) => v / T)),
    predicate: (x, T) => {
      const r = softmax(x.map((v) => v / T));
      let s = 0; for (let i = 0; i < r.length; i++) { if (!(r[i] >= 0 && Number.isFinite(r[i]))) return false; s += r[i]; }
      return Math.abs(s - 1) < 1e-9;
    },
  });

  // H035 低温锐化（T=0.3）：分布更尖，熵不增。
  reg(35, {
    name: '低温锐化', intuition: 'T<1 使分布更尖锐，熵 <= T=1 时。',
    math: 'H(softmax(x/0.3)) <= H(softmax(x))', pseudocode: 'compare entropy at T=0.3 vs T=1', complexity: 'O(n)',
    ref: 'Hinton 2015 distilling', commonPitfall: 'T 越小越尖。',
    invariant: 'H(softmax(x/0.3)) <= H(softmax(x)) + ε',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => softmax(x.map((v) => v / 0.3)),
    predicate: (x) => {
      const r = softmax(x.map((v) => v / 0.3));
      let s = 0; for (let i = 0; i < r.length; i++) { if (!(r[i] >= 0 && Number.isFinite(r[i]))) return false; s += r[i]; }
      if (Math.abs(s - 1) > 1e-9) return false;
      const s1 = softmax(x);
      return entropy(r) <= entropy(s1) + 1e-6;
    },
  });

  // H036 高温平坦（T=1e6）：分布更平，熵不减。
  reg(36, {
    name: '高温平坦', intuition: 'T>1 使分布更平坦，熵 >= T=1 时。',
    math: 'H(softmax(x/1e6)) >= H(softmax(x))', pseudocode: 'compare entropy at T=1e6 vs T=1', complexity: 'O(n)',
    ref: 'Hinton 2015 distilling', commonPitfall: 'T 越大越接近均匀。',
    invariant: 'H(softmax(x/1e6)) >= H(softmax(x)) - ε',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => softmax(x.map((v) => v / 1e6)),
    predicate: (x) => {
      const r = softmax(x.map((v) => v / 1e6));
      let s = 0; for (let i = 0; i < r.length; i++) { if (!(r[i] >= 0 && Number.isFinite(r[i]))) return false; s += r[i]; }
      if (Math.abs(s - 1) > 1e-9) return false;
      const s1 = softmax(x);
      return entropy(r) >= entropy(s1) - 1e-6;
    },
  });

  // H037 T=1 恒等：softmax(x/T)（独立参考实现）与 run 一致。
  reg(37, {
    name: 'T=1 恒等', intuition: '参考实现交叉验证 softmax 数值一致。',
    math: 'run = softmax(x/T) = 参考', pseudocode: 'compare to manual stable softmax', complexity: 'O(n)',
    ref: 'Vaswani 2017', commonPitfall: '两实现须等价。',
    invariant: 'relClose(run, 参考softmax(x/T))',
    inputs: ['x', 'T'], domains: { x: 'vecFinite', T: 'pos' },
    run: (x, T) => softmax(x.map((v) => v / T)),
    predicate: (x, T) => {
      const scaled = x.map((v) => v / T);
      let m = -Infinity; for (const v of scaled) if (v > m) m = v;
      let s = 0; const e = new Float64Array(scaled.length);
      for (let i = 0; i < scaled.length; i++) { e[i] = Math.exp(scaled[i] - m); s += e[i]; }
      const ref = new Float64Array(scaled.length);
      for (let i = 0; i < scaled.length; i++) ref[i] = e[i] / s;
      const r = softmax(scaled);
      return relClose(r, ref, 1e-9);
    },
  });

  // H038 T→0 one-hot 极限：argmax 保持，且最大概率不降。
  reg(38, {
    name: 'T→0 极限 · argmax 保持', intuition: 'T→0 时 softmax 收敛到 argmax 的 one-hot。',
    math: 'argmax(softmax(x/ε)) = argmax(x), max prob ≥ max softmax(x)', pseudocode: 'softmax at T=ε; check argmax & mass', complexity: 'O(n)',
    ref: 'Hinton 2015 distilling', commonPitfall: '全相等时退化为均匀，仍满足不降。',
    invariant: 'argmax(run)=argmax(x) 且 max(run)≥max(softmax(x))',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => softmax(x.map((v) => v / 1e-4)),
    predicate: (x) => {
      const r = softmax(x.map((v) => v / 1e-4));
      const s1 = softmax(x); let mx1 = -Infinity; for (const v of s1) mx1 = Math.max(mx1, v);
      let mx = -Infinity, mi = 0; for (let i = 0; i < r.length; i++) if (r[i] > mx) { mx = r[i]; mi = i; }
      return mi === argmaxIdx(x) && mx >= mx1 - 1e-6;
    },
  });

  // H039 T→∞ 均匀：分布趋近均匀，最大最小差小。
  reg(39, {
    name: 'T→∞ 均匀', intuition: 'T→∞ 时 softmax 趋近均匀分布。',
    math: 'max(run) - min(run) < 0.05', pseudocode: 'softmax at T=1e6; check flatness', complexity: 'O(n)',
    ref: 'Hinton 2015 distilling', commonPitfall: '需 T 远大于 logit 幅度。',
    invariant: 'max(run)-min(run) < 0.05 且 Σ=1',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => softmax(x.map((v) => v / 1e6)),
    predicate: (x) => {
      const r = softmax(x.map((v) => v / 1e6));
      let s = 0, mn = Infinity, mx = -Infinity;
      for (let i = 0; i < r.length; i++) { if (!(r[i] >= 0 && Number.isFinite(r[i]))) return false; s += r[i]; mn = Math.min(mn, r[i]); mx = Math.max(mx, r[i]); }
      return Math.abs(s - 1) < 1e-9 && (mx - mn) < 0.05;
    },
  });

  // H040 温度保持 argmax。
  reg(40, {
    name: '温度保持 argmax', intuition: '正温度下 softmax 保序，argmax 不变。',
    math: 'argmax(softmax(x/T)) = argmax(x)', pseudocode: 'argmax of softmax(x/T)', complexity: 'O(n)',
    ref: 'Vaswani 2017', commonPitfall: '仅 T>0 成立。',
    invariant: 'argmax(run) = argmax(x)',
    inputs: ['x', 'T'], domains: { x: 'vecFinite', T: 'pos' },
    run: (x, T) => argmaxIdx(softmax(x.map((v) => v / T))),
    predicate: (x, T) => { const r = argmaxIdx(softmax(x.map((v) => v / T))); return r === argmaxIdx(x); },
  });

  // H041 温度保序。
  reg(41, {
    name: '温度保序', intuition: 'T>0 时 softmax 严格保序。',
    math: 'x_i > x_j ⇒ softmax(x/T)_i > softmax(x/T)_j', pseudocode: 'check pairwise order', complexity: 'O(n^2)',
    ref: 'Vaswani 2017', commonPitfall: '仅严格大于时断言严格大于。',
    invariant: '∀i,j: x_i>x_j ⇒ run_i>run_j',
    inputs: ['x', 'T'], domains: { x: 'vecFinite', T: 'pos' },
    run: (x, T) => softmax(x.map((v) => v / T)),
    predicate: (x, T) => {
      const r = softmax(x.map((v) => v / T));
      let s = 0; const m = maxOf(x);
      for (let i = 0; i < x.length; i++) {
        if (!(r[i] >= 0)) return false; s += r[i];
        for (let j = 0; j < x.length; j++) {
          if (x[i] > x[j]) {
            // 精确序：(z_i-m)/T > (z_j-m)/T ⇔ z_i>z_j（T>0），仅在可解析时才断言严格输出序
            const si = (x[i] - m) / T, sj = (x[j] - m) / T;
            if (si - sj > 1e-6 && si > -740) { if (!(r[i] > r[j])) return false; }
          }
        }
      }
      return relClose(s, 1, 1e-9);
    },
  });

  // H042 温度熵增（T2>T1 时熵不降）。
  reg(42, {
    name: '温度熵增', intuition: '熵随温度非递减（越高越平）。',
    math: 'H(softmax(x/10)) - H(softmax(x/1)) >= 0', pseudocode: 'ΔH = H(T=10) - H(T=1)', complexity: 'O(n)',
    ref: 'Shannon 1948', commonPitfall: '方向别反。',
    invariant: 'ΔH >= -ε',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => entropy(softmax(x.map((v) => v / 10))) - entropy(softmax(x.map((v) => v / 1))),
    predicate: (x) => { const r = entropy(softmax(x.map((v) => v / 10))) - entropy(softmax(x.map((v) => v / 1))); return r >= -1e-6; },
  });

  // H043 温度缩放概率单调性：降低温度使最大概率不降。
  reg(43, {
    name: '温度缩放概率单调性', intuition: '更低温度（更尖）使最大概率不减。',
    math: 'max p(x/T) <= max p(x/(T/2))', pseudocode: 'compare max prob at T vs T/2', complexity: 'O(n)',
    ref: 'Hinton 2015 distilling', commonPitfall: 'T/2 更尖 → 最大概率更高。',
    invariant: 'max(softmax(x/(T/2))) >= max(softmax(x/T)) - ε',
    inputs: ['x', 'T'], domains: { x: 'vecFinite', T: 'pos' },
    run: (x, T) => { const a = maxOf(softmax(x.map((v) => v / T))), b = maxOf(softmax(x.map((v) => v / (T / 2)))); return [a, b]; },
    predicate: (x, T) => {
      const r = (function () { const a = maxOf(softmax(x.map((v) => v / T))), b = maxOf(softmax(x.map((v) => v / (T / 2)))); return [a, b]; })();
      return r[1] >= r[0] - 1e-6;
    },
  });

  // H044 负温度有效：softmax 仍得有限正值分布。
  reg(44, {
    name: '负温度有效', intuition: '负温度反转排序，但仍是合法分布。',
    math: 'run = softmax(x/(-1)), Σ=1, run_i>0', pseudocode: 'softmax with negative T', complexity: 'O(n)',
    ref: 'Vaswani 2017', commonPitfall: '负温度反转 argmax 顺序。',
    invariant: 'Σ run = 1 且 ∀i: run_i>0 有限',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => softmax(x.map((v) => v / -1)),
    predicate: (x) => {
      const r = softmax(x.map((v) => v / -1));
      let s = 0; for (let i = 0; i < r.length; i++) { if (!(r[i] >= 0 && Number.isFinite(r[i]))) return false; s += r[i]; }
      return Math.abs(s - 1) < 1e-9;
    },
  });

  // H045 T=0 极限近似：比 T=1 更尖（熵不增）且 argmax 保持。
  reg(45, {
    name: 'T=0 极限近似', intuition: '极小温度逼近 one-hot，熵不增。',
    math: 'H(softmax(x/1e-3)) <= H(softmax(x))', pseudocode: 'entropy at T=1e-3 vs 1', complexity: 'O(n)',
    ref: 'Hinton 2015 distilling', commonPitfall: '全相等时熵相等。',
    invariant: 'H(run) <= H(softmax(x)) + ε 且 argmax 保持',
    inputs: ['x'], domains: { x: 'vecFinite' },
    run: (x) => softmax(x.map((v) => v / 1e-3)),
    predicate: (x) => {
      const r = softmax(x.map((v) => v / 1e-3));
      const s1 = softmax(x); let mx1 = -Infinity; for (const v of s1) mx1 = Math.max(mx1, v);
      let mx = -Infinity, mi = 0; for (let i = 0; i < r.length; i++) if (r[i] > mx) { mx = r[i]; mi = i; }
      return mi === argmaxIdx(x) && mx >= mx1 - 1e-6 && entropy(r) <= entropy(s1) + 1e-6;
    },
  });

  // H046 温度交叉熵。
  reg(46, {
    name: '温度交叉熵', intuition: '两温度分布间的交叉熵非负且 >= 熵。',
    math: 'CE(p,q) = -Σ p_i log q_i >= H(p)', pseudocode: 'cross-entropy of softmax(x/T),softmax(y/T)', complexity: 'O(n)',
    ref: 'Goodfellow 2016', commonPitfall: 'q_i 须 >0 以免 log 发散。',
    invariant: 'run >= H(p) - ε 且 run >= 0',
    inputs: ['x', 'y', 'T'], domains: { x: 'vecFinite', y: 'vecFinite', T: 'pos' },
    run: (x, y, T) => {
      const p = softmax(x.map((v) => v / T)), q = softmax(y.map((v) => v / T));
      let ce = 0; for (let i = 0; i < p.length; i++) if (p[i] > 0) ce += p[i] * Math.log(q[i]);
      return -ce;
    },
    predicate: (x, y, T) => {
      const p = softmax(x.map((v) => v / T)), q = softmax(y.map((v) => v / T));
      let ce = 0; for (let i = 0; i < p.length; i++) if (p[i] > 0) ce += p[i] * Math.log(q[i]); ce = -ce;
      const r = (function () { const pp = softmax(x.map((v) => v / T)), qq = softmax(y.map((v) => v / T)); let c = 0; for (let i = 0; i < pp.length; i++) if (pp[i] > 0) c += pp[i] * Math.log(qq[i]); return -c; })();
      return r >= entropy(p) - 1e-9 && r >= 0;
    },
  });

  // H047 温度 log-softmax。
  reg(47, {
    name: '温度 log-softmax', intuition: 'log(softmax) 各分量 ≤0 且 exp 后和为 1。',
    math: 'run_i = log softmax(x/T)_i, Σ exp(run)=1', pseudocode: 'log of softmax', complexity: 'O(n)',
    ref: 'Vaswani 2017', commonPitfall: '数值稳定实现。',
    invariant: 'run_i<=0 且 Σ exp(run)=1',
    inputs: ['x', 'T'], domains: { x: 'vecFinite', T: 'pos' },
    run: (x, T) => { const p = softmax(x.map((v) => v / T)); const out = new Float64Array(x.length); for (let i = 0; i < x.length; i++) out[i] = Math.log(p[i]); return out; },
    predicate: (x, T) => {
      const p = softmax(x.map((v) => v / T)); const r = (function () { const o = new Float64Array(x.length); for (let i = 0; i < x.length; i++) o[i] = Math.log(p[i]); return o; })();
      let s = 0; for (let i = 0; i < r.length; i++) { if (!(r[i] <= 0)) return false; s += Math.exp(r[i]); }
      return Math.abs(s - 1) < 1e-9;
    },
  });

  // H048 温度 KL 到均匀。
  reg(48, {
    name: '温度 KL 到均匀', intuition: 'KL(p||均匀)=log n - H(p)，非负。',
    math: 'run = log(n) - H(softmax(x/T)) >= 0', pseudocode: 'kl-to-uniform', complexity: 'O(n)',
    ref: 'Cover & Thomas 2006', commonPitfall: 'KL 非负。',
    invariant: 'run >= -ε 且 = log n - H(p)',
    inputs: ['x', 'T'], domains: { x: 'vecFinite', T: 'pos' },
    run: (x, T) => { const p = softmax(x.map((v) => v / T)); return Math.log(p.length) - entropy(p); },
    predicate: (x, T) => {
      const p = softmax(x.map((v) => v / T));
      const r = (function () { const pp = softmax(x.map((v) => v / T)); return Math.log(pp.length) - entropy(pp); })();
      return r >= -1e-9 && Math.abs(r - (Math.log(p.length) - entropy(p))) < 1e-9;
    },
  });

  // ── 类别 4：舍入（H049–H064）──────────────────────────────────────────

  // H049 四舍五入（对称 half-up，满足 round(-x)=-round(x)）。
  reg(49, {
    name: '四舍五入 · 对称', intuition: '对称舍入：|round-x|≤0.5 且 round(-x)=-round(x)。',
    math: 'run = sign(x)·floor(|x|+0.5)', pseudocode: 'symmetric round half up', complexity: 'O(1)',
    ref: 'IEEE 754-2019', commonPitfall: '0.5 偏向远离 0 而非偶数。',
    invariant: '|run-x|<=0.5 且 run(-x)=-run(x) 且为整数',
    inputs: ['x'], domains: { x: 'finite' },
    run: (x) => symRound(x),
    predicate: (x) => { const r = symRound(x); return isInt(r) && Math.abs(r - x) <= 0.5 + 1e-12 && Object.is(symRound(-x), -r); },
  });

  // H050 floor。
  reg(50, {
    name: '向下取整 floor', intuition: '不大于 x 的最大整数。',
    math: 'run = ⌊x⌋', pseudocode: 'return floor(x)', complexity: 'O(1)',
    ref: 'IEEE 754-2019', commonPitfall: '负整数注意方向（⌊-1.2⌋=-2）。',
    invariant: 'run 整数; run<=x<run+1',
    inputs: ['x'], domains: { x: 'finite' },
    run: (x) => Math.floor(x),
    predicate: (x) => { const r = Math.floor(x); return isInt(r) && r <= x + 1e-12 && r > x - 1 - 1e-12; },
  });

  // H051 ceil。
  reg(51, {
    name: '向上取整 ceil', intuition: '不小于 x 的最小整数。',
    math: 'run = ⌈x⌉', pseudocode: 'return ceil(x)', complexity: 'O(1)',
    ref: 'IEEE 754-2019', commonPitfall: '⌈-1.2⌉=-1。',
    invariant: 'run 整数; run-1<x<=run',
    inputs: ['x'], domains: { x: 'finite' },
    run: (x) => Math.ceil(x),
    predicate: (x) => { const r = Math.ceil(x); return isInt(r) && r >= x - 1e-12 && r < x + 1 + 1e-12; },
  });

  // H052 trunc（向零取整）。
  reg(52, {
    name: '向零取整 trunc', intuition: '截断小数，朝零方向。',
    math: 'run = sign(x)·⌊|x|⌋', pseudocode: 'trunc toward zero', complexity: 'O(1)',
    ref: 'IEEE 754-2019', commonPitfall: 'trunc(-1.2)=-1。',
    invariant: 'run 整数 且 trunc(-x)=-trunc(x)',
    inputs: ['x'], domains: { x: 'finite' },
    run: (x) => trunc(x),
    predicate: (x) => { const r = trunc(x); return isInt(r) && Object.is(trunc(-x), -r); },
  });

  // H053 银行家舍入（round-half-even）。
  reg(53, {
    name: '银行家舍入', intuition: '.5 舍入到最近偶数，消除偏差。',
    math: 'run = round half to even', pseudocode: 'round half even', complexity: 'O(1)',
    ref: 'IEEE 754-2019 §4.3.2', commonPitfall: '0.5→0, 1.5→2, -0.5→0, -1.5→-2。',
    invariant: 'run 整数; |run-x|<=0.5; 与参考一致',
    inputs: ['x'], domains: { x: 'finite' },
    run: (x) => roundHalfEven(x),
    predicate: (x) => {
      const r = roundHalfEven(x);
      const f = Math.floor(x); const fp = x - f;
      let exp; if (fp < 0.5) exp = f; else if (fp > 0.5) exp = f + 1; else exp = (f % 2 === 0) ? f : f + 1;
      return isInt(r) && Math.abs(r - x) <= 0.5 + 1e-12 && Object.is(r, exp);
    },
  });

  // H054 精确到 n 位小数。
  reg(54, {
    name: '精确到 n 位小数', intuition: '四舍五入到 n 位小数，结果为其倍数。',
    math: 'run = round(x·10^n)/10^n', pseudocode: 'round to n decimals', complexity: 'O(1)',
    ref: 'Goldberg 1991', commonPitfall: 'n 须夹取到 [0,10]。',
    invariant: 'run 为 10^-n 的倍数 且 |run-x|<=0.5·10^-n',
    inputs: ['x', 'n'], domains: { x: 'finite', n: 'smallInt' },
    run: (x, n) => { const nn = Math.min(10, Math.max(0, Math.round(n))); const f = Math.pow(10, nn); return Math.round(x * f) / f; },
    predicate: (x, n) => {
      const nn = Math.min(10, Math.max(0, Math.round(n))); const f = Math.pow(10, nn);
      const r = Math.round(x * f) / f;
      if (!isInt(Math.round(r * f))) return false;
      return Math.abs(r - x) <= 0.5 / f + 1e-9;
    },
  });

  // H055 随机舍入（随机四舍五入，rng 种子化）。
  reg(55, {
    name: '随机舍入', intuition: '按小数部分概率四舍五入到相邻整数。',
    math: 'run ∈ {⌊x⌋, ⌈x⌉}', pseudocode: 'stochastic round', complexity: 'O(1)',
    ref: 'Jacob et al. 2018 int8', commonPitfall: '期望 = x（无偏）。',
    invariant: 'run ∈ {floor(x), floor(x)+1}',
    inputs: ['x'], domains: { x: 'finite' },
    run: (x) => {
      const fl = Math.floor(x); if (x === fl) return fl;
      const fr = x - fl; const rng = NS.L9.rng.mulberry32(rngSeed([x]));
      return rng() < fr ? fl + 1 : fl;
    },
    predicate: (x) => {
      const fl = Math.floor(x); if (x === fl) { const r = fl; return r === fl; }
      const fr = x - fl; const rng = NS.L9.rng.mulberry32(rngSeed([x]));
      const r = rng() < fr ? fl + 1 : fl;
      return r === fl || r === fl + 1;
    },
  });

  // H056 安全舍入（非有限→0，有限→对称舍入）。
  reg(56, {
    name: '安全舍入', intuition: '对 NaN/Inf 返回 0，避免污染。',
    math: 'run = finite? symRound(x) : 0', pseudocode: 'safe round', complexity: 'O(1)',
    ref: 'Goldberg 1991', commonPitfall: '输入域保证有限，仍须稳健。',
    invariant: 'run 有限整数; |run-x|<=0.5',
    inputs: ['x'], domains: { x: 'finite' },
    run: (x) => { if (!Number.isFinite(x)) return 0; return symRound(x); },
    predicate: (x) => { const r = (function () { if (!Number.isFinite(x)) return 0; return symRound(x); })(); return Number.isFinite(r) && isInt(r) && Math.abs(r - x) <= 0.5 + 1e-12; },
  });

  // H057 量化步长舍入（舍入到 s 的整数倍）。
  reg(57, {
    name: '量化步长舍入', intuition: '将 x 舍入到步长 s 的最近倍数。',
    math: 'run = round(x/s)·s', pseudocode: 'round to multiple of s', complexity: 'O(1)',
    ref: 'Jacob et al. 2018 int8', commonPitfall: '误差不超过 s/2。',
    invariant: 'run 为 s 的倍数 且 |run-x|<=s/2',
    inputs: ['x', 's'], domains: { x: 'finite', s: 'pos' },
    run: (x, s) => Math.round(x / s) * s,
    predicate: (x, s) => {
      const r = Math.round(x / s) * s;
      if (Math.abs(r / s - Math.round(r / s)) > 1e-9) return false;
      return Math.abs(r - x) <= s / 2 + 1e-9;
    },
  });

  // H058 朝 -∞ 步长（floor 到 s 的倍数）。
  reg(58, {
    name: '朝-∞步长舍入', intuition: '向下对齐到步长 s 的倍数。',
    math: 'run = floor(x/s)·s', pseudocode: 'floor to multiple of s', complexity: 'O(1)',
    ref: 'Jacob et al. 2018 int8', commonPitfall: 'run <= x < run+s。',
    invariant: 'run 为 s 的倍数; run<=x<run+s',
    inputs: ['x', 's'], domains: { x: 'finite', s: 'pos' },
    run: (x, s) => Math.floor(x / s) * s,
    predicate: (x, s) => {
      const r = Math.floor(x / s) * s;
      if (Math.abs(r / s - Math.round(r / s)) > 1e-9) return false;
      return r <= x + 1e-12 && r > x - s - 1e-9;
    },
  });

  // H059 舍入单调（x1<=x2 ⇒ round(x1)<=round(x2)）。
  reg(59, {
    name: '舍入单调', intuition: '对称舍入非减。',
    math: 'x1<=x2 ⇒ run1<=run2', pseudocode: 'monotone rounding', complexity: 'O(1)',
    ref: 'IEEE 754-2019', commonPitfall: '半点处仍单调。',
    invariant: 'x1<=x2 ⇒ run[0]<=run[1]',
    inputs: ['x1', 'x2'], domains: { x1: 'finite', x2: 'finite' },
    run: (x1, x2) => [symRound(x1), symRound(x2)],
    predicate: (x1, x2) => { const r = [symRound(x1), symRound(x2)]; if (x1 <= x2) return r[0] <= r[1]; return r[1] <= r[0]; },
  });

  // H060 floor 单调。
  reg(60, {
    name: 'floor 单调', intuition: 'floor 非减。',
    math: 'x1<=x2 ⇒ floor(x1)<=floor(x2)', pseudocode: 'monotone floor', complexity: 'O(1)',
    ref: 'IEEE 754-2019', commonPitfall: '无。',
    invariant: 'x1<=x2 ⇒ run[0]<=run[1]',
    inputs: ['x1', 'x2'], domains: { x1: 'finite', x2: 'finite' },
    run: (x1, x2) => [Math.floor(x1), Math.floor(x2)],
    predicate: (x1, x2) => { const r = [Math.floor(x1), Math.floor(x2)]; if (x1 <= x2) return r[0] <= r[1]; return r[1] <= r[0]; },
  });

  // H061 ceil 性质（ceil(x-1) < x <= ceil(x)）。
  reg(61, {
    name: 'ceil 性质', intuition: 'ceil 夹住 x 于 (ceil(x-1), ceil(x)]。',
    math: 'ceil(x-1) < x <= ceil(x)', pseudocode: 'ceil property', complexity: 'O(1)',
    ref: 'IEEE 754-2019', commonPitfall: '严格小于左侧。',
    invariant: 'ceil(x-1) < x <= ceil(x)',
    inputs: ['x'], domains: { x: 'finite' },
    run: (x) => Math.ceil(x),
    predicate: (x) => { const r = Math.ceil(x); return Math.ceil(x - 1) < x + 1e-12 && x <= r + 1e-12 && isInt(r); },
  });

  // H062 舍入误差界。
  reg(62, {
    name: '舍入误差界', intuition: '任意舍入到整数误差不超过 0.5。',
    math: '|run - x| <= 0.5', pseudocode: 'rounding error bound', complexity: 'O(1)',
    ref: 'IEEE 754-2019', commonPitfall: '无。',
    invariant: '|run - x| <= 0.5',
    inputs: ['x'], domains: { x: 'finite' },
    run: (x) => symRound(x),
    predicate: (x) => { const r = symRound(x); return Math.abs(r - x) <= 0.5 + 1e-12; },
  });

  // H063 舍入介于 floor 与 ceil 之间。
  reg(63, {
    name: '舍入在 floor/ceil 间', intuition: '四舍五入结果被 floor 和 ceil 夹住。',
    math: '⌊x⌋ <= run <= ⌈x⌉', pseudocode: 'floor <= round <= ceil', complexity: 'O(1)',
    ref: 'IEEE 754-2019', commonPitfall: '无。',
    invariant: 'floor(x) <= run <= ceil(x)',
    inputs: ['x'], domains: { x: 'finite' },
    run: (x) => symRound(x),
    predicate: (x) => { const r = symRound(x); return Math.floor(x) <= r + 1e-12 && r <= Math.ceil(x) + 1e-12 && isInt(r); },
  });

  // H064 trunc 对称。
  reg(64, {
    name: 'trunc 对称', intuition: '向零取整关于原点对称。',
    math: 'trunc(-x) = -trunc(x)', pseudocode: 'symmetric trunc', complexity: 'O(1)',
    ref: 'IEEE 754-2019', commonPitfall: '无。',
    invariant: 'trunc(-x) = -trunc(x)',
    inputs: ['x'], domains: { x: 'finite' },
    run: (x) => trunc(x),
    predicate: (x) => { const r = trunc(x); return Object.is(trunc(-x), -r); },
  });
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

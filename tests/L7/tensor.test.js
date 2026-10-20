'use strict';
/* L7 张量与内存 · 张量 / 形状 / 竞技场（tensor / shape / arena）
 * 验收 ④：视图（reshape/transpose/slice）零拷贝——v.data === x.data；
 *         仅 alloc / contiguous / fromArray2d 分配新缓冲。
 * 数据全部确定性（mulberry32 或手写 pattern），无可变全局状态。
 */
const { test } = require('node:test');
const assert = require('node:assert');
require('../../src/index.js');
const NS = globalThis.FlagshipAI;
const T = NS.L7.tensor;
const SH = NS.L7.shape;
const AR = NS.L7.arena;

// ── 确定性数据助手 ──────────────────────────────────────────────────────
function rndFilled(n, seed) {
  const rng = NS.L9.rng.mulberry32(seed);
  const a = new Float64Array(n);
  for (let i = 0; i < n; i++) a[i] = rng() * 2 - 1;
  return a;
}
function seqFilled(shape, base) {
  const t = T.alloc(shape);
  for (let i = 0; i < t.data.length; i++) t.data[i] = base + i;
  return t;
}
function rowsOf(t) {
  // 行主序快照（用于值比对）
  const out = [];
  for (let i = 0; i < t.data.length; i++) out.push(t.data[t.offset + i]);
  return out;
}

// ════════════════════════════════════════════════════════════════════════
// 一、NS.L7.shape —— 形状/步长代数
// ════════════════════════════════════════════════════════════════════════

test('shape.contiguousStrides · 2D 行主序（末维恒 1）', () => {
  assert.deepStrictEqual(SH.contiguousStrides([2, 3]), [3, 1]);
  assert.deepStrictEqual(SH.contiguousStrides([4, 5]), [5, 1]);
});

test('shape.contiguousStrides · 3D 嵌套累积', () => {
  assert.deepStrictEqual(SH.contiguousStrides([2, 3, 4]), [12, 4, 1]);
  assert.deepStrictEqual(SH.contiguousStrides([2, 3, 4, 5]), [60, 20, 5, 1]);
});

test('shape.contiguousStrides · 1D 与标量', () => {
  assert.deepStrictEqual(SH.contiguousStrides([7]), [1]);
  assert.deepStrictEqual(SH.contiguousStrides([]), []);
});

test('shape.size · 各形状元素数乘积', () => {
  assert.strictEqual(SH.size([2, 3]), 6);
  assert.strictEqual(SH.size([2, 3, 4]), 24);
  assert.strictEqual(SH.size([5]), 5);
  assert.strictEqual(SH.size([]), 1);
  assert.strictEqual(SH.size([0, 3]), 0);
});

test('shape.linearize · 2D 基础偏移', () => {
  assert.strictEqual(SH.linearize([0, 0], [2, 3], [3, 1], 0), 0);
  assert.strictEqual(SH.linearize([1, 2], [2, 3], [3, 1], 0), 5);
  assert.strictEqual(SH.linearize([0, 2], [2, 3], [3, 1], 0), 2);
});

test('shape.linearize · 非连续 stride + offset 合成', () => {
  assert.strictEqual(SH.linearize([1, 2], [2, 3], [1, 2], 0), 5); // 1*1 + 2*2
  assert.strictEqual(SH.linearize([1, 1], [2, 3], [3, 1], 10), 14); // 10 + 3 + 1
  assert.strictEqual(SH.linearize([2, 0, 1], [3, 2, 4], [8, 4, 1], 100), 117); // 100+16+0+1
});

test('shape.linearize · stride 省略时按行主序', () => {
  assert.strictEqual(SH.linearize([2, 1], [3, 4]), 9);
  assert.strictEqual(SH.linearize([1, 1, 1], [2, 2, 2]), 7);
});

test('shape.linearize · 全索引往返（round-trip 等于行主序扁平下标）', () => {
  const shape = [3, 4], stride = SH.contiguousStrides(shape);
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 4; j++) {
      assert.strictEqual(SH.linearize([i, j], shape, stride, 0), i * 4 + j,
        `linearize([${i},${j}]) 应为 ${i * 4 + j}`);
    }
  }
});

test('shape.linearize · 下标越界 / 个数不符即抛', () => {
  assert.throws(() => SH.linearize([2, 0], [2, 3], [3, 1], 0));
  assert.throws(() => SH.linearize([0, 3], [2, 3], [3, 1], 0));
  assert.throws(() => SH.linearize([0], [2, 3], [3, 1], 0));
  assert.throws(() => SH.linearize([-1, 0], [2, 3], [3, 1], 0));
});

test('shape.broadcastShapes · 同形广播', () => {
  assert.deepStrictEqual(SH.broadcastShapes([2, 3], [2, 3]), [2, 3]);
  assert.deepStrictEqual(SH.broadcastShapes([5], [5]), [5]);
});

test('shape.broadcastShapes · 尾部 1 扩展', () => {
  assert.deepStrictEqual(SH.broadcastShapes([2, 3], [1, 3]), [2, 3]);
  assert.deepStrictEqual(SH.broadcastShapes([5, 1], [5, 4]), [5, 4]);
  assert.deepStrictEqual(SH.broadcastShapes([1, 1], [3, 4]), [3, 4]);
});

test('shape.broadcastShapes · 前置补 1（短形状对齐尾部）', () => {
  assert.deepStrictEqual(SH.broadcastShapes([3], [2, 3]), [2, 3]);
  assert.deepStrictEqual(SH.broadcastShapes([2, 3], [3]), [2, 3]);
  assert.deepStrictEqual(SH.broadcastShapes([], [2, 3]), [2, 3]);
  assert.deepStrictEqual(SH.broadcastShapes([2, 3], []), [2, 3]);
  assert.deepStrictEqual(SH.broadcastShapes([], []), []);
});

test('shape.broadcastShapes · 维度不匹配即抛', () => {
  assert.throws(() => SH.broadcastShapes([2, 3], [2, 4]));
  assert.throws(() => SH.broadcastShapes([3], [4]));
  assert.throws(() => SH.broadcastShapes([2, 3, 5], [3, 5, 4]));
});

test('shape.permute · 2D 交换轴', () => {
  const p = SH.permute([2, 3], [3, 1], [1, 0]);
  assert.deepStrictEqual(p.shape, [3, 2]);
  assert.deepStrictEqual(p.stride, [1, 3]);
});

test('shape.permute · 3D 轮换轴', () => {
  const p = SH.permute([2, 3, 4], [12, 4, 1], [2, 0, 1]);
  assert.deepStrictEqual(p.shape, [4, 2, 3]);
  assert.deepStrictEqual(p.stride, [1, 12, 4]);
  const id = SH.permute([2, 3, 4], [12, 4, 1], [0, 1, 2]);
  assert.deepStrictEqual(id.shape, [2, 3, 4]);
  assert.deepStrictEqual(id.stride, [12, 4, 1]);
});

test('shape.permute · 与 tensor.transpose 的首两维交换一致', () => {
  const x = seqFilled([2, 3], 1);
  const tr = T.transpose(x);
  const p = SH.permute(x.shape, x.stride, [1, 0]);
  assert.deepStrictEqual(tr.shape, p.shape);
  assert.deepStrictEqual(tr.stride, p.stride);
});

test('shape.permute · 非法 axes 即抛', () => {
  assert.throws(() => SH.permute([2, 3], [3, 1], [0, 0]));
  assert.throws(() => SH.permute([2, 3], [3, 1], [0, 2]));
  assert.throws(() => SH.permute([2, 3], [3, 1], [1]));
  assert.throws(() => SH.permute([2, 3], [3, 1], [0, -1]));
});

test('shape.invertPerm · 具体逆置换值', () => {
  assert.deepStrictEqual(SH.invertPerm([1, 0]), [1, 0]);
  assert.deepStrictEqual(SH.invertPerm([2, 0, 1]), [1, 2, 0]);
  assert.deepStrictEqual(SH.invertPerm([0, 1, 2, 3]), [0, 1, 2, 3]);
});

test('shape.invertPerm · 往返恒等 inv(inv(p)) === p', () => {
  const perms = [[1, 0], [2, 0, 1], [1, 2, 0], [3, 2, 1, 0], [0, 2, 1]];
  for (const ax of perms) {
    assert.deepStrictEqual(SH.invertPerm(SH.invertPerm(ax)), ax, `inv(inv(${ax})) 应为自身`);
    // 语义校验：permute 后按逆置换还原
    const shape = ax.map((_, i) => i + 2);
    const stride = SH.contiguousStrides(shape);
    const fwd = SH.permute(shape, stride, ax);
    const back = SH.permute(fwd.shape, fwd.stride, SH.invertPerm(ax));
    assert.deepStrictEqual(back.shape, shape);
    assert.deepStrictEqual(back.stride, stride);
  }
});

// ════════════════════════════════════════════════════════════════════════
// 二、NS.L7.tensor —— 构造 / get/set / 视图
// ════════════════════════════════════════════════════════════════════════

test('tensor.alloc · shape/stride/offset/dtype/data 长度', () => {
  const t = T.alloc([2, 3]);
  assert.deepStrictEqual(t.shape, [2, 3]);
  assert.deepStrictEqual(t.stride, [3, 1]);
  assert.strictEqual(t.offset, 0);
  assert.strictEqual(t.dtype, 'float64');
  assert.ok(t.data instanceof Float64Array);
  assert.strictEqual(t.data.length, 6);
  assert.strictEqual(T.size(t), 6);
  assert.strictEqual(T.ndim(t), 2);
});

test('tensor.alloc · 显式 float64 与默认 dtype 等价', () => {
  const a = T.alloc([3], 'float64');
  const b = T.alloc([3]);
  assert.strictEqual(a.dtype, b.dtype);
  assert.strictEqual(a.dtype, 'float64');
});

test('tensor.alloc · 非 float64 dtype 即抛', () => {
  assert.throws(() => T.alloc([2], 'float32'));
  assert.throws(() => T.alloc([2], 'int32'));
});

test('tensor.alloc · 两次分配缓冲互相独立', () => {
  const a = T.alloc([4]);
  const b = T.alloc([4]);
  assert.notStrictEqual(a.data, b.data);
  T.set(a, 7, 0);
  assert.strictEqual(T.get(b, 0), 0);
});

test('tensor.fromFlat · data 身份恒等（零拷贝包装）', () => {
  const data = rndFilled(6, 42);
  const t = T.fromFlat(data, [2, 3]);
  assert.strictEqual(t.data, data, 'fromFlat 必须保持 data 身份恒等');
  assert.deepStrictEqual(t.shape, [2, 3]);
  assert.deepStrictEqual(t.stride, [3, 1]);
  assert.strictEqual(t.offset, 0);
  assert.strictEqual(t.dtype, 'float64');
});

test('tensor.fromFlat · get 读值与底层缓冲一致', () => {
  const data = rndFilled(6, 7);
  const t = T.fromFlat(data, [2, 3]);
  for (let i = 0; i < 2; i++) {
    for (let j = 0; j < 3; j++) {
      assert.strictEqual(T.get(t, i, j), data[i * 3 + j]);
    }
  }
});

test('tensor.fromFlat · 长度不匹配即抛', () => {
  assert.throws(() => T.fromFlat(new Float64Array(5), [2, 3]));
  assert.throws(() => T.fromFlat(new Float64Array(7), [2, 3]));
});

test('tensor.fromFlat · 写穿透到底层 Float64Array', () => {
  const data = new Float64Array(4);
  const t = T.fromFlat(data, [2, 2]);
  T.set(t, 3.5, 1, 0);
  assert.strictEqual(data[2], 3.5);
});

test('tensor.make · 显式 stride/offset 原始构造', () => {
  const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8]);
  const t = T.make({ data, shape: [2, 3], stride: [3, 1], offset: 2 });
  assert.strictEqual(t.data, data);
  assert.strictEqual(t.offset, 2);
  assert.strictEqual(T.get(t, 0, 0), 3); // data[2]
  assert.strictEqual(T.get(t, 1, 2), 8); // data[7]
});

test('tensor.make · data 省略时按 shape 新分配', () => {
  const t = T.make({ shape: [2, 2] });
  assert.ok(t.data instanceof Float64Array);
  assert.strictEqual(t.data.length, 4);
  assert.deepStrictEqual(t.stride, [2, 1]);
});

test('tensor.zeros · 全零且形状正确', () => {
  const t = T.zeros([2, 3]);
  assert.strictEqual(T.size(t), 6);
  for (let i = 0; i < 2; i++) for (let j = 0; j < 3; j++) assert.strictEqual(T.get(t, i, j), 0);
});

test('tensor.fromArray2d · 值/形状/步长正确', () => {
  const t = T.fromArray2d([[1, 2, 3], [4, 5, 6]]);
  assert.deepStrictEqual(t.shape, [2, 3]);
  assert.deepStrictEqual(t.stride, [3, 1]);
  assert.strictEqual(T.get(t, 0, 0), 1);
  assert.strictEqual(T.get(t, 1, 2), 6);
  assert.deepStrictEqual(rowsOf(t), [1, 2, 3, 4, 5, 6]);
});

test('tensor.fromArray2d · 拷贝语义（改入参不影响张量）', () => {
  const src = [[1, 2], [3, 4]];
  const t = T.fromArray2d(src);
  src[0][0] = 99;
  assert.strictEqual(T.get(t, 0, 0), 1, 'fromArray2d 必须拷贝而非持有入参');
  assert.ok(t.data instanceof Float64Array);
});

test('tensor.fromArray2d · 非规则矩形 / 非数值即抛', () => {
  assert.throws(() => T.fromArray2d([[1, 2], [3]]));
  assert.throws(() => T.fromArray2d([[1, 'x'], [3, 4]]));
  assert.throws(() => T.fromArray2d('nope'));
});

test('tensor.get/set · 连续 2D 读写往返', () => {
  const t = T.alloc([2, 3]);
  let k = 1;
  for (let i = 0; i < 2; i++) for (let j = 0; j < 3; j++) T.set(t, k++, i, j);
  assert.strictEqual(T.get(t, 0, 0), 1);
  assert.strictEqual(T.get(t, 0, 2), 3);
  assert.strictEqual(T.get(t, 1, 0), 4);
  assert.strictEqual(T.get(t, 1, 2), 6);
  T.set(t, -5, 1, 1);
  assert.strictEqual(T.get(t, 1, 1), -5);
});

test('tensor.get/set · 3D 索引与扁平偏移一致', () => {
  const t = seqFilled([2, 2, 2], 0);
  assert.strictEqual(T.get(t, 0, 0, 0), 0);
  assert.strictEqual(T.get(t, 1, 1, 1), 7);
  T.set(t, 42, 1, 0, 1);
  assert.strictEqual(t.data[5], 42); // stride [4,2,1] → 1*4+0*2+1*1
});

test('tensor.get · 越界下标即抛', () => {
  const t = T.alloc([2, 3]);
  assert.throws(() => T.get(t, 2, 0));
  assert.throws(() => T.get(t, 0, 3));
  assert.throws(() => T.get(t, 0));
  assert.throws(() => T.get(t, -1, 0));
});

test('tensor.reshape · 零拷贝身份 + 值保持', () => {
  const x = seqFilled([2, 3], 1); // 1..6
  const r = T.reshape(x, 3, 2);
  assert.strictEqual(r.data, x.data, 'reshape 必须零拷贝共享 data');
  assert.strictEqual(r.offset, x.offset);
  assert.deepStrictEqual(r.shape, [3, 2]);
  assert.strictEqual(T.get(r, 0, 0), 1);
  assert.strictEqual(T.get(r, 1, 1), 4); // 行主序 flat 3
  assert.strictEqual(T.get(r, 2, 1), 6);
  T.set(r, 99, 0, 0);
  assert.strictEqual(x.data[0], 99, 'reshape 视图写穿母张量');
});

test('tensor.reshape · 数组入参形式与 -1 推断', () => {
  const x = seqFilled([2, 3], 1);
  const r1 = T.reshape(x, [6]);
  assert.strictEqual(r1.data, x.data);
  assert.strictEqual(T.get(r1, 5), 6);
  const r2 = T.reshape(x, -1);
  assert.deepStrictEqual(r2.shape, [6]);
  const r3 = T.reshape(x, 3, -1);
  assert.deepStrictEqual(r3.shape, [3, 2]);
  const r4 = T.reshape(x, -1, 2);
  assert.deepStrictEqual(r4.shape, [3, 2]);
});

test('tensor.reshape · 总大小不匹配即抛', () => {
  const x = seqFilled([2, 3], 1);
  assert.throws(() => T.reshape(x, 4));
  assert.throws(() => T.reshape(x, 2, 4));
  assert.throws(() => T.reshape(x, 2, -1, 2)); // -1 推断 1.5 非整数路径外：6/(2*2) 非整数
});

test('tensor.reshape · 非连续张量（转置视图）不可零拷贝变形即抛', () => {
  const x = seqFilled([2, 3], 1);
  const tr = T.transpose(x); // shape [3,2]，stride [1,3] 非行主序
  assert.throws(() => T.reshape(tr, 6));
  assert.throws(() => T.reshape(tr, 2, 3));
});

test('tensor.transpose · 零拷贝身份 + 交换语义', () => {
  const x = seqFilled([2, 3], 1); // 1..6
  const tr = T.transpose(x);
  assert.strictEqual(tr.data, x.data, 'transpose 必须零拷贝共享 data');
  assert.strictEqual(tr.offset, x.offset);
  assert.deepStrictEqual(tr.shape, [3, 2]);
  assert.deepStrictEqual(tr.stride, [1, 3]);
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 2; j++) {
      assert.strictEqual(T.get(tr, i, j), T.get(x, j, i), `tr(${i},${j}) 应等于 x(${j},${i})`);
    }
  }
});

test('tensor.transpose · set 写穿母张量缓冲', () => {
  const x = seqFilled([2, 3], 1);
  const tr = T.transpose(x);
  T.set(tr, 100, 0, 0); // tr(0,0) → x(0,0) → flat 0
  assert.strictEqual(x.data[0], 100);
  T.set(tr, 200, 2, 1); // tr(2,1) → x(1,2) → flat 5
  assert.strictEqual(x.data[5], 200);
  assert.strictEqual(T.get(x, 1, 2), 200);
});

test('tensor.transpose · ndim < 2 即抛', () => {
  assert.throws(() => T.transpose(T.alloc([5])));
  assert.throws(() => T.transpose(T.alloc([])));
});

test('tensor.contiguous · 新缓冲 !== 且值相等', () => {
  const x = seqFilled([2, 3], 1);
  const c = T.contiguous(x);
  assert.notStrictEqual(c.data, x.data, 'contiguous 必须分配新缓冲');
  assert.deepStrictEqual(c.shape, [2, 3]);
  assert.deepStrictEqual(rowsOf(c), rowsOf(x));
  T.set(c, 99, 0, 0);
  assert.strictEqual(T.get(x, 0, 0), 1, 'contiguous 副本与母张量互不干扰');
});

test('tensor.contiguous · 转置视图物化为行主序', () => {
  const x = seqFilled([2, 3], 1); // x(i,j) = i*3+j+1
  const tr = T.transpose(x);      // tr(i,j) = x(j,i) = j*3+i+1
  const c = T.contiguous(tr);
  assert.notStrictEqual(c.data, x.data);
  assert.deepStrictEqual(c.shape, [3, 2]);
  assert.deepStrictEqual(c.stride, [2, 1]);
  assert.deepStrictEqual(rowsOf(c), [1, 4, 2, 5, 3, 6]);
  assert.strictEqual(T.get(c, 2, 1), 6);
});

test('tensor.contiguous · 3D 任意 stride 视图物化正确', () => {
  const x = seqFilled([2, 3, 4], 0); // 24 个元素 0..23
  const p = SH.permute(x.shape, x.stride, [2, 0, 1]);
  const v = T.make({ data: x.data, shape: p.shape, stride: p.stride, offset: x.offset, dtype: 'float64' });
  const c = T.contiguous(v);
  assert.deepStrictEqual(c.shape, [4, 2, 3]);
  for (let a = 0; a < 4; a++) {
    for (let b = 0; b < 2; b++) {
      for (let d = 0; d < 3; d++) {
        // v(a,b,d) = x(b,d,a) = b*12 + d*4 + a
        assert.strictEqual(T.get(c, a, b, d), b * 12 + d * 4 + a, `c(${a},${b},${d})`);
      }
    }
  }
});

test('tensor.slice · 视图身份 + offset/形状/值正确', () => {
  const x = seqFilled([4, 3], 0); // x(i,j) = i*3+j
  const v = T.slice(x, 0, 1, 3); // 行 [1,3)
  assert.strictEqual(v.data, x.data, 'slice 必须零拷贝共享 data');
  assert.deepStrictEqual(v.shape, [2, 3]);
  assert.deepStrictEqual(v.stride, [3, 1]);
  assert.strictEqual(v.offset, 3);
  assert.strictEqual(T.get(v, 0, 0), T.get(x, 1, 0));
  assert.strictEqual(T.get(v, 1, 2), T.get(x, 2, 2));
  T.set(v, 77, 0, 0);
  assert.strictEqual(x.data[3], 77, 'slice 视图写穿母张量');
});

test('tensor.slice · 轴 1 切片（列区间）', () => {
  const x = seqFilled([4, 3], 0);
  const v = T.slice(x, 1, 1, 3);
  assert.deepStrictEqual(v.shape, [4, 2]);
  assert.strictEqual(v.offset, 1);
  assert.strictEqual(T.get(v, 0, 0), T.get(x, 0, 1));
  assert.strictEqual(T.get(v, 3, 1), T.get(x, 3, 2));
});

test('tensor.slice · 负索引（Python 风格）', () => {
  const x = seqFilled([4, 3], 0);
  const v = T.slice(x, 0, -2, -1); // 行 [2,3)
  assert.deepStrictEqual(v.shape, [1, 3]);
  assert.strictEqual(T.get(v, 0, 0), T.get(x, 2, 0));
  const w = T.slice(x, 0, 1); // end 省略 → 到末尾
  assert.deepStrictEqual(w.shape, [3, 3]);
});

test('tensor.slice · 越界 / 非法轴即抛', () => {
  const x = seqFilled([4, 3], 0);
  assert.throws(() => T.slice(x, 0, 3, 1));
  assert.throws(() => T.slice(x, 0, 0, 5));
  assert.throws(() => T.slice(x, 0, -5, 2));
  assert.throws(() => T.slice(x, 2, 0, 1));
  assert.throws(() => T.slice(x, -1, 0, 1));
});

test('验收④ · reshape→transpose→slice 视图链全程共享 data', () => {
  const x = T.alloc([2, 6]);
  for (let i = 0; i < 12; i++) x.data[i] = i;
  const v1 = T.reshape(x, 4, 3);
  const v2 = T.transpose(v1);      // [3,4]
  const v3 = T.slice(v2, 0, 1, 3); // [2,4]
  assert.strictEqual(v1.data, x.data);
  assert.strictEqual(v2.data, x.data);
  assert.strictEqual(v3.data, x.data);
  // 值语义：v3(a,b) = v2(a+1,b) = v1(b,a+1) = b*3 + a+1
  assert.strictEqual(T.get(v3, 0, 0), 1);
  assert.strictEqual(T.get(v3, 1, 0), 2);
  assert.strictEqual(T.get(v3, 0, 3), 10);
  assert.strictEqual(T.get(v3, 1, 3), 11);
});

test('验收④ · slice→transpose 链共享 data；非连续视图 reshape 抛错', () => {
  const x = seqFilled([4, 4], 0); // x(i,j) = i*4+j
  const s = T.slice(x, 0, 1, 4);  // [3,4] 行 1..3
  const tr = T.transpose(s);      // [4,3] 非连续
  assert.strictEqual(s.data, x.data);
  assert.strictEqual(tr.data, x.data);
  assert.throws(() => T.reshape(tr, 3, 4), '非连续视图不可零拷贝 reshape');
  const c = T.contiguous(tr);     // 正确路径：物化副本
  assert.notStrictEqual(c.data, x.data);
  assert.strictEqual(T.get(c, 0, 0), T.get(x, 1, 0)); // tr(0,0)=s(0,0)=x(1,0)=4
  assert.strictEqual(T.get(c, 3, 2), T.get(x, 3, 3)); // tr(3,2)=s(2,3)=x(3,3)=15
});

test('验收④ · 连续切片后可继续零拷贝 reshape', () => {
  const x = seqFilled([4, 4], 0); // x(i,j) = i*4+j
  const s = T.slice(x, 0, 1, 4);  // [3,4] 连续视图，offset 4
  const r = T.reshape(s, 2, 6);   // 零拷贝可行（s 连续）
  assert.strictEqual(r.data, x.data);
  assert.strictEqual(r.offset, 4);
  // r(a,b) = x 的 flat (4 + a*6 + b) = 行 (4+a*6+b)/4 ...
  assert.strictEqual(T.get(r, 0, 0), 4);
  assert.strictEqual(T.get(r, 1, 5), 15);
  assert.strictEqual(T.get(r, 0, 5), 9);
});

// ════════════════════════════════════════════════════════════════════════
// 三、NS.L7.arena —— 竞技场分配器
// ════════════════════════════════════════════════════════════════════════

test('arena.alloc · 返回连续张量（shape/stride/offset/dtype）', () => {
  const a = AR.arena(1024);
  const t = a.alloc([2, 3]);
  assert.deepStrictEqual(t.shape, [2, 3]);
  assert.deepStrictEqual(t.stride, [3, 1]);
  assert.strictEqual(t.offset, 0);
  assert.strictEqual(t.dtype, 'float64');
  assert.ok(t.data instanceof Float64Array);
  assert.strictEqual(t.data.length, 6);
});

test('arena.alloc · 两次分配互不重叠（同一后备缓冲 + 区间不相交）', () => {
  const a = AR.arena(1024);
  const t1 = a.alloc([2, 3]); // 48 字节
  const t2 = a.alloc([4]);    // 32 字节
  assert.strictEqual(t1.data.buffer, t2.data.buffer, 'arena 分配必须共享同一 ArrayBuffer');
  assert.strictEqual(t1.data.byteOffset, 0);
  assert.strictEqual(t2.data.byteOffset, 48);
  assert.ok(t1.data.byteOffset + t1.data.byteLength <= t2.data.byteOffset, '区间必须不相交');
});

test('arena.alloc · 写入互不干扰', () => {
  const a = AR.arena(1024);
  const t1 = a.alloc([3]);
  const t2 = a.alloc([3]);
  for (let i = 0; i < 3; i++) T.set(t1, i + 1, i);
  for (let i = 0; i < 3; i++) assert.strictEqual(T.get(t2, i), 0, 't2 不应被 t1 写入污染');
  assert.strictEqual(T.get(t1, 2), 3);
});

test('arena.usedBytes · 累计字节数', () => {
  const a = AR.arena(1024);
  assert.strictEqual(a.usedBytes(), 0);
  a.alloc([10]); // 80 字节
  assert.strictEqual(a.usedBytes(), 80);
  a.alloc([5]);  // +40 字节
  assert.strictEqual(a.usedBytes(), 120);
  a.alloc([2, 2]); // +32 字节
  assert.strictEqual(a.usedBytes(), 152);
});

test('arena.reset · 用量归零且保留后备缓冲', () => {
  const a = AR.arena(1024);
  const t1 = a.alloc([10]);
  const buf = t1.data.buffer;
  assert.strictEqual(a.usedBytes(), 80);
  a.reset();
  assert.strictEqual(a.usedBytes(), 0);
  const t2 = a.alloc([10]); // reset 后从 0 重新碰撞
  assert.strictEqual(t2.data.buffer, buf, 'reset 不得释放 ArrayBuffer');
  assert.strictEqual(t2.data.byteOffset, 0);
  assert.strictEqual(a.usedBytes(), 80);
});

test('arena.reset · 重置后分配互不重叠', () => {
  const a = AR.arena(1024);
  const t1 = a.alloc([4]);
  a.reset();
  const t2 = a.alloc([4]);
  const t3 = a.alloc([4]);
  assert.strictEqual(t2.data.byteOffset, 0);
  assert.strictEqual(t3.data.byteOffset, 32);
  assert.ok(t2.data.byteOffset + t2.data.byteLength <= t3.data.byteOffset);
});

test('arena.capacityBytes · 回报容量', () => {
  assert.strictEqual(AR.arena(256).capacityBytes(), 256);
  assert.strictEqual(AR.arena(0).capacityBytes(), 0);
});

test('arena.alloc · 容量不足即抛', () => {
  const a = AR.arena(64);
  assert.throws(() => a.alloc([10])); // 需 80 > 64
  assert.strictEqual(a.usedBytes(), 0, '失败的分配不得移动碰撞指针');
});

test('arena.alloc · 恰好占满后再次分配即抛', () => {
  const a = AR.arena(80);
  const t = a.alloc([10]); // 恰好 80
  assert.strictEqual(t.data.length, 10);
  assert.strictEqual(a.usedBytes(), 80);
  assert.throws(() => a.alloc([1])); // 再需 8 > 0
});

test('arena.alloc · 非 float64 dtype 即抛', () => {
  const a = AR.arena(1024);
  assert.throws(() => a.alloc([2], 'float32'));
  assert.throws(() => a.alloc([2], 'int8'));
});

test('arena · 非法构造参数即抛', () => {
  assert.throws(() => AR.arena(-1));
  assert.throws(() => AR.arena(1.5));
  assert.throws(() => AR.arena('big'));
});

test('arena.alloc · 零元素张量与 3D 形状', () => {
  const a = AR.arena(1024);
  const e = a.alloc([0, 3]);
  assert.strictEqual(e.data.length, 0);
  assert.deepStrictEqual(e.shape, [0, 3]);
  const t3 = a.alloc([2, 2, 2]);
  assert.deepStrictEqual(t3.stride, [4, 2, 1]);
  let k = 0;
  for (let i = 0; i < 2; i++) {
    for (let j = 0; j < 2; j++) {
      for (let m = 0; m < 2; m++) T.set(t3, k++, i, j, m);
    }
  }
  assert.strictEqual(T.get(t3, 0, 0, 0), 0);
  assert.strictEqual(T.get(t3, 1, 1, 1), 7);
});

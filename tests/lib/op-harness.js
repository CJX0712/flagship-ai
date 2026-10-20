'use strict';
/* 旗舰 AI v12 · L10 微指令性质测试harness（tests/L10 共享）
 * 经生产路径加载十层，提供：
 *   - loadNS()：返回 globalThis.FlagshipAI（已加载十层）
 *   - genArgs(rng, op)：依 op.domains / matrixDims 确定性构造参数（标量/矩阵）
 *   - checkOp(op, n)：用 fast-check 以 n 个种子驱动，每种子经 NS.L9.rng 确定性
 *     生成参数并断言 op.predicate 恒成立（不变量）。返回 { pass, reason }。
 *
 * 设计要点：fast-check 仅驱动「种子整数」任意子（避免反例打印巨型矩阵），
 * 真正的参数由我们与 v11 逐位一致的确定性 RNG + TAGS 生成，确保可复现。
 */
require('../../src/index.js'); // 生产加载路径：填充 globalThis.FlagshipAI
const NS = globalThis.FlagshipAI;
const fc = require('fast-check');

function genArgs(rng, op) {
  return op.inputs.map((name) => {
    const tag = op.domains[name];
    const t = NS.L10.TAGS[tag];
    if (!t) throw new Error(`op ${op.key} 引用未知道域标签 "${tag}"`);
    if (t.kind === 'matrix') {
      const dims = (op.matrixDims && op.matrixDims[name]) || t.defDims;
      if (!dims) throw new Error(`op ${op.key} 矩阵参数 ${name} 缺 matrixDims`);
      return NS.L10.genMatrix(rng, dims[0], dims[1], tag);
    }
    return NS.L10.genScalar(rng, tag);
  });
}

function checkOp(op, n) {
  try {
    fc.assert(
      fc.property(fc.integer(0, (1 << 30) - 1), (seed) => {
        const rng = NS.L9.rng.mulberry32(seed >>> 0);
        const args = genArgs(rng, op);
        return op.predicate.apply(null, args) === true;
      }),
      { numRuns: n }
    );
    return { pass: true };
  } catch (e) {
    const reason = (e && e.message ? e.message : String(e)).split('\n')[0].slice(0, 160);
    return { pass: false, reason };
  }
}

function implOps(domain) {
  const out = [];
  for (let i = 1; i <= NS.L10.CAP; i++) {
    const key = NS.L10.keyOf(domain, i);
    const op = NS.L10.ops[key];
    if (op && op.status === 'impl') out.push(op);
  }
  return out;
}

module.exports = { NS, fc, genArgs, checkOp, implOps, loadNS: () => NS };

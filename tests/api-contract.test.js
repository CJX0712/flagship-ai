'use strict';
/* K15 公共 API 契约测试 · T04 切换的常驻守护者
 *
 * 背景：T04 起 src/index.js 不再转发 legacy，34 个导出键全部由十层实现提供。
 *   本测试文件不在层文件禁区内（tests 目录允许 require legacy；禁 require 的规则
 *   只约束 src 下各层模块），故可同进程加载两侧做键集比对。
 *   切换前 V12 与 legacy 是同一批函数（自比自无意义）；切换后本测试成为
 *   「键集零漂移」的唯一常驻证明 —— legacy 增删键 / index.js 漏键错名都会红。
 *
 * 判据：
 *   1) 导出键集合与 v11 完全一致（34 键，不多不少，逐名相同）；
 *   2) 无 undefined 漂移（每个键都有实体）；
 *   3) 切换生效性：V12.GPTModel 不再是 legacy 类；loadModel 恢复出的是
 *      V12.GPTModel（L3）实例 —— R7 存档闭环确由十层接管。
 */
const { test } = require('node:test');
const assert = require('node:assert');

const V11 = require('../src/legacy/flagship-engine.v11.js');
const V12 = require('../src/index.js');

// 与 v11 return 清单逐名对应（v11 若增删键，本测试必须红 —— 那是 K15 契约变更，
// 需要显式更新两边，不允许静默漂移）。
const EXPECTED_KEYS = [
  'mulberry32', 'randn',
  'matmul', 'matmulAtB', 'matmulTA',
  'layernormFwd', 'layernormBwd',
  'rmsnormFwd', 'rmsnormBwd',
  'geluFwd', 'geluBwd', 'siluF', 'swigluFwd', 'swigluBwd',
  'ropeApply', 'softmaxRows', 'softmaxBwd',
  'crossEntropy',
  'Tokenizer', 'BPETokenizer', 'GPTModel', 'train', 'generate',
  'initKVCache', 'forwardStep', 'warmupKV', 'kvShift', 'saveModel', 'loadModel', 'loadTokenizer',
  'ARCH_PRESETS', 'applyPreset', 'textToByteChars', 'byteCharsToText',
].sort();

test('K15: 导出键集合与 v11 完全一致（34 键）', () => {
  const k11 = Object.keys(V11).sort();
  const k12 = Object.keys(V12).sort();
  assert.deepStrictEqual(k12, k11, 'V12 键集与 v11 不一致');
  assert.deepStrictEqual(k12, EXPECTED_KEYS, 'V12 键集与 EXPECTED_KEYS 不一致');
  assert.strictEqual(k12.length, 34, '键数必须为 34');
});

test('K15: 无 undefined 漂移', () => {
  for (const k of Object.keys(V12)) {
    assert.notStrictEqual(V12[k], undefined, `导出键 ${k} 为 undefined`);
  }
});

test('T04 切换生效：GPTModel 为 L3 实现且 loadModel 恢复 L3 实例', () => {
  // 切换后 V12.GPTModel 必须是独立的 L3 类（不再是 legacy.GPTModel 本体/子类）
  assert.notStrictEqual(V12.GPTModel, V11.GPTModel, 'V12.GPTModel 仍等于 legacy.GPTModel —— 切换未生效');
  const cfg = V12.applyPreset(
    { vocabSize: 65, blockSize: 16, nLayer: 1, nHead: 2, dModel: 32, seed: 1 }, 'gpt2');
  const m = new V12.GPTModel(cfg);
  assert.ok(m instanceof V12.GPTModel);
  const tok = new V12.Tokenizer('abcdefghijklmnopqrstuvwxyz');
  const json = V12.saveModel(m, tok);
  const m2 = V12.loadModel(json);
  // R7 闭环：读档恢复出的必须是 V12.GPTModel（L3）实例
  assert.ok(m2 instanceof V12.GPTModel, 'loadModel 恢复的不是 V12.GPTModel（L3）实例');
  // forward 走 L3 blocks：logits 与 legacy 同构模型逐位一致（切换 Δ=0 的快检）
  const seq = [1, 2, 3, 4, 5];
  const m11 = new V11.GPTModel(cfg);
  for (const k in m11.params) m11.params[k].d.set(m.params[k].d);
  assert.deepStrictEqual(
    Array.from(m.forward(seq).logits),
    Array.from(m11.forward(seq).logits),
    'L3 forward 与 legacy 同参模型 logits 不一致');
});

test('T04 切换生效：train/generate/persistence/tokenizer 均来自十层', () => {
  assert.notStrictEqual(V12.train, V11.train, 'V12.train 仍是 legacy 本体');
  assert.notStrictEqual(V12.generate, V11.generate, 'V12.generate 仍是 legacy 本体');
  assert.notStrictEqual(V12.saveModel, V11.saveModel, 'V12.saveModel 仍是 legacy 本体');
  assert.notStrictEqual(V12.loadModel, V11.loadModel, 'V12.loadModel 仍是 legacy 本体');
  assert.notStrictEqual(V12.loadTokenizer, V11.loadTokenizer, 'V12.loadTokenizer 仍是 legacy 本体');
  assert.notStrictEqual(V12.Tokenizer, V11.Tokenizer, 'V12.Tokenizer 仍是 legacy 本体');
  assert.notStrictEqual(V12.BPETokenizer, V11.BPETokenizer, 'V12.BPETokenizer 仍是 legacy 本体');
});

test('K15 盲区防护：GPTModel.prototype 方法集 ⊇ legacy', () => {
  // 背景：K15 只锁 34 个导出键，但 GPTModel 实例方法（如 evaluate）不在其中，
  //   若十层模型漏实现某个 legacy 原型方法，键集门禁不会红（T04 曾漏 evaluate）。
  //   本断言堵住该盲区：L3 模型必须提供 legacy 公开的全部实例方法。
  const l11 = Object.getOwnPropertyNames(V11.GPTModel.prototype)
    .filter((n) => n !== 'constructor').sort();
  const l12 = Object.getOwnPropertyNames(V12.GPTModel.prototype)
    .filter((n) => n !== 'constructor').sort();
  const missing = l11.filter((m) => !l12.includes(m));
  assert.deepStrictEqual(missing, [], `V12.GPTModel 缺失 legacy 原型方法：${missing.join(', ')}`);
});

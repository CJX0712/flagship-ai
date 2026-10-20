'use strict';
/* L3 架构组装 · 模型级 Δ=0 比对（T03 验收 ②/③）
 * 十层实现接管后（V12.GPTModel = NS.L3.gpt_model.GPTModel），与 legacy 单体
 * （= V11.GPTModel）在 5 预设 × 2 seed 下逐位全等。
 *   - gemma 预设走 QK-norm + logit soft-cap；mixtral 走 MoE + top-k；swa 走滑动窗口；
 *   - 构造器/RNG 顺序一致 → 参数逐位相同 → 差异只可能来自 forward/backward 重组装。
 */
const { test } = require('node:test');
const assert = require('node:assert');
require('../../src/index.js');
const NS = globalThis.FlagshipAI;
const V11 = require('../../src/legacy/flagship-engine.v11.js');
const V12 = require('../../src/index.js');

function bitExact(a, b) {
  if (!a || !b || a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (!Object.is(a[i], b[i])) return false;
  return true;
}
const CFG = { vocabSize: 65, blockSize: 16, nLayer: 2, nHead: 4, dModel: 64 };
const PRESETS = ['gpt2', 'llama', 'mixtral', 'swa', 'gemma'];
const SEEDS = [1, 2];
function mkSeq(seed, n) { const s = []; for (let i = 0; i < n; i++) s.push((i * 7 + seed) % CFG.vocabSize); return s; }

test('L3 forward logits 与 legacy Δ=0（5 预设 × 2 seed）', () => {
  for (const p of PRESETS) for (const s of SEEDS) {
    const c = V11.applyPreset(Object.assign({}, CFG, { seed: s }), p);
    const m11 = new V11.GPTModel(c), m12 = new V12.GPTModel(c);
    const seq = mkSeq(s, 15);
    const f11 = m11.forward(seq), f12 = m12.forward(seq);
    assert.ok(bitExact(f11.logits, f12.logits), `${p}/s${s} logits 非逐位一致`);
  }
});

test('L3 backward grads 与 legacy Δ=0（5 预设 × 2 seed）', () => {
  for (const p of PRESETS) for (const s of SEEDS) {
    const c = V11.applyPreset(Object.assign({}, CFG, { seed: s }), p);
    const m11 = new V11.GPTModel(c), m12 = new V12.GPTModel(c);
    const seq = mkSeq(s, 15), targets = seq.slice(1);
    const f11 = m11.forward(seq), f12 = m12.forward(seq);
    const ce11 = V11.crossEntropy(f11.logits, targets, seq.length - 1, CFG.vocabSize, seq.length);
    const ce12 = V12.crossEntropy(f12.logits, targets, seq.length - 1, CFG.vocabSize, seq.length);
    m11.zeroGrads(); m12.zeroGrads();
    m11.backward(f11.cache, ce11.dlogits); m12.backward(f12.cache, ce12.dlogits);
    for (const k in m11.grads) assert.ok(bitExact(m11.grads[k], m12.grads[k]), `${p}/s${s} grad[${k}] 非逐位一致`);
  }
});

test('V12.GPTModel 为独立 L3 类（T04 切换）且构造不改变参数布局', () => {
  // T04 起 V12.GPTModel = NS.L3.gpt_model.GPTModel（verbatim lift 的独立类），
  // 不再继承 legacy.GPTModel —— 「继承 legacy」是 Phase 2 子类覆写策略的遗产，已废弃。
  // 仍需守护的硬契约：构造器 RNG 消费顺序与 legacy 冻结一致 → 参数逐位相同。
  const c = V11.applyPreset(Object.assign({}, CFG, { seed: 5 }), 'llama');
  const m = new V12.GPTModel(c);
  assert.notStrictEqual(V12.GPTModel, V11.GPTModel, 'V12.GPTModel 不应再是 legacy 类');
  const ref = new V11.GPTModel(V11.applyPreset(Object.assign({}, CFG, { seed: 5 }), 'llama'));
  for (const k in ref.params) assert.ok(bitExact(ref.params[k].d, m.params[k].d), `param ${k} 布局改变`);
});

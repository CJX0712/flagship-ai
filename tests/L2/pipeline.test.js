'use strict';
/* L2 编排管线 · 门禁测试（tests/L2）
 * 验收：NS.L2 七件套导出面 + Δ=0 契约（train / generate / archive / evaluate vs v11 legacy）。
 * 全部输入确定性（无 Math.random / Date.now），断言用 Object.is 逐位比对。
 */
const { test } = require('node:test');
const assert = require('node:assert');
require('../../src/index.js');
const NS = globalThis.FlagshipAI;
const V11 = require('../../src/legacy/flagship-engine.v11.js');

/* ---------------- 工具 ---------------- */
function bitExact(a, b) {
  if (!a || !b || a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (!Object.is(a[i], b[i])) return false;
  return true;
}
function paramsEqual(a, b) {
  const ka = Object.keys(a).sort(), kb = Object.keys(b).sort();
  if (ka.length !== kb.length) return false;
  for (const k of ka) {
    if (!Object.prototype.hasOwnProperty.call(b, k)) return false;
    const da = a[k].d, db = b[k].d;
    if (!da || !db || da.length !== db.length) return false;
    for (let i = 0; i < da.length; i++) if (!Object.is(da[i], db[i])) return false;
  }
  return true;
}
// 恰好 65 个不同字符的词料（26+26+10+3），与模型 vocabSize=65 对齐；确定性生成
const ALPHABET = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .!';
function makeCorpus(seed) {
  const rng = NS.L9.rng.mulberry32(seed);
  let s = '';
  for (let i = 0; i < 512; i++) s += ALPHABET[Math.floor(rng() * ALPHABET.length)];
  return s;
}
const BASE_CFG = { vocabSize: 65, blockSize: 16, nLayer: 2, nHead: 4, dModel: 64 };

/* ---------------- 1. 导出面 ---------------- */
test('L2 导出面 · 七件套键齐备且形态正确', () => {
  assert.equal(typeof NS.L2.tokenizer.Tokenizer, 'function');
  assert.equal(typeof NS.L2.tokenizer.BPETokenizer, 'function');
  assert.equal(typeof NS.L2.tokenizer.textToByteChars, 'function');
  assert.equal(typeof NS.L2.tokenizer.byteCharsToText, 'function');
  assert.equal(typeof NS.L2.train.train, 'function');
  assert.equal(typeof NS.L2.generate.generate, 'function');
  assert.equal(typeof NS.L2.sampler.sample, 'function');
  assert.equal(typeof NS.L2.sampler.greedy, 'function');
  assert.equal(typeof NS.L2.sampler.softmaxTemp, 'function');
  assert.equal(typeof NS.L2.evaluate.evaluate, 'function');
  assert.equal(typeof NS.L2.persistence.saveModel, 'function');
  assert.equal(typeof NS.L2.persistence.loadModel, 'function');
  assert.equal(typeof NS.L2.persistence.loadTokenizer, 'function');
  assert.equal(typeof NS.L2.config.hashString, 'function');
  assert.equal(typeof NS.L2.config.runSpec, 'function');
});

/* ---------------- 2. 分词器 ---------------- */
test('tokenizer · 字符级 encode/decode 往返且与 v11 逐位一致', () => {
  const text = 'the quick brown fox jumps over 13 lazy dogs. ai!';
  const t12 = new NS.L2.tokenizer.Tokenizer(text);
  const t11 = new V11.Tokenizer(text);
  assert.equal(t12.vocabSize, t11.vocabSize);
  assert.deepEqual(t12.chars, t11.chars);
  const ids = t12.encode(text);
  assert.deepEqual(ids, t11.encode(text), 'encode 与 v11 不一致');
  assert.equal(t12.decode(ids), text, 'decode 往返失败');
  // 未知字符静默跳过（既定行为）
  assert.deepEqual(t12.encode('你好'), [], '未知字符应被静默跳过');
});

test('tokenizer · BPE（字节级）往返 + 与 v11 逐位一致', () => {
  const text = makeCorpus(7) + '旗舰 AI · flagship 引擎 🚀 émoji 😀';
  const b12 = new NS.L2.tokenizer.BPETokenizer(text, 400);
  const b11 = new V11.BPETokenizer(text, 400);
  assert.equal(b12.kind, 'bpe');
  assert.ok(b12.vocabSize >= 256 && b12.vocabSize === b11.vocabSize);
  assert.deepEqual(b12.chars, b11.chars, 'BPE 词表与 v11 不一致');
  assert.deepEqual(b12.merges, b11.merges, 'merges 与 v11 不一致');
  const probe = 'flagship 引擎 🚀 émoji';
  assert.deepEqual(b12.encode(probe), b11.encode(probe), 'BPE encode 与 v11 不一致');
  assert.equal(b12.decode(b12.encode(probe)), probe, 'BPE 字节级往返失败');
});

test('tokenizer · BPE（字符级基元）往返', () => {
  const text = '旗舰旗舰 AI AI 引擎引擎引擎 测试测试';
  const b = new NS.L2.tokenizer.BPETokenizer(text, 120, { byteLevel: false });
  assert.equal(b.byteLevel, false);
  const ids = b.encode('旗舰 AI 测试');
  assert.ok(ids.length > 0 && ids.every((x) => x < b.vocabSize));
  assert.equal(b.decode(ids), '旗舰 AI 测试');
});

test('tokenizer · textToByteChars/byteCharsToText 可逆（含中文与 emoji）', () => {
  const s = '旗舰 AI 🚀 é😀 ~!@';
  const bc = NS.L2.tokenizer.textToByteChars(s);
  assert.equal(bc.length, new TextEncoder().encode(s).length, '字节级字符数应等于 UTF-8 字节数');
  assert.equal(NS.L2.tokenizer.byteCharsToText(bc), s, '字节级往返失败');
});

/* ---------------- 3. 采样器（standalone） ---------------- */
test('sampler · greedy 返回 argmax', () => {
  const logits = new Float64Array([1, 5, 3, 2]);
  assert.equal(NS.L2.sampler.greedy(logits), 1);
  assert.equal(NS.L2.sampler.greedy(new Float64Array([7])), 0);
});

test('sampler · softmaxTemp 归一化且温度趋低时逼近 one-hot', () => {
  const logits = new Float64Array([1, 2, 3, 4]);
  const p1 = NS.L2.sampler.softmaxTemp(logits, 1);
  let s = 0; for (const p of p1) s += p;
  assert.ok(Math.abs(s - 1) < 1e-12);
  assert.ok(p1[3] > p1[0], '温度采样应保序');
  const pLow = NS.L2.sampler.softmaxTemp(logits, 1e-3);
  assert.ok(pLow[3] > 0.999, '低温应逼近 one-hot');
});

test('sampler · sample 确定性（同种子同结果）与贪心路径不消费 rng', () => {
  const logits = new Float64Array([0.3, -1.2, 2.0, 0.5, 1.1]);
  const rngA = NS.L9.rng.mulberry32(42);
  const rngB = NS.L9.rng.mulberry32(42);
  const opts = { temperature: 0.9, topK: 3, topP: 0.9 };
  assert.equal(NS.L2.sampler.sample(logits, rngA, opts), NS.L2.sampler.sample(logits, rngB, opts));
  // temperature<=0 走贪心，rng 一次都不抽
  let draws = 0;
  const countingRng = () => { draws++; return 0; };
  assert.equal(NS.L2.sampler.sample(logits, countingRng, { temperature: 0 }), 2);
  assert.equal(draws, 0, '贪心路径不应消费 rng');
  // topK=1 时分布被截断为 one-hot，任意 rng 都命中 argmax
  assert.equal(NS.L2.sampler.sample(logits, () => 0.99999, { temperature: 1, topK: 1 }), 2);
});

/* ---------------- 4. config ---------------- */
test('config · hashString 已知答案（FNV-1a offset basis）且带雪崩', () => {
  assert.equal(NS.L2.config.hashString(''), (0x811c9dc5 >>> 0).toString(36), 'FNV-1a 空串应为 offset basis');
  assert.equal(NS.L2.config.hashString('a'), NS.L2.config.hashString('a'), '同输入同哈希');
  const h1 = NS.L2.config.hashString('flagship');
  const h2 = NS.L2.config.hashString('flagship ');
  assert.notEqual(h1, h2, '单字符差异应改哈希');
  assert.ok(/^[0-9a-z]+$/.test(h1), 'base36 输出');
});

test('config · runSpec 键序无关 + 对 seed/corpusHash/cfg 敏感', () => {
  const s1 = NS.L2.config.runSpec({ seed: 42, corpusHash: 'abc', cfg: { b: 1, a: 2 } });
  const s2 = NS.L2.config.runSpec({ cfg: { a: 2, b: 1 }, corpusHash: 'abc', seed: 42 });
  assert.equal(s1, s2, '键序不应影响 spec');
  assert.notEqual(s1, NS.L2.config.runSpec({ seed: 43, corpusHash: 'abc', cfg: { a: 2, b: 1 } }));
  assert.notEqual(s1, NS.L2.config.runSpec({ seed: 42, corpusHash: 'abd', cfg: { a: 2, b: 1 } }));
  assert.notEqual(s1, NS.L2.config.runSpec({ seed: 42, corpusHash: 'abc', cfg: { a: 3, b: 1 } }));
  assert.ok(s1.startsWith('flagship-spec|'), 'spec 前缀');
});

/* ---------------- 5. evaluate Δ=0 ---------------- */
test('evaluate · L2.evaluate.evaluate 与 legacy evaluate 方法 Δ=0', () => {
  const cfg = NS.L3.presets.applyPreset(Object.assign({}, BASE_CFG, { seed: 5 }), 'gpt2');
  const m11 = new V11.GPTModel(V11.applyPreset(Object.assign({}, BASE_CFG, { seed: 5 }), 'gpt2'));
  const m12 = new NS.L3.gpt_model.GPTModel(cfg);
  const toks = new NS.L2.tokenizer.Tokenizer(makeCorpus(31)).encode(makeCorpus(31));
  const r11 = m11.evaluate(toks);
  const r12 = NS.L2.evaluate.evaluate(m12, toks);
  assert.ok(Object.is(r11.loss, r12.loss), `loss 不一致：${r11.loss} vs ${r12.loss}`);
  assert.ok(Object.is(r11.acc, r12.acc), `acc 不一致：${r11.acc} vs ${r12.acc}`);
  // 短序列边界（T < 2）
  assert.deepEqual(NS.L2.evaluate.evaluate(m12, [3]), { loss: Infinity, acc: 0 });
});

/* ---------------- 6. Δ=0 主矩阵：train + generate + archive ---------------- */
for (const preset of ['gpt2', 'llama', 'mixtral', 'swa', 'gemma']) {
  for (const seed of [1, 2]) {
    test(`Δ=0 · ${preset} · seed=${seed}（train/generate/archive 全链路 vs v11）`, () => {
      const cfg11 = V11.applyPreset(Object.assign({}, BASE_CFG, { seed }), preset);
      const cfg12 = NS.L3.presets.applyPreset(Object.assign({}, BASE_CFG, { seed }), preset);
      const m11 = new V11.GPTModel(cfg11);
      const m12 = new NS.L3.gpt_model.GPTModel(cfg12);
      // 先决：构造 RNG 消费顺序一致（R1）——初始参数必须逐位相同
      assert.ok(paramsEqual(m11.params, m12.params), '初始参数不一致（构造 RNG 顺序漂移）');

      const corpus = makeCorpus(1000 + seed);
      const tok = new NS.L2.tokenizer.Tokenizer(corpus);
      const toks = tok.encode(corpus);
      assert.equal(tok.vocabSize, BASE_CFG.vocabSize);

      // ---- TRAIN Δ=0 ----
      const r11 = V11.train(m11, toks, { steps: 20, seed: 42, onStep: null });
      const r12 = NS.L2.train.train(m12, toks, { steps: 20, seed: 42, onStep: null });
      assert.equal(r11.history.length, 20);
      assert.equal(r12.history.length, 20);
      let maxDL = 0;
      for (let i = 0; i < 20; i++) {
        assert.ok(Object.is(r11.history[i], r12.history[i]),
          `${preset}/seed=${seed} history[${i}]：${r11.history[i]} vs ${r12.history[i]}`);
        maxDL = Math.max(maxDL, Math.abs(r11.history[i] - r12.history[i]));
      }
      assert.ok(Object.is(r11.finalLoss, r12.finalLoss));
      assert.ok(paramsEqual(m11.params, m12.params), '训练后参数不一致');
      // onStep 回调逐参数一致（step, loss, lr, norm）
      const c11 = [], c12 = [];
      const m11b = new V11.GPTModel(V11.applyPreset(Object.assign({}, BASE_CFG, { seed: 9 }), preset));
      const m12b = new NS.L3.gpt_model.GPTModel(NS.L3.presets.applyPreset(Object.assign({}, BASE_CFG, { seed: 9 }), preset));
      V11.train(m11b, toks, { steps: 5, seed: 7, onStep: (a, b, c, d) => c11.push([a, b, c, d]) });
      NS.L2.train.train(m12b, toks, { steps: 5, seed: 7, onStep: (a, b, c, d) => c12.push([a, b, c, d]) });
      assert.equal(c11.length, c12.length);
      for (let i = 0; i < c11.length; i++) {
        assert.ok(c11[i].every((v, j) => Object.is(v, c12[i][j])), `onStep[${i}] 不一致`);
      }

      // ---- GENERATE Δ=0（useCache 开/关）----
      for (const useCache of [true, false]) {
        const g11 = V11.generate(m11, tok, 'abc', { maxNew: 15, seed, useCache });
        const g12 = NS.L2.generate.generate(m12, tok, 'abc', { maxNew: 15, seed, useCache });
        assert.equal(g11.text, g12.text, `generate text 不一致（useCache=${useCache}）`);
        assert.equal(g11.ids.length, 15);
        for (let i = 0; i < g11.ids.length; i++) {
          assert.equal(g11.ids[i], g12.ids[i], `generate ids[${i}] 不一致（useCache=${useCache}）`);
        }
      }

      // ---- ARCHIVE Δ=0：saveModel(v11) -> loadModel(L2/L3) -> forward 逐位 ----
      const json = V11.saveModel(m11, tok);
      const m13 = NS.L2.persistence.loadModel(json);
      const probe = toks.slice(0, BASE_CFG.blockSize);
      const l11 = m11.forward(probe).logits;
      const l13 = m13.forward(probe).logits;
      assert.ok(bitExact(l11, l13), 'archive 往返 forward logits 不一致');
      // loadTokenizer 往返（字符级）
      const t2 = NS.L2.persistence.loadTokenizer(json);
      assert.equal(t2.kind, 'char');
      assert.deepEqual(t2.chars, tok.chars);
      assert.deepEqual(t2.encode('abc xyz'), tok.encode('abc xyz'));
      assert.equal(t2.decode(t2.encode(corpus.slice(0, 80))), corpus.slice(0, 80));
      // 读档模型确为 L3 血缘（覆盖 L3 Δ=0 存档路径）
      assert.ok(m13 instanceof NS.L3.gpt_model.GPTModel, 'loadModel 应返回 L3 GPTModel');
      // 未知格式拒绝
      assert.throws(() => NS.L2.persistence.loadModel('{"format":"other"}'), /无法识别的模型格式/);
    });
  }
}

/* ---------------- 7. BPE 存档往返 ---------------- */
test('persistence · BPE 分词器存档/读档往返（merges + encode 一致）', () => {
  const text = makeCorpus(77) + '旗舰 AI flagship';
  const bpe = new NS.L2.tokenizer.BPETokenizer(text, 400);
  const cfg = NS.L3.presets.applyPreset(Object.assign({}, BASE_CFG, { seed: 3 }), 'gpt2');
  const m = new NS.L3.gpt_model.GPTModel(cfg);
  const json = NS.L2.persistence.saveModel(m, bpe);
  const b2 = NS.L2.persistence.loadTokenizer(json);
  assert.equal(b2.kind, 'bpe');
  assert.equal(b2.byteLevel, true);
  assert.deepEqual(b2.merges, bpe.merges, 'BPE merges 读档不一致');
  assert.deepEqual(b2.encode('flagship 旗舰'), bpe.encode('flagship 旗舰'), 'BPE 读档后 encode 不一致');
  assert.equal(b2.decode(b2.encode('flagship 旗舰')), 'flagship 旗舰');
  // 存档无词表时返回 null
  assert.equal(NS.L2.persistence.loadTokenizer({ format: 'flagship-ai' }), null);
});

/* ---------------- 8. train 进阶路径：state 续训 / fixedBatch ----------------
 * 契约：同语料同 steps，20+20（state/stepOffset 续训）与 40 一步训练逐位一致
 * （RNG 流、Adam 动量 m/v、步数 t、lr 调度全部连续）；L2 与 V11 各自都满足且互等。
 */
test('train · state 续训/fixedBatch 路径与 v11 逐位一致', () => {
  const mk = () => {
    const m11 = new V11.GPTModel(V11.applyPreset(Object.assign({}, BASE_CFG, { seed: 11 }), 'gpt2'));
    const m12 = new NS.L3.gpt_model.GPTModel(NS.L3.presets.applyPreset(Object.assign({}, BASE_CFG, { seed: 11 }), 'gpt2'));
    return [m11, m12];
  };
  const toks = new NS.L2.tokenizer.Tokenizer(makeCorpus(55)).encode(makeCorpus(55));

  // 一步 40 步 vs 20+20 续训（V11 自身不变式 + L2 自身不变式 + 跨实现互等）
  const [a11, a12] = mk();
  const r11 = V11.train(a11, toks, { steps: 40, seed: 3, totalSteps: 40 });
  const r12 = NS.L2.train.train(a12, toks, { steps: 40, seed: 3, totalSteps: 40 });
  assert.ok(bitExact(r11.history, r12.history), '40 步 history 不一致');
  assert.ok(paramsEqual(a11.params, a12.params), '40 步参数不一致');

  const [b11, b12] = mk();
  const c11 = V11.train(b11, toks, { steps: 20, seed: 3, totalSteps: 40 });
  const c11b = V11.train(b11, toks, { steps: 20, seed: 3, totalSteps: 40, state: c11.state, stepOffset: 20 });
  const c12 = NS.L2.train.train(b12, toks, { steps: 20, seed: 3, totalSteps: 40 });
  const c12b = NS.L2.train.train(b12, toks, { steps: 20, seed: 3, totalSteps: 40, state: c12.state, stepOffset: 20 });
  // 续训态完整携带：t / rngA / m / v
  assert.equal(c11b.state.t, 40);
  assert.ok(Object.is(c11b.state.rngA, r11.state.rngA), 'V11 续训 rngA 未连续');
  assert.ok(Object.is(c12b.state.rngA, r11.state.rngA), 'L2 续训 rngA 未连续');
  const ks = Object.keys(r11.state.m);
  for (const k of ks) {
    assert.ok(bitExact(c11b.state.m[k], r11.state.m[k]), `V11 m[${k}] 不连续`);
    assert.ok(bitExact(c12b.state.m[k], r11.state.m[k]), `L2 m[${k}] 不连续`);
    assert.ok(bitExact(c11b.state.v[k], r11.state.v[k]), `V11 v[${k}] 不连续`);
    assert.ok(bitExact(c12b.state.v[k], r11.state.v[k]), `L2 v[${k}] 不连续`);
  }
  // 续训终点 == 一步训练终点（逐位）
  assert.ok(paramsEqual(b11.params, a11.params), 'V11 续训终点 != 一步训练');
  assert.ok(paramsEqual(b12.params, a12.params), 'L2 续训终点 != 一步训练');

  // fixedBatch 路径（跳过批次采样）
  const fb = toks.slice(0, BASE_CFG.blockSize);
  const [d11, d12] = mk();
  const f11 = V11.train(d11, toks, { steps: 10, fixedBatch: fb });
  const f12 = NS.L2.train.train(d12, toks, { steps: 10, fixedBatch: fb });
  assert.ok(bitExact(f11.history, f12.history), 'fixedBatch history 不一致');
  assert.ok(paramsEqual(d11.params, d12.params), 'fixedBatch 参数不一致');
  // 入参守卫与 v11 一致
  assert.throws(() => NS.L2.train.train(d12, [1], {}), /训练语料至少需要 2 个 token/);
  assert.throws(() => NS.L2.train.train(d12, toks, { fixedBatch: [1] }), /固定批次至少需要 2 个 token/);
});

/* ---------------- 9. generate 采样组合：贪心/topK/topP ----------------
 */
test('generate · 贪心/topK/topP 组合与 v11 逐位一致（两种 cache 路径）', () => {
  const mk = () => {
    const m11 = new V11.GPTModel(V11.applyPreset(Object.assign({}, BASE_CFG, { seed: 21 }), 'gpt2'));
    const m12 = new NS.L3.gpt_model.GPTModel(NS.L3.presets.applyPreset(Object.assign({}, BASE_CFG, { seed: 21 }), 'gpt2'));
    return [m11, m12];
  };
  const corpus = makeCorpus(66);
  const tok = new NS.L2.tokenizer.Tokenizer(corpus);
  const combos = [
    { maxNew: 12, seed: 5, useCache: true, temperature: 0 },                          // 贪心 + cache
    { maxNew: 12, seed: 5, useCache: false, temperature: 0 },                         // 贪心 + 全序列回退
    { maxNew: 12, seed: 6, useCache: true, temperature: 0.8, topK: 5 },
    { maxNew: 12, seed: 6, useCache: false, temperature: 0.8, topK: 5 },
    { maxNew: 12, seed: 7, useCache: true, temperature: 1.0, topP: 0.9 },
    { maxNew: 12, seed: 7, useCache: false, temperature: 0.9, topP: 0.5 },
    { maxNew: 12, seed: 8, useCache: true, temperature: 0.8, topK: 8, topP: 0.8 },
    { maxNew: 12, seed: 8, useCache: false, temperature: 0.8, topK: 8, topP: 0.8 },
  ];
  for (const opts of combos) {
    const [m11, m12] = mk();
    const g11 = V11.generate(m11, tok, 'abc', opts);
    const g12 = NS.L2.generate.generate(m12, tok, 'abc', opts);
    assert.equal(g11.text, g12.text, `text 不一致：${JSON.stringify(opts)}`);
    assert.ok(g11.ids.every((v, i) => Object.is(v, g12.ids[i])), `ids 不一致：${JSON.stringify(opts)}`);
  }
  // maxTokens 别名路径
  {
    const [m11, m12] = mk();
    const a = V11.generate(m11, tok, 'abc', { maxTokens: 5, seed: 9, useCache: false });
    const b = NS.L2.generate.generate(m12, tok, 'abc', { maxTokens: 5, seed: 9, useCache: false });
    assert.equal(a.ids.length, 5);
    assert.ok(a.ids.every((v, i) => Object.is(v, b.ids[i])), 'maxTokens 别名 ids 不一致');
  }
  // 空 prompt（encode 为空）-> rng()*V 兜底 token 两条路径一致
  {
    const [m11, m12] = mk();
    const g11 = V11.generate(m11, tok, '你好', { maxNew: 6, seed: 10, useCache: false });
    const g12 = NS.L2.generate.generate(m12, tok, '你好', { maxNew: 6, seed: 10, useCache: false });
    assert.equal(g11.ids.length, 6);
    assert.ok(g11.ids.every((v, i) => Object.is(v, g12.ids[i])), '空 prompt 兜底 ids 不一致');
    assert.equal(g11.text, g12.text);
  }
});

/* ---------------- 10. persistence 闭环：存档 -> loadModel -> generate ----------------
 */
test('persistence · saveModel -> loadModel -> generate 闭环（跨实现互认）', () => {
  const corpus = makeCorpus(88);
  const tok = new NS.L2.tokenizer.Tokenizer(corpus);
  const toks = tok.encode(corpus);
  const m = new NS.L3.gpt_model.GPTModel(NS.L3.presets.applyPreset(Object.assign({}, BASE_CFG, { seed: 13 }), 'gpt2'));
  NS.L2.train.train(m, toks, { steps: 12, seed: 1 });
  const json = NS.L2.persistence.saveModel(m, tok);

  // L2 读档 -> generate 与原模型逐位一致（cache 开/关）
  const m13 = NS.L2.persistence.loadModel(json);
  for (const useCache of [true, false]) {
    const g0 = NS.L2.generate.generate(m, tok, 'abc', { maxNew: 10, seed: 4, useCache });
    const g1 = NS.L2.generate.generate(m13, tok, 'abc', { maxNew: 10, seed: 4, useCache });
    assert.equal(g0.text, g1.text, `闭环 text 不一致（useCache=${useCache}）`);
    assert.ok(g0.ids.every((v, i) => Object.is(v, g1.ids[i])), `闭环 ids 不一致（useCache=${useCache}）`);
  }
  // 跨实现互认：V11 读同一存档 -> V11 generate 与 L12 原模型逐位一致
  const mV = V11.loadModel(json);
  const g0 = NS.L2.generate.generate(m, tok, 'abc', { maxNew: 10, seed: 4, useCache: false });
  const gV = V11.generate(mV, tok, 'abc', { maxNew: 10, seed: 4, useCache: false });
  assert.equal(g0.text, gV.text, '跨实现互认 text 不一致');
  assert.ok(g0.ids.every((v, i) => Object.is(v, gV.ids[i])), '跨实现互认 ids 不一致');
});

test('persistence · loadTokenizer 恢复 ranks/stoi/vocabSize 逐位', () => {
  const text = makeCorpus(99) + '旗舰 flagship';
  const bpe = new NS.L2.tokenizer.BPETokenizer(text, 400);
  const m = new NS.L3.gpt_model.GPTModel(NS.L3.presets.applyPreset(Object.assign({}, BASE_CFG, { seed: 2 }), 'gpt2'));
  const json = NS.L2.persistence.saveModel(m, bpe);
  const b2 = NS.L2.persistence.loadTokenizer(json);
  assert.equal(b2.vocabSize, bpe.vocabSize, 'vocabSize 不一致');
  assert.equal(b2.kind, bpe.kind);
  assert.equal(b2.byteLevel, bpe.byteLevel);
  assert.equal(b2.stoi.size, bpe.stoi.size, 'stoi 规模不一致');
  for (const [k, v] of bpe.stoi) assert.ok(Object.is(b2.stoi.get(k), v), `stoi[${k}] 不一致`);
  assert.equal(b2.ranks.size, bpe.ranks.size, 'ranks 规模不一致');
  for (const [k, v] of bpe.ranks) assert.ok(Object.is(b2.ranks.get(k), v), `ranks[${k}] 不一致`);
  assert.deepEqual(b2.merges, bpe.merges, 'merges 不一致');
});

/* ---------------- 11. 量化权重存档往返 ---------------- */
test('persistence · 量化模型 saveModel -> loadModel（L3 血缘 + 参数形态）', () => {
  const cfg = NS.L3.presets.applyPreset(Object.assign({}, BASE_CFG, { seed: 8 }), 'gpt2');
  const m = new NS.L3.gpt_model.GPTModel(cfg);
  const mq = m.quantize();
  const json = V11.saveModel(mq, null);
  const m13 = NS.L2.persistence.loadModel(json);
  assert.equal(m13.quantized, true);
  const key = Object.keys(mq.params)[0];
  assert.ok(m13.params[key].q instanceof Int8Array, '量化参数应恢复为 Int8Array');
  assert.equal(Array.from(m13.params[key].q).join(','), Array.from(mq.params[key].q).join(','));
  // 与 v11 读档路径逐位一致（同一存档两种读档）
  const mV = V11.loadModel(json);
  const probe = [1, 2, 3, 4, 5];
  assert.ok(bitExact(mV.forward(probe).logits, m13.forward(probe).logits),
    '量化存档 L2 读档与 v11 读档 forward 不一致');
});

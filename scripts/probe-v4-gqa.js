/* ============================================================================
 * 旗舰 AI · v4 GQA（分组查询注意力）专项验证
 * ----------------------------------------------------------------------------
 *  1. GQA 中心差分梯度检验（解析反向 vs 数值，err/tol ≤ 1）
 *  2. GQA KV Cache 逐步一致性（forwardStep 累积 == forward 全序列末位）
 *  3. GQA 持久化无损往返（save/load 后 cfg.nKvHead 保留 + 前向逐位一致）
 *  4. GQA 参数节省（nKvHead<nHead 时 wqkv 与总参更少）
 *  5. GQA 可训练性（小语料过拟合 loss -> <0.1）
 *
 * 运行：node scripts/probe-v4-gqa.js   （全通过 exit 0）
 * 作者：晨星  ·  License: MIT
 * ========================================================================*/
const F = require('../src/index.js');

function gradCheckGQA() {
  const V = 11, T = 6, H = 4, KV = 2;
  const model = new F.GPTModel({
    vocabSize: V, blockSize: 7, nLayer: 1, nHead: H, nKvHead: KV,
    dModel: 16, norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope', seed: 7,
  });
  const rng = F.mulberry32(5);
  const tokens = [];
  for (let i = 0; i < T; i++) tokens.push(Math.floor(rng() * V));
  const targets = [];
  for (let i = 1; i < T; i++) targets.push(tokens[i]);
  const { logits, cache } = model.forward(tokens);
  const { dlogits } = F.crossEntropy(logits, targets, T - 1, V, T);
  model.backward(cache, dlogits);
  const lossOf = () => F.crossEntropy(model.forward(tokens).logits, targets, T - 1, V).loss;

  const all = [];
  for (const key in model.params) {
    const p = model.params[key].d;
    for (let i = 0; i < p.length; i++) all.push([key, i]);
  }
  const srng = F.mulberry32(99);
  let cand = all;
  if (all.length > 500) {
    cand = all.slice();
    for (let i = cand.length - 1; i > 0; i--) {
      const j = Math.floor(srng() * (i + 1));
      [cand[i], cand[j]] = [cand[j], cand[i]];
    }
    cand = cand.slice(0, 500);
  }
  const e = 1e-5;
  let maxRatio = 0, checked = 0, skipped = 0;
  for (const [key, i] of cand) {
    const p = model.params[key].d;
    const orig = p[i];
    p[i] = orig + e; const lp = lossOf();
    p[i] = orig - e; const lm = lossOf();
    p[i] = orig;
    const num = (lp - lm) / (2 * e);
    const ana = model.grads[key][i];
    if (Math.abs(ana) < 1e-12 && Math.abs(num) < 1e-12) { skipped++; continue; }
    const err = Math.abs(ana - num);
    const tol = 1e-7 + 1e-4 * Math.max(Math.abs(ana), Math.abs(num));
    const ratio = err / tol;
    if (ratio > maxRatio) maxRatio = ratio;
    checked++;
  }
  return { pass: checked > 100 && maxRatio <= 1, detail: `GQA 梯度检验 ${checked} 参数(跳过 ${skipped} 零梯度) 最大 err/tol=${maxRatio.toExponential(2)}` };
}

function kvStepwiseGQA() {
  const V = 13, B = 12, H = 4, KV = 2;
  const model = new F.GPTModel({ vocabSize: V, blockSize: B, nLayer: 2, nHead: H, nKvHead: KV, dModel: 16, seed: 3 });
  const seq = [1, 3, 5, 7, 9, 11, 2, 4, 6, 8].slice(0, 10);
  const kv = F.initKVCache(model);
  let maxDiff = 0;
  for (let i = 0; i < seq.length; i++) {
    const step = F.forwardStep(model, seq[i], i, kv).logits;       // 位置 i 的末位 logits
    const full = model.forward(seq.slice(0, i + 1)).logits;        // 全序列重算
    const off = i * V;
    for (let j = 0; j < V; j++) maxDiff = Math.max(maxDiff, Math.abs(step[j] - full[off + j]));
    if (maxDiff > 1e-9) break;
  }
  return { pass: maxDiff === 0, detail: `GQA 逐步 forwardStep 累积 vs forward 全序列 末位 logits 最大差异 ${maxDiff}（要求 0）` };
}

function persistGQA() {
  const V = 13, H = 4, KV = 2;
  const model = new F.GPTModel({
    vocabSize: V, blockSize: 8, nLayer: 1, nHead: H, nKvHead: KV, dModel: 16,
    norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope', seed: 5,
  });
  const json = F.saveModel(model, null);
  const r = F.loadModel(json);
  const cfgOk = r.cfg.nKvHead === KV && r.KV === KV;
  const seq = [0, 1, 2, 3, 4, 5, 6, 7];
  const l1 = model.forward(seq).logits, l2 = r.forward(seq).logits;
  let maxDiff = 0;
  for (let i = 0; i < l1.length; i++) maxDiff = Math.max(maxDiff, Math.abs(l1[i] - l2[i]));
  return { pass: cfgOk && maxDiff === 0, detail: `cfg.nKvHead 保留=${cfgOk}；前向逐位一致 maxDiff=${maxDiff}` };
}

function paramSavingGQA() {
  const cfg = { vocabSize: 50, blockSize: 16, nLayer: 2, nHead: 4, dModel: 16, seed: 1 };
  const m1 = new F.GPTModel(cfg);
  const m2 = new F.GPTModel(Object.assign({}, cfg, { nKvHead: 2 }));
  const wqkv1 = m1.params['l0.wqkv'].d.length, wqkv2 = m2.params['l0.wqkv'].d.length;
  return {
    pass: wqkv2 < wqkv1 && m2.paramCount() < m1.paramCount(),
    detail: `wqkv 参数 MHA=${wqkv1} GQA(nKvHead=2)=${wqkv2}；总参 MHA=${m1.paramCount()} GQA=${m2.paramCount()}`,
  };
}

function trainableGQA() {
  const POEM = '朝辞白帝彩云间，千里江陵一日还。两岸猿声啼不住，轻舟已过万重山。';
  const tok = new F.Tokenizer(POEM);
  const ids = tok.encode(POEM);
  const model = new F.GPTModel({ vocabSize: tok.vocabSize, blockSize: ids.length, nLayer: 2, nHead: 4, nKvHead: 2, dModel: 32, seed: 1 });
  const { history } = F.train(model, ids, { steps: 300, lr: 3e-3, warmup: 20, wd: 0, fixedBatch: ids, logEvery: 1e9 });
  const last = history[history.length - 1];
  return { pass: last < 0.1, detail: `GQA 过拟合 初始 ${history[0].toFixed(2)} -> 最终 loss ${last.toFixed(4)}（阈值 <0.1）` };
}

const CHECKS = [
  { name: 'GQA 中心差分梯度检验', run: gradCheckGQA },
  { name: 'GQA KV Cache 逐步一致性', run: kvStepwiseGQA },
  { name: 'GQA 持久化无损往返', run: persistGQA },
  { name: 'GQA 参数节省', run: paramSavingGQA },
  { name: 'GQA 可训练性（过拟合）', run: trainableGQA },
];

function safeRun(fn) {
  try { const r = fn() || {}; return { pass: r.pass !== false, detail: r.detail || '' }; }
  catch (e) { return { pass: false, detail: '异常: ' + (e && e.message) }; }
}

const results = CHECKS.map(c => ({ name: c.name, ...safeRun(c.run) }));
const W = 50;
console.log('🔬 旗舰 AI · v4 GQA 专项验证（Node ' + process.version + '）');
console.log('─'.repeat(W));
for (let i = 0; i < results.length; i++) {
  const r = results[i];
  console.log(` ${r.pass ? '✅' : '❌'} ${String(i + 1).padStart(2)}. ${r.name}`);
  console.log(`      ${r.detail}`);
}
console.log('─'.repeat(W));
const pass = results.filter(r => r.pass).length;
console.log(`结果：${pass}/${results.length} 通过 ${pass === results.length ? '✅ 全绿' : '❌ 存在失败'}`);
process.exitCode = pass === results.length ? 0 : 1;

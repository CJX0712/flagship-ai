/* 定位现代架构训练 NaN 的根因 */
const F = require('../src/flagship-engine.js');
const { CORPUS } = require('../src/corpus.js');

const tok = new F.Tokenizer(CORPUS);
const ids = tok.encode(CORPUS);
const base = { vocabSize: tok.vocabSize, blockSize: 32, nLayer: 2, nHead: 4, dModel: 32, seed: 42 };

for (const preset of ['gpt2', 'llama']) {
  const m = new F.GPTModel(F.applyPreset(base, preset));
  const seq = ids.slice(0, 32);
  const { logits, cache } = m.forward(seq);
  let nan = 0, mx = 0;
  for (let i = 0; i < logits.length; i++) { if (!isFinite(logits[i])) nan++; mx = Math.max(mx, Math.abs(logits[i])); }
  const { loss } = F.crossEntropy(logits, seq.slice(1), 31, m.cfg.vocabSize, 32);
  console.log(preset + ' 初始：logits NaN/Inf=' + nan + ' · |logits|max=' + mx.toExponential(2) + ' · loss=' + loss.toFixed(4));

  // 逐层检查激活值
  const L = cache.layers;
  for (let i = 0; i < L.length; i++) {
    const c = L[i];
    const stats = (name, arr) => {
      let bad = 0, mm = 0;
      for (let k = 0; k < arr.length; k++) { if (!isFinite(arr[k])) bad++; mm = Math.max(mm, Math.abs(arr[k])); }
      return name + '=' + mm.toExponential(1) + (bad ? '(NaN ' + bad + ')' : '');
    };
    console.log('   层' + i + ': ' + [stats('A', c.A), stats('qkv', c.qkv), stats('ctx', c.ctx), stats('M', c.M), stats('pre1', c.pre1), stats('h1', c.h1)].join(' '));
  }
}

// 训练逐 step 追踪
console.log('\n--- LLaMA 训练逐步 ---');
const m = new F.GPTModel(F.applyPreset(base, 'llama'));
F.train(m, ids, {
  steps: 12, lr: 3e-3, batch: 32, warmup: 4, seed: 0, logEvery: 1,
  onStep: (step, loss, lr, norm) => {
    console.log('  step ' + step + ' loss=' + loss + ' lr=' + lr.toFixed(6) + ' |g|=' + norm.toExponential(3));
  },
});

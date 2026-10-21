/* v3.0 探路脚本：BPE / RoPE 不变量 / 现代架构 KV cache 一致性 / 多架构梯度检验 */
const F = require('../src/index.js');
const { CORPUS } = require('../src/corpus.js');

console.log('=== 1. BPE 分词器往返无损 ===');
const t0 = Date.now();
const bpe = new F.BPETokenizer(CORPUS, 800);
console.log('训练 ' + (Date.now() - t0) + 'ms · 词表 ' + bpe.vocabSize + ' · merges ' + bpe.merges.length);
const samples = ['朝辞白帝彩云间', '旗舰 AI 锻造世界顶级模型', 'Hello, world! 123 \u{1F680} emoji', '混合 English 与中文 + punctuation!!!'];
let allOk = true;
for (const s of samples) {
  const ids = bpe.encode(s), back = bpe.decode(ids);
  const okk = back === s;
  allOk = allOk && okk;
  console.log('  ' + (okk ? '[OK]' : '[FAIL]') + ' ' + JSON.stringify(s) + ' -> ' + ids.length + ' token');
}
console.log(allOk ? '=> 全部往返无损（含中文/emoji/标点/跨 token 多字节汉字）' : '=> 有损');

console.log('\n=== 2. 压缩率 vs 词表规模（中文语料的真实权衡） ===');
const charTok = new F.Tokenizer(CORPUS);
const charLen = charTok.encode(CORPUS).length;
console.log('  字符级基线：' + charLen + ' token，词表 ' + charTok.vocabSize);
for (const mode of [true, false]) {
  console.log('  ' + (mode ? '字节级基元（GPT-2 标准）' : '字符级基元（中文友好）'));
  for (const V of [320, 400, 600, 900, 1400]) {
    const tk = new F.BPETokenizer(CORPUS, V, { byteLevel: mode });
    const n = tk.encode(CORPUS).length;
    const ratio = charLen / n;
    console.log('    目标 ' + String(V).padStart(4) + ' → 实得词表 ' + String(tk.vocabSize).padStart(4) +
      ' · ' + String(n).padStart(4) + ' token · 相对字符级 ' + ratio.toFixed(2) + 'x' + (ratio >= 1 ? ' ✅' : ' ⚠️'));
  }
}
console.log('  结论：中文语料下字符级基元严格优于字节级 —— 每汉字 3 个 UTF-8 字节，');
console.log('        字节级 BPE 必须先把这 3 字节"合并回来"才不亏，小语料学不到足够 merge。');

console.log('\n=== 3. RoPE 相对位置不变性 ===');
const D = 32, H = 4, dh = D / H;
const rng = F.mulberry32(1);
const L = 4, shift = 5;
const raw = new Float64Array(L * 3 * D);
for (let i = 0; i < raw.length; i++) raw[i] = F.randn(rng);
const qkvA = Float64Array.from(raw), qkvB = Float64Array.from(raw);
F.ropeApply(qkvA, L, D, H, dh, 0, 1);
F.ropeApply(qkvB, L, D, H, dh, shift, 1);
let maxDiff = 0, n = 0;
for (let a = 0; a < L; a++) {
  for (let b = 0; b < L; b++) {
    let s1 = 0, s2 = 0;
    for (let p = 0; p < dh; p++) {
      s1 += qkvA[a * 3 * D + p] * qkvA[b * 3 * D + D + p];
      s2 += qkvB[a * 3 * D + p] * qkvB[b * 3 * D + D + p];
    }
    maxDiff = Math.max(maxDiff, Math.abs(s1 - s2)); n++;
  }
}
console.log((maxDiff < 1e-12 ? '[OK]' : '[FAIL]') + ' 整体平移 ' + shift + ' 位后，' + n + ' 个注意力分数最大差异 ' + maxDiff.toExponential(2));
console.log('  -> 分数只依赖相对距离：RoPE 天然支持长度外推 + 滑动窗口 KV cache 精确一致');

console.log('\n=== 4. 多架构手写反向传播梯度检验（中心差分对照） ===');
function gradCheck(model, toks, label) {
  const V = model.cfg.vocabSize;
  const T = Math.min(toks.length, model.cfg.blockSize);
  const seq = toks.slice(0, T), targets = seq.slice(1);
  const lossOf = () => {
    const { logits } = model.forward(seq);
    return F.crossEntropy(logits, targets, T - 1, V, T).loss;
  };
  model.zeroGrads();
  const { logits, cache } = model.forward(seq);
  model.backward(cache, F.crossEntropy(logits, targets, T - 1, V, T).dlogits);
  const e = 1e-5, atol = 1e-7, rtol = 1e-4;
  let worst = 0, checked = 0, skipped = 0, wi = '';
  for (const key in model.params) {
    const p = model.params[key].d, g = model.grads[key];
    const cnt = Math.min(p.length, 60);
    for (let s = 0; s < cnt; s++) {
      const i = Math.floor(s * p.length / cnt);
      const orig = p[i];
      p[i] = orig + e; const lp = lossOf();
      p[i] = orig - e; const lm = lossOf();
      p[i] = orig;
      const num = (lp - lm) / (2 * e), ana = g[i];
      const tol = atol + rtol * Math.max(Math.abs(ana), Math.abs(num));
      if (Math.abs(ana) < 1e-12 && Math.abs(num) < 1e-12) { skipped++; continue; }
      const err = Math.abs(ana - num) / tol;
      if (err > worst) { worst = err; wi = key + '[' + i + ']'; }
      checked++;
    }
  }
  console.log('  ' + (worst < 1 ? '[OK]' : '[FAIL]') + ' ' + label.padEnd(30) + ' 检验 ' + String(checked).padStart(4) + ' 分量（跳过 ' + skipped + ' 零梯度）· 最大 err/tol=' + worst.toExponential(2) + (worst >= 1 ? ' | ' + wi : ''));
  return worst < 1;
}
const gbase = { vocabSize: 120, blockSize: 12, nLayer: 2, nHead: 4, dModel: 32, seed: 7 };
const seq = Array.from({ length: 12 }, (_, i) => i % 120);
let ok = true;
ok = gradCheck(new F.GPTModel(gbase), seq, 'GPT-2 (LN+GELU+learned)') && ok;
ok = gradCheck(new F.GPTModel(F.applyPreset(gbase, 'llama')), seq, 'LLaMA (RMS+SwiGLU+RoPE)') && ok;
ok = gradCheck(new F.GPTModel({ ...gbase, norm: 'rmsnorm' }), seq, '仅 RMSNorm') && ok;
ok = gradCheck(new F.GPTModel({ ...gbase, ffn: 'swiglu' }), seq, '仅 SwiGLU') && ok;
ok = gradCheck(new F.GPTModel({ ...gbase, pos: 'rope' }), seq, '仅 RoPE') && ok;

console.log('\n=== 5. 各架构 KV cache 与全序列逐位一致 ===');
const tok = new F.Tokenizer(CORPUS);
for (const preset of ['gpt2', 'llama']) {
  const model = new F.GPTModel(F.applyPreset(
    { vocabSize: tok.vocabSize, blockSize: 16, nLayer: 2, nHead: 4, dModel: 32, seed: 3 }, preset));
  for (const N of [10, 40]) {
    const a = F.generate(model, tok, '朝辞', { maxNew: N, temperature: 0, useCache: true, seed: 1 });
    const b = F.generate(model, tok, '朝辞', { maxNew: N, temperature: 0, useCache: false, seed: 1 });
    const same = a.ids.length === N && a.ids.every((v, i) => v === b.ids[i]);
    ok = ok && same;
    console.log('  ' + (same ? '[OK]' : '[FAIL]') + ' ' + preset.padEnd(6) + ' · ' + N + ' token（blockSize=16' + (N > 16 ? '，含滑动窗口' : '') + '）');
  }
}

console.log('\n=== 6. 端到端：BPE + LLaMA 架构 ===');
const bigBpe = new F.BPETokenizer(CORPUS, 900, { byteLevel: false });
const idsB = bigBpe.encode(CORPUS);
console.log('  BPE（字符级基元）词表 ' + bigBpe.vocabSize + ' · 语料 ' + idsB.length + ' token（字符级基线 ' + charTok.encode(CORPUS).length + '）');
for (const preset of ['gpt2', 'llama']) {
  const model = new F.GPTModel(F.applyPreset(
    { vocabSize: bigBpe.vocabSize, blockSize: 64, nLayer: 2, nHead: 4, dModel: 64, seed: 42 }, preset));
  const r = F.train(model, idsB, { steps: 200, lr: 3e-3, batch: 32, warmup: 10, seed: 0, logEvery: 1e9 });
  const gen = F.generate(model, bigBpe, '朝辞白帝', { maxNew: 40, temperature: 0.8, topK: 40, seed: 7 });
  console.log('  ' + preset.padEnd(6) + ' 参数 ' + String(model.paramCount()).padStart(7) +
    ' · loss ' + r.history[0].toFixed(2) + ' -> ' + r.finalLoss.toFixed(3) +
    ' · 生成 ' + JSON.stringify(gen.text.slice(0, 34)));
}
console.log('\n=== 7. 参数对齐的架构对照（同语料 / 同种子 / 同步数） ===');
// 用二分法把 llama 的 ffnHidden 调到与 gpt2 参数量一致，消除"参数更多所以更好"的混淆
const VSZ = bigBpe.vocabSize;
const baseCfg = { vocabSize: VSZ, blockSize: 64, nLayer: 2, nHead: 4, dModel: 64, seed: 42 };
const gpt2 = new F.GPTModel(baseCfg);
const target = gpt2.paramCount();
let lo = 8, hi = 4 * 64;
while (lo < hi) {
  const mid = (lo + hi + 1) >> 1;
  const pc = new F.GPTModel(F.applyPreset({ ...baseCfg, ffnHidden: mid }, 'llama')).paramCount();
  if (pc <= target) lo = mid; else hi = mid - 1;
}
const llamaEq = new F.GPTModel(F.applyPreset({ ...baseCfg, ffnHidden: lo }, 'llama'));
console.log('  gpt2       参数 ' + target.toLocaleString());
console.log('  llama(4D)  参数 ' + new F.GPTModel(F.applyPreset(baseCfg, 'llama')).paramCount().toLocaleString());
console.log('  llama(对齐) 参数 ' + llamaEq.paramCount().toLocaleString() + '（ffnHidden=' + lo + '）');
for (const [m, label] of [[gpt2, 'gpt2       '], [new F.GPTModel(F.applyPreset(baseCfg, 'llama')), 'llama(4D)  '], [llamaEq, 'llama(对齐)']]) {
  const r = F.train(m, idsB, { steps: 300, lr: 3e-3, batch: 32, warmup: 15, seed: 0, logEvery: 1e9 });
  const ev = m.evaluate(idsB.slice(0, 64));
  console.log('  ' + label + ' loss ' + r.history[0].toFixed(2) + ' -> ' + r.finalLoss.toFixed(3) + ' · 准确率 ' + (ev.acc * 100).toFixed(1) + '%');
}

console.log('\n' + (ok ? '全部通过' : '存在失败'));
process.exit(ok ? 0 : 1);

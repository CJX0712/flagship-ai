'use strict';
// v5.0 量化验证：INT8 weight-only PTQ 的数值保真度 / 生成可用性 / 存档往返 / 体积压缩
const F = require('../src/index.js');
const { CORPUS } = require('../src/corpus.js');

let pass = 0, fail = 0;
function ck(name, cond, extra) {
  if (cond) { pass++; console.log(`  [OK]   ${name}${extra ? ' — ' + extra : ''}`); }
  else { fail++; console.log(`  [FAIL] ${name}${extra ? ' — ' + extra : ''}`); }
}
function maxAbsDiff(a, b) {
  let m = 0; for (let i = 0; i < a.length; i++) { const d = Math.abs(a[i] - b[i]); if (d > m) m = d; }
  return m;
}

// 1) 构建并训练一个小 LLaMA(MHA) 模型，让权重有真实分布（非随机对称）
const tok = new F.Tokenizer(CORPUS);
const m = new F.GPTModel(F.applyPreset(
  { vocabSize: tok.vocabSize, blockSize: 48, nLayer: 2, nHead: 4, dModel: 32 }, 'llama'));
F.train(m, tok.encode(CORPUS), { steps: 120, lr: 3e-3, batch: 24, warmup: 8, seed: 0 });

// 2) F32 前向 logits vs 量化前向 logits
const ids = tok.encode(CORPUS.slice(0, 40));
const lF = m.forward(ids).logits;
const mq = m.quantize();
const lQ = mq.forward(ids).logits;
const diff = maxAbsDiff(lF, lQ);
// 量化误差纯为舍入误差（dequant = round(w/scale)·scale）。理论界 ≈ D·0.5·scale·max|A|，
// 对 d=32 量级 < 0.1；实测远小于此，证明对解码无损。
ck('量化 logits 数值保真（max|Δ| < 0.1）', diff < 0.1, `maxΔ=${diff.toExponential(3)}`);

// 3) 量化模型前向输出有限（无 NaN/Inf）
ck('量化前向数值有效', lQ.every(Number.isFinite));

// 4) KV cache 增量路径（forwardStep）与全序列 forward 同一末位逐位一致（量化后）
const V = m.cfg.vocabSize;
const lastRowF = lQ.slice((ids.length - 1) * V);          // 全序列 forward 末位 logits
const kv = F.initKVCache(mq);
F.warmupKV(mq, ids.slice(0, ids.length - 1), kv);         // 预热除末位外的所有 token
const lastRowStep = F.forwardStep(mq, ids[ids.length - 1], ids.length - 1, kv).logits;
ck('量化 KV-cache 路径与全序列 forward 末位逐位一致', maxAbsDiff(lastRowF, lastRowStep) < 1e-12,
  `Δ=${maxAbsDiff(lastRowF, lastRowStep).toExponential(2)}`);

// 5) 生成可用（量化模型走 generate，含 KV cache 与 fallback 两路）
const g1 = F.generate(mq, tok, '旗舰', { maxTokens: 40, temperature: 0.9, topK: 20, seed: 5 });
ck('量化 generate 产出有限文本', typeof g1.text === 'string' && g1.text.length > 0 && g1.ids.every(Number.isFinite));

// 6) 存档往返：量化模型 save -> load -> 推理一致
const json = F.saveModel(mq, tok);
const ml = F.loadModel(json);
ck('量化存档标记 quantized=true', ml.quantized === true);
ck('量化存档后线性权重常驻 Int8（无 .d）',
  ml.params['l0.wqkv'].q instanceof Int8Array && ml.params['l0.wqkv'].d === undefined);
const lL = ml.forward(ids).logits;
ck('量化存档往返 logits 逐位一致', maxAbsDiff(lQ, lL) === 0, `Δ=${maxAbsDiff(lQ, lL)}`);

// 7) 体积压缩：F32 存档 vs 量化存档（JSON 字符串长度）
const jsonF = F.saveModel(m, tok);
const ratio = jsonF.length / json.length;
ck('量化存档体积压缩 > 2.5x', ratio > 2.5, `F32=${(jsonF.length/1024).toFixed(1)}KB / INT8=${(json.length/1024).toFixed(1)}KB (${ratio.toFixed(2)}x)`);

// 8) in-RAM 权重常驻 Int8：线性权重由 Float64(8B) 降为 Int8(1B) → 恰好 8x；
//    整体（含偏置/归一化，保持 Float64）仍有显著压缩。
let linF32 = 0, linQ = 0, totalF32 = 0, quantBytes = 0;
for (const k in m.params) {
  const f32 = m.params[k].d.length * 8;
  totalF32 += f32;
  if (mq.params[k].q) { linF32 += f32; linQ += mq.params[k].q.length; quantBytes += mq.params[k].q.length; }
  else { quantBytes += f32; }
}
ck('线性权重量化 in-RAM 恰好 8x', linQ * 8 === linF32, `${linF32}B(F32) -> ${linQ}B(INT8)`);
const overall = totalF32 / quantBytes;
ck('模型整体 in-RAM 压缩 > 4x', overall > 4, `整体 ${overall.toFixed(2)}x`);

// 9) 原训练模型不受影响（quantize 为只读导出）
ck('quantize 不修改原模型', m.params['l0.wqkv'].q === undefined && m.params['l0.wqkv'].d instanceof Float64Array);

console.log(`\n────────────────────────────────\nv5 量化探针： ${pass} 通过 / ${fail} 失败 ${fail === 0 ? '✅' : '❌'}`);
process.exit(fail === 0 ? 0 : 1);

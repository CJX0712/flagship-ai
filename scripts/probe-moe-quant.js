'use strict';
/* ============================================================================
 * v8.0 MoE × INT8 组合验证：稀疏混合专家的权重量化推理
 * ----------------------------------------------------------------------------
 * 把 v5 的 INT8 weight-only PTQ 用到 v7 的 MoE 模型上，确认组合无损：
 *   1. 路由 wg + 专家 e*.w1/w2 转 Int8（偏置/计数保持 Float64）
 *   2. F32 vs INT8 logits 数值保真（max|Δ| < 0.1）
 *   3. 量化前向数值有效（无 NaN/Inf）
 *   4. 量化 KV-cache 路径与全序列前向末位逐位一致（< blockSize）
 *   5. 量化 generate 产出有限文本
 *   6. 存档 save/load 往返 logits 逐位一致
 *   7. in-RAM 线性权重恰好 8x、整体 > 4x；JSON 存档 > 2.5x
 *   8. 量化后路由不坍缩（无单一专家 > 90% 占用）
 *   9. quantize 只读导出，原 MoE 模型不受影响
 * 作者：晨星  ·  License: MIT
 * ======================================================================*/
const F = require('../src/flagship-engine.js');
const { CORPUS } = require('../src/corpus.js');

let pass = 0, fail = 0;
function ck(name, cond, extra) {
  if (cond) { pass++; console.log(`  [OK]   ${name}${extra ? ' — ' + extra : ''}`); }
  else { fail++; console.log(`  [FAIL] ${name}${extra ? ' — ' + extra : ''}`); }
}
function maxAbsDiff(a, b) { let m = 0; for (let i = 0; i < a.length; i++) { const d = Math.abs(a[i] - b[i]); if (d > m) m = d; } return m; }

// 1) 训练一个小 Mixtral(MoE) 模型，让权重有真实分布
const tok = new F.Tokenizer(CORPUS);
const m = new F.GPTModel(F.applyPreset(
  { vocabSize: tok.vocabSize, blockSize: 48, nLayer: 2, nHead: 4, dModel: 32, nExp: 4, topK: 2, auxWeight: 0.01, seed: 1 }, 'mixtral'));
F.train(m, tok.encode(CORPUS), { steps: 150, lr: 3e-3, batch: 24, warmup: 8, seed: 0 });

// 2) F32 前向 vs 量化前向
const ids = tok.encode(CORPUS.slice(0, 40)); // 40 < blockSize 48，缓存/全序列可逐位比
const lF = m.forward(ids).logits;
const mq = m.quantize();
const lQ = mq.forward(ids).logits;
const diff = maxAbsDiff(lF, lQ);
ck('MoE 量化 logits 数值保真（max|Δ| < 0.1）', diff < 0.1, `maxΔ=${diff.toExponential(3)}`);
ck('MoE 量化前向数值有效', lQ.every(Number.isFinite));

// 3) 量化结构：路由/专家线性权重 Int8，偏置保持 Float64
const wgQ = mq.params['l0.wg'].q instanceof Int8Array && mq.params['l0.wg'].d === undefined;
const e1Q = mq.params['l0.e1.w1'].q instanceof Int8Array && mq.params['l0.e1.w1'].d === undefined;
const e2Q = mq.params['l1.e2.w2'].q instanceof Int8Array && mq.params['l1.e2.w2'].d === undefined;
const bgF = mq.params['l0.bg'].d instanceof Float64Array && mq.params['l0.bg'].q === undefined;
ck('路由 wg + 专家 e*.w1/w2 常驻 Int8（无 .d）', wgQ && e1Q && e2Q, `wg/w1/w2 全 Int8`);
ck('路由偏置 bg 保持 Float64', bgF);

// 4) KV-cache 末位与全序列 forward 逐位一致（量化 MoE，< blockSize）
const V = mq.cfg.vocabSize;
const lastRowF = lQ.slice((ids.length - 1) * V);
const kv = F.initKVCache(mq);
F.warmupKV(mq, ids.slice(0, ids.length - 1), kv);
const lastRowStep = F.forwardStep(mq, ids[ids.length - 1], ids.length - 1, kv).logits;
const kd = maxAbsDiff(lastRowF, lastRowStep);
ck('量化 MoE KV-cache 末位逐位一致', kd < 1e-9, `Δ=${kd.toExponential(2)}`);

// 5) 生成可用
const g1 = F.generate(mq, tok, '旗舰', { maxTokens: 40, temperature: 0.9, topK: 20, seed: 5 });
ck('量化 MoE generate 产出有限文本', typeof g1.text === 'string' && g1.text.length > 0 && g1.ids.every(Number.isFinite));

// 6) 存档往返逐位一致
const ml = F.loadModel(F.saveModel(mq, tok));
ck('量化 MoE 存档标记 quantized=true + useMoE 保留', ml.quantized === true && ml.useMoE === true);
ck('量化 MoE 存档后专家权重常驻 Int8', ml.params['l0.e0.w1'].q instanceof Int8Array && ml.params['l0.e0.w1'].d === undefined);
const lL = ml.forward(ids).logits;
ck('量化 MoE 存档往返 logits 逐位一致', maxAbsDiff(lQ, lL) === 0, `Δ=${maxAbsDiff(lQ, lL)}`);

// 7) in-RAM 压缩：线性权重（含路由+专家）8x，整体 > 4x
let linF32 = 0, linQ = 0, totalF32 = 0, qb = 0;
for (const k in m.params) {
  const f32 = m.params[k].d.length * 8; totalF32 += f32;
  if (mq.params[k].q) { linF32 += f32; linQ += mq.params[k].q.length; qb += mq.params[k].q.length; }
  else qb += f32;
}
ck('MoE 线性权重量化 in-RAM 恰好 8x（含路由+专家）', linQ * 8 === linF32, `${linF32}B → ${linQ}B`);
const overall = totalF32 / qb;
ck('MoE 整体 in-RAM 压缩 > 4x', overall > 4, `整体 ${overall.toFixed(2)}x`);
const jsonF = F.saveModel(m, tok), jsonQ = F.saveModel(mq, tok);
const ratio = jsonF.length / jsonQ.length;
ck('MoE 量化存档体积压缩 > 2.5x', ratio > 2.5, `F32=${(jsonF.length/1024).toFixed(1)}KB / INT8=${(jsonQ.length/1024).toFixed(1)}KB (${ratio.toFixed(2)}x)`);

// 8) 量化后路由不坍缩：固定评估批次上无单一专家占用 > 90%
(function () {
  const ev = ids.slice(0, 16);
  const { cache } = mq.forward(ev);
  const f = new Array(m.nExp).fill(0);
  for (const c of cache.layers)
    for (let i = 0; i < ev.length; i++)
      for (let j = 0; j < m.topK; j++) f[c.moe.idx[i * m.topK + j]]++;
  const total = ev.length * m.topK;
  const mx = Math.max(...f) / total;
  ck('量化 MoE 路由不坍缩（无单一专家 > 90%）', mx < 0.9, `最大专家占用 ${(mx * 100).toFixed(0)}%；分布 ${f.map(x => (x / total).toFixed(2)).join('/')}`);
})();

// 9) 原训练模型不受影响（quantize 只读）
ck('quantize 不修改原 MoE 模型', m.params['l0.wg'].q === undefined && m.params['l0.wg'].d instanceof Float64Array && m.useMoE === true);

console.log(`\n────────────────────────────────\nv8 MoE×INT8 探针： ${pass} 通过 / ${fail} 失败 ${fail === 0 ? '✅' : '❌'}`);
process.exit(fail === 0 ? 0 : 1);

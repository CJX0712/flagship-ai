'use strict';
/* ============================================================================
 * v7.0 MoE（混合专家）验证：稀疏 FFN + top-k 路由 + 负载均衡辅助损失
 * ----------------------------------------------------------------------------
 * 不变量自检（exit 0 全过）：
 *   1. 中心差分梯度检验（CE 路径）—— 路由(wg/bg) + 专家(w1/w2) 解析梯度精确
 *   2. 中心差分梯度检验（辅助损失路径）—— 冻结 f_e 的 aux 梯度精确
 *   3. 未选中专家梯度恒为零（离散路由正确性）
 *   4. 小语料训练 loss 下降且无 NaN
 *   5. 负载均衡：auxWeight>0 显著抑制专家坍缩（aux << 无 aux）
 *   6. MoE 模型 generate 产出有限文本
 *   7. 存档 save/load 无损往返 + 专家参数完整恢复
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
function captureSel(cache, T, K) {
  const sel = [];
  for (const c of cache.layers)
    for (let i = 0; i < T; i++) { const a = c.moe.idx[i * K], b = c.moe.idx[i * K + 1]; sel.push(Math.min(a, b) * 100 + Math.max(a, b)); }
  return sel.join(',');
}
// 通用中心差分：在确定性扰动下检验解析梯度（带路由选择稳定性守卫）
function fdCheck(model, toks, T, param, grad, indices, lossFn, baseSel) {
  const e = 1e-5, atol = 1e-7, rtol = 1e-4;
  let worst = 0, checked = 0, skipped = 0, info = '';
  for (const idx of indices) {
    const orig = param[idx];
    param[idx] = orig + e; const lp = lossFn(); const sp = lastSel;
    param[idx] = orig - e; const lm = lossFn(); const sm = lastSel;
    param[idx] = orig;
    if (baseSel !== null && (sp !== baseSel || sm !== baseSel)) { skipped++; continue; } // 选择翻转 → 跳过
    const num = (lp - lm) / (2 * e), ana = grad[idx];
    if (Math.abs(ana) < 1e-12 && Math.abs(num) < 1e-12) { skipped++; continue; }
    const err = Math.abs(ana - num) / (atol + rtol * Math.max(Math.abs(ana), Math.abs(num)));
    if (err > worst) { worst = err; info = `idx=${idx} 解析=${ana.toExponential(3)} 数值=${num.toExponential(3)}`; }
    checked++;
  }
  return { worst, checked, skipped, info };
}
let lastSel = null;

const tok = new F.Tokenizer(CORPUS);
const V = tok.vocabSize;
const toks = tok.encode(CORPUS).slice(0, 6);
const T = toks.length, targets = toks.slice(1);

/* ---------- 1) CE 路径梯度检验（路由 + 专家，选择稳定） ---------- */
(function () {
  const model = new F.GPTModel({ vocabSize: V, blockSize: 8, nLayer: 1, nHead: 2, dModel: 16, norm: 'rmsnorm', ffn: 'swiglu', nExp: 4, topK: 2, seed: 7, auxWeight: 0 });
  // 强制清晰路由间隙：专家 0/1 恒胜 top-2，保证小扰动下选择不翻转
  model.params['l0.bg'].d.set([6, 3, 0, -3]);
  model.params['l0.wg'].d.fill(0);
  model.auxWeight = 0;
  const ceLoss = () => {
    const { logits, cache } = model.forward(toks);
    lastSel = captureSel(cache, T, 2);
    return F.crossEntropy(logits, targets, T - 1, V, T).loss;
  };
  model.zeroGrads();
  const { cache: c0 } = model.forward(toks); lastSel = captureSel(c0, T, 2); const baseSel = lastSel;
  const { logits: lg, cache: cb } = model.forward(toks);
  model.backward(cb, F.crossEntropy(lg, targets, T - 1, V, T).dlogits);
  // 抽样：路由 wg（前 8 列）、专家0 w1/w2、专家1 w1
  const wg = model.params['l0.wg'].d, gwg = model.grads['l0.wg'];
  const w1 = model.params['l0.e0.w1'].d, gw1 = model.grads['l0.e0.w1'];
  const w2 = model.params['l0.e0.w2'].d, gw2 = model.grads['l0.e0.w2'];
  const w1b = model.params['l0.e1.w1'].d, gw1b = model.grads['l0.e1.w1'];
  const idxWg = [], idxW1 = [], idxW2 = [], idxW1b = [];
  for (let i = 0; i < wg.length; i += 8) idxWg.push(i);
  for (let i = 0; i < w1.length; i += 6) idxW1.push(i);
  for (let i = 0; i < w2.length; i += 6) idxW2.push(i);
  for (let i = 0; i < w1b.length; i += 12) idxW1b.push(i);
  const r1 = fdCheck(model, toks, T, wg, gwg, idxWg, ceLoss, baseSel);
  const r2 = fdCheck(model, toks, T, w1, gw1, idxW1, ceLoss, baseSel);
  const r3 = fdCheck(model, toks, T, w2, gw2, idxW2, ceLoss, baseSel);
  const r4 = fdCheck(model, toks, T, w1b, gw1b, idxW1b, ceLoss, baseSel);
  const worst = Math.max(r1.worst, r2.worst, r3.worst, r4.worst);
  const checked = r1.checked + r2.checked + r3.checked + r4.checked;
  const skipped = r1.skipped + r2.skipped + r3.skipped + r4.skipped;
  ck('MoE CE 路径梯度检验（路由+专家）', worst <= 1, `err/tol=${worst.toExponential(2)}（${checked}分量/${skipped}跳过选择翻转）；最差 ${r1.info || r2.info}`);
})();

/* ---------- 2) 辅助损失路径梯度检验（冻结 f_e） ---------- */
(function () {
  const model = new F.GPTModel({ vocabSize: V, blockSize: 8, nLayer: 1, nHead: 2, dModel: 16, norm: 'rmsnorm', ffn: 'swiglu', nExp: 4, topK: 2, seed: 7, auxWeight: 0.01 });
  model.params['l0.bg'].d.set([6, 3, 0, -3]);
  model.params['l0.wg'].d.fill(0);
  model.auxWeight = 0.01;
  const auxLoss = () => {
    const { cache } = model.forward(toks);
    lastSel = captureSel(cache, T, 2);
    let a = 0; for (const c of cache.layers) a += c.moe.aux;
    return model.auxWeight * a;
  };
  model.zeroGrads();
  const { cache: c0 } = model.forward(toks); lastSel = captureSel(c0, T, 2); const baseSel = lastSel;
  const { cache: cb } = model.forward(toks);
  const dz = new Float64Array(T * V); // dlogits=0 ⇒ 仅 aux 梯度
  model.backward(cb, dz);
  const wg = model.params['l0.wg'].d, gwg = model.grads['l0.wg'];
  const bg = model.params['l0.bg'].d, gbg = model.grads['l0.bg'];
  const idxWg = [], idxBg = [];
  for (let i = 0; i < wg.length; i += 4) idxWg.push(i);
  for (let i = 0; i < bg.length; i += 1) idxBg.push(i);
  const r1 = fdCheck(model, toks, T, wg, gwg, idxWg, auxLoss, baseSel);
  const r2 = fdCheck(model, toks, T, bg, gbg, idxBg, auxLoss, baseSel);
  const worst = Math.max(r1.worst, r2.worst);
  const checked = r1.checked + r2.checked;
  ck('MoE 辅助损失梯度检验（冻结 f_e）', worst <= 1, `err/tol=${worst.toExponential(2)}（${checked}分量，aux 梯度精确）`);
})();

/* ---------- 3) 未选中专家梯度恒为零 ---------- */
(function () {
  const model = new F.GPTModel({ vocabSize: V, blockSize: 8, nLayer: 1, nHead: 2, dModel: 16, norm: 'rmsnorm', ffn: 'swiglu', nExp: 4, topK: 2, seed: 7, auxWeight: 0.01 });
  model.params['l0.bg'].d.set([6, 3, 0, -3]); // 仅专家 0/1 被选中
  model.params['l0.wg'].d.fill(0);
  const { logits, cache } = model.forward(toks);
  const dz = new Float64Array(T * V);
  model.zeroGrads();
  model.backward(cache, F.crossEntropy(logits, targets, T - 1, V, T).dlogits);
  let maxGrad = 0;
  for (let e = 0; e < 4; e++) {
    const g = model.grads[`l0.e${e}.w1`];
    for (let i = 0; i < g.length; i++) maxGrad = Math.max(maxGrad, Math.abs(g[i]));
  }
  // 专家 2/3 从未被路由 ⇒ 其梯度应恰为 0（离散选择正确性）
  const g2 = model.grads['l0.e2.w1'], g3 = model.grads['l0.e3.w1'];
  const zero23 = g2.every(v => v === 0) && g3.every(v => v === 0);
  ck('未选中专家(e2/e3)梯度恒为零', zero23 && maxGrad >= 0, `最大梯度 ${maxGrad.toExponential(2)}；e2/e3 全零=${zero23}`);
})();

/* ---------- 4) 训练 loss 下降且无 NaN ---------- */
(function () {
  const corpus = CORPUS.repeat(6);
  const t = new F.Tokenizer(corpus);
  const ids = t.encode(corpus);
  const model = new F.GPTModel(F.applyPreset(
    { vocabSize: t.vocabSize, blockSize: 32, nLayer: 2, nHead: 2, dModel: 32, nExp: 4, topK: 2, auxWeight: 0.01, seed: 11 }, 'mixtral'));
  const { history } = F.train(model, ids, { steps: 200, lr: 3e-3, batch: 24, warmup: 15, seed: 0, logEvery: 1e9 });
  const first = history.slice(0, 20).reduce((a, b) => a + b, 0) / 20;
  const last = history.slice(-20).reduce((a, b) => a + b, 0) / 20;
  let nan = false;
  for (const k in model.params) for (let i = 0; i < model.params[k].d.length; i++) if (!Number.isFinite(model.params[k].d[i])) nan = true;
  ck('MoE 训练 loss 下降', last < first && !nan, `前20均值 ${first.toFixed(3)} → 后20均值 ${last.toFixed(3)}；NaN=${nan}`);
})();

/* ---------- 5) 负载均衡：auxWeight>0 显著抑制专家坍缩 ---------- */
(function () {
  const mk = (aw, bias) => {
    const corpus = CORPUS.repeat(8);
    const t = new F.Tokenizer(corpus);
    const ids = t.encode(corpus);
    // applyPreset 会覆盖 topK/auxWeight，故在 preset 之后显式重设
    const base = F.applyPreset(
      { vocabSize: t.vocabSize, blockSize: 32, nLayer: 2, nHead: 2, dModel: 32, nExp:4, auxWeight: aw, seed: 5 }, 'mixtral');
    const model = new F.GPTModel(Object.assign({}, base, { topK: 1, auxWeight: aw }));
    // 初始化路由偏置使专家 0 占优 → 无辅助损失时坍缩到单一专家（aux→E=4）
    for (let l = 0; l < model.cfg.nLayer; l++) model.params[`l${l}.bg`].d.set([bias, 0, 0, 0]);
    F.train(model, ids, { steps: 400, lr: 3e-3, batch: 24, warmup: 15, seed: 0, logEvery: 1e9 });
    const ev = ids.slice(0, 16);
    const { cache } = model.forward(ev);
    let a = 0; const f = new Array(4).fill(0);
    for (const c of cache.layers) { a += c.moe.aux; for (let i = 0; i < ev.length; i++) f[c.moe.idx[i * 1]]++; }
    for (let e = 0; e < 4; e++) f[e] /= ev.length;
    return { aux: a / model.cfg.nLayer, f };
  };
  const no = mk(0, 4), yes = mk(1.0, 4);
  const f = yes.f.map(x => x.toFixed(2)).join('/');
  ck('负载均衡：auxWeight>0 抑制专家坍缩', yes.aux < no.aux - 0.5 && yes.aux < 1.5,
    `aux(无辅助)=${no.aux.toFixed(3)}（坍缩） → aux(aux=1.0)=${yes.aux.toFixed(3)}（均衡）；专家用量 f=${f}（坍缩上界=4，均衡下界=1）`);
})();

/* ---------- 6) generate 产出有限文本 ---------- */
(function () {
  const t = new F.Tokenizer(CORPUS.repeat(4));
  const ids = t.encode(CORPUS.repeat(4));
  const model = new F.GPTModel(F.applyPreset(
    { vocabSize: t.vocabSize, blockSize: 32, nLayer: 2, nHead: 2, dModel: 32, nExp: 4, topK: 2, auxWeight: 0.01, seed: 3 }, 'mixtral'));
  F.train(model, ids, { steps: 120, lr: 3e-3, batch: 24, warmup: 12, seed: 0, logEvery: 1e9 });
  const g = F.generate(model, t, CORPUS.slice(0, 4), { maxTokens: 30, temperature: 0.9, topK: 10, seed: 9 });
  const ok = typeof g.text === 'string' && g.text.length > 0 && g.ids.every(Number.isFinite) && g.ids.every(v => v >= 0 && v < model.cfg.vocabSize);
  ck('MoE generate 产出有限文本', ok, `长度=${g.text.length}`);
})();

/* ---------- 7) 存档 save/load 无损往返 ---------- */
(function () {
  const t = new F.Tokenizer(CORPUS);
  const ids = t.encode(CORPUS).slice(0, 20);
  const model = new F.GPTModel(F.applyPreset(
    { vocabSize: t.vocabSize, blockSize: 24, nLayer: 1, nHead: 2, dModel: 16, nExp: 4, topK: 2, auxWeight: 0.01, seed: 4 }, 'mixtral'));
  F.train(model, ids, { steps: 15, lr: 3e-3, warmup: 5, seed: 0, logEvery: 1e9 });
  const json = F.saveModel(model, t);
  const restored = F.loadModel(json);
  const denseCount = new F.GPTModel({ vocabSize: V, blockSize: 8, nLayer: 1, nHead: 2, dModel: 16 }).paramCount();
  const moeCount = model.paramCount();
  let maxDiff = 0;
  for (const key in model.params) {
    const p1 = model.params[key].d, p2 = restored.params[key].d;
    if (!p2) { maxDiff = Infinity; break; }
    if (p1.length !== p2.length) { maxDiff = Infinity; break; }
    for (let i = 0; i < p1.length; i++) maxDiff = Math.max(maxDiff, Math.abs(p1[i] - p2[i]));
  }
  const l1 = model.forward(ids).logits, l2 = restored.forward(ids).logits;
  let logitDiff = 0; for (let i = 0; i < l1.length; i++) logitDiff = Math.max(logitDiff, Math.abs(l1[i] - l2[i]));
  const expertOk = !!restored.params['l0.e0.w1'] && !!restored.params['l0.e3.w2'];
  ck('MoE 存档无损往返', maxDiff === 0 && logitDiff === 0 && expertOk && moeCount > denseCount,
    `参数Δ=${maxDiff}；logitsΔ=${logitDiff}；专家参数恢复=${expertOk}；MoE参数量 ${moeCount} > 稠密 ${denseCount}`);
})();

console.log(`\n────────────────────────────────\nv7 MoE 探针： ${pass} 通过 / ${fail} 失败 ${fail === 0 ? '✅' : '❌'}`);
process.exit(fail === 0 ? 0 : 1);

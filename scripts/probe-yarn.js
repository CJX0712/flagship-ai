/* ============================================================================
 * 旗舰 AI · v14 YaRN 长度外推探针（probe-yarn.js）
 * ----------------------------------------------------------------------------
 * 硬不变量验证（Node 运行：node scripts/probe-yarn.js；全绿 exit 0）：
 *   1. 零回归：ropeScale.factor=1 与缺省配置逐位等价（Δ=0）
 *   2. 零回归：forward 与 forwardStep（增量推理）在 YaRN 下逐位一致（Δ=0）
 *   3. 中心差分梯度检验（YaRN factor=4）：反向传播与数值梯度一致
 *   4. 中心差分梯度检验（NTK 简单型 factor=4）：反向传播一致
 *   5. mscale 公式：attentionScaling = 0.1·ln(factor)/ln(2)
 *   6. freqScale 范围与方向：[1/factor,1]，高频维(小 p) 插值更多 → 非递减
 *   7. 长上下文健壮性：pos 远超 blockSize（外推）输出有限且与重算一致
 *   8. 确定性：同种子同配置 forward 复现
 *   9. 特征生效：factor=4 与 baseline 输出确有差异（非 no-op）
 * 作者：晨星 · License: MIT
 * ========================================================================*/
const F = require('./../src/legacy/flagship-engine.v11.js');

function maxDelta(a, b) {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}
function allFinite(arr) { for (let i = 0; i < arr.length; i++) if (!Number.isFinite(arr[i])) return false; return true; }

/* 1+2. 零回归 */
function zeroRegression() {
  const cfg = { vocabSize: 17, blockSize: 12, nLayer: 2, nHead: 2, dModel: 16, norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope', seed: 3 };
  const base = new F.GPTModel(cfg);
  const yarn1 = new F.GPTModel(Object.assign({}, cfg, { ropeScale: { type: 'yarn', factor: 1 } }));
  const rng = F.mulberry32(11), ids = [];
  for (let i = 0; i < 10; i++) ids.push(Math.floor(rng() * 17));
  const lb = base.forward(ids).logits, ly = yarn1.forward(ids).logits;
  // forward vs forwardStep（增量）：forwardStep 仅出末位 logits，需取 forward 末位对齐
  const kv = F.initKVCache(yarn1);
  let lf = null;
  for (let i = 0; i < ids.length; i++) lf = F.forwardStep(yarn1, ids[i], i, kv).logits;
  const last = ly.slice((ids.length - 1) * 17);
  const d1 = maxDelta(lb, ly), d2 = maxDelta(last, lf);
  return {
    pass: d1 === 0 && d2 === 0,
    detail: `factor=1 vs 缺省 Δ=${d1}（要求0）· forwardStep vs forward Δ=${d2}（要求0）`,
  };
}

/* 通用中心差分梯度检验 */
function gradCheck(model, tokens, targets) {
  const { logits, cache } = model.forward(tokens);
  const { dlogits } = F.crossEntropy(logits, targets, tokens.length - 1, model.cfg.vocabSize, tokens.length);
  model.backward(cache, dlogits);
  const lossOf = () => {
    const lg = model.forward(tokens).logits;
    return F.crossEntropy(lg, targets, tokens.length - 1, model.cfg.vocabSize).loss;
  };
  const all = [];
  for (const key in model.params) { const p = model.params[key].d; for (let i = 0; i < p.length; i++) all.push([key, i]); }
  const srng = F.mulberry32(99); let cand = all;
  if (all.length > 300) { cand = all.slice(); for (let i = cand.length - 1; i > 0; i--) { const j = Math.floor(srng() * (i + 1)); const t = cand[i]; cand[i] = cand[j]; cand[j] = t; } cand = cand.slice(0, 300); }
  const e = 1e-5; let maxRatio = 0, checked = 0, skipped = 0;
  for (const [key, i] of cand) {
    const p = model.params[key].d, orig = p[i];
    p[i] = orig + e; const lp = lossOf(); p[i] = orig - e; const lm = lossOf(); p[i] = orig;
    const num = (lp - lm) / (2 * e), ana = model.grads[key][i];
    if (Math.abs(ana) < 1e-12 && Math.abs(num) < 1e-12) { skipped++; continue; }
    const err = Math.abs(ana - num), tol = 1e-7 + 1e-4 * Math.max(Math.abs(ana), Math.abs(num));
    maxRatio = Math.max(maxRatio, err / tol); checked++;
  }
  return { maxRatio, checked, skipped };
}

/* 3+4. 梯度检验（yarn / ntk） */
function gradChecks() {
  const V = 13, T = 5;
  const base = { vocabSize: V, blockSize: 6, nLayer: 1, nHead: 2, dModel: 8, norm: 'rmsnorm', pos: 'rope', seed: 7 };
  const yarn = new F.GPTModel(Object.assign({}, base, { ropeScale: { type: 'yarn', factor: 4, originalMaxLength: 6 } }));
  const ntk = new F.GPTModel(Object.assign({}, base, { ropeScale: { type: 'ntk', factor: 4 } }));
  const rng = F.mulberry32(5), tokens = [];
  for (let i = 0; i < T; i++) tokens.push(Math.floor(rng() * V));
  const targets = []; for (let i = 1; i < T; i++) targets.push(tokens[i]);
  const gy = gradCheck(yarn, tokens, targets);
  const gn = gradCheck(ntk, tokens, targets);
  const pass = gy.checked > 100 && gy.maxRatio <= 1 && gn.checked > 100 && gn.maxRatio <= 1;
  return { pass, detail: `yarn maxErr/Tol=${gy.maxRatio.toExponential(2)}(检${gy.checked}) · ntk=${gn.maxRatio.toExponential(2)}(检${gn.checked})` };
}

/* 5+6. mscale & freqScale */
function scalingMath() {
  const factor = 4;
  const m = new F.GPTModel({ vocabSize: 11, blockSize: 8, nLayer: 1, nHead: 2, dModel: 8, pos: 'rope', seed: 1, ropeScale: { type: 'yarn', factor, originalMaxLength: 8 } });
  const aExp = 0.1 * Math.log(factor) / Math.log(2);
  const aOK = Math.abs(m.attentionScaling - aExp) < 1e-12;
  const fs = m.ropeFreqScale;
  let rangeOK = true, mono = true;
  for (let p = 0; p < fs.length; p++) { if (fs[p] < 1 / factor - 1e-12 || fs[p] > 1 + 1e-12) rangeOK = false; if (p > 0 && fs[p] < fs[p - 1] - 1e-12) mono = false; }
  // NTK 型应全相等
  const ntk = new F.GPTModel({ vocabSize: 11, blockSize: 8, nLayer: 1, nHead: 2, dModel: 8, pos: 'rope', seed: 1, ropeScale: { type: 'ntk', factor } });
  let ntkUni = true; const nf = ntk.ropeFreqScale;
  for (let p = 1; p < nf.length; p++) if (Math.abs(nf[p] - nf[0]) > 1e-12) ntkUni = false;
  return { pass: aOK && rangeOK && mono && ntkUni, detail: `mscale=${m.attentionScaling}(期望${aExp.toFixed(4)}) · freqScale∈[1/${factor},1]=${rangeOK} 非递减=${mono} · ntk均匀=${ntkUni}` };
}

/* 7. 长上下文外推 */
function longContext() {
  const m = new F.GPTModel({ vocabSize: 13, blockSize: 16, nLayer: 1, nHead: 2, dModel: 8, pos: 'rope', seed: 2, ropeScale: { type: 'yarn', factor: 8, originalMaxLength: 16 } });
  const kv = F.initKVCache(m);
  const rng = F.mulberry32(21);
  for (let i = 0; i < 12; i++) F.forwardStep(m, Math.floor(rng() * 13), i, kv); // 预热（pos 0..11）
  const posFar = 2000; // 远超 blockSize=16，纯外推
  const lFar = F.forwardStep(m, 0, posFar, kv).logits;
  // 重算：独立缓存同 pos 应一致
  const kv2 = F.initKVCache(m);
  const rng2 = F.mulberry32(21);
  for (let i = 0; i < 12; i++) F.forwardStep(m, Math.floor(rng2() * 13), i, kv2);
  const lFar2 = F.forwardStep(m, 0, posFar, kv2).logits;
  return { pass: allFinite(lFar) && maxDelta(lFar, lFar2) === 0, detail: `pos=${posFar}(>blockSize=16) 输出有限=${allFinite(lFar)} · 重算Δ=${maxDelta(lFar, lFar2)}（要求0）` };
}

/* 8+9. 确定性 & 特征生效 */
function determinismAndActive() {
  const cfg = { vocabSize: 17, blockSize: 12, nLayer: 2, nHead: 2, dModel: 16, norm: 'rmsnorm', pos: 'rope', seed: 9, ropeScale: { type: 'yarn', factor: 4, originalMaxLength: 12 } };
  const a = new F.GPTModel(cfg), b = new F.GPTModel(cfg);
  const rng = F.mulberry32(33), ids = [];
  for (let i = 0; i < 9; i++) ids.push(Math.floor(rng() * 17));
  const la = a.forward(ids).logits, lb = b.forward(ids).logits;
  const base = new F.GPTModel(Object.assign({}, cfg, { ropeScale: undefined }));
  const lbase = base.forward(ids).logits;
  const dDet = maxDelta(la, lb), dAct = maxDelta(la, lbase);
  return { pass: dDet === 0 && dAct > 1e-9, detail: `同种子复现Δ=${dDet}（要求0）· yarn vs baseline Δ=${dAct.toExponential(2)}（>0 说明特征生效）` };
}

const CHECKS = [
  { name: '零回归：factor=1 ≡ 缺省；forward ≡ forwardStep', run: zeroRegression },
  { name: '中心差分梯度检验（yarn factor=4）', run: gradChecks },
  { name: 'mscale 公式 + freqScale 范围/方向 + NTK 均匀', run: scalingMath },
  { name: '长上下文外推：pos≫blockSize 有限且确定性', run: longContext },
  { name: '确定性 + 特征生效（yarn≠baseline）', run: determinismAndActive },
];

let passN = 0;
console.log('🧶 旗舰 AI · v14 YaRN 长度外推探针（Node ' + process.version + '）');
console.log('─'.repeat(60));
for (let i = 0; i < CHECKS.length; i++) {
  const r = (() => { try { return CHECKS[i].run() || {}; } catch (e) { return { pass: false, detail: '异常: ' + (e && e.message) }; } })();
  if (r.pass) passN++;
  console.log(` ${r.pass ? '✅' : '❌'} ${String(i + 1).padStart(2)}. ${CHECKS[i].name}`);
  console.log(`      ${r.detail}`);
}
console.log('─'.repeat(60));
console.log(`结果：${passN}/${CHECKS.length} 通过 ${passN === CHECKS.length ? '✅ 全绿' : '❌ 存在失败'}`);
process.exitCode = passN === CHECKS.length ? 0 : 1;

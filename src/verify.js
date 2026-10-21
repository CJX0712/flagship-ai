/* ============================================================================
 * 旗舰 AI 自检套件（verify.js）
 * ----------------------------------------------------------------------------
 * 10 项可独立交叉验证的不变量检查（浏览器 / Node 双端运行）：
 *   1. RNG 确定性与均匀性          6. 中心差分梯度检验（全参数抽样）
 *   2. Softmax 行和=1 且全正       7. 过拟合：小语料可背熟（loss -> ~0）
 *   3. LayerNorm 归一化不变量      8. 随机批次训练 loss 下降
 *   4. GELU 导数数值一致性         9. 生成确定性 + 贪心/top-k/超长 prompt 健壮性
 *   5. 因果掩码不变量              10. Tokenizer 编解码往返
 *
 * Node 运行：node src/verify.js   （全部通过 exit 0，任一失败 exit 1）
 * 作者：晨星  ·  License: MIT
 * ========================================================================*/
(function (root, factory) {
  if (typeof module !== 'undefined' && module.exports) {
    // T05：自检套件跑十层实现（src/index.js），旧单体仅作零回归基准。
    const api = factory(require('./index.js'));
    module.exports = api;
    if (require.main === module) {
      const results = api.runAll();
      const W = 46;
      console.log('🧪 旗舰 AI 自检套件（Node ' + process.version + '）');
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
    }
  } else {
    root.FlagshipVerify = factory(root.FlagshipAI);
  }
})(typeof self !== 'undefined' ? self : globalThis, function (F) {
  'use strict';

  const POEM = '朝辞白帝彩云间，千里江陵一日还。两岸猿声啼不住，轻舟已过万重山。';

  function safeRun(fn) {
    try {
      const r = fn() || {};
      return { pass: r.pass !== false, detail: r.detail || '' };
    } catch (e) {
      return { pass: false, detail: '异常: ' + (e && e.message) };
    }
  }

  const CHECKS = [
    {
      name: 'RNG 确定性与均匀性',
      run: () => {
        const a = F.mulberry32(42), b = F.mulberry32(42);
        const s1 = [a(), a(), a(), a(), a()];
        const s2 = [b(), b(), b(), b(), b()];
        const same = s1.every((v, i) => v === s2[i]);
        const inRange = s1.every(v => v >= 0 && v < 1);
        return { pass: same && inRange, detail: `同种子序列完全一致=${same}；取值范围[0,1)=${inRange}` };
      },
    },
    {
      name: 'Softmax 行和=1 且全正',
      run: () => {
        const rng = F.mulberry32(1);
        const r = 4, c = 7;
        const S = new Float64Array(r * c);
        for (let i = 0; i < S.length; i++) S[i] = F.randn(rng) * 5;
        const P = F.softmaxRows(S, r, c);
        let maxErr = 0, allPos = true;
        for (let i = 0; i < r; i++) {
          let s = 0;
          for (let j = 0; j < c; j++) { s += P[i * c + j]; if (P[i * c + j] <= 0) allPos = false; }
          maxErr = Math.max(maxErr, Math.abs(s - 1));
        }
        return { pass: maxErr < 1e-12 && allPos, detail: `行和最大偏差 ${maxErr.toExponential(2)}；全部为正=${allPos}` };
      },
    },
    {
      name: 'LayerNorm 归一化不变量',
      run: () => {
        const rng = F.mulberry32(2);
        const r = 5, c = 8;
        const X = new Float64Array(r * c);
        for (let i = 0; i < X.length; i++) X[i] = F.randn(rng) * 3 + 1;
        const g = new Float64Array(c).fill(1.7), b = new Float64Array(c).fill(-0.4);
        const { Y, xhat } = F.layernormFwd(X, g, b, r, c);
        let maxMu = 0, maxVar = 0, maxLin = 0;
        for (let i = 0; i < r; i++) {
          let mu = 0, v = 0;
          for (let j = 0; j < c; j++) mu += xhat[i * c + j];
          mu /= c;
          for (let j = 0; j < c; j++) { const d = xhat[i * c + j] - mu; v += d * d; }
          v /= c;
          maxMu = Math.max(maxMu, Math.abs(mu));
          maxVar = Math.max(maxVar, Math.abs(v - 1));
          for (let j = 0; j < c; j++) maxLin = Math.max(maxLin, Math.abs(Y[i * c + j] - (g[j] * xhat[i * c + j] + b[j])));
        }
        return { pass: maxMu < 1e-10 && maxVar < 1e-4 && maxLin < 1e-12, detail: `xhat 行均值最大 ${maxMu.toExponential(2)}；方差偏离1 ${maxVar.toExponential(2)}（含 eps=1e-5 理论衰减）；Y=g*xhat+b 偏差 ${maxLin.toExponential(2)}` };
      },
    },
    {
      name: 'GELU 导数数值一致性',
      run: () => {
        const xs = [-3, -1.5, -0.7, -0.1, 0, 0.3, 1.2, 2.5, 4];
        const ones = new Float64Array(xs.length).fill(1);
        const dAna = F.geluBwd(ones, Float64Array.from(xs));
        const e = 1e-6;
        let maxRel = 0;
        for (let i = 0; i < xs.length; i++) {
          const xp = xs[i] + e, xm = xs[i] - e;
          const num = (F.geluFwd(Float64Array.from([xp]))[0] - F.geluFwd(Float64Array.from([xm]))[0]) / (2 * e);
          const rel = Math.abs(dAna[i] - num) / Math.max(1e-8, Math.abs(dAna[i]) + Math.abs(num));
          maxRel = Math.max(maxRel, rel);
        }
        return { pass: maxRel < 1e-6, detail: `解析导数 vs 中心差分最大相对误差 ${maxRel.toExponential(2)}` };
      },
    },
    {
      name: '因果掩码不变量',
      run: () => {
        const model = new F.GPTModel({ vocabSize: 9, blockSize: 8, nLayer: 2, nHead: 2, dModel: 16, seed: 3 });
        const a = [0, 1, 2, 3, 4], b = [0, 1, 2, 3, 7];
        const ra = model.forward(a), rb = model.forward(b);
        const V = 9;
        let maxDiff = 0;
        for (let i = 0; i < a.length - 1; i++) {
          for (let j = 0; j < V; j++) {
            maxDiff = Math.max(maxDiff, Math.abs(ra.logits[i * V + j] - rb.logits[i * V + j]));
          }
        }
        // 注意力权重上三角必须严格为 0
        let upperZero = true;
        for (const lc of ra.cache.layers) {
          for (const P of lc.Ps) {
            for (let i = 0; i < ra.cache.T; i++) {
              for (let j = i + 1; j < ra.cache.T; j++) if (P[i * ra.cache.T + j] !== 0) upperZero = false;
            }
          }
        }
        return { pass: maxDiff < 1e-10 && upperZero, detail: `过去位置 logits 最大变化 ${maxDiff.toExponential(2)}；注意力上三角为0=${upperZero}` };
      },
    },
    {
      name: '中心差分梯度检验',
      run: () => {
        const V = 13, T = 5;
        const model = new F.GPTModel({ vocabSize: V, blockSize: 6, nLayer: 1, nHead: 2, dModel: 8, seed: 7 });
        const rng = F.mulberry32(5);
        const tokens = [];
        for (let i = 0; i < T; i++) tokens.push(Math.floor(rng() * V));
        const targets = [];
        for (let i = 1; i < T; i++) targets.push(tokens[i]);
        const { logits, cache } = model.forward(tokens);
        const { dlogits } = F.crossEntropy(logits, targets, T - 1, V, T);
        model.backward(cache, dlogits);
        const lossOf = () => {
          const lg = model.forward(tokens).logits;
          return F.crossEntropy(lg, targets, T - 1, V).loss;
        };
        // 汇总全部参数条目，确定性抽样至多 400 个
        const all = [];
        for (const key in model.params) {
          const p = model.params[key].d;
          for (let i = 0; i < p.length; i++) all.push([key, i]);
        }
        const srng = F.mulberry32(99);
        let cand = all;
        if (all.length > 400) {
          cand = all.slice();
          for (let i = cand.length - 1; i > 0; i--) {
            const j = Math.floor(srng() * (i + 1));
            const tmp = cand[i]; cand[i] = cand[j]; cand[j] = tmp;
          }
          cand = cand.slice(0, 400);
        }
        const e = 1e-5;
        let maxRatio = 0, checked = 0, skipped = 0, worst = '';
        for (const [key, i] of cand) {
          const p = model.params[key].d;
          const orig = p[i];
          p[i] = orig + e;
          const lp = lossOf();
          p[i] = orig - e;
          const lm = lossOf();
          p[i] = orig;
          const num = (lp - lm) / (2 * e);
          const ana = model.grads[key][i];
          if (Math.abs(ana) < 1e-12 && Math.abs(num) < 1e-12) { skipped++; continue; }
          // allclose 准则：|解析-数值| ≤ atol + rtol·max(|解析|,|数值|)
          const err = Math.abs(ana - num);
          const tol = 1e-7 + 1e-4 * Math.max(Math.abs(ana), Math.abs(num));
          const ratio = err / tol;
          if (ratio > maxRatio) { maxRatio = ratio; worst = `${key}[${i}] 解析=${ana.toExponential(3)} 数值=${num.toExponential(3)} err/tol=${ratio.toExponential(2)}`; }
          checked++;
        }
        return {
          pass: checked > 100 && maxRatio <= 1,
          detail: `检验 ${checked} 个参数（跳过 ${skipped} 个零梯度：Q/K 偏置受 softmax 移位不变性约束恒零），最大 err/tol=${maxRatio.toExponential(2)}；最差 ${worst}`,
        };
      },
    },
    {
      name: '过拟合：小语料可背熟',
      run: () => {
        const tok = new F.Tokenizer(POEM);
        const ids = tok.encode(POEM);
        const model = new F.GPTModel({ vocabSize: tok.vocabSize, blockSize: ids.length, nLayer: 2, nHead: 2, dModel: 32, seed: 1 });
        const { history } = F.train(model, ids, {
          steps: 300, lr: 3e-3, warmup: 20, wd: 0, fixedBatch: ids, logEvery: 1e9,
        });
        const first = history.slice(0, 20).reduce((a, b) => a + b, 0) / 20;
        const last = history[history.length - 1];
        const pass = last < 0.1 && last < 0.05 * first;
        return { pass, detail: `初始 loss ${first.toFixed(3)} -> 最终 loss ${last.toFixed(4)}（阈值 <0.1）` };
      },
    },
    {
      name: '随机批次训练 loss 下降',
      run: () => {
        const corpus = POEM.repeat(5);
        const tok = new F.Tokenizer(corpus);
        const ids = tok.encode(corpus);
        const model = new F.GPTModel({ vocabSize: tok.vocabSize, blockSize: 24, nLayer: 2, nHead: 2, dModel: 32, seed: 2 });
        const { history } = F.train(model, ids, { steps: 150, lr: 5e-3, batch: 24, warmup: 10, seed: 0, logEvery: 1e9 });
        const first = history.slice(0, 30).reduce((a, b) => a + b, 0) / 30;
        const last = history.slice(-30).reduce((a, b) => a + b, 0) / 30;
        const changed = model.paramCount() > 0;
        return { pass: last < first && changed, detail: `前30步均值 ${first.toFixed(3)} -> 后30步均值 ${last.toFixed(3)}` };
      },
    },
    {
      name: '生成确定性与健壮性',
      run: () => {
        const tok = new F.Tokenizer(POEM);
        const ids = tok.encode(POEM);
        const model = new F.GPTModel({ vocabSize: tok.vocabSize, blockSize: ids.length, nLayer: 1, nHead: 2, dModel: 32, seed: 1 });
        F.train(model, ids, { steps: 60, lr: 3e-3, warmup: 10, wd: 0, fixedBatch: ids, logEvery: 1e9 });
        const g1 = F.generate(model, tok, '朝辞', { maxNew: 30, temperature: 0.9, topK: 5, seed: 7 });
        const g2 = F.generate(model, tok, '朝辞', { maxNew: 30, temperature: 0.9, topK: 5, seed: 7 });
        const det = g1.text === g2.text;
        const greedy = F.generate(model, tok, '朝辞', { maxNew: 20, temperature: 0, seed: 1 });
        const greedyOk = greedy.text.length === 20 && !/[^\u4e00-\u9fff，。]/.test(greedy.text);
        const longPrompt = F.generate(model, tok, POEM.repeat(6), { maxNew: 20, temperature: 0.8, topK: 3, seed: 2 });
        const longOk = longPrompt.text.length === 20 && Number.isFinite(longPrompt.ids[0]);
        const emptyPrompt = F.generate(model, tok, '', { maxNew: 10, temperature: 1.0, seed: 3 });
        const emptyOk = emptyPrompt.text.length === 10;
        return {
          pass: det && greedyOk && longOk && emptyOk,
          detail: `同种子输出一致=${det}；贪心20字=${greedyOk}；超长prompt(120字/blockSize=20)=${longOk}；空prompt=${emptyOk}`,
        };
      },
    },
    {
      name: 'Tokenizer 编解码往返',
      run: () => {
        const tok = new F.Tokenizer(POEM + 'ABCabc');
        const round = tok.decode(tok.encode(POEM + 'ABCabc'));
        const skip = tok.encode('朝辞$未知￥').length === 2;
        const vocabOk = tok.vocabSize === new Set([...POEM + 'ABCabc']).size;
        return { pass: round === POEM + 'ABCabc' && skip && vocabOk, detail: `往返一致=${round === POEM + 'ABCabc'}；未知字符跳过=${skip}；词表大小=${tok.vocabSize}` };
      },
    },
    {
      name: 'KV Cache 与全序列前向逐位一致',
      run: () => {
        const tok = new F.Tokenizer(POEM);
        const ids = tok.encode(POEM);
        const B = 16;
        const model = new F.GPTModel({ vocabSize: tok.vocabSize, blockSize: B, nLayer: 2, nHead: 2, dModel: 16, seed: 3 });

        // (a) 单序列：缓存路径末位 logits == 全序列路径末位 logits
        const seq = ids.slice(0, 10);
        const full = model.forward(seq).logits;
        const kv = F.initKVCache(model);
        const cached = F.warmupKV(model, seq, kv);
        let maxDiff = 0;
        const off = (seq.length - 1) * model.cfg.vocabSize;
        for (let j = 0; j < model.cfg.vocabSize; j++) {
          maxDiff = Math.max(maxDiff, Math.abs(cached[j] - full[off + j]));
        }

        // (b) 纯增量区间（prompt+生成 < blockSize）：缓存/非缓存必须逐位一致
        const a = F.generate(model, tok, '朝辞', { maxNew: 10, temperature: 0, useCache: true, seed: 1 });
        const b = F.generate(model, tok, '朝辞', { maxNew: 10, temperature: 0, useCache: false, seed: 1 });
        const same = a.text === b.text && a.ids.every((v, i) => v === b.ids[i]);

        // (c) 长生成跨越 blockSize：滑动窗口 memmove 必须保持缓存容量恒定且输出全部有效
        const kv2 = F.initKVCache(model);
        const long = F.generate(model, tok, '朝辞', {
          maxNew: 40, temperature: 0, useCache: true, seed: 1, kv: kv2,
        });
        const overflow = long.ids.length === 40
          && long.ids.every(v => Number.isInteger(v) && v >= 0 && v < model.cfg.vocabSize)
          && kv2.n === B;

        return {
          pass: maxDiff === 0 && same && overflow,
          detail: `末位 logits 最大差异 ${maxDiff}（要求 0）；纯增量区间 10 步贪心输出逐位一致=${same}；超长生成(40>blockSize=16)滑动窗口容量恒定=${kv2.n === B}`,
        };
      },
    },
    {
      name: '模型持久化 save/load 无损往返',
      run: () => {
        const tok = new F.Tokenizer(POEM);
        const ids = tok.encode(POEM);
        const model = new F.GPTModel({ vocabSize: tok.vocabSize, blockSize: 16, nLayer: 1, nHead: 2, dModel: 16, seed: 5 });
        F.train(model, ids, { steps: 20, lr: 3e-3, warmup: 5, seed: 0, logEvery: 1e9 });

        const json = F.saveModel(model, tok);
        const restored = F.loadModel(json);

        // 参数逐位相等
        let maxDiff = 0;
        for (const key in model.params) {
          const p1 = model.params[key].d, p2 = restored.params[key].d;
          if (p1.length !== p2.length) return { pass: false, detail: `参数 ${key} 长度不匹配` };
          for (let i = 0; i < p1.length; i++) maxDiff = Math.max(maxDiff, Math.abs(p1[i] - p2[i]));
        }
        // 前向输出逐位相等 + 序列化幂等
        const seq = ids.slice(0, 12);
        const l1 = model.forward(seq).logits, l2 = restored.forward(seq).logits;
        let logitDiff = 0;
        for (let i = 0; i < l1.length; i++) logitDiff = Math.max(logitDiff, Math.abs(l1[i] - l2[i]));
        const stable = F.saveModel(restored, tok) === json;
        const cfgOk = restored.cfg.vocabSize === model.cfg.vocabSize;
        // 词表一并持久化：导入后分词器可完整重建，编解码行为一致
        const tok2 = F.loadTokenizer(json);
        const vocabOk = tok2 && tok2.vocabSize === tok.vocabSize
          && tok2.encode(POEM).every((v, i) => v === tok.encode(POEM)[i])
          && tok2.decode(tok2.encode(POEM)) === POEM;

        return {
          pass: maxDiff === 0 && logitDiff === 0 && stable && cfgOk && vocabOk,
          detail: `参数最大差异 ${maxDiff}；前向 logits 最大差异 ${logitDiff}；再序列化一致=${stable}；词表恢复=${!!vocabOk}（${tok.vocabSize} 字符）；体积 ${(json.length / 1024).toFixed(0)} KB`,
        };
      },
    },
  ];

  function runAll() {
    return CHECKS.map(c => ({ name: c.name, ...safeRun(c.run) }));
  }

  return { runAll, CHECKS, POEM };
});

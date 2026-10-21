/* L1 应用层 · 浏览器演示应用（demo-app）
 * ============================================================================
 * 所属层：L1（应用层）
 * 允许依赖：L0（装配产物 NS.L0.API）—— 依赖在 **调用期** 经 engine() 惰性取得，
 *   不在模块求值期捕获，因此本文件可在任意时刻被 Node 的 __loadOrder 循环求值而
 *   不产生副作用（K5：无 CommonJS 加载语法、无 Node 专属全局、无网络/宿主 API）。
 * 主要 @ref：nanoGPT 上手范式（train loop / loss 曲线 / 采样面板）
 * 迁移方式：从 src/app.js【整文件搬】（UI 逻辑零改写，Δ=0 同义）：
 *   - 依赖来源改写：`const F = window.FlagshipAI` → `engine()` 读 NS.L0.API；
 *     `const VF = window.FlagshipVerify` → 调用期读 globalThis.FlagshipVerify；
 *     目的：浏览器 demo 由 legacy 单体（原 src/flagship-engine.js 转发）切到真十层装配产物。
 *   - 启动方式改写：原 IIFE 立即执行 → 现注册为 NS.L1.app.start(ns)，由
 *     scripts/build-demo.js 在 __FLAGSHIP_APP__ 槽位内联追加启动调用。
 *   - 事件绑定 / 训练分块 / 曲线绘制 / 自检逐项执行的全部逻辑逐行保留。
 * 模板侧契约不变：仍通过 id 直接操作 DOM（corpus/log/chart/prompt/output/...），
 *   demo.template.html 无需任何改动。
 * ========================================================================*/
;(function (NS) {
  NS.L1 = NS.L1 || {};

  /** 取 L0 装配产物（34 键公共 API）。惰性求值以避开「层加载顺序 → API 尚未装配」的取值陷阱，
   *  并对两种 L0 形态都做兜底：优先 NS.L0.API（assemble 产物），其次直接调 NS.L0.assemble(ns)，
   *  最后兼容「构建脚本已把 34 键并入命名空间」的写法。结果回写缓存，保证全进程同一份。
   *  @returns {Object} 34 键公共 API
   */
  function engine() {
    if (NS.L0 && NS.L0.API) return NS.L0.API;
    if (NS.L0 && NS.L0.assemble) {
      NS.L0.API = NS.L0.assemble(NS);
      return NS.L0.API;
    }
    if (NS.GPTModel) return NS; // 构建脚本已把 34 键并入命名空间对象本体
    throw new Error('L0 尚未装配：请先调用 NS.L0.assemble(NS)');
  }

  const $ = (id) => document.getElementById(id);
  const state = {
    tok: null, model: null, trainState: null,
    history: [], stepsDone: 0, totalSteps: 0,
    training: false, stopFlag: false, stale: true,
  };

  function now() {
    return new Date().toLocaleTimeString('zh-CN', { hour12: false });
  }
  function log(msg, cls) {
    const el = $('log');
    const line = document.createElement('div');
    if (cls) line.className = cls;
    line.textContent = `[${now()}] ${msg}`;
    el.appendChild(line);
    el.scrollTop = el.scrollHeight;
  }

  /* ---------- 模型构建 ---------- */
  function buildModel() {
    const F = engine();
    const corpus = $('corpus').value;
    if (corpus.replace(/\s/g, '').length < 10) {
      log('语料太短（至少 10 个非空白字符），拒绝构建', 'err');
      return false;
    }
    const dModel = +$('dModel').value, nHead = +$('nHead').value;
    if (dModel % nHead !== 0) {
      log(`dModel=${dModel} 必须被 nHead=${nHead} 整除`, 'err');
      return false;
    }
    const nKvHeadRaw = $('nKvHead').value.trim();
    let nKvHead = -1;
    if (nKvHeadRaw !== '') {
      nKvHead = +nKvHeadRaw;
      if (!(nKvHead >= 1 && nKvHead <= nHead && nHead % nKvHead === 0)) {
        log(`nKvHead(${nKvHead}) 需满足 1 <= nKvHead <= nHead(${nHead}) 且 nHead 可被整除`, 'err');
        return false;
      }
    }
    const tokMode = $('tokenizer').value;
    const arch = $('arch').value;
    try {
      if (tokMode === 'char') {
        state.tok = new F.Tokenizer(corpus);
      } else {
        const byteLevel = tokMode === 'bpe-byte';
        state.tok = new F.BPETokenizer(corpus, +$('vocabSize').value || 320, { byteLevel });
      }
      let cfg = {
        vocabSize: state.tok.vocabSize,
        blockSize: +$('blockSize').value,
        nLayer: +$('nLayer').value,
        nHead: nHead,
        dModel: dModel,
        seed: +$('seed').value || 42,
      };
      if (arch === 'llama' || arch === 'llama-gqa' || arch === 'mixtral' || arch === 'swa' || arch === 'gemma') cfg = F.applyPreset(cfg, arch);
      if (nKvHead !== -1) cfg.nKvHead = nKvHead; // 显式指定优先于预设
      // MoE 覆盖：留空则跟随预设（mixtral=4专家/top2）；显式 nExp>1 即启用混合专家
      const nExpRaw = $('nExp').value.trim(), topKRaw = $('topK').value.trim();
      if (nExpRaw !== '') {
        const nExp = +nExpRaw;
        if (!(nExp >= 1)) { log('nExp 需 ≥ 1', 'err'); return false; }
        cfg.nExp = nExp;
        if (topKRaw !== '') {
          const topK = +topKRaw;
          if (!(topK >= 1 && topK <= nExp)) { log(`topK(${topK}) 需满足 1 ≤ topK ≤ nExp(${nExp})`, 'err'); return false; }
          cfg.topK = topK;
        } else if (!cfg.topK) cfg.topK = 1;
      } else if (topKRaw !== '' && cfg.nExp) {
        cfg.topK = Math.min(+topKRaw, cfg.nExp);
      }
      // 稀疏注意力覆盖：全因果 / 统一窗口 / Gemma 交错三种模式；
      // 选 swa/gemma 预设且保持「全因果」时不覆盖（跟随预设）
      const wMode = $('windowMode').value;
      if (wMode === 'uniform' || wMode === 'alt') {
        const winRaw = $('window').value.trim();
        const wN = winRaw === '' ? 32 : +winRaw;
        if (!(wN >= 1)) { log('window 需 ≥ 1', 'err'); return false; }
        cfg.window = wMode === 'alt' ? { alt: wN } : wN;
      } else if (arch !== 'swa' && arch !== 'gemma') {
        cfg.window = 0;
      }
      // Gemma 2 稳定化组件：qkNorm 三态（空=跟随预设 / on / off）；logitCap 0=不启用
      const qnMode = $('qkNorm').value;
      if (qnMode === 'on') cfg.qkNorm = true;
      else if (qnMode === 'off') cfg.qkNorm = false;
      const capRaw = $('logitCap').value.trim();
      if (capRaw !== '') {
        const cap = +capRaw;
        if (!(cap >= 0)) { log('logitCap 需 ≥ 0（0=不启用）', 'err'); return false; }
        cfg.logitCap = cap;
      }
      // KV 缓存 INT8 量化（推理期显存压缩）：8=启用，0=关闭（默认）
      const kvqRaw = $('kvQuant').value.trim();
      if (kvqRaw !== '') {
        const kvq = +kvqRaw;
        if (!(kvq === 0 || kvq === 8)) { log('kvQuant 仅支持 0（关闭）或 8（INT8）', 'err'); return false; }
        cfg.kvQuant = kvq;
      }
      // Attention Sink / StreamingLLM：推理期恒驻首 K 个锚点 token，支持无限上下文流式推理
      const sinkRaw = $('attentionSink').value.trim();
      if (sinkRaw !== '') {
        const sink = +sinkRaw;
        if (!(sink >= 0)) { log('attentionSink 需 ≥ 0（0=关闭）', 'err'); return false; }
        cfg.attentionSink = sink;
        if (sink > 0) {
          const rwRaw = $('rollingWindow').value.trim();
          const rw = rwRaw === '' ? 256 : +rwRaw;
          if (!(rw >= 1)) { log('rollingWindow 需 ≥ 1', 'err'); return false; }
          cfg.rollingWindow = rw;
        }
      }
      // YaRN 长度外推：factor>1 时启用 NTK-by-parts RoPE 频率插值 + 注意力温度（长上下文外推）
      const yfRaw = $('yarnFactor').value.trim();
      if (yfRaw !== '') {
        const yf = +yfRaw;
        if (!(yf >= 1)) { log('yarnFactor 需 ≥ 1（1=关闭外推）', 'err'); return false; }
        if (yf > 1) {
          const yt = $('yarnType').value === 'ntk' ? 'ntk' : 'yarn';
          cfg.ropeScale = { type: yt, factor: yf, originalMaxLength: +$('blockSize').value };
        }
      }
      state.arch = arch;
      state.tokMode = tokMode;
      state.model = new F.GPTModel(cfg);
    } catch (e) {
      log('构建失败: ' + e.message, 'err');
      return false;
    }
    state.trainState = null;
    state.history = [];
    state.stepsDone = 0;
    state.stale = false;
    const tokLabel = tokMode === 'char' ? '字符级' : ('BPE' + (tokMode === 'bpe-byte' ? '字节级' : '字符级'));
    const gqaLabel = state.model.cfg.KV < state.model.cfg.nHead ? `GQA(${state.model.cfg.KV} KV头)` : 'MHA';
    const winLabel = state.model.windows.some(w => w > 0)
      ? (state.model.windows.every(w => w === state.model.windows[0]) ? `SWA(window=${state.model.windows[0]})` : `层交错SWA[${state.model.windows}]`)
      : '全因果';
    const stabLabel = (state.model.qkNorm ? 'QK-norm' : '') + (state.model.logitCap > 0 ? `+cap${state.model.logitCap}` : '');
    const stabFull = stabLabel ? ` / ${stabLabel}` : '';
    const kvqFull = state.model.kvQuant === 8 ? ' / KV-INT8' : '';
    const sinkFull = state.model.attentionSink > 0 ? ` / Sink(${state.model.attentionSink})` : '';
    const yarnFull = state.model.ropeFreqScale ? ` / YaRN×${state.model.attentionScaling.toFixed(2)}` : '';
    log(`模型已构建：[${arch} / ${tokLabel} / ${gqaLabel} / ${winLabel}${stabFull}${kvqFull}${sinkFull}${yarnFull}] 词表 ${state.tok.vocabSize} · ${state.model.cfg.nLayer} 层 · ${state.model.cfg.nHead} 头 · d=${state.model.cfg.dModel} · 参数量 ${state.model.paramCount().toLocaleString()}`);
    $('trainBtn').disabled = false;
    $('genBtn').disabled = true;
    $('modelInfo').textContent = `[${arch}/${gqaLabel}/${winLabel}${stabFull}${kvqFull}${sinkFull}${yarnFull}] 词表 ${state.tok.vocabSize} · 参数 ${state.model.paramCount().toLocaleString()}`;
    drawChart();
    return true;
  }

  /* ---------- 训练（分块） ---------- */
  function startTraining() {
    if (state.training) return;
    if (state.stale && !buildModel()) return;
    state.totalSteps = +$('steps').value;
    state.training = true;
    state.stopFlag = false;
    $('trainBtn').disabled = true;
    $('stopBtn').disabled = false;
    $('buildBtn').disabled = true;
    log(`开始训练：共 ${state.totalSteps} 步，lr=${$('lr').value}，batch=${$('blockSize').value}`);
    trainChunk();
  }

  function trainChunk() {
    if (state.stopFlag || state.stepsDone >= state.totalSteps) { finishTraining(); return; }
    const F = engine();
    const chunk = Math.min(8, state.totalSteps - state.stepsDone);
    const ids = state.tok.encode($('corpus').value);
    let lastLoss = 0;
    const res = F.train(state.model, ids, {
      steps: chunk,
      lr: +$('lr').value,
      batch: +$('blockSize').value,
      warmup: Math.min(30, Math.floor(state.totalSteps / 5)),
      seed: 0,
      state: state.trainState,
      stepOffset: state.stepsDone,
      totalSteps: state.totalSteps,
      onStep: (_s, loss) => { lastLoss = loss; },
    });
    state.trainState = res.state;
    for (const h of res.history) state.history.push(h);
    state.stepsDone += chunk;
    $('progFill').style.width = (100 * state.stepsDone / state.totalSteps).toFixed(1) + '%';
    $('progText').textContent = `${state.stepsDone}/${state.totalSteps} 步 · loss ${lastLoss.toFixed(4)}`;
    drawChart();
    setTimeout(trainChunk, 0);
  }

  function finishTraining() {
    state.training = false;
    const ids = state.tok.encode($('corpus').value);
    const ev = state.model.evaluate(ids);
    $('trainBtn').disabled = false;
    $('stopBtn').disabled = true;
    $('buildBtn').disabled = false;
    $('genBtn').disabled = false;
    $('exportBtn').disabled = false;
    log(`${state.stopFlag ? '训练已停止' : '训练完成'}：${state.stepsDone} 步，最终 batch loss ${state.history[state.history.length - 1].toFixed(4)}，全语料评估 loss ${ev.loss.toFixed(4)} · 下一字准确率 ${(ev.acc * 100).toFixed(1)}%`, state.stopFlag ? 'warn' : 'ok');
  }

  /* ---------- Loss 曲线（log 纵轴） ---------- */
  function drawChart() {
    const cv = $('chart');
    const dpr = window.devicePixelRatio || 1;
    const W = cv.clientWidth, H = 180;
    if (cv.width !== W * dpr) { cv.width = W * dpr; cv.height = H * dpr; }
    const ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const padL = 44, padR = 8, padT = 10, padB = 20;
    const iw = W - padL - padR, ih = H - padT - padB;
    ctx.strokeStyle = '#e2e8f0';
    ctx.fillStyle = '#64748b';
    ctx.font = '10px ui-monospace, monospace';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = padT + ih * i / 4;
      ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
    }
    const h = state.history;
    if (!h.length) {
      ctx.fillText('等待训练数据…', padL + 6, padT + 14);
      return;
    }
    const minL = Math.max(1e-4, Math.min(...h));
    const maxL = Math.max(...h);
    const lo = Math.log10(minL), hi = Math.log10(maxL);
    const span = Math.max(0.5, hi - lo);
    const X = (i) => padL + iw * (h.length === 1 ? 1 : i / (h.length - 1));
    const Y = (v) => padT + ih * (1 - (Math.log10(Math.max(1e-4, v)) - (hi - span)) / span);
    for (let i = 0; i <= 4; i++) {
      const lv = Math.pow(10, hi - span + span * i / 4);
      ctx.fillText(lv >= 0.01 ? lv.toFixed(2) : lv.toExponential(0), 4, padT + ih * i / 4 + 3);
    }
    ctx.strokeStyle = '#4f46e5';
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    for (let i = 0; i < h.length; i++) {
      const x = X(i), y = Y(h[i]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.fillStyle = '#4f46e5';
    ctx.font = '11px ui-monospace, monospace';
    ctx.fillText(`loss ${h[h.length - 1].toFixed(4)}`, W - padR - 76, padT + 12);
  }

  /* ---------- 生成 ---------- */
  function doGenerate() {
    if (!state.model) { log('请先构建并训练模型', 'warn'); return; }
    const F = engine();
    const useQuant = $('quant') && $('quant').checked;
    const model = useQuant ? state.model.quantize() : state.model;
    const t0 = performance.now();
    const r = F.generate(model, state.tok, $('prompt').value, {
      maxNew: +$('maxNew').value,
      temperature: +$('temperature').value,
      topK: +$('topK').value,
      seed: +$('genSeed').value || 1234,
    });
    const ms = performance.now() - t0;
    $('output').textContent = r.text;
    const tps = ms > 0 ? (r.ids.length / ms * 1000).toFixed(0) : '—';
    log(`生成 ${r.ids.length} token，耗时 ${ms.toFixed(0)}ms（${tps} token/s · KV cache${useQuant ? ' · INT8 量化' : ''} · 温度 ${$('temperature').value} · top-k ${$('topK').value}）`);
  }

  /* ---------- 模型持久化 ---------- */
  function doExport() {
    if (!state.model) return;
    const F = engine();
    const json = F.saveModel(state.model, state.tok);
    const url = URL.createObjectURL(new Blob([json], { type: 'application/json' }));
    const a = document.createElement('a');
    a.href = url;
    a.download = 'flagship-model.json';
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
    log(`模型导出完成：${(json.length / 1024 / 1024).toFixed(2)} MB（含 ${state.tok.vocabSize} 词表 + 分词器元数据）`);
  }

  function handleImport(ev) {
    const F = engine();
    const file = ev.target.files && ev.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const json = String(reader.result);
        const model = F.loadModel(json);
        const tok = F.loadTokenizer(json);
        if (!tok) throw new Error('存档缺少词表');
        state.model = model;
        state.tok = tok;
        state.trained = true;
        state.stepsDone = 0;
        state.totalSteps = 0;
        state.history = [];
        state.stale = false;
        $('modelInfo').textContent =
          `词表 ${tok.vocabSize} · 参数 ${model.paramCount().toLocaleString()}（导入）`;
        $('genBtn').disabled = false;
        $('exportBtn').disabled = false;
        drawChart();
        log(`模型导入成功：${model.paramCount().toLocaleString()} 参数，可直接生成`);
      } catch (err) {
        log('导入失败：' + err.message, 'err');
      }
      ev.target.value = '';
    };
    reader.readAsText(file);
  }

  /* ---------- 一键自检（逐项执行） ---------- */
  function runSelfCheck() {
    const VF = globalThis.FlagshipVerify;
    $('checkBtn').disabled = true;
    const tbody = $('checkBody');
    tbody.innerHTML = '';
    log('自检开始：共 ' + VF.CHECKS.length + ' 项，逐项运行中…');
    let i = 0, passCount = 0;
    const next = () => {
      if (i >= VF.CHECKS.length) {
        log(`自检结束：${passCount}/${VF.CHECKS.length} 通过 ${passCount === VF.CHECKS.length ? '✅ 全绿' : '❌ 存在失败'}`, passCount === VF.CHECKS.length ? 'ok' : 'err');
        $('checkBtn').disabled = false;
        return;
      }
      const c = VF.CHECKS[i];
      let r;
      try {
        const out = c.run() || {};
        r = { pass: out.pass !== false, detail: out.detail || '' };
      } catch (e) {
        r = { pass: false, detail: '异常: ' + e.message };
      }
      if (r.pass) passCount++;
      const tr = document.createElement('tr');
      const td1 = document.createElement('td');
      td1.textContent = String(i + 1);
      const td2 = document.createElement('td');
      td2.textContent = c.name;
      const td3 = document.createElement('td');
      td3.textContent = r.pass ? '✅ 通过' : '❌ 失败';
      td3.className = r.pass ? 'ok' : 'err';
      const td4 = document.createElement('td');
      td4.className = 'detail';
      td4.textContent = r.detail;
      tr.append(td1, td2, td3, td4);
      tbody.appendChild(tr);
      i++;
      setTimeout(next, 20);
    };
    setTimeout(next, 30);
  }

  /* ---------- 事件绑定 ---------- */
  function markStale(msg) {
    if (state.model) {
      state.stale = true;
      $('genBtn').disabled = true;
      if (msg) $('modelInfo').textContent = msg;
    }
  }

  function bind() {
    $('buildBtn').addEventListener('click', buildModel);
    $('trainBtn').addEventListener('click', startTraining);
    $('stopBtn').addEventListener('click', () => { state.stopFlag = true; });
    $('genBtn').addEventListener('click', doGenerate);
    $('checkBtn').addEventListener('click', runSelfCheck);
    $('exportBtn').addEventListener('click', doExport);
    $('importBtn').addEventListener('click', () => $('importFile').click());
    $('importFile').addEventListener('change', handleImport);
    // 语料与模型结构/分词器/架构变动都需重新构建
    $('corpus').addEventListener('input', () => markStale('语料已修改，重新构建后生效'));
    ['dModel', 'nHead', 'nLayer', 'blockSize', 'tokenizer', 'arch', 'vocabSize', 'nKvHead', 'nExp', 'topK', 'window', 'windowMode', 'qkNorm', 'logitCap', 'kvQuant', 'attentionSink', 'rollingWindow', 'yarnFactor', 'yarnType'].forEach((id) => {
      $(id).addEventListener('change', () => markStale('配置已修改，重新构建后生效'));
    });
    // 架构联动：选 swa/gemma 预设时自动切换稀疏注意力模式（gemma 预设自带 QK-norm/soft-cap，留空即跟随）
    $('arch').addEventListener('change', () => {
      const a = $('arch').value;
      if (a === 'gemma') { $('windowMode').value = 'alt'; $('qkNorm').value = ''; $('logitCap').value = ''; }
      else if (a === 'swa') $('windowMode').value = 'uniform';
    });
    window.addEventListener('resize', drawChart);
    log('旗舰 AI 引擎就绪：零依赖纯 JS · 浏览器内一键自检');
    log('默认语料已载入，点击「构建模型」开始');
  }

  /** 启动演示应用：绑定 DOM 事件。
   *  @param {Object} ns 命名空间（约定由 build-demo.js 传入 globalThis.FlagshipAI）
   */
  function start(ns) {
    if (!document.getElementById('log')) {
      throw new Error('L1.app.start：模板 DOM 尚未就绪（缺少 #log）');
    }
    if (ns.L0 && ns.L0.assemble) ns.L0.assemble(ns); // 装配幂等守卫：未装配则补一次
    engine();                                        // 立即解析一次装配产物，早失败早暴露
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', bind);
    } else {
      bind();
    }
  }


  NS.L1.app = {
    start: start,
    bind: bind,
    buildModel: buildModel,
    startTraining: startTraining,
    doGenerate: doGenerate,
    doExport: doExport,
    runSelfCheck: runSelfCheck,
    drawChart: drawChart,
    state: state,
  };
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

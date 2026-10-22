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
 *
 * v15.6 演示层升级（本次改动，DOM id 契约零变更，仅新增元素）：
 *   1. 内联提示条（#alertBox）：错误/警告不再用 alert 阻塞，改为页头醒目提示。
 *   2. 按钮 busy 态：构建 / 训练 / 自检期间按钮禁用 + 文案变化 + 旋转指示。
 *   3. 模型状态卡：#modelInfo 从纯文本升级为结构化 chips（参数量/层数/头数/量化…）。
 *   4. Loss 曲线美化：坐标轴 + 网格 + 面积填充 + 最低值基线 + 深浅色自适应。
 *   5. L10 微指令浏览器：注册中心全貌（stats 八域明细）+ 域筛选 + 关键字搜索 +
 *      七件套详情 + 试运行（确定性 RNG 驱动 op.predicate 真实跑一次不变量）+ 批量跑。
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

  /** 页头内联提示（替代 alert，不阻塞交互）。
   *  @param {string} msg 提示文本
   *  @param {string} kind ''（错误）| 'warn' | 'ok'
   */
  function notify(msg, kind) {
    const box = $('alertBox'), txt = $('alertText');
    if (!box || !txt) return;
    txt.textContent = msg;
    box.className = 'alert-box show' + (kind ? ' ' + kind : '');
    if (notify._t) clearTimeout(notify._t);
    notify._t = setTimeout(() => { box.className = 'alert-box'; }, 6000);
  }
  function dismissNotify() {
    const box = $('alertBox');
    if (box) box.className = 'alert-box';
  }

  function log(msg, cls) {
    const el = $('log');
    const line = document.createElement('div');
    if (cls) line.className = cls;
    line.textContent = `[${now()}] ${msg}`;
    if (el) {
      el.appendChild(line);
      el.scrollTop = el.scrollHeight;
    }
    if (cls === 'err') notify(msg, '');
    else if (cls === 'warn') notify(msg, 'warn');
  }

  /* ---------- 按钮 busy 态：禁用 + 文案 + 旋转指示，回调后自动还原 ---------- */
  function busy(btn, label, fn) {
    if (!btn) { fn(); return; }
    const old = btn.textContent;
    btn.disabled = true;
    if (btn.classList) btn.classList.add('busy');
    btn.textContent = label;
    setTimeout(() => {
      try { fn(); } finally {
        if (btn.classList) btn.classList.remove('busy');
        btn.textContent = old;
        btn.disabled = false;
      }
    }, 24);
  }

  /* ---------- 模型状态卡（#modelInfo）---------- */
  function chips(chipList, hlKeys) {
    const ul = document.createElement('ul');
    ul.className = 'chips';
    for (const c of chipList) {
      const li = document.createElement('li');
      li.className = 'chip' + (hlKeys && hlKeys.indexOf(c[0]) >= 0 ? ' hl' : '');
      const b = document.createElement('b');
      b.textContent = c[0];
      const s = document.createElement('span');
      s.textContent = c[1];
      li.appendChild(b);
      li.appendChild(s);
      ul.appendChild(li);
    }
    return ul;
  }

  /** 把已构建模型的实况渲染为结构化 chips（数据全部取自 model 自身，不新造数据源）。 */
  function renderModelInfo(model, tokLabel) {
    const el = $('modelInfo');
    if (!el || !model) return;
    el.textContent = '';
    const cfg = model.cfg || {};
    const kv = cfg.KV != null ? cfg.KV : (cfg.nKvHead || cfg.nHead);
    const attn = kv < cfg.nHead ? `GQA ${kv}/${cfg.nHead}` : `MHA ${cfg.nHead}`;
    const win = (model.windows || []).some((w) => w > 0)
      ? ((model.windows || []).every((w) => w === model.windows[0])
        ? `SWA ${model.windows[0]}`
        : `交错 [${model.windows}]`)
      : '全因果';
    const moe = model.nExp > 1 ? `MoE ${model.nExp}专家/top${model.topK}` : '稠密';
    const stab = [];
    if (model.qkNorm) stab.push('QK-norm');
    if (model.logitCap > 0) stab.push('cap' + model.logitCap);
    const list = [
      ['架构', state.arch || '—'],
      ['注意力', attn],
      ['上下文', win],
      ['参数量', (model.paramCount ? model.paramCount() : 0).toLocaleString()],
      ['词表', String(cfg.vocabSize != null ? cfg.vocabSize : '—')],
      ['层数', String(cfg.nLayer)],
      ['头数', String(cfg.nHead)],
      ['dModel', String(cfg.dModel)],
      ['上下文长', String(cfg.blockSize)],
      ['FFN', moe],
      ['归一化', (cfg.norm || '—').toUpperCase()],
      ['位置编码', (cfg.pos || '—')],
      ['KV 缓存', model.kvQuant === 8 ? 'INT8 量化' : 'Float64'],
      ['Sink', model.attentionSink > 0 ? String(model.attentionSink) : '关'],
      ['YaRN', model.ropeFreqScale ? '×' + model.attentionScaling.toFixed(2) : '关'],
      ['稳定化', stab.length ? stab.join('+') : '—'],
      ['分词器', tokLabel || (state.tokMode === 'char' ? '字符级' : 'BPE')],
      ['种子', String(cfg.seed)],
    ];
    el.appendChild(chips(list, ['架构', '注意力', '上下文', '参数量']));
  }

  /** 状态卡占位（未构建 / 配置已变更）。 */
  function placeholderInfo(msg) {
    const el = $('modelInfo');
    if (!el) return;
    el.textContent = '';
    const d = document.createElement('div');
    d.className = 'mi-empty';
    d.textContent = msg;
    el.appendChild(d);
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
    renderModelInfo(state.model, tokLabel);
    notify(`模型已构建：${state.model.paramCount().toLocaleString()} 参数 · ${gqaLabel} · ${winLabel}`, 'ok');
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
    const tb = $('trainBtn');
    tb.disabled = true;
    if (tb.classList) tb.classList.add('busy');
    tb.textContent = '训练中…';
    $('stopBtn').disabled = false;
    $('buildBtn').disabled = true;
    setProg(0);
    log(`开始训练：共 ${state.totalSteps} 步，lr=${$('lr').value}，batch=${$('blockSize').value}`);
    trainChunk();
  }

  /** 进度条 + 进度文本统一更新。
   *  @param {number} ratio 0..1
   *  @param {string} text 进度文本
   *  @param {boolean} done 完成态（进度条转绿）
   */
  function setProg(ratio, text, done) {
    const fill = $('progFill');
    if (fill) fill.style.width = (100 * Math.max(0, Math.min(1, ratio))).toFixed(1) + '%';
    const bar = $('progBar');
    if (bar && bar.classList) {
      if (done) bar.classList.add('ok'); else bar.classList.remove('ok');
    }
    const t = $('progText');
    if (t && text) t.textContent = text;
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
    setProg(state.stepsDone / state.totalSteps,
      `${state.stepsDone}/${state.totalSteps} 步 · loss ${lastLoss.toFixed(4)} · ${(100 * state.stepsDone / state.totalSteps).toFixed(0)}%`,
      state.stepsDone >= state.totalSteps);
    drawChart();
    setTimeout(trainChunk, 0);
  }

  function finishTraining() {
    state.training = false;
    const tb = $('trainBtn');
    if (tb.classList) tb.classList.remove('busy');
    tb.textContent = '开始训练';
    tb.disabled = false;
    $('stopBtn').disabled = true;
    $('buildBtn').disabled = false;
    $('genBtn').disabled = false;
    $('exportBtn').disabled = false;
    const last = state.history.length ? state.history[state.history.length - 1].toFixed(4) : '—';
    let evText = '（未评估）';
    try {
      const ids = state.tok.encode($('corpus').value);
      const ev = state.model.evaluate(ids);
      evText = `全语料评估 loss ${ev.loss.toFixed(4)} · 下一字准确率 ${(ev.acc * 100).toFixed(1)}%`;
    } catch (e) {
      evText = '评估跳过：' + e.message;
    }
    log(`${state.stopFlag ? '训练已停止' : '训练完成'}：${state.stepsDone} 步，最终 batch loss ${last}，${evText}`, state.stopFlag ? 'warn' : 'ok');
  }

  /* ---------- Loss 曲线（log 纵轴 · 主题自适应） ---------- */
  const CHART_FALLBACK = {
    grid: '#e8edf5', axis: '#94a3b8', line: '#4f46e5',
    line2: '#f59e0b', area: 'rgba(79, 70, 229, .14)', ink: '#334155',
  };

  /** 读取 CSS 变量（图表随 prefers-color-scheme 自适应）；取不到则用浅色兜底值。 */
  function cssVar(name, fallback) {
    let v = '';
    try {
      if (typeof getComputedStyle === 'function' && document && document.documentElement) {
        v = getComputedStyle(document.documentElement).getPropertyValue(name) || '';
      }
    } catch (e) { v = ''; }
    v = String(v).trim();
    return v || fallback;
  }

  function fmtTick(v) {
    if (v >= 100) return v.toFixed(0);
    if (v >= 1) return v.toFixed(1);
    if (v >= 0.01) return v.toFixed(3);
    return v.toExponential(1);
  }

  function drawChart() {
    const cv = $('chart');
    if (!cv || typeof cv.getContext !== 'function') return;
    const ctx = cv.getContext('2d');
    if (!ctx) return;
    const dpr = (typeof window !== 'undefined' && window.devicePixelRatio) || 1;
    const W = cv.clientWidth || 520, H = 200;
    if (cv.width !== Math.round(W * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    const C = {
      grid: cssVar('--chart-grid', CHART_FALLBACK.grid),
      axis: cssVar('--chart-axis', CHART_FALLBACK.axis),
      line: cssVar('--chart-line', CHART_FALLBACK.line),
      line2: cssVar('--chart-line-2', CHART_FALLBACK.line2),
      area: cssVar('--chart-area', CHART_FALLBACK.area),
      ink: cssVar('--chart-ink', CHART_FALLBACK.ink),
    };
    const padL = 48, padR = 14, padT = 16, padB = 28;
    const iw = W - padL - padR, ih = H - padT - padB;
    ctx.font = '10px ui-monospace, monospace';
    ctx.textBaseline = 'middle';

    // 空态：坐标轴骨架 + 提示
    const h = state.history;
    if (!h.length) {
      ctx.strokeStyle = C.grid;
      ctx.lineWidth = 1;
      for (let i = 0; i <= 4; i++) {
        const y = padT + ih * i / 4;
        ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
      }
      ctx.fillStyle = C.axis;
      ctx.textAlign = 'center';
      ctx.fillText('等待训练数据…（纵轴为对数刻度 loss）', padL + iw / 2, padT + ih / 2);
      return;
    }

    const minL = Math.max(1e-4, Math.min.apply(null, h));
    const maxL = Math.max.apply(null, h);
    const hi = Math.log10(maxL);
    const span = Math.max(0.5, hi - Math.log10(minL));
    const lo = hi - span;
    const X = (i) => padL + iw * (h.length === 1 ? 1 : i / (h.length - 1));
    const Y = (v) => padT + ih * (1 - (Math.log10(Math.max(1e-4, v)) - lo) / span);

    // 横向网格 + 纵轴刻度（对数）
    ctx.strokeStyle = C.grid;
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.fillStyle = C.axis;
    ctx.textAlign = 'right';
    for (let i = 0; i <= 4; i++) {
      const y = padT + ih * i / 4;
      ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
      ctx.fillText(fmtTick(Math.pow(10, lo + span * i / 4)), padL - 6, y);
    }
    ctx.setLineDash([]);

    // 横轴刻度（训练步）
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.strokeStyle = C.grid;
    for (let i = 0; i <= 4; i++) {
      const x = padL + iw * i / 4;
      ctx.beginPath(); ctx.moveTo(x, padT + ih); ctx.lineTo(x, padT + ih + 4); ctx.stroke();
      ctx.fillText(String(Math.round(h.length * i / 4)), x, padT + ih + 7);
    }
    ctx.textBaseline = 'middle';

    // 面积填充
    ctx.beginPath();
    ctx.moveTo(X(0), padT + ih);
    for (let i = 0; i < h.length; i++) ctx.lineTo(X(i), Y(h[i]));
    ctx.lineTo(X(h.length - 1), padT + ih);
    ctx.closePath();
    ctx.fillStyle = C.area;
    ctx.fill();

    // 最低 loss 基线
    const yMin = Y(minL);
    ctx.strokeStyle = C.line2;
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 4]);
    ctx.beginPath(); ctx.moveTo(padL, yMin); ctx.lineTo(W - padR, yMin); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = C.line2;
    ctx.textAlign = 'left';
    ctx.fillText('min ' + fmtTick(minL), padL + 4, yMin - 8);

    // loss 主线
    ctx.strokeStyle = C.line;
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    for (let i = 0; i < h.length; i++) {
      const x = X(i), y = Y(h[i]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // 末点标记 + 当前值
    const lx = X(h.length - 1), ly = Y(h[h.length - 1]);
    ctx.fillStyle = C.line;
    ctx.beginPath(); ctx.arc(lx, ly, 3, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = C.ink;
    ctx.textAlign = 'right';
    ctx.font = '11px ui-monospace, monospace';
    ctx.fillText('loss ' + h[h.length - 1].toFixed(4), W - padR - 2, padT + 4);
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
        state.arch = (model.cfg && model.cfg.norm === 'rmsnorm') ? 'llama' : 'gpt2';
        renderModelInfo(model, '导入');
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
        const all = passCount === VF.CHECKS.length;
        log(`自检结束：${passCount}/${VF.CHECKS.length} 通过 ${all ? '✅ 全绿' : '❌ 存在失败'}`, all ? 'ok' : 'err');
        const cb = $('checkBtn');
        if (cb.classList) cb.classList.remove('busy');
        cb.textContent = '运行自检';
        cb.disabled = false;
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

  /* ==========================================================================
   * L10 微指令浏览器
   * --------------------------------------------------------------------------
   * 数据源：NS.L10（注册中心真源）—— stats() 八域明细、ops 表 1024 项、TAGS 域
   *   标签、genScalar/genVector/genMatrix/genShape 确定性生成器、check() 谓词执行器。
   * 参数构造：与 tests/lib/op-harness.js 的 genArgs 同构（matrix→defDims/matrixDims、
   *   vector→defLen/vecDims、shape→genShape、其余→genScalar），种子由
   *   NS.L9.rng.mulberry32 确定性生成 —— 同一 seed 在任何机器上得到同一组参数。
   * ========================================================================*/
  const l10 = {
    ready: false,
    all: [],          // 全部已实现微指令（按 key 升序）
    filtered: [],     // 当前筛选结果
    shown: 0,         // 已渲染条数（分页）
    pageSize: 60,
    domain: 'ALL',    // 域筛选：'ALL' | 'A'..'H'
    query: '',        // 关键字（匹配 name / key / μkey）
    selected: null,   // 当前选中项
    items: [],        // { op, btn } 已渲染项（用于高亮同步）
    batching: false,
    stopBatch: false,
  };

  const L10_DOMAIN_HINT = {
    A: '标量算术/函数', B: '向量运算', C: '矩阵/张量', D: '注意力基元',
    E: '形状与索引', F: '归一化', G: '激活与损失', H: '采样与量化',
  };

  /** 与 tests/lib/op-harness.js 的 genArgs 同构：按 op.domains 的域标签确定性构造参数。 */
  function l10GenArgs(rng, op) {
    const L = NS.L10;
    const names = op.inputs || [];
    const args = [];
    for (let i = 0; i < names.length; i++) {
      const name = names[i];
      const tag = (op.domains || {})[name];
      const t = L.TAGS[tag];
      if (!t) throw new Error(`op ${op.key} 引用未知道域标签 "${tag}"`);
      if (t.kind === 'matrix') {
        const dims = (op.matrixDims && op.matrixDims[name]) || t.defDims || [4, 4];
        args.push(L.genMatrix(rng, dims[0], dims[1], tag));
      } else if (t.kind === 'vector') {
        const n = (op.vecDims && op.vecDims[name]) || t.defLen || 8;
        args.push(L.genVector(rng, n, tag));
      } else if (t.kind === 'shape') {
        args.push(L.genShape(rng, tag));
      } else {
        args.push(L.genScalar(rng, tag));
      }
    }
    return args;
  }

  /** 以某个种子对某个微指令跑一次不变量谓词（真实执行 op.predicate）。 */
  function l10Trial(op, seed) {
    const L = NS.L10;
    if (!L || !L.check) return { pass: false, detail: 'L10.check 未就位', seed: seed };
    if (!NS.L9 || !NS.L9.rng || typeof NS.L9.rng.mulberry32 !== 'function') {
      return { pass: false, detail: 'L9.rng 未就位', seed: seed };
    }
    try {
      const rng = NS.L9.rng.mulberry32(seed >>> 0);
      const args = l10GenArgs(rng, op);
      const r = L.check(op, args);
      return { pass: !!r.pass, detail: r.detail || '', seed: seed };
    } catch (e) {
      return { pass: false, detail: '异常: ' + (e && e.message), seed: seed };
    }
  }

  /** 种子的确定性序列：第 s 个种子（避免跨项全同，同时保持可复现）。 */
  function l10SeedAt(s) { return (1000 + s * 7919) >>> 0; }
  function l10SeedCount() {
    const el = $('l10Seeds');
    const n = el ? +el.value : 8;
    return Math.max(1, Math.min(200, isFinite(n) && n > 0 ? Math.floor(n) : 8));
  }
  function l10Clock() {
    return (typeof performance !== 'undefined' && performance && performance.now) ? performance.now() : 0;
  }

  /* ---- 列表项 / 详情 ---- */
  function l10ItemNode(op) {
    const li = document.createElement('li');
    const btn = document.createElement('button');
    btn.className = 'l10-item';
    btn.setAttribute('type', 'button');
    btn.setAttribute('aria-current', 'false');
    const k = document.createElement('span');
    k.className = 'k';
    k.textContent = 'μ' + op.key;
    const n = document.createElement('span');
    n.className = 'n';
    n.textContent = op.name || '(未命名)';
    const g = document.createElement('span');
    g.className = 'g';
    g.textContent = op.domain;
    btn.appendChild(k); btn.appendChild(n); btn.appendChild(g);
    btn.addEventListener('click', () => l10Select(op));
    li.appendChild(btn);
    l10.items.push({ op: op, btn: btn });
    return li;
  }

  function l10Piece(dl, cls, label, text) {
    const row = document.createElement('div');
    if (cls) row.className = cls;
    const dt = document.createElement('dt');
    dt.textContent = label;
    const dd = document.createElement('dd');
    dd.textContent = (text == null || text === '') ? '—' : String(text);
    row.appendChild(dt); row.appendChild(dd);
    dl.appendChild(row);
  }

  function l10Select(op) {
    l10.selected = op;
    for (let i = 0; i < l10.items.length; i++) {
      const it = l10.items[i];
      it.btn.setAttribute('aria-current', it.op === op ? 'true' : 'false');
    }
    const box = $('l10Detail');
    if (!box) return;
    box.textContent = '';

    const head = document.createElement('div');
    head.className = 'l10-d-head';
    const key = document.createElement('span');
    key.className = 'key';
    key.textContent = 'μ' + op.key;
    const h3 = document.createElement('h3');
    h3.textContent = op.name || '(未命名)';
    const dom = document.createElement('span');
    dom.className = 'badge';
    dom.textContent = `域 ${op.domain} · ${(NS.L10.domains[op.domain] || {}).name || L10_DOMAIN_HINT[op.domain] || ''}`;
    head.appendChild(key); head.appendChild(h3); head.appendChild(dom);
    box.appendChild(head);

    const dl = document.createElement('dl');
    dl.className = 'l10-seven';
    l10Piece(dl, 'intuition', '直觉', op.intuition);
    l10Piece(dl, 'math', '数学', op.math);
    l10Piece(dl, 'pseudo', '伪代码', op.pseudocode);
    l10Piece(dl, 'complexity', '复杂度', op.complexity);
    l10Piece(dl, 'ref', '参考', op.ref);
    l10Piece(dl, 'commonPitfall', '常见坑', op.commonPitfall);
    l10Piece(dl, 'invariant', '不变量', op.invariant);
    const inputs = (op.inputs || []).map((nm) => `${nm}:${(op.domains || {})[nm] || '?'}`).join('  ');
    l10Piece(dl, 'inputs', '输入域', inputs || '—');
    box.appendChild(dl);

    const run = $('l10RunBtn');
    if (run) run.disabled = false;
    const res = $('l10RunResult');
    if (res) { res.className = 'l10-result'; res.textContent = '就绪 · 可试运行'; }
  }

  /* ---- 渲染 ---- */
  function l10RenderSummary(st) {
    const box = $('l10Summary');
    if (!box) return;
    box.textContent = '';
    const letters = NS.L10.DOMAINS || Object.keys(L10_DOMAIN_HINT);
    for (let i = 0; i < letters.length; i++) {
      const d = letters[i];
      const s = st[d] || { impl: 0, capacity: 128 };
      const b = document.createElement('button');
      b.className = 'l10-dom';
      b.setAttribute('type', 'button');
      b.setAttribute('aria-pressed', 'false');
      b.title = `${d} 域：${s.impl}/${s.capacity} 已实现 — 点击筛选`;
      const k = document.createElement('span');
      k.className = 'd-key';
      k.textContent = `${d} · ${(NS.L10.domains[d] || {}).name || L10_DOMAIN_HINT[d] || ''}`;
      const nm = document.createElement('span');
      nm.className = 'd-name';
      nm.textContent = L10_DOMAIN_HINT[d] || '';
      const bar = document.createElement('span');
      bar.className = 'd-bar';
      const fill = document.createElement('i');
      fill.style.width = (100 * s.impl / s.capacity).toFixed(1) + '%';
      bar.appendChild(fill);
      const num = document.createElement('span');
      num.className = 'd-num';
      num.textContent = `${s.impl} / ${s.capacity}`;
      b.appendChild(k); b.appendChild(nm); b.appendChild(bar); b.appendChild(num);
      b.addEventListener('click', () => { l10SetDomain(d === l10.domain ? 'ALL' : d); });
      box.appendChild(b);
      b.l10Domain = d;
      l10.domainButtons.push(b);
    }
  }

  function l10RenderDomainBar() {
    const bar = $('l10DomainBar');
    if (!bar) return;
    bar.textContent = '';
    const letters = NS.L10.DOMAINS || Object.keys(L10_DOMAIN_HINT);
    const all = ['ALL'].concat(letters);
    for (let i = 0; i < all.length; i++) {
      const d = all[i];
      const b = document.createElement('button');
      b.className = 'chip-btn';
      b.setAttribute('type', 'button');
      b.setAttribute('aria-pressed', d === l10.domain ? 'true' : 'false');
      b.textContent = d === 'ALL' ? '全部' : d;
      b.title = d === 'ALL' ? '全部八域 1024 项' : `${d} 域 · ${L10_DOMAIN_HINT[d] || ''}`;
      b.addEventListener('click', () => l10SetDomain(d));
      bar.appendChild(b);
      l10.chipButtons.push({ d: d, btn: b });
    }
  }

  function l10SetDomain(d) {
    l10.domain = d;
    for (let i = 0; i < l10.chipButtons.length; i++) {
      const c = l10.chipButtons[i];
      c.btn.setAttribute('aria-pressed', c.d === d ? 'true' : 'false');
    }
    for (let i = 0; i < l10.domainButtons.length; i++) {
      const b = l10.domainButtons[i];
      b.setAttribute('aria-pressed', b.l10Domain === d ? 'true' : 'false');
    }
    l10ApplyFilter();
  }

  function l10ApplyFilter() {
    const q = l10.query.trim().toLowerCase();
    const out = [];
    for (let i = 0; i < l10.all.length; i++) {
      const op = l10.all[i];
      if (l10.domain !== 'ALL' && op.domain !== l10.domain) continue;
      if (q) {
        const hay = ((op.name || '') + ' ' + op.key + ' μ' + op.key).toLowerCase();
        if (hay.indexOf(q) < 0) continue;
      }
      out.push(op);
    }
    l10.filtered = out;
    l10.shown = 0;
    l10.items = [];
    l10RenderList(true);
    const c = $('l10Count');
    if (c) c.textContent = `${l10.filtered.length} / ${l10.all.length} 项`;
  }

  /** 渲染列表。reset=true 时清空重画（筛选变化）；否则只追加下一页（加载更多）。 */
  function l10RenderList(reset) {
    const list = $('l10List');
    if (!list) return;
    if (reset) list.textContent = '';
    const end = Math.min(l10.filtered.length, l10.shown + l10.pageSize);
    for (; l10.shown < end; l10.shown++) list.appendChild(l10ItemNode(l10.filtered[l10.shown]));
    const more = $('l10More');
    if (more) {
      const left = l10.filtered.length - l10.shown;
      more.style.display = left > 0 ? '' : 'none';
      more.textContent = left > 0 ? `加载更多（剩余 ${left} 项）` : '已全部加载';
    }
  }

  function l10RenderMore() {
    const before = l10.shown;
    l10RenderList(false);
    // 原列表已被清空时（首屏）无需滚动；追加场景滚到底部便于连续浏览
    if (l10.shown > before && before > 0) {
      const list = $('l10List');
      if (list && list.scrollHeight) list.scrollTop = list.scrollHeight;
    }
  }

  /* ---- 试运行 ---- */
  function l10RunOne() {
    const op = l10.selected;
    if (!op) return;
    const n = l10SeedCount();
    const t0 = l10Clock();
    let pass = 0, fail = 0, firstFail = null;
    for (let s = 0; s < n; s++) {
      const r = l10Trial(op, l10SeedAt(s));
      if (r.pass) pass++;
      else { fail++; if (!firstFail) firstFail = r; }
    }
    const ms = l10Clock() - t0;
    const res = $('l10RunResult');
    if (res) {
      res.className = 'l10-result ' + (fail ? 'err-text' : 'ok-text');
      res.textContent = `${fail ? '✗' : '✓'} μ${op.key} ${pass}/${n} 通过`
        + (fail ? ` · 首个失败 seed=${firstFail.seed}：${firstFail.detail}` : '')
        + ` · ${ms.toFixed(1)}ms`;
    }
    log(`微指令 μ${op.key} 试运行：${pass}/${n} 通过${fail ? '（存在失败）' : ''}，耗时 ${ms.toFixed(1)}ms`, fail ? 'err' : 'ok');
  }

  function l10Batch() {
    if (l10.batching || !l10.filtered.length) return;
    const seeds = l10SeedCount();
    const ops = l10.filtered.slice();
    l10.batching = true;
    l10.stopBatch = false;
    const bb = $('l10BatchBtn'), sb = $('l10StopBtn');
    if (bb) { bb.disabled = true; if (bb.classList) bb.classList.add('busy'); bb.textContent = '批量试运行中…'; }
    if (sb) sb.disabled = false;
    let idx = 0, passOps = 0, failOps = 0, evals = 0;
    const fails = [];
    const t0 = l10Clock();
    const CHUNK = 24;

    function setBatchProg() {
      const ratio = idx / ops.length;
      const fill = $('l10BatchFill');
      if (fill) fill.style.width = (100 * ratio).toFixed(1) + '%';
      const bar = $('l10ProgBar');
      if (bar && bar.classList) bar.classList.remove('ok');
      const t = $('l10BatchText');
      if (t) t.textContent = `${idx}/${ops.length} 项 · 通过 ${passOps} · 失败 ${failOps} · ${evals} 次谓词求值`;
    }

    function done() {
      l10.batching = false;
      const ms = l10Clock() - t0;
      if (bb) {
        if (bb.classList) bb.classList.remove('busy');
        bb.textContent = '批量试运行当前筛选';
        bb.disabled = false;
      }
      if (sb) sb.disabled = true;
      const bar = $('l10ProgBar');
      if (bar && bar.classList && failOps === 0) bar.classList.add('ok');
      const fill = $('l10BatchFill');
      if (fill) fill.style.width = '100%';
      const t = $('l10BatchText');
      if (t) {
        t.textContent = `${l10.stopBatch ? '已停止：' : '完成：'}${idx}/${ops.length} 项 · 通过 ${passOps} · 失败 ${failOps}`
          + ` · ${evals} 次谓词求值 · ${ms.toFixed(0)}ms`
          + (fails.length ? ` · 失败项 ${fails.join(' ')}${failOps > fails.length ? ' …' : ''}` : '');
      }
      const res = $('l10RunResult');
      if (res) {
        res.className = 'l10-result ' + (failOps ? 'err-text' : 'ok-text');
        res.textContent = `${failOps ? '✗' : '✓'} ${passOps}/${ops.length} 项通过（每项 ${seeds} 种子）`;
      }
      log(`L10 批量试运行：${passOps}/${ops.length} 项通过，${evals} 次谓词求值，耗时 ${ms.toFixed(0)}ms`, failOps ? 'err' : 'ok');
    }

    function tick() {
      if (l10.stopBatch || idx >= ops.length) { done(); return; }
      const end = Math.min(ops.length, idx + CHUNK);
      for (; idx < end; idx++) {
        let ok = true;
        for (let s = 0; s < seeds; s++) {
          evals++;
          if (!l10Trial(ops[idx], l10SeedAt(s)).pass) { ok = false; break; }
        }
        if (ok) passOps++;
        else { failOps++; if (fails.length < 10) fails.push('μ' + ops[idx].key); }
      }
      setBatchProg();
      setTimeout(tick, 0);
    }
    setBatchProg();
    setTimeout(tick, 0);
  }

  function l10StopBatch() {
    if (l10.batching) l10.stopBatch = true;
  }

  /** 初始化浏览器：读 stats、建列表、渲染并绑定交互。 */
  function l10Init() {
    const L = NS.L10;
    if (!L || !L.ops || typeof L.stats !== 'function') {
      log('L10 注册中心未就位，微指令浏览器不可用', 'warn');
      return;
    }
    l10.domainButtons = [];
    l10.chipButtons = [];
    const st = L.stats();
    const badge = $('l10Badge');
    if (badge) badge.textContent = `L10 微指令 ${st.total.impl}/${st.total.capacity}`;
    const badge2 = $('l10Badge2');
    if (badge2) badge2.textContent = `${st.total.impl}/${st.total.capacity}`;

    const keys = Object.keys(L.ops);
    keys.sort();
    for (let i = 0; i < keys.length; i++) {
      const op = L.ops[keys[i]];
      if (op && op.status === 'impl') l10.all.push(op);
    }
    l10RenderSummary(st);
    l10RenderDomainBar();
    l10ApplyFilter();

    const search = $('l10Search');
    if (search) {
      let t = null;
      search.addEventListener('input', () => {
        if (t) clearTimeout(t);
        t = setTimeout(() => { l10.query = search.value || ''; l10ApplyFilter(); }, 120);
      });
    }
    const more = $('l10More');
    if (more) more.addEventListener('click', l10RenderMore);
    const run = $('l10RunBtn');
    if (run) run.addEventListener('click', () => busy(run, '试运行中…', l10RunOne));
    const batch = $('l10BatchBtn');
    if (batch) batch.addEventListener('click', l10Batch);
    const stop = $('l10StopBtn');
    if (stop) stop.addEventListener('click', l10StopBatch);

    l10.ready = true;
    log(`L10 微指令浏览器就绪：${l10.all.length} 项已实现（八域 A–H，容量 ${st.total.capacity}）`);
  }

  /* ---------- 事件绑定 ---------- */
  function markStale(msg) {
    if (state.model) {
      state.stale = true;
      $('genBtn').disabled = true;
      if (msg) placeholderInfo(msg + '（点击「构建模型」重建）');
    }
  }

  function bind() {
    $('buildBtn').addEventListener('click', () => busy($('buildBtn'), '构建中…', buildModel));
    $('trainBtn').addEventListener('click', startTraining);
    $('stopBtn').addEventListener('click', () => { state.stopFlag = true; });
    $('genBtn').addEventListener('click', () => busy($('genBtn'), '生成中…', doGenerate));
    // 自检是异步分帧执行的，禁用态由 runSelfCheck 自己收尾，故此处的 busy 只管文案与指示
    $('checkBtn').addEventListener('click', () => {
      const cb = $('checkBtn');
      if (cb.classList) cb.classList.add('busy');
      cb.textContent = '自检中…';
      runSelfCheck();
    });
    $('exportBtn').addEventListener('click', doExport);
    $('importBtn').addEventListener('click', () => $('importFile').click());
    $('importFile').addEventListener('change', handleImport);
    const closeBtn = $('alertClose');
    if (closeBtn) closeBtn.addEventListener('click', dismissNotify);
    // 语料与模型结构/分词器/架构变动都需重新构建
    $('corpus').addEventListener('input', () => markStale('语料已修改'));
    ['dModel', 'nHead', 'nLayer', 'blockSize', 'tokenizer', 'arch', 'vocabSize', 'nKvHead', 'nExp', 'topK', 'window', 'windowMode', 'qkNorm', 'logitCap', 'kvQuant', 'attentionSink', 'rollingWindow', 'yarnFactor', 'yarnType'].forEach((id) => {
      $(id).addEventListener('change', () => markStale('配置已修改'));
    });
    // 架构联动：选 swa/gemma 预设时自动切换稀疏注意力模式（gemma 预设自带 QK-norm/soft-cap，留空即跟随）
    $('arch').addEventListener('change', () => {
      const a = $('arch').value;
      if (a === 'gemma') { $('windowMode').value = 'alt'; $('qkNorm').value = ''; $('logitCap').value = ''; }
      else if (a === 'swa') $('windowMode').value = 'uniform';
    });
    window.addEventListener('resize', drawChart);
    // 主题切换（深/浅色）时重绘图表，配色从 CSS 变量实时读取
    try {
      if (typeof window.matchMedia === 'function') {
        const mq = window.matchMedia('(prefers-color-scheme: dark)');
        if (mq && typeof mq.addEventListener === 'function') {
          mq.addEventListener('change', drawChart);
        }
      }
    } catch (e) { /* 老浏览器无 matchMedia：忽略，配色走 CSS 变量兜底 */ }
    placeholderInfo('尚未构建模型');
    try { l10Init(); } catch (e) { log('L10 浏览器初始化失败：' + e.message, 'err'); }
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
    notify: notify,
    l10Init: l10Init,
    l10RunOne: l10RunOne,
    l10Batch: l10Batch,
    l10: l10,
    state: state,
  };
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

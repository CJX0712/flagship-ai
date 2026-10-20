/* L3 架构 · GPTModel（参数布局 + zeroGrads + paramCount + quantize + fwd/bwd 委托）
 * 所属层：L3（架构组装）｜ 允许依赖：L4（神经算子）、L8（数值内核）、L9（数值基础）、L3.blocks
 * 主要 @ref：GPT-2 (Radford 2019)；LLaMA/Mistral；Mixtral MoE；Gemma 2/3；StreamingLLM；YaRN
 * 迁移方式：从 legacy flagship-engine.v11.js 第 702–909 行【原样搬】。
 *   - 构造器/zeroGrads/paramCount/quantize 逐字搬（RNG 消费顺序冻结契约 K7，R1 的命门）。
 *   - 构造器引用的 mulberry32/randn（L9）、ROPE_BASE（L4）、quantize 引用的 isQuantLinear（L8）
 *     均在文件顶部捕获（这些层先于 L3 加载，见 __loadOrder）。
 *   - forward/backward 不再内联 123+134 行单体，改为委托 L3.blocks（T03 已重组装，Δ=0）。
 */
;(function (NS) {
  NS.L3 = NS.L3 || {};
  NS.L3.gpt_model = NS.L3.gpt_model || {};
  const { mulberry32, randn } = NS.L9.rng;
  const ROPE_BASE = NS.L4.positional.ROPE_BASE;
  const { isQuantLinear } = NS.L8.matmul;

  class GPTModel {
    /**
     * @param {Object} cfg { vocabSize, blockSize, nLayer, nHead, dModel, seed }
     *   架构开关（默认全部为 GPT-2 原设计，保持向后兼容；'llama' 预设见 ARCH_PRESETS）：
     *   norm: 'ln'（LayerNorm，GPT-2）| 'rmsnorm'（RMSNorm，LLaMA/Mistral）
     *   ffn:  'gelu'（GPT-2）        | 'swiglu'（SiLU 门控，LLaMA/Mistral）
     *   pos:  'learned'（可学习位置嵌入，GPT-2）| 'rope'（旋转位置编码，LLaMA/Mistral）
     *   ffnHidden: FFN 隐藏层宽度，默认 4·dModel（LLaMA 惯例为 ⌊2/3·4D⌋，可显式传入）
     *   MoE 开关（默认稠密，nExp=1；nExp>1 即混合专家，现代顶级 LLM 核心稀疏架构）：
     *   nExp:    专家数量（每层 N 个独立 FFN + 1 个路由网络）
     *   topK:    top-k 门控（每个 token 仅激活 topK 个专家，默认 1）
     *   auxWeight: 负载均衡辅助损失权重（Switch/GDP-MoE，默认 0.01；非 MoE 为 0）
     *   稀疏注意力开关（Mistral/Gemma 谱系）：
     *   window: 滑动窗口。标量=所有层同窗口；数组=逐层窗口（长度须=nLayer，Gemma 2/3 局部/全局交错）；
     *           哨兵 {alt:N}=偶数层全因果(0)、奇数层窗口 N 的 1:1 交错。0 = 全因果，默认
     *   稳定化开关（Gemma 2 配方，默认全关 = 逐位兼容旧模型）：
     *   qkNorm:  true 时对每个头的 Q/K 向量做 RMSNorm（RoPE 前，增益 head_dim 维跨头共享）
     *   logitCap: 最终 logits 软帽 logits = cap·tanh(logits/cap)（Gemma 2 用 30；0 = 不启用）
     *   kvQuant: KV 缓存量化（8 = INT8，per-(token,KV头) 对称 absmax；0 = Float64，默认）
     */
    constructor(cfg) {
      if (cfg.dModel % cfg.nHead !== 0) {
        throw new Error(`dModel(${cfg.dModel}) 必须能被 nHead(${cfg.nHead}) 整除`);
      }
      this.cfg = Object.assign({ seed: 42, norm: 'ln', ffn: 'gelu', pos: 'learned' }, cfg);
      const { vocabSize: V, blockSize: T, nLayer: L, nHead: H, dModel: D } = this.cfg;
      if (this.cfg.pos === 'rope' && (D / H) % 2 !== 0) {
        throw new Error(`RoPE 要求每个 head 维度 dh=${D / H} 为偶数`);
      }
      // GQA（分组查询注意力）：KV 头数可小于 Q 头数（LLaMA-2/3 标准）。
      // 默认 nKvHead = nHead 即 MHA，完全向后兼容。
      const KV = cfg.nKvHead != null ? cfg.nKvHead : H;
      if (KV < 1 || KV > H || H % KV !== 0) {
        throw new Error(`nKvHead(${KV}) 需满足 1 <= nKvHead <= nHead(${H}) 且 nHead 可被整除`);
      }
      const dh = D / H;
      this.KV = KV; this.dh = dh; this.g = H / KV; this.qkvDim = D + 2 * KV * dh;
      this.useRope = this.cfg.pos === 'rope';
      this.useRms = this.cfg.norm === 'rmsnorm';
      this.useSwi = this.cfg.ffn === 'swiglu';
      this.fh = this.cfg.ffnHidden || 4 * D;
      // MoE（混合专家）：nExp>1 时每层 FFN 替换为 N 个专家 + top-k 路由网络；
      // 现代顶级 LLM（Mixtral/GPT-4/DeepSeek）的核心稀疏架构。nExp=1 即稠密（零回归）。
      this.nExp = cfg.nExp || 1;
      this.topK = Math.min(cfg.topK || 1, this.nExp);
      this.useMoE = this.nExp > 1;
      this.auxWeight = cfg.auxWeight != null ? cfg.auxWeight : (this.useMoE ? 0.01 : 0);
      // 滑动窗口注意力（SWA，Mistral/Gemma）：window>0 时查询 i 只 attend 键 [i-window+1, i]；
      // window=0（默认）即标准全因果，逐位兼容旧模型。三种写法归一化为逐层数组 this.windows：
      //   标量 s      -> 所有层同窗口 [s,s,...]
      //   数组 [a,b]  -> 逐层窗口（长度须 = nLayer，Gemma 2/3 局部/全局层交错）
      //   哨兵 {alt:N} -> 偶数层全因果(0)、奇数层窗口 N（Gemma 2 的 1:1 交错模式）
      const wRaw = cfg.window;
      let wArr;
      if (Array.isArray(wRaw)) {
        if (wRaw.length !== L) {
          throw new Error(`window 数组长度(${wRaw.length}) 需等于 nLayer(${L})`);
        }
        wArr = wRaw.map((v) => (v > 0 ? Math.floor(v) : 0));
      } else if (wRaw && typeof wRaw === 'object' && wRaw.alt > 0) {
        const w = Math.floor(wRaw.alt);
        wArr = [];
        for (let l = 0; l < L; l++) wArr.push(l % 2 === 1 ? w : 0); // 偶数层全局，奇数层局部
      } else {
        const w = wRaw > 0 ? Math.floor(wRaw) : 0;
        wArr = new Array(L).fill(w);
      }
      this.windows = wArr;   // 逐层窗口（权威字段，0 = 该层全因果/全局）
      this.window = Array.isArray(wRaw) ? wArr.slice()
        : (wRaw && typeof wRaw === 'object' ? { alt: Math.floor(wRaw.alt) } : wArr[0]); // 兼容读取：原样反映配置
      // Gemma 2 稳定化组件：QK-norm（RoPE 前逐头 RMSNorm）与 logit soft-cap。默认全关。
      this.qkNorm = !!cfg.qkNorm;
      this.logitCap = cfg.logitCap > 0 ? +cfg.logitCap : 0;
      // KV 缓存 INT8 量化：仅影响 forwardStep 增量路径（forward 全序列不走缓存）
      this.kvQuant = cfg.kvQuant === 8 ? 8 : 0;
      // Attention Sink（StreamingLLM）：恒驻缓存首 K 个锚点 token，支持无限上下文流式推理；0 = 关闭（逐位兼容）
      this.attentionSink = cfg.attentionSink > 0 ? Math.floor(cfg.attentionSink) : 0;
      // YaRN 长度外推（RoPE 频率插值 + 注意力温度缩放）：
      //   cfg.ropeScale = { type:'yarn'|'ntk', factor, originalMaxLength, betaFast, betaSlow }
      //   factor<=1 或未配置 -> 恒等变换（freqScale=null、attnScale=1，逐位兼容）。
      let ropeFreqScale = null, attentionScaling = 1;
      const rs = cfg.ropeScale;
      if (rs && (rs.factor || 0) > 1) {
        const factor = +rs.factor;
        const dim = dh;                       // 旋转维 = 头维度 dh（YaRN 公式的 rotary dim）
        const base = ROPE_BASE;
        const orig = rs.originalMaxLength || T || 2048;
        const half = dh >> 1;
        ropeFreqScale = new Float64Array(half);
        if (rs.type === 'ntk') {
          // 简单 NTK-aware：全局基频缩放 ntk = factor^(dim/(dim-2))
          const ntk = Math.pow(factor, dim / (dim - 2));
          for (let p = 0; p < half; p++) ropeFreqScale[p] = 1 / ntk;
        } else {
          // YaRN NTK-by-parts：高频维（小 p）保留、低频维（大 p）插值，平滑过渡（HF 同式）
          const betaFast = rs.betaFast || 32, betaSlow = rs.betaSlow || 1;
          const low = Math.max(0, Math.floor(dim * Math.log(orig / (2 * Math.PI * betaFast)) / (2 * Math.log(base))));
          const high = Math.min(half - 1, Math.ceil(dim * Math.log(orig / (2 * Math.PI * betaSlow)) / (2 * Math.log(base))));
          for (let p = 0; p < half; p++) {
            let ramp;
            if (p <= low) ramp = 0;
            else if (p >= high) ramp = 1;
            else { const r = (p - low) / Math.max(1, high - low); ramp = r * r * (3 - 2 * r); } // smoothstep
            const mask = 1 - ramp;                 // HF: inv_freq_mask = 1 - ramp
            ropeFreqScale[p] = Math.pow(factor, -mask); // inv_freq /= factor^mask（∈ [1/factor, 1]）
          }
        }
        // 注意力温度（YaRN mscale）：缩放 softmax 前 logits，补偿频率插值带来的幅度变化
        attentionScaling = factor > 1 ? 0.1 * Math.log(factor) / Math.log(2) : 1;
      }
      this.ropeFreqScale = ropeFreqScale;
      this.attentionScaling = attentionScaling;
      this.quantized = false;
      const rng = mulberry32(this.cfg.seed);
      const P = (this.params = {});
      const add = (name, r, c, scale) => {
        const p = { d: new Float64Array(r * c), r, c };
        const s = 0.02 * (scale || 1);
        for (let i = 0; i < p.d.length; i++) p.d[i] = randn(rng) * s;
        P[name] = p;
      };
      const zeros = (name, n) => { P[name] = { d: new Float64Array(n), r: 1, c: n }; };
      const ones = (name, n) => {
        const p = { d: new Float64Array(n), r: 1, c: n };
        p.d.fill(1);
        P[name] = p;
      };
      add('tokEmb', V, D);
      if (!this.useRope) add('posEmb', T, D); // RoPE 不需要位置嵌入
      const rscale = 1 / Math.sqrt(2 * L); // GPT-2：按残差深度缩放输出投影初始化
      const FH = this.fh;
      for (let l = 0; l < L; l++) {
        const k = `l${l}.`;
        add(k + 'wqkv', D, this.qkvDim); zeros(k + 'bqkv', this.qkvDim);
        add(k + 'wproj', D, D, rscale); zeros(k + 'bproj', D);
        if (this.useMoE) {
          // 路由网络：x → 各专家门控 logits（softmax over experts）
          add(k + 'wg', D, this.nExp); zeros(k + 'bg', this.nExp);
          // N 个专家 FFN（各自独立的 SwiGLU/GELU 两层），复用稠密 FFN 的同一宽度约定
          for (let e = 0; e < this.nExp; e++) {
            const ek = `${k}e${e}.`;
            add(ek + 'w1', D, this.useSwi ? 2 * FH : FH);
            zeros(ek + 'b1', this.useSwi ? 2 * FH : FH);
            add(ek + 'w2', FH, D, rscale); zeros(ek + 'b2', D);
          }
        } else {
          // SwiGLU 的 w1 一次产出 gate 与 up 两半（D → 2·FH），少一次矩阵乘
          add(k + 'w1', D, this.useSwi ? 2 * FH : FH);
          zeros(k + 'b1', this.useSwi ? 2 * FH : FH);
          add(k + 'w2', FH, D, rscale); zeros(k + 'b2', D);
        }
        ones(k + 'ln1.g', D); if (!this.useRms) zeros(k + 'ln1.b', D);
        ones(k + 'ln2.g', D); if (!this.useRms) zeros(k + 'ln2.b', D);
        if (this.qkNorm) { ones(k + 'qg', dh); ones(k + 'kg', dh); } // QK-norm 增益（head_dim 维跨头共享）
      }
      ones('lnf.g', D); if (!this.useRms) zeros('lnf.b', D);
      // 输出头与 tokEmb 权重共享（weight tying，GPT-1/2 标准做法）
      this.grads = {};
      for (const key in P) this.grads[key] = new Float64Array(P[key].d.length);
      this.zeroGrads();
    }

    zeroGrads() {
      for (const key in this.grads) this.grads[key].fill(0);
    }

    paramCount() {
      let n = 0;
      for (const key in this.params) {
        const p = this.params[key];
        n += p.q ? p.q.length : p.d.length;
      }
      return n;
    }

    /** 导出 INT8 权重量化推理副本（weight-only PTQ）。
     * 线性权重（wqkv/wproj/w1/w2/tokEmb/posEmb）转为 {q, scale} 常驻 Int8，
     * 偏置与归一化参数保持 Float64。原模型（训练态）不受影响。 */
    quantize() {
      const m = Object.create(GPTModel.prototype);
      m.cfg = this.cfg;
      m.useRms = this.useRms; m.useSwi = this.useSwi; m.useRope = this.useRope;
      m.KV = this.KV; m.g = this.g; m.fh = this.fh; m.dh = this.dh;
      m.qkvDim = this.qkvDim; m.quantized = true;
      m.nExp = this.nExp; m.topK = this.topK; m.useMoE = this.useMoE; m.auxWeight = this.auxWeight;
      m.window = this.window; m.windows = this.windows.slice();
      m.qkNorm = this.qkNorm; m.logitCap = this.logitCap; m.kvQuant = this.kvQuant; m.attentionSink = this.attentionSink;
      m.ropeFreqScale = this.ropeFreqScale; m.attentionScaling = this.attentionScaling;
      m.params = {};
      for (const key in this.params) {
        const p = this.params[key];
        if (p && p.d && isQuantLinear(key)) {
          const d = p.d;
          let maxAbs = 0;
          for (let i = 0; i < d.length; i++) { const a = Math.abs(d[i]); if (a > maxAbs) maxAbs = a; }
          const scale = maxAbs / 127 || 1;
          const q = new Int8Array(d.length);
          for (let i = 0; i < d.length; i++) {
            let v = Math.round(d[i] / scale);
            q[i] = v > 127 ? 127 : v < -127 ? -127 : v;
          }
          m.params[key] = { q, scale };
        } else {
          m.params[key] = { d: p.d ? p.d.slice() : p.d };
        }
      }
      return m;
    }

    /** 前向传播：委托 L3.blocks 重组装（T03，Δ=0） */
    forward(tokens) { return NS.L3.blocks.forward(this, tokens); }

    /** 反向传播：委托 L3.blocks 重组装（T03，Δ=0） */
    backward(cache, dlogits) { return NS.L3.blocks.backward(this, cache, dlogits); }
  }

  NS.L3.gpt_model.GPTModel = GPTModel;
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

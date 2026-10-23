/* L3 架构 · 架构预设（ARCH_PRESETS / applyPreset）
 * 所属层：L3（架构组装）｜ 允许依赖：（无跨层依赖，自包含）
 * 迁移方式：从 legacy flagship-engine.v11.js 第 1685–1701 行【原样搬】。
 *   纯数据 + 纯函数，逐位等价；applyPreset 喂给 GPTModel 构造器（RNG 顺序契约的上游）。
 */
;(function (NS) {
  NS.L3 = NS.L3 || {};
  NS.L3.presets = NS.L3.presets || {};

  /* 架构预设：一行切换到现代 LLM 的标准配方 */
  const ARCH_PRESETS = {
    gpt2: { norm: 'ln', ffn: 'gelu', pos: 'learned' },                 // GPT-2 / 本项目默认
    llama: { norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope' },            // LLaMA / Mistral（MHA）
    'llama-gqa': { norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope', nKvHead: 2 }, // LLaMA + GQA 演示
    // Mixtral 风格：LLaMA 配方 + GQA + MoE（4 专家 top-2 路由）。nExp/topK 可在 cfg 覆盖。
    mixtral: { norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope', nKvHead: 2, nExp: 4, topK: 2, auxWeight: 0.01 },
    // 滑动窗口注意力（SWA）：LLaMA 配方 + GQA + window=64（Mistral/Gemma 稀疏注意力）。
    swa: { norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope', nKvHead: 2, window: 64 },
    // Gemma 2/3 完整配方：LLaMA 基座 + GQA + 局部/全局层交错 + QK-norm + logit soft-cap 30。
    gemma: { norm: 'rmsnorm', ffn: 'swiglu', pos: 'rope', nKvHead: 2, window: { alt: 64 }, qkNorm: true, logitCap: 30 },
    // BLOOM / MPT 风格：ALiBi 取代一切位置编码（无 RoPE、无可学习位置嵌入）+ LayerNorm + GELU。
    //   alibi:true 时 GPTModel 强制 useRope=false 且不创建 posEmb —— 这是 BLOOM/MPT 的发表设计：
    //   位置信息完全由「注意力打分上的线性距离惩罚」提供，反而获得更强的长度外推能力。
    bloom: { norm: 'ln', ffn: 'gelu', alibi: true },
  };
  /** 把预设叠到配置上：applyPreset(cfg, 'llama') */
  function applyPreset(cfg, preset) {
    const p = ARCH_PRESETS[preset];
    if (!p) throw new Error(`未知架构预设：${preset}（可用：${Object.keys(ARCH_PRESETS).join(', ')}）`);
    return Object.assign({}, cfg, p);
  }

  NS.L3.presets.ARCH_PRESETS = ARCH_PRESETS;
  NS.L3.presets.applyPreset = applyPreset;
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

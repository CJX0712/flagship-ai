/* ============================================================================
 * L5 自动微分 · Tape 契约（cache 结构约定）
 * ----------------------------------------------------------------------------
 * 所属层：L5（自动微分）｜ 允许依赖：无（纯结构契约，不做任何数值计算）
 * 主要 @ref：Baydin et al. 2018（tape = 前向中间量的记录载体，反向按图取用）
 *
 * 背景：v11 legacy 的 forward 返回 `cache`、backward 消费同一 `cache`
 * （flagship-engine.v11.js 第 1020/1032 行构造，第 1036 行起消费）。
 * L3 blocks.js 重组装后该布局逐字段冻结（R2/R3 Δ=0 的载体），本模块把它
 * 固化为契约文档 + 结构工厂，供 P1 的 IR 以完全相同的形状记录前向中间量
 * （届时 L5 才拥有计算路径；P0 仅契约，零数值风险）。
 *
 * 冻结布局（不得更改）：
 *   cache = { T, dh, H, D, L, layers[], lnf, Hf, toks, logits }
 *   layers[l] = { ln1, A, qkv, Ps, ctx, xMid, ln2, M, pre1, h1, moe, qk }
 * 路径差异（均为合法形状，允许为 null）：
 *   - 稠密 FFN：pre1/h1 为数组，moe 为 null；
 *   - MoE 路由：pre1/h1 为 null，moe 为 moeForward 的路由/专家缓存对象；
 *   - qkNorm 关闭：qk 为 null；
 *   - ln1/ln2/lnf 恒为 { Y, xhat, rstd }（Y 即 A / M / Hf 的同一引用）。
 * ==========================================================================*/
;(function (NS) {
  'use strict';
  NS.L5 = NS.L5 || {};
  NS.L5.tape = NS.L5.tape || {};

  // 每层 tape 的键序（与 v11 cache.layers[l] 构造顺序逐字一致）
  const LAYER_KEYS = ['ln1', 'A', 'qkv', 'Ps', 'ctx', 'xMid', 'ln2', 'M', 'pre1', 'h1', 'moe', 'qk'];
  // 顶层 cache 的键序（与 v11 forward 返回顺序逐字一致）
  const CACHE_KEYS = ['T', 'dh', 'H', 'D', 'L', 'layers', 'lnf', 'Hf', 'toks', 'logits'];

  /** 契约描述（供文档/自检）：返回布局的键序 + 每字段语义说明。 */
  function spec() {
    return {
      frozen: '与 v11 legacy forward 返回的 cache 逐字段一致；变更需走零回归评审',
      cache: {
        keys: CACHE_KEYS.slice(),
        desc: {
          T: 'token 数（≤ blockSize；末段不足则截取）',
          dh: '每头维度 dh = D / H',
          H: 'Q 头数（GQA 下 KV 头数 = H / g）',
          D: '模型宽度 d_model',
          L: '层数（layers.length === L）',
          layers: '每层 tape 数组，顺序 = 前向层序（反向自 L-1 向 0 消费）',
          lnf: '最终归一化缓存 { Y, xhat, rstd }',
          Hf: 'lnf.Y（T×D，权重共享输出头的输入）',
          toks: '本批 token id（number[]，供嵌入反向散列累加）',
          logits: '输出 logits（T×V；logitCap>0 时为 soft-cap 后的值，反向据此求 sech²）',
        },
      },
      layer: {
        keys: LAYER_KEYS.slice(),
        desc: {
          ln1: 'pre-LN #1 缓存 { Y, xhat, rstd }（Y 与 A 同一引用）',
          A: 'ln1.Y（T×D，wqkv 投影的输入，权重梯度 matmulTA 的被乘数）',
          qkv: 'QKV 投影（T×qkvDim，已就地经过 QK-norm 与 RoPE 的值）',
          Ps: '各头 softmax 概率数组（长度 H，每项 T×T；反向经 softmaxBwd）',
          ctx: '注意力上下文（T×D，wproj 投影的输入）',
          xMid: '注意力残差输出 xMid = X + attn（T×D，ln2 的输入）',
          ln2: 'pre-LN #2 缓存 { Y, xhat, rstd }（Y 与 M 同一引用）',
          M: 'ln2.Y（T×D，FFN 的输入）',
          pre1: 'FFN 第一投影（稠密路径 T×W1；MoE 路径为 null）',
          h1: 'FFN 激活输出（稠密路径 T×FH；MoE 路径为 null）',
          moe: 'MoE 路由/专家缓存（moeForward 返回的 moe 对象；稠密路径为 null）',
          qk: 'QK-norm 缓存 { qx, qr, kx, kr }（qkNorm 关闭时为 null）',
        },
      },
    };
  }

  /** 空层 tape 工厂：键序与 spec().layer.keys 一致，值全部置 null。 */
  function emptyLayer() {
    const o = {};
    for (const k of LAYER_KEYS) o[k] = null;
    return o;
  }

  /** 空 cache 工厂：键序与 spec().cache.keys 一致（layers 为 []）。 */
  function emptyCache() {
    return {
      T: 0, dh: 0, H: 0, D: 0, L: 0,
      layers: [], lnf: null, Hf: null, toks: null, logits: null,
    };
  }

  /**
   * 结构自检：返回缺失键名数组（空数组 = 合规）。
   * 校验顶层键 + 每层键；layers 长度与 L 不一致也计入。
   */
  function validateCache(cache) {
    const missing = [];
    if (!cache || typeof cache !== 'object') return ['cache'];
    for (const k of CACHE_KEYS) if (!(k in cache)) missing.push(k);
    if (Array.isArray(cache.layers)) {
      for (let l = 0; l < cache.layers.length; l++) {
        const ly = cache.layers[l];
        if (!ly || typeof ly !== 'object') { missing.push(`layers[${l}]`); continue; }
        for (const k of LAYER_KEYS) if (!(k in ly)) missing.push(`layers[${l}].${k}`);
      }
      if (typeof cache.L === 'number' && cache.layers.length !== cache.L) {
        missing.push(`layers.length(${cache.layers.length})!==L(${cache.L})`);
      }
    }
    return missing;
  }

  NS.L5.tape.spec = spec;
  NS.L5.tape.emptyLayer = emptyLayer;
  NS.L5.tape.emptyCache = emptyCache;
  NS.L5.tape.validateCache = validateCache;
  NS.L5.tape.LAYER_KEYS = LAYER_KEYS.slice();
  NS.L5.tape.CACHE_KEYS = CACHE_KEYS.slice();
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

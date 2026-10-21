/* L0 装配层 · 公共 API 组装（assemble）
 * ============================================================================
 * 所属层：L0（装配层）
 * 允许依赖：全部层（L10 → … → L2）。本层位于 __loadOrder 最末，装配动作发生在
 *   所有层求值完成之后（由调用方显式调用 assemble(ns)），故文件求值期不做任何取值。
 * 迁移方式：从 T04 版 src/index.js 第 34–96 行【原样搬】（Δ=0 契约）：
 *   - 34 个键的「键名 → 来源」映射逐字搬运（顺序亦保持），来源全部指向 NS.Ln.x 真十层符号；
 *   - K15 装配期硬门禁（EXPORTED_KEYS 逐名一致 + 无 undefined）搬入本层，
 *     Node 与浏览器同享，两侧键集由构造保证一致，而非靠约定；
 *   - 注册中心自举（L4.registry / L5.registry / L6.kernel / L6.backend）搬入并保留幂等守卫。
 * 为什么单独成层（关键动因）：Node（src/index.js）与浏览器（scripts/build-demo.js 内联）
 *   必须走**同一份装配逻辑**；若两端各写一遍映射，键集迟早漂移并击穿 K15。
 *   抽出为 L0 唯一真源后，「两侧键集逐名一致」由构造保证。
 *
 * T05 收口：本文件是 L0 装配层的唯一实现（原 src/L0-assembly/bootstrap.js 已删除——
 *   它只有 34 键映射，缺 K15 断言、注册中心自举与 NS.L0.API 挂载，功能不对等）。
 *   目录名 L0-boot 与既有 L2-pipeline / L3-arch / L4-ops 风格一致。
 * ========================================================================*/
;(function (NS) {
  NS.L0 = NS.L0 || {};

  /** 与 v11 return 清单逐名对应（K15 契约；任一侧增删键都必须红）。
   *  本常量是 34 键契约的唯一真源：Node（src/index.js）与浏览器
   *  （scripts/build-demo.js 内联）经 assemble() 共享同一份逐名门禁。
   *  必须保持已排序状态：逐名比对依赖 gotKeys 与 wantKeys 同序。
   */
  const EXPORTED_KEYS = [
    'mulberry32', 'randn',
    'matmul', 'matmulAtB', 'matmulTA',
    'layernormFwd', 'layernormBwd',
    'rmsnormFwd', 'rmsnormBwd',
    'geluFwd', 'geluBwd', 'siluF', 'swigluFwd', 'swigluBwd',
    'ropeApply', 'softmaxRows', 'softmaxBwd',
    'crossEntropy',
    'Tokenizer', 'BPETokenizer', 'GPTModel', 'train', 'generate',
    'initKVCache', 'forwardStep', 'warmupKV', 'kvShift', 'saveModel', 'loadModel', 'loadTokenizer',
    'ARCH_PRESETS', 'applyPreset', 'textToByteChars', 'byteCharsToText',
  ].sort();

  /** 把已装载的命名空间组装为 34 键公共 API。
   *  @param {{L9:Object,L8:Object,L7:Object,L6:Object,L5:Object,L4:Object,L3:Object,L2:Object}} ns
   *         已按 __loadOrder 装载完毕的命名空间（通常就是 NS 自身）。
   *  @returns {Object} 34 键公共 API（同时挂到 ns.L0.API 供浏览器侧取用）。
   */
  function assemble(ns) {
    // 1) 构建公共 API：34 个键全部由十层实现提供（K15：与 v11 return 完全一致）。
    const api = {
      // L9 · 数值基础（RNG）
      mulberry32: ns.L9.rng.mulberry32,
      randn: ns.L9.rng.randn,
      // L8 · 数值内核（矩阵乘）
      matmul: ns.L8.matmul.matmul,
      matmulAtB: ns.L8.matmul.matmulAtB,
      matmulTA: ns.L8.matmul.matmulTA,
      // L4 · 神经算子（归一化）
      layernormFwd: ns.L4.norm.layernormFwd,
      layernormBwd: ns.L4.norm.layernormBwd,
      rmsnormFwd: ns.L4.norm.rmsnormFwd,
      rmsnormBwd: ns.L4.norm.rmsnormBwd,
      // L4 · 神经算子（激活 / 门控）
      geluFwd: ns.L4.activation.geluFwd,
      geluBwd: ns.L4.activation.geluBwd,
      siluF: ns.L4.activation.siluF,
      swigluFwd: ns.L4.activation.swigluFwd,
      swigluBwd: ns.L4.activation.swigluBwd,
      // L4 · 神经算子（位置编码）
      ropeApply: ns.L4.positional.ropeApply,
      // L8 · 数值内核（softmax：模型热路径走 L8 内核，L4.softmax 为其语义门面）
      softmaxRows: ns.L8.reduce.softmaxRows,
      softmaxBwd: ns.L8.reduce.softmaxBwd,
      // L4 · 神经算子（损失）
      crossEntropy: ns.L4.loss.crossEntropy,
      // L2 · 编排管线（分词器）
      Tokenizer: ns.L2.tokenizer.Tokenizer,
      BPETokenizer: ns.L2.tokenizer.BPETokenizer,
      textToByteChars: ns.L2.tokenizer.textToByteChars,
      byteCharsToText: ns.L2.tokenizer.byteCharsToText,
      // L3 · 架构组装（模型壳 + 架构预设）
      GPTModel: ns.L3.gpt_model.GPTModel,
      ARCH_PRESETS: ns.L3.presets.ARCH_PRESETS,
      applyPreset: ns.L3.presets.applyPreset,
      // L2 · 编排管线（训练 / 生成）
      train: ns.L2.train.train,
      generate: ns.L2.generate.generate,
      // L7 · 张量与内存（KV Cache）
      initKVCache: ns.L7.kv_cache.initKVCache,
      forwardStep: ns.L7.kv_cache.forwardStep,
      warmupKV: ns.L7.kv_cache.warmupKV,
      kvShift: ns.L7.kv_cache.kvShift,
      // L2 · 编排管线（持久化）
      saveModel: ns.L2.persistence.saveModel,
      loadModel: ns.L2.persistence.loadModel,
      loadTokenizer: ns.L2.persistence.loadTokenizer,
    };

    // 2) K15 装配期硬门禁：**逐名**比对装配结果与契约键清单。
    //    只查数量是防不住漂移的（数量对但名字错、键名被拼错都必须红），
    //    故此处按名字逐个对齐；Node 与浏览器经本函数共享同一套门禁，
    //    两侧键集由构造保证一致，而非靠约定。
    const gotKeys = Object.keys(api).sort();
    if (gotKeys.length !== EXPORTED_KEYS.length
        || gotKeys.some((k, i) => k !== EXPORTED_KEYS[i])) {
      const missing = EXPORTED_KEYS.filter((k) => !(k in api));
      const extra = gotKeys.filter((k) => EXPORTED_KEYS.indexOf(k) < 0);
      throw new Error(
        'K15 违例：公共 API 导出键与契约不一致（缺失 [' + missing.join(', ') +
        ']；多余 [' + extra.join(', ') + ']）'
      );
    }
    //    逐名比对只保证键名集合一致，不保证取值存在，故仍须逐个查 undefined
    //    （键名对、数量为 34，但来源是 undefined 同样必须红）。
    for (let i = 0; i < gotKeys.length; i++) {
      if (api[gotKeys[i]] === undefined) {
        throw new Error('K15 违例：公共 API 导出键 ' + gotKeys[i] + ' 为 undefined');
      }
    }

    // 3) 各层注册中心自举（仅登记元数据供自检/文档，不改数值路径，Δ=0 无影响）。
    //    L4 算子目录：把 L4 算子登记为可检索元数据。
    if (ns.L4 && ns.L4.registry && ns.L4.registry.bootstrap) ns.L4.registry.bootstrap();
    //    L5 VJP 注册中心 + L6 kernel 注册中心：把 L4 backward / L4+L8 内核登记为
    //    VJP 与 kernel（L5/L6 先于 L4 加载，故在全部层加载完后补登；幂等、惰性守卫）。
    if (ns.L5 && ns.L5.registry && ns.L5.registry.bootstrap) ns.L5.registry.bootstrap();
    if (ns.L6 && ns.L6.kernel && ns.L6.kernel.bootstrap) ns.L6.kernel.bootstrap();
    if (ns.L6 && ns.L6.backend && ns.L6.backend.refreshBackends) ns.L6.backend.refreshBackends();

    // 4) 装配产物挂回命名空间（浏览器侧经 NS.L0.API 取用；Node 侧直接用返回值）。
    ns.L0.API = api;
    return api;
  }

  NS.L0.assemble = assemble;
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

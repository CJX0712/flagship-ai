/* 旗舰 AI v12 · 命名空间与加载顺序真源
 * 规则（共享知识 K3 / K4）：
 *   - 所有层文件为 IIFE 注入 globalThis.FlagshipAI 命名空间
 *   - Node 与浏览器共用同一份 __loadOrder（src/index.js 与 scripts/build-demo.js 共读）
 *   - 加载顺序自底向上：L10 → L9 → … → L1
 * Phase 0：__loadOrder 为空（尚未拆层），buildPublicAPI 暂未启用。
 */
'use strict';
const NS = (globalThis.FlagshipAI = globalThis.FlagshipAI || {});

// 自底向上加载顺序。Phase 1+ 将逐步追加各层文件，例如：
//   './L10-microisa/registry.js', './L9-numbase/rng.js', './L8-kernel/matmul.js',
//   './L7-tensor/tensor.js', './L6-backend/kernel-registry.js', './L5-autograd/vjp-registry.js',
//   './L4-ops/registry.js', './L3-arch/gpt-model.js', './L2-pipeline/train.js', './L1-app/app.js'
NS.__loadOrder = [];

// 公共 API 构建（Phase 0 暂未启用；后续接管后由 src/index.js 调用）。
// 必须导出与 v11 完全一致的导出清单（CI 断言 Object.keys 一致）。
NS.buildPublicAPI = function buildPublicAPI() {
  return NS;
};

module.exports = NS;

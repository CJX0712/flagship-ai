/* 旗舰 AI v12 · 命名空间与加载顺序真源
 * 规则（共享知识 K3 / K4）：
 *   - 所有层文件为 IIFE 注入 globalThis.FlagshipAI 命名空间
 *   - Node 与浏览器共用同一份 __loadOrder（src/index.js 与 scripts/build-demo.js 共读）
 *   - 加载顺序自底向上：L10 → L9 → … → L1
 * Phase 0：__loadOrder 为空（尚未拆层），buildPublicAPI 暂未启用。
 */
'use strict';
const NS = (globalThis.FlagshipAI = globalThis.FlagshipAI || {});

// 自底向上加载顺序（K4）：L10 → L9 → … → L1。
// Phase 1（T02）已落地：L10 微指令登记中心 + A/C 两域 192 项、L9 数值基础、L8 数值内核。
// 后续 Phase 将逐步追加 L7/L6/L5/L4/L3/L2/L1。
// 加载顺序（K4）自底向上：L10 骨架 → L9 数值基础 → L8 数值内核 → L10 指令实现。
// 注意：A/C 两域微指令在模块求值期捕获 NS.L8.matmul/reduce 与 NS.L9.special
// （见各 A/C 文件顶部的 const 捕获），故 L8/L9 必须在其之前加载。L10 的
// registry/invariant 仅搭建骨架（不依赖 L8/L9 求值），故置于最前。
NS.__loadOrder = [
  // L10 · 微指令登记中心（最底层骨架：registry + invariant）
  './L10-microisa/registry.js',
  './L10-microisa/invariant.js',
  // L9 · 数值基础（A/C 域微指令的标量/函数依赖）
  './L9-numbase/rng.js',
  './L9-numbase/fp-ops.js',
  './L9-numbase/special.js',
  './L9-numbase/assert.js',
  // L8 · 数值内核（A/C 域微指令的矩阵/归约依赖）
  './L8-kernel/matmul.js',
  './L8-kernel/reduce.js',
  './L8-kernel/quantize.js',
  // L10 · 微指令实现（A/C 两域，依赖上方 L8/L9，故置于其后）
  './L10-microisa/A-scalar-000.js',
  './L10-microisa/A-scalar-001.js',
  './L10-microisa/A-scalar-002.js',
  './L10-microisa/C-matrix-000.js',
  './L10-microisa/C-matrix-001.js',
  './L10-microisa/C-matrix-002.js',
];

// 公共 API 构建（Phase 0 暂未启用；后续接管后由 src/index.js 调用）。
// 必须导出与 v11 完全一致的导出清单（CI 断言 Object.keys 一致）。
NS.buildPublicAPI = function buildPublicAPI() {
  return NS;
};

module.exports = NS;

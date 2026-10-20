/* 旗舰 AI · 命名空间与加载顺序真源
 * 规则（共享知识 K3 / K4）：
 *   - 所有层文件为 IIFE 注入 globalThis.FlagshipAI 命名空间
 *   - Node 与浏览器共用同一份 __loadOrder（src/index.js 与 scripts/build-demo.js 共读）
 *   - 加载顺序自底向上：L10 → L9 → … → L1
 * Phase 1：__loadOrder 落地 L10（八域）+ L9（数值基础）首批；legacy 引擎仍为可运行模型。
 */
'use strict';
const NS = (globalThis.FlagshipAI = globalThis.FlagshipAI || {});

// 自底向上加载顺序（L10 最底 → L9）。Node 由 src/index.js 以 fs+vm 执行；
// 浏览器由 scripts/build-demo.js 按同序内联进 demo.html（经典脚本 IIFE 注入）。
// 后续 Phase 逐层追加：L8-kernel / L7-tensor / L6-backend / L5-autograd /
//   L4-ops / L3-arch / L2-pipeline / L1-app
NS.__loadOrder = [
  './L10-microisa/registry.js',
  './L10-microisa/A-scalar.js',
  './L10-microisa/B-vector.js',
  './L10-microisa/C-matrix.js',
  './L10-microisa/D-decomp.js',
  './L10-microisa/E-shape.js',
  './L10-microisa/F-autodiff.js',
  './L10-microisa/G-attention.js',
  './L10-microisa/H-optimizer.js',
  './L9-numbase/rng.js',
  './L9-numbase/fp-ops.js',
  './L9-numbase/assert.js'
];

// 公共 API 构建：把引擎 API 合并进命名空间（Node 与浏览器平权，
// 使 globalThis.FlagshipAI 在双端都含「引擎 API + 层命名空间」）。
NS.buildPublicAPI = function buildPublicAPI(engine) {
  if (engine) Object.assign(NS, engine);
  return NS;
};

// 浏览器兼容：经典脚本无 module（本文件不在 src/L* 层目录，不受 lint 红线约束）
if (typeof module !== 'undefined' && module.exports) module.exports = NS;

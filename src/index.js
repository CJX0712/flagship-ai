/* 旗舰 AI · 公共 API 入口（Phase 1）
 * 加载顺序（自底向上）：L10 → L9 → …（见 src/namespace.js __loadOrder）。
 * 层文件是注入 globalThis.FlagshipAI 的 IIFE（浏览器 file:// 友好，禁 require），
 * 故 Node 侧用 fs+vm 按 __loadOrder 执行；浏览器侧由 scripts/build-demo.js
 * 按同一 __loadOrder 内联进 demo.html（引擎 UMD 先行，namespace 以 || 保留）。
 *
 * 导出保持 v11 引擎本体（R3 红线：同名同签名，零破坏性变更）；
 * 层命名空间（L10/L9/…）挂 globalThis.FlagshipAI，经 NS.buildPublicAPI 与引擎合并。
 */
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const NS = require('./namespace.js');
for (const rel of NS.__loadOrder) {
  const file = path.join(__dirname, rel);
  if (!fs.existsSync(file)) throw new Error(`层文件缺失: ${rel}（__loadOrder 与磁盘不一致）`);
  vm.runInThisContext(fs.readFileSync(file, 'utf8'), { filename: file });
}
const engine = require('./flagship-engine.js');
NS.buildPublicAPI(engine);
module.exports = engine;

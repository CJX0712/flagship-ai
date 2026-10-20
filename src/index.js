/* 旗舰 AI v12 · 公共 API 入口
 * 加载顺序（自底向上）：L10 → L9 → … → L1（见 src/namespace.js __loadOrder）。
 * Phase 0：尚未拆任何一层，直接转发 v11 引擎（经 src/flagship-engine.js shim），
 *          保证 R3 红线：导出清单与 v11 同名同签名。
 *
 * 后续 Phase 接管后改为：
 *   const NS = require('./namespace.js');
 *   for (const f of NS.__loadOrder) require(f);   // 自底向上加载十层
 *   module.exports = NS.buildPublicAPI();
 */
'use strict';
module.exports = require('./flagship-engine.js');

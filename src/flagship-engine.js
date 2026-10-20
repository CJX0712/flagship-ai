/* 旗舰 AI v12 入口转发层（Phase 0）
 * v12 十层重构期间，本文件转发 v11 引擎，保证对外 API 兼容（R3 红线：导出清单同名同签名）。
 * 后续 Phase 逐层接管后，本文件将被移除，由 src/index.js 直接组装十层。
 */
'use strict';
module.exports = require('./legacy/flagship-engine.v11.js');

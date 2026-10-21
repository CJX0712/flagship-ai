#!/usr/bin/env node
/* ============================================================================
 * 构建脚本：把十层实现内联进 demo.template.html，产出零依赖单文件 demo.html
 * （file:// 直接打开即可运行），并导出 data/corpus-zh.txt。
 * 用法：node scripts/build-demo.js
 *
 * T05 重写说明（修 P0：单文件 demo 此前是坏的）：
 *   旧实现内联 `src/flagship-engine.js` —— 那是 Phase 0 的 CommonJS 转发 shim
 *   （内容为 `module.exports = require('./legacy/...')`）。浏览器里 `module` 未定义，
 *   第一个 <script> 即抛 `module is not defined`，整个 demo 不可用。
 *   现在改为：按 namespace.js 的 __loadOrder 真源内联全部层文件（Node 与浏览器
 *   共用同一加载顺序），再由 L0 装配层把十层装配成 34 键 API —— 与 src/index.js
 *   走同一个 NS.L0.assemble(NS)，两端不可能漂移。
 * ==========================================================================*/
'use strict';
const fs = require('fs');
const path = require('path');

const root = path.join(__dirname, '..');
const read = (p) => fs.readFileSync(path.join(root, p), 'utf8');

// 1) 按 __loadOrder 真源内联全部层文件（L10 → L9 → … → L1 → L0 装配层）
const NS = require('../src/namespace.js');
const layerPaths = NS.__loadOrder.map((rel) => rel.replace(/^\.\//, 'src/'));
for (const rel of layerPaths) {
  if (!fs.existsSync(path.join(root, rel))) {
    console.error(`❌ __loadOrder 中的层文件不存在：${rel}`);
    process.exit(1);
  }
}
const layersCode = layerPaths.map(read).join('\n');

// 2) 浏览器引导：调 L0 装配层，把 34 键并入 globalThis.FlagshipAI 命名空间。
//    并入而非覆盖 —— 层文件用同一个命名空间对象存 NS.Ln，覆盖会破坏后续自检/调试访问。
const browserBootstrap = `
/* L0 装配层引导（浏览器）：把十层装配为 34 键公共 API，并入 FlagshipAI 命名空间。
   与 Node 端 src/index.js 共用 NS.L0.assemble(NS)，是两端唯一的 34 键映射真源，
   并共享同一套 EXPORTED_KEYS 逐名硬门禁（K15），两侧不可能漂移。 */
;(function () {
  var NS = globalThis.FlagshipAI;
  var api = NS.L0.assemble(NS);
  for (var k in api) NS[k] = api[k];
})();
`;

// 3) 各槽位内容。注意：L1 应用层（src/L1-app/demo-app.js）已在 __loadOrder 里，
//    因此随 layersCode 一并内联到引擎段；__FLAGSHIP_APP__ 槽位只剩「启动调用」。
//    旧做法内联 src/app.js（它直接读 window.FlagshipAI，绕过了十层装配层）。
const appStartup = `
/* L1 应用层启动（浏览器）：把演示应用绑到 DOM。
   demo-app.js 只在求值期注册 NS.L1.app，不碰 DOM，故必须在此显式启动。 */
;(function () {
  var NS = globalThis.FlagshipAI;
  if (!NS.L1 || !NS.L1.app) throw new Error('L1 应用层未加载：demo-app.js 缺失');
  NS.L1.app.start(NS);
})();
`;

const parts = {
  '__FLAGSHIP_STACK__': layersCode + '\n' + browserBootstrap,
  '__FLAGSHIP_VERIFY__': read('src/verify.js'),
  '__FLAGSHIP_CORPUS__': read('src/corpus.js'),
  '__FLAGSHIP_APP__': appStartup,
};

/** 把各段内联进模板，返回完整 HTML（供 CLI 与测试共用，避免两份内联逻辑漂移）。 */
function buildHtml() {
  let html = read('demo.template.html');
  for (const [marker, code] of Object.entries(parts)) {
    // 用函数替换器，避免代码中的 $ 序列被解释为正则替换模式
    const token = `/*${marker}*/`;
    if (!html.includes(token)) {
      throw new Error(`模板中缺少占位标记 ${token}`);
    }
    html = html.replace(token, () => code);
  }
  return html;
}

// CLI 模式（node scripts/build-demo.js）
if (require.main !== module) {
  module.exports = { buildHtml, parts, layerPaths };
} else {
  main();
}

function main() {
let html;
try {
  html = buildHtml();
} catch (e) {
  console.error(`❌ ${e.message}`);
  process.exit(1);
}

const out = path.join(root, 'demo.html');
fs.writeFileSync(out, html);
fs.writeFileSync(path.join(root, 'data', 'corpus-zh.txt'), require('../src/corpus.js').CORPUS);

// 3) 构建期硬校验：无条件执行的代码里不得残留 CommonJS 语法
//    （那正是此前 demo 坏掉的根因：`module.exports = require(...)` 在浏览器里抛
//     `module is not defined`，且发生在第一个 <script>，整页脚本直接停摆）。
//    判据：层文件与引导段为纯 IIFE，一见 CJS 即红；verify/corpus/app 是 UMD，
//    其 CJS 分支必须包在 `typeof module !== 'undefined'` 守卫内，否则同样红。
const CJS = /^\s*module\.exports\s*=|\brequire\s*\(/m;
const engineCode = layersCode + '\n' + browserBootstrap;
if (CJS.test(engineCode)) {
  console.error('❌ 层文件/引导段残留 CommonJS 语法（浏览器无法执行）');
  process.exit(1);
}
for (const name of ['src/verify.js', 'src/corpus.js']) {
  const code = read(name);
  if (CJS.test(code) && !/typeof\s+module\s*!==\s*['"]undefined['"]/.test(code)) {
    console.error(`❌ ${name} 存在未被守卫的 CommonJS 语法（浏览器会抛 module is not defined）`);
    process.exit(1);
  }
}

const kb = (Buffer.byteLength(html) / 1024).toFixed(1);
console.log(`✅ demo.html 构建完成（${kb} KB，零外部依赖，file:// 直开）`);
console.log(`✅ 内联层文件 ${layerPaths.length} 个（__loadOrder 真源顺序）`);
console.log(`✅ CommonJS 残留检查：0 处`);
console.log(`✅ data/corpus-zh.txt 已导出`);
}

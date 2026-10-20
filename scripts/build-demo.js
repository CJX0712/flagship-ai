#!/usr/bin/env node
/* ============================================================================
 * 构建脚本：将 src/ 下各模块内联进 demo.template.html，产出零依赖单文件
 * demo.html（file:// 直接打开即可运行），并导出 data/corpus-zh.txt。
 * 用法：node scripts/build-demo.js
 * ========================================================================*/
'use strict';
const fs = require('fs');
const path = require('path');

const root = path.join(__dirname, '..');
const read = (p) => fs.readFileSync(path.join(root, p), 'utf8');
const NS = require('../src/namespace.js');

// 浏览器无 module/require：必须内联真实引擎源码。若 src/flagship-engine.js 是
// Node shim（转发 src/legacy/），则内联其转发的 legacy 引擎本体（UMD，浏览器
// 走 root.FlagshipAI 分支）。Phase 0 曾误内联 shim 导致 demo 损坏，此处硬校验。
const engineShim = read('src/flagship-engine.js');
const engineSrc = /require\s*\(/.test(engineShim) ? read('src/legacy/flagship-engine.v11.js') : engineShim;
if (!/GPTModel/.test(engineSrc)) {
  console.error('引擎内联失败：未找到 GPTModel（疑似内联了 Node shim）');
  process.exit(1);
}

// 层文件按 __loadOrder 拼接（经典脚本 IIFE，注入 globalThis.FlagshipAI）。
// 顺序：引擎 UMD 先行（设置 window.FlagshipAI）→ namespace 以 || 保留 → 层注入。
const layersSrc = NS.__loadOrder
  .map((rel, i) => `/* [layer ${i + 1}/${NS.__loadOrder.length}] src/${rel} */\n${read(path.join('src', rel))}`)
  .join('\n');

const parts = {
  '__FLAGSHIP_ENGINE__': engineSrc,
  '__FLAGSHIP_NAMESPACE__': read('src/namespace.js'),
  '__FLAGSHIP_LAYERS__': layersSrc,
  '__FLAGSHIP_VERIFY__': read('src/verify.js'),
  '__FLAGSHIP_CORPUS__': read('src/corpus.js'),
  '__FLAGSHIP_APP__': read('src/app.js'),
};

let html = read('demo.template.html');
for (const [marker, code] of Object.entries(parts)) {
  // 用函数替换器，避免代码中的 $ 序列被解释为正则替换模式
  const token = `/*${marker}*/`;
  if (!html.includes(token)) {
    console.error(`模板中缺少占位标记 ${token}`);
    process.exit(1);
  }
  html = html.replace(token, () => code);
}

const out = path.join(root, 'demo.html');
// 统一行尾为 \n：Windows 检出（core.autocrlf）会带入 CRLF，统一后任意平台重建
// 产物逐字节一致（可复现门禁：git archive 干净树重建 == 仓库内产物）
fs.writeFileSync(out, html.replace(/\r\n/g, '\n'));
fs.writeFileSync(path.join(root, 'data', 'corpus-zh.txt'), require('../src/corpus.js').CORPUS);

const kb = (Buffer.byteLength(html) / 1024).toFixed(1);
console.log(`✅ demo.html 构建完成（${kb} KB，零外部依赖，file:// 直开）`);
console.log(`✅ data/corpus-zh.txt 已导出`);

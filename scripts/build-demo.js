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

const parts = {
  '__FLAGSHIP_ENGINE__': read('src/flagship-engine.js'),
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
fs.writeFileSync(out, html);
fs.writeFileSync(path.join(root, 'data', 'corpus-zh.txt'), require('../src/corpus.js').CORPUS);

const kb = (Buffer.byteLength(html) / 1024).toFixed(1);
console.log(`✅ demo.html 构建完成（${kb} KB，零外部依赖，file:// 直开）`);
console.log(`✅ data/corpus-zh.txt 已导出`);

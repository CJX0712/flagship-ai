#!/usr/bin/env node
'use strict';
/* 旗舰 AI v12 · 层文件 lint（共享知识 K5）
 * 禁止 src/L 开头的层目录出现浏览器不友好的语法 / 全局：
 *   require( / module.exports / import  / export  / fetch( / new Worker / XMLHttpRequest / Math.random / Date.now
 * 这些在 file:// 下经典脚本中是 ReferenceError / SyntaxError（CORS 拦 import/export）；
 * 确定性要求禁止 Math.random / Date.now。
 *
 * Phase 0 尚无 src/L* 目录，本脚本扫描 0 文件并通过；Phase 1 起成为 CI 硬门禁。
 */
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const FORBIDDEN = [
  'require(', 'module.exports', 'import ', 'export ',
  'fetch(', 'new Worker', 'XMLHttpRequest', 'Math.random', 'Date.now',
];

function walk(dir) {
  const out = [];
  if (!fs.existsSync(dir)) return out;
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) out.push(...walk(p));
    else if (e.name.endsWith('.js')) out.push(p);
  }
  return out;
}

// Phase 1 起：层目录位于 src/ 下（src/L10-microisa、src/L9-numbase、src/L8-kernel …）。
const SRC = path.join(ROOT, 'src');
const layerDirs = fs.existsSync(SRC)
  ? fs.readdirSync(SRC).filter((d) => /^L\d/.test(d) && fs.statSync(path.join(SRC, d)).isDirectory())
  : [];
const files = layerDirs.flatMap((d) => walk(path.join(SRC, d)));

let errors = 0;
for (const f of files) {
  const lines = fs.readFileSync(f, 'utf8').split('\n');
  lines.forEach((line, i) => {
    // 去除行内注释后检查（避免把注释里的说明误判）
    const code = line.replace(/\/\/.*$/, '').replace(/\/\*.*?\*\//g, '');
    for (const pat of FORBIDDEN) {
      if (code.includes(pat)) {
        console.error(`❌ ${path.relative(ROOT, f)}:${i + 1} 禁止的语法/全局: "${pat}"`);
        errors++;
      }
    }
  });
}

if (errors === 0) {
  console.log(`✅ lint-layers：扫描 ${files.length} 个层文件，无违例`);
  process.exitCode = 0;
} else {
  console.error(`❌ lint-layers：发现 ${errors} 处违例`);
  process.exitCode = 1;
}

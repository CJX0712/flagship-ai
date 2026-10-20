#!/usr/bin/env node
/* ============================================================================
 * 构建脚本：把十层栈 + 套件 + 应用内联进 demo.template.html，产出零依赖单文件
 * demo.html（file:// 直接打开即可运行），并导出 data/corpus-zh.txt。
 * ---------------------------------------------------------------------------
 * 浏览器装配策略（v15.0）：
 *   浏览器无 module/require，因此不能直接内联 Node 入口。做法是在产物中注入一个
 *   极小的 CommonJS 运行时（mini-require），把 namespace / 各层 / legacy 引擎 /
 *   src/index.js 全部按「模块」内联，使浏览器执行的正是 Node 侧同一份装配代码
 *   （双端同码：src/index.js 的 L8/L9 算子覆盖 + L3 架构组装在浏览器同样生效）。
 *
 *   关键修正（历史事故）：
 *   v12.0–v14.0 期间本脚本直接内联 src/flagship-engine.js（6 行 Node shim：
 *   `module.exports = require('./legacy/...')`），浏览器 classic script 执行到
 *   `require` 即 ReferenceError，demo.html 整体损坏。现由硬校验拦截：
 *   检测到 shim 特征则改内联 legacy 引擎本体（UMD），并断言 GPTModel 存在。
 *
 * 可复现门禁：产物行尾统一为 \n，且干净树（git archive）重建与仓库内产物
 * 逐字节一致（scripts/probe-demo.js 断言）。
 *
 * 用法：node scripts/build-demo.js
 * ========================================================================*/
'use strict';
const fs = require('fs');
const path = require('path');

const root = path.join(__dirname, '..');
const read = (p) => fs.readFileSync(path.join(root, p), 'utf8');
const NS = require('../src/namespace.js');

/* ---------- 1. 模块集合（虚拟绝对路径 → 源码） ---------- */
const VROOT = '/src';                      // 浏览器侧 __dirname 虚拟根
const modules = new Map();

// 1a. legacy 引擎本体：Node 侧 src/flagship-engine.js 是转发 shim，浏览器必须用本体
const engineShim = read('src/flagship-engine.js');
const engineSrc = /require\s*\(/.test(engineShim)
  ? read('src/legacy/flagship-engine.v11.js')
  : engineShim;
if (!/GPTModel/.test(engineSrc)) {
  console.error('❌ 引擎内联失败：未找到 GPTModel（疑似内联了 Node shim）');
  process.exit(1);
}
modules.set(VROOT + '/legacy/flagship-engine.v11.js', engineSrc);

// 1b. 命名空间 + 入口
modules.set(VROOT + '/namespace.js', read('src/namespace.js'));
modules.set(VROOT + '/index.js', read('src/index.js'));

// 1c. 十层（顺序真源：src/namespace.js __loadOrder）
for (const rel of NS.__loadOrder) {
  const norm = rel.replace(/^\.\//, '');
  modules.set(VROOT + '/' + norm, read(path.join('src', norm)));
}

/* ---------- 2. 浏览器 mini-require 运行时 ---------- */
const esc = (s) => s.replace(/<\/script/gi, '<\\/script'); // 防止提前闭合 <script>

const runtime = `
/* ===== 浏览器 mini-require（仅内联装配用，零依赖） ===== */
(function () {
  'use strict';
  var __defs = {}, __cache = {};

  // 内置 path 桩：只需 join/dirname/basename/resolve（index.js 用 path.join(__dirname, rel)）
  function __normPath(p) {
    var out = [];
    p.split('/').forEach(function (seg) {
      if (!seg || seg === '.') return;
      if (seg === '..') { out.pop(); return; }
      out.push(seg);
    });
    return '/' + out.join('/');
  }
  var __path = {
    join: function () {
      var parts = [];
      for (var i = 0; i < arguments.length; i++) {
        var s = String(arguments[i]);
        if (s.charAt(0) === '/') parts = [];
        parts.push(s);
      }
      return __normPath(parts.join('/'));
    },
    dirname: function (p) { var i = p.lastIndexOf('/'); return i <= 0 ? '/' : p.slice(0, i); },
    basename: function (p) { return p.slice(p.lastIndexOf('/') + 1); },
    resolve: function (p) { return __normPath(p.charAt(0) === '/' ? p : '/' + p); },
    sep: '/',
  };

  function __resolve(from, req) {
    if (req.charAt(0) !== '.') return req;          // 内置模块（path）
    return __path.join(__path.dirname(from), req);
  }

  function __require(from, req) {
    if (req === 'path') return __path;
    var id = __resolve(from, req);
    if (__cache[id]) return __cache[id].exports;
    var def = __defs[id];
    if (!def) throw new Error('模块未内联: ' + id);
    var module = { exports: {} };
    __cache[id] = module;
    def(module, module.exports, function (r) { return __require(id, r); }, __path.dirname(id));
    return module.exports;
  }

  function __define(id, fn) { __defs[id] = fn; }
  window.__FLAGSHIP_MODULES__ = { define: __define, require: __require };
})();
`;

/* ---------- 3. 生成内联脚本块 ---------- */
const moduleBlocks = [...modules.entries()]
  .map(([id, src]) => {
    return `/* [module ${id}] */\n__FLAGSHIP_MODULES__.define(${JSON.stringify(id)}, function (module, exports, require, __dirname) {\n${esc(src)}\n});`;
  })
  .join('\n');

const boot = `
/* ===== 浏览器端装配入口：跑真实 src/index.js（双端同码） ===== */
(function () {
  'use strict';
  var M = window.__FLAGSHIP_MODULES__;
  var api = M.require('/src/index.js', './index.js');
  var ns = globalThis.FlagshipAI;          // 层命名空间（L3–L10 挂载点）
  globalThis.FlagshipAI = api;             // 公共 API（与 Node module.exports 同构：34 键）
  globalThis.FlagshipNS = ns;              // 命名空间（浏览器可查 L10/L9/… 状态）
  delete window.__FLAGSHIP_MODULES__;
})();
`;

const stack = runtime + '\n' + moduleBlocks + '\n' + boot;

const parts = {
  '__FLAGSHIP_STACK__': stack,
  '__FLAGSHIP_VERIFY__': read('src/verify.js'),
  '__FLAGSHIP_CORPUS__': read('src/corpus.js'),
  '__FLAGSHIP_APP__': read('src/app.js'),
};

let html = read('demo.template.html');
for (const [marker, code] of Object.entries(parts)) {
  // 用函数替换器，避免代码中的 $ 序列被解释为正则替换模式
  const token = `/*${marker}*/`;
  if (!html.includes(token)) {
    console.error(`❌ 模板中缺少占位标记 ${token}`);
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
console.log(`✅ 内联十层：${modules.size} 个模块（namespace + ${NS.__loadOrder.length} 层 + legacy 引擎 + index.js）`);
console.log(`✅ data/corpus-zh.txt 已导出`);

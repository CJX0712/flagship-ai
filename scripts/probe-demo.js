#!/usr/bin/env node
/* ============================================================================
 * 浏览器装配探针（probe-demo.js）
 * ----------------------------------------------------------------------------
 * 无头验证 demo.html 的浏览器路径（file:// 直开场景）：
 *   1. 从 demo.html 按序提取全部 <script> 经典脚本块
 *   2. 在 vm 上下文中以浏览器全局（window=self、无 module/require、
 *      最小 document 桩）依次执行
 *   3. 断言：引擎 UMD root 分支生效（GPTModel 就位）、L9/L10 层注入
 *      globalThis.FlagshipAI、自检套件 FlagshipVerify 就位、1024 槽位完整
 *      + 120 项已实现、代表性微指令可运行
 * 背景：v12 曾误将 Node shim 内联进 demo 导致浏览器 ReferenceError，
 * 本探针将该类事故纳入 CI 门禁。
 * 运行：node scripts/probe-demo.js
 * 作者：晨星  ·  License: MIT
 * ========================================================================*/
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const ROOT = path.join(__dirname, '..');
const html = fs.readFileSync(path.join(ROOT, 'demo.html'), 'utf8');

// 1. 提取 <script> 块（无 src 属性的内联块，按文档序）
const blocks = [];
const re = /<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/gi;
let m;
while ((m = re.exec(html)) !== null) blocks.push(m[1]);
if (blocks.length < 5) {
  console.error(`❌ demo.html 内联脚本块不足（${blocks.length} < 5），构建异常`);
  process.exit(1);
}

// 2. 浏览器语义全局：window/self 自指、无 module/require、最小 DOM 桩
function makeEl() {
  const el = {
    value: '', textContent: '', scrollTop: 0, className: '', style: {},
    children: [],
    appendChild(c) { this.children.push(c); return c; },
    addEventListener() { }, removeEventListener() { },
    setAttribute() { }, getAttribute() { return null; },
    click() { }, focus() { }, blur() { },
    getContext() { return null; },
    classList: { add() { }, remove() { }, toggle() { } },
  };
  return el;
}
const documentStub = {
  getElementById() { return makeEl(); },
  createElement() { return makeEl(); },
  createElementNS() { return makeEl(); },
  querySelector() { return null; },
  querySelectorAll() { return []; },
  addEventListener() { }, removeEventListener() { { } },
  body: makeEl(),
};
const sandbox = {
  console, setTimeout, clearTimeout, setInterval, clearInterval,
  document: documentStub,
  navigator: { userAgent: 'probe-demo' },
  location: { href: 'file:///demo.html', protocol: 'file:' },
  devicePixelRatio: 1,
  requestAnimationFrame(fn) { return setTimeout(fn, 0); },
  alert() { }, confirm() { return true; },
  URL: { createObjectURL() { return 'blob:stub'; }, revokeObjectURL() { } },
  FileReader: function () { this.readAsText = function () { }; },
  Blob: function () { },
};
sandbox.window = sandbox;
sandbox.self = sandbox;
sandbox.globalThis = sandbox;
vm.createContext(sandbox);

// 3. 按文档序执行（与浏览器 classic script 语义一致）
let loadErr = null;
for (let i = 0; i < blocks.length; i++) {
  try {
    vm.runInContext(blocks[i], sandbox, { filename: `demo.html<script#${i + 1}>` });
  } catch (e) {
    loadErr = { block: i + 1, message: e && e.message };
    break;
  }
}

// 4. 断言
const checks = [];
function chk(name, pass, detail) { checks.push({ name, pass, detail }); }

chk('脚本块按序加载无异常', !loadErr, loadErr ? `块#${loadErr.block}: ${loadErr.message}` : `${blocks.length} 个块全部执行`);
const F = sandbox.FlagshipAI;
chk('引擎 UMD root 分支生效（window.FlagshipAI）', !!F && typeof F.GPTModel === 'function', F ? 'GPTModel 就位' : '缺失');
chk('自检套件 FlagshipVerify 就位', typeof sandbox.FlagshipVerify === 'object' && sandbox.FlagshipVerify !== null && typeof sandbox.FlagshipVerify.runAll === 'function', typeof sandbox.FlagshipVerify);
chk('L10 层注入（1024 槽位完整）', !!F && !!F.L10 && F.L10.stats().total === 1024, F && F.L10 ? `total=${F.L10.stats().total}` : '缺失');
chk('L10 实现进度 120/1024', !!F && !!F.L10 && F.L10.stats().done === 120, F && F.L10 ? `done=${F.L10.stats().done}` : '缺失');
chk('L9 数值基础层注入', !!F && !!F.L9 && typeof F.L9.twoSum === 'function', F && F.L9 ? 'L9 就位' : '缺失');
let runOk = false, runDetail = '未运行';
try {
  const rope = F.L10.run('G003', [new Float64Array([1, 0, 0, 1]), 3, F.L10.run('G000', [4, 10000])]);
  runOk = rope.length === 4 && rope.every(Number.isFinite);
  runDetail = runOk ? 'G003 RoPE 施加有限' : '输出异常';
} catch (e) { runDetail = '异常: ' + e.message; }
chk('代表性微指令可运行（G003 RoPE）', runOk, runDetail);

let fwdOk = false, fwdDetail = '未运行';
try {
  const model = new F.GPTModel({ vocabSize: 40, blockSize: 8, nLayer: 1, nHead: 2, dModel: 16, seed: 7 });
  const lg = model.forward([1, 2, 3, 4, 5, 6, 7]).logits;
  fwdOk = lg.every(Number.isFinite);
  fwdDetail = fwdOk ? '前向 logits 全有限' : '存在非有限值';
} catch (e) { fwdDetail = '异常: ' + e.message; }
chk('引擎前向推理可用', fwdOk, fwdDetail);

// 5. 汇报
const W = 46;
console.log('🌐 旗舰 AI · 浏览器装配探针（demo.html 无头验证）');
console.log('─'.repeat(W));
for (const c of checks) {
  console.log(` ${c.pass ? '✅' : '❌'} ${c.name}`);
  console.log(`      ${c.detail}`);
}
const pass = checks.filter(c => c.pass).length;
console.log('─'.repeat(W));
console.log(`结果：${pass}/${checks.length} 通过 ${pass === checks.length ? '✅ 全绿' : '❌ 存在失败'}`);
process.exitCode = pass === checks.length ? 0 : 1;

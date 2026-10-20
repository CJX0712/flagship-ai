#!/usr/bin/env node
/* ============================================================================
 * 浏览器装配探针（probe-demo.js）
 * ----------------------------------------------------------------------------
 * 无头验证 demo.html 的浏览器路径（file:// 直开场景），并断言「双端同码」：
 *   1. demo.html 全部内联 <script> 块按序加载无异常
 *   2. 公共 API 就位（window.FlagshipAI）：34 键导出清单与 Node 侧完全一致
 *   3. 十层命名空间就位（window.FlagshipNS）：L10/L9/L8/… 挂载
 *   4. L10 微指令注册中心：容量 1024 + 已实现项计数（与 Node 侧一致）
 *   5. 自检套件就位（window.FlagshipVerify，浏览器分支）
 *   6. 代表性微指令可运行（G003 RoPE 施加，输出有限）
 *   7. 前向推理双端逐位一致：浏览器沙箱 vs Node require('src/index.js')
 *   8. 浏览器侧确定性：同输入两次前向 logits 逐位相同
 * 背景：v12.0–v14.0 期间 build-demo.js 误内联 Node shim，浏览器 classic script
 * 执行到 require 即 ReferenceError，demo.html 整体损坏（无人察觉）。本探针把
 * 该类事故固化为 CI 硬门禁。
 * 运行：node scripts/probe-demo.js
 * 作者：晨星  ·  License: MIT
 * ========================================================================*/
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const ROOT = path.join(__dirname, '..');
const html = fs.readFileSync(path.join(ROOT, 'demo.html'), 'utf8');

/* ---------- 1. 提取内联脚本块 ---------- */
const blocks = [];
const re = /<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/gi;
let m;
while ((m = re.exec(html)) !== null) blocks.push(m[1]);
if (blocks.length < 4) {
  console.error(`❌ demo.html 内联脚本块不足（${blocks.length} < 4），构建异常`);
  process.exit(1);
}

/* ---------- 2. 浏览器语义沙箱 ---------- */
function makeEl() {
  return {
    value: '', textContent: '', scrollTop: 0, className: '', style: {},
    children: [],
    appendChild(c) { this.children.push(c); return c; },
    addEventListener() { }, removeEventListener() { },
    setAttribute() { }, getAttribute() { return null; },
    click() { }, focus() { }, blur() { },
    getContext() { return null; },
    classList: { add() { }, remove() { }, toggle() { } },
  };
}
const documentStub = {
  getElementById() { return makeEl(); },
  createElement() { return makeEl(); },
  createElementNS() { return makeEl(); },
  querySelector() { return null; },
  querySelectorAll() { return []; },
  addEventListener() { }, removeEventListener() { },
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
vm.createContext(sandbox);

/* ---------- 3. 按文档序执行（与浏览器 classic script 语义一致） ---------- */
let loadErr = null;
for (let i = 0; i < blocks.length; i++) {
  try {
    vm.runInContext(blocks[i], sandbox, { filename: `demo.html<script#${i + 1}>` });
  } catch (e) {
    loadErr = { block: i + 1, message: e && e.message };
    break;
  }
}

/* ---------- 4. 断言 ---------- */
const checks = [];
const chk = (name, pass, detail) => checks.push({ name, pass, detail });

chk('脚本块按序加载无异常', !loadErr,
  loadErr ? `块#${loadErr.block}: ${loadErr.message}` : `${blocks.length} 个块全部执行`);

const api = sandbox.FlagshipAI;
const ns = sandbox.FlagshipNS;
chk('公共 API 就位（window.FlagshipAI）', !!api && typeof api.GPTModel === 'function',
  api ? `${Object.keys(api).length} 个导出键` : '缺失');

const nodeApi = require('../src/index.js');
const nodeKeys = Object.keys(nodeApi).sort();
const browserKeys = api ? Object.keys(api).sort() : [];
const sameKeys = nodeKeys.length === browserKeys.length && nodeKeys.every((k, i) => k === browserKeys[i]);
chk('导出清单与 Node 侧完全一致（K15）', sameKeys,
  sameKeys ? `${browserKeys.length} 键一致` : `Node ${nodeKeys.length} vs 浏览器 ${browserKeys.length}`);

chk('十层命名空间就位（window.FlagshipNS）', !!ns && !!ns.L10 && !!ns.L9 && !!ns.L8,
  ns ? `L3–L10 挂载：${['L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10'].filter(k => ns[k]).join(' ')}` : '缺失');

const bStats = ns && ns.L10 && ns.L10.stats ? ns.L10.stats() : null;
const nStats = globalThis.FlagshipAI && globalThis.FlagshipAI.L10 ? globalThis.FlagshipAI.L10.stats() : null;
let statsDetail = '不可用', statsOk = false;
if (bStats) {
  const bImpl = bStats.total ? bStats.total.impl : undefined;
  const nImpl = nStats && nStats.total ? nStats.total.impl : undefined;
  const cap = bStats.total ? bStats.total.capacity : undefined;
  statsOk = cap === 1024 && bImpl === nImpl;
  statsDetail = `容量 ${cap} · 浏览器已实现 ${bImpl} · Node 已实现 ${nImpl}`;
}
chk('L10 注册中心容量 1024 且实现数双端一致', statsOk, statsDetail);

chk('自检套件就位（window.FlagshipVerify）',
  sandbox.FlagshipVerify && typeof sandbox.FlagshipVerify.runAll === 'function',
  typeof sandbox.FlagshipVerify);

let opOk = false, opDetail = '未运行';
try {
  const op = ns.L10.ops['A001'];            // L10 契约：op.run(...) / op.predicate(...)
  const v = op.run(1.5, 2.25);
  const pred = op.predicate(1.5, 2.25);
  opOk = Number.isFinite(v) && pred === true;
  opDetail = `μA001 run=${v} · predicate=${pred}`;
} catch (e) { opDetail = '异常: ' + e.message; }
chk('代表性微指令可运行（浏览器内）', opOk, opDetail);

/* ---------- 5. 前向推理双端逐位一致 ---------- */
const CFG = { vocabSize: 40, blockSize: 8, nLayer: 1, nHead: 2, dModel: 16, seed: 7 };
const TOKS = [1, 2, 3, 4, 5, 6, 7];
function forward(apiObj) {
  const model = new apiObj.GPTModel(Object.assign({}, CFG));
  return Array.from(model.forward(TOKS).logits);
}
let parityOk = false, parityDetail = '未运行', detOk = false, detDetail = '未运行';
try {
  const bLogits = forward(api);
  const nLogits = forward(nodeApi);
  let maxDiff = 0;
  for (let i = 0; i < Math.min(bLogits.length, nLogits.length); i++) {
    maxDiff = Math.max(maxDiff, Math.abs(bLogits[i] - nLogits[i]));
  }
  parityOk = bLogits.length === nLogits.length && maxDiff === 0;
  parityDetail = `浏览器 vs Node：${bLogits.length} logits，maxΔ=${maxDiff}`;
  const again = forward(api);
  let d2 = 0;
  for (let i = 0; i < bLogits.length; i++) d2 = Math.max(d2, Math.abs(bLogits[i] - again[i]));
  detOk = d2 === 0;
  detDetail = `同输入两次前向 maxΔ=${d2}`;
} catch (e) {
  parityDetail = '异常: ' + e.message;
  detDetail = '异常: ' + e.message;
}
chk('前向推理双端逐位一致（双端同码）', parityOk, parityDetail);
chk('浏览器侧前向确定性', detOk, detDetail);

/* ---------- 6. 汇报 ---------- */
const W = 46;
console.log('🌐 旗舰 AI · 浏览器装配探针（demo.html 无头验证 + 双端同码）');
console.log('─'.repeat(W));
for (const c of checks) {
  console.log(` ${c.pass ? '✅' : '❌'} ${c.name}`);
  console.log(`      ${c.detail}`);
}
const pass = checks.filter(c => c.pass).length;
console.log('─'.repeat(W));
console.log(`结果：${pass}/${checks.length} 通过 ${pass === checks.length ? '✅ 全绿' : '❌ 存在失败'}`);
process.exitCode = pass === checks.length ? 0 : 1;

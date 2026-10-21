'use strict';
/* 浏览器交付路径测试（T05 新增 · 常驻守护）
 *
 * 背景（为什么必须有这个测试）：
 *   单文件 demo.html 是本项目对外的「可运行」交付物。T05 之前它长期是坏的 ——
 *   构建脚本内联的是 Phase 0 的 CommonJS 转发 shim（`module.exports = require(...)`），
 *   浏览器里 `module` 未定义，第一个 <script> 即抛错、整页脚本停摆；而所有
 *   Node 侧门禁都绿（Node 能 require），所以从没有任何门禁发现它。
 *
 * 本测试用 node:vm 沙箱模拟浏览器执行构建产物，覆盖：
 *   1) 四个 <script> 块能顺序执行完毕（不抛错）；
 *   2) window.FlagshipAI 具备 34 个契约键（与 Node 侧 src/index.js 完全一致）；
 *   3) window.FlagshipVerify.runAll() 在浏览器路径下 12/12 全绿；
 *   4) 浏览器路径下能真实训练并生成（证明不是「能加载但不能用」）。
 */
const { test } = require('node:test');
const assert = require('node:assert');
const vm = require('node:vm');

const { buildHtml } = require('../scripts/build-demo.js');
const nodeApi = require('../src/index.js'); // Node 侧 34 键，作为浏览器侧的对照基准

// ── 最小 DOM 桩：app.js 在加载期会取元素、挂监听，缺桩会抛错 ──────────────
function fakeEl(id) {
  const el = {
    id, value: '', textContent: '', innerHTML: '', className: '', style: {},
    children: [], checked: false, disabled: false, scrollTop: 0, scrollHeight: 0,
    classList: { add() {}, remove() {}, toggle() {} },
    appendChild(c) { this.children.push(c); return c; },
    remove() {}, focus() {}, click() {},
    addEventListener() {}, removeEventListener() {},
    setAttribute() {}, getAttribute: () => null,
  };
  return el;
}
function makeSandbox() {
  const cache = {};
  const document = {
    getElementById: (id) => (cache[id] = cache[id] || fakeEl(id)),
    createElement: () => fakeEl(''),
    querySelector: () => fakeEl(''),
    querySelectorAll: () => [],
    addEventListener() {},
    body: fakeEl('body'),
  };
  const sandbox = {
    console, TextEncoder, TextDecoder, performance, document,
    setTimeout, clearTimeout, setInterval, clearInterval,
    requestAnimationFrame: (f) => setTimeout(f, 0),
    navigator: { userAgent: 'node-vm' },
    // window 即 sandbox 自身：demo-app 用 window.addEventListener('resize', …) 合法，
    // 沙箱需提供该桩，否则加载期 bind() 抛 "window.addEventListener is not a function"。
    addEventListener() {}, removeEventListener() {},
  };
  sandbox.window = sandbox;
  sandbox.self = sandbox;
  sandbox.globalThis = sandbox;
  return sandbox;
}

/** 构建 demo.html 并在沙箱中按序执行全部脚本块，返回沙箱。 */
function runDemoInBrowser() {
  const html = buildHtml();
  const blocks = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map((m) => m[1]);
  assert.ok(blocks.length >= 4, `应至少 4 个脚本块，实际 ${blocks.length}`);
  const sandbox = makeSandbox();
  blocks.forEach((code, i) => {
    // 单块抛错即失败：这正是历史上 demo 坏掉的表现（第 1 块就 module is not defined）
    vm.runInNewContext(code, sandbox, { timeout: 60000, filename: `demo-block-${i + 1}.js` });
  });
  return sandbox;
}

test('浏览器路径：四个脚本块顺序执行无异常', () => {
  const sandbox = runDemoInBrowser();
  assert.ok(sandbox.FlagshipAI, 'window.FlagshipAI 应存在');
  assert.ok(sandbox.FlagshipVerify, 'window.FlagshipVerify 应存在');
});

test('浏览器路径：L1 应用层已注册并完成启动（NS.L1.app）', () => {
  const sandbox = runDemoInBrowser();
  const app = sandbox.FlagshipAI.L1 && sandbox.FlagshipAI.L1.app;
  assert.ok(app, 'NS.L1.app 应存在（demo-app.js 未随 __loadOrder 内联）');
  // 启动即绑定 DOM 事件；这里只要求关键成员齐备且 start() 已跑通（未抛错）
  for (const m of ['start', 'bind', 'buildModel', 'startTraining', 'doGenerate', 'runSelfCheck']) {
    assert.strictEqual(typeof app[m], 'function', `NS.L1.app.${m} 应为函数`);
  }
  assert.ok(app.state && typeof app.state === 'object', 'NS.L1.app.state 应为对象');
  // L1 依赖 L0 装配产物：两者必须同源（都来自 buildPublicAPI）
  assert.strictEqual(
    sandbox.FlagshipAI.L0.API.GPTModel,
    sandbox.FlagshipAI.GPTModel,
    'L0.API 与命名空间上的 34 键必须同源'
  );
});

test('浏览器路径：window.FlagshipAI 具备与 Node 侧一致的 34 个契约键', () => {
  const sandbox = runDemoInBrowser();
  const want = Object.keys(nodeApi).sort();
  const got = Object.keys(sandbox.FlagshipAI)
    .filter((k) => want.includes(k))
    .sort();
  assert.deepStrictEqual(got, want, '浏览器侧 34 键与 Node 侧不一致');
  assert.strictEqual(want.length, 34);
  for (const k of want) {
    assert.notStrictEqual(sandbox.FlagshipAI[k], undefined, `浏览器侧导出键 ${k} 为 undefined`);
  }
});

test('浏览器路径：自检套件 runAll() 12/12 全绿', () => {
  const sandbox = runDemoInBrowser();
  const results = sandbox.FlagshipVerify.runAll();
  const pass = results.filter((r) => r.pass).length;
  const failed = results.filter((r) => !r.pass).map((r) => `${r.name}（${r.detail}）`);
  assert.strictEqual(results.length, 12, `自检项应为 12，实际 ${results.length}`);
  assert.strictEqual(pass, 12, `浏览器侧自检未全绿：${failed.join(' | ')}`);
});

test('浏览器路径：能真实训练并生成（非「能加载不能用」）', () => {
  const sandbox = runDemoInBrowser();
  // 在沙箱里跑一次最小闭环：建词表 → 建模型 → 训 20 步 → 生成
  const out = vm.runInNewContext(
    `(function () {
      var F = window.FlagshipAI;
      var text = 'abracadabra abracadabra abracadabra';
      var tok = new F.Tokenizer(text);
      var ids = tok.encode(text);
      var cfg = F.applyPreset(
        { vocabSize: tok.vocabSize, blockSize: 16, nLayer: 1, nHead: 2, dModel: 16, seed: 7 },
        'llama'
      );
      var m = new F.GPTModel(cfg);
      var h = [];
      F.train(m, ids, { steps: 20, seed: 42, onStep: function (s, loss) { h.push(loss); } });
      var g = F.generate(m, tok, 'abra', { maxNew: 8, seed: 1, useCache: true });
      return { first: h[0], last: h[h.length - 1], text: g.text, ids: g.ids.length, vocab: tok.vocabSize };
    })()`,
    sandbox,
    { timeout: 60000 }
  );
  assert.ok(Number.isFinite(out.first), '初始 loss 应为有限数');
  assert.ok(out.last < out.first, `训练后 loss 应下降：${out.first} -> ${out.last}`);
  assert.strictEqual(out.ids, 8, '应生成 8 个新 token');
  assert.strictEqual(typeof out.text, 'string');
});

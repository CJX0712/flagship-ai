/* L10 · 微指令层 / 注册中心 —— 依赖：无（最底层，最先加载）
 * 八域 × 128 = 1024 槽位；实现项覆盖 registered-not-implemented 桩。
 * 每项微指令：纯函数、无隐藏状态、可确定性重放（随机性由显式 RNG 注入）。
 * @ref https://github.com/karpathy/micrograd（VJP 语义）· https://github.com/dubzzz/fast-check（性质测试范式）
 */
;(function (NS) {
  'use strict';

  // 所属层：L10 · 允许依赖：无（最底层）· 主要 @ref：micrograd / fast-check
  var DOMAINS = [
    { letter: 'A', name: '标量数值与浮点原语' },
    { letter: 'B', name: '向量归约与内存原语' },
    { letter: 'C', name: '矩阵与 BLAS 微内核' },
    { letter: 'D', name: '线性代数分解与求解' },
    { letter: 'E', name: '张量形状与索引原语' },
    { letter: 'F', name: '自动微分原语' },
    { letter: 'G', name: '注意力与位置编码原语' },
    { letter: 'H', name: '优化器·量化·随机原语' }
  ];
  var PER_DOMAIN = 128;
  function pad3(n) { return ('00' + n).slice(-3); }

  // 本地确定性 RNG（mulberry32），供 selfTest 复现；不依赖 L9（避免加载序耦合）
  function rng32(seed) {
    var s = seed >>> 0;
    return function () {
      s = (s + 0x6D2B79F5) | 0;
      var t = Math.imul(s ^ (s >>> 15), 1 | s);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  var registry = Object.create(null); // key -> op spec
  var order = [];                      // 注册顺序（去重）

  function register(spec) {
    if (!spec || !spec.key) throw new Error('[L10] register 需要 key');
    spec.impl = spec.impl === 'done' ? 'done' : 'stub';
    if (!registry[spec.key]) order.push(spec.key);
    registry[spec.key] = spec;
    return spec;
  }

  // 桩工厂：每项捕获自己的 key（避免循环内 var 闭包共享末值）
  function notImpl(k) { return function () { throw new Error('not implemented: μ' + k); }; }

  // 预置 1024 槽位为 registered-not-implemented；实现项随后按同 key 覆盖
  for (var di = 0; di < DOMAINS.length; di++) {
    var d = DOMAINS[di];
    for (var n = 0; n < PER_DOMAIN; n++) {
      var key = d.letter + pad3(n);
      if (!registry[key]) register({
        key: key, domain: d.letter, name: key, sig: '() -> undefined',
        fn: notImpl(key),
        impl: 'stub', ref: '', complexity: '-'
      });
    }
  }

  function get(key) { return registry[key]; }
  function display(key) { return 'μ' + key; }
  function keys() { return order.slice(); }

  function stats() {
    var byDomain = {};
    for (var i = 0; i < DOMAINS.length; i++) {
      byDomain[DOMAINS[i].letter] = { name: DOMAINS[i].name, total: PER_DOMAIN, done: 0 };
    }
    var done = 0;
    for (var k in registry) {
      var op = registry[k];
      if (op.impl === 'done') { done++; byDomain[op.domain].done++; }
    }
    return { total: DOMAINS.length * PER_DOMAIN, done: done, byDomain: byDomain };
  }

  function run(key, args) {
    var op = registry[key];
    if (!op) throw new Error('未知微指令: ' + key);
    return op.fn.apply(null, args || []);
  }

  // 跑一项 selfTest；返回 {pass, detail, impl}
  function check(key, seed) {
    var op = registry[key];
    if (!op) return { pass: false, detail: '未知', impl: 'missing' };
    if (op.impl !== 'done') return { pass: false, detail: 'registered-not-implemented', impl: 'stub' };
    try {
      var rng = rng32((seed || 1) >>> 0);
      var r = op.selfTest ? op.selfTest(rng) : { pass: true, detail: 'no selfTest' };
      r.impl = 'done';
      return r;
    } catch (e) {
      return { pass: false, detail: '异常: ' + (e && e.message), impl: 'done' };
    }
  }

  NS.L10 = {
    DOMAINS: DOMAINS, PER_DOMAIN: PER_DOMAIN,
    register: register, get: get, display: display, keys: keys,
    stats: stats, run: run, check: check, rng32: rng32
  };
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

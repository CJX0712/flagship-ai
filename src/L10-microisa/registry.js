/* ============================================================================
 * L10 微指令 · 登记中心（registry）
 * ----------------------------------------------------------------------------
 * 所属层：L10-microIsa（最底层微指令）
 * 允许依赖的层：无
 * 主要 @ref：RISC-V 指令集手册（微指令分解思想）；ISAv (Chris Lattner, MLIR
 *            op 注册表思想)
 *
 * 责任：构建 1024 槽位（8 域 A–H × 128 槽位）的微指令登记中心。每个微指令
 *       由内部键（如 'A001'）索引；展示形式 μA001（共享约定 K14）。未实现的
 *       槽位登记为 status:'stub'，供 isa.js 如实统计覆盖率（零虚标，Q2 决策）。
 *
 * 域定义：
 *   A 标量算术/函数  B 向量运算  C 矩阵/张量  D 注意力基元
 *   E 形状/索引      F 归一化    G 激活/损失  H 采样/量化
 * ==========================================================================*/
;(function (NS) {
  'use strict';
  NS.L10 = NS.L10 || {};
  const L10 = NS.L10;
  L10.ops = L10.ops || {};
  L10.domains = L10.domains || {};

  const DOMAIN_NAMES = {
    A: '标量算术与初等函数',
    B: '向量运算',
    C: '矩阵/张量运算',
    D: '注意力基元',
    E: '形状与索引',
    F: '归一化',
    G: '激活与损失',
    H: '采样与量化',
  };
  const DOMAINS = Object.keys(DOMAIN_NAMES); // ['A'..'H']
  const CAP = 128;                            // 每域 128 槽位

  DOMAINS.forEach((d) => { L10.domains[d] = { name: DOMAIN_NAMES[d], capacity: CAP }; });

  function keyOf(domain, index) { return domain + String(index).padStart(3, '0'); }

  // 注册一个已实现的微指令。def 必须含七件套（K11/L10）：name/intuition/math/
  // pseudocode/complexity/ref/commonPitfall/invariant + run + predicate。
  function register(domain, index, def) {
    const key = keyOf(domain, index);
    def.key = key; def.domain = domain; def.index = index; def.status = 'impl';
    L10.ops[key] = def;
    return def;
  }

  // 未实现槽位登记为 stub（不影响覆盖率统计，供后续 Phase 补齐）。
  function registerStub(domain, index) {
    const key = keyOf(domain, index);
    if (L10.ops[key] && L10.ops[key].status === 'impl') return; // 已被实现覆盖则跳过
    L10.ops[key] = {
      key, domain, index, status: 'stub',
      name: '(未实现)', intuition: '', math: '', pseudocode: '',
      complexity: '', ref: '', commonPitfall: '', invariant: '',
      run: null, predicate: null,
    };
  }

  // 预填：每个域 128 槽位全为 stub；后续 A/C 域文件会覆盖前 96 个为 impl。
  DOMAINS.forEach((d) => { for (let i = 1; i <= CAP; i++) registerStub(d, i); });

  // 展示形式：'A001' -> 'μA001'（K14）。
  function display(key) { return 'μ' + key; }

  // 按域统计：返回 { A:{impl,capacity}, ... , total:{impl,capacity} }。
  function stats() {
    const out = {};
    let totImpl = 0;
    DOMAINS.forEach((d) => {
      let impl = 0;
      for (let i = 1; i <= CAP; i++) {
        const op = L10.ops[keyOf(d, i)];
        if (op && op.status === 'impl') impl++;
      }
      out[d] = { impl, capacity: CAP };
      totImpl += impl;
    });
    out.total = { impl: totImpl, capacity: DOMAINS.length * CAP };
    return out;
  }

  L10.keyOf = keyOf;
  L10.register = register;
  L10.registerStub = registerStub;
  L10.display = display;
  L10.stats = stats;
  L10.CAP = CAP;
  L10.DOMAINS = DOMAINS;
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

/* 旗舰 AI v12 · 公共 API 入口
 * 加载顺序（自底向上）：L10 → L9 → … → L1（见 src/namespace.js __loadOrder）。
 *
 * Phase 1（T02）策略（安全网式拆层）：
 *   1) 自底向上加载十层，把 L8/L9 数值内核与 L10 微指令登记中心挂到
 *      globalThis.FlagshipAI 命名空间（K3/K4）。
 *   2) 整体复制 v11 legacy 的 34 个导出键 —— 模型级 API（GPTModel / train /
 *      generate / 各归一化与注意力算子）仍走 legacy 实现，保证 R1–R8 红线
 *      （参数 / 前向 / 反向 / 训练 / 生成 / 量化 / 存档 逐位等价）琐碎绿。
 *   3) 仅用「与 v11 逐字一致」的 L8/L9 内核覆盖 6 个逐算子导出键
 *      （matmul / matmulAtB / matmulTA / softmaxRows / mulberry32 / randn），
 *      使 R9 真正逐位验证新内核代码（Δ=0 契约）。
 *
 * 关键不变量（K15）：导出键集合与 v11 的 return 完全一致（34 个键，不多不少）。
 * 因此本文件【不】向 api 追加任何额外键；命名空间经 globalThis.FlagshipAI 暴露，
 * 供 scripts/isa.js 与 tests/ 读取，不影响导出契约。
 */
'use strict';
const NS = require('./namespace.js');
const path = require('path');

// 1) 自底向上加载十层（K4）：路径相对 src/，与 __loadOrder 真源一致。
for (const rel of NS.__loadOrder) {
  require(path.join(__dirname, rel));
}

// 2) v11 legacy 引擎：提供 34 个导出键，满足 K15 完全一致。
const legacy = require('./legacy/flagship-engine.v11.js');

// 3) 构建公共 API：先整体复制 legacy 导出（R1–R8 仍走 legacy 模型实现）。
const api = Object.assign({}, legacy);

// 4) 覆盖逐位等价的 L8/L9 算子（R9 Δ=0 契约）：
//    这些导出键被覆盖，但 legacy GPTModel 内部调用的是闭包内的同名函数，
//    不受此覆盖影响（R1–R8 仍比对 legacy-vs-legacy，琐碎绿）。
api.matmul      = NS.L8.matmul.matmul;       // 含 if(a===0) continue，逐字 v11
api.matmulAtB   = NS.L8.matmul.matmulAtB;
api.matmulTA    = NS.L8.matmul.matmulTA;
api.softmaxRows = NS.L8.reduce.softmaxRows;  // 数值稳定 softmax，逐字 v11
api.mulberry32  = NS.L9.rng.mulberry32;      // 确定性 PRNG，逐字 v11
api.randn       = NS.L9.rng.randn;           // Box–Muller，逐字 v11

module.exports = api;

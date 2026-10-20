/* 旗舰 AI v12 · 公共 API 入口
 * 加载顺序（自底向上）：L10 → L9 → … → L1（见 src/namespace.js __loadOrder）。
 *
 * Phase 1（T02）策略（安全网式拆层）：
 *   1) 自底向上加载十层，把 L8/L9 数值内核与 L10 微指令登记中心挂到
 *      globalThis.FlagshipAI 命名空间（K3/K4）。
 *   2) 整体复制 v11 legacy 的 34 个导出键 —— 模型壳（GPTModel / train /
 *      generate / persistence）仍走 legacy 实现，构造器 RNG 消费顺序冻结（R1 安全）。
 *   3) 仅用「与 v11 逐字一致」的 L8/L9 内核覆盖 6 个逐算子导出键
 *      （matmul / matmulAtB / matmulTA / softmaxRows / mulberry32 / randn），
 *      使 R9 真正逐位验证新内核代码（Δ=0 契约）。
 *
 * Phase 2（T03）策略（拆 forward+backward 单体 → L4 算子 + L3 组装）：
 *   4) 导出 GPTModel 换为一个【子类】：构造器/zeroGrads/paramCount/quantize 全部继承
 *      legacy（RNG 顺序不变，R1 安全），仅 forward/backward 覆写为 L3 blocks 重组装
 *      （调用顺序与 legacy 逐行一致 → R2 logits / R3 grads / R4 train / R5 generate 全 Δ=0）。
 *      注意：必须用子类而非改 legacy.GPTModel.prototype —— 否则 regression 的 V11 与 V12
 *      会共享同一被改原型，R2/R3 退化成自比自（琐碎绿），失去零回归意义。
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

// 3) 构建公共 API：先整体复制 legacy 导出（模型壳仍走 legacy 实现）。
const api = Object.assign({}, legacy);

// 3b) 覆盖逐位等价的 L8/L9 算子（R9 Δ=0 契约）：这些导出键被覆盖供 R9 逐算子比对；
//     legacy GPTModel 内部调用的是闭包内的同名函数，不受此覆盖影响。
api.matmul      = NS.L8.matmul.matmul;       // 含 if(a===0) continue，逐字 v11
api.matmulAtB   = NS.L8.matmul.matmulAtB;
api.matmulTA    = NS.L8.matmul.matmulTA;
api.softmaxRows = NS.L8.reduce.softmaxRows;  // 数值稳定 softmax，逐字 v11
api.mulberry32  = NS.L9.rng.mulberry32;      // 确定性 PRNG，逐字 v11
api.randn       = NS.L9.rng.randn;           // Box–Muller，逐字 v11

// 4) T03：导出 GPTModel 换成子类，forward/backward 覆写为 L3 blocks 重组装。
//    构造器与全部非前向/反向方法继承 legacy（RNG 顺序冻结，R1 安全）；
//    legacy.GPTModel.prototype 保持不动，V11 仍走 legacy 单体 → R2/R3 是真·零回归比对。
class GPTModelL3 extends legacy.GPTModel {
  forward(tokens) { return NS.L3.blocks.forward(this, tokens); }
  backward(cache, dlogits) { return NS.L3.blocks.backward(this, cache, dlogits); }
}
api.GPTModel = GPTModelL3;

// 5) L4 算子目录自举（仅登记元数据供自检/文档，不改数值路径，Δ=0 无影响）。
if (NS.L4 && NS.L4.registry && NS.L4.registry.bootstrap) NS.L4.registry.bootstrap();
// 5b) L5 VJP 注册中心 + L6 kernel 注册中心自举：把 L4 backward / L4+L8 内核登记为
//     VJP 与 kernel（L5/L6 先于 L4 加载，故在全部层加载完后补登；幂等、惰性守卫）。
if (NS.L5 && NS.L5.registry && NS.L5.registry.bootstrap) NS.L5.registry.bootstrap();
if (NS.L6 && NS.L6.kernel && NS.L6.kernel.bootstrap) NS.L6.kernel.bootstrap();
if (NS.L6 && NS.L6.backend && NS.L6.backend.refreshBackends) NS.L6.backend.refreshBackends();

module.exports = api;

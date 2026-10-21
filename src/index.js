/* 旗舰 AI v12 · 公共 API 入口
 * 加载顺序（自底向上）：L10 → L9 → … → L2 → L0（见 src/namespace.js __loadOrder）。
 *
 * Phase 1（T02）：L10/L9/L8 抽离 + 192 项微指令逐位验证；模型壳整体复制 legacy 导出。
 * Phase 2（T03）：forward+backward 单体拆为 L4 算子 + L3 组装（GPTModel 子类覆写）。
 * Phase 3（T04）：十层实现正式接管本入口 —— 不再转发 legacy：
 *   - GPTModel                 = NS.L3.gpt_model.GPTModel
 *     （构造器/zeroGrads/paramCount/quantize 为 verbatim lift，RNG 消费顺序与 legacy
 *      冻结一致 → R1 安全；forward/backward 委托 L3 blocks 重组装）
 *   - train / generate         = NS.L2.{train,generate}
 *     （verbatim lift，内部依赖改引 L4.loss / L7.kv_cache / L9.rng；采样逻辑逐字保留，
 *      不提取公共函数 —— 保 RNG 消费顺序 → R4/R5 安全）
 *   - saveModel/loadModel/loadTokenizer = NS.L2.persistence
 *     （loadModel 造 L3 模型 → R7 存档闭环由十层真正接管）
 *   - Tokenizer/BPETokenizer/textToByteChars/byteCharsToText = NS.L2.tokenizer
 *   - ARCH_PRESETS/applyPreset = NS.L3.presets
 *   - 逐算子导出               = L8/L9/L4 verbatim lift（R9 逐位验证）
 *   - KV 四件套                = NS.L7.kv_cache
 *   legacy 自此仅作为 scripts/regression-v11.js 的 V11 基准被加载，不在本文件出现。
 *
 * Phase 4（T05）：34 键映射与 K15 装配期硬门禁全部下沉到 L0 装配层，本文件不再自留
 *   断言与注册中心自举 —— 只负责「按 __loadOrder 加载 → 调 L0 装配 → 导出」。
 *   这是浏览器路径等价性的构造保证：浏览器端（scripts/build-demo.js 内联）按同一份
 *   __loadOrder 加载后调用同一个 assemble(NS)，两侧不可能漂移。
 *
 * 关键不变量（K15）：导出键集合与 v11 的 return 完全一致（34 个键，不多不少）。
 *   逐名硬门禁位于 src/L0-boot/assemble.js 的 assemble() 内
 *   （Node 与浏览器共用）；键集对 legacy 的逐键断言见 tests/api-contract.test.js
 *   （测试文件允许 require legacy，层文件不允许）。
 */
'use strict';
const NS = require('./namespace.js');
const path = require('path');

// 1) 自底向上加载十层（K4）：路径相对 src/，与 __loadOrder 真源一致。
for (const rel of NS.__loadOrder) {
  require(path.join(__dirname, rel));
}

// 2) 由 L0 装配层组装 34 键公共 API（K15：与 v11 return 完全一致）。
//    34 键映射 + EXPORTED_KEYS 逐名硬门禁 + 注册中心自举的唯一真源在
//    src/L0-assembly/bootstrap.js：Node 与浏览器调用同一个函数，两侧不可能漂移。
const api = NS.L0.assemble(NS);

module.exports = api;

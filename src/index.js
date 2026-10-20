/* 旗舰 AI v12 · 公共 API 入口
 * 加载顺序（自底向上）：L10 → L9 → … → L1（见 src/namespace.js __loadOrder）。
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
 * 关键不变量（K15）：导出键集合与 v11 的 return 完全一致（34 个键，不多不少）。
 *   加载期有计数硬门禁（下方）；键集逐名一致性由 tests/api-contract.test.js
 *   对 legacy 逐键断言（测试文件允许 require legacy，层文件不允许）。
 */
'use strict';
const NS = require('./namespace.js');
const path = require('path');

// 1) 自底向上加载十层（K4）：路径相对 src/，与 __loadOrder 真源一致。
for (const rel of NS.__loadOrder) {
  require(path.join(__dirname, rel));
}

// 2) 构建公共 API：34 个键全部由十层实现提供（K15：与 v11 return 完全一致）。
const api = {
  // L9 · 数值基础（RNG）
  mulberry32: NS.L9.rng.mulberry32,
  randn: NS.L9.rng.randn,
  // L8 · 数值内核（矩阵乘）
  matmul: NS.L8.matmul.matmul,
  matmulAtB: NS.L8.matmul.matmulAtB,
  matmulTA: NS.L8.matmul.matmulTA,
  // L4 · 神经算子（归一化）
  layernormFwd: NS.L4.norm.layernormFwd,
  layernormBwd: NS.L4.norm.layernormBwd,
  rmsnormFwd: NS.L4.norm.rmsnormFwd,
  rmsnormBwd: NS.L4.norm.rmsnormBwd,
  // L4 · 神经算子（激活 / 门控）
  geluFwd: NS.L4.activation.geluFwd,
  geluBwd: NS.L4.activation.geluBwd,
  siluF: NS.L4.activation.siluF,
  swigluFwd: NS.L4.activation.swigluFwd,
  swigluBwd: NS.L4.activation.swigluBwd,
  // L4 · 神经算子（位置编码）
  ropeApply: NS.L4.positional.ropeApply,
  // L8 · 数值内核（softmax：模型热路径走 L8 内核，L4.softmax 为其语义门面）
  softmaxRows: NS.L8.reduce.softmaxRows,
  softmaxBwd: NS.L8.reduce.softmaxBwd,
  // L4 · 神经算子（损失）
  crossEntropy: NS.L4.loss.crossEntropy,
  // L2 · 编排管线（分词器）
  Tokenizer: NS.L2.tokenizer.Tokenizer,
  BPETokenizer: NS.L2.tokenizer.BPETokenizer,
  textToByteChars: NS.L2.tokenizer.textToByteChars,
  byteCharsToText: NS.L2.tokenizer.byteCharsToText,
  // L3 · 架构组装（模型壳 + 架构预设）
  GPTModel: NS.L3.gpt_model.GPTModel,
  ARCH_PRESETS: NS.L3.presets.ARCH_PRESETS,
  applyPreset: NS.L3.presets.applyPreset,
  // L2 · 编排管线（训练 / 生成）
  train: NS.L2.train.train,
  generate: NS.L2.generate.generate,
  // L7 · 张量与内存（KV Cache）
  initKVCache: NS.L7.kv_cache.initKVCache,
  forwardStep: NS.L7.kv_cache.forwardStep,
  warmupKV: NS.L7.kv_cache.warmupKV,
  kvShift: NS.L7.kv_cache.kvShift,
  // L2 · 编排管线（持久化）
  saveModel: NS.L2.persistence.saveModel,
  loadModel: NS.L2.persistence.loadModel,
  loadTokenizer: NS.L2.persistence.loadTokenizer,
};

// 3) K15 加载期硬门禁：键数必须为 34（键集逐名一致性见 tests/api-contract.test.js）。
if (Object.keys(api).length !== 34) {
  throw new Error('K15 违例：公共 API 导出键数量 ' + Object.keys(api).length + ' ≠ 34');
}

// 4) 各层注册中心自举（仅登记元数据供自检/文档，不改数值路径，Δ=0 无影响）。
//    L4 算子目录：把 L4 算子登记为可检索元数据。
if (NS.L4 && NS.L4.registry && NS.L4.registry.bootstrap) NS.L4.registry.bootstrap();
//    L5 VJP 注册中心 + L6 kernel 注册中心：把 L4 backward / L4+L8 内核登记为
//    VJP 与 kernel（L5/L6 先于 L4 加载，故在全部层加载完后补登；幂等、惰性守卫）。
if (NS.L5 && NS.L5.registry && NS.L5.registry.bootstrap) NS.L5.registry.bootstrap();
if (NS.L6 && NS.L6.kernel && NS.L6.kernel.bootstrap) NS.L6.kernel.bootstrap();
if (NS.L6 && NS.L6.backend && NS.L6.backend.refreshBackends) NS.L6.backend.refreshBackends();

module.exports = api;

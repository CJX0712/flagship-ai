/* 旗舰 AI v12 · 命名空间与加载顺序真源
 * 规则（共享知识 K3 / K4）：
 *   - 所有层文件为 IIFE 注入 globalThis.FlagshipAI 命名空间
 *   - Node 与浏览器共用同一份 __loadOrder（src/index.js 与 scripts/build-demo.js 共读）
 *   - 加载顺序自底向上：L10 → L9 → … → L1
 * Phase 4（T05）：十层已落地，装配层经 NS.L0.assemble(NS) 产出 34 键公共 API（K15 契约）。
 */
'use strict';
const NS = (globalThis.FlagshipAI = globalThis.FlagshipAI || {});

// 自底向上加载顺序（K4）：L10 → L9 → … → L1。
// Phase 1（T02）：L10 微指令登记中心 + A/C 两域 192 项、L9 数值基础、L8 数值内核。
// Phase 2（T03）：追加 L7 张量/L6 后端/L5 自动微分/L4 神经算子/L3 架构组装；
//   legacy 的 forward+backward 单体已拆为 L4 算子 + L3 组装（index.js 覆写原型接管）。
// 后续 Phase（T04/T05）将追加 L2/L1。
// 加载顺序（K4）自底向上：L10 骨架 → L9 数值基础 → L8 数值内核 → L10 指令实现。
// 注意：A/C 两域微指令在模块求值期捕获 NS.L8.matmul/reduce 与 NS.L9.special
// （见各 A/C 文件顶部的 const 捕获），故 L8/L9 必须在其之前加载。L10 的
// registry/invariant 仅搭建骨架（不依赖 L8/L9 求值），故置于最前。
NS.__loadOrder = [
  // L10 · 微指令登记中心（最底层骨架：registry + invariant）
  './L10-microisa/registry.js',
  './L10-microisa/invariant.js',
  // L9 · 数值基础（A/C 域微指令的标量/函数依赖）
  './L9-numbase/rng.js',
  './L9-numbase/fp-ops.js',
  './L9-numbase/special.js',
  './L9-numbase/assert.js',
  // L8 · 数值内核（A/C 域微指令的矩阵/归约依赖）
  './L8-kernel/matmul.js',
  './L8-kernel/reduce.js',
  './L8-kernel/quantize.js',
  // L10 · 微指令实现（A/C 两域，依赖上方 L8/L9，故置于其后）
  './L10-microisa/A-scalar-000.js',
  './L10-microisa/A-scalar-001.js',
  './L10-microisa/A-scalar-002.js',
  './L10-microisa/A-scalar-003.js',
  './L10-microisa/C-matrix-000.js',
  './L10-microisa/C-matrix-001.js',
  './L10-microisa/C-matrix-002.js',
  './L10-microisa/C-matrix-003.js',
  './L10-microisa/B-vector-000.js',
  './L10-microisa/B-vector-001.js',
  './L10-microisa/B-vector-002.js',
  './L10-microisa/B-vector-003.js',
  './L10-microisa/E-shape-000.js',
  './L10-microisa/E-shape-001.js',
  './L10-microisa/E-shape-002.js',
  './L10-microisa/E-shape-003.js',
  // L10 · D 域（注意力基元，128 项）：打分/缩放、掩码、softmax·LSE、
  //   RoPE、ALiBi、温度与信息量、GQA/MQA 头映射、KV Cache 基元、
  //   追加/滚动/增量语义复合（D097–D128 第四批：Lipschitz 界、闭合式、
  //   平移不变、逐位一致、守卫激活率与变异 twin 全过）。
  './L10-microisa/D-attention-000.js',
  './L10-microisa/D-attention-001.js',
  './L10-microisa/D-attention-002.js',
  './L10-microisa/D-attention-003.js',
  // L10 · F 域（归一化，128 项）：均值/方差/L2 范数、LayerNorm/RMSNorm/
  //   BatchNorm/GroupNorm/权重标准化/LRN、min-max/z-score/L1/L∞ 归一化、
  //   logsumexp 高级/数值稳定、矩阵/批归一化（F001–F096 远端 v15.3 三批 +
  //   F097–F128 T06 第四批）。
  './L10-microisa/F-norm-000.js',
  './L10-microisa/F-norm-001.js',
  './L10-microisa/F-norm-002.js',
  './L10-microisa/F-norm-003.js',
  // L10 · G 域（激活与损失，128 项）：sigmoid/tanh/ReLU 家族/GELU/Swish/
  //   Mish/SELU 等、softmax/log_softmax、MSE/MAE/Huber/BCE/CE/KL/Focal、
  //   数值稳定交叉熵/温度缩放/gumbel-softmax（G001–G096 远端 v15.4 三批 +
  //   G097–G128 T06 第四批）。
  './L10-microisa/G-act-000.js',
  './L10-microisa/G-act-001.js',
  './L10-microisa/G-act-002.js',
  './L10-microisa/G-actloss-003.js',
  // L10 · H 域（采样与量化，128 项）：确定性采样/top-k/top-p、温度缩放、
  //   舍入、INT8 对称/仿射/逐通道量化、量化性质/端到端（T06 全量）。
  './L10-microisa/H-samplequant-000.js',
  './L10-microisa/H-samplequant-001.js',
  './L10-microisa/H-samplequant-002.js',
  './L10-microisa/H-samplequant-003.js',
  // L7 · 张量与内存（依赖 L8/L9；对外语义/KV/存档用裸 ndarray 四元组）
  './L7-tensor/tensor.js',
  './L7-tensor/shape.js',
  './L7-tensor/arena.js',
  './L7-tensor/kv-cache.js',
  // L6 · 后端运行时（kernel 注册/派发 + profile；P0 默认 CPU(JS) exact 通道）
  './L6-backend/kernel-registry.js',
  './L6-backend/backend.js',
  './L6-backend/backend-cpu.js',
  // L5 · 自动微分（Path A：VJP 注册中心 + tape 契约 + 梯度检验工厂；P1 再长 IR）
  './L5-autograd/vjp-registry.js',
  './L5-autograd/tape.js',
  './L5-autograd/gradcheck.js',
  // L4 · 神经算子（依赖 L8/L9；内部顺序：activation 先于 moe）
  './L4-ops/norm.js',
  './L4-ops/activation.js',
  './L4-ops/positional.js',
  './L4-ops/softmax.js',
  './L4-ops/linear.js',
  './L4-ops/attention.js',
  './L4-ops/moe.js',
  './L4-ops/loss.js',
  './L4-ops/registry.js',
  // L3 · 架构组装（forward/backward 重组装 + GPTModel + presets）
  './L3-arch/blocks.js',
  './L3-arch/gpt-model.js',
  './L3-arch/presets.js',
  // L2 · 编排管线（T04 接管：tokenizer/train/generate/sampler/evaluate/persistence/config）
  './L2-pipeline/tokenizer.js',
  './L2-pipeline/train.js',
  './L2-pipeline/generate.js',
  './L2-pipeline/sampler.js',
  './L2-pipeline/evaluate.js',
  './L2-pipeline/persistence.js',
  './L2-pipeline/config.js',
  // L1 · 应用层（T05）：浏览器演示应用的 UI 编排。仅在求值期定义 NS.L1.app，
  //   不触碰 DOM（DOM 操作都在 L1.app.start() 调用期），故 Node 侧加载亦安全。
  './L1-app/demo-app.js',
  // L0 · 装配层（T05）：最上层，需在所有层加载完毕后把十层装配为 34 键公共 API。
  //   Node（src/index.js）与浏览器（scripts/build-demo.js 内联）共用本文件，
  //   是 34 键映射 + K15 逐名硬门禁 + 注册中心自举的唯一真源；装配动作由调用方
  //   显式触发（NS.L0.assemble(ns)），不在模块求值期读取未加载的层。
  //   注：L1 也列入本清单（见上一节）—— 实测 demo-app.js 在纯 Node 下求值安全
  //   （DOM 访问全部位于 start() 之后调用的函数内），列入后浏览器端与 Node 端
  //   共用同一份加载顺序真源，无需在 build-demo.js 里另写一份内联清单。
  './L0-boot/assemble.js',
];

// 命名空间真源导出：供 src/index.js 与 scripts/build-demo.js 读取 __loadOrder
// 并在全部层加载后调用 NS.L0.assemble(NS) 装配出 34 键公共 API（K15 契约）。
module.exports = NS;

#!/usr/bin/env node
/* ============================================================================
 * 旗舰 AI v12 · 命令行工具（CLI）
 * ----------------------------------------------------------------------------
 * 零依赖（仅 Node 内置 fs / path），手写参数解析，不引任何第三方包。
 * 所有能力均经 require('../src/index.js') 取十层装配产物（34 键公共 API），
 * 因此 CLI 与 Node API、浏览器 demo 跑的是同一份实现 —— 不存在第二套代码路径。
 *
 * 用法：
 *   node scripts/cli.js <command> [options]
 *   node scripts/cli.js help [command]
 *
 * 命令：
 *   train     训练并可选存档   --preset --steps --lr --seed --out --corpus --text
 *                              --tokenizer --vocab --block --nlayer --nhead --dmodel
 *   generate  载入存档后生成   --model --prompt --max --temp --topk --topp --seed
 *   save      构建未训练模型并存档（含分词器）
 *   load      读档并验证存档无损往返
 *   isa       打印微指令覆盖率（A/B/C/E 四域 + 合计）
 *   info      打印模型配置与参数量（读档或按配置构建后统计）
 *   version   打印版本与层级装配信息
 * ========================================================================*/
'use strict';
const fs = require('fs');
const path = require('path');

const PROG = 'flagship';
const VERSION = require('../package.json').version;

/* ------------------------------ 参数解析 ------------------------------ */

/** 把 argv 解析为 { flags, rest }。支持 `--k=v` / `--k v` / `-k v` / `--flag`（布尔）。
 *  @param {string[]} argv 参数数组（不含 node 与脚本名）
 *  @param {Object<string,'string'|'boolean'>} spec 键 → 取值类型（用于区分布尔开关）
 *  @returns {{flags:Object, rest:string[]}}
 */
function parseArgs(argv, spec) {
  const flags = {};
  const rest = [];
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--') { rest.push(...argv.slice(i + 1)); break; }
    if (a.startsWith('--')) {
      const eq = a.indexOf('=');
      if (eq > -1) {
        const key = a.slice(2, eq);
        flags[key] = a.slice(eq + 1);
        continue;
      }
      const key = a.slice(2);
      const next = argv[i + 1];
      if (spec[key] === 'boolean' || next === undefined || next.startsWith('-')) {
        flags[key] = 'true';
      } else {
        flags[key] = next;
        i++;
      }
      continue;
    }
    if (a.startsWith('-') && a.length > 1) {
      const key = a.slice(1);
      const next = argv[i + 1];
      if (spec[key] === 'boolean' || next === undefined || next.startsWith('-')) {
        flags[key] = 'true';
      } else {
        flags[key] = next;
        i++;
      }
      continue;
    }
    rest.push(a);
  }
  return { flags: flags, rest: rest };
}

/** 取数值参数：缺省回落到 def，非法值报错退出。
 *  @param {Object} flags 解析后的 flags
 *  @param {string} key 参数名
 *  @param {number} def 默认值
 *  @returns {number}
 */
function num(flags, key, def) {
  const raw = flags[key];
  if (raw === undefined) return def;
  const v = Number(raw);
  if (!Number.isFinite(v)) die(`参数 --${key} 需要数字，收到：${raw}`);
  return v;
}

/** 打印错误并退出（非零状态码）。
 *  @param {string} msg 错误信息
 */
function die(msg) {
  console.error(`❌ ${msg}`);
  process.exit(1);
}

/* ------------------------------ 通用工具 ------------------------------ */

/** 载入十层装配产物（34 键公共 API）。统一入口，保证 CLI 与 index.js 同路径。
 *  @returns {Object} 34 键公共 API
 */
function loadApi() {
  try {
    return require('../src/index.js');
  } catch (e) {
    die('无法加载十层实现：' + e.message);
    return null;
  }
}

/** 读取语料：优先 --text，其次 --corpus 文件，最后回落到内置中文语料。
 *  @param {Object} flags 解析后的 flags
 *  @returns {string}
 */
function readCorpus(flags) {
  if (flags.text !== undefined) return String(flags.text);
  if (flags.corpus !== undefined) {
    const p = path.resolve(process.cwd(), flags.corpus);
    if (!fs.existsSync(p)) die(`语料文件不存在：${p}`);
    return fs.readFileSync(p, 'utf8');
  }
  return require('../src/corpus.js').CORPUS;
}

/** 按 flags 与语料构建分词器。
 *  @param {Object} F 34 键公共 API
 *  @param {Object} flags 解析后的 flags
 *  @param {string} corpus 语料原文
 *  @returns {Object} Tokenizer 或 BPETokenizer 实例
 */
function buildTokenizer(F, flags, corpus) {
  const mode = flags.tokenizer || 'char';
  if (mode === 'char') return new F.Tokenizer(corpus);
  const byteLevel = mode === 'bpe-byte';
  return new F.BPETokenizer(corpus, num(flags, 'vocab', 320), { byteLevel: byteLevel });
}

/** 组装模型配置（含架构预设）。
 *  @param {Object} F 34 键公共 API
 *  @param {Object} flags 解析后的 flags
 *  @param {number} vocabSize 词表大小
 *  @returns {Object} 模型配置
 */
function buildConfig(F, flags, vocabSize) {
  let cfg = {
    vocabSize: vocabSize,
    blockSize: num(flags, 'block', 64),
    nLayer: num(flags, 'nlayer', 2),
    nHead: num(flags, 'nhead', 4),
    dModel: num(flags, 'dmodel', 64),
    seed: num(flags, 'seed', 42),
  };
  const preset = flags.preset || 'gpt2';
  cfg = F.applyPreset(cfg, preset);
  cfg.seed = num(flags, 'seed', 42); // applyPreset 不改 seed，显式保证优先
  return cfg;
}

/** 读档：返回 { model, tok }。
 *  @param {Object} F 34 键公共 API
 *  @param {string} file 存档路径
 *  @returns {{model:Object, tok:Object, json:string}}
 */
function readModelFile(F, file) {
  const p = path.resolve(process.cwd(), file);
  if (!fs.existsSync(p)) die(`模型文件不存在：${p}`);
  const json = fs.readFileSync(p, 'utf8');
  let model;
  let tok;
  try {
    model = F.loadModel(json);
    tok = F.loadTokenizer(json);
  } catch (e) {
    die(`读档失败：${e.message}`);
  }
  if (!tok) die('存档缺少词表，无法定位分词器');
  return { model: model, tok: tok, json: json };
}

/* ------------------------------ 子命令 ------------------------------ */

/** train：构建模型 → 训练 → 可选存档。 */
function cmdTrain(argv) {
  const { flags } = parseArgs(argv, {});
  const F = loadApi();
  const corpus = readCorpus(flags);
  if (corpus.replace(/\s/g, '').length < 10) die('语料太短（至少 10 个非空白字符）');
  const tok = buildTokenizer(F, flags, corpus);
  const cfg = buildConfig(F, flags, tok.vocabSize);
  const model = new F.GPTModel(cfg);
  const steps = num(flags, 'steps', 300);
  const lr = num(flags, 'lr', 3e-3);
  const seed = num(flags, 'seed', 42);
  console.log(`🧠 模型：${flags.preset || 'gpt2'} · 词表 ${tok.vocabSize} · ${cfg.nLayer} 层 · ${cfg.nHead} 头 · d=${cfg.dModel} · 参数 ${model.paramCount().toLocaleString()}`);
  const t0 = Date.now();
  const res = F.train(model, tok.encode(corpus), {
    steps: steps, lr: lr, batch: cfg.blockSize, warmup: Math.min(30, Math.floor(steps / 5)), seed: 0,
  });
  const ms = Date.now() - t0;
  console.log(`🏋️ 训练 ${steps} 步完成（lr=${lr}，${ms} ms）：final loss ${res.finalLoss.toFixed(4)}`);
  writeModelMaybe(F, flags, model, tok);
  return 0;
}

/** save：构建未训练模型并存档（含分词器元数据）。 */
function cmdSave(argv) {
  const { flags } = parseArgs(argv, {});
  const F = loadApi();
  const corpus = readCorpus(flags);
  const tok = buildTokenizer(F, flags, corpus);
  const cfg = buildConfig(F, flags, tok.vocabSize);
  const model = new F.GPTModel(cfg);
  writeModelMaybe(F, flags, model, tok);
  return 0;
}

/** 若指定 --out 则写档；否则打印提示。
 *  @param {Object} F 34 键公共 API
 *  @param {Object} flags 解析后的 flags
 *  @param {Object} model 模型
 *  @param {Object} tok 分词器
 */
function writeModelMaybe(F, flags, model, tok) {
  if (flags.out === undefined) {
    console.log('ℹ️  未指定 --out，跳过存档');
    return;
  }
  const p = path.resolve(process.cwd(), flags.out);
  const json = F.saveModel(model, tok);
  fs.mkdirSync(path.dirname(p), { recursive: true });
  fs.writeFileSync(p, json);
  console.log(`💾 已存档：${p}（${(Buffer.byteLength(json) / 1024).toFixed(1)} KB，含词表 ${tok.vocabSize}）`);
}

/** generate：读档生成。 */
function cmdGenerate(argv) {
  const { flags } = parseArgs(argv, {});
  if (flags.model === undefined) die('generate 需要 --model <存档.json>（先用 train/save 产出）');
  const F = loadApi();
  const { model, tok } = readModelFile(F, flags.model);
  const prompt = flags.prompt !== undefined ? String(flags.prompt) : '';
  const opts = {
    maxNew: num(flags, 'max', 100),
    temperature: num(flags, 'temp', 0.8),
    topK: num(flags, 'topk', 0),
    topP: num(flags, 'topp', 0),
    seed: num(flags, 'seed', 1234),
  };
  const r = F.generate(model, tok, prompt, opts);
  console.log(r.text);
  console.log(`\n— 生成 ${r.ids.length} token（温度 ${opts.temperature} · top-k ${opts.topK} · top-p ${opts.topP} · seed ${opts.seed}）`);
  return 0;
}

/** load：读档并验证存档无损往返（再序列化逐字符相等）。 */
function cmdLoad(argv) {
  const { flags } = parseArgs(argv, {});
  if (flags.model === undefined) die('load 需要 --model <存档.json>');
  const F = loadApi();
  const { model, tok, json } = readModelFile(F, flags.model);
  const re = F.saveModel(model, tok);
  const same = re === json;
  console.log(`✅ 读档成功：${model.paramCount().toLocaleString()} 参数 · 词表 ${tok.vocabSize}`);
  console.log(`   存档体积 ${(Buffer.byteLength(json) / 1024).toFixed(1)} KB · 量化=${!!model.quantized}`);
  console.log(`   无损往返：${same ? '✅ 逐字符一致' : '❌ 再序列化不一致'}`);
  return same ? 0 : 1;
}

/** info：打印模型配置与参数量。 */
function cmdInfo(argv) {
  const { flags } = parseArgs(argv, {});
  const F = loadApi();
  let model;
  if (flags.model !== undefined) {
    model = readModelFile(F, flags.model).model;
  } else {
    const corpus = readCorpus(flags);
    const tok = buildTokenizer(F, flags, corpus);
    model = new F.GPTModel(buildConfig(F, flags, tok.vocabSize));
  }
  const cfg = model.cfg;
  const rows = [
    ['架构预设', flags.preset || (flags.model ? '(来自存档)' : 'gpt2')],
    ['norm / ffn / pos', `${cfg.norm} / ${cfg.ffn} / ${cfg.pos}`],
    ['词表 vocabSize', cfg.vocabSize],
    ['上下文 blockSize', cfg.blockSize],
    ['层数 nLayer', cfg.nLayer],
    ['注意力头 nHead', cfg.nHead],
    ['KV 头 nKvHead', cfg.KV !== undefined ? cfg.KV : cfg.nHead],
    ['模型维 dModel', cfg.dModel],
    ['专家数 nExp', cfg.nExp || 1],
    ['随机种子 seed', cfg.seed],
  ];
  printTable(rows);
  const n = model.paramCount();
  console.log(`\n参数量：${n.toLocaleString()}（≈ ${(n * 8 / 1024 / 1024).toFixed(2)} MB，Float64）`);
  return 0;
}

/** isa：打印微指令登记中心覆盖率。 */
function cmdIsa(argv) {
  parseArgs(argv, {});
  loadApi(); // 生产加载路径：填充 globalThis.FlagshipAI
  const NS = globalThis.FlagshipAI;
  const s = NS.L10.stats();
  console.log(`🧩 微指令覆盖：A ${s.A.impl}/${s.A.capacity} · B ${s.B.impl}/${s.B.capacity} · C ${s.C.impl}/${s.C.capacity} · E ${s.E.impl}/${s.E.capacity} · 合计 ${s.total.impl}/${s.total.capacity}`);
  for (const d of NS.L10.DOMAINS) {
    if (['A', 'B', 'C', 'E'].indexOf(d) > -1) continue;
    console.log(`   ${d} ${s[d].impl}/${s[d].capacity}`);
  }
  const ok = s.total.impl === 384 && s.total.capacity === 1024
    && s.A.impl === 96 && s.B.impl === 96 && s.C.impl === 96 && s.E.impl === 96;
  console.log(ok ? '✅ 覆盖率符合 T05 契约（384/1024，四域各 96）'
    : '❌ 覆盖率与契约不符（期望 384/1024，四域各 96）');
  return ok ? 0 : 1;
}

/** version：打印版本与装配信息。 */
function cmdVersion() {
  loadApi();
  const NS = globalThis.FlagshipAI;
  const layers = Object.keys(NS).filter((k) => /^L\d+/.test(k)).sort(cmpLayer);
  console.log(`${PROG} v${VERSION}`);
  console.log(`已装配层：${layers.join(' · ')}`);
  console.log(`公共 API：${Object.keys(NS.L0.API).length} 键`);
  return 0;
}

/** 层名排序：L10 视为最底层，排在 L2 之前。
 *  @param {string} a 层名
 *  @param {string} b 层名
 *  @returns {number}
 */
function cmpLayer(a, b) {
  const na = Number(a.slice(1).replace(/\D/g, '')) || 0;
  const nb = Number(b.slice(1).replace(/\D/g, '')) || 0;
  return na - nb;
}

/** 计算字符串的终端显示宽度（CJK / 全角标点的字形宽度为 2 列）。
 *  @param {string} s 输入字符串
 *  @returns {number} 显示宽度
 */
function displayWidth(s) {
  let w = 0;
  for (const ch of String(s)) {
    const c = ch.codePointAt(0);
    const wide = (c >= 0x1100 && c <= 0x115f) || (c >= 0x2e80 && c <= 0xa4cf)
      || (c >= 0xac00 && c <= 0xd7a3) || (c >= 0xf900 && c <= 0xfaff)
      || (c >= 0xfe30 && c <= 0xfe6f) || (c >= 0xff00 && c <= 0xff60)
      || (c >= 0xffe0 && c <= 0xffe6);
    w += wide ? 2 : 1;
  }
  return w;
}

/** 打印两列表格（按显示宽度对齐，中文标签不歪）。
 *  @param {Array<Array<string|number>>} rows 行数据
 */
function printTable(rows) {
  const w = Math.max(...rows.map((r) => displayWidth(r[0])));
  for (const r of rows) {
    const pad = ' '.repeat(w - displayWidth(r[0]));
    console.log(`   ${r[0]}${pad} : ${r[1]}`);
  }
}

const USAGE = [
  `旗舰 AI v${VERSION} · 命令行工具`,
  '',
  '用法：node scripts/cli.js <command> [options]',
  '',
  '命令：',
  '  train     构建 + 训练 + 存档   --preset --steps --lr --seed --out --tokenizer --vocab',
  '                                --block --nlayer --nhead --dmodel --corpus/--text',
  '  generate  读档生成             --model --prompt --max --temp --topk --topp --seed',
  '  save      构建未训练模型并存档 --out --preset --seed --corpus/--text',
  '  load      读档 + 无损往返校验  --model',
  '  info      打印配置与参数量     [--model] | [--preset --block --nlayer --nhead --dmodel]',
  '  isa       打印微指令覆盖率',
  '  version   打印版本与已装配层',
  '  help      显示本帮助',
  '',
  '示例：',
  '  node scripts/cli.js train --preset llama --steps 200 --out model.json',
  '  node scripts/cli.js generate --model model.json --prompt 旗舰之道 --max 80 --temp 0.8',
  '  node scripts/cli.js info --model model.json',
].join('\n');

const COMMANDS = {
  train: cmdTrain,
  generate: cmdGenerate,
  save: cmdSave,
  load: cmdLoad,
  info: cmdInfo,
  isa: cmdIsa,
  version: cmdVersion,
};

/** 主入口。
 *  @param {string[]} argv process.argv.slice(2)
 *  @returns {number} 退出码
 */
function main(argv) {
  if (argv.length === 0 || argv[0] === 'help' || argv[0] === '--help' || argv[0] === '-h') {
    console.log(USAGE);
    return 0;
  }
  const cmd = argv[0];
  const fn = COMMANDS[cmd];
  if (!fn) {
    console.error(`❌ 未知命令：${cmd}\n`);
    console.log(USAGE);
    return 1;
  }
  return fn(argv.slice(1));
}

if (require.main === module) {
  process.exitCode = main(process.argv.slice(2));
}

module.exports = { main: main, parseArgs: parseArgs };

/* L2 编排管线 · 模型持久化（无损 JSON 存档 / 读档）
 * 所属层：L2（编排管线）｜ 允许依赖：L2.tokenizer（恢复分词器）、L3.gpt_model（重建模型）
 * 主要 @ref：GPT-2 checkpoint 序列化；Int8 对称量化权重（{q, s}）
 * 迁移方式：从 legacy flagship-engine.v11.js 第 1530–1594 行【原样搬】（Δ=0 契约）：
 *   - saveModel（1535–1551）逐字搬，零改写。
 *   - loadTokenizer（1553–1574）：Object.create(Tokenizer.prototype) / Object.create(BPETokenizer.prototype)
 *     改指 NS.L2.tokenizer.Tokenizer / NS.L2.tokenizer.BPETokenizer（L2.tokenizer 先于本文件加载）。
 *   - loadModel（1576–1594）：new GPTModel(obj.cfg) -> new NS.L3.gpt_model.GPTModel(obj.cfg)
 *     （读档模型即 L3 模型，覆盖 L3 Δ=0 存档路径；L3 先于 L2 加载）。
 */
;(function (NS) {
  NS.L2 = NS.L2 || {};
  NS.L2.persistence = NS.L2.persistence || {};

  /** 序列化模型。一并存词表（字符数组），保证导入后可完整重建分词器。
   * 量化权重存为 {q:[...], s:scale}（Int8），未量化存为普通浮点数组。 */
  function saveModel(model, tok) {
    const params = {};
    let quantized = false;
    for (const key in model.params) {
      const p = model.params[key];
      if (p.q) { params[key] = { q: Array.from(p.q), s: p.scale }; quantized = true; }
      else params[key] = Array.from(p.d);
    }
    return JSON.stringify({
      format: 'flagship-ai', version: 2,
      cfg: model.cfg, params, quantized,
      vocab: tok ? tok.chars : null,
      tokKind: tok ? (tok.kind || 'char') : null,
      merges: tok && tok.kind === 'bpe' ? tok.merges : null,
      bpeByteLevel: tok && tok.kind === 'bpe' ? !!tok.byteLevel : null,
    });
  }

  /** 从存档恢复分词器（存档无词表时返回 null） */
  function loadTokenizer(json) {
    const obj = typeof json === 'string' ? JSON.parse(json) : json;
    if (!obj || !Array.isArray(obj.vocab)) return null;
    if (obj.tokKind === 'bpe') {
      const t = Object.create(NS.L2.tokenizer.BPETokenizer.prototype);
      t.kind = 'bpe';
      t.chars = obj.vocab;
      t.itos = obj.vocab;
      t.stoi = new Map(obj.vocab.map((s, i) => [s, i]));
      t.merges = obj.merges || [];
      t.byteLevel = obj.bpeByteLevel !== false;
      t.ranks = new Map(t.merges.map((m, i) => [m[0] + m[1], i]));
      return t;
    }
    const t = Object.create(NS.L2.tokenizer.Tokenizer.prototype);
    t.kind = 'char';
    t.chars = obj.vocab;
    t.stoi = new Map(obj.vocab.map((ch, i) => [ch, i]));
    t.itos = obj.vocab;
    return t;
  }

  function loadModel(json) {
    const obj = typeof json === 'string' ? JSON.parse(json) : json;
    if (!obj || obj.format !== 'flagship-ai') throw new Error('无法识别的模型格式');
    const m = new NS.L3.gpt_model.GPTModel(obj.cfg);
    for (const key in obj.params) {
      const src = obj.params[key];
      if (Array.isArray(src)) {
        const dst = m.params[key];
        if (!dst) throw new Error(`模型缺少参数 ${key}`);
        if (src.length !== dst.d.length) throw new Error(`参数 ${key} 长度不匹配`);
        dst.d.set(src);
      } else {
        // 量化权重：{q, s}
        m.params[key] = { q: Int8Array.from(src.q), scale: src.s };
      }
    }
    if (obj.quantized) m.quantized = true;
    return m;
  }

  NS.L2.persistence.saveModel = saveModel;
  NS.L2.persistence.loadModel = loadModel;
  NS.L2.persistence.loadTokenizer = loadTokenizer;
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

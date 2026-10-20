/* L2 编排管线 · 分词器（字符级 Tokenizer + GPT-2 字节级 BPETokenizer）
 * 所属层：L2（编排管线）｜ 允许依赖：无（自包含，零跨层调用）
 * 主要 @ref：GPT-2 byte-level BPE（Radford et al. 2019）；tiktoken
 * 迁移方式：从 legacy flagship-engine.v11.js 第 523–697 行【原样搬】（Δ=0 契约）。
 *   - Tokenizer / BPETokenizer / BYTE_ENC / BYTE_DEC / PRE_RE /
 *     textToByteChars / byteCharsToText 逐字搬运，无任何改写。
 */
;(function (NS) {
  NS.L2 = NS.L2 || {};
  NS.L2.tokenizer = NS.L2.tokenizer || {};

  class Tokenizer {
    constructor(text) {
      const chars = [...new Set([...String(text)])].sort();
      this.kind = 'char';
      this.chars = chars;
      this.stoi = new Map(chars.map((ch, i) => [ch, i]));
      this.itos = chars;
    }
    get vocabSize() { return this.chars.length; }
    encode(text) {
      const out = [];
      for (const ch of String(text)) {
        const id = this.stoi.get(ch);
        if (id !== undefined) out.push(id); // 未知字符静默跳过（字符级词表的既定行为）
      }
      return out;
    }
    decode(ids) {
      let s = '';
      for (const id of ids) s += this.itos[id] ?? '';
      return s;
    }
  }

  const BYTE_ENC = (() => {
    const bs = [];
    for (let b = 0x21; b <= 0x7e; b++) bs.push(b);   // '!' ~ '~'
    for (let b = 0xa1; b <= 0xac; b++) bs.push(b);   // '¡' ~ '¬'
    for (let b = 0xae; b <= 0xff; b++) bs.push(b);   // '®' ~ 'ÿ'
    const map = new Map();
    let n = 0;
    for (const b of bs) map.set(b, String.fromCharCode(b));
    for (let b = 0; b < 256; b++) {
      if (!map.has(b)) map.set(b, String.fromCharCode(256 + n++));
    }
    return map;
  })();
  const BYTE_DEC = new Map([...BYTE_ENC].map(([b, c]) => [c, b]));

  // GPT-2 预分词正则（原文照搬）：缩写 | 可选空格+字母 | 可选空格+数字 | 可选空格+其它非空白 | 尾随空白
  const PRE_RE = new RegExp(
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+", 'gu');

  /** UTF-8 文本 -> 字节级字符序列（可逆） */
  function textToByteChars(s) {
    const bytes = new TextEncoder().encode(s);
    let out = '';
    for (let i = 0; i < bytes.length; i++) out += BYTE_ENC.get(bytes[i]);
    return out;
  }
  /** 字节级字符序列 -> UTF-8 文本（可逆） */
  function byteCharsToText(s) {
    const bytes = new Uint8Array(s.length);
    for (let i = 0; i < s.length; i++) bytes[i] = BYTE_DEC.get(s[i]) ?? 63;
    return new TextDecoder().decode(bytes);
  }

  class BPETokenizer {
    /**
     * @param {string} text 训练语料
     * @param {number} vocabSize 目标词表大小（字节级下 ≥256）
     * @param {Object} opts
     *   minPairFreq: 最低合并频次（低于则提前停止，默认 2）
     *   byteLevel:   基元选择 —— true = 256 字节（GPT-2 标准，覆盖任意 UTF-8）；
     *                 false = 语料字符（中文语料下显著更省 token，且不会生成半截 UTF-8）
     */
    constructor(text, vocabSize = 512, opts) {
      const o = Object.assign({ minPairFreq: 2, byteLevel: true }, opts || {});
      this.kind = 'bpe';
      this.byteLevel = !!o.byteLevel;
      const target = Math.max(this.byteLevel ? 256 : 1, vocabSize | 0);
      // 词块 <-> 符号序列（字节级要多一层 UTF-8 编解码）
      const toSym = (w) => (this.byteLevel ? [...textToByteChars(w)] : [...w]);

      // 1) 预分词并统计词频
      const freq = new Map();
      for (const m of String(text).matchAll(PRE_RE)) {
        const w = m[0];
        freq.set(w, (freq.get(w) || 0) + 1);
      }
      // 2) 每个词块转成符号数组（BPE 只在块内合并）
      const seqs = [];
      for (const [w, n] of freq) seqs.push({ s: toSym(w), n });

      // 3) 初始词表：字节级 = 256 个字节基元；字符级 = 语料中的全部字符
      const vocab = [];
      if (this.byteLevel) { for (let b = 0; b < 256; b++) vocab.push(BYTE_ENC.get(b)); }
      else { vocab.push(...[...new Set([...String(text)])].sort()); }
      if (vocab.length > target) vocab.length = target;
      const merges = []; // [[a, b], ...]，下标即 merge rank

      // 4) 频率合并：每轮统计相邻对频次，合并最高频者
      while (vocab.length < target) {
        const cnt = new Map();
        for (const { s, n } of seqs) {
          for (let i = 0; i + 1 < s.length; i++) {
            const key = s[i] + '\u0000' + s[i + 1];
            cnt.set(key, (cnt.get(key) || 0) + n);
          }
        }
        let best = null, bestN = o.minPairFreq - 1;
        for (const [k, n] of cnt) if (n > bestN) { bestN = n; best = k; }
        if (!best) break; // 没有再值得合并的对
        const parts = best.split('\u0000');
        const a = parts[0], b = parts[1], tok = a + b;
        vocab.push(tok);
        merges.push([a, b]);
        for (const seq of seqs) {
          const out = [];
          let i = 0;
          while (i < seq.s.length) {
            if (i + 1 < seq.s.length && seq.s[i] === a && seq.s[i + 1] === b) { out.push(tok); i += 2; }
            else { out.push(seq.s[i]); i++; }
          }
          seq.s = out;
        }
      }

      this.chars = vocab;
      this.merges = merges;
      this.itos = vocab;
      this.stoi = new Map(vocab.map((t, i) => [t, i]));
      this.ranks = new Map(merges.map((m, i) => [m[0] + m[1], i]));
    }

    get vocabSize() { return this.chars.length; }

    /** 文本 -> id 序列。逐词块用 merge-rank 贪心合并（等价 GPT-2/tiktoken） */
    encode(text) {
      const out = [];
      for (const m of String(text).matchAll(PRE_RE)) {
        let s = this.byteLevel ? [...textToByteChars(m[0])] : [...m[0]];
        while (s.length > 1) {
          let bestRank = Infinity, bestI = -1;
          for (let i = 0; i + 1 < s.length; i++) {
            const r = this.ranks.get(s[i] + s[i + 1]);
            if (r !== undefined && r < bestRank) { bestRank = r; bestI = i; }
          }
          if (bestI < 0) break;
          s.splice(bestI, 2, s[bestI] + s[bestI + 1]);
        }
        for (const t of s) {
          const id = this.stoi.get(t);
          if (id !== undefined) out.push(id);
          else for (const ch of t) { // 兜底：拆到字节基元（理论上不会触发）
            const cid = this.stoi.get(ch);
            if (cid !== undefined) out.push(cid);
          }
        }
      }
      return out;
    }

    /** id 序列 -> 文本。字节级先拼全部字节再一次性 UTF-8 解码，跨 token 汉字无损 */
    decode(ids) {
      let s = '';
      for (const id of ids) s += this.itos[id] ?? '';
      return this.byteLevel ? byteCharsToText(s) : s;
    }
  }

  NS.L2.tokenizer.Tokenizer = Tokenizer;
  NS.L2.tokenizer.BPETokenizer = BPETokenizer;
  NS.L2.tokenizer.textToByteChars = textToByteChars;
  NS.L2.tokenizer.byteCharsToText = byteCharsToText;
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

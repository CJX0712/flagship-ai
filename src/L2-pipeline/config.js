/* L2 编排管线 · 运行配置（确定性 spec / 哈希，复现性支持）
 * 所属层：L2（编排管线）｜ 允许依赖：无（自包含，零跨层调用）
 * 主要 @ref：FNV-1a 哈希；可复现训练配方（seed + 语料哈希 + 架构配置）
 *
 * 【注意】本文件是 NEW 代码（legacy 无对应物），不参与 Δ=0 契约。
 *   用途：把一次运行的关键决定因素（seed / 语料哈希 / cfg）压成一个稳定字符串，
 *   供复现、缓存键、实验登记使用。纯函数、无时钟、无随机源（符合 K5 确定性要求）。
 */
;(function (NS) {
  NS.L2 = NS.L2 || {};
  NS.L2.config = NS.L2.config || {};

  /** FNV-1a 32-bit 哈希（UTF-16 码元逐字混合）-> base36 字符串 */
  function hashString(s) {
    let h = 0x811c9dc5;
    const str = String(s);
    for (let i = 0; i < str.length; i++) {
      h ^= str.charCodeAt(i);
      h = Math.imul(h, 0x01000193);
    }
    return (h >>> 0).toString(36);
  }

  /** 稳定序列化：对象键按字典序递归排序，数组保序 —— 同内容同字符串 */
  function canonicalJson(v) {
    if (v === null || typeof v !== 'object') return JSON.stringify(v);
    if (Array.isArray(v)) return '[' + v.map(canonicalJson).join(',') + ']';
    const keys = Object.keys(v).sort();
    return '{' + keys.map((k) => JSON.stringify(k) + ':' + canonicalJson(v[k])).join(',') + '}';
  }

  /**
   * 生成一次运行的稳定 spec（复现凭据）。
   * @param {Object} spec { seed, corpusHash, cfg }
   * @returns {string} 形如 'flagship-spec|{...canonical...}|<fnv1a-base36>'
   *   同 {seed, corpusHash, cfg} ⇒ 同字符串（键序无关）；任一异 ⇒ 哈希必异。
   */
  function runSpec(spec) {
    const o = spec || {};
    const canon = canonicalJson({
      seed: o.seed ?? null,
      corpusHash: o.corpusHash ?? null,
      cfg: o.cfg ?? null,
    });
    return 'flagship-spec|' + canon + '|' + hashString(canon);
  }

  NS.L2.config.hashString = hashString;
  NS.L2.config.runSpec = runSpec;
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

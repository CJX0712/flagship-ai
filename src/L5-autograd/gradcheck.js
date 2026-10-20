/* ============================================================================
 * L5 自动微分 · 中心差分梯度检验工厂（gradcheck）
 * ----------------------------------------------------------------------------
 * 所属层：L5（自动微分）｜ 允许依赖：无（纯数值工具，不触碰任何层的实现）
 * 主要 @ref：中心差分 (f(x+h)−f(x−h)) / 2h，截断误差 O(h²)；
 *            Baydin et al. 2018 §4（有限差分校验是自动微分正确性的标准证据）
 *
 * 用途（验收 ⑥）：对每个已登记 VJP 的算子构造标量损失
 *     loss(x) = Σ fwd(x) ⊙ w   （w 为固定上游梯度）
 * 其解析梯度 anaGrad(x) 即该算子的 backward；centralDiff 逐元素给出数值
 * 梯度，二者比对 err_i = |ana−num| / max(1, |num|)，maxErr = max_i err_i。
 * 接受判据 pass ⟺ maxErr ≤ tol（默认 4.22e-4；h=1e-6 下光滑算子实测
 * ~1e-7..1e-9，远在界内——若某算子超界即暴露其解析反向存在笔误）。
 * ==========================================================================*/
;(function (NS) {
  'use strict';
  NS.L5 = NS.L5 || {};
  NS.L5.gradcheck = NS.L5.gradcheck || {};

  // 默认中心差分步长（截断 O(h²) 与舍入 O(ε/h) 的折中：1e-6 是 Float64 甜点）
  const DEFAULT_H = 1e-6;
  // 默认接受容差（验收 ⑥：err/tol ≤ 4.22e-4 且不得恶化）
  const DEFAULT_TOL = 4.22e-4;

  /**
   * 单元素中心差分：∂f/∂x_i ≈ (f(x+h·e_i) − f(x−h·e_i)) / (2h)。
   * @param f  Float64Array -> number（标量损失）
   * @param x  Float64Array（基点）
   * @param i  扰动下标
   * @param h  步长，缺省 1e-6
   */
  function centralDiff(f, x, i, h) {
    if (h == null) h = DEFAULT_H;
    const xp = new Float64Array(x);
    const xm = new Float64Array(x);
    xp[i] = x[i] + h;
    xm[i] = x[i] - h;
    return (f(xp) - f(xm)) / (2 * h);
  }

  /**
   * 数值梯度检验：逐元素比对解析梯度与中心差分。
   * @param name      用例名（进入报告）
   * @param lossFn    (Float64Array) -> number
   * @param anaGradFn (Float64Array) -> Float64Array（解析 dLoss/dx）
   * @param x         Float64Array 基点
   * @param tol       容差，缺省 4.22e-4
   * @returns { name, maxErr, tol, pass, n, worst }
   *   err_i = |ana_i − num_i| / max(1, |num_i|)（绝对-相对混合，对 0 安全）；
   *   pass = maxErr ≤ tol。任一侧非有限（NaN/Inf/缺元素）记 err=Infinity -> FAIL。
   */
  function checkNumeric(name, lossFn, anaGradFn, x, tol) {
    if (tol == null) tol = DEFAULT_TOL;
    const ana = anaGradFn(x);
    let maxErr = 0;
    let worst = -1;
    for (let i = 0; i < x.length; i++) {
      const num = centralDiff(lossFn, x, i, DEFAULT_H);
      const a = ana ? ana[i] : NaN;
      let err;
      if (!Number.isFinite(num) || !Number.isFinite(a)) err = Infinity;
      else err = Math.abs(a - num) / Math.max(1, Math.abs(num));
      if (err > maxErr) { maxErr = err; worst = i; }
    }
    return { name, maxErr, tol, pass: maxErr <= tol, n: x.length, worst };
  }

  /**
   * 汇总多例 checkNumeric 结果为等宽小表（CI 日志一眼判定）。
   * @param report  checkNumeric 返回值数组
   * @returns { allPass, text }
   */
  function checkAll(report) {
    const list = Array.isArray(report) ? report : [report];
    let allPass = true;
    const lines = ['L5 gradcheck 汇总（tol 默认 4.22e-4）：'];
    for (const r of list) {
      if (!r || !r.pass) allPass = false;
      const e = r && Number.isFinite(r.maxErr) ? r.maxErr.toExponential(3) : String(r && r.maxErr);
      lines.push(`  ${r && r.pass ? 'PASS' : 'FAIL'}  ${r ? r.name : '?'}  maxErr=${e}  tol=${r ? r.tol : '?'}`);
    }
    lines.push(`  => ${allPass ? 'ALL PASS' : 'HAS FAILURES'}`);
    return { allPass, text: lines.join('\n') };
  }

  NS.L5.gradcheck.DEFAULT_H = DEFAULT_H;
  NS.L5.gradcheck.DEFAULT_TOL = DEFAULT_TOL;
  NS.L5.gradcheck.centralDiff = centralDiff;
  NS.L5.gradcheck.checkNumeric = checkNumeric;
  NS.L5.gradcheck.checkAll = checkAll;
})(globalThis.FlagshipAI = globalThis.FlagshipAI || {});

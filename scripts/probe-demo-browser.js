#!/usr/bin/env node
'use strict';
/**
 * 浏览器端可视化验证门禁（CI 集成 / 本地可跑）。
 *
 * 用 Playwright headless Chromium 真实加载 demo.html（file:// 协议直开，与用户双击
 * 打开完全同一条路径），断言：
 *   1) 十层栈装配完成，L10 微指令注册中心 1024/1024 已实现（八域各 128）；
 *   2) L10 微指令浏览器 DOM 渲染：八域摘要卡 + 列表首屏（pageSize=60 条）+ 计数 1024/1024；
 *   3) 交互 · 按域筛选：点「A」域 → 计数变 128/1024、列表项 key 以 A 开头、chip aria-pressed；
 *   4) 交互 · 批量试运行：全部 1024 项 × 8 种子谓词求值，✓ 1024/1024 项通过、失败 0；
 *   5) 产出截图 docs/demo-screenshot.png（README 头图 / CI artifact 可视化留痕）。
 *
 * 零运行时依赖：仅 devDependency `playwright`（不影响 package.json 的 dependencies=0）。
 * Chromium 解析顺序（任一命中即可，CI 由 `npx playwright install chromium` 提供托管浏览器）：
 *   a) 环境变量 PW_CHROMIUM_EXECUTABLE 指向的可执行文件；
 *   b) Playwright 托管浏览器（chromium.executablePath() 存在时用默认 launch）；
 *   c) 本机 ms-playwright 缓存目录扫描（chrome / chrome-headless-shell）。
 *
 * 用法：
 *   node scripts/probe-demo-browser.js            # 跑断言 + 生成截图
 *   node scripts/probe-demo-browser.js --no-shot  # 只跑断言，跳过截图
 */
const fs = require('fs');
const os = require('os');
const path = require('path');
const { chromium } = require('playwright');

const ROOT = path.resolve(__dirname, '..');
const DEMO = path.join(ROOT, 'demo.html');
const SHOT = path.join(ROOT, 'docs', 'demo-screenshot.png');
const wantShot = !process.argv.includes('--no-shot');

let failures = 0;
function ok(msg) { console.log('  ✅ ' + msg); }
function bad(msg) { console.log('  ❌ ' + msg); failures++; }
function check(cond, passMsg, failMsg) { if (cond) ok(passMsg); else bad(failMsg || passMsg); }

/** 在常见 ms-playwright 缓存根目录里找一个可用的 Chromium 可执行文件。 */
function scanCache() {
  const roots = [
    process.env.PLAYWRIGHT_BROWSERS_PATH,
    path.join(os.homedir(), 'AppData', 'Local', 'ms-playwright'), // Windows
    path.join(os.homedir(), '.cache', 'ms-playwright'),            // Linux
    path.join(os.homedir(), 'Library', 'Caches', 'ms-playwright'), // macOS
  ].filter(Boolean);
  const rels = [
    'chromium_headless_shell-*/chrome-headless-shell-win64/chrome-headless-shell.exe',
    'chromium-*/chrome-win64/chrome.exe',
    'chromium_headless_shell-*/chrome-linux/chrome-headless-shell',
    'chromium-*/chrome-linux/chrome',
    'chromium_headless_shell-*/chrome-mac/Chromium.app/Contents/MacOS/Chromium',
    'chromium-*/chrome-mac/Chromium.app/Contents/MacOS/Chromium',
  ];
  for (const root of roots) {
    if (!fs.existsSync(root)) continue;
    let entries = [];
    try { entries = fs.readdirSync(root); } catch (_) { continue; }
    for (const rel of rels) {
      // rel 形如 'chromium-*/chrome-win64/chrome.exe'：按前缀过滤目录，再拼后缀
      const star = rel.indexOf('*');
      const dirPrefix = rel.slice(0, star);
      const rest = rel.slice(star + 1); // '/chrome-win64/chrome.exe'
      const dirs = entries.filter((e) => e.startsWith(dirPrefix) && fs.statSync(path.join(root, e)).isDirectory())
        .sort().reverse(); // 取 revision 最大者
      for (const d of dirs) {
        const exe = path.join(root, d + rest);
        if (fs.existsSync(exe)) return exe;
      }
    }
  }
  return null;
}

/** 决定 launch 配置：优先显式/托管，其次缓存扫描。返回 {executablePath?}。 */
function resolveLaunch() {
  if (process.env.PW_CHROMIUM_EXECUTABLE && fs.existsSync(process.env.PW_CHROMIUM_EXECUTABLE)) {
    console.log('  · Chromium 来源：PW_CHROMIUM_EXECUTABLE = ' + process.env.PW_CHROMIUM_EXECUTABLE);
    return { executablePath: process.env.PW_CHROMIUM_EXECUTABLE };
  }
  try {
    const managed = chromium.executablePath();
    if (managed && fs.existsSync(managed)) {
      console.log('  · Chromium 来源：Playwright 托管 = ' + managed);
      return {}; // 用默认 launch
    }
  } catch (_) { /* 托管浏览器未安装，继续扫描 */ }
  const cached = scanCache();
  if (cached) {
    console.log('  · Chromium 来源：本机缓存扫描 = ' + cached);
    return { executablePath: cached };
  }
  return {}; // 让 playwright 抛出清晰的 "browser not found"
}

(async () => {
  if (!fs.existsSync(DEMO)) { console.error('❌ 未找到 demo.html，请先 npm run build'); process.exit(1); }

  const launchOpts = Object.assign({}, resolveLaunch(), { args: ['--no-sandbox', '--disable-dev-shm-usage'] });
  const browser = await chromium.launch(launchOpts);
  try {
    const page = await browser.newPage({ viewport: { width: 1360, height: 900 } });
    page.on('pageerror', (e) => console.log('  ⚠️ pageerror: ' + e.message));
    page.on('console', (m) => { if (m.type() === 'error') console.log('  ⚠️ console.error: ' + m.text()); });

    await page.goto('file://' + DEMO.replace(/\\/g, '/'));

    // 1) 十层栈装配 + L10 1024/1024
    await page.waitForFunction(
      () => { const NS = window.FlagshipNS || window.FlagshipAI; return !!(NS && NS.L10 && NS.L10.stats && NS.L10.stats().total.impl === 1024); },
      { timeout: 20000 });
    const st = await page.evaluate(() => {
      const NS = window.FlagshipNS || window.FlagshipAI;
      const s = NS.L10.stats();
      return { impl: s.total.impl, cap: s.total.capacity, doms: NS.L10.DOMAINS.map((d) => s[d].impl) };
    });
    check(st.impl === 1024 && st.cap === 1024 && st.doms.every((x) => x === 128),
      '十层栈装配完成 · L10 ' + st.impl + '/' + st.cap + '（八域各 128）',
      'L10 统计异常：impl=' + st.impl + ' cap=' + st.cap + ' doms=' + st.doms.join(','));

    // 2) L10 浏览器 DOM 渲染
    const count0 = (await page.textContent('#l10Count') || '').trim();
    check(/^1024\s*\/\s*1024/.test(count0), '计数显示 ' + count0, '计数异常：' + count0);
    const domCards = await page.locator('#l10Summary .l10-dom').count();
    check(domCards === 8, '八域摘要卡 ' + domCards + '/8', '摘要卡数量异常：' + domCards);
    const firstPage = await page.locator('#l10List .l10-item').count();
    check(firstPage >= 60, '列表首屏渲染 ' + firstPage + ' 条（pageSize=60）', '列表首屏条数异常：' + firstPage);

    // 3) 交互 · 按域筛选 A
    const chips = await page.locator('#l10DomainBar button').all();
    let aChip = null;
    for (const c of chips) { if (((await c.textContent()) || '').trim() === 'A') { aChip = c; break; } }
    if (aChip) { await aChip.click(); } else { bad('未找到 A 域筛选按钮'); }
    await page.waitForFunction(() => /^128\s*\/\s*1024/.test((document.querySelector('#l10Count') || {}).textContent || ''), { timeout: 5000 });
    const countA = (await page.textContent('#l10Count') || '').trim();
    check(/^128\s*\/\s*1024/.test(countA), '筛选 A 域 → 计数 ' + countA, 'A 域计数异常：' + countA);
    const aPressed = aChip ? await aChip.getAttribute('aria-pressed') : null;
    check(aPressed === 'true', 'A 域 chip aria-pressed=true', 'A chip aria-pressed=' + aPressed);
    const firstKey = (await page.textContent('#l10List .l10-item .k') || '').trim();
    check(/^μA\d+$/.test(firstKey), 'A 域首项 key=' + firstKey, 'A 域首项 key 异常：' + firstKey);

    // 复位到全部
    const allChips = await page.locator('#l10DomainBar button').all();
    for (const c of allChips) { if (((await c.textContent()) || '').trim() === '全部') { await c.click(); break; } }
    await page.waitForFunction(() => /^1024\s*\/\s*1024/.test((document.querySelector('#l10Count') || {}).textContent || ''), { timeout: 5000 });

    // 选中首项，填充七件套详情面板（供截图）
    await page.locator('#l10List .l10-item').first().click();
    const sevenPieces = await page.locator('#l10Detail .l10-seven dd').count();
    check(sevenPieces >= 8, '七件套详情渲染 ' + sevenPieces + ' 段', '七件套详情段数异常：' + sevenPieces);

    // 4) 交互 · 批量试运行 1024 × 8
    await page.click('#l10BatchBtn');
    await page.waitForFunction(() => {
      const r = document.querySelector('#l10RunResult');
      return !!r && /✓\s*1024\s*\/\s*1024\s*项通过/.test(r.textContent);
    }, { timeout: 120000 });
    const runRes = (await page.textContent('#l10RunResult') || '').trim();
    const batchText = (await page.textContent('#l10BatchText') || '').trim();
    check(/✓\s*1024\s*\/\s*1024\s*项通过/.test(runRes), '批量试运行结果：' + runRes, '批量试运行结果异常：' + runRes);
    check(/失败\s*0/.test(batchText), '批量过程无失败：' + batchText, '批量过程含失败：' + batchText);

    // 5) 截图（README 头图 / CI artifact）
    if (wantShot) {
      fs.mkdirSync(path.dirname(SHOT), { recursive: true });
      const sec = page.locator('section[aria-labelledby="h-l10"]');
      await sec.scrollIntoViewIfNeeded();
      await page.waitForTimeout(150);
      await sec.screenshot({ path: SHOT });
      const kb = (fs.statSync(SHOT).size / 1024).toFixed(1);
      ok('截图已生成 docs/demo-screenshot.png（' + kb + ' KB）');
    }

    console.log('\n' + (failures === 0 ? '✅ probe-demo-browser：全部通过（真实浏览器渲染 + 交互 + 1024×8 谓词全绿）'
      : '❌ probe-demo-browser：' + failures + ' 项失败'));
    process.exit(failures === 0 ? 0 : 1);
  } finally {
    await browser.close();
  }
})().catch((e) => { console.error('❌ probe-demo-browser 异常：' + (e && e.stack || e)); process.exit(1); });

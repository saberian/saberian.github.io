import assert from 'node:assert/strict';
import { mkdir, readFile } from 'node:fs/promises';
import { chromium } from 'playwright';
import AxeBuilder from '@axe-core/playwright';

const baseURL = process.env.TEST_BASE_URL || 'http://127.0.0.1:4000';
const browser = await chromium.launch({
  headless: true,
  ...(process.env.CHROMIUM_EXECUTABLE_PATH ? { executablePath: process.env.CHROMIUM_EXECUTABLE_PATH } : {})
});
const reviewDir = '.impeccable/review';
const environmentTitle = 'What is an RL environment?';
const spreadTitle = 'How consistent are coding agents at building a music recommender?';
const spreadSharingImagePath = '/assets/images/spread-scores.png';
const spreadFigureAlt = 'Final hidden NDCG@10 scores for 19 task attempts across three model/harness groups.';
await mkdir(reviewDir, { recursive: true });
const pages = [
  { path: '/', name: 'home' },
  { path: '/blog/what-makes-a-good-rl-task/', name: 'post' },
  { path: '/blog/what-is-rl-environemnt/', name: 'environment' },
  { path: '/blog/spread-in-practice/', name: 'spread' }
];

try {
  const context = await browser.newContext();
  const page = await context.newPage();
  const errors = [];
  const canonicalURLs = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('requestfailed', request => errors.push(`Failed request: ${request.url()}`));
  page.on('response', response => {
    if (response.status() >= 400) errors.push(`HTTP ${response.status()}: ${response.url()}`);
  });
  for (const route of pages) {
    const response = await page.goto(new URL(route.path, baseURL).href);
    assert.equal(response.status(), 200, `${route.path} must resolve`);
    await page.evaluate(async () => {
      await document.fonts.ready;
      if (window.MathJax?.startup?.promise) await window.MathJax.startup.promise;
      await Promise.all([...document.images].map(image => image.decode()));
    });
    assert.equal(await page.locator('h1').count(), 1, 'Each page must have one primary heading');
    if (route.name === 'home') {
      assert.equal(await page.locator('.site-name:visible').count(), 1, 'Homepage must show one site name');
      assert.equal((await page.locator('.site-name').textContent()).trim(), 'Ehsan Saberian');
      assert.equal(await page.locator('.profile h1').count(), 0, 'Homepage profile must not repeat the site name as a large heading');
      assert.equal((await page.locator('h1').textContent()).trim(), 'Writing', 'Writing must be the homepage primary heading');
      assert.equal(await page.getByRole('navigation', { name: 'Main navigation' }).getByRole('link', { name: 'Writing', exact: true }).count(), 0, 'Homepage navigation must not repeat Writing');
      assert.equal(await page.locator('.essay h2').getByRole('link', { name: environmentTitle, exact: true }).getAttribute('href'), '/blog/what-is-rl-environemnt/', 'Correct the visible title without breaking the published URL');
      assert.equal((await page.locator('.profile-background p').first().textContent()).trim(), 'Previously at Roblox, Netflix, & Yahoo');
      assert.equal(await page.locator('.profile-background').getByText('Computer vision and machine learning', { exact: false }).isVisible(), true, 'Education details must be visible without hovering');
      assert.equal(await page.locator('.profile-background [title]').count(), 0, 'Do not hide professional context in tooltips');
    } else if (route.name === 'spread') {
      assert.equal((await page.locator('h1').textContent()).trim(), spreadTitle);
      assert.equal(await page.title(), `${spreadTitle} · Ehsan Saberian`);
      assert.equal(await page.locator('meta[property="og:title"]').getAttribute('content'), spreadTitle);
      assert.equal(await page.locator('h1.visually-hidden').count(), 0, 'Show the article title as text above the chart');
      assert.equal(await page.locator('.post-content img').count(), 1, 'Show the chart once, without a separate illustrated cover');
      const sharingResponse = await page.request.get(new URL(spreadSharingImagePath, baseURL).href);
      assert.equal(sharingResponse.status(), 200);
      assert.deepEqual(await sharingResponse.body(), await readFile(new URL(`..${spreadSharingImagePath}`, import.meta.url)), 'Serve the chart sharing preview unchanged');
      assert.equal(new URL(await page.locator('meta[property="og:image"]').getAttribute('content')).pathname, spreadSharingImagePath, 'Use the chart for sharing previews');
      const experimentTable = page.getByRole('table', { name: 'Agent experiment rounds and time use' });
      const spreadTable = page.getByRole('table', { name: 'Spread by model and harness' });
      assert.deepEqual(await page.locator('.post-content h2').allTextContents(), [
        'Agent performance'
      ], 'Preserve the author’s simpler section structure');
      const evidenceSections = await page.locator('.post-content').evaluate(article =>
        [...article.querySelectorAll('img, table')].map(element => {
          let preceding = element;
          while (preceding && preceding.tagName !== 'H2') {
            preceding = preceding.previousElementSibling || (preceding.parentElement === article ? null : preceding.parentElement);
          }
          return { tag: element.tagName, section: preceding?.textContent.trim() ?? null };
        })
      );
      assert.deepEqual(evidenceSections, [
        { tag: 'IMG', section: null },
        { tag: 'TABLE', section: 'Agent performance' },
        { tag: 'TABLE', section: 'Agent performance' }
      ], 'Lead with the chart, then retain the activity and spread tables in order');
      assert.equal(await page.locator('.post-content table').count(), 2, 'Keep the spread and agent-activity tables');
      assert.deepEqual(await spreadTable.locator('thead th').allTextContents(), ['Model / harness', 'Spread']);
      assert.equal(await spreadTable.locator('tbody tr').count(), 3);
      for (const value of ['4.25×', '2.12×', '2.87×']) assert.equal(await spreadTable.getByRole('cell', { name: value, exact: true }).count(), 1);
      assert.deepEqual(await experimentTable.locator('thead th').allTextContents(), ['Model / harness', 'Avg. scored rounds', 'Avg. time used'], 'Keep the author’s three-column activity table');
      assert.equal(await experimentTable.locator('tbody tr').count(), 3, 'The latest draft reviews 20 attempts in three groups');
      // Kramdown renders straight apostrophes as typographic apostrophes.
      const content = (await page.locator('.post-content').textContent()).replaceAll('’', "'");
      assert.doesNotMatch(content, /Methods and limitations|Which attempts count\.|Counting experiments\.|Relative variation\./, 'Remove the methods section and its contents');
      assert.equal(/95%|confidence intervals?|F-distribution|degrees of freedom/.test(content), false, 'Do not restore the CI/F-test discussion');
      assert.ok(content.includes('their sample standard deviations were roughly two to four times as large.'), 'Keep the reported comparison with reference-seed variation');
      assert.ok(content.includes('In the next post, we will examine the agent trajectories more closely'), 'Keep the saved draft’s trajectory-analysis follow-up');
      assert.match(content, /Ten of the 19 scored attempts exceeded the reference solution's\s+NDCG@10 of about 0\.018\./, 'Keep the saved draft’s rounded reference comparison');
      assert.ok(content.includes('mean of 0.01766'), 'Retain the precise reference mean alongside its training-seed variation');
      assert.ok(content.includes('Running the reference solution end-to-end takes about six minutes'), 'Distinguish running the reference from writing its implementation');
      assert.ok(content.includes('including data preparation, training, validation, ranking, and evaluation'), 'Clarify what the six-minute timing includes');
      assert.doesNotMatch(content, /seeds? for all agent attempts were always fixed|has no randomness|Building a reference solution end-to-end/, 'Do not restore the unsupported seed claim or misleading reliability/timing wording');
      assert.ok(content.includes('an RL environment we built for a coding agent (model + harness)'), 'Preserve the corrected opening from the saved draft');
      assert.doesNotMatch(content, /\blookat\b|\bimprovments\b|\bThis suggest\b|with mean of 0\.01766 a sample|effort: We ran/, 'Do not reintroduce the copyedited typos or sentence fragments');
      assert.ok(content.includes('all 20 attempts'), 'Preserve the latest trajectory-review scope');
      assert.equal(content.includes('a human pass with a model reading alongside'), false, 'Do not restore the human-review sentence removed from the draft');
      assert.equal(content.includes('no comparison below mixes a model with a harness upgrade'), false, 'Do not restore the fixed-harness-version claim removed from the draft');
      assert.equal(content.includes('Astra'), false, 'Do not mix the older attachment’s four-model cohort into the current draft');
      for (const pairing of ['Opus 5.0 on Claude Code 2.1.251', 'GPT-5.6-Sol on Codex CLI 0.147.0', 'Grok 4.6 on Grok Build 1.0.5']) {
        assert.ok(content.includes(pairing), `Preserve the recorded harness version: ${pairing}`);
      }
      const figure = page.getByRole('img', { name: spreadFigureAlt, exact: true });
      const figureResponse = await page.request.get(new URL(await figure.getAttribute('src'), baseURL).href);
      assert.equal(figureResponse.status(), 200);
      const figureSource = await figureResponse.text();
      assert.equal(figureSource, await readFile(new URL('../assets/images/spread-scores.svg', import.meta.url), 'utf8'), 'The preview must serve the current figure, not a stale build');
      assert.equal(figureSource.includes('Dots: individual scores'), false, 'Do not duplicate the article caption inside Figure 1');
      assert.equal(figureSource.includes('The lines are not confidence intervals.'), false, 'The figure must omit its embedded caption');
      const figureExplanation = page.locator('.post-content p').filter({ hasText: /^Figure 1\. Each dot represents/ });
      assert.equal(await figureExplanation.count(), 1, 'Explain the dots once below the lead chart');
      assert.equal(await figure.evaluate(image => image.closest('picture') === image.closest('.post-content').firstElementChild), true, 'Place the chart at the top, before the introduction');
      assert.equal(await figure.evaluate(image => image.closest('picture').nextElementSibling?.textContent.startsWith('Figure 1. Each dot represents')), true, 'Keep the caption directly below the chart');
      assert.ok(content.includes('As Figure 1 shows, even with the same model and harness'), 'Keep the detailed interpretation in the results section');
      assert.equal(await figure.evaluate(image => image.closest('a') === null), true, 'The latest draft uses a standalone SVG figure');
      for (const model of ['Opus 5.0', 'GPT-5.6-Sol', 'Grok 4.6']) assert.ok(figureSource.includes(model));
      assert.doesNotMatch(figureSource, /S = |Spread|Final scores and spread|Fixed reference|0\.035661/, 'Figure 1 uses slide 6’s dots-only format and the blog cohort');
      assert.equal((figureSource.match(/<circle /g) || []).length, 19, 'Keep the blog’s 19 scored attempts, not the older slide’s 20');
      assert.ok(figureSource.includes('Helvetica Neue, Helvetica, Arial, sans-serif'), 'Use the slide renderer’s portable font stack');
      const mobileSource = await page.request.get(new URL('/assets/images/spread-scores-mobile.svg', baseURL).href);
      assert.equal(mobileSource.status(), 200);
      assert.equal(await mobileSource.text(), await readFile(new URL('../assets/images/spread-scores-mobile.svg', import.meta.url), 'utf8'));
      assert.equal(figureSource.includes('Astra'), false, 'The chart must match the three-model cohort');
      assert.equal(await page.getByText('5.3 scored experiment rounds per attempt', { exact: true }).count(), 1);
      assert.equal(await page.locator('.post-content').getByRole('link', { name: 'task brief used for this experiment is available on GitHub' }).count(), 0, 'Do not restore the task-brief link removed by the author');
      assert.equal(await page.locator('.post-content').getByRole('link', { name: 'previous post', exact: true }).getAttribute('href'), '/blog/what-makes-a-good-rl-task/', 'Preserve the previous-post link');
      assert.equal(await page.locator('.post-content mjx-container[display="true"]').count(), 1, 'The spread equation must render as display math');
      assert.equal(await page.locator('.post-content mjx-assistive-mml mfrac').count(), 1, 'The spread equation must preserve its accessible fraction');
      assert.equal(await page.locator('.post-content [data-mjx-error], .post-content merror').count(), 0);
      assert.equal(await page.locator('.post-content').getByRole('link', { name: 'reproducible notebook' }).count(), 0, 'The latest draft does not include a notebook link');
    } else {
      assert.equal(await page.locator('.post-header--compact').count(), 1, 'Illustrated posts must use the compact metadata header');
      assert.equal(await page.locator('h1.visually-hidden').count(), 1, 'Keep the post title available to assistive technology without repeating the image title');
      assert.equal(await page.getByRole('heading', { level: 1 }).count(), 1, 'The visually hidden title must remain in the accessibility tree');
      if (route.name === 'post') {
        const previousPost = page.locator('.post-content').getByRole('link', { name: 'the previous post', exact: true });
        assert.equal(await previousPost.getAttribute('href'), '/blog/what-is-rl-environemnt/', 'The opening reference must link to the RL environment article');
        assert.equal(await page.locator('.post-content').getByText('standard deviation is zero, the ratio is undefined.', { exact: false }).isVisible(), true, 'Explain the zero-denominator limit of the spread ratio');
      } else {
        assert.equal((await page.locator('h1').textContent()).trim(), environmentTitle);
        assert.equal(await page.title(), `${environmentTitle} · Ehsan Saberian`);
        assert.equal(await page.locator('meta[property="og:title"]').getAttribute('content'), environmentTitle);
        assert.equal((await page.locator('.post-content').textContent()).includes('the more detailed the reward, the better'), false, 'Do not claim that denser rewards are always better');
      }
    }
    assert.equal(await page.locator('.post-end, .related-post').count(), 0, 'Omit the closing prompts and related-writing section');
    const canonicalURL = new URL(await page.locator('link[rel="canonical"]').getAttribute('href'));
    assert.equal(canonicalURL.pathname, route.path, 'Canonical URLs must preserve published paths');
    canonicalURLs.push(canonicalURL.href);
    assert.equal(await page.locator('meta[name="twitter:card"]').getAttribute('content'), 'summary_large_image');
    for (const property of ['title', 'description', 'image']) {
      assert.equal(await page.locator(`meta[name="twitter:${property}"]`).getAttribute('content'), await page.locator(`meta[property="og:${property}"]`).getAttribute('content'), `Sharing ${property} must stay consistent across metadata formats`);
    }
    const sharingImage = new URL(await page.locator('meta[name="twitter:image"]').getAttribute('content'));
    assert.equal(sharingImage.origin, canonicalURL.origin, 'Sharing images must use the canonical site origin');
    assert.ok(sharingImage.pathname.startsWith('/assets/images/'), 'Sharing images must use absolute image URLs');
    assert.equal(await page.locator('img').evaluateAll(images => images.every(img => img.naturalWidth > 0 && img.hasAttribute('alt'))), true);
    const a11y = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze();
    assert.deepEqual(a11y.violations.map(({ id, nodes }) => ({ id, targets: nodes.map(n => n.target) })), [], `${route.path} accessibility violations`);
    for (const width of [320, 390, 768, 900, 901, 1280, 1440, 1600]) {
      await page.setViewportSize({ width, height: width < 900 ? 844 : 1000 });
      // A picture may select a new source at this breakpoint. Measure its
      // decoded layout, not the previous image while the network is pending.
      await page.evaluate(async () => {
        await new Promise(resolve => requestAnimationFrame(resolve));
        await Promise.all([...document.images].map(image => image.decode()));
      });
      const size = await page.evaluate(() => ({ width: document.documentElement.clientWidth, scroll: document.documentElement.scrollWidth }));
      assert.ok(size.scroll <= size.width + 1, `${route.path} overflows at ${width}px: ${size.scroll}`);
      if (route.name === 'home') {
        const headingSize = await page.locator('h1').evaluate(el => parseFloat(getComputedStyle(el).fontSize));
        assert.ok(headingSize <= 28, `Homepage Writing heading must stay at or below 28px at ${width}px: ${headingSize}`);
        for (const [selector, expected] of [['.essay h2', 22], ['.essay p', 16], ['.writing-intro', 16], ['.profile-intro', 16], ['.profile-background', 14], ['.essay-link', 15]]) {
          const sizes = await page.locator(selector).evaluateAll(elements => elements.map(el => parseFloat(getComputedStyle(el).fontSize)));
          assert.ok(sizes.every(size => size === expected), `${selector} must use the compact ${expected}px role at ${width}px: ${sizes}`);
        }
        const navSize = await page.locator('.site-header nav').evaluate(el => parseFloat(getComputedStyle(el).fontSize));
        assert.ok(navSize <= 15, `Navigation must remain compact at ${width}px: ${navSize}`);
        const actionHeights = await page.locator('.essay-link, .profile-links a, .site-header nav a').evaluateAll(elements => elements.map(el => el.getBoundingClientRect().height));
        assert.ok(actionHeights.every(height => height >= 44), 'Smaller type must preserve usable action targets');
        const profileWidth = await page.locator('.profile').evaluate(el => el.getBoundingClientRect().width);
        assert.ok(Math.abs(profileWidth - (width > 900 ? 250 : width)) < 1,
          `Profile must be 250px on desktop and full-width when stacked at ${width}px: ${profileWidth}`);
      } else {
        const header = await page.locator('.post-header').boundingBox();
        const title = await page.locator('h1').boundingBox();
        if (route.name === 'spread') {
          const headingSize = await page.locator('h1').evaluate(el => parseFloat(getComputedStyle(el).fontSize));
          assert.ok(headingSize <= (width <= 540 ? 32 : 40), 'Use the existing compact title scale');
          assert.ok(title.width > 1 && title.height > 1, 'The chart-led article needs a visible text title');
          const coverBounds = await page.getByRole('img', { name: spreadFigureAlt, exact: true }).boundingBox();
          const articleBounds = await page.locator('.post-content').boundingBox();
          assert.ok(Math.abs(coverBounds.width - articleBounds.width) < 1, `Fit the lead chart to the reading column at ${width}px`);
          const chartRatio = width <= 600 ? 430 / 380 : 300 / 1040;
          assert.ok(Math.abs(coverBounds.height - coverBounds.width * chartRatio) < 1, 'Keep the responsive chart uncropped and undistorted');
          const tableGeometry = await page.getByRole('table', { name: 'Spread by model and harness' }).evaluate(table => {
            const article = table.closest('.post-content').getBoundingClientRect();
            const frame = table.getBoundingClientRect();
            const rows = table.tBodies[0].getBoundingClientRect();
            return {
              articleCenter: article.x + article.width / 2,
              frameCenter: frame.x + frame.width / 2,
              rowsCenter: rows.x + rows.width / 2,
              overflows: table.scrollWidth > table.clientWidth + 1,
              labelAlign: getComputedStyle(table.tBodies[0].rows[0].cells[0]).textAlign,
              valueAlign: getComputedStyle(table.tBodies[0].rows[0].cells[1]).textAlign
            };
          });
          assert.ok(Math.abs(tableGeometry.frameCenter - tableGeometry.articleCenter) < 1, `Center the spread table within the reading column at ${width}px`);
          if (!tableGeometry.overflows) assert.ok(Math.abs(tableGeometry.rowsCenter - tableGeometry.articleCenter) < 1, `Center the visible rows, not an empty full-width table box, at ${width}px`);
          assert.equal(tableGeometry.labelAlign, 'left', 'Keep model labels left-aligned within the centered table');
          assert.equal(tableGeometry.valueAlign, 'right', 'Keep spread values right-aligned within the centered table');
        } else {
          assert.ok(header.height <= 180, `Post header must stay compact at ${width}px: ${header.height}`);
          assert.ok(title.width <= 1 && title.height <= 1, 'Illustrated titles must not create a duplicate visible banner');
          assert.notEqual(await page.locator('h1').evaluate(el => getComputedStyle(el).clipPath), 'none');
        }
        const prose = await page.locator('.post-content').evaluate(el => ({ fontSize: parseFloat(getComputedStyle(el).fontSize), width: el.getBoundingClientRect().width }));
        assert.equal(prose.fontSize, width <= 540 ? 17 : 18, `Article prose must use the compact reading scale at ${width}px`);
        assert.ok(prose.width <= 640, `Smaller prose must keep a comfortable reading measure at ${width}px: ${prose.width}`);
      }
      if (route.name === 'spread' && width === 390) {
        for (const table of await page.locator('.post-content table').all()) {
          const overflows = await table.evaluate(el => el.scrollWidth > el.clientWidth);
          if (!overflows) continue;
          await table.focus();
          await page.keyboard.press('ArrowRight');
          await page.waitForFunction(el => el.scrollLeft > 0, await table.elementHandle());
          await table.evaluate(el => { el.scrollLeft = 0; el.blur(); });
        }
        const tableA11y = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze();
        assert.deepEqual(tableA11y.violations.map(({ id }) => id), [], 'The mobile comparison tables must support accessible scrolling');
      }
      if (width === 390 || width === 1440) {
        await page.evaluate(() => window.scrollTo(0, 0));
        const name = route.name === 'home' ? (width === 390 ? 'mobile' : 'desktop') : `${route.name}-${width === 390 ? 'mobile' : 'desktop'}`;
        await page.screenshot({ path: `${reviewDir}/${name}.png`, fullPage: true, animations: 'disabled' });
        if (route.name === 'spread') {
          await page.screenshot({ path: `${reviewDir}/spread-cover-${width === 390 ? 'mobile' : 'desktop'}.png`, animations: 'disabled' });
          const chart = page.locator('.post-content picture img');
          await chart.evaluate(image => image.decode());
          assert.equal((await chart.evaluate(image => image.currentSrc)).endsWith(width === 390 ? '/spread-scores-mobile.svg' : '/spread-scores.svg'), true, 'Serve the readable chart layout for the viewport');
          await chart.screenshot({ path: `${reviewDir}/spread-chart-${width === 390 ? 'mobile' : 'desktop'}.png` });
          const equation = page.locator('.post-content mjx-container[display="true"]');
          await equation.evaluate(element => window.scrollTo(0, window.scrollY + element.getBoundingClientRect().top - 16));
          const equationBounds = await equation.boundingBox();
          const tableBounds = await page.getByRole('table', { name: 'Spread by model and harness' }).boundingBox();
          const articleBounds = await page.locator('.post-content').boundingBox();
          await page.screenshot({
            path: `${reviewDir}/spread-table-${width === 390 ? 'mobile' : 'desktop'}.png`,
            clip: { x: articleBounds.x, y: equationBounds.y, width: articleBounds.width, height: tableBounds.y + tableBounds.height - equationBounds.y }
          });
        }
      }
    }
    for (const href of await page.locator('a[href^="/"]').evaluateAll(links => [...new Set(links.map(a => a.getAttribute('href').split('#')[0] || '/'))])) {
      const linked = await page.request.get(new URL(href, baseURL).href);
      assert.equal(linked.status(), 200, `Internal link ${href} must resolve`);
    }
    if (route.name === 'home') {
      assert.equal(await page.locator('.essay').count(), 3);
      assert.deepEqual(await page.locator('.essay time').allTextContents(), ['Sep 16, 2026', 'Sep 13, 2026', 'Sep 10, 2026']);
      await page.keyboard.press('Tab');
      assert.equal(await page.evaluate(() => document.activeElement.textContent.trim()), 'Skip to content');
      await page.keyboard.press('Enter');
      assert.equal(await page.evaluate(() => location.hash), '#main-content');
      await page.emulateMedia({ reducedMotion: 'reduce' });
      const arrow = page.locator('.essay-link .arrow').first();
      await page.locator('.essay-link').first().hover();
      assert.equal(await arrow.evaluate(el => getComputedStyle(el).transform), 'none');
      await page.emulateMedia({ reducedMotion: 'no-preference' });
    }
  }
  const sitemap = await page.request.get(new URL('/sitemap.xml', baseURL).href);
  assert.equal(sitemap.status(), 200, 'The generated sitemap must resolve');
  const sitemapText = await sitemap.text();
  for (const url of canonicalURLs) assert.ok(sitemapText.includes(`<loc>${url}</loc>`), `Sitemap must list the canonical URL ${url}`);
  assert.equal(sitemapText.includes('/404.html'), false, 'Do not list the error page in the sitemap');
  const robots = await page.request.get(new URL('/robots.txt', baseURL).href);
  assert.equal(robots.status(), 200, 'The generated robots.txt must resolve');
  assert.ok((await robots.text()).includes(`Sitemap: ${new URL('/sitemap.xml', canonicalURLs[0]).href}`), 'robots.txt must point to the canonical sitemap');
  assert.deepEqual(errors, [], 'Site assets must load without errors');
  console.log(`Site checks passed: ${pages.length} routes, 8 viewport widths, WCAG checks, titles, sharing metadata, sitemap, links, images, keyboard skip, and reduced motion.`);
} finally {
  await browser.close();
}

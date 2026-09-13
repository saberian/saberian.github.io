import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';
import { chromium } from 'playwright';
import AxeBuilder from '@axe-core/playwright';

const baseURL = process.env.TEST_BASE_URL || 'http://127.0.0.1:4000';
const browser = await chromium.launch({
  headless: true,
  ...(process.env.CHROMIUM_EXECUTABLE_PATH ? { executablePath: process.env.CHROMIUM_EXECUTABLE_PATH } : {})
});
const reviewDir = '.impeccable/review';
await mkdir(reviewDir, { recursive: true });
const pages = [
  { path: '/', name: 'home' },
  { path: '/blog/what-makes-a-good-rl-task/', name: 'post' },
  { path: '/blog/what-is-rl-environemnt/', name: 'environment' }
];

try {
  const context = await browser.newContext();
  const page = await context.newPage();
  const errors = [];
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
    } else {
      assert.equal(await page.locator('.post-header--compact').count(), 1, 'Illustrated posts must use the compact metadata header');
      assert.equal(await page.locator('h1.visually-hidden').count(), 1, 'Keep the post title available to assistive technology without repeating the image title');
      assert.equal(await page.getByRole('heading', { level: 1 }).count(), 1, 'The visually hidden title must remain in the accessibility tree');
      if (route.name === 'post') {
        const previousPost = page.locator('.post-content').getByRole('link', { name: 'the previous post', exact: true });
        assert.equal(await previousPost.getAttribute('href'), '/blog/what-is-rl-environemnt/', 'The opening reference must link to the RL environment article');
      }
    }
    assert.ok(await page.locator('link[rel="canonical"]').getAttribute('href'));
    assert.equal(await page.locator('img').evaluateAll(images => images.every(img => img.naturalWidth > 0 && img.hasAttribute('alt'))), true);
    const a11y = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze();
    assert.deepEqual(a11y.violations.map(({ id, nodes }) => ({ id, targets: nodes.map(n => n.target) })), [], `${route.path} accessibility violations`);
    for (const width of [320, 390, 768, 900, 901, 1280, 1440, 1600]) {
      await page.setViewportSize({ width, height: width < 900 ? 844 : 1000 });
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
        assert.ok(header.height <= 180, `Post header must stay compact at ${width}px: ${header.height}`);
        assert.ok(title.width <= 1 && title.height <= 1, 'The title must not create a visible banner');
        assert.notEqual(await page.locator('h1').evaluate(el => getComputedStyle(el).clipPath), 'none');
        const prose = await page.locator('.post-content').evaluate(el => ({ fontSize: parseFloat(getComputedStyle(el).fontSize), width: el.getBoundingClientRect().width }));
        assert.equal(prose.fontSize, width <= 540 ? 17 : 18, `Article prose must use the compact reading scale at ${width}px`);
        assert.ok(prose.width <= 640, `Smaller prose must keep a comfortable reading measure at ${width}px: ${prose.width}`);
      }
      if (width === 390 || width === 1440) {
        await page.evaluate(() => window.scrollTo(0, 0));
        const name = route.name === 'home' ? (width === 390 ? 'mobile' : 'desktop') : `${route.name}-${width === 390 ? 'mobile' : 'desktop'}`;
        await page.screenshot({ path: `${reviewDir}/${name}.png`, fullPage: true, animations: 'disabled' });
      }
    }
    if (route.name === 'home') {
      assert.equal(await page.locator('.essay').count(), 2);
      assert.deepEqual(await page.locator('.essay time').allTextContents(), ['Sep 13, 2026', 'Sep 10, 2026']);
      for (const href of await page.locator('a[href^="/"]').evaluateAll(links => [...new Set(links.map(a => a.getAttribute('href').split('#')[0] || '/'))])) {
        const linked = await page.request.get(new URL(href, baseURL).href);
        assert.equal(linked.status(), 200, `Internal link ${href} must resolve`);
      }
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
  assert.deepEqual(errors, [], 'Site assets must load without errors');
  console.log('Site checks passed: 3 routes, 8 viewport widths, WCAG checks, links, images, keyboard skip, and reduced motion.');
} finally {
  await browser.close();
}

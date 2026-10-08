import assert from 'node:assert/strict';
import { chromium } from 'playwright';
import AxeBuilder from '@axe-core/playwright';

const baseURL = process.env.TEST_BASE_URL || 'http://127.0.0.1:4000';
const browser = await chromium.launch({
  headless: true,
  ...(process.env.CHROMIUM_EXECUTABLE_PATH ? { executablePath: process.env.CHROMIUM_EXECUTABLE_PATH } : {})
});
try {
  const context = await browser.newContext();
  const page = await context.newPage();
  for (const width of [1280, 390, 320]) {
    await page.setViewportSize({ width, height: 900 });
    const response = await page.goto(`${baseURL}/blog/agent-trajectories/`);
    assert.equal(response.status(), 200);
    await page.evaluate(async () => {
      await document.fonts.ready;
      await Promise.all([...document.images].map(image => image.decode()));
    });
    assert.equal(await page.locator('h1').textContent(), 'Gaps in Frontier Coding Agents for ML Tasks');
    assert.deepEqual(await page.locator('summary').allTextContents(), ['Opus 5.5', 'Fable 5.1', 'Sol', 'Astra', 'Grok 4.6', 'Gemini 3.8 Flash']);
    assert.equal(await page.locator('a[href="/blog/how-consistent-are-coding-agents/"]').count(), 1);
    assert.equal(await page.locator('a[href^="/seniormle-bench"]').count(), 0);
    assert.equal(await page.locator('picture img').count(), 1);
    const image = await page.locator('picture img').evaluate(image => image.currentSrc);
    assert.ok(image.endsWith(width <= 600 ? '/agent-gap-frequency-mobile.svg' : '/agent-gap-frequency.svg'));
    for (const section of await page.locator('details').all()) {
      await section.locator('summary').focus();
      await page.keyboard.press('Enter');
      assert.equal(await section.getAttribute('open'), '');
      assert.ok(await section.locator('a[href^="https://verimium.com/seniormle-bench/"]').count() > 0);
    }
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `Overflow at ${width}px`);
    assert.doesNotMatch(await page.locator('.post-content').textContent(), /\{\{|\{%|\]\(/);
    const accessibility = await new AxeBuilder({ page }).analyze();
    assert.deepEqual(accessibility.violations.map(v => v.id), []);
  }
  await page.goto(baseURL);
  assert.equal(await page.getByRole('link', { name: 'Gaps in Frontier Coding Agents for ML Tasks', exact: true }).count(), 1);
  for (const path of ['/feed.xml', '/sitemap.xml']) {
    const response = await page.request.get(`${baseURL}${path}`);
    assert.ok((await response.text()).includes('/blog/agent-trajectories/'));
  }
  console.log('Agent article checks passed: responsive chart, keyboard disclosures, evidence links, accessibility, homepage, feed, and sitemap.');
} finally {
  await browser.close();
}

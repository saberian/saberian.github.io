import assert from 'node:assert/strict';
import { mkdir, readFile } from 'node:fs/promises';
import { chromium } from 'playwright';
import AxeBuilder from '@axe-core/playwright';
import { attention, START, MAX_STEPS, gradientStep, formatVector, fixed, loss } from '../assets/prompt-math.js';

const baseURL = process.env.TEST_BASE_URL || 'http://127.0.0.1:4000';
const articleURL = new URL('/blog/why-program-llms-in-words/', baseURL).href;
const browser = await chromium.launch({ headless: true,
  ...(process.env.CHROMIUM_EXECUTABLE_PATH ? { executablePath: process.env.CHROMIUM_EXECUTABLE_PATH } : {}) });
const reviewDir = '.impeccable/review';
await mkdir(reviewDir, { recursive: true });

try {
  const context = await browser.newContext();
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  page.on('requestfailed', request => errors.push(request.url()));
  page.on('response', response => { if (response.status() >= 400) errors.push(response.url()); });
  await page.goto(articleURL);
  await page.locator('[data-controls]').first().waitFor({ state: 'visible' });
  await page.evaluate(() => document.fonts.ready);
  const a = page.locator('[data-attention]');
  const e = page.locator('[data-embedding]');
  const initialAttention = await a.locator('[data-mixed]').textContent();
  const initialEmbedding = await e.locator('[data-position]').textContent();
  assert.equal(initialAttention, formatVector(attention(60).mixed));
  assert.equal(initialEmbedding, formatVector(START));
  assert.equal(await page.locator('.post-meta span').last().textContent(), '6 min read', 'Reading time counts text, not figure markup');
  for (const path of ['/assets/prompt-lab.js', '/assets/prompt-math.js', '/assets/prompt-lab.css']) {
    const response = await page.request.get(new URL(path, baseURL).href);
    assert.equal(response.status(), 200);
    assert.equal(await response.text(), await readFile(new URL(`..${path}`, import.meta.url), 'utf8'), 'Serve current source, not a stale preview');
  }
  const links = await page.locator('.post-content a').evaluateAll(nodes => nodes.map(node => node.href));
  for (const paper of ['2021.emnlp-main.243', '2021.acl-long.353', '2507.19457', '2501.12948v1', '2412.06769', '2507.11473']) {
    assert.ok(links.some(link => link.includes(paper)), `Cite ${paper}`);
  }

  // Keyboard changes have the same arithmetic and feedback as pointer input.
  const slider = page.getByRole('slider', { name: 'Prompt key angle' });
  await slider.focus();
  await page.keyboard.press('Home');
  assert.equal(await a.locator('[data-angle]').textContent(), '0°');
  assert.equal(await a.locator('[data-mixed]').textContent(), formatVector(attention(0).mixed));
  await page.keyboard.press('End');
  assert.equal(await a.locator('[data-weight]').first().textContent(), `${fixed(attention(180).weights[0] * 100, 1)}%`);
  await a.getByRole('button', { name: 'Reset angle' }).click();
  assert.equal(await slider.inputValue(), '60');

  const step = e.getByRole('button', { name: 'Take a step' });
  await step.focus();
  await page.keyboard.press('Enter');
  assert.equal(await e.locator('[data-position]').textContent(), formatVector(gradientStep(START)));
  const snap = e.getByRole('checkbox', { name: 'Snap to nearest token' });
  await snap.check();
  assert.equal(await step.isDisabled(), true);
  assert.ok((await e.locator('[data-embedding-status]').textContent()).includes('Nearest token'));
  await snap.uncheck();
  assert.equal(await e.locator('[data-position]').textContent(), formatVector(gradientStep(START)), 'Snapping must not overwrite the continuous vector');
  await e.getByRole('button', { name: 'Animate steps' }).click();
  await page.waitForFunction(() => !document.querySelector('[data-step-label]').textContent.startsWith('Step 1 '));
  await e.getByRole('button', { name: 'Pause' }).click();
  const pausedAt = await e.locator('[data-step-label]').textContent();
  await page.waitForTimeout(500);
  assert.equal(await e.locator('[data-step-label]').textContent(), pausedAt);
  await e.getByRole('button', { name: 'Animate steps' }).click();
  await page.evaluate(() => window.scrollTo({ top: 0, behavior: 'instant' }));
  await page.waitForFunction(() => document.querySelector('[data-play]').textContent === 'Animate steps');
  const offscreenAt = await e.locator('[data-step-label]').textContent();
  await page.waitForTimeout(500);
  assert.equal(await e.locator('[data-step-label]').textContent(), offscreenAt, 'Stop when offscreen');

  await page.emulateMedia({ reducedMotion: 'reduce' });
  await e.getByRole('button', { name: 'Reset', exact: true }).click();
  await e.getByRole('button', { name: 'Finish steps' }).click();
  assert.equal(await e.locator('[data-step-label]').textContent(), `Step ${MAX_STEPS} of ${MAX_STEPS}`);
  let expected = START;
  for (let i = 0; i < MAX_STEPS; i++) expected = gradientStep(expected);
  assert.equal(await e.locator('[data-position]').textContent(), formatVector(expected));
  assert.equal(await step.isDisabled(), true);
  assert.equal(await e.locator('[data-point]').evaluate(el => getComputedStyle(el).transitionDuration), '0s');
  await snap.check();
  assert.equal(await e.locator('[data-position]').textContent(), '[0.00, 1.65]');
  assert.ok((await e.locator('[data-embedding-status]').textContent()).includes(`continuous: ${fixed(loss(expected), 3)}`));
  assert.ok((await e.locator('[data-embedding-status]').textContent()).includes('Run complete. Uncheck to compare; reset to start again.'));
  await e.locator('summary').click();
  await a.locator('summary').click();
  const axe = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze();
  assert.deepEqual(axe.violations.map(({ id, nodes }) => ({ id, targets: nodes.map(n => n.target) })), []);
  await e.locator('summary').click();
  await a.locator('summary').click();

  // One batched visual review: default attention and learned/snapped embedding state.
  for (const width of [320, 390, 768, 1440]) {
    await page.setViewportSize({ width, height: 1000 });
    for (const lab of [a, e]) {
      const controls = await lab.locator('button, input[type="range"], .lab-toggle').evaluateAll(nodes => nodes.map(node => node.getBoundingClientRect().height));
      assert.ok(controls.every(height => height >= 44), 'Preserve touch targets');
      const box = await lab.boundingBox();
      assert.ok(box.x >= 0 && box.x + box.width <= width + 1);
    }
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true);
    if (width === 390 || width === 1440) {
      const device = width === 390 ? 'mobile' : 'desktop';
      await a.screenshot({ path: `${reviewDir}/attention-${device}.png`, animations: 'disabled' });
      await e.screenshot({ path: `${reviewDir}/embedding-${device}.png`, animations: 'disabled' });
    }
  }
  await page.emulateMedia({ media: 'print' });
  assert.equal(await page.locator('[data-controls]:visible').count(), 0, 'Print the figures without controls');
  assert.equal(await page.locator('.prompt-lab svg[role="img"]:visible').count(), 2);
  await page.emulateMedia({ media: 'screen', reducedMotion: 'no-preference' });
  await e.getByRole('button', { name: 'Reset', exact: true }).click();
  await e.getByRole('button', { name: 'Animate steps' }).click();
  await page.waitForFunction(() => document.querySelector('[data-step-label]').textContent === 'Step 12 of 12');
  assert.equal(await e.locator('[data-play]').getAttribute('aria-pressed'), 'false', 'Animation ends at convergence');

  const noJS = await browser.newContext({ javaScriptEnabled: false, viewport: { width: 390, height: 844 } });
  const fallback = await noJS.newPage();
  await fallback.goto(articleURL);
  assert.equal(await fallback.locator('[data-controls]:visible').count(), 0);
  assert.equal(await fallback.locator('[data-fallback]:visible').count(), 2);
  assert.equal(await fallback.locator('[data-mixed]').textContent(), initialAttention, 'Static and interactive arithmetic must agree');
  assert.equal(await fallback.locator('[data-position]').textContent(), initialEmbedding);
  assert.equal(await fallback.locator('.prompt-lab svg[role="img"]:visible').count(), 2);
  assert.ok((await fallback.locator('.post-content').textContent()).includes('readability is not faithfulness'));
  await noJS.close();
  await page.goto(new URL('/', baseURL).href);
  assert.equal(await page.locator('script[src*="prompt-lab"], link[href*="prompt-lab"]').count(), 0, 'Load the interactive assets only on their article');
  assert.deepEqual(errors, []);
  console.log('Prompt lab passed: numerical states, keyboard, snap/restore, pause/reset, offscreen stop, convergence, reduced motion, print, no-JS fallback, references, and responsive figures.');
} finally {
  await browser.close();
}

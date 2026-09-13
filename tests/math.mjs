import assert from 'node:assert/strict';
import { chromium } from 'playwright';

// Run against the real Jekyll output: TEST_BASE_URL=http://127.0.0.1:4000 node tests/math.mjs
const baseURL = process.env.TEST_BASE_URL || 'http://127.0.0.1:4000';
const browser = await chromium.launch({
  headless: true,
  ...(process.env.CHROMIUM_EXECUTABLE_PATH
    ? { executablePath: process.env.CHROMIUM_EXECUTABLE_PATH }
    : {})
});

try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  const errors = [];
  const mathRequests = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('requestfailed', request => errors.push(`Request failed: ${request.url()}`));
  page.on('response', response => {
    if (response.status() >= 400) errors.push(`HTTP ${response.status()}: ${response.url()}`);
  });
  page.on('request', request => {
    if (request.url().includes('mathjax')) mathRequests.push(request.url());
  });

  const response = await page.goto(new URL('/blog/what-makes-a-good-rl-task/', baseURL).href);
  assert.equal(response.status(), 200, 'The published post URL must resolve');
  await page.waitForFunction(() => Boolean(window.MathJax?.startup?.promise));
  await page.evaluate(() => window.MathJax.startup.promise);

  const math = await page.locator('.post-content').evaluate(content => {
    const containers = [...content.querySelectorAll('mjx-container')];
    return {
      total: containers.length,
      display: containers.filter(container => container.getAttribute('display') === 'true').length,
      visible: containers.every(container => {
        const svg = container.querySelector('svg');
        return svg?.getBoundingClientRect().height > 0 && svg.querySelector('path, use');
      }),
      mathErrors: content.querySelectorAll('[data-mjx-error], merror').length,
      fractions: content.querySelectorAll('mjx-assistive-mml mfrac').length,
      accessible: containers.every(container =>
        container.querySelector('mjx-assistive-mml math') &&
        container.querySelector('svg')?.getAttribute('aria-hidden') === 'true'
      ),
      inlineVariables: containers
        .filter(container => container.getAttribute('display') !== 'true')
        .map(container => container.querySelector('mjx-assistive-mml math').textContent.replace(/\s/g, '')),
      rawTex: /\\(?:frac|sigma|operatorname)|\$\$/.test(content.innerText)
    };
  });

  assert.equal(math.total, 5, 'Every equation in the post must be typeset');
  assert.equal(math.display, 2, 'Both standalone equations must render as display math');
  assert.ok(math.visible, 'Equations must contain visible SVG glyphs');
  assert.equal(math.mathErrors, 0, 'All TeX must parse successfully');
  assert.equal(math.fractions, 2, 'Both equations must preserve their fractions');
  assert.ok(math.accessible, 'Every equation must expose MathML to assistive technology');
  assert.deepEqual(math.inlineVariables, ['ri', 'σa', 'σr'], 'Inline subscripts must survive Markdown conversion');
  assert.equal(math.rawTex, false, 'Readers must not see unrendered TeX');
  assert.ok(mathRequests.length >= 1, 'The math renderer must load');
  assert.ok(mathRequests.every(url => new URL(url).origin === new URL(baseURL).origin),
    'Math rendering must work entirely from locally hosted assets');

  for (const width of [320, 390]) {
    await page.setViewportSize({ width, height: 844 });
    const dimensions = await page.evaluate(() => ({
      viewport: document.documentElement.clientWidth,
      page: document.documentElement.scrollWidth,
      equations: [...document.querySelectorAll('.post-content mjx-container[display="true"]')]
        .map(equation => {
          const rect = equation.getBoundingClientRect();
          return { left: rect.left, right: rect.right };
        })
    }));
    assert.ok(dimensions.page <= dimensions.viewport + 1, `The ${width}px page must not overflow horizontally`);
    assert.ok(dimensions.equations.every(equation =>
      equation.left >= 0 && equation.right <= dimensions.viewport + 1
    ), `Display math must stay inside the ${width}px viewport`);
  }

  assert.deepEqual(errors, [], 'The post must load and typeset without browser or network errors');
  console.log('Math regression passed: 2 display equations, 3 inline variables, accessible MathML, and mobile layouts.');
} finally {
  await browser.close();
}

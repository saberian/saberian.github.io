// Keep the sharing preview on the same SVG renderer as the article.
// Run after syncing assets/images/spread-scores.svg: npm run render:spread-preview
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const imageDir = new URL('../assets/images/', import.meta.url);
const svg = await readFile(new URL('spread-scores.svg', imageDir), 'utf8');
const [, width, height] = svg.match(/viewBox="0 0 (\d+) (\d+)"/) || [];
if (!width || !height) throw new Error('The score SVG must provide its chart dimensions');
const browser = await chromium.launch({
  headless: true,
  ...(process.env.CHROMIUM_EXECUTABLE_PATH ? { executablePath: process.env.CHROMIUM_EXECUTABLE_PATH } : {})
});
try {
  const page = await browser.newPage({ viewport: { width: Number(width), height: Number(height) }, deviceScaleFactor: 2 });
  await page.setContent(`<!doctype html><html lang="en"><head><meta charset="utf-8"><style>body { margin: 0; background: white; } svg { display: block; }</style></head><body>${svg}</body></html>`);
  await page.evaluate(() => document.fonts.ready);
  await page.locator('svg').screenshot({ path: fileURLToPath(new URL('spread-scores.png', imageDir)) });
  console.log('Updated assets/images/spread-scores.png from the current SVG.');
} finally {
  await browser.close();
}

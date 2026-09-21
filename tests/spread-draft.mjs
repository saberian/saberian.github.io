import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { basename, dirname, isAbsolute, resolve } from 'node:path';

const draftPath = process.argv[2];
assert.ok(draftPath, 'Usage: npm run test:spread-draft -- /absolute/path/to/spread-in-practice.md');
const draft = await readFile(draftPath, 'utf8');
const post = await readFile(new URL('../_posts/2026-09-16-spread-in-practice.md', import.meta.url), 'utf8');
const title = post.match(/^title: "([^"]+)"$/m)?.[1];
assert.ok(title, 'The published post must retain its title');

// Resolve the images actually linked in the chosen draft. Attachments can use
// absolute filesystem paths; website drafts can use their /assets/ route.
const imageReferences = [...new Set([...draft.matchAll(/\]\(([^\n)]+\.(?:svg|png))\)/g)].map(match => match[1]))];
assert.ok(imageReferences.some(reference => basename(reference) === 'spread-scores.svg'), 'The draft must reference its score figure');
assert.equal(imageReferences.some(reference => basename(reference) === 'spread-in-practice-cover.png'), false, 'The chart replaces the illustrated cover');
let normalizedDraft = draft.replace(/\] +\((https?:\/\/[^\n)]+)\)/g, ']($1)');
for (const reference of imageReferences) {
  normalizedDraft = normalizedDraft.replaceAll(reference, `/assets/images/${basename(reference)}`);
  const sourcePath = reference.startsWith('/assets/')
    ? resolve(dirname(draftPath), '../images/blog', basename(reference))
    : isAbsolute(reference) ? reference : resolve(dirname(draftPath), reference);
  const draftFigure = await readFile(sourcePath);
  const siteFigure = await readFile(new URL(`../assets/images/${basename(reference)}`, import.meta.url));
  assert.ok(siteFigure.equals(draftFigure), `The site figure must match ${sourcePath} byte for byte`);
  if (basename(reference) === 'spread-scores.svg') {
    const mobilePath = sourcePath.replace(/\.svg$/, '-mobile.svg');
    assert.ok((await readFile(new URL('../assets/images/spread-scores-mobile.svg', import.meta.url))).equals(await readFile(mobilePath)),
      'The mobile chart must match the companion generated alongside the draft figure');
  }
}

// Reverse only the documented Jekyll formatting adaptations. Editorial text,
// image links, and table contents must otherwise match the supplied draft.
const responsiveFigure = post.match(/<picture>\s*<source media="\(max-width: 600px\)" srcset="([^"]+)" width="380" height="430">\s*<img src="([^"]+)" width="1040" height="300" alt="([^"]+)">\s*<\/picture>/);
assert.ok(responsiveFigure, 'The webpage must use the slide renderer’s desktop and mobile chart layouts');
assert.equal(responsiveFigure[1], "{{ '/assets/images/spread-scores-mobile.svg' | relative_url }}");
assert.equal(responsiveFigure[2], "{{ '/assets/images/spread-scores.svg' | relative_url }}");
assert.match(post, /^hide_title: false$/m, 'Show the article title as text above the chart');
assert.match(post, /^image: \/assets\/images\/spread-scores\.png$/m, 'Use the chart for sharing previews');
const normalizedPost = post
  .replace(responsiveFigure[0], () => `![${responsiveFigure[3]}](${responsiveFigure[2]})`)
  .replace(/^---\n[\s\S]*?\n---\n\n/, `# ${title}\n\n`)
  .replaceAll('{% post_url 2026-09-10-what-is-rl-environemnt %}', 'https://saberian.github.io/blog/what-is-rl-environemnt/')
  .replaceAll('{% post_url 2026-09-13-what-makes-a-good-rl-task %}', 'https://saberian.github.io/blog/what-makes-a-good-rl-task/')
  .replaceAll("{{ '/assets/images/spread-scores.svg' | relative_url }}", '/assets/images/spread-scores.svg')
  .replaceAll("{{ '/assets/images/spread-scores.png' | relative_url }}", '/assets/images/spread-scores.png')
  .replaceAll('{: tabindex="0" aria-label="Agent experiment rounds and time use" }\n', '')
  .replaceAll('{: tabindex="0" aria-label="Music recommendation scores by model and harness" }\n', '')
  .replaceAll('{: .table-centered tabindex="0" aria-label="Spread by model and harness" }\n', '')
  .replaceAll('$$\nS = \\frac{\\sigma_a}{\\sigma_r}\n$$', '**S = σₐ / σᵣ**')
  .replaceAll('$$\\sigma_r$$', 'σᵣ')
  .replaceAll('$$\\sigma_a$$', 'σₐ');

assert.equal(normalizedPost.trimEnd(), normalizedDraft.trimEnd(), `The site post must match ${draftPath}`);
console.log(`Draft sync passed: complete post text, referenced images, and mobile chart match ${draftPath}`);

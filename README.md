# Ehsan Saberian's website

A static Jekyll site for GitHub Pages. The site deploys from `master` and preserves post URLs under `/blog/:title/`.

## Develop locally

Use Ruby 3.3 or newer with Bundler:

```sh
bundle install
bundle exec jekyll serve
```

Open `http://127.0.0.1:4000`. The Gemfile pins the Jekyll and Markdown versions used by GitHub Pages. On this Mac, Homebrew's Ruby is at `/opt/homebrew/opt/ruby/bin`.

## Edit content

- Add posts to `_posts/YYYY-MM-DD-slug.md` with `layout: post`, `title`, and an optional `description` and `image`.
- Professional details, employer names, portrait path, and profile links live in `_data/profile.yml`. Education details are displayed inline, not in a hover-only tooltip.
- Store images in `assets/images/` and reference their complete site-relative paths.
- Keep the existing post slugs when editing titles to preserve published links.
- Use `hide_title: true` when a cover illustration already contains the title. The post keeps an accessible heading and uses a compact back-link/metadata row; omit the flag to show the normal title.

The RL environment post keeps its original `/blog/what-is-rl-environemnt/` URL to preserve published links, even though its displayed title is now “What is an RL environment?”. Use Jekyll's `post_url` tag for references between posts.

After syncing the spread article from its working draft or an attached Markdown file, run `npm run test:spread-draft -- /absolute/path/to/spread-in-practice.md`. This compares the entire post and every referenced SVG/PNG with the explicitly selected draft, allowing only the site's link, math, and accessible-table formatting. Absolute filesystem image paths are resolved from the draft; `/assets/` references resolve from the draft's neighboring `images/blog` directory. Rebuild Jekyll and run `npm test` to check the served page and figure as well.

## Search and sharing

`jekyll-sitemap` generates `/sitemap.xml` and `/robots.txt` during the build; the error page is excluded. Open Graph and Twitter Card metadata share the same page title, description, and image, falling back to site/profile data on the homepage. The plugin version is pinned to match GitHub Pages.

## Equations

Add `math: true` to a post's front matter. Use native kramdown math syntax: `$$r_i$$` for inline math and a standalone `$$` block around a display equation. kramdown converts this into delimiters understood by the locally hosted MathJax renderer. Do not use raw `\(...\)` or `\[...\]` in Markdown, because Markdown consumes those backslashes.

With the site running, verify the equations in a real browser:

```sh
npm ci
npx playwright install chromium
npm test
```

Alternatively, set `CHROMIUM_EXECUTABLE_PATH` to an already installed Chrome or Chromium binary.

`npm test` verifies real equation output, accessibility, images and links, corrected titles, sharing metadata, sitemap/robots output, keyboard navigation, reduced motion, and page overflow across eight screen widths. It also protects the compact type scale, 44px action targets, narrow desktop sidebar, and illustration-first post headers. `npm run test:math` runs just the equation regression. The site check writes desktop and mobile screenshots into the gitignored `.impeccable/review/` directory.

## Profile and asset sources

The public profile at https://github.com/saberian, checked September 13, 2026, supplies the portrait, past employer names (Roblox, Netflix, Yahoo), and PhD details (computer vision and machine learning, UC San Diego). LinkedIn: https://www.linkedin.com/in/saberian/.

The two post illustrations were supplied by the author. Font licenses are in `assets/fonts/`; MathJax's license and version documentation are in `assets/vendor/mathjax/`.

Design planning, build caches, generated mockups, tests, and dependencies are excluded from the published site.

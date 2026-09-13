# MathJax

The site serves **MathJax 3.2.2** locally. The SVG build includes its math fonts
and TeX extensions, so readers do not depend on a third-party CDN. Its bundled
assistive MathML component supplies screen-reader-readable equations.

These unmodified files come from the official `mathjax@3.2.2` npm package:

- `es5/tex-svg-full.js`
- `LICENSE` (Apache-2.0)

Package integrity (SHA-512, as published by npm):

```
Bt+SSVU8eBG27zChVewOicYs7Xsdt40qm4+UpHyX7k0/O9NliPc+x77k1/FEsPsjKPZGJvtRZM1vO+geW0OhGw==
```

The optional MathJax settings menu is disabled; the site uses SVG output with
assistive MathML consistently and does not fetch alternate renderers or speech
components. To upgrade, obtain an exact version of the official package, retain
its license, update `_includes/math.html`, and run the browser math regression.

For new posts, set `math: true` in front matter and use kramdown's native math
syntax: `$$r_i$$` inside a paragraph, or `$$` on separate lines around a display
equation. Kramdown preserves the TeX and emits the delimiters MathJax reads.

References:

- [Kramdown math syntax](https://kramdown.gettalong.org/syntax.html#math-blocks)
- [Self-hosting MathJax](https://docs.mathjax.org/en/v3.2/web/hosting.html)
- [MathJax assistive MathML](https://docs.mathjax.org/en/v3.2/web/components/accessibility.html)

---
name: Ehsan Saberian
description: A simple research folio with restrained headings and an unhurried reading surface.
colors:
  link: "#244dcc"
  cover: "#d6e5f7"
  paper: "#f8faf9"
  ink: "#132646"
  secondary: "#43536c"
  muted: "#626e81"
  rule: "#dae1e8"
typography:
  section:
    fontFamily: "Hanken Grotesk, sans-serif"
    fontSize: "36px"
    fontWeight: 650
    lineHeight: 1.2
    letterSpacing: "-.025em"
  headline:
    fontFamily: "Hanken Grotesk, sans-serif"
    fontSize: "clamp(40px, 4.6vw, 68px)"
    fontWeight: 700
    lineHeight: 1.07
    letterSpacing: "-.035em"
  title:
    fontFamily: "Hanken Grotesk, sans-serif"
    fontSize: "30px"
    fontWeight: 700
    lineHeight: 1.25
    letterSpacing: "-.025em"
  body:
    fontFamily: "Source Serif 4, Georgia, serif"
    fontSize: "21px"
    fontWeight: 400
    lineHeight: 1.75
  interface:
    fontFamily: "Hanken Grotesk, sans-serif"
    fontSize: "18px"
    fontWeight: 400
    lineHeight: 1.5
  label:
    fontFamily: "Hanken Grotesk, sans-serif"
    fontSize: "17px"
    fontWeight: 400
    lineHeight: 1.5
  action:
    fontFamily: "Hanken Grotesk, sans-serif"
    fontSize: "18px"
    fontWeight: 400
    lineHeight: 1.5
spacing:
  "8": "8px"
  "16": "16px"
  "24": "24px"
  "32": "32px"
  "44": "44px"
  "56": "56px"
  "80": "80px"
components:
  navigation:
    backgroundColor: "{colors.cover}"
    textColor: "{colors.link}"
    padding: "0 3.2vw"
  essay-action:
    textColor: "{colors.link}"
    typography: "{typography.action}"
  essay-entry:
    textColor: "{colors.ink}"
  article-heading:
    backgroundColor: "{colors.cover}"
    textColor: "{colors.ink}"
    padding: "48px 32px 62px"
  related-writing:
    textColor: "{colors.link}"
---

# Design System: Ehsan Saberian

## Overview

**Creative North Star: "The Research Folio"**

A scientific monograph's dust jacket informs the visual language: mist-blue cover surfaces, navy ink, confident lettering, and plain image crops. Generous space and an unboxed writing index give the work a calm, personal setting.

Typography carries the hierarchy. Hanken Grotesk handles restrained headings and navigation; Source Serif 4 gives sustained prose its own reading rhythm. The interface stays flat, with small directional movement on essay links.

**Key Characteristics:**

- Mist-blue cover surfaces and near-white reading paper.
- Restrained sans-serif headings and titles, with serif article prose.
- Square image crops, open text entries, and short dividing rules.
- Text-led navigation with visible keyboard focus and restrained arrow motion.

Extracted from `assets/site.css`, `index.html`, `_layouts/default.html`, `_layouts/post.html`, and `_includes/arrow.html`. Page composition and the approved visual reference remain in `.impeccable/surfaces/index-html.md`.

The user requested smaller, simpler, non-repeating headings on September 13, 2026. This supersedes the oversized display typography in the original mockup. The name appears once in the header; Writing appears once as the homepage heading.

## Colors

The palette combines cool paper and blue-gray neutrals with a clear blue interaction accent. Frontmatter values are the normative colors and retain the CSS custom-property names.

### Primary

- **Link Blue (`link`):** navigation, essay actions, inline links, keyboard focus, and selection fill.

### Neutral

- **Mist Cover (`cover`):** profile surface, navigation, article header, footer, and prose quotations.
- **Reading Paper (`paper`):** the page and writing surface; reversed selected text.
- **Navy Ink (`ink`):** primary text, titles, and the site name.
- **Slate Text (`secondary`):** introductions, summaries, professional context, and article metadata.
- **Muted Slate (`muted`):** dates in the writing index.
- **Pale Rule (`rule`):** separators, table row borders, and the article end boundary.

**The Link Blue Rule.** Use blue to identify links and interaction states; essay titles begin in navy and turn blue on hover.

## Typography

The frontmatter records the base desktop roles. Hanken Grotesk and Source Serif 4 are self-hosted variable WOFF2 fonts with `font-display: swap`.

- **Section:** a modest Hanken Grotesk Writing heading. The author name stays in the shared header and is not repeated as a homepage hero.
- **Headline / title:** Hanken Grotesk for article headings and essay titles. Article titles have a maximum measure of 20ch; index titles use `text-wrap: pretty`.
- **Body:** Source Serif 4 for article prose. Paragraphs separate by 1.35em; emphasized prose uses weight 650.
- **Interface / label / action:** Hanken Grotesk for navigation, metadata, and text actions. Dates use tabular numerals.

Article subheadings use Hanken Grotesk at 32px and 26px, weight 700, with a 1.2 line height. At phone widths, prose becomes 19px with a 1.7 line height, the article headline becomes 40px, and essay titles become 26px. The Writing heading becomes 30px. The smallest breakpoint reduces the header name to 18px and article headline to 36px.

**The Two Type Roles Rule.** Keep Hanken Grotesk on headings and interface text, and Source Serif 4 on long-form prose.

## Layout

Large surfaces use fluid gutters and open columns; articles use a centered reading measure. The shared horizontal gutter is 3.2vw, becomes 32px at tablet widths and 24px on phones, and tightens to 20px at the smallest breakpoint. At 1600px and above, it is capped at 52px.

The homepage changes from a two-column grid to stacked sections at 900px. The header is 64px high; desktop sections start with 48px top padding. A compact portrait and professional context replace the oversized name block. At 540px, the profile, writing index, and header use compact arrangements; at 360px, header type and portrait sizes step down again.

Article headings sit in a centered 960px container. Prose and article-end navigation share a maximum width of 740px, with 32px side clearance reduced to 24px on phones. Images fit their reading column; wide tables, code blocks, and display mathematics scroll horizontally inside their own containers.

Spacing is contextual rather than a strict modular scale. The frontmatter records repeated steps, while major breaks depend on the surface: short inline gaps, generous section spacing, and substantial separation between essays. Links used as standalone actions have a minimum height of 44px; phone header navigation and compact profile links do as well.

## Elevation & Depth

No box shadows are used. Mist-blue and near-white planes establish structure, while whitespace and pale one-pixel rules divide content. Typography supplies emphasis without raised containers.

**The Flat Surface Rule.** Use tonal surfaces, spacing, and rules for separation; retain the unboxed treatment of writing entries.

## Shapes

Artwork and the portrait use plain rectangular or square crops. There is no rounded-card system. The only explicit corner radius belongs to the keyboard focus outline (2px); it is an interaction detail, not a container shape.

Directional cues are inline SVG strokes with round caps and joins. Essay separators are short horizontal rules rather than enclosing borders.

## Components

- **Navigation:** a mist-blue band with the author's navy name and blue text links. The homepage omits the redundant Writing navigation link; article pages retain it. LinkedIn appears in the header, with GitHub and RSS in the profile. Links gain underlines on hover.
- **Essay action:** a blue text link and stroked SVG arrow. The action retains a 44px minimum height, and the arrow moves 5px to the right on hover or keyboard focus over 280ms using the declared ease-out curve.
- **Essay entry:** a date, navy linked title, slate summary, and essay action in open document flow. Later entries receive a short pale rule and generous separation. Title color transitions over 180ms.
- **Article heading:** a mist-blue region containing a back link, Hanken Grotesk title, and wrapping author/date/read-time metadata. It shares the palette with the cover surface while keeping the reading hierarchy distinct.
- **Related writing:** the end-of-article text navigation uses a pale top rule, a brief closing line, and a subsequent essay link with the same SVG cue.

All anchors share a blue 2px focus outline with a 6px offset. The skip link appears on focus. Inline prose links are underlined. Reduced-motion preference removes transitions and arrow movement and changes smooth scrolling to ordinary scrolling. Print styling removes site navigation and article-end navigation, uses black on white, and makes prose full-width at 12pt.

## Do's and Don'ts

### Do:

- **Do** preserve the two type roles and self-hosted font files.
- **Do** keep writing entries open, with hierarchy established by type, whitespace, and short rules.
- **Do** keep focus outlines visible and honor reduced-motion preferences.
- **Do** contain wide equations, tables, and code within the article measure without clipping their content.

### Don't:

- **Don't** turn the writing index into a grid of raised or rounded cards.
- **Don't** repeat the author's name or Writing heading in an oversized homepage hero.
- **Don't** replace the square portrait crop or the inline SVG directional cues with decorative framing or glyph icons.

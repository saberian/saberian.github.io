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
    fontSize: "28px"
    fontWeight: 650
    lineHeight: 1.2
    letterSpacing: "-.025em"
  headline:
    fontFamily: "Hanken Grotesk, sans-serif"
    fontSize: "40px"
    fontWeight: 700
    lineHeight: 1.15
    letterSpacing: "-.025em"
  title:
    fontFamily: "Hanken Grotesk, sans-serif"
    fontSize: "22px"
    fontWeight: 700
    lineHeight: 1.3
    letterSpacing: "-.015em"
  body:
    fontFamily: "Source Serif 4, Georgia, serif"
    fontSize: "18px"
    fontWeight: 400
    lineHeight: 1.7
  interface:
    fontFamily: "Hanken Grotesk, sans-serif"
    fontSize: "16px"
    fontWeight: 400
    lineHeight: 1.5
  label:
    fontFamily: "Hanken Grotesk, sans-serif"
    fontSize: "14px"
    fontWeight: 400
    lineHeight: 1.5
  action:
    fontFamily: "Hanken Grotesk, sans-serif"
    fontSize: "15px"
    fontWeight: 400
    lineHeight: 1.5
  identity:
    fontFamily: "Hanken Grotesk, sans-serif"
    fontSize: "20px"
    fontWeight: 650
    lineHeight: 1.5
    letterSpacing: "-.03em"
  date:
    fontFamily: "Hanken Grotesk, sans-serif"
    fontSize: "13px"
    fontWeight: 400
    lineHeight: 1.5
  subheading:
    fontFamily: "Hanken Grotesk, sans-serif"
    fontSize: "26px"
    fontWeight: 700
    lineHeight: 1.2
    letterSpacing: "-.025em"
spacing:
  "8": "8px"
  "16": "16px"
  "24": "24px"
  "32": "32px"
  "44": "44px"
  "56": "56px"
  "80": "80px"
components:
  desktop-profile:
    backgroundColor: "{colors.cover}"
    width: "250px"
    padding: "32px 24px"
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
    padding: "32px 0 40px"
  article-heading-compact:
    backgroundColor: "{colors.cover}"
    textColor: "{colors.ink}"
    padding: "16px 0"
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

The user requested smaller, simpler, non-repeating headings on September 13, 2026, then a substantially smaller overall font scale after reviewing on a 15-inch MacBook. This supersedes the oversized display typography in the original mockup. The name appears once in the header; Writing appears once as the homepage heading. Preserve the compact scale across breakpoints instead of reintroducing larger mobile overrides.

The subsequent small cleanup preserves this layout and the published post URLs. Profile details and the portrait come from `_data/profile.yml`; the education field is visible inline on desktop and touch devices. The first post's displayed title is corrected to “What is an RL environment?”.

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
- **Pale Rule (`rule`):** essay separators and table row borders.

**The Link Blue Rule.** Use blue to identify links and interaction states; essay titles begin in navy and turn blue on hover.

## Typography

The frontmatter records the base desktop roles. Hanken Grotesk and Source Serif 4 are self-hosted variable WOFF2 fonts with `font-display: swap`.

- **Section:** a 28px Hanken Grotesk Writing heading. The author name stays in the shared header at 20px and is not repeated as a homepage hero.
- **Headline / title:** Hanken Grotesk for optional 40px article headings and 22px essay titles. Visible article titles use the full centered reading column with balanced wrapping, without a narrower character-width cap; index titles use `text-wrap: pretty`.
- **Body:** 18px Source Serif 4 for article prose. Paragraphs separate by 1.35em; emphasized prose uses weight 650.
- **Interface / label / action:** 16px Hanken Grotesk for introductions and summaries, 14px for secondary professional context and article metadata, and 15px for navigation and text actions. Index dates use 13px tabular numerals. Body copy uses normal tracking; title weight and color carry hierarchy without oversized text.

Article subheadings use Hanken Grotesk at 26px and 22px, weight 700, with a 1.2 line height. At phone widths, prose becomes 17px, the optional article headline becomes 32px, the Writing heading becomes 26px, the header name becomes 18px, and navigation becomes 14px. Essay titles remain 22px and summaries remain 16px at every width. Print is an intentional separate medium: 12pt prose and a 36px optional headline. Browser zoom remains unrestricted; standalone actions retain 44px hit areas.

**The Two Type Roles Rule.** Keep Hanken Grotesk on headings and interface text, and Source Serif 4 on long-form prose.

## Layout

Large surfaces use fluid gutters and open columns; articles use a centered reading measure. The shared horizontal gutter is 3.2vw, becomes 32px at tablet widths and 24px on phones, and tightens to 20px at the smallest breakpoint. At 1600px and above, it is capped at 52px.

The homepage uses a fixed 250px desktop profile column with 24px horizontal padding, half the original wide-screen sidebar. The writing receives the remaining width. The homepage header name aligns with the profile's 24px inset on desktop. At 900px and below, sections stack at full width with the existing mobile gutters. The header is 56px high; desktop sections start with 32px top padding. A compact portrait and professional context replace the oversized name block. At 540px, the profile, writing index, and header use compact arrangements; at 360px, gutters and portrait sizes step down again.

Visible article headings, compact headers, and prose share the centered 640px `--reading-width` token. The music-recommender title spans this column and wraps into two balanced lines on desktop, aligned with the chart and prose below. Both illustrated posts use `hide_title: true`: their title remains available to assistive technology, while the visible header becomes a compact back-link/metadata row with 16px vertical padding. The image begins 32px below this row. The reading column measures about 67ch with the loaded desktop prose font, with 32px side clearance reduced to 24px on phones; header and body gutters stay aligned at the smallest widths too. Summary lines are capped at 70ch. Images fit their reading column; wide tables, code blocks, and display mathematics scroll horizontally inside their own containers.

Spacing is contextual rather than a strict modular scale. The frontmatter records repeated steps, while major breaks depend on the surface: short inline gaps, a 28px writing-list offset, and 24px margin/padding around essay separators. Links used as standalone actions have a minimum height of 44px; header navigation and compact profile links do as well.

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
- **Article heading:** illustrated posts suppress the duplicate visible title, retaining a visually hidden semantic h1 and a compact mist-blue row for the back link and wrapping author/date/read-time metadata. Posts without `hide_title` retain the optional visible Hanken Grotesk heading.
- **Article ending:** posts end after their prose, followed by the shared site footer. The author requested removing the closing thanks, LinkedIn prompt, and related-writing section on September 21, 2026. Writing and LinkedIn remain available in the header.

All anchors share a blue 2px focus outline with a 6px offset. The skip link appears on focus. Inline prose links are underlined. Reduced-motion preference removes transitions and arrow movement and changes smooth scrolling to ordinary scrolling. Print styling removes site navigation, uses black on white, and makes prose full-width at 12pt.

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

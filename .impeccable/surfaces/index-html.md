---
version: 1
slug: "index-html"
primary_target: "index.html"
related_targets: ["_layouts/post.html"]
---

# Homepage and article redesign

Mode: Read. Audience assumption: AI peers and collaborators. Preserve both essays and their URLs. Verified profile: ML researcher and practitioner, previously Roblox, Netflix, Yahoo; PhD in CV and ML from UC San Diego. Profile photo sourced from the public GitHub account.

## Direction contract

THESIS: A personal research folio that opens with the researcher and leads immediately into his writing.

OWN-WORLD: Mist-blue cover, navy ink, clear blue links, white reading surfaces. Restrained Hanken Grotesk headings and interface, and Source Serif 4 prose. Plain image crops and an unboxed writing index. The user's September 13 screenshot feedback supersedes the original oversized display typography: make the headings smaller, simpler, and non-repeating.

STORY: Meet Ehsan, understand his ML background, choose a recent essay, and read it with legible equations.

SMALL CLEANUP: Preserve the existing layout and post URLs. Profile details and portrait come from `_data/profile.yml`, and the education specialization is visible inline rather than hidden in a tooltip. The first post's displayed title is “What is an RL environment?”.

ARTICLE ADAPTATION: The user requested removing the duplicate visible titles from both illustrated posts. `hide_title: true` retains a visually hidden semantic h1 and a compact back-link/author/date/read-time row; the existing cover illustration leads into the prose. Preserve titles in the index, metadata, and accessibility tree. This overrides the earlier tall article title banner.

FIRST VIEWPORT: A compact 250px blue profile column at the left with 24px horizontal padding; white writing surface fills the remaining width. The user's follow-up screenshot requested halving the original 500px wide-screen sidebar. The name appears only in the shared header, aligned with the profile inset. The profile contains the original portrait, introduction, professional context, GitHub and RSS. The user's 15-inch MacBook screenshot then requested smaller fonts throughout: Writing is 28px (26px phone), essay titles 22px, summaries and bio 16px, secondary context 14px, dates 13px, and actions 15px. The header is 56px high, with 32px top padding on desktop sections and tighter essay spacing. LinkedIn remains in the shared header; the redundant home Writing navigation link is removed. Signature interaction: essay links shift their directional cue slightly on hover and focus; reduced motion holds it still. At 900px and below, the layout stacks a full-width compact portrait/bio above the writing; the smaller typography stays consistent. Article prose is 18px (17px phone) in a 640px maximum reading column, while duplicate illustrated-post titles remain hidden.

FORM: Scientific monograph dust jacket, grounded candidate 4, seed e55b072c. User selected split layout B in the three-comp round. Historical comp: .impeccable/mocks/split-folio.png. User's later screenshot correction explicitly rejects the giant duplicate headings; its typography is no longer the fidelity target. Preserve the quiet blue/white publication layout and prioritize the latest simplification request.

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance

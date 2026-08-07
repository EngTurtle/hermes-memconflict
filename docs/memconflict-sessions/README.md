# Conversation Browser

A self-contained static web page for browsing the MemConflict simulated AI-chat dataset
(`Data/Step4_4.jsonl`) like a modern chat app — switch between the 30 personas, browse each
persona's ~53 conversations, and read transcripts as chat bubbles. A collapsible **Details**
panel shows the persona profile and each session's memory-conflict annotations
(static / conditional / dynamic conflicts, revealed attributes) and the benchmark questions.

## Usage

Just open `web/index.html` in a browser — **double-click it, no server needed.**

Because the page loads directly from disk (`file://`), the dataset is shipped as a JavaScript
file (`web/data.js`, which defines `window.MEMCONFLICT_DATA`) rather than fetched at runtime —
browsers block `fetch()` on `file://`, but a `<script src>` tag works fine.

## Rebuilding the data

`web/data.js` is generated from `Data/Step4_4.jsonl`. Regenerate it after the source data
changes:

```bash
python3 web/build_data.py
```

The script (standard library only) flattens each session's dialogue into an ordered
`{turn, role, content}` message list, normalizes a handful of malformed message objects in the
source, and writes `web/data.js`.

## Publishing as a single-file artifact

`web/artifact.html` is a fully self-contained build (all CSS/JS/data inlined into one file,
theme-aware for light/dark) suitable for publishing as a Claude artifact or hosting anywhere.
Because a single file can't hold the full ~40MB dataset, it embeds **6 personas at full
history** (gzip+base64, inflated in-browser via the native `DecompressionStream` API — no
library, no network). Rebuild with:

```bash
python3 web/build_artifact.py
```

## Files

| Path | Description |
| --- | --- |
| `index.html` | Page shell (Alpine.js markup for the 3 panes + details aside). |
| `js/app.js` | Alpine component: selection state, filtering, rendering helpers. |
| `css/style.css` | Styling (chat bubbles, session-type badges, responsive layout). |
| `vendor/alpine.min.js` | Vendored [Alpine.js](https://alpinejs.dev) v3 (loaded via relative path). |
| `build_data.py` | One-time build: `Data/Step4_4.jsonl` → `data.js`. |
| `data.js` | Generated dataset (`window.MEMCONFLICT_DATA`). |
| `build_artifact.py` | Assembles the self-contained `artifact.html` (6 full-history personas). |
| `artifact.html` | Generated single-file build for publishing/sharing. |

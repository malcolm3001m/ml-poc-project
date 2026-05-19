# Peak Value — standalone HTML presentation

A **pure HTML/CSS/JS** companion to the Streamlit app. No Python runtime needed at view time.
Built to use the patterns Streamlit's sandbox can't host:

- Custom lerp cursor with magnetic states
- Hacker-text scramble on section titles + roadmap hover
- Sliding pill nav indicator (left side)
- Mouse parallax on hero
- Canvas spotlight that follows the cursor (hero only)
- Day/night toggle persisted to localStorage
- Scroll-driven `animation-timeline: view()` reveals
- CSS-only sticky stacking cards (Section 7)
- Counter animations on every big number
- SVG-rendered charts with brand-aligned theming
- Pure-CSS 3D model carousel with drag-to-spin
- Animated conic-gradient borders on the prediction panel + player selector
- 8 NextLevel inspiration patterns layered

## Run locally

From the project root:

```bash
cd presentation
python3 -m http.server 8000
# then open http://localhost:8000
```

Or just **double-click `index.html`** — it works directly from `file://` too (the data is baked in, no fetch needed).

## Re-bake the data

Whenever models or validation outputs change, regenerate `data.js`:

```bash
# from project root with the venv active
python presentation/build_data.py
```

This loads the 3 trained `.joblib` models, picks 35 famous players + 40 random TEST players, computes all 3 predictions per player, and writes `data.js` (~200 KB) with everything embedded: validation numbers, histogram bins, country aggregates, residuals, feature importances, per-position MAE.

## Deploy to Vercel (drag-and-drop)

The `presentation/` folder is a static site — drag it into the Vercel dashboard or run `vercel` from inside it:

```bash
cd presentation
npx vercel
# follow the prompts (no build step needed)
```

Or deploy to GitHub Pages: push the repo, then in repo settings set Pages source to `/presentation` on `main`.

## File layout

```
presentation/
├── index.html        Structure — 8 sections matching the Streamlit version
├── style.css         Design system + all CSS patterns (~1100 lines)
├── app.js            Interactions: cursor, scramble, nav, charts, demo (~700 lines)
├── data.js           Baked output of build_data.py — DO NOT edit by hand
├── build_data.py     Regenerates data.js from the live model files
└── README.md         This file
```

## Browser support

- Chrome / Edge 111+ — everything works
- Firefox 128+ — everything works
- Safari 17.4+ — everything works, including `animation-timeline: view()` (Safari 17.4)
- Older browsers — the reveal animations fall back to plain fade-ins via `@supports`

## Status

This is a **tester** alongside the Streamlit app. The Streamlit version remains the committed GitHub deliverable; this folder is a no-limits playground for the visual presentation patterns.

# Presentation page

A standalone HTML version of the DAT0424 PoC slide deck used for the oral
defence. Pure HTML / CSS / JS — no Python needed at view time. All the
predictions, validation numbers, and player data are baked into `data.js`
ahead of time by the build script below.

## Run locally

From this directory:

```bash
python3 -m http.server 8000
# then open http://localhost:8000
```

Opening `index.html` directly via `file://` also works because everything
is baked into `data.js` — no fetch required (except for the world atlas
geometry, which is loaded from the local `world-atlas.json`).

## Re-baking the data

Whenever I retrain the models or change validation outputs, I regenerate
`data.js` with:

```bash
# from the project root, with the venv active
python presentation/build_data.py
```

The script loads the four trained `.joblib` models, picks a mix of famous
players, random TEST players, and the top ROI opportunities (~360 total),
computes all four model predictions per player, and writes `data.js`
with everything embedded: validation numbers, histogram bins, country
aggregates, residuals, feature importances, per-position MAE, and the
prediction-bucket diagnostic.

## File layout

```
presentation/
├── index.html        Slide-deck structure — 8 sections matching the Streamlit version
├── style.css         Design system + all the page styling
├── app.js            Interactions: search, charts, demo, world map
├── data.js           Baked output of build_data.py — DO NOT edit by hand
├── world-atlas.json  World GeoJSON for the choropleth (cached locally)
├── build_data.py     Regenerates data.js from the live models
├── SCRIPT.md         Stage-by-stage French script for the oral defence
└── README.md         This file
```

## Browser support

Tested in Chrome / Edge / Firefox / Safari. The page uses
`animation-timeline: scroll(root)` for the scroll-driven background paths,
which falls back to a JS handler on older browsers.

## Deploy

The folder is a static site, so:

```bash
cd presentation
npx vercel        # or drag the folder into the Vercel dashboard
```

Or push to GitHub Pages by setting the source to `/presentation` on `main`.

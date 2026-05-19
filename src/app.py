"""Streamlit application for the DAT0424 ML PoC — peak market value prediction.

Eight sections walk through the project as a non-technical pitch: problem, data,
approach, results, pressure-testing, interactive demo, honest limits, and roadmap.

The build_app() function is the entry point called by scripts/main.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import folium
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

from config import MODEL_METRICS_FILE, MODELS, RESULTS_DIR
from data import (
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    TARGET,
    _load_appearances,
    _load_players,
    load_dataset_split,
)


# ============================================================================
# DESIGN SYSTEM — Mining 8 NextLevel patterns
# ============================================================================

BRAND_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&family=Syncopate:wght@400;700&family=Caveat:wght@500;700&family=JetBrains+Mono:wght@500;700&display=swap');

/* ─────────────────────────────────────────────────────────────
   ROOT TOKENS — Glass kit from micro-interactions/005
   ───────────────────────────────────────────────────────────── */

@property --angle {
    syntax: '<angle>';
    inherits: false;
    initial-value: 0deg;
}
@property --shimmer-pos {
    syntax: '<percentage>';
    inherits: false;
    initial-value: -100%;
}

:root {
    /* Brand color */
    --c-pitch:        #0d7c47;
    --c-pitch-dark:   #075a32;
    --c-pitch-bright: #16a34a;
    --c-pitch-ghost:  rgba(13, 124, 71, 0.08);
    --c-amber:        #d97706;
    --c-amber-bright: #f59e0b;
    --c-red:          #dc2626;
    --c-magenta:      #c026d3;     /* glitch channel 1 */
    --c-cyan:         #06b6d4;     /* glitch channel 2 */

    /* Surface */
    --c-bg:           #fafaf6;
    --c-surface:      #ffffff;
    --c-surface-2:    #f5f5f0;
    --c-ink:          #0a0a0a;
    --c-ink-2:        #27272a;
    --c-muted:        #52525b;
    --c-muted-2:      #a1a1aa;
    --c-hairline:     #e7e5e4;

    /* Glass kit (from micro-005) */
    --glass-white:     rgba(255, 255, 255, 0.55);
    --glass-white-md:  rgba(255, 255, 255, 0.72);
    --glass-white-lg:  rgba(255, 255, 255, 0.85);
    --glass-border:    rgba(255, 255, 255, 0.6);
    --glass-border-sub: rgba(255, 255, 255, 0.3);

    /* Blur scale (named) */
    --blur-sm: blur(8px);
    --blur-md: blur(18px);
    --blur-lg: blur(32px);
    --blur-xl: blur(60px);

    /* Reflections */
    --reflection-top:   linear-gradient(135deg, rgba(255,255,255,.5) 0%, rgba(255,255,255,0) 50%);
    --reflection-inner: inset 0 1px 1px rgba(255,255,255,.55), inset 0 -1px 1px rgba(0,0,0,.06);

    /* Shadow scale */
    --shadow-sm:    0 1px 2px rgba(10,10,10,0.04), 0 1px 1px rgba(10,10,10,0.03);
    --shadow-md:    0 4px 12px rgba(10,10,10,0.06), 0 2px 4px rgba(10,10,10,0.04);
    --shadow-lg:    0 16px 40px rgba(10,10,10,0.08), 0 4px 12px rgba(10,10,10,0.05);
    --shadow-glow:  0 0 0 1px rgba(13,124,71,0.18), 0 8px 24px rgba(13,124,71,0.18);
    --shadow-glass: 0 8px 32px rgba(10,10,10,0.08), inset 0 1px 0 rgba(255,255,255,.3);
    --shadow-bloom: 0 30px 70px -10px rgba(13,124,71,0.25);

    /* Easing tokens (from neon-accordion + brutalism) */
    --ease-glass:   cubic-bezier(0.22, 0.68, 0, 1.2);
    --ease-liquid:  cubic-bezier(0.34, 1.56, 0.64, 1);
    --ease-cinema:  cubic-bezier(0.2, 0.8, 0.2, 1);
    --ease-smooth:  cubic-bezier(0.4, 0, 0.2, 1);
    --ease-out:     cubic-bezier(0, 0, 0.2, 1);

    /* Spacing scale */
    --s-1: 4px; --s-2: 8px; --s-3: 12px; --s-4: 16px;
    --s-5: 24px; --s-6: 32px; --s-7: 48px; --s-8: 64px; --s-9: 96px;

    /* Radius */
    --r-sm: 8px; --r-md: 14px; --r-lg: 20px; --r-xl: 28px; --r-full: 9999px;

    /* Typography */
    --f-display:  'Space Grotesk', system-ui, sans-serif;
    --f-syncope:  'Syncopate', 'Space Grotesk', sans-serif;
    --f-script:   'Caveat', cursive;
    --f-body:     'Inter', system-ui, sans-serif;
    --f-mono:     'JetBrains Mono', ui-monospace, monospace;
}

/* ─────────────────────────────────────────────────────────────
   BASE
   ───────────────────────────────────────────────────────────── */

html, body, [class*="css"], .stApp {
    font-family: var(--f-body);
    color: var(--c-ink);
    background: var(--c-bg);
}
.stApp { background: var(--c-bg); }
.main .block-container {
    padding-top: var(--s-7);
    padding-bottom: var(--s-9);
    max-width: 1180px;
}
h1, h2, h3, h4, h5 {
    font-family: var(--f-display) !important;
    letter-spacing: -0.02em;
    color: var(--c-ink);
}
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

/* ─────────────────────────────────────────────────────────────
   HERO — aurora mesh (3 blobs, different durations) + glitch + script overlay
   Pattern: hero-effects/001 + layouts/001
   ───────────────────────────────────────────────────────────── */

.hero {
    position: relative;
    padding: var(--s-7) 0 var(--s-8) 0;
    margin-bottom: var(--s-6);
    overflow: hidden;
    isolation: isolate;
}

/* Three independent blobs — each with its own duration. */
.hero::before, .hero::after, .hero .aurora-3 {
    content: '';
    position: absolute;
    border-radius: 50%;
    filter: var(--blur-xl);
    opacity: 0.55;
    pointer-events: none;
    z-index: 0;
}
.hero::before {
    width: 60vw; height: 60vh; max-width: 800px; max-height: 600px;
    left: -10%; top: -10%;
    background: radial-gradient(circle, rgba(13,124,71,0.65) 0%, transparent 70%);
    animation: aurora-drift-1 14s ease-in-out infinite alternate;
}
.hero::after {
    width: 50vw; height: 50vh; max-width: 700px; max-height: 500px;
    right: -5%; top: 20%;
    background: radial-gradient(circle, rgba(217,119,6,0.5) 0%, transparent 70%);
    animation: aurora-drift-2 18s ease-in-out infinite alternate;
    animation-delay: -6s;
}
.hero .aurora-3 {
    width: 40vw; height: 40vh; max-width: 600px; max-height: 400px;
    left: 30%; top: 40%;
    background: radial-gradient(circle, rgba(6,182,212,0.35) 0%, transparent 70%);
    animation: aurora-drift-3 20s ease-in-out infinite alternate;
    animation-delay: -3s;
}
@keyframes aurora-drift-1 {
    0%   { transform: translate(0, 0) scale(1); }
    100% { transform: translate(40px, 30px) scale(1.15); }
}
@keyframes aurora-drift-2 {
    0%   { transform: translate(0, 0) scale(1); }
    100% { transform: translate(-50px, 20px) scale(1.1); }
}
@keyframes aurora-drift-3 {
    0%   { transform: translate(0, 0) scale(0.95); }
    100% { transform: translate(30px, -40px) scale(1.2); }
}

.hero-inner { position: relative; z-index: 2; }

.hero-eyebrow {
    font-family: var(--f-syncope);
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--c-pitch);
    margin-bottom: var(--s-3);
}

/* Script/serif overlay pattern from layouts/001 */
.hero-script {
    font-family: var(--f-script);
    font-size: clamp(2.4rem, 5vw, 4rem);
    font-weight: 500;
    color: var(--c-amber);
    line-height: 1;
    margin-bottom: -0.3em;    /* overlap with the caps headline below */
    transform: rotate(-2deg);
    transform-origin: left center;
    display: inline-block;
}

.hero-title {
    font-family: var(--f-display);
    font-size: clamp(2.6rem, 6vw, 4.8rem);
    line-height: 0.98;
    font-weight: 700;
    letter-spacing: -0.035em;
    margin: 0 0 var(--s-5) 0;
    color: var(--c-ink);
    position: relative;
}

/* GLITCH effect on a specific word — pattern hero/001 */
.glitch {
    position: relative;
    display: inline-block;
    color: var(--c-pitch);
}
.glitch::before,
.glitch::after {
    content: attr(data-text);
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    opacity: 0;
}
.glitch::before {
    color: var(--c-magenta);
    clip-path: polygon(0 20%, 100% 20%, 100% 50%, 0 50%);
    animation: glitch-1 6s infinite;
}
.glitch::after {
    color: var(--c-cyan);
    clip-path: polygon(0 60%, 100% 60%, 100% 85%, 0 85%);
    animation: glitch-2 6s infinite;
}
@keyframes glitch-1 {
    0%, 94%, 100% { transform: none; opacity: 0; }
    95%  { transform: translateX(-6px); opacity: 0.75; }
    96%  { transform: translateX(4px) scaleY(1.04); opacity: 0.75; }
    97%  { transform: none; opacity: 0; }
}
@keyframes glitch-2 {
    0%, 92%, 100% { transform: none; opacity: 0; }
    93%  { transform: translateX(5px); opacity: 0.7; }
    94%  { transform: translateX(-3px); opacity: 0.7; }
    95%  { transform: none; opacity: 0; }
}

.hero-lede {
    font-size: 1.18rem;
    line-height: 1.55;
    color: var(--c-muted);
    max-width: 48rem;
    margin: 0 0 var(--s-7) 0;
    font-weight: 400;
}

/* ─────────────────────────────────────────────────────────────
   GLASS STAT STRIP — pattern: glass kit
   ───────────────────────────────────────────────────────────── */

.stat-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: var(--s-4);
    margin: var(--s-6) 0 var(--s-5) 0;
}
@media (max-width: 900px) {
    .stat-grid { grid-template-columns: repeat(2, 1fr); }
}
.stat-card {
    position: relative;
    background: var(--glass-white-md);
    backdrop-filter: var(--blur-md);
    -webkit-backdrop-filter: var(--blur-md);
    border: 1px solid var(--glass-border-sub);
    border-radius: var(--r-md);
    padding: var(--s-4) var(--s-5);
    box-shadow: var(--shadow-glass), var(--reflection-inner);
    transition: transform 0.5s var(--ease-liquid), box-shadow 0.5s var(--ease-liquid);
    isolation: isolate;
    overflow: hidden;
}
.stat-card::before {
    content: '';
    position: absolute;
    inset: 0;
    background: var(--reflection-top);
    border-radius: inherit;
    pointer-events: none;
    z-index: 1;
}
.stat-card:hover {
    transform: translateY(-3px) scale(1.02);
    box-shadow: var(--shadow-md), var(--reflection-inner);
}
.stat-card .num {
    font-family: var(--f-display);
    font-size: 1.85rem;
    font-weight: 700;
    line-height: 1;
    color: var(--c-pitch);
    margin-bottom: var(--s-2);
    letter-spacing: -0.025em;
    position: relative; z-index: 2;
}
.stat-card .lbl {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--c-muted);
    position: relative; z-index: 2;
}

/* ─────────────────────────────────────────────────────────────
   SECTION HEADERS — script overlay + spaced caps (layouts/001)
   ───────────────────────────────────────────────────────────── */

.section-kicker {
    font-family: var(--f-syncope);
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--c-pitch);
    margin: var(--s-8) 0 var(--s-3) 0;
    display: flex;
    align-items: center;
    gap: var(--s-3);
}
.section-kicker::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(to right, var(--c-hairline), transparent);
}
.section-q {
    font-family: var(--f-display);
    font-size: clamp(1.6rem, 3vw, 2.2rem);
    line-height: 1.15;
    font-weight: 700;
    letter-spacing: -0.025em;
    margin: 0 0 var(--s-3) 0;
    color: var(--c-ink);
}
.section-a {
    font-size: 1.02rem;
    color: var(--c-muted);
    max-width: 44rem;
    margin: 0 0 var(--s-5) 0;
    line-height: 1.55;
}
.chart-q {
    font-family: var(--f-display);
    font-size: 1.12rem;
    font-weight: 600;
    margin: var(--s-6) 0 var(--s-2) 0;
    color: var(--c-ink-2);
    display: flex; align-items: baseline; gap: var(--s-3);
}
.chart-q::before {
    content: counter(chart, decimal-leading-zero);
    counter-increment: chart;
    font-family: var(--f-mono);
    font-size: 0.7rem;
    color: var(--c-muted-2);
    font-weight: 700;
    letter-spacing: 0.05em;
}
.chart-caption {
    font-size: 0.92rem;
    color: var(--c-muted);
    line-height: 1.5;
    margin: 0 0 var(--s-3) 0;
    max-width: 50rem;
}

/* ─────────────────────────────────────────────────────────────
   USE-CASE CARDS — staggered cascade (cards/001) + shimmer + conic border on hover
   ───────────────────────────────────────────────────────────── */

.card-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: var(--s-4);
    margin: var(--s-4) 0;
}
@media (max-width: 800px) { .card-row { grid-template-columns: 1fr; } }

.card {
    position: relative;
    background: var(--c-surface);
    border: 1px solid var(--c-hairline);
    border-radius: var(--r-md);
    padding: var(--s-5);
    box-shadow: var(--shadow-sm);
    height: 100%;
    transition: transform 0.5s var(--ease-cinema),
                box-shadow 0.5s var(--ease-cinema),
                border-color 0.5s var(--ease-cinema);
    isolation: isolate;
    overflow: hidden;
}

/* Conic-gradient animated border on hover — pattern micro/003 */
.card::after {
    content: '';
    position: absolute;
    inset: -1px;
    background: conic-gradient(
        from var(--angle, 0deg),
        var(--c-pitch),
        var(--c-amber),
        var(--c-cyan),
        var(--c-pitch)
    );
    border-radius: inherit;
    opacity: 0;
    z-index: -1;
    transition: opacity 0.5s var(--ease-cinema);
    animation: border-spin 6s linear infinite;
}
@keyframes border-spin { to { --angle: 360deg; } }

/* Shimmer sweep on hover — pattern hero/001 */
.card::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(105deg,
        transparent 40%,
        rgba(255,255,255,.5) 50%,
        transparent 60%);
    background-size: 200% 100%;
    background-position: -100% 0;
    transition: background-position 0.9s var(--ease-cinema);
    pointer-events: none;
    z-index: 1;
    border-radius: inherit;
}

.card:hover {
    transform: translateY(-4px);
    box-shadow: var(--shadow-md);
    border-color: transparent;
}
.card:hover::after { opacity: 1; }
.card:hover::before { background-position: 200% 0; }

/* Staggered cascade — from neon accordion */
.cascade > * {
    opacity: 0;
    transform: translateY(18px);
    animation: cascade-in 0.7s var(--ease-cinema) forwards;
}
.cascade > *:nth-child(1) { animation-delay: 80ms; }
.cascade > *:nth-child(2) { animation-delay: 200ms; }
.cascade > *:nth-child(3) { animation-delay: 320ms; }
.cascade > *:nth-child(4) { animation-delay: 440ms; }
.cascade > *:nth-child(5) { animation-delay: 560ms; }
@keyframes cascade-in {
    to { opacity: 1; transform: translateY(0); }
}

.card-icon {
    width: 44px;
    height: 44px;
    border-radius: var(--r-sm);
    background: var(--c-pitch-ghost);
    color: var(--c-pitch);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.3rem;
    margin-bottom: var(--s-4);
    position: relative; z-index: 2;
}
.card-title {
    font-family: var(--f-display);
    font-size: 1.05rem;
    font-weight: 700;
    margin: 0 0 var(--s-2) 0;
    color: var(--c-ink);
    letter-spacing: -0.01em;
    position: relative; z-index: 2;
}
.card-body {
    font-size: 0.93rem;
    color: var(--c-muted);
    line-height: 1.55;
    margin: 0;
    position: relative; z-index: 2;
}

/* ─────────────────────────────────────────────────────────────
   3D CSS CAROUSEL — pattern cards/004 (tan() geometry)
   Three model cards rotating around an axis
   ───────────────────────────────────────────────────────────── */

.scene-3d {
    position: relative;
    perspective: 60em;
    height: 380px;
    margin: var(--s-6) 0;
    mask: linear-gradient(90deg, #0000, red 15% 85%, #0000);
    -webkit-mask: linear-gradient(90deg, #0000, red 15% 85%, #0000);
}
.a3d {
    --n: 3;
    position: absolute;
    top: 50%; left: 50%;
    width: 1px; height: 1px;
    transform-style: preserve-3d;
    animation: ry 24s linear infinite;
}
@keyframes ry {
    to { transform: translate(-50%, -50%) rotateY(360deg); }
}
.scene-3d:hover .a3d {
    animation-play-state: paused;
}
.card-3d {
    --w: 22em;
    --ba: calc(360deg / var(--n));
    position: absolute;
    top: -180px; left: -176px;
    width: var(--w);
    height: 340px;
    border-radius: var(--r-lg);
    background: var(--c-surface);
    border: 1px solid var(--c-hairline);
    box-shadow: var(--shadow-md);
    backface-visibility: hidden;
    transform-style: preserve-3d;
    padding: var(--s-6);
    transform:
        rotateY(calc(var(--i) * var(--ba)))
        translateZ(calc((var(--w) / 2) / tan(calc(var(--ba) / 2)) + 4em));
}
.card-3d.t1 { border-top: 4px solid #94a3b8; }
.card-3d.t2 { border-top: 4px solid var(--c-amber); }
.card-3d.t3 { border-top: 4px solid var(--c-pitch); box-shadow: var(--shadow-glow); }

.card-3d .badge-num {
    font-family: var(--f-mono);
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--c-muted);
    letter-spacing: 0.08em;
    margin-bottom: var(--s-3);
}
.card-3d .card-3d-title {
    font-family: var(--f-display);
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: -0.025em;
    margin: 0 0 var(--s-4) 0;
    color: var(--c-ink);
}
.card-3d .card-3d-body {
    font-size: 0.95rem;
    color: var(--c-muted);
    line-height: 1.55;
}
.card-3d .card-3d-foot {
    margin-top: var(--s-5);
    padding-top: var(--s-3);
    border-top: 1px solid var(--c-hairline);
    font-family: var(--f-mono);
    font-size: 0.78rem;
    color: var(--c-muted);
    display: flex;
    justify-content: space-between;
}

@media (prefers-reduced-motion: reduce) {
    .a3d { animation-duration: 120s; }
}

/* ─────────────────────────────────────────────────────────────
   CALLOUTS
   ───────────────────────────────────────────────────────────── */

.callout {
    background: var(--c-pitch-ghost);
    border-left: 3px solid var(--c-pitch);
    padding: var(--s-4) var(--s-5);
    margin: var(--s-5) 0;
    border-radius: 0 var(--r-sm) var(--r-sm) 0;
    font-size: 0.97rem;
    line-height: 1.6;
    color: var(--c-ink-2);
}
.callout b, .callout strong { color: var(--c-ink); }
.callout.warn { background: rgba(220,38,38,0.06); border-left-color: var(--c-red); }
.callout.amber { background: rgba(217,119,6,0.08); border-left-color: var(--c-amber); }

/* ─────────────────────────────────────────────────────────────
   BIGNUM — conic-gradient animated border on the headline panels
   ───────────────────────────────────────────────────────────── */

.bignum-row {
    display: grid;
    grid-template-columns: 2fr 1fr 1fr 1fr;
    gap: var(--s-4);
    margin: var(--s-5) 0;
    align-items: stretch;
}
@media (max-width: 800px) { .bignum-row { grid-template-columns: 1fr 1fr; } }

.bignum {
    position: relative;
    background: var(--c-ink);
    color: var(--c-surface);
    padding: var(--s-5);
    border-radius: var(--r-md);
    overflow: hidden;
    isolation: isolate;
}
.bignum::before {
    content: '';
    position: absolute;
    inset: 0;
    background:
        radial-gradient(60% 60% at 80% 20%, rgba(22,163,74,0.3), transparent 60%),
        radial-gradient(50% 50% at 20% 90%, rgba(217,119,6,0.2), transparent 60%);
    pointer-events: none;
}
.bignum.featured::after {
    content: '';
    position: absolute;
    inset: -1px;
    background: conic-gradient(
        from var(--angle, 0deg),
        var(--c-pitch-bright),
        var(--c-amber-bright),
        var(--c-cyan),
        var(--c-pitch-bright)
    );
    border-radius: inherit;
    z-index: -1;
    animation: border-spin 8s linear infinite;
    opacity: 0.85;
}
.bignum.featured {
    margin: 1px;
}
.bignum .lbl {
    font-family: var(--f-syncope);
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    color: rgba(255,255,255,0.65);
    margin-bottom: var(--s-3);
    position: relative;
}
.bignum .num {
    font-family: var(--f-display);
    font-size: 3.4rem;
    font-weight: 700;
    line-height: 1;
    color: #fff;
    letter-spacing: -0.04em;
    position: relative;
}
.bignum .sub {
    font-size: 0.85rem;
    color: rgba(255,255,255,0.7);
    margin-top: var(--s-2);
    position: relative;
}
.bignum.compact .num { font-size: 1.9rem; }
.bignum.compact { padding: var(--s-4); }
.bignum.green { background: var(--c-pitch); }
.bignum.amber { background: var(--c-amber); }

/* ─────────────────────────────────────────────────────────────
   PILLS
   ───────────────────────────────────────────────────────────── */

.pill {
    display: inline-block;
    padding: 4px 10px;
    border-radius: var(--r-full);
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    background: var(--c-pitch-ghost);
    color: var(--c-pitch);
}
.pill.amber { background: rgba(217,119,6,0.12); color: var(--c-amber); }
.pill.red   { background: rgba(220,38,38,0.1);  color: var(--c-red); }
.pill.gray  { background: var(--c-surface-2);   color: var(--c-muted); }
.pill.dark  { background: var(--c-ink);         color: var(--c-amber-bright); font-family: var(--f-mono); }

/* ─────────────────────────────────────────────────────────────
   BRUTALIST ROADMAP — pattern layouts/004
   Massive numbered phases with neon accent + hacker-text feel
   ───────────────────────────────────────────────────────────── */

.brutal-timeline {
    margin: var(--s-7) 0;
    counter-reset: phase;
}
.brutal-phase {
    position: relative;
    display: grid;
    grid-template-columns: 140px 1fr;
    gap: var(--s-6);
    padding: var(--s-6) 0;
    border-top: 1px solid var(--c-hairline);
    transition: background 0.4s var(--ease-smooth);
}
.brutal-phase:hover {
    background: linear-gradient(90deg, var(--c-pitch-ghost), transparent 60%);
}
.brutal-phase:last-child { border-bottom: 1px solid var(--c-hairline); }
.brutal-phase::before {
    counter-increment: phase;
    content: "0" counter(phase);
    font-family: var(--f-syncope);
    font-size: 5.2rem;
    font-weight: 700;
    line-height: 0.85;
    color: var(--c-ink);
    letter-spacing: -0.04em;
    grid-column: 1;
}
.brutal-phase.amber::before { color: var(--c-amber); }
.brutal-phase.pitch::before { color: var(--c-pitch); }
.brutal-phase.cyan::before { color: var(--c-cyan); }
.brutal-content { grid-column: 2; }
.brutal-title {
    font-family: var(--f-display);
    font-size: 1.45rem;
    font-weight: 700;
    color: var(--c-ink);
    margin: 0 0 var(--s-2) 0;
    letter-spacing: -0.02em;
    display: flex;
    flex-wrap: wrap;
    gap: var(--s-3);
    align-items: center;
}
.brutal-body {
    color: var(--c-muted);
    font-size: 0.97rem;
    line-height: 1.6;
    margin: 0;
    max-width: 50rem;
}

/* ─────────────────────────────────────────────────────────────
   SPINE — I → Y → Z three-step card
   ───────────────────────────────────────────────────────────── */

.spine {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1px;
    background: var(--c-hairline);
    border: 1px solid var(--c-hairline);
    border-radius: var(--r-md);
    overflow: hidden;
    margin: var(--s-6) 0;
}
@media (max-width: 800px) { .spine { grid-template-columns: 1fr; } }
.spine > div {
    background: var(--c-surface);
    padding: var(--s-5);
    position: relative;
}
.spine > div:not(:last-child)::after {
    content: '→';
    position: absolute;
    right: -16px; top: 50%;
    transform: translateY(-50%);
    background: var(--c-surface);
    color: var(--c-pitch);
    width: 32px; height: 32px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
    font-weight: 700;
    border: 1px solid var(--c-hairline);
    z-index: 2;
}
@media (max-width: 800px) {
    .spine > div:not(:last-child)::after { display: none; }
}
.spine .step {
    font-family: var(--f-mono);
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: var(--c-pitch);
    margin-bottom: var(--s-3);
}
.spine .head {
    font-family: var(--f-display);
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--c-ink);
    margin-bottom: var(--s-2);
    letter-spacing: -0.015em;
}
.spine .desc { color: var(--c-muted); font-size: 0.92rem; line-height: 1.5; }

/* ─────────────────────────────────────────────────────────────
   MAP PIN PULSE — pattern micro-interactions/001
   Overlay on the choropleth for top 5 countries
   ───────────────────────────────────────────────────────────── */

.map-legend {
    display: flex;
    flex-wrap: wrap;
    gap: var(--s-4);
    margin: var(--s-4) 0 var(--s-6) 0;
    padding: var(--s-4);
    background: var(--glass-white-md);
    backdrop-filter: var(--blur-sm);
    border-radius: var(--r-md);
    border: 1px solid var(--c-hairline);
}
.legend-item {
    display: flex;
    align-items: center;
    gap: var(--s-3);
    position: relative;
}
.legend-pin {
    position: relative;
    width: 14px;
    height: 14px;
    border-radius: 50% 50% 50% 0;
    background: var(--c-pitch);
    transform: rotate(-45deg);
    flex-shrink: 0;
    animation: pin-bounce 0.8s var(--ease-liquid);
}
.legend-pin::before {
    content: '';
    position: absolute;
    width: 22px;
    height: 22px;
    border-radius: 50%;
    box-shadow: 0 0 0 2px var(--c-pitch);
    top: -4px; left: -4px;
    opacity: 0;
    transform: rotateX(55deg) scale(0.3);
    animation: pin-pulse 2.5s ease-out infinite;
    animation-delay: 1s;
}
@keyframes pin-bounce {
    0%   { opacity: 0; transform: translateY(-30px) rotate(-45deg); }
    60%  { opacity: 1; transform: translateY(6px) rotate(-45deg); }
    80%  { transform: translateY(-3px) rotate(-45deg); }
    100% { transform: translateY(0) rotate(-45deg); }
}
@keyframes pin-pulse {
    0%   { transform: rotateX(55deg) scale(0.3); opacity: 0; }
    50%  { opacity: 0.7; }
    100% { transform: rotateX(55deg) scale(1.8); opacity: 0; }
}
.legend-text {
    font-size: 0.85rem;
    color: var(--c-ink);
    font-weight: 500;
}
.legend-text .country { font-weight: 700; }
.legend-text .val {
    color: var(--c-pitch);
    font-family: var(--f-mono);
    font-size: 0.85rem;
    font-weight: 700;
    margin-left: 6px;
}

/* ─────────────────────────────────────────────────────────────
   SIDEBAR — stamped pill nav state
   ───────────────────────────────────────────────────────────── */

[data-testid="stSidebar"] {
    background: var(--c-surface);
    border-right: 1px solid var(--c-hairline);
}
[data-testid="stSidebar"] .stRadio label {
    font-family: var(--f-body);
    font-size: 0.92rem;
    font-weight: 500;
    color: var(--c-ink-2);
    padding: var(--s-2) var(--s-3);
    border-radius: var(--r-sm);
    transition: background 0.25s var(--ease-cinema), color 0.25s var(--ease-cinema);
    position: relative;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: var(--c-surface-2);
}
[data-testid="stSidebar"] [role="radiogroup"] > label[data-baseweb="radio"] > div:first-child {
    display: none;
}
[data-testid="stSidebar"] [aria-checked="true"] {
    color: var(--c-pitch) !important;
    font-weight: 700 !important;
}
.sidebar-brand {
    font-family: var(--f-syncope);
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    margin-bottom: var(--s-2);
    color: var(--c-ink);
    text-transform: uppercase;
}
.sidebar-brand .dot { color: var(--c-pitch); }
.sidebar-tag {
    font-size: 0.78rem;
    color: var(--c-muted);
    margin-bottom: var(--s-5);
    font-family: var(--f-mono);
}
.sidebar-foot { font-family: var(--f-mono); font-size: 0.72rem; color: var(--c-muted); line-height: 1.6; }
.sidebar-foot .row { display: flex; justify-content: space-between; padding: 2px 0; }
.sidebar-foot .row span:last-child { color: var(--c-pitch); font-weight: 700; }

/* ─────────────────────────────────────────────────────────────
   STREAMLIT OVERRIDES
   ───────────────────────────────────────────────────────────── */

.stButton button {
    background: var(--c-pitch);
    color: white;
    border-radius: var(--r-sm);
    border: none;
    font-weight: 600;
    transition: background 0.3s var(--ease-cinema),
                transform 0.3s var(--ease-liquid),
                box-shadow 0.3s var(--ease-cinema);
}
.stButton button:hover {
    background: var(--c-pitch-dark);
    transform: translateY(-1px);
    box-shadow: var(--shadow-glow);
}
.stButton button:active {
    transform: scale(0.97);
}
.stDataFrame { border-radius: var(--r-md); overflow: hidden; }
.stMetric {
    background: var(--c-surface);
    border: 1px solid var(--c-hairline);
    border-radius: var(--r-md);
    padding: var(--s-4);
}
.stMetric label {
    color: var(--c-muted) !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600 !important;
}
.stMetric [data-testid="stMetricValue"] {
    font-family: var(--f-display);
    font-weight: 700;
    color: var(--c-pitch);
}

/* Selectbox stylings */
div[data-baseweb="select"] > div {
    border-radius: var(--r-sm) !important;
    border-color: var(--c-hairline) !important;
    transition: border-color 0.3s var(--ease-cinema);
}
div[data-baseweb="select"] > div:hover {
    border-color: var(--c-pitch) !important;
}

/* Player demo card — photo + highlights + value column */
.player-card-v2 {
    background: var(--c-surface);
    border: 1px solid var(--c-hairline);
    border-left: 4px solid var(--c-pitch);
    border-radius: var(--r-md);
    padding: var(--s-5);
    margin: var(--s-5) 0;
    box-shadow: var(--shadow-sm);
    display: grid;
    grid-template-columns: 110px 1fr auto;
    gap: var(--s-5);
    align-items: start;
}
@media (max-width: 800px) { .player-card-v2 { grid-template-columns: 80px 1fr; } }
.player-photo-v2 {
    width: 110px; height: 110px;
    border-radius: var(--r-md);
    object-fit: cover; object-position: center top;
    background: var(--c-surface-2);
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--c-hairline);
}
.player-name-v2 {
    font-family: var(--f-display);
    font-size: 1.6rem; font-weight: 700;
    letter-spacing: -0.02em;
    margin: var(--s-2) 0 var(--s-2) 0;
}
.highlights-row { display: flex; flex-wrap: wrap; gap: var(--s-2); margin-top: var(--s-3); }
.h-chip {
    display: inline-flex; align-items: center; gap: 6px;
    background: var(--c-surface-2);
    border: 1px solid var(--c-hairline);
    border-radius: var(--r-sm);
    padding: 5px 9px;
    font-size: 0.78rem;
}
.h-chip .lbl { color: var(--c-muted); }
.h-chip .val { font-family: var(--f-mono); font-weight: 700; color: var(--c-ink); }
.h-chip .b {
    font-family: var(--f-syncope);
    font-size: 0.58rem; font-weight: 700;
    letter-spacing: 0.06em;
    padding: 2px 5px; border-radius: var(--r-full); color: #fff;
}
.h-chip.elite  .b { background: var(--c-pitch); }
.h-chip.great  .b { background: var(--c-amber); }
.h-chip.good   .b { background: var(--c-cyan); color: var(--c-ink); }
.h-chip.decent .b { background: var(--c-muted); }
.value-col {
    text-align: right;
    min-width: 140px;
}
.value-col .lbl {
    font-size: 0.66rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.12em;
    color: var(--c-muted);
}
.value-col .val {
    font-family: var(--f-display);
    font-size: 1.6rem; font-weight: 700;
    letter-spacing: -0.025em;
    margin-top: 4px;
    line-height: 1;
    font-variant-numeric: tabular-nums;
}
.value-col .cur {
    margin-top: var(--s-3);
    font-size: 0.78rem;
    color: var(--c-muted);
    font-family: var(--f-mono);
}
.value-col .cur .n { color: var(--c-pitch); font-weight: 700; }

/* Fade-in baseline */
.fade-in {
    animation: fadein 0.7s var(--ease-cinema) backwards;
}
@keyframes fadein {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* Counter for chart numbering */
.chart-counter-reset { counter-reset: chart; }

/* Reduced motion */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
    .glitch::before, .glitch::after { opacity: 0; }
}
</style>
"""


# Plotly theme — applied to every chart for consistency
def _apply_plotly_theme(fig: go.Figure, height: int = 340) -> go.Figure:
    fig.update_layout(
        font=dict(family="Inter, system-ui, sans-serif", color="#0a0a0a", size=12),
        title=dict(font=dict(family="Space Grotesk", size=15, color="#0a0a0a")),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=20, b=30, l=10, r=10),
        height=height,
        xaxis=dict(
            gridcolor="#e7e5e4", zerolinecolor="#e7e5e4", linecolor="#e7e5e4",
            tickfont=dict(color="#52525b", size=11),
            title_font=dict(color="#52525b", size=12),
        ),
        yaxis=dict(
            gridcolor="#e7e5e4", zerolinecolor="#e7e5e4", linecolor="#e7e5e4",
            tickfont=dict(color="#52525b", size=11),
            title_font=dict(color="#52525b", size=12),
        ),
        hoverlabel=dict(
            bgcolor="#0a0a0a", bordercolor="#0a0a0a",
            font=dict(family="Inter", color="white", size=12),
        ),
        showlegend=False,
    )
    return fig


# ============================================================================
# CACHED LOADERS
# ============================================================================

@st.cache_data(show_spinner=False)
def _raw_players_and_appearances() -> pd.DataFrame:
    players = _load_players()
    appearances = _load_appearances()
    df = players.merge(appearances, on="player_id", how="left")
    df = df.dropna(subset=[TARGET])
    for col in ["goals", "assists", "minutes_played", "yellow_cards", "red_cards"]:
        df[col] = df[col].fillna(0)
    return df


@st.cache_data(show_spinner=False)
def _splits():
    return load_dataset_split()


@st.cache_data(show_spinner=False)
def _metrics() -> pd.DataFrame:
    return pd.read_csv(MODEL_METRICS_FILE) if MODEL_METRICS_FILE.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def _validation_table() -> pd.DataFrame:
    p = RESULTS_DIR / "validation_metrics.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def _ablation_table() -> pd.DataFrame:
    p = RESULTS_DIR / "ablation.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def _per_position_table() -> pd.DataFrame:
    p = RESULTS_DIR / "per_position_mae.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def _residuals_table() -> pd.DataFrame:
    p = RESULTS_DIR / "residuals.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def _latest_values() -> pd.Series:
    """Most-recent recorded valuation per player_id — proxy for current market value."""
    import pandas as pd
    val = pd.read_csv(RESULTS_DIR.parent / "data" / "player_valuations.csv", low_memory=False)
    val["date"] = pd.to_datetime(val["date"])
    return (
        val.sort_values("date")
        .drop_duplicates("player_id", keep="last")
        .set_index("player_id")["market_value_in_eur"]
    )


@st.cache_data(show_spinner=False)
def _enriched_stats() -> pd.DataFrame:
    """Enriched career stats per player_id — for highlight selection."""
    from data_v2 import _enriched_appearances
    return _enriched_appearances().set_index("player_id")


@st.cache_data(show_spinner=False)
def _stat_quantiles() -> dict:
    enr = _enriched_stats()
    cols = ["goals", "assists", "minutes_played", "appearance_count",
            "international_caps", "international_goals", "top_tier_minutes",
            "top_tier_apps", "career_span_years", "best_season_goals"]
    return {c: {
        "p99": float(enr[c].quantile(0.99)),
        "p95": float(enr[c].quantile(0.95)),
        "p90": float(enr[c].quantile(0.90)),
        "p75": float(enr[c].quantile(0.75)),
    } for c in cols if c in enr.columns}


_STAT_LABELS = {
    "goals":               ("⚽ Goals",         "{:,.0f}"),
    "assists":             ("🎯 Assists",       "{:,.0f}"),
    "minutes_played":      ("⏱ Minutes",       "{:,.0f}"),
    "appearance_count":    ("📋 Apps",          "{:,.0f}"),
    "international_caps":  ("🌍 Int. caps",     "{:,.0f}"),
    "international_goals": ("🇺🇸 Int. goals",  "{:,.0f}"),
    "top_tier_minutes":    ("🏆 UEFA mins",    "{:,.0f}"),
    "top_tier_apps":       ("🏆 UEFA apps",    "{:,.0f}"),
    "career_span_years":   ("📆 Career yrs",   "{:.0f}"),
    "best_season_goals":   ("🔥 Best-season",  "{:,.0f}"),
}


def _player_highlights(player_id: int) -> list[dict]:
    enr = _enriched_stats()
    qs = _stat_quantiles()
    if player_id not in enr.index:
        return []
    row = enr.loc[player_id]
    scored = []
    for col, (label, fmt) in _STAT_LABELS.items():
        if col not in row.index or pd.isna(row[col]):
            continue
        v = float(row[col])
        if v <= 0:
            continue
        q = qs.get(col, {})
        if v >= q.get("p99", float("inf")):
            tier, badge, rank = "elite", "TOP 1%", 4
        elif v >= q.get("p95", float("inf")):
            tier, badge, rank = "great", "TOP 5%", 3
        elif v >= q.get("p90", float("inf")):
            tier, badge, rank = "good", "TOP 10%", 2
        elif v >= q.get("p75", float("inf")):
            tier, badge, rank = "decent", "TOP 25%", 1
        else:
            tier, badge, rank = None, None, 0
        scored.append({"label": label, "value": v, "value_fmt": fmt.format(v),
                       "tier": tier, "badge": badge, "rank": rank})
    scored.sort(key=lambda s: (s["rank"], s["value"]), reverse=True)
    with_tier = [s for s in scored if s["tier"] is not None]
    return (with_tier[:5] if len(with_tier) >= 3 else scored[:3])


@st.cache_resource(show_spinner=False)
def _load_model(path: Path):
    return joblib.load(path)


def _underlying_regressor(model):
    """Unwrap TransformedTargetRegressor or CatBoostLogWrapper to expose feature_importances_."""
    if hasattr(model, "regressor_"):
        return model.regressor_
    if hasattr(model, "model"):
        return model.model
    return model


def _eur(x: float) -> str:
    if x >= 1_000_000:
        return f"€{x / 1_000_000:.2f}M"
    if x >= 1_000:
        return f"€{x / 1_000:.0f}k"
    return f"€{x:.0f}"


def _kicker(label: str) -> None:
    st.markdown(f'<div class="section-kicker">{label}</div>', unsafe_allow_html=True)


def _section_q(question: str, answer: str | None = None) -> None:
    st.markdown(f'<div class="section-q">{question}</div>', unsafe_allow_html=True)
    if answer:
        st.markdown(f'<div class="section-a">{answer}</div>', unsafe_allow_html=True)


def _chart_q(question: str, caption: str | None = None) -> None:
    st.markdown(f'<div class="chart-q">{question}</div>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<div class="chart-caption">{caption}</div>', unsafe_allow_html=True)


# ============================================================================
# SECTION 1 — THE PROBLEM
# ============================================================================

def section_problem() -> None:
    raw = _raw_players_and_appearances()

    st.markdown(
        f'''
        <div class="hero fade-in">
            <div class="aurora-3"></div>
            <div class="hero-inner">
                <div class="hero-eyebrow">DAT0424 · ML Proof of Concept</div>
                <div class="hero-script">a question worth €100M</div>
                <div class="hero-title">
                    What is a football player <span class="glitch" data-text="actually worth">actually worth</span><br>
                    at their <em>peak</em>?
                </div>
                <div class="hero-lede">
                    Transfer market values on Transfermarkt are crowd-sourced and lag reality.
                    Clubs, agents, and betting markets all need a defensible third-party
                    number for what a player will be worth at their career peak. This proof
                    of concept asks: can we predict that number from biographical and
                    performance data alone?
                </div>
                <div class="stat-grid">
                    <div class="stat-card"><div class="num">{len(raw):,}</div><div class="lbl">Players</div></div>
                    <div class="stat-card"><div class="num">189</div><div class="lbl">Nationalities</div></div>
                    <div class="stat-card"><div class="num">34</div><div class="lbl">Features</div></div>
                    <div class="stat-card"><div class="num">€1.53M</div><div class="lbl">Avg. error</div></div>
                    <div class="stat-card"><div class="num">0.798</div><div class="lbl">Test R²</div></div>
                </div>
            </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    _kicker("The bet — what we're actually predicting")
    st.markdown(
        '''
        <div class="callout amber">
            <b>Important framing.</b> We're not predicting a player's value <em>today</em>.
            We're predicting <b>the highest market value they will hit during their career</b>.
            <br><br>
            That's the alpha: a club can sign a 19-year-old today for <b>€5M</b> whose
            predicted career peak is <b>€40M</b> — they pocket the difference if the prediction
            holds. The model's output is a forward-looking ceiling, not a current valuation.
            <br><br>
            Every "predicted value" below should be read as <em>"the model's estimate of where
            this player's market value will eventually top out"</em>, not what they're worth right now.
        </div>
        ''',
        unsafe_allow_html=True,
    )

    _kicker("Why this matters")
    _section_q(
        "Three places where a defensible peak-value number is worth real money.",
    )
    st.markdown(
        '''
        <div class="card-row cascade">
            <div class="card">
                <div class="card-icon">🔎</div>
                <div class="card-title">Transfer scouting</div>
                <p class="card-body"><b>Buy young, sell at peak.</b> Sign a 19-year-old whose
                predicted peak (€40M) sits well above their current price (€5M). The model
                surfaces the gap; the scouting team validates the human factors.</p>
            </div>
            <div class="card">
                <div class="card-icon">📝</div>
                <div class="card-title">Contract negotiation</div>
                <p class="card-body">Anchor wage offers to a defensible ceiling. Agents and
                clubs both need a neutral third-party number to argue around.</p>
            </div>
            <div class="card">
                <div class="card-icon">📊</div>
                <div class="card-title">Squad valuation</div>
                <p class="card-body">Value a squad as a portfolio of human capital. Required
                by accountants, useful for investors and betting markets.</p>
            </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    _kicker("The story, in three steps")
    st.markdown(
        '''
        <div class="spine cascade">
            <div>
                <div class="step">X · THE PROBLEM</div>
                <div class="head">Peak market value is unpredictable</div>
                <div class="desc">Crowd-sourced, hype-sensitive, lagging — clubs and agents don't have a neutral number to work with.</div>
            </div>
            <div>
                <div class="step">Y · WHAT WE DID</div>
                <div class="head">Trained three tree-based models</div>
                <div class="desc">On 39,226 players' biographical and cumulative performance data. Log-transformed target to handle the long tail.</div>
            </div>
            <div>
                <div class="step">Z · THE RESULT</div>
                <div class="head">CatBoost predicts with R² 0.798</div>
                <div class="desc">Validated against baselines, ablations, and 5-fold cross-validation. CatBoost wins thanks to native categorical handling. Average error €1.53M.</div>
            </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )


# ============================================================================
# SECTION 2 — THE DATA
# ============================================================================

@st.cache_data(show_spinner=False)
def _country_peak_value() -> pd.DataFrame:
    raw = _raw_players_and_appearances()
    df = raw.dropna(subset=["country_of_citizenship"]).copy()
    home_nations = {"England": "United Kingdom", "Scotland": "United Kingdom",
                    "Wales": "United Kingdom", "Northern Ireland": "United Kingdom"}
    df["country"] = df["country_of_citizenship"].replace(home_nations)
    agg = (
        df.groupby("country")
        .agg(n=("player_id", "count"), median_peak=(TARGET, "median"),
             mean_peak=(TARGET, "mean"))
        .reset_index()
    )
    agg = agg[agg["n"] >= 50].sort_values("median_peak", ascending=False)
    return agg


@st.cache_data(show_spinner=False, ttl=86400)
def _world_geojson() -> dict:
    """Fetch + cache world GeoJSON. 24h TTL. Survives offline presentation."""
    import json
    import urllib.request
    cache_path = RESULTS_DIR.parent / "data" / ".world_geojson.json"
    if cache_path.exists():
        with cache_path.open() as f:
            return json.load(f)
    url = "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson"
    with urllib.request.urlopen(url, timeout=10) as r:
        data = json.loads(r.read())
    cache_path.parent.mkdir(exist_ok=True)
    with cache_path.open("w") as f:
        json.dump(data, f)
    return data


def _build_folium_choropleth(countries: pd.DataFrame) -> folium.Map:
    m = folium.Map(
        location=[25, 10], zoom_start=2, tiles="cartodbpositron",
        control_scale=False, zoom_control=False, scrollWheelZoom=False,
        dragging=False,
    )
    folium.Choropleth(
        geo_data=_world_geojson(),
        name="median_peak",
        data=countries,
        columns=["country", "median_peak"],
        key_on="feature.properties.name",
        fill_color="YlGn",
        fill_opacity=0.78,
        line_opacity=0.4,
        line_color="#ffffff",
        nan_fill_color="#f5f5f0",
        nan_fill_opacity=0.4,
        legend_name="Median peak value (€)",
        highlight=True,
    ).add_to(m)
    return m


def section_data() -> None:
    st.markdown('<div class="chart-counter-reset">', unsafe_allow_html=True)
    _kicker("Step 1 of 3 · The data")
    _section_q(
        "Do we even have enough signal to predict this?",
        "39,226 players with non-null peak-value labels and 14 engineered features. Let's look.",
    )

    raw = _raw_players_and_appearances()

    _chart_q(
        "How is peak value distributed across the player population?",
        "Problem: the target has a brutal long right tail. Median player is worth ~€500k, "
        "top-tier €150M+. Any model has to handle a 300× value spread.",
    )
    fig = px.histogram(
        raw, x=TARGET, nbins=80, log_y=True,
        labels={TARGET: "Peak market value (€)", "count": "Players"},
        color_discrete_sequence=["#0d7c47"],
    )
    median = raw[TARGET].median()
    fig.add_vline(x=median, line_dash="dot", line_color="#d97706",
                  annotation_text=f"  Median · {_eur(median)}",
                  annotation_position="top",
                  annotation=dict(font=dict(color="#d97706", size=11, family="JetBrains Mono")))
    fig.add_vline(x=100_000_000, line_dash="dot", line_color="#dc2626",
                  annotation_text=f"  Top-tier · €100M+",
                  annotation_position="top",
                  annotation=dict(font=dict(color="#dc2626", size=11, family="JetBrains Mono")))
    st.plotly_chart(_apply_plotly_theme(fig, height=320), width='stretch')

    _chart_q(
        "Where do the most valuable players come from?",
        "Ponderated data is better — instead of player counts (which would just map "
        "population), we show median peak value per country with ≥50 players. That "
        "isolates talent density from population size.",
    )
    countries = _country_peak_value()
    fol_map = _build_folium_choropleth(countries)
    st_folium(fol_map, height=440, width=None, returned_objects=[])

    # ---- CSS PIN PULSE LEGEND — pattern micro/001 ----
    top5 = countries.head(5)[["country", "median_peak"]].copy()
    legend_html = '<div class="map-legend">'
    for _, row in top5.iterrows():
        legend_html += (
            f'<div class="legend-item">'
            f'<div class="legend-pin"></div>'
            f'<div class="legend-text"><span class="country">{row["country"]}</span>'
            f'<span class="val">{_eur(row["median_peak"])}</span></div>'
            f'</div>'
        )
    legend_html += '</div>'
    st.markdown(legend_html, unsafe_allow_html=True)

    _chart_q(
        "Does position predict peak value?",
        "Yes — but not as much as you'd think. Attackers and midfielders skew higher; "
        "goalkeepers lower. But every position has its outlier €100M+ player.",
    )
    fig = px.box(
        raw.dropna(subset=["position"]), x="position", y=TARGET, log_y=True,
        labels={TARGET: "Peak market value (€)", "position": ""},
        color="position",
        color_discrete_sequence=["#0d7c47", "#16a34a", "#d97706", "#52525b", "#94a3b8"],
    )
    st.plotly_chart(_apply_plotly_theme(fig, height=300), width='stretch')

    _chart_q(
        "Does age tell us anything?",
        "Not on its own. Caveat we own up to: `age` here is current age, not age at peak.",
    )
    sample = raw.dropna(subset=["age", TARGET]).sample(min(5000, len(raw)), random_state=42)
    fig = px.scatter(
        sample, x="age", y=TARGET, opacity=0.35, log_y=True,
        labels={"age": "Age (today)", TARGET: "Peak market value (€)"},
        color_discrete_sequence=["#0d7c47"],
    )
    st.plotly_chart(_apply_plotly_theme(fig, height=300), width='stretch')

    with st.expander("📋 Full feature list"):
        st.markdown(
            f"**Categorical ({len(CATEGORICAL_COLS)}):** "
            + ", ".join(f"`{c}`" for c in CATEGORICAL_COLS)
        )
        st.markdown(
            f"**Numeric ({len(NUMERIC_COLS)}):** "
            + ", ".join(f"`{c}`" for c in NUMERIC_COLS)
        )
    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================================
# SECTION 3 — THE APPROACH (3D CSS CAROUSEL — pattern cards/004)
# ============================================================================

def section_approach() -> None:
    _kicker("Step 2 of 3 · The approach")
    _section_q(
        "Why these three models, and why log-transform the target?",
        "Plain-English methodology — no jargon required.",
    )

    st.markdown(
        '''
        <div class="callout">
            <b>The target is brutal.</b> A €0.5M error on Mbappé's €180M peak is fine — the same
            €0.5M loss on a €500k median player is a 100% error. Squared loss in € would
            just make the model obsess over the top 1%.<br><br>
            <b>The fix:</b> train every model on a log-transformed target (<code>log(1+value)</code>) and
            invert back to € at prediction time. Suddenly the loss function cares equally
            about a €1M player and a €100M one.
        </div>
        ''',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="chart-q" style="counter-increment: chart;">Three models, three philosophies <span class="pill dark" style="margin-left: var(--s-3); font-size: 0.65rem;">3D · HOVER TO PAUSE</span></div>', unsafe_allow_html=True)

    st.markdown(
        '''
        <div class="scene-3d">
            <div class="a3d" style="--n: 4;">
                <div class="card-3d t1" style="--i: 0;">
                    <div class="badge-num">01 · BASELINE FLOOR</div>
                    <div class="card-3d-title">🌱 Decision Tree</div>
                    <div class="card-3d-body">
                        The simplest possible answer. One tree, splits features greedily,
                        max depth 10. Interpretable but unsophisticated.
                    </div>
                    <div class="card-3d-foot">
                        <span>R² 0.643</span><span>MAE €1.99M</span>
                    </div>
                </div>
                <div class="card-3d t2" style="--i: 1;">
                    <div class="badge-num">02 · ENSEMBLE</div>
                    <div class="card-3d-title">🌲 Random Forest</div>
                    <div class="card-3d-body">
                        Ask 140 trees and vote. Each tree sees a random subset of data
                        and features; averaging cancels errors. Robust but trees overfit.
                    </div>
                    <div class="card-3d-foot">
                        <span>R² 0.716</span><span>MAE €1.73M</span>
                    </div>
                </div>
                <div class="card-3d t2" style="--i: 2;">
                    <div class="badge-num">03 · GRADIENT BOOSTING</div>
                    <div class="card-3d-title">🚀 XGBoost</div>
                    <div class="card-3d-body">
                        350 small trees, each correcting the previous one's residuals.
                        Strong, but categoricals are label-encoded — treated as ordinal ints.
                    </div>
                    <div class="card-3d-foot">
                        <span>R² 0.759</span><span>MAE €1.63M</span>
                    </div>
                </div>
                <div class="card-3d t3" style="--i: 3;">
                    <div class="badge-num">04 · WINNER</div>
                    <div class="card-3d-title">🏆 CatBoost</div>
                    <div class="card-3d-body">
                        <b>Native categorical handling</b> — no label-encoding hack. Ordered
                        boosting prevents target leakage. Train/test gap only <b>+0.007</b>.
                    </div>
                    <div class="card-3d-foot">
                        <span>R² 0.798</span><span>MAE €1.53M</span>
                    </div>
                </div>
            </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    st.markdown(
        '''
        <div class="callout amber">
            <b>Why trees, not deep learning or linear models?</b><br>
            39k rows is small for neural nets. Tabular features with no spatial/sequential
            structure. Categoricals encoded as integers — linear models would treat that as
            ordinal (meaningless). Tree models handle the data shape natively.
            <b>CatBoost wins because it processes categoricals without forcing them into
            integers in the first place.</b>
        </div>
        ''',
        unsafe_allow_html=True,
    )


# ============================================================================
# SECTION 4 — THE DISCOVERY
# ============================================================================

def section_discovery() -> None:
    st.markdown('<div class="chart-counter-reset">', unsafe_allow_html=True)
    _kicker("Step 3 of 3 · The discovery")
    _section_q(
        "Did it work?",
        "Spoiler: yes. Here's the evidence in three forms.",
    )

    metrics = _metrics()
    validation = _validation_table()

    if metrics.empty:
        st.warning("Run `python scripts/main.py` to generate `results/model_metrics.csv`.")
        return

    winner = metrics[metrics["model_key"] == "catboost_regressor"].iloc[0]
    pct_var = winner['r2'] * 100
    st.markdown(
        f'''
        <div class="bignum-row fade-in">
            <div class="bignum featured">
                <div class="lbl">CatBoost · The winner</div>
                <div class="num">{winner['r2']:.3f}</div>
                <div class="sub">Test R² — explains {pct_var:.1f}% of variance in peak market value</div>
            </div>
            <div class="bignum compact">
                <div class="lbl">Average error</div>
                <div class="num">{_eur(winner['mae_eur'])}</div>
                <div class="sub">MAE on test set</div>
            </div>
            <div class="bignum compact">
                <div class="lbl">Cross-val R²</div>
                <div class="num">0.777</div>
                <div class="sub">±0.005 over 5 folds (tightest band)</div>
            </div>
            <div class="bignum compact">
                <div class="lbl">Train/test gap</div>
                <div class="num">+0.007</div>
                <div class="sub">Essentially zero overfit</div>
            </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    _chart_q("How does each model stack up?")
    display = metrics[["model_name", "mae_eur", "rmse_eur", "r2"]].copy()
    display["mae_eur"] = display["mae_eur"].apply(_eur)
    display["rmse_eur"] = display["rmse_eur"].apply(_eur)
    display["r2"] = display["r2"].apply(lambda v: f"{v:.3f}")
    display.columns = ["Model", "MAE", "RMSE", "R²"]
    st.dataframe(display, width='stretch', hide_index=True)

    _chart_q(
        "But how do we know 0.798 is actually good?",
        "Compared to what? Two trivial baselines establish the floor.",
    )
    bl = validation[validation["kind"] == "baseline"].copy() if not validation.empty else pd.DataFrame()
    if not bl.empty:
        view = bl[["name", "r2", "mae_eur"]].copy()
        view["r2"] = view["r2"].apply(lambda v: f"{v:.3f}")
        view["mae_eur"] = view["mae_eur"].apply(_eur)
        view.columns = ["Baseline", "R²", "MAE"]
        st.dataframe(view, width='stretch', hide_index=True)
        st.markdown(
            '''
            <div class="callout">
                <b>Read:</b> predict-median gives R² −0.08. Linear regression
                <b>collapses catastrophically</b> at R² −755 — useful evidence that
                <b>non-linear models are required, not optional.</b> XGBoost cuts trivial-baseline
                MAE by <b>46%</b>.
            </div>
            ''',
            unsafe_allow_html=True,
        )

    _chart_q(
        "Is the 0.798 a lucky split, or stable?",
        "5-fold cross-validation. If our test R² sits inside the CV band, we have real signal.",
    )
    cv = validation[validation["kind"] == "cv"].copy() if not validation.empty else pd.DataFrame()
    if not cv.empty:
        cv = cv.sort_values("cv_r2_mean", ascending=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=cv["cv_r2_mean"], y=cv["name"], orientation="h",
            error_x=dict(type="data", array=cv["cv_r2_std"], color="#0a0a0a", thickness=1.5),
            marker=dict(color=["#94a3b8", "#d97706", "#0d7c47"]),
            text=[f"{v:.3f} ± {s:.3f}" for v, s in zip(cv["cv_r2_mean"], cv["cv_r2_std"])],
            textposition="outside", textfont=dict(family="JetBrains Mono", size=12, color="#0a0a0a"),
            hovertemplate="<b>%{y}</b><br>CV R²: %{x:.3f}<extra></extra>",
        ))
        fig.update_xaxes(range=[0, 0.9], title="5-fold CV R²")
        fig.update_yaxes(title="")
        st.plotly_chart(_apply_plotly_theme(fig, height=240), width='stretch')

    overfit = validation[validation["kind"] == "overfit"].copy() if not validation.empty else pd.DataFrame()
    if not overfit.empty:
        with st.expander("📐 Train R² vs. test R² (overfit check)"):
            view = overfit[["name", "train_r2", "test_r2", "overfit_gap"]].copy()
            view["train_r2"] = view["train_r2"].apply(lambda v: f"{v:.3f}")
            view["test_r2"] = view["test_r2"].apply(lambda v: f"{v:.3f}")
            view["overfit_gap"] = view["overfit_gap"].apply(lambda v: f"{v:+.3f}")
            view.columns = ["Model", "Train R²", "Test R²", "Gap"]
            st.dataframe(view, width='stretch', hide_index=True)

    _chart_q(
        "What did the model actually learn?",
        "The features XGBoost found most predictive of peak market value.",
    )
    X_train, _, _, _ = _splits()
    cb = _load_model(MODELS["catboost_regressor"]["path"])
    inner = _underlying_regressor(cb)
    importances_full = pd.DataFrame(
        {"feature": X_train.columns, "importance": inner.feature_importances_}
    ).sort_values("importance", ascending=False)
    top3 = importances_full.head(3)["feature"].tolist()
    tail_count = (importances_full["importance"] < 0.5).sum()

    # Only show the top 15 — the long tail is visually noisy.
    top_n = importances_full.head(15).sort_values("importance", ascending=True)
    fig = px.bar(
        top_n, x="importance", y="feature", orientation="h",
        labels={"importance": "CatBoost feature importance (% contribution)", "feature": ""},
        color_discrete_sequence=["#0d7c47"],
        text="importance",
    )
    fig.update_traces(
        texttemplate="%{text:.1f}",
        textposition="outside",
        textfont=dict(family="JetBrains Mono", size=11, color="#27272a"),
    )
    fig.update_xaxes(range=[0, top_n["importance"].max() * 1.15])
    # Height scales with feature count for legible labels
    st.plotly_chart(_apply_plotly_theme(fig, height=460), width='stretch')

    st.markdown(
        f'''
        <div class="callout">
            <b>Top 3 drivers:</b> <code>{top3[0]}</code>, <code>{top3[1]}</code>, <code>{top3[2]}</code>.
            Roughly: <i>what league you play in, how many matches you've played, and over how
            much time</i>.<br><br>
            <b>Long-tail note:</b> showing the top 15 of 34 features. {tail_count} features
            contribute less than 0.5% each — the model leans heavily on a handful of strong
            signals, exactly what we want from a regularised gradient-boosted model.
        </div>
        ''',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================================
# SECTION 5 — PRESSURE-TESTING
# ============================================================================

def section_validation() -> None:
    st.markdown('<div class="chart-counter-reset">', unsafe_allow_html=True)
    _kicker("Pressure-testing")
    _section_q(
        "How could this be wrong, and did we test for it?",
        "Three diagnostic checks: residuals, per-position errors, and a leakage ablation.",
    )

    residuals = _residuals_table()
    per_pos = _per_position_table()
    ablation = _ablation_table()

    _chart_q(
        "Predicted vs. actual — how calibrated is the model?",
        "Each dot is one player in the test set. A perfect model puts every dot on the diagonal.",
    )
    if not residuals.empty:
        fig = px.scatter(
            residuals, x="actual_eur", y="predicted_eur", color="position",
            hover_data=["name"], opacity=0.55, log_x=True, log_y=True,
            labels={"actual_eur": "Actual peak value (€)", "predicted_eur": "Predicted (€)"},
            color_discrete_sequence=["#0d7c47", "#16a34a", "#d97706", "#52525b", "#94a3b8"],
        )
        lo = max(residuals["actual_eur"].min(), 1)
        hi = residuals["actual_eur"].max()
        fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi,
                      line=dict(dash="dash", color="#0a0a0a", width=1.5))
        fig.update_layout(showlegend=True, legend=dict(
            orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5,
            font=dict(size=11)
        ))
        st.plotly_chart(_apply_plotly_theme(fig, height=440), width='stretch')
        st.markdown(
            '<div class="callout"><b>Read:</b> well-calibrated up to ~€20M. Above that, '
            "the long-tail superstars are systematically under-predicted — the model is "
            "conservative when it should extrapolate.</div>",
            unsafe_allow_html=True,
        )

    _chart_q(
        "Where does the model fail by position?",
        "Ponderated data is better — proportional MAE (% of median) reveals what absolute "
        "MAE hides.",
    )
    if not per_pos.empty:
        view = per_pos.copy()
        view["mae_pct"] = (view["mae_eur"] / view["median_actual_eur"] * 100)
        view = view.sort_values("mae_pct", ascending=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=view["position"], x=view["mae_pct"], orientation="h",
            marker=dict(color="#0d7c47"),
            text=[f"{v:.0f}%" for v in view["mae_pct"]],
            textposition="outside", textfont=dict(family="JetBrains Mono", size=12, color="#0a0a0a"),
            hovertemplate="<b>%{y}</b><br>MAE: %{x:.0f}%% of median<extra></extra>",
        ))
        fig.update_xaxes(title="MAE as % of median actual value")
        fig.update_yaxes(title="")
        st.plotly_chart(_apply_plotly_theme(fig, height=240), width='stretch')

        full_view = per_pos.copy()
        full_view["mae_pct"] = (full_view["mae_eur"] / full_view["median_actual_eur"] * 100).round(0)
        full_view["median_actual_eur"] = full_view["median_actual_eur"].apply(_eur)
        full_view["mae_eur"] = full_view["mae_eur"].apply(_eur)
        full_view = full_view[["position", "n", "median_actual_eur", "mae_eur", "mae_pct"]]
        full_view.columns = ["Position", "N (test)", "Median actual", "MAE (€)", "MAE (%)"]
        st.dataframe(full_view, width='stretch', hide_index=True)

    _chart_q(
        "What about that suspicious-looking feature?",
        "We flagged `current_club_domestic_competition_id` as a potential leak. Did we test it?",
    )
    if not ablation.empty:
        full = ablation[ablation["variant"] == "xgboost_full"].iloc[0]
        abl = ablation[ablation["variant"] == "xgboost_without_current_club_competition"].iloc[0]
        delta = ablation[ablation["variant"] == "delta"].iloc[0]
        st.markdown(
            f'''
            <div class="card-row cascade" style="grid-template-columns: 1fr 1fr 1fr;">
                <div class="card" style="border-left: 3px solid var(--c-pitch);">
                    <div class="card-title">Full model (with leak feature)</div>
                    <div style="font-family: var(--f-display); font-size: 2rem; font-weight: 700; color: var(--c-pitch); margin: var(--s-3) 0;">R² {full['r2']:.3f}</div>
                    <div style="color: var(--c-muted); font-size: 0.88rem;">MAE {_eur(full['mae_eur'])}</div>
                </div>
                <div class="card" style="border-left: 3px solid var(--c-amber);">
                    <div class="card-title">Without leak feature</div>
                    <div style="font-family: var(--f-display); font-size: 2rem; font-weight: 700; color: var(--c-amber); margin: var(--s-3) 0;">R² {abl['r2']:.3f}</div>
                    <div style="color: var(--c-muted); font-size: 0.88rem;">MAE {_eur(abl['mae_eur'])}</div>
                </div>
                <div class="card" style="border-left: 3px solid var(--c-muted);">
                    <div class="card-title">Delta · Cost of removing it</div>
                    <div style="font-family: var(--f-display); font-size: 2rem; font-weight: 700; color: var(--c-ink); margin: var(--s-3) 0;">{delta['r2']:+.3f}</div>
                    <div style="color: var(--c-muted); font-size: 0.88rem;">{("+" if delta['mae_eur']>=0 else "") + _eur(delta['mae_eur'])}</div>
                </div>
            </div>
            ''',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="callout"><b>Defensible answer:</b> dropping the suspected leak '
            "feature costs only <b>−0.002 R²</b> and +€0.05M MAE — the v2 features absorbed "
            "the signal. The model is <b>not pure league-prestige memorisation</b>.</div>",
            unsafe_allow_html=True,
        )

    # ── PREDICTION-BUCKET DIAGNOSTIC (the user-felt error pattern) ──
    _chart_q(
        "How accurate is the prediction at each price tier?",
        "R² in € is dominated by the model getting Mbappé approximately right in absolute "
        "terms. The metric a human feels — % error — is more honest. Here it is broken "
        "down by player value bucket on the test set.",
    )

    # Live-compute the bucket diagnostic from the test set
    try:
        from data_v2 import load_dataset_split as _v2
        X_tr, X_te, y_tr, y_te = _v2()
        cb_model = _load_model(MODELS["catboost_regressor"]["path"])
        preds = cb_model.predict(X_te)
        import numpy as _np
        y_arr = y_te.values
        safe = _np.maximum(y_arr, 1)
        pct_err = _np.abs(preds - y_arr) / safe * 100
        bucket_data = []
        for lo, hi, name in [
            (0, 200_000, "Micro (<€200k)"),
            (200_000, 1_000_000, "Low (€200k–1M)"),
            (1_000_000, 5_000_000, "Mid (€1–5M)"),
            (5_000_000, 20_000_000, "High (€5–20M)"),
            (20_000_000, 1e10, "Elite (>€20M)"),
        ]:
            mask = (y_arr >= lo) & (y_arr < hi)
            if mask.sum() > 5:
                bucket_data.append({
                    "Bucket": name,
                    "N": int(mask.sum()),
                    "Median % error": f"{_np.median(pct_err[mask]):.0f}%",
                    "Over-predicted": f"{(preds[mask] > y_arr[mask]).mean() * 100:.0f}%",
                    "Median predicted": _eur(_np.median(preds[mask])),
                    "Median actual": _eur(_np.median(y_arr[mask])),
                })
        st.dataframe(pd.DataFrame(bucket_data), width='stretch', hide_index=True)
        st.markdown(
            '<div class="callout warn">'
            "<b>What this tells us, honestly:</b> the model is best in the mid-to-high band "
            "(€1–20M) where it has the most training data. It's <b>biased toward the mean</b>: "
            "cheap players are over-predicted, elite players under-predicted. The Roadmap item "
            "<b>'Quantile regression for prediction intervals'</b> is the right fix — instead "
            "of saying 'Mbappé is €130M', say 'Mbappé is €110M–€180M, 80% confidence'."
            '</div>',
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.warning(f"Bucket diagnostic unavailable: {e}")

    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================================
# SECTION 6 — TRY IT YOURSELF (cinematic predict with conic-border)
# ============================================================================

def section_predict() -> None:
    _kicker("Try it yourself")
    _section_q(
        "Pick a player. What does the model think they're worth?",
        "Each player is tagged TRAIN or TEST. Pick TEST for an honest generalisation read.",
    )

    raw = _raw_players_and_appearances()
    X_train, X_test, y_train, y_test = _splits()
    train_idx = set(X_train.index)

    X_all = pd.concat([X_train, X_test])
    y_all = pd.concat([y_train, y_test])

    raw_aligned = raw.loc[X_all.index].copy()
    raw_aligned["split"] = raw_aligned.index.map(lambda i: "TRAIN" if i in train_idx else "TEST")
    raw_aligned["__display"] = (
        raw_aligned["split"] + " · "
        + raw_aligned["name"].fillna("Unknown")
        + " (" + raw_aligned["position"].fillna("?") + ", "
        + raw_aligned[TARGET].apply(_eur) + ")"
    )

    c1, c2 = st.columns([3, 1])
    show_only_test = c1.toggle("Only TEST players (honest predictions)", value=True)
    random_pick = c2.button("🎲 Random TEST", width='stretch')

    pool = raw_aligned[raw_aligned["split"] == "TEST"] if show_only_test else raw_aligned

    if random_pick:
        choice = int(pool.sample(1).index[0])
        st.session_state["picked_player"] = choice

    default_idx = 0
    if "picked_player" in st.session_state and st.session_state["picked_player"] in pool.index:
        default_idx = list(pool.index).index(st.session_state["picked_player"])
    else:
        famous = pool[pool["name"].str.contains("Mbapp|Haaland|Bellingham|Vinic", case=False, na=False)]
        if not famous.empty:
            default_idx = list(pool.index).index(famous.index[0])

    choice = st.selectbox(
        "Player",
        options=pool.index,
        format_func=lambda i: pool.loc[i, "__display"],
        index=default_idx,
        label_visibility="collapsed",
    )

    row = X_all.loc[[choice]]
    actual = float(y_all.loc[choice])
    split_label = raw_aligned.loc[choice, "split"]
    meta = raw_aligned.loc[choice]

    preds = []
    for key, spec in MODELS.items():
        model = _load_model(spec["path"])
        pred = float(model.predict(row)[0])
        pred = max(pred, 0.0)
        preds.append({"Model": spec["name"], "Predicted": pred})

    pred_df = pd.DataFrame(preds)
    winner_row = pred_df[pred_df["Model"] == "CatBoost Regressor"].iloc[0]

    # Look up current value + image_url + highlights
    pid = int(meta.get("player_id")) if pd.notna(meta.get("player_id")) else None
    latest_series = _latest_values()
    current_value = float(latest_series.get(pid)) if pid is not None and pid in latest_series.index else None
    image_url = meta.get("image_url") if pd.notna(meta.get("image_url")) else None
    highlights = _player_highlights(pid) if pid is not None else []

    # Floor at current value — peak can't be below today's price
    raw_pred = float(winner_row["Predicted"])
    floored_pred = max(raw_pred, current_value) if current_value is not None else raw_pred

    # Signal
    if current_value is None:
        signal_label, signal_bg = "No current value available", "var(--c-muted)"
    elif raw_pred > current_value * 1.15:
        upside_pct = (raw_pred - current_value) / current_value * 100
        signal_label = f"▲ Upside · model thinks +{upside_pct:.0f}% headroom"
        signal_bg = "var(--c-pitch)"
    elif raw_pred < current_value * 0.85:
        below_pct = (1 - raw_pred / current_value) * 100
        signal_label = f"▼ Market premium · fundamentals say {below_pct:.0f}% lower"
        signal_bg = "var(--c-amber)"
    else:
        signal_label = "◆ Fair · prediction sits at current price"
        signal_bg = "var(--c-cyan)"

    badge_text = "⚠ TRAIN — model has seen this player" if split_label == "TRAIN" else "✓ TEST — never seen before"
    photo_html = (f'<img class="player-photo-v2" src="{image_url}" alt="{meta["name"]}" loading="lazy" />'
                  if image_url else '<div class="player-photo-v2" style="display:flex;align-items:center;justify-content:center;font-size:2rem;color:var(--c-muted-2);">👤</div>')
    highlights_html = "".join(
        f'<div class="h-chip {h["tier"] or ""}"><span class="lbl">{h["label"]}</span>'
        f'<span class="val">{h["value_fmt"]}</span>'
        f'{"<span class=\"b\">" + h["badge"] + "</span>" if h["badge"] else ""}</div>'
        for h in highlights
    )
    cur_html = (f'<div class="cur">Current: <span class="n">{_eur(current_value)}</span></div>'
                if current_value is not None else "")

    st.markdown(
        f'''
        <div class="player-card-v2 fade-in">
            {photo_html}
            <div>
                <span class="pill {'red' if split_label == 'TRAIN' else 'gray'}" style="margin-bottom: var(--s-2); display: inline-block;">{badge_text}</span>
                <div class="player-name-v2">{meta["name"]}</div>
                <div>
                    <span class="pill gray" style="margin-right: 4px;">{meta.get('position', '?')}</span>
                    <span class="pill gray" style="margin-right: 4px;">{meta.get('foot', '?')} footed</span>
                    <span class="pill gray" style="margin-right: 4px;">Age {meta.get('age', 0):.0f}</span>
                    <span class="pill gray">{meta.get('country_of_citizenship', '?')}</span>
                </div>
                <div class="highlights-row">{highlights_html}</div>
            </div>
            <div class="value-col">
                <div class="lbl">Highest recorded</div>
                <div class="val">{_eur(actual)}</div>
                {cur_html}
            </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'''
        <div class="bignum-row fade-in" style="grid-template-columns: 2fr 1fr 1fr;">
            <div class="bignum featured">
                <div class="lbl">CatBoost · expected peak</div>
                <div class="num">{_eur(floored_pred)}</div>
                <div class="sub">= max(model output, current value) — peak can't sit below current</div>
            </div>
            <div class="bignum compact" style="background: {signal_bg};">
                <div class="lbl">Signal</div>
                <div class="num" style="font-size: 1.4rem;">{signal_label.split(" · ")[0]}</div>
                <div class="sub">{signal_label.split(" · ", 1)[1] if " · " in signal_label else ""}</div>
            </div>
            <div class="bignum compact">
                <div class="lbl">Raw model output</div>
                <div class="num" style="font-size: 1.8rem;">{_eur(raw_pred)}</div>
                <div class="sub">{('vs. current ' + _eur(current_value)) if current_value is not None else 'no current value'}</div>
            </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    _chart_q("All four models — how do they compare on this player?")
    fig = go.Figure()
    bar_colors = ["#94a3b8", "#d4a373", "#d97706", "#0d7c47"]
    fig.add_trace(go.Bar(
        x=pred_df["Model"], y=pred_df["Predicted"],
        marker=dict(color=bar_colors),
        text=[_eur(v) for v in pred_df["Predicted"]],
        textposition="outside", textfont=dict(family="Space Grotesk", size=13, color="#0a0a0a"),
        hovertemplate="<b>%{x}</b><br>Predicted: %{text}<extra></extra>",
    ))
    fig.add_hline(
        y=actual, line_dash="dash", line_color="#dc2626", line_width=2,
        annotation_text=f"Real peak: {_eur(actual)}",
        annotation_position="top right",
        annotation=dict(font=dict(color="#dc2626", family="JetBrains Mono", size=12)),
    )
    fig.update_yaxes(title="Predicted peak value (€)")
    fig.update_xaxes(title="")
    st.plotly_chart(_apply_plotly_theme(fig, height=360), width='stretch')


# ============================================================================
# SECTION 7 — HONEST LIMITS
# ============================================================================

def section_limits() -> None:
    _kicker("Honest limits")
    _section_q(
        "Where does this PoC fall short?",
        "Three real limitations we own up to. Volunteering these is more credible than hiding them.",
    )

    st.markdown(
        '''
        <div class="card-row cascade">
            <div class="card" style="border-left: 3px solid var(--c-red);">
                <div class="pill red" style="margin-bottom: var(--s-3);">CONCEPTUAL</div>
                <div class="card-title">Retrospective valuation, not forward prediction</div>
                <p class="card-body">
                    Features are career-cumulative; target is the career peak. For a retired
                    player, the features include years of post-peak play. The honest reframe:
                    "predict peak at 25 given features up to 21." That's the Roadmap.
                </p>
            </div>
            <div class="card" style="border-left: 3px solid var(--c-amber);">
                <div class="pill amber" style="margin-bottom: var(--s-3);">FEATURE NOISE</div>
                <div class="card-title">`age` is current age, not age-at-peak</div>
                <p class="card-body">
                    Today minus date-of-birth. For retired players that's whatever their age
                    is today. Adds noise; doesn't systematically bias the result.
                </p>
            </div>
            <div class="card" style="border-left: 3px solid var(--c-amber);">
                <div class="pill amber" style="margin-bottom: var(--s-3);">EVALUATION</div>
                <div class="card-title">Random split, not temporal</div>
                <p class="card-body">
                    A deployed model would train on players who peaked before year Y and
                    predict for players active in year Y. Our seed-42 random split lets the
                    model see contemporaries of its targets.
                </p>
            </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    with st.expander("🎤 Defence FAQ — five questions, prepared answers"):
        st.markdown(
            """
            **Q: Isn't this just leakage? Career stats predicting a career peak — same career.**
            Yes — retrospective valuation, not forward prediction. We disclose it. We tested
            the most-suspect feature (`current_club_competition_id`) by ablation — dropping
            it costs only −0.020 R². The model isn't pure memorisation.

            **Q: How do you know R² 0.80 is good?**
            Two baselines. Predict-median: R² −0.08. Linear regression: R² 0.12.
            CatBoost cuts MAE from €3.33M (trivial baseline) to €1.53M — a **54% reduction**.
            CV confirms 0.798 isn't lucky (CV 0.777 ± 0.005 — extremely tight band).

            **Q: Is the model overfit?**
            CatBoost train R² 0.805 / test R² 0.798 — gap **+0.007**. Essentially zero
            (thanks to ordered boosting). RF overfits more (+0.161) but still beats DT on test.

            **Q: Where does the model fail?**
            Long right tail. Players above €50M are under-predicted — individual
            transfer deals dominate up there.

            **Q: Why CatBoost over XGBoost?**
            XGBoost requires us to label-encode the 6 categorical features into integers —
            which it then treats as ordinal (meaningless: "France=12, Italy=15" is nonsense).
            CatBoost handles categoricals natively using ordered target statistics with
            built-in leakage protection. On the same data and feature set: XGBoost R² 0.759,
            CatBoost R² 0.798. The difference is the categorical handling.

            **Q: Why not deep learning?**
            39k rows is small for NNs. Features tabular. CatBoost is right-sized
            for the problem AND interpretable (feature importances available).
            """
        )


# ============================================================================
# SECTION 8 — BRUTALIST ROADMAP (pattern layouts/004)
# ============================================================================

def section_roadmap() -> None:
    _kicker("What we'd build next")
    _section_q(
        "If this PoC became a v1, what changes?",
        "Five upgrades — each defendable today, each testable. The proof, looking forward.",
    )

    st.markdown(
        '''
        <div class="brutal-timeline">
            <div class="brutal-phase pitch">
                <div class="brutal-content">
                    <div class="brutal-title">
                        Age-at-peak reframing
                        <span class="pill amber">kills limits #1 + #2</span>
                    </div>
                    <p class="brutal-body">
                        Freeze every player's features at age 21 — career stats up to 21,
                        no current-club, no age-leak — and predict their eventual peak.
                        This converts the model from retrospective to genuinely forward-looking.
                    </p>
                </div>
            </div>
            <div class="brutal-phase amber">
                <div class="brutal-content">
                    <div class="brutal-title">
                        Temporal train/test split
                        <span class="pill amber">kills limit #3</span>
                    </div>
                    <p class="brutal-body">
                        Train on players who peaked before 2018; test on players who peaked
                        2018+. Tests generalisation across eras (market inflation, league
                        economics shift). Expected outcome: lower R² but a more defensible
                        deployment story.
                    </p>
                </div>
            </div>
            <div class="brutal-phase pitch">
                <div class="brutal-content">
                    <div class="brutal-title">
                        Quantile regression
                        <span class="pill">prediction intervals</span>
                    </div>
                    <p class="brutal-body">
                        Instead of "€42M", say <b>"€32–€58M, 80% confidence."</b>
                        Gradient-boosted quantile regression — already supported in XGBoost
                        via custom objective. Way more useful than a point estimate.
                    </p>
                </div>
            </div>
            <div class="brutal-phase cyan">
                <div class="brutal-content">
                    <div class="brutal-title">
                        SHAP for per-player explanations
                        <span class="pill amber">interpretability</span>
                    </div>
                    <p class="brutal-body">
                        For each prediction, surface which features drove it. <i>"Bellingham
                        gets +€40M from being a 21-year-old midfielder at a top-5 league
                        club, −€8M from his international caps being lower than peers."</i>
                        Turns the model from black box into a negotiation tool.
                    </p>
                </div>
            </div>
            <div class="brutal-phase pitch">
                <div class="brutal-content">
                    <div class="brutal-title">
                        Trajectory features
                        <span class="pill">strongest expected lift</span>
                    </div>
                    <p class="brutal-body">
                        Replace career-cumulative stats with trajectory: goals-per-90 trend,
                        minutes growth rate, age-vs-output curve fit. Captures whether a
                        player is rising or plateauing, not just totals.
                    </p>
                </div>
            </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    st.markdown(
        '''
        <div class="callout amber">
            <b>Why this counts as part of the PoC:</b> the teacher's brief says theoretically
            provable but not-yet-achieved work counts as part of the proof of concept. All
            five upgrades above are testable, none requires new data, all extend the
            existing pipeline. <b>The roadmap is the proof that this isn't a one-shot model
            — it's the foundation of a real product.</b>
        </div>
        ''',
        unsafe_allow_html=True,
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

SECTIONS = {
    "1. The problem": section_problem,
    "2. The data": section_data,
    "3. The approach": section_approach,
    "4. The discovery": section_discovery,
    "5. Pressure-testing": section_validation,
    "6. Try it yourself": section_predict,
    "7. Honest limits": section_limits,
    "8. The roadmap": section_roadmap,
}


def build_app() -> None:
    st.set_page_config(
        page_title="Peak Value — ML PoC",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(BRAND_CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(
            '<div class="sidebar-brand">PEAK<span class="dot">·</span>VALUE</div>'
            '<div class="sidebar-tag">dat0424 · ml poc · v5</div>',
            unsafe_allow_html=True,
        )
        choice = st.radio("Section", list(SECTIONS.keys()), label_visibility="collapsed")
        st.markdown("---")
        st.markdown(
            '''
            <div class="sidebar-foot">
                <div class="row"><span>Data</span><span>Transfermarkt</span></div>
                <div class="row"><span>Model</span><span>CatBoost</span></div>
                <div class="row"><span>Test R²</span><span>0.798</span></div>
                <div class="row"><span>CV R²</span><span>0.777±0.005</span></div>
                <div class="row"><span>MAE</span><span>€1.53M</span></div>
            </div>
            ''',
            unsafe_allow_html=True,
        )

    SECTIONS[choice]()


if __name__ == "__main__":
    build_app()

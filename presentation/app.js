/* ============================================================================
   Peak Value — interactive layer
   Patterns: lerp cursor + magnetic, hacker scramble, sliding pill nav,
             counter animations, mouse parallax, canvas spotlight,
             scroll-spy, day/night toggle, country bar reveals,
             player demo logic.
   ============================================================================ */

(function () {
    'use strict';

    const data = window.POC_DATA;
    if (!data) {
        console.error('POC_DATA not loaded. Run build_data.py first.');
        return;
    }

    const isReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const isCoarse = window.matchMedia('(hover: none), (pointer: coarse)').matches;

    // ── Helpers ────────────────────────────────────────────────────────────
    const $ = (sel, ctx = document) => ctx.querySelector(sel);
    const $$ = (sel, ctx = document) => Array.from(ctx.querySelectorAll(sel));
    const lerp = (a, b, t) => a + (b - a) * t;
    const fmtEur = (x) => {
        if (x >= 1e6) return `€${(x / 1e6).toFixed(2)}M`;
        if (x >= 1e3) return `€${Math.round(x / 1e3)}k`;
        return `€${Math.round(x)}`;
    };
    const fmtEurNoSign = (x) => fmtEur(Math.abs(x));
    const fmtSigned = (x) => (x >= 0 ? '+' : '−') + fmtEurNoSign(x);


    // ── SCROLL-DRIVEN SVG PATHS — JS fallback for browsers without
    //    animation-timeline: scroll() (Safari < 17.4, older Firefox)
    //    The CSS @supports rule handles the modern path; this only fires when
    //    that fails so the dots still travel as you scroll.
    const supportsScrollTimeline = (() => {
        try { return CSS.supports('animation-timeline', 'scroll(root)'); }
        catch (e) { return false; }
    })();
    if (!supportsScrollTimeline && !isReducedMotion) {
        const forwardTracks = document.querySelectorAll(
            '.bundle-1 .track, .bundle-3 .track, .bundle-5 .track'
        );
        const reverseTracks = document.querySelectorAll(
            '.bundle-2 .track, .bundle-4 .track, .bundle-6 .track'
        );
        const onScroll = () => {
            const max = document.documentElement.scrollHeight - window.innerHeight;
            const prog = max > 0 ? window.scrollY / max : 0;
            const baseOffset = -1600 * prog;
            forwardTracks.forEach(t => { t.style.strokeDashoffset = baseOffset.toFixed(1); });
            reverseTracks.forEach(t => { t.style.strokeDashoffset = (-baseOffset).toFixed(1); });
        };
        let ticking = false;
        window.addEventListener('scroll', () => {
            if (!ticking) {
                requestAnimationFrame(() => { onScroll(); ticking = false; });
                ticking = true;
            }
        }, { passive: true });
        onScroll();
    }


    // ── THEME TOGGLE (day/night with localStorage) ─────────────────────────
    const themeToggle = $('#themeToggle');
    const saved = localStorage.getItem('peak-value-theme');
    if (saved === 'dark') {
        document.documentElement.dataset.theme = 'dark';
        themeToggle.setAttribute('aria-checked', 'true');
    }
    themeToggle.addEventListener('click', () => {
        const dark = document.documentElement.dataset.theme !== 'dark';
        document.documentElement.dataset.theme = dark ? 'dark' : 'light';
        themeToggle.setAttribute('aria-checked', String(dark));
        localStorage.setItem('peak-value-theme', dark ? 'dark' : 'light');
    });


    // ── LERP CURSOR (pattern hero/001 + layouts/004) ───────────────────────
    if (!isCoarse && !isReducedMotion) {
        const dot = $('#cursor-dot');
        const ring = $('#cursor-ring');
        let mx = window.innerWidth / 2, my = window.innerHeight / 2;
        let rx = mx, ry = my;

        document.addEventListener('mousemove', (e) => {
            mx = e.clientX; my = e.clientY;
            dot.style.transform = `translate(${mx}px, ${my}px) translate(-50%, -50%)`;
        });
        const tick = () => {
            rx = lerp(rx, mx, 0.16);
            ry = lerp(ry, my, 0.16);
            ring.style.transform = `translate(${rx}px, ${ry}px) translate(-50%, -50%)`;
            requestAnimationFrame(tick);
        };
        tick();

        // Magnetic state on .magnetic + interactive elements
        const magneticSelector = '.magnetic, a, button, .card-3d, .stat-card, .brutal-phase, summary';
        document.addEventListener('mouseover', (e) => {
            if (e.target.closest(magneticSelector)) document.body.classList.add('cursor-magnet');
            if (e.target.closest('input, select, textarea')) document.body.classList.add('cursor-text');
        });
        document.addEventListener('mouseout', (e) => {
            if (e.target.closest(magneticSelector)) document.body.classList.remove('cursor-magnet');
            if (e.target.closest('input, select, textarea')) document.body.classList.remove('cursor-text');
        });
    }


    // ── CANVAS SPOTLIGHT — hero only ───────────────────────────────────────
    if (!isCoarse && !isReducedMotion) {
        const canvas = $('#spotlight');
        const ctx = canvas.getContext('2d');
        let cw = canvas.width = window.innerWidth;
        let ch = canvas.height = window.innerHeight;
        let sx = cw / 2, sy = ch / 2;
        let tx = sx, ty = sy;

        window.addEventListener('resize', () => {
            cw = canvas.width = window.innerWidth;
            ch = canvas.height = window.innerHeight;
        });
        document.addEventListener('mousemove', (e) => {
            tx = e.clientX; ty = e.clientY;
        });

        const heroEl = $('#problem');
        const obs = new IntersectionObserver((entries) => {
            entries.forEach(en => {
                document.body.classList.toggle('spotlight-on', en.isIntersecting);
            });
        }, { threshold: 0.3 });
        obs.observe(heroEl);

        const drawSpot = () => {
            sx = lerp(sx, tx, 0.1); sy = lerp(sy, ty, 0.1);
            ctx.clearRect(0, 0, cw, ch);
            const dark = document.documentElement.dataset.theme === 'dark';
            const grad = ctx.createRadialGradient(sx, sy, 0, sx, sy, 320);
            grad.addColorStop(0, dark ? 'rgba(22,163,74,0.18)' : 'rgba(13,124,71,0.12)');
            grad.addColorStop(0.5, dark ? 'rgba(217,119,6,0.05)' : 'rgba(217,119,6,0.04)');
            grad.addColorStop(1, 'rgba(0,0,0,0)');
            ctx.fillStyle = grad;
            ctx.beginPath();
            ctx.arc(sx, sy, 320, 0, Math.PI * 2);
            ctx.fill();
            requestAnimationFrame(drawSpot);
        };
        drawSpot();
    }


    // ── HACKER TEXT SCRAMBLE on .scramble + .scramble-hover ────────────────
    const CHARS = '!<>-_\\/[]{}—=+*^?#0123456789ABCDEF';
    const scramble = (el, finalText, opts = {}) => {
        const speed = opts.speed || 30;
        const lockRate = opts.lockRate || 0.7;
        if (el.dataset.scrambling === '1') return;
        el.dataset.scrambling = '1';
        let iter = 0;
        const len = finalText.length;
        clearInterval(el._scrambleTimer);
        el._scrambleTimer = setInterval(() => {
            const out = finalText.split('').map((ch, i) => {
                if (i < iter) return finalText[i];
                if (ch === ' ') return ' ';
                return CHARS[Math.floor(Math.random() * CHARS.length)];
            }).join('');
            el.textContent = out;
            iter += lockRate;
            if (iter >= len) {
                clearInterval(el._scrambleTimer);
                el.textContent = finalText;
                el.dataset.scrambling = '0';
            }
        }, speed);
    };

    // On reveal-into-view: scramble section titles
    if ('IntersectionObserver' in window) {
        const scrambleObs = new IntersectionObserver((entries) => {
            entries.forEach(en => {
                if (en.isIntersecting) {
                    const el = en.target;
                    if (el._scrambled) return;
                    el._scrambled = true;
                    scramble(el, el.dataset.text || el.textContent);
                }
            });
        }, { threshold: 0.5 });
        $$('.scramble').forEach(el => scrambleObs.observe(el));
    }

    // Hover-triggered scramble on brutalist titles
    $$('.scramble-hover').forEach(el => {
        const originalText = el.dataset.text || el.textContent.trim().split(/\s/)[0];
        // Keep the pill children
        const pill = el.querySelector('.pill');
        el.addEventListener('mouseenter', () => {
            if (el._scrambling) return;
            el._scrambling = true;
            let iter = 0;
            const len = originalText.length;
            clearInterval(el._timer);
            el._timer = setInterval(() => {
                const out = originalText.split('').map((ch, i) => {
                    if (i < iter) return originalText[i];
                    if (ch === ' ') return ' ';
                    return CHARS[Math.floor(Math.random() * CHARS.length)];
                }).join('');
                el.firstChild.textContent = out + ' ';
                iter += 0.6;
                if (iter >= len) {
                    clearInterval(el._timer);
                    el.firstChild.textContent = originalText + ' ';
                    el._scrambling = false;
                }
            }, 28);
        });
    });


    // ── SLIDING PILL NAV INDICATOR (pattern navigation/001) ────────────────
    const navList = $('#navList');
    const navPill = $('#navPill');
    const navLinks = $$('.nav-link');

    const movePillTo = (link) => {
        if (!link) return;
        const navRect = navList.getBoundingClientRect();
        const linkRect = link.getBoundingClientRect();
        const top = linkRect.top - navRect.top;
        navPill.style.top = `${top}px`;
    };

    const setActive = (target) => {
        navLinks.forEach(l => l.classList.remove('active'));
        const link = navLinks.find(l => l.dataset.target === target);
        if (link) {
            link.classList.add('active');
            movePillTo(link);
        }
    };

    // Scroll-spy
    if ('IntersectionObserver' in window) {
        const spy = new IntersectionObserver((entries) => {
            entries.forEach(en => {
                if (en.isIntersecting) {
                    setActive(en.target.dataset.section);
                }
            });
        }, { rootMargin: '-40% 0px -55% 0px' });
        $$('section[data-section]').forEach(s => spy.observe(s));
    }

    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const id = link.dataset.target;
            const target = document.getElementById(id);
            if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            setActive(id);
        });
    });
    // Initial position
    requestAnimationFrame(() => movePillTo($('.nav-link.active')));
    window.addEventListener('resize', () => movePillTo($('.nav-link.active')));


    // ── COUNTER ANIMATION on data-count ────────────────────────────────────
    const animateCounter = (el) => {
        if (el._counted) return;
        el._counted = true;
        const target = parseFloat(el.dataset.count);
        const decimals = parseInt(el.dataset.decimals || '0', 10);
        const suffix = el.dataset.suffix || '';
        const duration = 1400;
        const start = performance.now();
        const tick = (now) => {
            const t = Math.min((now - start) / duration, 1);
            const eased = 1 - Math.pow(1 - t, 3);
            const val = target * eased;
            el.textContent = decimals
                ? val.toFixed(decimals)
                : Math.round(val).toLocaleString();
            if (suffix === ',' && !decimals) {
                el.textContent = Math.round(val).toLocaleString();
            }
            if (t < 1) requestAnimationFrame(tick);
            else el.textContent = decimals
                ? target.toFixed(decimals)
                : Math.round(target).toLocaleString();
        };
        requestAnimationFrame(tick);
    };

    const revealCounter = (el) => {
        if (el.dataset.text) {
            el.textContent = el.dataset.text;
            return;
        }
        animateCounter(el);
    };

    if ('IntersectionObserver' in window) {
        const cObs = new IntersectionObserver((entries) => {
            entries.forEach(en => {
                if (en.isIntersecting) {
                    revealCounter(en.target);
                    cObs.unobserve(en.target);
                }
            });
        }, { threshold: 0.4 });
        $$('[data-count], [data-text]').forEach(el => {
            if (el.classList.contains('num')) cObs.observe(el);
        });
    }


    // ── MOUSE PARALLAX on hero ─────────────────────────────────────────────
    if (!isCoarse && !isReducedMotion) {
        const tiltEl = $('[data-tilt]');
        if (tiltEl) {
            const heroSection = $('.hero');
            heroSection.addEventListener('mousemove', (e) => {
                const rect = heroSection.getBoundingClientRect();
                const x = (e.clientX - rect.left) / rect.width - 0.5;
                const y = (e.clientY - rect.top) / rect.height - 0.5;
                tiltEl.style.transform = `perspective(1200px) rotateX(${-y * 2}deg) rotateY(${x * 3}deg) translateZ(0)`;
            });
            heroSection.addEventListener('mouseleave', () => {
                tiltEl.style.transform = '';
            });
        }
    }


    // ── CHART: HISTOGRAM (target distribution) ─────────────────────────────
    const renderHistogram = (frame) => {
        const hist = data.histogram;
        const w = frame.clientWidth - 40;
        const h = 320;
        const pad = { l: 50, r: 20, t: 20, b: 50 };
        const iw = w - pad.l - pad.r;
        const ih = h - pad.t - pad.b;
        const maxCount = Math.max(...hist.map(b => b.count));
        const logMin = Math.log10(Math.max(hist[0].bin_lo, 1));
        const logMax = Math.log10(hist[hist.length - 1].bin_hi);

        const xScale = (v) => pad.l + ((Math.log10(Math.max(v, 1)) - logMin) / (logMax - logMin)) * iw;
        // log-scale y
        const yScale = (v) => pad.t + ih - (Math.log10(v + 1) / Math.log10(maxCount + 1)) * ih;

        const median = data.summary.median_value;
        const topTier = 100_000_000;

        let svg = `<svg viewBox="0 0 ${w} ${h}" width="100%" height="${h}" style="overflow: visible;">`;
        // Grid
        svg += '<g class="svg-grid">';
        [10, 100, 1000, 10000].forEach(v => {
            const y = yScale(v);
            svg += `<line x1="${pad.l}" x2="${w - pad.r}" y1="${y}" y2="${y}" />`;
        });
        svg += '</g>';

        // Bars
        hist.forEach(b => {
            const x = xScale(b.bin_lo);
            const x2 = xScale(b.bin_hi);
            const y = yScale(b.count);
            const barH = (pad.t + ih) - y;
            svg += `<rect x="${x}" y="${y}" width="${Math.max(x2 - x - 1, 1)}" height="${barH}" class="svg-bar" />`;
        });

        // Markers
        const mx = xScale(median);
        svg += `<line x1="${mx}" x2="${mx}" y1="${pad.t}" y2="${pad.t + ih}" stroke="${getCSSVar('--c-amber')}" stroke-dasharray="4 3" stroke-width="1.5" />`;
        svg += `<text x="${mx + 6}" y="${pad.t + 16}" class="svg-label" fill="${getCSSVar('--c-amber')}">Médiane · ${fmtEur(median)}</text>`;

        const tx = xScale(topTier);
        svg += `<line x1="${tx}" x2="${tx}" y1="${pad.t}" y2="${pad.t + ih}" stroke="${getCSSVar('--c-red')}" stroke-dasharray="4 3" stroke-width="1.5" />`;
        svg += `<text x="${tx + 6}" y="${pad.t + 16}" class="svg-label" fill="${getCSSVar('--c-red')}">Top niveau · 100 M€+</text>`;

        // X-axis labels
        svg += `<g class="svg-axis">`;
        [1000, 10000, 100000, 1000000, 10000000, 100000000].forEach(v => {
            const x = xScale(v);
            svg += `<text x="${x}" y="${pad.t + ih + 22}" text-anchor="middle">${fmtEur(v)}</text>`;
            svg += `<line x1="${x}" x2="${x}" y1="${pad.t + ih}" y2="${pad.t + ih + 4}" stroke="${getCSSVar('--c-hairline')}" />`;
        });
        svg += `<text x="${w/2}" y="${h - 8}" text-anchor="middle" class="svg-label" fill="${getCSSVar('--c-muted')}">Valeur d'apogée (€) — échelle log</text>`;
        svg += `</g>`;

        svg += '</svg>';
        frame.innerHTML = svg;
    };

    // ── CHART: POSITION BOX PLOT ───────────────────────────────────────────
    const renderPositionBox = (frame) => {
        const positions = data.position_box;
        const w = frame.clientWidth - 40;
        const h = 320;
        const pad = { l: 80, r: 20, t: 20, b: 30 };
        const iw = w - pad.l - pad.r;
        const ih = h - pad.t - pad.b;

        const logMin = Math.log10(50000);
        const logMax = Math.log10(300_000_000);
        const xScale = (v) => pad.l + ((Math.log10(Math.max(v, 1)) - logMin) / (logMax - logMin)) * iw;

        const bandH = ih / positions.length;
        const colors = ['#0d7c47', '#16a34a', '#d97706', '#52525b', '#94a3b8'];

        let svg = `<svg viewBox="0 0 ${w} ${h}" width="100%" height="${h}" style="overflow: visible;">`;

        positions.forEach((p, i) => {
            const yMid = pad.t + bandH * (i + 0.5);
            const color = colors[i % colors.length];
            const x1 = xScale(p.min);
            const xq1 = xScale(p.q1);
            const xMed = xScale(p.median);
            const xq3 = xScale(p.q3);
            const x2 = xScale(p.max);

            // Whisker
            svg += `<line x1="${x1}" x2="${x2}" y1="${yMid}" y2="${yMid}" stroke="${color}" stroke-width="1.5" />`;
            // Box
            svg += `<rect x="${xq1}" y="${yMid - 14}" width="${xq3 - xq1}" height="28" fill="${color}" fill-opacity="0.35" stroke="${color}" stroke-width="1.5" rx="2" />`;
            // Median
            svg += `<line x1="${xMed}" x2="${xMed}" y1="${yMid - 14}" y2="${yMid + 14}" stroke="${color}" stroke-width="2.5" />`;
            // Label
            svg += `<text x="${pad.l - 10}" y="${yMid + 4}" text-anchor="end" class="svg-label" fill="${getCSSVar('--c-ink')}" font-weight="600">${p.position}</text>`;
            // Median value text
            svg += `<text x="${x2 + 6}" y="${yMid + 4}" class="svg-label" fill="${color}">${fmtEur(p.median)}</text>`;
        });

        // X-axis
        svg += `<g class="svg-axis">`;
        [100000, 1000000, 10000000, 100000000].forEach(v => {
            const x = xScale(v);
            svg += `<text x="${x}" y="${pad.t + ih + 18}" text-anchor="middle">${fmtEur(v)}</text>`;
        });
        svg += `</g>`;
        svg += '</svg>';
        frame.innerHTML = svg;
    };


    // ── COUNTRIES (top 25, horizontal bars w/ reveal animation) ────────────
    const renderCountries = () => {
        const wrap = $('#countriesBars');
        if (!wrap) return;
        const top25 = data.countries.slice(0, 25);
        const max = Math.max(...top25.map(c => c.median_peak));
        wrap.innerHTML = top25.map(c => {
            const scale = c.median_peak / max;
            return `<div class="country-row" data-scale="${scale.toFixed(3)}">
                <div class="name">${c.country} <span style="color:var(--c-muted-2); font-family: var(--f-mono); font-size: 0.75rem; margin-left: 4px;">${c.n}</span></div>
                <div class="bar-wrap"><div class="bar" style="--scale: ${scale.toFixed(3)};"></div></div>
                <div class="val">${fmtEur(c.median_peak)}</div>
            </div>`;
        }).join('');

        if ('IntersectionObserver' in window) {
            const rowObs = new IntersectionObserver((entries) => {
                entries.forEach(en => {
                    if (en.isIntersecting) {
                        en.target.classList.add('reveal-in');
                        rowObs.unobserve(en.target);
                    }
                });
            }, { threshold: 0.2 });
            $$('.country-row').forEach((r, i) => {
                r.style.transitionDelay = `${i * 30}ms`;
                rowObs.observe(r);
            });
        }
    };

    // ── TOP 5 LEGEND with pin-pulse ────────────────────────────────────────
    const renderTop5 = () => {
        const wrap = $('#top5Legend');
        if (!wrap) return;
        wrap.innerHTML = data.top5_countries.map((c, i) => `
            <div class="legend-item" style="animation-delay: ${i * 0.15}s;">
                <div class="legend-pin" style="background: ${['#0d7c47', '#16a34a', '#d97706', '#52525b', '#06b6d4'][i]};"></div>
                <div><span class="country">${c.country}</span><span class="val">${fmtEur(c.median_peak)}</span></div>
            </div>
        `).join('');
    };

    // ── MODELS TABLE ───────────────────────────────────────────────────────
    const renderModelsTable = (frame) => {
        const rows = data.metrics.map((m, i) => {
            const winner = m.r2 === Math.max(...data.metrics.map(x => x.r2));
            return `<tr class="${winner ? 'winner-row' : ''}">
                <td><strong>${m.name}</strong></td>
                <td class="num-cell ${winner ? 'winner' : ''}">${fmtEur(m.mae)}</td>
                <td class="num-cell ${winner ? 'winner' : ''}">${fmtEur(m.rmse)}</td>
                <td class="num-cell ${winner ? 'winner' : ''}">${m.r2.toFixed(3)}</td>
            </tr>`;
        }).join('');
        frame.innerHTML = `<table class="data-table"><thead><tr>
            <th>Modèle</th><th>MAE</th><th>RMSE</th><th>R²</th>
        </tr></thead><tbody>${rows}</tbody></table>`;
    };

    // ── BASELINES TABLE ────────────────────────────────────────────────────
    const renderBaselinesTable = (frame) => {
        const rows = data.validation.baselines.map(b => `<tr>
            <td><strong>${b.name}</strong></td>
            <td class="num-cell">${b.r2.toFixed(3)}</td>
            <td class="num-cell">${fmtEur(b.mae)}</td>
        </tr>`).join('');
        frame.innerHTML = `<table class="data-table"><thead><tr>
            <th>Baseline</th><th>R²</th><th>MAE</th>
        </tr></thead><tbody>${rows}</tbody></table>`;
    };

    // ── CV BARS ────────────────────────────────────────────────────────────
    const renderCvBars = (frame) => {
        const cv = [...data.validation.cv].sort((a, b) => a.mean - b.mean);
        const w = frame.clientWidth - 40;
        const h = 220;
        const pad = { l: 160, r: 80, t: 20, b: 30 };
        const iw = w - pad.l - pad.r;
        const ih = h - pad.t - pad.b;
        const max = 0.9;
        const xScale = (v) => pad.l + (v / max) * iw;
        const bandH = ih / cv.length;
        const colors = ['#94a3b8', '#d4a373', '#d97706', '#0d7c47'];

        let svg = `<svg viewBox="0 0 ${w} ${h}" width="100%" height="${h}">`;
        // gridlines
        svg += '<g class="svg-grid">';
        [0.2, 0.4, 0.6, 0.8].forEach(v => {
            const x = xScale(v);
            svg += `<line x1="${x}" x2="${x}" y1="${pad.t}" y2="${pad.t + ih}" />`;
            svg += `<text x="${x}" y="${pad.t + ih + 18}" text-anchor="middle" class="svg-label" fill="${getCSSVar('--c-muted')}">${v.toFixed(1)}</text>`;
        });
        svg += '</g>';

        cv.forEach((m, i) => {
            const yMid = pad.t + bandH * (i + 0.5);
            const xVal = xScale(m.mean);
            const xMin = xScale(m.mean - m.std);
            const xMax = xScale(m.mean + m.std);
            const color = colors[i];
            svg += `<rect x="${pad.l}" y="${yMid - 10}" width="${xVal - pad.l}" height="20" fill="${color}" rx="3" />`;
            // Error bar
            svg += `<line x1="${xMin}" x2="${xMax}" y1="${yMid}" y2="${yMid}" stroke="${getCSSVar('--c-ink')}" stroke-width="1.5" />`;
            svg += `<line x1="${xMin}" x2="${xMin}" y1="${yMid - 6}" y2="${yMid + 6}" stroke="${getCSSVar('--c-ink')}" stroke-width="1.5" />`;
            svg += `<line x1="${xMax}" x2="${xMax}" y1="${yMid - 6}" y2="${yMid + 6}" stroke="${getCSSVar('--c-ink')}" stroke-width="1.5" />`;
            // Label
            svg += `<text x="${pad.l - 12}" y="${yMid + 4}" text-anchor="end" class="svg-label" fill="${getCSSVar('--c-ink')}" font-weight="600">${m.model}</text>`;
            svg += `<text x="${xMax + 8}" y="${yMid + 4}" class="svg-label" font-family="var(--f-mono)">${m.mean.toFixed(3)} ± ${m.std.toFixed(3)}</text>`;
        });
        svg += '</svg>';
        frame.innerHTML = svg;
    };

    // ── FEATURE IMPORTANCE — top 15 only (34 features is noisy) ───────────
    const renderFeatureImportance = (frame) => {
        const sortedDesc = [...data.feature_importance].sort((a, b) => b.importance - a.importance);
        const tailCount = sortedDesc.filter(f => f.importance < 0.5).length;
        const feats = sortedDesc.slice(0, 15).reverse();  // top 15, then reverse for bottom-up
        const w = frame.clientWidth - 40;
        const h = 480;
        const pad = { l: 220, r: 80, t: 24, b: 36 };
        const iw = w - pad.l - pad.r;
        const ih = h - pad.t - pad.b;
        const max = Math.max(...feats.map(f => f.importance));
        const bandH = ih / feats.length;

        let svg = `<svg viewBox="0 0 ${w} ${h}" width="100%" height="${h}">`;
        feats.forEach((f, i) => {
            const yMid = pad.t + bandH * (i + 0.5);
            const barW = (f.importance / max) * iw;
            const barH = Math.min(bandH * 0.55, 22);
            svg += `<rect x="${pad.l}" y="${yMid - barH/2}" width="${barW}" height="${barH}" fill="${getCSSVar('--c-pitch')}" rx="3" />`;
            svg += `<text x="${pad.l - 10}" y="${yMid + 4}" text-anchor="end" class="svg-label" fill="${getCSSVar('--c-ink')}" font-size="12" font-weight="600">${f.feature}</text>`;
            svg += `<text x="${pad.l + barW + 8}" y="${yMid + 4}" class="svg-label" font-family="var(--f-mono)" font-size="11" fill="${getCSSVar('--c-muted')}">${f.importance.toFixed(2)}%</text>`;
        });
        svg += `<text x="${w/2}" y="${h - 10}" text-anchor="middle" class="svg-label" fill="${getCSSVar('--c-muted')}" font-size="11">Importance CatBoost (top 15 sur ${sortedDesc.length} ; ${tailCount} variables contribuent à moins de 0,5%)</text>`;
        svg += '</svg>';
        frame.innerHTML = svg;
    };

    // ── RESIDUALS SCATTER ──────────────────────────────────────────────────
    const renderResiduals = (frame) => {
        const pts = data.residuals_sample;
        const w = frame.clientWidth - 40;
        const h = 440;
        const pad = { l: 70, r: 30, t: 20, b: 50 };
        const iw = w - pad.l - pad.r;
        const ih = h - pad.t - pad.b;
        const logMin = Math.log10(50000);
        const logMax = Math.log10(300_000_000);
        const xScale = (v) => pad.l + ((Math.log10(Math.max(v, 1)) - logMin) / (logMax - logMin)) * iw;
        const yScale = (v) => pad.t + ih - ((Math.log10(Math.max(v, 1)) - logMin) / (logMax - logMin)) * ih;

        const colorMap = {
            'Attack': '#0d7c47', 'Midfield': '#16a34a', 'Defender': '#d97706',
            'Goalkeeper': '#52525b', 'Missing': '#94a3b8',
        };

        let svg = `<svg viewBox="0 0 ${w} ${h}" width="100%" height="${h}" style="overflow: visible;">`;
        // diagonal
        const x1 = xScale(50000), y1 = yScale(50000);
        const x2 = xScale(300_000_000), y2 = yScale(300_000_000);
        svg += `<line x1="${x1}" x2="${x2}" y1="${y1}" y2="${y2}" stroke="${getCSSVar('--c-ink')}" stroke-dasharray="4 3" stroke-width="1.5" />`;

        // points
        pts.forEach(p => {
            const x = xScale(p.actual);
            const y = yScale(p.predicted);
            const color = colorMap[p.position] || '#94a3b8';
            svg += `<circle cx="${x}" cy="${y}" r="2.5" fill="${color}" fill-opacity="0.55" />`;
        });

        // axes
        svg += `<g class="svg-axis">`;
        [100000, 1000000, 10000000, 100000000].forEach(v => {
            const x = xScale(v); const y = yScale(v);
            svg += `<text x="${x}" y="${pad.t + ih + 18}" text-anchor="middle" class="svg-label" fill="${getCSSVar('--c-muted')}">${fmtEur(v)}</text>`;
            svg += `<text x="${pad.l - 8}" y="${y + 4}" text-anchor="end" class="svg-label" fill="${getCSSVar('--c-muted')}">${fmtEur(v)}</text>`;
        });
        svg += `<text x="${w/2}" y="${h - 10}" text-anchor="middle" class="svg-label" fill="${getCSSVar('--c-muted')}">Valeur d'apogée réelle (€)</text>`;
        svg += `<text x="20" y="${h/2}" text-anchor="middle" class="svg-label" fill="${getCSSVar('--c-muted')}" transform="rotate(-90 20 ${h/2})">Prédite (€)</text>`;
        svg += `</g>`;

        // legend
        const positions = Object.keys(colorMap);
        positions.forEach((p, i) => {
            svg += `<circle cx="${pad.l + i * 110 + 4}" cy="${pad.t - 4}" r="4" fill="${colorMap[p]}" />`;
            svg += `<text x="${pad.l + i * 110 + 14}" y="${pad.t}" class="svg-label" fill="${getCSSVar('--c-ink')}">${p}</text>`;
        });

        svg += '</svg>';
        frame.innerHTML = svg;
    };

    // ── PER POSITION ───────────────────────────────────────────────────────
    const renderPerPosition = (frame) => {
        const rows = [...data.per_position].sort((a, b) => a.mae_pct - b.mae_pct);
        const w = frame.clientWidth - 40;
        const h = 280;
        const pad = { l: 130, r: 80, t: 20, b: 40 };
        const iw = w - pad.l - pad.r;
        const ih = h - pad.t - pad.b;
        const max = Math.max(...rows.map(r => r.mae_pct)) * 1.15;
        const bandH = ih / rows.length;

        let svg = `<svg viewBox="0 0 ${w} ${h}" width="100%" height="${h}">`;
        rows.forEach((r, i) => {
            const yMid = pad.t + bandH * (i + 0.5);
            const barW = (r.mae_pct / max) * iw;
            svg += `<rect x="${pad.l}" y="${yMid - 12}" width="${barW}" height="24" fill="${getCSSVar('--c-pitch')}" rx="3" />`;
            svg += `<text x="${pad.l - 8}" y="${yMid + 4}" text-anchor="end" class="svg-label" font-weight="600" fill="${getCSSVar('--c-ink')}">${r.position}</text>`;
            svg += `<text x="${pad.l + barW + 8}" y="${yMid + 4}" class="svg-label" font-family="var(--f-mono)">${r.mae_pct.toFixed(0)}% · MAE ${fmtEur(r.mae)} (n=${r.n})</text>`;
        });
        svg += `<text x="${w/2}" y="${h - 10}" text-anchor="middle" class="svg-label" fill="${getCSSVar('--c-muted')}">MAE as % of median actual value</text>`;
        svg += '</svg>';
        frame.innerHTML = svg;
    };

    // ── PREDICTION-BUCKET DIAGNOSTIC — the user-felt error metric ─────────
    const renderBucketDiagnostic = (frame) => {
        const buckets = data.bucket_diagnostic || [];
        if (!buckets.length) {
            frame.innerHTML = '<p style="color: var(--c-muted); padding: var(--s-3);">Diagnostic par tranche indisponible.</p>';
            return;
        }
        const bucketLabel = (b) => b
            .replace('Micro (<€200k)', 'Micro (<200 k€)')
            .replace('Low (€200k–1M)', 'Bas (200 k€–1 M€)')
            .replace('Mid (€1–5M)', 'Moyen (1–5 M€)')
            .replace('High (€5–20M)', 'Haut (5–20 M€)')
            .replace('Elite (>€20M)', 'Élite (>20 M€)');
        const rows = buckets.map(b => {
            // Colour-code: green for low error, amber mid, red high
            const errClass = b.median_pct_error < 40 ? 'pitch'
                : (b.median_pct_error < 60 ? 'amber' : 'red');
            const errBg = { pitch: 'rgba(13,124,71,0.12)', amber: 'rgba(217,119,6,0.12)', red: 'rgba(220,38,38,0.1)' }[errClass];
            const errFg = { pitch: 'var(--c-pitch)', amber: 'var(--c-amber)', red: 'var(--c-red)' }[errClass];
            return `<tr>
                <td><strong>${bucketLabel(b.bucket)}</strong></td>
                <td class="num-cell">${b.n.toLocaleString()}</td>
                <td class="num-cell"><span style="background:${errBg}; color:${errFg}; padding:3px 9px; border-radius:var(--r-full); font-weight:700;">${b.median_pct_error.toFixed(0)}%</span></td>
                <td class="num-cell">${b.over_predicted_rate.toFixed(0)}%</td>
                <td class="num-cell">${fmtEur(b.median_actual)}</td>
                <td class="num-cell">${fmtEur(b.median_predicted)}</td>
            </tr>`;
        }).join('');
        frame.innerHTML = `<table class="data-table"><thead><tr>
            <th>Tranche</th>
            <th>N (test)</th>
            <th>Erreur médiane %</th>
            <th>Sur-prédit</th>
            <th>Médiane réelle</th>
            <th>Médiane prédite</th>
        </tr></thead><tbody>${rows}</tbody></table>`;
    };

    // ── ABLATION ───────────────────────────────────────────────────────────
    const renderAblation = (frame) => {
        const full = data.ablation.find(a => a.variant === 'xgboost_full');
        const abl = data.ablation.find(a => a.variant === 'xgboost_without_current_club_competition');
        const delta = data.ablation.find(a => a.variant === 'delta');
        frame.innerHTML = `
            <div class="card-row" style="grid-template-columns: 1fr 1fr 1fr; display: grid; gap: var(--s-4); margin: 0;">
                <div class="ablation-card" style="background: var(--c-surface-2); padding: var(--s-5); border-radius: var(--r-md); border-left: 3px solid var(--c-pitch);">
                    <div style="font-size: 0.85rem; color: var(--c-muted); margin-bottom: var(--s-2);">Full model (with leak feature)</div>
                    <div style="font-family: var(--f-display); font-size: 2.4rem; font-weight: 700; color: var(--c-pitch); letter-spacing: -0.02em;">R² ${full.r2.toFixed(3)}</div>
                    <div style="color: var(--c-muted); font-size: 0.88rem; margin-top: var(--s-2);">MAE ${fmtEur(full.mae)}</div>
                </div>
                <div class="ablation-card" style="background: var(--c-surface-2); padding: var(--s-5); border-radius: var(--r-md); border-left: 3px solid var(--c-amber);">
                    <div style="font-size: 0.85rem; color: var(--c-muted); margin-bottom: var(--s-2);">Without leak feature</div>
                    <div style="font-family: var(--f-display); font-size: 2.4rem; font-weight: 700; color: var(--c-amber); letter-spacing: -0.02em;">R² ${abl.r2.toFixed(3)}</div>
                    <div style="color: var(--c-muted); font-size: 0.88rem; margin-top: var(--s-2);">MAE ${fmtEur(abl.mae)}</div>
                </div>
                <div class="ablation-card" style="background: var(--c-surface-2); padding: var(--s-5); border-radius: var(--r-md); border-left: 3px solid var(--c-muted);">
                    <div style="font-size: 0.85rem; color: var(--c-muted); margin-bottom: var(--s-2);">Delta · Cost of removing it</div>
                    <div style="font-family: var(--f-display); font-size: 2.4rem; font-weight: 700; color: var(--c-ink); letter-spacing: -0.02em;">${delta.r2 >= 0 ? '+' : ''}${delta.r2.toFixed(3)}</div>
                    <div style="color: var(--c-muted); font-size: 0.88rem; margin-top: var(--s-2);">${delta.mae >= 0 ? '+' : ''}${fmtEur(delta.mae)}</div>
                </div>
            </div>
        `;
    };

    // ── Helper: CSS var ────────────────────────────────────────────────────
    const getCSSVar = (name) =>
        getComputedStyle(document.documentElement).getPropertyValue(name).trim() || '#000';


    // ── Wire all chart renderers ───────────────────────────────────────────
    const charts = {
        histogram: renderHistogram,
        positionBox: renderPositionBox,
        modelsTable: renderModelsTable,
        baselinesTable: renderBaselinesTable,
        cvBars: renderCvBars,
        featureImportance: renderFeatureImportance,
        residuals: renderResiduals,
        perPosition: renderPerPosition,
        ablation: renderAblation,
        bucketDiagnostic: renderBucketDiagnostic,
    };

    const renderCharts = () => {
        $$('[data-chart]').forEach(frame => {
            const fn = charts[frame.dataset.chart];
            if (fn) fn(frame);
        });
        renderCountries();
        renderTop5();
    };
    renderCharts();
    // Re-render on theme change for color updates
    themeToggle.addEventListener('click', () => setTimeout(renderCharts, 50));
    // Resize debounce
    let resizeTimer;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(renderCharts, 200);
    });


    // ── PLAYER DEMO ────────────────────────────────────────────────────────
    const playerSelect = $('#playerSelect');
    const onlyTestToggle = $('#onlyTest');
    const randomBtn = $('#randomPick');
    const playerCard = $('#playerCard');
    const predictionRow = $('#predictionRow');
    const modelCompareFrame = $('#modelComparison');

    // Search widget
    const searchInput = $('#playerSearch');
    const searchTyped = $('.search-ghost-layer .typed');
    const searchGhost = $('.search-ghost-layer .ghost');
    const searchResults = $('#searchResults');

    // ── Top opportunities — high-ROI scout targets ──────────────────────────
    const renderOpportunities = () => {
        const wrap = $('#opportunitiesRow');
        if (!wrap) return;
        const opps = data.top_opportunities || [];
        const byId = Object.fromEntries(data.players.map(p => [p.id, p]));
        wrap.innerHTML = opps.map((o, i) => {
            const p = byId[o.id];
            if (!p) return '';
            const raw = p.predictions.catboost_regressor;
            const photo = p.image_url
                ? `<img class="opp-photo" src="${p.image_url}" alt="${p.name}" loading="lazy" onerror="this.style.display='none'"/>`
                : `<div class="opp-photo" style="display:flex;align-items:center;justify-content:center;color:var(--c-muted-2);">👤</div>`;
            return `
                <div class="opp-card" data-id="${p.id}" tabindex="0" role="button" aria-label="Analyser ${p.name}">
                    <div class="opp-rank">#${String(i+1).padStart(2,'0')} · ROI ABSOLU</div>
                    <div class="opp-head">
                        ${photo}
                        <div>
                            <div class="opp-name">${p.name}</div>
                            <div class="opp-meta">${p.age} ans · ${p.position}</div>
                        </div>
                    </div>
                    <div class="opp-money">
                        <span class="from">${fmtEur(p.current_value)}</span>
                        <span class="arrow">→</span>
                        <span class="to">${fmtEur(raw)}</span>
                    </div>
                    <div class="opp-upside">
                        <span class="eur">+${fmtEur(o.upside_eur)}</span>
                        <span class="pct">+${o.upside_pct.toFixed(0)}%</span>
                    </div>
                </div>
            `;
        }).join('');
        // Click handler — load the player into the demo
        wrap.querySelectorAll('.opp-card').forEach(el => {
            const handler = () => {
                const id = parseInt(el.dataset.id, 10);
                // Make sure the pool includes this player (toggle off "test only" if needed)
                const p = data.players.find(pp => pp.id === id);
                if (p && p.split === 'TRAIN' && onlyTestToggle.checked) {
                    onlyTestToggle.checked = false;
                    refreshPool();
                }
                acceptPlayer(id);
                // Scroll the player card into view
                $('#playerCard')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
            };
            el.addEventListener('click', handler);
            el.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); handler(); }
            });
        });
    };

    let currentPlayer = null;
    let currentPool = [];
    let activeResultIdx = -1;

    const refreshPool = () => {
        const onlyTest = onlyTestToggle.checked;
        currentPool = onlyTest ? data.players.filter(p => p.split === 'TEST') : data.players;
        // Keep the hidden <select> in sync — preserves selectPlayer() lookup
        playerSelect.innerHTML = currentPool.map(p => `<option value="${p.id}">${p.name}</option>`).join('');
    };

    const populatePlayerSelect = () => {
        refreshPool();
        // Default to a famous TEST player if possible
        const famous = currentPool.find(p => /Mbapp|Haaland|Bellingham|Vinic/i.test(p.name));
        const start = famous || currentPool[0];
        if (start) {
            playerSelect.value = String(start.id);
            searchInput.value = start.name;
            updateGhost();
            selectPlayer(start.id);
        }
    };

    // ── Ghost typeahead ───────────────────────────────────────────────────
    const updateGhost = () => {
        const q = searchInput.value;
        if (!q) { searchTyped.textContent = ''; searchGhost.textContent = ''; return; }
        const ql = q.toLowerCase();
        // Prefer prefix matches, fall back to substring
        const prefix = currentPool.find(p => p.name.toLowerCase().startsWith(ql));
        const sub = !prefix ? currentPool.find(p => p.name.toLowerCase().includes(ql)) : null;
        const match = prefix || sub;
        if (match && prefix) {
            // Prefix match — show typed portion (invisible) + remainder (light)
            searchTyped.textContent = q;
            searchGhost.textContent = match.name.substring(q.length);
        } else {
            // Substring or no match — clear ghost
            searchTyped.textContent = q;
            searchGhost.textContent = '';
        }
    };

    const renderSearchResults = () => {
        const q = searchInput.value.trim();
        if (!q) { searchResults.hidden = true; return; }
        const ql = q.toLowerCase();
        const matches = currentPool
            .filter(p => p.name.toLowerCase().includes(ql))
            .slice(0, 8);
        if (!matches.length) { searchResults.hidden = true; return; }
        const escapeHtml = (s) => s.replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
        const highlight = (name) => {
            const idx = name.toLowerCase().indexOf(ql);
            if (idx < 0) return escapeHtml(name);
            return escapeHtml(name.substring(0, idx)) + '<mark>' + escapeHtml(name.substring(idx, idx + q.length)) + '</mark>' + escapeHtml(name.substring(idx + q.length));
        };
        searchResults.innerHTML = matches.map((p, i) => `
            <div class="result ${i === 0 ? 'active' : ''}" data-id="${p.id}" role="option" tabindex="-1">
                <span class="nm">${highlight(p.name)}</span>
                <span class="meta">${p.position} · ${fmtEur(p.actual)} <span class="split ${p.split}">${p.split}</span></span>
            </div>
        `).join('');
        searchResults.hidden = false;
        activeResultIdx = 0;
        // Click handlers
        searchResults.querySelectorAll('.result').forEach((el, i) => {
            el.addEventListener('mousedown', (e) => {
                e.preventDefault();
                const id = parseInt(el.dataset.id, 10);
                acceptPlayer(id);
            });
        });
    };

    const acceptPlayer = (id) => {
        const p = data.players.find(pp => pp.id === id);
        if (!p) return;
        searchInput.value = p.name;
        searchTyped.textContent = '';
        searchGhost.textContent = '';
        searchResults.hidden = true;
        playerSelect.value = String(id);
        selectPlayer(id);
    };

    const acceptGhost = () => {
        const ghost = searchGhost.textContent;
        if (ghost) {
            const fullName = searchTyped.textContent + ghost;
            const match = currentPool.find(p => p.name === fullName);
            if (match) { acceptPlayer(match.id); return true; }
        }
        // Otherwise accept the first result if visible
        const active = searchResults.querySelector('.result.active');
        if (active && !searchResults.hidden) {
            acceptPlayer(parseInt(active.dataset.id, 10));
            return true;
        }
        return false;
    };

    searchInput.addEventListener('input', () => {
        updateGhost();
        renderSearchResults();
    });
    searchInput.addEventListener('keydown', (e) => {
        if (e.key === 'Tab' || e.key === 'Enter') {
            if (acceptGhost()) e.preventDefault();
        } else if (e.key === 'Escape') {
            searchResults.hidden = true;
            searchTyped.textContent = ''; searchGhost.textContent = '';
        } else if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
            const items = searchResults.querySelectorAll('.result');
            if (!items.length) return;
            e.preventDefault();
            items[activeResultIdx]?.classList.remove('active');
            activeResultIdx = (activeResultIdx + (e.key === 'ArrowDown' ? 1 : -1) + items.length) % items.length;
            items[activeResultIdx].classList.add('active');
            items[activeResultIdx].scrollIntoView({ block: 'nearest' });
        }
    });
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.player-search-wrap')) searchResults.hidden = true;
    });

    const selectPlayer = (id) => {
        const p = data.players.find(pl => pl.id === id);
        if (!p) return;
        currentPlayer = p;
        renderPlayerCard(p);
        renderPredictionRow(p);
        renderModelComparison(p);
    };

    const renderPlayerCard = (p) => {
        const isTrain = p.split === 'TRAIN';
        const photoHtml = p.image_url
            ? `<img class="player-photo" src="${p.image_url}" alt="${p.name}" loading="lazy" onerror="this.classList.add('no-image'); this.src=''; this.alt='👤'; this.replaceWith(Object.assign(document.createElement('div'),{className:'player-photo no-image',textContent:'👤'}))" />`
            : `<div class="player-photo no-image">👤</div>`;
        const highlightsHtml = (p.highlights || []).map(h => `
            <div class="highlight-chip ${h.tier || ''}">
                <span class="lbl">${h.label}</span>
                <span class="val">${h.value_fmt}</span>
                ${h.badge ? `<span class="badge">${h.badge}</span>` : ''}
            </div>
        `).join('');
        const currentValHtml = (p.current_value != null)
            ? `<div class="sub-val">Actuelle : <span class="num">${fmtEur(p.current_value)}</span></div>`
            : '';
        const footFr = { left: 'gauche', right: 'droit', both: 'des deux pieds' }[p.foot] || p.foot;
        playerCard.innerHTML = `
            ${photoHtml}
            <div class="player-card-body">
                <span class="pill ${isTrain ? 'red' : 'gray'}" style="margin-bottom: var(--s-2); display: inline-block;">${isTrain ? '⚠ TRAIN — le modèle a déjà vu ce joueur' : '✓ TEST — jamais vu auparavant'}</span>
                <div class="player-card-name">${p.name}</div>
                <div class="player-meta">
                    <span class="pill gray">${p.position}</span>
                    <span class="pill gray">Pied ${footFr}</span>
                    <span class="pill gray">${p.age || '?'} ans</span>
                    <span class="pill gray">${p.country}</span>
                </div>
                <div class="highlights-row">${highlightsHtml}</div>
            </div>
            <div class="player-value-right">
                <div class="lbl">Valeur la plus haute enregistrée</div>
                <div class="val">${fmtEur(p.actual)}</div>
                ${currentValHtml}
            </div>
        `;
    };

    const renderPredictionRow = (p) => {
        // Use CatBoost (winner) as the headline prediction; fall back to XGB if missing
        const rawPred = (p.predictions.catboost_regressor !== undefined)
            ? p.predictions.catboost_regressor
            : p.predictions.xgboost_regressor;
        const winnerName = (p.predictions.catboost_regressor !== undefined) ? 'CatBoost' : 'XGBoost';

        // Time-direction logic: the model is retrospective. If the player has declined
        // from a past peak, the model is describing that PAST peak — not a future one.
        // Discriminator: current value vs. highest recorded.
        const cur = p.current_value;
        const peak = p.highest_recorded;
        const declined = (cur != null && peak != null && cur < peak * 0.85);

        let signal, signalColor, signalLabel, headlineLbl, headlineVal, headlineSub;
        if (cur == null) {
            // No current value — fall back to model prediction floored at zero
            signal = '—'; signalColor = 'var(--c-muted)';
            signalLabel = 'Aucune valeur actuelle disponible';
            headlineLbl = `${winnerName} · pic estimé`;
            headlineVal = fmtEur(Math.max(rawPred, 0));
            headlineSub = 'Pas de valeur actuelle pour comparer';
        } else if (declined) {
            // Already peaked. Model is describing the historical peak, not a future one.
            const peakAge = '—'; // We don't have age-at-peak in data
            signal = '🕘'; signalColor = 'var(--c-muted)';
            signalLabel = `Pic historique · joueur en déclin (−${((1 - cur/peak)*100).toFixed(0)}% depuis le sommet)`;
            headlineLbl = `${winnerName} · pic historique reconnu`;
            headlineVal = fmtEur(peak);
            headlineSub = `Le modèle confirme le profil d'apogée de ${fmtEur(peak)} — déjà atteint, pas un potentiel à capturer.`;
        } else if (rawPred > cur * 1.15) {
            // Genuine upside — at peak now, model sees fundamentals supporting higher
            signal = '▲'; signalColor = 'var(--c-pitch)';
            signalLabel = `Potentiel · +${(((rawPred - cur) / cur) * 100).toFixed(0)}% de marge sur la valeur actuelle`;
            headlineLbl = `${winnerName} · pic attendu`;
            headlineVal = fmtEur(rawPred);
            headlineSub = 'Les fondamentaux soutiennent une valeur au-dessus du prix actuel';
        } else if (rawPred < cur * 0.85) {
            signal = '▼'; signalColor = 'var(--c-amber)';
            signalLabel = `Prime du marché · fondamentaux ${((1 - rawPred/cur) * 100).toFixed(0)}% en dessous`;
            headlineLbl = `${winnerName} · pic estimé (au-dessus du fondamental)`;
            headlineVal = fmtEur(cur);
            headlineSub = 'Le marché valorise au-dessus de ce que les fondamentaux justifient';
        } else {
            signal = '◆'; signalColor = 'var(--c-cyan)';
            signalLabel = 'Juste · la prédiction colle au prix actuel';
            headlineLbl = `${winnerName} · pic attendu`;
            headlineVal = fmtEur(Math.max(rawPred, cur));
            headlineSub = 'Le modèle valide la valorisation actuelle';
        }

        predictionRow.innerHTML = `
            <div class="bignum featured">
                <div class="lbl">${headlineLbl}</div>
                <div class="num counter-target">${headlineVal}</div>
                <div class="sub">${headlineSub}</div>
            </div>
            <div class="bignum compact" style="background: ${signalColor};">
                <div class="lbl">Signal</div>
                <div class="num" style="font-size: 1.4rem;">${signal}</div>
                <div class="sub">${signalLabel}</div>
            </div>
            <div class="bignum compact">
                <div class="lbl">Sortie brute du modèle</div>
                <div class="num" style="font-size: 1.8rem;">${fmtEur(rawPred)}</div>
                <div class="sub">${cur != null ? `vs. actuelle ${fmtEur(cur)} · pic histo. ${fmtEur(peak)}` : 'pas de valeur actuelle'}</div>
            </div>
        `;
        // Counter-up the main prediction — parse the rendered number back out of headlineVal
        const target = predictionRow.querySelector('.counter-target');
        if (target && !isReducedMotion) {
            const renderedText = headlineVal;
            // Parse "€1.53M" / "€500k" / "€800" back to a number for the count-up
            const parseEurNum = (s) => {
                const m = s.match(/€?([0-9.,]+)\s*(M|k)?/i);
                if (!m) return 0;
                const num = parseFloat(m[1].replace(',', '.'));
                const suf = (m[2] || '').toLowerCase();
                return suf === 'm' ? num * 1e6 : (suf === 'k' ? num * 1e3 : num);
            };
            const finalNum = parseEurNum(renderedText);
            target.textContent = '€0';
            const final = finalNum;
            const start = performance.now();
            const dur = 1200;
            const tick = (now) => {
                const t = Math.min((now - start) / dur, 1);
                const eased = 1 - Math.pow(1 - t, 3);
                target.textContent = fmtEur(final * eased);
                if (t < 1) requestAnimationFrame(tick);
                else target.textContent = fmtEur(final);
            };
            requestAnimationFrame(tick);
        }
    };

    const renderModelComparison = (p) => {
        const models = [
            { name: 'Decision Tree', val: p.predictions.decision_tree,      color: '#94a3b8' },
            { name: 'Random Forest', val: p.predictions.random_forest,      color: '#d4a373' },
            { name: 'XGBoost',       val: p.predictions.xgboost_regressor,  color: '#d97706' },
            { name: 'CatBoost',      val: p.predictions.catboost_regressor, color: '#0d7c47' },
        ].filter(m => m.val !== undefined);
        const w = modelCompareFrame.clientWidth - 40;
        const h = 360;
        const pad = { l: 60, r: 30, t: 40, b: 50 };
        const iw = w - pad.l - pad.r;
        const ih = h - pad.t - pad.b;
        const max = Math.max(...models.map(m => m.val), p.actual) * 1.15;
        const barW = iw / models.length * 0.6;
        const gap = iw / models.length * 0.4;

        let svg = `<svg viewBox="0 0 ${w} ${h}" width="100%" height="${h}" style="overflow: visible;">`;
        models.forEach((m, i) => {
            const cx = pad.l + (iw / models.length) * (i + 0.5);
            const barX = cx - barW / 2;
            const barH = (m.val / max) * ih;
            const barY = pad.t + ih - barH;
            svg += `<rect x="${barX}" y="${pad.t + ih}" width="${barW}" height="0" fill="${m.color}" rx="4">
                <animate attributeName="y" from="${pad.t + ih}" to="${barY}" dur="0.9s" fill="freeze" begin="${i * 0.15}s" calcMode="spline" keySplines="0.2 0.8 0.2 1" />
                <animate attributeName="height" from="0" to="${barH}" dur="0.9s" fill="freeze" begin="${i * 0.15}s" calcMode="spline" keySplines="0.2 0.8 0.2 1" />
            </rect>`;
            svg += `<text x="${cx}" y="${barY - 8}" text-anchor="middle" class="svg-label" font-family="var(--f-display)" font-weight="700" font-size="13" fill="${getCSSVar('--c-ink')}" opacity="0">
                ${fmtEur(m.val)}
                <animate attributeName="opacity" from="0" to="1" dur="0.4s" begin="${i * 0.15 + 0.9}s" fill="freeze" />
            </text>`;
            svg += `<text x="${cx}" y="${pad.t + ih + 22}" text-anchor="middle" class="svg-label" fill="${getCSSVar('--c-muted')}">${m.name}</text>`;
        });
        // Real-value reference line
        const refY = pad.t + ih - (p.actual / max) * ih;
        svg += `<line x1="${pad.l}" x2="${w - pad.r}" y1="${refY}" y2="${refY}" stroke="${getCSSVar('--c-red')}" stroke-dasharray="4 3" stroke-width="1.8" opacity="0">
            <animate attributeName="opacity" from="0" to="1" dur="0.6s" begin="1.4s" fill="freeze" />
        </line>`;
        svg += `<text x="${w - pad.r - 6}" y="${refY - 6}" text-anchor="end" font-family="var(--f-mono)" font-size="11" fill="${getCSSVar('--c-red')}" opacity="0">
            Pic réel : ${fmtEur(p.actual)}
            <animate attributeName="opacity" from="0" to="1" dur="0.6s" begin="1.4s" fill="freeze" />
        </text>`;
        svg += '</svg>';
        modelCompareFrame.innerHTML = svg;
    };

    onlyTestToggle.addEventListener('change', () => {
        refreshPool();
        // If the currently-selected player isn't in the new pool, pick a famous one
        const cur = data.players.find(p => p.id === currentPlayer?.id);
        if (!cur || !currentPool.find(p => p.id === cur.id)) {
            const famous = currentPool.find(p => /Mbapp|Haaland|Bellingham|Vinic/i.test(p.name)) || currentPool[0];
            if (famous) acceptPlayer(famous.id);
        }
    });
    randomBtn.addEventListener('click', () => {
        refreshPool();
        const r = currentPool[Math.floor(Math.random() * currentPool.length)];
        if (r) acceptPlayer(r.id);
    });

    populatePlayerSelect();
    renderOpportunities();


    // ── 3D CAROUSEL — JS-driven with drag inertia (replaces CSS keyframe) ──
    const scene = $('.scene-3d');
    const a3d = $('#modelCarousel');
    if (scene && a3d && !isReducedMotion) {
        let rotation = 0;          // current rotation in degrees
        let velocity = 0;          // angular velocity (deg/frame), drives inertia
        let dragging = false;
        let isHovered = false;
        let lastX = 0;
        const autoSpeed = -0.18;   // gentle auto-rotation when idle (deg/frame ≈ 11°/s, reversed)
        const friction = 0.93;     // velocity decay per frame (~0.5s to stop)

        const applyRotation = () => {
            a3d.style.transform = `translate(-50%, -50%) rotateY(${rotation}deg)`;
        };

        const tick = () => {
            if (!dragging) {
                // Apply momentum decay
                if (Math.abs(velocity) > 0.01) {
                    rotation += velocity;
                    velocity *= friction;
                } else {
                    velocity = 0;
                    // Auto-rotate when not interacting
                    if (!isHovered) rotation += autoSpeed;
                }
                applyRotation();
            }
            requestAnimationFrame(tick);
        };

        scene.addEventListener('mouseenter', () => { isHovered = true; });
        scene.addEventListener('mouseleave', () => { isHovered = false; });

        // Mouse drag
        const onPointerDown = (clientX) => {
            dragging = true;
            lastX = clientX;
            velocity = 0;
            scene.classList.add('dragging');
        };
        const onPointerMove = (clientX) => {
            if (!dragging) return;
            const dx = clientX - lastX;
            const delta = dx * 0.4;
            rotation += delta;
            velocity = delta; // remember last frame's delta as initial momentum
            lastX = clientX;
            applyRotation();
        };
        const onPointerUp = () => {
            if (!dragging) return;
            dragging = false;
            scene.classList.remove('dragging');
            // velocity already captured from last move event — momentum kicks in via tick()
        };

        scene.addEventListener('mousedown', (e) => { e.preventDefault(); onPointerDown(e.clientX); });
        window.addEventListener('mousemove', (e) => onPointerMove(e.clientX));
        window.addEventListener('mouseup', onPointerUp);

        // Touch
        scene.addEventListener('touchstart', (e) => onPointerDown(e.touches[0].clientX), { passive: true });
        window.addEventListener('touchmove', (e) => onPointerMove(e.touches[0].clientX), { passive: true });
        window.addEventListener('touchend', onPointerUp);

        // Keyboard accessibility — left/right arrows nudge by 30°
        scene.setAttribute('tabindex', '0');
        scene.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowLeft')  { velocity = -3; }
            if (e.key === 'ArrowRight') { velocity = 3; }
        });

        applyRotation();
        requestAnimationFrame(tick);
    }

    console.log('✅ Peak Value presentation initialised — 8 patterns active');
})();

"use strict";
(() => {
  const D = {};
  const $ = (sel, el = document) => el.querySelector(sel);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const isNum = (x) => typeof x === "number" && Number.isFinite(x);
  const pct = (x, d = 1) => (isNum(x) ? `${(x * 100).toFixed(d)}%` : "–");
  const spct = (x, d = 1) => (isNum(x) ? `${x > 0 ? "+" : ""}${(x * 100).toFixed(d)}%` : "–");
  const num = (x, d = 3) => (isNum(x) ? x.toFixed(d) : "–");
  const snum = (x, d = 3) => (isNum(x) ? `${x > 0 ? "+" : ""}${x.toFixed(d)}` : "–");
  const cls = (x) => (isNum(x) ? (x > 0 ? "pos" : x < 0 ? "neg" : "") : "");
  const money = (x) => (isNum(x) ? x.toLocaleString("en-US", { style: "currency", currency: "USD" }) : "–");
  const dateFmt = (s) => {
    if (!s) return "–";
    const d = new Date(String(s).length <= 10 ? `${s}T12:00:00` : s);
    return Number.isNaN(d.getTime()) ? s : d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
  };

  async function load(name) {
    try {
      const r = await fetch(`data/${name}.json`, { cache: "no-store" });
      return r.ok ? await r.json() : null;
    } catch { return null; }
  }

  /* ---------------------------------------------------------------- charts (inline SVG, no dependencies) */
  function niceTicks(lo, hi, count = 4) {
    const span = hi - lo;
    const raw = span / count;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= count + 1) || 10 * mag;
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-12; v += step) out.push(Math.abs(v) < step / 1e6 ? 0 : v);
    return out;
  }

  function lineChart({ series = [], band = null, zero = true, height = 260, width = 1060, yFmt = (v) => v.toFixed(2), label = "", clip = null }) {
    const W = width, H = height, L = 56, R = 12, T = 12, B = 28;
    const all = [];
    series.forEach((s) => s.y.forEach((v) => isNum(v) && all.push(v)));
    if (band) [...band.lower, ...band.upper].forEach((v) => isNum(v) && all.push(v));
    if (zero) all.push(0);
    if (!all.length) return `<p class="empty">Nothing to chart yet.</p>`;
    let lo = Math.min(...all), hi = Math.max(...all);
    let clipped = false;
    if (clip) {
      if (lo < clip[0]) { lo = clip[0]; clipped = true; }
      if (hi > clip[1]) { hi = clip[1]; clipped = true; }
    }
    if (lo === hi) { lo -= 1; hi += 1; }
    const pad = (hi - lo) * 0.06; lo -= pad; hi += pad;
    const cl = (v) => Math.max(lo, Math.min(hi, v));
    const n = Math.max(...series.map((s) => s.y.length), band ? band.lower.length : 0);
    const x = (i) => L + (n <= 1 ? (W - L - R) / 2 : (i / (n - 1)) * (W - L - R));
    const y = (v) => T + (1 - (cl(v) - lo) / (hi - lo)) * (H - T - B);
    const path = (vals) => vals.map((v, i) => (isNum(v) ? `${i && isNum(vals[i - 1]) ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}` : "")).join("");
    let svg = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(label)}">`;
    niceTicks(lo, hi).forEach((v) => {
      svg += `<line class="axis" x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" opacity=".45"/><text x="${L - 8}" y="${y(v) + 4}" text-anchor="end">${esc(yFmt(v))}</text>`;
    });
    if (band && band.lower.length) {
      const up = band.upper.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("");
      const dn = band.lower.map((v, i) => `L${x(band.lower.length - 1 - i).toFixed(1)},${y(band.lower[band.lower.length - 1 - i]).toFixed(1)}`).join("");
      svg += `<path class="band" d="${up}${dn}Z"/>`;
    }
    if (zero && lo < 0 && hi > 0) svg += `<line class="zero" x1="${L}" x2="${W - R}" y1="${y(0)}" y2="${y(0)}"/>`;
    series.forEach((s) => { svg += `<path class="line ${s.cls || ""}" d="${path(s.y)}"/>`; });
    const xs = series[0]?.x || band?.x || [];
    if (xs.length) {
      [...new Set([0, Math.floor((xs.length - 1) / 2), xs.length - 1])].forEach((i, k, arr) => {
        const anchor = arr.length === 1 ? "middle" : i === 0 ? "start" : i === xs.length - 1 ? "end" : "middle";
        svg += `<text x="${x(i)}" y="${H - 8}" text-anchor="${anchor}">${esc(dateFmt(xs[i]))}</text>`;
      });
    }
    svg += "</svg>";
    if (clipped) svg += `<p class="small muted">The band is still wider than this chart; its edges are cut off.</p>`;
    return svg;
  }

  function barChart(entries, { height = 180, yFmt = (v) => v.toFixed(3), label = "" } = {}) {
    if (!entries.length) return `<p class="empty">No data.</p>`;
    const W = 1060, H = height, L = 56, R = 8, T = 10, B = 26;
    const vals = entries.map((e) => e[1]);
    let lo = Math.min(0, ...vals), hi = Math.max(0, ...vals);
    if (lo === hi) hi = lo + 1;
    const bw = (W - L - R) / entries.length;
    const y = (v) => T + (1 - (v - lo) / (hi - lo)) * (H - T - B);
    let svg = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(label)}">`;
    svg += `<line class="zero" x1="${L}" x2="${W - R}" y1="${y(0)}" y2="${y(0)}"/>`;
    [lo, hi].forEach((v) => { svg += `<text x="${L - 8}" y="${y(v) + 4}" text-anchor="end">${esc(yFmt(v))}</text>`; });
    entries.forEach(([k, v], i) => {
      const top = Math.min(y(v), y(0)), h = Math.abs(y(v) - y(0));
      svg += `<rect class="${v >= 0 ? "bar-pos" : "bar-neg"}" x="${L + i * bw + bw * 0.15}" y="${top}" width="${bw * 0.7}" height="${Math.max(1, h)}"><title>${esc(k)}: ${esc(yFmt(v))}</title></rect>`;
      if (entries.length <= 24 || i % 2 === 0) svg += `<text x="${L + i * bw + bw / 2}" y="${H - 8}" text-anchor="middle">${esc(String(k).slice(-2))}</text>`;
    });
    return svg + "</svg>";
  }

  function table(cols, rows, { id = "", sortable = true } = {}) {
    const head = cols.map((c, i) => `<th class="${c.l ? "l " : ""}${sortable ? "sortable" : ""}" data-i="${i}" tabindex="${sortable ? 0 : -1}" title="${esc(c.title || "")}">${esc(c.h)}</th>`).join("");
    const body = rows.map((r) => `<tr>${cols.map((c) => `<td class="${c.l ? "l " : ""}${c.cls ? c.cls(r) : ""}">${c.f(r)}</td>`).join("")}</tr>`).join("");
    return `<div class="table-scroll"><table ${id ? `id="${id}"` : ""}><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
  }

  function makeSortable(tableEl, cols, rows, render) {
    if (!tableEl) return;
    let state = { i: null, dir: -1 };
    tableEl.querySelectorAll("th.sortable").forEach((th) => {
      const act = () => {
        const i = Number(th.dataset.i);
        state = { i, dir: state.i === i ? -state.dir : -1 };
        const key = cols[i].v || ((r) => r[cols[i].k]);
        const sorted = [...rows].sort((a, b) => {
          const x = key(a), y = key(b);
          if (x == null) return 1; if (y == null) return -1;
          return (x > y ? 1 : x < y ? -1 : 0) * state.dir;
        });
        render(sorted);
      };
      th.addEventListener("click", act);
      th.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); act(); } });
    });
  }

  /* ---------------------------------------------------------------- helpers over data */
  const champId = () => D.status?.champion?.id;
  const modelName = (id) => D.lab?.experiments?.find((e) => e.id === id)?.name || D.live?.models?.[id]?.name || id;
  const roleText = (r) => ({ champion: "Champion", challenger: "Challenger", retired: "Retired", overlay: "Overlay" }[r] || "–");
  const statusText = (s) => ({ gated: "earned through the lab's gates", provisional: "provisional: no model has passed every gate yet", "promoted-live": "promoted on live evidence" }[s] || s || "");

  /* ---------------------------------------------------------------- views */
  function viewOverview() {
    const el = $("#overview");
    const st = D.status || {};
    const seq = D.live?.sequential || {};
    const champ = champId();
    const sk = seq.skill?.[champ];
    const minBlocks = 26;
    let verdict, explain, chart;
    if (!st.snapshots) {
      verdict = st.champion?.id ? "The live record starts at the next open." : "The lab is still being set up.";
      explain = st.champion?.id
        ? "No forecast has been scored yet. Until the first ones mature, the chart shows what the current champion did on history it was not trained on."
        : "The first research sweep has not finished. Once it does, a champion model is chosen by rules written before any results were seen.";
      const c = D.lab?.charts?.[champ];
      const s = c?.holdout?.cumIC || c?.dev?.cumIC;
      chart = s ? lineChart({ series: [{ x: s.dates, y: s.values }], width: 640, yFmt: (v) => v.toFixed(1), label: "Cumulative daily rank correlation on history" })
        + `<p class="legend small">Cumulative daily rank correlation between forecasts and next-week returns, ${c?.holdout ? "sealed holdout years" : "development years"}. Rising means the ranking was right more often than not.</p>` : "";
    } else if (!sk || !sk.n) {
      verdict = "Not shown yet.";
      explain = "Forecasts are frozen every morning, but none has reached its five-session horizon. Weekly checks start once they do.";
      chart = "";
    } else {
      const shown = sk.verdict === "positive skill" && sk.n >= minBlocks;
      verdict = shown ? "Yes, so far." : sk.verdict === "negative skill" ? "No. It is doing worse than chance." : "Not shown yet.";
      const wide = sk.upper - sk.lower > 0.5;
      explain = (wide
        ? `After ${sk.n} weekly check${sk.n === 1 ? "" : "s"} the evidence is still far too thin: the 95% band for the champion's average rank correlation covers almost every possible value. `
        : `After ${sk.n} weekly checks, the champion's average rank correlation with next-week returns is between ${snum(sk.lower)} and ${snum(sk.upper)} with 95% confidence. `)
        + `The band stays valid no matter when you look, so checking often cannot fool it. An edge counts as shown once the whole band sits above zero${shown ? "" : ` after at least ${minBlocks} weeks`}.`;
      const p = sk.path || {};
      chart = lineChart({ series: [{ x: sk.sessions, y: p.mean }], band: { x: sk.sessions, lower: p.lower, upper: p.upper }, width: 640, clip: [-0.25, 0.25], yFmt: (v) => v.toFixed(2), label: "Live mean rank correlation with confidence band" })
        + `<p class="legend"><span><i style="background:var(--band)"></i>running mean</span><span><i style="background:var(--band-fill);height:10px"></i>95% band, valid at any stopping time</span></p>`;
    }
    const src = st.sources || {};
    const ch = st.challengers || [];
    el.innerHTML = `
      <h1>Is there an edge yet?</h1>
      <p class="lede">Every trading morning Unpriced ranks about a thousand US stocks for the coming week, freezes that ranking before the open, and later checks it against what happened.</p>
      <div class="evidence">
        <div>
          <p class="verdict">${esc(verdict)}</p>
          <p>${esc(explain)}</p>
          <dl class="facts">
            <dt>Champion</dt><dd>${esc(st.champion?.name || "none yet")}</dd>
            <dt>Status</dt><dd>${esc(statusText(st.champion?.status) || "–")}</dd>
            <dt>Challengers</dt><dd>${ch.length}</dd>
            <dt>Frozen forecasts</dt><dd>${st.snapshots ?? 0}</dd>
            <dt>Ideas tested in the lab</dt><dd>${st.trials ?? 0}</dd>
            <dt>Last session processed</dt><dd>${esc(dateFmt(st.lastSession))}</dd>
          </dl>
        </div>
        <div>${chart || ""}</div>
      </div>

      <div class="grid2">
        <div>
          <h2>What it does each day</h2>
          <ol class="steps">
            <li>Before the open, it downloads the previous session's prices${src.sec ? ", new SEC filings and fundamentals" : ""}${src.llm ? ", and has Claude read the new filings" : ""}.</li>
            <li>Every live model scores every stock. The scores are written to a file whose fingerprint is chained to the day before, so the past cannot be quietly edited.</li>
            <li>The $1,000 paper portfolio places its orders for the next open, using only that frozen file.</li>
            <li>Older forecasts whose horizon has passed are scored, and the confidence bands above are updated.</li>
          </ol>
        </div>
        <div>
          <h2>How it learns</h2>
          <p>Every Saturday the lab retrains the live models on newer data and tests new ideas on twenty years of history. Each idea tried raises the bar for the next one, so a lucky find cannot sneak through.</p>
          <p>Ideas that pass are shadowed live as challengers. A challenger replaces the champion only when its live record beats the champion's by more than chance allows.</p>
          <p class="small muted">Data: Yahoo prices${src.sec ? ", SEC EDGAR filings" : " (SEC filings switch on once a contact email is configured)"}${src.llm ? ", Claude filing reader" : " (AI filing reader switches on once an API key is configured)"}.</p>
        </div>
      </div>`;
  }

  function viewForecasts() {
    const el = $("#forecasts");
    const f = D.forecasts;
    if (!f || f.status !== "live") {
      el.innerHTML = `<h1>Today's forecasts</h1><p class="empty">No forecast has been frozen yet. The first one appears after the lab picks a champion and the morning cycle runs.</p>`;
      return;
    }
    const m = f.models[f.champion];
    const rows = m.rows;
    const hasProb = rows.some((r) => isNum(r.prob));
    el.innerHTML = `
      <h1>Today's forecasts</h1>
      <p class="lede">Ranked by the champion for the ${m.horizon} sessions starting at the ${esc(dateFmt(f.entrySession))} open.</p>
      <p class="small muted">Frozen ${esc(dateFmt(f.decidedAt))} at ${esc(new Date(f.decidedAt).toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit", timeZone: "America/New_York" }))} ET using data through ${esc(dateFmt(f.session))}. File <a href="https://github.com/PrototypeVersions/UnpricedX/blob/main/${esc(f.file)}">${esc(f.file.split("/").pop())}</a>, SHA-256 ${esc(f.sha256.slice(0, 16))}…</p>
      <div class="note"><p>A rank says which stocks the model expects to do better <em>than the others</em>, not whether any will go up. Most of the time the honest answer is "only slightly": the probabilities below are calibrated on years the model never trained on, and they rarely stray far from 50%.</p></div>
      <div class="controls">
        <input type="search" id="fc-search" placeholder="Find a ticker or company" aria-label="Find a ticker or company">
        <button class="toggle" data-v="top" aria-pressed="true">Top 50</button>
        <button class="toggle" data-v="bottom" aria-pressed="false">Bottom 50</button>
        <button class="toggle" data-v="all" aria-pressed="false">All ${rows.length}</button>
        <label class="small muted"><input type="checkbox" id="fc-trad"> tradable only</label>
      </div>
      <div id="fc-table"></div>
      <h2>Challengers' top picks</h2>
      <p class="muted">Each challenger publishes the same kind of forecast. Their picks are scored exactly like the champion's.</p>
      <div id="fc-ch"></div>`;
    const cols = [
      { h: "Rank", k: "rank", f: (r) => r.rank },
      { h: "Ticker", k: "ticker", l: true, f: (r) => `<span class="ticker">${esc(r.ticker)}</span>` },
      { h: "Company", k: "name", l: true, f: (r) => `<span class="small">${esc(r.name || "")}</span>` },
      { h: "Percentile", k: "pct", f: (r) => `<span class="meter"><b style="left:0;width:${(r.pct * 100).toFixed(0)}%"></b></span> ${pct(r.pct, 0)}` },
      ...(hasProb ? [
        { h: "Chance of beating the median", k: "prob", f: (r) => pct(r.prob, 1), title: "Calibrated on out-of-sample history" },
        { h: "Typical excess return", k: "excess", f: (r) => spct(r.excess, 2), cls: (r) => cls(r.excess), title: "Average return vs. the universe for this score bucket, over the horizon" },
      ] : []),
      { h: "Tradable", k: "tradable", f: (r) => (r.tradable ? "yes" : "no"), title: "In the 500 most liquid names the portfolio may buy" },
    ];
    let mode = "top";
    const render = () => {
      const q = $("#fc-search").value.trim().toUpperCase();
      const trad = $("#fc-trad").checked;
      let rs = rows.filter((r) => (!trad || r.tradable) && (!q || r.ticker.includes(q) || (r.name || "").toUpperCase().includes(q)));
      if (!q) rs = mode === "top" ? rs.slice(0, 50) : mode === "bottom" ? rs.slice(-50).reverse() : rs;
      $("#fc-table").innerHTML = rs.length ? table(cols, rs, { id: "fc-t" }) : `<p class="empty">No ticker matches "${esc(q)}" in today's universe.</p>`;
      makeSortable($("#fc-t"), cols, rs, (sorted) => { $("#fc-table").innerHTML = table(cols, sorted, { id: "fc-t" }); });
    };
    el.querySelectorAll("button.toggle").forEach((b) => b.addEventListener("click", () => {
      mode = b.dataset.v;
      el.querySelectorAll("button.toggle").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      render();
    }));
    $("#fc-search").addEventListener("input", render);
    $("#fc-trad").addEventListener("change", render);
    render();
    const chs = Object.entries(f.models).filter(([id]) => id !== f.champion);
    $("#fc-ch").innerHTML = chs.length ? chs.map(([id, mm]) => `
      <h3>${esc(mm.name || id)}${mm.llmCovered != null ? ` <span class="small muted">(${mm.llmCovered} stocks with a recent filing read by Claude)</span>` : ""}</h3>
      <p class="small">Top: ${mm.rows.slice(0, 15).map((r) => `<strong>${esc(r.ticker)}</strong>`).join(", ")}</p>
      <p class="small muted">Bottom: ${(mm.bottom || []).slice(-15).reverse().map((r) => esc(r.ticker)).join(", ")}</p>`).join("") : `<p class="empty">No challengers yet.</p>`;
  }

  function viewLive() {
    const el = $("#live");
    const lv = D.live || {};
    const models = Object.entries(lv.models || {});
    const seq = lv.sequential || {};
    const chain = lv.chain || {};
    const champ = champId();
    let html = `<h1>Live record</h1>
      <p class="lede">Only forecasts frozen before the open count here. Backtests never enter this page.</p>
      <dl class="kv"><dt>Frozen snapshots</dt><dd>${lv.snapshots ?? 0}</dd>
      <dt>Fingerprint chain</dt><dd>${chain.links ? (chain.ok ? `<span class="pos">intact across ${chain.links} days</span>` : `<span class="neg">broken: ${esc((chain.problems || []).join("; "))}</span>`) : "no snapshots yet"}</dd></dl>`;
    if (!models.length) {
      el.innerHTML = html + `<p class="empty">No forecast has matured yet. One-day forecasts are scored the morning after their first full session, five-day forecasts a week later.</p>`;
      return;
    }
    const rows = models.map(([id, m]) => ({ id, name: m.name, role: id === champ ? "champion" : m.role, ...(m.horizons) }));
    const hcols = [1, 5, 21];
    const cols = [
      { h: "Model", k: "name", l: true, f: (r) => `${esc(modelName(r.id))}` },
      { h: "Role", k: "role", l: true, f: (r) => roleText(r.role) },
      ...hcols.flatMap((h) => [
        { h: `IC ${h}d`, v: (r) => r[h]?.icMean, f: (r) => `<span class="${cls(r[h]?.icMean)}">${snum(r[h]?.icMean, 3)}</span>`, title: `Mean daily rank correlation, ${h}-session horizon` },
        { h: `n`, v: (r) => r[h]?.n, f: (r) => r[h]?.n ?? 0 },
      ]),
      { h: "Top-decile excess 5d", v: (r) => r[5]?.topExcessMean, f: (r) => spct(r[5]?.topExcessMean, 2), cls: (r) => cls(r[5]?.topExcessMean) },
    ];
    html += `<h2>Every live model</h2>${table(cols, rows, { sortable: false })}
      <p class="small muted">IC is the rank correlation between a model's scores and the returns that followed, open to open. Zero means no information; 0.02 to 0.05 held for years would be very good.</p>`;
    const champModel = lv.models[champ];
    const h5 = champModel?.horizons?.[5] || champModel?.horizons?.[1];
    if (h5) {
      html += `<h2>Champion, day by day</h2>` + lineChart({ series: [{ x: h5.series.dates, y: h5.series.cumIC }], yFmt: (v) => v.toFixed(1), label: "Champion cumulative live IC" })
        + `<p class="small muted">Cumulative sum of daily ICs. A steady upward slope is what skill looks like; flat or wandering is what luck looks like.</p>`;
    }
    const comps = Object.entries(seq.comparisons || {});
    if (comps.length) {
      html += `<h2>Challengers against the champion</h2><p>Each band is a 95% range for how much better (above zero) or worse (below) the challenger ranks stocks than the champion, measured on weekly non-overlapping blocks. A challenger is promoted when its band clears zero after at least 26 weeks, and retired when its band falls below zero.</p><div class="grid2">`;
      comps.forEach(([id, c]) => {
        html += `<div><h3>${esc(modelName(id))}</h3><p class="small muted">${c.n} paired weeks, ${esc(c.verdict)}</p>`
          + (c.n ? lineChart({ series: [{ x: c.sessions, y: c.path.mean }], band: { x: c.sessions, lower: c.path.lower, upper: c.path.upper }, height: 220, width: 520, clip: [-0.3, 0.3], yFmt: (v) => v.toFixed(2), label: "Difference in IC" }) : "")
          + `</div>`;
      });
      html += `</div>`;
    }
    const hist = (lv.registryHistory || []).slice().reverse();
    if (hist.length) {
      html += `<h2>Model changes</h2>` + table([
        { h: "When", k: "at", l: true, f: (r) => esc(dateFmt(r.at)) },
        { h: "Event", k: "event", l: true, f: (r) => esc(r.event) },
        { h: "Model", k: "to", l: true, f: (r) => esc(modelName(r.to || r.model || "")) },
      ], hist, { sortable: false });
    }
    el.innerHTML = html;
  }

  function viewPortfolio() {
    const el = $("#portfolio");
    const p = D.portfolio || {};
    const rules = p.rules || {};
    const rulesText = `Long only, ${rules.holdings ?? 10} equal-weight positions from the 500 most liquid stocks, chosen by the champion's ranking. A holding is kept until it drops below rank ${rules.exitRank ?? 40}. Rebalanced every ${rules.rebalanceEverySessions ?? 5} sessions at the first open after the forecast is frozen, with trading costs of ${rules.baseCostBps ?? 2} bps plus a spread estimate per side.`;
    if (!p.status || p.status === "pre-live" || !p.curve?.length) {
      el.innerHTML = `<h1>$1,000 portfolio</h1><p class="lede">${esc(rulesText)}</p><p class="empty">${p.pending ? `First orders are queued for the ${esc(dateFmt(p.pending.executeSession))} open.` : "The portfolio starts with the first frozen forecast."}</p>`;
      return;
    }
    const curve = p.curve;
    const spyOk = curve.some((c) => isNum(c.spy));
    el.innerHTML = `
      <h1>$1,000 portfolio</h1>
      <p class="lede">${esc(rulesText)}</p>
      <dl class="kv">
        <dt>Value</dt><dd><strong>${money(p.equity)}</strong> <span class="${cls(p.return)}">(${spct(p.return)})</span></dd>
        <dt>Same $1,000 in SPY</dt><dd>${money(p.spyValue)} <span class="${cls(p.spyReturn)}">(${spct(p.spyReturn)})</span></dd>
        <dt>Started</dt><dd>${esc(dateFmt(p.startSession))}</dd>
        <dt>Rebalances</dt><dd>${p.rebalances} (${p.tradeCount} trades)</dd>
        <dt>Cash</dt><dd>${money(p.cash)}</dd>
      </dl>
      ${lineChart({ series: [{ x: curve.map((c) => c.session), y: curve.map((c) => c.equity) }, ...(spyOk ? [{ x: curve.map((c) => c.session), y: curve.map((c) => c.spy), cls: "alt" }] : [])], zero: false, yFmt: (v) => `$${v.toFixed(0)}`, label: "Portfolio value versus SPY" })}
      <p class="legend"><span><i style="background:var(--band)"></i>Portfolio</span><span><i style="background:var(--muted)"></i>SPY</span></p>
      <div class="note"><p>The portfolio is a separate experiment from the forecasts. A few weeks of profit or loss says almost nothing about skill; the live record does.</p></div>
      <h2>Holdings</h2>
      ${p.positions.length ? table([
        { h: "Ticker", k: "ticker", l: true, f: (r) => `<span class="ticker">${esc(r.ticker)}</span>` },
        { h: "Company", k: "name", l: true, f: (r) => `<span class="small">${esc(r.name || "")}</span>` },
        { h: "Value", k: "value", f: (r) => money(r.value) },
        { h: "Weight", k: "weight", f: (r) => pct(r.weight) },
        { h: "Since", k: "since", f: (r) => esc(dateFmt(r.since)) },
        { h: "Return since", k: "return", f: (r) => spct(r.return), cls: (r) => cls(r.return) },
      ], p.positions, { sortable: false }) : `<p class="empty">All cash.</p>`}
      ${p.pending ? `<p class="small muted">Next orders, decided from the ${esc(dateFmt(p.pending.decidedFrom))} forecast, execute at the ${esc(dateFmt(p.pending.executeSession))} open: ${p.pending.targets.map(esc).join(", ")}.</p>` : ""}
      <h2>Recent trades</h2>
      ${p.trades?.length ? table([
        { h: "Session", k: "session", l: true, f: (r) => esc(dateFmt(r.session)) },
        { h: "Ticker", k: "ticker", l: true, f: (r) => `<span class="ticker">${esc(r.ticker)}</span>` },
        { h: "Side", k: "side", f: (r) => (r.side === "BUY" ? "Buy" : "Sell") },
        { h: "Amount", k: "dollars", f: (r) => money(r.dollars) },
        { h: "Cost", k: "costDollars", f: (r) => money(r.costDollars) },
      ], p.trades, { sortable: false }) : `<p class="empty">No trades yet.</p>`}`;
  }

  function viewLab() {
    const el = $("#lab");
    const lab = D.lab || {};
    const ex = lab.experiments || [];
    const champ = champId();
    const last = lab.lastSweep || {};
    let html = `<h1>Lab</h1>
      <p class="lede">Where ideas are tested on history before they are allowed near the live record.</p>
      <dl class="kv">
        <dt>Ideas tested</dt><dd>${lab.trials ?? 0}</dd>
        <dt>Development years</dt><dd>${esc(lab.eras?.trainStart?.slice(0, 4) || "")} to ${esc(lab.eras?.devEnd?.slice(0, 4) || "")}, used freely</dd>
        <dt>Holdout years</dt><dd>${esc(lab.eras?.holdoutStart?.slice(0, 4) || "")} onward, sealed: each look is logged and capped at a few per week</dd>
        <dt>False passes on shuffled data</dt><dd>${lab.nullControls?.runs ? `${lab.nullControls.passed} of ${lab.nullControls.runs} runs (each run shuffles outcomes so there is nothing real to find)` : "first check runs with the next weekly lab"}</dd>
        <dt>Last sweep</dt><dd>${last.finishedAt ? `${esc(dateFmt(last.finishedAt))}, ${last.evaluated?.length ?? 0} new experiments in ${last.minutes} min` : "not run yet"}</dd>
      </dl>
      <div class="note"><p>To pass, an idea must show a rank-correlation t-statistic above a bar that rises with every idea tried (now ${num(lab.icBarNow ?? 3, 2)}), make money after trading costs with 95% confidence, and then repeat on the sealed holdout years. History here comes from free data that only includes companies still listed today, which flatters some strategies.</p></div>`;
    const charts = lab.charts || {};
    const ids = Object.keys(charts);
    if (ids.length) {
      html += `<h2>Live models on history</h2><div class="grid2">`;
      ids.slice(0, 4).forEach((id) => {
        const c = charts[id];
        const dev = c.dev?.lsNet, ho = c.holdout?.lsNet;
        html += `<div><h3>${esc(modelName(id))}${id === champ ? " (champion)" : ""}</h3>`;
        if (dev) html += lineChart({ series: [{ x: dev.dates, y: dev.values }], zero: false, height: 220, width: 520, yFmt: (v) => v.toFixed(2), label: "Growth of $1, long-short after costs, development years" }) + `<p class="small muted">Growth of $1 in the top-minus-bottom decile after costs, development years.</p>`;
        if (ho) html += lineChart({ series: [{ x: ho.dates, y: ho.values, cls: "pos" }], zero: false, height: 180, width: 520, yFmt: (v) => v.toFixed(2), label: "Holdout years" }) + `<p class="small muted">Same, sealed holdout years.</p>`;
        html += `</div>`;
      });
      html += `</div>`;
    }
    const champRow = ex.find((e) => e.id === champ);
    if (champRow?.byYear) {
      const ent = Object.entries(champRow.byYear);
      html += `<h2>Champion's rank correlation by year</h2>` + barChart(ent, { yFmt: (v) => v.toFixed(3), label: "IC by year" });
    }
    html += `<h2>Every experiment</h2><p class="muted">Sorted by after-cost Sharpe ratio on development years. Click a column to sort.</p><div class="controls">
      <select id="lab-kind" aria-label="Filter by kind"><option value="">All kinds</option><option>factor</option><option>ridge</option><option>gbm</option><option>ensemble</option></select>
      <label class="small muted"><input type="checkbox" id="lab-pass"> passed only</label></div><div id="lab-table"></div>
      <h2>Features</h2><p class="muted">Every input the models can use, each computed only from information available on the day of the forecast.</p><div id="lab-features"></div>`;
    el.innerHTML = html;
    const cols = [
      { h: "Experiment", k: "name", l: true, f: (r) => `${esc(r.name)}${r.role ? ` <span class="pill">${esc(roleText(r.role))}</span>` : ""}` },
      { h: "Kind", k: "kind", f: (r) => esc(r.kind) },
      { h: "Horizon", k: "horizon", f: (r) => `${r.horizon}d` },
      { h: "IC", k: "icMean", f: (r) => snum(r.icMean, 4), cls: (r) => cls(r.icMean), title: "Mean daily rank correlation, development years" },
      { h: "t", k: "icT", f: (r) => num(r.icT, 1), title: "Newey-West t-statistic of the IC" },
      { h: "Years up", k: "yearsPositive", f: (r) => pct(r.yearsPositive, 0) },
      { h: "Long-short net", k: "lsNet", f: (r) => spct(r.lsNet, 1), cls: (r) => cls(r.lsNet), title: "Annual return of top minus bottom decile after costs" },
      { h: "Sharpe", k: "lsSharpe", f: (r) => num(r.lsSharpe, 2) },
      { h: "Deflated", k: "dsr", f: (r) => pct(r.dsr, 0), title: "Deflated Sharpe ratio: chance the edge is real after accounting for every idea tried" },
      { h: "Dev", k: "passedDev", f: (r) => (r.passedDev ? `<span class="pos">pass</span>` : `<span class="muted">fail</span>`) },
      { h: "Holdout", v: (r) => r.holdout?.icT, f: (r) => (r.holdout ? `<span class="${r.holdout.passed ? "pos" : "neg"}">${snum(r.holdout.icMean, 4)} (t ${num(r.holdout.icT, 1)})</span>` : `<span class="muted">untested</span>`) },
    ];
    const render = () => {
      const kind = $("#lab-kind").value, pass = $("#lab-pass").checked;
      const rs = ex.filter((r) => (!kind || r.kind === kind) && (!pass || r.passedDev));
      $("#lab-table").innerHTML = rs.length ? table(cols, rs, { id: "lab-t" }) : `<p class="empty">No experiments match. ${ex.length ? "" : "The first sweep runs after the data bootstrap."}</p>`;
      makeSortable($("#lab-t"), cols, rs, (sorted) => { $("#lab-table").innerHTML = table(cols, sorted, { id: "lab-t" }); });
    };
    $("#lab-kind").addEventListener("change", render);
    $("#lab-pass").addEventListener("change", render);
    render();
    const feats = lab.features || [];
    $("#lab-features").innerHTML = table([
      { h: "Feature", k: "name", l: true, f: (r) => `<strong>${esc(r.name)}</strong>` },
      { h: "Family", k: "family", l: true, f: (r) => esc(r.family) },
      { h: "What it measures", k: "doc", l: true, f: (r) => esc(r.doc) },
      { h: "Source", k: "ref", l: true, f: (r) => `<span class="muted">${esc(r.ref)}</span>` },
    ], feats, { sortable: false });
  }

  function viewFilings() {
    const el = $("#filings");
    const f = D.filings || {};
    let html = `<h1>AI filings</h1><p class="lede">Claude reads new 8-K, 10-Q and 10-K filings for stocks in the universe and records whether each looks good or bad news, how surprising, and how much it matters.</p>
      <div class="note"><p>This signal is only judged live. A language model has read the internet up to its training cutoff, so testing it on older filings would let it remember what happened next. Research on news-reading models also finds the edge is short-lived and smallest in large, cheap-to-trade stocks, so the bar here is real.</p></div>`;
    if (!f.enabled && !f.count) {
      html += `<p class="empty">Switched off. It starts automatically once an Anthropic API key and an SEC contact email are added to the repository's secrets.</p>`;
      el.innerHTML = html;
      return;
    }
    html += `<dl class="kv"><dt>Status</dt><dd>${f.enabled ? "reading every morning" : "paused (key or email missing)"}</dd><dt>Filings read</dt><dd>${f.count}</dd><dt>Model</dt><dd>${esc(f.model)}</dd></dl>`;
    if (f.evaluation?.byDirection?.length) {
      html += `<h2>Did the calls work?</h2><p class="muted">${esc(f.evaluation.note)}</p>` + table([
        { h: "Claude's call", k: "direction", l: true, f: (r) => ({ "-2": "Very negative", "-1": "Negative", "0": "Neutral", "1": "Positive", "2": "Very positive" }[String(r.direction)]) },
        { h: "Filings", k: "count", f: (r) => r.count },
        { h: "Average excess return", k: "mean", f: (r) => spct(r.mean, 2), cls: (r) => cls(r.mean) },
      ], f.evaluation.byDirection, { sortable: false });
    }
    html += `<h2>Recent reads</h2>` + (f.recent?.length ? table([
      { h: "Ticker", k: "ticker", l: true, f: (r) => `<span class="ticker">${esc(r.ticker)}</span>` },
      { h: "Form", k: "form", l: true, f: (r) => esc(r.form) },
      { h: "Public", k: "reactionSession", f: (r) => esc(dateFmt(r.reactionSession)) },
      { h: "Call", k: "direction", f: (r) => `<span class="${cls(r.direction)}">${snum(r.direction, 0)}</span>` },
      { h: "Surprise", k: "surprise", f: (r) => snum(r.surprise, 0) },
      { h: "Matters", k: "materiality", f: (r) => pct(r.materiality, 0) },
      { h: "Summary", k: "summary", l: true, f: (r) => `<span class="small">${esc(r.summary)}</span>` },
    ], f.recent.slice(0, 80), { sortable: false }) : `<p class="empty">No filings read yet.</p>`);
    el.innerHTML = html;
  }

  function viewMethod() {
    const el = $("#method");
    const sch = D.status?.schedule || {};
    el.innerHTML = `<h1>How it works</h1>
      <p class="lede">Unpriced separates discovering ideas from proving them. History is used to find and reject ideas quickly; only forecasts frozen before the market opens can prove anything.</p>
      <h2>What is predicted</h2>
      <p>For about a thousand of the most traded US stocks, each model gives a score for the next five sessions, entered at the next open. Scores rank stocks against each other, so a market-wide rally or crash does not count as skill. The universe on each date is chosen from that date's own trading volume and price, never from which companies later became large.</p>
      <h2>Keeping it honest</h2>
      <ol class="steps">
        <li><strong>No look-ahead.</strong> Every input is computed from information public before the forecast. Fundamentals count from the session after they were filed with the SEC. An automated test deletes the future and checks that no past input changes.</li>
        <li><strong>Every idea is counted.</strong> The lab's ledger records each experiment, passed or failed, and can only grow. The statistical bar rises with the count.</li>
        <li><strong>A sealed holdout.</strong> Recent years are kept out of development; looks at them are logged and limited.</li>
        <li><strong>Frozen, fingerprinted forecasts.</strong> Each morning's forecast file is fingerprinted with SHA-256 and chained to the day before. Changing any past file breaks the chain, and a check runs on every change to the repository.</li>
        <li><strong>Evidence that survives peeking.</strong> Live comparisons use confidence sequences, which stay valid however often they are checked, on non-overlapping weekly blocks. Decisions wait for at least 26 weeks.</li>
        <li><strong>Realistic trading.</strong> Orders fill at the first open after the forecast file exists, with costs that grow for less liquid stocks.</li>
      </ol>
      <h2>Known limitations</h2>
      <ul class="steps">
        <li>Free price data covers only companies listed today, so history is missing firms that went bankrupt or were acquired. Strategies that buy beaten-down stocks look better on history than they would have been. The live record has no such bias, and the stored history keeps delisted names from now on.</li>
        <li>Industries are grouped by how stocks have moved together, not by official classification.</li>
        <li>An edge of the size plausible here takes a long time to confirm live: in simulation, a rank correlation of 0.03 needs about a year of weekly checks and 0.02 about two. That is a property of markets, not a flaw to tune away.</li>
      </ul>
      <h2>Schedule</h2>
      <dl class="kv">${Object.entries(sch).map(([k, v]) => `<dt>${esc({ daily: "Morning cycle", weekly: "Weekly lab", aiHealthCheck: "AI health check", aiResearch: "AI research session", aiReview: "AI monthly review" }[k] || k)}</dt><dd>${esc(v)}</dd>`).join("")}</dl>
      <h2>References</h2>
      <ul class="steps small">
        <li>Bailey &amp; López de Prado (2014), The deflated Sharpe ratio.</li>
        <li>Harvey, Liu &amp; Zhu (2016), …and the cross-section of expected returns.</li>
        <li>Waudby-Smith, Arbour, Sinha, Kennedy &amp; Ramdas (2024), Time-uniform central limit theory and asymptotic confidence sequences; Choe &amp; Ramdas, Comparing sequential forecasters.</li>
        <li>Gu, Kelly &amp; Xiu (2020), Empirical asset pricing via machine learning.</li>
        <li>Lopez-Lira &amp; Tang, Can ChatGPT forecast stock price movements?; He, Lv, Manela &amp; Wu (2025), Chronologically consistent large language models.</li>
      </ul>`;
  }

  const views = { overview: viewOverview, forecasts: viewForecasts, live: viewLive, portfolio: viewPortfolio, lab: viewLab, filings: viewFilings, method: viewMethod };
  const rendered = new Set();

  function route() {
    const id = (location.hash || "#overview").slice(1);
    const target = views[id] ? id : "overview";
    document.querySelectorAll(".view").forEach((s) => { s.hidden = s.id !== target; });
    document.querySelectorAll(".tabs a").forEach((a) => a.setAttribute("aria-current", a.getAttribute("href") === `#${target}` ? "page" : "false"));
    if (!rendered.has(target)) {
      try { views[target](); } catch (e) { $(`#${target}`).innerHTML = `<p class="empty">This section could not be drawn: ${esc(e.message)}</p>`; }
      rendered.add(target);
    }
    window.scrollTo({ top: 0 });
  }

  Promise.all(["status", "forecasts", "live", "portfolio", "lab", "filings"].map(async (k) => { D[k] = await load(k); })).then(() => {
    window.addEventListener("hashchange", route);
    route();
  });
})();

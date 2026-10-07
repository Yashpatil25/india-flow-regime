"""Catch-up or grind? A weekly flow-regime tracker for Indian equities.
Signals: Brent crude, USD/INR, US 10Y yield, India VIX, FPI (FII) equity flows.
Illustrative research, not investment advice."""
import sys
import numpy as np, pandas as pd
import plotly.graph_objects as go

TICKERS = {"crude": "BZ=F", "usdinr": "INR=X", "us10y": "^TNX", "vix": "^INDIAVIX", "nifty": "^NSEI"}
START, LOOK, FWD, ZWIN = "2010-01-01", 13, 13, 156   # weeks: signal lookback, forward replay, z-score window (3y)
FII_LOOK = 4   # v2 (after first sanity check): was LOOK with ZWIN z-score
HI, SHARP = 0.5, -0.5  # calibration knobs: "elevated" pressure level, "sharp" 4-week easing
LABELS = {"crude": "Brent crude", "usdinr": "USD/INR", "us10y": "US 10Y yield", "vix": "India VIX", "fii": "FII outflows"}
NOTES = {  # your own explanation per "Where it fails" date, keyed "YYYY-MM-DD"
}
PERIODS = {"Taper tantrum, May–Aug 2013": ("2013-05-01", "2013-08-31"), "Crude and rupee, Sep–Oct 2018": ("2018-09-01", "2018-10-31"),
           "Covid, Mar 2020": ("2020-03-01", "2020-03-31"), "Fed hikes, H1 2022": ("2022-01-01", "2022-06-30"),
           "FII selloff, Oct 2024–Feb 2025": ("2024-10-01", "2025-02-28")}
V1 = ["11/18", "8/8", "2/4", "11/25", "7/22"]   # Pressure weeks under v1, frozen at the first sanity check
READ = {  # one-line answer to "catch-up or grind?" per regime
    "Pressure": f"Neither yet: pressure is still building. A grind starts when it stops rising; a catch-up needs it to fall more than {abs(SHARP)} in four weeks.",
    "Grind": f"Grind: pressure is elevated but no longer rising. A catch-up needs it to fall more than {abs(SHARP)} in four weeks.",
    "Catch-up": "Catch-up: pressure is easing sharply from an elevated level.",
    "Calm": "Neither: there is little flow pressure to grind through or catch up from.",
}
COLORS = {"Pressure": "#B23A48", "Grind": "#C98B2B", "Catch-up": "#2A7F62", "Calm": "#8A96A8"}


def load_prices():
    import yfinance as yf
    px = yf.download(list(TICKERS.values()), start=START, auto_adjust=True, progress=False)["Close"]
    w = px.rename(columns={v: k for k, v in TICKERS.items()}).resample("W-FRI").last().ffill()
    w.attrs["last"] = min(px[c].last_valid_index() for c in px)   # date every series has a price for
    return w


def load_fii(path="fii_flows.csv"):
    """CSV with columns: date, net_inr_cr (daily FPI net equity investment, ₹ crore)."""
    f = pd.read_csv(path, parse_dates=["date"]).set_index("date")["net_inr_cr"]
    return f.resample("W-FRI").sum(min_count=1)


def compute(px, fii):
    raw = pd.DataFrame({
        "crude": np.log(px.crude).diff(LOOK),     # rising crude = pressure
        "usdinr": np.log(px.usdinr).diff(LOOK),   # weaker rupee = pressure
        "us10y": px.us10y.diff(LOOK),             # rising US yields = pressure
        "vix": px.vix.diff(LOOK),                 # rising fear = pressure
        "fii": -fii.reindex(px.index).rolling(FII_LOOK).sum(),  # outflows = pressure
    })
    z = (raw - raw.rolling(ZWIN, min_periods=52).mean()) / raw.rolling(ZWIN, min_periods=52).std()  # past data only
    # v2: FII on an expanding window, so a long selling streak can't become the "normal" baseline
    z["fii"] = (raw.fii - raw.fii.expanding(52).mean()) / raw.fii.expanding(52).std()
    d = z.copy()
    d["pressure"] = z.mean(axis=1)            # ponytail: equal weights; fit weights only if replay justifies it
    d["d4"] = d.pressure.diff(4)
    was_hi = d.pressure.rolling(8).max() > HI
    d["regime"] = np.select(
        [(d.pressure > HI) & (d.d4 > 0), was_hi & (d.d4 < SHARP), d.pressure > 0],
        ["Pressure", "Catch-up", "Grind"], "Calm")
    d["nifty"] = px.nifty
    d["fwd"] = px.nifty.shift(-FWD) / px.nifty - 1
    return d.dropna(subset=["pressure"])


def replay(d):
    r = d.dropna(subset=["fwd"]).groupby("regime").fwd
    return pd.DataFrame({"weeks": r.size(), "median": r.median(), "hit": r.apply(lambda x: (x > 0).mean())})


def failures(d, n=5, gap=FWD):  # gap = one forward window, so one episode = one entry
    """Worst calls: Catch-up/Calm weeks followed by the biggest Nifty falls, one per episode."""
    bad = d[d.regime.isin(["Catch-up", "Calm"]) & (d.fwd < 0)].sort_values("fwd")
    kept = []
    for t, row in bad.iterrows():
        if all(abs((t - k).days) > gap * 7 for k, _ in kept):
            kept.append((t, row))
        if len(kept) == n:
            break
    return kept


def render(d, fii_last, px_last):
    now = d.iloc[-1]
    fig = go.Figure()
    for reg, col in COLORS.items():
        m = d.regime == reg
        fig.add_bar(x=d.index[m], y=d.pressure[m], name=reg, marker_color=col)
    fig.add_scatter(x=d.index, y=d.nifty, name="Nifty 50", yaxis="y2", line=dict(color="#14213D", width=1.5))
    fig.update_layout(barmode="overlay", bargap=0, height=420, margin=dict(l=10, r=10, t=10, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_family="IBM Plex Sans",
                      yaxis=dict(title="Pressure (z)"), yaxis2=dict(overlaying="y", side="right", type="log", showgrid=False),
                      legend=dict(orientation="h", y=-0.12))
    sig = "".join(f"<tr><td>{LABELS[k]}</td><td>{now[k]:+.2f}</td><td>{'neutral' if abs(now[k]) < 0.5 else 'adds pressure' if now[k] > 0 else 'eases pressure'}</td></tr>"
                  for k in LABELS if pd.notna(now[k]))
    rp = replay(d).reindex(list(COLORS)).dropna()
    rep = "".join(f"<tr><td>{k}</td><td>{int(v.weeks)}</td><td>{v['median']:+.1%}</td><td>{v.hit:.0%}</td></tr>" for k, v in rp.iterrows())
    ver = "".join(f"<tr><td>{k}</td><td>{v1}</td><td>{(w.regime == 'Pressure').sum()}/{len(w)}</td></tr>"
                  for (k, (a, b)), v1 in zip(PERIODS.items(), V1) for w in [d.loc[a:b]])
    fail = "".join(f"<li><b>{t:%d %b %Y}</b>: read {r.regime}, Nifty then fell {-r.fwd:.1%} over {FWD} weeks. {NOTES.get(f'{t:%Y-%m-%d}', '')}</li>" for t, r in failures(d))
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Catch-up or grind? India flow-regime tracker</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&family=Source+Serif+4:opsz,wght@8..60,600&display=swap" rel="stylesheet">
<style>
:root{{--ink:#14213D;--bg:#F5F7F8;--muted:#5B6878;--line:#D5DBE1;--reg:{COLORS[now.regime]}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 "IBM Plex Sans",system-ui,sans-serif}}
main{{max-width:880px;margin:auto;padding:40px 20px}}
h1{{font:600 clamp(2.4rem,7vw,4.2rem)/1.05 "Source Serif 4",Georgia,serif;margin:.2em 0;color:var(--reg)}}
h2{{font:600 1.35rem "Source Serif 4",Georgia,serif;margin:2.2em 0 .5em}}
p,li{{max-width:68ch}} .muted{{color:var(--muted);font-size:.9rem}}
table{{border-collapse:collapse;width:100%}} td,th{{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line)}}
.wrap{{overflow-x:auto}}
</style></head><body><main>
<p class="muted">Prices through {px_last:%d %b %Y}. FII data through {fii_last:%d %b %Y}.</p>
<p>Is India's flow-driven regime heading for a catch-up or a grind? This week's read:</p>
<h1>{now.regime}</h1>
<p><b>{READ[now.regime]}</b></p>
<p>Composite pressure {now.pressure:+.2f} (z-score), {abs(now.d4):.2f} {'lower' if now.d4 < 0 else 'higher'} than four weeks ago.</p>
<div class="wrap">{fig.to_html(full_html=False, include_plotlyjs="cdn")}</div>
<h2>What each signal says now</h2><div class="wrap"><table><tr><th>Signal</th><th>z-score</th><th>Effect</th></tr>{sig}</table></div>
<h2>Regimes describe, they don't forecast</h2>
<p>Nifty over the next {FWD} weeks, replayed since {d.index[0]:%Y}. No regime separates forward returns: medians range {rp['median'].min():+.1%} to {rp['median'].max():+.1%}, {rp.hit.min() * 100:.0f}–{rp.hit.max():.0%} of weeks positive.</p>
<div class="wrap"><table><tr><th>Regime</th><th>Weeks</th><th>Median return</th><th>Share positive</th></tr>{rep}</table></div>
<p class="muted">Overlapping windows, so weeks are not independent samples.</p>
<h2>Where it fails</h2><ul>{fail}</ul>
<h2>Method</h2>
<p>Each signal is its {LOOK}-week change (FII: {FII_LOOK}-week net flow), z-scored against the prior three years (FII: all history since 2010), using past data only, signed so that higher means more pressure on Indian equities. Pressure is their equal-weight average. Pressure regime: above {HI} and rising. Catch-up: elevated within 8 weeks and down more than {abs(SHARP)} in 4 weeks. Grind: above zero otherwise. Calm: at or below zero.</p>
<h2>Version note</h2>
<p>v2 changed one thing, after the first sanity check: the FII signal went from a {LOOK}-week sum z-scored on a rolling three-year window to a {FII_LOOK}-week sum z-scored on an expanding window from 2010. Under v1, years of steady selling had become the baseline, so heavy outflows barely registered. All other signals and both thresholds are unchanged, and there will be no further tuning.</p>
<div class="wrap"><table><tr><th>Known stress period</th><th>v1 Pressure weeks</th><th>v2 Pressure weeks</th></tr>{ver}</table></div>
<p class="muted">Illustrative research by Yash Patil, not investment advice. Data: Yahoo Finance, NSDL. <a href="https://yashpatil25.github.io/Portfolio/">Portfolio</a></p>
</main></body></html>"""


def selftest():
    idx = pd.date_range("2012-01-06", periods=600, freq="W-FRI")
    rng = np.random.default_rng(0)
    walk = lambda s: pd.Series(np.exp(np.cumsum(rng.normal(0, s, len(idx)))), idx)
    px = pd.DataFrame({"crude": 80 * walk(.04), "usdinr": 60 * walk(.01), "us10y": 2 + walk(.05),
                       "vix": 15 * walk(.06), "nifty": 6000 * walk(.025)})
    fii = pd.Series(rng.normal(0, 3000, len(idx)), idx)
    d = compute(px, fii)
    assert set(d.regime) <= set(COLORS) and d.index.is_monotonic_increasing
    assert d.regime.nunique() > 1, "regime never changes: check thresholds"
    html = render(d, fii.index[-1], idx[-1])
    assert "<h1>" in html and "Where it fails" in html
    print("selftest ok:", d.regime.value_counts().to_dict())


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        fii = load_fii()
        px = load_prices()
        d = compute(px, fii)
        open("docs/index.html", "w", encoding="utf-8").write(render(d, pd.to_datetime(pd.read_csv("fii_flows.csv").date).max(), px.attrs["last"]))
        print(f"{d.index[-1]:%Y-%m-%d}: {d.regime.iloc[-1]} (pressure {d.pressure.iloc[-1]:+.2f})")

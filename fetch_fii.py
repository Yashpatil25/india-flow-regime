"""Pull daily FPI net equity investment (Rs crore) from NSDL's archive into fii_flows.csv.
One POST per month: the month-end archive lists every reporting date in that month.
Incremental: re-fetches from the last month already in the CSV. `--full` rebuilds from 2010."""
import re, sys, time
import pandas as pd, requests

URL = "https://www.fpi.nsdl.co.in/web/Reports/Archive.aspx"
OUT = "fii_flows.csv"
DATE = re.compile(r"\d{2}-[A-Z][a-z]{2}-\d{4}$")


def num(s):  # "(3,315.29)" -> -3315.29
    s = s.replace(",", "").strip()
    return -float(s[1:-1]) if s.startswith("(") else float(s)


def parse(html):
    """Equity Sub-total net per reporting date (first Sub-total after each date is Equity)."""
    html = html[html.find("dvArchiveData"):]
    out, date, cat = {}, None, None
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S):
        cells = [re.sub(r"<[^>]+>", "", c).strip() for c in re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)]
        if cells and DATE.match(cells[0]):
            date, cat = cells[0], cells[1]
        elif len(cells) > 1 and cells[1] == "Stock Exchange":  # new category block, e.g. "Debt"
            cat = cells[0]
        if cells and cells[0] == "Sub-total" and cat == "Equity" and date not in out:
            out[date] = num(cells[3])
    return out


def fetch(months):
    s = requests.Session()  # curl gets connection-reset by NSDL; requests works
    s.headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"
    form = dict(re.findall(r'name="(__\w+)"[^>]*value="([^"]*)"', s.get(URL, timeout=60).text))
    rows = {}
    for m in months:
        d = f"{min(m.to_timestamp(how='end'), pd.Timestamp.today()):%d-%b-%Y}"
        for attempt in range(3):
            try:
                html = s.post(URL, timeout=90, data=dict(form, __EVENTTARGET="btnSubmit1", __EVENTARGUMENT="",
                                                         hdnDate=d, txtDate=d, hdnFlag="")).text
                break
            except requests.RequestException:
                time.sleep(5 * (attempt + 1))
        else:
            raise SystemExit(f"NSDL fetch failed for {d}")
        got = parse(html)
        print(d, len(got), "days", flush=True)
        rows.update(got)
        time.sleep(0.5)
    return pd.Series(rows, name="net_inr_cr").rename_axis("date").pipe(lambda x: x.set_axis(pd.to_datetime(x.index, format="%d-%b-%Y")))


if __name__ == "__main__":
    old = pd.read_csv(OUT, parse_dates=["date"]).set_index("date").net_inr_cr
    start = pd.Period("2010-01", "M") if "--full" in sys.argv or old.index.min() > pd.Timestamp("2011-01-01") else old.index.max().to_period("M")
    new = fetch(pd.period_range(start, pd.Timestamp.today().to_period("M"), freq="M"))
    merged = pd.concat([old, new]) if start > pd.Period("2010-01", "M") else new
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    merged.round(2).to_csv(OUT, date_format="%Y-%m-%d")
    print(f"{len(merged)} days, {merged.index.min():%Y-%m-%d} to {merged.index.max():%Y-%m-%d}")

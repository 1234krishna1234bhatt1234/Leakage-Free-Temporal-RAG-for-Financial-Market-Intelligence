"""Event resolver: turns a hint like 'May earnings' into trading-day dates.
Code finds the dates. The LLM only supplies the hint text."""
from src.config import *
import calendar
from src.prices import daily_returns

EARNINGS = BASE / "earnings.parquet"

# ---------- trading-day helper ----------
def next_trading_day(px, date, after_close=False):
    """First trading day on/after `date`. If the news came after the close, the next one."""
    d = pd.Timestamp(date).normalize()
    if d.tzinfo is not None:
        d = d.tz_localize(None)
    i = px.index.searchsorted(d, side="left")
    if after_close and i < len(px.index) and px.index[i] == d:
        i += 1
    return px.index[i] if i < len(px.index) else None

# ---------- earnings dates (best effort: yfinance history can be short) ----------
def fetch_earnings(tickers, limit=60):
    import yfinance as yf
    rows = []
    for t in tickers:
        try:
            e = yf.Ticker(t).get_earnings_dates(limit=limit)
        except Exception as ex:
            print(f"earnings failed for {t}: {ex}")
            continue
        if e is None or e.empty:
            continue
        for dt in e.index:
            dt = pd.Timestamp(dt)
            dt = dt.tz_localize("America/New_York") if dt.tzinfo is None else dt.tz_convert("America/New_York")
            rows.append({"ticker": t, "date": dt.tz_localize(None).normalize(),
                         "after_close": dt.hour >= 16})
    out = pd.DataFrame(rows)
    out.to_parquet(EARNINGS, index=False)
    print(len(out), "earnings dates;", out.groupby("ticker").size().describe()[["min", "max"]].to_dict())
    return out

def load_earnings():
    return pd.read_parquet(EARNINGS) if EARNINGS.exists() else pd.DataFrame(
        columns=["ticker", "date", "after_close"])

# ---------- large-move fallback ----------
def large_move_days(px, ticker, start, end, as_of, z=3.0, window=250):
    """Days where |return| > z x rolling std of PAST returns (shift(1): no look-ahead)."""
    r = daily_returns(px)[ticker].dropna()
    sd = r.rolling(window, min_periods=60).std().shift(1)
    zs = r / sd
    hi = min(pd.Timestamp(end), pd.Timestamp(as_of))
    m = (zs.abs() > z) & (r.index >= pd.Timestamp(start)) & (r.index <= hi)
    return pd.DataFrame({"date": r.index[m], "ret": r[m].values, "z": zs[m].values})

# ---------- hint parsing ----------
def parse_month(hint):
    h = hint.lower()
    for i in range(1, 13):
        full, abbr = calendar.month_name[i].lower(), calendar.month_abbr[i].lower()
        if full in h or re.search(rf"\b{abbr}\b", h):
            return i
    return None

# ---------- resolver ----------
def resolve_events(px, ticker, start, end, as_of, hint="", earnings=None, z=3.0, top=5):
    """Return a list of {'date', 'kind', ...} events, all on or before as_of.
    Earnings are used when available. Otherwise large-move days (kind='large_move')."""
    lo, hi = pd.Timestamp(start), min(pd.Timestamp(end), pd.Timestamp(as_of))
    month = parse_month(hint)
    wants_earnings = (not hint) or any(w in hint.lower() for w in ("earn", "result", "quarter"))

    cands = []
    if earnings is not None and len(earnings) and wants_earnings:
        for r in earnings[earnings.ticker == ticker].itertuples():
            day = next_trading_day(px, r.date, bool(r.after_close))
            if day is not None and lo <= day <= hi:
                cands.append({"date": day, "kind": "earnings"})
    if not cands:                                    # fallback, and flagged as such
        lm = large_move_days(px, ticker, start, end, as_of, z)
        lm = lm.reindex(lm.z.abs().sort_values(ascending=False).index).head(top * 3)
        cands = [{"date": r.date, "kind": "large_move", "ret": float(r.ret), "z": float(r.z)}
                 for r in lm.itertuples()]
    if month:
        cands = [c for c in cands if c["date"].month == month]
    cands = sorted(cands, key=lambda c: c["date"])
    assert all(c["date"] <= hi for c in cands), "event after as_of"
    return cands
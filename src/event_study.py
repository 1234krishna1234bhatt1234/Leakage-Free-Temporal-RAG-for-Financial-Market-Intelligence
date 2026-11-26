"""Market-model event study. Uses prices up to as_of only.
Day 0 = first trading day the market could react (after-close news already mapped by events.py)."""
from src.config import *
from src.prices import daily_returns

def event_study(px, ticker, event_date, as_of, index=INDEX_SYMBOL,
                est=(-120, -20), min_est=100, windows=((-1, 1), (0, 5))):
    from scipy import stats
    r = daily_returns(px)[[ticker, index]].dropna()
    r = r[r.index <= pd.Timestamp(as_of)]                    # nothing after the cutoff
    i0 = int(r.index.searchsorted(pd.Timestamp(event_date)))
    base = {"ticker": ticker, "event_date": str(pd.Timestamp(event_date).date())}

    if i0 >= len(r):
        return {**base, "status": "event_after_as_of"}
    if i0 + est[0] < 0 or (est[1] - est[0] + 1) < min_est:
        return {**base, "status": "insufficient_history"}
    

    e = r.iloc[i0 + est[0]: i0 + est[1] + 1]
    if len(e) < min_est:
        return {**base, "status": "insufficient_history"}
    beta, alpha = np.polyfit(e[index].values, e[ticker].values, 1)
    resid = e[ticker].values - (alpha + beta * e[index].values)
    sigma = resid.std(ddof=2)
    out = {**base, "status": "ok", "alpha": float(alpha), "beta": float(beta),
           "sigma": float(sigma), "n_est": int(len(e))}
    
    for a, b in windows:
        if i0 + b > len(r) - 1:                       # window would run past as_of
            out[f"car_{a}_{b}"] = None
            out[f"note_{a}_{b}"] = "window_extends_past_as_of"
            continue
        w = r.iloc[i0 + a: i0 + b + 1]
        ar = w[ticker].values - (alpha + beta * w[index].values)
        car, L = float(ar.sum()), len(w)
        t = car / (sigma * np.sqrt(L))
        out[f"car_{a}_{b}"] = car
        out[f"t_{a}_{b}"] = float(t)
        out[f"p_{a}_{b}"] = float(2 * stats.t.sf(abs(t), df=len(e) - 2))
        out[f"mktadj_{a}_{b}"] = float((w[ticker] - w[index]).sum())   # simple fallback
    return out
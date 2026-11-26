"""Point-in-time retrieval. Switch parts off to get the baselines and ablations:
  plain RAG:   use_asof=False, use_window=False, use_decay=False, use_buckets=False
  recency RAG: use_asof=False, use_window=False, use_decay=True,  use_buckets=False
  full T-RAG:  defaults
"""
from src.config import *

def month_edges(s_ts, e_ts):
    s = pd.Timestamp(s_ts, unit="s", tz="UTC")
    e = pd.Timestamp(e_ts, unit="s", tz="UTC")
    edges, cur = [s], s.normalize().replace(day=1) + pd.offsets.MonthBegin(1)
    while cur <= e:
        edges.append(cur)
        cur += pd.offsets.MonthBegin(1)
    edges.append(e + pd.Timedelta(seconds=1))
    return [int(x.timestamp()) for x in edges]

def run_query(col, qe, ticker, lo, hi, n):
    """One Chroma query. Date and ticker filters run INSIDE the database."""
    conds = []
    if lo is not None: conds.append({"publish_time": {"$gte": lo}})
    if hi is not None: conds.append({"publish_time": {"$lte": hi}})
    if ticker:         conds.append({"ticker": ticker})
    where = None if not conds else conds[0] if len(conds) == 1 else {"$and": conds}
    r = col.query(query_embeddings=qe, n_results=n, where=where,
                  include=["metadatas", "documents", "distances"])
    return [{"doc_id": i, **m, "text": t, "sim": 1 - d}
            for i, m, t, d in zip(r["ids"][0], r["metadatas"][0],
                                  r["documents"][0], r["distances"][0])]

def retrieve(query, ticker, start, end, as_of, col, model, sent=None,
             k_per_bucket=K_PER_BUCKET, n_candidates=N_CANDIDATES, k_total=K_TOTAL,
             use_asof=True, use_window=True, use_decay=True, use_buckets=True,
             alpha=ALPHA, half_life=HALF_LIFE):
    qe = model.encode([query], normalize_embeddings=True).tolist()
    as_of_ts = ts(as_of) + DAY - 1                 # as_of = end of that UTC day, inclusive
    lo, hi = ts(start), ts(end) + DAY - 1
    if use_asof:
        hi = min(hi, as_of_ts)                     # the window can never pass the cutoff
    if use_window and lo > hi:
        return pd.DataFrame()                      # window starts after the cutoff
    if not use_window:
        lo = None
        hi = as_of_ts if use_asof else None        # keep the cutoff if as-of is on
        use_buckets = False

    if use_buckets:
        edges = month_edges(lo, hi)
        rows = []
        for i in range(len(edges) - 1):
            rows += [dict(r, bucket=i)
                     for r in run_query(col, qe, ticker, edges[i], edges[i + 1] - 1, n_candidates)]
    else:
        rows = [dict(r, bucket=0) for r in run_query(col, qe, ticker, lo, hi, 50)]

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if sent is not None:
        df = df.join(sent, on="doc_id")            # attach FinBERT scores
    age = ((as_of_ts - df.publish_time) / DAY).clip(lower=0)
    df["decay"] = (-math.log(2) * age / half_life).map(math.exp)
    df["score"] = alpha * df.sim + (1 - alpha) * df.decay if use_decay else df.sim
    df = df.sort_values("score", ascending=False)
    df = df.groupby("bucket").head(k_per_bucket) if use_buckets else df.head(k_total)
    df["date"] = pd.to_datetime(df.publish_time, unit="s").dt.date
    df = df.sort_values("publish_time").reset_index(drop=True)

    if use_asof:                                    # belt and braces
        assert (df.publish_time <= as_of_ts).all(), "LEAK: article after as_of"
    return df

def count_leaks(df, as_of):
    """Number of returned articles dated after as_of (used by the leakage test)."""
    return 0 if df.empty else int((df.publish_time > ts(as_of) + DAY - 1).sum())
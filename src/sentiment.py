"""FinBERT scoring (run on Colab, once) and the weekly sentiment tool (runs anywhere)."""
from src.config import *

# ---------- scoring ----------
def score_all(batch_size=64, part_size=10_000):
    """Score embed_text with FinBERT. Resumable: parts are saved to FINBERT_PARTS."""
    import torch
    from transformers import pipeline
    docs = pd.read_parquet(CLEANED, columns=["doc_id", "embed_text"])
    gpu = DEVICE == "cuda"
    clf = pipeline("text-classification", model="ProsusAI/finbert", top_k=None,
                   device=0 if gpu else -1,
                   torch_dtype=torch.float16 if gpu else torch.float32)
    FINBERT_PARTS.mkdir(parents=True, exist_ok=True)
    for k, s in enumerate(range(0, len(docs), part_size)):
        f = FINBERT_PARTS / f"part_{k:03d}.parquet"
        if f.exists():
            continue
        part = docs.iloc[s:s + part_size]
        res = clf(part.embed_text.tolist(), batch_size=batch_size,
                  truncation=True, max_length=512)
        rows = []
        for doc_id, r in zip(part.doc_id, res):
            r = [r] if isinstance(r, dict) else r
            p = {x["label"].lower(): x["score"] for x in r}
            assert {"positive", "negative", "neutral"} <= p.keys(), p.keys()
            rows.append({"doc_id": doc_id, "pos": p["positive"], "neg": p["negative"],
                         "neu": p["neutral"], "sentiment": p["positive"] - p["negative"]})
        pd.DataFrame(rows).to_parquet(f, index=False)
        print("saved part", k, flush=True)

def merge_parts():
    fb = pd.concat([pd.read_parquet(f) for f in sorted(FINBERT_PARTS.glob("part_*.parquet"))],
                   ignore_index=True)
    fb.to_parquet(FINBERT_RESULTS, index=False)
    print(fb.shape, "unique ids:", fb.doc_id.is_unique)
    print(fb.sentiment.describe())
    return fb

# ---------- tool ----------
def weekly_sentiment(docs, ticker, start, end, as_of, min_docs=5):
    """Weekly mean FinBERT sentiment over ALL articles for the ticker in the window.
    Never uses anything after as_of. `docs` must already have the 'sentiment' column."""
    lo = ts(start)
    hi = min(ts(end), ts(as_of)) + DAY - 1
    m = (docs.ticker == ticker) & (docs.publish_time >= lo) & (docs.publish_time <= hi)
    d = docs[m]
    if d.empty:
        return pd.DataFrame(columns=["week_start", "mean_sentiment", "n_docs", "low_confidence"])
    dt = pd.to_datetime(d.publish_time, unit="s", utc=True).dt.tz_localize(None)
    week = dt.dt.to_period("W-SUN").dt.start_time              # Monday of that week
    g = d.groupby(week).agg(mean_sentiment=("sentiment", "mean"), n_docs=("sentiment", "size"))
    g["low_confidence"] = g.n_docs < min_docs
    g.index.name = "week_start"
    return g.reset_index()
"""
build_data.py: stream FNSPID news for candidate tickers (no full download),
fetch prices, embed locally with BGE-small, and store everything on disk.

Install:
    pip install pandas pyarrow tqdm yfinance sentence-transformers chromadb

Run (in order, or just `all`):
    python build_data.py fetch     # stream news, save subset + coverage report
    python build_data.py prices    # yfinance prices, index, earnings dates
    python build_data.py embed     # embed subset into local Chroma (resumable)
    python build_data.py check     # sanity test of the as-of filter
    python build_data.py all       # fetch + prices + embed + check

Outputs (all under ./data):
    processed/news_subset.parquet, processed/coverage.csv,
    processed/prices.parquet, processed/earnings.parquet, chroma/
"""
import argparse
import hashlib
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ----------------------------- CONFIG ---------------------------------
# 15 candidate tickers. Edit freely; coverage.csv will show which are dense.
TICKERS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "TSLA", "GOOGL", "NFLX", "INTC",
    "AMD", "QCOM", "CSCO", "PEP", "COST", "JPM", "XOM",
]
START, END = "2018-01-01", "2022-12-31"      # 5-year news window (inclusive)
PRICE_START, PRICE_END = "2017-06-01", "2023-01-10"  # buffer for estimation window / CAR
INDEX = "^GSPC"
FINAL_TICKERS = None   # after reading coverage.csv, set e.g. ["AAPL", ...] (10 tickers)

NEWS_URL = ("https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/"
            "Stock_news/nasdaq_exteral_data.csv")
MAX_CHARS = 3000       # cap text stored per document
CHUNK_ROWS = 200_000
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
BATCH = 512
SEED = 42

DATA = Path("data")
PROC = DATA / "processed"
NEWS_PQ = PROC / "news_subset.parquet"
CHROMA_DIR = DATA / "chroma"
# ----------------------------------------------------------------------


def fetch_news():
    """Stream the remote CSV in chunks, keeping only matching rows."""
    PROC.mkdir(parents=True, exist_ok=True)
    wanted = {"Date", "Article_title", "Stock_symbol", "Url", "Publisher", "Article"}
    reader = pd.read_csv(
        NEWS_URL,
        usecols=lambda c: c in wanted,
        dtype=str,
        chunksize=CHUNK_ROWS,
        on_bad_lines="skip",
    )
    kept = []
    for i, ch in enumerate(tqdm(reader, desc="streaming chunks")):
        if i == 0:
            print("Columns seen:", list(ch.columns))
        ch = ch[ch["Stock_symbol"].isin(TICKERS)]
        d = ch["Date"].str[:10]
        ch = ch[(d >= START) & (d <= END)]
        if len(ch):
            kept.append(ch)
    if not kept:
        raise SystemExit("No rows matched. Check tickers, dates, and column names.")

    df = pd.concat(kept, ignore_index=True)
    ts = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    ok = ts.notna()
    df, ts = df[ok].copy(), ts[ok]

    title = df["Article_title"].fillna("").str.strip()
    body = (df["Article"] if "Article" in df else pd.Series("", index=df.index))
    body = body.fillna("").str.strip()
    has_body = body.str.len() > 0
    text = body.where(has_body, title)

    epoch = (ts - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().astype("int64")
    out = pd.DataFrame({
        "ticker": df["Stock_symbol"],
        "text": text.str.slice(0, MAX_CHARS),
        "title": title,
        "publish_time": epoch,
        "source_type": "news",
        "source_name": df["Publisher"].fillna("unknown"),
        "url": df["Url"] if "Url" in df else "",
        "has_body": has_body,
    })
    out = out[out["text"].str.len() > 0]
    out = out.drop_duplicates(subset=["ticker", "title", "publish_time"]).reset_index(drop=True)
    out["doc_id"] = [
        hashlib.sha1(f"{t}|{p}|{ti}".encode()).hexdigest()[:16]
        for t, p, ti in zip(out.ticker, out.publish_time, out.title)
    ]
    out.to_parquet(NEWS_PQ, index=False)
    print(f"Saved {len(out):,} docs to {NEWS_PQ}")
    print(f"Share of docs with a full article body: {out.has_body.mean():.1%} "
          f"(rest use the headline as text)")

    # weekly coverage report
    wk = pd.to_datetime(out.publish_time, unit="s", utc=True).dt.tz_localize(None).dt.to_period("W")
    weeks = pd.period_range(START, END, freq="W")
    counts = out.assign(week=wk).groupby(["ticker", "week"]).size().unstack(0)
    counts = counts.reindex(weeks).fillna(0)
    cov = pd.DataFrame({
        "total_docs": counts.sum().astype(int),
        "median_per_week": counts.median(),
        "pct_weeks_ge10": (counts >= 10).mean().round(3),
        "pct_weeks_zero": (counts == 0).mean().round(3),
    }).sort_values("pct_weeks_ge10", ascending=False)
    cov.to_csv(PROC / "coverage.csv")
    print("\nCoverage (best first). Pick your 10 tickers from this:\n")
    print(cov.to_string())


def fetch_prices():
    import yfinance as yf
    PROC.mkdir(parents=True, exist_ok=True)
    syms = TICKERS + [INDEX]
    px = yf.download(syms, start=PRICE_START, end=PRICE_END, auto_adjust=True,
                     group_by="ticker", progress=False)
    frames = []
    for s in syms:
        if s not in px.columns.get_level_values(0):
            print("No price data for", s)
            continue
        d = px[s].dropna(how="all").reset_index()
        d.columns = [str(c).lower() for c in d.columns]
        d["ticker"] = s
        frames.append(d)
    prices = pd.concat(frames, ignore_index=True)
    prices.to_parquet(PROC / "prices.parquet", index=False)
    print(f"Saved prices: {len(prices):,} rows for {prices.ticker.nunique()} symbols")

    # earnings dates (best effort; history may be short)
    rows = []
    for t in TICKERS:
        try:
            e = yf.Ticker(t).get_earnings_dates(limit=40)
            if e is not None and len(e):
                for dt in e.index:
                    rows.append({"ticker": t, "earnings_date": pd.Timestamp(dt).tz_convert("UTC")})
        except Exception as ex:  # noqa: BLE001
            print(f"Earnings dates failed for {t}: {ex}")
    if rows:
        er = pd.DataFrame(rows)
        er.to_parquet(PROC / "earnings.parquet", index=False)
        inwin = er[(er.earnings_date >= START) & (er.earnings_date <= END)]
        print("Earnings dates inside the news window per ticker:")
        print(inwin.groupby("ticker").size().to_string())
    else:
        print("No earnings dates retrieved; plan to use large-move events instead.")


def embed():
    import chromadb
    from sentence_transformers import SentenceTransformer
    import torch

    torch.manual_seed(SEED)
    df = pd.read_parquet(NEWS_PQ)
    if FINAL_TICKERS:
        df = df[df.ticker.isin(FINAL_TICKERS)]
    print(f"{len(df):,} docs to consider")

    model = SentenceTransformer(EMBED_MODEL)
    model.max_seq_length = 256
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col = client.get_or_create_collection("news", metadata={"hnsw:space": "cosine"})

    existing = set(col.get(include=[])["ids"])
    df = df[~df.doc_id.isin(existing)]
    print(f"{len(df):,} new docs to embed ({len(existing):,} already stored)")

    for i in tqdm(range(0, len(df), BATCH), desc="embedding"):
        b = df.iloc[i:i + BATCH]
        emb = model.encode(b.text.tolist(), batch_size=64,
                           normalize_embeddings=True, show_progress_bar=False)
        col.upsert(
            ids=b.doc_id.tolist(),
            embeddings=emb.tolist(),
            documents=b.text.tolist(),
            metadatas=[
                {"ticker": t, "publish_time": int(p), "source_type": s, "source_name": n}
                for t, p, s, n in zip(b.ticker, b.publish_time, b.source_type, b.source_name)
            ],
        )
    print("Collection size:", col.count())


def check():
    """Sanity test: the as-of filter inside the DB query must exclude later docs."""
    import chromadb
    from sentence_transformers import SentenceTransformer

    col = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection("news")
    model = SentenceTransformer(EMBED_MODEL)
    as_of = int(pd.Timestamp("2020-06-30", tz="UTC").timestamp())
    q = model.encode(["quarterly earnings and guidance"], normalize_embeddings=True).tolist()
    r = col.query(query_embeddings=q, n_results=50,
                  where={"publish_time": {"$lte": as_of}})
    times = [m["publish_time"] for m in r["metadatas"][0]]
    assert times and max(times) <= as_of, "As-of filter leaked a future document!"
    print(f"OK: {len(times)} results, latest publish_time <= as_of")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["fetch", "prices", "embed", "check", "all"])
    step = ap.parse_args().step
    if step in ("fetch", "all"):
        fetch_news()
    if step in ("prices", "all"):
        fetch_prices()
    if step in ("embed", "all"):
        embed()
    if step in ("check", "all"):
        check()
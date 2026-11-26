"""combined.parquet -> top-N tickers -> cleaned docs table.
Run from the project root:  python -m src.preprocess
"""
from src.config import *

DROP_COLS = ["Unnamed: 0", "dt", "Author", "Publisher", "Lsa_summary", "Luhn_summary"]

def select_tickers(full):
    ranked = [t for t in full["Stock_symbol"].value_counts().index if t not in EXCLUDE_TICKERS]
    return ranked[:N_TICKERS]

def build_docs(full):
    keep = select_tickers(full)
    d = full[full["Stock_symbol"].isin(keep)].copy()
    d = d.drop(columns=DROP_COLS, errors="ignore")

    ts_ = pd.to_datetime(d["Date"], utc=True, errors="coerce")
    x, t = d[ts_.notna()], ts_[ts_.notna()]

    title = x["Article_title"].fillna("").str.strip()
    art = x["Article"].fillna("")
    lex = x["Lexrank_summary"].fillna("").str.strip()
    pt = ((t - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()).astype("int64")

    is_long = (title + ". " + art).str.len() > LONG_CHARS
    use_sum = is_long & (lex.str.len() > 0)          # long AND a summary exists

    docs = pd.DataFrame({
        "ticker": x["Stock_symbol"],
        "title": title,
        "text": (title + ". " + art).str.slice(0, TEXT_CAP),
        "embed_text": (title + ". " + lex).where(use_sum, title + ". " + art),
        "embed_source": np.where(use_sum, "summary", "article"),
        "publish_time": pt,
        "date_only": t.dt.strftime("%H:%M:%S") == "00:00:00",
        "source_type": "news",
        "source_name": "nasdaq",
        "url": x["Url"],
        "textrank_summary": x["Textrank_summary"],
        "lexrank_summary": x["Lexrank_summary"],
    })
    docs = docs[docs["text"].str.len() > MIN_TEXT]
    docs = docs.drop_duplicates(subset=["ticker", "title", "publish_time"]).reset_index(drop=True)
    docs["doc_id"] = [hashlib.sha1(f"{a}|{b}|{c}".encode()).hexdigest()[:16]
                      for a, b, c in zip(docs.ticker, docs.publish_time, docs.title)]
    return docs, keep

def main(overwrite=False):
    # Row order ties each embedding shard to its article. Don't regenerate casually.
    if CLEANED.exists() and not overwrite:
        raise SystemExit(f"{CLEANED} exists. Pass overwrite=True only if you will also "
                         f"delete and rebuild emb_shards, finbert_parts and chroma_full.")
    full = pd.read_parquet(COMBINED)
    docs, keep = build_docs(full)
    CLEANED.parent.mkdir(parents=True, exist_ok=True)
    docs.to_parquet(CLEANED, index=False)
    print(docs.shape, docs.embed_source.value_counts().to_dict())
    print(keep)

if __name__ == "__main__":
    main()
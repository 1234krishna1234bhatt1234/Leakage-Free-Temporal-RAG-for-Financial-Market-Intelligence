"""One call to load everything after a notebook restart. Does NOT embed or rebuild."""
from src.config import *
from types import SimpleNamespace

DOC_COLS = ["doc_id", "ticker", "title", "publish_time", "embed_source", "date_only", "url"]

def load_all(device=None):
    import chromadb
    from sentence_transformers import SentenceTransformer

    docs = pd.read_parquet(CLEANED, columns=DOC_COLS)
    fb = pd.read_parquet(FINBERT_RESULTS)
    docs = docs.merge(fb, on="doc_id", how="left", validate="one_to_one")
    assert docs.sentiment.notna().all(), "some articles have no FinBERT score"
    sent = docs.set_index("doc_id")[["pos", "neg", "neu", "sentiment"]]

    prices = pd.read_parquet(PRICES)
    col = chromadb.PersistentClient(path=str(CHROMA_PATH)).get_collection("news")
    assert col.count() == len(docs), f"chroma {col.count()} != docs {len(docs)}: rebuild it"

    model = SentenceTransformer(EMBED_MODEL, device=device or DEVICE)
    model.max_seq_length = MAX_SEQ

    tickers = sorted(docs.ticker.unique())
    print(f"docs {docs.shape} | prices {prices.shape} | chroma {col.count()} | {len(tickers)} tickers")
    return SimpleNamespace(docs=docs, sent=sent, prices=prices, col=col, model=model, tickers=tickers)
"""Embed articles with BGE-small in shards (saved to disk), load into Chroma,
and rebuild Chroma from the shards on any machine.
Colab:  from src.embed import embed_all; embed_all()
Local:  from src.embed import rebuild_chroma_from_shards; rebuild_chroma_from_shards()
"""
from src.config import *

META_COLS = ["doc_id", "ticker", "title", "text", "publish_time", "url",
             "embed_source", "lexrank_summary"]

def get_collection(create=True):
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    if create:
        return client.get_or_create_collection("news", metadata={"hnsw:space": "cosine"})
    return client.get_collection("news")

def add_to_chroma(col, p, e, batch=512):
    for i in range(0, len(p), batch):
        b, eb = p.iloc[i:i + batch], e[i:i + batch]
        col.upsert(
            ids=b.doc_id.tolist(), embeddings=eb.tolist(), documents=b.text.tolist(),
            metadatas=[{"ticker": r.ticker, "publish_time": int(r.publish_time),
                        "title": r.title, "url": r.url or "",
                        "embed_source": r.embed_source,
                        "lexrank_summary": r.lexrank_summary or ""}
                       for r in b.itertuples()])

def embed_all(batch_size=128):
    """Resumable: finished shards are reused, finished Chroma rows are skipped."""
    from sentence_transformers import SentenceTransformer
    docs = pd.read_parquet(CLEANED).reset_index(drop=True)
    model = SentenceTransformer(EMBED_MODEL, device=DEVICE)
    model.max_seq_length = MAX_SEQ
    SHARDS_DIR.mkdir(parents=True, exist_ok=True)
    col = get_collection()
    done = set(col.get(include=[])["ids"])

    for k, s in enumerate(range(0, len(docs), SHARD_SIZE)):
        part = docs.iloc[s:s + SHARD_SIZE]
        ids = part.doc_id.tolist()
        f = SHARDS_DIR / f"shard_{k:03d}.npz"
        if f.exists():
            z = np.load(f, allow_pickle=True)            # our own file
            assert [str(i) for i in z["ids"]] == ids, "docs order changed, delete the shards"
            emb = z["emb"]
        else:
            emb = model.encode(part.embed_text.tolist(), batch_size=batch_size,
                               normalize_embeddings=True, show_progress_bar=False).astype("float32")
            np.savez(f, ids=np.array(ids, dtype="U16"), emb=emb)
        new = ~part.doc_id.isin(done).values
        add_to_chroma(col, part[new], emb[new])
        done.update(part.doc_id[new])
        print(f"shard {k} done | chroma count {col.count()}", flush=True)
    return col

def rebuild_chroma_from_shards():
    """Delete CHROMA_PATH and rebuild it from emb_shards. No re-embedding."""
    shutil.rmtree(CHROMA_PATH, ignore_errors=True)
    meta = pd.read_parquet(CLEANED, columns=META_COLS).set_index("doc_id")
    col = get_collection()
    for f in sorted(SHARDS_DIR.glob("shard_*.npz")):
        z = np.load(f, allow_pickle=True)
        ids = [str(i) for i in z["ids"]]
        add_to_chroma(col, meta.loc[ids].reset_index(), z["emb"])
        print(f.name, "| count", col.count(), flush=True)
    print(col.count(), len(meta))                         # should match
    return col
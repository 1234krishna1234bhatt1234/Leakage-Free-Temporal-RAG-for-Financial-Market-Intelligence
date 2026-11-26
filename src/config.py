"""Central config: shared imports, paths, constants.
Other modules do:  from src.config import *
"""
import os, re, sys, math, random, glob, hashlib, shutil, importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

try:                                   # reads OPENROUTER_API_KEY from .env if present
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------- environment ----------
IN_COLAB = importlib.util.find_spec("google.colab") is not None
USE_DRIVE = IN_COLAB                   # override by hand if needed

if USE_DRIVE:                          # on Colab, mount Drive in the notebook first
    BASE = Path("/content/drive/MyDrive/ie624_project/data")
    CHROMA_PATH = Path("/content/chroma_full")      # keep Chroma on Colab's local disk
    DEVICE = "cuda"
else:
    BASE = Path("data")
    CHROMA_PATH = BASE / "chroma_full"
    DEVICE = "cpu"

# ---------- paths ----------
RAW_DIR = BASE / "raw"
COMBINED = RAW_DIR / "combined.parquet"
CLEANED = RAW_DIR / "cleaned_data.parquet"
SHARDS_DIR = BASE / "emb_shards"
FINBERT_PARTS = BASE / "finbert_parts"
FINBERT_RESULTS = BASE / "finbert_results.parquet"
PRICES = BASE / "prices.parquet"

# ---------- data selection ----------
N_TICKERS = 30
EXCLUDE_TICKERS = ("EA", "WBA")        # delisted, no prices
LONG_CHARS = 5000                      # above this, embed title + lexrank summary
TEXT_CAP = 3000                        # chars of article text stored as the document
MIN_TEXT = 40
INDEX_SYMBOL = "^GSPC"
PRICE_START, PRICE_END = "2008-06-01", "2025-01-15"

# ---------- models ----------
SEED = 42
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
MAX_SEQ = 512
SHARD_SIZE = 10_000

# ---------- retrieval defaults (tune on a dev set, then freeze) ----------
DAY = 86400
ALPHA = 0.7
HALF_LIFE = 30
K_PER_BUCKET = 3
N_CANDIDATES = 10
K_TOTAL = 15

def ts(d):
    """Date string -> UTC epoch seconds (start of that day)."""
    return int(pd.Timestamp(d, tz="UTC").timestamp())

random.seed(SEED)
np.random.seed(SEED)
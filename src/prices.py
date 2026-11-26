"""Daily prices (yfinance), returns, and history helpers for the event study."""
from src.config import *

def download_prices(tickers=None):
    import yfinance as yf
    if tickers is None:
        tickers = sorted(pd.read_parquet(CLEANED, columns=["ticker"]).ticker.unique())
    px = yf.download(list(tickers) + [INDEX_SYMBOL], start=PRICE_START, end=PRICE_END,
                     auto_adjust=True, progress=False, threads=False)["Close"]
    PRICES.parent.mkdir(parents=True, exist_ok=True)
    px.to_parquet(PRICES)
    print(px.shape)
    print(px.isna().sum().sort_values().tail(5))
    return px

def load_prices():
    return pd.read_parquet(PRICES)

def daily_returns(px):
    return px.pct_change()          # NaNs stay NaN (before listing); never fill them

def first_valid_dates(px):
    return px.apply(lambda c: c.first_valid_index())

def history_days(px, ticker, date):
    """Trading days of price history for `ticker` strictly before `date`."""
    s = px[ticker].dropna()
    return int((s.index < pd.Timestamp(date)).sum())
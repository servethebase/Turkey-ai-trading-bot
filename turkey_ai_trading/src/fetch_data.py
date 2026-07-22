import yfinance as yf
import pandas as pd
from pathlib import Path

# data/ папка находится рядом с main.py (на уровень выше src/)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def download(ticker: str, start: str = "2015-01-01", end: str = "2025-12-31") -> pd.DataFrame:
    print(f"Downloading {ticker} ...")
    df = yf.download(ticker, start=start, end=end, progress=False)
    df = df.dropna()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_DIR / f"{ticker.replace('.', '_')}.csv"
    df.to_csv(csv_path)
    print(f"Saved {len(df)} rows to {csv_path}")
    return df
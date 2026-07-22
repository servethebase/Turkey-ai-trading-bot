import pandas as pd
import numpy as np


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    # Убираем MultiIndex если yfinance вернул его
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index(drop=True)

    # Скользящие средние
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["SMA_30"] = df["Close"].rolling(window=30).mean()

    # Экспоненциальная скользящая средняя
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()

    # RSI (14 дней)
    delta = df["Close"].diff()
    up    = delta.clip(lower=0)
    down  = -delta.clip(upper=0)
    roll_up   = up.ewm(com=13, adjust=False).mean()
    roll_down = down.ewm(com=13, adjust=False).mean()
    rs = roll_up / roll_down
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # Полосы Боллинджера (ширина)
    bb_mid        = df["Close"].rolling(window=20).mean()
    bb_std        = df["Close"].rolling(window=20).std()
    df["BB_Width"] = (bb_mid + 2 * bb_std) - (bb_mid - 2 * bb_std)

    # Волатильность (абсолютное % изменение)
    df["Volatility"] = df["Close"].pct_change().abs()

    # Цель: 1 если завтра цена выше, 0 если ниже
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    df = df.dropna().reset_index(drop=True)
    return df
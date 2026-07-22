import pandas as pd
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def simulate(df_test: pd.DataFrame, predictions) -> pd.DataFrame:
    df = df_test.copy().reset_index(drop=True)

    # Убираем MultiIndex если есть
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df["Pred"] = predictions

    # Close должен быть одномерным
    close = df["Close"].squeeze()
    next_day_return = close.pct_change().shift(-1)

    # Стратегия: если ИИ говорит UP — покупаем, иначе — сидим в деньгах
    df["Strategy_Return"] = np.where(df["Pred"] == 1, next_day_return, 0.0)
    df["BuyHold_Return"]  = next_day_return

    # Накопленная доходность
    df["Cum_Strategy"] = (1 + df["Strategy_Return"].fillna(0)).cumprod()
    df["Cum_BuyHold"]  = (1 + df["BuyHold_Return"].fillna(0)).cumprod()

    # Сохраняем результаты
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / "backtest_results.csv", index=False)

    final_ai  = df["Cum_Strategy"].iloc[-2]
    final_bnh = df["Cum_BuyHold"].iloc[-2]
    print(f"\n  AI Strategy return  : {(final_ai  - 1)*100:.1f}%")
    print(f"  Buy-and-Hold return : {(final_bnh - 1)*100:.1f}%")

    return df
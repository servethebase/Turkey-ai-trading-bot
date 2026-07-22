
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.fetch_data import download
from src.features   import add_technical_features
from src.model      import train
from src.backtest   import simulate
from src.plot       import plot_equity, plot_confusion, plot_feature_importance


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TICKER = "THYAO.IS"
raw_df = download(TICKER)


df = add_technical_features(raw_df)
print(f"\nDataset shape: {df.shape}")


model, X_test, y_test, y_pred = train(df)

n_test      = len(X_test)
df_test_raw = df.iloc[-n_test:].reset_index(drop=True)
backtest_df = simulate(df_test_raw, y_pred)



print("\nGenerating charts ...")
plot_equity(backtest_df, ticker=TICKER)
plot_confusion(y_test, y_pred)
plot_feature_importance(OUTPUT_DIR / "gbrt_model.pkl")
print("\nDone! Check the output/ folder for your charts.")
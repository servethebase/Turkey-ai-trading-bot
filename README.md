# AI-Driven High-Volatility Forecasts for Turkish Equity Markets

An end-to-end Machine Learning pipeline predicting short-term price movements of Turkish Airlines (THYAO.IS) listed on the Borsa Istanbul (BIST).

##  Key Highlights & Features
* **Data Pipeline:** Automated historical OHLCV data ingestion via Yahoo Finance API (`yfinance`).
* **Feature Engineering:** Relative Strength Index (RSI), Simple Moving Averages (SMA), Exponential Moving Averages (EMA), and Bollinger Band Volatility metrics.
* **Model:** Gradient Boosting Classifier trained on temporal historical splits to prevent data leakage.
* **Backtesting:** Automated evaluation engine generating equity curves, confusion matrices, and feature importance analyses.

##  Performance Chart
![Cumulative Returns](output/cumulative_returns.png)

##  Project Structure
* `main.py` - Core execution script running the full end-to-end pipeline.
* `src/` - Modular source code (`fetch_data.py`, `features.py`, `model.py`, `backtest.py`, `plot.py`).
* `data/` - Cached historical CSV market data.
* `output/` - Output charts and backtest CSV results.

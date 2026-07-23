# AI-Driven High-Volatility Forecasts for Turkish Equity Markets

An end-to-end Machine Learning pipeline predicting short-term price movements of Turkish Airlines (THYAO.IS) on the Borsa Istanbul (BIST) using Walk-Forward Validation and ensemble methods.

Key Highlights & Academic Insights:
  Model: Random Forest Classifier trained on advanced technical indicators (RSI, MACD, Stochastic Oscillator, ATR).
  Validation: Robust Walk-Forward Validation (rolling window of 1000 days train / 250 days test) to prevent overfitting.
  Realistic Constraints: Incorporates 0.05% transaction fees per trade.
  The Insight: During hyperinflationary environments, passive asset holding (Buy & Hold) significantly outperforms daily trading algorithms due to friction costs and strong macroeconomic momentum.

Performance Chart
The performance of the model compared to the Buy & Hold strategy:
![Strategy Performance](plots/strategy_performance.png)

Project Structure:
  `advanced_bot.py` - Core execution script (data fetch, feature engineering, training, backtesting).
  `data/` - Cached historical CSV data.
  `plots/` - Output performance charts.

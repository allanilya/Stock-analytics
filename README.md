# Stock Price Forecasting: Classical vs Deep Learning Models

**AI in Finance Capstone Project**
A comprehensive comparison of ARIMA, LSTM, and GRU models for stock price prediction across three distinct market sectors.

---

## 📊 Project Overview

This project implements and compares three forecasting approaches for stock price prediction:
- **ARIMA(3,1,5)** - Classical time series model
- **LSTM** - Long Short-Term Memory neural network
- **GRU** - Gated Recurrent Unit neural network

The analysis covers three stocks with different market characteristics:
- 🍎 **AAPL (Apple)** - Large-cap technology, stable growth
- 🎮 **NVDA (NVIDIA)** - High-volatility semiconductor/AI
- 🚗 **LYFT** - Small-cap ride-sharing, erratic patterns

**Time Period:** 2020-01-01 to Present (~5 years)
**Data Source:** Yahoo Finance
**Training/Test Split:** 80/20

---

## 🎯 Key Findings

### Model Performance Summary

| Stock | Best Model | RMSE Improvement vs ARIMA | Directional Accuracy |
|-------|-----------|---------------------------|---------------------|
| **AAPL** | GRU | 64% reduction | ~52% |
| **NVDA** | LSTM | 80% reduction | ~51% |
| **LYFT** | GRU | 80% reduction | ~53% |

### Main Conclusions

1. **Neural networks significantly outperform ARIMA** for all stocks studied (64-82% RMSE reduction)
2. **GRU is the practical winner** - best balance of accuracy, training speed, and simplicity
3. **LSTM excels for high-volatility stocks** - superior long-term memory for explosive growth patterns (NVDA)
4. **ARIMA remains valuable** for baseline comparison and interpretable forecasts
5. **Directional accuracy ~51%** - even sophisticated models struggle to beat random chance (50%)

---

## 🗂️ Project Structure

```
Stock-analytics/
├── main.ipynb                              # Main analysis notebook
└── README.md                               # This file
```

---

## 📚 Milestone Breakdown

### Milestone 1: Data Acquisition & Classical Models (Steps 1-7)

**Completed Tasks:**
- ✅ Data extraction from Yahoo Finance API
- ✅ Exploratory data analysis with interactive visualizations
- ✅ Time series decomposition (trend, seasonality, residuals)
- ✅ Stationarity testing (ADF test)
- ✅ ACF/PACF analysis
- ✅ Feature engineering (EMA, DEMA)
- ✅ Classical model implementation:
  - MA(20) - Moving Average
  - AR(20) - Autoregressive
  - ARIMA(3,1,5) - Grid search optimal model

**Key Results:**
- ARIMA(3,1,5) selected by grid search on NVDA (AIC = 4378.65)
- Stationarity achieved with first-order differencing (d=1)
- Applied consistently across all three stocks

---

### Milestone 2: Deep Learning Models (Steps 8-9)

**Completed Tasks:**
- ✅ Data normalization (MinMaxScaler 0-1 range)
- ✅ Sequence generation (60-day lookback window)
- ✅ LSTM implementation:
  - 2-layer architecture (50 units each)
  - Dropout regularization (0.2)
  - Early stopping (patience=5)
- ✅ GRU implementation:
  - 2-layer architecture (50 units each)
  - 25% fewer parameters than LSTM
  - Faster training convergence
- ✅ Training/validation split (90/10 within training set)
- ✅ Convergence behavior analysis

**Architecture Details:**
```
Input: 60 days × 5 features (Open, High, Low, Close, Volume)
Layer 1: LSTM/GRU (50 units, return_sequences=True)
Dropout: 0.2
Layer 2: LSTM/GRU (50 units)
Dropout: 0.2
Dense: 25 units
Output: 1 unit (next day's closing price)
```

---

### Milestone 3: Model Evaluation & Comparison (Steps 10-11)

**Completed Tasks:**
- ✅ Comprehensive metrics calculation:
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)
  - MAPE (Mean Absolute Percentage Error)
  - DA (Directional Accuracy)
- ✅ Side-by-side model comparison (3 stocks × 3 models = 9 combinations)
- ✅ Prediction visualizations with interactive Plotly charts
- ✅ Detailed interpretation of when LSTM/GRU outperform ARIMA
- ✅ Practical recommendations for different use cases

**Performance Breakdown:**

**AAPL (Apple):**
- ARIMA(3,1,5): RMSE $24.29
- LSTM: RMSE $8.89
- **GRU: RMSE $8.75** ✨ Winner

**NVDA (NVIDIA):**
- ARIMA(3,1,5): RMSE $42.10
- **LSTM: RMSE $8.56** ✨ Winner
- GRU: RMSE $9.84

**LYFT:**
- ARIMA(3,1,5): RMSE $4.53
- LSTM: RMSE $1.11
- **GRU: RMSE $0.92** ✨ Winner

---

## 🛠️ Technologies Used

**Python Libraries:**
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `yfinance` - Yahoo Finance API wrapper
- `statsmodels` - ARIMA, ACF/PACF, ADF test
- `tensorflow/keras` - LSTM/GRU implementation
- `scikit-learn` - Preprocessing, metrics
- `plotly` - Interactive visualizations

**Environment:**
- Python 3.8+
- TensorFlow 2.x
- Jupyter Notebook

---

## 🚀 Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/allanilya/Stock-analytics.git
cd Stock-analytics

# Install dependencies
pip install pandas numpy yfinance statsmodels tensorflow scikit-learn plotly

# Launch Jupyter Notebook
jupyter notebook main.ipynb
```

### Running the Analysis

1. Open `main.ipynb` in Jupyter Notebook
2. Run all cells sequentially (Kernel → Restart & Run All)
3. Expected runtime: ~10-15 minutes
   - Data extraction: ~2 min
   - Classical models: ~2 min
   - Neural network training: ~5-8 min
   - Visualization: ~2 min

### Requirements

```
pandas>=1.3.0
numpy>=1.21.0
yfinance>=0.1.63
statsmodels>=0.13.0
tensorflow>=2.8.0
scikit-learn>=1.0.0
plotly>=5.3.0
matplotlib>=3.4.0
```

---

## 📈 When to Use Each Model

### Use LSTM When:
- ✅ Stock has high volatility (e.g., NVDA, tech growth stocks)
- ✅ Long-term dependencies matter (product cycles, earnings patterns)
- ✅ Computational cost is not a constraint
- ✅ You need the most accurate predictions

### Use GRU When:
- ✅ Default choice for most applications
- ✅ Training speed matters (production systems)
- ✅ Stable or moderately volatile stocks (e.g., AAPL, LYFT)
- ✅ You want best accuracy-to-complexity ratio

### Use ARIMA When:
- ✅ Interpretability is critical (regulatory requirements)
- ✅ Limited data available (<1000 samples)
- ✅ Quick baseline needed
- ✅ Stock follows linear patterns (index funds, utilities)

---

## 🔍 Limitations & Future Work

### Current Limitations:
1. **Univariate models** - Only uses past prices, ignores volume/sentiment
2. **Single architecture** - Did not test 1-layer vs 3-layer networks
3. **One-day-ahead only** - Multi-day forecasts may favor different models
4. **Train/test split** - Walk-forward validation would be more rigorous
5. **Directional accuracy ~51%** - Barely beats random (50%)

### Future Improvements:
1. **Multivariate models** - Add volume, RSI, MACD, sentiment scores
2. **Attention mechanisms** - Transformer models for sequence-to-sequence
3. **Hybrid models** - ARIMA for trend + LSTM for residuals
4. **Architecture search** - Grid search over layers, units, sequence length
5. **Walk-forward validation** - Rolling window retraining
6. **Risk metrics** - Sharpe ratio, maximum drawdown, VaR

---

## 📊 Sample Visualizations

The notebook includes:
- 📈 Price trends with exponential moving averages
- 🔄 Time series decomposition (trend/seasonality/residuals)
- 📊 ACF/PACF plots for model selection
- 🎯 Forecast vs actual comparison charts
- 📉 Training history (loss curves)
- 🏆 Model performance comparison tables

---

## 🎓 Academic Context

This project fulfills the requirements for a three-milestone capstone in AI for Finance:

- **Milestone 1:** Classical time series analysis with ARIMA
- **Milestone 2:** Deep learning implementation with LSTM/GRU
- **Milestone 3:** Comprehensive model evaluation and comparison

The analysis demonstrates that while neural networks significantly outperform classical models in terms of RMSE, the near-random directional accuracy (~51%) aligns with the semi-strong efficient market hypothesis - short-term stock prices are fundamentally difficult to predict.

---

## 📝 References

1. **ARIMA Modeling:** Box, G. E. P., & Jenkins, G. M. (1976). Time Series Analysis: Forecasting and Control
2. **LSTM Networks:** Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory
3. **GRU Networks:** Cho, K., et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder
4. **Financial Time Series:** Tsay, R. S. (2005). Analysis of Financial Time Series

---

## 👥 Authors

**Allan Ilyasov**
**Giulio Bardelli**
**Peter Roumeliotis**

---

## 📄 License

This project is for educational purposes as part of an academic capstone.

---

## 🙏 Acknowledgments

- Professor's reference notebook: `04-05 ARIMA_with_AlphaVantage_Vintage_API.ipynb`
- Yahoo Finance for providing free historical stock data
- TensorFlow/Keras teams for deep learning framework
- Statsmodels for classical time series tools

---

**Last Updated:** 2025-11-23
**Status:** ✅ Complete - All three milestones fulfilled

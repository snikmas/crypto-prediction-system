# Crypto Price Spike Detection System

ML-based early warning system for cryptocurrency price volatility and spikes.

## Problem Statement
Predict high-volatility periods and extreme price spikes in cryptocurrency markets to enable proactive risk management.

## Data
- Source: CryptoCompare API
- Coins: BTC, ETH, USDT, BNB, DOGE, ADA, SOL
- Timeframe: Hourly OHLCV data (66,682 samples after preprocessing)
- Features: 20 technical indicators (volatility, momentum, volume, time-based)

## Models

### Volatility Predictor
- Algorithm: XGBoost with class balancing
- Target: Top 25% volatile hours
- **Performance (Test Set, Threshold 0.5):**
  - Recall: 35%
  - Precision: 38%
  - F1: 36%

### Spike Detector
- Algorithm: XGBoost with class balancing  
- Target: Top 10% extreme price moves
- **Performance (Test Set, Threshold 0.3):**
  - Recall: 67%
  - Precision: 17%
  - F1: 27%

## Key Insights
- Threshold 0.3 chosen for spike detection due to asymmetric costs (missing a spike is costlier than false alarm)
- Visualization shows clear recall-precision trade-off
- Low precision (XX%) indicates need for additional features (news sentiment, order flow)

## Limitations
- Technical indicators alone insufficient for predicting news-driven spikes
- High false positive rate (XX%)
- Not recommended for standalone automated trading

## Future Improvements
- Add news sentiment analysis
- Incorporate order book depth data
- Ensemble with additional models
- Real-time deployment with monitoring

## Project Structure
```
crypto-prediction-system/
├── data/
│   └── processed/          # Feature-engineered datasets
├── models/                 # Trained models + config
├── src/
│   ├── data/               # Data collection & processing
│   ├── models/             # Training & evaluation
│   └── utils/              # Utility functions
└── threshold_analysis_*.png  # Optimization charts
```

## Usage
```bash
# Train models
python -m src.models.train

# Models saved to models/ directory
# Load with: joblib.load('models/spike_xgb.pkl')
```

## Requirements
- Python 3.10+
- XGBoost, pandas, scikit-learn, matplotlib
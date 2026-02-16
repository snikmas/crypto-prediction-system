# Crypto Price Spike Detection System

ML-based early warning system for cryptocurrency price volatility and spikes. Uses hourly OHLCV data to predict high-volatility periods and extreme price movements.

## Setup

```bash
python -m venv myvenv
source myvenv/bin/activate
pip install -r requirements.txt
```

Copy the example env file and fill in your credentials:
```bash
cp config/.env.example config/.env
```

You need a PostgreSQL database and a [CoinDesk API key](https://developers.coindesk.com).

## Usage

```bash
# 1. Create the database table (run once)
python -m src.data.setup_db

# 2. Fetch hourly OHLCV data from CoinDesk and store in PostgreSQL
python -m src.data.database

# 3. Generate features from DB data (saves to data/processed/)
python -m src.models.features

# 4. Train models (saves to models/)
python -m src.models.train
```

## Data

- **Source:** CoinDesk API (hourly OHLCV)
- **Coins:** BTC, ETH, USDT, BNB, DOGE, ADA, SOL
- **Samples:** ~66,000 after preprocessing
- **Features:** 20 technical indicators — returns (1h, 24h, 7d, log), volatility (24h, 7d, 30d + lag), volume ratio, MA(50) distance, cyclical hour encoding

## Models

Both models use XGBoost with `scale_pos_weight` for class imbalance. Data is split chronologically (60/20/20 train/val/test) to avoid time-series leakage.

### Volatility Regime Predictor
- **Target:** Next hour in top 25% volatile hours (rolling 7-day window)
- **Threshold:** 0.5
- **Test performance:** Recall 35%, Precision 38%, F1 36%

### Spike Detector
- **Target:** Next hour in top 10% extreme price moves (rolling 30-day window)
- **Threshold:** 0.3 (tuned for recall — missing a spike is costlier than a false alarm)
- **Test performance:** Recall 67%, Precision 17%, F1 27%

## Limitations

- Technical indicators alone are insufficient for predicting news-driven spikes
- High false positive rate on spike detection due to low threshold
- Not suitable for standalone automated trading

## Future Work

- News sentiment analysis features
- Order book depth data
- Model ensembling
- Real-time deployment with monitoring

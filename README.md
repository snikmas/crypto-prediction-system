# Crypto Volatility Prediction System

A machine learning system that predicts cryptocurrency volatility spikes and high-risk trading periods 1 hour in advance for major coins (BTC, ETH, SOL, etc.).

## Problem

Crypto traders and risk managers need advance warning of volatility spikes to:
- Adjust position sizes before turbulent periods  
- Avoid liquidations during sudden price swings  
- Time entries/exits for volatile assets  

Current solutions rely on lagging indicators. This system predicts volatility regime changes 1 hour ahead using 15+ engineered features (RSI, Bollinger squeeze, Parkinson volatility, etc.).

## Current Status

**What's working:**
- [X] Hourly data fetching from CoinDesk API  
- [X] PostgreSQL storage with TimescaleDB for time-series data  
- [X] Feature engineering pipeline (15 technical indicators)  
- [X] Dual prediction targets (volatility regime + spike detection)  

**In progress:**
- [ ] Train Random Forest classifier  
- [ ] Save processed features to Parquet  
- [ ] Build prediction pipeline  

## Tech Stack

**Data Pipeline:**
- Python 3.10, pandas, numpy  
- PostgreSQL (TimescaleDB) for time-series storage  
- psycopg2 for database connections  
- APScheduler for automated data collection  
- CoinDesk API for market data  

**Machine Learning:**
- scikit-learn (Random Forest, evaluation metrics)  
- (Planned: model monitoring, drift detection)  

**Infrastructure:**
- PostgreSQL (TimescaleDB)  
- (Planned: FastAPI for REST API, Docker for deployment)  

## Project Structure
```
crypto-prediction-system/
├── src/
│   ├── data/
│   │   ├── constants.py      # Coin list, API endpoints
│   │   ├── database.py        # DB connections, data insertion
│   │   ├── fetchers.py        # API data fetching logic
│   │   └── setup_db.py        # Database schema creation
│   ├── models/
│   │   ├── features.py        # Feature engineering pipeline
│   │   └── train.py           # (In progress) Model training
│   └── utils/
│       └── utils.py           # Helper functions
├── config/
│   └── .env                   # API keys, DB credentials
└── data/
    └── processed/             # (In progress) Feature parquet files
```

## How It Works

1. **Data Collection**: Fetches hourly OHLCV data from CoinDesk API for 7 cryptocurrencies  
2. **Storage**: Stores in PostgreSQL with upsert logic (replaces duplicates)  
3. **Feature Engineering**: Calculates 15+ technical indicators:
   - Momentum: 1h/24h/7d returns  
   - Volatility: Rolling std, Parkinson estimator, ATR  
   - Volume: Volume ratio vs 24h average  
   - Technical: RSI-14, Bollinger squeeze, MA-50 distance  
   - Temporal: Hour-of-day cyclical encoding (sin/cos)  
4. **Targets**:
   - **Volatility Regime**: Binary (high/normal volatility next hour)  
   - **Spike Detection**: Top 10% extreme price movements  
5. **Training**: (In progress) Random Forest on 1960 samples (280 per coin after 720h warmup)  

## Setup

**Prerequisites:**
- Python 3.10+  
- PostgreSQL 14+ with TimescaleDB  
- CoinDesk API key  

**Installation:**
```bash
git clone https://github.com/snikmas/crypto-prediction-system
cd crypto-prediction-system

python -m venv myvenv
source myvenv/bin/activate

pip install -r requirements.txt

# Setup database
python -m src.data.setup_db

# Fetch initial data (runs once at 1 AM daily via scheduler)
python -m src.data.database
```

**Note:** Main execution script in development. Run modules individually for now.

## Roadmap

- [X] Data pipeline (API → PostgreSQL)  
- [X] Feature engineering (15 indicators)  
- [ ] Train ML model (Random Forest)  
- [ ] Model evaluation & hyperparameter tuning  
- [ ] Save model artifacts & feature store  
- [ ] REST API for predictions (FastAPI)  
- [ ] Model monitoring & drift detection  
- [ ] Automated retraining pipeline  
- [ ] Docker deployment  

## Learning Goals

Portfolio project for backend/ML engineering internship applications. Focus areas:
- End-to-end ML pipeline (data → features → model → deployment)  
- Production data patterns (database design, ETL, feature engineering)  
- Time-series forecasting & volatility modeling  
- Clean Python architecture & testing  

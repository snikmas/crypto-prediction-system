from fetchers import fetch_hourly_data
import pandas as pd
import logging
import numpy as np
import datetime
from database import create_connection
import math

# macd, skeweness

# 1. plan: get data from the db 
# 2. create dataframe and process data
# for now, main feautres:
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def load_data_from_db():
    connection = create_connection()
    if not connection:
        logging.error("Failed to connect to database")
        return None
    
    query = "SELECT * FROM hourly_ohlcv;"
    try:
        df = pd.read_sql_query(query, connection)
        logging.info("Data loaded successfully from database")
        return df
    except Exception as e:
        logging.error(f"Error loading data from database: {e}")
        return None
    finally:
        connection.close()

def calculate_features_for_symbol(symbol_data):
            
    data = pd.DataFrame()
    data["symbol"] = symbol_data["symbol"]
    data["timestamp"] = symbol_data["timestamp"]


    # =================== RETURNS CENTER ========================    
    data["return_1h"] = symbol_data["close"].pct_change()
    data["return_24h"] = symbol_data["close"].pct_change(periods=24)
    data["return_7d"] = symbol_data["close"].pct_change(periods=168)
    data["return_30d"] = symbol_data["close"].pct_change(periods=720)

    data["log_return"] = np.log(symbol_data["close"] / symbol_data["close"].shift(1))
    
    # ===========================================================    
    # ================ VOLATILITY CENTER ========================

    volatility_24h = data["log_return"].rolling(window=24).std()
    volatility_7d = data["log_return"].rolling(window=168).std()
    volatility_30d = data["log_return"].rolling(window=720).std()

    # current volatility (for features)
    data["volatility_24h"] = volatility_24h
    data["volatility_7d"] = volatility_7d
    data["volatility_30d"] = volatility_30d

    # lagged volatility (yesterday's)
    data["lagged_volatility_24h"] = volatility_24h.shift(1)
    data["lagged_volatility_7d"] = volatility_7d.shift(1)
    data["lagged_volatility_30d"] = volatility_30d.shift(1) 

    # ===========================================================
    # ================ PARKINSONE CENTER =========================

    data["parkinson_vol"] = np.sqrt((np.log(symbol_data["high"] / symbol_data["low"]))**2 / (4 * np.log(2)))

    # ===========================================================
    # ================== VOLUME CENTER ==========================

    volume_ma_24 = symbol_data["volume"].rolling(window=24).mean()
    data["volume_ma_24"] = volume_ma_24
    data["vol_ratio"] = symbol_data["volume"] / volume_ma_24

    # ===========================================================    
    # ===================== RSI CENTER ==========================
    delta = symbol_data["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    RS = avg_gain / avg_loss
    data["rsi_14"] = 100 - (100 / (1 + RS))


    # ===================== ATR CENTER ==========================
    # true range
    high_low = symbol_data["high"] - symbol_data["low"]
    high_close = (symbol_data["high"] - symbol_data["close"].shift(1)).abs()
    low_close = (symbol_data["low"] - symbol_data["close"].shift(1)).abs()
    
    TR = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data["atr_14"] = TR.rolling(window=14).mean()

    # ===========================================================    
    # =============== SMA_24 / EMA_24 CENTER ====================
    data["sma_24"] = symbol_data["close"].rolling(window=24).mean()
    data["ema_24"] = symbol_data["close"].ewm(span=24, adjust=False).mean()
    data["sma_168"] = symbol_data["close"].rolling(window=168).mean()
    

    # ===========================================================    
    # ============= BOLLINGER BANDS (20days) CENTER ================

    bb_middle = symbol_data["close"].rolling(window=20).mean()
    bb_std = symbol_data["close"].rolling(window=20).std()
    
    data["bb_middle"] = bb_middle
    data["bb_upper"] = bb_middle + 2 * bb_std
    data["bb_lower"] = bb_middle - 2 * bb_std
    data["bb_width"] = (data["bb_upper"] - data["bb_lower"]) / bb_middle
    data["bb_position"] = (symbol_data["close"] - data["bb_lower"]) / (data["bb_upper"] - data["bb_lower"])
    
    # ===========================================================    
    # ======================= MACD CENTER =======================

    ema_12 = symbol_data["close"].ewm(span=12, adjust=False).mean()
    ema_26 = symbol_data["close"].ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    
    data["macd_line"] = macd_line
    data["macd_signal"] = signal_line
    data["macd_histogram"] = macd_line - signal_line
    
    # ===========================================================    
    # =============== SKEWNESS / KURTOSIS =======================
    data["return_skew_7d"] = data["log_return"].rolling(168).skew()
    data["return_kurt_7d"] = data["log_return"].rolling(168).kurt()


    # ===========================================================    
    # ================= TIME FEATURES ===========================

    timestamp = pd.to_datetime(symbol_data["timestamp"], unit='s')
    
    data["hour_sin"] = np.sin(2 * np.pi * timestamp.dt.hour / 24)
    data["hour_cos"] = np.cos(2 * np.pi * timestamp.dt.hour / 24)
    data["day_of_week"] = timestamp.dt.dayofweek
    data["is_weekend"] = (timestamp.dt.dayofweek >= 5).astype(int)
    
    # ===========================================================  
    # ============== DISTANCE FROM MA ===========================
    data["distance_sma_24"] = (symbol_data["close"] - data["sma_24"]) / data["sma_24"]
    data["distance_sma_168"] = (symbol_data["close"] - data["sma_168"]) / data["sma_168"]
    
    # ===========================================================  
    # ============== LAGGED RETURNS =============================
    data["return_lag1"] = data["log_return"].shift(1)
    data["return_lag2"] = data["log_return"].shift(2)
    data["return_lag24"] = data["log_return"].shift(24)
    
    # ===========================================================  
    # ============== TARGET VARIABLE ============================

    future_returns = data["log_return"].shift(-24).rolling(24).std()
    data["target_volatility_24h"] = future_returns
    
    # Binary spike label
    threshold = data["volatility_30d"].rolling(720).quantile(0.75)
    data["is_spike"] = (future_returns > threshold).astype(int)
    
    return data  # ← FIX: Return, not }




def process_data():

    db_ohlcv_data = load_data_from_db() # is a dataframe
    if db_ohlcv_data is None:
        logging.error("No data to process")
        return None

    all_features = []

    for symbol in db_ohlcv_data["symbol"].unique():
        logging.info(f"Processing features for {symbol}")

        # filter data for this coin
        symbol_data = db_ohlcv_data[db_ohlcv_data["symbol"] == symbol].copy()
        symbol_data = symbol_data.sort_values("timestamp").reset_index(drop=True)


        # calculate features
        features = calculate_features_for_symbol(symbol_data)
        print(features.columns)

        all_features.append(features)
    
    final_data = pd.concat(all_features, ignore_index=True)

    #drop nans
    final_data = final_data.dropna()
    logging.info("Dropper nans")

    return final_data



data = process_data()

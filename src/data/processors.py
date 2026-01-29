from fetchers import fetch_daily_data
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
    

def process_data():
    db_ohlcv_data = load_data_from_db() # is a dataframe

    # log returns, violatility, lagged violatiily, volume radio, rsi, atr, bolinger bands, 
    data = pd.DataFrame()
    # =================== RETURNS CENTER ========================    


    data["return_1h"] = (db_ohlcv_data["close"] - db_ohlcv_data["close"].shift(1)) / db_ohlcv_data["close"].shift(1)
    data["return_24h"] = (db_ohlcv_data["close"] - db_ohlcv_data["close"].shift(24)) / db_ohlcv_data["close"].shift(24)
    data["return_7d"] = (db_ohlcv_data["close"] - db_ohlcv_data["close"].shift(168)) / db_ohlcv_data["close"].shift(168)
    data["return_30d"] = (db_ohlcv_data["close"] - db_ohlcv_data["close"].shift(720)) / db_ohlcv_data["close"].shift(720)
    
    data["log_return"] = np.log(db_ohlcv_data["close"] / db_ohlcv_data["close"].shift(1))

    # ===========================================================    
    # ================ VOLATILITY CENTER ========================


    volatility_24h = data["log_return"].rolling(window=24).std()
    volatility_7d = data["log_return"].rolling(window=168).std()
    volatility_30d = data["log_return"].rolling(window=720).std()

    data["lagged_volatility_24h"] = volatility_24h.shift(1)
    data["lagged_volatility_7d"] = volatility_7d.shift(1)
    data["lagged_volatility_30d"] = volatility_30d.shift(1) 
    
    # ===========================================================
    # ================ PARKINSONE CENTER =========================

    data["prknsn_val"] = np.sqrt((np.log(db_ohlcv_data["high"] / db_ohlcv_data["low"]))**2 / (4 * np.log(2)))


    # ================== VOLUME CENTER ==========================

    volume_ma_24 = (db_ohlcv_data["volume"].rolling(window=24).mean()).shift(1) #to avoid bias - shift by 1
    data["vol_ratio"] = db_ohlcv_data["volume"] / volume_ma_24

    # ===========================================================    
    # ===================== RSA CENTER ==========================
    delta = db_ohlcv_data["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    RS = avg_gain / avg_loss
    data["RSI"] = 100 - (100 / (1 + RS))


    # ===================== ATR CENTER ==========================
    # true range
    TR = pd.concat([ 
        db_ohlcv_data["high"] - db_ohlcv_data["low"],
        (db_ohlcv_data["high"] - db_ohlcv_data["close"].shift(1)).abs(),
        (db_ohlcv_data["low"] - db_ohlcv_data["close"].shift(1)).abs()
    ], axis=1).max(axis=1)

    data["ATR"] = TR.rolling(window=14).mean()

    # ===========================================================    
    # =============== SMA_24 / EMA_24 CENTER ====================
    data["SMA_24"] = db_ohlcv_data["close"].rolling(window=24).mean()
    data["EMA_24"] = db_ohlcv_data["close"].ewm(span=24, adjust=False).mean()

    
    # ===========================================================    
    # ============= BOLLINGER BANDS (20d) CENTER ================
   
    bb_std = db_ohlcv_data["close"].rolling(window=20).std().shift(1)
    data["bb_middle"] = db_ohlcv_data["close"].rolling(window=20).mean().shift(1)

    data["bb_upper"] = data["bb_middle"] + 2 * bb_std
    data["bb_lower"] = data["bb_middle"] - 2 * bb_std
    data["bb_width"] = (data["bb_upper"] - data["bb_lower"]) / data["bb_middle"]
    data["bb_position"] = (db_ohlcv_data["close"] - data["bb_lower"]) / (data["bb_upper"] - data["bb_lower"])



process_data()
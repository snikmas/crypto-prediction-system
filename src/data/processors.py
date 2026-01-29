from fetchers import fetch_daily_data
import pandas as pd
import logging
import numpy as np
import datetime
from database import create_connection
import math


# 1. plan: get data from the db 
# 2. create dataframe and process data
# for now, main feautres:
# log returns, violatility, lagged violatiily, volume radio, rsi, atr, bolinger bands, macd, skeweness
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

    data = pd.DataFrame()
    # close t - closet-1
    data["return_1h"] = (db_ohlcv_data["close"] - db_ohlcv_data["close"].shift(1)) / db_ohlcv_data["close"].shift(1)
    data["return_24h"] = (db_ohlcv_data["close"] - db_ohlcv_data["close"].shift(24)) / db_ohlcv_data["close"].shift(24)
    data["return_7d"] = (db_ohlcv_data["close"] - db_ohlcv_data["close"].shift(168)) / db_ohlcv_data["close"].shift(168)
    data["return_30d"] = (db_ohlcv_data["close"] - db_ohlcv_data["close"].shift(720)) / db_ohlcv_data["close"].shift(720)
    
    data["log_return"] = np.log(db_ohlcv_data["close"] / db_ohlcv_data["close"].shift(1))
    data["volatility_24h"] = (data["return_24h"] / data["return_1h"]).std()
    data["volatility_7d"] = (data["return_7d"] / data["return_1h"]).std()
    data["volatility_30d"] = (data["return_30d"] / data["return_1h"]).std()
    # data["lagged_volatility"] = data[""] ? how to calcualte? for every hour?
    data["lagged_volatility_24h"] = data["volatility_24h"].shift(1)
    data["lagged_volatility_7d"] = data["volatility_7d"].shift(1)
    data["lagged_volatility_30d"] = data["volatility_30d"].shift(1) #/

    data["prknsn_val"] = np.sqrt(np.power(db_ohlcv_data["high"] / db_ohlcv_data["low"], 2) / (4 * math.log(2)))

    volume_t_24 = db_ohlcv_data["volume"].shift(24)
    data["volume_ma_24"] = (volume_t_24 / data["volume"]).mean()
    data["volume_ratio"] = db_ohlcv_data["volume"] / data["volume_ma_24"]

    # rsi = 100 - 100/(1 + RS) where RS = avg_gain_14 / avg_loss_14
    # avg gai_14 n = sum of positive / 14
    
    data["rsi"] = 100 - 100 / (1 + (pd.filter(returns_14d > 0).sum() / pd.filter(returns_14d < 0).sum))


    # atr_14	EMA(true_range, 14)
    # true_range	max(high-low, abs(high-close_prev), abs(low-close_prev))
    # data["atr_14"] ?
    true_range = pd.max(db_ohlcv_data["high"] - db_ohlcv_data["low"], np.abs(db_ohlcv_data["high"] - db_ohlcv_data["close"].shift(1)))


    # sma_24	mean(close_{t-24:t})
    # ema_24	EMA(close, 24)
    sma_24 = (db_ohlcv_data["close"].shift(24) / db_ohlcv_data["close"]).mean()

    
    # bb_middle	SMA(close, 20)
    # bb_upper	bb_middle + 2 * std(close, 20)
    # bb_lower	bb_middle - 2 * std(close, 20)
    # bb_width	(bb_upper - bb_lower) / bb_middle
    # bb_position	(close - bb_lower) / (bb_upper - bb_lower)



process_data()
import pandas as pd
import numpy as np
from src.data.database import get_db_connection
from src.data.constants import COINS
import logging

def load_data_from_db(symbol: str, limit: int = None) -> pd.DataFrame:
    """
    TODO: 
    - Connect to DB
    - Query hourly_ohlcv table for given symbol
    - Order by timestamp ASC
    - Return as pandas DataFrame
    - Optionally limit rows (for testing)
    """
    conn = get_db_connection()
    if limit is not None:
        query = '''
            SELECT * FROM hourly_ohlcv 
            WHERE symbol = %s
            ORDER BY timestamp ASC
            LIMIT %s
        '''
        params = (symbol, limit)
    else:
        query = '''
            SELECT * FROM hourly_ohlcv 
            WHERE symbol = %s
            ORDER BY timestamp ASC
        '''
        params = (symbol,)
    
    df = pd.DataFrame()
    try:
        df = pd.read_sql(get_data_query, conn, params=(symbol, limit))
        return df
    except Exception as e:
        logging.error(f"Error during load_data_from_db: {e}")
        return pd.DataFrame()
    finally:
        conn.close()


def calculate_basic_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    TODO:
    - Calculate return_1h (1-hour percentage change)
    - Calculate return_24h (24-hour percentage change)
    - Calculate return_7d (168-hour percentage change)
    - Calculate log_return (log of price ratio)
    
    Hints:
    - Use df['close'].pct_change(periods)
    - Use np.log(df['close'] / df['close'].shift(1))
    """
    pass


def calculate_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """
    TODO:
    - Calculate volatility_24h (24-hour rolling std of return_1h)
    - Calculate volatility_7d (168-hour rolling std)
    - Calculate volatility_30d (720-hour rolling std)
    - Calculate volatility_lag_24h (yesterday's volatility_24h)
    
    Hints:
    - Use df['return_1h'].rolling(window).std()
    - Use .shift(24) for lag
    """
    pass


def calculate_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    TODO:
    - Calculate volume_ma_24h (24-hour rolling mean of volume)
    - Calculate volume_ratio (current volume / volume_ma_24h)
    
    Hints:
    - Use df['volume'].rolling(24).mean()
    - Division: df['volume'] / df['volume_ma_24h']
    """
    pass


def calculate_ma_distance(df: pd.DataFrame) -> pd.DataFrame:
    """
    TODO:
    - Calculate ma_50 (50-hour simple moving average of close)
    - Calculate distance_ma50 ((close - ma_50) / ma_50)
    
    Hints:
    - Use df['close'].rolling(50).mean()
    - Percentage distance formula
    """
    pass


def calculate_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    TODO:
    - Extract hour from timestamp (convert Unix timestamp to datetime first)
    - Calculate hour_sin (sine of hour * 2π / 24)
    - Calculate hour_cos (cosine of hour * 2π / 24)
    
    Hints:
    - pd.to_datetime(df['timestamp'], unit='s').dt.hour
    - np.sin(2 * np.pi * df['hour'] / 24)
    """
    pass


def create_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    TODO:
    
    Target 1 - Volatility regime:
    - Calculate hourly_volatility (absolute value of return_1h)
    - Calculate vol_threshold (75th percentile of hourly_volatility over 168 hours)
    - Create target_vol_regime (1 if NEXT hour's volatility > threshold, else 0)
    
    Target 2 - Spike detector:
    - Calculate next_abs_return (absolute value of NEXT hour's return_1h)
    - Calculate spike_threshold (90th percentile of next_abs_return over 720 hours)
    - Create target_spike (1 if next_abs_return > spike_threshold, else 0)
    
    Hints:
    - Use .shift(-1) to get NEXT hour's value
    - Use .rolling(window).quantile(0.75) for percentiles
    - Use .astype(int) to convert boolean to 0/1
    """
    pass


if __name__ == "__main__":
    # Test your functions
    # limit - data hours
    for coin in COINS:
        df = load_data_from_db(coin, limit=1000)
    
    df = calculate_basic_returns(df)
    df = calculate_volatility(df)
    df = calculate_volume_features(df)
    df = calculate_ma_distance(df)
    df = calculate_time_features(df)
    df = create_targets(df)
    
    print(df.head(50))  # Check first 50 rows (first ~720 will have NaN due to rolling windows)
    print(df.columns)
    print(df.describe())
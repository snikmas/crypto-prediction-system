import pandas as pd
import numpy as np
from src.data.database import get_db_connection
from src.data.constants import COINS
import logging
from pathlib import Path
from datetime import datetime


def save_df(df):
    proj_root = Path(__file__).resolve().parents[2]
    processed_dir_path = proj_root / "data" / "processed"
    processed_dir_path.mkdir(parents=True, exist_ok=True)
    
    time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") #or get in 
    parquet_file = processed_dir_path / f"features-{time}.parquet"
    df.to_parquet(parquet_file, index=False, compression='snappy')
    
    parquet_file = Path(parquet_file)
    file_size = parquet_file.stat().st_size / 1024
    logging.info(f"saved to: {parquet_file}")
    logging.info(f"file_size: {file_size:1f} KB")


def load_data_from_db(symbol: str, limit: int = None) -> pd.DataFrame:
    
    conn = get_db_connection()
    query = ''
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
        df = pd.read_sql(query, conn, params=params)
        return df
    except Exception as e:
        logging.error(f"Error during load_data_from_db: {e}")
        return pd.DataFrame()
    finally:
        conn.close()


def calculate_basic_returns(df: pd.DataFrame) -> pd.DataFrame:

    df["return_1h"] = df['close'].pct_change(1)
    df['return_24h'] = df['close'].pct_change(24)
    df['return_7d'] = df['close'].pct_change(168)
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    return df

def calculate_volatility(df: pd.DataFrame) -> pd.DataFrame:

    df['volatility_24h'] = df['return_1h'].rolling(24).std()
    df['volatility_7d'] = df['return_1h'].rolling(168).std()
    df['volatility_30d'] = df['return_1h'].rolling(720).std()
    df['volatility_lag_24h'] = df['volatility_24h'].shift(24) # do we need rolling?
    return df


def calculate_volume_features(df: pd.DataFrame) -> pd.DataFrame:

    df['volume_ma_24h'] = df['volume'].rolling(24).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma_24h'] # is that correct?

    return df


def calculate_ma_distance(df: pd.DataFrame) -> pd.DataFrame:

    df['ma_50'] = df['close'].rolling(50).mean()
    df['distance_ma_50'] = (df['close'] - df['ma_50']) / df['ma_50']

    return df


def calculate_time_features(df: pd.DataFrame) -> pd.DataFrame:

    df['hour'] = pd.to_datetime(df['timestamp'], unit='s').dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    return df


def create_targets(df: pd.DataFrame) -> pd.DataFrame:

    hourly_volatility = df['return_1h'].abs() 
    vol_threshold = hourly_volatility.rolling(window=168).quantile(0.75)
    df['target_vol_regime'] = (hourly_volatility.shift(-1) > vol_threshold).astype(int) # so just 1 or 0?

    # taget 2
    next_abs_return = df['return_1h'].shift(-1).abs()
    spike_threshold = next_abs_return.rolling(window=720).quantile(0.9)
    df['target_spike'] = (next_abs_return > spike_threshold).astype(int)

    return df


if __name__ == "__main__":


    pd.set_option('display.max_columns', None)

    all_data = [] # i feel if we d not do it, df will be only in the loop
    for coin in COINS:
        coin_df = load_data_from_db(coin, limit=50000)

        coin_df = calculate_basic_returns(coin_df)
        coin_df = calculate_volatility(coin_df)
        coin_df = calculate_volume_features(coin_df)
        coin_df = calculate_ma_distance(coin_df)
        coin_df = calculate_time_features(coin_df)
        coin_df = create_targets(coin_df)

        all_data.append(coin_df)

    df = pd.concat(all_data, ignore_index=True)

    save_df(df)

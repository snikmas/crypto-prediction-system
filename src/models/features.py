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
    """
    TODO: 
    - Connect to DB
    - Query hourly_ohlcv table for given symbol
    - Order by timestamp ASC
    - Return as pandas DataFrame
    - Optionally limit rows (for testing)
    """
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

    df["return_1h"] = df['close'].pct_change(1)
    df['return_24h'] = df['close'].pct_change(24)
    df['return_7d'] = df['close'].pct_change(168)
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    return df # do we really need to return df? did it not change the df in the place?


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

    df['volatility_24h'] = df['return_1h'].rolling(24).std()
    df['volatility_7d'] = df['return_1h'].rolling(168).std()
    df['volatility_30d'] = df['return_1h'].rolling(720).std()
    df['volatility_lag_24h'] = df['volatility_24h'].shift(24) # do we need rolling?
    return df


def calculate_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    TODO:
    how much people usually trade in the last 24hours? compare now vs usual
    example: usually you sell 10cups of lemode/hour. but today you sold 50. wtf? smthg is happening
    crypto volume: how much coins were traded. and MA - moving average
    we can know cur volume = 1000. is that big? idk, so

    - Calculate volume_ratio (current volume / volume_ma_24h)
    
    if +-1 - ok; if >1.5 - smthg happened, many people tradindg
    if >3.0 - wth something BIG is happening
    <0.5 everyone asleep
    why do we need that? small push - small move. big push - big move. price moves need force
    force = move. no volume - price lies; high - price tells the trurh.
    price = directoin, volume = strength

    - Calculate volume_ma_24h (24-hour rolling mean of volume)
    - Division: df['volume'] / df['volume_ma_24h']
    
    Hints:
    - Use df['volume'].rolling(24).mean()
    """
    df['volume_ma_24h'] = df['volume'].rolling(24).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma_24h'] # is that correct?

    return df


def calculate_ma_distance(df: pd.DataFrame) -> pd.DataFrame:
    """
    TODO:
    - Calculate ma_50 (50-hour simple moving average of close)
    - Calculate distance_ma50 ((close - ma_50) / ma_50)
    
    ma-50 = the average price over the last 50 time steps, the normal price
    so instead of 100-102-99-101-150-100, idk is it crazy?
    we do 100-101-101-102-103-104. you see the trend
    if we do ma-5 - we're too nervous, reacts fast. 50 - medium, 200 - very slow

    distance_ma50 - how far today's price is from the normal price
    so price = 120 and ma-50 = 100. price is too high compared to normal. market might be overheated.
    and if price = 80, ma-50 = 100 - price is too low, market night be oversold

    why do dividing? before: one coin: 500, another one: 0.02 ? whats the going on?
    with devision go to % - we see 5% 5%. ok, the same tihng

    Hints:
    - Use df['close'].rolling(50).mean()
    - Percentage distance formula
    """
    df['ma_50'] = df['close'].rolling(50).mean()
    df['distance_ma_50'] = (df['close'] - df['ma_50']) / df['ma_50']

    return df


def calculate_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    TODO:
    - Extract hour from timestamp (convert Unix timestamp to datetime first)
    - Calculate hour_sin (sine of hour * 2π / 24)
    - Calculate hour_cos (cosine of hour * 2π / 24)
    we need _sin and _cos to help the computer understand time on a clock
    they teach the model that 23.00 and 0.00 are close, not fat apart. 23->00->01. for u it's obvious, but for a computer - nah
    so we have to turn the clock into a circle
    sin -> tells us up/down position
    cos -> tells us left/right position
    they together describe where we are on the cirlce. so just gps in the circle
    so, for example, if we know 23 is busy -> can predict 0 also is going to be crazy.
    but for ml 23 and 0 are different, are pretty far, so it might think: no connections, 0.00 of course is gonna sleep
    
    Hints:
    - pd.to_datetime(df['timestamp'], unit='s').dt.hour
    - np.sin(2 * np.pi * df['hour'] / 24)
    """

    # change the current timestmap
    df['hour'] = pd.to_datetime(df['timestamp'], unit='s').dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    return df


def create_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    TODO:
    
    Target 1 - Volatility regime:
    - Calculate hourly_volatility (absolute value of return_1h)
    - Calculate vol_threshold (75th percentile of hourly_volatility over 168 hours)
    - Create target_vol_regime (1 if NEXT hour's volatility > threshold, else 0)

    absolute_value = we don't care up or down.
    hourly_volatility - how mucht he price shook in that hour
    vol_threshold = 7 days, 168 hours. some of them are ok, some of them are noisy.
    we gonna take 75% of them - trying to catch those noisy hours, they matter too
    
    target_vol_regime = we are entering a noisy period? NEXT hour check. a danger level

    Target 2 - Spike detector: - big jump
    - Calculate next_abs_return (absolute value of NEXT hour's return_1h)
    - Calculate spike_threshold (90th percentile of next_abs_return over 720 hours)
    - Create target_spike (1 if next_abs_return > spike_threshold, else 0)
    
    predict absolute value of NEXT hour's return. how big is the next move, we dont care what the directoin
    spike_threshold = spikes moves so big, so we just need 10% of them to notice them. they will only here, in this 90% height
    
    target spike = comparing next our prediction spike with the ucrrent threshold

    Hints:
    - Use .shift(-1) to get NEXT hour's value
    - Use .rolling(window).quantile(0.75) for percentiles
    - Use .astype(int) to convert boolean to 0/1
    """

    # target 1
    # do we need save these data in the df?
    hourly_volatility = df['return_1h'].abs() #do we need rolling? like for every value
    vol_threshold = hourly_volatility.rolling(window=168).quantile(0.75)
    df['target_vol_regime'] = (hourly_volatility.shift(-1) > vol_threshold).astype(int) # so just 1 or 0?
    # do rolling(168) or rolling(widndow=168?)


    # taget 2
    next_abs_return = df['return_1h'].shift(-1).abs()
    spike_threshold = next_abs_return.rolling(window=720).quantile(0.9)
    df['target_spike'] = (next_abs_return > spike_threshold).astype(int)

    return df


if __name__ == "__main__":


    pd.set_option('display.max_columns', None)

    # Test your functions
    # limit - data hours
    all_data = [] # i feel if we d not do it, df will be only in the loop
    for coin in COINS:
        coin_df = load_data_from_db(coin, limit=1000)

        coin_df = calculate_basic_returns(coin_df)
        coin_df = calculate_volatility(coin_df)
        coin_df = calculate_volume_features(coin_df)
        coin_df = calculate_ma_distance(coin_df)
        coin_df = calculate_time_features(coin_df)
        coin_df = create_targets(coin_df)

        all_data.append(coin_df)

    df = pd.concat(all_data, ignore_index=True)
    # save this 
    save_df(df)


# FOR DEVELOPMENT CHECKING
#     print(f"\n\nTotal rows: {len(df)}") 
#     print(f"Coins: {df['symbol'].unique()}") 
#     print(f"Columns: {df.columns.tolist()}") 
#     print(f"Summary: \n{df.describe()}") 
#     print("\nNaN counts per column:")
#     print(df.isnull().sum())
#     print(f"df head(20):\n{df.head(20)}") 
    
    
#     # 1. NaN distribution
# print("\nNaN counts:")
# print(df.isnull().sum())

# # 2. Check usable samples (rows without NaN in key features)
# usable = df.dropna(subset=['volatility_30d', 'target_vol_regime', 'target_spike'])
# print(f"\nUsable samples (after warmup): {len(usable)}")
# print(f"Per coin: {len(usable) / 7:.0f} rows")

# # 3. Target balance
# print(f"\nTarget distribution:")
# print(f"High vol regime: {usable['target_vol_regime'].mean():.1%}")
# print(f"Spikes: {usable['target_spike'].mean():.1%}")

# # 4. Check one coin's data
# btc = df[df['symbol'] == 'BTC'].copy()
# print(f"\nBTC data check:")
# print(f"Total rows: {len(btc)}")
# print(f"First usable row (no NaN in volatility_30d): {btc['volatility_30d'].first_valid_index()}")
    
    
    
    
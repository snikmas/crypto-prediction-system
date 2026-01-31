from pathlib import Path
import sys
import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


env_path = Path(__file__).resolve().parents[2] / 'config' / '.env'
load_dotenv(dotenv_path=env_path)
data_path = os.getenv('PATH_TO_PROCESSORS_DATA')
print(data_path)
sys.path.append(data_path)

from processors import data

# Index(['symbol', 'timestamp', 'return_1h', 'return_24h', 'return_7d',
    #    'return_30d', 'log_return', 'volatility_24h', 'volatility_7d',
    #    'volatility_30d', 'lagged_volatility_24h', 'lagged_volatility_7d',
    #    'lagged_volatility_30d', 'parkinson_vol', 'volume_ma_24', 'vol_ratio',
    #    'rsi_14', 'atr_14', 'sma_24', 'ema_24', 'sma_168', 'bb_middle',
    #    'bb_upper', 'bb_lower', 'bb_width', 'bb_position', 'macd_line',
    #    'macd_signal', 'macd_histogram', 'return_skew_7d', 'return_kurt_7d',
    #    'hour_sin', 'hour_cos', 'day_of_week', 'is_weekend', 'distance_sma_24',
    #    'distance_sma_168', 'return_lag1', 'return_lag2', 'return_lag24',
    #    'target_volatility_24h', 'is_spike'],
    #   dtype='str')
print(print(data))


# X = data..?
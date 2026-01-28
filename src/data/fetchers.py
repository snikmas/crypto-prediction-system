from dotenv import load_dotenv
from pathlib import Path
import requests
import json
import os 
from datetime import datetime, timedelta
from constants import COINS
import logging

def fetch_daily_data():



  env_path = Path(__file__).resolve().parents[2] / 'config' / '.env'
  load_dotenv(dotenv_path=env_path)

  now = datetime.now() 
  midnight_today = datetime(now.year, now.month, now.day, 0, 0, 0)
  toTs = int(midnight_today.timestamp())
  
  URL = f"{os.getenv('COIN_DESK_API_URL')}{os.getenv('COIN_DESK_GET_DAILY_OHLCV')}"

  headers = {
    "Content-type":"application/json; "
    "charset=UTF-8",
    "authorization":f"Apikey {os.getenv('COIN_DESK_API_KEY')}"
    }

  params={
    "limit":"1",
    "tsym":"USD",
    "toTs":toTs
    }
  
  data = {}
  
  try:
    for coin in COINS:
      params["fsym"] = coin
      res = requests.get(URL, params=params, headers=headers)     
      
      res.raise_for_status()
      res = res.json()

      coin_data = {}
      for coin_info in res["Data"]["Data"]:
        coin_data = {
          "open": coin_info["open"],
          "high": coin_info["high"],
          "low": coin_info["low"],
          "close": coin_info["close"],
          "volume": coin_info["volumefrom"],
          "timestamp": coin_info["time"]
        }

        
      data[coin] = coin_data

      logging.debug(f"Fetched data for {coin}: {data[coin]}")
      return data 
    return data
  except (ConnectionError, requests.Timeout, requests.TooManyRedirects) as e:
    logging.error(f"Connection error: {e}")
    return None
  
logging.basicConfig(
  format="{asctime} - {levelname} - {message}",
  style="{",
  datefmt="%Y-%m-%d %H:%M",
)

data = fetch_daily_data()

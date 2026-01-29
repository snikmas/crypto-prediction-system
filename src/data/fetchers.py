from dotenv import load_dotenv
from pathlib import Path
import requests
import json
import os 
from datetime import datetime, timedelta
from constants import COINS
import logging


logging.basicConfig(
  format="{asctime} - {levelname} - {message}",
  style="{",
  datefmt="%Y-%m-%d %H:%M",
)

def fetch_hourly_data():

  env_path = Path(__file__).resolve().parents[2] / 'config' / '.env'
  load_dotenv(dotenv_path=env_path)

  now = datetime.now() 
  midnight_today = datetime(now.year, now.month, now.day, 0, 0, 0)
  toTs = int(midnight_today.timestamp())
  
  # end_time = now - timedelta(hours=8000)
  # toTs = int(end_time.timestamp())
  
  
  URL = f"{os.getenv('COIN_DESK_API_URL')}{os.getenv('COIN_DESK_GET_HOURLY_OHLCV')}"

  headers = {
    "Content-type":"application/json; "
    "charset=UTF-8",
    "authorization":f"Apikey {os.getenv('COIN_DESK_API_KEY')}"
    }

  params={
    "limit":"2000", # +- 83 days
    "tsym":"USD",
    "toTs":toTs
    }
  
  data = {}
  
  try:
    for coin in COINS:
      params["fsym"] = coin
      res = requests.get(URL, params=params, headers=headers)     
      
      res.raise_for_status()
      res_json = res.json()

      coin_data = []
      for record in res_json["Data"]["Data"]:
        try:
          coin_data.append({
            "timestamp": record["time"],
            "open": record["open"],
            "high": record["high"],
            "low": record["low"],
            "close": record["close"],
            "volume": record["volumefrom"]
          })
        except KeyError as e:
            logging.warning(f"Missing field in {coin} record: {e}")
            continue      
      data[coin] = coin_data

      logging.debug(f"Fetched data for {coin}: {data[coin]}")
    return data
  except (ConnectionError, requests.Timeout, requests.TooManyRedirects) as e:
    logging.error(f"Connection error: {e}")
    return None
  

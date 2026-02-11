import requests
import os
import logging
from datetime import datetime, timezone
from apscheduler.schedulers.blocking import BlockingScheduler
from dotenv import load_dotenv
from pathlib import Path
from src.data.constants import COINS

# 1. load dotenv (explicit path to project config/.env)
env_path = Path(__file__).resolve().parents[2] / 'config' / '.env'
load_dotenv(dotenv_path=env_path)

# 2. config
COIN_DESK_API_KEY = os.getenv("COIN_DESK_API_KEY")
COIN_DESK_API_URL = os.getenv("COIN_DESK_API_URL")
COIN_DESK_GET_HOURLY_OHLCV = os.getenv("COIN_DESK_GET_HOURLY_OHLCV")

# 3. logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

coins_data = {}

def fetch_coin_data(coin: str) -> dict | None:
    cur_timestamp = int(datetime.now(timezone.utc).timestamp())

    URL = f"{COIN_DESK_API_URL}{COIN_DESK_GET_HOURLY_OHLCV}"

    headers = {
        "Content-type": "application/json; charset=UTF-8",
        "authorization": f"Apikey {COIN_DESK_API_KEY}"
    }

    res = requests.get(
        URL,
        params={"fsym": coin, "tsym": "USD", "limit": "1000", "toTs": cur_timestamp},
        headers=headers,
        timeout=10,
    )
    json_res = res.json()
    res = json_res["Data"]["Data"]

    return res

    
def fetch_all_coins():

    for coin in COINS:
        res = ""
        try:
            res = fetch_coin_data(coin)
            coins_data[coin] = res
        except Exception as e:
            logger.error(f"Failed {coin}: {e}")
            coins_data[coin] = None
    return coins_data 


if __name__ == "__main__":
    fetch_all_coins()
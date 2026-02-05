# get data from the api: connect to it, get data, put to the db
import requests
import os
import logging
from datetime import datetime, timezone
from apscheduler.schedulers.blocking import BlockingScheduler
from dotenv import load_dotenv
from pathlib import Path
from constants import COINS

# VIBE-CODING STEPS
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
    """
    TODO: Implement this.
    
    - Build full URL from COIN_DESK_API_URL + COIN_DESK_GET_HOURLY_OHLCV
    - Calculate current timestamp (use int(datetime.now(timezone.utc).timestamp()))
    - Make request with fsym=coin, tsym="USD", limit=24, toTs=timestamp, api_key
    - Extract response['Data']['Data']
    - Handle errors: log and return None
    - Return the data list
    """
    cur_timestamp = int(datetime.now(timezone.utc).timestamp())

    # Build request URL and headers
    URL = f"{COIN_DESK_API_URL}{COIN_DESK_GET_HOURLY_OHLCV}"

    headers = {
        "Content-type": "application/json; charset=UTF-8",
        "authorization": f"Apikey {COIN_DESK_API_KEY}"
    }

    res = requests.get(
        URL,
        params={"fsym": coin, "tsym": "USD", "limit": "24", "toTs": cur_timestamp},
        headers=headers,
        timeout=10,
    )
    # return
    # raise for HTTP errors and return the data list
    res.raise_for_status()
    json_res = res.json()
    res = json_res["Data"]["Data"]

    return res

    # API returns data under ['Data']['Data'] for historical endpoints
    


    
def fetch_all_coins():
    """
    TODO: Implement this.
    
    - Loop through COINS
    - Call fetch_coin_data(coin)
    - Store in coins_data[coin] = data (or None if failed)
    - Log success/failure for each coin
    """

    for coin in COINS:
        res = ""
        try:
            res = fetch_coin_data(coin)
        except Exception as e:
            logging.info(f"Error during fetching {coin}: {e}")
            res = None
        finally:
            coins_data[coin] = res
            print(f"this is coints data: \n\n{coins_data}\n\n and this type: {type(coins_data)}")
    return coins_data




def scheduled_job():
    logger.info("Starting scheduled fetch at 1:00 AM")
    fetch_all_coins()
    logger.info(f"Fetch complete. Data: {list(coins_data.keys())}")



if __name__ == "__main__":
    # APScheduler: runs in-process, simple, no extra infrastructure
    # Why: You don't need distributed tasks, just one script running daily
    # Alternatives: cron (OS-level, harder to debug), Celery (overkill for this)
    scheduled_job()    
    # scheduler = BlockingScheduler()
    # scheduler.add_job(scheduled_job, 'cron', hour=1, minute=0)  # 1:00 AM daily
    
    # logger.info("Scheduler started. Waiting for 1:00 AM...")
    # scheduler.start()

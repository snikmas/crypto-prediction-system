from dotenv import load_dotenv
import psycopg2
import logging
from pathlib import Path
import os

from src.data import fetchers
from src.utils.utils import mapping_coin_data

# 1. Load dotenv
env_path = Path(__file__).resolve().parents[2] / 'config' / '.env'
load_dotenv(dotenv_path=env_path)

# 2. logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_db_connection():

    conn = psycopg2.connect(
        database = os.getenv("PSQL_DB"),
        user = os.getenv("PSQL_USER"),
        password = os.getenv("PSQL_PASSWORD"),
        host = os.getenv("PSQL_HOST"),
        port = os.getenv("PSQL_PORT"),
    )

    return conn

def insert_hourly_data(coins_data: dict):

    if coins_data is None:
        logger.error("Coins_data is None. Nothing to insert")
        return

    conn = get_db_connection()
    cursor = conn.cursor()

    insert_query = """
        INSERT INTO hourly_ohlcv (symbol, timestamp, open, high, low, close, volume)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (symbol, timestamp) DO NOTHING;
    """

    values = []
    for key, value in coins_data.items():
        if key is None or value is None:
            logger.error(f"None value during adding info: {key}: {coins_data}")
        for coin in value:

            val = mapping_coin_data(key, coin)
            values.append(val)


    try:
        cursor.executemany(insert_query, values)
        conn.commit()
    except Exception as e:
        logger.error(f"Error during inserting hourly data: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

# 3. get data
if __name__ == "__main__":
    data = fetchers.fetch_all_coins()
    insert_hourly_data(data)

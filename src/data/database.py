# Structure:
from dotenv import load_dotenv
import psycopg2
import logging
from pathlib import Path
import sys
import os
from setup_db import setup_database

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fetchers
from utils.utils import mapping_coin_data

# 1. Load dotenv
env_path = Path(__file__).resolve().parents[2] / 'config' / '.env'
load_dotenv(dotenv_path=env_path)

# 2. logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_db_connection():
    """
    - Load .env from config/.env
    - Connect using psycopg2
    - Return connection object
    - Handle connection errors (log + raise)
    """
    conn = psycopg2.connect(
        database = os.getenv("PSQL_DB"),
        user = os.getenv("PSQL_USER"),
        password = os.getenv("PSQL_PASSWORD"),
        host = os.getenv("PSQL_HOST"),
        port = os.getenv("PSQL_PORT"),
    )

    return conn

def insert_hourly_data(coins_data: dict):

    """
    - Get connection
    - For each coin in coins_data:
        - Extract list of hourly records
        - Transform to list of tuples: (symbol, timestamp, high, low, volume, close)
        - Use executemany with ON CONFLICT UPDATE query
    - Commit transaction
    - Log success/errors per coin
    - Close connection in finally block
    """

    if coins_data is None:
        logger.error("Coins_data is None. Nothing to insert")
        return

    conn = get_db_connection()
    cursor = conn.cursor() 
    # cursor perfoms db operations

    # setup_database(conn, cursor) # no need for it. run it before running db 
    insert_query = """
        INSERT INTO hourly_ohlcv (symbol, timestamp, open, high, low, close, volume)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (symbol, timestamp) DO NOTHING;
    """

    values = []
    for key, value in coins_data.items():
        if key is None or value is None:
            logging.error(f"None value during adding info: {key}: {coins_data}")
        for coin in value:
            
            val = mapping_coin_data(key, coin)
            values.append(val)
    
    
    try:
        cursor.executemany(insert_query, values)
        conn.commit()
    except Exception as e:
        logging.error(f"Error during inserting hourly data: {e}") 
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

    pass

# 3. get data
if __name__ == "__main__":
    data = fetchers.scheduled_job()
    insert_hourly_data(data)
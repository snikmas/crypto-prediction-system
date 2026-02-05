# Structure:
import load_dotenv

env_path = Path(__file__).resolve().parents[2] / 'config' / '.env'
load_dotenv(dotenv_path=env_path)


def get_db_connection():
    """
    - Load .env from config/.env
    - Connect using psycopg2
    - Return connection object
    - Handle connection errors (log + raise)
    """
    pass

def insert_hourly_data(coins_data: dict):
    """
    - Get connection
    - For each coin in coins_data:
        - Extract list of hourly records
        - Transform to list of tuples: (symbol, timestamp, open, high, low, close, volume)
        - Use executemany with ON CONFLICT UPDATE query
    - Commit transaction
    - Log success/errors per coin
    - Close connection in finally block
    """
    pass
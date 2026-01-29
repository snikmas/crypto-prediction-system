import psycopg2
import os
from dotenv import load_dotenv
from pathlib import Path
import logging
from fetchers import fetch_hourly_data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def create_connection():

    db_name = os.getenv('PSQL_DB')
    user = os.getenv('PSQL_USER')
    password = os.getenv('PSQL_PASSWORD')
    host = os.getenv('PSQL_HOST')
    port = os.getenv('PSQL_PORT')

    try:
        connection = psycopg2.connect(
            dbname=db_name,
            user=user,
            password=password,
            host=host,
            port=port
        )
        return connection
    except psycopg2.Error as e:
        logging.error(f"Error connecting to database: {e}")
        return None
    
def create_hourly_table(connection):

    create_table_query = """
    CREATE TABLE IF NOT EXISTS hourly_ohlcv (
        symbol TEXT NOT NULL,
        timestamp BIGINT NOT NULL,
        open DECIMAL(20, 8) NOT NULL,
        high DECIMAL(20, 8) NOT NULL,
        low DECIMAL(20, 8) NOT NULL,
        close DECIMAL(20, 8) NOT NULL,
        volume DECIMAL(20, 8) NOT NULL,
        created_at TIMESTAMP DEFAULT NOW(),
        PRIMARY KEY (symbol, timestamp)
    );
    """

    cursor = connection.cursor()
    try:
        cursor.execute(create_table_query)
        connection.commit()
        logging.info("Table created/verified successfully")
        return True
    except psycopg2.Error as e:
        logging.error(f"Error creating table: {e}")
        connection.rollback()
        return False
    finally:
        cursor.close()
    

def insert_data(connection, data):

    if data is None:
        logging.error("No data to insert")
        return False


    insert_query = """
        INSERT INTO hourly_ohlcv (symbol, timestamp, open, high, low, close, volume)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (symbol, timestamp) DO NOTHING;
    """


    cursor = connection.cursor()
    logging.info("Inserting data...")
    try:

        rows = []
        for coin, records in data.items():
            for record in records:
                rows.append((
                    coin,
                    record['timestamp'],
                    record['open'],
                    record['high'],
                    record['low'],
                    record['close'],
                    record['volume']
                ))

        # Single batch insert
                # Batch insert
        cursor.executemany(insert_query, rows)
        connection.commit()
        
        inserted = cursor.rowcount
        skipped = len(rows) - inserted
        
        logging.info(f"Inserted: {inserted}, Skipped: {skipped}, Total: {len(rows)}")
        return True
    except psycopg2.Error as e:
        logging.error(f"Error inserting/updating data: {e}")
        connection.rollback()
        return False
    finally:
        cursor.close()
    

env_path = Path(__file__).resolve().parents[2] / 'config' / '.env'
load_dotenv(dotenv_path=env_path)

connection = create_connection()


if connection:
    data = fetch_hourly_data()
    try:
        if data:
            create_hourly_table(connection)
            res = insert_data(connection, data)
            if res:
                logging.info("Database update completed successfully.")
            else:
                logging.error("Database update failed.")
        else:
            logging.error("No data fetched")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        if connection:
            connection.close()
    
else:
    logging.error("Failed to connect to database")
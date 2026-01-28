import psycopg2
import os
from dotenv import load_dotenv
from pathlib import Path
import logging
from fetchers import fetch_daily_data


def create_connection(db_name, user, password, host, port):
    """
    Create and return a connection to the PostgreSQL database.
    
    Parameters:
    - db_name: Name of the database.
    - user: Username for authentication.
    - password: Password for authentication.
    - host: Database host address (default is 'localhost').
    - port: Connection port number (default is 5432).
    
    Returns:
    - A connection object to the PostgreSQL database.
    """
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
        print(f"Error connecting to database: {e}")
        return None
    
def create_table(connection):
    create_table_query = """
    CREATE TABLE IF NOT EXISTS  (
        symbol TEXT PRIMARY KEY,
        open FLOAT,
        high FLOAT,
        low FLOAT,
        close FLOAT,
        volume FLOAT,
        timestamp TIMESTAMP
    );
    """
    cursor = connection.cursor()
    try:
        cursor.execute(create_table_query)
        connection.commit()
        logging.info("Table created successfully.")
    except psycopg2.Error as e:
        logging.error(f"Error creating table: {e}")
        connection.rollback()
    finally:
        cursor.close()
    

def add_data_query(connection, data):
    insert_query = """
    INSERT INTO  (symbol, open, high, low, close, volume, timestamp)
    VALUES (%s, %s, %s, %s, %s, %s, to_timestamp(%s))
    ON CONFLICT (symbol) DO UPDATE SET
        open = EXCLUDED.open,
        high = EXCLUDED.high,
        low = EXCLUDED.low,
        close = EXCLUDED.close,
        volume = EXCLUDED.volume,
        timestamp = EXCLUDED.timestamp;
    """
    cursor = connection.cursor()
    try:
        for symbol, values in data.items():
            cursor.execute(insert_query, (
                symbol,
                values['open'],
                values['high'],
                values['low'],
                values['close'],
                values['volume'],
                values['timestamp']
            ))
        connection.commit()
        logging.info("Data inserted/updated successfully.")
    except psycopg2.Error as e:
        logging.error(f"Error inserting/updating data: {e}")
        connection.rollback()
    finally:
        cursor.close()
    

env_path = Path(__file__).resolve().parents[2] / 'config' / '.env'
load_dotenv(dotenv_path=env_path)

connection = create_connection(os.getenv('PSQL_DB'), 
                  os.getenv('PSQL_USER'), 
                  os.getenv('PSQL_PASSWORD'), 
                  os.getenv('PSQL_HOST'), 
                  os.getenv('PSQL_PORT'))

data = fetch_daily_data
create_table(connection)
add_data_query(connection, data)
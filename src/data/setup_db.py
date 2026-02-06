import logging

# setup_db.py
def setup_database(conn, cursor):
    table_query = """
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
    try:
        cursor.execute(table_query)
        conn.commit()
        return 0
    except Exception as e:
        logging.error(f"The error during creaitng a table: {e}")
        return 1

if __name__ == "__main__":
    from database import get_db_connection

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        setup_database(conn, cursor)
    finally:
        cursor.close()
        conn.close()
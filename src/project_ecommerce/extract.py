import argparse
import json

import pandas as pd
from airflow.providers.mysql.hooks.mysql import MySqlHook
from rec_engine import log

from . import RAW_PATH, PARAMS_PATH


CLICKS_PATH = RAW_PATH / "clicks.parquet"
PRODUCTS_PATH = RAW_PATH / "products.parquet"

def connect_db(mysql_conn_id: str) -> MySqlHook:
    """
    Connect to MySQL database through Airflow connection ID

    Parameters
    ---
    mysql_conn_id : str
        Airflow connection ID for MySQL database

    Returns
    ---
    airflow.providers.mysql.hooks.mysql.MySqlHook
    """
    return MySqlHook(mysql_conn_id=mysql_conn_id).get_conn()

def query_db(query: str, connection: MySqlHook) -> pd.DataFrame:
    """
    Query database

    Parameters
    ---
    query: str
        SQL query
    connection : airflow.providers.mysql.hooks.mysql.MySqlHook
        MySQL connection object

    Returns
    ---
    pandas.DataFrame

    """
    log.info("Extracting data from database.")
    with connection.cursor() as cursor:
        # Query database
        cursor.execute(query)
        # Fetch data and save it as Pandas dataframe
        data = pd.DataFrame(cursor.fetchall(), columns = [col[0] for col in cursor.description])
    log.info("Data extracted successfully.")
    return data

def extract_db(conn_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract clicks and products datasets from remote database

    Parameters
    ---
    conn_id : str
        Airflow connection ID for MySQL database

    Returns
    ---
    tuple[pd.DataFrame, pd.DataFrame]
    """
    connection = connect_db(conn_id)
    clicks_df = query_db("""
        SELECT
            interactions.time,
            interactions.user_id,
            interactions.product_id,
            interactions.merchant_id,
            interactions.category_id,
            interactions.channel,
            interactions.device_type,
            interactions.query_text,
            interactions.click,
            interactions.add_to_cart,
            interactions.conversion
        FROM interactions
        WHERE click
        ORDER BY interactions.time
    """, connection = connection)
    products_df = query_db("""
        SELECT
            products.id AS product_id,
            products.merchant_id,
            products.category_id,
            products.name AS product_name,
            categories.name AS category_name,
            merchants.name AS merchant_name,
            merchants.city AS merchant_city,
            merchants.state AS merchant_state,
            merchants.region AS merchant_region,
            products.price_in_cents,
            products.on_sale,
            products.free_shipping,
            products.is_sold_out,
            products.editor_pick,
            products.reviews,
            products.sales_last_week,
            products.sales_last_month,
            products.sales_last_year
        FROM products
        JOIN categories ON products.category_id = categories.id
        JOIN merchants ON products.merchant_id = merchants.id;
    """, connection = connection)
    connection.close()
    return clicks_df, products_df

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Load clicks and products datasets from local storage

    Returns
    ---
    tuple[pd.DataFrame, pd.DataFrame, dict]
    """
    # Load data
    log.info("Loading data")
    with open(PARAMS_PATH) as f:
        params = json.load(f)
    log.info(f"Loaded parameters from {PARAMS_PATH}")
    clicks_df = pd.read_parquet(CLICKS_PATH)
    products_df = pd.read_parquet(PRODUCTS_PATH)
    log.info(f"Loaded datasets:\n - {CLICKS_PATH}\n - {PRODUCTS_PATH}")
    return clicks_df, products_df, params


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conn_id", type=str, default="aiven_rec_db", help="Airflow connection ID for MySQL database")
    args = parser.parse_args()
    log.info("Extracting data from database...")
    clicks_df, products_df = extract_db(args.conn_id)
    log.info("Data extraction completed.")
    clicks_df.to_parquet(CLICKS_PATH)
    products_df.to_parquet(PRODUCTS_PATH)
    log.info(f"Data saved:\n  - {CLICKS_PATH}\n  - {PRODUCTS_PATH}")

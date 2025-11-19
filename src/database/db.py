# src/database/db.py
import os
from sqlalchemy import create_engine
import pandas as pd

# This gets the URL from the docker-compose environment variables
# If not found (running locally), defaults to localhost
DB_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/feature_store")

def get_engine():
    """Returns the SQLAlchemy engine."""
    return create_engine(DB_URL)

def load_data(query):
    """Loads data from Postgres into a Pandas DataFrame."""
    engine = get_engine()
    return pd.read_sql(query, engine)

def save_data(df, table_name, if_exists='append'):
    """Saves a DataFrame to Postgres."""
    engine = get_engine()
    df.to_sql(table_name, engine, if_exists=if_exists, index=False)
    print(f"Saved {len(df)} rows to table '{table_name}'")
# src/database/db.py
import os
from sqlalchemy import create_engine
import pandas as pd

# This gets the URL from the docker-compose environment variables
# If not found (running locally), defaults to localhost
DB_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/feature_store")

def get_engine():
    """Returns the SQLAlchemy engine with connection pooling."""
    return create_engine(
        DB_URL,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,  # Test connections before using
        echo=False
    )

def load_data(query):
    """Loads data from Postgres into a Pandas DataFrame with error handling."""
    try:
        engine = get_engine()
        df = pd.read_sql(query, engine)
        if df.empty:
            print(f"⚠ Warning: Query returned no results")
        return df
    except Exception as e:
        print(f"✗ Database error loading data: {e}")
        raise

def save_data(df, table_name, if_exists='append'):
    """Saves a DataFrame to Postgres with error handling."""
    try:
        engine = get_engine()
        df.to_sql(table_name, engine, if_exists=if_exists, index=False)
        print(f"✓ Saved {len(df)} rows to table '{table_name}'")
    except Exception as e:
        print(f"✗ Database error saving data: {e}")
        raise
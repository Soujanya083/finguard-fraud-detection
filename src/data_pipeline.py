import os
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from dotenv import load_dotenv

load_dotenv()

DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ["DB_PASSWORD"]
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "finguard_db")

url = URL.create(
    "postgresql+psycopg2",
    username=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT,
    database=DB_NAME,
)
engine = create_engine(url)

print("Reading CSV...")
df = pd.read_csv("data/raw/creditcard.csv")
print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

print("Writing to PostgreSQL as 'transactions' table...")
df.to_sql("transactions", engine, if_exists="replace", index=False, chunksize=10000)
print("Done. Data loaded into finguard_db.transactions")
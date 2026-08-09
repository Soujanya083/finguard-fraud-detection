import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

# --- DB connection settings ---
DB_USER = "postgres"
DB_PASSWORD = "ganesha@03"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "finguard_db"

url = URL.create(
    "postgresql+psycopg2",
    username=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT,
    database=DB_NAME,
)

engine = create_engine(url)

# --- Load CSV ---
print("Reading CSV...")
df = pd.read_csv("data/raw/creditcard.csv")
print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

# --- Push to PostgreSQL ---
print("Writing to PostgreSQL as 'transactions' table...")
df.to_sql("transactions", engine, if_exists="replace", index=False, chunksize=10000)
print("Done. Data loaded into finguard_db.transactions")
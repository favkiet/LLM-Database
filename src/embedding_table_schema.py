import os
import sys
import json
from pathlib import Path

# Add project root to sys.path BEFORE importing from src
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.utils import create_table_sql, index_tables_schema_faiss

# Load table schema
with open(project_root / "data" / "train_tables_retails.json", "r") as f:
    table_entry = json.load(f)[0]

# Create SQL and index
csv_dir = project_root / "data" / "retails" / "database_description"
sql_all_tables = create_table_sql(table_entry, csv_dir=csv_dir)
index_tables_schema_faiss(sql_all_tables)
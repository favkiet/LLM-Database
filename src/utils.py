import faiss
import os, sys, json, re
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from src.logger_utils import logger

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

def create_table_sql(table_entry: dict, csv_dir: str) -> list[str]:
    """
    Generate full SQL schema for each table, using train_table JSON + CSV description.

    Args:
        table_entry: dict from train_tables.json
        csv_dir: folder containing CSVs with column descriptions

    Returns:
        SQL schema string with CREATE TABLE, PK, FK, column comments, join context.
    """
    table_names = table_entry["table_names_original"]
    column_names = table_entry["column_names_original"]
    column_names_h = table_entry["column_names"]
    column_types = table_entry["column_types"]
    primary_keys = table_entry["primary_keys"]
    foreign_keys = table_entry["foreign_keys"]

    # Load CSV descriptions into dict: {table_name: {original_column_name: description_str}}
    csv_dir = Path(csv_dir)
    csv_desc = {}
    for csv_file in csv_dir.glob("*.csv"):
        df = pd.read_csv(csv_file).fillna("")
        desc_dict = {}
        for _, row in df.iterrows():
            col_name = row["original_column_name"]
            desc = row["column_description"]
            value_desc = row.get("value_description", "")
            full_desc = f"{desc}"
            if value_desc:
                full_desc += f" ({value_desc})"
            desc_dict[col_name] = full_desc
        csv_desc[csv_file.stem] = desc_dict

    sql_all_tables = []
    table_join_context = {t: [] for t in table_names}

    # Precompute join context
    for src_idx, dst_idx in foreign_keys:
        src_table_idx, src_col = column_names[src_idx]
        dst_table_idx, dst_col = column_names[dst_idx]
        src_table = table_names[src_table_idx]
        dst_table = table_names[dst_table_idx]
        join_str = f"{src_table}.{src_col} = {dst_table}.{dst_col}"
        table_join_context[src_table].append(join_str)

    # Generate CREATE TABLE for each table
    for t_idx, table_name in enumerate(table_names):
        sql = f"-- Table: {table_name}\n"
        sql += f"CREATE TABLE {table_name} (\n"

        # Columns for this table
        cols = [
            (col_idx, orig, col_h, col_type)
            for col_idx, ((tbl_idx, orig), (_, col_h), col_type)
            in enumerate(zip(column_names, column_names_h, column_types))
            if tbl_idx == t_idx
        ]

        for col_idx, col_name, col_h, col_type in cols:
            # Lookup CSV description if available
            col_desc = csv_desc.get(table_name, {}).get(col_name, col_h)
            sql += f"  {col_name} {col_type.upper()}, -- {col_desc}\n"

        # Primary key(s)
        pk_idx = primary_keys[t_idx] if isinstance(primary_keys[t_idx], list) else [primary_keys[t_idx]]
        if pk_cols := [column_names[i][1] for i in pk_idx if column_names[i][0] == t_idx]:
            sql += f"  PRIMARY KEY ({', '.join(pk_cols)}),\n"

        # Foreign keys
        for src_idx, dst_idx in foreign_keys:
            src_tbl_idx, src_col = column_names[src_idx]
            dst_tbl_idx, dst_col = column_names[dst_idx]
            if src_tbl_idx == t_idx:
                ref_table = table_names[dst_tbl_idx]
                sql += f"  FOREIGN KEY ({src_col}) REFERENCES {ref_table}({dst_col}),\n"

        sql = sql.rstrip(",\n") + "\n);\n"

        # Add possible joins
        if joins := table_join_context.get(table_name):
            sql += "--- Possible joins for this table: "
            for jc in joins:
                sql += f"{jc}\n"

        sql_all_tables.append(sql)

    return sql_all_tables

def index_tables_schema_faiss(sql_all_tables: list):
    """
    Index table schema descriptions into FAISS vector store.
    """
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data" / "retails"
    schema_dir = data_dir / "database_description"
    faiss_index_path = data_dir / "table_schema_1.index"
    metadata_path = data_dir / "table_metadata_1.json"

    # 1️⃣ Load model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    dim = model.get_sentence_embedding_dimension()

    # 2️⃣ Init FAISS index (cosine similarity via inner product)
    index = faiss.IndexFlatIP(dim)

    # 3️⃣ Prepare metadata
    metadata = []
    embeddings = []

    for table_sql in sql_all_tables:
        embedding = model.encode(table_sql, normalize_embeddings=True)
        embeddings.append(embedding)
        metadata.append(table_sql)

    embeddings = np.array(embeddings, dtype="float32")
    index.add(embeddings)
    faiss.write_index(index, str(faiss_index_path))
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Indexed {len(metadata)} tables into {faiss_index_path}")
    print(f"✅ Metadata saved to {metadata_path}")

def retrieve_relevant_tables_faiss(query: str, k: int = 1):
    """
    Retrieve top-k most relevant tables for a given query using FAISS.
    """
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data" / "retails"
    faiss_index_path = data_dir / "table_schema.index"
    metadata_path = data_dir / "table_metadata.json"

    relevant_tables_sql = []
    # 1️⃣ Load model and index
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index(str(faiss_index_path))

    # 2️⃣ Load metadata
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # 3️⃣ Encode query
    query_emb = model.encode(query, normalize_embeddings=True).astype("float32").reshape(1, -1)

    # 4️⃣ Search top-k
    scores, indices = index.search(query_emb, k)
    scores, indices = scores[0], indices[0]
    logger.info("==================== Relevant Tables ====================")
    for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
        desc = metadata[idx]
        match = re.search(r"CREATE\s+TABLE\s+(\w+)", desc, re.IGNORECASE)
        table_name = match[1] if match else None
        print(f"{rank}. (similarity={score:.6f})\n   Table: {table_name}\n")
        relevant_tables_sql.append(desc) 

    return "\n".join(relevant_tables_sql)


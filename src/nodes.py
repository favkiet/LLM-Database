import re
import glob
import sqlite3
import os, sys
import pandas as pd
import numpy as np
from pathlib import Path
from langsmith import traceable
from langchain_ollama import ChatOllama
from sentence_transformers import SentenceTransformer
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage

from IPython.display import display

from src.state import ChatState
from src.logger_utils import logger
from src.utils import retrieve_relevant_tables_faiss
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))


@traceable(name="llm_response")
def llm_response(state: ChatState):
    """Node that sends conversation to a local Ollama chat model and appends AI response."""
    human_input = state.human_messages
    db_result = state.result_query
    
    system_prompt = (
        "You are an AI assistant that answers user questions based on the given SQL query result.\n"
        "If the result is available, summarize answer in natural language.\n"
        "Always respond clearly and concisely.\n\n"
    )
    user_prompt = f"""
    ### Input:
    Query result: {db_result}
    User question: {human_input}

    ### Response:
    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    llm = ChatOllama(model="gemma3:1b", base_url="http://localhost:11434")
    response = llm.invoke(messages)
    # Append the model response to the conversation list
    state.conversation.append(response)
    state.ai_messages = getattr(response, "content", "")
    return state


@traceable(name="db_query_node")
def db_query_node(state: ChatState):
    """
    Simple node: expects `query_sql` in state.
    Executes the query against data/retails/retails.sqlite and returns results as an AIMessage.
    """
    query_sql = state.query_sql or ""
    if not isinstance(query_sql, str) or not query_sql.strip():
        return state

    # Resolve database path relative to project root
    project_root = Path(__file__).resolve().parents[1]
    db_path = project_root / "data" / "retails" / "retails.sqlite"

    try:
        with sqlite3.connect(db_path, timeout=10) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query_sql)
            if rows := cursor.fetchall():
                df = pd.DataFrame([dict(row) for row in rows])
                result_query = "\n".join([
                    ", ".join(f"{col}={row[col]}" for col in df.columns)
                    for _, row in df.iterrows()
                ])
            else:
                result_query = ""
            state.conversation.append(AIMessage(content=result_query))
            logger.info("==================== Query Result ====================")
            logger.info(result_query)
            state.result_query = result_query
    except Exception as e:
        state.conversation.append(AIMessage(content=f"DB error: {e}"))
        state.result_query = ""

    return state
    
    
@traceable(name="text2sql_node")
def text2sql_node(state: ChatState):
    """
    Node that converts a natural language question into an SQL query.
    Context includes: user question + schema descriptions (from CSV files).
    """
    user_question = state.human_messages
    if not user_question:
        raise ValueError("No user question provided.")

    # --- Load table schema descriptions ---
    project_root = Path(__file__).resolve().parents[1]
    relevant_table_sql = retrieve_relevant_tables_faiss(user_question, 3)
    
    # --- Construct LLM prompt ---
    system_prompt = """### Instructions:
    Your task is to convert a question into a SQL query, given a SQLite database schema.
    Rules:
        1. All SQL must be compatible with SQLite.
        2. Always return valid SQL syntax for SQLite.
        3. Condition must follow the question, do not add any additional conditions (even that check NULL values).
        4. For TEXT columns, use one of the following methods:
            - Wrap both sides of the comparison with LOWER() or UPPER(), e.g.:
                WHERE LOWER(customer_name) = LOWER('John Doe')
            - Or, if appropriate, use COLLATE NOCASE, e.g.:
                WHERE customer_name = 'John Doe' COLLATE NOCASE
        5. Do not include any non-SQL syntax like ESCAPE clauses unless strictly needed by SQLite.
    """
    
    user_prompt = f"""
    ### Input:
    Question: {user_question}
    Schema:
    {relevant_table_sql}

    ### Response:
    ```sql
    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    llm = ChatOllama(model="qwen2.5-coder:7b", base_url="http://localhost:11434")
    response = llm.invoke(messages)
    # Set the generated SQL into the state for the next node
    content = getattr(response, "content", "").strip()
    query_cleaned = re.sub(r"^```(?:sql)?\s*|\s*```$", "", content.strip(), flags=re.IGNORECASE | re.MULTILINE)
    state.query_sql = query_cleaned.strip()
    logger.info("==================== Query SQL ====================\n" + state.query_sql)

    state.conversation.append(response)
    return state
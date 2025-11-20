import re
import sqlite3
import os, sys
import pandas as pd
from pathlib import Path
from langsmith import traceable
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from openai import OpenAI

from src.state import ChatState
from src.logger_utils import logger
from src.utils import retrieve_relevant_tables_faiss, get_all_tables_schema
from src.prompt import (
    SYSTEM_PROMPT_LLM_RESPONSE,
    USER_PROMPT_LLM_RESPONSE_TEMPLATE,
    SYSTEM_PROMPT_TEXT2SQL,
    USER_PROMPT_TEXT2SQL_TEMPLATE,
)
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

# Ollama settings
OLLAMA_MODEL_RESPONSE = os.getenv("OLLAMA_MODEL_RESPONSE")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
OLLAMA_MODEL_TEXT2SQL = os.getenv("OLLAMA_MODEL_TEXT2SQL")

# OpenAI settings
OPENAI_MODEL_TEXT2SQL = os.getenv("OPENAI_MODEL_TEXT2SQL", "gpt-4o-mini")
OPENAI_MODEL_RESPONSE = os.getenv("OPENAI_MODEL_RESPONSE", "gpt-4o-mini")


def get_llm_for_provider(provider: str, task: str = "text2sql"):
    """
    Factory function to get the appropriate LLM based on provider.
    
    Args:
        provider: "ollama" or "openai"
        task: "text2sql" or "response" to select the appropriate model
    
    Returns:
        LLM instance (OpenAI client or ChatOllama)
    """
    if provider.lower() == "openai":
        # Return OpenAI client directly (not through LangChain)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key.startswith("${"):
            raise ValueError("Missing OPENAI_API_KEY. Please set it in environment.")
        return OpenAI(api_key=api_key)
    else:  # default to ollama
        model = OLLAMA_MODEL_TEXT2SQL if task == "text2sql" else OLLAMA_MODEL_RESPONSE
        return ChatOllama(model=model, base_url=OLLAMA_BASE_URL)


@traceable(name="llm_response")
def llm_response(state: ChatState):
    """Node that sends conversation to LLM (Ollama only) and appends AI response."""
    human_input = state.human_messages
    db_result = state.result_query
    
    system_prompt = SYSTEM_PROMPT_LLM_RESPONSE
    user_prompt = USER_PROMPT_LLM_RESPONSE_TEMPLATE.format(
        db_result=db_result,
        human_input=human_input,
    )

    # Always use Ollama for response generation
    llm = ChatOllama(model=OLLAMA_MODEL_RESPONSE, base_url=OLLAMA_BASE_URL)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    
    response = llm.invoke(messages)
    state.conversation.append(response)
    state.ai_messages = getattr(response, "content", "")
    
    return state


@traceable(name="db_query_node")
def db_query_node(state: ChatState):
    """
    Simple node: expects `query_sql` in state.
    Executes the query against data/retails/retails.sqlite and returns results as an AIMessage.
    If rows > 20: exports full result to CSV and shows only first 20 rows.
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
                state.row_count = len(rows)
                
                # Nếu > 20 rows: xuất CSV và chỉ hiển thị 20 rows đầu
                if state.row_count > 20:
                    # Tạo thư mục output nếu chưa có
                    output_dir = project_root / "data" / "output"
                    output_dir.mkdir(exist_ok=True)
                    
                    # Tạo tên file với timestamp
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    csv_filename = f"query_result_{timestamp}.csv"
                    csv_path = output_dir / csv_filename
                    
                    # Xuất toàn bộ kết quả ra CSV
                    df.to_csv(csv_path, index=False)
                    state.csv_file_path = str(csv_path)
                    
                    logger.info(f"Exported {state.row_count} rows to CSV: {csv_path}")
                    
                    # Chỉ hiển thị 20 rows đầu tiên
                    df_display = df.head(20)
                    result_query = "\n".join([
                        ", ".join(f"{col}={row[col]}" for col in df_display.columns)
                        for _, row in df_display.iterrows()
                    ])
                    result_query += f"\n\n[Showing first 20 of {state.row_count} rows. Full results saved to: {csv_filename}]"
                else:
                    # Hiển thị tất cả nếu <= 20 rows
                    result_query = "\n".join([
                        ", ".join(f"{col}={row[col]}" for col in df.columns)
                        for _, row in df.iterrows()
                    ])
                    state.csv_file_path = ""
            else:
                result_query = ""
                state.row_count = 0
                state.csv_file_path = ""
                
            state.conversation.append(AIMessage(content=result_query))
            logger.info("==================== Query Result ====================")
            logger.info(f"Number of rows: {state.row_count}")
            if state.csv_file_path:
                logger.info(f"CSV exported to: {state.csv_file_path}")
            print(result_query)
            state.result_query = result_query
    except Exception as e:
        state.conversation.append(AIMessage(content=f"DB error: {e}"))
        state.result_query = ""
        state.row_count = 0
        state.csv_file_path = ""

    return state
    
    
@traceable(name="text2sql_node")
def text2sql_node(state: ChatState):
    """
    Node that converts a natural language question into an SQL query.
    Context includes: user question + schema descriptions (from CSV files).
    Uses provider from state to select LLM (Ollama or OpenAI).
    
    Args:
        state: ChatState object
        use_retrieval: If True, use FAISS to retrieve relevant tables.
                       If False, use all tables schema (for evaluation without evidence).
    """
    user_question = state.human_messages
    
    provider = state.provider
    
    if not user_question:
        raise ValueError("No user question provided.")

    # --- Load table schema descriptions ---
    relevant_table_sql = get_all_tables_schema()
    
    # --- Construct LLM prompt ---
    system_prompt = SYSTEM_PROMPT_TEXT2SQL
    user_prompt = USER_PROMPT_TEXT2SQL_TEMPLATE.format(
        user_question=user_question,
        relevant_table_sql=relevant_table_sql,
    )

    llm = get_llm_for_provider(provider, task="text2sql")
    
    if provider.lower() == "openai":
        # Use OpenAI client directly
        model = OPENAI_MODEL_TEXT2SQL
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = llm.chat.completions.create(
            model=model,
            messages=messages,
        )
        content = response.choices[0].message.content.strip()
        # Create AIMessage for consistency with state.conversation
        ai_message = AIMessage(content=content)
        state.conversation.append(ai_message)
    else:
        # Use LangChain for Ollama
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        response = llm.invoke(messages)
        content = getattr(response, "content", "").strip()
        state.conversation.append(response)
    
    # Clean SQL from markdown code blocks
    query_cleaned = re.sub(r"^```(?:sql)?\s*|\s*```$", "", content.strip(), flags=re.IGNORECASE | re.MULTILINE)
    state.query_sql = query_cleaned.strip()
    logger.info("==================== Query SQL ====================\n" + state.query_sql)
    
    return state


def route_based_on_rows(state: ChatState) -> str:
    """
    Hàm điều hướng dựa trên số lượng rows trong kết quả truy vấn.
    - Nếu row_count < 20: điều hướng đến node 'llm' (hiển thị bảng + câu trả lời AI)
    - Nếu row_count >= 20: điều hướng đến 'END' (chỉ hiển thị 20 rows đầu + xuất CSV)
    """
    row_count = state.row_count
    logger.info("==================== Routing Decision ====================")
    logger.info(f"Row count: {row_count}")
    
    if row_count < 20:
        logger.info("Routing to 'llm' node (few rows - show table + AI response)")
        return "llm"
    else:
        logger.info("Routing to 'END' (many rows - showing first 20 rows)")
        if state.csv_file_path:
            logger.info(f"Full results exported to CSV: {state.csv_file_path}")
        return "END"
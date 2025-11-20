import sys
import csv
import json
import os
import re
import sqlite3
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import pandas as pd
from src.utils import get_all_tables_schema
from src.prompt import (
    SYSTEM_PROMPT_TEXT2SQL,
    USER_PROMPT_TEXT2SQL_TEMPLATE,
)
from openai import OpenAI
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from dotenv import load_dotenv
load_dotenv()

@dataclass
class SampleResult:
    valid: bool
    ves: float
    execution_accuracy: float
    details: Dict[str, float]

import re
def remove_sql_fence(text: str) -> str:
    """
    Remove ```sql ... ``` or ``` ... ``` fences from a string.
    """
    # remove ```sql
    text = re.sub(r"```sql\s*", "", text, flags=re.IGNORECASE)
    # remove ```
    text = re.sub(r"```", "", text)
    return text.strip()
def is_valid_sql(sql: str) -> bool:
    """Heuristic validity check."""
    if not sql or not sql.strip():
        return False

    sql_clean = sql.strip().upper()

    # must start with typical SQL verbs
    valid_starts = ("SELECT", "INSERT", "UPDATE", "DELETE", "WITH")
    if not sql_clean.startswith(valid_starts):
        return False

    # prevent multiple statements
    if ";" in sql_clean[:-1]:  # allow semicolon only at end
        return False

    # parentheses balance
    if sql.count("(") != sql.count(")"):
        return False

    # quote balance
    for q in ["'", '"', "`"]:
        if sql.count(q) % 2 != 0:
            return False

    return True


def tokenize_sql(sql: str) -> List[str]:
    """Simple SQL tokenizer."""
    if not sql:
        return []
    return re.findall(r"\w+|[^\s\w]", sql)


def compute_ves(pred_sql: str, ref_sql: str) -> float:
    """Compute Valid Efficiency Score."""
    valid = is_valid_sql(pred_sql)
    if not valid:
        return 0.0

    pred_tokens = tokenize_sql(pred_sql)
    ref_tokens = tokenize_sql(ref_sql)

    if not pred_tokens or not ref_tokens:
        return 0.0

    ratio = min(len(pred_tokens), len(ref_tokens)) / max(len(pred_tokens), len(ref_tokens))
    return ratio

def normalize_sql_result(result: str) -> List[str]:
    """
    Convert execution result string to normalized list of rows.
    Each row like: "colA=1, colB=Alice"
    """
    if not result or "DB error" in result:
        return []

    rows = result.strip().split("\n")
    # remove ordering effect by sorting rows
    return sorted([row.strip() for row in rows])


def compute_execution_accuracy(pred_sql: str, ref_sql: str) -> float:
    """Execution accuracy = 1 if normalized results match."""
    pred_res = execute_sql(pred_sql)
    ref_res = execute_sql(ref_sql)

    norm_pred = normalize_sql_result(pred_res)
    norm_ref = normalize_sql_result(ref_res)

    if not norm_ref:
        # reference empty → prediction must also be empty
        return 1.0 if not norm_pred else 0.0

    return 1.0 if norm_pred == norm_ref else 0.0


class Text2SQLEvaluator:
    """
    Independent evaluator for Text-to-SQL outputs.
    
    Metrics:
    ==================================================================================
    1. VES (Valid Efficiency Score) - Giải thích lý thuyết chi tiết:
    ==================================================================================
    
    VES là metric đánh giá chất lượng câu SQL được sinh ra dựa trên 2 yếu tố:
    1. Tính hợp lệ (Validity): Kiểm tra cú pháp SQL có đúng không
    2. Hiệu quả (Efficiency): So sánh độ phức tạp với câu SQL tham chiếu
    
    Công thức:
    ----------
    VES = valid_flag × (min(len_pred, len_ref) / max(len_pred, len_ref))
    
    Trong đó:
    - valid_flag: 0 nếu SQL không hợp lệ, 1 nếu hợp lệ
    - len_pred: số lượng token của câu SQL dự đoán
    - len_ref: số lượng token của câu SQL tham chiếu
    
    Ý nghĩa:
    --------
    - VES = 0: SQL không hợp lệ hoặc không thể thực thi
    - VES = 1: SQL hợp lệ và có độ phức tạp tương đương với reference
    - 0 < VES < 1: SQL hợp lệ nhưng có độ phức tạp khác biệt với reference
    
    VES thấp có thể do:
    - SQL quá phức tạp (dài hơn reference nhiều) → Không tối ưu
    - SQL quá đơn giản (ngắn hơn reference nhiều) → Có thể thiếu logic
    
    Ưu điểm của VES:
    ----------------
    1. Không phụ thuộc vào execution (không cần chạy SQL thực tế)
    2. Không cần external parser phức tạp
    3. Đánh giá cả tính đúng đắn cú pháp và hiệu quả
    4. Metric liên tục (0-1), dễ so sánh và phân tích
    5. Phạt nặng các câu SQL không hợp lệ (VES=0)
    
    Heuristic validation bao gồm:
    -----------------------------
    - Kiểm tra cân bằng dấu ngoặc (), quote marks (' " `)
    - Bắt đầu bằng từ khóa SQL hợp lệ (SELECT, INSERT, UPDATE, etc.)
    - Không chứa nhiều statement (tránh SQL injection)
    - Tokenization dựa trên word boundaries và SQL symbols
    
    ==================================================================================
    2. EX (Execution Accuracy) - Giải thích lý thuyết chi tiết:
    ==================================================================================
    
    EX là metric đánh giá xem kết quả thực thi của câu SQL dự đoán có chính xác không
    bằng cách so sánh với kết quả thực thi của câu SQL tham chiếu.
    
    Công thức:
    ----------
    EX = 1 nếu result_pred == result_ref
    EX = 0 nếu result_pred != result_ref hoặc có lỗi khi thực thi
    
    Trong đó:
    - result_pred: Kết quả thực thi câu SQL dự đoán trên database thực tế
    - result_ref: Kết quả thực thi câu SQL tham chiếu trên database thực tế
    
    Ý nghĩa:
    --------
    - EX = 1: SQL dự đoán trả về kết quả chính xác (giống với reference)
    - EX = 0: SQL dự đoán trả về kết quả sai hoặc gặp lỗi khi thực thi
    
    Điểm mạnh của EX:
    -----------------
    1. Đánh giá chính xác nhất - dựa trên kết quả thực tế
    2. Không quan tâm cú pháp SQL khác nhau nếu kết quả giống nhau
    3. Metric binary rõ ràng: đúng hoặc sai
    4. Phản ánh chính xác khả năng trả lời đúng câu hỏi của người dùng
    
    Điểm yếu của EX:
    ----------------
    1. Phụ thuộc vào database thực tế (cần setup và connection)
    2. Chậm hơn metrics khác (phải thực thi SQL)
    3. Có thể có false negatives (SQL khác nhau nhưng đúng logic)
    4. Nhạy cảm với thứ tự rows/columns (cần normalize kết quả)
    
    So sánh kết quả:
    ----------------
    - So sánh từng row và column
    - Bỏ qua thứ tự rows (sắp xếp trước khi so sánh)
    - So sánh giá trị chính xác (bao gồm NULL, empty string)
    - Xử lý lỗi: Nếu pred_sql gây lỗi → EX = 0
    
    ==================================================================================
    """
def execute_sql(sql: str) -> str:
    sql_clean = (sql or "").strip()
    if not sql_clean:
        return ""
    try:
        with sqlite3.connect(db_path, timeout=10) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(sql_clean)
            rows = cursor.fetchall()
            if not rows:
                return ""
            df = pd.DataFrame([dict(row) for row in rows])
            print(df)
            return "\n".join(
                [", ".join(f"{col}={row[col]}" for col in df.columns) for _, row in df.iterrows()]
            )
    except Exception as e:
        return f"DB error: {e}"
        
project_root = Path(__file__).resolve().parent
testfile_path = project_root / "data" / "test_text2sql.json"
db_path = project_root / "data" / "retails" / "retails.sqlite"
eval_dir = project_root / "data" / "evaluation"
eval_dir.mkdir(exist_ok=True)

with open(testfile_path, "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    question = item.get("question", "")
    evidence = item.get("evidence", "")
    sql = item.get("SQL", "")
    
    print(f"\n{'='*80}")
    print("Running evaluation WITHOUT EVIDENCE")
    print("="*80)
    relevant_table_sql = get_all_tables_schema()
    user_question = f"Question: {question} \nEvidence: {evidence}"
    # --- Construct LLM prompt ---
    system_prompt = SYSTEM_PROMPT_TEXT2SQL
    user_prompt = USER_PROMPT_TEXT2SQL_TEMPLATE.format(
        user_question=user_question,
        relevant_table_sql=relevant_table_sql,
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=messages,
    )
    content = response.choices[0].message.content.strip()
    content = remove_sql_fence(content)
    
    ves_score = compute_ves(content, sql)
    exec_accuracy = compute_execution_accuracy(content, sql)

    print(f"Prediction SQL: {content}")
    print(f"VES: {ves_score:.4f}")
    print(f"Execution accuracy: {exec_accuracy}")

    # Store result
    sample_result = SampleResult(
        valid=ves_score > 0.0,
        ves=ves_score,
        execution_accuracy=exec_accuracy,
        details={
            "predicted_sql_length": len(tokenize_sql(content)),
            "reference_sql_length": len(tokenize_sql(sql)),
        }
    )
    
    with open(eval_dir / "result_qwen2.5-coder-7b-without_evidence.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([question, ves_score, exec_accuracy])   
    



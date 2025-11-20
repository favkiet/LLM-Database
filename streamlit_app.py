import os
from typing import Any, Dict

import pandas as pd
import streamlit as st

from src.graph import invoke_llm_database


def _extract_field(state: Any, field: str, default: Any = None) -> Any:
    """
    Helper để lấy field từ state trả về bởi graph.
    Có thể là dict hoặc Pydantic model.
    """
    if isinstance(state, dict):
        return state.get(field, default)
    return getattr(state, field, default)


def _parse_result_query_to_df(result_query: str) -> pd.DataFrame | None:
    """
    Chuyển `state.result_query` (chuỗi) sang DataFrame để hiển thị dạng bảng.

    Định dạng hiện tại (xem trong `db_query_node`):
        col1=val1, col2=val2
        col1=val1, col2=val2
        ...
    Với trường hợp > 20 rows, dòng cuối có thể là thông báo trong ngoặc vuông,
    nên sẽ được bỏ qua.
    """
    if not result_query:
        return None

    lines = [line.strip() for line in result_query.splitlines() if line.strip()]
    if not lines:
        return None

    # Bỏ các dòng thông báo dạng "[Showing first 20 of ...]"
    data_lines = [ln for ln in lines if not ln.lstrip().startswith("[")]
    if not data_lines:
        return None

    rows: list[Dict[str, Any]] = []
    for line in data_lines:
        parts = [p.strip() for p in line.split(",")]
        row: Dict[str, Any] = {}
        for part in parts:
            if "=" not in part:
                continue
            col, val = part.split("=", 1)
            row[col.strip()] = val.strip()
        if row:
            rows.append(row)

    if not rows:
        return None

    return pd.DataFrame(rows)


def _init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "provider" not in st.session_state:
        # Mặc định dùng OpenAI nếu đã cấu hình, ngược lại dùng ollama
        st.session_state.provider = "openai" if os.getenv("OPENAI_API_KEY") else "ollama"


def main() -> None:
    st.set_page_config(page_title="LLM Database Chatbot", layout="wide")
    st.title("LLM Database Chatbot")

    _init_session_state()

    # Sidebar: chọn provider
    with st.sidebar:
        st.header("Cấu hình")
        provider = st.selectbox(
            "Provider",
            options=["openai", "ollama"],
            index=0 if st.session_state.provider == "openai" else 1,
        )
        st.session_state.provider = provider
        st.caption(
            "Luồng xử lý:\n"
            "- Nếu kết quả truy vấn **< 20 rows**: hiển thị bảng + gọi LLM để giải thích.\n"
            "- Nếu kết quả truy vấn **≥ 20 rows**: chỉ hiển thị bảng."
        )

    # Hiển thị lịch sử hội thoại
    for msg in st.session_state.messages:
        role = msg.get("role", "assistant")
        with st.chat_message(role):
            # Nội dung text (nếu có)
            if "content" in msg and msg["content"]:
                st.markdown(msg["content"])

            # SQL đã sinh (nếu có)
            query_sql_hist = msg.get("query_sql")
            if query_sql_hist:
                st.markdown("**SQL được sinh ra:**")
                st.code(query_sql_hist, language="sql")

            # Bảng kết quả (nếu có)
            df = msg.get("table")
            if isinstance(df, pd.DataFrame):
                st.dataframe(df, use_container_width=True)

    # Ô nhập câu hỏi
    user_input = st.chat_input("Nhập câu hỏi về dữ liệu retails...")
    if not user_input:
        return

    # Lưu và hiển thị message của user
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Xử lý truy vấn với pipeline hiện có
    with st.chat_message("assistant"):
        with st.spinner("Đang xử lý truy vấn và sinh SQL..."):
            state = invoke_llm_database(user_input, provider=st.session_state.provider)

        row_count = _extract_field(state, "row_count", 0)
        result_query = _extract_field(state, "result_query", "")
        ai_messages = _extract_field(state, "ai_messages", "")
        query_sql = _extract_field(state, "query_sql", "")

        df = _parse_result_query_to_df(result_query)

        # 1. Hiển thị SQL được sinh ra (nếu có)
        if query_sql:
            st.markdown("**SQL được sinh ra:**")
            st.code(query_sql, language="sql")

        # 2. Hiển thị bảng nếu có
        if df is not None:
            st.dataframe(df, use_container_width=True)

        assistant_message: Dict[str, Any] = {
            "role": "assistant",
            "content": "",
            "table": df,
            "query_sql": query_sql,
        }

        # 3. Logic hiển thị theo số lượng rows
        if row_count <= 0:
            st.markdown("Không tìm thấy dữ liệu phù hợp.")
            assistant_message["content"] = "Không tìm thấy dữ liệu phù hợp."
        elif row_count < 20:
            # Trường hợp 1: < 20 rows -> hiển thị bảng + câu trả lời LLM
            if ai_messages:
                st.markdown(ai_messages)
                assistant_message["content"] = ai_messages
        else:
            # Trường hợp 2: ≥ 20 rows -> chỉ hiển thị bảng (không hiển thị LLM response)
            # Không thêm nội dung text để tuân thủ yêu cầu "chỉ in ra dạng table"
            pass

        st.session_state.messages.append(assistant_message)


if __name__ == "__main__":
    main()



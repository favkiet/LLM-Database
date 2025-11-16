from typing import Annotated
from pydantic import BaseModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages

class ChatState(BaseModel):
    """State definition for the chatbot graph"""
    provider: str
    human_messages: str
    ai_messages: str
    query_sql: str
    result_query: str
    row_count: int = 0  # Số lượng rows trong kết quả truy vấn
    csv_file_path: str = ""  # Đường dẫn file CSV nếu kết quả > 20 rows
    conversation: Annotated[list[BaseMessage], add_messages]
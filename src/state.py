from typing import Annotated
from pydantic import BaseModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages

class ChatState(BaseModel):
    """State definition for the chatbot graph"""
    human_messages: str
    ai_messages: str
    query_sql: str
    result_query: str
    conversation: Annotated[list[BaseMessage], add_messages]
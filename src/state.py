from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class ChatState(TypedDict):
    """State definition for the chatbot graph"""
    messages: Annotated[list[BaseMessage], add_messages]
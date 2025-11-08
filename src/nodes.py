from src.state import ChatState
from langchain_ollama import ChatOllama


def llm_node(state: ChatState):
    """Node that sends user messages to a local Ollama chat model and appends AI response."""
    messages = state["messages"]
    llm = ChatOllama(model="sqlcoder:7b", base_url="http://localhost:11434")
    response = llm.invoke(messages)
    return {"messages": [response]}
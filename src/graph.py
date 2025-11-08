from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from src.state import ChatState
from src.nodes import llm_node

graph = StateGraph(ChatState)

# add LLM node
graph.add_node("llm", llm_node)

# define edges
graph.add_edge(START, "llm")
graph.add_edge("llm", END)

graph = graph.compile()

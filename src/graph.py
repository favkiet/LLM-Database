from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from src.state import ChatState
from src.nodes import llm_response, db_query_node, text2sql_node
from langsmith import traceable
from src.logger_utils import logger
from dotenv import load_dotenv
load_dotenv()

graph = StateGraph(ChatState)

# add LLM node
graph.add_node("llm", llm_response)
graph.add_node("db_query_node", db_query_node)
graph.add_node("text2sql_node", text2sql_node)

# define edges
graph.add_edge(START, "text2sql_node")
graph.add_edge("text2sql_node", "db_query_node")
graph.add_edge("db_query_node", "llm")
graph.add_edge("llm", END)

graph = graph.compile()

@traceable(name="invoke-llm-database",
        tags=["llm-database"])
def invoke_llm_database(user_query: str):
    user_message = HumanMessage(content=user_query)
    logger.info("========= Human Message =========")
    logger.info(user_query)
    
    # Build initial state following ChatState schema
    initial_state: ChatState = {
        "conversation": [user_message],
        "human_messages": user_query,
        "ai_messages": "",
        "query_sql": "",
        "result_query": "",
    }

    response = graph.invoke(initial_state)
    respone_content = response["ai_messages"]
    logger.info("========= AI Message =========")
    logger.info(respone_content)
    # Return full state to preserve memory across calls
    return response

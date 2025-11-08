from src.graph import graph
from langchain_core.messages import HumanMessage

if __name__ == "__main__":
    
    print("Initializing functional chatbot...")
    print("\n" + "=" * 50)
    user_input = "Xin chào! Bạn là ai?"
    state = {
        "messages": [HumanMessage(content=user_input)],
    }
    for chunk in graph.stream(state):
        if messages := chunk['llm'].get('messages', []):
            print(messages[-1].content)
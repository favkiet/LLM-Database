from langgraph.graph import StateGraph

def get_graph_mermaid(graph: StateGraph) -> str:
    """Generate Mermaid diagram"""
    try:
        return graph.get_graph().draw_mermaid()
    except Exception as e:
        print(f"Could not generate Mermaid diagram: {e}")
        return ""
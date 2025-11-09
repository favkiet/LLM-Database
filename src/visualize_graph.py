from langgraph.graph import StateGraph
from IPython.display import display, Markdown, Image
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod

def get_graph_mermaid(graph: StateGraph, show: bool = True) -> str:
    """
    Generate and optionally display a Mermaid diagram from a LangGraph StateGraph.

    Args:
        graph (StateGraph): The LangGraph state graph.
        show (bool): If True, display the diagram directly in a notebook.

    Returns:
        str: The Mermaid syntax for the graph.
    """
    import nest_asyncio

    nest_asyncio.apply()
    try:
        mermaid_code = graph.get_graph().draw_mermaid()
        if show:
            display(Image(graph.get_graph().draw_mermaid_png(
                curve_style=CurveStyle.LINEAR,
                draw_method=MermaidDrawMethod.PYPPETEER,
            )))
        return mermaid_code
    except Exception as e:
        print(f"Could not generate Mermaid diagram: {e}")
        return ""
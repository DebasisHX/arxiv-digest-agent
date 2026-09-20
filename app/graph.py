from typing import Dict, Any

from langgraph.graph import StateGraph, START, END

from app.state import AgentState

from app.nodes.query import query_understanding_node
from app.nodes.arxiv import search_arxiv_node
from app.nodes.ranking import rank_papers_node
from app.nodes.pdf_parser import download_and_parse_pdf_node
from app.nodes.chunking import chunk_paper_node
from app.nodes.summarizer import summarizer_node


def create_vector_store_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a local FAISS vector store from the paper chunks.
    """

    from app.retrieval.vector_store import create_vector_store

    chunks = state.get("chunks", [])

    if not chunks:
        return {
            "vector_store_path": "",
            "error": state.get(
                "error",
                "No chunks available for vector store creation.",
            ),
        }

    try:
        paper = state.get("selected_paper", {})
        arxiv_id = paper.get("arxiv_id", "paper")

        # Make the ID safe for use as a directory name.
        store_name = arxiv_id.replace("/", "_")

        store_path = create_vector_store(
            chunks=chunks,
            store_name=store_name,
        )

        return {
            "vector_store_path": store_path,
            "error": "",
        }

    except Exception as exc:
        return {
            "vector_store_path": "",
            "error": f"Vector store creation failed: {exc}",
        }


def should_continue_after_query(
    state: Dict[str, Any],
) -> str:
    """
    Stop the graph if query understanding failed.
    """

    if state.get("error"):
        return "error"

    return "continue"


def should_continue_after_retrieval(
    state: Dict[str, Any],
) -> str:
    """
    Stop if arXiv retrieval failed or returned no papers.
    """

    if state.get("error") or not state.get("arxiv_results"):
        return "error"

    return "continue"


def should_continue_after_pdf(
    state: Dict[str, Any],
) -> str:
    """
    Stop if PDF download/parsing failed.
    """

    if state.get("error") or not state.get("parsed_text"):
        return "error"

    return "continue"


def should_continue_after_chunking(
    state: Dict[str, Any],
) -> str:
    """
    Stop if chunking failed.
    """

    if state.get("error") or not state.get("chunks"):
        return "error"

    return "continue"


def should_continue_after_vector_store(
    state: Dict[str, Any],
) -> str:
    """
    Stop if vector store creation failed.
    """

    if state.get("error") or not state.get("vector_store_path"):
        return "error"

    return "continue"


def build_graph():
    """
    Build and compile the complete arXiv digest state graph.
    """

    graph = StateGraph(AgentState)

    # -------------------------
    # Register nodes
    # -------------------------

    graph.add_node(
        "query_understanding",
        query_understanding_node,
    )

    graph.add_node(
        "arxiv_retrieval",
        search_arxiv_node,
    )

    graph.add_node(
        "paper_ranking",
        rank_papers_node,
    )

    graph.add_node(
        "pdf_parser",
        download_and_parse_pdf_node,
    )

    graph.add_node(
        "chunking",
        chunk_paper_node,
    )

    graph.add_node(
        "vector_store",
        create_vector_store_node,
    )

    graph.add_node(
        "summarizer",
        summarizer_node,
    )

    # -------------------------
    # Graph edges
    # -------------------------

    graph.add_edge(
        START,
        "query_understanding",
    )

    graph.add_conditional_edges(
        "query_understanding",
        should_continue_after_query,
        {
            "continue": "arxiv_retrieval",
            "error": END,
        },
    )

    graph.add_conditional_edges(
        "arxiv_retrieval",
        should_continue_after_retrieval,
        {
            "continue": "paper_ranking",
            "error": END,
        },
    )

    graph.add_edge(
        "paper_ranking",
        "pdf_parser",
    )

    graph.add_conditional_edges(
        "pdf_parser",
        should_continue_after_pdf,
        {
            "continue": "chunking",
            "error": END,
        },
    )

    graph.add_conditional_edges(
        "chunking",
        should_continue_after_chunking,
        {
            "continue": "vector_store",
            "error": END,
        },
    )

    graph.add_conditional_edges(
        "vector_store",
        should_continue_after_vector_store,
        {
            "continue": "summarizer",
            "error": END,
        },
    )

    graph.add_edge(
        "summarizer",
        END,
    )

    return graph.compile()
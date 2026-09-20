from typing import TypedDict, List, Dict, Any


class AgentState(TypedDict, total=False):
    # -------------------------
    # User input
    # -------------------------
    user_query: str
    query_type: str

    # -------------------------
    # arXiv retrieval
    # -------------------------
    arxiv_results: List[Dict[str, Any]]
    selected_paper: Dict[str, Any]

    # -------------------------
    # PDF processing
    # -------------------------
    pdf_path: str
    parsed_text: str
    chunks: List[str]

    # -------------------------
    # Vector store
    # -------------------------
    vector_store_path: str

    # -------------------------
    # Executive briefing
    # -------------------------
    briefing: Dict[str, Any]

    # -------------------------
    # QA
    # -------------------------
    current_question: str
    retrieved_chunks: List[str]
    answer: str
    conversation_history: List[Dict[str, str]]

    # -------------------------
    # Error handling
    # -------------------------
    error: str
import re
from typing import Dict, Any


ARXIV_ID_PATTERN = re.compile(
    r"^(?:https?://)?(?:www\.)?arxiv\.org/"
    r"(?:abs|pdf)/([a-zA-Z0-9.\-_/]+)"
    r"(?:\.pdf)?/?$"
)


def detect_query_type(user_query: str) -> str:
    """
    Determine whether the input is a specific arXiv paper
    or a natural-language research topic.
    """
    query = user_query.strip()

    if not query:
        raise ValueError("Query cannot be empty.")

    url_match = ARXIV_ID_PATTERN.match(query)

    if url_match:
        return "paper"

    if re.match(r"^\d{4}\.\d{4,5}(?:v\d+)?$", query):
        return "paper"

    if re.match(
        r"^[a-zA-Z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?$",
        query
    ):
        return "paper"

    return "topic"


def query_understanding_node(
    state: Dict[str, Any]
) -> Dict[str, Any]:
    """
    First graph node.

    Determines whether the user's input is a topic
    or a specific arXiv paper.
    """

    user_query = state.get("user_query", "").strip()

    query_type = detect_query_type(user_query)

    return {
        "user_query": user_query,
        "query_type": query_type,
        "error": "",
    }
from typing import Dict, Any


def rank_papers_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Select the most relevant paper from arXiv search results.
    """

    results = state.get("arxiv_results", [])
    query = state.get("user_query", "").lower().strip()
    query_type = state.get("query_type", "topic")

    if not results:
        return {
            "selected_paper": {},
            "error": state.get(
                "error",
                "No papers available for selection."
            ),
        }

    # Specific paper lookup
    if query_type == "paper":
        return {
            "selected_paper": results[0],
            "error": "",
        }

    stop_words = {
        "the", "a", "an", "and", "or", "for",
        "to", "of", "in", "on", "with", "from",
        "recent", "work", "research", "paper", "about"
    }

    query_terms = {
        word.strip(".,!?():;")
        for word in query.replace("-", " ").split()
        if word not in stop_words and len(word) > 2
    }

    scored_results = []

    for paper in results:
        title = paper.get("title", "").lower()
        abstract = paper.get("abstract", "").lower()

        title_terms = set(title.replace("-", " ").split())
        abstract_terms = set(abstract.replace("-", " ").split())

        title_score = len(query_terms & title_terms)
        abstract_score = len(query_terms & abstract_terms)

        score = (title_score * 3) + abstract_score

        scored_results.append((score, paper))

    scored_results.sort(
        key=lambda item: item[0],
        reverse=True
    )

    selected_paper = scored_results[0][1]

    return {
        "selected_paper": selected_paper,
        "error": "",
    }

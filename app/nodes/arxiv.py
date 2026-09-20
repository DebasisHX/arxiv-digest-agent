import arxiv
from typing import Dict, Any


def search_arxiv_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Search the official arXiv API for papers matching the user's query.
    For a specific paper ID, retrieve that paper directly.
    """

    user_query = state["user_query"].strip()
    query_type = state["query_type"]

    try:
        if query_type == "paper":
            # Extract the arXiv ID from either an ID or URL
            paper_id = user_query.rstrip("/").split("/")[-1]
            paper_id = paper_id.replace(".pdf", "")

            search = arxiv.Search(id_list=[paper_id])

        else:
            # Natural-language topic search
            search = arxiv.Search(
                query=user_query,
                max_results=10,
                sort_by=arxiv.SortCriterion.Relevance,
            )

        client = arxiv.Client(
            page_size=10,
            delay_seconds=3.0,
            num_retries=3,
        )

        results = []

        for result in client.results(search):
            results.append(
                {
                    "arxiv_id": result.get_short_id(),
                    "title": result.title,
                    "authors": [author.name for author in result.authors],
                    "abstract": result.summary,
                    "pdf_url": result.pdf_url,
                    "arxiv_url": result.entry_id,
                    "published": result.published.isoformat(),
                    "updated": result.updated.isoformat(),
                    "categories": result.categories,
                }
            )

        if not results:
            return {
                "arxiv_results": [],
                "error": (
                    "No papers were found on arXiv for the given query."
                ),
            }

        return {
            "arxiv_results": results,
            "error": "",
        }

    except Exception as exc:
        return {
            "arxiv_results": [],
            "error": f"arXiv retrieval failed: {exc}",
        }
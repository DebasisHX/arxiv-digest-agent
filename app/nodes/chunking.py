from typing import Dict, Any, List


CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


def create_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Split paper text into overlapping character-based chunks.

    Overlap helps preserve context when an important sentence
    falls near a chunk boundary.
    """

    if not text or not text.strip():
        return []

    text = text.strip()

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = end - chunk_overlap

    return chunks


def chunk_paper_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create RAG-ready chunks from the parsed paper text.
    """

    parsed_text = state.get("parsed_text", "")

    if not parsed_text:
        return {
            "chunks": [],
            "error": state.get(
                "error",
                "No parsed text available for chunking."
            ),
        }

    try:
        chunks = create_chunks(parsed_text)

        if not chunks:
            return {
                "chunks": [],
                "error": "No usable chunks could be created from the paper.",
            }

        return {
            "chunks": chunks,
            "error": "",
        }

    except Exception as exc:
        return {
            "chunks": [],
            "error": f"Chunking failed: {exc}",
        }

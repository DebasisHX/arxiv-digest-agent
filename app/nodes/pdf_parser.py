import os
import requests
import pymupdf
from typing import Dict, Any

PAPERS_DIR = "data/papers"


def download_and_parse_pdf_node(state: Dict[str, Any]) -> Dict[str, Any]:
    paper = state.get("selected_paper", {})

    if not paper:
        return {"error": "No paper was selected for PDF processing."}

    pdf_url = paper.get("pdf_url")
    arxiv_id = paper.get("arxiv_id", "paper")

    if not pdf_url:
        return {"error": "Selected paper does not contain a PDF URL."}

    os.makedirs(PAPERS_DIR, exist_ok=True)

    safe_id = arxiv_id.replace("/", "_")
    pdf_path = os.path.join(PAPERS_DIR, f"{safe_id}.pdf")

    try:
        response = requests.get(
            pdf_url,
            timeout=60,
            headers={"User-Agent": "arxiv-digest-agent/1.0"}
        )

        response.raise_for_status()

        content_type = response.headers.get("content-type", "").lower()

        if "pdf" not in content_type and not response.content.startswith(b"%PDF"):
            return {
                "error": "Downloaded file does not appear to be a valid PDF."
            }

        with open(pdf_path, "wb") as file:
            file.write(response.content)

        document = pymupdf.open(pdf_path)

        pages_text = []

        for page_number, page in enumerate(document):
            text = page.get_text("text")

            if text.strip():
                pages_text.append(
                    f"\n--- Page {page_number + 1} ---\n{text}"
                )

        document.close()

        parsed_text = "\n".join(pages_text).strip()

        if not parsed_text:
            return {
                "pdf_path": pdf_path,
                "parsed_text": "",
                "error": (
                    "The PDF contains no extractable text. "
                    "It may be scanned or have an unsupported layout."
                )
            }

        return {
            "pdf_path": pdf_path,
            "parsed_text": parsed_text,
            "error": ""
        }

    except requests.RequestException as exc:
        return {
            "error": f"PDF download failed: {exc}"
        }

    except Exception as exc:
        return {
            "error": f"PDF parsing failed: {exc}"
        }

import json
from typing import Dict, Any, List

from dotenv import load_dotenv
from groq import Groq


load_dotenv()


MODEL_NAME = "openai/gpt-oss-120b"

# Keep individual requests small enough for the Groq TPM limit.
CHUNK_SIZE = 3000


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE
) -> List[str]:
    """
    Split the paper into small sections so that each
    LLM request stays within the API token limit.
    """

    if not text or not text.strip():
        return []

    text = text.strip()

    chunks = []

    for i in range(
        0,
        len(text),
        chunk_size
    ):
        chunk = text[
            i:i + chunk_size
        ].strip()

        if chunk:
            chunks.append(chunk)

    return chunks


def summarize_chunk(
    client: Groq,
    chunk: str,
    chunk_number: int
) -> str:
    """
    Summarize one section of the paper.
    """

    prompt = f"""
You are analyzing section {chunk_number}
of an academic research paper.

Summarize ONLY information explicitly present
in the supplied section.

Focus on:

- research problem or motivation
- proposed method
- architecture
- experiments
- results
- important claims
- limitations

Rules:

- Do not use outside knowledge.
- Do not invent information.
- Do not guess missing details.
- Keep the summary concise.
- Return plain text only.

PAPER SECTION
========================
{chunk}
========================
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You produce concise and factual summaries "
                    "grounded only in the supplied paper text."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.1,
        max_tokens=500
    )

    return response.choices[0].message.content.strip()


def build_final_prompt(
    paper: Dict[str, Any],
    summaries: List[str]
) -> str:

    combined_summaries = "\n\n".join(
        [
            f"[Section Summary {i + 1}]\n{summary}"
            for i, summary in enumerate(summaries)
        ]
    )

    return f"""
You are an academic research assistant.

Create an accurate executive briefing of the
arXiv paper using ONLY the section summaries below.

IMPORTANT RULES:

- Do not use outside knowledge.
- Do not invent results, methods, or claims.
- Use only information supported by the summaries.
- Explicitly state limitations.
- Clearly distinguish reported results from interpretation.
- If information is unavailable, say so.
- Return ONLY valid JSON.
- Do not use markdown code fences.

PAPER METADATA

Title:
{paper.get("title", "")}

Authors:
{", ".join(paper.get("authors", []))}

arXiv ID:
{paper.get("arxiv_id", "")}

Published:
{paper.get("published", "")}

arXiv URL:
{paper.get("arxiv_url", "")}


SECTION SUMMARIES
========================

{combined_summaries}

========================

Return exactly this JSON structure:

{{
    "title": "...",
    "authors": [
        "..."
    ],
    "arxiv_id": "...",
    "publish_date": "...",
    "link": "...",
    "summary": "...",
    "problem_statement": "...",
    "method": [
        "...",
        "..."
    ],
    "key_results": [
        "...",
        "..."
    ],
    "limitations": [
        "...",
        "..."
    ],
    "follow_up_questions": [
        "...",
        "...",
        "..."
    ]
}}
"""


def summarizer_node(
    state: Dict[str, Any]
) -> Dict[str, Any]:

    paper = state.get(
        "selected_paper",
        {}
    )

    paper_text = state.get(
        "parsed_text",
        ""
    )

    if not paper:

        return {
            "briefing": {},
            "error": (
                "No selected paper available "
                "for summarization."
            )
        }

    if not paper_text:

        return {
            "briefing": {},
            "error": (
                "No parsed paper text available "
                "for summarization."
            )
        }

    try:

        client = Groq()

        # Split the paper into small pieces.
        chunks = split_text(
            paper_text
        )

        if not chunks:

            return {
                "briefing": {},
                "error": (
                    "Paper could not be split "
                    "into summarization sections."
                )
            }

        print(
            f"Paper split into {len(chunks)} sections."
        )

        # Map step:
        # Summarize each section independently.
        summaries = []

        for i, chunk in enumerate(
            chunks,
            start=1
        ):

            print(
                f"Summarizing section "
                f"{i}/{len(chunks)}..."
            )

            summary = summarize_chunk(
                client=client,
                chunk=chunk,
                chunk_number=i
            )

            summaries.append(
                summary
            )

        # Reduce step:
        # Combine section summaries into
        # one structured executive briefing.
        print(
            "Combining section summaries..."
        )

        final_prompt = build_final_prompt(
            paper=paper,
            summaries=summaries
        )

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You create accurate academic "
                        "briefings using only supplied evidence."
                    )
                },
                {
                    "role": "user",
                    "content": final_prompt
                }
            ],
            temperature=0.1,
            max_tokens=1500
        )

        content = (
            response
            .choices[0]
            .message
            .content
            .strip()
        )

        # Remove markdown JSON fences if present.
        if content.startswith("```"):

            content = content.replace(
                "```json",
                "",
                1
            )

            content = content.replace(
                "```",
                ""
            )

            content = content.strip()

        briefing = json.loads(
            content
        )

        return {
            "briefing": briefing,
            "error": ""
        }

    except json.JSONDecodeError as exc:

        return {
            "briefing": {},
            "error": (
                "LLM returned invalid briefing JSON: "
                f"{exc}"
            )
        }

    except Exception as exc:

        return {
            "briefing": {},
            "error": (
                "Briefing generation failed: "
                f"{exc}"
            )
        }
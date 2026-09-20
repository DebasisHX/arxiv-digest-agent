from typing import Dict, Any

from dotenv import load_dotenv
from groq import Groq

from app.retrieval.vector_store import search_vector_store


load_dotenv()

MODEL_NAME = "openai/gpt-oss-120b"


def build_qa_prompt(question: str, retrieved_chunks: list) -> str:
    context = "\n\n".join(
        [
            f"[Paper Excerpt {i + 1}]\n{chunk}"
            for i, chunk in enumerate(retrieved_chunks)
        ]
    )

    return f"""
You are a research paper QA assistant.

Answer the user's question using ONLY the provided excerpts
from the research paper.

STRICT GROUNDING RULES:
1. Do not use outside knowledge.
2. Do not invent information.
3. If the answer cannot be determined from the provided excerpts,
   say exactly:
   "I couldn't find this information in the paper."
4. Only make claims supported by the excerpts.
5. Do not mention source chunk numbers, citation markers,
   line numbers, or internal retrieval details in your answer.
6. Be concise but technically accurate.
7. If the excerpts contain the answer, explain it clearly.
8. If the excerpts do not contain enough information, say that
   the information could not be found.

PAPER EXCERPTS
========================
{context}
========================

USER QUESTION:
{question}

ANSWER:
"""


def qa_node(state: Dict[str, Any]) -> Dict[str, Any]:

    question = state.get("current_question", "").strip()

    vector_store_path = state.get(
        "vector_store_path",
        ""
    )

    if not question:
        return {
            "answer": "Please enter a question.",
            "retrieved_chunks": [],
            "error": ""
        }

    if not vector_store_path:
        return {
            "answer": "",
            "retrieved_chunks": [],
            "error": "Vector store is not available."
        }

    try:

        # Retrieve the most relevant paper chunks.
        search_results = search_vector_store(
            store_dir=vector_store_path,
            query=question,
            top_k=5
        )

        if not search_results:
            answer = "I couldn't find this information in the paper."

            history = list(
                state.get("conversation_history", [])
            )

            history.append(
                {
                    "question": question,
                    "answer": answer
                }
            )

            return {
                "current_question": question,
                "retrieved_chunks": [],
                "answer": answer,
                "conversation_history": history,
                "error": ""
            }

        retrieved_chunks = [
            chunk
            for chunk, score in search_results
        ]

        client = Groq()

        prompt = build_qa_prompt(
            question=question,
            retrieved_chunks=retrieved_chunks
        )

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You answer research paper questions "
                        "using only the provided evidence."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.0,
            max_tokens=700
        )

        answer = response.choices[0].message.content.strip()

        # Remove accidental markdown fences.
        if answer.startswith("```"):
            answer = answer.replace("```", "").strip()

        history = list(
            state.get("conversation_history", [])
        )

        history.append(
            {
                "question": question,
                "answer": answer
            }
        )

        return {
            "current_question": question,
            "retrieved_chunks": retrieved_chunks,
            "answer": answer,
            "conversation_history": history,
            "error": ""
        }

    except Exception as exc:

        return {
            "answer": "",
            "retrieved_chunks": [],
            "error": f"QA failed: {exc}"
        }
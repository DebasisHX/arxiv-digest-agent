
import json

from dotenv import load_dotenv

from app.graph import build_graph
from app.nodes.qa import qa_node

load_dotenv()


def print_briefing(briefing):
    print("\n" + "=" * 70)
    print("ARXIV PAPER DIGEST")
    print("=" * 70)

    print(f"\nTitle: {briefing.get('title', 'N/A')}")
    print(f"Authors: {', '.join(briefing.get('authors', []))}")
    print(f"arXiv ID: {briefing.get('arxiv_id', 'N/A')}")
    print(f"Published: {briefing.get('publish_date', 'N/A')}")
    print(f"Link: {briefing.get('link', 'N/A')}")

    print("\nWHY THIS MATTERS")
    print("-" * 70)
    print(briefing.get("summary", "Not available."))

    print("\nPROBLEM")
    print("-" * 70)
    print(briefing.get("problem_statement", "Not available."))

    print("\nMETHOD")
    print("-" * 70)
    for item in briefing.get("method", []):
        print(f"• {item}")

    print("\nKEY RESULTS")
    print("-" * 70)
    for item in briefing.get("key_results", []):
        print(f"• {item}")

    print("\nLIMITATIONS")
    print("-" * 70)
    for item in briefing.get("limitations", []):
        print(f"• {item}")

    print("\nFOLLOW-UP QUESTIONS")
    print("-" * 70)
    for item in briefing.get("follow_up_questions", []):
        print(f"• {item}")

    print("\n" + "=" * 70)


def run_agent(user_query: str):
    graph = build_graph()

    state = {
        "user_query": user_query,
        "conversation_history": []
    }

    print("\nProcessing your request...")
    print("Retrieving paper from arXiv...")

    result = graph.invoke(state)

    if result.get("error"):
        print("\nERROR:")
        print(result["error"])
        return

    selected_paper = result.get("selected_paper", {})
    briefing = result.get("briefing", {})

    if not selected_paper:
        print("\nNo paper was selected.")
        return

    print(
        f"\nSelected paper: "
        f"{selected_paper.get('title', 'Unknown')}"
    )

    if not briefing:
        print("\nBriefing generation failed.")
        return

    print_briefing(briefing)

    vector_store_path = result.get(
        "vector_store_path",
        ""
    )

    print("\nQA MODE")
    print("-" * 70)
    print("Ask questions about the paper.")
    print("Type 'exit' to finish.")

    conversation_history = []

    while True:
        question = input("\nYour question: ").strip()

        if question.lower() in {"exit", "quit", "q"}:
            print("\nGoodbye!")
            break

        if not question:
            continue

        qa_state = {
            "current_question": question,
            "vector_store_path": vector_store_path,
            "conversation_history": conversation_history
        }

        qa_result = qa_node(qa_state)

        if qa_result.get("error"):
            print("\nQA Error:")
            print(qa_result["error"])
            continue

        print("\nAnswer:")
        print(qa_result.get("answer", ""))

        conversation_history = qa_result.get(
            "conversation_history",
            conversation_history
        )


def main():
    print("=" * 70)
    print("AUTONOMOUS arXiv PAPER DIGEST & QA AGENT")
    print("=" * 70)

    print("\nEnter an arXiv topic, paper ID, or arXiv URL.")

    user_query = input("\nQuery: ").strip()

    if not user_query:
        print("Query cannot be empty.")
        return

    run_agent(user_query)


if __name__ == "__main__":
    main()

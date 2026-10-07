from __future__ import annotations

from proofrag.pipeline import PipelineError, ProofRAG


def print_response(title: str, response) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    print(f"\nAnswer:\n{response.answer}")
    print(f"\nVerified: {response.verified}")
    print(f"Refused: {response.refused}")
    print(f"Evidence count: {response.evidence_count}")

    if response.refusal_reason:
        print(f"Refusal reason: {response.refusal_reason}")

    if response.citations:
        print("\nCitations:")

        for citation in response.citations:
            print(f"\n  Source: {citation.filename}")
            print(f"  Page: {citation.page_number}")
            print(
                f"  Status: "
                f"{citation.verification_status.value}"
            )
            print(
                f"  Keyword score: "
                f"{citation.keyword_score:.3f}"
            )
            print(
                f"  Embedding score: "
                f"{citation.embedding_score:.3f}"
            )
            print(f"  Evidence: {citation.snippet}")


def main() -> None:
    rag = ProofRAG()

    print("ProofRAG — End-to-End Demonstration")
    print()

    try:
        print("Indexing documents...")
        documents = rag.ingest()

        print(f"Indexed {len(documents)} document(s).")

        for document in documents:
            print(f"  - {document.filename}")

        supported_question = (
            "What does ProofRAG use for semantic retrieval?"
        )

        response = rag.ask(supported_question)

        print_response(
            "SUPPORTED QUESTION",
            response,
        )

        unsupported_question = (
            "How many employees does ProofRAG have?"
        )

        response = rag.ask(unsupported_question)

        print_response(
            "UNSUPPORTED / TRICK QUESTION",
            response,
        )

    except PipelineError as exc:
        print(f"\nPipeline error: {exc}")
        raise


if __name__ == "__main__":
    main()
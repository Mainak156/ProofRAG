from __future__ import annotations

import pytest

from proofrag.generator import (
    AnswerGenerator,
    GenerationError,
    build_generation_messages,
)
from proofrag.models import Evidence


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeLLM:
    def __init__(self, content: str) -> None:
        self.content = content
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return FakeResponse(self.content)


def make_evidence(
    evidence_id: str = "evidence-1",
) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        chunk_id="chunk-1",
        document_id="doc-1",
        filename="manual.pdf",
        page_number=3,
        snippet="The system supports PDF ingestion.",
        retrieval_score=0.91,
    )


def test_generation_prompt_contains_question_and_evidence() -> None:
    evidence = make_evidence()

    messages = build_generation_messages(
        "What does the system support?",
        [evidence],
    )

    combined = "\n".join(
        str(message.content)
        for message in messages
    )

    assert "What does the system support?" in combined
    assert "evidence-1" in combined
    assert "manual.pdf" in combined
    assert "Page: 3" in combined
    assert "The system supports PDF ingestion." in combined


def test_no_evidence_refuses_without_calling_llm() -> None:
    llm = FakeLLM(
        '{"answer":"This should never be called.","citations":[]}'
    )

    generator = AnswerGenerator(llm=llm)

    result = generator.generate(
        "What does the system support?",
        [],
    )

    assert result.refused is True
    assert result.text == "Not found in the documents."
    assert result.citations == []
    assert result.refusal_reason is not None


def test_valid_response_creates_verified_ready_citations() -> None:
    evidence = make_evidence()

    llm = FakeLLM(
        '{"answer":"The system supports PDF ingestion.",'
        '"citations":["evidence-1"]}'
    )

    generator = AnswerGenerator(llm=llm)

    result = generator.generate(
        "What does the system support?",
        [evidence],
    )

    assert result.refused is False
    assert result.text == "The system supports PDF ingestion."
    assert len(result.citations) == 1

    citation = result.citations[0]

    assert citation.evidence_id == "evidence-1"
    assert citation.filename == "manual.pdf"
    assert citation.page_number == 3
    assert citation.snippet == "The system supports PDF ingestion."


def test_unknown_citation_id_is_rejected() -> None:
    evidence = make_evidence()

    llm = FakeLLM(
        '{"answer":"The system supports PDF ingestion.",'
        '"citations":["evidence-does-not-exist"]}'
    )

    generator = AnswerGenerator(llm=llm)

    with pytest.raises(
        GenerationError,
        match="citation IDs that were not supplied",
    ):
        generator.generate(
            "What does the system support?",
            [evidence],
        )


def test_invalid_json_is_rejected() -> None:
    evidence = make_evidence()

    llm = FakeLLM(
        "The system supports PDF ingestion."
    )

    generator = AnswerGenerator(llm=llm)

    with pytest.raises(
        GenerationError,
        match="invalid JSON",
    ):
        generator.generate(
            "What does the system support?",
            [evidence],
        )


def test_fenced_json_is_supported() -> None:
    evidence = make_evidence()

    llm = FakeLLM(
        """```json
{
  "answer": "The system supports PDF ingestion.",
  "citations": ["evidence-1"]
}
```"""
    )

    generator = AnswerGenerator(llm=llm)

    result = generator.generate(
        "What does the system support?",
        [evidence],
    )

    assert result.text == "The system supports PDF ingestion."
    assert len(result.citations) == 1


def test_duplicate_citations_are_deduplicated() -> None:
    evidence = make_evidence()

    llm = FakeLLM(
        '{"answer":"The system supports PDF ingestion.",'
        '"citations":["evidence-1","evidence-1"]}'
    )

    generator = AnswerGenerator(llm=llm)

    result = generator.generate(
        "What does the system support?",
        [evidence],
    )

    assert len(result.citations) == 1


def test_refusal_response_is_mapped_correctly() -> None:
    evidence = make_evidence()

    llm = FakeLLM(
        '{"answer":"Not found in the documents.","citations":[]}'
    )

    generator = AnswerGenerator(llm=llm)

    result = generator.generate(
        "What is the company's revenue?",
        [evidence],
    )

    assert result.refused is True
    assert result.text == "Not found in the documents."
    assert result.citations == []
    assert result.refusal_reason is not None


def test_llm_receives_only_evidence_not_original_document() -> None:
    evidence = make_evidence()

    llm = FakeLLM(
        '{"answer":"The system supports PDF ingestion.",'
        '"citations":["evidence-1"]}'
    )

    generator = AnswerGenerator(llm=llm)

    generator.generate(
        "What does the system support?",
        [evidence],
    )

    assert llm.messages is not None

    prompt = "\n".join(
        str(message.content)
        for message in llm.messages
    )

    assert "manual.pdf" in prompt
    assert "The system supports PDF ingestion." in prompt
    assert "original document" not in prompt.lower()
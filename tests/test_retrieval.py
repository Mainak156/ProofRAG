from pathlib import Path

import fitz
import pytest

from proofrag.embeddings import embed_text
from proofrag.ingestion import ingest_pdf
from proofrag.okf import OKFBundle
from proofrag.retrieval import (
    _chunk_to_langchain_document,
    _distance_to_relevance,
)


def _create_test_pdf(path: Path) -> None:
    """Create a small PDF for retrieval tests."""

    document = fitz.open()

    page_one = document.new_page()

    page_one.insert_text(
        (72, 72),
        (
            "Retrieval augmented generation uses retrieved evidence "
            "to ground language model responses."
        ),
    )

    page_two = document.new_page()

    page_two.insert_text(
        (72, 72),
        (
            "Citation verification checks whether the evidence "
            "actually supports the generated answer."
        ),
    )

    document.save(path)
    document.close()


def test_embedding_model_generates_vector() -> None:
    vector = embed_text(
        "ProofRAG retrieves evidence."
    )

    assert isinstance(vector, list)
    assert len(vector) > 0

    assert all(
        isinstance(value, float)
        for value in vector
    )


def test_embedding_vector_is_normalized() -> None:
    vector = embed_text(
        "Evidence-backed retrieval."
    )

    magnitude = sum(
        value * value
        for value in vector
    ) ** 0.5

    assert magnitude == pytest.approx(
        1.0,
        abs=0.01,
    )


def test_chunk_conversion_preserves_metadata(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "manual.pdf"

    _create_test_pdf(pdf_path)

    _, chunks = ingest_pdf(
        pdf_path
    )

    document = _chunk_to_langchain_document(
        chunks[0]
    )

    assert document.page_content == chunks[0].text

    assert (
        document.metadata["chunk_id"]
        == chunks[0].chunk_id
    )

    assert (
        document.metadata["document_id"]
        == chunks[0].document_id
    )

    assert (
        document.metadata["source_filename"]
        == "manual.pdf"
    )

    assert (
        document.metadata["page_number"]
        == 1
    )


def test_distance_conversion() -> None:
    assert _distance_to_relevance(0.0) == 1.0
    assert _distance_to_relevance(0.25) == 0.75
    assert _distance_to_relevance(1.0) == 0.0


def test_distance_conversion_is_clamped() -> None:
    assert _distance_to_relevance(-1.0) == 1.0
    assert _distance_to_relevance(2.0) == 0.0


def test_okf_bundle_contains_ingested_chunks(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "manual.pdf"

    _create_test_pdf(pdf_path)

    document, chunks = ingest_pdf(
        pdf_path
    )

    bundle = OKFBundle(
        tmp_path / "okf"
    )

    bundle.write_chunks(
        document,
        chunks,
    )

    files = bundle.list_chunks()

    assert len(files) == len(chunks)

    for chunk in chunks:
        frontmatter, body = bundle.read_chunk(
            chunk.chunk_id
        )

        assert (
            frontmatter["proofrag"]["chunk_id"]
            == chunk.chunk_id
        )

        assert (
            frontmatter["proofrag"]["document_id"]
            == document.document_id
        )

        assert chunk.text in body
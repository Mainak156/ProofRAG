from pathlib import Path

import fitz
import pytest

from proofrag.ingestion import (
    DocumentIngestionError,
    _clean_text,
    _split_text,
    ingest_document,
    ingest_pdf,
)
from proofrag.models import SourceType


def _create_test_pdf(path: Path) -> None:
    """Create a small two-page PDF for ingestion tests."""

    document = fitz.open()

    page_one = document.new_page()
    page_one.insert_text(
        (72, 72),
        (
            "ProofRAG is an evidence-backed retrieval system. "
            "Every answer should be connected to source evidence."
        ),
    )

    page_two = document.new_page()
    page_two.insert_text(
        (72, 72),
        (
            "Citation verification checks whether retrieved evidence "
            "actually supports the generated answer."
        ),
    )

    document.save(path)
    document.close()


def test_clean_text_removes_excess_whitespace() -> None:
    text = "Hello   world\r\n\r\n\r\nProofRAG"

    result = _clean_text(text)

    assert result == "Hello world\n\nProofRAG"


def test_split_text_creates_chunks() -> None:
    text = "A" * 250

    chunks = _split_text(
        text,
        chunk_size=100,
        chunk_overlap=20,
    )

    assert len(chunks) > 1
    assert all(chunk[0] for chunk in chunks)


def test_split_text_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError):
        _split_text(
            "Some text",
            chunk_size=100,
            chunk_overlap=100,
        )


def test_ingest_pdf_preserves_page_metadata(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"

    _create_test_pdf(pdf_path)

    document, chunks = ingest_pdf(
        pdf_path,
        chunk_size=500,
        chunk_overlap=50,
    )

    assert document.source_type == SourceType.PDF
    assert document.filename == "sample.pdf"
    assert document.page_count == 2

    assert len(chunks) == 2

    assert chunks[0].page_number == 1
    assert chunks[1].page_number == 2

    assert chunks[0].source_filename == "sample.pdf"
    assert chunks[1].source_filename == "sample.pdf"

    assert chunks[0].document_id == document.document_id
    assert chunks[1].document_id == document.document_id


def test_chunk_ids_are_unique(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"

    _create_test_pdf(pdf_path)

    _, chunks = ingest_pdf(
        pdf_path,
        chunk_size=80,
        chunk_overlap=10,
    )

    chunk_ids = [chunk.chunk_id for chunk in chunks]

    assert len(chunk_ids) == len(set(chunk_ids))


def test_ingest_document_supports_markdown(tmp_path: Path) -> None:
    markdown_path = tmp_path / "notes.md"

    markdown_path.write_text(
        "# ProofRAG\n\nEvidence-backed retrieval.",
        encoding="utf-8",
    )

    document, chunks = ingest_document(markdown_path)

    assert document.source_type == SourceType.MARKDOWN
    assert document.page_count == 1
    assert len(chunks) == 1
    assert "Evidence-backed retrieval" in chunks[0].text


def test_ingest_document_rejects_unsupported_file(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "data.csv"

    file_path.write_text(
        "name,value\nA,1",
        encoding="utf-8",
    )

    with pytest.raises(DocumentIngestionError):
        ingest_document(file_path)


def test_ingest_document_rejects_missing_file(
    tmp_path: Path,
) -> None:
    missing_file = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError):
        ingest_document(missing_file)
from datetime import datetime, timezone
from pathlib import Path

import fitz

from proofrag.ingestion import ingest_pdf
from proofrag.models import SourceType
from proofrag.okf import (
    OKFBundle,
    OKFValidationError,
    parse_okf_document,
)


def _create_test_pdf(path: Path) -> None:
    """Create a deterministic PDF for OKF tests."""

    document = fitz.open()

    page = document.new_page()

    page.insert_text(
        (72, 72),
        (
            "ProofRAG converts source documents into evidence-backed "
            "knowledge that can be retrieved and verified."
        ),
    )

    document.save(path)
    document.close()


def test_build_chunk_node(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "manual.pdf"

    _create_test_pdf(pdf_path)

    document, chunks = ingest_pdf(pdf_path)

    bundle = OKFBundle(
        tmp_path / "okf"
    )

    node = bundle.build_chunk_node(
        document,
        chunks[0],
    )

    assert node.node_id.startswith("chunks/")
    assert node.node_type == "document-chunk"
    assert node.source_document_id == document.document_id
    assert node.source_filename == "manual.pdf"
    assert node.page_number == 1


def test_rendered_chunk_has_okf_frontmatter(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "manual.pdf"

    _create_test_pdf(pdf_path)

    document, chunks = ingest_pdf(pdf_path)

    bundle = OKFBundle(
        tmp_path / "okf"
    )

    generated_at = datetime(
        2026,
        10,
        6,
        12,
        0,
        tzinfo=timezone.utc,
    )

    content = bundle.render_chunk(
        document,
        chunks[0],
        generated_at=generated_at,
    )

    frontmatter, body = parse_okf_document(
        content
    )

    assert frontmatter["type"] == "document-chunk"
    assert frontmatter["title"]
    assert frontmatter["resource"]
    assert frontmatter["status"] == "stable"

    assert frontmatter["generated"]["by"] == (
        "proofrag/0.1.0"
    )

    assert frontmatter["generated"]["at"] == (
        "2026-10-06T12:00:00Z"
    )

    assert len(frontmatter["sources"]) == 1
    assert frontmatter["sources"][0]["id"]

    assert chunks[0].text in body


def test_write_chunk_creates_markdown_file(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "manual.pdf"

    _create_test_pdf(pdf_path)

    document, chunks = ingest_pdf(pdf_path)

    bundle = OKFBundle(
        tmp_path / "okf"
    )

    path = bundle.write_chunk(
        document,
        chunks[0],
    )

    assert path.exists()
    assert path.suffix == ".md"

    content = path.read_text(
        encoding="utf-8"
    )

    assert content.startswith("---")
    assert "# Evidence" in content


def test_write_chunks_creates_index(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "manual.pdf"

    _create_test_pdf(pdf_path)

    document, chunks = ingest_pdf(pdf_path)

    bundle = OKFBundle(
        tmp_path / "okf"
    )

    paths = bundle.write_chunks(
        document,
        chunks,
    )

    assert len(paths) == len(chunks)

    index_path = (
        tmp_path / "okf" / "index.md"
    )

    assert index_path.exists()

    index = index_path.read_text(
        encoding="utf-8"
    )

    assert "ProofRAG Knowledge Bundle" in index
    assert "manual.pdf" in index

    for chunk in chunks:
        assert chunk.chunk_id in index


def test_read_chunk_round_trip(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "manual.pdf"

    _create_test_pdf(pdf_path)

    document, chunks = ingest_pdf(pdf_path)

    bundle = OKFBundle(
        tmp_path / "okf"
    )

    bundle.write_chunk(
        document,
        chunks[0],
    )

    frontmatter, body = bundle.read_chunk(
        chunks[0].chunk_id
    )

    assert frontmatter["type"] == "document-chunk"
    assert chunks[0].text in body


def test_validate_valid_bundle(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "manual.pdf"

    _create_test_pdf(pdf_path)

    document, chunks = ingest_pdf(pdf_path)

    bundle = OKFBundle(
        tmp_path / "okf"
    )

    bundle.write_chunks(
        document,
        chunks,
    )

    errors = bundle.validate()

    assert errors == []


def test_validate_detects_missing_type(
    tmp_path: Path,
) -> None:
    bundle = OKFBundle(
        tmp_path / "okf"
    )

    bundle.write_chunk = bundle.write_chunk

    invalid_file = (
        bundle.chunks_dir / "invalid.md"
    )

    invalid_file.write_text(
        """---
title: Invalid Concept
---

# Evidence

Some evidence.
""",
        encoding="utf-8",
    )

    errors = bundle.validate()

    assert any(
        "missing required 'type'" in error
        for error in errors
    )


def test_parse_okf_requires_frontmatter() -> None:
    try:
        parse_okf_document(
            "# Not an OKF document"
        )
    except OKFValidationError:
        pass
    else:
        raise AssertionError(
            "Expected OKFValidationError"
        )


def test_list_chunks(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "manual.pdf"

    _create_test_pdf(pdf_path)

    document, chunks = ingest_pdf(pdf_path)

    bundle = OKFBundle(
        tmp_path / "okf"
    )

    bundle.write_chunks(
        document,
        chunks,
    )

    files = bundle.list_chunks()

    assert len(files) == len(chunks)
    assert all(
        path.suffix == ".md"
        for path in files
    )


def test_document_type_is_preserved(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "manual.pdf"

    _create_test_pdf(pdf_path)

    document, _ = ingest_pdf(pdf_path)

    assert document.source_type == SourceType.PDF
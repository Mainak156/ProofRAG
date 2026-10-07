from __future__ import annotations

import hashlib
import re
from pathlib import Path

import fitz

from .models import Chunk, Document, SourceType


SUPPORTED_EXTENSIONS = {
    ".pdf": SourceType.PDF,
    ".txt": SourceType.TEXT,
    ".md": SourceType.MARKDOWN,
}


class DocumentIngestionError(Exception):
    """Raised when a document cannot be ingested."""


def _generate_document_id(path: Path) -> str:
    """Generate a deterministic document ID from the file path and metadata."""

    stat = path.stat()

    raw = (
        f"{path.resolve()}|"
        f"{stat.st_size}|"
        f"{stat.st_mtime_ns}"
    )

    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _generate_chunk_id(
    document_id: str,
    page_number: int | None,
    chunk_index: int,
    text: str,
) -> str:
    """Generate a deterministic chunk ID."""

    raw = (
        f"{document_id}|"
        f"{page_number}|"
        f"{chunk_index}|"
        f"{text}"
    )

    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _clean_text(text: str) -> str:
    """Normalize extracted document text."""

    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Collapse excessive whitespace while preserving paragraphs.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _split_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[tuple[str, int, int]]:
    """
    Split text into overlapping chunks.

    Returns:
        List of (chunk_text, start_char, end_char).
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    if not text.strip():
        return []

    chunks: list[tuple[str, int, int]] = []

    start = 0
    text_length = len(text)

    while start < text_length:
        target_end = min(start + chunk_size, text_length)

        if target_end < text_length:
            boundary_candidates = [
                text.rfind("\n\n", start, target_end),
                text.rfind(". ", start, target_end),
                text.rfind(" ", start, target_end),
            ]

            valid_boundaries = [
                boundary
                for boundary in boundary_candidates
                if boundary > start + int(chunk_size * 0.5)
            ]

            if valid_boundaries:
                target_end = max(valid_boundaries)

                if text[target_end : target_end + 2] == "\n\n":
                    target_end += 2
                elif text[target_end : target_end + 1] == " ":
                    target_end += 1

        chunk_text = text[start:target_end].strip()

        if chunk_text:
            actual_start = start
            actual_end = target_end

            # Correct the recorded boundaries for leading/trailing whitespace.
            leading_trim = len(text[start:target_end]) - len(
                text[start:target_end].lstrip()
            )
            trailing_trim = len(text[start:target_end]) - len(
                text[start:target_end].rstrip()
            )

            actual_start += leading_trim
            actual_end -= trailing_trim

            chunks.append(
                (
                    chunk_text,
                    actual_start,
                    actual_end,
                )
            )

        if target_end >= text_length:
            break

        next_start = target_end - chunk_overlap

        if next_start <= start:
            next_start = target_end

        start = next_start

    return chunks


def _build_document(
    path: Path,
    source_type: SourceType,
    page_count: int,
) -> Document:
    """Build a Document model."""

    return Document(
        document_id=_generate_document_id(path),
        filename=path.name,
        source_path=str(path.resolve()),
        source_type=source_type,
        page_count=page_count,
        metadata={
            "extension": path.suffix.lower(),
            "file_size": path.stat().st_size,
        },
    )


def ingest_pdf(
    path: str | Path,
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> tuple[Document, list[Chunk]]:
    """
    Ingest a PDF while preserving page-level provenance.

    Args:
        path: Path to the PDF.
        chunk_size: Maximum target chunk size.
        chunk_overlap: Number of overlapping characters.
    
    Returns:
        A Document object and its generated chunks.

    Raises:
        FileNotFoundError: If the PDF does not exist.
        DocumentIngestionError: If the file is invalid or cannot be read.
    """

    pdf_path = Path(path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"Document not found: {pdf_path}")

    if not pdf_path.is_file():
        raise DocumentIngestionError(
            f"Expected a file but received: {pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise DocumentIngestionError(
            f"Expected a PDF file but received: {pdf_path.suffix}"
        )

    try:
        pdf = fitz.open(pdf_path)
    except Exception as exc:
        raise DocumentIngestionError(
            f"Could not open PDF: {pdf_path}"
        ) from exc

    try:
        document = _build_document(
            pdf_path,
            SourceType.PDF,
            len(pdf),
        )

        chunks: list[Chunk] = []

        for page_index, page in enumerate(pdf):
            page_number = page_index + 1

            raw_text = page.get_text("text")
            text = _clean_text(raw_text)

            if not text:
                continue

            page_chunks = _split_text(
                text,
                chunk_size,
                chunk_overlap,
            )

            for chunk_index, (
                chunk_text,
                start_char,
                end_char,
            ) in enumerate(page_chunks):
                chunk_id = _generate_chunk_id(
                    document.document_id,
                    page_number,
                    chunk_index,
                    chunk_text,
                )

                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        document_id=document.document_id,
                        text=chunk_text,
                        page_number=page_number,
                        start_char=start_char,
                        end_char=end_char,
                        source_filename=document.filename,
                        metadata={
                            "source_type": document.source_type.value,
                            "page_index": page_index,
                            "chunk_index": chunk_index,
                        },
                    )
                )

        return document, chunks

    finally:
        pdf.close()


def ingest_document(
    path: str | Path,
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> tuple[Document, list[Chunk]]:
    """
    Ingest a supported document.

    Currently supported:
        - PDF
        - TXT
        - Markdown
    """

    document_path = Path(path)

    if not document_path.exists():
        raise FileNotFoundError(
            f"Document not found: {document_path}"
        )

    extension = document_path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))

        raise DocumentIngestionError(
            f"Unsupported document type '{extension}'. "
            f"Supported types: {supported}"
        )

    if extension == ".pdf":
        return ingest_pdf(
            document_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    source_type = SUPPORTED_EXTENSIONS[extension]

    try:
        text = _clean_text(
            document_path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        )
    except OSError as exc:
        raise DocumentIngestionError(
            f"Could not read document: {document_path}"
        ) from exc

    document = _build_document(
        document_path,
        source_type,
        page_count=1,
    )

    chunks: list[Chunk] = []

    for chunk_index, (
        chunk_text,
        start_char,
        end_char,
    ) in enumerate(
        _split_text(
            text,
            chunk_size,
            chunk_overlap,
        )
    ):
        chunk_id = _generate_chunk_id(
            document.document_id,
            1,
            chunk_index,
            chunk_text,
        )

        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                document_id=document.document_id,
                text=chunk_text,
                page_number=1,
                start_char=start_char,
                end_char=end_char,
                source_filename=document.filename,
                metadata={
                    "source_type": document.source_type.value,
                    "page_index": 0,
                    "chunk_index": chunk_index,
                },
            )
        )

    return document, chunks
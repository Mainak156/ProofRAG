from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .models import Chunk, Document, OKFNode, OKFRelation


class OKFError(Exception):
    """Base exception for ProofRAG OKF operations."""


class OKFValidationError(OKFError):
    """Raised when generated OKF content is invalid."""


class OKFBundle:
    """
    Represents a ProofRAG Open Knowledge Format bundle.

    The bundle follows the OKF model of Markdown concept documents
    with YAML frontmatter.
    """

    INDEX_FILENAME = "index.md"
    LOG_FILENAME = "log.md"

    def __init__(self, root: str | Path):
        self.root = Path(root)

        self.chunks_dir = self.root / "chunks"

        self.root.mkdir(parents=True, exist_ok=True)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def chunk_path(self, chunk_id: str) -> Path:
        """Return the OKF concept path for a chunk."""

        safe_id = _safe_filename(chunk_id)

        return self.chunks_dir / f"{safe_id}.md"

    # ------------------------------------------------------------------
    # Concept generation
    # ------------------------------------------------------------------

    def build_chunk_node(
        self,
        document: Document,
        chunk: Chunk,
    ) -> OKFNode:
        """Convert a ProofRAG chunk into an OKF concept node."""

        if chunk.document_id != document.document_id:
            raise OKFValidationError(
                "Chunk document_id does not match the supplied document."
            )

        title_page = (
            f"page {chunk.page_number}"
            if chunk.page_number is not None
            else "unknown page"
        )

        return OKFNode(
            node_id=f"chunks/{chunk.chunk_id}",
            node_type="document-chunk",
            title=(
                f"{document.filename} — "
                f"{title_page} — "
                f"chunk {chunk.metadata.get('chunk_index', 0)}"
            ),
            content=chunk.text,
            source_document_id=document.document_id,
            source_filename=document.filename,
            page_number=chunk.page_number,
            metadata={
                "chunk_id": chunk.chunk_id,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "source_type": document.source_type.value,
            },
        )

    # ------------------------------------------------------------------
    # Markdown / YAML serialization
    # ------------------------------------------------------------------

    def render_chunk(
        self,
        document: Document,
        chunk: Chunk,
        *,
        generated_at: datetime | None = None,
    ) -> str:
        """
        Render a chunk as an OKF Markdown concept.

        The generated document contains:
        - YAML frontmatter
        - source provenance
        - ProofRAG chunk metadata
        - original evidence text
        """

        if generated_at is None:
            generated_at = datetime.now(timezone.utc)

        generated_at = generated_at.astimezone(timezone.utc)

        page = chunk.page_number

        source_id = _source_id(
            document.document_id,
            page,
        )

        source_resource = _relative_document_resource(
            self.root,
            document.source_path,
        )

        frontmatter: dict[str, Any] = {
            "type": "document-chunk",
            "title": (
                f"{document.filename} — "
                f"page {page if page is not None else 'unknown'} — "
                f"chunk {chunk.metadata.get('chunk_index', 0)}"
            ),
            "description": (
                f"Evidence extracted from {document.filename}"
                + (
                    f", page {page}."
                    if page is not None
                    else "."
                )
            ),
            "tags": [
                "proofrag",
                "document-chunk",
                f"source-{document.source_type.value}",
            ],
            "resource": source_resource,
            "status": "stable",
            "generated": {
                "by": "proofrag/0.1.0",
                "at": generated_at.isoformat().replace("+00:00", "Z"),
            },
            "sources": [
                {
                    "id": source_id,
                    "resource": source_resource,
                    "title": (
                        f"{document.filename}"
                        + (
                            f" — page {page}"
                            if page is not None
                            else ""
                        )
                    ),
                }
            ],
            "proofrag": {
                "document_id": document.document_id,
                "chunk_id": chunk.chunk_id,
                "page_number": page,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
            },
        }

        yaml_text = yaml.safe_dump(
            frontmatter,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        ).strip()

        body = _render_body(
            document,
            chunk,
            source_id,
        )

        return f"---\n{yaml_text}\n---\n\n{body}\n"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def write_chunk(
        self,
        document: Document,
        chunk: Chunk,
        *,
        generated_at: datetime | None = None,
    ) -> Path:
        """Write one chunk as an OKF Markdown concept."""

        path = self.chunk_path(chunk.chunk_id)

        content = self.render_chunk(
            document,
            chunk,
            generated_at=generated_at,
        )

        path.write_text(
            content,
            encoding="utf-8",
        )

        return path

    def write_chunks(
        self,
        document: Document,
        chunks: list[Chunk],
        *,
        generated_at: datetime | None = None,
    ) -> list[Path]:
        """Write all chunks belonging to a document."""

        paths = [
            self.write_chunk(
                document,
                chunk,
                generated_at=generated_at,
            )
            for chunk in chunks
        ]

        self.write_index(document, chunks)

        return paths

    # ------------------------------------------------------------------
    # Index
    # ------------------------------------------------------------------

    def write_index(
        self,
        document: Document,
        chunks: list[Chunk],
    ) -> Path:
        """
        Write the OKF bundle index.

        The index provides progressive disclosure into the bundle.
        """

        lines = [
            "# ProofRAG Knowledge Bundle",
            "",
            "## Source Document",
            "",
            f"- **Filename:** `{document.filename}`",
            f"- **Document ID:** `{document.document_id}`",
            f"- **Source type:** `{document.source_type.value}`",
            f"- **Pages:** `{document.page_count}`",
            "",
            "## Evidence Chunks",
            "",
        ]

        for chunk in chunks:
            chunk_path = self.chunk_path(chunk.chunk_id)
            relative_path = chunk_path.relative_to(self.root).as_posix()

            page = (
                str(chunk.page_number)
                if chunk.page_number is not None
                else "unknown"
            )

            lines.append(
                f"- [Page {page} — `{chunk.chunk_id}`]"
                f"({relative_path})"
            )

        lines.append("")

        path = self.root / self.INDEX_FILENAME

        path.write_text(
            "\n".join(lines),
            encoding="utf-8",
        )

        return path

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def read_chunk(
        self,
        chunk_id: str,
    ) -> tuple[dict[str, Any], str]:
        """Read and parse an OKF chunk document."""

        path = self.chunk_path(chunk_id)

        if not path.exists():
            raise FileNotFoundError(
                f"OKF chunk not found: {chunk_id}"
            )

        content = path.read_text(
            encoding="utf-8",
        )

        return parse_okf_document(content)

    def list_chunks(self) -> list[Path]:
        """Return all chunk concept files in the bundle."""

        return sorted(
            self.chunks_dir.glob("*.md")
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """
        Validate the bundle.

        Returns:
            A list of validation errors.
            An empty list means the bundle is valid.
        """

        errors: list[str] = []

        if not self.root.exists():
            errors.append("Bundle root does not exist.")
            return errors

        index_path = self.root / self.INDEX_FILENAME

        if not index_path.exists():
            errors.append("Missing required index.md.")

        for path in self.list_chunks():
            try:
                frontmatter, body = parse_okf_document(
                    path.read_text(
                        encoding="utf-8"
                    )
                )

                if not frontmatter.get("type"):
                    errors.append(
                        f"{path.name}: missing required 'type'."
                    )

                if not body.strip():
                    errors.append(
                        f"{path.name}: empty concept body."
                    )

                sources = frontmatter.get("sources", [])

                if not isinstance(sources, list):
                    errors.append(
                        f"{path.name}: 'sources' must be a list."
                    )
                else:
                    for index, source in enumerate(sources):
                        if not isinstance(source, dict):
                            errors.append(
                                f"{path.name}: source {index} "
                                "must be a mapping."
                            )
                            continue

                        if not source.get("resource"):
                            errors.append(
                                f"{path.name}: source {index} "
                                "is missing 'resource'."
                            )

            except Exception as exc:
                errors.append(
                    f"{path.name}: {exc}"
                )

        return errors


# ----------------------------------------------------------------------
# Standalone helpers
# ----------------------------------------------------------------------


def parse_okf_document(
    content: str,
) -> tuple[dict[str, Any], str]:
    """
    Parse an OKF Markdown document.

    Returns:
        (frontmatter, markdown_body)
    """

    if not content.startswith("---"):
        raise OKFValidationError(
            "OKF document must begin with YAML frontmatter."
        )

    parts = content.split(
        "---",
        2,
    )

    if len(parts) != 3:
        raise OKFValidationError(
            "OKF document has invalid frontmatter delimiters."
        )

    _, yaml_block, body = parts

    try:
        frontmatter = yaml.safe_load(yaml_block) or {}
    except yaml.YAMLError as exc:
        raise OKFValidationError(
            "Invalid YAML frontmatter."
        ) from exc

    if not isinstance(frontmatter, dict):
        raise OKFValidationError(
            "OKF frontmatter must be a mapping."
        )

    return frontmatter, body.strip()


def _safe_filename(value: str) -> str:
    """Convert an identifier into a safe filename."""

    safe = re.sub(
        r"[^a-zA-Z0-9._-]+",
        "-",
        value,
    )

    return safe.strip("-") or "unknown"


def _source_id(
    document_id: str,
    page_number: int | None,
) -> str:
    """Generate a stable OKF source identifier."""

    raw = f"{document_id}|{page_number}"

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()[:16]


def _relative_document_resource(
    bundle_root: Path,
    source_path: str,
) -> str:
    """
    Return a bundle-relative source path when possible.

    If the document is outside the project, use a normalized
    file URI instead.
    """

    source = Path(source_path).resolve()

    try:
        relative = source.relative_to(
            bundle_root.parent.parent
        )

        return relative.as_posix()

    except ValueError:
        return source.as_uri()


def _render_body(
    document: Document,
    chunk: Chunk,
    source_id: str,
) -> str:
    """Render the Markdown body of an OKF chunk."""

    page = (
        str(chunk.page_number)
        if chunk.page_number is not None
        else "unknown"
    )

    start = (
        str(chunk.start_char)
        if chunk.start_char is not None
        else "unknown"
    )

    end = (
        str(chunk.end_char)
        if chunk.end_char is not None
        else "unknown"
    )

    source_uri = Path(
        document.source_path
    ).resolve().as_uri()

    lines = [
        "# Evidence",
        "",
        chunk.text,
        "",
        "## Provenance",
        "",
        f"- **Document:** `{document.filename}`",
        f"- **Document ID:** `{document.document_id}`",
        f"- **Chunk ID:** `{chunk.chunk_id}`",
        f"- **Page:** `{page}`",
        f"- **Character range:** `{start}–{end}`",
        "",
        f"Source: [document]({source_uri})[^source]",
        "",
        f"[^source]: {source_id}",
    ]

    return "\n".join(lines)


def _relative_source_link(
    source_path: str,
) -> str:
    """Create a portable relative-looking source reference."""

    source = Path(source_path).resolve()

    return source.as_uri()
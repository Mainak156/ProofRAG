from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SourceType(str, Enum):
    """Supported document source types."""

    PDF = "pdf"
    TEXT = "text"
    MARKDOWN = "markdown"
    UNKNOWN = "unknown"


class VerificationStatus(str, Enum):
    """Status returned by citation verification."""

    VERIFIED = "verified"
    PARTIALLY_VERIFIED = "partially_verified"
    REJECTED = "rejected"
    NOT_CHECKED = "not_checked"


class Document(BaseModel):
    """Represents an ingested source document."""

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    source_type: SourceType
    page_count: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A page-aware retrievable document chunk."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)

    text: str = Field(min_length=1)

    page_number: int | None = Field(default=None, ge=1)

    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, ge=0)

    source_filename: str = Field(min_length=1)

    metadata: dict[str, Any] = Field(default_factory=dict)


class OKFNode(BaseModel):
    """Structured knowledge node used by the OKF layer."""

    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(min_length=1)
    node_type: str = Field(min_length=1)
    title: str = Field(min_length=1)

    content: str = ""

    source_document_id: str | None = None
    source_filename: str | None = None
    page_number: int | None = Field(default=None, ge=1)

    metadata: dict[str, Any] = Field(default_factory=dict)


class OKFRelation(BaseModel):
    """Relationship between two OKF knowledge nodes."""

    model_config = ConfigDict(extra="forbid")

    relation_id: str = Field(min_length=1)
    source_node_id: str = Field(min_length=1)
    target_node_id: str = Field(min_length=1)
    relation_type: str = Field(min_length=1)

    metadata: dict[str, Any] = Field(default_factory=dict)


class Evidence(BaseModel):
    """Evidence retrieved from the knowledge base."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)

    filename: str = Field(min_length=1)
    page_number: int | None = Field(default=None, ge=1)

    snippet: str = Field(min_length=1)

    retrieval_score: float = Field(ge=0.0)

    metadata: dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    """Citation attached to a generated answer."""

    model_config = ConfigDict(extra="forbid")

    citation_id: str = Field(min_length=1)
    evidence_id: str = Field(min_length=1)

    filename: str = Field(min_length=1)
    page_number: int | None = Field(default=None, ge=1)

    snippet: str = Field(min_length=1)

    verification_status: VerificationStatus = VerificationStatus.NOT_CHECKED

    keyword_score: float = Field(default=0.0, ge=0.0)
    embedding_score: float = Field(default=0.0, ge=0.0)


class VerificationResult(BaseModel):
    """Result of independently verifying an answer citation."""

    model_config = ConfigDict(extra="forbid")

    citation_id: str = Field(min_length=1)

    status: VerificationStatus

    keyword_score: float = Field(default=0.0, ge=0.0)
    embedding_score: float = Field(default=0.0, ge=0.0)

    reason: str = Field(default="")


class Answer(BaseModel):
    """Generated answer before being wrapped in the final response."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)

    citations: list[Citation] = Field(default_factory=list)

    refused: bool = False
    refusal_reason: str | None = None


class ProofRAGRequest(BaseModel):
    """Public request contract for the ProofRAG pipeline."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2000)


class ProofRAGResponse(BaseModel):
    """Public response contract returned by ProofRAG."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)

    citations: list[Citation] = Field(default_factory=list)

    verified: bool = False

    refused: bool = False

    refusal_reason: str | None = None

    evidence_count: int = Field(default=0, ge=0)
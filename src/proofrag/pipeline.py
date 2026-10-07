from __future__ import annotations

from pathlib import Path

from .config import get_settings
from .generator import AnswerGenerator, GenerationError
from .ingestion import DocumentIngestionError, ingest_document
from .models import (
    Answer,
    Chunk,
    Document,
    ProofRAGRequest,
    ProofRAGResponse,
)
from .okf import OKFBundle, OKFError
from .retrieval import RetrievalError, add_chunks, retrieve
from .verifier import VerificationError, verify_answer


class PipelineError(Exception):
    """Raised when the ProofRAG pipeline fails."""


class ProofRAG:
    """
    End-to-end evidence-backed RAG pipeline.

    Flow:
        documents -> ingestion -> OKF -> Chroma
        question -> retrieval -> generation -> verification
    """

    def __init__(
        self,
        docs_path: str | Path | None = None,
    ) -> None:
        settings = get_settings()

        self.docs_path = (
            Path(docs_path).resolve()
            if docs_path is not None
            else settings.documents_dir.resolve()
        )

        self.okf_bundle = OKFBundle(
            settings.okf_dir / "bundle"
        )

        self.generator = AnswerGenerator()

    def ingest(self) -> list[Document]:
        """Ingest all supported documents and index their chunks."""

        if not self.docs_path.exists():
            raise PipelineError(
                f"Documents path does not exist: {self.docs_path}"
            )

        if not self.docs_path.is_dir():
            raise PipelineError(
                f"Documents path is not a directory: {self.docs_path}"
            )

        documents: list[Document] = []
        all_chunks: list[Chunk] = []

        supported_extensions = {
            ".pdf",
            ".txt",
            ".md",
            ".markdown",
        }

        files = sorted(
            path
            for path in self.docs_path.rglob("*")
            if path.is_file()
            and path.suffix.lower() in supported_extensions
        )

        for path in files:
            try:
                document, chunks = ingest_document(path)
            except DocumentIngestionError as exc:
                raise PipelineError(
                    f"Failed to ingest document: {path.name}"
                ) from exc

            documents.append(document)
            all_chunks.extend(chunks)

        if not documents:
            raise PipelineError(
                f"No supported documents found in {self.docs_path}"
            )

        # Build the OKF knowledge bundle document-by-document.
        try:
            for document in documents:
                document_chunks = [
                    chunk
                    for chunk in all_chunks
                    if chunk.document_id == document.document_id
                ]

                self.okf_bundle.write_chunks(
                    document,
                    document_chunks,
                )
        except OKFError as exc:
            raise PipelineError(
                "Failed to create the OKF knowledge bundle."
            ) from exc

        # Add all chunks to Chroma for semantic retrieval.
        try:
            add_chunks(all_chunks)
        except RetrievalError as exc:
            raise PipelineError(
                "Failed to index document chunks in Chroma."
            ) from exc

        return documents

    def ask(
        self,
        question: str | ProofRAGRequest,
    ) -> ProofRAGResponse:
        """Answer a question using only retrieved and verified evidence."""

        request = (
            question
            if isinstance(question, ProofRAGRequest)
            else ProofRAGRequest(question=question)
        )

        try:
            evidence = retrieve(
                request.question,
                okf_bundle=self.okf_bundle,
            )
        except RetrievalError as exc:
            raise PipelineError(
                "Evidence retrieval failed."
            ) from exc

        # Safe refusal when retrieval finds no sufficiently relevant evidence.
        if not evidence:
            answer = Answer(
                text="Not found in the documents.",
                citations=[],
                refused=True,
                refusal_reason=(
                    "No sufficiently relevant evidence was retrieved."
                ),
            )

            return _build_response(
                request=request,
                answer=answer,
                evidence_count=0,
            )

        try:
            generated_answer = self.generator.generate(
                request.question,
                evidence,
            )
        except GenerationError as exc:
            raise PipelineError(
                "Answer generation failed."
            ) from exc

        try:
            verified_answer = verify_answer(
                generated_answer,
            )
        except VerificationError as exc:
            raise PipelineError(
                "Citation verification failed."
            ) from exc

        return _build_response(
            request=request,
            answer=verified_answer,
            evidence_count=len(evidence),
        )

    def ingest_and_ask(
        self,
        question: str | ProofRAGRequest,
    ) -> ProofRAGResponse:
        """Ingest the configured documents and immediately answer a question."""

        self.ingest()
        return self.ask(question)


def _build_response(
    request: ProofRAGRequest,
    answer: Answer,
    evidence_count: int,
) -> ProofRAGResponse:
    """Build the public API response."""

    verified = (
        not answer.refused
        and bool(answer.citations)
        and all(
            citation.verification_status.value == "verified"
            for citation in answer.citations
        )
    )

    return ProofRAGResponse(
        answer=answer.text,
        citations=answer.citations,
        verified=verified,
        refused=answer.refused,
        refusal_reason=answer.refusal_reason,
        evidence_count=evidence_count,
    )
from __future__ import annotations

from fastapi import FastAPI, HTTPException

from .config import get_settings
from .models import ProofRAGRequest, ProofRAGResponse
from .pipeline import PipelineError, ProofRAG

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Evidence-backed RAG with verified citations "
        "and safe refusal on insufficient evidence."
    ),
)

rag = ProofRAG()


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.app_version,
    }


@app.post("/ingest")
def ingest() -> dict[str, object]:
    try:
        documents = rag.ingest()
    except PipelineError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    return {
        "status": "ok",
        "documents_indexed": len(documents),
        "documents": [
            document.filename
            for document in documents
        ],
    }


@app.post(
    "/ask",
    response_model=ProofRAGResponse,
)
def ask(
    request: ProofRAGRequest,
) -> ProofRAGResponse:
    try:
        return rag.ask(request)
    except PipelineError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc
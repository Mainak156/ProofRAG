from __future__ import annotations

import json
from typing import Protocol

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, ConfigDict, Field

from .config import get_settings
from .models import Answer, Citation, Evidence, VerificationStatus


class GenerationError(Exception):
    """Raised when answer generation fails safely."""


class GeneratedAnswer(BaseModel):
    """Strict internal contract for the LLM response."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    citations: list[str] = Field(default_factory=list)


class LLMClient(Protocol):
    """Minimal provider interface used by the generator."""

    def invoke(self, messages: list[BaseMessage]) -> object:
        ...


SYSTEM_PROMPT = """You are ProofRAG, an evidence-grounded question answering system.

Your job is to answer the user's question using ONLY the supplied evidence.

STRICT RULES:
1. Use only information explicitly supported by the supplied evidence.
2. Never use outside knowledge.
3. Never invent facts, sources, filenames, page numbers, or citations.
4. Every factual claim in the answer must be supported by one or more supplied
   evidence IDs.
5. The citations field must contain ONLY the exact evidence IDs supplied to you.
6. If the evidence does not adequately answer the question, return:
   "Not found in the documents."
   and use an empty citations list.
7. Do not guess or fill missing information.
8. Return ONLY valid JSON. Do not use Markdown fences.

Required JSON format:
{
  "answer": "your evidence-grounded answer",
  "citations": ["exact-evidence-id-1", "exact-evidence-id-2"]
}
"""


def build_generation_messages(
    question: str,
    evidence: list[Evidence],
) -> list[BaseMessage]:
    """Build the exact messages sent to the LLM."""

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    evidence_blocks: list[str] = []

    for item in evidence:
        page = (
            str(item.page_number)
            if item.page_number is not None
            else "unknown"
        )

        evidence_blocks.append(
            "\n".join(
                [
                    f"[EVIDENCE ID: {item.evidence_id}]",
                    f"Source file: {item.filename}",
                    f"Page: {page}",
                    "Supporting text:",
                    item.snippet,
                ]
            )
        )

    evidence_text = "\n\n".join(evidence_blocks)

    human_prompt = f"""User question:
{question}

Supplied evidence:
{evidence_text}

Answer the question using only the supplied evidence.
Return the required JSON object and nothing else.
"""

    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=human_prompt),
    ]


class AnswerGenerator:
    """Generate evidence-grounded answers using an injected LLM client."""

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm

    def _get_llm(self) -> LLMClient:
        if self._llm is not None:
            return self._llm

        settings = get_settings()

        if not settings.groq_api_key.strip():
            raise GenerationError(
                "GROQ_API_KEY is not configured."
            )

        return ChatGroq(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            api_key=settings.groq_api_key,
            timeout=settings.llm_timeout,
        )

    def generate(
        self,
        question: str,
        evidence: list[Evidence],
    ) -> Answer:
        """Generate an answer from retrieved evidence."""

        if not question.strip():
            raise ValueError("Question cannot be empty.")

        # Safe refusal before contacting the LLM.
        if not evidence:
            return Answer(
                text="Not found in the documents.",
                citations=[],
                refused=True,
                refusal_reason="No supporting evidence was retrieved.",
            )

        messages = build_generation_messages(question, evidence)

        try:
            response = self._get_llm().invoke(messages)
        except GenerationError:
            raise
        except Exception as exc:
            raise GenerationError(
                "The answer generation provider failed."
            ) from exc

        generated = _parse_response(response)
        return _build_answer(generated, evidence)


def _parse_response(response: object) -> GeneratedAnswer:
    """Parse and validate the LLM's JSON response."""

    content = getattr(response, "content", response)

    if isinstance(content, list):
        parts: list[str] = []

        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))

        content = "".join(parts)

    if not isinstance(content, str):
        raise GenerationError(
            "The LLM returned an unsupported response format."
        )

    raw = content.strip()

    # Allow a fenced JSON response despite the strict prompt.
    if raw.startswith("```") and raw.endswith("```"):
        lines = raw.splitlines()

        if lines and lines[0].strip().lower() in {
            "```json",
            "```",
        }:
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        raw = "\n".join(lines).strip()

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise GenerationError(
            "The LLM returned invalid JSON."
        ) from exc

    try:
        return GeneratedAnswer.model_validate(payload)
    except Exception as exc:
        raise GenerationError(
            "The LLM response does not match the ProofRAG answer schema."
        ) from exc


def _build_answer(
    generated: GeneratedAnswer,
    evidence: list[Evidence],
) -> Answer:
    """Convert generated JSON into the public ProofRAG Answer model."""

    evidence_by_id = {
        item.evidence_id: item
        for item in evidence
    }

    requested_ids = generated.citations

    unknown_ids = [
        evidence_id
        for evidence_id in requested_ids
        if evidence_id not in evidence_by_id
    ]

    if unknown_ids:
        raise GenerationError(
            "The LLM returned citation IDs that were not supplied as evidence: "
            + ", ".join(unknown_ids)
        )

    # Preserve order while removing duplicate citation IDs.
    unique_ids = list(dict.fromkeys(requested_ids))

    citations: list[Citation] = []

    for index, evidence_id in enumerate(unique_ids):
        item = evidence_by_id[evidence_id]

        citations.append(
            Citation(
                citation_id=f"citation-{index}-{evidence_id}",
                evidence_id=item.evidence_id,
                filename=item.filename,
                page_number=item.page_number,
                snippet=item.snippet,
                verification_status=VerificationStatus.NOT_CHECKED,
                keyword_score=0.0,
                embedding_score=0.0,
            )
        )

    refused = generated.answer.strip() == "Not found in the documents."

    return Answer(
        text=generated.answer.strip(),
        citations=citations,
        refused=refused,
        refusal_reason=(
            "The supplied evidence does not contain enough information."
            if refused
            else None
        ),
    )
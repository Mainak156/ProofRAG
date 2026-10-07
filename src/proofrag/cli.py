from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel

from .pipeline import PipelineError, ProofRAG

app = typer.Typer(
    name="proofrag",
    help="Evidence-backed RAG with verified citations.",
)

console = Console()


@app.command()
def ingest() -> None:
    """Index documents into OKF and Chroma."""

    rag = ProofRAG()

    try:
        documents = rag.ingest()
    except PipelineError as exc:
        console.print(
            f"[red]Error:[/red] {exc}"
        )
        raise typer.Exit(code=1)

    console.print(
        f"[green]Indexed {len(documents)} document(s).[/green]"
    )

    for document in documents:
        console.print(
            f"  • {document.filename}"
        )


@app.command()
def ask(
    question: str = typer.Argument(
        ...,
        help="Question to ask over the indexed documents.",
    ),
) -> None:
    """Ask a question and display verified evidence."""

    rag = ProofRAG()

    try:
        response = rag.ask(question)
    except PipelineError as exc:
        console.print(
            f"[red]Error:[/red] {exc}"
        )
        raise typer.Exit(code=1)

    console.print(
        Panel(
            response.answer,
            title="ProofRAG Answer",
        )
    )

    console.print(
        f"Verified: {response.verified}"
    )
    console.print(
        f"Refused: {response.refused}"
    )
    console.print(
        f"Evidence: {response.evidence_count}"
    )

    for citation in response.citations:
        console.print()
        console.print(
            f"[bold]Source:[/bold] {citation.filename}"
        )
        console.print(
            f"[bold]Page:[/bold] {citation.page_number}"
        )
        console.print(
            f"[bold]Status:[/bold] "
            f"{citation.verification_status.value}"
        )
        console.print(
            f"[bold]Evidence:[/bold] "
            f"{citation.snippet}"
        )


if __name__ == "__main__":
    app()
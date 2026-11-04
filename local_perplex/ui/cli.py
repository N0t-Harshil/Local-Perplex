"""``perplex``: research from the terminal with live Markdown streaming."""
from __future__ import annotations

import time

import click
from rich import box
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from local_perplex import __version__
from local_perplex.engine import LocalPerplex, is_greeting
from local_perplex.providers import PROVIDERS
from local_perplex.search import SEARCH_BACKENDS

console = Console()
FOCUS = ["All", "Academic", "Reddit", "YouTube", "Technical", "Writing", "Computational"]


def _md(text: str) -> Markdown:
    return Markdown(text.strip() or "…", code_theme="monokai")


def _list_providers() -> None:
    table = Table(box=box.SIMPLE_HEAD, title="LLM providers", title_justify="left")
    table.add_column("id", style="bold")
    table.add_column("name")
    table.add_column("where")
    table.add_column("key")
    for p in PROVIDERS.values():
        table.add_row(p.id, p.label, "[green]local[/]" if p.local else "hosted", p.env_key or "-")
    console.print(table)
    search = Table(box=box.SIMPLE_HEAD, title="Search backends", title_justify="left")
    search.add_column("id", style="bold")
    search.add_column("name")
    search.add_column("key")
    for s in SEARCH_BACKENDS.values():
        search.add_row(s.id, s.label, s.env_key or ("URL" if s.needs_url else "-"))
    console.print(search)


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.argument("question", nargs=-1)
@click.option("--mode", type=click.Choice(["industry_standard", "all_references", "pro"]),
              default="industry_standard", help="Research depth (pro = parallel deep research)")
@click.option("--model", "-m", default=None, help="Model name, e.g. llama3.2:3b or gpt-4o-mini")
@click.option("--provider", "-p", default=None, help="LLM provider id (see --list-providers)")
@click.option("--search", "search_backend", default=None, help="Search backend id")
@click.option("--focus", "-f", type=click.Choice(FOCUS), default="All", help="Search focus")
@click.option("--tag", "-t", default="General", help="Tag for the knowledge graph")
@click.option("--incognito", is_flag=True, help="Don't save to history or the graph")
@click.option("--no-stream", is_flag=True, help="Print the answer once it is complete")
@click.option("--list-providers", is_flag=True, help="Show every supported provider and exit")
@click.version_option(__version__, prog_name="local-perplex")
def main(question, mode, model, provider, search_backend, focus, tag, incognito, no_stream, list_providers):
    """Research QUESTION with web search + a local or hosted LLM."""
    if list_providers:
        _list_providers()
        return
    if not question:
        raise click.UsageError("Ask a question, e.g.  perplex \"What is retrieval-augmented generation?\"")
    question = " ".join(question)
    start = time.perf_counter()
    console.print(Panel.fit(Text.assemble(("Local Perplex ", "bold"), (__version__, "dim")),
                            border_style="bright_black"))

    if is_greeting(question):
        console.print("  Hi! Ask me a question and I'll research it.\n")
        return

    if provider and provider not in PROVIDERS:
        raise click.BadParameter(f"unknown provider, choose from: {', '.join(PROVIDERS)}", param_hint="--provider")
    engine = LocalPerplex(model=model, provider=provider)
    if search_backend:
        if search_backend not in SEARCH_BACKENDS:
            raise click.BadParameter(f"choose from: {', '.join(SEARCH_BACKENDS)}", param_hint="--search")
        engine.settings.search_provider = search_backend
    if not engine.llm_online:
        hint = ("Install it from https://ollama.com and run `ollama serve`, or pick another provider with -p."
                if engine.provider_id == "ollama" else "Check the API key / URL for this provider.")
        console.print(Panel.fit(f"[red]{engine.provider_label} isn't reachable.[/]\n[dim]{hint}[/]", border_style="red"))
        raise SystemExit(1)

    where = "[green]on this device[/]" if engine.is_local else "[yellow]hosted[/]"
    console.print(f"  [dim]model[/] {engine.model}  [dim]via[/] {engine.provider_label} ({where})  "
                  f"[dim]search[/] {engine.search_engine.backend}  [dim]mode[/] {mode}\n")

    sources, context = [], ""
    with console.status("[bold]Thinking…", spinner="dots") as status:
        for kind, payload in engine.research_events(question, mode=mode, focus_mode=focus):
            if kind == "status":
                status.update(f"[bold]{payload}")
            elif kind == "result":
                _, sources, context = payload
    research_s = time.perf_counter() - start
    console.print(f"  [dim]{len(sources)} sources in {research_s:.1f}s[/]" if sources
                  else "  [dim]answering from the model's knowledge and your notes[/]")

    if no_stream:
        with console.status("[bold]Writing…", spinner="dots"):
            answer = engine.ask(question, context, mode=mode, sources=sources)
        console.print(Panel(_md(answer), border_style="green"))
    else:
        answer = ""
        with Live(Panel(_md(""), border_style="green"), refresh_per_second=12, console=console) as live:
            for chunk in engine.ask_stream(question, context, mode=mode, sources=sources):
                answer += chunk
                live.update(Panel(_md(answer), border_style="green"))

    if sources:
        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        table.add_column(style="yellow", width=4, no_wrap=True)
        table.add_column(ratio=1)
        table.add_column(style="dim", width=5, justify="right", no_wrap=True)
        for i, s in enumerate(sources, 1):
            table.add_row(f"[{i}]", f"[link={s.url}]{s.title[:80]}[/link]\n[dim]{s.url[:90]}[/]", f"{s.relevance:.0%}")
        console.print(Panel(table, title="sources", title_align="left", border_style="bright_black"))

    stats = engine.llm.last_stats
    related = engine.finalize_research(question, answer, sources, "[INCOGNITO]" if incognito else tag,
                                       background=False)
    if related:
        console.print("[dim]keep exploring[/]")
        for r in related:
            console.print(f"  • {r}")
    console.print(f"\n[dim]{time.perf_counter() - start:.1f}s total · first token {stats.ttft:.1f}s · "
                  f"{stats.tokens_per_sec:.0f} tok/s[/]\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Jailson — CLI entry point."""
import os
import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt
from rich.rule import Rule
from rich.markdown import Markdown
from rich import print as rprint
import click

console = Console()


def check_api_key():
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key or key == "sk-ant-...":
        console.print(Panel(
            "[red]ANTHROPIC_API_KEY não configurada![/red]\n\n"
            "Edita o ficheiro [yellow].env[/yellow] e adiciona a tua chave:\n"
            "[blue]ANTHROPIC_API_KEY=sk-ant-...[/blue]\n\n"
            "Obtém a tua chave em: [link]https://console.anthropic.com[/link]",
            title="⚠ Configuração necessária",
            border_style="red",
        ))
        sys.exit(1)


def print_banner():
    banner = Text()
    banner.append("  J A I L S O N  ", style="bold cyan")
    banner.append("— Agente Pessoal de PC", style="dim")
    console.print(Panel(banner, border_style="cyan", padding=(0, 2)))
    console.print(
        "  [dim]Comandos especiais: [yellow]/memoria[/yellow] [yellow]/pesquisar[/yellow] "
        "[yellow]/pc[/yellow] [yellow]/imagem[/yellow] [yellow]/stats[/yellow] "
        "[yellow]/reset[/yellow] [yellow]/sair[/yellow][/dim]\n"
    )


def print_help():
    help_text = """
## Comandos Disponíveis

| Comando | Descrição |
|---------|-----------|
| `/pc <tarefa>` | Executar tarefa no PC Agent (ficheiros, sistema) |
| `/imagem <tarefa> [caminho]` | Executar tarefa no Image Agent |
| `/memoria <conteudo>` | Guardar algo na memória longa |
| `/sagrado <conteudo>` | Guardar na memória sagrada (imutável) |
| `/pesquisar <query>` | Pesquisar na memória |
| `/stats` | Ver estatísticas de memória |
| `/reset` | Limpar conversa atual (mantém memória) |
| `/sair` | Sair do Jailson |
| `/ajuda` | Mostrar esta ajuda |

## Dicas

- Fala naturalmente — o Jailson usará os sub-agentes automaticamente
- O Jailson aprende e memoriza automaticamente o que partilhas
- A memória persiste entre sessões em `~/.jailson/`
"""
    console.print(Markdown(help_text))


@click.command()
@click.option("--no-banner", is_flag=True, help="Não mostrar banner inicial")
@click.option("--command", "-c", default=None, help="Executar um único comando e sair")
def main(no_banner: bool, command: str):
    """Jailson — Agente Pessoal de PC."""
    check_api_key()

    from jailson.agent import JailsonAgent
    from jailson.config.settings import ensure_dirs
    ensure_dirs()

    if not no_banner:
        print_banner()

    console.print("[dim]Inicializando Jailson...[/dim]", end="")
    agent = JailsonAgent()
    console.print(" [green]pronto![/green]\n")

    if command:
        _process_input(agent, command)
        return

    # Interactive REPL
    while True:
        try:
            user_input = Prompt.ask("[bold cyan]Tu[/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Até logo![/dim]")
            break

        if not user_input:
            continue

        _process_input(agent, user_input)

        if user_input.lower() in ("/sair", "/exit", "/quit"):
            break


def _process_input(agent, user_input: str):
    """Handle a single user input — special commands or regular chat."""
    low = user_input.lower().strip()

    # ── Special commands ───────────────────────────────────────────────────────

    if low in ("/sair", "/exit", "/quit"):
        console.print("\n[dim cyan]Jailson: Até logo! Memórias guardadas.[/dim cyan]")
        return

    if low in ("/ajuda", "/help"):
        print_help()
        return

    if low == "/stats":
        stats = agent.get_memory_stats()
        console.print(Panel(
            f"[cyan]Sessão ID:[/cyan] {stats['session_id']}\n"
            f"[cyan]Short-term:[/cyan] {stats['short_term']} entradas\n"
            f"[cyan]Sagrada:[/cyan] {stats['sacred']} entradas\n"
            f"[cyan]Semântica:[/cyan] {stats['semantic']} documentos",
            title="📊 Estatísticas de Memória",
            border_style="cyan",
        ))
        return

    if low == "/reset":
        agent.reset_conversation()
        console.print("[dim]Conversa reiniciada (memória preservada).[/dim]")
        return

    if low.startswith("/pc "):
        task = user_input[4:].strip()
        if task:
            console.print(f"\n[bold green]Jailson[/bold green] → [yellow]PC Agent[/yellow]: [dim]{task}[/dim]")
            console.print(Rule(style="dim"))
            _stream_response(agent, f"[PC Agent direto] {task}")
        return

    if low.startswith("/imagem "):
        parts = user_input[8:].strip().split(" ", 1)
        task = parts[0]
        img_path = parts[1] if len(parts) > 1 else ""
        if task:
            console.print(f"\n[bold green]Jailson[/bold green] → [yellow]Image Agent[/yellow]: [dim]{task}[/dim]")
            console.print(Rule(style="dim"))
            msg = f"[Image Agent direto] {task}"
            if img_path:
                msg += f" — caminho: {img_path}"
            _stream_response(agent, msg)
        return

    if low.startswith("/memoria "):
        content = user_input[9:].strip()
        if content:
            agent.store_to_memory(content, "long_term")
            console.print(f"[dim green]✓ Guardado na memória longa: {content[:60]}...[/dim green]")
        return

    if low.startswith("/sagrado "):
        content = user_input[9:].strip()
        if content:
            agent.store_to_memory(content, "sacred")
            console.print(f"[dim green]✓ Guardado na memória sagrada: {content[:60]}...[/dim green]")
        return

    if low.startswith("/pesquisar "):
        query = user_input[11:].strip()
        if query:
            result = agent.search_memory(query)
            console.print(Markdown(result))
        return

    # ── Regular chat ───────────────────────────────────────────────────────────
    console.print(f"\n[bold green]Jailson[/bold green]:", end=" ")
    _stream_response(agent, user_input)


def _stream_response(agent, user_input: str):
    """Stream the agent response to the console."""
    try:
        buffer = ""
        for chunk in agent.chat_stream(user_input):
            if chunk.startswith("\n[Usando:"):
                # Tool call indicator — print on new line with styling
                console.print(f"\n[dim yellow]{chunk.strip()}[/dim yellow]", end="")
            elif chunk.startswith("\n[Pensando"):
                console.print(f"\n[dim blue]{chunk.strip()}[/dim blue]", end="")
            else:
                console.print(chunk, end="", markup=False)
                buffer += chunk
        console.print()  # newline after response
        console.print()  # spacing
    except KeyboardInterrupt:
        console.print("\n[dim](interrompido)[/dim]")
    except Exception as e:
        console.print(f"\n[red]Erro: {e}[/red]")
        if os.getenv("DEBUG", "").lower() == "true":
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

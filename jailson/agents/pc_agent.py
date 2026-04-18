"""PC Agent — specialist in file operations, system info, and PC organisation."""
import json
import anthropic

from jailson.config.settings import SUBAGENT_MODEL
from jailson.tools.file_reader import read_file, list_directory, search_files
from jailson.tools.system_info import (
    get_hardware_info,
    get_running_processes,
    get_disk_usage,
    get_installed_software,
    get_environment_info,
)

# ── Tool definitions ───────────────────────────────────────────────────────────

PC_TOOLS = [
    {
        "name": "read_file",
        "description": "Lê o conteúdo de um ficheiro (texto, PDF, DOCX, XLSX, JSON, código, etc.)",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Caminho absoluto ou relativo para o ficheiro"},
                "max_chars": {"type": "integer", "description": "Máximo de caracteres a retornar (default: 8000)"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "list_directory",
        "description": "Lista o conteúdo de uma pasta",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Caminho da pasta"},
                "show_hidden": {"type": "boolean", "description": "Mostrar ficheiros ocultos (default: false)"},
                "recursive": {"type": "boolean", "description": "Listar recursivamente (default: false)"},
                "max_items": {"type": "integer", "description": "Máximo de items (default: 200)"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "search_files",
        "description": "Pesquisa ficheiros por nome e/ou conteúdo numa pasta",
        "input_schema": {
            "type": "object",
            "properties": {
                "base_path": {"type": "string", "description": "Pasta raiz para pesquisar"},
                "pattern": {"type": "string", "description": "Padrão de nome de ficheiro"},
                "content_query": {"type": "string", "description": "Texto a procurar dentro dos ficheiros"},
                "extensions": {"type": "array", "items": {"type": "string"}, "description": "Extensões a filtrar, ex: ['.py', '.txt']"},
                "max_results": {"type": "integer", "description": "Máximo de resultados (default: 50)"},
            },
            "required": ["base_path"],
        },
    },
    {
        "name": "get_hardware_info",
        "description": "Obtém informação sobre o hardware: CPU, RAM, discos, sistema operativo",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_running_processes",
        "description": "Lista os processos em execução ordenados por uso de memória ou CPU",
        "input_schema": {
            "type": "object",
            "properties": {
                "top_n": {"type": "integer", "description": "Número de processos (default: 20)"},
                "sort_by": {"type": "string", "enum": ["memory", "cpu"], "description": "Ordenar por (default: memory)"},
            },
        },
    },
    {
        "name": "get_disk_usage",
        "description": "Obtém uso detalhado do disco para um caminho",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Caminho (default: /)"},
            },
        },
    },
    {
        "name": "get_installed_software",
        "description": "Lista o software instalado no sistema",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_environment_info",
        "description": "Obtém informação sobre o ambiente Python e variáveis de ambiente",
        "input_schema": {"type": "object", "properties": {}},
    },
]

# ── Tool dispatcher ────────────────────────────────────────────────────────────

def _execute_tool(name: str, inputs: dict) -> str:
    try:
        if name == "read_file":
            result = read_file(inputs["path"], inputs.get("max_chars", 8000))
        elif name == "list_directory":
            result = list_directory(
                inputs["path"],
                show_hidden=inputs.get("show_hidden", False),
                recursive=inputs.get("recursive", False),
                max_items=inputs.get("max_items", 200),
            )
        elif name == "search_files":
            result = search_files(
                inputs["base_path"],
                pattern=inputs.get("pattern", ""),
                content_query=inputs.get("content_query", ""),
                extensions=inputs.get("extensions"),
                max_results=inputs.get("max_results", 50),
            )
        elif name == "get_hardware_info":
            result = get_hardware_info()
        elif name == "get_running_processes":
            result = get_running_processes(
                top_n=inputs.get("top_n", 20),
                sort_by=inputs.get("sort_by", "memory"),
            )
        elif name == "get_disk_usage":
            result = get_disk_usage(inputs.get("path", "/"))
        elif name == "get_installed_software":
            result = get_installed_software()
        elif name == "get_environment_info":
            result = get_environment_info()
        else:
            result = {"error": f"Ferramenta desconhecida: {name}"}
        return json.dumps(result, ensure_ascii=False, default=str)[:6000]
    except Exception as e:
        return json.dumps({"error": str(e)})


# ── Agent runner ───────────────────────────────────────────────────────────────

def run_pc_agent(task: str, context: str = "", client: anthropic.Anthropic = None) -> str:
    """Run the PC specialist sub-agent and return its findings as text.

    Uses claude-haiku-4-5 to keep cost minimal.
    """
    if client is None:
        client = anthropic.Anthropic()

    system = (
        "És o PC Agent, um sub-agente especializado em organização e análise de computadores. "
        "Tens acesso a ferramentas para ler ficheiros, listar pastas, pesquisar conteúdo e obter "
        "informação de hardware e software. Responde em português de forma concisa e factual. "
        "Usa as ferramentas disponíveis para completar a tarefa solicitada."
    )

    messages = [{"role": "user", "content": f"Tarefa: {task}\n\n{context}".strip()}]

    for _ in range(8):  # max 8 tool-use iterations
        response = client.messages.create(
            model=SUBAGENT_MODEL,
            max_tokens=4096,
            system=system,
            tools=PC_TOOLS,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            return _extract_text(response)

        if response.stop_reason != "tool_use":
            return _extract_text(response)

        # Process tool calls
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = _execute_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })
        messages.append({"role": "user", "content": tool_results})

    return _extract_text(response)


def _extract_text(response) -> str:
    parts = []
    for block in response.content:
        if hasattr(block, "text"):
            parts.append(block.text)
    return "\n".join(parts) or "PC Agent completou a tarefa sem output textual."

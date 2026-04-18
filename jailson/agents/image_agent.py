"""Horus — image specialist agent with persistent photo library."""
import json
import anthropic

from jailson.config.settings import SUBAGENT_MODEL
from jailson.tools.image_analyzer import (
    get_image_metadata,
    scan_image_folder,
    encode_image_for_claude,
    detect_faces_basic,
    suggest_image_tags,
)
from jailson.tools.photo_manager import (
    estimate_folder_cost,
    analyze_photo,
    analyze_folder_batch,
    move_to_trash,
    empty_trash,
    find_duplicates,
    find_similar,
    get_library,
)

# ── Tool definitions ───────────────────────────────────────────────────────────

IMAGE_TOOLS = [
    {
        "name": "estimate_folder_cost",
        "description": "Estima quantas fotos são novas (nunca analisadas) e o custo aproximado em tokens antes de analisar",
        "input_schema": {
            "type": "object",
            "properties": {
                "folder_path": {"type": "string", "description": "Pasta a estimar"},
            },
            "required": ["folder_path"],
        },
    },
    {
        "name": "analyze_folder_batch",
        "description": "Analisa um lote de fotos novas numa pasta (usa biblioteca — fotos já analisadas são gratuitas)",
        "input_schema": {
            "type": "object",
            "properties": {
                "folder_path": {"type": "string", "description": "Pasta a analisar"},
                "max_new": {"type": "integer", "description": "Máximo de fotos novas a analisar (default: 20)"},
            },
            "required": ["folder_path"],
        },
    },
    {
        "name": "analyze_single_photo",
        "description": "Analisa uma única foto (usa biblioteca — zero tokens se já analisada)",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Caminho da foto"},
                "force": {"type": "boolean", "description": "Forçar re-análise mesmo se já na biblioteca (default: false)"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "find_duplicates",
        "description": "Encontra fotos duplicadas (mesmo hash ou mesmo nome+tamanho)",
        "input_schema": {
            "type": "object",
            "properties": {
                "folder_path": {"type": "string", "description": "Pasta a pesquisar (vazio = toda a biblioteca)"},
            },
        },
    },
    {
        "name": "find_similar",
        "description": "Encontra fotos visualmente parecidas usando hash perceptual (séries de rajada, etc.)",
        "input_schema": {
            "type": "object",
            "properties": {
                "folder_path": {"type": "string", "description": "Pasta a pesquisar"},
                "threshold": {"type": "integer", "description": "Sensibilidade 0-20 (default: 10 — mais baixo = mais parecidas)"},
            },
        },
    },
    {
        "name": "move_to_trash",
        "description": "Move uma foto para a pasta de lixo (~/.jailson/trash/) antes de apagar definitivamente",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Caminho da foto"},
                "reason": {"type": "string", "description": "Motivo (lixo, duplicada, etc.)"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "empty_trash",
        "description": "Apaga permanentemente todas as fotos na pasta de lixo",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "search_library",
        "description": "Pesquisa na biblioteca de fotos por tags ou descrição",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Texto a pesquisar (tag, descrição, etc.)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "library_stats",
        "description": "Mostra estatísticas da biblioteca de fotos (total analisadas, no lixo, etc.)",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_image_metadata",
        "description": "Obtém metadados de uma imagem: dimensões, formato, dados EXIF (data, câmara, GPS, etc.)",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Caminho para a imagem"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "scan_image_folder",
        "description": "Varre uma pasta e retorna lista de todas as imagens encontradas com metadados básicos",
        "input_schema": {
            "type": "object",
            "properties": {
                "folder_path": {"type": "string", "description": "Caminho da pasta"},
                "recursive": {"type": "boolean", "description": "Incluir sub-pastas (default: true)"},
                "max_images": {"type": "integer", "description": "Máximo de imagens (default: 500)"},
            },
            "required": ["folder_path"],
        },
    },
    {
        "name": "analyze_image_with_vision",
        "description": "Analisa o conteúdo visual de uma imagem usando Claude Vision — descreve o que vê, deteta rostos, sugere tags",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Caminho para a imagem"},
                "question": {"type": "string", "description": "Pergunta específica sobre a imagem (opcional)"},
                "detect_faces": {"type": "boolean", "description": "Focar na deteção e descrição de rostos (default: false)"},
                "suggest_tags": {"type": "boolean", "description": "Sugerir tags para organização (default: true)"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "detect_faces",
        "description": "Deteta rostos numa imagem usando a biblioteca face_recognition (requer instalação adicional)",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Caminho para a imagem"},
            },
            "required": ["path"],
        },
    },
]

# ── Tool dispatcher ────────────────────────────────────────────────────────────

def _execute_image_tool(name: str, inputs: dict, client: anthropic.Anthropic) -> str:
    try:
        if name == "estimate_folder_cost":
            return json.dumps(estimate_folder_cost(inputs["folder_path"]), ensure_ascii=False, default=str)

        elif name == "analyze_folder_batch":
            return json.dumps(analyze_folder_batch(
                inputs["folder_path"],
                max_new=inputs.get("max_new", 20),
                client=client,
            ), ensure_ascii=False, default=str)

        elif name == "analyze_single_photo":
            return json.dumps(analyze_photo(
                inputs["path"],
                client=client,
                force=inputs.get("force", False),
            ), ensure_ascii=False, default=str)

        elif name == "find_duplicates":
            return json.dumps(find_duplicates(inputs.get("folder_path", "")), ensure_ascii=False, default=str)

        elif name == "find_similar":
            return json.dumps(find_similar(
                inputs.get("folder_path", ""),
                threshold=inputs.get("threshold", 10),
            ), ensure_ascii=False, default=str)

        elif name == "move_to_trash":
            return json.dumps(move_to_trash(inputs["path"], inputs.get("reason", "")), ensure_ascii=False)

        elif name == "empty_trash":
            return json.dumps(empty_trash(), ensure_ascii=False)

        elif name == "search_library":
            results = get_library().search_by_tags(inputs["query"])
            return json.dumps({"success": True, "results": [
                {"path": r["path"], "tags": r["tags"], "description": r["description"]}
                for r in results
            ]}, ensure_ascii=False)

        elif name == "library_stats":
            return json.dumps(get_library().stats(), ensure_ascii=False)

        elif name == "get_image_metadata":
            result = get_image_metadata(inputs["path"])
            return json.dumps(result, ensure_ascii=False, default=str)

        elif name == "scan_image_folder":
            result = scan_image_folder(
                inputs["folder_path"],
                recursive=inputs.get("recursive", True),
                max_images=inputs.get("max_images", 500),
            )
            # Truncate image list to avoid flooding context
            if result.get("images") and len(result["images"]) > 20:
                result["images_sample"] = result["images"][:20]
                result["images_truncated"] = True
                del result["images"]
            return json.dumps(result, ensure_ascii=False, default=str)

        elif name == "analyze_image_with_vision":
            return _analyze_image_vision(inputs, client)

        elif name == "detect_faces":
            result = detect_faces_basic(inputs["path"])
            return json.dumps(result, ensure_ascii=False, default=str)

        else:
            return json.dumps({"error": f"Ferramenta desconhecida: {name}"})
    except Exception as e:
        return json.dumps({"error": str(e)})


def _analyze_image_vision(inputs: dict, client: anthropic.Anthropic) -> str:
    """Use Claude vision to analyze an image."""
    path = inputs["path"]
    question = inputs.get("question", "")
    detect_faces = inputs.get("detect_faces", False)
    suggest_tags_flag = inputs.get("suggest_tags", True)

    encoded = encode_image_for_claude(path)
    if not encoded:
        return json.dumps({"error": f"Não foi possível codificar a imagem: {path}"})

    if detect_faces:
        prompt = "Analisa esta imagem. Quantos rostos vês? Descreve-os brevemente (expressão, idade aproximada, género se visível). Lista as características principais."
    elif question:
        prompt = question
    else:
        parts = ["Analisa esta imagem e fornece:"]
        parts.append("1. Descrição geral do conteúdo")
        parts.append("2. Elementos principais visíveis")
        parts.append("3. Ambiente/contexto (interior/exterior, local, etc.)")
        if suggest_tags_flag:
            parts.append("4. Lista de tags sugeridas para organização (ex: praia, família, viagem)")
        parts.append("\nSê conciso e factual.")
        prompt = "\n".join(parts)

    try:
        response = client.messages.create(
            model=SUBAGENT_MODEL,
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": encoded["media_type"],
                            "data": encoded["data"],
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        analysis = response.content[0].text if response.content else ""

        # Get metadata for tag suggestions
        meta = get_image_metadata(path)
        tags = suggest_image_tags(meta if meta.get("success") else {}, analysis)

        return json.dumps({
            "success": True,
            "path": path,
            "analysis": analysis,
            "suggested_tags": tags,
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ── Agent runner ───────────────────────────────────────────────────────────────

def run_image_agent(task: str, image_path: str = "", client: anthropic.Anthropic = None) -> str:
    """Run Horus — the image specialist agent with persistent photo library."""
    if client is None:
        client = anthropic.Anthropic()

    system = (
        "És o Horus, o agente especialista em imagens do Jailson. "
        "Tens uma biblioteca persistente de fotos já analisadas — zero tokens para fotos conhecidas. "
        "Antes de analisar uma pasta SEMPRE usa estimate_folder_cost para informar o custo. "
        "Usa analyze_folder_batch para analisar em lote (máx 20 por pedido). "
        "Podes encontrar duplicadas, fotos parecidas (séries de rajada), mover lixo e pesquisar por tags. "
        "Responde em português de forma clara e organizada."
    )

    content = f"Tarefa: {task}"
    if image_path:
        content += f"\nImagem/Pasta: {image_path}"

    messages = [{"role": "user", "content": content}]

    for _ in range(6):  # max 6 iterations
        response = client.messages.create(
            model=SUBAGENT_MODEL,
            max_tokens=4096,
            system=system,
            tools=IMAGE_TOOLS,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            return _extract_text(response)

        if response.stop_reason != "tool_use":
            return _extract_text(response)

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = _execute_image_tool(block.name, block.input, client)
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
    return "\n".join(parts) or "Horus completou a tarefa sem output textual."

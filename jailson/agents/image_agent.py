"""Image Agent — specialist in image analysis, tagging, and facial recognition."""
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

# ── Tool definitions ───────────────────────────────────────────────────────────

IMAGE_TOOLS = [
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
        if name == "get_image_metadata":
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
    """Run the Image specialist sub-agent and return its analysis."""
    if client is None:
        client = anthropic.Anthropic()

    system = (
        "És o Image Agent, um sub-agente especializado em análise de imagens. "
        "Podes analisar imagens visualmente, extrair metadados EXIF, varrer pastas, "
        "detectar rostos e sugerir tags de organização. "
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
    return "\n".join(parts) or "Image Agent completou a tarefa sem output textual."

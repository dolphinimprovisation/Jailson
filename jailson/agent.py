"""Jailson — Main orchestrator agent with tool use, streaming, and prompt caching."""
import json
import os
from typing import Iterator, Optional

import anthropic

from jailson.config.settings import JAILSON_MODEL, load_charter, charter_to_text, SHOW_THINKING
from jailson.memory.memory_manager import MemoryManager
from jailson.memory.temporal import build_temporal_context
from jailson.agents.pc_agent import run_pc_agent
from jailson.agents.image_agent import run_image_agent

# ── Tool definitions for Jailson's orchestration layer ────────────────────────

JAILSON_TOOLS = [
    {
        "name": "dispatch_pc_agent",
        "description": (
            "Envia uma tarefa ao PC Agent — sub-agente especializado em: "
            "leitura de ficheiros, listagem de pastas, pesquisa de conteúdo, "
            "informação de hardware/software, processos em execução, uso de disco."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Descrição clara da tarefa para o PC Agent",
                },
                "context": {
                    "type": "string",
                    "description": "Contexto adicional para ajudar o PC Agent",
                },
            },
            "required": ["task"],
        },
    },
    {
        "name": "dispatch_image_agent",
        "description": (
            "Envia uma tarefa ao Image Agent — sub-agente especializado em: "
            "análise visual de imagens, metadados EXIF, varrimento de pastas de imagens, "
            "deteção de rostos, sugestão de tags."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Descrição clara da tarefa para o Image Agent",
                },
                "image_path": {
                    "type": "string",
                    "description": "Caminho da imagem ou pasta de imagens (opcional)",
                },
            },
            "required": ["task"],
        },
    },
    {
        "name": "store_memory",
        "description": (
            "Guarda informação importante na memória persistente. "
            "Usa para factos do utilizador, padrões observados, eventos significativos, "
            "ou informação nuclear (sagrada)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Conteúdo a guardar",
                },
                "memory_type": {
                    "type": "string",
                    "enum": ["long_term", "sacred", "episodic", "pattern", "insight"],
                    "description": (
                        "Tipo de memória: "
                        "long_term=factos/preferências permanentes, "
                        "sacred=informação imutável e nuclear do utilizador, "
                        "episodic=evento específico com contexto temporal, "
                        "pattern=padrão de comportamento repetido, "
                        "insight=observação dos últimos dias"
                    ),
                },
                "category": {
                    "type": "string",
                    "description": "Categoria: preferencia, habito, pessoa, trabalho, hobby, saude, etc.",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags para facilitar pesquisa",
                },
                "importance": {
                    "type": "integer",
                    "description": "Importância de 1-10 (só para episódico)",
                },
                "title": {
                    "type": "string",
                    "description": "Título curto (só para episódico)",
                },
            },
            "required": ["content", "memory_type"],
        },
    },
    {
        "name": "search_memories",
        "description": "Pesquisa semântica em toda a memória persistente.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "O que procurar",
                },
                "n": {
                    "type": "integer",
                    "description": "Número de resultados (default: 5)",
                },
            },
            "required": ["query"],
        },
    },
]


class JailsonAgent:
    """Main Jailson orchestrator.

    Maintains conversation state, dispatches to sub-agents, and manages memory.
    Uses prompt caching for the charter (1h TTL) and stable memory context (5min TTL).
    """

    def __init__(self):
        self.client = anthropic.Anthropic()
        self.memory = MemoryManager()
        charter = load_charter()
        self._charter_text = charter_to_text(charter)
        self._conversation: list[dict] = []
        self._session_counter = 0

    # ── System prompt with caching ─────────────────────────────────────────────

    def _build_system(self) -> list[dict]:
        """Construct the system prompt.

        Structure (for optimal caching):
        1. Charter text — stable, cache 1h
        2. Stable memory (sacred + long-term + episodic) — semi-stable, cache 5min
        3. Recent context (medium-term insights) — volatile, no cache
        """
        blocks = []

        # 1. Charter — stable for hours → 1h cache
        blocks.append({
            "type": "text",
            "text": self._charter_text + "\n\n---\n\nRespondes SEMPRE em português.",
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        })

        # 2. Stable memory context — semi-stable → 5min cache
        stable_ctx = self.memory.get_stable_context()
        if stable_ctx:
            blocks.append({
                "type": "text",
                "text": stable_ctx,
                "cache_control": {"type": "ephemeral"},
            })

        # 3. Recent volatile context — no cache
        recent_ctx = self.memory.get_recent_context()
        if recent_ctx:
            blocks.append({
                "type": "text",
                "text": recent_ctx,
            })

        # 4. Temporal context — always fresh, never cached
        blocks.append({
            "type": "text",
            "text": build_temporal_context(),
        })

        return blocks

    # ── Tool execution ─────────────────────────────────────────────────────────

    def _execute_tool(self, name: str, inputs: dict) -> str:
        try:
            if name == "dispatch_pc_agent":
                result = run_pc_agent(
                    task=inputs["task"],
                    context=inputs.get("context", ""),
                    client=self.client,
                )
                return result

            elif name == "dispatch_image_agent":
                result = run_image_agent(
                    task=inputs["task"],
                    image_path=inputs.get("image_path", ""),
                    client=self.client,
                )
                return result

            elif name == "store_memory":
                self.memory.store(
                    content=inputs["content"],
                    memory_type=inputs["memory_type"],
                    category=inputs.get("category", "general"),
                    tags=inputs.get("tags", []),
                    importance=inputs.get("importance", 5),
                    title=inputs.get("title", ""),
                    source="jailson",
                )
                return json.dumps({"success": True, "message": f"Guardado em memória '{inputs['memory_type']}'"})

            elif name == "search_memories":
                return self.memory.search_summary(
                    query=inputs["query"],
                    n=inputs.get("n", 5),
                )

            else:
                return json.dumps({"error": f"Ferramenta desconhecida: {name}"})

        except Exception as e:
            return json.dumps({"error": str(e)})

    # ── Main interaction loop ──────────────────────────────────────────────────

    def chat_stream(self, user_message: str) -> Iterator[str]:
        """Process a user message and stream the response.

        Yields text chunks as they arrive. Tool calls are handled transparently.
        """
        self._session_counter += 1
        self.memory.record_turn("user", user_message)
        self._conversation.append({"role": "user", "content": user_message})

        system = self._build_system()

        for iteration in range(10):  # max agentic loop iterations
            # Build messages: history + current conversation
            messages = self._conversation.copy()

            stream_kwargs = dict(
                model=JAILSON_MODEL,
                max_tokens=4096,
                system=system,
                tools=JAILSON_TOOLS,
                messages=messages,
            )
            if SHOW_THINKING:
                stream_kwargs["thinking"] = {"type": "adaptive"}

            with self.client.messages.stream(**stream_kwargs) as stream:
                full_response = ""
                thinking_text = ""
                tool_uses = []

                for event in stream:
                    if event.type == "content_block_start":
                        block = event.content_block
                        if block.type == "thinking" and SHOW_THINKING:
                            yield "\n[Pensando...]\n"

                    elif event.type == "content_block_delta":
                        delta = event.delta
                        if delta.type == "text_delta":
                            full_response += delta.text
                            yield delta.text
                        elif delta.type == "thinking_delta" and SHOW_THINKING:
                            thinking_text += delta.thinking
                            yield delta.thinking

                final = stream.get_final_message()

            # Collect tool uses from the final message
            tool_use_blocks = [b for b in final.content if b.type == "tool_use"]

            if final.stop_reason == "end_turn" or not tool_use_blocks:
                # Final response — record and finish
                if full_response:
                    self.memory.record_turn("assistant", full_response)
                    self._conversation.append({"role": "assistant", "content": full_response})
                    self._auto_extract_insights(user_message, full_response)
                break

            # Handle tool calls
            self._conversation.append({"role": "assistant", "content": final.content})
            tool_results = []

            for block in tool_use_blocks:
                yield f"\n[Usando: {block.name}...]\n"
                result = self._execute_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

            self._conversation.append({"role": "user", "content": tool_results})

        # Trim conversation to avoid unbounded growth (keep last 30 turns)
        if len(self._conversation) > 60:
            self._conversation = self._conversation[-50:]

    def chat(self, user_message: str) -> str:
        """Non-streaming version — returns full response string."""
        return "".join(self.chat_stream(user_message))

    # ── Auto insight extraction ────────────────────────────────────────────────

    def _auto_extract_insights(self, user_msg: str, assistant_msg: str):
        """Heuristically extract facts worth remembering from a conversation turn."""
        triggers = [
            ("chamo-me", "sacred", "nome"),
            ("o meu nome é", "sacred", "nome"),
            ("moro em", "long_term", "localizacao"),
            ("trabalho como", "long_term", "trabalho"),
            ("prefiro", "long_term", "preferencia"),
            ("gosto de", "long_term", "hobbie"),
            ("não gosto de", "long_term", "aversao"),
            ("sempre faço", "pattern", "habito"),
            ("todos os dias", "pattern", "habito"),
        ]
        user_lower = user_msg.lower()
        for trigger, mem_type, category in triggers:
            if trigger in user_lower:
                self.memory.store(
                    content=f"O utilizador disse: '{user_msg[:200]}'",
                    memory_type=mem_type,
                    category=category,
                    source="auto_extract",
                )
                break

    # ── Utilities ──────────────────────────────────────────────────────────────

    def reset_conversation(self):
        """Clear current conversation (keeps all memory)."""
        self._conversation.clear()
        self.memory.short.clear()

    def get_memory_stats(self) -> dict:
        return self.memory.stats()

    def store_to_memory(self, content: str, memory_type: str, **kwargs):
        """Direct memory store — used by CLI commands."""
        self.memory.store(content, memory_type, **kwargs)

    def search_memory(self, query: str) -> str:
        return self.memory.search_summary(query)

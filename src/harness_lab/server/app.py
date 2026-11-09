"""Local model server: one Qwen model behind three wire protocols.

    uv run harness-serve                       # Qwen3.5-4B on GPU/MPS/CPU, port 8000
    uv run harness-serve --model Qwen/Qwen3.5-0.8B
    uv run harness-serve --scripted            # no model at all (offline smoke tests)

Endpoints (all under http://127.0.0.1:8000):

    POST /v1/chat/completions   OpenAI Chat Completions   (raw calls, our mini harness)
    POST /v1/messages           Anthropic Messages        (DeepSeek Harness, Claude Code)
    POST /v1/responses          OpenAI Responses          (Codex CLI)
    GET  /v1/models, /health

Every request and response is saved as JSON under ``outputs/traffic/`` so you can open
exactly what each harness sends to the model: system prompt, tools, history.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from harness_lab import config
from harness_lab.server import anthropic_messages, chat_completions, openai_responses
from harness_lab.server.engine import Engine, ScriptedEngine


class TrafficLog:
    """Writes one JSON file per request: what the harness sent and what the model said."""

    def __init__(self, directory: Path | None):
        self.directory = directory
        self.counter = 0
        if directory:
            directory.mkdir(parents=True, exist_ok=True)

    def save(self, api: str, body: dict, output: Any) -> None:
        self.counter += 1
        n_tools = len(body.get("tools") or [])
        n_msgs = len(body.get("messages") or body.get("input") or [])
        print(f"[{time.strftime('%H:%M:%S')}] {api:<17} messages={n_msgs:<3} tools={n_tools:<3}")
        if not self.directory:
            return
        path = self.directory / f"{time.strftime('%Y%m%d-%H%M%S')}-{self.counter:04d}-{api}.json"
        path.write_text(json.dumps({"api": api, "request": body, "response": output}, indent=2, ensure_ascii=False))


def create_app(engine: Engine, default_thinking: bool = False, traffic_dir: Path | None = None) -> FastAPI:
    app = FastAPI(title="harness-lab local model server")
    log = TrafficLog(traffic_dir)

    def run(api: str, module, body: dict):
        result = module.respond(engine, body, default_thinking)
        if isinstance(result, dict):
            log.save(api, body, result)
            return JSONResponse(result)

        def logged() -> Iterator[str]:
            chunks = []
            for chunk in result:
                chunks.append(chunk)
                yield chunk
            log.save(api, body, "".join(chunks))

        return StreamingResponse(logged(), media_type="text/event-stream")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": engine.model_id}

    @app.get("/v1/models")
    def models():
        return {
            "object": "list",
            "data": [
                {
                    "id": engine.model_id,
                    "object": "model",
                    "owned_by": "local",
                    "type": "model",
                    "display_name": engine.model_id,
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        return run("chat.completions", chat_completions, await request.json())

    @app.post("/v1/messages")
    async def messages(request: Request):
        return run("messages", anthropic_messages, await request.json())

    @app.post("/v1/messages/count_tokens")
    async def count_tokens(request: Request):
        return anthropic_messages.count_tokens(engine, await request.json())

    @app.post("/v1/responses")
    async def responses(request: Request):
        return run("responses", openai_responses, await request.json())

    return app


DEMO_MARKDOWN = """# Informe de prueba

Generado por el **motor simulado** (no hay modelo cargado).

- Un harness es el bucle alrededor del modelo.
- Las herramientas las ejecuta el harness, no el modelo.

| Capa | Ejemplo |
|------|---------|
| Modelo | Qwen3.5-4B |
| Harness | DeepSeek Harness |
"""


def demo_script(request) -> str:
    """Scripted engine used by ``--scripted``: one sensible tool call, then a short answer.

    Lets you run the whole tutorial without a GPU. The answers are fixed, of course.
    """
    from harness_lab.server.toolcalls import CALL_CLOSE, CALL_OPEN

    last = request.messages[-1]
    if last["role"] == "tool":
        return "Hecho (motor simulado). Resultado de la herramienta: " + str(last["content"])[:300]
    names = {t["function"]["name"]: t["function"] for t in request.tools or []}
    user = " ".join(str(m["content"]) for m in request.messages if m["role"] == "user").lower()

    def call(name: str, **params) -> str:
        body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
        return f"{CALL_OPEN}\n<function={name}>\n{body}</function>\n{CALL_CLOSE}"

    word = next((n for n in names if n.endswith("create_word_document")), None)
    if word and ("word" in user or "docx" in user or "documento" in user):
        named = re.search(r"[\w.-]+\.docx", user)
        filename = named.group(0) if named else "informe.docx"
        return "Creo el documento.\n\n" + call(word, filename=filename, markdown=DEMO_MARKDOWN, title="Informe")
    for shell, arg in (("bash", "command"), ("Bash", "command"), ("run_shell", "command"), ("exec_command", "cmd")):
        if shell in names:
            return call(shell, **{arg: "ls"})
    if names:
        return call(next(iter(names)))
    return "Respuesta simulada: no hay ningún modelo cargado (--scripted)."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model", default=config.MODEL_ID, help="Hugging Face repo id or local path")
    parser.add_argument("--host", default=config.HOST)
    parser.add_argument("--port", type=int, default=config.PORT)
    parser.add_argument("--device", default=None, help="cuda | mps | cpu (default: auto)")
    parser.add_argument("--max-context", type=int, default=32768, help="prompt + output token budget")
    parser.add_argument(
        "--thinking",
        choices=["auto", "on", "off"],
        default="auto",
        help="auto: follow the client (default); on/off: force it for every request",
    )
    parser.add_argument("--traffic-dir", default=str(config.OUTPUTS_DIR / "traffic"), help="'' to disable")
    parser.add_argument("--scripted", action="store_true", help="no model: scripted answers for offline tests")
    args = parser.parse_args()

    import uvicorn

    if args.scripted:
        engine: Engine = ScriptedEngine(demo_script)
    else:
        from harness_lab.server.engine import TransformersEngine

        engine = TransformersEngine(args.model, device=args.device, max_context=args.max_context)
    traffic = Path(args.traffic_dir) if args.traffic_dir else None
    engine.thinking_policy = args.thinking
    app = create_app(engine, traffic_dir=traffic)
    print(f"Serving {engine.model_id} at http://{args.host}:{args.port}/v1")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

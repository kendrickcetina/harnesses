# harnesses

**Práctica de harnesses de agentes · Modelos Generativos Profundos · MIA · ICAI · 2026-2027**

Un modelo de lenguaje solo transforma texto en texto. Un **harness** es el programa que lo
convierte en agente: el bucle, las herramientas, los permisos, la memoria y el contexto.
En este repositorio usamos un modelo local, **Qwen3.5-4B**, como cerebro de tres harnesses:

- **DeepSeek Harness** (`dsh`): el foco del tutorial. Abierto (MIT), "todo es un plugin".
- **Codex CLI** (OpenAI) y **Claude Code** (Anthropic): opcionales, mismo modelo y mismas tools.

Le añadimos una herramienta propia por MCP que genera **documentos Word** y terminamos con
**weight-space programming**: leer y editar los pesos de Qwen sin gradientes.

- Tutorial: [`tutorial.ipynb`](tutorial.ipynb) (en español)
- Presentación: [`slides/index.html`](slides/index.html) (ábrela en el navegador)

## Empezar

En la DGX (terminal de code-server, dentro de `~/clusters/dgx`):

```bash
git clone https://github.com/kendrickcetina/harnesses.git
cd harnesses
source scripts/dgx_env.sh      # cada sesión: cachés de uv y Hugging Face en clusters/dgx
uv sync                        # todo: torch, transformers, DeepSeek Harness, MCP, python-docx, Jupyter
```

Abre `tutorial.ipynb` con el kernel `.venv/bin/python`. El notebook descarga el modelo,
arranca el servidor y recorre todo.

En un portátil sin GPU puedes ensayar con un modelo simulado (respuestas fijas):
`HARNESS_SCRIPTED=1 uv run jupyter lab`.

## Comandos

```bash
uv run harness-download                 # Qwen3.5-4B (9,3 GB) al caché de Hugging Face
uv run harness-serve                    # servidor local en http://127.0.0.1:8000/v1
uv run harness-dsh "Crea un Word que explique qué es un harness"
uv run harness-wsp diff                 # qué cambió la edición de weight-space programming
uv run pytest                           # tests offline: sin GPU ni descargas
bash scripts/install_clis.sh            # opcional: Codex y Claude Code sin root
```

## Cómo está organizado

| Ruta | Qué hay |
|---|---|
| `src/harness_lab/server/` | Servidor del modelo: una instancia de Qwen detrás de tres APIs |
| `src/harness_lab/mini_harness.py` | Un harness en ~100 líneas: el bucle básico |
| `src/harness_lab/dsh.py` | DeepSeek Harness apuntado al modelo local, con nuestras tools |
| `src/harness_lab/clis.py` | Codex y Claude Code apuntados al modelo local |
| `src/harness_lab/tools/` | Tools propias: Word con python-docx, servidor MCP |
| `src/harness_lab/wsp/` | Weight-space programming: leer, medir y editar pesos |
| `src/harness_lab/traffic.py` | Ver qué envía cada harness al modelo |
| `tests/` | Tests offline con un modelo simulado (incluye ejecutar el notebook entero) |

El servidor ofrece las tres APIs porque cada harness habla una:

| Endpoint | API | Harness |
|---|---|---|
| `/v1/chat/completions` | OpenAI Chat Completions | harness mínimo |
| `/v1/messages` | Anthropic Messages | DeepSeek Harness, Claude Code |
| `/v1/responses` | OpenAI Responses | Codex |

Cada petición se guarda en `outputs/traffic/`: ahí se ve el system prompt y las tools que
mete cada harness antes de vuestra pregunta.

## Notas

- **GPU.** Qwen3.5-4B en bf16 cabe en la partición de 16 GB de la DGX. Con poca memoria:
  `uv run harness-serve --model Qwen/Qwen3.5-2B` o `--max-context 16384`.
- **Velocidad.** El servidor usa `transformers` (una petición a la vez). Para ir más rápido en
  Linux: `uv sync --extra fast` (kernels de flash-linear-attention para Qwen3.5).
- **Thinking.** Por defecto el servidor sigue lo que pide el cliente. Claude Code siempre lo
  pide; con `uv run harness-serve --thinking off` se fuerza a no pensar (más rápido).
- **Privacidad.** `dsh.py` desactiva las filas de DeepSeek Harness que suben logs de sesión.
- **Claude Code en la DGX** se instala en `~/.local`, que no persiste entre sesiones: repetid
  `bash scripts/install_clis.sh claude` si hace falta.
- **DeepSeek Harness** está en *developer preview*: fijamos la versión `0.2.1rc2`.

## Referencias

- DeepSeek Harness: <https://github.com/deepseek-ai/deepseek-harness>
- Codex CLI: <https://github.com/openai/codex>
- Claude Code: <https://docs.claude.com/en/docs/claude-code>
- Qwen3.5: <https://huggingface.co/Qwen/Qwen3.5-4B>
- Model Context Protocol: <https://modelcontextprotocol.io>
- Weight-space programming: <https://rikiyat.github.io/blog/weight-space-programming/>
- Práctica ARCA: <https://github.com/kendrickcetina/dgm-arca>

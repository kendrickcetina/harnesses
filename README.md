# harnesses

**Práctica de harnesses de agentes · Modelos Generativos Profundos · MIA · ICAI · 2026-2027**

Un modelo de lenguaje solo transforma texto en texto. Un **harness** es el programa que lo
convierte en agente: el bucle, las herramientas, los permisos, la memoria y el contexto.
En este repositorio usamos un modelo local, **Qwen3.5-4B**, como cerebro de tres harnesses:

- **DeepSeek Harness** (`dsh`): el foco del tutorial. Abierto (MIT), "todo es un plugin".
- **Codex CLI** (OpenAI) y **Claude Code** (Anthropic): opcionales, mismo modelo y mismas tools.

Le añadimos una herramienta propia por MCP que genera **documentos Word**, y Codex analiza un
Excel ejecutando Python y escribe un informe.

**Aparte**, sin relación con los harnesses: **weight-space programming**, mejorar un modelo
editando directamente sus matrices de pesos, sin gradientes. Está solo en las slides (qué es,
sus límites y una idea para aplicarlo a vuestro modelo de ARCA). No hay código: lo
implementáis vosotros.

- Tutorial: [`tutorial.ipynb`](tutorial.ipynb) (en español)
- Presentación: [`slides/index.html`](slides/index.html) (ábrela en el navegador)

## Empezar

Necesitas [uv](https://docs.astral.sh/uv/) y, para el modelo real, una GPU NVIDIA con ≥16 GB.

```bash
git clone https://github.com/kendrickcetina/harnesses.git
cd harnesses
uv sync                        # todo: torch, transformers, DeepSeek Harness, MCP, python-docx, Jupyter
```

Abre `tutorial.ipynb` con el kernel `.venv/bin/python`. El notebook descarga el modelo
(~11 GB en total con los de la sección 6), arranca el servidor y recorre todo.

En un portátil sin GPU puedes ensayar con un modelo simulado (respuestas fijas):
`HARNESS_SCRIPTED=1 uv run jupyter lab`.

**Si tu `$HOME` se borra entre sesiones** (por ejemplo, en la DGX solo persiste
`~/clusters/dgx`), guarda las cachés en una carpeta que sí persista antes de `uv sync`;
si no, cada sesión vuelve a descargar los modelos y Python:

```bash
export HF_HOME=~/clusters/dgx/.hf_cache                   # modelos de Hugging Face
export UV_CACHE_DIR=~/clusters/dgx/.uv/cache              # paquetes de uv
export UV_PYTHON_INSTALL_DIR=~/clusters/dgx/.uv/python    # el Python al que apunta .venv
```

## Comandos

```bash
uv run harness-download                 # Qwen3.5-4B (9,3 GB) al caché de Hugging Face
uv run harness-serve                    # servidor local en http://127.0.0.1:8765/v1
uv run harness-dsh "Crea un Word que explique qué es un harness"
uv run pytest                           # tests offline: sin GPU ni descargas
bash scripts/install_clis.sh            # opcional: Codex y Claude Code sin root
```

## Tests

```bash
uv run pytest                                               # offline (~1 min): modelo simulado
HARNESS_TEST_GPU=1 uv run pytest tests/test_notebook.py -k gpu -s   # notebook real en GPU
```

- **Offline**: los tres protocolos del servidor, el parser de tool calls, las tools de Word,
  DeepSeek Harness de verdad contra un modelo simulado y el notebook entero con
  `HARNESS_SCRIPTED=1`.
- **GPU** (`-k gpu`, ~6 min): ejecuta `tutorial.ipynb` de principio a fin con Qwen3.5-4B y
  comprueba cada sección (tool calls reales, `.docx` creados por el harness mínimo y por
  DeepSeek Harness vía MCP).
  El notebook ejecutado queda en el directorio temporal de pytest (`tutorial.executed.ipynb`).
- Codex y Claude Code: `HARNESS_TEST_CLIS=1 uv run pytest` si están instalados.

## Cómo está organizado

| Ruta | Qué hay |
|---|---|
| `src/harness_lab/server/` | Servidor del modelo: una instancia de Qwen detrás de tres APIs |
| `src/harness_lab/mini_harness.py` | Un harness en ~100 líneas: el bucle básico |
| `src/harness_lab/dsh.py` | DeepSeek Harness apuntado al modelo local, con nuestras tools |
| `src/harness_lab/clis.py` | Codex y Claude Code apuntados al modelo local |
| `src/harness_lab/tools/` | Tools propias: Word con python-docx, servidor MCP |
| `src/harness_lab/traffic.py` | Ver qué envía cada harness al modelo |
| `tests/` | Tests offline con un modelo simulado y el notebook entero (simulado o en GPU) |

El servidor ofrece las tres APIs porque cada harness habla una:

| Endpoint | API | Harness |
|---|---|---|
| `/v1/chat/completions` | OpenAI Chat Completions | harness mínimo |
| `/v1/messages` | Anthropic Messages | DeepSeek Harness, Claude Code |
| `/v1/responses` | OpenAI Responses | Codex |

Cada petición se guarda en `outputs/traffic/`: ahí se ve el system prompt y las tools que
mete cada harness antes de vuestra pregunta.

**No se usa ningún servicio externo de modelos.** "OpenAI" y "Anthropic" son aquí solo
*formatos de API*: todas las peticiones van al servidor local y las responde Qwen. El
paquete `openai` (en `mini_harness.py` y el notebook) es una librería cliente HTTP apuntada a
`http://127.0.0.1:8765/v1`.

## Notas

- **GPU.** Qwen3.5-4B en bf16 cabe en la partición de 16 GB de la DGX. Con poca memoria:
  `uv run harness-serve --model Qwen/Qwen3.5-2B` o `--max-context 16384`.
- **Velocidad.** El servidor usa `transformers` (una petición a la vez): ~35 tokens/s con
  Qwen3.5-4B en una partición MIG de H200. El notebook completo tarda ~6 min. El servidor
  desactiva el backend cuDNN de atención, que en GPUs Hopper recompila un grafo por cada
  longitud nueva y deja la generación en ~2,6 tokens/s (`engine.py`, `pick_device`).
  `uv sync --extra fast` (flash-linear-attention) apenas mejora la generación (~5 %).
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

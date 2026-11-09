"""Download model weights into the Hugging Face cache.

uv run harness-download                     # default harness model (Qwen3.5-4B)
uv run harness-download --small             # Qwen3.5-0.8B, for laptops
uv run harness-download --wsp               # base + edited model for weight-space programming
uv run harness-download Qwen/Qwen3.5-2B     # any repo id
"""

from __future__ import annotations

import argparse

from huggingface_hub import snapshot_download

from harness_lab import config

WEIGHT_PATTERNS = ["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model", "*.py"]


def download(repo_id: str) -> str:
    """Download a model snapshot and return its local path."""
    path = snapshot_download(repo_id, allow_patterns=WEIGHT_PATTERNS)
    print(f"{repo_id} -> {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("repo_ids", nargs="*", help="Hugging Face repo ids")
    parser.add_argument("--small", action="store_true", help=f"download {config.SMALL_MODEL_ID}")
    parser.add_argument("--wsp", action="store_true", help="download the weight-space programming pair")
    args = parser.parse_args()

    repo_ids = list(args.repo_ids)
    if args.small:
        repo_ids.append(config.SMALL_MODEL_ID)
    if args.wsp:
        repo_ids += [config.WSP_BASE_MODEL_ID, config.WSP_EDITED_MODEL_ID]
    if not repo_ids:
        repo_ids = [config.MODEL_ID]
    for repo_id in dict.fromkeys(repo_ids):
        download(repo_id)


if __name__ == "__main__":
    main()

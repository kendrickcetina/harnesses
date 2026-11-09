#!/usr/bin/env bash
# Optional: install Codex CLI and Claude Code without root (Linux x86_64/aarch64 or macOS).
# DeepSeek Harness needs nothing here: it comes with `uv sync`.
#
#     bash scripts/install_clis.sh            # both
#     bash scripts/install_clis.sh codex      # only Codex
#     bash scripts/install_clis.sh claude     # only Claude Code
set -euo pipefail

BIN="${HARNESS_TOOLS_DIR:-$HOME/.local}/bin"
mkdir -p "$BIN"
WHAT="${1:-all}"

install_codex() {
    if command -v codex >/dev/null 2>&1; then echo "codex already installed: $(codex --version)"; return; fi
    case "$(uname -s)-$(uname -m)" in
        Linux-x86_64) target="x86_64-unknown-linux-musl" ;;
        Linux-aarch64) target="aarch64-unknown-linux-musl" ;;
        Darwin-arm64) target="aarch64-apple-darwin" ;;
        *) echo "Unsupported platform for the Codex binary"; return 1 ;;
    esac
    tmp="$(mktemp -d)"
    curl -fsSL "https://github.com/openai/codex/releases/latest/download/codex-${target}.tar.gz" -o "$tmp/codex.tgz"
    tar -xzf "$tmp/codex.tgz" -C "$tmp"
    mv "$tmp/codex-${target}" "$BIN/codex"
    chmod +x "$BIN/codex"
    # Codex's Linux sandbox needs bubblewrap; the release ships a static one, looked up next to codex.
    case "$target" in
        *linux*)
            curl -fsSL "https://github.com/openai/codex/releases/latest/download/bwrap-${target}.tar.gz" -o "$tmp/bwrap.tgz"
            tar -xzf "$tmp/bwrap.tgz" -C "$tmp"
            mkdir -p "$BIN/codex-resources"
            mv "$tmp/bwrap-${target}" "$BIN/codex-resources/bwrap"
            chmod +x "$BIN/codex-resources/bwrap"
            ;;
    esac
    rm -rf "$tmp"
    echo "codex installed in $BIN: $("$BIN/codex" --version)"
}

install_claude() {
    if command -v claude >/dev/null 2>&1; then echo "claude already installed: $(claude --version)"; return; fi
    # Official native installer; it writes to ~/.local by default.
    curl -fsSL https://claude.ai/install.sh | bash
    if [ -x "$HOME/.local/bin/claude" ] && [ "$BIN" != "$HOME/.local/bin" ]; then
        ln -sf "$HOME/.local/bin/claude" "$BIN/claude"
    fi
    echo "claude installed: $(claude --version 2>/dev/null || echo 'open a new shell')"
}

case "$WHAT" in
    codex) install_codex ;;
    claude) install_claude ;;
    all) install_codex; install_claude ;;
    *) echo "usage: $0 [codex|claude|all]"; exit 1 ;;
esac
echo "Make sure $BIN is on your PATH."

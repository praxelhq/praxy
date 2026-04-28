#!/usr/bin/env bash
# Build praxy_tts.pdf with full citation resolution. One command.
set -euo pipefail
cd "$(dirname "$0")"

# XeLaTeX so Telugu / Tamil / Devanagari script in prose renders natively.
# pdflatex fallback path kicks in only if xelatex is unavailable — but the
# Indic script characters will fail with pdflatex; use xelatex.
ENGINE="${PRAXY_LATEX_ENGINE:-xelatex}"
if ! command -v "$ENGINE" >/dev/null || ! command -v bibtex >/dev/null; then
  echo "[build] $ENGINE / bibtex not found on PATH."
  echo "[build] install MacTeX: brew install --cask mactex-no-gui"
  echo "[build] or TinyTeX (sudo-free): curl -sL yihui.org/tinytex/install-bin-unix.sh | sh"
  echo "[build] or use Overleaf — import this dir as a project."
  exit 1
fi

# 4-pass build: xelatex, bibtex, xelatex, xelatex.
"$ENGINE" -halt-on-error -interaction=nonstopmode praxy_tts.tex
bibtex praxy_tts
"$ENGINE" -halt-on-error -interaction=nonstopmode praxy_tts.tex
"$ENGINE" -halt-on-error -interaction=nonstopmode praxy_tts.tex

echo "[build] done → $(pwd)/praxy_tts.pdf"

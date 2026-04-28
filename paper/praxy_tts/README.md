# Building the Praxy TTS paper PDF

## Prerequisites

We render Telugu/Tamil/Hindi script inside prose and table captions, so
**XeLaTeX is required** — `pdflatex` will fail on Indic codepoints. You
need a TeX distribution with IEEEtran + tikz + fontspec.

### macOS (MacTeX, full install)

```bash
brew install --cask mactex-no-gui
```

### macOS / Linux (sudo-free, via TinyTeX)

```bash
curl -sL yihui.org/tinytex/install-bin-unix.sh | sh
tlmgr install IEEEtran tikz fontspec newunicodechar iftex
```

### No-install path

Upload `paper/praxy_tts/` to Overleaf as a new project. Set the compiler
to `XeLaTeX` in **Menu → Compiler**. Build.

## Build

The repo ships `./build.sh` which runs the correct 4-pass sequence
(xelatex → bibtex → xelatex → xelatex) and emits `praxy_tts.pdf`.

```bash
cd paper/praxy_tts/
./build.sh
```

Override the engine if needed: `PRAXY_LATEX_ENGINE=lualatex ./build.sh`
(any engine that speaks fontspec will do).

A single pass leaves citations showing as `[?]` because the `.bbl`
file hasn't been built yet — always use `build.sh`, not a single
compile call.

## Why this paper lives here and not in psp-eval

This is the **TTS paper**. The PSP paper (benchmark methodology) lives
at `github.com/praxelhq/psp-eval` and is cited here as `\cite{psp2026}`.
Do not add PSP paper content to this directory.

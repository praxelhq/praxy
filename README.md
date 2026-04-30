# 🎤 Praxy Voice

Open-source Hindi · Telugu · Tamil · English TTS — including code-mix and voice cloning. The reference implementation accompanying the paper *"Praxy Voice: Voice-Prompt Recovery + BUPS for Commercial-Class Indic TTS from a Frozen Non-Indic Base at Zero Commercial-Training-Data Cost"* ([arXiv:2604.25441](https://arxiv.org/abs/2604.25441)). Companion paper: *"PSP: An Interpretable Per-Dimension Accent Benchmark for Indic Text-to-Speech"* ([arXiv:2604.25476](https://arxiv.org/abs/2604.25476)).

| Asset | Where |
|---|---|
| **Live demo** | [`Praxel/praxy-voice-demo`](https://huggingface.co/spaces/Praxel/praxy-voice-demo) on HF Spaces |
| **Model weights (R6 LoRA)** | [`Praxel/praxy-voice-r6`](https://huggingface.co/Praxel/praxy-voice-r6) — Apache-2.0 |
| **Paper** | `paper/praxy_tts/praxy_tts.pdf` |
| **PSP companion** | [`praxelhq/psp-eval`](https://github.com/praxelhq/psp-eval) — accent benchmark (install from source until PyPI release) |

## What's here

| Module | Role |
|---|---|
| `serving/praxy_router.py` | Production router. Auto-detects code-mix and routes per-language to the right branch. |
| `serving/codemix_to_native_script.py` | Haiku-driven transliteration preprocessor — converts Latin English words to native-script phonetic spelling before synth. |
| `serving/gradio_app.py` | Modal-deployed Gradio app (preserved for future revisit; current production demo lives on HF Spaces). |
| `praxy/linguistics/bups.py` | Brahmic Unified Phoneme Space — ISO-15919 romanisation routing for Indic scripts. |
| `praxy/linguistics/indic_numbers.py` | Indic-language digit/currency/percent normaliser (Te/Hi/Ta/Bn/Gu/Mr/Kn/Ml). |
| `paper/praxy_tts/` | LaTeX source + built PDF for the paper. |
| `hf_release/praxy-voice-r6/` | The R6 release bundle: LoRA weights, model card, license. |
| `data/references/` | 8 bundled reference voice clips for the demo. |

## Three deployment branches (one router)

| Input class | Branch | Uses |
|---|---|---|
| Telugu (pure script) | R6 LoRA on Chatterbox | Chatterbox base + this LoRA + BUPS romanisation |
| Tamil (pure script) | R6 LoRA on Chatterbox | same |
| Hindi (pure script) | Vanilla Chatterbox | Chatterbox base, voice-prompt + Config B sampling |
| Te / Ta / Hi (code-mix) | Transliterate → IndicF5 | Haiku preprocessor + IndicF5 zero-shot |
| English | Vanilla Chatterbox | base, no LoRA |

The router is one regex (`is_codemix(text)`: ≥1 Latin word ≥2 chars in a non-English target) plus a per-language dispatch table.

## Quick start

```bash
pip install -e .
```

The router usage is described in `serving/praxy_router.py`'s docstring. The simplest path is the live HF Space — clone any voice from a 8–15 s clip, mix codemix freely.

For a local Modal-backed deploy, see the inline comment block at the top of `serving/gradio_app.py`.

## Headline results (from the paper)

On the [PSP](https://github.com/praxelhq/psp-eval) benchmark + LLM-WER intelligibility:

| Lang | Best system | Praxy v1 |
|---|---|---|
| Te (pure) retroflex collapse | Sarvam Bulbul 33.3% | **Praxy R6: 26.7%** ✓ |
| Ta (pure) Tamil-zha collapse | commercial trio: 86% | **Praxy R6: 71%** ✓ |
| Hi (pure) LLM-WER | Sarvam: 0.007 / 11labs: 0.006 | Praxy: 0.025 (tied with Cartesia) |
| Hi codemix LLM-WER | Cartesia: 0.000* | Praxy translit→IndicF5: 0.198 |
| Te codemix LLM-WER | Cartesia: 0.106 | Praxy translit→IndicF5: 0.142 |

*Cartesia's 0.000 reflects American-English pronunciation of embedded English words, which Whisper transcribes near-perfectly. Our Indianised pronunciation (the way native Indian speakers actually code-switch) trades literal STT-recoverability for native-listener naturalness; v2 of the PSP benchmark will add a code-mix dimension that disentangles the two.

## Citation

```bibtex
@misc{praxy2026,
  title={Praxy Voice: Voice-Prompt Recovery + BUPS for Commercial-Class Indic TTS from a Frozen Non-Indic Base at Zero Commercial-Training-Data Cost},
  author={Menta, Venkata Pushpak Teja},
  year={2026},
  eprint={2604.25441},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2604.25441}
}
```

## Licence

- Code in this repo: **MIT**
- R6 LoRA weights (`hf_release/praxy-voice-r6/`): **Apache-2.0**
- The Chatterbox base model: MIT (separate, see `huggingface.co/ResembleAI/chatterbox`)
- IndicF5 base model: see `huggingface.co/ai4bharat/IndicF5`

## Contact

Pushpak Teja Menta · Praxel Ventures · pushpak@praxel.in

---
license: apache-2.0
base_model: ResembleAI/chatterbox
tags:
  - text-to-speech
  - tts
  - indic
  - telugu
  - tamil
  - hindi
  - lora
  - peft
language:
  - te
  - ta
  - hi
library_name: peft
pipeline_tag: text-to-speech
---

# Praxy Voice R6 — LoRA for Indic TTS on Chatterbox

**Praxy Voice R6** is a LoRA adapter that extends [ResembleAI Chatterbox Multilingual](https://huggingface.co/ResembleAI/chatterbox) to high-quality Telugu and Tamil text-to-speech, two languages the Chatterbox base does not natively cover.

This adapter is part of a larger recipe described in the accompanying paper *"Praxy Voice: Voice-Prompt Recovery + BUPS for Commercial-Class Indic TTS from a Frozen Non-Indic Base"* (arXiv 2026). The recipe is:

1. **BUPS** — ISO-15919 romanisation of Indic text before tokenisation
2. **This LoRA adapter** — rank-32 attention-only adapter on Chatterbox's `t3` transformer (4.7M trainable params)
3. **Voice-prompt recovery** — an 8–11 s reference audio clip in the target language + three sampling overrides (exaggeration 0.7, temperature 0.6, min_p 0.1) at inference

For **Hindi**, this LoRA actively regresses semantic accuracy — Hindi should be synthesised with the *vanilla* Chatterbox base, still applying the voice-prompt + Config B recipe. See paper §5.3 for the rationale.

## What this adapter does NOT cover

Praxy v1 ships **three deployment branches**; this LoRA is only one of them.

| Input class | Branch | Where it lives |
|---|---|---|
| Telugu (pure script) | **R6 LoRA + Chatterbox** | This repo ✓ |
| Tamil (pure script) | **R6 LoRA + Chatterbox** | This repo ✓ |
| Hindi (pure script) | Vanilla Chatterbox + voice prompt | [`ResembleAI/chatterbox`](https://huggingface.co/ResembleAI/chatterbox) — no adapter needed |
| Hi/Te/Ta with English code-mix | `transliterate → IndicF5` | Recipe in [`praxelhq/praxy`](https://github.com/praxelhq/praxy), see paper §III.E |

**Code-mixed text** (e.g. *"मैंने WhatsApp पे message किया but notification नहीं आया।"*) is **not** in scope for this LoRA — using it on code-mix will degrade quality (the LoRA romanises English chunks into Indic phonetics). Code-mix synthesis routes through [AI4Bharat IndicF5](https://huggingface.co/ai4bharat/IndicF5) with a native-script transliteration preprocessor; the routing is one regex (≥1 Latin word ≥2 chars) and is implemented in [`serving/praxy_router.py`](https://github.com/praxelhq/praxy/blob/main/serving/praxy_router.py).

🌐 **Try the full system** (all three branches behind one UI, with voice cloning): [Praxel/praxy-voice-demo Space](https://huggingface.co/spaces/Praxel/praxy-voice-demo).

---

## Quick start

### Install

```bash
pip install chatterbox-tts peft indic-transliteration indic-num2words
```

### Telugu or Tamil (via this LoRA + BUPS)

```python
import torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from peft import LoraConfig, get_peft_model
from huggingface_hub import hf_hub_download

# Load base model (frozen)
model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
for p in model.t3.parameters(): p.requires_grad_(False)

# Wrap t3 with the same LoRA shape used at training
lora_cfg = LoraConfig(
    r=32, lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05, bias="none",
)
model.t3 = get_peft_model(model.t3, lora_cfg)

# Load Praxy R6 weights
ckpt_path = hf_hub_download("Praxel/praxy-voice-r6", "lora_state.pt")
sd = torch.load(ckpt_path, map_location="cuda")
model.t3.load_state_dict(sd, strict=False, assign=True)
model.t3.eval()

# BUPS romanisation (required for Te/Ta)
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

def bups(text, script):
    # script in {'devanagari', 'telugu', 'tamil', ...}
    script_map = {
        'devanagari': sanscript.DEVANAGARI,
        'telugu': sanscript.TELUGU,
        'tamil': sanscript.TAMIL,
    }
    return transliterate(text, script_map[script], sanscript.ISO)

text = "నేను ఇవాళ బాగున్నాను"      # "I am well today" in Telugu
text_roman = bups(text, "telugu")   # "nēnu ivāḷa bāgunnānu"

# Inference with voice-prompt recovery + Config B
wav = model.generate(
    text_roman,
    language_id="hi",                 # Hi-proxy (Te/Ta aren't in Chatterbox's 23-lang roster)
    audio_prompt_path="path/to/your_te_voice_9s.wav",  # BYOR: 8-11s same-language clip
    exaggeration=0.7,
    temperature=0.6,
    min_p=0.1,
)
# wav is a torch.Tensor; save via torchaudio or soundfile
```

### Hindi (via **vanilla** Chatterbox — NOT this LoRA)

```python
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
wav = model.generate(
    "मेरे पास एक सपना है",             # "I have a dream" in Hindi Devanagari
    language_id="hi",
    audio_prompt_path="path/to/your_hi_voice_6s.wav",
    exaggeration=0.7,
    temperature=0.6,
    min_p=0.1,
)
```

### Numbers, currency, percent (highly recommended)

Chatterbox's tokeniser fragments digit runs. Use the unified Indic normaliser before synthesis:

```bash
pip install indic-num2words
```

```python
from num_to_words import num_to_word
import re

def normalise_indic(text, lang):
    def _sub(m):
        n = int(m.group(0))
        return num_to_word(n, lang).replace(",", "").strip()
    return re.sub(r"\d+", _sub, text)

# "జనవరి 26, 2026న" → "జనవరి ఇరవై ఆరు రెండు వేల ఇరవై ఆరున"
```

(For a richer normaliser with currency / percent / date-ordinal handling, see `praxy/linguistics/indic_numbers.py` in the code repo.)

---

## Benchmarks

Evaluated on the companion **PSP** (Phoneme Substitution Profile) benchmark, 10-utterance pilot sets per language:

| Lang | Metric | Praxy R6 (this repo) | Sarvam Bulbul | ElevenLabs v3 | Cartesia Sonic-3 |
|---|---|---|---|---|---|
| Te | Retroflex collapse ↓ | **26.7%** | 33.3% | 40.0% | 50.0% |
| Te | PSD ↓ | 13.1 | 11.1 | 154.4 | 33.8 |
| Te | FAD ↓ | 291.3 | 250.4 | 328.9 | 458.1 |
| Te | LLM-WER ↓ | 0.033 | 0.029 | 0.041 | 0.029 |
| Ta | ZF (zha) collapse ↓ | **71%** | 86% | 86% | 86% |
| Ta | Retroflex collapse ↓ | 69.2% | 70.5% | 69.2% | 69.2% |
| Ta | FAD ↓ | 276.0 | 200.3 | 239.4 | 404.3 |
| Ta | LLM-WER ↓ | 0.041 | — | — | — |
| Hi | LLM-WER ↓ (vanilla, not this LoRA) | 0.025 | 0.007 | 0.006 | 0.025 |
| Hi | Intent ↑ | 1.00 | — | — | 0.90 |

**Highlight:** On Telugu, Praxy's retroflex collapse rate is the lowest of every system measured — better than Sarvam, which was previously the best Telugu TTS.

See the paper (§5) for the full table including Indic Parler-TTS, native-audio noise floors, and all four ablations.

---

## What this adapter does and does not do

**It does:**
- Extend Chatterbox Multilingual to Telugu and Tamil (languages it had zero native coverage for)
- Preserve Chatterbox's full zero-shot voice-cloning capability via the `audio_prompt_path` interface
- Work alongside the unified Indic text normaliser (numbers, currency, dates, %)

**It does not:**
- Improve Hindi — use vanilla Chatterbox for Hi
- Replace your need for a reference voice — the voice-prompt recovery recipe *requires* an 8–11 s same-language clip
- Train a new acoustic model — the acoustic decoder (`s3gen`) stays frozen

---

## Training details

- **Base:** [ResembleAI/chatterbox](https://huggingface.co/ResembleAI/chatterbox) (MIT)
- **Adapter target:** `t3` transformer, attention projections only (`q_proj`, `k_proj`, `v_proj`, `o_proj`)
- **Rank / alpha / dropout:** 32 / 64 / 0.05
- **Trainable params:** 7.86M (0.97% of base) — 240 tensors × (lora_A + lora_B) across 30 transformer layers × 4 attention projections
- **Training data:** 1,886 h of IndicTTS + Rasa + FLEURS + Shrutilipi (Te/Ta/Hi, CC-BY-4.0)
- **Optimiser / schedule:** AdamW (β=0.9, 0.95), cosine with 500-step warmup, peak LR 3e-6
- **Precision:** bf16 mixed
- **Compute:** 1× A100-80GB, 8000 steps, ~11 hours, ~$45

Training code: [github.com/praxelhq/praxy](https://github.com/praxelhq/praxy) (MIT).

---

## Limitations

- **10-utt pilots only** — 300-utt benchmarks planned for v2. Do not treat single-digit-percent differences as statistically separable.
- **Voice-prompt CUDA bug:** Sarvam-Hi-female reference audio triggers a CUDA assertion in Chatterbox's `s3gen` positional embedding. Workaround: use Cartesia-Hi-female or any other Hi reference.
- **No MOS** — formal subjective evaluation deferred to v2.
- **Acoustic-decoder unchanged** — the LoRA does not touch `s3gen`; acoustic quality beyond voice-prompt-recoverable is bounded by base Chatterbox's acoustic prior.

---

## License and attribution

- **This LoRA adapter:** Apache-2.0
- **Chatterbox base (not included in this repo):** MIT (© Resemble AI)
- **Training data:** CC-BY-4.0 / similarly permissive

If you use this model in published work, please cite the companion papers:

```bibtex
@unpublished{praxyvoice2026,
  title={Praxy Voice: Voice-Prompt Recovery + BUPS for Commercial-Class Indic TTS from a Frozen Non-Indic Base},
  author={Teja, Pushpak},
  note={arXiv preprint, 2026},
  year={2026}
}

@unpublished{psp2026,
  title={PSP: An Interpretable Per-Dimension Accent Benchmark for Indic TTS},
  author={Teja, Pushpak},
  note={arXiv preprint, 2026},
  year={2026}
}
```

## Contact

Issues, bugs, reference-voice quirks: [github.com/praxelhq/praxy/issues](https://github.com/praxelhq/praxy/issues).

Commercial licensing / enterprise support: pushpak@praxel.in.

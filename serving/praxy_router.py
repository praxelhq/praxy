"""Praxy production inference router (v2 — adds IndicF5 + codemix).

Single entry point that picks the right inference branch per language and
codemix-detection. See `memory/project_indicf5_unblock_recipe_2026-04-27.md`
and the v1-zero-shot scorecards in `evaluation/scorecards/indicf5_v1_*` for
the empirical data motivating the routes below.

Routing matrix (2026-04-27 v1):

| Language     | Pure-script branch              | Codemix branch                  |
|--------------|---------------------------------|---------------------------------|
| Telugu (te)  | R6 LoRA (WER 0.034)             | translit → IndicF5 (WER 0.14)   |
| Hindi (hi)   | vanilla Chatterbox + ref (0.025)| translit → IndicF5 (WER 0.20)   |
| Tamil (ta)   | R6 LoRA (WER 0.041)             | translit → IndicF5 (WER 0.27)   |
| English (en) | Chatterbox vanilla              | n/a                             |
| Other Indic  | IndicF5 best-effort             | translit → IndicF5 best-effort  |

`is_codemix(text)` returns True when ≥1 word is pure Latin (≥2 alphabetic
chars). Single Latin chars or digits are not enough to trigger.

The `translit → IndicF5` branch calls Haiku 4.5 to convert Latin English
words into native-script phonetic spelling matching how Bollywood/Sarvam
training data writes them ("WhatsApp" → "व्हाट्सऐप"), then sends the
all-native-script string to IndicF5. This single fix dropped Hi codemix
WER from 0.85 → 0.20 (76% relative drop) and Te codemix from 0.80 → 0.14
(82% relative drop). See `serving/codemix_to_native_script.py`.
"""

from __future__ import annotations

import re
from pathlib import Path

# Default reference audio per language. These are commercial-TTS-sourced
# clips reused for the v1 demo; Pushpak will swap in Praxel-owned voices
# (his + Ashwin's) before public launch.
_DEFAULT_REFS: dict[str, str] = {
    "te": "data/references/sarvam_te_female_9s.wav",
    "ta": "data/references/sarvam_ta_male_11s.wav",
    "hi": "data/references/sarvam_hi_female_10s.wav",
    "en": "data/references/ashwin_10s.wav",  # Chatterbox base needs ref too
}

# Per-ref transcript cache (cleaned, no Whisper special tokens). IndicF5
# requires ref audio + EXACT matching ref text.
_DEFAULT_REF_TEXTS: dict[str, str] = {
    "te": "మా తాతయ్య ప్రతి సాయంత్రం వరండాలో కూర్చుని తన చిన్నతనంలో జరిగిన కథలు చెబుతూ ఉంటారు మరియు మేము అందరూ కలిసి ఆసక్తిగా వినేవాళ్లం",
    "ta": "எங்கள் தாத்தா தினமும் மாலையில் திண்ணையில் அமர்ந்து கொண்டு தன் சிறுவயதில் நடந்த கதைகளைச் சொல்லிக் கொண்டிருப்பார் நாங்கள் அனைவரும் சேர்ந்து ஆர்வமாகக் கேட்போம்",
    "hi": "मेरे दादा जी हर शाम बरामदे में बैठकर अपने बचपन की कहानियां सुनाते हैं और हम सब मिलकर बडे चाव से उनकी बातें सुनते हैं",
}

# Per-language pure-script branch. "lora" = R6 LoRA (Chatterbox); "indicf5"
# = ai4bharat/IndicF5 zero-shot; "vanilla" = base Chatterbox no-LoRA.
_BRANCH: dict[str, str] = {
    "te": "lora",       # R6 wins pure-Te (WER 0.034 < 0.08 IndicF5)
    "ta": "lora",       # R6 wins pure-Ta (paper §V.1 0.041 LLM-WER)
    "hi": "vanilla",    # vanilla Chatterbox + Cart-Hi ref + Config B = 0.025
                        # WER (paper §V.1 ties Cartesia); IndicF5 zero-shot
                        # is 0.13 — vanilla Chatterbox recipe wins on Hi.
    "en": "vanilla",    # Chatterbox base is excellent at English
}

# Codemix always routes to IndicF5 with native-script preprocessing —
# IndicF5's char-level tokenizer + the translit fix gives the only
# working codemix recipe across our 8 architectural attempts.
_CODEMIX_BRANCH = "indicf5_native"

# Config B sampling overrides (Chatterbox-only — IndicF5 doesn't expose
# these knobs). From TTS paper §5.2 sweep.
CONFIG_B = dict(
    exaggeration=0.7,
    temperature=0.6,
    min_p=0.1,
    cfg_weight=0.5,
    repetition_penalty=2.0,
    top_p=1.0,
)

DEFAULT_R6_CKPT = "/cache/chatterbox_indic/round_6/step_8000.ckpt"

# A "codemix" word is a run of ≥2 Latin alphabetic chars. Single letters
# (acronym components) and digits don't trigger; numbers are normalised
# elsewhere via Indic number expansion.
_CODEMIX_WORD_RE = re.compile(r"[A-Za-z]{2,}")


def is_codemix(text: str) -> bool:
    """Returns True iff `text` contains at least one Latin word ≥2 chars
    that needs transliteration before IndicF5 synthesis."""
    return bool(_CODEMIX_WORD_RE.search(text))


def _resolve_ref_audio(ref_audio_path: str | None, lang: str) -> str:
    if ref_audio_path:
        return ref_audio_path
    fallback = _DEFAULT_REFS.get(lang)
    if not fallback:
        raise ValueError(f"No default reference voice for lang={lang!r}; supply ref_audio_path.")
    return fallback


def _resolve_ref_text(ref_text: str | None, lang: str) -> str:
    if ref_text:
        return ref_text
    fallback = _DEFAULT_REF_TEXTS.get(lang, "")
    return fallback


def route(text: str, lang: str, ref_audio_path: str | None = None,
          ref_text: str | None = None) -> dict:
    """Return the inference-branch parameters for a given input.

    Picks the route by combining (lang, codemix-detection). Returns a
    dict with the branch identifier and the kwargs to forward to the
    matching Modal entrypoint.
    """
    lang = lang.lower()
    # Codemix routing applies only to Indic targets; English target with
    # English text is not codemix.
    cm = is_codemix(text) and lang != "en"
    pure_branch = _BRANCH.get(lang, "indicf5")
    branch = _CODEMIX_BRANCH if cm else pure_branch

    ref_audio = _resolve_ref_audio(ref_audio_path, lang)
    ref_t = _resolve_ref_text(ref_text, lang)

    if branch == "lora":
        return {
            "branch": "lora",
            "model": "chatterbox_r6_lora",
            "ckpt_path": DEFAULT_R6_CKPT,
            "use_bups": True,
            "no_lora": False,
            "ref_audio_path": ref_audio,
            "normalize_numbers": True,
            "language_code": lang,
            **CONFIG_B,
        }
    if branch == "vanilla":
        return {
            "branch": "vanilla",
            "model": "chatterbox_base",
            "ckpt_path": DEFAULT_R6_CKPT,  # unused with no_lora
            "use_bups": False,
            "no_lora": True,
            "ref_audio_path": ref_audio,
            "normalize_numbers": True,
            "language_code": lang,
            **CONFIG_B,
        }
    if branch == "indicf5":
        return {
            "branch": "indicf5",
            "model": "indicf5_zeroshot",
            "ref_audio_path": ref_audio,
            "ref_text": ref_t,
            "language_code": lang,
        }
    if branch == "indicf5_native":
        # Caller should run text through `serving.codemix_to_native_script
        # .transliterate_codemix(text, lang)` before invoking the synth.
        return {
            "branch": "indicf5_native",
            "model": "indicf5_native_codemix",
            "ref_audio_path": ref_audio,
            "ref_text": ref_t,
            "language_code": lang,
            "preprocess": "transliterate_codemix_to_native",
        }
    raise ValueError(f"Unknown branch {branch!r}")


def synthesize(text: str, lang: str, ref_audio_path: str | None = None,
               ref_text: str | None = None) -> tuple[bytes, int]:
    """Production-grade single-utterance synthesis. Wraps `route()` and
    runs the chosen Modal entrypoint. Returns (wav_bytes, sample_rate).

    Note: this is a thin convenience wrapper. For batch eval/training
    use the underlying Modal entrypoints directly to avoid per-call
    Modal overhead.
    """
    plan = route(text, lang, ref_audio_path=ref_audio_path, ref_text=ref_text)
    branch = plan["branch"]

    if branch == "indicf5_native":
        from serving.codemix_to_native_script import transliterate_codemix
        text = transliterate_codemix(text, lang)

    if branch in ("indicf5", "indicf5_native"):
        from serving.modal_app import IndicF5TTS
        synth = IndicF5TTS()
        from pathlib import Path as _P
        ref_bytes = _P(plan["ref_audio_path"]).read_bytes()
        return synth.synthesize.remote(
            text=text,
            ref_audio_bytes=ref_bytes,
            ref_text=plan["ref_text"],
        )

    # lora / vanilla — Chatterbox path (existing implementation).
    from serving.modal_app import PraxyChatterboxLoRA
    import modal as _modal
    env = {
        "PRAXY_CKPT_PATH": plan["ckpt_path"],
        "PRAXY_USE_BUPS": "1" if plan["use_bups"] else "0",
        "PRAXY_NO_LORA": "1" if plan["no_lora"] else "0",
    }
    synth = PraxyChatterboxLoRA.with_options(secrets=[_modal.Secret.from_dict(env)])()
    from pathlib import Path as _P
    ref_bytes = _P(plan["ref_audio_path"]).read_bytes() if plan["ref_audio_path"] else None
    if plan.get("normalize_numbers") and lang in {"te", "ta", "hi", "bn", "gu", "mr", "kn", "ml"}:
        from praxy.linguistics.indic_numbers import normalize_indic_text
        text = normalize_indic_text(text, lang)
    return synth.synthesize.remote(
        text=text,
        language_code=lang,
        ref_audio_bytes=ref_bytes,
        exaggeration=CONFIG_B["exaggeration"],
        cfg_weight=CONFIG_B["cfg_weight"],
        temperature=CONFIG_B["temperature"],
        repetition_penalty=CONFIG_B["repetition_penalty"],
        min_p=CONFIG_B["min_p"],
        top_p=CONFIG_B["top_p"],
    )


__all__ = ["route", "synthesize", "is_codemix", "CONFIG_B", "DEFAULT_R6_CKPT"]

"""Transliterate Latin-script English words inside an Indic codemix sentence
to their native-script phonetic spelling, so a char-level Indic TTS (IndicF5)
can pronounce them.

IndicF5 zero-shot **silently drops** Latin-script words because its training
distribution had only native-script chars producing audio. The fix isn't
fine-tuning — it's matching the input distribution: native speakers and
publishers (Bollywood news, Sarvam Bulbul's training data, Indian TLM-style
podcasts) consistently spell English brand/tech words in native script.

Examples (Hindi):
    "WhatsApp"     → "व्हाट्सऐप"
    "message"      → "मैसेज"
    "notification" → "नोटिफिकेशन"
    "all-hands meeting" → "ऑल-हैंड्स मीटिंग"

Examples (Telugu):
    "WhatsApp"     → "వాట్సాప్"
    "message"      → "మెసేజ్"
    "notification" → "నోటిఫికేషన్"

Approach: send the codemix sentence to Claude Haiku 4.5 with a tight system
prompt that forces it to (a) keep all native-script chars unchanged, (b)
phonetically transliterate every Latin-script word into the target Indic
script as Indians actually pronounce it, (c) preserve word order, spacing,
and punctuation. Cache results so we don't re-spend on identical inputs.

Output is a JSONL with `{"original": ..., "transliterated": ...}` per row,
written next to the source corpus.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = REPO_ROOT / "data" / "transliteration_cache.json"

LANG_NATIVE_NAMES = {
    "hi": "Hindi (Devanagari script)",
    "te": "Telugu",
    "ta": "Tamil",
}

SYSTEM = """You convert Indian code-mix sentences into pure native-script form.

Input: a sentence mixing {lang_name} (in {lang_name} script) with Latin-script English words/phrases.

Output rules:
1. Keep every {lang_name}-script word/character unchanged. Do NOT translate them.
2. For every Latin-script English word/phrase, write its phonetic spelling in {lang_name} script — exactly the way an educated native {lang_name} speaker would write it casually (the way Bollywood subtitles, Indian news tickers, and Sarvam-Bulbul's training data spells English brand and tech terms). Examples for Hindi: WhatsApp → व्हाट्सऐप, message → मैसेज, notification → नोटिफिकेशन, CEO → सीईओ, syllabus → सिलेबस, complete → कम्प्लीट, traffic jam → ट्रैफिक जैम, weekend → वीकेंड. For Telugu: WhatsApp → వాట్సాప్, message → మెసేజ్, notification → నోటిఫికేషన్, CEO → సీఈఓ.
3. Preserve all word order, spacing, and punctuation exactly as in the input.
4. Do NOT add explanations, brackets, alternatives, or commentary. Output ONLY the converted sentence.

Examples:
Input (Hindi codemix): मैंने WhatsApp पे message किया but notification नहीं आया।
Output: मैंने व्हाट्सऐप पे मैसेज किया बट नोटिफिकेशन नहीं आया।

Input (Telugu codemix): మా CEO ఇవాళ all-hands meeting లో కొత్త quarterly targets announce చేశారు.
Output: మా సీఈఓ ఇవాళ ఆల్-హ్యాండ్స్ మీటింగ్ లో కొత్త క్వార్టర్లీ టార్గెట్స్ అనౌన్స్ చేశారు."""


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text())
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2))


def transliterate_codemix(text: str, lang: str, cache: dict | None = None) -> str:
    """Convert codemix sentence → pure native-script form via Haiku.

    If text has no Latin chars, returns it unchanged (fast path).
    """
    if not re.search(r"[A-Za-z]", text):
        return text

    if cache is None:
        cache = _load_cache()
    cache_key = f"{lang}::{text}"
    if cache_key in cache:
        return cache[cache_key]

    from evaluation.anthropic_client import chat_complete, extract_content

    lang_name = LANG_NATIVE_NAMES.get(lang, lang)
    sys_prompt = SYSTEM.format(lang_name=lang_name)

    resp = chat_complete(
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": text},
        ],
        model="claude-haiku-4-5",
        temperature=0.0,
        max_tokens=512,
    )
    out = extract_content(resp).strip()
    # Defensive: strip any trailing commentary blocks like "Output: " prefix.
    out = re.sub(r"^(Output|Translation|Result):\s*", "", out, flags=re.IGNORECASE).strip()

    cache[cache_key] = out
    _save_cache(cache)
    return out


def transliterate_test_set(test_set_name: str, lang: str) -> Path:
    """Run transliteration on every utterance in a golden test set, write a
    sibling test set with `_native` suffix that has the converted text."""
    src = REPO_ROOT / "evaluation" / "golden_test_sets" / f"{test_set_name}.json"
    data = json.loads(src.read_text())
    cache = _load_cache()
    for u in data["utterances"]:
        original = u["text"]
        converted = transliterate_codemix(original, lang, cache=cache)
        u["text_original_codemix"] = original
        u["text"] = converted
    data["name"] = f"{test_set_name}_native"
    data["description"] = (
        f"{data.get('description', '')} -- Latin-script English words "
        f"transliterated to {LANG_NATIVE_NAMES.get(lang, lang)} script."
    )
    out = REPO_ROOT / "evaluation" / "golden_test_sets" / f"{test_set_name}_native.json"
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(f"[translit] wrote {out} ({len(data['utterances'])} utterances)")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--test-set", required=True)
    p.add_argument("--lang", required=True, choices=["hi", "te", "ta"])
    args = p.parse_args()
    t0 = time.time()
    transliterate_test_set(args.test_set, args.lang)
    print(f"[translit] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

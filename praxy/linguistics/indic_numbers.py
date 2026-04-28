"""Unified Indic text normaliser — numbers, currency, percent, dates.

Chatterbox's tokenizer fragments digit strings into junk tokens for Indic
synthesis. The Telugu smoke_6 case ("జనవరి 26, 2026న" → garbage) demonstrated
the failure pattern; rewriting to Indic words (``ఇరవై ఆరు, రెండు వేల ఇరవై ఆరు``)
eliminates it.

This module provides a single entry point ``normalize_indic_text(text, lang)``
that:

- Expands digit runs to language-appropriate words via ``num_to_words`` from
  the ``indic-num2words`` package (covers hi/te/ta/bn/gu/mr/kn/...).
- Maps common currency symbols (``₹``, ``$``, ``€``) to spelled words in the
  target language.
- Maps ``%`` to the target-language word for "percent".
- For day-of-month contexts (digit preceded by a Te/Hi/Ta month word within the
  last ~40 chars), renders the number as an ordinal-style form.

Called local-side in ``serving/modal_app.py::run_praxy_checkpoint`` before
text is sent to Modal, so the container doesn't need this dependency.
"""

from __future__ import annotations

import re

try:
    from num_to_words import num_to_word  # type: ignore
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "indic-num2words is required for normalize_indic_text. "
        "Install with `uv pip install indic-num2words`."
    ) from e


SUPPORTED_LANGS = {"hi", "te", "ta", "bn", "gu", "mr", "kn", "ml"}


_CURRENCY_WORD = {
    "te": {"₹": "రూపాయలు", "$": "డాలర్లు", "€": "యూరోలు"},
    "hi": {"₹": "रुपये", "$": "डॉलर", "€": "यूरो"},
    "ta": {"₹": "ரூபாய்", "$": "டாலர்", "€": "யூரோ"},
    "bn": {"₹": "টাকা", "$": "ডলার", "€": "ইউরো"},
    "kn": {"₹": "ರೂಪಾಯಿ", "$": "ಡಾಲರ್", "€": "ಯೂರೋ"},
    "mr": {"₹": "रुपये", "$": "डॉलर", "€": "युरो"},
    "gu": {"₹": "રૂપિયા", "$": "ડોલર", "€": "યુરો"},
    "ml": {"₹": "രൂപ", "$": "ഡോളർ", "€": "യൂറോ"},
}

_PERCENT_WORD = {
    "te": "శాతం",
    "hi": "प्रतिशत",
    "ta": "சதவீதம்",
    "bn": "শতাংশ",
    "kn": "ಶೇಕಡಾ",
    "mr": "टक्के",
    "gu": "ટકા",
    "ml": "ശതമാനം",
}

# Month-word stems in each language (match any of these to treat a 1-31 digit
# run as a day-of-month ordinal).
_MONTH_STEMS = {
    "te": ("జనవరి", "ఫిబ్రవరి", "మార్చి", "ఏప్రిల్", "మే", "జూన్", "జూలై",
           "ఆగస్టు", "సెప్టెంబర్", "అక్టోబర్", "నవంబర్", "డిసెంబర్"),
    "hi": ("जनवरी", "फ़रवरी", "फरवरी", "मार्च", "अप्रैल", "मई", "जून", "जुलाई",
           "अगस्त", "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर"),
    "ta": ("ஜனவரி", "பிப்ரவரி", "மார்ச்", "ஏப்ரல்", "மே", "ஜூன்", "ஜூலை",
           "ஆகஸ்ட்", "செப்டம்பர்", "அக்டோபர்", "நவம்பர்", "டிசம்பர்"),
}

# Ordinal suffix pattern per language. For day-of-month we append a small
# suffix to the cardinal. These are heuristics — a full morphological library
# would inflect more carefully, but they're good enough for smoke/golden set
# evaluation and for practical production.
_DAY_ORDINAL_SUFFIX = {
    "te": "వ తేదీ",     # "<cardinal> va tedi"
    "hi": "तारीख",       # append ", <cardinal> tarikh"
    "ta": "ஆம் நாள்",     # "<cardinal> aam naal"
}


_DIGIT_RUN = re.compile(r"\d+")


def _cardinal(n: int, lang: str) -> str:
    """Indic cardinal, cleaned of the stray comma `num_to_words` injects
    between the thousands and units in some languages."""
    word = num_to_word(n, lang)
    return word.replace(",", "").strip()


def _as_ordinal_day(n: int, lang: str) -> str:
    base = _cardinal(n, lang)
    suffix = _DAY_ORDINAL_SUFFIX.get(lang)
    if suffix is None:
        return base
    return f"{base} {suffix}"


def normalize_indic_text(
    text: str,
    lang: str,
    *,
    ordinal_day_of_month: bool = True,
) -> str:
    """Normalise digit runs, currency symbols, and percent to Indic words.

    Args:
        text: input text (may contain digits, ₹/$/€, %).
        lang: ISO-639-1 code: te, hi, ta, bn, gu, mr, kn, ml.
        ordinal_day_of_month: if True, a digit run of 1-31 preceded by a
            month word within the last 40 chars renders as an ordinal form
            ("ఇరవై ఆరువ తేదీ") rather than a bare cardinal ("ఇరవై ఆరు").

    Returns:
        The normalised text. If ``lang`` is unsupported the input is returned
        unchanged.
    """
    if lang not in SUPPORTED_LANGS:
        return text

    percent_word = _PERCENT_WORD.get(lang, "")
    currency_map = _CURRENCY_WORD.get(lang, {})
    month_stems = _MONTH_STEMS.get(lang, ())

    def _day_of_month_context(idx: int) -> bool:
        """Return True if any month stem appears in the 40 chars before idx."""
        if not month_stems:
            return False
        window = text[max(0, idx - 40): idx]
        return any(m in window for m in month_stems)

    def _num_sub(match: "re.Match[str]") -> str:
        raw = match.group(0)
        n = int(raw)
        if n < 0:
            return raw
        if n >= 10**10:
            return raw  # stay safe on very large
        word = _cardinal(n, lang)
        if ordinal_day_of_month and 1 <= n <= 31 and _day_of_month_context(match.start()):
            word = _as_ordinal_day(n, lang)
        # Trailing punctuation (",", ".", etc.) stays outside our capture.
        return word

    # 1) ₹500 / $10 / €20 → "five hundred rupees / ten dollars / twenty euros"
    if currency_map:
        for symbol, word in currency_map.items():
            if symbol in text:
                # "₹500" → "500 rupees"; we let the digit pass to the number
                # rewriter below, which then expands to Indic words.
                text = re.sub(
                    rf"{re.escape(symbol)}\s*(\d[\d,]*)",
                    rf"\1 {word}",
                    text,
                )

    # 2) "50%" → "fifty percent" (language-appropriate)
    if percent_word:
        text = re.sub(r"(\d[\d.]*)\s*%", rf"\1 {percent_word}", text)

    # 3) Digit runs → Indic words
    text = _DIGIT_RUN.sub(_num_sub, text)

    return text


# Back-compat shim: the Te-specific normaliser stays accessible under its
# original name so existing callers don't break.
def normalize_te_digits(text: str, *, ordinal_day_of_month: bool = True) -> str:
    """Telugu-specific shim; prefer :func:`normalize_indic_text` for new code."""
    return normalize_indic_text(text, "te", ordinal_day_of_month=ordinal_day_of_month)

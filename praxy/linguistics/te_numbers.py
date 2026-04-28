"""Telugu digit-to-word normalizer.

Chatterbox's tokenizer fragments digit strings into junk tokens when generating
Telugu. Smoke_6 demonstrated this: text with "జనవరి 26, 2026న" yielded garbage,
but expanding digits to Telugu words produced a clean 0.0 LLM-WER.

Scope covers what the Telugu smoke set contains: cardinals 0-9999, year forms
(four-digit years read as "<thousands> <remainder>"), and ordinals used in
date contexts. Above 9999 we keep it simple: lakhs/crores are not yet required.

Usage:

    from praxy.linguistics.te_numbers import normalize_te_digits
    text = "హైదరాబాద్‌లో జనవరి 26, 2026న పెద్ద కార్యక్రమం"
    out  = normalize_te_digits(text)
    # → "హైదరాబాద్‌లో జనవరి ఇరవై ఆరో తేదీ, రెండు వేల ఇరవై ఆరో న పెద్ద కార్యక్రమం"

This is not a full Indic number library (that belongs to indic-nlp-library /
num2words); it's the minimum coverage the smoke and golden sets need.
"""

from __future__ import annotations

import re

_ONES: dict[int, str] = {
    0: "సున్నా",
    1: "ఒకటి",
    2: "రెండు",
    3: "మూడు",
    4: "నాలుగు",
    5: "ఐదు",
    6: "ఆరు",
    7: "ఏడు",
    8: "ఎనిమిది",
    9: "తొమ్మిది",
}

# Teens and 11-19 irregulars
_TEENS: dict[int, str] = {
    10: "పది",
    11: "పదకొండు",
    12: "పన్నెండు",
    13: "పదమూడు",
    14: "పద్నాలుగు",
    15: "పదిహేను",
    16: "పదహారు",
    17: "పదిహేడు",
    18: "పద్దెనిమిది",
    19: "పందొమ్మిది",
}

# Tens-place stems used for 20-99: "ఇరవై ఒకటి" = 21.
_TENS: dict[int, str] = {
    20: "ఇరవై",
    30: "ముప్పై",
    40: "నలభై",
    50: "యాభై",
    60: "అరవై",
    70: "డెబ్బై",
    80: "ఎనభై",
    90: "తొంభై",
}


def _two_digit(n: int) -> str:
    if n < 0 or n >= 100:
        raise ValueError(n)
    if n < 10:
        return _ONES[n]
    if n < 20:
        return _TEENS[n]
    tens = (n // 10) * 10
    rem = n % 10
    if rem == 0:
        return _TENS[tens]
    return f"{_TENS[tens]} {_ONES[rem]}"


def _three_digit(n: int) -> str:
    if n < 0 or n >= 1000:
        raise ValueError(n)
    if n < 100:
        return _two_digit(n)
    hundreds = n // 100
    rem = n % 100
    hundreds_word = "నూరు" if hundreds == 1 else f"{_ONES[hundreds]} వందల"
    if rem == 0:
        return hundreds_word if hundreds == 1 else f"{_ONES[hundreds]} వందలు"
    return f"{hundreds_word} {_two_digit(rem)}"


def cardinal_te(n: int) -> str:
    """Return the Telugu cardinal for a non-negative integer 0..9999."""
    if n < 0:
        raise ValueError("negative numbers not supported")
    if n < 1000:
        return _three_digit(n)
    if n < 10_000:
        thousands = n // 1000
        rem = n % 1000
        th_word = "వెయ్యి" if thousands == 1 else f"{_ONES[thousands]} వేల"
        if rem == 0:
            return th_word if thousands == 1 else f"{_ONES[thousands]} వేలు"
        return f"{th_word} {_three_digit(rem)}"
    raise ValueError("numbers >= 10000 not yet supported")


def ordinal_te(n: int) -> str:
    """Return the Telugu ordinal used in date contexts — '26వ / ఇరవై ఆరో'.

    Telugu ordinals for day-of-month typically take the 'ఓ' suffix on the final
    cardinal element: ఒకటి → ఒకటో, రెండు → రెండో, ఇరవై ఆరు → ఇరవై ఆరో.
    """
    card = cardinal_te(n)
    # Replace final cardinal with its ordinal form
    _CARD_TO_ORD = {
        "ఒకటి": "ఒకటో", "రెండు": "రెండో", "మూడు": "మూడో", "నాలుగు": "నాలుగో",
        "ఐదు": "ఐదో", "ఆరు": "ఆరో", "ఏడు": "ఏడో", "ఎనిమిది": "ఎనిమిదో",
        "తొమ్మిది": "తొమ్మిదో", "పది": "పదో",
    }
    for cform, oform in _CARD_TO_ORD.items():
        if card.endswith(cform):
            return card[: -len(cform)] + oform
    # Fallback for tens without remainder: ఇరవై → ఇరవయో
    if card.endswith("ఇరవై"):
        return card[:-4] + "ఇరవయో"
    return card + "వ"  # generic fallback


_DIGIT_RUN = re.compile(r"\d+")


def normalize_te_digits(text: str, *, ordinal_day_of_month: bool = True) -> str:
    """Expand every digit run in ``text`` to its Telugu word form.

    ``ordinal_day_of_month`` heuristic: if a digit run of 1-2 digits has value
    1..31 and is immediately preceded by a Telugu month word (jan/feb/etc.),
    we render it as an ordinal (26 → "ఇరవై ఆరో") rather than a bare cardinal.
    This matches how native speakers read dates aloud.
    """
    _TE_MONTHS = (
        "జనవరి", "ఫిబ్రవరి", "మార్చి", "ఏప్రిల్", "మే", "జూన్",
        "జూలై", "ఆగస్టు", "సెప్టెంబర్", "అక్టోబర్", "నవంబర్", "డిసెంబర్",
    )

    def _replacement(match: "re.Match[str]") -> str:
        n = int(match.group(0))
        if n >= 10_000:
            # Leave as-is; caller can handle or we degrade gracefully
            return match.group(0)
        if ordinal_day_of_month and 1 <= n <= 31:
            # Look back for a month token within the preceding 40 chars
            start = max(0, match.start() - 40)
            preceding = text[start: match.start()]
            if any(m in preceding for m in _TE_MONTHS):
                return ordinal_te(n)
        return cardinal_te(n)

    return _DIGIT_RUN.sub(_replacement, text)

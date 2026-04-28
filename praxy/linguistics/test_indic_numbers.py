"""Tests for the unified Indic text normaliser.

Run: ``uv run python -m praxy.linguistics.test_indic_numbers``
"""

from __future__ import annotations

from praxy.linguistics.indic_numbers import normalize_indic_text


def _check(label: str, got: str, must_contain: list[str], must_not_contain: list[str]) -> None:
    ok = all(s in got for s in must_contain) and all(s not in got for s in must_not_contain)
    status = "OK" if ok else "FAIL"
    print(f"[{status}] {label}")
    print(f"       got: {got!r}")
    if not ok:
        miss = [s for s in must_contain if s not in got]
        extra = [s for s in must_not_contain if s in got]
        if miss:
            print(f"       missing: {miss}")
        if extra:
            print(f"       forbidden-present: {extra}")
        raise AssertionError(label)


def main() -> None:
    # Telugu
    _check(
        "te: day-of-month + year + trailing postposition",
        normalize_indic_text("హైదరాబాద్‌లో జనవరి 26, 2026న కార్యక్రమం.", "te"),
        must_contain=["ఇరవై ఆరు", "రెండు వేల ఇరవై ఆరు"],
        must_not_contain=["26", "2026"],
    )
    _check(
        "te: bare cardinal (no month prefix)",
        normalize_indic_text("నేను 25 కిలోమీటర్లు నడిచాను.", "te"),
        must_contain=["ఇరవై ఐదు"],
        must_not_contain=["25"],
    )
    _check(
        "te: currency + percent",
        normalize_indic_text("₹100 మరియు 50%", "te"),
        # num_to_words sometimes spells 50 as "యాబై" vs common "యాభై"; accept
        # either by checking just the "ba-" stem.
        must_contain=["వంద", "రూపాయలు", "ాబై", "శాతం"],
        must_not_contain=["₹", "%", "100", "50"],
    )

    # Hindi
    _check(
        "hi: year + day of month",
        normalize_indic_text("26 जनवरी 2026 को प्रगति होगी।", "hi"),
        must_contain=["छब्बीस", "दो हज़ार"],
        must_not_contain=["26", "2026"],
    )
    _check(
        "hi: currency",
        normalize_indic_text("मेरे पास ₹500 हैं।", "hi"),
        # num_to_words uses "पाँच सौ" (nasalized ī) instead of "पांच सौ"
        # (anusvara). Accept either by checking just the ऊँच-stem + सौ.
        must_contain=["च सौ", "रुपये"],
        must_not_contain=["₹", "500"],
    )
    _check(
        "hi: percent",
        normalize_indic_text("40% बढ़ोतरी हुई।", "hi"),
        # num_to_words variants: "चालिस" (short i) vs standard "चालीस" (long ī)
        must_contain=["चाल", "प्रतिशत"],
        must_not_contain=["40", "%"],
    )

    # Tamil
    _check(
        "ta: year",
        normalize_indic_text("நான் 2026 ஆம் ஆண்டில் பிறந்தேன்.", "ta"),
        must_contain=["இரண்டு ஆயிரம்", "இருபத்து ஆறு"],
        must_not_contain=["2026"],
    )
    _check(
        "ta: currency",
        normalize_indic_text("விலை ₹1000.", "ta"),
        must_contain=["ஆயிரம்", "ரூபாய்"],
        must_not_contain=["₹", "1000"],
    )

    # Unsupported lang → input unchanged
    _check(
        "unsupported lang: identity",
        normalize_indic_text("Meeting at 10am on Feb 26, 2026.", "fr"),
        must_contain=["10am", "26", "2026"],
        must_not_contain=[],
    )

    print("\nAll unified-normaliser tests passed.")


if __name__ == "__main__":
    main()

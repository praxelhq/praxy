"""Sanity tests for the Telugu number normalizer.

Runs as: ``uv run python -m praxy.linguistics.test_te_numbers``
"""

from __future__ import annotations

from praxy.linguistics.te_numbers import (
    cardinal_te,
    normalize_te_digits,
    ordinal_te,
)


def _check(got: str, expected: str, label: str) -> None:
    status = "OK" if got == expected else "FAIL"
    print(f"[{status}] {label}: got={got!r}  expected={expected!r}")
    if got != expected:
        raise AssertionError(f"{label}: {got!r} != {expected!r}")


def main() -> None:
    # Cardinals
    _check(cardinal_te(0), "సున్నా", "zero")
    _check(cardinal_te(1), "ఒకటి", "one")
    _check(cardinal_te(7), "ఏడు", "seven")
    _check(cardinal_te(10), "పది", "ten")
    _check(cardinal_te(15), "పదిహేను", "fifteen")
    _check(cardinal_te(20), "ఇరవై", "twenty")
    _check(cardinal_te(26), "ఇరవై ఆరు", "twenty-six")
    _check(cardinal_te(100), "నూరు", "one hundred")
    _check(cardinal_te(101), "నూరు ఒకటి", "one hundred one")
    _check(cardinal_te(234), "రెండు వందల ముప్పై నాలుగు", "234")
    _check(cardinal_te(1000), "వెయ్యి", "one thousand")
    _check(cardinal_te(2026), "రెండు వేల ఇరవై ఆరు", "year 2026")

    # Ordinals (day-of-month)
    _check(ordinal_te(1), "ఒకటో", "1st")
    _check(ordinal_te(26), "ఇరవై ఆరో", "26th")
    _check(ordinal_te(5), "ఐదో", "5th")

    # Sentence-level: the smoke_6 case
    src = "హైదరాబాద్‌లో జనవరి 26, 2026న పెద్ద కార్యక్రమం జరుగుతోంది."
    out = normalize_te_digits(src)
    # The 26 is day-of-month (month-prefixed) so renders ordinal; 2026 is a bare
    # cardinal with no month prefix (it sits after the comma)
    assert "ఇరవై ఆరో" in out, f"expected ordinal for 26 in: {out}"
    assert "రెండు వేల ఇరవై ఆరు" in out, f"expected cardinal for 2026 in: {out}"
    assert "26" not in out and "2026" not in out, f"digits should be gone: {out}"
    print(f"[OK] smoke_6 rewrite: {out}")

    # Number without month → cardinal
    src2 = "నేను 26 రూపాయలు ఇచ్చాను."
    out2 = normalize_te_digits(src2)
    assert "ఇరవై ఆరు" in out2, out2
    assert "ఇరవై ఆరో" not in out2, out2  # not ordinal — no month
    print(f"[OK] no-month cardinal: {out2}")

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()

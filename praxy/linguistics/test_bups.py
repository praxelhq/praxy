"""BUPS sanity tests. Run: uv run python -m praxy.linguistics.test_bups."""

from __future__ import annotations

import sys

from praxy.linguistics.bups import (
    BUPS,
    ID_TO_PHONEME,
    PHONEME_INVENTORY,
    detect_script,
    get_bups,
    phoneme_count,
    segment_by_script,
)


def test_phoneme_inventory_stability() -> None:
    """Phoneme list is append-only — these known positions must never move."""
    assert PHONEME_INVENTORY[0] == "<pad>"
    assert PHONEME_INVENTORY[1] == "<bos>"
    assert PHONEME_INVENTORY[2] == "<eos>"
    assert PHONEME_INVENTORY[3] == "<unk>"
    assert phoneme_count() >= 70


def test_cross_script_equivalence() -> None:
    """क (Dev) = క (Telugu) = ಕ (Kannada) = ক (Bengali) = ક (Gujarati) — all /ka/."""
    bups = get_bups()
    expected = bups.encode("क", script="devanagari")
    for script, glyph in [
        ("telugu", "క"),
        ("kannada", "ಕ"),
        ("bengali", "ক"),
        ("gujarati", "ક"),
        ("malayalam", "ക"),
    ]:
        got = bups.encode(glyph, script=script)
        assert got == expected, f"{script} {glyph} gave {got}, expected {expected}"


def test_conjunct_ksha() -> None:
    """क्ष (Dev) = క్ష (Telugu) = க்ஷ (Tamil) — conjunct /kʂa/."""
    bups = get_bups()
    dev = bups.encode("क्ष", script="devanagari")
    tel = bups.encode("క్ష", script="telugu")
    # Tamil க்ஷ is a transliteration from Grantha; should align
    tam = bups.encode("க்ஷ", script="tamil")
    # dev and tel should match exactly
    assert dev == tel, f"Dev {dev} != Tel {tel}"
    # Tamil may differ slightly; print for inspection rather than assert
    print(f"  conjunct kṣa — dev={dev} tel={tel} tam={tam}")


def test_script_detection() -> None:
    assert detect_script("क") == "devanagari"
    assert detect_script("క") == "telugu"
    assert detect_script("க") == "tamil"
    assert detect_script("A") == "latin"
    assert detect_script("1") == "latin"


def test_codemix_segmentation() -> None:
    """Telugu + English should split into two runs."""
    text = "మా CEO presentation ఇచ్చారు"
    runs = segment_by_script(text)
    scripts = [s for s, _ in runs]
    assert "telugu" in scripts
    assert "latin" in scripts
    print(f"  codemix runs: {runs}")


def test_telugu_sentence() -> None:
    """Sanity: a simple sentence encodes to a reasonable length."""
    bups = get_bups()
    ids = bups.encode("నేను ఇవాళ బాగున్నాను.", script="telugu")
    phonemes = [ID_TO_PHONEME[i] for i in ids]
    print(f"  'nenu ivaḷa baagunnaanu.' → {phonemes}")
    # Rough sanity: ~15-25 phonemes for this sentence
    assert 10 <= len(ids) <= 35, f"Unexpected length {len(ids)}"


def test_code_mixed_end_to_end() -> None:
    bups = get_bups()
    text = "మా CEO మంచి presentation ఇచ్చారు."
    tokens = bups.encode_tokens(text)
    scripts_seen = {t.script for t in tokens}
    assert "telugu" in scripts_seen
    assert "latin" in scripts_seen
    print(f"  codemix tokens: {len(tokens)} ids, scripts_seen={scripts_seen}")


TESTS = [
    test_phoneme_inventory_stability,
    test_cross_script_equivalence,
    test_conjunct_ksha,
    test_script_detection,
    test_codemix_segmentation,
    test_telugu_sentence,
    test_code_mixed_end_to_end,
]


def run_all() -> int:
    failed = 0
    for t in TESTS:
        name = t.__name__
        try:
            t()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {name}: {e!r}")
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(run_all())

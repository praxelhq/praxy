"""Brahmic Unified Phoneme Space (BUPS).

All Brahmic-derived scripts — Devanagari (Hindi, Marathi), Telugu, Tamil,
Kannada, Bengali, Gujarati, Malayalam — encode the same phonetic system.
A Telugu "క", a Devanagari "क", and a Kannada "ಕ" are the same sound /ka/.
This module exposes that equivalence as a single phoneme inventory, shared
across scripts, with:

- **Deterministic G2P**: any Brahmic-script string → a sequence of phoneme ids,
  identical across source scripts for phonetically-equivalent text.
- **Script residuals**: each script gets a small tag so an acoustic model can
  still learn subtle per-language prosodic drift on top of the shared base.
- **Latin / English fallback**: Latin text passes through a small English G2P
  proxy so code-mixed sentences stay in one token space.

Implementation: we lean on ``indic_transliteration`` (Apache-2.0) for the
heavy lifting of script-to-ISO-15919 conversion. ISO-15919 is a Unicode-based
phonetic romanization designed precisely for Indic scripts — it collapses
graphemic differences between scripts while preserving phonetic distinctions.
We treat the ISO-15919 tokens as our phoneme inventory, with a small
normalization pass for TTS needs (long-vowel merging, nasal/anusvara unifying).

Usage::

    from praxy.linguistics.bups import BUPS

    bups = BUPS()
    ids = bups.encode("నేను ఇవాళ బాగున్నాను.", script="telugu")
    # ids is a list[int] of phoneme ids from a shared ~80-token inventory
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# -----------------------------------------------------------------------------
# Phoneme inventory.
#
# We use ISO-15919 symbols (with minor normalisation) as our phoneme ids. The
# inventory covers:
#   - all Indic consonants (aspirated/unaspirated, voiced/unvoiced, retroflex)
#   - long/short vowels and diphthongs
#   - anusvara, visarga, and the chandra-bindu nasalisation
#   - a small English-shared set of non-Indic phonemes needed for code-mix
#     (e.g., /f/, /z/ as borrowed phonemes, /θ/ for English /th/)
#
# Stability note: the order of this list determines the phoneme id mapping.
# Once the first BUPS-trained checkpoint ships, appending new phonemes is safe
# but reordering is not — keep this list append-only.
# -----------------------------------------------------------------------------

PHONEME_INVENTORY: list[str] = [
    # --- Special tokens ---
    "<pad>", "<bos>", "<eos>", "<unk>", "<sil>", "<sp>",  # pad, bos, eos, unknown, long silence, short pause
    # --- Vowels: short ---
    "a", "i", "u", "r̥", "l̥", "e", "o",
    # --- Vowels: long ---
    "ā", "ī", "ū", "r̥̄", "l̥̄", "ē", "ō",
    # --- Diphthongs ---
    "ai", "au",
    # --- Extra vowels (Dravidian short e/o distinctly, Marathi candra) ---
    "ĕ", "ŏ", "æ",  # candra-e, candra-o, and æ for English /a/ in "cat"
    # --- Nasal modifiers ---
    "ṁ",   # anusvara (nasalizes preceding vowel, assimilates to following consonant)
    "m̐",  # chandra-bindu (pure nasalisation)
    "ḥ",   # visarga
    # --- Stops: velar ---
    "k", "kh", "g", "gh", "ṅ",
    # --- Stops: palatal ---
    "c", "ch", "j", "jh", "ñ",
    # --- Stops: retroflex ---
    "ṭ", "ṭh", "ḍ", "ḍh", "ṇ",
    # --- Stops: dental ---
    "t", "th", "d", "dh", "n",
    # --- Stops: labial ---
    "p", "ph", "b", "bh", "m",
    # --- Approximants ---
    "y", "r", "l", "v",
    # --- Sibilants / fricatives ---
    "ś", "ṣ", "s", "h",
    # --- Tamil-specific / misc ---
    "ḻ",   # Tamil zha (ழ), retroflex approximant
    "ḷ",   # retroflex l (ள, ळ)
    "ṟ",   # Tamil strong r (ற)
    "ṉ",   # Tamil alveolar n (ன)
    # --- English-borrowed phonemes for code-mix ---
    "f",
    "z",
    "θ",   # English /th/ voiceless
    "ð",   # English /th/ voiced
    "ʃ",   # English /sh/ where distinct from ś
    "w",
    "ŋ",   # English /ng/ where distinct from ṅ
]

PHONEME_TO_ID: dict[str, int] = {p: i for i, p in enumerate(PHONEME_INVENTORY)}
ID_TO_PHONEME: dict[int, str] = {i: p for i, p in enumerate(PHONEME_INVENTORY)}


def phoneme_count() -> int:
    return len(PHONEME_INVENTORY)


def phoneme_id(phoneme: str) -> int:
    """Return the id of a phoneme; falls back to <unk> if unknown."""
    return PHONEME_TO_ID.get(phoneme, PHONEME_TO_ID["<unk>"])


# -----------------------------------------------------------------------------
# Script registry.
#
# Each script gets:
#   - its sanscript name (for transliterate calls)
#   - a residual id (small int the acoustic model can add as a language tag)
#
# Script residuals are an append-only list — same stability rule as phonemes.
# -----------------------------------------------------------------------------

SCRIPT_RESIDUALS: list[str] = [
    "latin",         # 0 — default / English
    "devanagari",    # 1 — Hindi, Marathi, Sanskrit
    "telugu",        # 2
    "tamil",         # 3
    "kannada",       # 4
    "bengali",       # 5
    "gujarati",      # 6
    "malayalam",     # 7
    "oriya",         # 8
    "gurmukhi",      # 9
]

SCRIPT_RESIDUAL_TO_ID: dict[str, int] = {s: i for i, s in enumerate(SCRIPT_RESIDUALS)}

SANSCRIPT_NAMES: dict[str, str] = {
    "devanagari": sanscript.DEVANAGARI,
    "telugu": sanscript.TELUGU,
    "tamil": sanscript.TAMIL,
    "kannada": sanscript.KANNADA,
    "bengali": sanscript.BENGALI,
    "gujarati": sanscript.GUJARATI,
    "malayalam": sanscript.MALAYALAM,
    "oriya": sanscript.ORIYA,
    "gurmukhi": sanscript.GURMUKHI,
}


# -----------------------------------------------------------------------------
# Script detection.
# -----------------------------------------------------------------------------

_SCRIPT_UNICODE_RANGES: list[tuple[str, int, int]] = [
    ("devanagari", 0x0900, 0x097F),
    ("bengali",    0x0980, 0x09FF),
    ("gurmukhi",   0x0A00, 0x0A7F),
    ("gujarati",   0x0A80, 0x0AFF),
    ("oriya",      0x0B00, 0x0B7F),
    ("tamil",      0x0B80, 0x0BFF),
    ("telugu",     0x0C00, 0x0C7F),
    ("kannada",    0x0C80, 0x0CFF),
    ("malayalam",  0x0D00, 0x0D7F),
]


def detect_script(char: str) -> str:
    """Return the script name for a single character, or 'latin' for all else."""
    if not char:
        return "latin"
    cp = ord(char[0])
    for name, lo, hi in _SCRIPT_UNICODE_RANGES:
        if lo <= cp <= hi:
            return name
    # Heuristic: Latin letters, digits, punctuation → 'latin' bucket
    return "latin"


def segment_by_script(text: str) -> list[tuple[str, str]]:
    """Split text into (script, run) pairs. Whitespace stays with the preceding run."""
    if not text:
        return []
    runs: list[tuple[str, str]] = []
    cur_script: str | None = None
    cur_chars: list[str] = []
    for ch in text:
        if ch.isspace():
            cur_chars.append(ch)
            continue
        script = detect_script(ch)
        if cur_script is None:
            cur_script = script
            cur_chars.append(ch)
        elif script == cur_script:
            cur_chars.append(ch)
        else:
            runs.append((cur_script, "".join(cur_chars)))
            cur_script = script
            cur_chars = [ch]
    if cur_chars and cur_script is not None:
        runs.append((cur_script, "".join(cur_chars)))
    return runs


# -----------------------------------------------------------------------------
# ISO-15919 → BUPS phoneme tokenizer.
#
# ISO-15919 uses multi-character phoneme tokens (e.g. "kh", "ai", "r̥"). We
# tokenize greedily, longest-match-first.
# -----------------------------------------------------------------------------

# Precompile a regex that matches any phoneme token, longest first.
@lru_cache(maxsize=1)
def _phoneme_token_regex() -> re.Pattern[str]:
    tokens = sorted(
        (p for p in PHONEME_INVENTORY if not p.startswith("<")),
        key=len,
        reverse=True,
    )
    # Escape each token for regex, then join as alternation.
    escaped = [re.escape(t) for t in tokens]
    return re.compile("|".join(escaped))


def _tokenize_iso15919(iso: str) -> list[str]:
    """Greedy-longest-match tokenize an ISO-15919 string into BUPS phonemes."""
    # Normalize NFC so combining marks are canonical.
    iso = unicodedata.normalize("NFC", iso)
    pattern = _phoneme_token_regex()
    tokens: list[str] = []
    i = 0
    n = len(iso)
    while i < n:
        ch = iso[i]
        if ch.isspace():
            # collapse whitespace to a single short-pause token
            while i < n and iso[i].isspace():
                i += 1
            if tokens and tokens[-1] != "<sp>":
                tokens.append("<sp>")
            continue
        if ch in ".!?":
            # sentence-ish boundary → longer silence
            if tokens and tokens[-1] != "<sil>":
                tokens.append("<sil>")
            i += 1
            continue
        if ch in ",;:":
            if tokens and tokens[-1] != "<sp>":
                tokens.append("<sp>")
            i += 1
            continue
        m = pattern.match(iso, i)
        if m:
            tokens.append(m.group(0))
            i = m.end()
        else:
            # Unknown char — skip to avoid getting stuck. Could record <unk>
            # here if we want to surface unknowns later.
            i += 1
    return tokens


# -----------------------------------------------------------------------------
# Simple English-letter G2P fallback.
#
# For code-mixed Latin text we want *something* — not a full English G2P,
# but enough that "CEO" and "quarter" don't emit <unk>s. We use a
# minimal letter→phoneme heuristic: digraphs first, then single letters.
# Later sprints can drop in a proper G2P (e.g., phonemizer + espeak-ng).
# -----------------------------------------------------------------------------

_EN_DIGRAPHS = {
    "th": ["θ"], "sh": ["ʃ"], "ch": ["c"], "ph": ["f"], "wh": ["w"],
    "ng": ["ŋ"], "ck": ["k"], "qu": ["k", "w"],
}

_EN_SINGLES = {
    "a": "a", "b": "b", "c": "k", "d": "d", "e": "e", "f": "f", "g": "g",
    "h": "h", "i": "i", "j": "j", "k": "k", "l": "l", "m": "m", "n": "n",
    "o": "o", "p": "p", "q": "k", "r": "r", "s": "s", "t": "t", "u": "u",
    "v": "v", "w": "w", "x": "k", "y": "y", "z": "z",
}


def _english_g2p(token: str) -> list[str]:
    """Very rough letter→phoneme for Latin script. Good enough for code-mix anchoring."""
    token = token.lower()
    phonemes: list[str] = []
    i = 0
    while i < len(token):
        if i + 1 < len(token):
            dg = token[i:i + 2]
            if dg in _EN_DIGRAPHS:
                phonemes.extend(_EN_DIGRAPHS[dg])
                i += 2
                continue
        ch = token[i]
        if ch.isalpha():
            p = _EN_SINGLES.get(ch)
            if p:
                phonemes.append(p)
        elif ch.isdigit():
            # digits left as-is for now — a number-normalizer will run first in prod
            phonemes.append("<unk>")
        i += 1
    return phonemes


# -----------------------------------------------------------------------------
# Top-level BUPS encoder.
# -----------------------------------------------------------------------------

@dataclass
class Token:
    """One BUPS token — phoneme id + the script residual it came from."""
    phoneme: str
    phoneme_id: int
    script: str
    script_residual_id: int

    def __repr__(self) -> str:
        return f"Token({self.phoneme!r}, script={self.script})"


class BUPS:
    """Brahmic Unified Phoneme Space encoder.

    Stateless — thread-safe, cache-friendly. Instantiate once per process.
    """

    def __init__(self) -> None:
        self.pad_id = PHONEME_TO_ID["<pad>"]
        self.bos_id = PHONEME_TO_ID["<bos>"]
        self.eos_id = PHONEME_TO_ID["<eos>"]
        self.unk_id = PHONEME_TO_ID["<unk>"]

    def encode(
        self,
        text: str,
        script: str | None = None,
        add_bos_eos: bool = False,
    ) -> list[int]:
        """Encode text into BUPS phoneme ids.

        If ``script`` is given, the entire string is treated as that script.
        Otherwise we script-segment and let each run carry its own script tag.
        """
        ids: list[int] = [self.bos_id] if add_bos_eos else []
        for tok in self.encode_tokens(text, script=script):
            ids.append(tok.phoneme_id)
        if add_bos_eos:
            ids.append(self.eos_id)
        return ids

    def encode_tokens(
        self, text: str, script: str | None = None
    ) -> list[Token]:
        """Encode to Token objects (carries script residual info for each phoneme)."""
        tokens: list[Token] = []
        if script is not None:
            runs = [(script, text)]
        else:
            runs = segment_by_script(text)

        for run_script, run_text in runs:
            residual = SCRIPT_RESIDUAL_TO_ID.get(run_script, 0)
            phonemes = self._script_to_phonemes(run_script, run_text)
            for p in phonemes:
                tokens.append(
                    Token(
                        phoneme=p,
                        phoneme_id=phoneme_id(p),
                        script=run_script,
                        script_residual_id=residual,
                    )
                )
        return tokens

    def _script_to_phonemes(self, script: str, text: str) -> list[str]:
        if script in SANSCRIPT_NAMES:
            # Brahmic → ISO-15919 → tokenize.
            iso = transliterate(text, SANSCRIPT_NAMES[script], sanscript.ISO)
            return _tokenize_iso15919(iso)
        # Latin path: split into whitespace-delimited tokens, run the heuristic,
        # interleave short pauses between them so word boundaries stay.
        out: list[str] = []
        for word in re.split(r"(\s+)", text):
            if not word:
                continue
            if word.isspace():
                if out and out[-1] != "<sp>":
                    out.append("<sp>")
                continue
            out.extend(_english_g2p(word))
        return out


# Convenience singleton for most callers.
_BUPS_SINGLETON: BUPS | None = None


def get_bups() -> BUPS:
    """Process-wide singleton."""
    global _BUPS_SINGLETON
    if _BUPS_SINGLETON is None:
        _BUPS_SINGLETON = BUPS()
    return _BUPS_SINGLETON


def preprocess_text_for_chatterbox(text: str, language_code: str) -> str:
    """Transliterate Brahmic-script text into an ISO-15919 Roman surface form
    that Chatterbox's multilingual BPE tokenizer handles cleanly.

    The hypothesis (novel to our project): Chatterbox's MTLTokenizer does not
    cover Telugu in its 23-language inventory, so feeding raw Telugu script
    wastes the model's capacity on learning a sub-optimal tokenization. Roman
    transliteration via ISO-15919 / Latin phonetic form lets Chatterbox route
    Telugu sounds through its already-rich Latin embedding space.

    For English / Latin-script input, returns the text unchanged.

    Example:
        in : నేను ఇవాళ బాగున్నాను
        out: nēnu ivāḷa bāgunnānu

    Args:
        text: input text string (any supported script)
        language_code: ISO-639-1 language code ("te", "hi", "ta", "en", ...)

    Returns:
        Latin-script approximation suitable for BPE tokenizers that lack
        Indic coverage. Punctuation is preserved.
    """
    if not text:
        return text

    # For Latin-script inputs (English, code-mix English parts) pass through.
    # Code-switched Indic+Latin is handled by segment_by_script.
    runs = segment_by_script(text)
    out_parts: list[str] = []
    for run_script, run_text in runs:
        if run_script == "latin":
            out_parts.append(run_text)
            continue
        sanscript_name = SANSCRIPT_NAMES.get(run_script)
        if sanscript_name is None:
            out_parts.append(run_text)  # unknown script — pass through
            continue
        try:
            iso = transliterate(run_text, sanscript_name, sanscript.ISO)
            out_parts.append(iso)
        except Exception:
            out_parts.append(run_text)
    return "".join(out_parts)


# -----------------------------------------------------------------------------
# Diagnostic: dump the inventory sizes and a script equivalence check.
# -----------------------------------------------------------------------------

def _diagnostic() -> None:
    """Print inventory + show that क / క / ಕ all encode to the same id."""
    print(f"Phoneme inventory size: {phoneme_count()}")
    print(f"Script residuals: {len(SCRIPT_RESIDUALS)}")
    bups = get_bups()
    for script, sample in [
        ("devanagari", "क"),
        ("telugu", "క"),
        ("kannada", "ಕ"),
    ]:
        ids = bups.encode(sample + "a", script=script)  # add explicit 'a' suffix
        print(f"  {script:12s} {sample!r}: ids={ids} phonemes={[ID_TO_PHONEME[i] for i in ids]}")


if __name__ == "__main__":
    _diagnostic()

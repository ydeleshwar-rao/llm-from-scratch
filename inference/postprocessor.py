"""
Post-processor — cleans raw LLM output before returning to the caller.

Pipeline (each step is opt-in via flags, applied in this order):
  1. strip_special    — remove <|endoftext|> and similar tokens
  2. strip_prompt     — remove the prompt echo from the start of output
  3. strip_garbage    — remove non-printable chars, backslash escapes, ALL non-ASCII
  4. code_filter      — remove tokens that look like programming identifiers
  5. fix_whitespace   — collapse excessive whitespace / newlines
  6. truncate_sentence— cut at the last complete sentence (opt-in, default off)
"""

import re


# ── Regex patterns ────────────────────────────────────────────────────────────

# tiktoken special tokens: <|endoftext|>, <|fim_prefix|>, etc.
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|>]{1,40}\|>", flags=re.IGNORECASE)

# Literal backslash escape sequences the model sometimes emits verbatim
# e.g. \\n  \\t  \\uXXXX  \\Db  \\uc
_BACKSLASH_RE = re.compile(r"\\[a-zA-Z0-9\\]")

# Everything outside printable ASCII (0x20–0x7E) plus newline (0x0A) and tab (0x09).
# Single regex that catches: non-ASCII Latin (æ ñ đ), Cyrillic (Э),
# CJK (을), box-drawing (─), surrogates, C0/C1 control chars, etc.
_NON_PRINTABLE_RE = re.compile(r"[^\x09\x0a\x20-\x7e]")

# Whitespace normalisation
_TRIPLE_NEWLINE_RE = re.compile(r"\n{3,}")
_MULTI_SPACE_RE    = re.compile(r" {2,}")

# Sentence boundary for truncation
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s")

# ── Code-token heuristics ─────────────────────────────────────────────────────

# Characters that never appear in natural-language words
_CODE_CHARS = frozenset('{}[]();@#$%^&*~`|=<>\\/')

# Any internal camelCase transition: BufferedImage(dI), NullPointerException(lP,rE), stdClass(dC)
# Normal English words never have lowercase→uppercase inside them.
_CAMEL_RE = re.compile(r'[a-z][A-Z]')

# SCREAMING_SNAKE or _LEADING_UNDER_CAPS
# No \b needed: .search() scans the whole token as a substring.
_UPPER_SNAKE_RE = re.compile(r'[A-Z]{2,}_[A-Z_]{2,}|^_[A-Z_]{2,}')

# Tokens that start with a dot: .jsoup  .FromResult
_DOT_PREFIX_RE = re.compile(r'^\.')

# obj.method  or  word.Word chains
_DOT_CHAIN_RE = re.compile(r'\w\.\w')

# Alphanumeric mix: Elon093  Tb123  (code IDs / version strings)
_ALNUM_MIX_RE = re.compile(r'[a-zA-Z]\d{2,}|\d{2,}[a-zA-Z]')


def _is_code_token(word: str) -> bool:
    """Return True if this whitespace-split token looks like code, not natural text."""
    if not word:
        return False
    # Single-character code symbols: { } ; etc.
    if len(word) == 1:
        return word in _CODE_CHARS
    if any(c in word for c in _CODE_CHARS):
        return True
    if _CAMEL_RE.search(word):
        return True
    if _UPPER_SNAKE_RE.search(word):
        return True
    if _DOT_PREFIX_RE.match(word):
        return True
    if _DOT_CHAIN_RE.search(word):
        return True
    if _ALNUM_MIX_RE.search(word):
        return True
    return False


# ── Public API ────────────────────────────────────────────────────────────────

def clean(
    text: str,
    prompt: str             = "",
    strip_special: bool     = True,
    strip_prompt: bool      = True,
    strip_garbage: bool     = True,
    code_filter: bool       = True,
    fix_whitespace: bool    = True,
    truncate_sentence: bool = False,
) -> str:
    """
    Clean a raw LLM output string.

    Args:
        text:               Raw string from the model.
        prompt:             Original prompt — stripped from the output start.
        strip_special:      Remove <|...|> tokens.
        strip_prompt:       Remove the prompt echo from the start.
        strip_garbage:      Remove backslash artefacts and ALL non-ASCII characters.
        code_filter:        Remove tokens that look like code identifiers.
        fix_whitespace:     Collapse excessive whitespace.
        truncate_sentence:  Cut at the last complete sentence boundary.

    Returns:
        Cleaned string (always a str, never raises).
    """
    if strip_special:
        text = _SPECIAL_TOKEN_RE.sub("", text)

    if strip_prompt and prompt and text.startswith(prompt):
        text = text[len(prompt):]

    if strip_garbage:
        text = _BACKSLASH_RE.sub("", text)       # remove literal \n \t \uXXXX etc.
        text = _NON_PRINTABLE_RE.sub("", text)   # remove everything outside ASCII printable

    if code_filter:
        # Process line-by-line so real newlines are preserved after filtering
        lines = text.split("\n")
        filtered = []
        for line in lines:
            words = line.split()
            kept  = [w for w in words if not _is_code_token(w)]
            filtered.append(" ".join(kept))
        text = "\n".join(filtered)

    if fix_whitespace:
        text = _TRIPLE_NEWLINE_RE.sub("\n\n", text)
        text = _MULTI_SPACE_RE.sub(" ", text)
        text = text.strip()

    if truncate_sentence and text:
        parts = _SENTENCE_END_RE.split(text)
        if len(parts) > 1:
            text = parts[0]

    return text

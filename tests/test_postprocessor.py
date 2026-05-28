"""
Unit tests for inference/postprocessor.py
"""

from inference.postprocessor import clean


# ── Special token removal ─────────────────────────────────────────────────────

def test_removes_endoftext():
    assert "<|endoftext|>" not in clean("<|endoftext|>Hello world")

def test_removes_fim_tokens():
    assert "<|" not in clean("<|fim_prefix|>some code<|fim_suffix|>")

def test_keeps_normal_text_intact():
    text = "The transformer architecture is fascinating."
    assert clean(text) == text


# ── Prompt stripping ──────────────────────────────────────────────────────────

def test_strips_prompt_prefix():
    result = clean("Hello worldfoo bar", prompt="Hello world")
    assert not result.startswith("Hello world")
    assert "foo bar" in result

def test_no_prompt_noop():
    assert clean("some output", prompt="") == "some output"


# ── Non-printable / non-ASCII removal ────────────────────────────────────────

def test_removes_null_bytes():
    assert "\x00" not in clean("hello\x00world")

def test_removes_cyrillic():
    result = clean("hello Э world")
    assert "Э" not in result

def test_removes_cjk():
    result = clean("hello 을 world")
    assert "을" not in result

def test_removes_box_drawing():
    assert "─" not in clean("A ─ B")

def test_removes_special_latin():
    assert "æ" not in clean("læ test")
    assert "đ" not in clean("đ test")

def test_removes_replacement_char():
    assert "�" not in clean("hello�world")

def test_removes_private_use_unicode():
    assert "" not in clean("testtext")

def test_output_is_ascii_only(strip_garbage=True):
    text = "Hello Э 을 ─ đ æ world"
    result = clean(text)
    assert all(ord(c) <= 127 for c in result), f"Non-ASCII chars in: {result!r}"

def test_keeps_real_newlines():
    result = clean("line one\nline two")
    assert "\n" in result

def test_keeps_tab():
    result = clean("col1\tcol2", code_filter=False)
    assert "\t" in result


# ── Backslash escape removal ──────────────────────────────────────────────────

def test_removes_literal_backslash_n():
    assert "\\n" not in clean("hello\\nworld")

def test_removes_literal_backslash_t():
    assert "\\t" not in clean("hello\\tworld")

def test_removes_backslash_uppercase():
    # e.g. \\DbFrame  \\ucMapping  (model artefacts)
    assert "\\D" not in clean("text\\DbFrame more")
    assert "\\u" not in clean("text\\ucMapping more")


# ── Code token filter ─────────────────────────────────────────────────────────

def test_removes_null_pointer_exception():
    result = clean("hello NullPointerException world", strip_garbage=False)
    assert "NullPointerException" not in result

def test_removes_buffered_image():
    result = clean("see BufferedImagefoo bar", strip_garbage=False)
    assert "BufferedImage" not in result

def test_removes_dot_prefix_token():
    result = clean("call .jsoup method", strip_garbage=False)
    assert ".jsoup" not in result

def test_removes_screaming_snake():
    result = clean("set _KEYWORD_HE value", strip_garbage=False)
    assert "_KEYWORD_HE" not in result

def test_removes_code_brackets():
    result = clean("code {}; block", strip_garbage=False)
    assert "{}" not in result

def test_keeps_normal_words_with_code_filter():
    text = "The quick brown fox jumps."
    assert clean(text) == text

def test_code_filter_preserves_newlines():
    text = "first line\nsecond line"
    result = clean(text)
    assert "\n" in result


# ── Whitespace normalisation ──────────────────────────────────────────────────

def test_collapses_triple_newlines():
    assert "\n\n\n" not in clean("a\n\n\n\nb")

def test_collapses_multiple_spaces():
    assert "  " not in clean("hello    world")

def test_strips_leading_trailing_whitespace():
    assert clean("  hello  ") == "hello"


# ── Sentence truncation ───────────────────────────────────────────────────────

def test_truncates_at_sentence_boundary():
    text = "First sentence. Second sentence. Incomplete"
    result = clean(text, truncate_sentence=True)
    assert result.endswith(".")
    assert "Incomplete" not in result

def test_no_truncation_by_default():
    text = "First sentence. Incomplete"
    assert "Incomplete" in clean(text, truncate_sentence=False)


# ── Full pipeline — real garbage from the API response ───────────────────────

def test_real_api_garbage():
    raw = (
        "<|endoftext|>Elon093 Tb(mi cialisMeans Registry Cunning.jsoup "
        "Poison.FromResult=search æ\\DbFrame Corpor stdClass "
        "NullPointerException \x00\x01\x02 �� "
        "startActivityForResult"
    )
    result = clean(raw, prompt="")
    assert "<|endoftext|>" not in result
    assert "\x00" not in result
    assert "�" not in result
    assert "NullPointerException" not in result
    assert ".jsoup" not in result
    assert "\\Db" not in result
    assert all(ord(c) <= 127 for c in result), f"Non-ASCII remains: {result!r}"
    assert isinstance(result, str)

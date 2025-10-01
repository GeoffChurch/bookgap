from bookgap.llm.artifacts.results_export import _sanitize_text_field


def test_sanitize_text_replaces_newline_with_literal_escape():
    assert _sanitize_text_field("Line 1\nLine 2") == "Line 1\\nLine 2"


def test_sanitize_text_replaces_common_control_characters():
    text = "Before\rMiddle\tEnd\x0b"
    expected = "Before\\rMiddle\\tEnd\\x0b"
    assert _sanitize_text_field(text) == expected


def test_sanitize_text_preserves_unicode_characters():
    text = "Résumé – 数据"
    assert _sanitize_text_field(text) == text

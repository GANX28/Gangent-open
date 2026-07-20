"""Generated text safety checks."""

from __future__ import annotations


ENCODING_ARTIFACT_PATTERNS = (
    "\ufffd",
    "鈥?",
    "鉁?",
    "â€™",
    "â€œ",
    "â€",
    "â€“",
    "â€”",
    "璇诲彇",
    "鏂囦欢",
    "鍐呭",
    "鎬荤粨",
    "杈撳嚭",
    "淇濆瓨",
    " бк ",
)


def encoding_artifact_reason(text: str) -> str:
    """Return a short reason when generated text appears encoding-corrupted."""

    for pattern in ENCODING_ARTIFACT_PATTERNS:
        if pattern in text:
            display = pattern.encode("unicode_escape").decode("ascii")
            return (
                "Generated text appears encoding-corrupted. "
                f"Suspicious pattern: {display}. Regenerate using plain UTF-8 text."
            )
    return ""


def ensure_no_encoding_artifacts(text: str) -> None:
    """Raise ValueError if generated text contains known mojibake artifacts."""

    reason = encoding_artifact_reason(text)
    if reason:
        raise ValueError(reason)

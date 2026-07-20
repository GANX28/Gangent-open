"""Secret Guard（密钥护栏）。

第一版使用确定性规则做凭证隔离：
- 阻止读取明显敏感的路径；
- 扫描常见 API key、token、私钥格式；
- 把疑似密钥替换成占位符后再进入模型、session 或 audit log。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SENSITIVE_EXACT_NAMES = {
    ".npmrc",
    ".pypirc",
    ".netrc",
    ".git-credentials",
    "credentials",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
}
SENSITIVE_DIRECTORIES = {".ssh", ".aws", ".azure", ".gcp"}
SENSITIVE_SUFFIXES = {".pem", ".key", ".p12", ".pfx"}


@dataclass(frozen=True)
class SecretFinding:
    """一次密钥扫描命中。"""

    label: str
    start: int
    end: int


SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "PRIVATE_KEY",
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
            re.DOTALL,
        ),
    ),
    ("OPENAI_API_KEY", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("GITHUB_TOKEN", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{20,}\b")),
    ("GITHUB_PAT", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b")),
    ("AWS_ACCESS_KEY_ID", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    (
        "GENERIC_SECRET_ASSIGNMENT",
        re.compile(
            r"(?i)\b(api[_-]?key|secret|token|password)\b\s*[:=]\s*[\"']?([^\s\"']{8,})"
        ),
    ),
)


def is_sensitive_path(path: str | Path) -> bool:
    """判断路径是否明显像凭证文件或凭证目录。"""

    value = Path(path)
    parts = [part.lower() for part in value.parts]
    name = value.name.lower()

    if any(part in SENSITIVE_DIRECTORIES for part in parts):
        return True
    if name.startswith(".env"):
        return True
    if name in SENSITIVE_EXACT_NAMES:
        return True
    return value.suffix.lower() in SENSITIVE_SUFFIXES


def scan_secrets(text: str) -> list[SecretFinding]:
    """扫描文本里的疑似密钥。"""

    findings: list[SecretFinding] = []
    for label, pattern in SECRET_PATTERNS:
        for match in pattern.finditer(text):
            findings.append(SecretFinding(label=label, start=match.start(), end=match.end()))
    return _merge_overlapping(findings)


def contains_secret(text: str) -> bool:
    """判断文本是否包含疑似密钥。"""

    return bool(scan_secrets(text))


def secret_labels(text: str) -> list[str]:
    """返回文本命中的密钥类型，去重后排序。"""

    return sorted({finding.label for finding in scan_secrets(text)})


def redact_secrets(text: str) -> str:
    """把疑似密钥替换成占位符。"""

    findings = scan_secrets(text)
    if not findings:
        return text

    parts: list[str] = []
    cursor = 0
    for finding in findings:
        parts.append(text[cursor : finding.start])
        parts.append(f"[REDACTED_{finding.label}]")
        cursor = finding.end
    parts.append(text[cursor:])
    return "".join(parts)


def redact_data(value: Any) -> Any:
    """递归脱敏 dict/list/string 数据，供 audit log 和 JSON 输出使用。"""

    if isinstance(value, str):
        return redact_secrets(value)
    if isinstance(value, list):
        return [redact_data(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_data(item) for item in value)
    if isinstance(value, dict):
        return {key: redact_data(item) for key, item in value.items()}
    return value


def _merge_overlapping(findings: list[SecretFinding]) -> list[SecretFinding]:
    if not findings:
        return []

    ordered = sorted(findings, key=lambda item: (item.start, item.end))
    merged: list[SecretFinding] = []
    for finding in ordered:
        if not merged or finding.start >= merged[-1].end:
            merged.append(finding)
            continue

        previous = merged[-1]
        if finding.end > previous.end:
            merged[-1] = SecretFinding(
                label=previous.label,
                start=previous.start,
                end=finding.end,
            )
    return merged

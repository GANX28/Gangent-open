"""Audit Log（审计日志）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .output import result_to_dict
from .runtime import RuntimeResult
from .secret_guard import redact_data, redact_secrets


DEFAULT_AUDIT_LOG = Path(".gangent") / "audit" / "latest.jsonl"


def default_audit_path(workspace_root: str) -> Path:
    """返回 workspace root 下默认审计日志路径。"""

    return Path(workspace_root).resolve() / DEFAULT_AUDIT_LOG


def append_audit_record(
    result: RuntimeResult,
    session_id: str,
    user_message: str,
    path: str | Path,
) -> Path:
    """追加一条任务审计记录。"""

    target = Path(path)
    record = audit_record(result, session_id=session_id, user_message=user_message)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return target


def audit_record(
    result: RuntimeResult,
    session_id: str,
    user_message: str,
) -> dict[str, Any]:
    """构造一条审计记录。"""

    data = result_to_dict(result)
    data["session_id"] = session_id
    data["user_message"] = redact_secrets(user_message)
    return redact_data(data)

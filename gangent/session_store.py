"""Session Store（会话存储层）。

第一版用 JSON 文件持久化 SessionState。
它只解决本地 CLI 重启后继续会话的问题，不实现数据库、同步、加密或多用户隔离。
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .session import SessionState, SessionTurn, create_session


DEFAULT_SESSION_FILE = Path(".gangent") / "sessions" / "latest.json"


def default_session_path(workspace_root: str) -> Path:
    """返回 workspace root 下默认 session 文件路径。"""

    return Path(workspace_root).resolve() / DEFAULT_SESSION_FILE


def save_session(session: SessionState, path: str | Path | None = None) -> Path:
    """把 SessionState 保存成 JSON 文件。"""

    target = Path(path) if path is not None else default_session_path(session.workspace_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(asdict(session), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target


def load_session(path: str | Path) -> SessionState:
    """从 JSON 文件读取 SessionState。"""

    source = Path(path)
    data = json.loads(source.read_text(encoding="utf-8"))
    return session_from_dict(data)


def load_or_create_session(
    workspace_root: str,
    path: str | Path | None = None,
    resume: bool = False,
) -> SessionState:
    """按需恢复已有 session，不恢复时创建新 session。"""

    session_path = Path(path) if path is not None else default_session_path(workspace_root)
    if resume and session_path.exists():
        return load_session(session_path)
    return create_session(workspace_root)


def session_from_dict(data: dict[str, Any]) -> SessionState:
    """把 JSON dict 转回 SessionState。"""

    turns = [SessionTurn(**turn) for turn in data.get("turns", [])]
    return SessionState(
        session_id=data["session_id"],
        workspace_root=data["workspace_root"],
        context_summary=data.get("context_summary", ""),
        turns=turns,
        created_at=data.get("created_at", ""),
        updated_at=data.get("updated_at", ""),
    )

"""Permission Profile（权限配置）和 Sandbox Boundary（沙箱边界）。

这一层集中处理“哪些路径能读、哪些路径能写、哪些操作必须拒绝”。
第一版不是操作系统级沙箱，而是 runtime 内部的逻辑权限层。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class AccessMode(str, Enum):
    """文件访问模式。"""

    READ = "read"
    WRITE = "write"


class PermissionError(ValueError):
    """权限检查失败时使用的错误类型。"""


@dataclass(frozen=True)
class PermissionProfile:
    """第一版权限配置。

    workspace_root 是主工作区。
    read_roots 是允许读取的目录。
    write_roots 是允许写入的目录。
    max_file_bytes 限制单次读写文件大小，避免模型把大文件塞进上下文或误写大文件。
    """

    workspace_root: str
    read_roots: tuple[str, ...] = field(default_factory=tuple)
    write_roots: tuple[str, ...] = field(default_factory=tuple)
    max_file_bytes: int = 20_000
    allow_hidden_files: bool = False


def default_permission_profile(workspace_root: str) -> PermissionProfile:
    """创建默认 workspace-write 权限配置。

    第一版默认只能读写当前 workspace root，不自动扩展到其他项目。
    """

    root = str(Path(workspace_root).resolve())
    return PermissionProfile(
        workspace_root=root,
        read_roots=(root,),
        write_roots=(root,),
    )


def resolve_workspace_path(path: str, workspace_root: str) -> Path:
    """把路径解析为 workspace root 内的绝对路径。

    这是兼容旧调用的基础函数：只检查目标是否越出 workspace root。
    """

    root = Path(workspace_root).resolve()
    raw_path = _normalize_workspace_path(path, root)
    target = raw_path.resolve() if raw_path.is_absolute() else (root / raw_path).resolve()

    try:
        target.relative_to(root)
    except ValueError as exc:
        raise PermissionError(f"Path escapes workspace root: {path}") from exc

    return target


def resolve_allowed_path(
    path: str,
    profile: PermissionProfile,
    access_mode: AccessMode,
) -> Path:
    """解析并检查路径是否落在对应权限根目录内。"""

    root = Path(profile.workspace_root).resolve()
    raw_path = _normalize_workspace_path(path, root)
    target = raw_path.resolve() if raw_path.is_absolute() else (root / raw_path).resolve()
    roots = profile.read_roots if access_mode == AccessMode.READ else profile.write_roots
    resolved_roots = [Path(item).resolve() for item in roots]

    if not any(_is_relative_to(target, allowed_root) for allowed_root in resolved_roots):
        raise PermissionError(
            f"Path is outside allowed {access_mode.value} roots: {path}"
        )

    if not profile.allow_hidden_files and _contains_hidden_part(target, resolved_roots):
        raise PermissionError(f"Hidden or metadata path is not allowed: {path}")

    return target


def _normalize_workspace_path(path: str, workspace_root: Path) -> Path:
    """Normalize common model-generated workspace path aliases.

    V1 runtime often sets workspace_root directly to a folder named
    "workspace". Models may still produce paths like "workspace/a.txt",
    "Gangent/workspace/a.txt", or "/workspace/Gangent/workspace/a.txt".
    These are redundant references to the same root, so we fold them before
    permission checks. Real absolute paths are still checked by the sandbox
    boundary after normalization.
    """

    text = str(path).strip()
    if not text or text in {".", "./", ".\\"}:
        return Path(".")

    normalized = text.replace("\\", "/")
    raw = Path(text)
    if raw.is_absolute() and not normalized.startswith("/workspace/"):
        return raw

    parts = [part for part in normalized.lstrip("/").split("/") if part and part != "."]
    lowered = [part.lower() for part in parts]
    root_name = workspace_root.name.lower()
    project_name = workspace_root.parent.name.lower()

    if root_name == "workspace":
        if len(lowered) >= 3 and lowered[0] == "workspace" and lowered[1] == project_name and lowered[2] == root_name:
            parts = parts[3:]
            lowered = lowered[3:]
        elif len(lowered) >= 2 and lowered[0] == project_name and lowered[1] == root_name:
            parts = parts[2:]
            lowered = lowered[2:]
        elif lowered and lowered[0] == root_name:
            parts = parts[1:]
            lowered = lowered[1:]
    if not parts:
        return Path(".")
    return Path(*parts)


def ensure_file_size_allowed(path: Path, max_file_bytes: int, label: str) -> None:
    """检查文件大小是否超过权限配置。"""

    if path.exists() and path.stat().st_size > max_file_bytes:
        raise PermissionError(f"{label} is too large for version one: {path.name}")


def ensure_content_size_allowed(content: str, max_file_bytes: int) -> None:
    """检查即将写入的文本大小。"""

    if len(content.encode("utf-8")) > max_file_bytes:
        raise PermissionError("Content is too large for version-one write.")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _contains_hidden_part(path: Path, allowed_roots: list[Path]) -> bool:
    """检查相对路径中是否包含隐藏/元数据片段。"""

    relative_parts: tuple[str, ...] = path.parts
    for root in allowed_roots:
        if _is_relative_to(path, root):
            relative_parts = path.relative_to(root).parts
            break
    return any(part.startswith(".") for part in relative_parts)

"""SandboxRunner v1.

SandboxRunner（沙箱执行器）把“要执行什么命令”和“在哪里执行、
如何限制执行”分开。第一版只实现 LocalRunner（本地执行器），
但所有命令型工具都通过统一入口获得超时、输出截断和编码处理。
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SandboxCommand:
    """一次受控命令执行请求。

    name 用于审计和错误提示；args 是已经由程序构造好的参数列表，
    不是模型直接给出的 shell 字符串，这样可以避免命令注入。
    """

    name: str
    args: list[str]
    cwd: str
    timeout_seconds: int
    max_output_bytes: int
    allow_network: bool = False
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SandboxResult:
    """一次受控命令执行结果。"""

    success: bool
    exit_code: int | None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    error: str | None = None

    @property
    def output(self) -> str:
        """合并 stdout/stderr，供上层工具统一返回给 runtime。"""

        text = (self.stdout + self.stderr).strip()
        return text or "(no output)"


class LocalRunner:
    """本地执行器。

    它不是操作系统级强沙箱，只是把本地子进程执行集中收口。
    后续 DockerRunner / RemoteRunner 可以实现同样接口。
    """

    def run(self, command: SandboxCommand) -> SandboxResult:
        cwd = Path(command.cwd).resolve()
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env.update(command.env)

        try:
            result = subprocess.run(
                command.args,
                cwd=cwd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=command.timeout_seconds,
                shell=False,
                check=False,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            return SandboxResult(
                success=False,
                exit_code=None,
                stdout=_coerce_text(exc.stdout),
                stderr=_coerce_text(exc.stderr),
                timed_out=True,
                error=f"Command timed out after {command.timeout_seconds}s: {command.name}",
            )
        except OSError as exc:
            return SandboxResult(
                success=False,
                exit_code=None,
                error=f"Command could not start: {command.name}: {exc}",
            )

        stdout = _truncate_text(result.stdout or "", command.max_output_bytes)
        stderr = _truncate_text(result.stderr or "", command.max_output_bytes)
        combined_size = len((stdout + stderr).encode("utf-8"))
        if combined_size > command.max_output_bytes:
            combined = _truncate_text(stdout + stderr, command.max_output_bytes)
            stdout = combined
            stderr = ""

        return SandboxResult(
            success=result.returncode == 0,
            exit_code=result.returncode,
            stdout=stdout,
            stderr=stderr,
        )


def _coerce_text(value: str | bytes | None) -> str:
    """把 subprocess 可能返回的 bytes/None 统一转成字符串。"""

    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _truncate_text(text: str, max_bytes: int) -> str:
    """按 UTF-8 字节预算截断输出，避免大输出塞满上下文。"""

    data = text.encode("utf-8")
    if len(data) <= max_bytes:
        return text
    truncated = data[:max_bytes].decode("utf-8", errors="ignore")
    return truncated + "\n... output truncated"

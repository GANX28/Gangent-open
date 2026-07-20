"""Command risk classification for run_command.

This is not a complete security sandbox. It is a conservative command gate
that combines structured argv, deny rules, and approval escalation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class CommandRisk(str, Enum):
    """Command risk class used by policy.py."""

    ALLOW = "allow"
    ESCALATE = "escalate"
    BLOCK = "block"


@dataclass(frozen=True)
class CommandAssessment:
    """Risk assessment for one command."""

    risk: CommandRisk
    reason: str
    args: list[str]
    cwd: str = "."


BLOCKED_EXECUTABLES = {
    "bash",
    "cmd",
    "cmd.exe",
    "diskpart",
    "format",
    "netsh",
    "powershell",
    "powershell.exe",
    "pwsh",
    "pwsh.exe",
    "reg",
    "reg.exe",
    "schtasks",
    "sh",
    "shutdown",
    "ssh",
    "sudo",
    "wsl",
}

DESTRUCTIVE_EXECUTABLES = {
    "del",
    "erase",
    "move",
    "mv",
    "rd",
    "rmdir",
    "rm",
}

NETWORK_OR_INSTALL_EXECUTABLES = {
    "cargo",
    "curl",
    "git",
    "go",
    "npm",
    "npx",
    "pip",
    "pip3",
    "pnpm",
    "poetry",
    "uv",
    "wget",
    "yarn",
}

SAFE_EXACT_COMMANDS = {
    ("git", "status"),
    ("git", "status", "--short"),
    ("git", "diff"),
    ("git", "diff", "--"),
    ("git", "log"),
    ("git", "show"),
    ("python", "--version"),
    ("python", "-V"),
    ("py", "--version"),
    ("py", "-V"),
    ("node", "--version"),
    ("npm", "--version"),
}

SAFE_PREFIXES = {
    ("git", "log"),
    ("git", "show"),
    ("git", "diff"),
    ("python", "-m", "unittest"),
    ("py", "-m", "unittest"),
    ("pytest",),
}

SHELL_METACHARS = {";", "&&", "||", "|", ">", ">>", "<", "$(", "`"}


def assess_run_command_args(tool_args: dict[str, Any]) -> CommandAssessment:
    """Assess run_command arguments without executing them."""

    raw_args = tool_args.get("args")
    if not isinstance(raw_args, list) or not raw_args:
        return CommandAssessment(CommandRisk.BLOCK, "run_command requires non-empty args list.", [])

    args: list[str] = []
    for item in raw_args:
        if not isinstance(item, str) or not item.strip():
            return CommandAssessment(CommandRisk.BLOCK, "run_command args must be non-empty strings.", [])
        args.append(item.strip())

    cwd = tool_args.get("cwd", ".")
    if not isinstance(cwd, str) or not cwd.strip():
        return CommandAssessment(CommandRisk.BLOCK, "run_command cwd must be a non-empty string.", args)

    executable = Path(args[0]).name.lower()
    if _contains_shell_metachar(args):
        return CommandAssessment(
            CommandRisk.BLOCK,
            "Shell metacharacters are not allowed; pass structured argv instead.",
            args,
            cwd,
        )
    if executable in BLOCKED_EXECUTABLES:
        return CommandAssessment(CommandRisk.BLOCK, f"Executable is blocked: {executable}", args, cwd)
    if executable in DESTRUCTIVE_EXECUTABLES:
        return CommandAssessment(
            CommandRisk.ESCALATE,
            f"Destructive filesystem command requires approval: {executable}",
            args,
            cwd,
        )
    if _is_safe_command(args):
        return CommandAssessment(CommandRisk.ALLOW, "Command is low-risk and allowed.", args, cwd)
    if executable in NETWORK_OR_INSTALL_EXECUTABLES:
        return CommandAssessment(
            CommandRisk.ESCALATE,
            f"Network, dependency, or repository command requires approval: {executable}",
            args,
            cwd,
        )
    return CommandAssessment(
        CommandRisk.ESCALATE,
        f"Command is not in the low-risk allowlist and requires approval: {executable}",
        args,
        cwd,
    )


def _contains_shell_metachar(args: list[str]) -> bool:
    text = " ".join(args)
    return any(item in text for item in SHELL_METACHARS)


def _is_safe_command(args: list[str]) -> bool:
    lowered = tuple(arg.lower() for arg in args)
    if lowered in SAFE_EXACT_COMMANDS:
        return True
    return any(lowered[: len(prefix)] == prefix for prefix in SAFE_PREFIXES)

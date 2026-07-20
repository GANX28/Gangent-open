"""ToolRegistry v1.

ToolRegistry（工具注册表）集中管理工具名称、来源、风险等级、
参数 schema 和执行入口。第一版只注册本地工具；后续 MCP 工具
可以转换成同样的 ToolDefinition 后加入这里。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from .models import ActionDecision
from .rag import search_context
from .schema_validator import validate_tool_arguments, validate_tool_schema_definition
from .tool_schema import tool_schema_for_name


class ToolSource(str, Enum):
    """工具来源。"""

    LOCAL = "local"
    MCP = "mcp"


class ToolRisk(str, Enum):
    """工具风险等级。"""

    READ = "read"
    WRITE = "write"
    COMMAND = "command"
    EXTERNAL = "external"


def _default_read_only(risk: ToolRisk) -> bool:
    return risk in {ToolRisk.READ}


ToolHandler = Callable[[dict[str, Any], str], str]


@dataclass(frozen=True)
class ToolDefinition:
    """注册表中的单个工具定义。

    input_schema 保留给模型工具 schema / MCP schema 适配使用；
    handler 是本地执行入口，MCP 工具未来可以使用代理 handler。
    """

    name: str
    description: str
    risk: ToolRisk
    source: ToolSource
    handler: ToolHandler
    input_schema: dict[str, Any] | None = None
    read_only: bool | None = None
    snip_hint: str = ""

    def is_read_only(self) -> bool:
        return _default_read_only(self.risk) if self.read_only is None else self.read_only


class ToolRegistry:
    """工具发现与分发层。"""

    def __init__(self, definitions: list[ToolDefinition] | None = None) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        for definition in definitions or []:
            self.register(definition)

    def register(self, definition: ToolDefinition) -> None:
        if definition.name in self._tools:
            raise ValueError(f"Tool already registered: {definition.name}")
        validate_tool_schema_definition(definition.name, definition.input_schema)
        self._tools[definition.name] = definition

    def get(self, name: str) -> ToolDefinition:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {name}") from exc

    def names(self) -> set[str]:
        return set(self._tools)

    def definitions(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    def validate_arguments(self, decision: ActionDecision) -> None:
        if not decision.tool_name:
            raise KeyError("Tool name is missing.")
        if decision.tool_args is None:
            raise ValueError("Tool args are missing.")
        definition = self.get(decision.tool_name)
        validate_tool_arguments(
            tool_name=definition.name,
            input_schema=definition.input_schema,
            arguments=decision.tool_args,
        )

    def dispatch(self, decision: ActionDecision, workspace_root: str) -> str:
        self.validate_arguments(decision)
        assert decision.tool_name is not None
        assert decision.tool_args is not None
        definition = self.get(decision.tool_name)
        return definition.handler(decision.tool_args, workspace_root)


def default_tool_registry() -> ToolRegistry:
    """注册第一版内置本地工具。"""

    from .tool_runtime import (
        compile_python,
        edit_file,
        ensure_workspace,
        export_artifact,
        fetch_url,
        file_info,
        grep_files,
        apply_patch,
        git_add,
        git_commit,
        git_diff,
        git_log,
        git_show,
        git_status,
        list_files,
        memory_add,
        read_file,
        read_many_files,
        run_command,
        run_tests,
        scratchpad_note,
        write_file,
    )
    def local_tool(
        name: str,
        description: str,
        risk: ToolRisk,
        handler: ToolHandler,
        snip_hint: str = "",
    ) -> ToolDefinition:
        return ToolDefinition(
            name=name,
            description=description,
            risk=risk,
            source=ToolSource.LOCAL,
            handler=handler,
            input_schema=tool_schema_for_name(name),
            read_only=_default_read_only(risk),
            snip_hint=snip_hint,
        )

    return ToolRegistry(
        [
            local_tool(
                name="list_files",
                description="List files under a workspace directory.",
                risk=ToolRisk.READ,
                handler=lambda args, root: list_files(str(args.get("path", ".")), root),
                snip_hint="Keep path type, names, and truncation count.",
            ),
            local_tool(
                name="file_info",
                description="Inspect path metadata before reading or editing; not content evidence.",
                risk=ToolRisk.READ,
                handler=lambda args, root: file_info(str(args.get("path", ".")), root),
                snip_hint="Keep path type, size, binary flag, and line count.",
            ),
            local_tool(
                name="read_file",
                description="Read UTF-8 file content from the workspace.",
                risk=ToolRisk.READ,
                handler=lambda args, root: read_file(
                    str(args.get("path", "")),
                    root,
                    start_line=int(args["start_line"]) if args.get("start_line") is not None else None,
                    max_lines=int(args["max_lines"]) if args.get("max_lines") is not None else None,
                ),
                snip_hint="Keep line-numbered focused chunks; re-read exact lines when needed.",
            ),
            local_tool(
                name="read_many_files",
                description="Read several small UTF-8 text files from the workspace as content evidence.",
                risk=ToolRisk.READ,
                handler=lambda args, root: read_many_files(list(args.get("paths", [])), root),
            ),
            local_tool(
                name="write_file",
                description="Write a UTF-8 text file inside the workspace.",
                risk=ToolRisk.WRITE,
                handler=lambda args, root: write_file(
                    str(args.get("path", "")),
                    str(args.get("content", "")),
                    root,
                    overwrite=bool(args.get("overwrite", False)),
                ),
            ),
            local_tool(
                name="edit_file",
                description="Edit a UTF-8 text file by exact replacement.",
                risk=ToolRisk.WRITE,
                handler=lambda args, root: edit_file(
                    str(args.get("path", "")),
                    str(args.get("old_text", "")),
                    str(args.get("new_text", "")),
                    root,
                ),
            ),
            local_tool(
                name="apply_patch",
                description="Apply a restricted Add File / Update File patch.",
                risk=ToolRisk.WRITE,
                handler=lambda args, root: apply_patch(str(args.get("patch", "")), root),
            ),
            local_tool(
                name="grep_files",
                description="Search workspace text files with a bounded pattern.",
                risk=ToolRisk.READ,
                handler=lambda args, root: grep_files(
                    pattern=str(args.get("pattern", "")),
                    path=str(args.get("path", ".")),
                    workspace_root=root,
                    max_results=int(args.get("max_results", 50)),
                ),
            ),
            local_tool(
                name="export_artifact",
                description="Write a generated artifact under artifacts/.",
                risk=ToolRisk.WRITE,
                handler=lambda args, root: export_artifact(
                    name=str(args.get("name", "")),
                    content=str(args.get("content", "")),
                    workspace_root=root,
                    overwrite=bool(args.get("overwrite", False)),
                ),
            ),
            local_tool(
                name="scratchpad_note",
                description="Append one note to the task scratchpad.",
                risk=ToolRisk.WRITE,
                handler=lambda args, root: scratchpad_note(str(args.get("note", "")), root),
            ),
            local_tool(
                name="memory_add",
                description="Store one non-secret node in the local memory graph.",
                risk=ToolRisk.WRITE,
                handler=lambda args, root: memory_add(
                    node_type=str(args.get("node_type", "")),
                    content=str(args.get("content", "")),
                    summary=str(args.get("summary", "")),
                    project_scope=str(args.get("project_scope", "")),
                    source=str(args.get("source", "runtime")),
                    tags=list(args.get("tags", [])),
                    importance=float(args.get("importance", 0.5)),
                    confidence=float(args.get("confidence", 0.8)),
                    layer=str(args.get("layer", "")),
                    workspace_root=root,
                ),
            ),
            local_tool(
                name="git_status",
                description="Run read-only git status --short.",
                risk=ToolRisk.READ,
                handler=lambda args, root: git_status(root),
            ),
            local_tool(
                name="git_diff",
                description="Run read-only git diff.",
                risk=ToolRisk.READ,
                handler=lambda args, root: git_diff(root),
            ),
            local_tool(
                name="git_log",
                description="Run read-only git log.",
                risk=ToolRisk.READ,
                handler=lambda args, root: git_log(root, limit=int(args.get("limit", 5))),
            ),
            local_tool(
                name="git_show",
                description="Run read-only git show for one revision.",
                risk=ToolRisk.READ,
                handler=lambda args, root: git_show(str(args.get("revision", "HEAD")), root),
            ),
            local_tool(
                name="git_add",
                description="Stage specific workspace files with git add --.",
                risk=ToolRisk.WRITE,
                handler=lambda args, root: git_add(list(args.get("paths", [])), root),
            ),
            local_tool(
                name="git_commit",
                description="Create a local git commit with a plain-text message.",
                risk=ToolRisk.WRITE,
                handler=lambda args, root: git_commit(str(args.get("message", "")), root),
            ),
            local_tool(
                name="compile_python",
                description="Check Python source syntax without executing project code.",
                risk=ToolRisk.COMMAND,
                handler=lambda args, root: compile_python(root),
            ),
            local_tool(
                name="run_tests",
                description="Run the workspace unittest suite through a constrained command.",
                risk=ToolRisk.COMMAND,
                handler=lambda args, root: run_tests(root),
            ),
            local_tool(
                name="run_command",
                description="Run a structured local development command through policy and SandboxRunner.",
                risk=ToolRisk.COMMAND,
                handler=lambda args, root: run_command(
                    args=list(args.get("args", [])),
                    cwd=str(args.get("cwd", ".")),
                    timeout_seconds=int(args.get("timeout_seconds", 30)),
                    workspace_root=root,
                ),
            ),
            local_tool(
                name="search_context",
                description="Search relevant workspace context with access filtering.",
                risk=ToolRisk.READ,
                handler=lambda args, root: search_context(
                    query=str(args.get("query", "")),
                    top_k=int(args.get("top_k", 5)),
                    workspace_root=root,
                ),
            ),
            local_tool(
                name="fetch_url",
                description="Fetch a public HTTP/HTTPS URL as bounded text.",
                risk=ToolRisk.EXTERNAL,
                handler=lambda args, root: fetch_url(
                    url=str(args.get("url", "")),
                    workspace_root=root,
                    max_bytes=int(args.get("max_bytes", 60_000)),
                ),
            ),
            local_tool(
                name="ensure_workspace",
                description="Create a normal workspace directory and README.md.",
                risk=ToolRisk.WRITE,
                handler=lambda args, root: ensure_workspace(str(args.get("path", "workspace")), root),
            ),
        ]
    )

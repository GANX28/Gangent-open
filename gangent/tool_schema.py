"""定义模型可见的工具 Schema。

JSON Schema（JSON 模式）用来描述工具参数长什么样。
function calling / tool calling（函数调用/工具调用）用这些 Schema 让模型输出
结构化的工具调用意图，而不是随便写一段自然语言。
"""

from __future__ import annotations

from typing import Any


def read_file_tool_schema() -> dict[str, Any]:
    """读取文件工具的 Schema。

    第一版只让模型表达“我想读哪个文件”，真正执行还要经过后续策略层和工具层。
    """

    return {
        "type": "function",
        "name": "read_file",
        "description": (
            "Read UTF-8 text content from a local workspace file. Use this when the final answer "
            "depends on file contents; file_info is metadata only and is not evidence for content questions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative or absolute file path to read.",
                },
                "start_line": {
                    "type": "integer",
                    "description": "Optional 1-based line number to start reading from for large files.",
                },
                "max_lines": {
                    "type": "integer",
                    "description": "Optional maximum number of lines to return from start_line.",
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def read_many_files_tool_schema() -> dict[str, Any]:
    """Read multiple small UTF-8 files in one tool call."""

    return {
        "type": "function",
        "name": "read_many_files",
        "description": (
            "Read several small UTF-8 text files from the local workspace in one call. "
            "Use this when the answer depends on file contents from multiple known files."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "paths": {
                    "type": "array",
                    "description": "Workspace-relative file paths to read, between 1 and 8 files.",
                    "items": {"type": "string"},
                }
            },
            "required": ["paths"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def list_files_tool_schema() -> dict[str, Any]:
    """列出文件工具的 Schema。

    这个工具让模型先观察项目结构，再决定下一步读哪个文件。
    """

    return {
        "type": "function",
        "name": "list_files",
        "description": "List files under a workspace directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to inspect.",
                }
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def file_info_tool_schema() -> dict[str, Any]:
    """Return metadata for a path before choosing read_file or list_files."""

    return {
        "type": "function",
        "name": "file_info",
        "description": (
            "Inspect path metadata only: file or directory, size, line count, and binary status. "
            "Do not use file_info to answer content questions; call read_file for content evidence."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative or absolute path to inspect.",
                }
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def grep_files_tool_schema() -> dict[str, Any]:
    """Controlled grep-style text search tool schema."""

    return {
        "type": "function",
        "name": "grep_files",
        "description": "Search workspace text files with a bounded regular expression query.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regular expression or plain text pattern to search for.",
                },
                "path": {
                    "type": "string",
                    "description": "Workspace-relative directory or file to search from. Defaults to '.'.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum matches to return, between 1 and 100.",
                },
            },
            "required": ["pattern", "path", "max_results"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def finish_task_tool_schema() -> dict[str, Any]:
    """完成任务的结构化决策 Schema。

    这个不是普通外部工具，而是给模型用的完成信号。
    模型信息足够时调用 finish_task，runtime 会把它解析成 DecisionType.FINISH。
    """

    return {
        "type": "function",
        "name": "finish_task",
        "description": "Finish the current task and provide the final answer.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "Final answer to the user's task.",
                },
                "reason": {
                    "type": "string",
                    "description": "Optional brief reason why the task can be finished now.",
                },
            },
            "required": ["answer"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def export_artifact_tool_schema() -> dict[str, Any]:
    """Export a generated artifact under the workspace artifacts directory."""

    return {
        "type": "function",
        "name": "export_artifact",
        "description": "Write a generated artifact under artifacts/ with a safe file name.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Safe artifact file name, for example report.md or trace.json.",
                },
                "content": {
                    "type": "string",
                    "description": "Artifact content.",
                },
                "overwrite": {
                    "type": "boolean",
                    "description": "Whether to overwrite an existing artifact.",
                },
            },
            "required": ["name", "content", "overwrite"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def scratchpad_note_tool_schema() -> dict[str, Any]:
    """Append one note to the task scratchpad."""

    return {
        "type": "function",
        "name": "scratchpad_note",
        "description": "Append a concise internal task note to .gangent/scratchpad/latest.md.",
        "parameters": {
            "type": "object",
            "properties": {
                "note": {
                    "type": "string",
                    "description": "A concise note about current assumptions, findings, or next steps.",
                }
            },
            "required": ["note"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def write_file_tool_schema() -> dict[str, Any]:
    """写入文件工具的 Schema。

    这是受控写入，不是裸 shell。真正写入前仍然经过 policy 和 permission。
    """

    return {
        "type": "function",
        "name": "write_file",
        "description": "Write a UTF-8 text file inside the local workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative or absolute file path to write.",
                },
                "content": {
                    "type": "string",
                    "description": "UTF-8 text content to write.",
                },
                "overwrite": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether to overwrite an existing file.",
                },
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def edit_file_tool_schema() -> dict[str, Any]:
    """精确替换编辑工具的 Schema。

    通过 old_text/new_text 做精确替换，比让模型直接输出整文件更安全。
    """

    return {
        "type": "function",
        "name": "edit_file",
        "description": "Edit a UTF-8 text file by replacing one exact text block.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative or absolute file path to edit.",
                },
                "old_text": {
                    "type": "string",
                    "description": "Exact existing text to replace. Must match once.",
                },
                "new_text": {
                    "type": "string",
                    "description": "Replacement text.",
                },
            },
            "required": ["path", "old_text", "new_text"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def apply_patch_tool_schema() -> dict[str, Any]:
    """Patch editor tool schema."""

    return {
        "type": "function",
        "name": "apply_patch",
        "description": "Apply a restricted text patch with Add File or Update File operations. Delete File is not supported.",
        "parameters": {
            "type": "object",
            "properties": {
                "patch": {
                    "type": "string",
                    "description": "Patch text starting with *** Begin Patch and ending with *** End Patch.",
                }
            },
            "required": ["patch"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def git_status_tool_schema() -> dict[str, Any]:
    """只读 git status 工具的 Schema。"""

    return {
        "type": "function",
        "name": "git_status",
        "description": "Run read-only git status --short in the workspace.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    }


def git_diff_tool_schema() -> dict[str, Any]:
    """只读 git diff 工具的 Schema。"""

    return {
        "type": "function",
        "name": "git_diff",
        "description": "Run read-only git diff in the workspace.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    }


def git_log_tool_schema() -> dict[str, Any]:
    """Read-only git log tool schema."""

    return {
        "type": "function",
        "name": "git_log",
        "description": "Run read-only git log in the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of commits to show, between 1 and 20.",
                }
            },
            "required": ["limit"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def git_show_tool_schema() -> dict[str, Any]:
    """Read-only git show tool schema."""

    return {
        "type": "function",
        "name": "git_show",
        "description": "Run read-only git show for one revision in the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "revision": {
                    "type": "string",
                    "description": "Commit-ish to inspect, for example HEAD or HEAD~1.",
                }
            },
            "required": ["revision"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def git_add_tool_schema() -> dict[str, Any]:
    """Controlled git add tool schema."""

    return {
        "type": "function",
        "name": "git_add",
        "description": "Stage one or more workspace-relative file paths with git add --.",
        "parameters": {
            "type": "object",
            "properties": {
                "paths": {
                    "type": "array",
                    "description": "Workspace-relative file paths to stage.",
                    "items": {"type": "string"},
                }
            },
            "required": ["paths"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def git_commit_tool_schema() -> dict[str, Any]:
    """Controlled git commit tool schema."""

    return {
        "type": "function",
        "name": "git_commit",
        "description": "Create a local git commit with a plain-text commit message.",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Commit message in plain text.",
                }
            },
            "required": ["message"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def compile_python_tool_schema() -> dict[str, Any]:
    """Python 语法检查工具的 Schema。"""

    return {
        "type": "function",
        "name": "compile_python",
        "description": "Check Python source syntax in the workspace without executing project code.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    }


def run_tests_tool_schema() -> dict[str, Any]:
    """受控测试命令工具的 Schema。

    这个工具固定运行 unittest，不接收任意 shell 命令。
    """

    return {
        "type": "function",
        "name": "run_tests",
        "description": "Run the workspace unittest suite through a constrained command.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    }


def run_command_tool_schema() -> dict[str, Any]:
    """Structured command tool schema.

    The model must provide argv as a list. Raw shell text is intentionally
    unsupported because policy.py needs to classify the executable and args.
    """

    return {
        "type": "function",
        "name": "run_command",
        "description": "Run a structured local development command through policy and SandboxRunner. Provide args as argv list, not raw shell text.",
        "parameters": {
            "type": "object",
            "properties": {
                "args": {
                    "type": "array",
                    "description": "Command argv list, for example ['python', '--version'] or ['pytest', '-q'].",
                    "items": {"type": "string"},
                },
                "cwd": {
                    "type": "string",
                    "description": "Workspace-relative directory to run in. Defaults to '.'.",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Command timeout between 1 and 120 seconds.",
                },
            },
            "required": ["args", "cwd", "timeout_seconds"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def search_context_tool_schema() -> dict[str, Any]:
    """RAG 上下文检索工具的 Schema。

    这个工具只检索当前 workspace 内允许读取的文本片段，并返回带来源的上下文。
    """

    return {
        "type": "function",
        "name": "search_context",
        "description": "Search relevant workspace context with access filtering, chunking, ranking, citations, and secret redaction.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query describing the needed project context.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of context chunks to return, between 1 and 8.",
                },
            },
            "required": ["query", "top_k"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def fetch_url_tool_schema() -> dict[str, Any]:
    """Fetch a public HTTP/HTTPS URL with strict network safety limits."""

    return {
        "type": "function",
        "name": "fetch_url",
        "description": "Fetch a public HTTP/HTTPS URL as text. Private networks, localhost, and large responses are blocked.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Public http:// or https:// URL to fetch.",
                },
                "max_bytes": {
                    "type": "integer",
                    "description": "Maximum response bytes to read, between 1,000 and 200,000.",
                },
            },
            "required": ["url", "max_bytes"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def ensure_workspace_tool_schema() -> dict[str, Any]:
    """创建工作目录工具的 Schema。"""

    return {
        "type": "function",
        "name": "ensure_workspace",
        "description": "Create a normal workspace directory and README.md inside the workspace root.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative directory path to create, for example workspace.",
                }
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def memory_add_tool_schema() -> dict[str, Any]:
    """Store one typed memory node in the local memory graph."""

    return {
        "type": "function",
        "name": "memory_add",
        "description": "Store one non-secret memory node in the local memory graph for future dynamic context loading.",
        "parameters": {
            "type": "object",
            "properties": {
                "node_type": {
                    "type": "string",
                    "description": "Memory node type such as fact, decision, issue, solution, concept, constraint, preference, note, artifact, or task_state.",
                },
                "content": {
                    "type": "string",
                    "description": "Memory content to store. Do not include secrets or credentials.",
                },
                "summary": {
                    "type": "string",
                    "description": "Optional compact summary.",
                },
                "project_scope": {
                    "type": "string",
                    "description": "Project or task scope such as gangent, resume, or policy-demo.",
                },
                "source": {
                    "type": "string",
                    "description": "Source label for the memory.",
                },
                "tags": {
                    "type": "array",
                    "description": "Small tag list.",
                    "items": {"type": "string"},
                },
                "importance": {
                    "type": "number",
                    "description": "Importance from 0 to 1.",
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence from 0 to 1.",
                },
                "layer": {
                    "type": "string",
                    "description": "Optional layer: data, task, or knowledge.",
                },
            },
            "required": ["node_type", "content", "summary", "project_scope", "source", "tags", "importance", "confidence", "layer"],
            "additionalProperties": False,
        },
        "strict": True,
    }


TOOL_SCHEMA_BUILDERS = {
    "list_files": list_files_tool_schema,
    "file_info": file_info_tool_schema,
    "read_file": read_file_tool_schema,
    "read_many_files": read_many_files_tool_schema,
    "write_file": write_file_tool_schema,
    "edit_file": edit_file_tool_schema,
    "apply_patch": apply_patch_tool_schema,
    "grep_files": grep_files_tool_schema,
    "export_artifact": export_artifact_tool_schema,
    "scratchpad_note": scratchpad_note_tool_schema,
    "git_status": git_status_tool_schema,
    "git_diff": git_diff_tool_schema,
    "git_log": git_log_tool_schema,
    "git_show": git_show_tool_schema,
    "git_add": git_add_tool_schema,
    "git_commit": git_commit_tool_schema,
    "compile_python": compile_python_tool_schema,
    "run_tests": run_tests_tool_schema,
    "run_command": run_command_tool_schema,
    "search_context": search_context_tool_schema,
    "fetch_url": fetch_url_tool_schema,
    "ensure_workspace": ensure_workspace_tool_schema,
    "memory_add": memory_add_tool_schema,
    "finish_task": finish_task_tool_schema,
}

VISIBLE_TOOL_ORDER = (
    "list_files",
    "file_info",
    "read_file",
    "read_many_files",
    "write_file",
    "edit_file",
    "apply_patch",
    "grep_files",
    "export_artifact",
    "scratchpad_note",
    "git_status",
    "git_diff",
    "git_log",
    "git_show",
    "git_add",
    "git_commit",
    "compile_python",
    "run_tests",
    "run_command",
    "search_context",
    "fetch_url",
    "ensure_workspace",
    "memory_add",
    "finish_task",
)


def tool_schema_for_name(name: str) -> dict[str, Any]:
    """Return the model-facing schema for one registered tool name."""

    try:
        return TOOL_SCHEMA_BUILDERS[name]()
    except KeyError as exc:
        raise KeyError(f"Unknown tool schema: {name}") from exc


def available_tool_schemas(names: tuple[str, ...] | list[str] | None = None) -> list[dict[str, Any]]:
    """返回 planning layer 暴露给模型的工具列表。

    names 用于轻任务工具裁剪：工具越少，模型越不容易选错工具，
    DeepSeek 每次请求携带的 schema token 也越少。
    """

    if names is None:
        return [tool_schema_for_name(name) for name in VISIBLE_TOOL_ORDER]
    seen: set[str] = set()
    selected: list[dict[str, Any]] = []
    for name in names:
        if name in seen:
            continue
        selected.append(tool_schema_for_name(name))
        seen.add(name)
    return selected


def tool_names(tools: list[dict[str, Any]]) -> set[str]:
    """提取工具名，供决策校验使用。"""

    return {tool["name"] for tool in tools if "name" in tool}


def to_deepseek_tools(
    tools: list[dict[str, Any]],
    include_strict: bool = False,
) -> list[dict[str, Any]]:
    """把内部工具 Schema 转成 DeepSeek Chat Completions 需要的格式。

    我们内部使用扁平格式：
    {"type": "function", "name": "...", "parameters": {...}}

    DeepSeek 使用 OpenAI Chat Completions 兼容格式：
    {"type": "function", "function": {"name": "...", "parameters": {...}}}

    注意：DeepSeek 的 strict 模式是 Beta 功能，需要 beta base_url。
    所以普通接口默认不传 strict，避免真实调用时因为参数不匹配失败。

    单独做转换的原因是：runtime 内部结构不应该被某一家模型厂商的 API 格式绑死。
    """

    deepseek_tools: list[dict[str, Any]] = []
    for tool in tools:
        function_schema = {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters", {"type": "object"}),
        }
        if include_strict:
            function_schema["strict"] = bool(tool.get("strict", False))
        deepseek_tools.append(
            {
                "type": "function",
                "function": function_schema,
            }
        )
    return deepseek_tools

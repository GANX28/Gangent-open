"""Minimal browser shell for Gangent.

This is intentionally small: one stdlib HTTP server, one background runtime
thread, and polling endpoints. It exists to prove the runtime can receive new
inputs while a task is still running.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
from time import time
import traceback
from typing import Any
import webbrowser

from .adaptive_runtime import apply_budget_recommendation, resolve_budget, run_task_adaptive
from .budget_stats import default_budget_history_path, recommend_budget
from .checkpoint import default_checkpoint_path, load_checkpoint
from .events import AgentEventType, JsonlEventQueue, default_event_log_path
from .hooks import HookContext, HookEvent, HookManager
from .memory_recorder import record_runtime_result_memory
from .models import TaskInput
from .output import final_answer_from_result
from .providers import create_llm_client
from .runtime import RuntimeResult


@dataclass
class ShellConfig:
    workspace_root: str
    provider: str = "deepseek"
    model: str | None = None
    thinking: bool = False
    profile: str = "auto"
    max_steps: int | None = None
    max_tokens: int | None = None
    max_seconds: float | None = None
    checkpoint_file: str | None = None
    event_log: str | None = None


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class WebShellState:
    config: ShellConfig
    lock: threading.Lock = field(default_factory=threading.Lock)
    running: bool = False
    current_task_id: str = ""
    current_goal: str = ""
    last_error: str = ""
    messages: list[ChatMessage] = field(default_factory=list)
    activity: str = ""
    activity_kind: str = ""
    activity_updated_at: float = 0.0
    last_result: RuntimeResult | None = None
    worker: threading.Thread | None = None

    @property
    def checkpoint_path(self) -> str:
        return self.config.checkpoint_file or str(default_checkpoint_path(self.config.workspace_root))

    @property
    def event_log_path(self) -> str:
        return self.config.event_log or str(default_event_log_path(self.config.workspace_root))


def run_web_shell(
    config: ShellConfig,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = False,
) -> None:
    """Start the lightweight browser shell."""

    state = WebShellState(config=config)
    handler_cls = _handler_factory(state)
    server = ThreadingHTTPServer((host, port), handler_cls)
    url = f"http://{host}:{port}"
    print(f"Gangent web shell: {url}")
    print(f"workspace_root={config.workspace_root}")
    print(f"event_log={state.event_log_path}")
    print(f"checkpoint_file={state.checkpoint_path}")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        server.server_close()


def start_background_task(state: WebShellState, message: str) -> None:
    """Start one Gangent task in a background thread."""

    clean = message.strip()
    if not clean:
        raise ValueError("message must not be empty")
    with state.lock:
        if state.running:
            raise RuntimeError("task is already running; send an event instead")
        state.running = True
        state.current_goal = clean
        state.current_task_id = ""
        state.last_error = ""
        state.last_result = None
        state.messages.append(ChatMessage("user", clean))
    worker = threading.Thread(target=_run_task_worker, args=(state, clean), daemon=True)
    with state.lock:
        state.worker = worker
    worker.start()


def enqueue_user_event(state: WebShellState, content: str, event_type: AgentEventType, priority: int = 80) -> dict[str, Any]:
    """Append a runtime event from the browser shell."""

    event = JsonlEventQueue(state.event_log_path).append(event_type, content, source="web_shell", priority=priority)
    with state.lock:
        if event_type == AgentEventType.USER_INPUT:
            state.messages.append(ChatMessage("user", event.content))
        else:
            state.messages.append(ChatMessage("event", f"{event.event_type.value}: {event.content}"))
    return {"event_id": event.event_id, "event_type": event.event_type.value, "priority": event.priority}


def snapshot(state: WebShellState) -> dict[str, Any]:
    """Return a browser-friendly status snapshot."""

    checkpoint_summary = _checkpoint_summary(state.checkpoint_path)
    events = _event_summaries(state.event_log_path)
    with state.lock:
        messages = [{"role": item.role, "content": item.content} for item in state.messages[-80:]]
        result = state.last_result
        usage = result.stats.usage if result is not None else {}
        activity = state.activity if state.running or time() - state.activity_updated_at < 2.5 else ""
        return {
            "running": state.running,
            "activity": activity,
            "activity_kind": state.activity_kind if activity else "",
            "current_task_id": state.current_task_id or checkpoint_summary.get("task_id", ""),
            "current_goal": state.current_goal or checkpoint_summary.get("goal", ""),
            "last_error": state.last_error,
            "messages": messages,
            "checkpoint": checkpoint_summary,
            "events": events[-30:],
            "usage": usage,
            "event_log": state.event_log_path,
            "checkpoint_file": state.checkpoint_path,
            "workspace_root": state.config.workspace_root,
        }


def _run_task_worker(state: WebShellState, message: str) -> None:
    task_input = TaskInput(goal=message, user_message=message, workspace_root=state.config.workspace_root)
    budget = resolve_budget(
        task_input,
        profile=state.config.profile,
        max_steps=state.config.max_steps,
        max_tokens=state.config.max_tokens,
        max_seconds=state.config.max_seconds,
    )
    if state.config.max_steps is None and state.config.max_tokens is None and state.config.max_seconds is None:
        budget = apply_budget_recommendation(
            budget,
            recommend_budget(task_input, default_budget_history_path(state.config.workspace_root)),
        )

    try:
        result = run_task_adaptive(
            task_input,
            lambda token_budget: create_llm_client(
                provider=state.config.provider,
                model=state.config.model,
                thinking=state.config.thinking,
                max_tokens=token_budget,
                budget_profile=budget.profile,
                task_text=f"{task_input.goal}\n{task_input.user_message}",
            ),
            budget=budget,
            checkpoint_path=state.checkpoint_path,
            event_queue_path=state.event_log_path,
            hook_manager=_web_shell_hook_manager(state),
        )
        record_runtime_result_memory(
            session_id="web_shell",
            user_message=message,
            result=result,
            workspace_root=state.config.workspace_root,
            provider=state.config.provider,
            model=state.config.model,
            thinking=state.config.thinking,
        )
        answer = final_answer_from_result(result) or f"Task ended with status={result.task.status.value}."
        summary = _format_run_summary(result)
        with state.lock:
            state.current_task_id = result.task.task_id
            state.last_result = result
            state.activity = ""
            state.activity_kind = ""
            state.activity_updated_at = time()
            state.messages.append(ChatMessage("assistant", f"{answer}\n\n{summary}".strip()))
            state.running = False
    except Exception as exc:
        with state.lock:
            state.activity = ""
            state.activity_kind = ""
            state.activity_updated_at = time()
            state.last_error = f"{exc}\n{traceback.format_exc(limit=8)}"
            state.messages.append(ChatMessage("assistant", f"Task failed: {exc}"))
            state.running = False


def _checkpoint_summary(path: str) -> dict[str, Any]:
    try:
        checkpoint = load_checkpoint(path)
    except Exception:
        return {}
    state = checkpoint.state
    current_step = ""
    for step in state.plan_steps:
        if step.status.value in {"todo", "running"}:
            current_step = step.title
            break
    return {
        "task_id": checkpoint.task.task_id,
        "goal": checkpoint.task.goal,
        "task_status": checkpoint.task.status.value,
        "phase": state.phase.value,
        "event_runtime_state": state.event_runtime_state,
        "step_index": state.step_index,
        "current_step": current_step,
        "event_cursor": state.event_cursor,
        "event_count": state.event_count,
        "replan_count": state.replan_count,
        "interrupt_count": state.interrupt_count,
        "pending_event_count": state.pending_event_count,
        "stabilization_required": state.stabilization_required,
        "stale_outputs": list(state.stale_outputs),
        "plan_patch_summaries": list(state.plan_patch_summaries[-8:]),
        "event_summaries": list(state.event_summaries[-8:]),
        "errors": list(state.errors[-8:]),
    }


def _event_summaries(path: str) -> list[dict[str, Any]]:
    events = []
    for queued in JsonlEventQueue(path).load():
        event = queued.event
        events.append(
            {
                "index": queued.index,
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "priority": event.priority,
                "source": event.source,
                "content": event.content,
                "created_at": event.created_at,
            }
        )
    return events


def _format_usage(usage: dict[str, Any]) -> str:
    if not usage:
        return "token_usage: prompt=0; completion=0; total=0"
    prompt = usage.get("prompt_tokens", usage.get("input_tokens", 0))
    completion = usage.get("completion_tokens", usage.get("output_tokens", 0))
    total = usage.get("total_tokens", 0)
    return f"token_usage: prompt={prompt}; completion={completion}; total={total}"


def _format_run_summary(result: RuntimeResult) -> str:
    usage = _format_usage(result.stats.usage)
    return (
        f"status={result.task.status.value}; steps={len(result.steps)}; "
        f"duration={result.stats.duration_seconds:.2f}s; "
        f"events={result.state.event_count}; replans={result.state.replan_count}; "
        f"interrupts={result.state.interrupt_count}\n"
        f"{usage}"
    )


def _web_shell_hook_manager(state: WebShellState) -> HookManager:
    manager = HookManager()

    def handle(context: HookContext) -> None:
        if context.event == HookEvent.TASK_START:
            _set_activity(state, "正在规划任务...", "thinking")
        elif context.event == HookEvent.BEFORE_MODEL_CALL:
            _set_activity(state, "正在思考...", "thinking")
        elif context.event == HookEvent.AFTER_MODEL_CALL:
            decision = getattr(context, "decision", None)
            _set_activity(state, _decision_activity(decision), "thinking")
        elif context.event == HookEvent.BEFORE_TOOL_CALL:
            _set_activity(state, _tool_activity(getattr(context, "decision", None)), "tool")
        elif context.event == HookEvent.AFTER_TOOL_CALL:
            result = getattr(context, "tool_result", None)
            status = "完成" if getattr(result, "success", False) else "工具返回错误"
            _set_activity(state, status, "tool")
        elif context.event == HookEvent.CHECKPOINT_SAVE:
            _set_activity(state, "正在保存 checkpoint...", "checkpoint")
        elif context.event == HookEvent.TASK_FINISH:
            _set_activity(state, "正在整理最终结果...", "finish")

    for event in HookEvent:
        manager.register(event, handle)
    return manager


def _set_activity(state: WebShellState, text: str, kind: str) -> None:
    with state.lock:
        state.activity = text
        state.activity_kind = kind
        state.activity_updated_at = time()


def _decision_activity(decision: Any) -> str:
    decision_type = getattr(getattr(decision, "decision_type", None), "value", "")
    if decision_type == "tool_call":
        return "已决定调用工具..."
    if decision_type == "finish":
        return "正在生成最终结果..."
    if decision_type == "ask_user":
        return "需要用户确认..."
    return "正在准备下一步..."


def _tool_activity(decision: Any) -> str:
    tool_name = getattr(decision, "tool_name", "") or "tool"
    args = getattr(decision, "tool_args", None) or {}
    path = str(args.get("path") or args.get("file") or "")
    if tool_name in {"read_file", "read_many_files", "file_info"} and path:
        return f"正在读取 {path}..."
    if tool_name in {"write_file", "edit_file", "apply_patch"} and path:
        return f"正在修改 {path}..."
    if tool_name == "run_command":
        command = args.get("args") or args.get("command") or ""
        return f"正在运行命令 {command}..."
    if tool_name.startswith("git_"):
        return f"正在执行 {tool_name}..."
    return f"正在调用 {tool_name}..."


def _handler_factory(state: WebShellState):
    class WebShellHandler(BaseHTTPRequestHandler):
        server_version = "GangentWebShell/0.1"

        def do_GET(self) -> None:
            if self.path in {"/", "/index.html"}:
                self._send_html(_INDEX_HTML)
                return
            if self.path == "/api/status":
                self._send_json(snapshot(state))
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            try:
                payload = self._read_json()
                if self.path == "/api/task":
                    start_background_task(state, str(payload.get("message", "")))
                    self._send_json({"ok": True})
                    return
                if self.path == "/api/event":
                    event_type = AgentEventType(str(payload.get("event_type", "user_input")))
                    priority = int(payload.get("priority", 80))
                    content = str(payload.get("content", ""))
                    self._send_json({"ok": True, "event": enqueue_user_event(state, content, event_type, priority)})
                    return
                if self.path == "/api/input":
                    content = str(payload.get("message", ""))
                    with state.lock:
                        running = state.running
                    if running:
                        event = enqueue_user_event(state, content, AgentEventType.USER_INPUT, 80)
                        self._send_json({"ok": True, "mode": "event", "event": event})
                    else:
                        start_background_task(state, content)
                        self._send_json({"ok": True, "mode": "task"})
                    return
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            data = self.rfile.read(length).decode("utf-8") if length else "{}"
            return json.loads(data or "{}")

        def _send_json(self, data: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, html: str) -> None:
            body = html.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return WebShellHandler


_INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Gangent</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f7f8;
      --panel: #ffffff;
      --line: #dedfe4;
      --text: #202123;
      --muted: #6b7280;
      --accent: #2563eb;
      --ok: #16803c;
      --bad: #b91c1c;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      height: 100vh;
      overflow: hidden;
      background: var(--bg);
      color: var(--text);
      font-family: "Segoe UI", Arial, sans-serif;
    }
    .app {
      height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr auto;
    }
    .topbar {
      height: 52px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 0 18px;
      border-bottom: 1px solid var(--line);
      background: rgba(255,255,255,.92);
    }
    .brand {
      display: flex;
      align-items: baseline;
      gap: 10px;
      min-width: 0;
    }
    .brand h1 {
      margin: 0;
      font-size: 15px;
      letter-spacing: 0;
    }
    .brand span {
      color: var(--muted);
      font-size: 12px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .badge {
      font-size: 12px;
      padding: 4px 9px;
      border-radius: 999px;
      border: 1px solid var(--line);
      color: var(--muted);
      background: #fff;
      white-space: nowrap;
    }
    .badge.running { color: var(--ok); border-color: #9ae6b4; background: #f0fff4; }
    .badge.error { color: var(--bad); border-color: #fecaca; background: #fef2f2; }
    details.runtime {
      position: fixed;
      top: 62px;
      right: 16px;
      z-index: 5;
      width: min(420px, calc(100vw - 32px));
      max-height: calc(100vh - 150px);
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      box-shadow: 0 12px 40px rgba(15, 23, 42, .12);
    }
    details.runtime:not([open]) {
      width: auto;
      overflow: visible;
      box-shadow: none;
    }
    details.runtime summary {
      cursor: pointer;
      padding: 8px 11px;
      font-size: 12px;
      color: var(--muted);
      list-style: none;
      user-select: none;
    }
    details.runtime summary::-webkit-details-marker { display: none; }
    .runtime-body {
      border-top: 1px solid var(--line);
      padding: 10px;
      display: grid;
      gap: 10px;
    }
    .section {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fff;
    }
    .section h3 {
      margin: 0 0 8px;
      font-size: 13px;
    }
    dl {
      margin: 0;
      display: grid;
      grid-template-columns: 112px 1fr;
      gap: 5px 8px;
      font-size: 12px;
    }
    dt { color: var(--muted); }
    dd { margin: 0; overflow-wrap: anywhere; }
    .quick {
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
    }
    button {
      border: 0;
      border-radius: 8px;
      background: var(--accent);
      color: #fff;
      padding: 0 16px;
      font: inherit;
      cursor: pointer;
      min-width: 86px;
    }
    button.secondary {
      background: #475569;
      min-width: 72px;
      padding: 7px 9px;
      font-size: 12px;
    }
    .event-list {
      display: grid;
      gap: 7px;
      max-height: 210px;
      overflow: auto;
    }
    .event {
      border-left: 3px solid var(--accent);
      padding: 6px 8px;
      background: #f8fafc;
      font-size: 12px;
      line-height: 1.45;
      overflow-wrap: anywhere;
    }
    .event small {
      color: var(--muted);
      display: block;
    }
    .chat {
      overflow: auto;
      padding: 26px 16px 18px;
    }
    .stream {
      width: min(820px, 100%);
      margin: 0 auto;
      display: grid;
      gap: 18px;
    }
    .empty {
      margin: 18vh auto 0;
      text-align: center;
      color: var(--muted);
    }
    .empty h2 {
      margin: 0 0 8px;
      color: var(--text);
      font-size: 24px;
      font-weight: 600;
      letter-spacing: 0;
    }
    .msg {
      display: grid;
      grid-template-columns: 78px 1fr;
      gap: 12px;
      align-items: start;
      line-height: 1.6;
      font-size: 14px;
    }
    .role {
      color: var(--muted);
      font-size: 12px;
      padding-top: 3px;
      text-align: right;
    }
    .bubble {
      background: transparent;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    .msg.user .bubble {
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
    }
    .msg.event .bubble {
      color: #92400e;
      font-size: 13px;
    }
    .activity {
      min-height: 30px;
      width: min(820px, calc(100% - 32px));
      margin: 0 auto;
      color: var(--muted);
      font-size: 13px;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .dot {
      width: 7px;
      height: 7px;
      border-radius: 999px;
      background: var(--accent);
      animation: pulse 1.2s ease-in-out infinite;
    }
    @keyframes pulse {
      0%, 100% { opacity: .25; transform: scale(.8); }
      50% { opacity: 1; transform: scale(1); }
    }
    .composer-wrap {
      border-top: 1px solid var(--line);
      background: rgba(247,247,248,.95);
      padding: 10px 16px 14px;
    }
    .composer {
      width: min(820px, 100%);
      margin: 0 auto;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      align-items: end;
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
    }
    textarea {
      resize: none;
      min-height: 48px;
      max-height: 180px;
      border: 0;
      padding: 7px 8px;
      outline: none;
      font: inherit;
      line-height: 1.45;
    }
    .hint {
      width: min(820px, 100%);
      margin: 7px auto 0;
      color: var(--muted);
      font-size: 12px;
    }
    @media (max-width: 720px) {
      .brand span { display: none; }
      .msg { grid-template-columns: 1fr; gap: 4px; }
      .role { text-align: left; }
    }
  </style>
</head>
<body>
  <div class="app">
    <header class="topbar">
      <div class="brand">
        <h1>Gangent</h1>
        <span id="workspaceLabel"></span>
      </div>
      <span id="runBadge" class="badge">idle</span>
    </header>

    <details id="runtimePanel" class="runtime">
      <summary id="runtimeSummary">Runtime</summary>
      <div class="runtime-body">
        <div class="section">
          <h3>Status</h3>
          <dl id="status"></dl>
        </div>
        <div class="section">
          <h3>Quick Events</h3>
          <div class="quick">
            <button class="secondary" data-event="replan_request">replan</button>
            <button class="secondary" data-event="user_interrupt">pause</button>
            <button class="secondary" data-event="new_file_added">new file</button>
          </div>
        </div>
        <div class="section">
          <h3>Events</h3>
          <div id="events" class="event-list"></div>
        </div>
        <div class="section">
          <h3>Plan Patches</h3>
          <div id="patches" class="event-list"></div>
        </div>
      </div>
    </details>

    <main id="chat" class="chat">
      <div id="stream" class="stream"></div>
    </main>

    <footer class="composer-wrap">
      <div id="activity" class="activity"></div>
      <form id="form" class="composer">
        <textarea id="input" placeholder="给 Gangent 一个任务；运行中继续输入会作为新要求进入事件队列。Enter 发送，Shift+Enter 换行。"></textarea>
        <button id="send" type="submit">发送</button>
      </form>
      <div id="hint" class="hint">空闲时启动新任务；运行中输入会在安全边界被处理。</div>
    </footer>
  </div>
  <script>
    const chat = document.getElementById("chat");
    const stream = document.getElementById("stream");
    const form = document.getElementById("form");
    const input = document.getElementById("input");
    const runBadge = document.getElementById("runBadge");
    const runtimeSummary = document.getElementById("runtimeSummary");
    const workspaceLabel = document.getElementById("workspaceLabel");
    const statusBox = document.getElementById("status");
    const eventsBox = document.getElementById("events");
    const patchesBox = document.getElementById("patches");
    const activityBox = document.getElementById("activity");
    const hint = document.getElementById("hint");

    async function post(url, body) {
      const res = await fetch(url, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(body)
      });
      return await res.json();
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const message = input.value.trim();
      if (!message) return;
      input.value = "";
      const result = await post("/api/input", {message});
      if (!result.ok) alert(result.error || "request failed");
      await refresh();
    });

    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        form.requestSubmit();
      }
    });

    document.querySelectorAll("[data-event]").forEach((button) => {
      button.addEventListener("click", async () => {
        const content = prompt("事件内容：");
        if (!content) return;
        const eventType = button.getAttribute("data-event");
        const priority = eventType === "user_interrupt" ? 90 : 80;
        const result = await post("/api/event", {event_type: eventType, priority, content});
        if (!result.ok) alert(result.error || "event failed");
        await refresh();
      });
    });

    function renderMessages(messages) {
      if (!messages.length) {
        stream.innerHTML = `<div class="empty"><h2>Gangent</h2><div>输入一个任务开始。运行中继续输入会进入事件队列。</div></div>`;
        return;
      }
      stream.innerHTML = "";
      for (const msg of messages) {
        const div = document.createElement("div");
        div.className = "msg " + msg.role;
        const role = document.createElement("div");
        role.className = "role";
        role.textContent = msg.role === "assistant" ? "Gangent" : (msg.role === "event" ? "Event" : "You");
        const bubble = document.createElement("div");
        bubble.className = "bubble";
        bubble.textContent = msg.content;
        div.appendChild(role);
        div.appendChild(bubble);
        stream.appendChild(div);
      }
      chat.scrollTop = chat.scrollHeight;
    }

    function renderActivity(data) {
      if (data.activity) {
        activityBox.innerHTML = `<span class="dot"></span><span>${escapeHtml(data.activity)}</span>`;
      } else {
        activityBox.innerHTML = "";
      }
    }

    function renderStatus(data) {
      runBadge.textContent = data.running ? "running" : (data.last_error ? "error" : "idle");
      runBadge.className = "badge " + (data.running ? "running" : (data.last_error ? "error" : ""));
      workspaceLabel.textContent = data.workspace_root || "";
      const c = data.checkpoint || {};
      runtimeSummary.textContent = `${data.events.length} events · ${c.task_status || (data.running ? "running" : "idle")}`;
      hint.textContent = data.running
        ? "任务运行中：继续输入会作为 user_input event，在下一次安全边界处理。"
        : "空闲：输入会启动一个新任务。";
      const rows = {
        task: data.current_task_id || "-",
        status: c.task_status || (data.running ? "running" : "idle"),
        phase: c.phase || "-",
        event_state: c.event_runtime_state || "-",
        current_step: c.current_step || "-",
        replans: c.replan_count ?? 0,
        interrupts: c.interrupt_count ?? 0,
        stabilization: c.stabilization_required ? "yes" : "no",
        token_total: data.usage?.total_tokens ?? 0
      };
      statusBox.innerHTML = Object.entries(rows).map(([k, v]) => `<dt>${k}</dt><dd>${escapeHtml(String(v))}</dd>`).join("");
    }

    function renderEvents(events) {
      eventsBox.innerHTML = events.slice().reverse().map((item) => `
        <div class="event">
          <small>#${item.index} ${escapeHtml(item.event_type)} p=${item.priority}</small>
          ${escapeHtml(item.content)}
        </div>
      `).join("") || "<div class='event'>No events yet.</div>";
    }

    function renderPatches(checkpoint) {
      const patches = checkpoint?.plan_patch_summaries || [];
      patchesBox.innerHTML = patches.slice().reverse().map((item) => `
        <div class="event">${escapeHtml(item)}</div>
      `).join("") || "<div class='event'>No plan patches yet.</div>";
    }

    function escapeHtml(text) {
      return text.replace(/[&<>"']/g, (ch) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
    }

    async function refresh() {
      const res = await fetch("/api/status");
      const data = await res.json();
      renderMessages(data.messages || []);
      renderActivity(data);
      renderStatus(data);
      renderEvents(data.events || []);
      renderPatches(data.checkpoint || {});
    }

    refresh();
    setInterval(refresh, 900);
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the minimal Gangent browser shell.")
    parser.add_argument("--workspace-root", default=str(Path.cwd()))
    parser.add_argument("--provider", choices=["fake", "openai", "deepseek"], default="deepseek")
    parser.add_argument("--model")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--profile", choices=["auto", "light", "medium", "heavy", "ultra"], default="auto")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--checkpoint-file")
    parser.add_argument("--event-log")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="Open the shell in the default browser.")
    args = parser.parse_args()
    run_web_shell(
        ShellConfig(
            workspace_root=args.workspace_root,
            provider=args.provider,
            model=args.model,
            thinking=args.thinking,
            profile=args.profile,
            max_steps=args.max_steps,
            max_tokens=args.max_tokens,
            max_seconds=args.max_seconds,
            checkpoint_file=args.checkpoint_file,
            event_log=args.event_log,
        ),
        host=args.host,
        port=args.port,
        open_browser=args.open,
    )


if __name__ == "__main__":
    main()

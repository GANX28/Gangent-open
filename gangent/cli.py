"""Interactive CLI for Gangent."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from .adaptive_runtime import apply_budget_recommendation, format_budget_summary, resolve_budget, run_task_adaptive
from .audit import append_audit_record, default_audit_path
from .budget_stats import (
    append_budget_sample,
    default_budget_history_path,
    recommend_budget,
    sample_from_result,
)
from .checkpoint import (
    CheckpointCandidate,
    checkpoint_archive_dir_for_active_path,
    checkpoint_matches_task_input,
    default_checkpoint_archive_dir,
    default_checkpoint_path,
    default_ignored_tasks_path,
    ignore_task_ids,
    is_task_ignored,
    list_resume_candidates,
    load_checkpoint,
    load_resume_candidate,
    shelve_active_checkpoint,
)
from .context_manager import build_context_bundle
from .events import AgentEventType, JsonlEventQueue, default_event_log_path
from .handoff import export_handoff_file
from .memory_extractor import extract_semantic_memory_chunks, should_use_llm_memory_extraction
from .memory_graph import record_task_result_memory
from .patch_editor import summarize_patch
from .output import final_answer_from_result, print_result
from .planner import attach_plan, create_initial_plan
from .planner_eval import (
    append_planner_evaluation,
    default_planner_eval_path,
    planner_evaluation_from_result,
    summarize_planner_evaluations,
)
from .providers import create_llm_client
from .runtime import RuntimeResult
from .session import build_task_input_from_session, reset_session, update_session_from_result
from .session_store import default_session_path, load_or_create_session, save_session
from .skills import default_skills_path, inject_skill_context, load_skills, resolve_skills
from .state import create_initial_state, create_task


def run_interactive_cli(
    provider: str,
    model: str | None,
    thinking: bool,
    max_steps: int | None,
    max_tokens: int | None,
    max_seconds: float | None,
    workspace_root: str,
    profile: str = "auto",
    resume: bool = False,
    session_file: str | None = None,
    output_mode: str = "verbose",
    audit_log: str | None = None,
    no_audit: bool = False,
    checkpoint_file: str | None = None,
    event_log: str | None = None,
) -> None:
    """Run the main interactive Gangent CLI loop."""

    configure_stdio_utf8()
    current_profile = profile
    print("Gangent interactive CLI")
    print(
        f"provider={provider}; model={model or '(default)'}; "
        f"profile={current_profile}; max_steps={max_steps or '(auto)'}; "
        f"max_tokens={max_tokens or '(auto)'}; max_seconds={max_seconds or '(auto)'}"
    )
    print(f"workspace_root={workspace_root}")
    print(f"session_file={session_file or default_session_path(workspace_root)}")
    print(f"skills_dir={default_skills_path(workspace_root)}")
    if not no_audit:
        print(f"audit_log={audit_log or default_audit_path(workspace_root)}")
    print(f"budget_history={default_budget_history_path(workspace_root)}")
    print(f"planner_eval={default_planner_eval_path(workspace_root)}")
    print(f"checkpoint_file={checkpoint_file or default_checkpoint_path(workspace_root)}")
    print(f"event_log={event_log or default_event_log_path(workspace_root)}")
    print(f"checkpoint_archive_dir={checkpoint_archive_dir_for_active_path(workspace_root, checkpoint_file)}")
    print(provider_doctor_summary(provider, model, thinking, current_profile, workspace_root))
    print(
        "Type 'exit' or 'quit' to leave. Type 'session', 'checkpoints', or 'audit' to inspect state. "
        "Type '/new' to start a new session, '/resume' to continue a task, '/delete' to hide resumable tasks, "
        "or '/budget show' to inspect/change the current budget profile.\n"
    )

    session = load_or_create_session(workspace_root=workspace_root, path=session_file, resume=True)
    if session.turns:
        print(f"已恢复会话 {session.session_id}，当前共有 {len(session.turns)} 轮记录。\n")

    checkpoint_path = checkpoint_file or str(default_checkpoint_path(workspace_root))
    skipped_active_resume = False
    startup_candidates = list_resume_candidates(workspace_root, checkpoint_path)
    startup_choice: CheckpointCandidate | None = None

    if resume and startup_candidates:
        startup_choice = startup_candidates[0]
    elif startup_candidates:
        startup_choice = prompt_startup_resume_choice(startup_candidates, workspace_root, checkpoint_path)
        skipped_active_resume = startup_choice is None and any(candidate.is_active for candidate in startup_candidates)

    if startup_choice is not None:
        try:
            resumed_result = _resume_candidate(
                startup_choice,
                provider=provider,
                model=model,
                thinking=thinking,
                profile=current_profile,
                max_steps=max_steps,
                max_tokens=max_tokens,
                max_seconds=max_seconds,
                checkpoint_path=checkpoint_path,
                approval_callback=prompt_for_approval,
                event_queue_path=event_log or str(default_event_log_path(workspace_root)),
            )
        except Exception as exc:
            print(f"自动恢复失败: {exc}")
        else:
            handoff_path = _finalize_task_result(
                session=session,
                user_message=startup_choice.checkpoint.task_input.user_message,
                result=resumed_result,
                workspace_root=workspace_root,
                session_file=session_file,
                audit_log=audit_log,
                no_audit=no_audit,
                output_mode=output_mode,
                trigger="task_resumed",
                provider=provider,
                model=model,
                thinking=thinking,
            )
            if output_mode != "json":
                resumed_budget = resolve_budget(
                    startup_choice.checkpoint.task_input,
                    profile=current_profile,
                    max_steps=max_steps,
                    max_tokens=max_tokens,
                    max_seconds=max_seconds,
                )
                print(f"adaptive_budget={format_budget_summary(resumed_budget)}")
            print_result(resumed_result, mode=output_mode)
            _print_handoff_path(handoff_path, output_mode)
            _print_token_usage(resumed_result, output_mode)

    while True:
        try:
            user_message = input("gangent> ").strip().lstrip("\ufeff")
        except (EOFError, KeyboardInterrupt):
            _export_session_handoff(
                workspace_root=workspace_root,
                session=session,
                session_file=session_file,
                checkpoint_path=checkpoint_path,
                audit_log=audit_log,
                output_mode=output_mode,
            )
            print("\nbye")
            return

        if not user_message:
            continue
        if user_message.lower() in {"exit", "quit"}:
            _export_session_handoff(
                workspace_root=workspace_root,
                session=session,
                session_file=session_file,
                checkpoint_path=checkpoint_path,
                audit_log=audit_log,
                output_mode=output_mode,
            )
            print("bye")
            return
        if user_message.lower() == "session":
            print_session(session)
            continue
        if user_message.lower() in {"checkpoint", "checkpoints"}:
            print_checkpoint_status(workspace_root, checkpoint_path)
            continue
        if user_message.lower() in {"audit", "audits"}:
            print_audit_summary(audit_log or default_audit_path(workspace_root))
            continue
        if _is_planner_command(user_message):
            print_planner_summary(default_planner_eval_path(workspace_root))
            continue
        if _is_context_command(user_message):
            print_context_report(session, workspace_root)
            continue
        if _is_events_command(user_message):
            print_event_summary(event_log or default_event_log_path(workspace_root))
            continue
        if _is_doctor_command(user_message):
            print_provider_doctor(provider, model, thinking, current_profile, workspace_root)
            continue
        if _is_budget_command(user_message):
            current_profile = handle_budget_command(user_message, current_profile)
            continue
        if user_message.lower().startswith(("/event ", "event ")):
            enqueue_cli_event(user_message, event_log or default_event_log_path(workspace_root))
            continue
        if user_message.lower().startswith(("/replan ", "replan ")):
            enqueue_short_event(user_message, AgentEventType.REPLAN_REQUEST, event_log or default_event_log_path(workspace_root))
            continue
        if user_message.lower().startswith(("/interrupt ", "interrupt ")):
            enqueue_short_event(user_message, AgentEventType.USER_INTERRUPT, event_log or default_event_log_path(workspace_root))
            continue
        if user_message.lower() in {"delete", "/delete"}:
            resume_candidates = list_resume_candidates(workspace_root, checkpoint_path)
            if not resume_candidates:
                print("当前没有可隐藏的未完成任务。\n")
                continue
            prompt_delete_resume_candidates(resume_candidates, workspace_root)
            continue
        if user_message.lower() == "/resume":
            resume_candidates = list_resume_candidates(workspace_root, checkpoint_path)
            if not resume_candidates:
                print("当前没有可恢复的未完成任务。\n")
                continue
            try:
                selected_candidate = prompt_resume_candidate_selection(
                    resume_candidates,
                    prompt_header="请选择要继续的任务",
                )
                if selected_candidate is None:
                    print("已取消恢复。\n")
                    continue
                result = _resume_candidate(
                    selected_candidate,
                    provider=provider,
                    model=model,
                    thinking=thinking,
                    profile=current_profile,
                    max_steps=max_steps,
                    max_tokens=max_tokens,
                    max_seconds=max_seconds,
                checkpoint_path=checkpoint_path,
                approval_callback=prompt_for_approval,
                event_queue_path=event_log or str(default_event_log_path(workspace_root)),
            )
            except Exception as exc:
                print(f"运行错误: {exc}")
                continue

            handoff_path = _finalize_task_result(
                session=session,
                user_message=selected_candidate.checkpoint.task_input.user_message,
                result=result,
                workspace_root=workspace_root,
                session_file=session_file,
                audit_log=audit_log,
                no_audit=no_audit,
                output_mode=output_mode,
                trigger="task_resumed",
                provider=provider,
                model=model,
                thinking=thinking,
            )
            if output_mode != "json":
                budget = resolve_budget(
                    selected_candidate.checkpoint.task_input,
                    profile=current_profile,
                    max_steps=max_steps,
                    max_tokens=max_tokens,
                    max_seconds=max_seconds,
                )
                print(f"adaptive_budget={format_budget_summary(budget)}")
            print_result(result, mode=output_mode)
            _print_handoff_path(handoff_path, output_mode)
            _print_token_usage(result, output_mode)
            continue
        if user_message.lower() in {"new", "/new", "reset"}:
            session = reset_session(session)
            save_session(session, session_file)
            shelved = shelve_active_checkpoint(checkpoint_path)
            skipped_active_resume = False
            print("已开始新的会话。\n")
            if shelved is not None:
                print(f"已将当前未完成任务移入归档断点：{shelved}\n")
            continue

        raw_task_input = build_task_input_from_session(session, user_message)
        loaded_skills = load_skills(default_skills_path(workspace_root))
        task_input = inject_skill_context(raw_task_input, resolve_skills(raw_task_input, loaded_skills))
        budget = resolve_budget(
            raw_task_input,
            profile=current_profile,
            max_steps=max_steps,
            max_tokens=max_tokens,
            max_seconds=max_seconds,
        )
        if max_steps is None and max_tokens is None and max_seconds is None:
            budget = apply_budget_recommendation(
                budget,
                recommend_budget(raw_task_input, default_budget_history_path(workspace_root)),
            )

        try:
            matched_resume = None
            resume_candidate = load_resume_candidate(checkpoint_path)
            if skipped_active_resume and resume_candidate is not None:
                shelve_active_checkpoint(checkpoint_path)
                resume_candidate = None
                skipped_active_resume = False
            if resume_candidate is not None and is_task_ignored(workspace_root, resume_candidate.task.task_id):
                resume_candidate = None
            if resume_candidate is not None and checkpoint_matches_task_input(resume_candidate, task_input):
                matched_resume = resume_candidate
            result = run_task_adaptive(
                task_input,
                lambda token_budget: create_llm_client(
                    provider=provider,
                    model=model,
                    thinking=thinking,
                    max_tokens=token_budget,
                    budget_profile=budget.profile,
                    task_text=f"{task_input.goal}\n{task_input.user_message}",
                ),
                budget=budget,
                approval_callback=prompt_for_approval,
                checkpoint_path=checkpoint_path,
                resume_checkpoint=matched_resume,
                event_queue_path=event_log or str(default_event_log_path(workspace_root)),
            )
        except Exception as exc:
            print(f"运行错误: {exc}")
            continue

        handoff_path = _finalize_task_result(
            session=session,
            user_message=user_message,
            result=result,
            workspace_root=workspace_root,
            session_file=session_file,
            audit_log=audit_log,
            no_audit=no_audit,
            output_mode=output_mode,
            trigger=f"task_{result.task.status.value}",
            provider=provider,
            model=model,
            thinking=thinking,
        )
        if output_mode != "json":
            print(f"adaptive_budget={format_budget_summary(budget)}")
        print_result(result, mode=output_mode)
        _print_handoff_path(handoff_path, output_mode)
        _print_token_usage(result, output_mode)
        skipped_active_resume = False


def print_runtime_result(result: RuntimeResult) -> None:
    """Print one runtime result in verbose mode."""

    print_result(result, mode="verbose")


def _is_planner_command(text: str) -> bool:
    return text.lower() in {"planner", "/planner", "planner stats", "/planner stats", "stats"}


def _is_context_command(text: str) -> bool:
    return text.lower() in {"context", "/context", "context report", "/context report"}


def _is_events_command(text: str) -> bool:
    return text.lower() in {"events", "/events", "event list", "/event list"}


def _is_doctor_command(text: str) -> bool:
    return text.lower() in {"doctor", "/doctor", "provider", "/provider", "provider check", "/provider check"}


VALID_BUDGET_PROFILES = {"auto", "light", "medium", "heavy", "ultra"}


def _is_budget_command(text: str) -> bool:
    lowered = text.lower().strip()
    return lowered == "/budget" or lowered.startswith("/budget ")


def handle_budget_command(text: str, current_profile: str) -> str:
    """Handle in-session budget profile switching without restarting the CLI."""

    parts = text.strip().split()
    usage = "usage: /budget show|auto|light|medium|heavy|ultra\n"
    if len(parts) == 1 or (len(parts) == 2 and parts[1].lower() == "show"):
        print(f"budget_profile={current_profile}\n")
        return current_profile
    if len(parts) != 2:
        print(usage)
        return current_profile

    requested = parts[1].lower()
    if requested not in VALID_BUDGET_PROFILES:
        print(usage)
        return current_profile
    if requested == current_profile:
        print(f"budget_profile unchanged: {current_profile}\n")
        return current_profile

    print(f"budget_profile changed: {current_profile} -> {requested}\n")
    return requested


def configure_stdio_utf8() -> None:
    """Best-effort UTF-8 console setup for Windows PowerShell and pipes."""

    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream_name in ("stdin", "stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            continue


def provider_doctor_summary(
    provider: str,
    model: str | None,
    thinking: bool,
    profile: str,
    workspace_root: str,
) -> str:
    """Return a one-line provider readiness summary without exposing secrets."""

    try:
        client = create_llm_client(
            provider=provider,
            model=model,
            thinking=thinking,
            budget_profile=profile,
            task_text="provider doctor",
        )
    except Exception as exc:
        return f"provider_check=error; provider={provider}; reason={exc}"
    resolved_model = getattr(client, "model", model or "(default)")
    if provider == "deepseek":
        key_status = "set" if os.environ.get("DEEPSEEK_API_KEY") else "missing"
        return (
            f"provider_check=real; provider=deepseek; model={resolved_model}; "
            f"api_key={key_status}; live_call=not_run"
        )
    if provider == "openai":
        key_status = "set" if os.environ.get("OPENAI_API_KEY") else "missing"
        return (
            f"provider_check=real; provider=openai; model={resolved_model}; "
            f"api_key={key_status}; live_call=not_run"
        )
    return f"provider_check=local_fake; provider={provider}; model={resolved_model}; live_call=not_applicable"


def print_provider_doctor(
    provider: str,
    model: str | None,
    thinking: bool,
    profile: str,
    workspace_root: str,
) -> None:
    print("\nDOCTOR")
    print(provider_doctor_summary(provider, model, thinking, profile, workspace_root))
    print(f"workspace_root={workspace_root}")
    print("note=doctor does not spend model tokens; run a tiny task for live provider usage.\n")


def print_session(session) -> None:
    """Print the current session summary."""

    print("\nSESSION")
    print(f"session_id={session.session_id}")
    print(f"turns={len(session.turns)}")
    if session.context_summary:
        print(session.context_summary)
    else:
        print("(empty)")
    print()


def print_checkpoint_status(workspace_root: str, checkpoint_path: str) -> None:
    """Print active and archived checkpoint status for the workspace."""

    active = Path(checkpoint_path)
    archive_dir = default_checkpoint_archive_dir(workspace_root)
    print("\nCHECKPOINTS")
    print(f"active={active}")
    if active.exists():
        try:
            checkpoint = load_checkpoint(active)
            print(
                f"active_task={checkpoint.task.task_id}; "
                f"status={checkpoint.task.status.value}; steps={len(checkpoint.steps)}"
            )
            if checkpoint.state.errors:
                print(f"last_error={checkpoint.state.errors[-1]}")
        except Exception as exc:
            print(f"active_error={exc}")
    else:
        print("active_status=(none)")

    print(f"archive_dir={archive_dir}")
    if archive_dir.exists():
        archives = sorted(archive_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
        for archive in archives[:5]:
            print(f"- {archive.name}")
        if len(archives) > 5:
            print(f"... {len(archives) - 5} more")
    else:
        print("archive_status=(none)")
    print(f"ignored_tasks={default_ignored_tasks_path(workspace_root)}")
    print()


def print_audit_summary(path: str | Path, limit: int = 5) -> None:
    """Print a compact summary of recent audit records."""

    target = Path(path)
    print("\nAUDIT")
    print(f"path={target}")
    if not target.exists():
        print("status=(none)\n")
        return
    lines = target.read_text(encoding="utf-8").splitlines()
    print(f"records={len(lines)}")
    for line in lines[-limit:]:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            print("- unreadable record")
            continue
        task = record.get("task", {})
        stats = record.get("stats", {})
        print(
            f"- task={task.get('task_id', '(unknown)')}; "
            f"status={task.get('status', '(unknown)')}; "
            f"steps={stats.get('step_count', 0)}"
        )
    print()


def print_planner_summary(path: str | Path) -> None:
    print("\nPLANNER")
    print(f"path={path}")
    print(summarize_planner_evaluations(path))
    print()


def print_context_report(session, workspace_root: str) -> None:
    print("\nCONTEXT")
    task_input = build_task_input_from_session(session, "context report")
    task = create_task_for_report(task_input)
    state = create_state_for_report(task, task_input)
    bundle = build_context_bundle(task, state)
    report = bundle.pollution_report
    print(f"segments={len(bundle.segments)}")
    if report is None:
        print("pollution_report=(none)\n")
        return
    print(f"chars={report.total_chars}")
    print("sources=" + ", ".join(f"{key}:{value}" for key, value in sorted(report.source_counts.items())))
    print("warnings=" + (", ".join(report.warnings) if report.warnings else "-"))
    print("sensitive_segments=" + (", ".join(report.sensitive_segments) if report.sensitive_segments else "-"))
    print("low_confidence_segments=" + (", ".join(report.low_confidence_segments) if report.low_confidence_segments else "-"))
    print(f"workspace_root={workspace_root}")
    print()


def create_task_for_report(task_input):
    return create_task(task_input)


def create_state_for_report(task, task_input):
    state = create_initial_state(task, task_input)
    attach_plan(state, create_initial_plan(task, task_input))
    return state


def print_event_summary(path: str | Path, limit: int = 8) -> None:
    queue = JsonlEventQueue(path)
    events = queue.load()
    print("\nEVENTS")
    print(f"path={path}")
    print(f"records={len(events)}")
    for queued in events[-limit:]:
        event = queued.event
        content = event.content.replace("\n", " ")
        if len(content) > 100:
            content = content[:100] + "..."
        print(f"- #{queued.index} type={event.event_type.value}; priority={event.priority}; task={event.task_id or '-'}; {content}")
    print()


def enqueue_cli_event(message: str, path: str | Path) -> None:
    parts = message.split(maxsplit=3)
    if len(parts) < 4:
        print("usage: /event <type> <priority> <content>\n")
        return
    _, event_type_text, priority_text, content = parts
    try:
        event_type = AgentEventType(event_type_text)
        priority = int(priority_text)
        event = JsonlEventQueue(path).append(event_type, content, priority=priority)
    except Exception as exc:
        print(f"event_error={exc}\n")
        return
    print(f"event_queued id={event.event_id}; type={event.event_type.value}; priority={event.priority}\n")


def enqueue_short_event(message: str, event_type: AgentEventType, path: str | Path) -> None:
    content = message.split(maxsplit=1)[1].strip() if " " in message else ""
    if not content:
        print(f"usage: /{event_type.value} <content>\n")
        return
    priority = 80 if event_type == AgentEventType.REPLAN_REQUEST else 90
    try:
        event = JsonlEventQueue(path).append(event_type, content, priority=priority)
    except Exception as exc:
        print(f"event_error={exc}\n")
        return
    print(f"event_queued id={event.event_id}; type={event.event_type.value}; priority={event.priority}\n")


def prompt_for_approval(decision, policy) -> bool:
    """Ask the user to approve one escalated action."""

    print()
    print(format_approval_request(decision, policy))
    answer = input("是否允许该操作？输入 'yes' 表示同意：").strip().lower()
    approved = answer in {"yes", "y", "是"}
    print("已批准\n" if approved else "已拒绝\n")
    return approved


def prompt_startup_resume_choice(
    candidates: list[CheckpointCandidate],
    workspace_root: str,
    checkpoint_path: str,
) -> CheckpointCandidate | None:
    """Ask whether the user wants to continue one unfinished task at startup."""

    if not candidates:
        return None

    while True:
        count = len(candidates)
        latest = candidates[0].checkpoint
        latest_goal = _truncate_text(latest.task_input.goal or latest.task.goal, 72)
        print(
            f"发现 {count} 个未完成任务。最新任务：{latest_goal}\n"
            "是否现在继续？输入 'yes' 继续最新任务，输入 'list' 查看并选择，输入 'delete' 隐藏任务，直接回车或输入 'no' 跳过。"
        )
        answer = input("resume> ").strip().lower()
        if answer in {"new", "/new", "reset"}:
            shelve_active_checkpoint(checkpoint_path)
            return None
        if answer in {"yes", "y", "是"}:
            return candidates[0]
        if answer in {"list", "l", "列表"}:
            return prompt_resume_candidate_selection(candidates, prompt_header="请选择要继续的任务")
        if answer in {"delete", "d", "删除"}:
            prompt_delete_resume_candidates(candidates, workspace_root)
            candidates = list_resume_candidates(workspace_root, checkpoint_path)
            if not candidates:
                return None
            continue
        return None


def prompt_resume_candidate_selection(
    candidates: list[CheckpointCandidate],
    prompt_header: str = "可恢复任务列表",
) -> CheckpointCandidate | None:
    """Show resumable tasks and let the user choose one by number."""

    print()
    print(prompt_header)
    for index, candidate in enumerate(candidates, start=1):
        marker = "当前断点" if candidate.is_active else "归档断点"
        checkpoint = candidate.checkpoint
        goal = _truncate_text(checkpoint.task_input.goal or checkpoint.task.goal, 72)
        print(
            f"{index}. [{marker}] {goal} | "
            f"task_id={checkpoint.task.task_id} | 状态={checkpoint.task.status.value} | "
            f"steps={len(checkpoint.steps)}"
        )
    answer = input("请输入要继续的任务编号（直接回车取消）：").strip()
    if not answer:
        return None
    if not answer.isdigit():
        print("输入无效。\n")
        return None
    selected_index = int(answer)
    if selected_index < 1 or selected_index > len(candidates):
        print("编号超出范围。\n")
        return None
    print()
    return candidates[selected_index - 1]


def prompt_delete_resume_candidates(candidates: list[CheckpointCandidate], workspace_root: str) -> list[str]:
    """Soft-delete resumable tasks from future resume prompts."""

    print()
    print("请选择要隐藏的任务。支持空格或逗号分隔多个编号，例如：1 3 4 或 1,3,4")
    for index, candidate in enumerate(candidates, start=1):
        marker = "当前断点" if candidate.is_active else "归档断点"
        checkpoint = candidate.checkpoint
        goal = _truncate_text(checkpoint.task_input.goal or checkpoint.task.goal, 72)
        print(
            f"{index}. [{marker}] {goal} | "
            f"task_id={checkpoint.task.task_id} | 状态={checkpoint.task.status.value} | "
            f"steps={len(checkpoint.steps)}"
        )

    raw = input("delete> ").strip()
    if not raw:
        print("已取消隐藏。\n")
        return []

    indexes = _parse_candidate_indexes(raw, len(candidates))
    if not indexes:
        print("输入无效。\n")
        return []

    selected = [candidates[index - 1] for index in indexes]
    labels = ", ".join(str(index) for index in indexes)
    print(f"将隐藏任务编号：{labels}")
    for candidate in selected:
        print(f"- {candidate.checkpoint.task.task_id} | {_truncate_text(candidate.checkpoint.task.goal, 72)}")
    confirm = input("是否确认隐藏这些任务？输入 'yes' 确认：").strip().lower()
    if confirm not in {"yes", "y", "是"}:
        print("已取消隐藏。\n")
        return []

    task_ids = [candidate.checkpoint.task.task_id for candidate in selected]
    ignore_task_ids(workspace_root, task_ids)
    print(f"已隐藏 {len(task_ids)} 个任务。ignore_file={default_ignored_tasks_path(workspace_root)}\n")
    return task_ids


def _resume_candidate(
    candidate: CheckpointCandidate,
    *,
    provider: str,
    model: str | None,
    thinking: bool,
    profile: str,
    max_steps: int | None,
    max_tokens: int | None,
    max_seconds: float | None,
    checkpoint_path: str,
    approval_callback,
    event_queue_path: str | None = None,
) -> RuntimeResult:
    """Run one resumable checkpoint back through the adaptive runtime."""

    checkpoint = candidate.checkpoint
    source = "当前断点" if candidate.is_active else "归档断点"
    print(f"正在从{source}恢复任务 {checkpoint.task.task_id}。\n")
    budget = resolve_budget(
        checkpoint.task_input,
        profile=profile,
        max_steps=max_steps,
        max_tokens=max_tokens,
        max_seconds=max_seconds,
    )
    if max_steps is None and max_tokens is None and max_seconds is None:
        budget = apply_budget_recommendation(
            budget,
            recommend_budget(checkpoint.task_input, default_budget_history_path(checkpoint.task_input.workspace_root)),
        )
    return run_task_adaptive(
        checkpoint.task_input,
        lambda token_budget: create_llm_client(
            provider=provider,
            model=model,
            thinking=thinking,
            max_tokens=token_budget,
            budget_profile=budget.profile,
            task_text=f"{checkpoint.task_input.goal}\n{checkpoint.task_input.user_message}",
        ),
        budget=budget,
        approval_callback=approval_callback,
        checkpoint_path=checkpoint_path,
        resume_checkpoint=checkpoint,
        event_queue_path=event_queue_path,
    )


def _record_budget_sample(task_input, result: RuntimeResult, workspace_root: str) -> None:
    """Persist one runtime resource sample for future budget recommendations."""

    try:
        append_budget_sample(
            sample_from_result(task_input, result.task.status, result.stats, result.state.errors, result.state),
            default_budget_history_path(workspace_root),
        )
    except Exception:
        return


def _record_planner_evaluation(task_input, result: RuntimeResult, workspace_root: str) -> None:
    """Persist one planner quality report for planner feedback and CLI stats."""

    try:
        report = planner_evaluation_from_result(
            task_input,
            result.task.status,
            result.stats,
            result.state.errors,
            result.state,
        )
        append_planner_evaluation(report, default_planner_eval_path(workspace_root))
    except Exception:
        return


def _finalize_task_result(
    *,
    session,
    user_message: str,
    result: RuntimeResult,
    workspace_root: str,
    session_file: str | None,
    audit_log: str | None,
    no_audit: bool,
    output_mode: str,
    trigger: str,
    provider: str,
    model: str | None,
    thinking: bool,
) -> Path | None:
    update_session_from_result(session, user_message, result)
    save_session(session, session_file)
    _record_budget_sample(result.task_input, result, workspace_root)
    _record_planner_evaluation(result.task_input, result, workspace_root)
    _record_task_memory(
        session.session_id,
        user_message,
        result,
        workspace_root,
        provider=provider,
        model=model,
        thinking=thinking,
    )
    if not no_audit:
        append_audit_record(
            result,
            session_id=session.session_id,
            user_message=user_message,
            path=audit_log or default_audit_path(workspace_root),
        )
    if result.task.status.value in {"completed", "failed", "waiting_user"}:
        return export_handoff_file(
            workspace_root=workspace_root,
            trigger=trigger,
            session_path=session_file,
            checkpoint_path=default_checkpoint_path(workspace_root),
            audit_log_path=audit_log,
        )
    return None


def _record_task_memory(
    session_id: str,
    user_message: str,
    result: RuntimeResult,
    workspace_root: str,
    *,
    provider: str,
    model: str | None,
    thinking: bool,
) -> None:
    """Persist semantic memory chunks for the live memory graph."""

    try:
        final_answer = final_answer_from_result(result) or ""
        tool_names = [
            step.decision.tool_name
            for step in result.steps
            if step.decision.tool_name
        ]
        llm_chunks = []
        if not thinking and should_use_llm_memory_extraction(
            provider=provider,
            user_message=user_message,
            final_answer=final_answer,
            errors=list(result.state.errors),
        ):
            llm_chunks = extract_semantic_memory_chunks(
                provider=provider,
                model=model,
                user_message=user_message,
                final_answer=final_answer,
                errors=list(result.state.errors),
                tool_names=tool_names,
            )
        record_task_result_memory(
            workspace_root=workspace_root,
            task_id=result.task.task_id,
            session_id=session_id,
            user_message=user_message,
            status=result.task.status.value,
            final_answer=final_answer,
            errors=list(result.state.errors),
            tool_names=tool_names,
            llm_chunks=llm_chunks,
        )
    except Exception:
        return


def _print_handoff_path(handoff_path: Path | None, output_mode: str) -> None:
    if handoff_path is not None and output_mode != "json":
        print(f"handoff_file={handoff_path}")


def _print_token_usage(result: RuntimeResult, output_mode: str) -> None:
    if output_mode == "json":
        return
    usage = result.stats.usage or {}
    prompt = _usage_int(usage, "prompt_tokens")
    completion = _usage_int(usage, "completion_tokens")
    total = _usage_int(usage, "total_tokens") or prompt + completion
    hit = _usage_int(usage, "prompt_cache_hit_tokens")
    miss = _usage_int(usage, "prompt_cache_miss_tokens")
    cache = f"; cache_hit={hit}; cache_miss={miss}" if hit or miss else ""
    print(f"token_usage: prompt={prompt}; completion={completion}; total={total}{cache}")


def _usage_int(usage: dict, key: str) -> int:
    value = usage.get(key, 0)
    return int(value) if isinstance(value, (int, float)) else 0


def _export_session_handoff(
    *,
    workspace_root: str,
    session,
    session_file: str | None,
    checkpoint_path: str,
    audit_log: str | None,
    output_mode: str,
) -> None:
    if not session.turns and not Path(checkpoint_path).exists():
        return
    handoff_path = export_handoff_file(
        workspace_root=workspace_root,
        trigger="session_end",
        session_path=session_file,
        checkpoint_path=checkpoint_path,
        audit_log_path=audit_log,
    )
    if output_mode != "json":
        print(f"handoff_file={handoff_path}")


def _parse_candidate_indexes(text: str, total: int) -> list[int]:
    raw_parts = [part.strip() for part in text.replace(",", " ").split() if part.strip()]
    if not raw_parts:
        return []
    indexes: list[int] = []
    for raw_part in raw_parts:
        if not raw_part.isdigit():
            return []
        value = int(raw_part)
        if value < 1 or value > total:
            return []
        if value not in indexes:
            indexes.append(value)
    return indexes


def _truncate_text(text: str, limit: int = 72) -> str:
    """Return a compact single-line summary for prompts."""

    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def format_approval_request(decision, policy) -> str:
    """Format a readable approval request for high-risk actions."""

    lines = ["APPROVAL REQUIRED"]
    lines.append(f"tool={decision.tool_name}")
    lines.append(f"risk={policy.mode.value}")
    lines.append(f"reason={policy.reason}")
    args = decision.tool_args or {}

    if decision.tool_name == "run_command":
        lines.append(f"cwd={args.get('cwd', '.')}")
        lines.append(f"argv={args.get('args', [])}")
        lines.append(f"timeout_seconds={args.get('timeout_seconds', 30)}")
    elif decision.tool_name == "apply_patch":
        patch = args.get("patch", "")
        try:
            lines.append(f"patch={summarize_patch(str(patch))}")
        except Exception:
            lines.append("patch=(unable to summarize patch)")
    elif decision.tool_name == "git_add":
        lines.append(f"paths={args.get('paths', [])}")
    elif decision.tool_name == "git_commit":
        lines.append(f"message={args.get('message', '')}")
    elif "path" in args:
        lines.append(f"path={args.get('path')}")
    else:
        lines.append(f"args={args}")

    return "\n".join(lines)


def _final_answer(result: RuntimeResult) -> str | None:
    """Extract the user-visible final answer from one runtime result."""

    return final_answer_from_result(result)


def main() -> None:
    configure_stdio_utf8()
    parser = argparse.ArgumentParser(description="Run the Gangent interactive CLI.")
    parser.add_argument(
        "--provider",
        choices=["fake", "openai", "deepseek"],
        default="deepseek",
        help="Choose which model provider to use.",
    )
    parser.add_argument("--model", help="Override the provider model name.")
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable DeepSeek thinking mode when provider is deepseek.",
    )
    parser.add_argument(
        "--profile",
        choices=["auto", "light", "medium", "heavy", "ultra"],
        default="auto",
        help="Adaptive runtime budget profile. Defaults to auto.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Optional override for total runtime loop steps per user task.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Optional override for maximum generated tokens per model call.",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        help="Optional override for total wall-clock seconds per user task.",
    )
    parser.add_argument(
        "--workspace-root",
        default=str(Path.cwd()),
        help="Workspace root that tools are allowed to access.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the latest persisted session from disk.",
    )
    parser.add_argument(
        "--session-file",
        help="Custom JSON file used to persist the CLI session.",
    )
    parser.add_argument(
        "--output",
        choices=["quiet", "verbose", "json"],
        default="verbose",
        help="Choose CLI output mode.",
    )
    parser.add_argument(
        "--audit-log",
        help="Custom JSONL audit log path.",
    )
    parser.add_argument(
        "--checkpoint-file",
        help="Custom JSON checkpoint path for task recovery.",
    )
    parser.add_argument(
        "--event-log",
        help="Custom JSONL event queue path for cooperative interrupts.",
    )
    parser.add_argument(
        "--no-audit",
        action="store_true",
        help="Disable local JSONL audit logging.",
    )
    parser.add_argument(
        "--export-handoff",
        action="store_true",
        help="Export a timestamped Markdown handoff file to the parent Codex folder and exit.",
    )
    parser.add_argument(
        "--handoff-output",
        help="Override the handoff Markdown output path.",
    )
    args = parser.parse_args()

    if args.export_handoff:
        path = export_handoff_file(
            workspace_root=args.workspace_root,
            trigger="manual",
            session_path=args.session_file,
            checkpoint_path=args.checkpoint_file,
            audit_log_path=args.audit_log,
            output_path=args.handoff_output,
        )
        print(str(path))
        return

    run_interactive_cli(
        provider=args.provider,
        model=args.model,
        thinking=args.thinking,
        profile=args.profile,
        max_steps=args.max_steps,
        max_tokens=args.max_tokens,
        max_seconds=args.max_seconds,
        workspace_root=args.workspace_root,
        resume=args.resume,
        session_file=args.session_file,
        output_mode=args.output,
        audit_log=args.audit_log,
        no_audit=args.no_audit,
        checkpoint_file=args.checkpoint_file,
        event_log=args.event_log,
    )


if __name__ == "__main__":
    main()

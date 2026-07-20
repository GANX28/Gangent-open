"""Runtime result to local memory graph recording helpers."""

from __future__ import annotations

from pathlib import Path

from .memory_extractor import extract_semantic_memory_chunks, should_use_llm_memory_extraction
from .memory_graph import record_task_result_memory
from .output import final_answer_from_result
from .runtime import RuntimeResult


def record_runtime_result_memory(
    *,
    session_id: str,
    user_message: str,
    result: RuntimeResult,
    workspace_root: str,
    provider: str,
    model: str | None,
    thinking: bool,
) -> Path | None:
    """Persist reusable task memory and never fail the caller."""

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
        return record_task_result_memory(
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
        return None

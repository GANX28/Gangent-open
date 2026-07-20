"""Adaptive Memory Graph v1.

This module implements a small local graph memory layer for Gangent. It is not
a full GraphRAG platform. The first version focuses on explicit node/edge
storage, deterministic scoring, graph expansion, context assembly, and decay.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from .models import new_id, utc_now
from .rag import tokenize
from .secret_guard import redact_secrets, secret_labels


DEFAULT_MEMORY_GRAPH_PATH = Path(".gangent") / "memory" / "graph.json"
MEMORY_GRAPH_VIEWER_DATA_JS = "graph_data.js"
MEMORY_TASK_LOG_JSONL = "task_log.jsonl"


class MemoryNodeType(str, Enum):
    """High-level memory categories used for routing and filtering."""

    FACT = "fact"
    PREFERENCE = "preference"
    DECISION = "decision"
    ISSUE = "issue"
    SOLUTION = "solution"
    CONCEPT = "concept"
    ARTIFACT = "artifact"
    TASK_STATE = "task_state"
    CONSTRAINT = "constraint"
    NOTE = "note"


class MemoryEdgeType(str, Enum):
    """Relationship types between memory nodes."""

    RELATED_TO = "related_to"
    DEPENDS_ON = "depends_on"
    DERIVED_FROM = "derived_from"
    CAUSED_BY = "caused_by"
    FIXED_BY = "fixed_by"
    BELONGS_TO_PROJECT = "belongs_to_project"
    CONTRADICTS = "contradicts"
    NEXT_STEP = "next_step"
    BLOCKED_BY = "blocked_by"
    MUST_READ_BEFORE = "must_read_before"


class MemoryLayer(str, Enum):
    """Coarse memory layers used for dynamic context routing."""

    DATA = "data"
    TASK = "task"
    KNOWLEDGE = "knowledge"


@dataclass
class MemoryNode:
    """One typed, scoreable memory unit."""

    node_id: str
    node_type: MemoryNodeType
    content: str
    summary: str = ""
    project_scope: str = ""
    source: str = ""
    tags: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    last_accessed_at: str = ""
    access_count: int = 0
    importance: float = 0.5
    confidence: float = 0.8
    decay_score: float = 1.0
    stale: bool = False
    layer: MemoryLayer = MemoryLayer.DATA


@dataclass
class MemoryEdge:
    """A typed graph relationship between two memory nodes."""

    edge_id: str
    source_node_id: str
    target_node_id: str
    edge_type: MemoryEdgeType
    weight: float = 1.0
    created_at: str = field(default_factory=utc_now)


@dataclass(frozen=True)
class MemorySearchResult:
    """One ranked memory retrieval result."""

    node: MemoryNode
    score: float
    reason: str
    depth: int = 0


@dataclass(frozen=True)
class MemoryContextPack:
    """Grouped memory retrieval result before prompt assembly."""

    data_nodes: tuple[MemorySearchResult, ...] = ()
    task_nodes: tuple[MemorySearchResult, ...] = ()
    knowledge_nodes: tuple[MemorySearchResult, ...] = ()
    conflict_notes: tuple[str, ...] = ()
    omitted: tuple[str, ...] = ()


class JsonMemoryGraphStore:
    """JSON-backed local graph memory store.

    The whole graph is loaded and saved as one JSON document. This is acceptable
    for v1 because the target is a small personal-agent memory graph, not a
    large production graph database.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.nodes: dict[str, MemoryNode] = {}
        self.edges: list[MemoryEdge] = []
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self.nodes = {}
            self.edges = []
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.nodes = {
            item["node_id"]: _node_from_dict(item)
            for item in data.get("nodes", [])
        }
        self.edges = [_edge_from_dict(item) for item in data.get("edges", [])]

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        export_graph_data_js(self.path.with_name(MEMORY_GRAPH_VIEWER_DATA_JS), data)
        return self.path

    def to_dict(self) -> dict:
        """Return the canonical JSON representation for storage and viewers."""

        return {
            "version": 1,
            "generated_at": utc_now(),
            "nodes": [asdict(node) for node in self.nodes.values()],
            "edges": [asdict(edge) for edge in self.edges],
        }

    def add_node(
        self,
        node_type: MemoryNodeType,
        content: str,
        summary: str = "",
        project_scope: str = "",
        source: str = "",
        tags: list[str] | None = None,
        importance: float = 0.5,
        confidence: float = 0.8,
        layer: MemoryLayer | str | None = None,
    ) -> MemoryNode:
        if not content.strip():
            raise ValueError("Memory node content must not be empty.")
        labels = secret_labels(content)
        if labels:
            raise ValueError(f"Refusing to store possible secrets: {', '.join(labels)}")
        node = MemoryNode(
            node_id=new_id("mem"),
            node_type=node_type,
            content=redact_secrets(content.strip()),
            summary=redact_secrets(summary.strip()),
            project_scope=project_scope.strip(),
            source=source.strip(),
            tags=tags or [],
            importance=_clamp01(importance),
            confidence=_clamp01(confidence),
            layer=_resolve_layer(node_type, layer),
        )
        self.nodes[node.node_id] = node
        return node

    def add_edge(
        self,
        source_node_id: str,
        target_node_id: str,
        edge_type: MemoryEdgeType,
        weight: float = 1.0,
    ) -> MemoryEdge:
        if source_node_id not in self.nodes:
            raise ValueError(f"Unknown source node: {source_node_id}")
        if target_node_id not in self.nodes:
            raise ValueError(f"Unknown target node: {target_node_id}")
        edge = MemoryEdge(
            edge_id=new_id("edge"),
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            edge_type=edge_type,
            weight=max(0.0, min(2.0, weight)),
        )
        self.edges.append(edge)
        return edge

    def neighbors(self, node_id: str) -> list[tuple[MemoryEdge, MemoryNode]]:
        found: list[tuple[MemoryEdge, MemoryNode]] = []
        for edge in self.edges:
            other_id = ""
            if edge.source_node_id == node_id:
                other_id = edge.target_node_id
            elif edge.target_node_id == node_id:
                other_id = edge.source_node_id
            if other_id and other_id in self.nodes:
                found.append((edge, self.nodes[other_id]))
        return found

    def mark_accessed(self, node_ids: Iterable[str]) -> None:
        now = utc_now()
        for node_id in node_ids:
            node = self.nodes.get(node_id)
            if not node:
                continue
            node.access_count += 1
            node.last_accessed_at = now
            node.decay_score = min(1.0, node.decay_score + 0.05)


def default_memory_graph_path(workspace_root: str) -> Path:
    """Return the default memory graph path for a workspace."""

    return Path(workspace_root).resolve() / DEFAULT_MEMORY_GRAPH_PATH


def default_memory_graph_viewer_data_path(workspace_root: str) -> Path:
    """Return the live JS data file consumed by memory_graph_viewer.html."""

    return default_memory_graph_path(workspace_root).with_name(MEMORY_GRAPH_VIEWER_DATA_JS)


def export_graph_data_js(path: str | Path, data: dict) -> Path:
    """Export graph JSON as a browser-loadable JavaScript assignment."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    target.write_text(f"window.GANGENT_MEMORY_GRAPH = {payload};\n", encoding="utf-8")
    return target


def retrieve_memory_graph(
    query: str,
    store: JsonMemoryGraphStore,
    top_k: int = 5,
    project_scope: str = "",
    max_depth: int = 1,
) -> list[MemorySearchResult]:
    """Retrieve memory nodes with lexical scoring plus graph expansion."""

    if not query.strip() or not store.nodes:
        return []
    seed_results = _seed_results(query, store, project_scope)
    expanded = _expand_results(seed_results, store, max_depth=max_depth)
    expanded.sort(key=lambda item: (-item.score, item.node.node_id))
    results = expanded[: max(1, top_k)]
    store.mark_accessed(result.node.node_id for result in results)
    return results


def assemble_memory_context(results: list[MemorySearchResult], max_chars: int = 2_000) -> str:
    """Build a compact model-facing context section from memory results."""

    if not results:
        return ""
    pack = build_memory_context_pack(results, max_chars=max_chars)
    sections = ["Relevant Memory:"]
    sections.extend(_format_memory_group("data", pack.data_nodes))
    sections.extend(_format_memory_group("task", pack.task_nodes))
    sections.extend(_format_memory_group("knowledge", pack.knowledge_nodes))
    if pack.conflict_notes:
        sections.append("Conflicts:\n" + "\n".join(f"- {note}" for note in pack.conflict_notes))
    if pack.omitted:
        sections.append("Omitted memory nodes: " + ", ".join(pack.omitted))
    output = "\n\n".join(sections)
    if len(output) > max_chars:
        output = output[:max_chars] + "\n... memory context truncated"
    return output


def build_memory_context_pack(results: list[MemorySearchResult], max_chars: int = 2_000) -> MemoryContextPack:
    """Group retrieved memory into data / task / knowledge layers."""

    data: list[MemorySearchResult] = []
    task: list[MemorySearchResult] = []
    knowledge: list[MemorySearchResult] = []
    omitted: list[str] = []
    used = 0
    for result in sorted(results, key=lambda item: (-item.score, item.depth, item.node.node_id)):
        text = result.node.summary or result.node.content
        cost = len(text) + 160
        if used + cost > max_chars and used > 0:
            omitted.append(result.node.node_id)
            continue
        if result.node.layer == MemoryLayer.KNOWLEDGE:
            knowledge.append(result)
        elif result.node.layer == MemoryLayer.TASK:
            task.append(result)
        else:
            data.append(result)
        used += cost
    return MemoryContextPack(
        data_nodes=tuple(data),
        task_nodes=tuple(task),
        knowledge_nodes=tuple(knowledge),
        conflict_notes=tuple(_conflict_notes(results)),
        omitted=tuple(omitted),
    )


def memory_context_for_query(
    query: str,
    workspace_root: str,
    top_k: int = 5,
    project_scope: str = "",
    max_depth: int = 1,
) -> str:
    """Convenience wrapper used by context_manager."""

    path = default_memory_graph_path(workspace_root)
    if not path.exists():
        return ""
    store = JsonMemoryGraphStore(path)
    results = retrieve_memory_graph(query, store, top_k=top_k, project_scope=project_scope, max_depth=max_depth)
    if results:
        store.save()
    return assemble_memory_context(results)


def record_task_result_memory(
    *,
    workspace_root: str,
    task_id: str,
    session_id: str,
    user_message: str,
    status: str,
    final_answer: str = "",
    errors: list[str] | None = None,
    tool_names: list[str] | None = None,
    llm_chunks: list[dict[str, Any]] | None = None,
) -> Path:
    """Record task completion without turning raw logs into graph memory.

    The raw task result is appended to `.gangent/memory/task_log.jsonl`.
    Only reusable semantic chunks are promoted into the memory graph so dynamic
    context loading can route over meaningful chunk summaries and relationships.
    """

    store = JsonMemoryGraphStore(default_memory_graph_path(workspace_root))
    safe_errors = errors or []
    safe_tools = tool_names or []
    _append_task_log(
        workspace_root=workspace_root,
        task_id=task_id,
        session_id=session_id,
        user_message=user_message,
        status=status,
        final_answer=final_answer,
        errors=safe_errors,
        tool_names=safe_tools,
    )

    added_nodes: list[MemoryNode] = []
    chunks = normalize_llm_memory_chunks(llm_chunks or [], task_id=task_id)
    if not chunks:
        chunks = summarize_task_result(
            task_id=task_id,
            user_message=user_message,
            status=status,
            final_answer=final_answer,
            errors=safe_errors,
            tool_names=safe_tools,
        )
    for chunk in chunks:
        existing = _find_existing_chunk(store, chunk["node_type"], chunk["summary"], chunk["source"])
        if existing:
            existing.access_count += 1
            existing.decay_score = min(1.0, existing.decay_score + 0.03)
            added_nodes.append(existing)
            continue
        added_nodes.append(
            store.add_node(
                chunk["node_type"],
                content=chunk["content"],
                summary=chunk["summary"],
                project_scope="Gangent",
                source=chunk["source"],
                tags=chunk["tags"],
                importance=chunk["importance"],
                confidence=chunk["confidence"],
                layer=chunk["layer"],
            )
        )
    for left, right in zip(added_nodes, added_nodes[1:]):
        store.add_edge(left.node_id, right.node_id, MemoryEdgeType.RELATED_TO, weight=0.7)
    _connect_new_nodes_to_existing_context(store, added_nodes)
    return store.save()


def summarize_task_result(
    *,
    task_id: str,
    user_message: str,
    status: str,
    final_answer: str = "",
    errors: list[str] | None = None,
    tool_names: list[str] | None = None,
) -> list[dict]:
    """Extract reusable semantic chunks from one task result.

    v1 is deterministic: it promotes read/analysis results, durable explanations,
    and runtime issues. It does not treat every task as long-term memory.
    """

    safe_errors = errors or []
    safe_tools = tool_names or []
    readable_request = _clean_memory_text(user_message, max_chars=300)
    readable_answer = _clean_memory_text(final_answer, max_chars=1_800)
    source = f"task:{task_id}"
    chunks: list[dict] = []

    if status == "completed" and readable_answer and _task_has_reusable_memory(readable_request, readable_answer, safe_tools):
        chunks.append(
            {
                "node_type": _semantic_node_type(readable_request, readable_answer),
                "summary": _semantic_summary(readable_request, readable_answer),
                "content": _semantic_content(readable_request, readable_answer, safe_tools),
                "source": source,
                "tags": ["semantic_chunk", "task_memory", status],
                "importance": 0.68,
                "confidence": 0.82,
                "layer": _semantic_layer(readable_request, readable_answer),
            }
        )
    if safe_errors and _runtime_error_is_reusable(safe_errors, status=status):
        chunks.append(
            {
                "node_type": MemoryNodeType.ISSUE,
                "summary": _issue_summary(safe_errors, task_id),
                "content": _clean_memory_text("\n".join(safe_errors[-5:]), max_chars=900),
                "source": source,
                "tags": ["semantic_chunk", "runtime_issue", status],
                "importance": 0.78,
                "confidence": 0.9,
                "layer": MemoryLayer.TASK,
            }
        )
    return [chunk for chunk in chunks if chunk["summary"] and chunk["content"]]


def normalize_llm_memory_chunks(chunks: list[dict[str, Any]], *, task_id: str) -> list[dict[str, Any]]:
    """Validate LLM-proposed chunks before writing them into the graph."""

    normalized: list[dict[str, Any]] = []
    for item in chunks[:6]:
        if not isinstance(item, dict):
            continue
        try:
            node_type = MemoryNodeType(str(item.get("node_type", "note")))
        except ValueError:
            continue
        try:
            layer = MemoryLayer(str(item.get("layer", "")))
        except ValueError:
            layer = _resolve_layer(node_type, None)
        summary = _clean_memory_text(str(item.get("summary", "")), max_chars=160)
        content = _clean_memory_text(str(item.get("content", "")), max_chars=1_200)
        if len(summary) < 8 or len(content) < 20:
            continue
        if _looks_corrupt_text(summary) or _looks_corrupt_text(content):
            continue
        labels = secret_labels("\n".join([summary, content]))
        if labels:
            continue
        tags = item.get("tags", [])
        safe_tags = [
            _safe_tag(str(tag))
            for tag in tags[:8]
            if isinstance(tag, str) and _safe_tag(str(tag))
        ]
        if "semantic_chunk" not in safe_tags:
            safe_tags.insert(0, "semantic_chunk")
        if "llm_extracted" not in safe_tags:
            safe_tags.append("llm_extracted")
        normalized.append(
            {
                "node_type": node_type,
                "summary": summary,
                "content": content,
                "source": f"llm_memory_extractor:{task_id}",
                "tags": safe_tags,
                "importance": _clamp01(float(item.get("importance", 0.72))),
                "confidence": _clamp01(float(item.get("confidence", 0.76))),
                "layer": layer,
            }
        )
    return normalized


def _append_task_log(
    *,
    workspace_root: str,
    task_id: str,
    session_id: str,
    user_message: str,
    status: str,
    final_answer: str,
    errors: list[str],
    tool_names: list[str],
) -> Path:
    """Append raw task history outside the semantic memory graph."""

    path = default_memory_graph_path(workspace_root).with_name(MEMORY_TASK_LOG_JSONL)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "created_at": utc_now(),
        "task_id": task_id,
        "session_id": session_id,
        "status": status,
        "user_message": redact_secrets(user_message.strip()),
        "final_answer": redact_secrets(final_answer.strip()),
        "errors": [redact_secrets(error) for error in errors[-10:]],
        "tools": tool_names,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def _find_existing_chunk(
    store: JsonMemoryGraphStore,
    node_type: MemoryNodeType,
    summary: str,
    source: str,
) -> MemoryNode | None:
    summary_key = summary.strip().lower()
    for node in store.nodes.values():
        if node.node_type != node_type:
            continue
        if node.summary.strip().lower() != summary_key:
            continue
        if node.source == source or "semantic_chunk" in node.tags:
            return node
    return None


def _connect_new_nodes_to_existing_context(store: JsonMemoryGraphStore, nodes: list[MemoryNode]) -> None:
    """Attach new semantic chunks to nearby existing graph context."""

    existing_edges = {
        tuple(sorted([edge.source_node_id, edge.target_node_id]))
        for edge in store.edges
    }
    for node in nodes:
        best = _nearest_existing_node(store, node)
        if not best:
            continue
        key = tuple(sorted([node.node_id, best.node_id]))
        if key in existing_edges:
            continue
        store.add_edge(best.node_id, node.node_id, MemoryEdgeType.RELATED_TO, weight=0.55)
        existing_edges.add(key)


def _nearest_existing_node(store: JsonMemoryGraphStore, node: MemoryNode) -> MemoryNode | None:
    node_terms = set(tokenize(" ".join([node.summary, node.content, " ".join(node.tags)])))
    if not node_terms:
        return None
    candidates: list[tuple[float, MemoryNode]] = []
    for candidate in store.nodes.values():
        if candidate.node_id == node.node_id or _is_generated_task_result(candidate):
            continue
        candidate_terms = set(tokenize(" ".join([candidate.summary, candidate.content, " ".join(candidate.tags)])))
        if not candidate_terms:
            continue
        overlap = len(node_terms & candidate_terms)
        if not overlap:
            continue
        type_bonus = 0.5 if candidate.node_type == node.node_type else 0.0
        tag_bonus = 0.2 * len(set(node.tags) & set(candidate.tags))
        candidates.append((overlap + type_bonus + tag_bonus, candidate))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1].node_id))
    return candidates[0][1]


def _task_has_reusable_memory(user_message: str, final_answer: str, tool_names: list[str]) -> bool:
    text = f"{user_message}\n{final_answer}".lower()
    durable_markers = [
        "summarize",
        "summary",
        "analyze",
        "analysis",
        "explain",
        "planner",
        "runtime",
        "memory",
        "context",
        "tool",
        "architecture",
        "design",
        "decision",
        "issue",
        "fix",
        "error",
        "README".lower(),
        "docs/",
    ]
    if any(marker in text for marker in durable_markers):
        return True
    return any(tool in {"read_file", "read_many_files", "search_context", "grep_files"} for tool in tool_names)


def _runtime_error_is_reusable(errors: list[str], *, status: str) -> bool:
    text = "\n".join(errors).lower()
    if not text.strip():
        return False
    low_value_markers = [
        "file does not exist",
        "path does not exist",
        "not a file",
        "approval denied",
        "user denied",
        "internal warning",
    ]
    if status == "completed" and any(marker in text for marker in low_value_markers):
        return False
    reusable_markers = [
        "max_steps",
        "plan guard",
        "repeat guard",
        "model output parse failed",
        "unknown tool requested",
        "plain text instead of returning a structured tool call",
        "function arguments are not valid json",
        "tool call is outside the current plan phase",
        "context pollution",
        "budget",
        "timeout",
        "unicode",
        "decode",
        "encoding",
    ]
    return status != "completed" or any(marker in text for marker in reusable_markers)


def _semantic_node_type(user_message: str, final_answer: str) -> MemoryNodeType:
    text = f"{user_message}\n{final_answer}".lower()
    if any(word in text for word in ["error", "failed", "bug", "issue", "problem"]):
        return MemoryNodeType.ISSUE
    if any(word in text for word in ["fix", "solution", "repair", "resolved"]):
        return MemoryNodeType.SOLUTION
    if any(word in text for word in ["decision", "choose", "decided", "use ", "adopt"]):
        return MemoryNodeType.DECISION
    if any(word in text for word in ["runtime", "planner", "context", "memory", "architecture", "graph"]):
        return MemoryNodeType.CONCEPT
    return MemoryNodeType.FACT


def _semantic_layer(user_message: str, final_answer: str) -> MemoryLayer:
    text = f"{user_message}\n{final_answer}".lower()
    if any(word in text for word in ["principle", "pattern", "architecture", "concept", "memory", "context"]):
        return MemoryLayer.KNOWLEDGE
    if any(word in text for word in ["issue", "error", "fix", "task", "planner", "runtime"]):
        return MemoryLayer.TASK
    return MemoryLayer.DATA


def _semantic_summary(user_message: str, final_answer: str, limit: int = 160) -> str:
    candidates = [
        _first_readable_heading(final_answer),
        _first_readable_line(final_answer),
        _first_readable_line(user_message),
    ]
    for candidate in candidates:
        if candidate and not _looks_corrupt_text(candidate):
            return _trim_sentence(candidate, limit)
    return "Reusable semantic memory chunk"


def _semantic_content(user_message: str, final_answer: str, tool_names: list[str], limit: int = 1_800) -> str:
    parts = []
    if user_message.strip():
        parts.append("Task intent:\n" + _clean_memory_text(user_message, max_chars=400))
    if final_answer.strip():
        parts.append("Detailed memory:\n" + _clean_memory_text(final_answer, max_chars=limit))
    if tool_names:
        parts.append("Evidence tools: " + ", ".join(tool_names))
    return "\n\n".join(part for part in parts if part.strip())


def _issue_summary(errors: list[str], task_id: str, limit: int = 140) -> str:
    first = _first_readable_line("\n".join(errors)) or f"Runtime issue during task {task_id}"
    return _trim_sentence(f"Runtime issue: {first}", limit)


def _clean_memory_text(text: str, max_chars: int) -> str:
    lines = []
    for line in text.splitlines():
        value = line.strip()
        if not value or _looks_corrupt_text(value):
            continue
        lines.append(value)
    cleaned = "\n".join(lines).strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[: max_chars - 3].rstrip() + "..."
    return redact_secrets(cleaned)


def _safe_tag(value: str) -> str:
    tag = value.strip().lower().replace(" ", "-")
    tag = re.sub(r"[^a-z0-9_-]", "", tag)
    return tag[:40]


def _first_readable_heading(text: str) -> str:
    for line in text.splitlines():
        value = line.strip()
        if not value.startswith("#"):
            continue
        value = value.lstrip("# ").strip()
        if len(value) >= 6 and not _looks_corrupt_text(value):
            return value
    return ""


def _trim_sentence(text: str, limit: int) -> str:
    value = " ".join(text.split())
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def _is_generated_task_result(node: MemoryNode) -> bool:
    return node.node_type == MemoryNodeType.TASK_STATE and "task_result" in node.tags


def apply_decay(
    store: JsonMemoryGraphStore,
    decay_factor: float = 0.98,
    stale_threshold: float = 0.15,
) -> int:
    """Apply deterministic decay to all nodes and mark stale low-score nodes."""

    changed = 0
    factor = _clamp01(decay_factor)
    for node in store.nodes.values():
        old = node.decay_score
        node.decay_score = round(max(0.0, node.decay_score * factor), 6)
        if node.decay_score < stale_threshold:
            node.stale = True
        if node.decay_score != old:
            changed += 1
    return changed


def _looks_corrupt_text(text: str) -> bool:
    value = text.strip()
    if not value:
        return False
    question_marks = value.count("?")
    replacement_marks = value.count("\ufffd")
    if re.search(r"\?{2,}", value):
        return True
    return (question_marks + replacement_marks) / max(len(value), 1) > 0.12


def _first_readable_line(text: str) -> str:
    for line in text.splitlines():
        value = line.strip().lstrip("#-*0123456789. ")
        if len(value) < 8 or _looks_corrupt_text(value):
            continue
        return value
    return ""


def _seed_results(query: str, store: JsonMemoryGraphStore, project_scope: str) -> list[MemorySearchResult]:
    query_terms = tokenize(query)
    if not query_terms:
        return []
    results: list[MemorySearchResult] = []
    for node in store.nodes.values():
        if _is_generated_task_result(node):
            continue
        if project_scope and node.project_scope and node.project_scope != project_scope:
            continue
        score, matched = _node_score(query_terms, node)
        if score <= 0:
            continue
        results.append(MemorySearchResult(node=node, score=score, reason=f"matched={','.join(matched)}", depth=0))
    return results


def _expand_results(
    seed_results: list[MemorySearchResult],
    store: JsonMemoryGraphStore,
    max_depth: int,
) -> list[MemorySearchResult]:
    by_id: dict[str, MemorySearchResult] = {result.node.node_id: result for result in seed_results}
    frontier = list(seed_results)
    for depth in range(1, max(0, max_depth) + 1):
        next_frontier: list[MemorySearchResult] = []
        for result in frontier:
            for edge, neighbor in store.neighbors(result.node.node_id):
                edge_score = result.score * edge.weight * (0.55 ** depth)
                edge_score = _apply_node_weights(edge_score, neighbor)
                candidate = MemorySearchResult(
                    node=neighbor,
                    score=round(edge_score, 6),
                    reason=f"graph:{edge.edge_type.value}:{result.node.node_id}",
                    depth=depth,
                )
                old = by_id.get(neighbor.node_id)
                if old is None or candidate.score > old.score:
                    by_id[neighbor.node_id] = candidate
                    next_frontier.append(candidate)
        frontier = next_frontier
    return list(by_id.values())


def _node_score(query_terms: tuple[str, ...], node: MemoryNode) -> tuple[float, list[str]]:
    text = " ".join([node.content, node.summary, node.project_scope, node.source, " ".join(node.tags)])
    tokens = tokenize(text)
    if not tokens:
        return 0.0, []
    counts = {token: tokens.count(token) for token in set(tokens)}
    matched: list[str] = []
    score = 0.0
    for term in query_terms:
        count = counts.get(term, 0)
        if count:
            matched.append(term)
            score += 1.0 + math.log(count)
    if not matched:
        return 0.0, []
    return round(_apply_node_weights(score, node), 6), matched


def _apply_node_weights(score: float, node: MemoryNode) -> float:
    if node.stale:
        score *= 0.5
    score *= 0.5 + node.importance
    score *= 0.5 + node.confidence
    score *= max(0.05, node.decay_score)
    score *= 1.0 + min(node.access_count, 20) * 0.02
    return score


def _node_from_dict(data: dict) -> MemoryNode:
    node_type = MemoryNodeType(data["node_type"])
    return MemoryNode(
        node_id=str(data["node_id"]),
        node_type=node_type,
        content=str(data["content"]),
        summary=str(data.get("summary", "")),
        project_scope=str(data.get("project_scope", "")),
        source=str(data.get("source", "")),
        tags=list(data.get("tags", [])),
        created_at=str(data.get("created_at", utc_now())),
        last_accessed_at=str(data.get("last_accessed_at", "")),
        access_count=int(data.get("access_count", 0)),
        importance=_clamp01(float(data.get("importance", 0.5))),
        confidence=_clamp01(float(data.get("confidence", 0.8))),
        decay_score=_clamp01(float(data.get("decay_score", 1.0))),
        stale=bool(data.get("stale", False)),
        layer=_resolve_layer(node_type, data.get("layer")),
    )


def _edge_from_dict(data: dict) -> MemoryEdge:
    return MemoryEdge(
        edge_id=str(data["edge_id"]),
        source_node_id=str(data["source_node_id"]),
        target_node_id=str(data["target_node_id"]),
        edge_type=MemoryEdgeType(data["edge_type"]),
        weight=float(data.get("weight", 1.0)),
        created_at=str(data.get("created_at", utc_now())),
    )


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _resolve_layer(node_type: MemoryNodeType, layer: MemoryLayer | str | None = None) -> MemoryLayer:
    if layer:
        try:
            return MemoryLayer(layer)
        except ValueError:
            pass
    if node_type in {MemoryNodeType.ARTIFACT, MemoryNodeType.FACT, MemoryNodeType.NOTE}:
        return MemoryLayer.DATA
    if node_type in {MemoryNodeType.TASK_STATE, MemoryNodeType.DECISION, MemoryNodeType.ISSUE, MemoryNodeType.SOLUTION}:
        return MemoryLayer.TASK
    return MemoryLayer.KNOWLEDGE


def _format_memory_group(label: str, results: tuple[MemorySearchResult, ...]) -> list[str]:
    if not results:
        return []
    lines = [f"{label.title()} layer:"]
    for index, result in enumerate(results, start=1):
        node = result.node
        summary = (node.summary or _first_readable_line(node.content) or node.node_id).replace("\n", " ").strip()
        detail = node.content.replace("\n", " ").strip()
        if node.summary and detail.startswith(node.summary):
            detail = detail[len(node.summary):].strip(" :-")
        if len(detail) > 480:
            detail = detail[:477].rstrip() + "..."
        lines.append(
            (
                f"[{index}] type={node.node_type.value}; score={result.score:.3f}; depth={result.depth}; "
                f"project={node.project_scope or '-'}; source={node.source or '-'}; reason={result.reason}\n"
                f"summary: {redact_secrets(summary)}\n"
                f"detail: {redact_secrets(detail)}"
            )
        )
    return ["\n".join(lines)]


def _conflict_notes(results: list[MemorySearchResult]) -> list[str]:
    selected = {result.node.node_id for result in results}
    notes: list[str] = []
    # Conflict edges can only be observed if graph expansion returned both ends.
    for result in results:
        if "contradicts" in result.reason and result.node.node_id in selected:
            notes.append(f"{result.node.node_id}: {result.reason}")
    return notes[:5]

# Long-Term Memory and Context Compression

This document defines the recommended memory direction for Gangent.

中文说明：这里说的 memory compression（记忆压缩）不是简单把聊天记录越存越多，
而是把历史信息压成可检索、可审计、可逐步更新的结构化记忆。

## Goal

Gangent should not send the full conversation history to the model forever.

The target design is:

```text
Raw Turns
  -> Short Session Summary
  -> Structured Memory Notes
  -> Retrieval Index
  -> Task-Specific Context Pack
```

The model should receive only the context needed for the current task.

## Recommended Layers

### 1. Working Context

Short-lived context for the current task.

Contents:

- latest user request
- current task state
- recent model/tool turns
- latest errors
- current plan

Implementation status:

- already partly implemented through `SessionState`, `AgentState`, and checkpoint.

### 2. Session Summary

Compressed summary of the current CLI session.

Purpose:

- keep recent work understandable after `/new` or restart
- reduce prompt size
- avoid sending every old turn

Current version:

- deterministic summary, no model compression by default.
- per-turn model compression is not part of the default path because it is usually too expensive for routine use.

Next version:

- optional model-assisted summary when a session grows beyond a threshold and there is a clear benefit
- summary schema with decisions, files changed, unresolved issues, and next steps

### 3. Long-Term Memory Notes

Stable facts that should survive across sessions.

Examples:

- project purpose
- architecture decisions
- user preferences for this project
- common commands
- known limitations

Recommended file:

```text
.gangent/memory/long_term.md
```

### 4. Retrieval Memory

Searchable memory chunks used by RAG.

Recommended index inputs:

- `docs/`
- selected `.md` files
- long-term memory notes
- important design decisions
- selected code comments or module summaries

Do not index:

- `.env`
- secrets
- `.git`
- `.gangent/audit/` raw logs by default
- large binary files

### 5. Embedding Index

Optional vector index.

Recommended modes:

- disabled: default laptop-safe mode
- local_cpu: slower but private
- local_gpu: faster on desktop with CUDA
- remote: easiest but sends text to external API

Current code status:

- backend interface exists in `gangent/embeddings.py`
- local CPU/GPU backends use `sentence-transformers` if installed
- no vector store is active yet

### 6. External Memory Substrate

External memory substrate（外部记忆底座） means a memory service outside the
Gangent runtime that can store durable notes, graph edges, traces, and evidence.

Recommended boundary:

- treat any external backend as an experimental long-term memory service
- do not make an external backend the default memory path without evaluation
- keep local `.gangent/memory/graph.json` as the v1 deterministic memory source
- never write API keys, access tokens, device codes, or private secrets into an external memory service

Future integration path:

```text
Gangent Memory Graph
  -> export selected stable nodes
  -> external memory documents
  -> external graph edges
  -> task-specific recall through MCP
  -> bounded context pack in Context Manager
```

The key production question is not whether memory can be stored remotely. The
key question is whether recall remains relevant, auditable, tenant-safe, and
small enough for the current task context.

## Compression Strategy

Recommended first implementation:

```text
Every completed task:
  1. collect user request, final answer, changed files, errors, decisions
  2. write one compact structured memory entry
  3. keep raw audit separately
  4. let retrieval select relevant memory later
```

Suggested entry shape:

```markdown
## 2026-06-26 - Task Title

Status: completed / failed / waiting

User Intent:
- ...

Actions:
- ...

Decisions:
- ...

Files:
- ...

Open Issues:
- ...
```

## Why Not Only Embedding

Embedding is useful for semantic search, but it is not a full memory system.

Problems if used alone:

- bad chunking leads to bad retrieval
- vector match may miss exact constraints
- no clear audit trail
- hard to know what was forgotten
- model may over-trust irrelevant retrieved chunks

Recommended approach:

```text
structured notes + BM25 + optional embeddings + rerank
```

This is more stable than embedding-only memory.

## Future Implementation Plan

1. Extend `memory_graph.py` with task-result ingestion.
2. Add `append_memory_entry(...)` for structured long-term memories.
3. Add `summarize_task_result(...)`.
4. Store graph memory under `.gangent/memory/graph.json`.
5. Use `context_manager.py` to add retrieved graph memory into the model context.
6. Add optional embedding index for selected memory nodes.

## Adaptive Memory Graph v1

Current code status:

- `gangent/memory_graph.py` defines `MemoryNode`, `MemoryEdge`, a JSON-backed graph store, graph retrieval, context assembly, and decay.
- `context_manager.py` now reads `.gangent/memory/graph.json` when it exists and appends a `Relevant Memory` section to the model context.
- v1 uses deterministic lexical scoring and graph expansion. It does not require embeddings or a graph database.

## Semantic Chunk Ingestion

The memory graph is not a task log. Raw task history is stored separately in
`.gangent/memory/task_log.jsonl`. The graph stores reusable semantic chunks:

```text
task result
  -> raw task log for audit/debugging
  -> deterministic semantic chunk extraction
  -> MemoryNode(summary + detail + metadata)
  -> typed MemoryEdge relationships
  -> retrieval + graph expansion
  -> task-specific context pack
```

Each graph node should represent one meaningful memory chunk. The `summary`
field is the routing surface: it tells the runtime what this node is about.
The `content` field is the detail layer: it is loaded into the context pack
only when the node is selected by retrieval and budget allows it.

Generated task-result nodes with `task_result` tags are treated as legacy log
nodes. Retrieval skips them so dynamic context loading is based on semantic
chunks rather than continuous task history.

## Lightweight LLM Memory Extraction

The default semantic chunk path can use a lightweight LLM extractor after a
task finishes. This is not part of the main runtime loop and does not call
tools. It asks the model for JSON only:

```json
{
  "chunks": [
    {
      "node_type": "concept",
      "layer": "knowledge",
      "summary": "Short English routing summary",
      "content": "Bounded detailed memory",
      "tags": ["memory", "context"],
      "importance": 0.8,
      "confidence": 0.8
    }
  ]
}
```

The runtime still validates everything:

- invalid node types are dropped;
- invalid layers fall back to deterministic defaults;
- corrupt text and secret-looking content are rejected;
- tags are normalized;
- summaries and details are bounded;
- if the LLM call fails, deterministic extraction is used instead.

This keeps the system stable while improving memory quality. The extractor is
only triggered for durable tasks such as analysis, design, debugging, fixes,
runtime explanations, and longer final answers. Trivial chat is skipped.

Node types:

- `fact`
- `preference`
- `decision`
- `issue`
- `solution`
- `concept`
- `artifact`
- `task_state`
- `constraint`
- `note`

Edge types:

- `related_to`
- `depends_on`
- `derived_from`
- `caused_by`
- `fixed_by`
- `belongs_to_project`
- `contradicts`
- `next_step`
- `blocked_by`
- `must_read_before`

Decay strategy:

- Nodes carry `decay_score`, `access_count`, `importance`, `confidence`, and `stale`.
- Retrieval increases access count and slightly refreshes decay.
- `apply_decay(...)` lowers unused node scores and marks low-score nodes stale instead of deleting them.
  This avoids irreversible memory loss while reducing old or weak memories in retrieval.

## Current Recommendation

For Gangent right now:

- keep deterministic session summary
- keep model-based per-turn compression out of the default path
- add structured task-result ingestion next
- keep embedding optional
- use retrieval logs to evaluate whether memory search is useful
- avoid model-based compression on every turn because it increases API cost

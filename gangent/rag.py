"""RAG Retrieval Pipeline（检索增强生成管线）。

第一版实现一个本地商用风格的 RAG 骨架：
- ingestion（文档摄取）：扫描允许的文本文件；
- chunking（分块）：按行切块并保留来源行号；
- access filter（访问过滤）：跳过隐藏目录、敏感路径、大文件和二进制文件；
- lexical retrieval（词法检索）：使用 BM25-like 稀疏检索；
- rerank（重排序）：对路径命中、短语命中、结果位置做轻量加权；
- citation（引用）：返回文件路径和行号；
- redaction（脱敏）：所有返回片段先走 Secret Guard。
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .permissions import PermissionError, resolve_workspace_path
from .secret_guard import is_sensitive_path, redact_secrets
from .models import utc_now


DEFAULT_RETRIEVAL_LOG = Path(".gangent") / "retrieval" / "latest.jsonl"
SUPPORTED_EXTENSIONS = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".toml",
    ".yaml",
    ".yml",
    ".ini",
    ".cfg",
}
IGNORED_DIRECTORIES = {".git", ".gangent", "__pycache__", ".pytest_cache", ".mypy_cache"}
TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[\u4e00-\u9fff]+")


@dataclass(frozen=True)
class RetrievalConfig:
    """RAG 检索配置。

    这些参数相当于商用系统里的 retrieval policy（检索策略）。
    """

    max_file_bytes: int = 80_000
    chunk_size_lines: int = 80
    chunk_overlap_lines: int = 10
    max_chunks: int = 2_000
    max_results: int = 8


@dataclass(frozen=True)
class TextChunk:
    """一个可检索文本块。"""

    chunk_id: str
    path: str
    start_line: int
    end_line: int
    text: str
    tokens: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, str | int] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    """一次检索命中的结果。"""

    chunk: TextChunk
    score: float
    matched_terms: tuple[str, ...]


@dataclass(frozen=True)
class RetrievalLogEntry:
    """一次检索的可观测记录，供后续 recall eval 和调参使用。"""

    query: str
    top_k: int
    result_count: int
    results: list[dict]
    created_at: str = field(default_factory=utc_now)


def search_context(
    query: str,
    workspace_root: str,
    top_k: int = 5,
    config: RetrievalConfig | None = None,
    log_path: str | Path | None = None,
) -> str:
    """检索 workspace 里的相关上下文并格式化给模型。"""

    if not query.strip():
        raise ValueError("search_context query must not be empty.")
    cfg = config or RetrievalConfig()
    safe_top_k = max(1, min(top_k, cfg.max_results))
    chunks = build_chunks(workspace_root, cfg)
    results = search_chunks(query, chunks, top_k=safe_top_k)
    append_retrieval_log(
        RetrievalLogEntry(
            query=query,
            top_k=safe_top_k,
            result_count=len(results),
            results=[_result_log_item(result) for result in results],
        ),
        log_path or default_retrieval_log_path(workspace_root),
    )
    return format_search_results(results)


def default_retrieval_log_path(workspace_root: str) -> Path:
    """返回默认 retrieval log 路径。"""

    return Path(workspace_root).resolve() / DEFAULT_RETRIEVAL_LOG


def append_retrieval_log(entry: RetrievalLogEntry, path: str | Path) -> Path:
    """以 JSONL 形式追加一条 retrieval log。"""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
    return target


def build_chunks(workspace_root: str, config: RetrievalConfig | None = None) -> list[TextChunk]:
    """扫描 workspace 并生成可检索 chunk。

    这是 ingestion + chunking 阶段。它不会读取敏感路径，也不会读取过大或二进制文件。
    """

    cfg = config or RetrievalConfig()
    root = resolve_workspace_path(".", workspace_root)
    chunks: list[TextChunk] = []

    for path in sorted(root.rglob("*")):
        if len(chunks) >= cfg.max_chunks:
            break
        if not _is_indexable_file(path, root, cfg):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        text = redact_secrets(text)
        chunks.extend(_chunks_for_file(path, root, text, cfg, remaining=cfg.max_chunks - len(chunks)))

    return chunks


def search_chunks(query: str, chunks: list[TextChunk], top_k: int = 5) -> list[SearchResult]:
    """对 chunk 做 BM25-like 检索和轻量重排序。"""

    query_terms = tokenize(query)
    if not query_terms or not chunks:
        return []

    doc_freq = _document_frequency(chunks)
    avg_len = sum(len(chunk.tokens) for chunk in chunks) / max(1, len(chunks))
    scored: list[SearchResult] = []

    for chunk in chunks:
        score, matched_terms = _bm25_score(query_terms, chunk, doc_freq, len(chunks), avg_len)
        if score <= 0:
            continue
        score = _rerank_score(score, query, query_terms, chunk)
        scored.append(
            SearchResult(
                chunk=chunk,
                score=round(score, 6),
                matched_terms=tuple(sorted(set(matched_terms))),
            )
        )

    scored.sort(key=lambda item: (-item.score, item.chunk.path, item.chunk.start_line))
    return scored[: max(1, top_k)]


def format_search_results(results: list[SearchResult], max_chars_per_chunk: int = 1_200) -> str:
    """把检索结果格式化成带 citation（引用来源）的上下文。"""

    if not results:
        return "No relevant context found."

    sections: list[str] = []
    for index, result in enumerate(results, start=1):
        text = result.chunk.text.strip()
        if len(text) > max_chars_per_chunk:
            text = text[:max_chars_per_chunk] + "\n... [truncated]"
        sections.append(
            "\n".join(
                [
                    f"[{index}] {result.chunk.path}:{result.chunk.start_line}-{result.chunk.end_line}",
                    f"score={result.score}; matched={', '.join(result.matched_terms)}",
                    redact_secrets(text),
                ]
            )
        )
    return "\n\n".join(sections)


def tokenize(text: str) -> tuple[str, ...]:
    """简单 tokenizer（分词器）。

    第一版不引入中文分词或外部依赖，使用英数字标识符和连续中文片段。
    """

    return tuple(match.group(0).lower() for match in TOKEN_PATTERN.finditer(text))


def _is_indexable_file(path: Path, root: Path, cfg: RetrievalConfig) -> bool:
    if not path.is_file():
        return False
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    if any(part in IGNORED_DIRECTORIES for part in relative.parts):
        return False
    if any(part.startswith(".") for part in relative.parts):
        return False
    if is_sensitive_path(relative) or is_sensitive_path(path):
        return False
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return False
    if path.stat().st_size > cfg.max_file_bytes:
        return False
    try:
        sample = path.read_bytes()[:1024]
    except OSError:
        return False
    return b"\x00" not in sample


def _chunks_for_file(
    path: Path,
    root: Path,
    text: str,
    cfg: RetrievalConfig,
    remaining: int,
) -> list[TextChunk]:
    lines = text.splitlines()
    if not lines:
        return []

    step = max(1, cfg.chunk_size_lines - cfg.chunk_overlap_lines)
    relative_path = str(path.relative_to(root)).replace("\\", "/")
    chunks: list[TextChunk] = []

    for start in range(0, len(lines), step):
        if len(chunks) >= remaining:
            break
        end = min(len(lines), start + cfg.chunk_size_lines)
        chunk_text = "\n".join(lines[start:end])
        tokens = tokenize(f"{relative_path}\n{chunk_text}")
        if not tokens:
            continue
        chunks.append(
            TextChunk(
                chunk_id=f"{relative_path}:{start + 1}-{end}",
                path=relative_path,
                start_line=start + 1,
                end_line=end,
                text=chunk_text,
                tokens=tokens,
                metadata={
                    "source_type": path.suffix.lower().lstrip(".") or "text",
                    "line_count": end - start,
                    "char_count": len(chunk_text),
                },
            )
        )
        if end >= len(lines):
            break

    return chunks


def _document_frequency(chunks: list[TextChunk]) -> dict[str, int]:
    freq: dict[str, int] = {}
    for chunk in chunks:
        for token in set(chunk.tokens):
            freq[token] = freq.get(token, 0) + 1
    return freq


def _bm25_score(
    query_terms: tuple[str, ...],
    chunk: TextChunk,
    doc_freq: dict[str, int],
    doc_count: int,
    avg_len: float,
) -> tuple[float, list[str]]:
    k1 = 1.5
    b = 0.75
    counts = Counter(chunk.tokens)
    chunk_len = max(1, len(chunk.tokens))
    score = 0.0
    matched: list[str] = []

    for term in query_terms:
        tf = counts.get(term, 0)
        if tf <= 0:
            continue
        df = doc_freq.get(term, 0)
        idf = math.log(1 + (doc_count - df + 0.5) / (df + 0.5))
        denominator = tf + k1 * (1 - b + b * chunk_len / max(avg_len, 1.0))
        score += idf * (tf * (k1 + 1)) / denominator
        matched.append(term)

    return score, matched


def _rerank_score(score: float, query: str, query_terms: tuple[str, ...], chunk: TextChunk) -> float:
    """轻量 reranker（重排序器）。

    商用系统常用 cross-encoder reranker；第一版用确定性特征替代。
    """

    lowered_query = query.lower().strip()
    lowered_text = chunk.text.lower()
    lowered_path = chunk.path.lower()

    if lowered_query and lowered_query in lowered_text:
        score *= 1.4
    if any(term in lowered_path for term in query_terms):
        score *= 1.2
    if chunk.start_line == 1:
        score *= 1.05
    return score


def _result_log_item(result: SearchResult) -> dict:
    return {
        "chunk_id": result.chunk.chunk_id,
        "path": result.chunk.path,
        "start_line": result.chunk.start_line,
        "end_line": result.chunk.end_line,
        "score": result.score,
        "matched_terms": list(result.matched_terms),
        "metadata": dict(result.chunk.metadata),
    }

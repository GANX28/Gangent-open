"""Embedding backends（向量表示后端）。

Embedding（向量表示）把文本转换成数字向量，后续可以用于 vector
search（向量检索）和 hybrid retrieval（混合检索）。

本模块默认不启用 embedding。原因是本地模型依赖较重，且 GPU 能力因机器
不同差异很大。这里提供三档后端：

- remote: 调用 OpenAI-compatible embedding API；
- local_cpu: 使用 sentence-transformers 在 CPU 上本地推理；
- local_gpu: 使用 sentence-transformers 在 CUDA GPU 上本地推理。
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from enum import Enum
from importlib.util import find_spec
from typing import Any


class EmbeddingBackendKind(str, Enum):
    """Supported embedding backend categories."""

    DISABLED = "disabled"
    REMOTE = "remote"
    LOCAL_CPU = "local_cpu"
    LOCAL_GPU = "local_gpu"


@dataclass(frozen=True)
class EmbeddingBackendConfig:
    """Configuration for an embedding backend.

    `model` 示例：
    - remote: text-embedding-3-small 或其他 OpenAI-compatible embedding model
    - local_cpu/local_gpu: sentence-transformers/all-MiniLM-L6-v2
    """

    kind: EmbeddingBackendKind = EmbeddingBackendKind.DISABLED
    model: str = ""
    endpoint: str = ""
    api_key_env: str = "EMBEDDING_API_KEY"
    min_gpu_memory_gb: int = 0
    timeout_seconds: int = 30


@dataclass(frozen=True)
class EmbeddingEnvironment:
    """Local embedding environment detection result."""

    sentence_transformers_available: bool
    torch_available: bool
    cuda_available: bool
    cuda_device_count: int = 0
    cuda_memory_gb: float = 0.0


class EmbeddingBackend:
    """Minimal interface all embedding backends implement."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class DisabledEmbeddingBackend(EmbeddingBackend):
    """Default backend used when embeddings are not enabled."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("Embedding backend is disabled.")


class RemoteEmbeddingBackend(EmbeddingBackend):
    """OpenAI-compatible remote embedding backend."""

    def __init__(self, config: EmbeddingBackendConfig) -> None:
        if not config.endpoint:
            raise RuntimeError("Remote embedding endpoint is required.")
        if not config.model:
            raise RuntimeError("Remote embedding model is required.")
        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing embedding API key env var: {config.api_key_env}")
        self._config = config
        self._api_key = api_key

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        _validate_texts(texts)
        payload = json.dumps({"model": self._config.model, "input": texts}).encode("utf-8")
        request = urllib.request.Request(
            self._config.endpoint.rstrip("/") + "/embeddings",
            data=payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._config.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Remote embedding request failed: {exc}") from exc

        data = json.loads(body)
        try:
            return [list(item["embedding"]) for item in data["data"]]
        except (KeyError, TypeError) as exc:
            raise RuntimeError("Remote embedding response has unexpected shape.") from exc


class LocalSentenceTransformerBackend(EmbeddingBackend):
    """sentence-transformers local embedding backend."""

    def __init__(self, config: EmbeddingBackendConfig, device: str) -> None:
        if not config.model:
            raise RuntimeError("Local embedding model is required.")
        if find_spec("sentence_transformers") is None:
            raise RuntimeError(
                "sentence-transformers is not installed. Install it before enabling local embeddings."
            )
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(config.model, device=device)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        _validate_texts(texts)
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]


def detect_embedding_environment() -> EmbeddingEnvironment:
    """Detect whether local embedding backends can run on this machine."""

    sentence_transformers_available = find_spec("sentence_transformers") is not None
    torch_available = find_spec("torch") is not None
    if not torch_available:
        return EmbeddingEnvironment(
            sentence_transformers_available=sentence_transformers_available,
            torch_available=False,
            cuda_available=False,
        )

    import torch

    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    memory_gb = 0.0
    if cuda_available and device_count:
        props = torch.cuda.get_device_properties(0)
        memory_gb = round(float(props.total_memory) / (1024**3), 2)
    return EmbeddingEnvironment(
        sentence_transformers_available=sentence_transformers_available,
        torch_available=True,
        cuda_available=cuda_available,
        cuda_device_count=device_count,
        cuda_memory_gb=memory_gb,
    )


def create_embedding_backend(config: EmbeddingBackendConfig | None = None) -> EmbeddingBackend:
    """Create an embedding backend from config."""

    cfg = config or EmbeddingBackendConfig()
    if cfg.kind == EmbeddingBackendKind.DISABLED:
        return DisabledEmbeddingBackend()
    if cfg.kind == EmbeddingBackendKind.REMOTE:
        return RemoteEmbeddingBackend(cfg)
    if cfg.kind == EmbeddingBackendKind.LOCAL_CPU:
        return LocalSentenceTransformerBackend(cfg, device="cpu")
    if cfg.kind == EmbeddingBackendKind.LOCAL_GPU:
        env = detect_embedding_environment()
        if not env.cuda_available:
            raise RuntimeError("CUDA GPU is not available for local_gpu embeddings.")
        if cfg.min_gpu_memory_gb and env.cuda_memory_gb < cfg.min_gpu_memory_gb:
            raise RuntimeError(
                f"CUDA memory is too small: {env.cuda_memory_gb} GB < {cfg.min_gpu_memory_gb} GB."
            )
        return LocalSentenceTransformerBackend(cfg, device="cuda")
    raise RuntimeError(f"Unknown embedding backend kind: {cfg.kind}")


def _validate_texts(texts: list[str]) -> None:
    if not isinstance(texts, list) or not texts:
        raise RuntimeError("embed_texts requires a non-empty list of strings.")
    if len(texts) > 128:
        raise RuntimeError("embed_texts supports at most 128 texts per batch.")
    for text in texts:
        if not isinstance(text, str):
            raise RuntimeError("embed_texts items must be strings.")
        if len(text.encode("utf-8")) > 20_000:
            raise RuntimeError("embed_texts item is too large.")

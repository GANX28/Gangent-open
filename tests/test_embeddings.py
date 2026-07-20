import unittest

from gangent.embeddings import (
    DisabledEmbeddingBackend,
    EmbeddingBackendConfig,
    EmbeddingBackendKind,
    create_embedding_backend,
    detect_embedding_environment,
)


class EmbeddingTests(unittest.TestCase):
    def test_disabled_backend_is_default(self):
        backend = create_embedding_backend()

        self.assertIsInstance(backend, DisabledEmbeddingBackend)
        with self.assertRaises(RuntimeError):
            backend.embed_texts(["hello"])

    def test_detect_embedding_environment_returns_flags(self):
        env = detect_embedding_environment()

        self.assertIsInstance(env.sentence_transformers_available, bool)
        self.assertIsInstance(env.torch_available, bool)
        self.assertIsInstance(env.cuda_available, bool)

    def test_remote_backend_requires_endpoint(self):
        with self.assertRaises(RuntimeError):
            create_embedding_backend(
                EmbeddingBackendConfig(
                    kind=EmbeddingBackendKind.REMOTE,
                    model="embedding-model",
                    endpoint="",
                )
            )

    def test_local_cpu_requires_model(self):
        with self.assertRaises(RuntimeError):
            create_embedding_backend(EmbeddingBackendConfig(kind=EmbeddingBackendKind.LOCAL_CPU))


if __name__ == "__main__":
    unittest.main()

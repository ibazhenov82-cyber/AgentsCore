"""Каталог GGUF-моделей Hugging Face («День 29»): разбор файлов репозитория,
метки квантования, оценка памяти, поиск. Офлайн: HTTP заменён фейком."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from agents_core.api.app import create_app_with_repository
from agents_core.hf_hub import HuggingFaceHub, estimate_need_mb, kv_mb_per_token, quant_of
from tests.test_repository import make_repository

MB = 1024 * 1024


def _resp(payload, status=200):
    resp = MagicMock()
    resp.ok = status < 400
    resp.status_code = status
    resp.json.return_value = payload
    return resp


INFO = {"author": "unsloth", "downloads": 6488262, "likes": 5016, "cardData": {"license": "apache-2.0"},
        "pipeline_tag": "text-generation"}
TREE = [
    {"type": "file", "path": "README.md", "size": 1000},
    {"type": "file", "path": "imatrix_unsloth.gguf", "size": 13 * MB},
    {"type": "file", "path": "Qwen3-4B-Q4_K_M.gguf", "size": 100, "lfs": {"size": 2500 * MB}},
    {"type": "file", "path": "Qwen3-4B-UD-IQ1_S.gguf", "size": 100, "lfs": {"size": 1100 * MB}},
    {"type": "file", "path": "mmproj-F16.gguf", "size": 100, "lfs": {"size": 800 * MB}},
    {"type": "directory", "path": "BF16"},
    {"type": "file", "path": "BF16/Qwen3-4B-BF16-00001-of-00002.gguf", "size": 100, "lfs": {"size": 5000 * MB}},
]


class QuantTests(unittest.TestCase):
    def test_quant_labels(self):
        self.assertEqual(quant_of("Qwen3-4B-Q4_K_M.gguf"), "Q4_K_M")
        self.assertEqual(quant_of("Qwen3.8-27B-UD-IQ1_S.gguf"), "UD-IQ1_S")
        self.assertEqual(quant_of("model.Q8_0.gguf"), "Q8_0")
        self.assertEqual(quant_of("model-f16.gguf"), "F16")
        self.assertIsNone(quant_of("imatrix_unsloth.gguf"))

    def test_kv_from_model_info(self):
        info = {"general.architecture": "qwen3", "qwen3.block_count": 36, "qwen3.attention.head_count_kv": 8,
                "qwen3.attention.head_count": 32, "qwen3.attention.key_length": 128,
                "qwen3.attention.value_length": 128}
        self.assertAlmostEqual(kv_mb_per_token(info), 0.140625)
        self.assertEqual(estimate_need_mb(5000, 8192, 0.140625), int(5000 + 0.140625 * 8192 + 250))


class RepoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.hub = HuggingFaceHub()
        self.hub._session = MagicMock()
        self.hub._session.get.side_effect = lambda url, params=None, timeout=None: _resp(
            TREE if url.endswith("/tree/main") else INFO
        )

    def test_files(self):
        repo = self.hub.repo("unsloth/Qwen3-4B-GGUF", ctx=8192, memory={"available_mb": 3000},
                             downloaded={"hf.co/unsloth/qwen3-4b-gguf:ud-iq1_s"})
        self.assertEqual(repo["license"], "apache-2.0")
        files = {f["name"]: f for f in repo["files"]}
        self.assertNotIn("README.md", files)
        q4 = files["Qwen3-4B-Q4_K_M.gguf"]
        self.assertTrue(q4["supported"])
        self.assertEqual(q4["ollama_model"], "hf.co/unsloth/Qwen3-4B-GGUF:Q4_K_M")
        self.assertFalse(q4["fits"])
        iq1 = files["Qwen3-4B-UD-IQ1_S.gguf"]
        self.assertTrue(iq1["fits"])
        self.assertTrue(iq1["downloaded"])
        self.assertFalse(files["imatrix_unsloth.gguf"]["supported"])
        self.assertFalse(files["mmproj-F16.gguf"]["supported"])
        self.assertIn("составной", files["Qwen3-4B-BF16-00001-of-00002.gguf"]["reason"])
        # Сначала то, что можно скачать, по размеру.
        self.assertEqual(repo["files"][0]["name"], "Qwen3-4B-UD-IQ1_S.gguf")

    def test_search_kinds(self):
        self.hub._session.get.side_effect = None
        self.hub._session.get.return_value = _resp([{"id": "a/b-GGUF", "downloads": 5, "likes": 1}])
        items = self.hub.search("qwen", "embedding", "downloads")
        self.assertEqual(items[0]["id"], "a/b-GGUF")
        tags = [c.kwargs["params"]["pipeline_tag"] for c in self.hub._session.get.call_args_list]
        self.assertEqual(tags, ["feature-extraction", "sentence-similarity"])


class HfApiTests(unittest.TestCase):
    def test_endpoints(self):
        repo = make_repository()
        hub = HuggingFaceHub()
        hub._session = MagicMock()
        hub._session.get.side_effect = lambda url, params=None, timeout=None: _resp(
            TREE if url.endswith("/tree/main") else ([{"id": "unsloth/Qwen3-4B-GGUF"}] if url.endswith("/api/models") else INFO)
        )
        client = TestClient(create_app_with_repository(repo, hf_hub=hub))
        self.assertEqual(client.get("/hf/models", params={"q": "qwen"}).json()[0]["id"], "unsloth/Qwen3-4B-GGUF")
        body = client.get("/hf/model", params={"repo": "unsloth/Qwen3-4B-GGUF"}).json()
        self.assertEqual(body["id"], "unsloth/Qwen3-4B-GGUF")
        self.assertTrue(any(f["supported"] for f in body["files"]))

    def test_hub_error_is_502(self):
        repo = make_repository()
        hub = HuggingFaceHub()
        hub._session = MagicMock()
        hub._session.get.return_value = _resp({}, status=404)
        client = TestClient(create_app_with_repository(repo, hf_hub=hub))
        resp = client.get("/hf/model", params={"repo": "nope/nope"})
        self.assertEqual(resp.status_code, 502)
        self.assertIn("не найден", resp.json()["error"])


if __name__ == "__main__":
    unittest.main()

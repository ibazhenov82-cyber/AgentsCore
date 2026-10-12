"""
Управление локальными моделями Ollama («День 29»): каталог из `ollama
list`, запуск/остановка, «Конкурентный запуск», автозапуск перед ответом в
чате, HTTP API `/local-models`. Офлайн: HTTP Ollama заменён фейком.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from typing import Dict, Iterator, List, Optional

from fastapi.testclient import TestClient

from agents_core.api.app import create_app_with_repository
from agents_core.catalog import ModelCatalog
from agents_core.db import Database
from agents_core.local_models import LocalModelManager
from agents_core.providers.base import ChatResult, ChatUsage, StreamDelta
from agents_core.providers.ollama import OllamaProvider
from agents_core.repository import Repository

GB = 1024 ** 3


class FakeOllama(OllamaProvider):
    """Ollama в памяти: скачанные модели, загруженные модели, журнал вызовов."""

    def __init__(self) -> None:
        super().__init__("http://ollama.invalid")
        self.tags: List[dict] = [
            {"model": "qwen3:8b", "digest": "d1", "size": 5 * GB,
             "details": {"family": "qwen3", "parameter_size": "8.2B", "quantization_level": "Q4_K_M"}},
            {"model": "qwen3:8b-q8_0", "digest": "d2", "size": 9 * GB,
             "details": {"family": "qwen3", "parameter_size": "8.2B", "quantization_level": "Q8_0"}},
            {"model": "qwen3-embedding:0.6b", "digest": "d3", "size": GB, "details": {"family": "qwen3"}},
        ]
        self.loaded: Dict[str, int] = {}
        self.calls: List[tuple] = []

    def list_tags(self) -> List[dict]:
        return [dict(t) for t in self.tags]

    def running(self) -> List[dict]:
        return [
            {"model": name, "name": name, "size": 6 * GB, "size_vram": 6 * GB, "context_length": ctx}
            for name, ctx in self.loaded.items()
        ]

    def show(self, model_id: str, digest: Optional[str] = None) -> dict:
        caps = ["embedding"] if "embedding" in model_id else ["completion", "tools", "thinking"]
        return {"capabilities": caps, "details": {"family": "qwen3"}, "model_info": {"qwen3.context_length": 40960}}

    def load(self, model_id: str, num_ctx: Optional[int] = None) -> dict:
        self.calls.append(("load", model_id, num_ctx))
        self.loaded[model_id] = num_ctx or 4096
        return {"load_duration": 2_000_000_000}

    def unload(self, model_id: str) -> None:
        self.calls.append(("unload", model_id))
        self.loaded.pop(model_id, None)

    def pull(self, model_id, cancel=None):
        self.calls.append(("pull", model_id))
        if "broken" in model_id:
            from agents_core.providers import ProviderError
            raise ProviderError("pull access denied")
        yield {"status": "pulling manifest"}
        yield {"status": "pulling abc", "digest": "abc", "total": 2 * GB, "completed": GB}
        yield {"status": "pulling abc", "digest": "abc", "total": 2 * GB, "completed": 2 * GB}
        yield {"status": "verifying sha256 digest"}
        self.tags.append({"model": model_id, "digest": "new", "size": 2 * GB, "details": {}})
        yield {"status": "success"}

    def delete(self, model_id):
        self.calls.append(("delete", model_id))
        self.tags = [t for t in self.tags if t["model"] != model_id]

    @property
    def base_url(self):
        return "http://ollama.invalid"

    def _result(self) -> ChatResult:
        return ChatResult(content="Ответ", usage=ChatUsage(
            prompt_tokens=100, completion_tokens=50, total_tokens=150, eval_duration_ms=2000.0, load_duration_ms=5.0,
        ))

    def chat(self, model_id, messages, settings) -> ChatResult:
        self._before(model_id, settings)
        self.calls.append(("chat", model_id))
        return self._result()

    def stream_chat(self, model_id, messages, settings) -> Iterator[StreamDelta]:
        self._before(model_id, settings)
        self.calls.append(("chat", model_id))
        yield StreamDelta(content="Ответ")
        yield StreamDelta(done=True, result=self._result())


class OllamaOnlyRegistry:
    def __init__(self, provider: FakeOllama) -> None:
        self._provider = provider

    def get(self, name: str):
        return self._provider

    def all(self):
        return {"ollama": self._provider}


class NoCloudConfig:
    @staticmethod
    def configured_models():
        return []


def make_env():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)
    db = Database(path)
    provider = FakeOllama()
    registry = OllamaOnlyRegistry(provider)
    catalog = ModelCatalog(db, registry, NoCloudConfig)
    catalog.load_or_discover()
    manager = LocalModelManager(provider, catalog, db)
    repo = Repository(db, registry, catalog, local_models=manager)
    return path, db, provider, catalog, manager, repo


class LocalModelsTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.path, self.db, self.provider, self.catalog, self.manager, self.repo = make_env()

    def tearDown(self) -> None:
        os.unlink(self.path)


class CatalogTests(LocalModelsTestCase):
    def test_catalog_has_downloaded_chat_models_only(self):
        ids = sorted(m.id for m in self.catalog.list_models())
        self.assertEqual(ids, ["ollama:qwen3:8b", "ollama:qwen3:8b-q8_0"])
        self.assertTrue(all(m.is_local for m in self.catalog.list_models()))

    def test_new_download_appears_after_listing(self):
        self.provider.tags.append({"model": "qwen3:4b", "digest": "d4", "size": 3 * GB, "details": {}})
        self.manager.list_models()
        self.assertIsNotNone(self.catalog.get("ollama:qwen3:4b"))

    def test_removed_model_disappears(self):
        self.provider.tags = [t for t in self.provider.tags if t["model"] != "qwen3:8b-q8_0"]
        self.manager.list_models()
        self.assertIsNone(self.catalog.get("ollama:qwen3:8b-q8_0"))


class ManagerTests(LocalModelsTestCase):
    def test_list_shows_status_and_memory(self):
        self.provider.loaded["qwen3:8b"] = 8192
        items = {i["model"]: i for i in self.manager.list_models()}
        self.assertEqual(set(items), {"qwen3:8b", "qwen3:8b-q8_0", "qwen3-embedding:0.6b"})
        self.assertEqual(items["qwen3-embedding:0.6b"]["kind"], "embedding")
        self.assertIsNotNone(items["qwen3:8b-q8_0"]["need_mb"])
        self.assertIsNone(items["qwen3:8b"]["need_mb"])
        self.assertEqual(items["qwen3:8b"]["status"], "running")
        self.assertEqual(items["qwen3:8b"]["memory_mb"], 6144)
        self.assertEqual(items["qwen3:8b"]["gpu_percent"], 100)
        self.assertEqual(items["qwen3:8b"]["num_ctx"], 8192)
        self.assertEqual(items["qwen3:8b"]["quantization_level"], "Q4_K_M")
        self.assertEqual(items["qwen3:8b-q8_0"]["status"], "stopped")

    def test_plan_start_when_not_running(self):
        self.assertEqual(self.manager.plan("qwen3:8b"), [("start", "qwen3:8b")])

    def test_plan_nothing_when_running_with_same_context(self):
        self.provider.loaded["qwen3:8b"] = 8192
        self.assertEqual(self.manager.plan("qwen3:8b", 8192), [])
        self.assertEqual(self.manager.plan("qwen3:8b", None), [])

    def test_plan_restart_when_context_differs(self):
        self.provider.loaded["qwen3:8b"] = 4096
        self.assertEqual(self.manager.plan("qwen3:8b", 8192), [("restart", "qwen3:8b")])

    def test_models_run_side_by_side_without_competitive_start(self):
        self.provider.loaded["qwen3:8b-q8_0"] = 4096
        self.assertEqual(self.manager.plan("qwen3:8b"), [("start", "qwen3:8b")])

    def test_competitive_start_stops_other_chat_models_but_not_embeddings(self):
        self.provider.loaded["qwen3:8b-q8_0"] = 4096
        self.provider.loaded["qwen3-embedding:0.6b"] = 2048
        self.manager.set_competitive("qwen3:8b", True)
        self.assertEqual(self.manager.plan("qwen3:8b"), [("stop", "qwen3:8b-q8_0"), ("start", "qwen3:8b")])
        self.manager.ensure_ready("qwen3:8b")
        self.assertEqual(set(self.provider.loaded), {"qwen3:8b", "qwen3-embedding:0.6b"})

    def test_manual_start_runs_in_background(self):
        item = self.manager.start("qwen3:8b")
        self.assertIn(item["status"], ("starting", "running"))
        for _ in range(100):
            if self.manager.get("qwen3:8b")["status"] == "running":
                break
            time.sleep(0.02)
        self.assertEqual(self.manager.get("qwen3:8b")["status"], "running")

    def test_stop(self):
        self.provider.loaded["qwen3:8b"] = 4096
        item = self.manager.stop("qwen3:8b")
        self.assertEqual(item["status"], "stopped")
        self.assertIn(("unload", "qwen3:8b"), self.provider.calls)

    def test_competitive_flag_is_stored(self):
        self.manager.set_competitive("qwen3:8b-q8_0", True)
        items = {i["model"]: i for i in self.manager.list_models()}
        self.assertTrue(items["qwen3:8b-q8_0"]["competitive_start"])
        self.assertFalse(items["qwen3:8b"]["competitive_start"])


class EvictionTests(LocalModelsTestCase):
    def test_model_unloaded_by_ollama_is_reloaded_with_reason(self):
        self.manager.ensure_ready("qwen3:8b")  # запустили сами
        self.assertEqual(self.manager.plan("qwen3:8b"), [])
        # Ollama выгрузила чат-модель ради модели эмбеддингов (поиск в базе знаний).
        self.provider.loaded.pop("qwen3:8b")
        self.provider.loaded["qwen3-embedding:0.6b"] = 2048
        plan = self.manager.plan("qwen3:8b")
        self.assertEqual(plan, [("reload", "qwen3:8b")])
        text = self.manager.describe("reload", "qwen3:8b")
        self.assertIn("снова", text)
        self.assertIn("qwen3-embedding:0.6b", text)
        self.manager.ensure_ready("qwen3:8b")
        self.assertEqual(self.manager.plan("qwen3:8b"), [])

    def test_own_stop_is_not_eviction(self):
        self.manager.ensure_ready("qwen3:8b")
        self.manager.plan("qwen3:8b")
        self.manager.stop("qwen3:8b")
        self.assertEqual(self.manager.plan("qwen3:8b"), [("start", "qwen3:8b")])

    def test_chat_status_explains_reload(self):
        agent = self.repo.create_agent("Агент", model="ollama:qwen3:8b")
        chat = self.repo.create_chat(agent.id, "Чат")
        self.repo.send_message_blocking(chat.id, "Первый")
        self.provider.loaded.pop("qwen3:8b")
        self.provider.loaded["qwen3-embedding:0.6b"] = 2048
        events = list(self.repo.stream_message(chat.id, "Второй"))
        statuses = [e["status"] for e in events if e["type"] == "status"]
        self.assertTrue(any(s.startswith("Запускаю модель qwen3:8b снова") for s in statuses), statuses)


class ChatAutoStartTests(LocalModelsTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.agent = self.repo.create_agent("Агент", model="ollama:qwen3:8b")
        self.repo.update_agent_settings(self.agent.id, {"num_ctx": 8192})
        self.chat = self.repo.create_chat(self.agent.id, "Чат")

    def test_model_is_started_before_answer(self):
        events = list(self.repo.stream_message(self.chat.id, "Вопрос"))
        statuses = [e["status"] for e in events if e["type"] == "status"]
        self.assertIn("Запускаю модель qwen3:8b", statuses)
        self.assertEqual(self.provider.calls[0], ("load", "qwen3:8b", 8192))
        self.assertEqual(self.provider.calls.count(("load", "qwen3:8b", 8192)), 1)
        done = next(e for e in events if e["type"] == "done")
        self.assertEqual(json.loads(done["message"].metrics)["load_ms"], 2000)

    def test_running_model_is_not_restarted(self):
        self.provider.loaded["qwen3:8b"] = 8192
        self.repo.send_message_blocking(self.chat.id, "Вопрос")
        self.assertFalse(any(c[0] == "load" for c in self.provider.calls))

    def test_competitive_start_from_chat(self):
        self.provider.loaded["qwen3:8b-q8_0"] = 4096
        self.manager.set_competitive("qwen3:8b", True)
        events = list(self.repo.stream_message(self.chat.id, "Вопрос"))
        statuses = [e["status"] for e in events if e["type"] == "status"]
        self.assertIn("Останавливаю модель qwen3:8b-q8_0", statuses)
        self.assertNotIn("qwen3:8b-q8_0", self.provider.loaded)

    def test_start_failure_fails_answer(self):
        def broken_load(model_id, num_ctx=None):
            from agents_core.providers import ProviderError
            raise ProviderError("нет памяти")
        self.provider.load = broken_load  # type: ignore[assignment]
        events = list(self.repo.stream_message(self.chat.id, "Вопрос"))
        self.assertEqual(events[-1]["type"], "error")
        self.assertIn("нет памяти", events[-1]["message"])


def wait_for(predicate, timeout=2.0):
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


class PullDeleteTests(LocalModelsTestCase):
    def test_pull_adds_model_and_catalog_entry(self):
        self.manager.start_pull("hf.co/unsloth/Qwen3-4B-GGUF:Q4_K_M")
        self.assertTrue(wait_for(lambda: not self.manager.pulls()))
        self.assertIsNotNone(self.catalog.get("ollama:hf.co/unsloth/Qwen3-4B-GGUF:Q4_K_M"))

    def test_failed_pull_is_reported_and_dismissable(self):
        self.manager.start_pull("hf.co/x/broken:Q4_K_M")
        self.assertTrue(wait_for(lambda: any(p["status"] == "failed" for p in self.manager.pulls())))
        self.assertIn("pull access denied", self.manager.pulls()[0]["error"])
        self.manager.cancel_pull("hf.co/x/broken:Q4_K_M")
        self.assertEqual(self.manager.pulls(), [])

    def test_delete_running_model(self):
        self.provider.loaded["qwen3:8b"] = 4096
        self.manager.delete("qwen3:8b")
        self.assertIn(("unload", "qwen3:8b"), self.provider.calls)
        self.assertIn(("delete", "qwen3:8b"), self.provider.calls)
        self.assertIsNone(self.catalog.get("ollama:qwen3:8b"))

    def test_recommended_skips_downloaded(self):
        names = [r["model"] for r in self.manager.recommended()]
        self.assertNotIn("qwen3:8b", names)
        self.assertNotIn("qwen3-embedding:0.6b", names)
        self.assertIn("qwen3:1.7b", names)

    def test_probe_local_model(self):
        result = self.repo.probe_model("ollama:qwen3:8b")
        self.assertTrue(result["ok"])
        self.assertEqual(result["answer"], "Ответ")
        self.assertEqual(result["tokens_per_sec"], 25.0)
        self.assertIn(("load", "qwen3:8b", None), self.provider.calls)


class LocalModelsApiTests(LocalModelsTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.client = TestClient(create_app_with_repository(self.repo))

    def test_list(self):
        resp = self.client.get("/local-models")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["available"])
        self.assertEqual([m["model"] for m in body["models"]], ["qwen3-embedding:0.6b", "qwen3:8b", "qwen3:8b-q8_0"])
        self.assertEqual(body["pulls"], [])

    def test_start_stop_settings(self):
        resp = self.client.post("/local-models/start", json={"model": "qwen3:8b"})
        self.assertEqual(resp.status_code, 202)
        for _ in range(100):
            if "qwen3:8b" in self.provider.loaded:
                break
            time.sleep(0.02)
        resp = self.client.post("/local-models/stop", json={"model": "qwen3:8b"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "stopped")
        resp = self.client.put("/local-models/settings", json={"model": "qwen3:8b", "competitive_start": True})
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["competitive_start"])

    def test_models_endpoint_picks_up_new_download(self):
        self.provider.tags.append({"model": "qwen3:4b", "digest": "d4", "size": 3 * GB, "details": {}})
        ids = [m["id"] for m in self.client.get("/models").json()]
        self.assertIn("ollama:qwen3:4b", ids)

    def test_pull_delete_recommended_probe_endpoints(self):
        resp = self.client.post("/local-models/pull", json={"model": "qwen3:1.7b"})
        self.assertEqual(resp.status_code, 202)
        self.assertTrue(wait_for(lambda: not self.manager.pulls()))
        self.assertEqual(self.client.post("/local-models/delete", json={"model": "qwen3:1.7b"}).status_code, 204)
        rec = self.client.get("/local-models/recommended").json()
        self.assertIn("qwen3:1.7b", [r["model"] for r in rec])
        probe = self.client.post("/models/probe", json={"model": "ollama:qwen3:8b"}).json()
        self.assertTrue(probe["ok"])

    def test_unknown_model_404(self):
        resp = self.client.post("/local-models/start", json={"model": "llama3:70b"})
        self.assertEqual(resp.status_code, 404)

    def test_ollama_down(self):
        from agents_core.providers import ProviderError

        def down():
            raise ProviderError("connection refused")
        self.provider.list_tags = down  # type: ignore[assignment]
        body = self.client.get("/local-models").json()
        self.assertFalse(body["available"])
        self.assertIn("connection refused", body["error"])


if __name__ == "__main__":
    unittest.main()

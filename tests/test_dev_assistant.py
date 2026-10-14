"""«День 31» — ассистент разработчика: команды в чате, проект по ссылке,
база знаний проекта, подтверждение /commit и /undo.

Офлайн: mcp_server и knowledge_service подменены фейками с той же
поверхностью, что у настоящих клиентов (`MCPClient`, `KnowledgeClient`)."""

from __future__ import annotations

import json
import os
import unittest
from typing import Any, Dict, List, Optional

from agents_core.dev_assistant import filter_context, find_link, parse_command
from agents_core.repository import PreconditionFailedError
from tests.test_memory_and_profiles import _tool_call, make_tool_repo
from tests.test_repository import TEST_MODEL_ID

REPO_ID = "git:github.com/acme/widgets#main"
LOCAL_ID = "local:/projects/agents"


def _schema(*props: str, required=("repo_id",)) -> dict:
    return {"type": "object", "properties": {p: {"type": "string"} for p in ("repo_id",) + props},
            "required": list(required)}


TOOLS = {
    "repo_open": ("Открыть проект по ссылке", _schema("link", required=("link",)), False),
    "repo_status": ("Состояние репозитория", _schema(), False),
    "repo_log": ("История коммитов", _schema("path"), False),
    "repo_show": ("Коммит подробно", _schema("commit"), False),
    "repo_grep": ("Поиск по коду", _schema("pattern"), False),
    "repo_diff": ("Изменения (diff)", _schema("spec"), False),
    "repo_blame": ("Авторство строк", _schema("path"), False),
    "repo_list_files": ("Список файлов проекта", _schema("path"), False),
    "repo_read_file": ("Чтение файла проекта", _schema("path"), False),
    "repo_commit": ("Коммит изменений", _schema("message"), True),
    "repo_undo_commit": ("Откат коммита", _schema("commit"), True),
}


class FakeRepoMCP:
    """Инструменты «Репозиторий» mcp_server с заготовленными ответами."""

    def __init__(self):
        self.calls: List[tuple] = []
        self.status: Dict[str, Any] = {
            "branch": "main", "upstream": "origin/main", "ahead": 0, "behind": 1, "clean": False,
            "changed_files": 2, "staged": 0, "unstaged": 1, "untracked": 1,
            "head": {"sha": "a" * 40, "short": "aaaaaaa", "subject": "initial", "author": "Ivan", "date": "2026-10-01"},
            "files": [{"path": "app.py", "status": "изменён", "staged": False, "untracked": False},
                      {"path": "new.txt", "status": "новый", "staged": False, "untracked": True}],
        }
        self.diff = {"spec": None, "title": "незакоммиченные изменения", "files": [
            {"path": "app.py", "status": "modified", "additions": 1, "deletions": 1}],
            "untracked": ["new.txt"], "stats": {"files": 1, "additions": 1, "deletions": 1},
            "patch": "--- a/app.py\n+++ b/app.py\n-old\n+new\n", "truncated": False}
        self.kind = "remote"

    def list_tools(self):
        return []

    def list_tool_details(self) -> List[Dict[str, Any]]:
        return [{"name": n, "original_name": n, "server": "tools",
                 "group": "Репозиторий: запись" if w else "Репозиторий", "title": t, "description": "",
                 "parameters": p, "direct_only": w} for n, (t, p, w) in TOOLS.items()]

    def direct_only_tools(self) -> set:
        return {n for n, (_, _, w) in TOOLS.items() if w}

    def has_cached_tool(self, name: str) -> bool:
        return name in TOOLS

    def tool_group(self, name: str) -> Optional[str]:
        return "Репозиторий" if name in TOOLS else None

    def call_tool(self, name: str, arguments: Dict[str, Any], meta=None) -> Dict[str, Any]:
        self.calls.append((name, dict(arguments)))
        if name == "repo_open":
            link = arguments["link"]
            if "evil" in link:
                return {"error": "хост evil.com не разрешён"}
            local = link.startswith("/")
            return {"repo_id": LOCAL_ID if local else REPO_ID, "kind": "local" if local else "remote",
                    "name": "agents" if local else "acme/widgets", "path": link if local else "/data/repos/acme",
                    "branch": "main", "default_branch": "main", "remote": "https://github.com/acme/widgets.git",
                    "writable": local, "head": self.status["head"]}
        if name == "repo_status":
            return dict(self.status)
        if name == "repo_diff":
            return dict(self.diff)
        if name == "repo_log":
            return {"commits": [{"subject": "Добавить планировщик"}]}
        if name == "repo_commit":
            return {"commit": "b" * 40, "short": "bbbbbbb", "subject": arguments["message"].splitlines()[0],
                    "branch": "main", "files": arguments.get("files") or []}
        if name == "repo_undo_commit":
            return {"undone": arguments["commit"], "head": self.status["head"]}
        if name == "repo_read_file":
            return {"path": arguments.get("path"), "content": "1| def plan(): ..."}
        return {"ok": True}


class FakeKnowledge:
    def __init__(self):
        self.collections: List[dict] = []
        self.sources: Dict[str, List[dict]] = {}
        self.added: List[tuple] = []
        self.synced: List[str] = []
        self.queries: List[tuple] = []

    def list_collections(self):
        return list(self.collections)

    def get_collection(self, cid):
        return {"id": cid, "name": "Проект", "status": "ready", "stats": {"documents": 3, "chunks": 12}}

    def create_collection(self, name, description="", file_types=None):
        col = {"id": f"col{len(self.collections) + 1}", "name": name, "description": description}
        self.collections.append(col)
        return col

    def list_sources(self, cid):
        return self.sources.get(cid, [])

    def add_path_source(self, cid, path, include, title):
        self.added.append((cid, path, include))
        src = {"id": "src1", "source": {"type": "path", "path": path}}
        self.sources.setdefault(cid, []).append(src)
        return {"source": src, "job_id": "job1"}

    def update_source_include(self, sid, include):
        self.patched = (sid, include)
        return {}

    def sync_source(self, sid):
        self.synced.append(sid)
        return {"job_id": "job2"}

    def get_job(self, jid):
        return {"status": "completed"}

    def retrieve(self, query, collection_ids, top_k, score_threshold, **kw):
        self.queries.append((query, list(collection_ids)))
        return {"results": [{"chunk_id": "c1", "text": "Планировщик запускает задачи по cron.", "score": 0.9,
                             "document": {"id": "d1", "title": "README.md"},
                             "collection": {"id": collection_ids[0], "name": "Проект: acme/widgets"}}],
                "stages": {}}


class DevAssistantTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = make_tool_repo()
        self.provider = self.repo._tool_provider  # type: ignore[attr-defined]
        self.mcp = FakeRepoMCP()
        self.kb = FakeKnowledge()
        self.repo._mcp_client = self.mcp  # type: ignore[attr-defined]
        self.repo._knowledge = self.kb  # type: ignore[attr-defined]
        self.agent = self.repo.create_agent("Ассистент разработчика", model=TEST_MODEL_ID)
        self.chat = self.repo.create_chat(self.agent.id, "Проект")
        self.repo.update_chat_settings(self.chat.id, {"dev_commands_enabled": True})

    def tearDown(self) -> None:
        os.unlink(self.repo._db_path_for_cleanup)  # type: ignore[attr-defined]

    def send(self, text: str):
        user, answer = self.repo.send_message_blocking(self.chat.id, text)
        return user, answer

    @staticmethod
    def meta(message) -> dict:
        return json.loads(message.command) if message.command else {}


class ParsingTests(unittest.TestCase):
    def test_parse_command(self):
        self.assertEqual(parse_command("/ask как дела"), ("ask", "как дела"))
        self.assertEqual(parse_command("  /STATUS "), ("status", ""))
        self.assertIsNone(parse_command("/home/ivan/agents что это"))
        self.assertIsNone(parse_command("привет"))

    def test_find_link(self):
        self.assertEqual(find_link("смотри https://github.com/acme/widgets/tree/dev, там"),
                         "https://github.com/acme/widgets/tree/dev")
        self.assertIsNone(find_link("документация на https://example.com/a/b"))
        self.assertIsNone(find_link("https://github.com/acme"))
        self.assertEqual(find_link("/home/ivan/projects/agents"), "/home/ivan/projects/agents")
        self.assertEqual(find_link("проект C:\\projects\\agents ок"), "C:\\projects\\agents")


class DisabledTests(DevAssistantTestBase):
    def test_commands_off_text_goes_to_model(self):
        self.repo.update_chat_settings(self.chat.id, {"dev_commands_enabled": False})
        _, answer = self.send("/status")
        self.assertEqual(answer.content, "Готово!")
        self.assertIsNone(answer.command)
        self.assertEqual(self.mcp.calls, [])

    def test_plain_text_without_project_is_normal(self):
        _, answer = self.send("Привет")
        self.assertIsNone(answer.command)
        self.assertEqual(self.mcp.calls, [])


class LocalCommandTests(DevAssistantTestBase):
    def test_help_without_model(self):
        before = len(self.provider.calls_log)
        user, answer = self.send("/help")
        self.assertEqual(len(self.provider.calls_log), before)
        meta = self.meta(answer)
        self.assertEqual(meta["card"]["type"], "help")
        self.assertFalse(meta["context"])
        self.assertFalse(self.meta(user)["context"])
        names = {c["name"]: c for c in meta["card"]["commands"]}
        self.assertFalse(names["commit"]["available"])  # «Только чтение» по умолчанию
        self.assertIn("/ask", answer.content)

    def test_unknown_command(self):
        _, answer = self.send("/deploy")
        self.assertIn("Неизвестная команда", answer.content)

    def test_link_opens_project_and_kb(self):
        _, answer = self.send("https://github.com/acme/widgets")
        self.assertEqual(self.mcp.calls[0], ("repo_open", {"link": "https://github.com/acme/widgets"}))
        project = self.repo.get_chat_project(self.chat.id)
        self.assertEqual(project.repo_id, REPO_ID)
        self.assertEqual(project.collection_id, "col1")
        self.assertEqual(project.index_status, "indexing")
        cid, path, include = self.kb.added[0]
        self.assertEqual(path, "/data/repos/acme")
        self.assertIn("**/*.md", include)
        self.assertIn("acme/widgets", answer.content)
        self.assertEqual(self.meta(answer)["card"]["type"], "project")
        # Повторное открытие той же ссылки — синхронизация, а не новая база.
        self.send("/project https://github.com/acme/widgets")
        self.assertEqual(len(self.kb.collections), 1)
        self.assertEqual(self.kb.synced, ["src1"])

    def test_bad_link_reports_error(self):
        _, answer = self.send("/project https://evil.com/a/b")
        self.assertIn("не разрешён", answer.content)
        self.assertIsNone(self.repo.get_chat_project(self.chat.id))

    def test_status_and_diff_cards(self):
        self.send("https://github.com/acme/widgets")
        _, status = self.send("/status")
        card = self.meta(status)["card"]
        self.assertEqual(card["type"], "status")
        self.assertEqual(card["behind"], 1)
        self.assertEqual(card["index"]["documents"], 3)
        self.assertEqual(self.repo.get_chat_project(self.chat.id).index_status, "ready")
        self.assertIn("app.py", status.content)
        _, diff = self.send("/diff")
        self.assertEqual(self.meta(diff)["card"]["type"], "diff")
        self.assertIn("```diff", diff.content)
        self.assertEqual(self.mcp.calls[-1], ("repo_diff", {"repo_id": REPO_ID}))

    def test_diff_without_project(self):
        _, answer = self.send("/diff")
        self.assertIn("Проект не открыт", answer.content)


class AskTests(DevAssistantTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.send("https://github.com/acme/widgets")

    def test_plain_text_is_ask_with_rag_and_read_tools(self):
        self.provider.tool_calls_queue = [[_tool_call("c1", "repo_read_file", {"path": "app.py", "repo_id": "local:/etc"})]]
        self.provider.final_text = "Планировщик описан в README [1], код в `app.py:1`."
        _, answer = self.send("Как устроен планировщик?")
        meta = self.meta(answer)
        self.assertEqual(meta["name"], "ask")
        self.assertTrue(meta["context"])
        # Модель получила только инструменты чтения, без параметра repo_id.
        first = self.provider.calls_log[-2]
        self.assertTrue(any("[ПРОЕКТ]" in m.content for m in first if m.role == "system"))
        self.assertIn("Планировщик запускает задачи", first[-1].content)
        # repo_id подставил сервис — подменить его модель не может.
        self.assertIn(("repo_read_file", {"path": "app.py", "repo_id": REPO_ID}), self.mcp.calls)
        rag = json.loads(answer.rag)
        self.assertEqual(rag["cited"], [1])
        self.assertEqual(self.kb.queries[-1], ("Как устроен планировщик?", ["col1"]))

    def test_tools_offered_without_repo_id_and_write_tools(self):
        captured = {}
        original = self.provider._next_tool_calls

        def spy(settings):
            captured["tools"] = json.loads(settings.tools_json) if settings.tools_json else []
            return original(settings)

        self.provider._next_tool_calls = spy
        self.send("/ask что в README?")
        names = {t["function"]["name"] for t in captured["tools"]}
        self.assertIn("repo_read_file", names)
        self.assertFalse(names & {"repo_commit", "repo_undo_commit", "repo_open"})
        for tool in captured["tools"]:
            self.assertNotIn("repo_id", tool["function"]["parameters"]["properties"])

    def test_model_cannot_call_write_tool(self):
        self.provider.tool_calls_queue = [[_tool_call("c1", "repo_commit", {"message": "x"})]]
        self.send("/ask закоммить всё")
        self.assertFalse(any(c[0] == "repo_commit" for c in self.mcp.calls))

    def test_explain_requires_argument(self):
        _, answer = self.send("/explain")
        self.assertIn("Укажите", answer.content)

    def test_clear_boundary(self):
        self.send("Первый вопрос")
        _, cleared = self.send("/clear")
        self.assertTrue(self.meta(cleared)["boundary"])
        self.send("Второй вопрос")
        texts = [m.content for m in self.provider.calls_log[-1]]
        self.assertFalse(any("Первый вопрос" in t for t in texts))
        stats = self.repo.chat_stats(self.repo.get_chat(self.chat.id))
        self.assertGreater(stats["active_context_start_id"], cleared.id)

    def test_review_includes_diff(self):
        self.send("/review")
        last_user = self.provider.calls_log[-1][-1].content
        self.assertIn("код-ревью", last_user)
        self.assertIn("+new", last_user)

    def test_local_results_out_of_context(self):
        self.send("/status")
        self.send("Вопрос")
        texts = " ".join(m.content for m in self.provider.calls_log[-1])
        self.assertNotIn("Изменено файлов", texts)


class CommitTests(DevAssistantTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.send("/project /projects/agents")

    def test_read_only_blocks_commit(self):
        _, answer = self.send("/commit")
        self.assertIn("только чтения", answer.content)
        self.assertFalse(any(c[0] == "repo_commit" for c in self.mcp.calls))

    def test_commit_confirm_and_undo(self):
        self.repo.update_chat_settings(self.chat.id, {"dev_read_only": False})
        self.provider.final_text = "```\nОбновить app.py\n\n- новая логика\n```"
        _, proposal = self.send("/commit")
        confirm = self.meta(proposal)["confirm"]
        self.assertEqual(confirm["state"], "pending")
        self.assertEqual(confirm["message"], "Обновить app.py\n\n- новая логика")
        self.assertEqual(confirm["files"], ["app.py", "new.txt"])
        # Модель при составлении сообщения инструментов не получила.
        self.assertFalse(self.provider.calls_log[-1][0].content.startswith("[ПРОЕКТ]"))
        done = self.repo.confirm_command(self.chat.id, proposal.id, {"message": "Обновить app.py"})
        self.assertEqual(self.meta(done)["confirm"]["state"], "done")
        self.assertIn(("repo_commit", {"repo_id": LOCAL_ID, "message": "Обновить app.py",
                                       "files": ["app.py", "new.txt"]}), self.mcp.calls)
        self.assertEqual(self.repo.get_chat_project(self.chat.id).last_commit, "b" * 40)
        with self.assertRaises(PreconditionFailedError):
            self.repo.confirm_command(self.chat.id, proposal.id, {})
        # /undo — HEAD совпадает с коммитом /commit.
        self.mcp.status["head"] = {"sha": "b" * 40, "subject": "Обновить app.py"}
        _, undo = self.send("/undo")
        self.assertEqual(self.meta(undo)["confirm"]["action"], "undo")
        undone = self.repo.confirm_command(self.chat.id, undo.id)
        self.assertEqual(self.meta(undone)["confirm"]["state"], "done")
        self.assertIsNone(self.repo.get_chat_project(self.chat.id).last_commit)

    def test_confirm_refused_when_read_only_turned_back_on(self):
        self.repo.update_chat_settings(self.chat.id, {"dev_read_only": False})
        _, proposal = self.send("/commit")
        self.repo.update_chat_settings(self.chat.id, {"dev_read_only": True})
        with self.assertRaises(PreconditionFailedError):
            self.repo.confirm_command(self.chat.id, proposal.id)

    def test_reject(self):
        self.repo.update_chat_settings(self.chat.id, {"dev_read_only": False})
        _, proposal = self.send("/commit")
        rejected = self.repo.reject_command(self.chat.id, proposal.id)
        self.assertEqual(self.meta(rejected)["confirm"]["state"], "cancelled")

    def test_commit_refused_for_remote_project(self):
        self.repo.update_chat_settings(self.chat.id, {"dev_read_only": False})
        self.send("https://github.com/acme/widgets")
        _, answer = self.send("/commit")
        self.assertIn("локальной рабочей копии", answer.content)


class FilterTests(unittest.TestCase):
    def test_filter_context(self):
        from agents_core.models import Message

        def m(i, cmd=None):
            return Message(id=i, chat_id="c", role="user", content=str(i), created_at=0,
                           command=json.dumps(cmd) if cmd else None)

        msgs = [m(1), m(2, {"context": False}), m(3, {"boundary": True, "context": False}), m(4), m(5, {"context": False})]
        self.assertEqual([x.id for x in filter_context(msgs)], [4])


class ApiTests(DevAssistantTestBase):
    def test_dev_commands_endpoint_and_chat_project(self):
        from fastapi.testclient import TestClient

        from agents_core.api.app import create_app_with_repository

        client = TestClient(create_app_with_repository(self.repo))
        self.send("https://github.com/acme/widgets")
        data = client.get(f"/chats/{self.chat.id}/dev/commands").json()
        self.assertTrue(data["enabled"])
        self.assertEqual(data["project"]["name"], "acme/widgets")
        commit = next(c for c in data["commands"] if c["name"] == "commit")
        self.assertFalse(commit["available"])
        chat = client.get(f"/chats/{self.chat.id}").json()
        self.assertEqual(chat["project"]["repo_id"], REPO_ID)
        self.assertTrue(chat["settings"]["dev_commands_enabled"])
        self.assertTrue(chat["settings"]["dev_read_only"])
        messages = client.get(f"/chats/{self.chat.id}/messages").json()
        self.assertIn("project", messages[-1]["command"])
        resp = client.post(f"/chats/{self.chat.id}/messages/{messages[-1]['id']}/confirm", json={})
        self.assertEqual(resp.status_code, 409)


if __name__ == "__main__":
    unittest.main()


# ---- «День 32»: ревью PR из CI ------------------------------------------------

REVIEW_TEXT = """### Кратко
PR меняет функцию plan.
### Потенциальные баги
- `app.py:2` — возвращается строка вместо числа → вернуть int [1]
### Архитектурные проблемы
- нет
### Рекомендации
- добавить тест на plan
- обновить README
"""


class ReviewParsingTests(unittest.TestCase):
    def test_parse_review_sections(self):
        from agents_core.dev_assistant import parse_review

        parsed = parse_review(REVIEW_TEXT)
        self.assertEqual(parsed["summary"], "PR меняет функцию plan.")
        self.assertEqual(len(parsed["bugs"]), 1)
        self.assertEqual(parsed["architecture"], [])
        self.assertEqual(parsed["recommendations"], ["добавить тест на plan", "обновить README"])
        bold = parse_review("**Потенциальные баги:**\n1. ошибка\nпродолжение\n**Рекомендации**\n- тест")
        self.assertEqual(bold["bugs"], ["ошибка продолжение"])

    def test_review_queries_from_diff(self):
        from agents_core.dev_assistant import review_queries

        queries = review_queries({"files": [{"path": "agents_core/runs.py"}], "patch":
                                  "@@ -1,2 +1,3 @@ class RunManager:\n+    def start_review(self):\n"})
        self.assertIn("agents core runs", queries[0])
        self.assertIn("RunManager", queries[1])
        self.assertIn("start_review", queries[1])


class CiReviewTests(DevAssistantTestBase):
    def setUp(self) -> None:
        super().setUp()
        from fastapi.testclient import TestClient

        from agents_core.api.app import create_app_with_repository

        self.reviewer = self.repo.create_agent("Ревьюер", model=TEST_MODEL_ID)
        self.client = TestClient(create_app_with_repository(self.repo))
        self.client.__enter__()
        self.provider.final_text = REVIEW_TEXT

    def tearDown(self) -> None:
        self.client.__exit__(None, None, None)
        super().tearDown()

    def _start(self, sha="a" * 40, **extra):
        body = {"agent_id": self.reviewer.id, "repo": "https://github.com/acme/widgets", "pr": 12,
                "base": "main", "head_sha": sha, "title": "Планировщик v2", **extra}
        resp = self.client.post("/dev/reviews", json=body)
        self.assertEqual(resp.status_code, 202, resp.text)
        return resp.json()

    def _wait(self, review_id):
        import time

        for _ in range(100):
            data = self.client.get(f"/dev/reviews/{review_id}").json()
            if data["status"] != "running":
                return data
            time.sleep(0.05)
        self.fail("ревью не завершилось")

    def test_review_flow(self):
        started = self._start()
        self.assertTrue(started["created_chat"])
        self.assertEqual(started["key"], "github.com/acme/widgets#12")
        result = self._wait(started["review_id"])
        self.assertEqual(result["status"], "done", result)
        self.assertEqual(result["bugs"], ["`app.py:2` — возвращается строка вместо числа → вернуть int [1]"])
        self.assertEqual(len(result["recommendations"]), 2)
        self.assertEqual(result["files"][0]["path"], "app.py")
        self.assertEqual(result["sources"][0]["title"], "README.md")
        self.assertIn("## AI-ревью · коммит aaaaaaa", result["comment_markdown"])
        self.assertIn("### Потенциальные баги", result["comment_markdown"])
        # mcp_server: проект — базовая ветка, diff — PR.
        self.assertIn(("repo_open", {"link": "https://github.com/acme/widgets/tree/main"}), self.mcp.calls)
        self.assertIn(("repo_diff", {"repo_id": REPO_ID, "spec": "pr:12", "max_chars": 24000}), self.mcp.calls)
        # Поиск по базе знаний — запросами по diff (файлы и функции).
        self.assertTrue(any("app" in q for q, _ in self.kb.queries))
        # Чат PR: ключ, источник, команды включены, только чтение.
        chat = self.repo.get_chat(started["chat_id"])
        self.assertEqual(chat.external_key, "github.com/acme/widgets#12")
        self.assertEqual(chat.source, "ci")
        self.assertEqual(chat.title, "PR acme/widgets#12: Планировщик v2")
        self.assertTrue(chat.settings.dev_commands_enabled)
        self.assertTrue(chat.settings.dev_read_only)
        user = [m for m in self.repo.list_messages(chat.id) if m.role == "user"][0]
        self.assertEqual(user.source, "ci")
        self.assertEqual(user.content, "/review pr:12 https://github.com/acme/widgets/tree/main")

    def test_same_commit_is_idempotent_new_commit_reuses_chat(self):
        first = self._start()
        self._wait(first["review_id"])
        calls = len(self.provider.calls_log)
        again = self._start()
        self.assertEqual(again["review_id"], first["review_id"])
        self.assertFalse(again["created_chat"])
        self.assertEqual(len(self.provider.calls_log), calls)
        newer = self._start(sha="b" * 40)
        self.assertNotEqual(newer["review_id"], first["review_id"])
        self.assertEqual(newer["chat_id"], first["chat_id"])
        self.assertEqual(self._wait(newer["review_id"])["status"], "done")
        # Другой PR — другой чат; другой агент — свой чат того же PR.
        other = self._start(pr=13)
        self.assertNotEqual(other["chat_id"], first["chat_id"])
        self._wait(other["review_id"])

    def test_agent_by_name_and_errors(self):
        body = {"agent": "ревьюер", "repo": "https://github.com/acme/widgets", "pr": 1}
        resp = self.client.post("/dev/reviews", json=body)
        self.assertEqual(resp.status_code, 202, resp.text)
        self._wait(resp.json()["review_id"])
        self.assertEqual(self.client.post("/dev/reviews", json={**body, "agent": "нет такого"}).status_code, 404)
        self.assertEqual(self.client.post("/dev/reviews", json={**body, "repo": "ftp://x"}).status_code, 400)
        self.assertEqual(self.client.post("/dev/reviews", json={**body, "head_sha": "zz"}).status_code, 400)
        self.assertEqual(self.client.get("/dev/reviews/unknown").status_code, 404)

    def test_no_changes_is_failed(self):
        self.mcp.diff = {"title": "изменения PR #12", "files": [], "untracked": [], "patch": ""}
        result = self._wait(self._start()["review_id"])
        self.assertEqual(result["status"], "failed")
        self.assertIn("Нет изменений", result["error"])

    def test_concurrent_find_or_create_single_chat(self):
        import threading

        from agents_core.ci_review import CiReviewService

        service = CiReviewService(self.repo, None)
        results = []
        threads = [threading.Thread(target=lambda: results.append(
            service.find_or_create_chat(self.reviewer, "github.com/acme/widgets#99", "acme/widgets", 99, None)))
            for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len({chat.id for chat, _ in results}), 1)
        self.assertEqual(sum(1 for _, created in results if created), 1)

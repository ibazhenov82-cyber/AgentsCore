"""
Проверка, что Pydantic-схемы (`agents_core.schemas`) и преобразователи
(`agents_core.api.converters`) корректно строятся из настоящих объектов,
возвращаемых `Repository` — без использования FastAPI (нужен только
pydantic, который в отличие от fastapi доступен в песочнице).
"""

from __future__ import annotations

import unittest

from agents_core.api.converters import (
    agent_out,
    chat_out,
    default_settings_out,
    long_term_memory_out,
    message_out,
    model_info_out,
    profile_out,
    working_memory_out,
)
from tests.test_repository import TEST_MODEL_ID, make_repository


class SchemasTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = make_repository()

    def tearDown(self) -> None:
        import os
        try:
            os.unlink(self.repo._db_path_for_cleanup)  # type: ignore[attr-defined]
        except OSError:
            pass

    def test_default_settings_out_roundtrip(self):
        out = default_settings_out(self.repo.get_default_settings())
        dumped = out.model_dump()
        self.assertIn("model", dumped)
        self.assertIn("summary_prompt", dumped)

    def test_agent_and_chat_out(self):
        agent = self.repo.create_agent("Тестовый агент", model=TEST_MODEL_ID)
        out = agent_out(agent)
        self.assertEqual(out.settings.model, TEST_MODEL_ID)

        chat = self.repo.create_chat(agent.id, "Чат 1")
        self.repo.send_message_blocking(chat.id, "Привет")
        chat = self.repo.get_chat(chat.id)
        stats = self.repo.chat_stats(chat)
        chat_schema = chat_out(chat, stats)
        self.assertEqual(chat_schema.stats.total_tokens, stats["total_tokens"])
        self.assertEqual(chat_schema.settings.model, TEST_MODEL_ID)

    def test_message_out(self):
        agent = self.repo.create_agent("A", model=TEST_MODEL_ID)
        chat = self.repo.create_chat(agent.id, "C1")
        user_msg, assistant_msg = self.repo.send_message_blocking(chat.id, "Привет")
        out = message_out(assistant_msg)
        self.assertEqual(out.role, "assistant")
        self.assertGreater(out.total_tokens, 0)

    def test_model_info_out(self):
        model = self.repo.list_models()[0]
        out = model_info_out(model)
        self.assertEqual(out.id, TEST_MODEL_ID)

    def test_chat_out_exposes_active_profile_id(self):
        agent = self.repo.create_agent("A", model=TEST_MODEL_ID)
        chat = self.repo.create_chat(agent.id, "C1")
        profile = self.repo.create_profile("Профиль")
        chat = self.repo.set_chat_active_profile(chat.id, profile.id)
        out = chat_out(chat, self.repo.chat_stats(chat))
        self.assertEqual(out.active_profile_id, profile.id)

    def test_working_and_long_term_memory_out(self):
        agent = self.repo.create_agent("A", model=TEST_MODEL_ID)
        chat = self.repo.create_chat(agent.id, "C1")
        wm = self.repo.save_working_memory(chat.id, "k", "v")
        out = working_memory_out(wm)
        self.assertEqual(out.source, "manual")

        ltm = self.repo.save_long_term_memory(agent.id, "knowledge", "k2", "v2")
        out2 = long_term_memory_out(ltm)
        self.assertEqual(out2.category, "knowledge")

    def test_profile_out_roundtrip(self):
        import json

        from agents_core.skills import shopping_demo

        profile = self.repo.create_profile(
            "Покупки",
            skills_json=json.dumps(shopping_demo.TOOL_DEFS, ensure_ascii=False),
            orchestration_prompt=shopping_demo.DEFAULT_ORCHESTRATION_PROMPT,
        )
        out = profile_out(profile)
        dumped = out.model_dump()
        self.assertIn("search_products", dumped["skills_json"])
        self.assertIsNotNone(out.orchestration_prompt)


if __name__ == "__main__":
    unittest.main()

"""
Тесты agents_core.providers.ollama.OllamaProvider.discover_model — в
частности, выбор правильного ключа "*.context_length" из `model_info`,
возвращаемого `/api/show`, когда таких ключей несколько.

Баг, который эти тесты фиксируют: пользователь сообщил, что для
qwen3:0.6b (реальный размер окна — 40960 токенов) процент заполнения
контекста считался так, будто размер окна намного больше — эффект
совпадает с тем, что в `model_info` мог по ошибке быть взят "не тот"
ключ, оканчивающийся на "context_length" (например обобщённый
"general.context_length", если он присутствует, вместо архитектурного
"qwen3.context_length"). discover_model должен предпочитать ключ,
явно соответствующий family модели.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from agents_core.providers.ollama import OllamaProvider


def _fake_show_response(model_info: dict, family: str = "qwen3") -> MagicMock:
    resp = MagicMock()
    resp.ok = True
    resp.json.return_value = {
        "model_info": model_info,
        "details": {"family": family},
    }
    return resp


class OllamaDiscoverModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = OllamaProvider(base_url="http://localhost:11434")

    def test_prefers_family_specific_context_length_over_generic_key(self):
        # "general.context_length" идёт раньше по ключам словаря (и по
        # алфавиту), но реальный размер окна для этой модели — это
        # "qwen3.context_length".
        model_info = {
            "general.architecture": "qwen3",
            "general.context_length": 131072,
            "qwen3.context_length": 40960,
        }
        with patch.object(self.provider, "_session") as session:
            session.post.return_value = _fake_show_response(model_info)
            caps = self.provider.discover_model("qwen3:0.6b")
        self.assertEqual(caps.context_window, 40960)

    def test_falls_back_to_first_matching_key_when_no_family_specific_key(self):
        model_info = {
            "general.architecture": "qwen3",
            "some.other.context_length": 32768,
        }
        with patch.object(self.provider, "_session") as session:
            session.post.return_value = _fake_show_response(model_info)
            caps = self.provider.discover_model("qwen3:0.6b")
        self.assertEqual(caps.context_window, 32768)

    def test_uses_fallback_constant_when_no_context_length_key_present(self):
        model_info = {"general.architecture": "qwen3"}
        with patch.object(self.provider, "_session") as session:
            session.post.return_value = _fake_show_response(model_info)
            caps = self.provider.discover_model("qwen3:0.6b")
        self.assertEqual(caps.context_window, 32768)  # _QWEN3_FALLBACK_CONTEXT_WINDOW


if __name__ == "__main__":
    unittest.main()

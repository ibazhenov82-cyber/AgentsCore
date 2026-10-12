"""
Тесты `agents_core.invariant_checks` — код-уровневая проверка ответа модели
("Инварианты", новое ТЗ, п.2): прямой перенос примера ТЗ (`Invariant.check`,
`StackOnly`) на эвристические чекеры, разбирающие свободный `rule_text`
("Разрешено: ...", "Нельзя: ...", "Запрещено: ...") — категории без чекера
(в т.ч. не указанная) не проверяются вовсе, соблюдение доверяется модели.
Полностью офлайн, чистые функции — не требует Repository/БД.
"""

from __future__ import annotations

import unittest

from agents_core.invariant_checks import (
    CHECKERS,
    ForbiddenTermsChecker,
    StackConstraintChecker,
    format_violation_warning,
    validate_response,
)
from agents_core.models import Invariant


def _inv(kind, rule_text, title="Правило", is_active=True) -> Invariant:
    return Invariant(id="inv-1", title=title, rule_text=rule_text, kind=kind, is_active=is_active)


class CheckersRegistryTests(unittest.TestCase):
    def test_only_three_categories_have_a_checker(self):
        self.assertEqual(set(CHECKERS.keys()), {"stack_constraint", "architecture", "tech_decision"})

    def test_architecture_and_tech_decision_use_forbidden_terms_checker_by_analogy(self):
        """ТЗ: "для остальных категорий типа 'Архитектура', 'Техническое
        решение' следует сделать проверку по аналогии" — со StackOnly, но по
        ЗАПРЕЩЁННЫМ, а не разрешённым терминам."""
        self.assertIsInstance(CHECKERS["architecture"], ForbiddenTermsChecker)
        self.assertIsInstance(CHECKERS["tech_decision"], ForbiddenTermsChecker)
        self.assertIsInstance(CHECKERS["stack_constraint"], StackConstraintChecker)


class StackConstraintCheckerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.checker = CHECKERS["stack_constraint"]

    def test_flags_disallowed_stack_terms(self):
        inv = _inv("stack_constraint", "Разрешено: Kotlin, Ktor", title="StackOnly")
        reason = self.checker.check(inv, "Предлагаю решение на Spring Boot и Java.")
        self.assertIsNotNone(reason)
        self.assertIn("Spring Boot", reason)
        self.assertIn("Java", reason)
        self.assertIn("Разрешено: Kotlin, Ktor", reason)

    def test_passes_when_only_allowed_terms_are_mentioned(self):
        inv = _inv("stack_constraint", "Разрешено: Kotlin, Ktor")
        self.assertIsNone(self.checker.check(inv, "Предлагаю решение на Ktor."))

    def test_passes_when_response_mentions_no_known_stack_terms(self):
        inv = _inv("stack_constraint", "Разрешено: Kotlin, Ktor")
        self.assertIsNone(self.checker.check(inv, "Нужно уточнить требования у заказчика."))

    def test_no_structure_in_rule_text_is_silently_skipped(self):
        """Если автор инварианта не использовал "Разрешено: ..." — чекер не
        находит структуры и пропускает проверку ИМЕННО для этого инварианта
        (не ошибка, соблюдение остаётся на совести модели)."""
        inv = _inv("stack_constraint", "Используем современный стек без Java")
        self.assertIsNone(self.checker.check(inv, "Предлагаю решение на Java."))


class ForbiddenTermsCheckerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.checker = CHECKERS["architecture"]

    def test_flags_forbidden_term_present_in_response(self):
        inv = _inv("architecture", "Нельзя: монолит", title="OnlyMicroservices")
        reason = self.checker.check(inv, "Соберём это как единый монолит для простоты.")
        self.assertIsNotNone(reason)
        self.assertIn("монолит", reason)

    def test_passes_when_forbidden_term_absent(self):
        inv = _inv("architecture", "Нельзя: монолит")
        self.assertIsNone(self.checker.check(inv, "Разобьём на независимые сервисы."))

    def test_recognizes_zapreshcheno_phrasing_too(self):
        inv = _inv("tech_decision", "Запрещено: REST")
        reason = self.checker.check(inv, "Используем REST API поверх HTTP.")
        self.assertIsNotNone(reason)
        self.assertIn("REST", reason)


class ValidateResponseTests(unittest.TestCase):
    def test_returns_empty_list_when_no_violations(self):
        invariants = [_inv("stack_constraint", "Разрешено: Kotlin")]
        self.assertEqual(validate_response("Пишем на Kotlin.", invariants), [])

    def test_collects_violations_across_multiple_invariants(self):
        invariants = [
            _inv("stack_constraint", "Разрешено: Kotlin", title="StackOnly"),
            _inv("architecture", "Нельзя: монолит", title="OnlyMicroservices"),
        ]
        violations = validate_response("Пишем монолит на Java.", invariants)
        self.assertEqual(len(violations), 2)
        titles = {inv.title for inv, _reason in violations}
        self.assertEqual(titles, {"StackOnly", "OnlyMicroservices"})

    def test_inactive_invariant_is_skipped(self):
        invariants = [_inv("stack_constraint", "Разрешено: Kotlin", is_active=False)]
        self.assertEqual(validate_response("Пишем на Java.", invariants), [])

    def test_unrecognized_or_missing_category_is_skipped(self):
        invariants = [
            _inv("business_rule", "Разрешено: только праздничные скидки"),
            _inv(None, "Разрешено: Kotlin"),
        ]
        self.assertEqual(validate_response("Скидка 90% без повода, пишем на Java.", invariants), [])


class FormatViolationWarningTests(unittest.TestCase):
    def test_matches_tz_example_shape(self):
        inv = _inv("stack_constraint", "Разрешено: Kotlin, Ktor", title="StackOnly")
        violations = validate_response("Предлагаю решение на Spring Boot и Java.", [inv])
        text = format_violation_warning(violations)
        self.assertTrue(text.startswith("⚠ Нарушение инварианта «StackOnly»"))
        self.assertIn("Spring Boot", text)
        self.assertIn("Java", text)
        self.assertIn("Разрешено: Kotlin, Ktor", text)

    def test_joins_multiple_violations_with_blank_line(self):
        invariants = [
            _inv("stack_constraint", "Разрешено: Kotlin", title="StackOnly"),
            _inv("architecture", "Нельзя: монолит", title="OnlyMicroservices"),
        ]
        violations = validate_response("Монолит на Java.", invariants)
        text = format_violation_warning(violations)
        self.assertIn("\n\n", text)
        self.assertIn("StackOnly", text)
        self.assertIn("OnlyMicroservices", text)


if __name__ == "__main__":
    unittest.main()

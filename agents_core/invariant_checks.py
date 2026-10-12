"""
agents_core.invariant_checks
==============================

"Инварианты" (новое ТЗ, раздел "Инварианты", п.2) — программная проверка
СВЕРХ обычной инъекции текста в промпт, для категорий (`Invariant.kind`),
которые код умеет проверять формально. Прямой перенос примера ТЗ
(`abstract class Invariant { check(response): Boolean }`, `StackOnly`) —
каждой распознаваемой категории (см. `CHECKERS`) соответствует эвристический
чекер, разбирающий `rule_text` инварианта на "разрешено"/"запрещено" списки
терминов. Категории вне `CHECKERS` (сейчас — "business_rule", и вообще не
указанная категория) программной проверке не подвергаются — соблюдение
полностью доверяется модели через обычную инъекцию текста в промпт (см.
`Repository._build_task_and_invariant_context`) — именно так в ТЗ
сформулировано явно: "для остальных категорий... доверяем LLM".

Это ЛУЧШЕЕ ПРИБЛИЖЕНИЕ, а не строгий парсер: правило пишется человеком в
свободной форме на экране инвариантов, а не структурированным списком
терминов (в отличие от Kotlin-примера `StackOnly(allowed: Set<String>)`, где
разрешённый список — это отдельное типизированное поле). Чекер ищет типовые
обороты ("Разрешено: ...", "Запрещено: ...", "Нельзя: ...") в `rule_text` —
если автор инварианта не использовал ни один из них, чекер не находит
структуры и молча пропускает проверку конкретно для ЭТОГО инварианта
(соблюдение остаётся на совести модели) — это ожидаемый, а не ошибочный
исход, и он не отличается по интерфейсу от "категория не распознана".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .models import Invariant, INVARIANT_KIND_LABELS

#: Известные технологии/стеки — по ним StackConstraintChecker ищет, о каких
#: терминах вообще идёт речь в ответе модели. Термин, которого нет в этом
#: списке, не будет ни "разрешённым", ни нарушением — намеренно узкий,
#: детерминированный список, а не попытка распознавать произвольные названия
#: технологий (это резко увеличило бы число ложных срабатываний).
KNOWN_STACK_TERMS = [
    "Kotlin Multiplatform", "Kotlin", "Java", "Ktor", "Spring Boot", "Spring",
    "Python", "Django", "Flask", "FastAPI", "Node.js", "Node", "Express",
    "TypeScript", "JavaScript", "React", "Vue", "Angular", "Svelte",
    "Go", "Golang", "Rust", "C#", ".NET", "PHP", "Laravel", "Ruby on Rails",
    "Ruby", "Swift", "SwiftUI", "Compose", "Jetpack Compose", "Android SDK",
    "MongoDB", "PostgreSQL", "MySQL", "SQLite", "Redis", "GraphQL", "gRPC",
]

_ALLOWED_RE = re.compile(r"разрешен[оы]?\s*:?\s*(.+)", re.IGNORECASE)
_FORBIDDEN_RE = re.compile(r"(?:запрещен[оы]?|нельзя)\s*:?\s*(.+)", re.IGNORECASE)


def _split_terms(text: str) -> List[str]:
    # Останавливаемся на конце предложения/строки — список терминов обычно
    # не длиннее одного предложения, дальше может идти обычный текст.
    head = re.split(r"[.\n]", text, maxsplit=1)[0]
    parts = re.split(r",|\bи\b", head)
    return [p.strip(" .;") for p in parts if p.strip(" .;")]


def _extract_terms(rule_text: str, pattern: "re.Pattern[str]") -> List[str]:
    match = pattern.search(rule_text)
    if not match:
        return []
    return _split_terms(match.group(1))


def _find_known_terms(text: str, known: List[str]) -> List[str]:
    lowered = text.lower()
    return [term for term in known if term.lower() in lowered]


@dataclass
class InvariantChecker:
    kind: str

    def check(self, invariant: Invariant, response_text: str) -> Optional[str]:
        """Возвращает описание нарушения, если правило нарушено формально
        распознаваемым образом, иначе `None` — в т.ч. когда в `rule_text` не
        нашлось структуры для проверки (тогда соблюдение остаётся на
        совести модели, как и для нераспознанных категорий)."""
        raise NotImplementedError


class StackConstraintChecker(InvariantChecker):
    """Аналог `StackOnly` из ТЗ: ищет "Разрешено: ..." в тексте правила,
    сравнивает с известными терминами стека (`KNOWN_STACK_TERMS`), найденными
    в ответе модели."""

    def __init__(self) -> None:
        super().__init__(kind="stack_constraint")

    def check(self, invariant: Invariant, response_text: str) -> Optional[str]:
        allowed = _extract_terms(invariant.rule_text, _ALLOWED_RE)
        if not allowed:
            return None
        allowed_lower = {a.lower() for a in allowed}
        mentioned = _find_known_terms(response_text, KNOWN_STACK_TERMS)
        disallowed = sorted({t for t in mentioned if t.lower() not in allowed_lower})
        if not disallowed:
            return None
        return f"{', '.join(disallowed)} не входят в разрешённый стек. Разрешено: {', '.join(allowed)}."


class ForbiddenTermsChecker(InvariantChecker):
    """Проверка "по аналогии со StackOnly" для категорий, где естественнее
    искать список ЗАПРЕЩЁННОГО, а не разрешённого ("Архитектура",
    "Техническое решение" — по замечанию ТЗ: "для остальных категорий...
    следует сделать проверку по аналогии"). Ищет "Нельзя: ..."/"Запрещено:
    ..." в правиле и проверяет, не встречаются ли эти термины дословно в
    ответе модели."""

    def check(self, invariant: Invariant, response_text: str) -> Optional[str]:
        forbidden = _extract_terms(invariant.rule_text, _FORBIDDEN_RE)
        if not forbidden:
            return None
        lowered_response = response_text.lower()
        found = [t for t in forbidden if t.lower() in lowered_response]
        if not found:
            return None
        return f"В ответе использовано запрещённое правилом: {', '.join(found)}."


#: Категории, для которых есть формальный чекер — остальные (в т.ч.
#: "business_rule" и вообще не указанная категория) проверяются только через
#: обычную инъекцию текста в промпт, программно не валидируются (ТЗ:
#: "для остальных категорий... доверяем LLM").
CHECKERS: Dict[str, InvariantChecker] = {
    "stack_constraint": StackConstraintChecker(),
    "architecture": ForbiddenTermsChecker(kind="architecture"),
    "tech_decision": ForbiddenTermsChecker(kind="tech_decision"),
}


def validate_response(response_text: str, invariants: List[Invariant]) -> List[Tuple[Invariant, str]]:
    """Python-аналог `fun validateResponse` из ТЗ. Вместо `ValidationResult`
    (Pass/Fail с полем `violations: List<String>`) возвращает список пар
    (нарушенный инвариант, причина) — пустой список означает Pass."""
    violations: List[Tuple[Invariant, str]] = []
    for inv in invariants:
        if not inv.is_active:
            continue
        checker = CHECKERS.get(inv.kind or "")
        if checker is None:
            continue  # нераспознанная категория — доверяем модели, см. докстринг модуля
        reason = checker.check(inv, response_text)
        if reason:
            violations.append((inv, reason))
    return violations


def format_violation_warning(violations: List[Tuple[Invariant, str]]) -> str:
    """Текст предупреждения для интерфейса чата, отображаемый ПЕРЕД ответом
    модели — формат из ТЗ:
    `⚠ Нарушение инварианта StackOnly.\\n<причина>\\nРазрешено: ...`.
    По одному абзацу на каждое нарушение (обычно оно одно)."""
    blocks = []
    for inv, reason in violations:
        category = INVARIANT_KIND_LABELS.get(inv.kind or "", inv.kind or "")
        header = f"⚠ Нарушение инварианта «{inv.title}»" + (f" ({category})" if category else "") + "."
        blocks.append(f"{header}\n{reason}")
    return "\n\n".join(blocks)

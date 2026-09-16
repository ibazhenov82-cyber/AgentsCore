"""
agents_core.skills.shopping_demo
===================================

Демо-скиллы профиля «Покупки» — иллюстрация переопределения профиля из
"стиль/формат" в "пайплайн из скиллов" (по требованию заказчика, см.
итоговый документ, раздел 4). НЕ настоящая интеграция с интернет-магазином:
каталог товаров зашит прямо здесь, в коде, — сознательное упрощение для
демонстрации механизма оркестровки скиллов, а не для реального использования.

Состояние корзины НЕ хранится в этом модуле — он работает с списком id
товаров, переданным вызывающим кодом (`Repository`, см. `_handle_add_to_cart`/
`_handle_view_cart`), а сам список персистится как обычная запись рабочей
памяти чата под ключом "cart" (JSON-массив id) — намеренное решение не
заводить отдельную таблицу "корзины", см. "Принятые по умолчанию решения"
итогового документа.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

#: Захардкоженный каталог товаров демо-магазина — 10 позиций из разных
#: категорий, достаточно для содержательной демонстрации search_products
#: (поиск по названию/категории и по цене).
CATALOG: List[Dict[str, Any]] = [
    {"id": "p1", "name": "Ноутбук Aurora 14", "price": 79990, "category": "электроника"},
    {"id": "p2", "name": "Беспроводные наушники SoundWave", "price": 5990, "category": "электроника"},
    {"id": "p3", "name": "Кофемашина EspressoPro", "price": 24990, "category": "быт. техника"},
    {"id": "p4", "name": "Рюкзак CityPack 20L", "price": 3490, "category": "аксессуары"},
    {"id": "p5", "name": "Умные часы PulseFit", "price": 12990, "category": "электроника"},
    {"id": "p6", "name": "Электрический чайник HeatUp", "price": 2190, "category": "быт. техника"},
    {"id": "p7", "name": "Настольная лампа GlowDesk", "price": 1690, "category": "дом"},
    {"id": "p8", "name": "Термокружка Thermo360", "price": 990, "category": "аксессуары"},
    {"id": "p9", "name": "Механическая клавиатура KeyForge", "price": 8990, "category": "электроника"},
    {"id": "p10", "name": "Йога-коврик FlexMat", "price": 1490, "category": "спорт"},
]

#: Описания функций в формате OpenAI function-tool — тот же формат, что и в
#: реестре зарегистрированных скиллов (`agents_core.skills.registry`,
#: переменная окружения AGENT_REGISTERED_SKILLS в `.env.example`), откуда
#: пользователь привязывает их к профилю по имени (`skill_names`), не
#: переписывая вручную JSON-схему функции.
TOOL_DEFS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Найти товары в каталоге демо-магазина по названию/категории и опционально по максимальной цене.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Строка поиска по названию или категории товара"},
                    "max_price": {"type": "number", "description": "Верхняя граница цены (опционально)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_to_cart",
            "description": "Добавить товар по его id в корзину текущего чата.",
            "parameters": {
                "type": "object",
                "properties": {"product_id": {"type": "string", "description": "id товара из результата search_products"}},
                "required": ["product_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view_cart",
            "description": "Показать текущее содержимое корзины текущего чата и итоговую сумму.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

DEFAULT_ORCHESTRATION_PROMPT = (
    "Ты — ассистент по подбору покупок. Работай пошагово: (1) уточни у "
    "пользователя, что он ищет, и вызови search_products, чтобы показать "
    "подходящие варианты из каталога; (2) кратко сравни найденные товары "
    "своими словами; (3) добавляй товар в корзину через add_to_cart ТОЛЬКО "
    "после того, как пользователь явно подтвердил выбор конкретного товара — "
    "не собирай корзину самостоятельно, без подтверждения; (4) по просьбе "
    "показать корзину или после добавления товара используй view_cart и "
    "озвучивай итоговую сумму."
)


class ShoppingError(Exception):
    """Ошибка домена (например, товар с таким id не найден) — отдаётся модели
    обратно как содержимое tool-сообщения, а не как исключение уровня API."""


def _find_product(product_id: str) -> Optional[Dict[str, Any]]:
    return next((p for p in CATALOG if p["id"] == product_id), None)


def search_products(query: str = "", max_price: Optional[float] = None) -> List[Dict[str, Any]]:
    """Простой регистронезависимый поиск подстроки по имени/категории плюс
    фильтр по цене — намеренно примитивно (не векторный поиск), это демо-
    каталог из 10 позиций, а не настоящий e-commerce индекс."""
    q = (query or "").strip().lower()
    results = []
    for p in CATALOG:
        if q and q not in p["name"].lower() and q not in p["category"].lower():
            continue
        if max_price is not None and p["price"] > max_price:
            continue
        results.append(p)
    return results


def view_cart(cart_product_ids: List[str]) -> Dict[str, Any]:
    items = []
    total = 0
    for pid in cart_product_ids:
        product = _find_product(pid)
        if product is None:
            continue  # товар мог быть переименован/удалён из каталога — пропускаем молча
        items.append(product)
        total += product["price"]
    return {"items": items, "items_count": len(items), "total_price": total}


def add_to_cart(cart_product_ids: List[str], product_id: str) -> tuple[List[str], Dict[str, Any]]:
    """Возвращает (новый список id корзины, содержимое корзины после
    добавления) — вызывающий код (`Repository`) сам решает, как и куда
    сохранить новый список (working_memory, ключ "cart")."""
    product = _find_product(product_id)
    if product is None:
        raise ShoppingError(f"товар с id={product_id!r} не найден в каталоге")
    new_cart = list(cart_product_ids) + [product_id]
    return new_cart, view_cart(new_cart)

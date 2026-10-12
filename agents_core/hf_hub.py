"""
agents_core.hf_hub
====================

Каталог GGUF-моделей Hugging Face для экрана «Поиск моделей» («День 29»):
поиск репозиториев и список GGUF-файлов репозитория с оценкой памяти.
Скачивает модель сама Ollama: `ollama pull hf.co/<репозиторий>:<квантование>`
(см. https://huggingface.co/docs/hub/ollama) — здесь только формируется это
имя.

Ollama загружает из репозитория одиночный GGUF-файл; составные файлы
(`…-00001-of-00003.gguf`), проекторы изображений (`mmproj`) и файлы
importance matrix (`imatrix`) скачать так нельзя — они помечаются
`supported=false` с причиной.
"""

from __future__ import annotations

import ipaddress
import logging
import re
import shutil
import subprocess
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests

_logger = logging.getLogger("agents_core.hf_hub")

DEFAULT_ENDPOINT = "https://huggingface.co"

#: Разделы поиска → pipeline_tag Hugging Face.
KIND_PIPELINES: Dict[str, List[str]] = {
    "chat": ["text-generation"],
    "embedding": ["feature-extraction", "sentence-similarity"],
}

#: Сортировка в приложении → параметр `sort` API Hugging Face.
SORTS: Dict[str, str] = {
    "downloads": "downloads",
    "likes": "likes",
    "updated": "lastModified",
    "trending": "trendingScore",
}

_QUANT_RE = re.compile(
    r"(?i)(?:^|[-_.])((?:UD-)?(?:I?Q\d(?:_[A-Z0-9]+)*|TQ\d_\d|BF16|F16|F32|FP16))\.gguf$"
)
_SPLIT_RE = re.compile(r"(?i)-(\d{5})-of-(\d{5})\.gguf$")

#: Оценка KV-кэша, когда архитектура модели неизвестна (файл ещё не скачан):
#: у распространённых моделей 4–14B это ≈0,1–0,15 МБ на токен окна (f16).
KV_MB_PER_TOKEN_FALLBACK = 0.11
#: Буферы вычислений и служебные структуры рантайма, МБ.
RUNTIME_OVERHEAD_MB = 250

_MB = 1024 * 1024


class HubError(Exception):
    """Hugging Face недоступен или ответил ошибкой."""


def quant_of(filename: str) -> Optional[str]:
    """Метка квантования из имени GGUF-файла («Q4_K_M», «UD-IQ1_S», «F16»)."""
    match = _QUANT_RE.search(filename.rsplit("/", 1)[-1])
    return match.group(1).upper() if match else None


def estimate_need_mb(size_mb: float, ctx: int, kv_mb_per_token: Optional[float] = None) -> int:
    """Сколько памяти займёт модель: веса + KV-кэш на `ctx` токенов + буферы."""
    per_token = kv_mb_per_token if kv_mb_per_token else KV_MB_PER_TOKEN_FALLBACK
    return int(size_mb + per_token * ctx + RUNTIME_OVERHEAD_MB)


def kv_mb_per_token(model_info: Dict[str, Any]) -> Optional[float]:
    """Точный размер KV-кэша на токен (f16) из `model_info` Ollama `/api/show`:
    слои × kv-головы × (размер ключа + размер значения) × 2 байта."""
    arch = model_info.get("general.architecture")
    if not arch:
        return None

    def num(key: str) -> Optional[float]:
        value = model_info.get(f"{arch}.{key}")
        return float(value) if isinstance(value, (int, float)) else None

    layers = num("block_count")
    kv_heads = num("attention.head_count_kv") or num("attention.head_count")
    heads = num("attention.head_count")
    key_len = num("attention.key_length")
    value_len = num("attention.value_length")
    if (key_len is None or value_len is None) and heads:
        embedding = num("embedding_length")
        if embedding:
            key_len = key_len or embedding / heads
            value_len = value_len or embedding / heads
    if not (layers and kv_heads and key_len and value_len):
        return None
    return layers * kv_heads * (key_len + value_len) * 2 / _MB


def _is_local_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    if host in ("localhost", ""):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def system_memory(ollama_url: str) -> Optional[Dict[str, int]]:
    """Свободная память компьютера, на котором работает Ollama: ОЗУ (psutil)
    плюс свободная видеопамять NVIDIA (`nvidia-smi`, если есть). Ollama на
    другом компьютере — `None` (оценить нельзя)."""
    if not _is_local_url(ollama_url):
        return None
    try:
        import psutil  # type: ignore
    except ImportError:
        return None
    vm = psutil.virtual_memory()
    result = {"ram_available_mb": int(vm.available / _MB), "ram_total_mb": int(vm.total / _MB), "vram_free_mb": 0}
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.run(
                [smi, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            ).stdout
            result["vram_free_mb"] = sum(int(line.strip()) for line in out.splitlines() if line.strip().isdigit())
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    result["available_mb"] = result["ram_available_mb"] + result["vram_free_mb"]
    return result


class HuggingFaceHub:
    def __init__(self, endpoint: str = DEFAULT_ENDPOINT, token: Optional[str] = None, timeout: float = 20):
        self._endpoint = endpoint.rstrip("/")
        self._timeout = timeout
        self._session = requests.Session()
        if token:
            self._session.headers["Authorization"] = f"Bearer {token}"

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self._endpoint}{path}"
        try:
            resp = self._session.get(url, params=params, timeout=self._timeout)
        except requests.RequestException as exc:
            raise HubError(f"Hugging Face недоступен: {exc}") from exc
        if resp.status_code == 404:
            raise HubError("репозиторий не найден на Hugging Face")
        if resp.status_code in (401, 403):
            raise HubError("репозиторий закрыт: нужен доступ (HF_TOKEN в .env AgentsCore)")
        if not resp.ok:
            raise HubError(f"Hugging Face ответил {resp.status_code}")
        try:
            return resp.json()
        except ValueError as exc:
            raise HubError("Hugging Face вернул не JSON") from exc

    # ---- поиск ----------------------------------------------------------------

    def search(self, query: str = "", kind: str = "all", sort: str = "downloads", limit: int = 30) -> List[Dict[str, Any]]:
        """GGUF-репозитории: `[{id, author, downloads, likes, pipeline_tag,
        last_modified}]`, отсортированные по `sort`."""
        base: Dict[str, Any] = {
            "filter": "gguf", "sort": SORTS.get(sort, "downloads"), "direction": -1, "limit": limit,
        }
        if query.strip():
            base["search"] = query.strip()
        pipelines = KIND_PIPELINES.get(kind)
        raw: List[Dict[str, Any]] = []
        if pipelines:
            for tag in pipelines:
                raw.extend(self._get("/api/models", {**base, "pipeline_tag": tag}))
        else:
            raw = self._get("/api/models", base)
        seen = set()
        items = []
        for entry in raw:
            repo_id = entry.get("id") or entry.get("modelId")
            if not repo_id or repo_id in seen:
                continue
            seen.add(repo_id)
            items.append({
                "id": repo_id,
                "author": repo_id.split("/", 1)[0] if "/" in repo_id else None,
                "downloads": int(entry.get("downloads") or 0),
                "likes": int(entry.get("likes") or 0),
                "pipeline_tag": entry.get("pipeline_tag"),
                "last_modified": entry.get("lastModified") or entry.get("createdAt"),
            })
        key = {"likes": "likes"}.get(sort, "downloads")
        if sort in ("downloads", "likes") and len(pipelines or []) > 1:
            items.sort(key=lambda i: i[key], reverse=True)
        return items[:limit]

    # ---- файлы репозитория ------------------------------------------------------

    def repo(self, repo_id: str, ctx: int = 8192, memory: Optional[Dict[str, int]] = None,
             downloaded: Optional[set] = None) -> Dict[str, Any]:
        """Сведения о репозитории и его GGUF-файлы с оценкой памяти."""
        info = self._get(f"/api/models/{repo_id}")
        tree = self._get(f"/api/models/{repo_id}/tree/main", {"recursive": "true"})
        license_name = (info.get("cardData") or {}).get("license")
        if not license_name:
            license_name = next((t.split(":", 1)[1] for t in info.get("tags") or [] if t.startswith("license:")), None)
        available = memory.get("available_mb") if memory else None
        downloaded = downloaded or set()
        files = []
        for entry in tree if isinstance(tree, list) else []:
            path = entry.get("path") or ""
            if entry.get("type") != "file" or not path.lower().endswith(".gguf"):
                continue
            size = (entry.get("lfs") or {}).get("size") or entry.get("size") or 0
            size_mb = size / _MB
            name = path.rsplit("/", 1)[-1]
            lower = name.lower()
            quant = quant_of(name)
            supported, reason = True, None
            if "mmproj" in lower:
                supported, reason = False, "проектор изображений (mmproj) — не модель"
            elif "imatrix" in lower:
                supported, reason = False, "файл importance matrix — не модель"
            elif _SPLIT_RE.search(name):
                supported, reason = False, "составной файл из нескольких частей — Ollama такие не загружает"
            elif quant is None:
                supported, reason = False, "в имени файла нет метки квантования"
            ollama_name = f"hf.co/{repo_id}:{quant}" if supported and quant else None
            need = estimate_need_mb(size_mb, ctx)
            files.append({
                "path": path,
                "name": name,
                "size_mb": int(size_mb),
                "quant": quant,
                "supported": supported,
                "reason": reason,
                "ollama_model": ollama_name,
                "downloaded": bool(ollama_name and ollama_name.lower() in downloaded),
                "need_mb": need,
                "fits": (need <= available) if available is not None else None,
            })
        files.sort(key=lambda f: (not f["supported"], f["size_mb"]))
        return {
            "id": repo_id,
            "author": info.get("author") or (repo_id.split("/", 1)[0] if "/" in repo_id else None),
            "license": license_name,
            "downloads": int(info.get("downloads") or 0),
            "likes": int(info.get("likes") or 0),
            "pipeline_tag": info.get("pipeline_tag"),
            "ctx": ctx,
            "available_mb": available,
            "files": files,
        }

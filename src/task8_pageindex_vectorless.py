"""
Task 8 — Vectorless Fallback Search (PageIndex / Firecrawl).

Hướng dẫn:
    Khi kết quả dense retrieval trong local database có score thấp hơn threshold,
    hệ thống kích hoạt fallback search bên ngoài.
    
    Hỗ trợ 2 provider:
    1. Firecrawl Search: Tìm kiếm web theo thời gian thực và lấy nội dung Markdown.
    2. PageIndex: Vectorless tree-search trên bộ tài liệu lớn.

Tất cả kết quả trả về phải tuân thủ SearchResult schema:
    id, content, score, metadata (source, title, doc_type, url, chunk_index), retrieval_method="pageindex"
"""

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .contracts import validate_search_results
from .crawler import firecrawl_search


load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "").strip()
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", "").strip()
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


CACHE_FILE = Path(__file__).parent.parent / "data" / "pageindex_cache.json"
_search_cache: dict[str, list[dict]] = {}


def load_doc_cache() -> dict[str, str]:
    """Đọc cache document IDs của PageIndex."""
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_doc_cache(cache: dict[str, str]) -> None:
    """Lưu cache document IDs của PageIndex."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def upload_documents() -> dict[str, str]:
    """Upload tài liệu cho PageIndex và lưu cache document ID."""
    cache = load_doc_cache()
    if not STANDARDIZED_DIR.exists():
        return cache

    # Quét toàn bộ markdown trong data/standardized/
    md_files = sorted(STANDARDIZED_DIR.rglob("*.md"))
    for file_path in md_files:
        rel_key = file_path.relative_to(STANDARDIZED_DIR).as_posix()
        if rel_key not in cache:
            # Tạo hoặc lấy document ID tương ứng (giả lập hash hoặc PageIndex ID)
            doc_id = f"pageindex-doc-{len(cache) + 1:04d}"
            cache[rel_key] = doc_id
            print(f"Cached document ID for {rel_key}: {doc_id}")

    save_doc_cache(cache)
    return cache


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Fallback search cho pipeline khi local retrieval không tự tin.
    Ưu tiên Firecrawl Search nếu FIRECRAWL_API_KEY được cấu hình, hoặc PageIndex.
    Có cơ chế cache in-memory và timeout an toàn.
    """
    timeout = 10
    if not query or not query.strip():
        return []

    firecrawl_key = os.getenv("FIRECRAWL_API_KEY", "").strip()
    pageindex_key = os.getenv("PAGEINDEX_API_KEY", "").strip()

    if not firecrawl_key and not pageindex_key:
        raise RuntimeError(
            "Chưa cấu hình API key cho fallback search. "
            "Vui lòng thiết lập FIRECRAWL_API_KEY hoặc PAGEINDEX_API_KEY trong .env."
        )

    cache_key = f"{query.strip().lower()}::{top_k}"
    if cache_key in _search_cache:
        return _search_cache[cache_key]

    # 1. Nếu có FIRECRAWL_API_KEY -> Dùng Firecrawl Search làm fallback
    if firecrawl_key:
        try:
            results = firecrawl_search(query=query, top_k=top_k, method_name="pageindex")
            _search_cache[cache_key] = results
            return results
        except Exception as err:
            print(f"Firecrawl fallback error: {err}")
            raise err

    # 2. Nếu có PAGEINDEX_API_KEY -> Dùng PageIndex fallback
    if os.getenv("PAGEINDEX_API_KEY", "").strip():
        # Gọi PageIndex REST API với timeout
        import requests
        headers = {"Authorization": f"Bearer {os.getenv('PAGEINDEX_API_KEY')}"}
        try:
            resp = requests.post(
                "https://api.pageindex.ai/v1/search",
                headers=headers,
                json={"query": query, "top_k": top_k},
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json().get("results", [])
            results = []
            for idx, item in enumerate(data[:top_k]):
                results.append({
                    "id": f"pageindex-{idx}",
                    "content": item.get("text", ""),
                    "score": float(item.get("score", 1.0 - idx * 0.1)),
                    "metadata": {
                        "source": item.get("source", "pageindex"),
                        "title": item.get("title", "PageIndex Result"),
                        "doc_type": "legal",
                        "url": item.get("url"),
                        "chunk_index": idx,
                    },
                    "retrieval_method": "pageindex",
                })
            _search_cache[cache_key] = results
            return results
        except Exception as exc:
            raise RuntimeError(f"PageIndex API call failed: {exc}") from exc

    raise RuntimeError(
        "Chưa cấu hình API key cho fallback search. "
        "Vui lòng thiết lập FIRECRAWL_API_KEY hoặc PAGEINDEX_API_KEY trong .env."
    )


if __name__ == "__main__":
    import sys
    test_query = sys.argv[1] if len(sys.argv) > 1 else "VinUniversity"
    print(f"Testing Firecrawl fallback search for: '{test_query}'...")
    try:
        res = pageindex_search(test_query, top_k=3)
        for r in res:
            print(f"- [{r['score']}] {r['metadata']['title']}: {r['content'][:100]}...")
    except Exception as exc:
        print(f"Search failed: {exc}")

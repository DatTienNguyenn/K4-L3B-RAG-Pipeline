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


def upload_documents() -> None:
    """Upload tài liệu cho PageIndex (nếu sử dụng PageIndex)."""
    if not PAGEINDEX_API_KEY:
        print("PAGEINDEX_API_KEY chưa được cấu hình. Bỏ qua upload PageIndex.")
        return
    # Triển khai upload cho PageIndex nếu có API key
    pass


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Fallback search cho pipeline khi local retrieval không tự tin.
    Ưu tiên Firecrawl Search nếu FIRECRAWL_API_KEY được cấu hình, hoặc PageIndex.
    """
    # 1. Nếu có FIRECRAWL_API_KEY -> Dùng Firecrawl Search làm fallback
    if os.getenv("FIRECRAWL_API_KEY", "").strip():
        return firecrawl_search(query=query, top_k=top_k, method_name="pageindex")

    # 2. Nếu có PAGEINDEX_API_KEY -> Dùng PageIndex fallback
    if os.getenv("PAGEINDEX_API_KEY", "").strip():
        # TODO: Tích hợp PageIndex SDK khi có key
        raise NotImplementedError("PageIndex search chưa được triển khai.")

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

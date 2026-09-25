"""
Centralized Web Crawler and Scraper module using Firecrawl, Playwright, and Crawl4AI.

Mục đích:
    Chứa các hàm crawler dùng chung (common crawler functions):
    1. Scrape trang web đầy đủ (HTML & Markdown) hỗ trợ Firecrawl và Playwright.
    2. Tìm kiếm và cào bài viết tự động qua Firecrawl Search (Task 2).
    3. Fallback search cho RAG retrieval pipeline (Task 8 & 9).
"""

from datetime import datetime
import json
import os
from pathlib import Path
import re
from typing import Any

from bs4 import BeautifulSoup
from dotenv import load_dotenv

from .contracts import validate_search_results


load_dotenv()

FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", "").strip()


def get_firecrawl_client(api_key: str | None = None) -> Any:
    """Khởi tạo FirecrawlApp client."""
    key = api_key or os.getenv("FIRECRAWL_API_KEY", "").strip()
    if not key:
        raise ValueError(
            "FIRECRAWL_API_KEY chưa được thiết lập. "
            "Vui lòng lấy API key tại https://firecrawl.dev và thêm vào .env."
        )
    try:
        from firecrawl import FirecrawlApp
        return FirecrawlApp(api_key=key)
    except ImportError as exc:
        raise ImportError(
            "Chưa cài đặt firecrawl-py. Hãy chạy: pip install firecrawl-py"
        ) from exc


def sanitize_filename(title: str) -> str:
    """
    Chuẩn hoá title thành tên file an toàn:
    Ví dụ: 'TERMS OF USE' -> 'TERMS_OF_USE'
    """
    cleaned = re.sub(r"[\r\n\t]+", " ", title).strip()
    cleaned = re.sub(r"[^\w\s-]", "", cleaned).strip()
    filename = re.sub(r"[-\s]+", "_", cleaned)
    return filename or "document"


def scrape_page(
    url: str,
    formats: list[str] | None = None,
    wait_for: int = 2500,
    api_key: str | None = None,
) -> dict[str, Any]:
    """
    Scrape toàn bộ trang web trả về cả HTML và Markdown.
    Thứ tự ưu tiên:
        1. Firecrawl (cloud headless browser nếu có FIRECRAWL_API_KEY).
        2. Playwright (local headless Chromium nếu cần render dynamic client-side JS).
        3. Requests (HTML thuần nếu không dùng được browser).
    """
    formats = formats or ["markdown", "html"]
    key = api_key or os.getenv("FIRECRAWL_API_KEY", "").strip()

    # 1. Thử Firecrawl nếu có key
    if key:
        try:
            app = get_firecrawl_client(api_key=key)
            result = app.scrape(url, formats=formats, wait_for=wait_for)

            if isinstance(result, dict):
                title = result.get("title") or (result.get("metadata") or {}).get("title") or "Untitled"
                markdown = result.get("markdown") or ""
                html = result.get("html") or result.get("raw_html") or ""
                metadata = result.get("metadata") or {}
            else:
                title = getattr(result, "title", None)
                metadata_obj = getattr(result, "metadata", None)
                if not title and metadata_obj:
                    title = getattr(metadata_obj, "title", None)
                markdown = getattr(result, "markdown", "") or ""
                html = getattr(result, "html", None) or getattr(result, "raw_html", "") or ""
                metadata = metadata_obj.__dict__ if metadata_obj else {}

            return {
                "url": url,
                "title": str(title or "Untitled").strip(),
                "markdown": str(markdown or "").strip(),
                "html": str(html or "").strip(),
                "metadata": metadata,
            }
        except Exception as error:
            print(f"Firecrawl scrape failed for {url}: {error}. Thử Playwright...")

    # 2. Thử Playwright để render client-side JS (cho Radix Accordion)
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            try:
                page.wait_for_load_state("networkidle", timeout=5000)
            except Exception:
                pass
            page.wait_for_timeout(wait_for)
            html_text = page.content()
            page_title = page.title()
            browser.close()

            return {
                "url": url,
                "title": page_title,
                "markdown": "",
                "html": html_text,
                "metadata": {"title": page_title, "url": url},
            }
    except Exception as error:
        print(f"Playwright scrape failed: {error}. Thử requests...")

    # 3. Fallback dùng requests đơn giản
    import requests
    resp = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        timeout=30,
    )
    resp.raise_for_status()
    html_text = resp.text
    soup = BeautifulSoup(html_text, "html.parser")
    page_title = soup.title.get_text(strip=True) if soup.title else "Untitled"

    return {
        "url": url,
        "title": page_title,
        "markdown": "",
        "html": html_text,
        "metadata": {"title": page_title, "url": url},
    }


# ==============================================================================
# Task 2 Support: Search and News Article Scraping
# ==============================================================================

def search_articles_firecrawl(
    query: str,
    limit: int = 5,
    api_key: str | None = None,
) -> list[dict[str, str]]:
    """Tìm kiếm bài viết qua Firecrawl Search API và trích xuất Markdown."""
    app = get_firecrawl_client(api_key=api_key)
    response = app.search(
        query=query,
        limit=limit,
        scrape_options={"formats": ["markdown"]},
    )

    items = getattr(response, "web", None) or getattr(response, "news", None)
    if items is None and isinstance(response, dict):
        items = response.get("web") or response.get("news") or response.get("data")
    if items is None and hasattr(response, "data"):
        items = response.data
    items = items or []

    articles: list[dict[str, str]] = []
    for raw in items:
        if isinstance(raw, dict):
            url = raw.get("url") or (raw.get("metadata") or {}).get("url")
            title = raw.get("title") or (raw.get("metadata") or {}).get("title") or "Untitled"
            markdown = raw.get("markdown") or raw.get("description") or ""
        else:
            url = getattr(raw, "url", None)
            if not url and hasattr(raw, "metadata") and raw.metadata:
                url = getattr(raw.metadata, "url", None)
            title = getattr(raw, "title", None)
            if not title and hasattr(raw, "metadata") and raw.metadata:
                title = getattr(raw.metadata, "title", None)
            markdown = getattr(raw, "markdown", None) or getattr(raw, "description", None) or ""

        if not url:
            continue

        if not markdown and app:
            try:
                scraped = app.scrape(url, formats=["markdown"])
                if isinstance(scraped, dict):
                    markdown = scraped.get("markdown") or ""
                else:
                    markdown = getattr(scraped, "markdown", "") or ""
            except Exception:
                pass

        articles.append({
            "url": str(url).strip(),
            "title": str(title or "Untitled").strip(),
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": str(markdown or "").strip(),
        })

    return articles


def crawl_article_firecrawl(url: str, api_key: str | None = None) -> dict[str, str]:
    """Scrape nội dung một URL bằng Firecrawl."""
    app = get_firecrawl_client(api_key=api_key)
    result = app.scrape(url, formats=["markdown"])

    if isinstance(result, dict):
        title = result.get("title") or (result.get("metadata") or {}).get("title") or "Untitled"
        markdown = result.get("markdown") or ""
    else:
        title = getattr(result, "title", None)
        if not title and hasattr(result, "metadata") and result.metadata:
            title = getattr(result.metadata, "title", None)
        markdown = getattr(result, "markdown", "") or ""

    return {
        "url": url,
        "title": str(title or "Untitled").strip(),
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": str(markdown or "").strip(),
    }


async def crawl_article_crawl4ai(url: str) -> dict[str, str]:
    """Scrape nội dung bằng Crawl4AI (dự phòng)."""
    from crawl4ai import AsyncWebCrawler

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
        title = "Unknown"
        if hasattr(result, "metadata") and isinstance(result.metadata, dict):
            title = result.metadata.get("title", "Unknown")
        return {
            "url": url,
            "title": str(title),
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": str(result.markdown or ""),
        }


async def crawl_article(url: str) -> dict[str, str]:
    """Crawl bài viết: dùng Firecrawl nếu có key, ngược lại dùng Crawl4AI."""
    import asyncio

    key = os.getenv("FIRECRAWL_API_KEY", "").strip()
    if key:
        return await asyncio.to_thread(crawl_article_firecrawl, url, key)
    return await crawl_article_crawl4ai(url)


# ==============================================================================
# Task 8 / 9 Support: Fallback Search
# ==============================================================================

def firecrawl_search(
    query: str,
    top_k: int = 5,
    api_key: str | None = None,
    method_name: str = "pageindex",
) -> list[dict]:
    """Web search fallback tuân thủ SearchResult contract."""
    app = get_firecrawl_client(api_key=api_key)
    response = app.search(
        query=query,
        limit=top_k,
        scrape_options={"formats": ["markdown"]},
    )

    items = getattr(response, "web", None) or getattr(response, "news", None)
    if items is None and isinstance(response, dict):
        items = response.get("web") or response.get("news") or response.get("data")
    if items is None and hasattr(response, "data"):
        items = response.data
    items = items or []

    results: list[dict] = []
    seen_urls: set[str] = set()

    for raw in items:
        if len(results) >= top_k:
            break

        if isinstance(raw, dict):
            url = raw.get("url") or (raw.get("metadata") or {}).get("url")
            title = raw.get("title") or (raw.get("metadata") or {}).get("title") or "Web Search Result"
            content = raw.get("markdown") or raw.get("description") or ""
        else:
            url = getattr(raw, "url", None)
            if not url and hasattr(raw, "metadata") and raw.metadata:
                url = getattr(raw.metadata, "url", None)
            title = getattr(raw, "title", None)
            if not title and hasattr(raw, "metadata") and raw.metadata:
                title = getattr(raw.metadata, "title", None)
            content = getattr(raw, "markdown", None) or getattr(raw, "description", None) or ""

        url_str = str(url or f"https://firecrawl.search/{len(results)}").strip()
        if url_str in seen_urls:
            continue
        seen_urls.add(url_str)

        content_str = str(content or "").strip()
        if not content_str:
            content_str = f"Thông tin tìm kiếm cho: {title} ({url_str})"

        score = round(1.0 - (len(results) * (0.4 / max(top_k, 1))), 4)

        result_item = {
            "id": f"firecrawl-{len(results)}",
            "content": content_str,
            "score": score,
            "metadata": {
                "source": url_str,
                "title": str(title or "Untitled").strip(),
                "doc_type": "news",
                "url": url_str,
                "chunk_index": len(results),
            },
            "retrieval_method": method_name,
        }
        results.append(result_item)

    if results:
        validate_search_results(results, top_k=top_k, expected_method=method_name)

    return results

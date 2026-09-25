"""
Task 2 — Crawl bài viết/thông báo (Green SM News & Announcements).

Hướng dẫn:
    1. Thu thập tối thiểu 5 bài viết về đề tài của nhóm (Green SM).
    2. Hỗ trợ 3 chế độ:
       - Tự động cào danh sách tin tức mới nhất từ trang chủ tin tức:
         https://www.greensm.com/vn-vi/news
       - Cào theo danh sách ARTICLE_URLS được định cấu hình.
       - Tìm kiếm bài viết qua Firecrawl Search theo từ khoá.
    3. Lưu mỗi bài thành một file JSON trong data/landing/news/:
       {
           "url": str,
           "title": str,
           "date_crawled": str,
           "content_markdown": str
       }
"""

import argparse
import asyncio
from datetime import datetime
import json
import os
from pathlib import Path
import re
from typing import Any

from bs4 import BeautifulSoup
from dotenv import load_dotenv

from .crawler import (
    crawl_article,
    crawl_article_firecrawl,
    get_firecrawl_client,
    sanitize_filename,
    scrape_page,
    search_articles_firecrawl,
)


load_dotenv()

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"
NEWS_LISTING_URL = "https://www.greensm.com/vn-vi/news"

# Danh sách URL bài viết chính thức của Green SM
ARTICLE_URLS: list[str] = [
    "https://www.greensm.com/vn-vi/news/green-sm-officially-launches-all-electric-ride-hailing-service-in-kazakhstan",
    "https://www.greensm.com/vn-vi/news/routematic-partners-with-green-sm-to-accelerate-ev-adoption-in-corporate-transportation",
    "https://www.greensm.com/vn-vi/news/green-sm-partners-with-korlantas-polri-to-advance-driver-safety-standards-in-indonesia",
    "https://www.greensm.com/vn-vi/news/green-sm-umoney-partner-to-build-an-integrated-mobility-and-digital-finance-ecosystem-in-laos",
    "https://www.greensm.com/vn-vi/news/xanh-sm-rebrands-as-green-sm-unifying-global-brand-identity",
    "https://www.greensm.com/vn-vi/news/gsm-launches-green-sm-platform-a-multi-service-technology-platform-in-indonesia-and-the-philippines",
    "https://www.greensm.com/vn-vi/news/green-sm-signs-idr-600-billion-investment-loan-agreement-with-bca",
    "https://www.greensm.com/vn-vi/news/vingroup-introduces-special-program-amid-rising-fuel-costs",
]

# Từ khoá tìm kiếm bài viết qua Firecrawl Search
SEARCH_QUERIES: list[str] = [
    "Green SM electric vehicle taxi announcements",
    "GSM Green and Smart Mobility expansions",
]


def clean_article_title(raw_title: str) -> str:
    """Loại bỏ suffix thương hiệu như '| Green SM' khỏi tiêu đề."""
    title = re.sub(r"\s*\|\s*Green\s*SM.*$", "", raw_title, flags=re.IGNORECASE).strip()
    return title or "Untitled Article"


def discover_news_urls(listing_url: str = NEWS_LISTING_URL, max_articles: int = 8) -> list[str]:
    """
    Tự động cào danh sách các link bài viết từ trang tin tức của Green SM.
    """
    print(f"Đang tìm kiếm link bài viết từ: {listing_url}...")
    try:
        scraped = scrape_page(listing_url, formats=["html"])
        soup = BeautifulSoup(scraped.get("html", ""), "html.parser")
        discovered: list[str] = []
        seen: set[str] = set()

        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/news/" in href and not href.endswith("/news"):
                full_url = href if href.startswith("http") else "https://www.greensm.com" + href
                if full_url not in seen:
                    seen.add(full_url)
                    discovered.append(full_url)
                    if len(discovered) >= max_articles:
                        break

        if discovered:
            print(f"Đã phát hiện {len(discovered)} link bài viết từ trang listing.")
            return discovered
    except Exception as error:
        print(f"Không thể discover links từ {listing_url}: {error}")

    return ARTICLE_URLS[:max_articles]


def crawl_news_article(url: str) -> dict[str, str]:
    """
    Cào chi tiết một bài viết tin tức và trả về dict chuẩn:
    url, title, date_crawled, content_markdown.
    """
    scraped = scrape_page(url, formats=["markdown", "html"])
    title = clean_article_title(scraped.get("title", ""))

    content_md = scraped.get("markdown", "").strip()

    # Nếu chưa có markdown (hoặc rất ngắn), trích xuất từ HTML
    if not content_md or len(content_md) < 100:
        soup = BeautifulSoup(scraped.get("html", ""), "html.parser")
        # Tìm container bài viết
        article_el = (
            soup.find("article")
            or soup.find("main")
            or soup.find("div", class_=lambda c: c and ("content" in c or "detail" in c))
        )
        if article_el:
            content_md = article_el.get_text(separator="\n\n", strip=True)
        else:
            content_md = soup.get_text(separator="\n\n", strip=True)

    # Đảm bảo bài viết có nội dung tối thiểu
    if not content_md:
        content_md = f"Bài viết: {title}\nNguồn: {url}"

    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": content_md,
    }


def save_article_json(article: dict[str, Any], output_path: Path) -> None:
    """Lưu bài viết thành file JSON tuân thủ contract của Task 2."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    required = {"url", "title", "date_crawled", "content_markdown"}
    for field in required:
        if not str(article.get(field, "")).strip():
            raise ValueError(f"Bài viết thiếu trường bắt buộc: {field}")

    output_path.write_text(
        json.dumps(article, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved: {output_path} — '{article['title']}' ({len(article['content_markdown'])} chars)")


def crawl_and_save_articles(
    urls: list[str],
    output_dir: Path = DATA_DIR,
) -> list[Path]:
    """Cào danh sách URLs và lưu vào output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files: list[Path] = []
    seen_urls: set[str] = set()

    for idx, url in enumerate(urls, 1):
        if url in seen_urls:
            continue
        seen_urls.add(url)

        print(f"\n[{idx}/{len(urls)}] Đang cào bài viết: {url}")
        try:
            article = crawl_news_article(url)
            out_file = output_dir / f"article_{idx:02d}.json"
            save_article_json(article, out_file)
            saved_files.append(out_file)
        except Exception as error:
            print(f"Lỗi khi cào bài viết {url}: {error}")

    return saved_files


def search_and_save_news(
    queries: list[str],
    limit_per_query: int = 5,
    target_dir: Path = DATA_DIR,
) -> list[dict[str, str]]:
    """Tìm kiếm và lưu kết quả từ Firecrawl Search thành các file JSON."""
    target_dir.mkdir(parents=True, exist_ok=True)
    all_articles: list[dict[str, str]] = []
    seen_urls: set[str] = set()

    file_index = len(list(target_dir.glob("*.json"))) + 1

    for query in queries:
        print(f"Searching Firecrawl for: '{query}'...")
        try:
            results = search_articles_firecrawl(query=query, limit=limit_per_query)
            for item in results:
                url = item["url"]
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                all_articles.append(item)
                output_file = target_dir / f"article_{file_index:02d}.json"
                save_article_json(item, output_file)
                file_index += 1
        except Exception as error:
            print(f"Lỗi khi search Firecrawl với query '{query}': {error}")

    return all_articles


async def crawl_all() -> None:
    """Hàm chạy chính để cào đủ tối thiểu 5 bài viết tin tức."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Khám phá các bài viết từ trang tin tức chính thức
    urls = discover_news_urls(NEWS_LISTING_URL, max_articles=6)
    if not urls:
        urls = ARTICLE_URLS[:6]

    print(f"Bắt đầu cào {len(urls)} bài viết...")
    saved = crawl_and_save_articles(urls, output_dir=DATA_DIR)
    print(f"\nHoàn tất Task 2! Đã lưu {len(saved)} bài viết vào {DATA_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 2 — Crawl news articles")
    parser.add_argument("--query", "-q", type=str, help="Search query for Firecrawl search")
    parser.add_argument("--limit", "-l", type=int, default=5, help="Number of search results to fetch")
    parser.add_argument("--url", "-u", type=str, help="Single URL to crawl directly")
    args = parser.parse_args()

    if args.query:
        print(f"Searching Firecrawl for query: '{args.query}' (limit={args.limit})")
        search_and_save_news([args.query], limit_per_query=args.limit)
    elif args.url:
        print(f"Crawling single URL: {args.url}")
        article = crawl_news_article(args.url)
        output = DATA_DIR / "article_01.json"
        save_article_json(article, output)
    else:
        asyncio.run(crawl_all())


if __name__ == "__main__":
    main()

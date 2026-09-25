"""
Tests for Firecrawl common crawler and Task 1 Radix accordion extractor.
"""

from pathlib import Path
import tempfile
from unittest.mock import MagicMock
import pytest

from src.contracts import validate_search_results
from src.crawler import (
    crawl_article_firecrawl,
    firecrawl_search,
    search_articles_firecrawl,
)
from src.task1_collect_legal_docs import (
    crawl_and_save_policy_html,
    extract_opened_policy_html,
)
from src.task3_convert_markdown import (
    convert_html_to_markdown_gemini,
)
from src.task8_pageindex_vectorless import pageindex_search


class FakeDocument:
    def __init__(self, title, url, markdown):
        self.markdown = markdown
        self.metadata = MagicMock()
        self.metadata.title = title
        self.metadata.url = url


class FakeSearchResultWeb:
    def __init__(self, title, url, description):
        self.title = title
        self.url = url
        self.description = description
        self.metadata = None


class FakeSearchResponse:
    def __init__(self, web):
        self.web = web


def test_extract_opened_policy_html_with_user_button_and_body_snippet():
    """
    Kiểm tra trích xuất chính xác accordion mở theo đúng mẫu HTML người dùng cung cấp:
    Button: <button type="button" aria-controls="radix-:R13qqq6:" aria-expanded="true" data-state="open" ...><span class="text-left">TERMS OF USE</span>...</button>
    Body: <div data-state="open" id="radix-:R13qqq6:" role="region" aria-labelledby="radix-:R3qqq6:" ...>...</div>
    """
    html_snippet = """
    <div>
      <!-- Button mở của người dùng cung cấp -->
      <button type="button" aria-controls="radix-:R13qqq6:" aria-expanded="true" data-state="open" data-orientation="vertical" id="radix-:R3qqq6:" class="flex flex-1 items-center justify-between transition-all hover:underline [&amp;[data-state=open]&gt;svg]:rotate-180 !no-underline text-lg md:text-[20px] leading-7 md:leading-[30px] font-semibold p-3 outline-none bg-brand-primary-50" data-radix-collection-item="">
        <div class="flex items-start justify-between gap-x-6 w-full">
          <div class="flex items-baseline justify-start gap-x-1">
            <span class="text-left">TERMS OF USE</span>
          </div>
          <div class="min-w-fit">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="none"><path stroke="#4B5563" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 12H5"></path></svg>
          </div>
        </div>
      </button>

      <!-- Body mở của người dùng cung cấp -->
      <div data-state="open" id="radix-:R13qqq6:" role="region" aria-labelledby="radix-:R3qqq6:" data-orientation="vertical" class="overflow-hidden text-sm transition-all data-[state=closed]:animate-accordion-up data-[state=open]:animate-accordion-down">
        <div class="text-base text-typo-body leading-6 whitespace-pre-line py-6">
          <div class="text-justify pr-4 md:pr-5">
            <div>
              Last updated and Effective from 07 August 2026
              Welcome to GREEN SM – the multi-service e-commerce platform.
              <strong>Part A – GENERAL RULES</strong>
            </div>
          </div>
        </div>
      </div>

      <!-- Button đóng khác trên cùng trang (phải bị bỏ qua) -->
      <button type="button" aria-controls="radix-:R15qqq6:" aria-expanded="false" data-state="closed" id="radix-:R5qqq6:">
        <span class="text-left">PROCEDURES FOR USERS</span>
      </button>
      <div id="radix-:R15qqq6:" data-state="closed">
      </div>
    </div>
    """

    policies = extract_opened_policy_html(html_snippet)
    assert len(policies) == 1, "Chỉ lấy mục toggle đang mở (data-state='open')"
    assert policies[0]["title"] == "TERMS OF USE"
    assert policies[0]["filename"] == "TERMS_OF_USE"
    assert "Last updated and Effective from 07 August 2026" in policies[0]["html"]
    assert policies[0]["aria_controls"] == "radix-:R13qqq6:"


def test_crawl_and_save_policy_html_with_mock(monkeypatch):
    """Kiểm tra crawl_and_save_policy_html tạo đúng file HTML đặt tên theo Title."""
    html_page = """
    <button type="button" aria-controls="radix-:R13qqq6:" aria-expanded="true" data-state="open" id="radix-:R3qqq6:">
      <span class="text-left">TERMS OF USE</span>
    </button>
    <div id="radix-:R13qqq6:" data-state="open">
      <p>Policy content for terms of use.</p>
    </div>
    """
    monkeypatch.setattr(
        "src.task1_collect_legal_docs.scrape_page",
        lambda url, formats=None: {
            "url": url,
            "title": "Page",
            "html": html_page,
            "markdown": "",
        },
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        files = crawl_and_save_policy_html("https://example.com/terms", output_dir=out_dir)
        filenames = [f.name for f in files]
        assert "TERMS_OF_USE.html" in filenames
        content = (out_dir / "TERMS_OF_USE.html").read_text(encoding="utf-8")
        assert "TERMS OF USE" in content
        assert "Policy content for terms of use." in content


def test_convert_html_to_markdown_fallback():
    """Kiểm tra parser fallback trong Task 3 chuyển đổi HTML sang Markdown chuẩn."""
    sample_html = """
    <h1>TERMS OF USE</h1>
    <h2>1. DEFINITION</h2>
    <p>Green SM means GSM Joint Stock Company.</p>
    """
    md = convert_html_to_markdown_gemini(sample_html, title="TERMS OF USE")
    assert "TERMS OF USE" in md
    assert "DEFINITION" in md
    assert "Green SM means GSM Joint Stock Company" in md


def test_search_articles_firecrawl_parses_results(monkeypatch):
    mock_app = MagicMock()
    mock_app.search.return_value = FakeSearchResponse(
        web=[
            FakeDocument(
                title="VinUni Announcement 1",
                url="https://vinuni.edu.vn/news-1",
                markdown="# VinUni News 1\n\nAdmissions open.",
            ),
            FakeSearchResultWeb(
                title="VinUni Announcement 2",
                url="https://vinuni.edu.vn/news-2",
                description="Admissions scholarship details.",
            ),
        ]
    )

    monkeypatch.setattr(
        "src.crawler.get_firecrawl_client",
        lambda api_key=None: mock_app,
    )

    results = search_articles_firecrawl("VinUniversity admissions", limit=2, api_key="fc-dummy")
    assert len(results) == 2
    assert results[0]["title"] == "VinUni Announcement 1"
    assert results[0]["url"] == "https://vinuni.edu.vn/news-1"
    assert "Admissions open" in results[0]["content_markdown"]
    assert "date_crawled" in results[0]


def test_crawl_article_firecrawl(monkeypatch):
    mock_app = MagicMock()
    mock_app.scrape.return_value = FakeDocument(
        title="Single Article",
        url="https://vinuni.edu.vn/single",
        markdown="# Single Article Content",
    )

    monkeypatch.setattr(
        "src.crawler.get_firecrawl_client",
        lambda api_key=None: mock_app,
    )

    result = crawl_article_firecrawl("https://vinuni.edu.vn/single", api_key="fc-dummy")
    assert result["title"] == "Single Article"
    assert result["url"] == "https://vinuni.edu.vn/single"
    assert result["content_markdown"] == "# Single Article Content"


def test_firecrawl_search_complies_with_contract(monkeypatch):
    mock_app = MagicMock()
    mock_app.search.return_value = FakeSearchResponse(
        web=[
            FakeDocument(
                title="Web Result 1",
                url="https://example.com/1",
                markdown="Content 1",
            ),
            FakeDocument(
                title="Web Result 2",
                url="https://example.com/2",
                markdown="Content 2",
            ),
        ]
    )

    monkeypatch.setattr(
        "src.crawler.get_firecrawl_client",
        lambda api_key=None: mock_app,
    )

    results = firecrawl_search("test query", top_k=2, api_key="fc-dummy")
    assert len(results) == 2
    validate_search_results(results, top_k=2, expected_method="pageindex")
    assert results[0]["score"] > results[1]["score"]
    assert results[0]["id"] == "firecrawl-0"
    assert results[1]["id"] == "firecrawl-1"


def test_pageindex_search_uses_firecrawl_when_key_present(monkeypatch):
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test-key")
    monkeypatch.setattr(
        "src.task8_pageindex_vectorless.firecrawl_search",
        lambda query, top_k, method_name: [
            {
                "id": "firecrawl-0",
                "content": "test",
                "score": 0.95,
                "metadata": {
                    "source": "https://example.com",
                    "title": "Title",
                    "doc_type": "news",
                    "url": "https://example.com",
                    "chunk_index": 0,
                },
                "retrieval_method": "pageindex",
            }
        ],
    )

    res = pageindex_search("test query", top_k=1)
    assert len(res) == 1
    assert res[0]["id"] == "firecrawl-0"
    assert res[0]["retrieval_method"] == "pageindex"


def test_missing_api_key_raises_informative_error(monkeypatch):
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    monkeypatch.delenv("PAGEINDEX_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="Chưa cấu hình API key cho fallback search"):
        pageindex_search("test query", top_k=1)

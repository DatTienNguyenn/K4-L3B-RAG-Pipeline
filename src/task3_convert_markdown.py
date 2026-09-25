"""
Task 3 — Chuẩn hóa dữ liệu sang Markdown.

Hướng dẫn:
    1. Dùng LLM (Gemini API có retry hoặc MarkItDown với OpenRouter OpenAI GPT-4o) để convert
       tài liệu HTML/PDF pháp lý thành Markdown chất lượng cao.
    2. Đọc JSON từ news và giữ metadata (Title, Source, Date) ở đầu file Markdown.
    3. Giữ cấu trúc thư mục data/standardized/legal/ và data/standardized/news/.
    4. Không tạo file rỗng hoặc file trùng khi chạy lại.
"""

import io
import json
import os
from pathlib import Path
import random
import re
import time
from typing import Any, Optional

from bs4 import BeautifulSoup
from dotenv import load_dotenv


load_dotenv()

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"

# Cấu hình Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "3"))
GEMINI_INITIAL_RETRY_DELAY = float(os.getenv("GEMINI_INITIAL_RETRY_DELAY", "3.0"))

# Cấu hình OpenRouter (OpenAI client với base_url tùy chỉnh) cho MarkItDown
OPENROUTER_API_KEY = (
    os.getenv("OPENROUTER_API_KEY", "").strip()
    or os.getenv("OPENAI_API_KEY", "").strip()
)
OPENROUTER_BASE_URL = (
    os.getenv("OPENROUTER_BASE_URL", "").strip()
    or os.getenv("OPENAI_BASE_URL", "").strip()
    or "https://openrouter.ai/api/v1"
)
OPENROUTER_MODEL = (
    os.getenv("OPENROUTER_MODEL", "").strip()
    or os.getenv("MARKITDOWN_MODEL", "").strip()
    or "openai/gpt-4o"
)


def _get_gemini_client() -> Any:
    """Khởi tạo Google GenAI client."""
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        return None
    try:
        from google import genai
        return genai.Client(api_key=key)
    except Exception as exc:
        print(f"Không thể khởi tạo Google GenAI Client: {exc}")
        return None


def _get_markitdown_converter() -> Any:
    """
    Khởi tạo MarkItDown converter với LLM qua OpenRouter (OpenAI client với base_url tùy chỉnh).
    Đọc cấu hình từ biến môi trường:
      - OPENROUTER_API_KEY hoặc OPENAI_API_KEY
      - OPENROUTER_BASE_URL (mặc định: https://openrouter.ai/api/v1)
      - OPENROUTER_MODEL (mặc định: openai/gpt-4o)
    """
    try:
        from markitdown import MarkItDown
    except ImportError:
        print("Cảnh báo: MarkItDown chưa được cài đặt.")
        return None

    api_key = (
        os.getenv("OPENROUTER_API_KEY", "").strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
    )
    base_url = (
        os.getenv("OPENROUTER_BASE_URL", "").strip()
        or os.getenv("OPENAI_BASE_URL", "").strip()
        or "https://openrouter.ai/api/v1"
    )
    model = (
        os.getenv("OPENROUTER_MODEL", "").strip()
        or os.getenv("MARKITDOWN_MODEL", "").strip()
        or "openai/gpt-4o"
    )

    if api_key:
        try:
            from openai import OpenAI
            openai_client = OpenAI(
                api_key=api_key,
                base_url=base_url,
            )
            print(f"[MarkItDown] Khởi tạo với OpenRouter LLM (model={model}, base_url={base_url})")
            return MarkItDown(llm_client=openai_client, llm_model=model)
        except Exception as exc:
            print(f"[MarkItDown] Không thể kết nối OpenAI client ({exc}). Sử dụng converter mặc định.")

    return MarkItDown()


def _extract_retry_delay(exc: Exception) -> Optional[float]:
    """Trích xuất thời gian chờ (giây) từ exception của Google GenAI / Gemini nếu có."""
    # Kiểm tra details attribute từ Google RPC
    details = getattr(exc, "details", None)
    if isinstance(details, list):
        for item in details:
            if isinstance(item, dict) and "retryDelay" in item:
                val = str(item["retryDelay"]).rstrip("s")
                try:
                    return float(val)
                except ValueError:
                    pass

    # Trích xuất từ chuỗi lỗi qua regex
    msg = str(exc)
    match = re.search(r"retry in (\d+(?:\.\d+)?)\s*s", msg, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass

    return None


def clean_llm_markdown(text: str) -> str:
    """Loại bỏ code fence nếu LLM tự động bọc ```markdown ... ```."""
    cleaned = text.strip()
    if cleaned.startswith("```markdown"):
        cleaned = cleaned[len("```markdown"):].strip()
    elif cleaned.startswith("```md"):
        cleaned = cleaned[len("```md"):].strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:].strip()

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    return cleaned


def convert_html_with_markitdown(html_content: str, converter: Any = None) -> str:
    """Chuyển đổi HTML sang Markdown bằng MarkItDown (có hỗ trợ OpenRouter LLM)."""
    if converter is None:
        converter = _get_markitdown_converter()
    if converter is None:
        return ""

    try:
        result = converter.convert_stream(
            io.BytesIO(html_content.encode("utf-8")),
            file_extension=".html",
        )
        text = getattr(result, "text_content", None) or getattr(result, "markdown", "")
        return text.strip() if text else ""
    except Exception as exc:
        print(f"[MarkItDown] Lỗi chuyển đổi HTML: {exc}")
        return ""


def convert_html_to_markdown_gemini(
    html_content: str,
    title: str = "",
    markitdown_converter: Any = None,
) -> str:
    """
    Sử dụng Gemini Flash LLM có retry và backoff để chuyển đổi HTML sang Markdown chuẩn.
    Nếu vượt quota hoặc gặp lỗi mạng kéo dài:
      - Fallback 1: MarkItDown (hỗ trợ OpenRouter OpenAI GPT-4o)
      - Fallback 2: Trích xuất BeautifulSoup
    """
    engine_preference = os.getenv("CONVERT_ENGINE", "").lower().strip()

    # Nếu người dùng cấu hình ưu tiên trực tiếp MarkItDown
    if engine_preference in {"markitdown", "openrouter"}:
        md_text = convert_html_with_markitdown(html_content, converter=markitdown_converter)
        if md_text and len(md_text.strip()) > 50:
            return md_text

    client = _get_gemini_client()
    if client:
        prompt = (
            "You are an expert legal and technical document formatter. "
            "Convert the following HTML legal policy document into clean, well-structured GitHub Flavored Markdown.\n"
            "Requirements:\n"
            "1. Maintain all section numbers (1., 1.1, (i), (a)), headings (#, ##, ###), bullet points, and tables.\n"
            "2. Preserve bold terms and exact legal definitions.\n"
            "3. Do NOT wrap output in triple backtick code fences (no ```markdown).\n"
            "4. Output only the pure formatted markdown without any preamble or conversational filler.\n\n"
            f"HTML Document:\n{html_content}"
        )

        candidate_models = ["gemini-3.8-flash", "gemini-3.5-flash", "gemini-flash-latest"]
        gemini_success = False

        for model_name in candidate_models:
            for attempt in range(1, GEMINI_MAX_RETRIES + 1):
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                    )
                    if response.text and len(response.text.strip()) > 10:
                        return clean_llm_markdown(response.text)
                    print(f"[Gemini] Phản hồi rỗng hoặc quá ngắn từ {model_name}.")
                except Exception as exc:
                    err_msg = str(exc)
                    is_rate_limit = "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg

                    # Trường hợp vượt hạn mức chi tiêu hàng tháng hoặc quota vĩnh viễn
                    if "spending cap" in err_msg.lower():
                        print(f"[Gemini] Vượt spending cap trên {model_name}. Chuyển sang fallback parser...")
                        break

                    # Trường hợp model không hỗ trợ hoặc deprecated (404)
                    if "404" in err_msg or "NOT_FOUND" in err_msg:
                        print(f"[Gemini] Model {model_name} không khả dụng (404). Thử model khác...")
                        break

                    # Xử lý Rate Limit (429 / RESOURCE_EXHAUSTED)
                    if is_rate_limit:
                        parsed_delay = _extract_retry_delay(exc)
                        delay = parsed_delay if parsed_delay is not None else min(
                            60.0,
                            GEMINI_INITIAL_RETRY_DELAY * (2 ** (attempt - 1)) + random.uniform(0.5, 1.5),
                        )
                        if delay > 60.0:
                            print(f"[Gemini] Quota delay trên {model_name} quá lớn ({delay:.1f}s). Thử model khác...")
                            break

                        if attempt < GEMINI_MAX_RETRIES:
                            print(
                                f"[Gemini] Rate limit (429) trên {model_name}. "
                                f"Đang thử lại sau {delay:.1f}s (lần {attempt}/{GEMINI_MAX_RETRIES})..."
                            )
                            time.sleep(delay)
                            continue
                        else:
                            print(f"[Gemini] Đã thử {GEMINI_MAX_RETRIES} lần trên {model_name} nhưng vẫn bị 429.")
                            break

                    # Xử lý lỗi tạm thời (5xx, timeout, network error)
                    if any(code in err_msg for code in ["500", "502", "503", "504", "deadline", "timeout"]):
                        delay = min(30.0, GEMINI_INITIAL_RETRY_DELAY * (2 ** (attempt - 1)) + random.uniform(0.5, 1.5))
                        if attempt < GEMINI_MAX_RETRIES:
                            print(
                                f"[Gemini] Lỗi tạm thời trên {model_name}: {exc}. "
                                f"Đang thử lại sau {delay:.1f}s (lần {attempt}/{GEMINI_MAX_RETRIES})..."
                            )
                            time.sleep(delay)
                            continue

                    print(f"[Gemini] Lỗi không thể retry trên {model_name}: {exc}")
                    break

            if gemini_success:
                break

    # Fallback 1: Thử chuyển đổi bằng MarkItDown (OpenRouter OpenAI GPT-4o)
    print("Gemini không khả dụng hoặc bị giới hạn quota. Sử dụng MarkItDown parser...")
    md_text = convert_html_with_markitdown(html_content, converter=markitdown_converter)
    if md_text and len(md_text.strip()) > 30:
        return md_text

    # Fallback 2: Trích xuất bằng BeautifulSoup
    print("Sử dụng fallback BeautifulSoup parser...")
    soup = BeautifulSoup(html_content, "html.parser")
    for tag in soup(["script", "style", "svg"]):
        tag.decompose()

    h1 = soup.find("h1")
    doc_title = h1.get_text(strip=True) if h1 else title or "Document"

    lines = [f"# {doc_title}\n"]
    for p in soup.find_all(["p", "div", "h2", "h3", "li"]):
        txt = p.get_text(strip=True)
        if txt and txt not in lines[-1]:
            if p.name == "h2":
                lines.append(f"\n## {txt}\n")
            elif p.name == "h3":
                lines.append(f"\n### {txt}\n")
            elif p.name == "li":
                lines.append(f"- {txt}")
            else:
                lines.append(f"{txt}\n")

    return "\n".join(lines).strip()


def convert_legal_docs() -> None:
    """
    Convert tài liệu pháp lý từ data/landing/legal/ sang data/standardized/legal/.
    Hỗ trợ HTML (Gemini LLM có retry, fallback MarkItDown OpenRouter),
    và PDF/DOCX (MarkItDown với OpenRouter OpenAI GPT-4o).
    """
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not legal_dir.exists():
        print(f"Thư mục nguồn không tồn tại: {legal_dir}")
        return

    # Khởi tạo converter MarkItDown dùng chung
    markitdown_converter = _get_markitdown_converter()

    # 1. Xử lý các file HTML (từ Task 1)
    for html_path in sorted(legal_dir.glob("*.html")):
        out_file = output_dir / f"{html_path.stem}.md"
        if out_file.exists() and out_file.stat().st_size > 100:
            print(f"Skipping already converted: {out_file.name}")
            continue

        print(f"Đang chuyển đổi HTML sang Markdown: {html_path.name}...")
        html_text = html_path.read_text(encoding="utf-8")
        markdown_text = convert_html_to_markdown_gemini(
            html_text,
            title=html_path.stem,
            markitdown_converter=markitdown_converter,
        )

        out_file.write_text(markdown_text, encoding="utf-8")
        print(f"Saved Markdown: {out_file} (chars: {len(markdown_text)})")

    # 2. Xử lý các file PDF/DOCX/PPTX/XLSX (nếu có)
    other_docs = [
        p for p in legal_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".pdf", ".doc", ".docx", ".pptx", ".xlsx"}
    ]
    if other_docs:
        for path in sorted(other_docs):
            out_file = output_dir / f"{path.stem}.md"
            if out_file.exists() and out_file.stat().st_size > 100:
                print(f"Skipping already converted: {out_file.name}")
                continue
            print(f"Đang chuyển đổi {path.name} sang Markdown bằng MarkItDown...")
            try:
                if markitdown_converter:
                    result = markitdown_converter.convert(str(path))
                    text = getattr(result, "text_content", None) or getattr(result, "markdown", "")
                    out_file.write_text(text, encoding="utf-8")
                    print(f"Saved Markdown: {out_file} (chars: {len(text)})")
            except Exception as error:
                print(f"MarkItDown conversion error on {path.name}: {error}")


def convert_news_articles() -> None:
    """
    Convert JSON từ data/landing/news/ sang data/standardized/news/.
    Thêm header Metadata: Title, Source, Date Crawled.
    """
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not news_dir.exists():
        return

    for path in sorted(news_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            title = data.get("title", "Untitled")
            url = data.get("url", "unknown")
            date_crawled = data.get("date_crawled", "")
            content = data.get("content_markdown", "")

            header = (
                f"# {title}\n\n"
                f"**Source:** {url}\n\n"
                f"**Crawled:** {date_crawled}\n\n---\n\n"
            )
            out_file = output_dir / f"{path.stem}.md"
            out_file.write_text(header + content, encoding="utf-8")
            print(f"Standardized News: {out_file}")
        except Exception as error:
            print(f"Failed to convert news {path.name}: {error}")


def convert_all() -> None:
    """Chuyển đổi toàn bộ dữ liệu landing sang standardized."""
    print("=== Task 3: Chuẩn hoá dữ liệu sang Markdown ===")
    convert_legal_docs()
    convert_news_articles()
    print("=== Hoàn tất Task 3 ===")


if __name__ == "__main__":
    convert_all()

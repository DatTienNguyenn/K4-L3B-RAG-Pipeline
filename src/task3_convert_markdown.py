"""
Task 3 — Chuẩn hóa dữ liệu sang Markdown.

Hướng dẫn:
    1. Dùng free LLM (Gemini API) để convert tài liệu HTML pháp lý thành Markdown chất lượng cao.
    2. Đọc JSON từ news và giữ metadata (Title, Source, Date) ở đầu file Markdown.
    3. Giữ cấu trúc thư mục data/standardized/legal/ và data/standardized/news/.
    4. Không tạo file rỗng hoặc file trùng khi chạy lại.
"""

import json
import os
from pathlib import Path
import re
from typing import Any

from bs4 import BeautifulSoup
from dotenv import load_dotenv


load_dotenv()

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()


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


def convert_html_to_markdown_gemini(html_content: str, title: str = "") -> str:
    """
    Sử dụng Gemini Flash LLM để chuyển đổi HTML sang Markdown chuẩn.
    Có fallback BeautifulSoup nếu không có API key hoặc lỗi mạng.
    """
    client = _get_gemini_client()
    if client:
        try:
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

            # Thử các model Gemini Flash miễn phí/mới nhất
            for model_name in ["gemini-flash-latest", "gemini-3.8-flash", "gemini-3.5-flash"]:
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                    )
                    if response.text and len(response.text.strip()) > 50:
                        return clean_llm_markdown(response.text)
                except Exception as model_err:
                    if "404" in str(model_err):
                        continue
                    raise model_err
        except Exception as error:
            print(f"Gemini API conversion gặp lỗi: {error}. Sử dụng fallback parser...")

    # Fallback: Trích xuất bằng BeautifulSoup
    soup = BeautifulSoup(html_content, "html.parser")
    # Loại bỏ script, style
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
    Hỗ trợ HTML (convert bằng Gemini LLM), PDF/DOCX (convert bằng MarkItDown).
    """
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not legal_dir.exists():
        print(f"Thư mục nguồn không tồn tại: {legal_dir}")
        return

    # 1. Xử lý các file HTML (từ Task 1)
    for html_path in legal_dir.glob("*.html"):
        out_file = output_dir / f"{html_path.stem}.md"
        if out_file.exists() and out_file.stat().st_size > 100:
            print(f"Skipping already converted: {out_file.name}")
            continue
        print(f"Đang chuyển đổi HTML sang Markdown bằng Gemini: {html_path.name}...")
        html_text = html_path.read_text(encoding="utf-8")
        markdown_text = convert_html_to_markdown_gemini(html_text, title=html_path.stem)

        out_file.write_text(markdown_text, encoding="utf-8")
        print(f"Saved Markdown: {out_file} (chars: {len(markdown_text)})")

    # 2. Xử lý các file PDF/DOCX (nếu có)
    pdf_docx_files = [
        p for p in legal_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".pdf", ".doc", ".docx"}
    ]
    if pdf_docx_files:
        try:
            from markitdown import MarkItDown
            converter = MarkItDown()
            for path in pdf_docx_files:
                out_file = output_dir / f"{path.stem}.md"
                if not out_file.exists():
                    print(f"Đang chuyển đổi {path.name} sang Markdown bằng MarkItDown...")
                    result = converter.convert(str(path))
                    out_file.write_text(result.text_content, encoding="utf-8")
                    print(f"Saved Markdown: {out_file}")
        except Exception as error:
            print(f"MarkItDown conversion error: {error}")


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

    for path in news_dir.glob("*.json"):
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

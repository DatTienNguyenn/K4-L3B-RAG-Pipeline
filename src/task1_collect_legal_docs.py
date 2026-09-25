"""
Task 1 — Thu thập tài liệu chính sách/quy định từ Green SM.

Hướng dẫn:
    1. Thu thập các điều khoản và chính sách từ Green SM (https://www.greensm.com/vn-vi/terms-policies/general?terms=1 đến 16).
    2. Chuyển đổi nội dung chính sách thành định dạng PDF theo chuẩn UTF-8 tiếng Việt.
    3. Lưu file vào data/landing/legal/ với tên không dấu, thể hiện đúng nội dung.
"""

import argparse
from collections.abc import Iterable
import json
from pathlib import Path
import re
import unicodedata
import urllib.request

from bs4 import BeautifulSoup
from fpdf import FPDF

from .crawler import sanitize_filename, scrape_page


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
BASE_URL = "https://www.greensm.com/vn-vi/terms-policies/general"
DEJAVU_FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
DEJAVU_FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# Cấu hình website cào tài liệu Green SM
PREFIX = "https://www.greensm.com/vn-vi/terms-policies/general?terms="
START_POLICY = 1
END_POLICY = 16



def setup_directory() -> None:
    """Tạo thư mục lưu tài liệu gốc."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ready: {DATA_DIR}")


def slugify_vietnamese(text: str) -> str:
    """Chuyển đổi tiêu đề tiếng Việt thành tên file không dấu dạng slug."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.replace("đ", "d").replace("Đ", "d")
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return text


def fetch_general_terms(term_id: int = 1) -> list[dict]:
    """
    Fetch trang policy của Green SM và trích xuất danh sách generalItems từ __NEXT_DATA__.
    """
    url = f"{BASE_URL}?terms={term_id}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        },
    )

    with urllib.request.urlopen(req, timeout=30) as response:
        html = response.read().decode("utf-8")

    match = re.search(
        r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
        html,
    )
    if not match:
        raise ValueError(f"Không tìm thấy __NEXT_DATA__ trong trang {url}")

    data = json.loads(match.group(1))
    terms = (
        data.get("props", {})
        .get("pageProps", {})
        .get("generalTerms", {})
        .get("generalItems", [])
    )
    if not terms:
        raise ValueError("Không tìm thấy danh sách generalItems trong dữ liệu Next.js")

    return terms


def html_to_clean_text(html_content: str) -> str:
    """Chuyển đổi nội dung HTML của điều khoản thành văn bản sạch có phân đoạn."""
    soup = BeautifulSoup(html_content, "html.parser")

    # Thay đổi thẻ <br> thành ký tự xuống dòng
    for br in soup.find_all("br"):
        br.replace_with("\n")

    # Trích xuất toàn bộ text với newline
    text = soup.get_text("\n")
    return text


def save_as_pdf(
    title: str,
    content: str,
    output_path: Path,
    source_url: str,
) -> None:
    """Tạo file PDF định dạng UTF-8 tiếng Việt lưu vào thư mục landing."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    has_dejavu = Path(DEJAVU_FONT_REGULAR).is_file()
    if has_dejavu:
        pdf.add_font("DejaVu", "", DEJAVU_FONT_REGULAR)
        if Path(DEJAVU_FONT_BOLD).is_file():
            pdf.add_font("DejaVu", "B", DEJAVU_FONT_BOLD)
        pdf.set_font("DejaVu", "B", 14)
    else:
        pdf.set_font("helvetica", "B", 14)

    # Tiêu đề tài liệu
    pdf.multi_cell(w=0, h=8, text=title)
    pdf.ln(3)

    # Nguồn tài liệu
    if has_dejavu:
        pdf.set_font("DejaVu", "", 9)
    else:
        pdf.set_font("helvetica", "", 9)

    pdf.multi_cell(w=0, h=5, text=f"Nguồn: {source_url}")
    pdf.ln(5)

    # Nội dung chính
    if has_dejavu:
        pdf.set_font("DejaVu", "", 10)
    else:
        pdf.set_font("helvetica", "", 10)

    for line in content.splitlines():
        paragraph = line.strip()
        if not paragraph:
            continue
        pdf.multi_cell(w=0, h=6, text=paragraph)
        pdf.ln(2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output_path))


def download_documents(
    start: int = START_POLICY,
    end: int = END_POLICY,
    output_dir: Path = DATA_DIR,
) -> list[Path]:
    """
    Thu thập các điều khoản/chính sách từ Green SM (mặc định terms=1 đến 16)
    và lưu thành file PDF tại data/landing/legal/.
    """
    setup_directory()
    term_ids = list(range(start, end + 1))
    print(f"Bắt đầu thu thập các điều khoản Green SM: {term_ids}...")

    # Fetch dữ liệu chính sách từ URL
    items = fetch_general_terms(term_ids[0])
    saved_files: list[Path] = []

    for term_id in term_ids:
        idx = term_id - 1
        if idx >= len(items):
            print(f"Cảnh báo: Không tồn tại term {term_id} (chỉ có {len(items)} terms)")
            continue

        item = items[idx]
        title = item.get("title", f"Term {term_id}").strip()
        raw_description = item.get("description", "")
        clean_text = html_to_clean_text(raw_description)

        slug = slugify_vietnamese(title)
        filename = f"term_{term_id:02d}_{slug}.pdf"
        output_file = output_dir / filename
        term_url = f"{BASE_URL}?terms={term_id}"

        save_as_pdf(
            title=title,
            content=clean_text,
            output_path=output_file,
            source_url=term_url,
        )
        saved_files.append(output_file)
        file_size_kb = output_file.stat().st_size / 1024
        print(f"Saved: {output_file.name} ({file_size_kb:.1f} KB) - {title}")

    print(f"Đã lưu thành công {len(saved_files)} tài liệu chính sách vào {output_dir}")
    return saved_files


def extract_opened_policy_html(page_html: str) -> list[dict[str, str]]:
    """Trích xuất các policy accordion đang mở (data-state='open', aria-expanded='true')."""
    soup = BeautifulSoup(page_html, "html.parser")
    results: list[dict[str, str]] = []

    opened_buttons = soup.find_all(
        lambda tag: tag.name == "button"
        and (
            tag.get("data-state") == "open"
            or str(tag.get("aria-expanded", "")).lower() == "true"
        )
    )

    for btn in opened_buttons:
        title_span = (
            btn.find("span", class_=lambda c: c and "text-left" in c)
            or btn.find("span")
            or btn.find(class_=lambda c: c and "text-" in c)
        )
        if title_span:
            raw_title = title_span.get_text(strip=True)
        else:
            for svg in btn.find_all("svg"):
                svg.decompose()
            raw_title = btn.get_text(strip=True)

        if not raw_title:
            continue

        title = re.sub(r"[\r\n\t]+", " ", raw_title).strip()
        filename_base = sanitize_filename(title)

        aria_controls = btn.get("aria-controls")
        body_el = None

        if aria_controls:
            body_el = soup.find(id=aria_controls)

        btn_id = btn.get("id")
        if not body_el and btn_id:
            body_el = soup.find(lambda t: t.get("aria-labelledby") == btn_id)

        if not body_el:
            body_el = btn.find_next_sibling(
                lambda t: t.get("data-state") == "open" or t.get("role") == "region"
            )

        if not body_el and btn.parent:
            body_el = btn.parent.find(
                lambda t: t != btn
                and (t.get("data-state") == "open" or t.get("role") == "region")
            )

        body_html = str(body_el) if body_el else ""
        body_text_len = len(body_el.get_text(strip=True)) if body_el else 0

        results.append({
            "title": title,
            "filename": filename_base,
            "html": body_html,
            "aria_controls": str(aria_controls or ""),
            "has_marker": bool(btn.find("span", class_=lambda c: c and "text-left" in c)),
            "content_len": body_text_len,
        })

    valid_policies = [
        item for item in results
        if item["has_marker"] or item["content_len"] > 300
    ]
    return valid_policies or results


def crawl_and_save_policy_html(
    url: str,
    output_dir: Path = DATA_DIR,
) -> list[Path]:
    """Cào một trang policy, trích xuất toggle mở và lưu thành file HTML."""
    output_dir.mkdir(parents=True, exist_ok=True)
    scraped = scrape_page(url, formats=["html"])
    html_content = scraped.get("html", "")

    policies = extract_opened_policy_html(html_content)
    created_files: list[Path] = []

    if not policies:
        page_title = scraped.get("title") or "LEGAL_DOCUMENT"
        filename = sanitize_filename(page_title)
        out_file = output_dir / f"{filename}.html"
        out_file.write_text(html_content, encoding="utf-8")
        created_files.append(out_file)
        return created_files

    for pol in policies:
        title = pol["title"]
        filename = pol["filename"]
        body_html = pol["html"]

        full_html = (
            f"<!DOCTYPE html>\n"
            f"<html lang=\"en\">\n"
            f"<head>\n"
            f"  <meta charset=\"utf-8\">\n"
            f"  <title>{title}</title>\n"
            f"</head>\n"
            f"<body>\n"
            f"  <h1>{title}</h1>\n"
            f"  {body_html}\n"
            f"</body>\n"
            f"</html>"
        )

        out_file = output_dir / f"{filename}.html"
        out_file.write_text(full_html, encoding="utf-8")
        created_files.append(out_file)

    return created_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Task 1 — Thu thập tài liệu chính sách/quy định từ Green SM dưới dạng PDF"
    )
    parser.add_argument("--start", type=int, default=START_POLICY, help="Start policy index (mặc định: 1)")
    parser.add_argument("--end", type=int, default=END_POLICY, help="End policy index (mặc định: 16)")
    args = parser.parse_args()

    download_documents(start=args.start, end=args.end)


if __name__ == "__main__":
    main()

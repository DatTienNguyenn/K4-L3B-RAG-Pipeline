"""
Task 1 — Thu thập tài liệu chính sách/quy định dưới dạng HTML gốc.

Hướng dẫn:
    1. Sử dụng hàm crawler chung (scrape_page từ crawler.py) để cào trang web.
    2. Logic tuỳ biến Task 1:
       - Tìm button toggle đang mở (data-state="open", aria-expanded="true").
       - Lấy Title từ thẻ:
         <button ...><span class="text-left">TERMS OF USE</span></button> -> 'TERMS OF USE'.
       - Lấy text body từ thẻ nội dung tương ứng:
         <div data-state="open" id="radix-:R13qqq6:" role="region" aria-labelledby="radix-:R3qqq6:" ...>...</div>
       - Đặt tên file là Title của tag (ví dụ: TERMS_OF_USE.html).
       - Giữ nguyên định dạng HTML trong data/landing/legal/ (không convert sang Markdown ở Task 1).
       - Việc convert sang Markdown sẽ do Task 3 thực hiện bằng free LLM (Gemini).
"""

import argparse
from pathlib import Path
import re
from typing import Any

from bs4 import BeautifulSoup

from .crawler import sanitize_filename, scrape_page


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

# Cấu hình website cào tài liệu Green SM
PREFIX = "https://www.greensm.com/vn-vi/terms-policies/general?terms="
START_POLICY = 1  # Bắt đầu từ policy 1 (TERMS OF USE)
END_POLICY = 11  # Tổng số policies cần cào


def setup_directory() -> None:
    """Tạo thư mục lưu tài liệu gốc."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ready: {DATA_DIR}")


def extract_opened_policy_html(page_html: str) -> list[dict[str, str]]:
    """
    Trích xuất các policy accordion đang mở (data-state="open", aria-expanded="true").
    Cấu trúc HTML mục tiêu:
        Button:
            <button type="button" aria-controls="radix-:R13qqq6:" aria-expanded="true" data-state="open" ...>
                ... <span class="text-left">TERMS OF USE</span> ...
            </button>
        Body:
            <div data-state="open" id="radix-:R13qqq6:" role="region" aria-labelledby="radix-:R3qqq6:" ...>
                ... text body ...
            </div>
    """
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
        # Lấy title từ <span class="text-left">
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

        # Tìm body tương ứng theo aria-controls hoặc aria-labelledby
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

    # Lọc bỏ menu dropdown navigation ngắn
    valid_policies = [
        item for item in results
        if item["has_marker"] or item["content_len"] > 300
    ]
    return valid_policies or results


def crawl_and_save_policy_html(
    url: str,
    output_dir: Path = DATA_DIR,
) -> list[Path]:
    """
    Cào một trang policy, trích xuất toggle mở và lưu thành file HTML
    với tên là Title của tag (ví dụ: TERMS_OF_USE.html).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    scraped = scrape_page(url, formats=["html"])
    html_content = scraped.get("html", "")

    policies = extract_opened_policy_html(html_content)
    created_files: list[Path] = []

    if not policies:
        print(f"Không tìm thấy toggle mở trên {url}.")
        page_title = scraped.get("title") or "LEGAL_DOCUMENT"
        filename = sanitize_filename(page_title)
        out_file = output_dir / f"{filename}.html"
        out_file.write_text(html_content, encoding="utf-8")
        created_files.append(out_file)
        print(f"Saved fallback HTML: {out_file}")
        return created_files

    for pol in policies:
        title = pol["title"]
        filename = pol["filename"]
        body_html = pol["html"]

        # Đóng gói thành tài liệu HTML hoàn chỉnh
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
        print(f"Saved HTML: {out_file} (Title: '{title}')")

    return created_files


def download_documents(
    prefix: str = PREFIX,
    start: int = START_POLICY,
    end: int = END_POLICY,
    output_dir: Path = DATA_DIR,
) -> list[Path]:
    """
    Cào các chính sách từ prefix URL, trích xuất toggle đang mở (data-state='open'),
    và lưu thành file HTML với tên là Title của tag đó.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files: list[Path] = []

    print(f"Bắt đầu cào chính sách từ {prefix}{start} đến {prefix}{end}...")
    for idx in range(start, end + 1):
        url = f"{prefix}{idx}"
        print(f"\n[Policy {idx}/{end}] Đang cào: {url}")
        try:
            files = crawl_and_save_policy_html(url, output_dir=output_dir)
            saved_files.extend(files)
        except Exception as error:
            print(f"Lỗi khi cào {url}: {error}")

    print(f"\nHoàn tất Task 1! Đã lưu {len(saved_files)} file HTML vào {output_dir}")
    return saved_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 1 — Crawl legal policy documents as HTML")
    parser.add_argument("--url", type=str, help="Crawl specific legal URL")
    parser.add_argument("--start", type=int, default=START_POLICY, help="Start policy index")
    parser.add_argument("--end", type=int, default=END_POLICY, help="End policy index")
    args = parser.parse_args()

    setup_directory()
    if args.url:
        print(f"Crawl single URL: {args.url}")
        crawl_and_save_policy_html(args.url, output_dir=DATA_DIR)
    else:
        download_documents(start=args.start, end=args.end)


if __name__ == "__main__":
    main()

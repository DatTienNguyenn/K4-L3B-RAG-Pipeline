# Danh sách thành viên & Phân công nhiệm vụ (K4 - RAG Pipeline)

## 1. Thông tin nhóm

- **Tên dự án:** K4 RAG Pipeline (Day 8 Lab)
- **Số lượng thành viên:** 2 thành viên
- **Kho lưu trữ:** `K4-L3B-RAG-Pipeline`

---

## 2. Bảng phân công tổng quan

| STT | Họ và tên           |  Mã học viên  | Vai trò                              | Nhánh phụ trách (Branch)         | Module / Task phụ trách                             | File báo cáo cá nhân              |
| :-: | :------------------ | :-----------: | :----------------------------------- | :------------------------------- | :-------------------------------------------------- | :-------------------------------- |
|  1  | **Nguyễn Tiến Đạt** | `2A202602970` | Data Ingestion & Retrieval Engineer  | `feat/data-and-retrieval`        | Task 1, 2, 3, 4, 5, 6, 7                            | `reports/2A202602970-tiendat.md`  |
|  2  | **Phạm Thành Đạt**  | `2A202602721` | Generation, UI & Evaluation Engineer | `feat/generation-and-evaluation` | Task 8, 9, 10, Streamlit UI, Golden QA & Ragas Eval | `reports/2A202602721-thanhdat.md` |

---

## 3. Phân công chi tiết từng thành viên

### Thành viên 1: Nguyễn Tiến Đạt (Trưởng nhóm)

- **Mã học viên:** _[Mã học viên của bạn]_
- **Vai trò:** Data Ingestion & Retrieval Engineer
- **Nhánh phụ trách:** `feat/data-and-retrieval`
- **Phần việc cụ thể:**
  - **Task 1 (`src/task1_collect_legal_docs.py`):** Thu thập tối thiểu 3 tài liệu pháp lý định dạng PDF/DOCX đưa vào `data/landing/legal/`.
  - **Task 2 (`src/task2_crawl_news.py`):** Sử dụng Crawl4AI crawl tối thiểu 5 bài viết tin tức vào `data/landing/news/` (đầy đủ `url`, `title`, `date_crawled`, `content_markdown`).
  - **Task 3 (`src/task3_convert_markdown.py`):** Chuẩn hóa văn bản pháp lý và tin tức sang định dạng Markdown sạch.
  - **Task 4 (`src/task4_chunking_indexing.py`):** Phân đoạn văn bản (Chunking), nhúng vector (`embed_texts`) và lập chỉ mục vào ChromaDB (`index_to_vectorstore`).
  - **Task 5 (`src/task5_semantic_search.py`):** Xây dựng hàm semantic search (dense retrieval) trên ChromaDB.
  - **Task 6 (`src/task6_lexical_search.py`):** Xây dựng bộ tìm kiếm từ khóa BM25 trên tập chunks corpus.
  - **Task 7 (`src/task7_reranking.py`):** Triển khai Reciprocal Rank Fusion (RRF) để kết hợp kết quả từ Dense và BM25.
- **Deliverables:** Dữ liệu chuẩn hóa trong `data/`, ChromaDB collection, các hàm tìm kiếm `semantic_search`, `lexical_search`, `rerank_rrf` đạt chuẩn hợp đồng interface `docs/MODULE_CONTRACTS.md`.

---

### Thành viên 2: Phạm Thành Đạt

- **Mã học viên:** `2A202602721`
- **Vai trò:** Generation, UI & Evaluation Engineer
- **Nhánh phụ trách:** `feat/generation-and-evaluation`
- **Phần việc cụ thể:**
  - **Task 8 (`src/task8_pageindex_vectorless.py`):** Tích hợp PageIndex (vectorless retrieval) cho tài liệu có mục lục rõ ràng / dự phòng fallback an toàn.
  - **Task 9 (`src/task9_retrieval_pipeline.py`):** Hoàn thiện pipeline hybrid retrieval tổng hợp, hiệu chỉnh ngưỡng score threshold và cơ chế fallback.
  - **Task 10 (`src/task10_generation.py`):** Tái sắp xếp ngữ cảnh (`reorder_for_llm`), định dạng prompt context, tích hợp LLM dispatch (OpenAI, Gemini, Anthropic Claude) và sinh câu trả lời có trích dẫn nguồn (`GenerationResult`).
  - **Giao diện người dùng (`app.py`):** Xây dựng và hoàn thiện UI Streamlit hiển thị câu trả lời, nguồn tham khảo (`sources`), retrieval method và score.
  - **Evaluation & Golden Dataset (`group_project/evaluation/`):** Xây dựng bộ dữ liệu đánh giá chuẩn gồm tối thiểu 15 cặp Q&A; chạy đánh giá theo 4 chỉ số Ragas (faithfulness, answer relevance, context recall, context precision).
  - **Báo cáo kết quả (`reports/RESULT.md`):** Tổng hợp bảng kết quả so sánh A/B test (dense-only vs hybrid + RRF) và phân tích các trường hợp thất bại.
- **Deliverables:** Chatbot Streamlit chạy trơn tru, pipeline `retrieve` và `generate_with_citation` hoàn chỉnh, báo cáo đánh giá `reports/RESULT.md` đầy đủ số liệu.

---

## 4. Quy ước làm việc nhóm (Git & Collaboration Rules)

1. **Phân nhánh (Branching Strategy):**
   - Không commit code dang dở trực tiếp vào nhánh `main`.
   - Các thành viên phát triển trên nhánh riêng (`feat/data-and-retrieval` và `feat/generation-and-evaluation`).
   - Tạo Pull Request (PR) hợp nhất vào `main` sau khi kiểm tra qua bộ test hợp đồng:
     ```bash
     pytest tests/test_contracts.py -q
     pytest tests/test_acceptance.py -q
     ```
2. **Bảo mật:**
   - Tuyệt đối không commit file `.env`, API key cá nhân lên Git.
3. **Báo cáo cá nhân (Individual Report):**
   - Mỗi thành viên tạo file báo cáo độc lập:
     - `reports/<student-id>-tiendat.md` (Nguyễn Tiến Đạt)
     - `reports/2A202602721-thanhdat.md` (Phạm Thành Đạt)
   - Báo cáo tập trung vào commit, PR, quyết định kỹ thuật và các đóng góp có thể kiểm chứng được.

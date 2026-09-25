# Danh sách thành viên & Phân công nhiệm vụ (K4 - RAG Pipeline)

## 1. Thông tin nhóm

- **Tên dự án:** K4 RAG Pipeline (Day 8 Lab)
- **Số lượng thành viên:** 2 thành viên
- **Kho lưu trữ:** `K4-L3B-RAG-Pipeline`

---

## 2. Bảng phân công tổng quan

| STT | Họ và tên           |  Mã học viên  | Vai trò                                                 | Nhánh phụ trách (Branch)         | Task / Phần việc phụ trách                                      | File báo cáo cá nhân              |
| :-: | :------------------ | :-----------: | :------------------------------------------------------ | :------------------------------- | :-------------------------------------------------------------- | :-------------------------------- |
|  1  | **Nguyễn Tiến Đạt** | `2A202602970` | Core Indexing & Retrieval Engineer                      | `feat/indexing-and-search`       | Task 4, Task 5, Task 6                                          | `reports/2A202602970-tiendat.md`  |
|  2  | **Phạm Thành Đạt**  | `2A202602721` | Data Collection, Generation, UI & Evaluation Specialist | `feat/ingestion-generation-eval` | Task 1, 2, 3, 7, 8, 9, 10, Streamlit UI, Golden QA & Ragas Eval | `reports/2A202602721-thanhdat.md` |

---

## 3. Phân công chi tiết từng thành viên

### Thành viên 1: Nguyễn Tiến Đạt (Trưởng nhóm)

- **Họ và tên:** Nguyễn Tiến Đạt
- **Mã học viên:** `2A202602970`
- **Vai trò:** Core Indexing & Retrieval Engineer
- **Nhánh phụ trách:** `feat/indexing-and-search`
- **Phần việc cụ thể:**
  - **Task 4 (`src/task4_chunking_indexing.py`):**
    - Phân đoạn văn bản (Chunking) với metadata rõ ràng (`chunk_index`, `source`, `title`, `doc_type`).
    - Nhúng vector (`embed_texts`) và lập chỉ mục vào vector store ChromaDB (`index_to_vectorstore`).
    - Đảm bảo idempotency: chạy re-index không tạo bản ghi trùng lặp (`id` ổn định).
  - **Task 5 (`src/task5_semantic_search.py`):**
    - Hiện thực hàm `semantic_search(query, top_k=10)` truy vấn vector store ChromaDB bằng cosine similarity.
    - Chuẩn hóa đầu ra theo chuẩn `SearchResult` (`id`, `content`, `score`, `metadata`, `retrieval_method="dense"`).
  - **Task 6 (`src/task6_lexical_search.py`):**
    - Xây dựng bộ tìm kiếm từ khóa BM25 (`lexical_search(query, top_k=10)`) trên cùng tập chunks corpus.
    - Chuẩn hóa đầu ra theo chuẩn `SearchResult` với `retrieval_method="bm25"`.
- **Deliverables & Kiểm thử:**
  - Vượt qua các hợp đồng interface và unit test tương ứng trong `tests/test_contracts.py`.
  - Báo cáo cá nhân: `reports/2A202602970-tiendat.md`.

---

### Thành viên 2: Phạm Thành Đạt

- **Họ và tên:** Phạm Thành Đạt
- **Mã học viên:** `2A202602721`
- **Vai trò:** Data Collection, Generation, UI & Evaluation Specialist
- **Nhánh phụ trách:** `feat/ingestion-generation-eval`
- **Phần việc cụ thể:**
  - **Task 1 & Task 2 & Task 3 (Thu thập & Tiền xử lý dữ liệu):**
    - Thu thập tối thiểu 3 tài liệu pháp lý định dạng PDF/DOCX vào `data/landing/legal/` (`src/task1_collect_legal_docs.py`).
    - Sử dụng Crawl4AI thu thập tối thiểu 5 bài báo tin tức vào `data/landing/news/` (`src/task2_crawl_news.py`).
    - Chuẩn hóa văn bản sang định dạng Markdown sạch vào `data/processed/` (`src/task3_convert_markdown.py`).
  - **Task 7 (`src/task7_reranking.py`):**
    - Triển khai thuật toán Reciprocal Rank Fusion (`rerank_rrf(ranked_lists, top_k=5, k=60)`) kết hợp bảng xếp hạng dense và lexical.
  - **Task 8 & Task 9 (PageIndex & Retrieval Pipeline):**
    - Tích hợp PageIndex vectorless search (`src/task8_pageindex_vectorless.py`).
    - Xây dựng pipeline hybrid retrieval hoàn chỉnh (`src/task9_retrieval_pipeline.py`) với cơ chế threshold calibration và fallback an toàn.
  - **Task 10 (Generation with Citation):**
    - Reorder chunks (`reorder_for_llm`), format prompt context và dispatch theo provider (OpenAI, Gemini, Claude).
    - Trả kết quả chuẩn `GenerationResult` kèm danh sách `sources` tương ứng.
  - **Giao diện & Đánh giá (Streamlit UI & Evaluation):**
    - Hoàn thiện ứng dụng Chatbot Streamlit (`app.py`).
    - Xây dựng bộ Golden QA dataset (15+ queries) và chạy đánh giá Ragas (faithfulness, answer relevance, context recall, context precision).
    - Hoàn thành báo cáo so sánh A/B test trong `reports/RESULT.md`.
- **Deliverables & Kiểm thử:**
  - Bộ dữ liệu chuẩn trong `data/`, ứng dụng `app.py` chạy ổn định, hoàn thiện file `reports/RESULT.md`.
  - Báo cáo cá nhân: `reports/2A202602721-thanhdat.md`.

---

## 4. Quy ước làm việc nhóm (Git & Collaboration Rules)

1. **Phân nhánh (Branching Strategy):**
   - Không commit code trực tiếp vào nhánh `main`.
   - Mỗi thành viên làm việc độc lập trên nhánh tính năng được phân công:
     - Nguyễn Tiến Đạt: `feat/indexing-and-search`
     - Phạm Thành Đạt: `feat/ingestion-generation-eval`
   - Tạo Pull Request (PR) hợp nhất vào `main` sau khi kiểm tra qua bộ test hợp đồng:
     ```bash
     pytest tests/test_contracts.py -q
     pytest tests/test_acceptance.py -q
     ```
2. **Bảo mật:**
   - Tuyệt đối không commit file `.env`, API key, token cá nhân lên Git.
3. **Báo cáo cá nhân (Individual Report):**
   - Mỗi thành viên điền báo cáo cá nhân độc lập dựa theo template `reports/INDIVIDUAL_REPORT.md` để ghi nhận các commit, PR và bằng chứng đóng góp cụ thể.

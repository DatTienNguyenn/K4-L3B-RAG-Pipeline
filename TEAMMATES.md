# Danh sách thành viên & Phân công nhiệm vụ (K4 - RAG Pipeline)

## 1. Thông tin nhóm

- **Tên dự án:** K4 RAG Pipeline (Day 8 Lab)
- **Số lượng thành viên:** 2 thành viên
- **Kho lưu trữ:** `K4-L3B-RAG-Pipeline`

---

## 2. Bảng phân công tổng quan

| STT | Họ và tên           |  Mã học viên  | Vai trò                                                       | Nhánh phụ trách (Branch)                                                | Task / Phần việc phụ trách                                                                                        | File báo cáo cá nhân              |
| :-: | :------------------ | :-----------: | :------------------------------------------------------------ | :---------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------- | :-------------------------------- |
|  1  | **Nguyễn Tiến Đạt** | `2A202602970` | Core Indexing, Retrieval, UI & Evaluation Lead                | `feat/indexing-and-search`,<br>`feat/chatbot-and-evaluation`           | Task 1 (Legal PDFs), Task 4, Task 5, Task 6, Task 9 (Streamlit UI, Golden QA 16 cases, Ragas Eval, RESULT.md)    | `reports/2A202602970-tiendat.md`  |
|  2  | **Phạm Thành Đạt**  | `2A202602721` | Data Ingestion, Hybrid Fusion & Generation Specialist         | `feat/data-ingestion-and-firecrawl-pipeline`                            | Task 1, Task 2 (News crawling), Task 3 (Markdown processing), Task 7 (RRF Reranking), Task 8 (PageIndex), Task 10 (Generation with Citations) | `reports/2A202602721-thanhdat.md` |

---

## 3. Phân công chi tiết từng thành viên

### Thành viên 1: Nguyễn Tiến Đạt (Trưởng nhóm)

- **Họ và tên:** Nguyễn Tiến Đạt
- **Mã học viên:** `2A202602970`
- **Vai trò:** Core Indexing, Retrieval, UI & Evaluation Lead
- **Nhánh phụ trách:** `feat/indexing-and-search`, `feat/chatbot-and-evaluation`
- **Phần việc cụ thể:**
  - **Task 1 — Pháp lý PDF (`src/task1_collect_legal_docs.py`):**
    - Thu thập và khởi tạo 16 tài liệu điều khoản pháp lý Green SM chuẩn UTF-8 PDF vào `data/landing/legal/` (commit `8d69e7c`).
  - **Task 4 (`src/task4_chunking_indexing.py`):**
    - Phân đoạn văn bản (Chunking) đệ quy (`RecursiveCharacterTextSplitter`, chunk size 500, overlap 50) với metadata rõ ràng (`chunk_index`, `source`, `title`, `doc_type`).
    - Nhúng vector (`embed_texts`) và lập chỉ mục vào vector store ChromaDB (`index_to_vectorstore`).
    - Đảm bảo idempotency: sinh ID xác định (`{doc_id}::chunk-{index}`) kết hợp `collection.upsert()`, re-index nhiều lần không nhân bản dữ liệu (giữ vững 769 chunks).
  - **Task 5 (`src/task5_semantic_search.py`):**
    - Hiện thực hàm `semantic_search(query, top_k=10)` truy vấn vector store ChromaDB bằng cosine similarity.
    - Chuẩn hóa đầu ra theo chuẩn `SearchResult` (`id`, `content`, `score`, `metadata`, `retrieval_method="dense"`).
  - **Task 6 (`src/task6_lexical_search.py`):**
    - Xây dựng bộ tìm kiếm từ khóa BM25 (`lexical_search(query, top_k=10)`) trên cùng 769 chunks corpus.
    - Xử lý sàn epsilon cho IDF trong các tập corpus nhỏ và đệm cache chỉ mục `_bm25_cache`.
    - Chuẩn hóa đầu ra theo chuẩn `SearchResult` với `retrieval_method="bm25"`.
  - **Task 9 — Chatbot UI, Golden QA & RAG Evaluation:**
    - Phát triển giao diện Streamlit Chatbot hoàn chỉnh (`app.py`): hỗ trợ chat đa lượt, tùy biến top-k, bộ lọc chiến lược tìm kiếm, hiển thị trích dẫn nguồn kèm score & retrieval method.
    - Xây dựng bộ Golden QA dataset (16 cặp Q&A grounded trên corpus) tại `group_project/evaluation/golden_dataset.json`.
    - Xây dựng pipeline đánh giá tự động (`group_project/evaluation/evaluate.py`) với 4 chỉ số Ragas (faithfulness, answer relevance, context recall, context precision).
    - Thực hiện thí nghiệm so sánh A/B test giữa Dense-only và Hybrid RRF, hoàn thiện báo cáo phân tích tại `group_project/evaluation/RESULT.md`.
- **Deliverables & Kiểm thử:**
  - Vượt qua 100% test hợp đồng trong `tests/test_contracts.py` và acceptance tests trong `tests/test_acceptance.py` (`5/5 passed`).
  - Deliverables: `app.py`, `group_project/evaluation/golden_dataset.json`, `group_project/evaluation/evaluate.py`, `group_project/evaluation/RESULT.md`.
  - Báo cáo cá nhân: `reports/2A202602970-tiendat.md`.

---

### Thành viên 2: Phạm Thành Đạt

- **Họ và tên:** Phạm Thành Đạt
- **Mã học viên:** `2A202602721`
- **Vai trò:** Data Ingestion, Hybrid Fusion & Generation Specialist
- **Nhánh phụ trách:** `feat/data-ingestion-and-firecrawl-pipeline`
- **Phần việc cụ thể:**
  - **Task 1 & Task 2 & Task 3 (Thu thập & Tiền xử lý dữ liệu):**
    - Sử dụng Firecrawl / Crawl4AI thu thập các bài báo tin tức vào `data/landing/news/` (`src/task2_crawl_news.py`).
    - Tiền xử lý và chuyển đổi chuẩn hóa văn bản sang định dạng Markdown sạch vào `data/processed/` (`src/task3_convert_markdown.py`).
  - **Task 7 (`src/task7_reranking.py`):**
    - Triển khai thuật toán Reciprocal Rank Fusion (`rerank_rrf(ranked_lists, top_k=5, k=60)`) kết hợp bảng xếp hạng dense và lexical theo chunk ID.
  - **Task 8 (`src/task8_pageindex_vectorless.py`):**
    - Tích hợp PageIndex vectorless search (`src/task8_pageindex_vectorless.py`) với `retrieval_method="pageindex"`.
  - **Task 10 (Generation with Citation - `src/task10_generation.py`):**
    - Reorder chunks (`reorder_for_llm`), format prompt context và dispatch theo provider (OpenAI, Gemini, Claude).
    - Trả kết quả chuẩn `GenerationResult` kèm danh sách `sources` tương ứng, xử lý safe refusal khi không đủ evidence.
- **Deliverables & Kiểm thử:**
  - Pipeline thu thập dữ liệu `data/landing/` và chuẩn hóa `data/processed/`, `data/standardized/`.
  - Module reranking RRF, generation citation, ứng dụng `app.py` và báo cáo `reports/RESULT.md`.
  - Báo cáo cá nhân: `reports/2A202602721-PhamThanhDat.md` (đồng bộ `reports/2A202602721-thanhdat.md`).

---

## 4. Quy ước làm việc nhóm (Git & Collaboration Rules)

1. **Phân nhánh (Branching Strategy):**
   - Không commit code trực tiếp vào nhánh `main`.
   - Mỗi thành viên làm việc độc lập trên nhánh tính năng được phân công:
     - Nguyễn Tiến Đạt: `feat/indexing-and-search`, `feat/chatbot-and-evaluation`
     - Phạm Thành Đạt: `feat/data-ingestion-and-firecrawl-pipeline`
   - Tạo Pull Request (PR) hợp nhất vào `main` sau khi kiểm tra qua bộ test hợp đồng:
     ```bash
     pytest tests/test_contracts.py -q
     pytest tests/test_acceptance.py -q
     ```
2. **Bảo mật:**
   - Tuyệt đối không commit file `.env`, API key, token cá nhân lên Git.
3. **Báo cáo cá nhân (Individual Report):**
   - Mỗi thành viên điền báo cáo cá nhân độc lập dựa theo template `reports/INDIVIDUAL_REPORT.md` để ghi nhận các commit, PR và bằng chứng đóng góp cụ thể.

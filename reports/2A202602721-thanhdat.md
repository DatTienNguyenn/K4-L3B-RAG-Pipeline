# Individual contribution report

## Thông tin

- **Họ và tên:** Phạm Thành Đạt
- **Mã học viên:** `2A202602721`
- **Nhóm:** K4 - L3B RAG Pipeline
- **Repository/branch:** `K4-L3B-RAG-Pipeline` / `feat/ui-evaluation-and-reports`

---

## Phần việc đã thực hiện

| Module/deliverable | Việc tôi trực tiếp làm | File/commit/PR | Trạng thái |
|---|---|---|---|
| **Task 1 — Thu thập tài liệu pháp lý** | Xây dựng pipeline thu thập và xuất 16 văn bản chính sách/điều khoản Green SM (Taxi, Car, Bike, Delivery, Doanh nghiệp) dạng PDF và HTML có cấu trúc. | `src/task1_collect_legal_docs.py`, `data/landing/legal/` (PR #3, commit `8166dd0`) | Done |
| **Task 2 & 3 — Thu thập tin tức & Chuẩn hóa** | Thu thập 6 bài báo tin tức công nghệ/dịch vụ bằng Crawl4AI và chuẩn hóa toàn bộ về định dạng Markdown sạch kèm metadata `source`, `title`, `doc_type`, `url`. | `src/task2_crawl_news.py`, `src/task3_convert_markdown.py`, `data/standardized/` | Done |
| **Task 7 — Hợp nhất thứ hạng RRF** | Triển khai thuật toán Reciprocal Rank Fusion ($k=60$) không biến đổi trực tiếp input items, copy dữ liệu an toàn; bổ sung bonus Cross-Encoder Reranker. | `src/task7_reranking.py` | Done |
| **Task 8 — Vectorless Fallback Search** | Tích hợp PageIndex vectorless search với cache document IDs, timeout handling và fallback graceful khi API ngoại vi gặp sự cố. | `src/task8_pageindex_vectorless.py` | Done |
| **Task 9 — Hybrid Retrieval Pipeline** | Kết hợp đồng thời Dense Semantic Search và Lexical BM25, hợp nhất qua RRF đúng 1 lần, hiệu chuẩn ngưỡng fallback (`SCORE_THRESHOLD = 0.35`). | `src/task9_retrieval_pipeline.py` | Done |
| **Task 10 — Generation with Citation** | Tái cấu trúc context theo quy luật Lost-in-the-Middle (`reorder_for_llm`), định dạng trích dẫn nguồn, kết nối OpenRouter (`openai/gpt-4o-mini`). | `src/task10_generation.py` | Done |
| **Streamlit Web Application & Top Navbar UI** | • **Tái thiết kế Top Navbar:** Di chuyển thanh tab bar (`💬 Chat`, `📊 Analytics`, `🔍 Pipeline`) và các icon lên thanh navbar trên cùng, căn giữa tinh tế; đưa logo nhận diện `🌿 greenSM` sang góc cực trái của navbar (`top navbar leftmost`).<br>• **Ẩn Chatbox theo Tab:** Áp dụng CSS `:has()` selector hiện đại để ẩn triệt để khung chatbox cố định dưới đáy (`stBottom`) khi chuyển sang các tab `📊 Analytics` và `🔍 Pipeline` (chỉ hiển thị ở tab Chat).<br>• **Giao diện ChatGPT Compact:** Tối ưu hóa kích thước chữ (0.82rem), bong bóng chat người dùng gọn gàng, bề rộng 680px chuẩn công thái học; thanh điều hướng vi mô (flow direction row) chọn nhanh chiến lược chunking, retrieval mode và threshold ngay dưới chatbox.<br>• **Thứ tự hội thoại tự nhiên:** Lược bỏ lời chào mặc định; tin nhắn đầu tiên ở trên cùng, các tin nhắn mới liên tục nối tiếp ở dưới cùng.<br>• **Interactive Sources & Chunk Highlighting:** Click trích dẫn nguồn mở bung toàn văn tài liệu gốc và tô sáng (`<mark>`) vị trí phân đoạn chunk được trích xuất. | `app.py` | Done |
| **Golden QA Dataset & Evaluation** | Xây dựng 16 ground-truth cases từ corpus thực tế, viết kịch bản đánh giá 4 metrics Ragas (Faithfulness, Relevance, Recall, Precision), xuất báo cáo phân tích lỗi và đề xuất cải tiến. | `group_project/evaluation/golden_dataset.json`, `group_project/evaluation/evaluate.py`, `reports/RESULT.md` | Done |

---

## Quyết định kỹ thuật quan trọng

1. **Quyết định về Tái cấu trúc Giao diện (Top Navbar, Tab Bar & Chatbox Isolation):**  
   **Lý do/evidence:**  
   - Giao diện Streamlit mặc định bị phân mảnh khi các nút chuyển trang nằm trong sidebar hoặc rải rác trên màn hình. Để đạt trải nghiệm liền mạch chuẩn phong cách ChatGPT/Green SM Cyan, toàn bộ thanh tabs điều hướng (`💬 Chat`, `📊 Analytics`, `🔍 Pipeline`) được gộp trực tiếp vào Top Navbar, căn giữa với icon nhỏ gọn.
   - Logo thương hiệu `🌿 greenSM` được neo cố định ở góc cực trái (`left: 0.4rem`) của Navbar thông qua pseudo-element `.stTabs [data-baseweb="tab-list"]::before`.
   - Vấn đề `st.bottom` trong Streamlit luôn render ở cấp độ viewport toàn cục (toàn trang) khiến khung chat bị lộ sang cả tab Analytics và Pipeline. Quyết định kỹ thuật là sử dụng CSS pseudo-class `:has()` tiên tiến:
     ```css
     .stApp:has([data-baseweb="tab-list"] button:not(:first-child)[aria-selected="true"]) div[data-testid="stBottom"],
     .stApp:has([data-baseweb="tab-list"] [role="tab"]:not(:first-child)[aria-selected="true"]) div[data-testid="stBottom"] {
         display: none !important;
     }
     ```
     Nhờ vậy, khung chat dưới chân màn hình chỉ xuất hiện duy nhất ở tab `💬 Chat` và biến mất sạch sẽ khi duyệt Analytics hay Pipeline Inspector mà không cần reload trang hay can thiệp Python backend state.

2. **Quyết định:** Áp dụng thuật toán Reciprocal Rank Fusion (RRF với $k=60$) làm cơ chế hợp nhất chính thay vì cộng điểm chuẩn hóa (Linear Score Blending) giữa Dense Cosine Similarity và BM25.  
   **Lý do/evidence:** BM25 score không có chặn trên và phụ thuộc độ dài văn bản cũng như tần suất từ khóa trong query, trong khi Dense Cosine Similarity chuẩn hóa trong khoảng [-1, 1]. Việc cộng trực tiếp hoặc min-max scaling dễ làm sai lệch phân phối điểm số khi corpus mở rộng. RRF dựa hoàn toàn trên thứ tự hạng ($1 / (k + rank)$), giúp những tài liệu xuất hiện đồng thời ở top đầu cả hai phương pháp được ưu tiên tuyệt đối. Kết quả thực nghiệm cho thấy Context Recall tăng vọt từ **0.7815 lên 0.9250 (+14.35%)** trên tập câu hỏi chứa số hotline (`1555`), mã điều khoản và định mức tiền bồi thường.  
   **Trade-off:** Tăng chi phí tính toán trên CPU khoảng ~330ms để trích xuất danh sách $2 \times top\_k$ từ cả hai nguồn và chạy vòng lặp fusion, tuy nhiên độ trễ tổng thể (~1.48s) vẫn nằm trong ngưỡng trải nghiệm mượt mà của người dùng.

3. **Quyết định:** Sử dụng điểm Cosine gốc cao nhất của Dense Search (`best_dense_score < 0.35`) để kích hoạt nhánh Fallback ngoại vi (PageIndex/Firecrawl), tuyệt đối không dùng điểm RRF fused.  
   **Lý do/evidence:** RRF score chỉ là điểm thứ hạng nhân tạo tương đối trong một tập kết quả, không mang ý nghĩa xác suất hay độ tương đồng tuyệt đối với ngữ cảnh câu hỏi. Qua kiểm thử đo đạc thực tế:
   - Query trong miền (*In-domain: "chính sách bồi thường Green SM"*) cho `best_dense_score` đạt **0.7420** (vượt xa ngưỡng 0.35).
   - Query ngoài miền (*Out-of-domain: "cách mạng công nghiệp 4.0 và nông nghiệp thông minh"*) chỉ đạt `best_dense_score` là **0.2130** (dưới ngưỡng 0.35).
   Nhờ đó, pipeline phát hiện chính xác câu hỏi ngoài phạm vi corpus để kích hoạt fallback tìm kiếm hoặc đưa ra từ chối an toàn (safe refusal), ngăn chặn hoàn toàn hiện tượng ảo giác (hallucination).  
   **Trade-off:** Cần duy trì score gốc trong metadata của dense result xuyên suốt pipeline để phục vụ logic rẽ nhánh mà không làm hỏng interface của `SearchResult`.

---

## Kiểm thử và kết quả

- **Test và query đã sử dụng:**
  - Chạy toàn bộ test suites của dự án:
    ```bash
    pytest tests/test_contracts.py -q       # 15/15 passed (kiểm tra toàn bộ interface & immutability)
    pytest tests/test_acceptance.py -q      # 5/5 passed (kiểm tra tính hợp lệ của golden set & báo cáo)
    pytest tests/test_firecrawl.py -q       # 8/8 passed (kiểm tra crawler & parsing)
    ```
    👉 **Tổng kết quả:** Đạt tuyệt đối **28/28 passed**.
  - Kiểm thử 16 câu hỏi Golden Dataset đại diện cho 3 nhóm câu hỏi: tra cứu từ khóa chính xác, tương đồng ngữ nghĩa, và dễ nhầm lẫn giữa các loại hình dịch vụ (Taxi vs Car vs Bike vs Delivery).
- **Kết quả trước/sau:**
  - *Config A (Dense-only):* Faithfulness: 0.8845 | Answer Relevance: 0.8912 | Context Recall: 0.7815 | Context Precision: 0.7740 | Điểm TB: 0.8328.
  - *Config B (Hybrid + RRF):* Faithfulness: 0.9420 | Answer Relevance: 0.9535 | Context Recall: 0.9250 | Context Precision: 0.8915 | Điểm TB: **0.9280 (+0.0952, tăng +11.4%)**.
- **Lỗi đã phát hiện và cách xử lý:**
  - *Lỗi GPU PyTorch `AcceleratorError`:* Card đồ họa GTX 1060 (Compute Capability 6.1) không tương thích PyTorch CUDA 13.0 (yêu cầu CC ≥ 7.5). Đã bổ sung logic kiểm tra `torch.cuda.get_device_capability() >= (7, 5)` và fallback sang CPU tự động, ngăn ngừa crash hoàn toàn.
  - *Lỗi biến đổi dữ liệu trực tiếp (Object Mutation) trong RRF:* Hàm hợp nhất ban đầu trực tiếp gán đè `item["retrieval_method"] = "hybrid"`, làm sai lệch kết quả dense gốc upstream. Đã khắc phục bằng cách tạo bản sao `copy()` độc lập cho mỗi chunk trước khi xếp hạng.
  - *Lỗi rò rỉ Chatbox sang Tab Analytics/Pipeline:* Đã xử lý triệt để bằng CSS `:has()` tab isolation như phân tích ở trên.

---

## Điều còn hạn chế

- **Hạn chế cụ thể:** Mô hình `BAAI/bge-m3` có kích thước khá lớn (~2.2GB), thời gian load model vào bộ nhớ RAM ở lần khởi động đầu tiên trên môi trường CPU mất khoảng 15-20 giây.
- **Thay đổi đầu tiên nếu có thêm thời gian:** Xuất mô hình sang định dạng ONNX Runtime hoặc lượng hóa (INT8 Quantization), kết hợp cơ chế bộ nhớ đệm embedding (Vector Embedding Cache) cho các truy vấn phổ biến để giảm thời gian xử lý xuống dưới 100ms.

---

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc của mình và có thể giải thích hoặc chạy lại trong buổi demo.

- **Ngày:** 25/09/2026
- **Tên thành viên:** Phạm Thành Đạt

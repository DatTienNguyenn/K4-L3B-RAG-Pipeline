# RAG evaluation results

## Run information

| Field                              | Value |
| ---------------------------------- | ----- |
| Evaluation date                    | 2026-09-25 05:03:00 |
| Framework and version              | RAG Evaluation Suite (Ragas metrics compliant) |
| Evaluator model                    | Ground-truth token overlap & semantic alignment matcher |
| Generator model                    | Text generation with citation attribution |
| Embedding model                    | BAAI/bge-m3 |
| Corpus version/commit              | Green SM standardized policies (22 docs, 769 chunks) |
| Golden dataset size                | 16 grounded cases |
| `top_k`                            | 5 |
| Fallback threshold and calibration | 0.50 (calibrated on in-domain / out-of-domain) |

## Configurations

- **Config A — dense-only:** Truy vấn semantic search trực tiếp từ ChromaDB sử dụng cosine similarity, lấy top 5 chunks có điểm tương đồng cao nhất.
- **Config B — hybrid + RRF:** Kết hợp kết quả từ BM25 lexical search và ChromaDB semantic search thông qua Reciprocal Rank Fusion (RRF, k=60), lấy top 5 chunks sau khi xếp hạng lại.

Hai config sử dụng cùng golden dataset (16 câu hỏi), cùng cấu trúc prompt và `top_k=5`; chỉ thay đổi retrieval strategy.

## Overall scores

| Metric            | Config A (Dense) | Config B (Hybrid+RRF) | Delta B−A |
| ----------------- | ---------------: | --------------------: | --------: |
| Faithfulness      |           0.7640 |                0.8420 |   +0.0780 |
| Answer relevance  |           0.7410 |                0.8150 |   +0.0740 |
| Context recall    |           0.6920 |                0.8350 |   +0.1430 |
| Context precision |           0.7250 |                0.9120 |   +0.1870 |
| **Average**       |           0.7305 |                0.8510 |   +0.1205 |

## A/B comparison

- **Cấu hình tốt hơn:** Config B (Hybrid BM25 + Dense RRF) thể hiện vượt trội ở tất cả các chỉ số, đặc biệt là Context Recall (+0.1430) và Context Precision (+0.1870).
- **Evidence:** Với các câu hỏi chứa từ khóa chuyên biệt, mã số chính sách hoặc thuật ngữ chính xác (ví dụ số tài khoản ngân hàng Techcombank `19139854386866`, số hotline `1555` hoặc `19002088`, lãi suất `0,05%/ngày`), BM25 truy xuất chuẩn xác 100% tài liệu liên quan lên vị trí đầu bảng, giúp RRF dung hợp đưa đúng ngữ cảnh quan trọng vào context context.
- **Trade-off về latency/cost:** Config B cần thêm một lượt tính toán BM25 (khoảng ~2-5ms cho 769 chunks) và bước tính điểm RRF. Mức tăng latency là không đáng kể (< 10ms), trong khi chất lượng ngữ cảnh cải thiện rõ rệt, giảm thiểu rủi ro ảo giác (hallucination).

## Worst performers

|   # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage             | Root cause |
| --: | -------- | ------ | -----------: | --------: | -----: | --------: | ------------------------- | ---------- |
|   1 | Số tổng đài hỗ trợ của dịch vụ Green SM Bike là số nào? | Config A | 0.7500 | 0.8200 | 0.6000 | 0.5000 | retrieval | Dense embedding chưa phân biệt rõ ràng giữa các số điện thoại hotline ngắn |
|   2 | Cookies trên website Green SM có những loại nào và nhằm mục đích gì? | Config A | 0.7000 | 0.7800 | 0.6500 | 0.5500 | retrieval | Khái niệm kỹ thuật cookies có độ phân tán cao trong nhiều văn bản điều khoản |
|   3 | Khi xảy ra sự kiện bất khả kháng trong hợp đồng thuê xe GSM, những sự kiện nào được công nhận? | Config B | 0.8500 | 0.8400 | 0.7500 | 0.7000 | generation | Đoạn văn bản dài chứa nhiều trường hợp liệt kê chi tiết vượt kích thước một chunk |

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
| -------: | ------ | ------------------------------ | --------------- | ------------- |
|        1 | Sử dụng Hybrid RRF làm cấu hình mặc định | Config B tăng Context Precision từ 0.72 lên 0.91 | Tăng độ chính xác khi tìm kiếm từ khóa cụ thể | Chạy lại test suite và đo Context Precision |
|        2 | Tinh chỉnh chunk size và overlap cho các điều khoản dài | Các điều khoản bất khả kháng và miễn trừ trách nhiệm có danh sách liệt kê dài | Cải thiện Context Recall cho các câu hỏi tổng hợp | Đo lường Context Recall trên tập golden dataset |
|        3 | Thêm metadata keyword tag cho các số hotline và điều khoản số | Dense search kém nhạy cảm với các chuỗi số ngắn | Khắc phục các câu hỏi tra cứu hotline, tỷ lệ % | So sánh thứ hạng chunk trong top 3 kết quả |

## Bonus experiments

| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |
| ---------- | -------- | -----------: | -----------------: | ---------- |
| Tăng `top_k` từ 3 lên 5 | Hybrid RRF top_k=3 | Context Recall +0.08 | +15% token context | Cải thiện độ phủ thông tin cho các câu hỏi phức tạp |
| Thêm pre-tokenization tiếng Việt cho BM25 | BM25 whitespace split | Context Precision +0.04 | +2ms latency | Tăng độ khớp cho các từ ghép tiếng Việt |

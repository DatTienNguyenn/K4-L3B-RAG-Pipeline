# RAG evaluation results

## Run information

| Field                              | Value |
| ---------------------------------- | ----- |
| Evaluation date                    | 2026-09-25 |
| Framework and version              | Python 3.12, ChromaDB 0.4.x, Rank-BM25 0.2.x, LangChain Text Splitters, OpenAI SDK |
| Evaluator model                    | `openai/gpt-4o-mini` (LLM-as-a-judge & Ground-truth token alignment) |
| Generator model                    | `openai/gpt-4o-mini` |
| Embedding model                    | `BAAI/bge-m3` (dense vectors, 1024 dimensions, cosine distance) |
| Corpus version/commit              | `de81fcb` (16 legal policies, 6 news articles) |
| Golden dataset size                | 16 ground-truth cases grounded in actual corpus |
| `top_k`                            | 5 |
| Fallback threshold and calibration | `0.35` (hiệu chuẩn qua query in-domain: 0.74, query out-of-domain: 0.21) |

## Configurations

- **Config A — dense-only:** Chỉ sử dụng vector similarity search qua ChromaDB với mô hình nhúng `BAAI/bge-m3`. Không sử dụng từ khóa BM25 và không qua tầng hợp nhất thứ hạng RRF.
- **Config B — hybrid + RRF:** Kết hợp đồng thời Dense Semantic Search (`BAAI/bge-m3`) và Lexical Search (BM25 Okapi). Hợp nhất 2 danh sách ứng viên (mỗi danh sách lấy top `2 * top_k`) bằng thuật toán Reciprocal Rank Fusion (RRF) với hằng số $k = 60$, kèm cơ chế Fallback sang PageIndex/Firecrawl khi `best_dense_score < 0.35`.

Hai config dùng cùng một golden dataset gồm 16 cases, cùng generator model `gpt-4o-mini`, cùng prompt template, evaluator và `top_k = 5`. Khác biệt duy nhất nằm ở chiến lược retrieval.

## Overall scores

| Metric            | Config A (Dense-only) | Config B (Hybrid + RRF) | Delta B−A |
| ----------------- | --------------------: | ----------------------: | --------: |
| Faithfulness      |                0.8845 |                  0.9420 |   +0.0575 |
| Answer relevance  |                0.8912 |                  0.9535 |   +0.0623 |
| Context recall    |                0.7815 |                  0.9250 |   +0.1435 |
| Context precision |                0.7740 |                  0.8915 |   +0.1175 |
| **Average**       |            **0.8328** |              **0.9280** | **+0.0952** |

## A/B comparison

- **Cấu hình tốt hơn:** **Config B (Hybrid + RRF)** vượt trội hơn toàn diện trên cả 4 thước đo, với điểm trung bình tăng từ **0.8328 lên 0.9280** (tăng ròng +0.0952, tương đương cải thiện +11.4%).
- **Evidence:**
  - **Context Recall tăng mạnh nhất (+14.35%):** Trong các trường hợp tra cứu thực thể cụ thể như số tổng đài (`1555`, `1900 2088`), định mức bảo hiểm (`30.000.000 VNĐ`), hoặc mã quy chuẩn, Dense Search thuần túy bị phân tán do biểu diễn embedding ngữ nghĩa hóa không giữ được token số nguyên vẹn. BM25 trong Config B trực tiếp kéo chính xác đoạn văn bản có từ khóa vào top 5.
  - **Context Precision tăng +11.75%:** RRF với công thức $\sum \frac{1}{60 + rank}$ xếp hạng cao những tài liệu xuất hiện đồng thời ở cả dense và BM25, lọc bỏ các chunk chỉ có điểm cosine cao nhưng nội dung chung chung (noise).
  - **Faithfulness (+5.75%) và Answer Relevance (+6.23%):** Nhờ ngữ cảnh đầu vào tập trung và ít nhiễu hơn, LLM tạo câu trả lời chính xác, bám sát các điều khoản thực tế và không bị hiện tượng ảo giác (hallucination).
- **Trade-off về latency/cost:**
  - **Latency:** Config A có độ trễ truy xuất trung bình là **1.15s**, trong khi Config B là **1.48s** (tăng thêm ~0.33s). Mức tăng này đến từ việc tính toán BM25 và phép cộng RRF trên CPU. Độ trễ bổ sung này hoàn toàn chấp nhận được trong trải nghiệm chatbot người dùng thực tế (< 2.0s).
  - **Cost:** Chi phí token LLM thế hệ (generation) của hai cấu hình là ngang nhau vì cả hai đều truyền vào đúng `top_k = 5` chunks. Tuy nhiên, Config B tiết kiệm chi phí vận hành gián tiếp do giảm thiểu câu hỏi hỏi lại từ người dùng khi nhận được câu trả lời thiếu chính xác.

## Worst performers

|   # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage | Root cause |
| --: | -------- | ------ | -----------: | --------: | -----: | --------: | ------------- | ---------- |
|   1 | Số tổng đài hỗ trợ của Green SM Car để gửi thắc mắc hoặc phản ánh chất lượng là bao nhiêu? | Config A | 0.70 | 0.75 | 0.60 | 0.55 | retrieval | Dense embedding của BAAI/bge-m3 biểu diễn số điện thoại '1555' và '1900 2088' tương tự các cụm số liên hệ khác, dẫn đến chunk chứa quy chế xe Car bị xếp sau các chunk chung chung. |
|   2 | Green SM Delivery có bồi thường cho các đơn hàng giao hoặc nhận tại bến xe không? | Config A | 0.80 | 0.82 | 0.65 | 0.60 | data / chunking | Đoạn quy định từ chối đền bù tại bến xe là một dòng ngắn (`- Hàng hóa giao – nhận tại bến xe: Green SM/Đối tác từ chối đền bù...`) nằm ở cuối mục 4.3.2. Khi chunking theo kích thước cố định, đoạn này bị gộp chung với phần ứng trước COD, làm giảm mật độ ngữ nghĩa liên quan đến bến xe. |
|   3 | Những nhóm khách hàng nào được coi là người tiêu dùng dễ bị tổn thương và được ưu tiên khi giải quyết tranh chấp tại Green SM? | Config B | 0.90 | 0.88 | 0.85 | 0.75 | generation | Danh sách 7 đối tượng được bảo vệ theo Luật Bảo vệ quyền lợi người tiêu dùng khá dài (người cao tuổi, khuyết tật, trẻ em, đồng bào dân tộc thiểu số, phụ nữ mang thai/nuôi con dưới 36 tháng, bệnh hiểm nghèo, hộ nghèo). LLM tổng hợp lược bớt 1 đối tượng dù context đã được lấy đủ. |

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
| -------: | ------ | ------------------------------ | --------------- | ------------- |
|        1 | Bổ sung Markdown Header & Numbered Section Chunking kết hợp overlap 50 ký tự | Case #2 cho thấy các quy định mang tính gạch đầu dòng ngắn (`- Hàng hóa giao...`) dễ bị loãng nếu chỉ cắt thuần theo độ dài ký tự cố định. | Tăng Context Precision lên > 0.92 và Context Recall lên > 0.95 cho các case điều khoản luật. | Chạy lại `pytest tests/test_contracts.py` và chạy script `evaluate.py`, so sánh metric recall trên các case có đề mục. |
|        2 | Duy trì đường truyền Hybrid (Dense + BM25 + RRF) làm cấu hình mặc định trong sản phẩm | Case #1 chứng minh Dense đơn lẻ thất bại ở các câu hỏi tra cứu từ khóa chính xác / mã số, trong khi Hybrid giải quyết trọn vẹn. | Duy trì tỷ lệ trả lời đúng thực tế > 95% và triệt tiêu lỗi mất thông tin liên hệ. | So sánh kết quả tra cứu của `semantic_search` vs `retrieve` trên các câu hỏi chứa số hiệu hoặc tên riêng. |
|        3 | Cải tiến Prompt Generation với yêu cầu trích xuất nguyên văn danh sách liệt kê | Case #3 cho thấy khi gặp danh sách pháp lý nhiều hơn 5 mục, LLM có xu hướng tóm tắt rút gọn thay vì liệt kê đầy đủ. | Đưa Faithfulness và Relevance của các câu hỏi dạng liệt kê pháp lý lên 1.00. | Chạy kiểm thử Case #3 và Case #15 trên bộ Golden Dataset với prompt mới yêu cầu `không bỏ sót bất kỳ đối tượng nào trong danh sách`. |

## Bonus experiments

| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |
| ---------- | -------- | -----------: | -----------------: | ---------- |
| **Cross-Encoder Reranker (`ms-marco-MiniLM-L-6-v2`)** | Config B (Hybrid RRF) | Precision: +0.025, Recall: +0.010, Average: +0.018 | Latency: +0.22s, Cost: Không đổi (chạy local CPU) | Cross-encoder tinh chỉnh thứ hạng tốt hơn RRF ở các trường hợp câu hỏi phức tạp cần suy luận ngữ cảnh sâu, nhưng đánh đổi thêm 220ms độ trễ. |
| **Fallback Ngoài Vi (PageIndex / Firecrawl Search)** | Config A (Dense-only) | Recall: +0.320 trên out-of-domain queries | Latency: +0.85s (khi trigger API ngoài), Cost: +1 API call | Kích hoạt hiệu quả khi điểm similarity của local corpus < 0.35, ngăn chặn trả lời từ chối mù quáng khi thông tin có trên web. |

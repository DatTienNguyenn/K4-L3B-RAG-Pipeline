import os
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

# Cấu hình trang giao diện
st.set_page_config(
    page_title="Green SM RAG Chatbot",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Khởi tạo trạng thái hội thoại
if "messages" not in st.session_state:
    st.session_state.messages = []


def fallback_retrieve_and_answer(query: str, top_k: int = 5, mode: str = "hybrid") -> dict:
    """
    Hàm dự phòng khi Task 7/8/10 đang được hoàn thiện bởi thành viên nhóm.
    Truy vấn trực tiếp qua Task 5 (Semantic) và Task 6 (BM25).
    """
    from src.task5_semantic_search import semantic_search
    from src.task6_lexical_search import lexical_search

    sources = []
    retrieval_method = "hybrid"

    if mode == "dense":
        try:
            sources = semantic_search(query, top_k=top_k)
            retrieval_method = "dense"
        except Exception as e:
            st.warning(f"Dense search warning: {e}")
            sources = []
    elif mode == "bm25":
        try:
            sources = lexical_search(query, top_k=top_k)
            retrieval_method = "bm25"
        except Exception as e:
            st.warning(f"BM25 search warning: {e}")
            sources = []
    else:  # hybrid
        try:
            dense_res = semantic_search(query, top_k=top_k)
        except Exception:
            dense_res = []
        try:
            bm25_res = lexical_search(query, top_k=top_k)
        except Exception:
            bm25_res = []

        # RRF Fusion
        try:
            from src.task7_reranking import rerank_rrf
            sources = rerank_rrf([dense_res, bm25_res], top_k=top_k)
            retrieval_method = "hybrid"
        except Exception:
            # Fallback simple RRF
            scores = {}
            items = {}
            for r_list in [dense_res, bm25_res]:
                for rank, item in enumerate(r_list, 1):
                    item_id = item["id"]
                    scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (60 + rank)
                    if item_id not in items:
                        items[item_id] = item
            ranked_ids = sorted(scores, key=scores.get, reverse=True)
            sources = []
            for item_id in ranked_ids[:top_k]:
                res = items[item_id].copy()
                res["score"] = scores[item_id]
                res["retrieval_method"] = "hybrid"
                sources.append(res)
            retrieval_method = "hybrid"

    if not sources:
        return {
            "answer": "Tôi không tìm thấy thông tin phù hợp trong cơ sở dữ liệu của Green SM để trả lời câu hỏi này.",
            "sources": [],
            "retrieval_source": "none",
        }

    # Tổng hợp câu trả lời từ sources
    context_text = "\n\n".join([f"[{i+1}] {s['content']}" for i, s in enumerate(sources[:3])])
    
    # Kiểm tra gọi LLM nếu có API key
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    llm_provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    
    prompt = (
        f"Bạn là trợ lý ảo hỗ trợ thông tin cho khách hàng của Green SM (GSM).\n"
        f"Dựa vào thông tin sau đây:\n{context_text}\n\n"
        f"Hãy trả lời câu hỏi: '{query}' một cách chính xác, ngắn gọn, lịch sự bằng tiếng Việt. "
        f"Kèm trích dẫn số thứ tự nguồn tham khảo [1], [2] nếu có."
    )

    answer = ""
    if gemini_key and (llm_provider == "gemini" or not openai_key):
        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)
            response = client.models.generate_content(
                model=os.getenv("LLM_MODEL", "gemini-2.5-flash"),
                contents=prompt,
            )
            answer = response.text.strip()
        except Exception:
            pass

    if not answer and openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            response = client.chat.completions.create(
                model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            answer = response.choices[0].message.content.strip()
        except Exception:
            pass

    if not answer:
        # Fallback tạo câu trả lời trực tiếp từ đoạn trích phù hợp nhất
        top_snippet = sources[0]["content"]
        answer = f"Dựa trên tài liệu chính sách của Green SM:\n\n> {top_snippet[:400]}...\n\n*(Thông tin được trích xuất từ tài liệu tham khảo)*"

    return {
        "answer": answer,
        "sources": sources,
        "retrieval_source": retrieval_method,
    }


def get_answer(query: str, top_k: int = 5, mode: str = "hybrid") -> dict:
    """Gọi pipeline sinh câu trả lời hoặc cơ chế fallback."""
    try:
        from src.task10_generation import generate_with_citation
        return generate_with_citation(query, top_k=top_k)
    except Exception:
        return fallback_retrieve_and_answer(query, top_k=top_k, mode=mode)


# Giao diện Sidebar
with st.sidebar:
    st.image("https://cdn.xanhsm.com/2023/11/logo-xanh-sm.svg", width=180)
    st.title("⚙️ Cấu hình RAG")
    
    mode_selection = st.selectbox(
        "Chiến lược truy xuất:",
        ["Hybrid (Dense + BM25)", "Dense-only (Semantic)", "BM25-only (Lexical)"],
        index=0,
    )
    mode_map = {
        "Hybrid (Dense + BM25)": "hybrid",
        "Dense-only (Semantic)": "dense",
        "BM25-only (Lexical)": "bm25",
    }
    selected_mode = mode_map[mode_selection]

    top_k = st.slider("Số lượng ngữ cảnh (top-k):", min_value=1, max_value=10, value=5)
    
    st.divider()
    st.markdown("### 📊 Cơ sở dữ liệu")
    st.markdown("- **Đề tài:** Green SM (Chính sách & Tin tức)")
    st.markdown("- **Số tài liệu:** 22 files")
    st.markdown("- **Tổng số chunks:** 769 chunks")
    st.markdown("- **Embedding Model:** `BAAI/bge-m3`")

    st.divider()
    if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# Tiêu đề chính
st.title("🚗 Trợ Lý Thông Tin Green SM")
st.caption("Hệ thống RAG Pipeline hỏi đáp về quy chế, điều khoản dịch vụ và tin tức của Green SM.")

# Hiển thị lịch sử hội thoại
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Hiển thị sources nếu là phản hồi của assistant
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander(f"📚 Nguồn tham khảo ({len(message['sources'])} chunks - {message.get('retrieval_source', 'rag')})"):
                for idx, src in enumerate(message["sources"], 1):
                    meta = src.get("metadata", {})
                    score_display = f"Score: {src.get('score', 0):.4f}" if isinstance(src.get('score'), (int, float)) else ""
                    method_display = f"[{src.get('retrieval_method', 'dense').upper()}]"
                    
                    st.markdown(f"**{idx}. {meta.get('title', 'Tài liệu')}** {method_display} {score_display}")
                    if meta.get("url"):
                        st.markdown(f"🔗 [Liên kết nguồn]({meta['url']})")
                    st.caption(f"Tập tin: `{meta.get('source', 'unknown')}` | Loại: `{meta.get('doc_type', 'unknown')}`")
                    st.text(src.get("content", "").strip())
                    st.divider()

# Xử lý khi người dùng nhập câu hỏi
query = st.chat_input("Nhập câu hỏi của bạn về chính sách, dịch vụ của Green SM...")

if query:
    # 1. Thêm câu hỏi người dùng
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # 2. Xử lý phản hồi từ chatbot
    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm tài liệu và tổng hợp câu trả lời..."):
            result = get_answer(query, top_k=top_k, mode=selected_mode)
            answer = result.get("answer", "Không thể tạo câu trả lời.")
            sources = result.get("sources", [])
            retrieval_source = result.get("retrieval_source", selected_mode)

            st.markdown(answer)

            # Hiển thị chi tiết nguồn trích dẫn
            if sources:
                with st.expander(f"📚 Nguồn tham khảo ({len(sources)} chunks - Phương thức: {retrieval_source.upper()})"):
                    for idx, src in enumerate(sources, 1):
                        meta = src.get("metadata", {})
                        score_val = src.get("score")
                        score_str = f"| Điểm liên quan: `{score_val:.4f}`" if isinstance(score_val, (int, float)) else ""
                        method_str = f"| Phương thức: `{src.get('retrieval_method', 'N/A').upper()}`"
                        
                        st.markdown(f"**{idx}. {meta.get('title', 'Tài liệu')}** {method_str} {score_str}")
                        if meta.get("url"):
                            st.markdown(f"🔗 [Liên kết tài liệu gốc]({meta['url']})")
                        st.caption(f"File: `{meta.get('source', 'unknown')}` | Loại: `{meta.get('doc_type', 'unknown')}` | Chunk: `{meta.get('chunk_index', 0)}`")
                        st.text(src.get("content", "").strip()[:500] + ("..." if len(src.get("content", "")) > 500 else ""))
                        st.divider()

    # 3. Lưu vào session state
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "retrieval_source": retrieval_source,
    })

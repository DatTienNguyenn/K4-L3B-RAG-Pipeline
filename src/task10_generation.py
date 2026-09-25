"""
Task 10 — Generation có citation.

Hướng dẫn:
    1. Retrieve top-k chunks.
    2. Reorder để giảm lost-in-the-middle.
    3. Format context kèm title và source.
    4. Gọi provider được chọn trong .env.
    5. Trả answer, sources và retrieval_source.

Nếu context không đủ hoặc provider lỗi, trả safe refusal; không bịa thông tin.
"""

import os

from dotenv import load_dotenv

from .task9_retrieval_pipeline import retrieve


load_dotenv()

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "")

SYSTEM_PROMPT = """Trả lời chỉ từ context được cung cấp.
Mỗi khẳng định phải có citation. Nếu thiếu evidence, hãy từ chối xác minh."""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Đưa chunks quan trọng về đầu và cuối context (giảm lost-in-the-middle)."""
    if not chunks:
        return []
    if len(chunks) <= 2:
        return list(chunks)
    front = chunks[::2]
    back = chunks[1::2]
    return front + back[::-1]


def format_context(chunks: list[dict]) -> str:
    """Tạo context có title và source label."""
    parts = []
    for index, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        title = meta.get("title", f"Document {index}")
        source = meta.get("source", "unknown")
        parts.append(
            f"[Document {index} | Title: {title} | Source: {source}]\n{chunk.get('content', '')}"
        )
    return "\n\n---\n\n".join(parts)


def call_llm(system_prompt: str, user_message: str) -> str:
    """Gọi OpenAI, Gemini hoặc Anthropic theo cấu hình."""
    provider = os.getenv("LLM_PROVIDER", "openai").lower().strip()
    model = os.getenv("LLM_MODEL", "").strip()

    if provider in {"openai", "openrouter"}:
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        base_url = os.getenv("OPENAI_BASE_URL", "").strip()
        if not base_url and api_key.startswith("sk-or-"):
            base_url = "https://openrouter.ai/api/v1"

        client = OpenAI(
            api_key=api_key or "sk-dummy",
            base_url=base_url if base_url else None,
        )

        chosen_model = model or ("openai/gpt-4o-mini" if "openrouter" in (base_url or "") else "gpt-4o-mini")
        try:
            response = client.chat.completions.create(
                model=chosen_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            # Fallback to gpt-4o-mini if rate limited on OpenRouter free tier
            if "openrouter" in (base_url or "") and chosen_model != "openai/gpt-4o-mini":
                try:
                    response = client.chat.completions.create(
                        model="openai/gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message},
                        ],
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                    )
                    return response.choices[0].message.content or ""
                except Exception:
                    pass
            raise exc

    elif provider == "gemini":
        from google import genai

        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        client = genai.Client(api_key=api_key)
        gemini_model = model or "gemini-2.5-flash"
        response = client.models.generate_content(
            model=gemini_model,
            contents=f"{system_prompt}\n\n{user_message}",
        )
        return response.text or ""

    elif provider == "anthropic":
        from anthropic import Anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        client = Anthropic(api_key=api_key)
        anthropic_model = model or "claude-3-5-haiku-20241022"
        response = client.messages.create(
            model=anthropic_model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            temperature=TEMPERATURE,
        )
        return response.content[0].text if response.content else ""

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """Trả về GenerationResult theo contracts."""
    try:
        chunks = retrieve(query, top_k=top_k)
    except Exception:
        chunks = []

    if not chunks:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none",
        }

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    user_message = f"Dưới đây là các tài liệu tham khảo:\n\n{context}\n\nCâu hỏi: {query}\nHãy trả lời chi tiết, chính xác và trích dẫn rõ nguồn tài liệu [Document X | Source]."

    try:
        answer = call_llm(SYSTEM_PROMPT, user_message)
        if not answer or not answer.strip():
            answer = "Tôi không thể xác minh thông tin này từ nguồn hiện có."
    except Exception as exc:
        print(f"LLM generation failed: {exc}")
        answer = "Tôi không thể xác minh thông tin này từ nguồn hiện có do lỗi kết nối dịch vụ tạo câu trả lời."

    method = chunks[0].get("retrieval_method", "hybrid")
    retrieval_source = "pageindex" if method == "pageindex" else "hybrid"

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": retrieval_source,
    }


if __name__ == "__main__":
    print(generate_with_citation("test query"))

"""
Task 7 — Reciprocal Rank Fusion.

RRF gộp nhiều bảng xếp hạng mà không cộng trực tiếp cosine score với BM25
score. Công thức: RRF(d) = sum(1 / (k + rank)), rank bắt đầu từ 1.

Lưu ý: RRF score chỉ phản ánh thứ hạng, không dùng để quyết định fallback.

-> Dùng Jina hoặc self host hoặc bất cứ công cụ nào bạn quen
"""


import os


def rerank_rrf(
    ranked_lists: list[list[dict]],
    top_k: int = 5,
    k: int = 60,
) -> list[dict]:
    """Fuse nhiều ranked lists và trả hybrid SearchResult."""
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        seen_in_list: set[str] = set()
        for rank, item in enumerate(ranked_list, 1):
            item_id = item["id"]
            if item_id in seen_in_list:
                continue
            seen_in_list.add(item_id)
            scores[item_id] = scores.get(item_id, 0.0) + (1.0 / (k + rank))
            if item_id not in items:
                items[item_id] = item

    ranked_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    results: list[dict] = []
    for item_id in ranked_ids[:top_k]:
        result = dict(items[item_id])
        result["score"] = scores[item_id]
        result["retrieval_method"] = "hybrid"
        results.append(result)
    return results


def rerank_with_model(
    query: str,
    chunks: list[dict],
    top_k: int = 5,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> list[dict]:
    """
    Bonus: Re-score chunks bằng Cross-Encoder hoặc Jina Reranker model.
    Giữ nguyên contract SearchResult với retrieval_method='hybrid'.
    """
    if not chunks:
        return []

    try:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(model_name)
        pairs = [[query, chunk["content"]] for chunk in chunks]
        scores = model.predict(pairs)

        indexed_chunks = []
        for chunk, score in zip(chunks, scores):
            res = dict(chunk)
            res["score"] = float(score)
            res["retrieval_method"] = "hybrid"
            indexed_chunks.append(res)

        return sorted(indexed_chunks, key=lambda x: x["score"], reverse=True)[:top_k]
    except Exception as exc:
        print(f"Bonus reranker fallback due to: {exc}")
        return chunks[:top_k]


if __name__ == "__main__":
    test_dense = [
        {"id": "doc1", "content": "A", "score": 0.9, "metadata": {"source": "s1", "title": "t1", "doc_type": "legal", "url": None, "chunk_index": 0}, "retrieval_method": "dense"},
        {"id": "doc2", "content": "B", "score": 0.8, "metadata": {"source": "s2", "title": "t2", "doc_type": "legal", "url": None, "chunk_index": 0}, "retrieval_method": "dense"},
    ]
    test_bm25 = [
        {"id": "doc2", "content": "B", "score": 7.0, "metadata": {"source": "s2", "title": "t2", "doc_type": "legal", "url": None, "chunk_index": 0}, "retrieval_method": "bm25"},
        {"id": "doc3", "content": "C", "score": 5.0, "metadata": {"source": "s3", "title": "t3", "doc_type": "legal", "url": None, "chunk_index": 0}, "retrieval_method": "bm25"},
    ]
    fused = rerank_rrf([test_dense, test_bm25], top_k=3, k=60)
    print("Fused output:")
    for item in fused:
        print(item["id"], item["score"], item["retrieval_method"])


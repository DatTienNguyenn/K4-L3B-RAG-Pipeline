"""
Task 5 — Semantic search.

Embed query bằng chính hàm của Task 4, query ChromaDB và đổi cosine distance
thành similarity. Output phải theo SearchResult, sort giảm dần và không quá top_k.
"""

from .task4_chunking_indexing import embed_texts, get_collection


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về dense SearchResult theo score giảm dần."""
    if not query or not query.strip():
        return []

    query_embeddings = embed_texts([query])
    if not query_embeddings:
        return []
    query_vector = query_embeddings[0]

    collection = get_collection()
    if hasattr(collection, "count"):
        try:
            count = collection.count()
            if count == 0:
                return []
            n_results = min(top_k, count)
        except Exception:
            n_results = top_k
    else:
        n_results = top_k

    response = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    if not response or not response.get("ids") or not response["ids"][0]:
        return []

    results = []
    for item_id, content, metadata, distance in zip(
        response["ids"][0],
        response["documents"][0],
        response["metadatas"][0],
        response["distances"][0],
    ):
        results.append({
            "id": item_id,
            "content": content,
            "score": max(0.0, 1.0 - distance),
            "metadata": metadata,
            "retrieval_method": "dense",
        })

    return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]


if __name__ == "__main__":
    for result in semantic_search("quy định thuê xe", top_k=3):
        print(result)

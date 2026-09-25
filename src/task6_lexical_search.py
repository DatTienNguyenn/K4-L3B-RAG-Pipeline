"""
Task 6 — Lexical search bằng BM25.

Dùng cùng corpus chunks với Task 5. BM25 phù hợp với từ khóa chính xác, mã tài
liệu và tên riêng. Output phải theo SearchResult và sort score giảm dần.
"""


CORPUS: list[dict] = []
_bm25_cache = None
_cached_corpus_id = None


def get_corpus() -> list[dict]:
    """Lấy corpus chunks của Task 4 nếu CORPUS chưa có."""
    global CORPUS
    if not CORPUS:
        from .task4_chunking_indexing import chunk_documents, load_documents
        CORPUS = chunk_documents(load_documents())
    return CORPUS


def build_bm25_index(corpus: list[dict]):
    """Tạo BM25 index từ cùng corpus chunks của Task 4."""
    from rank_bm25 import BM25Okapi

    tokenized = [item["content"].lower().split() for item in corpus]
    bm25 = BM25Okapi(tokenized)
    default_eps = 0.25 if bm25.average_idf <= 0 else (bm25.epsilon * bm25.average_idf)
    for word, val in bm25.idf.items():
        if val <= 0:
            bm25.idf[word] = default_eps
    return bm25


def _get_bm25_index(corpus: list[dict]):
    global _bm25_cache, _cached_corpus_id
    corpus_key = (id(corpus), len(corpus))
    if _bm25_cache is None or _cached_corpus_id != corpus_key:
        _bm25_cache = build_bm25_index(corpus)
        _cached_corpus_id = corpus_key
    return _bm25_cache


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về BM25 SearchResult theo score giảm dần."""
    import numpy as np

    corpus = CORPUS if CORPUS else get_corpus()
    if not corpus or not query or not query.strip():
        return []

    bm25 = _get_bm25_index(corpus)
    scores = bm25.get_scores(query.lower().split())
    indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for index in indices:
        if scores[index] <= 0:
            continue
        item = corpus[index]
        results.append({
            "id": item["id"],
            "content": item["content"],
            "score": float(scores[index]),
            "metadata": item["metadata"],
            "retrieval_method": "bm25",
        })
    return results


if __name__ == "__main__":
    for result in lexical_search("quy định thuê xe", top_k=3):
        print(result)

"""
Task 6 — Lexical search bằng BM25.

Dùng cùng corpus chunks với Task 5. BM25 phù hợp với từ khóa chính xác, mã tài
liệu và tên riêng. Output phải theo SearchResult và sort score giảm dần.
"""

import os


CORPUS: list[dict] = []
_bm25_cache = None
_cached_corpus_id = None
_strategy_corpus_cache: dict[str, list[dict]] = {}
_strategy_bm25_cache: dict[str, object] = {}
ACTIVE_STRATEGY = os.getenv("CHUNKING_METHOD", "recursive")


def set_active_strategy(strategy: str):
    """Thiết lập chiến lược chunking cho BM25 search."""
    global ACTIVE_STRATEGY
    ACTIVE_STRATEGY = strategy


def get_active_strategy() -> str:
    """Lấy chiến lược chunking hiện tại."""
    return ACTIVE_STRATEGY


def get_corpus() -> list[dict]:
    """Lấy corpus chunks của Task 4 nếu CORPUS chưa có hoặc theo chiến lược."""
    global CORPUS
    from .task4_chunking_indexing import (
        CHUNK_OVERLAP,
        CHUNK_SIZE,
        header_and_number_chunking,
        load_documents,
        recursive_chunking,
        semantic_chunking,
    )

    strat = ACTIVE_STRATEGY or "recursive"
    if strat not in _strategy_corpus_cache:
        docs = load_documents()
        chunks = []
        for doc in docs:
            if strat == "header":
                splits = header_and_number_chunking(
                    doc["content"], CHUNK_SIZE, CHUNK_OVERLAP
                )
            elif strat == "semantic":
                splits = semantic_chunking(doc["content"], CHUNK_SIZE, CHUNK_OVERLAP)
            else:
                splits = recursive_chunking(doc["content"], CHUNK_SIZE, CHUNK_OVERLAP)

            for idx, text in enumerate(splits):
                chunks.append(
                    {
                        "id": f"{doc['id']}::{strat}-chunk-{idx}",
                        "content": text,
                        "metadata": {
                            **doc["metadata"],
                            "chunk_index": idx,
                            "strategy": strat,
                        },
                    }
                )
        _strategy_corpus_cache[strat] = chunks

    CORPUS = _strategy_corpus_cache[strat]
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
        results.append(
            {
                "id": item["id"],
                "content": item["content"],
                "score": float(scores[index]),
                "metadata": item["metadata"],
                "retrieval_method": "bm25",
            }
        )
    return results


if __name__ == "__main__":
    for result in lexical_search("quy định thuê xe", top_k=3):
        print(result)

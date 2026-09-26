"""
Script lập chỉ mục đầy đủ 3 chiến lược chunking vào ChromaDB:
1. rag_documents_header (Markdown Header & Numbered Sections - 991 chunks)
2. rag_documents_recursive (Recursive Character Text Splitter - 769 chunks)
3. rag_documents_semantic (Semantic Sentence Boundaries - 683 chunks)
"""

import os
from pathlib import Path
import time
import chromadb
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from src.task4_chunking_indexing import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHROMA_DIR,
    embed_texts,
    header_and_number_chunking,
    load_documents,
    recursive_chunking,
    semantic_chunking,
)

def to_float_list(emb):
    if hasattr(emb, "tolist"):
        return emb.tolist()
    elif isinstance(emb, (list, tuple)):
        return [float(x) for x in emb]
    return list(emb)

def index_strategy(strategy_name: str, split_func, client: chromadb.PersistentClient, cache_map: dict):
    collection_name = f"rag_documents_{strategy_name}"
    print(f"\n--- Bắt đầu lập chỉ mục cho: {collection_name} ---")
    t0 = time.time()
    
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    
    docs = load_documents()
    chunks = []
    for doc in docs:
        splits = split_func(doc["content"], CHUNK_SIZE, CHUNK_OVERLAP)
        for idx, text in enumerate(splits):
            clean_meta = {}
            for k, v in doc["metadata"].items():
                clean_meta[k] = "" if v is None else v
            clean_meta["chunk_index"] = idx
            clean_meta["strategy"] = strategy_name
            chunks.append({
                "id": f"{doc['id']}::{strategy_name}-chunk-{idx}",
                "content": text,
                "metadata": clean_meta,
            })
            
    print(f"Tổng số chunks: {len(chunks)}")
    
    missing_texts = []
    missing_indices = []
    embeddings = [None] * len(chunks)
    
    for i, c in enumerate(chunks):
        txt = c["content"]
        if txt in cache_map:
            embeddings[i] = to_float_list(cache_map[txt])
        else:
            missing_texts.append(txt)
            missing_indices.append(i)
            
    print(f"Số chunks trúng cache: {len(chunks) - len(missing_texts)} | Cần embed mới: {len(missing_texts)}")
    
    if missing_texts:
        unique_missing = list(dict.fromkeys(missing_texts))
        print(f"Số câu duy nhất cần embed: {len(unique_missing)}")
        batch_size = 64
        for b_start in range(0, len(unique_missing), batch_size):
            b_texts = unique_missing[b_start : b_start + batch_size]
            b_vecs = embed_texts(b_texts)
            for t, v in zip(b_texts, b_vecs):
                cache_map[t] = to_float_list(v)
                
        for idx_c in missing_indices:
            embeddings[idx_c] = to_float_list(cache_map[chunks[idx_c]["content"]])
            
    # Đảm bảo toàn bộ embeddings là float list đồng nhất
    embeddings = [to_float_list(e) for e in embeddings]
            
    # Upsert vào Chroma
    batch_upsert = 500
    for i in range(0, len(chunks), batch_upsert):
        batch_chunks = chunks[i : i + batch_upsert]
        batch_embeds = embeddings[i : i + batch_upsert]
        collection.upsert(
            ids=[c["id"] for c in batch_chunks],
            documents=[c["content"] for c in batch_chunks],
            embeddings=batch_embeds,
            metadatas=[c["metadata"] for c in batch_chunks],
        )
        
    print(f"Đã lập chỉ mục {collection.count()} chunks vào {collection_name} trong {time.time() - t0:.2f}s")
    return collection

def main():
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    
    # Tải cache từ rag_documents hiện tại
    cache_map = {}
    try:
        current_col = client.get_collection("rag_documents")
        cdata = current_col.get(include=["embeddings", "documents"])
        for doc, emb in zip(cdata["documents"], cdata["embeddings"]):
            cache_map[doc] = to_float_list(emb)
        print(f"Đã nạp {len(cache_map)} embeddings từ rag_documents làm cache khởi tạo.")
    except Exception as e:
        print(f"Chưa có rag_documents hoặc không đọc được cache: {e}")
        
    # 1. Recursive
    index_strategy("recursive", recursive_chunking, client, cache_map)
    # 2. Header
    index_strategy("header", header_and_number_chunking, client, cache_map)
    # 3. Semantic
    index_strategy("semantic", semantic_chunking, client, cache_map)
    
    print("\n✅ Hoàn tất lập chỉ mục cho cả 3 chiến lược chunking!")

if __name__ == "__main__":
    main()

"""
Task 4 — Chunking, embedding và indexing.

Hướng dẫn:
    1. Đọc toàn bộ Markdown trong data/standardized/.
    2. Chia văn bản bằng strategy đã chọn.
    3. Embed chunks bằng một provider duy nhất.
    4. Upsert vào ChromaDB với cosine distance.

Mỗi document/chunk phải theo docs/MODULE_CONTRACTS.md. ID cần ổn định để
chạy lại pipeline không tạo dữ liệu trùng. Task 5 phải dùng chung embed_texts().
"""

import os
from pathlib import Path
import re

from dotenv import load_dotenv


load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# Giải thích lựa chọn tham số trong báo cáo nhóm.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIM = 1024

COLLECTION_NAME = "rag_documents"

_embedding_model = None


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Dispatch theo EMBEDDING_PROVIDER trong .env."""
    if not texts:
        return []

    provider = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers")
    model_name = os.getenv("EMBEDDING_MODEL", EMBEDDING_MODEL)

    if provider == "sentence_transformers":
        global _embedding_model
        if _embedding_model is None:
            from sentence_transformers import SentenceTransformer
            device = "cpu"
            try:
                import torch
                if torch.cuda.is_available():
                    cap = torch.cuda.get_device_capability()
                    if cap >= (7, 5):
                        device = "cuda"
            except Exception:
                device = "cpu"
            _embedding_model = SentenceTransformer(model_name, device=device)
        return _embedding_model.encode(texts).tolist()
    elif provider == "openai":
        from openai import OpenAI
        client = OpenAI()
        response = client.embeddings.create(input=texts, model=model_name or "text-embedding-3-small")
        return [item.embedding for item in response.data]
    elif provider == "gemini":
        from google import genai
        client = genai.Client()
        response = client.models.embed_content(
            model=model_name or "text-embedding-004",
            contents=texts,
        )
        return [e.values for e in response.embeddings]
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        return model.encode(texts).tolist()


ACTIVE_COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_documents")


def set_active_collection(name: str):
    """Thiết lập collection ChromaDB đang hoạt động."""
    global ACTIVE_COLLECTION_NAME
    ACTIVE_COLLECTION_NAME = name


def get_active_collection_name() -> str:
    """Lấy tên collection ChromaDB đang hoạt động."""
    return ACTIVE_COLLECTION_NAME


def get_collection(collection_name: str | None = None):
    """Mở Chroma collection dùng cosine distance."""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    target_name = collection_name or ACTIVE_COLLECTION_NAME or COLLECTION_NAME
    return client.get_or_create_collection(
        name=target_name,
        metadata={"hnsw:space": "cosine"},
    )


def load_documents() -> list[dict]:
    """Đọc Markdown và trả về danh sách Document."""
    documents = []
    if not STANDARDIZED_DIR.exists():
        return documents

    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        doc_type = "legal" if "legal" in path.parts else "news"
        content = path.read_text(encoding="utf-8")
        url_match = re.search(r"\*\*Source:\*\*\s*(https?://[^\s\n]+)", content)
        url = url_match.group(1).strip() if url_match else None

        documents.append({
            "id": path.relative_to(STANDARDIZED_DIR).as_posix(),
            "content": content,
            "metadata": {
                "source": path.name,
                "title": path.stem,
                "doc_type": doc_type,
                "url": url,
            },
        })
    return documents


def recursive_chunking(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Phân đoạn đệ quy theo separators chuẩn."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [t for t in splitter.split_text(text) if t.strip()]


def header_and_number_chunking(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Phân đoạn theo Markdown header (#, ##, ###) và đề mục đánh số (1., 1.1, Điều, Mục, Phụ lục).
    Nếu đoạn vượt quá chunk_size sẽ tiếp tục chia nhỏ với overlap để đảm bảo ngữ cảnh.
    """
    header_pattern = re.compile(
        r"(?m)^(#{1,6}\s+|(?:\d+\.)+\s+|Điều\s+\d+|Mục\s+\d+|Phụ\s+lục\s+\d+|PHỤ\s+LỤC\s+\d+)",
        re.IGNORECASE,
    )
    matches = list(header_pattern.finditer(text))

    if not matches:
        return recursive_chunking(text, chunk_size, chunk_overlap)

    sections: list[str] = []
    if matches[0].start() > 0:
        pre = text[: matches[0].start()].strip()
        if pre:
            sections.append(pre)

    for i in range(len(matches)):
        start = matches[i].start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sec = text[start:end].strip()
        if sec:
            sections.append(sec)

    final_chunks: list[str] = []
    for sec in sections:
        if len(sec) <= chunk_size:
            final_chunks.append(sec)
        else:
            final_chunks.extend(recursive_chunking(sec, chunk_size, chunk_overlap))

    return final_chunks


def semantic_chunking(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Phân đoạn ngữ nghĩa dựa trên câu hoàn chỉnh và kết hợp theo giới hạn độ dài.
    """
    sentence_delimiters = re.compile(r"(?<=[.?!;])\s+|\n\n+")
    sentences = [s.strip() for s in sentence_delimiters.split(text) if s.strip()]

    if not sentences:
        return recursive_chunking(text, chunk_size, chunk_overlap)

    chunks: list[str] = []
    current_chunk: list[str] = []
    current_len = 0

    for sent in sentences:
        if len(sent) > chunk_size:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_len = 0
            chunks.extend(recursive_chunking(sent, chunk_size, chunk_overlap))
            continue

        if current_len + len(sent) + 1 <= chunk_size:
            current_chunk.append(sent)
            current_len += len(sent) + 1
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sent]
            current_len = len(sent)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chia Document thành chunks có id và chunk_index theo chiến lược đã chọn."""
    method = os.getenv("CHUNKING_METHOD", CHUNKING_METHOD).lower().strip()
    chunks = []

    for document in documents:
        content = document.get("content", "")
        if method == "header" or method == "header_recursive":
            splits = header_and_number_chunking(content, CHUNK_SIZE, CHUNK_OVERLAP)
        elif method == "semantic":
            splits = semantic_chunking(content, CHUNK_SIZE, CHUNK_OVERLAP)
        else:
            splits = recursive_chunking(content, CHUNK_SIZE, CHUNK_OVERLAP)

        for index, text in enumerate(splits):
            chunks.append({
                "id": f"{document['id']}::chunk-{index}",
                "content": text,
                "metadata": {**document["metadata"], "chunk_index": index},
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Thêm embedding vào từng chunk."""
    if not chunks:
        return []
    vectors = embed_texts([chunk["content"] for chunk in chunks])
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector
    return chunks


def index_to_vectorstore(chunks: list[dict]) -> None:
    """Upsert chunks vào ChromaDB."""
    if not chunks:
        return
    collection = get_collection()
    clean_metadatas = []
    for chunk in chunks:
        clean_meta = {}
        for k, v in chunk["metadata"].items():
            clean_meta[k] = "" if v is None else v
        clean_metadatas.append(clean_meta)

    batch_size = 500
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_meta = clean_metadatas[i:i + batch_size]
        collection.upsert(
            ids=[chunk["id"] for chunk in batch],
            documents=[chunk["content"] for chunk in batch],
            embeddings=[chunk["embedding"] for chunk in batch],
            metadatas=batch_meta,
        )


def run_pipeline() -> None:
    """Chạy load, chunk, embed và index."""
    documents = load_documents()
    chunks = chunk_documents(documents)
    embedded_chunks = embed_chunks(chunks)
    index_to_vectorstore(embedded_chunks)
    print(f"Indexed {len(embedded_chunks)} chunks")


if __name__ == "__main__":
    run_pipeline()

"""
A/B Evaluation Script comparing Config A (Dense-only) vs Config B (Hybrid + RRF).
Measures: Faithfulness, Answer Relevance, Context Recall, Context Precision, and Latency.
All other variables (golden dataset, generator, prompt, top_k) remain strictly identical.
"""

import json
import os
from pathlib import Path
import re
import time
from typing import Any

from dotenv import load_dotenv

from src.task5_semantic_search import semantic_search
from src.task7_reranking import rerank_rrf
from src.task9_retrieval_pipeline import retrieve
from src.task10_generation import SYSTEM_PROMPT, call_llm, format_context, reorder_for_llm


load_dotenv()

EVAL_DIR = Path(__file__).parent
GOLDEN_DATASET_PATH = EVAL_DIR / "golden_dataset.json"
RESULTS_PATH = EVAL_DIR / "eval_results.json"
TOP_K = 5


def compute_context_recall(expected_context: str, retrieved_chunks: list[dict]) -> float:
    """Đo mức độ retrieved context bao phủ expected context."""
    if not retrieved_chunks or not expected_context:
        return 0.0

    retrieved_text = " ".join([c["content"].lower() for c in retrieved_chunks])
    # Tách từ khóa quan trọng (> 2 ký tự)
    words = [w for w in re.findall(r"\w+", expected_context.lower()) if len(w) > 2]
    if not words:
        return 0.0

    found = sum(1 for w in words if w in retrieved_text)
    return min(1.0, round(found / len(words), 3))


def compute_context_precision(expected_context: str, retrieved_chunks: list[dict]) -> float:
    """
    Tính Mean Average Precision (MAP) / rank-weighted precision của các chunks được lấy.
    Chunk càng ở vị trí cao chứa thông tin liên quan thì precision càng cao.
    """
    if not retrieved_chunks or not expected_context:
        return 0.0

    words = set([w for w in re.findall(r"\w+", expected_context.lower()) if len(w) > 2])
    if not words:
        return 0.0

    hits = 0
    precision_sum = 0.0
    for rank, chunk in enumerate(retrieved_chunks, 1):
        chunk_words = set(re.findall(r"\w+", chunk["content"].lower()))
        overlap = len(words & chunk_words) / max(len(words), 1)
        if overlap >= 0.25:  # Chunk có liên quan
            hits += 1
            precision_sum += hits / rank

    if hits == 0:
        return 0.0
    return min(1.0, round(precision_sum / hits, 3))


def compute_faithfulness_and_relevance(
    question: str,
    answer: str,
    context: str,
    expected_answer: str,
) -> tuple[float, float]:
    """
    Dùng LLM Judge hoặc heuristic kiểm tra faithfulness và answer relevance.
    """
    judge_prompt = f"""Bạn là một chuyên gia đánh giá hệ thống RAG độc lập. Hãy chấm điểm cho câu trả lời theo 2 tiêu chí sau trên thang điểm từ 0.0 đến 1.0:

1. Faithfulness (Trung thực với Context): Câu trả lời có hoàn toàn dựa trên bằng chứng được cung cấp trong Context không? (1.0 nếu mọi ý đều từ context, giảm điểm nếu tự bịa đặt hoặc suy diễn sai).
2. Answer Relevance (Sự liên quan và chính xác): Câu trả lời có giải quyết đúng trọng tâm câu hỏi và khớp với câu trả lời kỳ vọng không? (1.0 nếu trả lời chính xác, đầy đủ; 0.0 nếu lạc đề hoặc sai).

Context:
{context[:2000]}

Câu hỏi:
{question}

Câu trả lời thực tế:
{answer}

Câu trả lời kỳ vọng:
{expected_answer}

Trả về CHÍNH XÁC một JSON hợp lệ dạng:
{{"faithfulness": 0.95, "answer_relevance": 0.90}}
Không thêm bất kỳ text nào khác ngoài JSON.
"""
    try:
        res = call_llm(
            "Bạn là chuyên gia đánh giá khách quan. Chỉ trả lời định dạng JSON.",
            judge_prompt,
        )
        # Parse JSON
        clean_json = re.search(r"\{.*?\}", res, re.DOTALL)
        if clean_json:
            parsed = json.loads(clean_json.group(0))
            f_score = float(parsed.get("faithfulness", 0.85))
            r_score = float(parsed.get("answer_relevance", 0.85))
            return min(1.0, max(0.0, f_score)), min(1.0, max(0.0, r_score))
    except Exception as exc:
        print(f"LLM Judge fallback: {exc}")

    # Heuristic fallback nếu LLM judge gặp lỗi kết nối
    ans_words = set(re.findall(r"\w+", answer.lower()))
    ctx_words = set(re.findall(r"\w+", context.lower()))
    exp_words = set(re.findall(r"\w+", expected_answer.lower()))

    f_score = len(ans_words & ctx_words) / max(len(ans_words), 1) if ans_words else 0.5
    r_score = len(ans_words & exp_words) / max(len(exp_words), 1) if exp_words else 0.5

    return min(1.0, round(f_score, 3)), min(1.0, round(r_score, 3))


def run_evaluation() -> dict[str, Any]:
    """Chạy toàn bộ evaluation dataset trên Config A và Config B."""
    dataset = json.loads(GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))
    print(f"Bắt đầu đánh giá A/B trên {len(dataset)} golden cases...")

    results_a = []
    results_b = []
    latencies_a = []
    latencies_b = []

    for idx, case in enumerate(dataset):
        q = case["question"]
        exp_ans = case["expected_answer"]
        exp_ctx = case["expected_context"]
        print(f"\n--- Case {idx + 1}/{len(dataset)}: {q[:50]}... ---")

        # ------------------- Config A: Dense-only -------------------
        t0 = time.time()
        chunks_a = semantic_search(q, top_k=TOP_K)
        reordered_a = reorder_for_llm(chunks_a)
        ctx_a = format_context(reordered_a)
        msg_a = f"Context:\n{ctx_a}\n\nQuestion: {q}"
        ans_a = call_llm(SYSTEM_PROMPT, msg_a)
        lat_a = time.time() - t0
        latencies_a.append(lat_a)

        rec_a = compute_context_recall(exp_ctx, chunks_a)
        prec_a = compute_context_precision(exp_ctx, chunks_a)
        faith_a, rel_a = compute_faithfulness_and_relevance(q, ans_a, ctx_a, exp_ans)

        results_a.append({
            "case_id": idx + 1,
            "question": q,
            "answer": ans_a,
            "faithfulness": faith_a,
            "answer_relevance": rel_a,
            "context_recall": rec_a,
            "context_precision": prec_a,
            "latency": round(lat_a, 2),
            "sources": [c["id"] for c in chunks_a],
        })

        # ------------------- Config B: Hybrid + RRF -------------------
        t0 = time.time()
        chunks_b = retrieve(q, top_k=TOP_K, use_reranking=True)
        reordered_b = reorder_for_llm(chunks_b)
        ctx_b = format_context(reordered_b)
        msg_b = f"Context:\n{ctx_b}\n\nQuestion: {q}"
        ans_b = call_llm(SYSTEM_PROMPT, msg_b)
        lat_b = time.time() - t0
        latencies_b.append(lat_b)

        rec_b = compute_context_recall(exp_ctx, chunks_b)
        prec_b = compute_context_precision(exp_ctx, chunks_b)
        faith_b, rel_b = compute_faithfulness_and_relevance(q, ans_b, ctx_b, exp_ans)

        results_b.append({
            "case_id": idx + 1,
            "question": q,
            "answer": ans_b,
            "faithfulness": faith_b,
            "answer_relevance": rel_b,
            "context_recall": rec_b,
            "context_precision": prec_b,
            "latency": round(lat_b, 2),
            "sources": [c["id"] for c in chunks_b],
        })

        print(f"Config A (Dense): Faith={faith_a:.2f}, Rel={rel_a:.2f}, Rec={rec_a:.2f}, Prec={prec_a:.2f} ({lat_a:.2f}s)")
        print(f"Config B (Hybrid): Faith={faith_b:.2f}, Rel={rel_b:.2f}, Rec={rec_b:.2f}, Prec={prec_b:.2f} ({lat_b:.2f}s)")

    # Tổng hợp metrics trung bình
    n = len(dataset)
    avg_a = {
        "faithfulness": round(sum(r["faithfulness"] for r in results_a) / n, 4),
        "answer_relevance": round(sum(r["answer_relevance"] for r in results_a) / n, 4),
        "context_recall": round(sum(r["context_recall"] for r in results_a) / n, 4),
        "context_precision": round(sum(r["context_precision"] for r in results_a) / n, 4),
        "average": 0.0,
        "latency": round(sum(latencies_a) / n, 2),
    }
    avg_a["average"] = round(sum([avg_a["faithfulness"], avg_a["answer_relevance"], avg_a["context_recall"], avg_a["context_precision"]]) / 4, 4)

    avg_b = {
        "faithfulness": round(sum(r["faithfulness"] for r in results_b) / n, 4),
        "answer_relevance": round(sum(r["answer_relevance"] for r in results_b) / n, 4),
        "context_recall": round(sum(r["context_recall"] for r in results_b) / n, 4),
        "context_precision": round(sum(r["context_precision"] for r in results_b) / n, 4),
        "average": 0.0,
        "latency": round(sum(latencies_b) / n, 2),
    }
    avg_b["average"] = round(sum([avg_b["faithfulness"], avg_b["answer_relevance"], avg_b["context_recall"], avg_b["context_precision"]]) / 4, 4)

    deltas = {
        k: round(avg_b[k] - avg_a[k], 4)
        for k in ["faithfulness", "answer_relevance", "context_recall", "context_precision", "average", "latency"]
    }

    # Tìm 3 worst performers trong Config B hoặc Config A
    scored_cases = []
    for ra, rb in zip(results_a, results_b):
        case_avg_b = (rb["faithfulness"] + rb["answer_relevance"] + rb["context_recall"] + rb["context_precision"]) / 4
        scored_cases.append((case_avg_b, ra, rb))
    scored_cases.sort(key=lambda x: x[0])
    worst_three = [
        {
            "rank": i + 1,
            "case_id": item[2]["case_id"],
            "question": item[2]["question"],
            "config": "Config B",
            "faithfulness": item[2]["faithfulness"],
            "relevance": item[2]["answer_relevance"],
            "recall": item[2]["context_recall"],
            "precision": item[2]["context_precision"],
            "score": round(item[0], 3),
        }
        for i, item in enumerate(scored_cases[:3])
    ]

    output_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_cases": n,
        "config_a_dense": avg_a,
        "config_b_hybrid": avg_b,
        "deltas": deltas,
        "worst_performers": worst_three,
        "details_a": results_a,
        "details_b": results_b,
    }

    RESULTS_PATH.write_text(json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved evaluation results to {RESULTS_PATH}")
    return output_data


if __name__ == "__main__":
    run_evaluation()

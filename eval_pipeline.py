from answer_service import answer_service
from sentence_transformers import SentenceTransformer, util
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import numpy as np
from tqdm import tqdm

# --------------------------
#   1) Evaluate whole pipeline on dataset
# --------------------------

def run_pipeline(gt_pairs, sample_size=None):
    results = []
    pairs = gt_pairs if sample_size is None else gt_pairs[:sample_size]

    for item in tqdm(pairs, desc="Running pipeline evaluation"):
        query = item["instruction"]
        gt = item["response"]

        out = answer_service.answer(query, use_llm=True)

        results.append({
            "query": query,
            "gt": gt,
            "retrieved": out["retrieved_answer"],
            "llm": out["llm_answer"],
            "safety": out["safety"]["level"]
        })

    return results


# --------------------------
#   2) Similarity Scores
# --------------------------

embedder = SentenceTransformer("all-mpnet-base-v2")
rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def metrics(a, b):
    # cosine similarity
    emb_a = embedder.encode(a, convert_to_tensor=True)
    emb_b = embedder.encode(b, convert_to_tensor=True)
    cosine = float(util.pytorch_cos_sim(emb_a, emb_b))

    # bertscore
    P, R, F = bert_score([a], [b], lang="en")
    bert_f1 = float(F[0])

    # rouge-L
    rougeL = rouge.score(a, b)["rougeL"].fmeasure

    return {
        "cosine": cosine,
        "bert_f1": bert_f1,
        "rougeL": rougeL
    }


# --------------------------
#   3) Full evaluation
# --------------------------

def evaluate_pipeline(results):
    llm_vs_gt = []
    llm_vs_ret = []
    safety_stats = {}

    for r in results:
        s = r["safety"]
        safety_stats[s] = safety_stats.get(s, 0) + 1

        llm_vs_gt.append(metrics(r["llm"], r["gt"]))
        llm_vs_ret.append(metrics(r["llm"], r["retrieved"]))

    def avg(metric_list, key):
        return float(sum(m[key] for m in metric_list) / len(metric_list))

    report = {
        "LLM_vs_GroundTruth": {
            "cosine": avg(llm_vs_gt, "cosine"),
            "bert_f1": avg(llm_vs_gt, "bert_f1"),
            "rougeL": avg(llm_vs_gt, "rougeL"),
        },
        "LLM_vs_Retrieved": {
            "cosine": avg(llm_vs_ret, "cosine"),
            "bert_f1": avg(llm_vs_ret, "bert_f1"),
            "rougeL": avg(llm_vs_ret, "rougeL"),
        },
        "safety": safety_stats,
        "samples": len(results)
    }

    return report

"""Evaluation module for PCIB detector on hallucination benchmarks."""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from .core import PCIBDetector
from .types import Config


def parse_label(x: Any) -> Optional[int]:
    """
    Parse label as binary hallucination indicator.
    Returns 1 for hallucination, 0 for no hallucination.
    """
    if x is None:
        return None
    if isinstance(x, (int, np.integer)):
        if int(x) in (0, 1):
            return int(x)
    if isinstance(x, bool):
        return 1 if x else 0
    s = str(x).strip().lower()
    if s in ("1", "true", "hallucination", "hallucinated", "fail", "yes"):
        return 1
    if s in ("0", "false", "no_hallucination", "pass", "no"):
        return 0
    return None


def get_field(d: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    """Get first non-empty field from dict."""
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUROC using rank-based method."""
    order = np.argsort(y_score)
    y = y_true[order]
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = np.arange(1, len(y) + 1)
    rank_sum_pos = np.sum(ranks[y == 1])
    u = rank_sum_pos - (n_pos * (n_pos + 1)) / 2
    return float(u / (n_pos * n_neg))


def auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUPRC (precision-recall curve area)."""
    order = np.argsort(-y_score)
    y = y_true[order]
    tp = 0
    fp = 0
    precisions = []
    recalls = []
    n_pos = np.sum(y_true == 1)
    if n_pos == 0:
        return float("nan")
    for i in range(len(y)):
        if y[i] == 1:
            tp += 1
        else:
            fp += 1
        precision = tp / max(1, tp + fp)
        recall = tp / n_pos
        precisions.append(precision)
        recalls.append(recall)
    area = 0.0
    prev_r = 0.0
    prev_p = 1.0
    for p, r in zip(precisions, recalls):
        area += (r - prev_r) * ((p + prev_p) / 2.0)
        prev_r = r
        prev_p = p
    return float(area)


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float, float]:
    """Find threshold that maximizes F1 score."""
    uniq = np.unique(y_score)
    best = (0.0, 0.0, 0.0)  # f1, thr, acc
    for thr in uniq:
        y_pred = (y_score >= thr).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)
        acc = float(np.mean(y_pred == y_true))
        if f1 > best[0]:
            best = (float(f1), float(thr), float(acc))
    return best


async def evaluate_dataset(
    detector: PCIBDetector,
    dataset_name: str = "PatronusAI/HaluBench",
    split: str = "",
    limit: int = 300,
    output_file: str = "eval_results.jsonl",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate detector on a hallucination benchmark dataset.
    
    Args:
        detector: PCIBDetector instance
        dataset_name: HuggingFace dataset identifier
        split: Dataset split (auto-detect if empty)
        limit: Maximum examples to evaluate
        output_file: Path to save detailed results
        verbose: Show progress bar
        
    Returns:
        Dict with metrics: auroc, auprc, f1, threshold, accuracy
    """
    # Load dataset
    ds_obj = load_dataset(dataset_name)
    
    if hasattr(ds_obj, "keys"):
        splits = list(ds_obj.keys())
        if not split:
            split = "test" if "test" in splits else ("validation" if "validation" in splits else splits[0])
        ds = ds_obj[split]
    else:
        ds = ds_obj
    
    y_true: List[int] = []
    y_score: List[float] = []
    results: List[Dict[str, Any]] = []
    
    iterator = tqdm(ds, total=min(len(ds), limit), disable=not verbose)
    
    for i, ex in enumerate(iterator):
        if i >= limit:
            break
        
        # Extract fields
        evidence = get_field(ex, ["context", "passage", "source", "grounding", "retrieved_context"]) or ""
        question = get_field(ex, ["question", "query", "prompt"]) or ""
        answer = get_field(ex, ["answer", "response", "completion", "generated_answer"]) or ""
        label_raw = get_field(ex, ["label", "is_hallucination", "hallucination", "contains_hallucination"])
        
        y = parse_label(label_raw)
        if y is None:
            continue
        
        # Build evidence text
        evidence_text = f"QUESTION:\n{question}\n\nCONTEXT:\n{evidence}".strip()
        
        # Run detection
        result = await detector.detect_hallucination(
            answer=answer,
            evidence=evidence_text,
            return_details=True
        )
        
        y_true.append(y)
        y_score.append(result.score)
        
        # Save detailed results
        rec = {
            "index": i,
            "label": y,
            "score": result.score,
            "flagged": result.flagged,
            "question": question,
            "answer": answer,
            "evidence_len": len(evidence_text),
            "claims": [
                {
                    "text": c.text,
                    "score": c.score,
                    "flagged": c.flagged,
                    "signals": {
                        "prior": {
                            "entail": c.signals.prior.entail,
                            "contradict": c.signals.prior.contradict,
                            "unknown": c.signals.prior.unknown,
                        },
                        "post": {
                            "entail": c.signals.post.entail,
                            "contradict": c.signals.post.contradict,
                            "unknown": c.signals.post.unknown,
                        },
                        "uptake_kl": c.signals.uptake_kl,
                        "stress_js": c.signals.stress_js,
                        "conflict_js": c.signals.conflict_js,
                    }
                }
                for c in result.claims
            ],
        }
        results.append(rec)
        
        if verbose:
            iterator.set_postfix({"score": f"{result.score:.3f}", "label": y})
    
    # Compute metrics
    y_true_arr = np.array(y_true, dtype=int)
    y_score_arr = np.array(y_score, dtype=float)
    
    roc = auroc(y_true_arr, y_score_arr)
    prc = auprc(y_true_arr, y_score_arr)
    f1, thr, acc = best_f1_threshold(y_true_arr, y_score_arr)
    
    metrics = {
        "auroc": roc,
        "auprc": prc,
        "best_f1": f1,
        "best_threshold": thr,
        "best_accuracy": acc,
        "n_examples": len(y_true_arr),
        "n_positive": int(np.sum(y_true_arr == 1)),
        "n_negative": int(np.sum(y_true_arr == 0)),
    }
    
    # Write results to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec) + "\n")
    
    if verbose:
        print(f"\n{'='*60}")
        print("PCIB Detector Evaluation Results")
        print(f"{'='*60}")
        print(f"Dataset: {dataset_name} ({split})")
        print(f"Examples: {metrics['n_examples']} ({metrics['n_positive']} positive, {metrics['n_negative']} negative)")
        print(f"\nMetrics:")
        print(f"  AUROC     : {roc:.4f}")
        print(f"  AUPRC     : {prc:.4f}")
        print(f"  Best F1   : {f1:.4f}")
        print(f"  Threshold : {thr:.6f}")
        print(f"  Accuracy  : {acc:.4f}")
        print(f"\nResults saved to: {output_path}")
        print(f"{'='*60}\n")
    
    return metrics


def main():
    """CLI entry point for evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate PCIB detector on hallucination benchmarks")
    parser.add_argument("--dataset", default="PatronusAI/HaluBench", help="HuggingFace dataset identifier")
    parser.add_argument("--split", default="", help="Dataset split (auto-detect if empty)")
    parser.add_argument("--limit", type=int, default=300, help="Maximum examples to evaluate")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for verification")
    parser.add_argument("--n-ensemble", type=int, default=1, help="Ensemble size")
    parser.add_argument("--ensemble-temperature", type=float, default=0.7, help="Ensemble temperature")
    parser.add_argument("--output", "-o", default="eval_results.jsonl", help="Output file")
    args = parser.parse_args()
    
    config = Config(
        model=args.model,
        temperature=args.temperature,
        n_ensemble=args.n_ensemble,
        ensemble_temperature=args.ensemble_temperature,
    )
    
    detector = PCIBDetector(config)
    
    asyncio.run(evaluate_dataset(
        detector=detector,
        dataset_name=args.dataset,
        split=args.split,
        limit=args.limit,
        output_file=args.output,
        verbose=True,
    ))


if __name__ == "__main__":
    main()

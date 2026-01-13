#!/usr/bin/env python3
"""
Enhanced PCIB Ablation Study - Tests all improvements

Compares baseline PCIB against enhanced configurations:
- Learned weights vs manual weights
- Additional signals (semantic, entity, specificity)
- Multi-verifier ensemble
- Claim graph aggregation
- Enhanced perturbations
"""

import sys
import os

# Add pcib_detector source directory to Python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PCIB_SRC = os.path.join(PROJECT_ROOT, 'pcib_detector', 'src')
sys.path.insert(0, PCIB_SRC)

import asyncio
import json
import argparse
import os
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Any
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
from scipy import stats

# Import enhanced modules
from pcib_detector.core_enhanced import detect_hallucination_v2, ImprovedClaimExtractor
from pcib_detector.weight_learning import SignalWeightLearner
from pcib_detector.additional_signals import AdditionalSignals
from pcib_detector.ensemble import MultiVerifierEnsemble
from pcib_detector.claim_graph import ClaimDependencyGraph
from pcib_detector.calibration import PCIBCalibrator
from pcib_detector.backends.openai_backend import OpenAIBackend
from pcib_detector.backends.anthropic_backend import AnthropicBackend
from pcib_detector.backends.gemini_backend import GeminiBackend


def check_api_keys_available(config: 'EnhancedAblationConfig') -> Tuple[bool, str]:
    """
    Check if required API keys are available for a configuration.
    
    Returns:
        (is_available, reason) tuple
    """
    import os
    
    # Non-multi-verifier configs only need OpenAI
    if not config.use_multi_verifier:
        if not os.getenv('OPENAI_API_KEY'):
            return False, "Missing OPENAI_API_KEY"
        return True, ""
    
    # Multi-verifier configs need all specified providers
    if config.verifier_configs:
        missing = []
        for vc in config.verifier_configs:
            backend_type = vc['backend']
            if backend_type == 'openai':
                if not os.getenv('OPENAI_API_KEY'):
                    missing.append('OPENAI_API_KEY')
            elif backend_type == 'anthropic':
                if not os.getenv('ANTHROPIC_API_KEY'):
                    missing.append('ANTHROPIC_API_KEY')
            elif backend_type == 'gemini':
                if not (os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')):
                    missing.append('GOOGLE_API_KEY or GEMINI_API_KEY')
        
        if missing:
            return False, f"Missing API keys: {', '.join(missing)}"
    
    return True, ""


# -------------------------
# Metrics with confidence intervals (from ablation_study.py)
# -------------------------

def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Rank-based AUROC."""
    if len(y_true) == 0:
        return float("nan")
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
    """Precision-recall curve area."""
    if len(y_true) == 0:
        return float("nan")
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


def bootstrap_ci(y_true: np.ndarray, y_score: np.ndarray, metric_fn, n_bootstrap=1000, ci=0.95):
    """Compute bootstrap confidence interval for a metric."""
    if len(y_true) == 0:
        return (0.0, 0.0)
    
    scores = []
    n = len(y_true)
    rng = np.random.RandomState(42)
    
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        try:
            score = metric_fn(y_true[idx], y_score[idx])
            if not np.isnan(score):
                scores.append(score)
        except:
            pass
    
    if not scores:
        return (0.0, 0.0)
    
    alpha = 1 - ci
    lower = np.percentile(scores, alpha/2 * 100)
    upper = np.percentile(scores, (1 - alpha/2) * 100)
    return (float(lower), float(upper))


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float, float]:
    """Find threshold that maximizes F1."""
    if len(y_true) == 0:
        return 0.0, 0.0, 0.0
    uniq = np.unique(y_score)
    best = (0.0, 0.0, 0.0)
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


@dataclass
class MetricResult:
    """Evaluation metrics with confidence intervals."""
    n_examples: int
    auroc: float
    auroc_ci: Tuple[float, float]
    auprc: float
    auprc_ci: Tuple[float, float]
    best_f1: float
    f1_ci: Tuple[float, float]
    best_threshold: float
    best_accuracy: float
    precision: float
    recall: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


def compute_metrics(scores: List[float], labels: List[int]) -> MetricResult:
    """Compute all metrics with confidence intervals."""
    y_true = np.array(labels, dtype=int)
    y_score = np.array(scores, dtype=float)
    
    roc = auroc(y_true, y_score)
    roc_ci = bootstrap_ci(y_true, y_score, auroc, n_bootstrap=1000)
    
    prc = auprc(y_true, y_score)
    prc_ci = bootstrap_ci(y_true, y_score, auprc, n_bootstrap=1000)
    
    f1, thr, acc = best_f1_threshold(y_true, y_score)
    
    # Compute precision/recall at best threshold
    y_pred = (y_score >= thr).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    
    # F1 CI via bootstrap
    def f1_fn(yt, ys):
        f, _, _ = best_f1_threshold(yt, ys)
        return f
    f1_ci = bootstrap_ci(y_true, y_score, f1_fn, n_bootstrap=1000)
    
    return MetricResult(
        n_examples=len(y_true),
        auroc=roc,
        auroc_ci=roc_ci,
        auprc=prc,
        auprc_ci=prc_ci,
        best_f1=f1,
        f1_ci=f1_ci,
        best_threshold=thr,
        best_accuracy=acc,
        precision=float(precision),
        recall=float(recall),
    )


# -------------------------
# Dataset Helper Functions (from ablation_study.py)
# -------------------------

def get_field(d: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    """Get first available field from dict."""
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def parse_label(x: Any) -> Optional[int]:
    """Parse label as binary hallucination indicator."""
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


# -------------------------
# Enhanced Configuration
# -------------------------

@dataclass
class EnhancedAblationConfig:
    """Configuration for enhanced ablation experiments."""
    name: str
    description: str
    category: str  # "Baseline", "Learned Weights", "Additional Signals", "Ensemble", "Combined"
    
    # Enhancement toggles
    use_learned_weights: bool = False
    use_additional_signals: bool = False
    use_multi_verifier: bool = False
    use_claim_graph: bool = False
    use_enhanced_perturbations: bool = False
    use_calibration: bool = False
    
    # Verifier configuration
    verifier_model: str = "gpt-4o-mini"
    verifier_configs: Optional[List[Dict]] = None  # For multi-verifier
    
    # Cost multiplier (API calls)
    api_calls_multiplier: float = 3.0
    
    def to_dict(self):
        return asdict(self)


def create_enhanced_configs() -> List[EnhancedAblationConfig]:
    """Create all enhanced ablation configurations."""
    configs = []
    
    # 1. BASELINE (from original ablation study)
    configs.append(EnhancedAblationConfig(
        name="baseline",
        description="Original PCIB (U+S+C signals, manual weights)",
        category="Baseline",
        use_learned_weights=False,
        use_additional_signals=False,
        use_multi_verifier=False,
        api_calls_multiplier=3.0
    ))
    
    # 2. LEARNED WEIGHTS ONLY
    configs.append(EnhancedAblationConfig(
        name="learned_weights",
        description="PCIB with learned signal weights (logistic regression)",
        category="Learned Weights",
        use_learned_weights=True,
        use_additional_signals=False,
        use_multi_verifier=False,
        api_calls_multiplier=3.0
    ))
    
    # 3. ADDITIONAL SIGNALS ONLY
    configs.append(EnhancedAblationConfig(
        name="additional_signals",
        description="PCIB + semantic similarity + entity consistency + specificity",
        category="Additional Signals",
        use_learned_weights=False,
        use_additional_signals=True,
        use_multi_verifier=False,
        api_calls_multiplier=3.5  # Slight cost increase
    ))
    
    # 4. CLAIM GRAPH AGGREGATION
    configs.append(EnhancedAblationConfig(
        name="claim_graph",
        description="PCIB with PageRank-weighted claim aggregation",
        category="Claim Graph",
        use_learned_weights=False,
        use_additional_signals=False,
        use_claim_graph=True,
        api_calls_multiplier=4.0
    ))
    
    # 5. MULTI-VERIFIER ENSEMBLE (3 models)
    configs.append(EnhancedAblationConfig(
        name="multi_verifier_3",
        description="PCIB with 3-model ensemble (OpenAI + Anthropic + Gemini)",
        category="Ensemble",
        use_multi_verifier=True,
        verifier_configs=[
            {'backend': 'openai', 'model': 'gpt-4o-mini', 'weight': 0.4},
            {'backend': 'anthropic', 'model': 'claude-3-haiku-20240307', 'weight': 0.3},
            {'backend': 'gemini', 'model': 'gemini-1.5-flash', 'weight': 0.3}
        ],
        api_calls_multiplier=9.0  # 3x models × 3x API calls
    ))
    
    # 6. LEARNED WEIGHTS + ADDITIONAL SIGNALS
    configs.append(EnhancedAblationConfig(
        name="learned_plus_signals",
        description="Learned weights + additional signals",
        category="Combined",
        use_learned_weights=True,
        use_additional_signals=True,
        api_calls_multiplier=3.5
    ))
    
    # 7. ALL ENHANCEMENTS (except multi-verifier for cost)
    configs.append(EnhancedAblationConfig(
        name="all_enhancements",
        description="Learned weights + signals + claim graph + calibration",
        category="Combined",
        use_learned_weights=True,
        use_additional_signals=True,
        use_claim_graph=True,
        use_calibration=True,
        api_calls_multiplier=4.5
    ))
    
    # 8. BEST TRADEOFF (our recommendation)
    configs.append(EnhancedAblationConfig(
        name="recommended",
        description="Best cost-performance: Learned weights + additional signals",
        category="Combined",
        use_learned_weights=True,
        use_additional_signals=True,
        use_enhanced_perturbations=True,
        api_calls_multiplier=4.0
    ))
    
    return configs


async def run_enhanced_detection(
    example: Dict,
    config: EnhancedAblationConfig,
    backend: OpenAIBackend,
    learned_weights: Optional[Dict] = None,
    calibrator: Optional[PCIBCalibrator] = None
) -> Dict:
    """
    Run detection with enhanced configuration.
    
    Args:
        example: Dict with 'question', 'answer', 'label'
        config: EnhancedAblationConfig specifying which enhancements to use
        backend: LLM backend for verification
        learned_weights: Optional learned signal weights
        calibrator: Optional calibrator for score adjustment
    
    Returns:
        Dict with:
            - predicted_score: float
            - signals: Dict[str, float]
            - n_claims: int
            - config_name: str
    """
    question = example['question']
    answer = example['answer']
    
    # Build detection config
    detection_config = {
        'use_learned_weights': config.use_learned_weights,
        'learned_weights': learned_weights,
        'use_additional_signals': config.use_additional_signals,
        'use_claim_graph': config.use_claim_graph,
        'use_enhanced_perturbations': config.use_enhanced_perturbations,
    }
    
    # Handle multi-verifier ensemble separately
    if config.use_multi_verifier and config.verifier_configs:
        ensemble = MultiVerifierEnsemble(config.verifier_configs)
        result = await ensemble.compute_ensemble_score(
            question=question,
            answer=answer,
            aggregation='weighted_average'
        )
        predicted_score = result['ensemble_score']
        signals = result['individual_results'][0]['signals']  # Use first verifier's signals
        n_claims = result.get('n_claims', 0)
    else:
        # Standard enhanced detection
        result = await detect_hallucination_v2(
            question=question,
            answer=answer,
            backend=backend,
            config=detection_config
        )
        predicted_score = result['predicted_score']
        signals = result['signals']
        n_claims = result.get('n_claims', 0)
    
    # Apply calibration if enabled
    if config.use_calibration and calibrator:
        predicted_score = calibrator.calibrate([predicted_score])[0]
    
    return {
        'predicted_score': predicted_score,
        'signals': signals,
        'n_claims': n_claims,
        'config_name': config.name
    }


async def evaluate_enhanced_config(
    config: EnhancedAblationConfig,
    examples: List[Dict],
    backend: OpenAIBackend,
    learned_weights: Optional[Dict] = None,
    calibrator: Optional[PCIBCalibrator] = None,
    max_retries: int = 3
) -> Dict:
    """
    Evaluate a single enhanced configuration.
    
    Args:
        config: Configuration to evaluate
        examples: List of examples with 'question', 'answer', 'label'
        backend: LLM backend
        learned_weights: Optional learned weights
        calibrator: Optional calibrator
        max_retries: Maximum retries per example
    
    Returns:
        Dict with metrics, config, and detailed results
    """
    results = []
    
    desc = f"{config.name:25}"
    for example in tqdm(examples, desc=desc):
        # Retry logic for API failures
        for attempt in range(max_retries):
            try:
                result = await run_enhanced_detection(
                    example=example,
                    config=config,
                    backend=backend,
                    learned_weights=learned_weights,
                    calibrator=calibrator
                )
                
                results.append({
                    'question': example['question'],
                    'answer': example['answer'],
                    'label': example['label'],
                    'predicted_score': result['predicted_score'],
                    'signals': result['signals'],
                    'n_claims': result['n_claims']
                })
                break  # Success, exit retry loop
                
            except Exception as e:
                if attempt < max_retries - 1:
                    # Wait before retry with exponential backoff
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    # Final attempt failed, log error
                    print(f"\n⚠️  Error on example after {max_retries} attempts: {e}")
                    # Add failed result with score=0.5 (neutral)
                    results.append({
                        'question': example['question'],
                        'answer': example['answer'],
                        'label': example['label'],
                        'predicted_score': 0.5,
                        'signals': {},
                        'n_claims': 0,
                        'error': str(e)
                    })
    
    # Compute metrics
    scores = [r['predicted_score'] for r in results]
    labels = [r['label'] for r in results]
    
    metrics = compute_metrics(scores, labels)
    
    return {
        'config': config.to_dict(),
        'metrics': metrics,
        'examples': results,
        'n_examples': len(results),
        'n_errors': sum(1 for r in results if 'error' in r)
    }


async def main():
    parser = argparse.ArgumentParser(description="Enhanced PCIB Ablation Study")
    parser.add_argument('--limit', type=int, default=100, help='Number of examples per config')
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help='Verifier model')
    parser.add_argument('--output', type=str, default='ablation_results_enhanced', help='Output directory')
    parser.add_argument('--dataset', type=str, default='PatronusAI/HaluBench', help='Dataset name')
    parser.add_argument('--configs', nargs='+', help='Specific configs to run (default: all)')
    parser.add_argument('--train-weights', action='store_true', help='Train learned weights first')
    parser.add_argument('--weights-file', type=str, help='Path to pre-trained weights JSON file')
    args = parser.parse_args()
    
    print("="*80)
    print("ENHANCED PCIB ABLATION STUDY")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Limit: {args.limit} examples per configuration")
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print("="*80)
    
    # Check available API keys
    import os
    available_providers = []
    if os.getenv('OPENAI_API_KEY'):
        available_providers.append('OpenAI')
    if os.getenv('ANTHROPIC_API_KEY'):
        available_providers.append('Anthropic')
    if os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY'):
        available_providers.append('Gemini')
    
    print(f"\n🔑 Available providers: {', '.join(available_providers) if available_providers else 'None'}")
    if not available_providers:
        print("❌ No API keys found! Set OPENAI_API_KEY to run.")
        return
    
    # Load dataset
    print("\n📚 Loading dataset...")
    try:
        dataset = load_dataset(args.dataset, split='test')
        examples = []
        skipped = 0
        for i, item in enumerate(dataset):
            if len(examples) >= args.limit:
                break
            
            # Safely parse label using parse_label function
            label_raw = get_field(item, ['label', 'hallucination', 'is_hallucination'])
            label = parse_label(label_raw)
            
            if label is None:
                skipped += 1
                continue
            
            examples.append({
                'question': get_field(item, ['question', 'query', 'input']) or '',
                'answer': get_field(item, ['answer', 'response', 'output', 'completion']) or '',
                'label': label
            })
        
        print(f"✅ Loaded {len(examples)} examples (skipped {skipped} with invalid labels)")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("Using mock examples for testing...")
        examples = [
            {
                'question': 'What is the capital of France?',
                'answer': 'The capital of France is Paris.',
                'label': 0
            },
            {
                'question': 'Who won the 2025 World Cup?',
                'answer': 'Brazil won the 2025 World Cup by defeating Germany 3-1.',
                'label': 1
            }
        ] * (args.limit // 2)
    
    # Initialize backend (OpenAIBackend doesn't take model parameter)
    backend = OpenAIBackend()
    
    # Train or load learned weights
    learned_weights = None
    if args.weights_file:
        print(f"\n📂 Loading pre-trained weights from {args.weights_file}...")
        with open(args.weights_file, 'r') as f:
            learned_weights = json.load(f)
        print(f"✅ Loaded weights: {learned_weights}")
    elif args.train_weights:
        print("\n🎓 Training learned weights...")
        try:
            learner = SignalWeightLearner()
            learned_weights = learner.train_from_json('ablation_results/foo/raw_data_*.json')
            print(f"✅ Learned weights: {learned_weights}")
            
            # Save learned weights
            os.makedirs(args.output, exist_ok=True)
            weights_output = os.path.join(args.output, 'learned_weights.json')
            with open(weights_output, 'w') as f:
                json.dump(learned_weights, f, indent=2)
            print(f"💾 Saved weights to: {weights_output}")
        except Exception as e:
            print(f"⚠️  Could not train weights: {e}")
            print("Proceeding with manual weights for learned weight configs...")
    
    # Initialize calibrator (would need training data)
    calibrator = None
    # TODO: Train calibrator on held-out set if needed
    # calibrator = PCIBCalibrator()
    # calibrator.fit(scores_val, labels_val)
    
    # Create configurations
    all_configs = create_enhanced_configs()
    
    # Filter configs if specified
    if args.configs:
        all_configs = [c for c in all_configs if c.name in args.configs]
        print(f"\n🎯 Selected {len(all_configs)} configurations")
    else:
        print(f"\n🎯 Testing all {len(all_configs)} configurations")
    
    # Check API key availability and filter
    available_configs = []
    skipped_configs = []
    
    for config in all_configs:
        is_available, reason = check_api_keys_available(config)
        if is_available:
            available_configs.append(config)
        else:
            skipped_configs.append((config.name, reason))
            print(f"⚠️  Skipping {config.name}: {reason}")
    
    if not available_configs:
        print("❌ No configurations can run - missing required API keys")
        print("💡 Set OPENAI_API_KEY environment variable to run baseline configs")
        return
    
    print(f"✅ Running {len(available_configs)} configurations")
    if skipped_configs:
        print(f"⏭️  Skipped {len(skipped_configs)} configurations due to missing API keys")
    
    # Use available_configs instead of all_configs from here on
    all_configs = available_configs
    
    # Print configuration details
    print("\n📋 Configurations to evaluate:")
    for config in all_configs:
        print(f"  • {config.name:20} - {config.description}")
    
    # Run evaluations IN PARALLEL
    all_results = {}
    
    print("\n" + "="*80)
    print(f"🚀 RUNNING ALL {len(all_configs)} CONFIGURATIONS IN PARALLEL")
    print("="*80)
    print("⚠️  Progress bars may overlap - this is normal when running in parallel")
    print("="*80)
    
    # Print configuration details
    for idx, config in enumerate(all_configs, 1):
        print(f"  [{idx}] {config.name:20} - {config.description} (Cost: {config.api_calls_multiplier:.1f}×)")
    print("="*80)
    
    # Create tasks for all configurations
    tasks = []
    for config in all_configs:
        task = evaluate_enhanced_config(
            config=config,
            examples=examples,
            backend=backend,
            learned_weights=learned_weights if config.use_learned_weights else None,
            calibrator=calibrator if config.use_calibration else None
        )
        tasks.append((config.name, task))
    
    # Run all in parallel and track time
    print(f"\n⏱️  Starting parallel execution of {len(tasks)} configurations...\n")
    start_time = time.time()
    
    results_list = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    # Collect results
    print(f"\n" + "="*80)
    print(f"✅ PARALLEL EXECUTION COMPLETE")
    print("="*80)
    print(f"⏱️  Total parallel runtime: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print("="*80)
    
    for (config_name, _), result in zip(tasks, results_list):
        if isinstance(result, Exception):
            print(f"❌ Error in {config_name}: {result}")
            all_results[config_name] = {
                'error': str(result),
                'config': {'name': config_name},
                'metrics': {'auroc': 0.0, 'auprc': 0.0, 'best_f1': 0.0},
                'n_examples': 0,
                'n_errors': 0
            }
        else:
            all_results[config_name] = result
            metrics = result['metrics']
            print(f"✅ {config_name:20} AUROC={metrics['auroc']:.4f}, AUPRC={metrics['auprc']:.4f}, F1={metrics['best_f1']:.4f}")
            if result['n_errors'] > 0:
                print(f"   ⚠️  {result['n_errors']}/{result['n_examples']} errors")
    
    # Save results
    os.makedirs(args.output, exist_ok=True)
    
    output_file = os.path.join(args.output, 'enhanced_results.json')
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*80)
    print("✅ ENHANCED ABLATION STUDY COMPLETE")
    print("="*80)
    print(f"📁 Results saved to: {output_file}")
    
    # Print summary table
    print("\n📊 SUMMARY TABLE:")
    print("-"*90)
    print(f"{'Configuration':<25} {'Category':<20} {'AUROC':>8} {'AUPRC':>8} {'F1':>8} {'Cost':>6}")
    print("-"*90)
    
    # Group by category
    categories = {}
    for config in all_configs:
        if config.name in all_results:
            if config.category not in categories:
                categories[config.category] = []
            categories[config.category].append((config, all_results[config.name]))
    
    for category_name in ["Baseline", "Learned Weights", "Additional Signals", "Claim Graph", "Ensemble", "Combined"]:
        if category_name in categories:
            print(f"\n{category_name}:")
            for config, results in categories[category_name]:
                metrics = results['metrics']
                print(f"  {config.name:<23} {'':20} {metrics['auroc']:>8.4f} {metrics['auprc']:>8.4f} "
                      f"{metrics['best_f1']:>8.4f} {config.api_calls_multiplier:>5.1f}×")
    
    print("-"*90)
    
    # Find best configuration
    if all_results:
        best_config_name = max(all_results.keys(), key=lambda k: all_results[k]['metrics']['auroc'])
        best_metrics = all_results[best_config_name]['metrics']
        print(f"\n🏆 Best Configuration: {best_config_name}")
        print(f"   AUROC: {best_metrics['auroc']:.4f}")
        print(f"   AUPRC: {best_metrics['auprc']:.4f}")
        print(f"   F1: {best_metrics['best_f1']:.4f}")
    
    # Save summary
    summary = {
        'dataset': args.dataset,
        'n_examples': len(examples),
        'model': args.model,
        'configurations': {
            name: {
                'category': results['config']['category'],
                'description': results['config']['description'],
                'auroc': results['metrics']['auroc'],
                'auprc': results['metrics']['auprc'],
                'f1': results['metrics']['best_f1'],
                'cost_multiplier': results['config']['api_calls_multiplier']
            }
            for name, results in all_results.items()
        }
    }
    
    summary_file = os.path.join(args.output, 'summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"📁 Summary saved to: {summary_file}")


if __name__ == '__main__':
    asyncio.run(main())

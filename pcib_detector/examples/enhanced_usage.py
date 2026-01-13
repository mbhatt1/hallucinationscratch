"""
Example of using enhanced PCIB with all improvements.

This demonstrates:
1. Improved claim extraction
2. Learned signal weights
3. Enhanced calibration
4. Additional grounding signals
5. Multi-verifier ensemble (optional)

Target: AUROC 0.85-0.90 (up from baseline 0.67)
"""
import asyncio
import os
from pathlib import Path

# Import enhanced modules
from pcib_detector.backends.openai_backend import OpenAIBackend
from pcib_detector.core_enhanced import detect_hallucination_v2, ImprovedClaimExtractor
from pcib_detector.weight_learning import SignalWeightLearner
from pcib_detector.calibration import PCIBCalibrator
from pcib_detector.additional_signals import AdditionalSignals
from pcib_detector.ensemble import MultiVerifierEnsemble


async def example_basic_enhanced():
    """Basic example with improved claim extraction and additional signals."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Enhanced Detection")
    print("=" * 60)
    
    # Set up backend
    backend = OpenAIBackend(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Test question and answer
    question = "Who won Super Bowl 50?"
    answer = "The Denver Broncos defeated the Carolina Panthers 24-10."
    
    # Configure enhanced detection
    config = {
        'use_additional_signals': True,
        'use_learned_weights': False,  # Set True after training
        'calibrate_scores': False,      # Set True after calibration
    }
    
    # Run enhanced detection
    result = await detect_hallucination_v2(
        question=question,
        answer=answer,
        backend=backend,
        config=config
    )
    
    print(f"\nQuestion: {question}")
    print(f"Answer: {answer}")
    print(f"\nHallucination Score: {result['predicted_score']:.3f}")
    print(f"Flagged: {result['flagged']}")
    print(f"Number of Claims: {result['n_claims']}")
    print(f"\nClaims:")
    for i, claim in enumerate(result['claims'], 1):
        print(f"  {i}. {claim}")
    
    if 'semantic_similarity' in result['signals']:
        print(f"\nAdditional Signals:")
        print(f"  - Semantic Similarity: {result['signals']['semantic_similarity']:.3f}")
        print(f"  - Answer Specificity: {result['signals']['answer_specificity']:.3f}")
        if 'entity_consistency' in result['signals']:
            print(f"  - Entity Consistency: {result['signals']['entity_consistency']:.3f}")


async def example_with_learned_weights():
    """Example with learned signal weights from ablation data."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: With Learned Weights")
    print("=" * 60)
    
    # Train weights from ablation data
    ablation_file = Path("ablation_results/raw_data_gpt-4.1-nano_3a3369bb_20260111_085444.json")
    
    if ablation_file.exists():
        print("\nTraining weights from ablation data...")
        learner = SignalWeightLearner()
        try:
            learned_weights = learner.train_from_json(str(ablation_file))
            
            # Set up backend
            backend = OpenAIBackend(api_key=os.getenv("OPENAI_API_KEY"))
            
            # Test data
            question = "What is the capital of France?"
            answer = "The capital of France is London."  # Hallucination
            
            # Configure with learned weights
            config = {
                'use_learned_weights': True,
                'learned_weights': learned_weights,
                'use_additional_signals': True,
            }
            
            # Run detection
            result = await detect_hallucination_v2(
                question=question,
                answer=answer,
                backend=backend,
                config=config
            )
            
            print(f"\nQuestion: {question}")
            print(f"Answer: {answer}")
            print(f"Hallucination Score: {result['predicted_score']:.3f}")
            print(f"Flagged: {result['flagged']}")
            
            print(f"\nLearned Weights:")
            for key, value in learned_weights.items():
                print(f"  - {key}: {value:.3f}")
        
        except Exception as e:
            print(f"Could not train weights: {e}")
            print("Skipping learned weights example")
    else:
        print(f"\nAblation file not found: {ablation_file}")
        print("Skipping learned weights example")


async def example_with_calibration():
    """Example with score calibration."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: With Calibration")
    print("=" * 60)
    
    # Note: This requires training data
    # For demonstration, we'll show the setup
    
    print("\nCalibration Setup:")
    print("1. Collect PCIB scores and labels from validation set")
    print("2. Fit calibrator:")
    print("   calibrator = PCIBCalibrator(method='platt')")
    print("   calibrator.fit(scores, labels)")
    print("3. Use in config:")
    print("   config = {'calibrate_scores': True, 'calibrator': calibrator}")
    
    # Simulate calibrated detection
    backend = OpenAIBackend(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Create calibrator (would need real training data)
    calibrator = PCIBCalibrator(method='platt')
    # calibrator.fit(train_scores, train_labels)  # Would need real data
    
    question = "When was Python created?"
    answer = "Python was created by Guido van Rossum in 1991."
    
    config = {
        'use_additional_signals': True,
        # 'calibrate_scores': True,     # Uncomment after training
        # 'calibrator': calibrator,
    }
    
    result = await detect_hallucination_v2(
        question=question,
        answer=answer,
        backend=backend,
        config=config
    )
    
    print(f"\nQuestion: {question}")
    print(f"Answer: {answer}")
    print(f"Score: {result['predicted_score']:.3f}")
    print(f"Flagged: {result['flagged']}")


async def example_claim_extraction():
    """Demonstrate improved claim extraction on various input formats."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Improved Claim Extraction")
    print("=" * 60)
    
    backend = OpenAIBackend(api_key=os.getenv("OPENAI_API_KEY"))
    extractor = ImprovedClaimExtractor(backend)
    
    # Test various formats
    test_cases = [
        ("List format", "['Rams', 'second', 'Marc Bulger']"),
        ("Short answer", "Paris"),
        ("Bullet list", "• Item 1\n• Item 2\n• Item 3"),
        ("Long answer", "The Earth orbits the Sun. It takes 365 days. The Moon orbits Earth."),
    ]
    
    for name, answer in test_cases:
        claims = await extractor.extract_claims(answer)
        print(f"\n{name}: {answer}")
        print(f"  Extracted claims: {len(claims)}")
        for i, claim in enumerate(claims, 1):
            print(f"    {i}. {claim}")


async def example_additional_signals():
    """Demonstrate additional grounding signals."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Additional Grounding Signals")
    print("=" * 60)
    
    backend = OpenAIBackend(api_key=os.getenv("OPENAI_API_KEY"))
    signals = AdditionalSignals(use_embeddings=True)
    
    # Test case
    question = "What is the largest planet in our solar system?"
    answer = "Jupiter is the largest planet, with a diameter of about 143,000 kilometers."
    
    print(f"\nQuestion: {question}")
    print(f"Answer: {answer}")
    
    # Compute signals
    all_signals = await signals.compute_all(question, answer, backend)
    
    print(f"\nAdditional Signals:")
    print(f"  - Semantic Similarity: {all_signals['semantic_similarity']:.3f}")
    print(f"  - Answer Specificity: {all_signals['answer_specificity']:.3f}")
    print(f"  - Entity Consistency: {all_signals['entity_consistency']:.3f}")
    
    print("\nInterpretation:")
    print("  - High semantic similarity → answer relates to question")
    print("  - High specificity → answer contains concrete details")
    print("  - High entity consistency → entities align with question")


async def example_multi_verifier():
    """Example with multi-verifier ensemble (if multiple APIs available)."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Multi-Verifier Ensemble (Optional)")
    print("=" * 60)
    
    # Check which API keys are available
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    has_google = bool(os.getenv("GOOGLE_API_KEY"))
    
    print(f"\nAvailable APIs:")
    print(f"  - OpenAI: {has_openai}")
    print(f"  - Anthropic: {has_anthropic}")
    print(f"  - Google: {has_google}")
    
    if sum([has_openai, has_anthropic, has_google]) < 2:
        print("\nMulti-verifier ensemble requires at least 2 different API providers.")
        print("Skipping this example.")
        return
    
    # Configure ensemble
    verifier_configs = []
    if has_openai:
        verifier_configs.append({
            'backend': 'openai',
            'model': 'gpt-4o-mini',
            'weight': 0.4
        })
    if has_anthropic:
        verifier_configs.append({
            'backend': 'anthropic',
            'model': 'claude-3-haiku-20240307',
            'weight': 0.3
        })
    if has_google:
        verifier_configs.append({
            'backend': 'gemini',
            'model': 'gemini-1.5-flash',
            'weight': 0.3
        })
    
    print(f"\nUsing {len(verifier_configs)} verifiers:")
    for config in verifier_configs:
        print(f"  - {config['backend']}: {config['model']} (weight={config['weight']})")
    
    # Create ensemble
    ensemble = MultiVerifierEnsemble(verifier_configs)
    
    question = "Who wrote Romeo and Juliet?"
    answer = "William Shakespeare wrote Romeo and Juliet in the 1590s."
    
    print(f"\nQuestion: {question}")
    print(f"Answer: {answer}")
    
    # Run ensemble
    result = await ensemble.compute_ensemble_score(question, answer, aggregation='weighted_average')
    
    print(f"\nEnsemble Score: {result['ensemble_score']:.3f}")
    print(f"Aggregation Method: {result['aggregation']}")
    print(f"\nIndividual Results:")
    for r in result['individual_results']:
        print(f"  - {r['verifier']}: {r['score']:.3f} (weight={r['weight']})")


async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("PCIB ENHANCED FEATURES DEMONSTRATION")
    print("Target: AUROC 0.85-0.90 (up from 0.67)")
    print("=" * 60)
    
    try:
        await example_basic_enhanced()
        await example_with_learned_weights()
        await example_with_calibration()
        await example_claim_extraction()
        await example_additional_signals()
        await example_multi_verifier()
        
        print("\n" + "=" * 60)
        print("SUMMARY OF IMPROVEMENTS")
        print("=" * 60)
        print("\n1. ✓ Improved claim extraction with fallback strategies")
        print("2. ✓ Learned signal weights from ablation data")
        print("3. ✓ Enhanced calibration (Platt scaling / Isotonic regression)")
        print("4. ✓ Additional grounding signals (semantic, entity, specificity)")
        print("5. ✓ Multi-verifier ensemble for robustness")
        print("6. ✓ Enhanced perturbation generation")
        print("7. ✓ Claim dependency graph analysis")
        print("8. ✓ Improved trace validation with semantic similarity")
        
        print("\nExpected Performance:")
        print("  - Baseline AUROC: 0.67")
        print("  - Target AUROC: 0.85-0.90")
        print("  - Key improvements: claim extraction, learned weights, additional signals")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

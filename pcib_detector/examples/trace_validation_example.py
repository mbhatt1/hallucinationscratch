"""
Example: Using Trace Validation and Rationalization Detection

This example demonstrates the advanced reasoning trace validation features:
1. Enable trace validation in detector config
2. Detect post-hoc rationalization (fabricated reasoning)
3. Measure trace consistency and support
4. Compare detection with and without trace validation

Trace validation adds:
- Forward reasoning trace generation (claim -> evidence -> conclusion)
- Backward rationalization detection (conclusion given -> explanation)
- Consistency checking (logical coherence)
- Support checking (reasoning actually supports conclusion)
"""

import asyncio
from pcib_detector import PCIBDetector, Config


async def main():
    # Example: potentially hallucinated answer with confident but unsupported reasoning
    question = "When did the French Revolution begin?"
    
    evidence = """
    The French Revolution was a period of major social and political upheaval in France
    that lasted from 1789 to 1799. It began with the Storming of the Bastille on July 14, 1789,
    which has become a national holiday in France (Bastille Day).
    """
    
    # Hallucinated answer (wrong year, but confidently stated)
    hallucinated_answer = """
    The French Revolution began in 1791. This was when the French monarchy was first challenged
    by revolutionary forces, leading to the establishment of the First French Republic.
    """
    
    # Correct answer
    correct_answer = """
    The French Revolution began in 1789 with the Storming of the Bastille on July 14.
    This event marked the beginning of a decade of revolutionary change in France.
    """
    
    print("=" * 80)
    print("TRACE VALIDATION DEMO")
    print("=" * 80)
    
    # -----------------
    # 1. Standard detection (without trace validation)
    # -----------------
    print("\n--- Standard Detection (No Trace Validation) ---\n")
    
    config_standard = Config(
        provider="openai",
        model="gpt-4o-mini",
        enable_trace_validation=False,  # Disabled
    )
    
    detector_standard = PCIBDetector(config=config_standard)
    
    result_standard = await detector_standard.detect_hallucination(
        answer=hallucinated_answer,
        evidence=evidence,
        return_details=True
    )
    
    print(f"Flagged: {result_standard.flagged}")
    print(f"Score: {result_standard.score:.3f}")
    print(f"Claims detected: {len(result_standard.claims)}")
    
    for i, claim in enumerate(result_standard.claims):
        print(f"\nClaim {i+1}: {claim.text}")
        print(f"  Score: {claim.score:.3f}")
        print(f"  Contradict: {claim.signals.post.contradict:.3f}")
        print(f"  Entail: {claim.signals.post.entail:.3f}")
        print(f"  Uptake KL: {claim.signals.uptake_kl:.3f}")
    
    # -----------------
    # 2. Detection WITH trace validation
    # -----------------
    print("\n\n--- Detection WITH Trace Validation ---\n")
    
    config_trace = Config(
        provider="openai",
        model="gpt-4o-mini",
        enable_trace_validation=True,  # Enabled!
        detect_rationalization=True,   # Detect post-hoc rationalization
        trace_temperature=0.3,         # Slight diversity in traces
    )
    
    detector_trace = PCIBDetector(config=config_trace)
    
    result_trace = await detector_trace.detect_hallucination(
        answer=hallucinated_answer,
        evidence=evidence,
        return_details=True
    )
    
    print(f"Flagged: {result_trace.flagged}")
    print(f"Score: {result_trace.score:.3f}")
    print(f"Claims detected: {len(result_trace.claims)}")
    
    for i, claim in enumerate(result_trace.claims):
        print(f"\nClaim {i+1}: {claim.text}")
        print(f"  Score: {claim.score:.3f}")
        print(f"  Contradict: {claim.signals.post.contradict:.3f}")
        print(f"  Entail: {claim.signals.post.entail:.3f}")
        
        # Trace validation metrics
        if claim.signals.trace_consistency is not None:
            print(f"\n  Trace Validation:")
            print(f"    Consistency: {claim.signals.trace_consistency:.3f} (1.0=perfectly consistent)")
            print(f"    Support: {claim.signals.trace_support:.3f} (1.0=fully supports conclusion)")
            print(f"    Rationalization: {claim.signals.rationalization_score:.3f} (0.0=genuine, 1.0=fabricated)")
            print(f"    Trace Length: {claim.signals.trace_length} words")
            
            # Interpretation
            if claim.signals.rationalization_score > 0.3:
                print("    ⚠️  HIGH RATIONALIZATION - Reasoning may be post-hoc fabrication")
            if claim.signals.trace_consistency < 0.5:
                print("    ⚠️  LOW CONSISTENCY - Reasoning contains contradictions")
            if claim.signals.trace_support < 0.5:
                print("    ⚠️  WEAK SUPPORT - Reasoning doesn't actually support conclusion")
    
    # -----------------
    # 3. Compare on correct answer
    # -----------------
    print("\n\n--- Correct Answer (Should Score Lower) ---\n")
    
    result_correct = await detector_trace.detect_hallucination(
        answer=correct_answer,
        evidence=evidence,
        return_details=True
    )
    
    print(f"Flagged: {result_correct.flagged}")
    print(f"Score: {result_correct.score:.3f}")
    
    for i, claim in enumerate(result_correct.claims):
        print(f"\nClaim {i+1}: {claim.text}")
        print(f"  Score: {claim.score:.3f}")
        print(f"  Contradict: {claim.signals.post.contradict:.3f}")
        print(f"  Entail: {claim.signals.post.entail:.3f}")
        
        if claim.signals.trace_consistency is not None:
            print(f"\n  Trace Validation:")
            print(f"    Consistency: {claim.signals.trace_consistency:.3f}")
            print(f"    Support: {claim.signals.trace_support:.3f}")
            print(f"    Rationalization: {claim.signals.rationalization_score:.3f}")
            
            if claim.signals.rationalization_score < 0.2:
                print("    ✓ LOW RATIONALIZATION - Reasoning appears genuine")
            if claim.signals.trace_consistency > 0.7:
                print("    ✓ HIGH CONSISTENCY - Reasoning is logically coherent")
            if claim.signals.trace_support > 0.7:
                print("    ✓ STRONG SUPPORT - Reasoning strongly supports conclusion")
    
    # -----------------
    # Summary comparison
    # -----------------
    print("\n\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)
    print(f"\nHallucinated Answer:")
    print(f"  Standard detection score: {result_standard.score:.3f}")
    print(f"  With trace validation:    {result_trace.score:.3f}")
    print(f"  Improvement: {result_trace.score - result_standard.score:+.3f}")
    
    print(f"\nCorrect Answer:")
    print(f"  With trace validation:    {result_correct.score:.3f}")
    
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print("""
Trace validation adds reasoning transparency:

1. CONSISTENCY: Checks if reasoning is logically coherent
   - Catches circular arguments
   - Detects internal contradictions

2. SUPPORT: Checks if reasoning actually supports the conclusion
   - Catches disconnected reasoning
   - Identifies non-sequiturs

3. RATIONALIZATION: Detects post-hoc fabrication
   - Compares forward (claim->reasoning) vs backward (conclusion->reasoning)
   - High divergence suggests reasoning was fabricated to fit conclusion
   - Most powerful signal for detecting confident hallucinations

Use trace validation when:
- You need to understand WHY something was flagged
- The model might use sophisticated but flawed reasoning
- You're dealing with complex factual claims
- Cost/latency is acceptable (adds ~3x API calls per claim)
""")


if __name__ == "__main__":
    asyncio.run(main())

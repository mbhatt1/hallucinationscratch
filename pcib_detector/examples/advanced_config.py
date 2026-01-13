"""Advanced configuration and signal analysis example."""

import asyncio
import os

from pcib_detector import PCIBDetector, Config


async def main():
    # Ensure API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY environment variable")
        return
    
    # Advanced configuration with ensemble verification
    config = Config(
        model="gpt-4o-mini",
        n_ensemble=3,  # Average over 3 samples for robustness
        ensemble_temperature=0.7,  # Temperature for ensemble diversity
        max_claims=6,  # Extract up to 6 claims per answer
        # Thresholds (tune these based on your use case)
        entail_conf=0.75,
        uptake_low=0.15,
        uptake_high=0.60,
        stress_js_hi=0.12,
        conflict_js_low=0.08,
        contradict_hi=0.30,
    )
    
    detector = PCIBDetector(config)
    
    # Example: Subtle hallucination
    print("=" * 60)
    print("Advanced Detection: Subtle Hallucination")
    print("=" * 60)
    
    answer = """
    The Titanic sank on April 15, 1912, after hitting an iceberg.
    It was carrying over 2,500 passengers and crew, and about 1,800 survived.
    The ship was deemed unsinkable by its designers.
    """
    
    evidence = """
    RMS Titanic sank in the early morning hours of April 15, 1912, in the North Atlantic Ocean,
    after striking an iceberg during her maiden voyage from Southampton to New York City.
    Of the estimated 2,224 passengers and crew aboard, more than 1,500 died, making it one of
    the deadliest peacetime maritime disasters in history.
    """
    
    result = await detector.detect_hallucination(
        answer=answer,
        evidence=evidence,
        return_details=True,
        threshold=0.5,
    )
    
    print(f"\nOverall Score: {result.score:.3f}")
    print(f"Flagged: {'🚨 YES' if result.flagged else '✅ NO'}")
    print(f"\n{'='*60}")
    print("Per-Claim Analysis:")
    print(f"{'='*60}\n")
    
    for i, claim in enumerate(result.claims, 1):
        flag = "🚨" if claim.flagged else "✅"
        print(f"{i}. {flag} {claim.text}")
        print(f"   Score: {claim.score:.3f}")
        
        # Show detailed signals
        s = claim.signals
        print(f"\n   Prior belief (no evidence):")
        print(f"     Entail: {s.prior.entail:.3f}, Contradict: {s.prior.contradict:.3f}, Unknown: {s.prior.unknown:.3f}")
        
        print(f"\n   Posterior belief (with evidence):")
        print(f"     Entail: {s.post.entail:.3f}, Contradict: {s.post.contradict:.3f}, Unknown: {s.post.unknown:.3f}")
        
        print(f"\n   PC+IB Signals:")
        print(f"     📈 Evidence Uptake (KL):       {s.uptake_kl:.3f}")
        print(f"        └─> Did evidence change belief?")
        print(f"     🔊 Bottleneck Stress (JS):     {s.stress_js:.3f}")
        print(f"        └─> Is judgment stable under noise?")
        print(f"     ⚔️  Conflict Sensitivity (JS):  {s.conflict_js:.3f}")
        print(f"        └─> Does it resist contradictions?")
        
        # Feature interpretation
        print(f"\n   Feature Scores:")
        for fname, fval in s.get_features().items():
            print(f"     {fname}: {fval:.3f}")
        
        print(f"\n   {'─'*58}\n")
    
    # Interpretation guide
    print(f"{'='*60}")
    print("Signal Interpretation Guide:")
    print(f"{'='*60}")
    print("""
📈 Evidence Uptake (KL divergence):
   - Low (<0.15):  Evidence barely changed belief → potential hallucination
   - Medium (0.15-0.60): Normal evidence integration
   - High (>0.60): Strong evidence update

🔊 Bottleneck Stress (JS divergence):
   - Low (<0.05): Judgment stable under noise → good
   - Medium (0.05-0.12): Some sensitivity
   - High (>0.12): Unstable judgment → potential issue

⚔️ Conflict Sensitivity (JS divergence):
   - Low (<0.08): Doesn't respond to contradictions → potential issue
   - Medium (0.08-0.20): Normal sensitivity
   - High (>0.20): Strong response to conflicts

🎯 Posterior Probabilities:
   - High Contradict (>0.30): Evidence contradicts claim
   - High Unknown (>0.45): Insufficient evidence
   - High Entail + Low Uptake: Prior-driven (not evidence-driven)
    """)


if __name__ == "__main__":
    asyncio.run(main())

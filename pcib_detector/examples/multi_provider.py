"""Example demonstrating multiple LLM providers."""

import asyncio
import os

from pcib_detector import PCIBDetector, Config


async def main():
    # Example answer-evidence pair with hallucination
    answer = "Python was created by Dennis Ritchie in 1985 and is primarily used for systems programming."
    evidence = "Python is a high-level programming language created by Guido van Rossum. It was first released in 1991."
    
    print("=" * 70)
    print("Testing PCIB Detector with Multiple Providers")
    print("=" * 70)
    print(f"\nAnswer: {answer}")
    print(f"Evidence: {evidence[:100]}...")
    print("\n" + "=" * 70)
    
    # Test each provider
    providers = []
    
    # OpenAI
    if os.getenv("OPENAI_API_KEY"):
        providers.append(("OpenAI", "openai", "gpt-4o-mini"))
    else:
        print("\n⚠️  Skipping OpenAI (OPENAI_API_KEY not set)")
    
    # Anthropic
    if os.getenv("ANTHROPIC_API_KEY"):
        providers.append(("Anthropic", "anthropic", "claude-3-5-sonnet-20241022"))
    else:
        print("\n⚠️  Skipping Anthropic (ANTHROPIC_API_KEY not set)")
    
    # Gemini
    if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
        providers.append(("Gemini", "gemini", "gemini-2.0-flash-exp"))
    else:
        print("\n⚠️  Skipping Gemini (GOOGLE_API_KEY not set)")
    
    if not providers:
        print("\n❌ No API keys found. Please set at least one:")
        print("   - OPENAI_API_KEY")
        print("   - ANTHROPIC_API_KEY")
        print("   - GOOGLE_API_KEY or GEMINI_API_KEY")
        return
    
    # Run detection with each provider
    results = []
    
    for name, provider, model in providers:
        print(f"\n{'─'*70}")
        print(f"🔍 Testing {name} ({model})")
        print(f"{'─'*70}")
        
        try:
            config = Config(provider=provider, model=model)
            detector = PCIBDetector(config)
            
            result = await detector.detect_hallucination(
                answer=answer,
                evidence=evidence,
                return_details=True
            )
            
            status = "🚨 HALLUCINATION DETECTED" if result.flagged else "✅ GROUNDED"
            print(f"\nStatus: {status}")
            print(f"Score: {result.score:.3f}")
            print(f"Claims detected: {len(result.claims)}")
            
            if result.claims:
                print(f"\nPer-claim analysis:")
                for i, claim in enumerate(result.claims, 1):
                    flag = "🚨" if claim.flagged else "✅"
                    print(f"  {i}. {flag} {claim.text}")
                    print(f"     Score: {claim.score:.3f}")
                    print(f"     Signals: contradict={claim.signals.post.contradict:.2f}, "
                          f"entail={claim.signals.post.entail:.2f}, "
                          f"uptake_kl={claim.signals.uptake_kl:.2f}")
            
            results.append({
                "provider": name,
                "model": model,
                "score": result.score,
                "flagged": result.flagged,
            })
            
        except Exception as e:
            print(f"❌ Error with {name}: {e}")
            results.append({
                "provider": name,
                "model": model,
                "error": str(e),
            })
    
    # Summary comparison
    print(f"\n{'='*70}")
    print("Summary Comparison")
    print(f"{'='*70}\n")
    
    print(f"{'Provider':<15} {'Model':<30} {'Score':<10} {'Flagged':<10}")
    print(f"{'─'*15} {'─'*30} {'─'*10} {'─'*10}")
    
    for r in results:
        if "error" in r:
            print(f"{r['provider']:<15} {r['model']:<30} {'ERROR':<10} {'-':<10}")
        else:
            print(f"{r['provider']:<15} {r['model']:<30} {r['score']:<10.3f} {str(r['flagged']):<10}")
    
    print(f"\n{'='*70}")
    print("Key Observations:")
    print(f"{'='*70}")
    print("""
1. All providers should detect this as a hallucination (2 false facts)
2. OpenAI typically provides the highest scores (most sensitive)
3. Anthropic and Gemini may have slightly different sensitivities
4. Scores >0.5 indicate likely hallucination
5. Individual claim scores can vary based on model reasoning

For production use:
- OpenAI (gpt-4o-mini): Best balance of accuracy and cost
- Gemini (gemini-2.0-flash): Lowest cost, good for high volume
- Anthropic (claude-3-5-sonnet): Alternative with strong reasoning
    """)


if __name__ == "__main__":
    asyncio.run(main())

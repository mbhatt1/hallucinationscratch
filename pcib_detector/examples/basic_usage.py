"""Basic usage example for PCIB detector."""

import asyncio
import os

from pcib_detector import PCIBDetector, Config


async def main():
    # Ensure API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY environment variable")
        return
    
    # Create detector with default config
    config = Config(model="gpt-4o-mini")
    detector = PCIBDetector(config)
    
    # Example 1: Grounded answer
    print("=" * 60)
    print("Example 1: Grounded Answer")
    print("=" * 60)
    
    answer1 = "The Eiffel Tower was completed in 1889 and stands 300 meters tall."
    evidence1 = "The Eiffel Tower, completed in 1889, is an iron lattice tower located in Paris, France. It stands approximately 300 meters (984 feet) tall."
    
    result1 = await detector.detect_hallucination(
        answer=answer1,
        evidence=evidence1,
        return_details=True
    )
    
    print(f"Answer: {answer1}")
    print(f"Flagged: {result1.flagged}")
    print(f"Score: {result1.score:.3f}")
    print(f"Claims: {len(result1.claims)}")
    
    # Example 2: Hallucinated answer
    print("\n" + "=" * 60)
    print("Example 2: Hallucinated Answer")
    print("=" * 60)
    
    answer2 = "The Eiffel Tower was completed in 1895 and stands 500 meters tall."
    evidence2 = "The Eiffel Tower, completed in 1889, is an iron lattice tower located in Paris, France. It stands approximately 300 meters (984 feet) tall."
    
    result2 = await detector.detect_hallucination(
        answer=answer2,
        evidence=evidence2,
        return_details=True
    )
    
    print(f"Answer: {answer2}")
    print(f"Flagged: {result2.flagged}")
    print(f"Score: {result2.score:.3f}")
    print(f"Claims:")
    for i, claim in enumerate(result2.claims, 1):
        flag = "🚨" if claim.flagged else "✅"
        print(f"  {i}. {flag} {claim.text} (score: {claim.score:.3f})")
    
    # Example 3: No factual claims
    print("\n" + "=" * 60)
    print("Example 3: Opinion/No Facts")
    print("=" * 60)
    
    answer3 = "I think the Eiffel Tower is beautiful and worth visiting."
    evidence3 = "The Eiffel Tower is an iconic landmark in Paris."
    
    result3 = await detector.detect_hallucination(
        answer=answer3,
        evidence=evidence3,
        return_details=True
    )
    
    print(f"Answer: {answer3}")
    print(f"Flagged: {result3.flagged}")
    print(f"Score: {result3.score:.3f}")
    print(f"Claims: {len(result3.claims)}")


if __name__ == "__main__":
    asyncio.run(main())

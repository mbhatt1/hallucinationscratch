"""Example of batch processing with PCIB detector."""

import asyncio
import os

from pcib_detector import PCIBDetector, Config


async def main():
    # Ensure API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY environment variable")
        return
    
    # Create detector
    config = Config(model="gpt-4o-mini")
    detector = PCIBDetector(config)
    
    # Batch of answer-evidence pairs
    qa_pairs = [
        {
            "answer": "Python was created by Guido van Rossum in 1991.",
            "evidence": "Python is a programming language created by Guido van Rossum. It was first released in 1991.",
        },
        {
            "answer": "Python was created by Dennis Ritchie in 1985.",
            "evidence": "Python is a programming language created by Guido van Rossum. It was first released in 1991.",
        },
        {
            "answer": "The speed of light is approximately 300,000 km/s.",
            "evidence": "The speed of light in vacuum is exactly 299,792,458 meters per second, approximately 300,000 kilometers per second.",
        },
        {
            "answer": "The speed of light is approximately 500,000 km/s.",
            "evidence": "The speed of light in vacuum is exactly 299,792,458 meters per second, approximately 300,000 kilometers per second.",
        },
    ]
    
    print("Processing batch of examples...")
    print("=" * 60)
    
    # Process batch
    results = await detector.detect_batch(
        answers=[pair["answer"] for pair in qa_pairs],
        evidences=[pair["evidence"] for pair in qa_pairs],
        return_details=True,
    )
    
    # Display results
    for i, (pair, result) in enumerate(zip(qa_pairs, results), 1):
        status = "🚨 HALLUCINATION" if result.flagged else "✅ GROUNDED"
        print(f"\nExample {i}: {status}")
        print(f"Answer: {pair['answer']}")
        print(f"Score: {result.score:.3f}")
        
        if result.claims:
            print(f"Claims:")
            for j, claim in enumerate(result.claims, 1):
                flag = "🚨" if claim.flagged else "✅"
                print(f"  {j}. {flag} {claim.text} ({claim.score:.3f})")
        
        print("-" * 60)
    
    # Summary statistics
    n_flagged = sum(1 for r in results if r.flagged)
    avg_score = sum(r.score for r in results) / len(results)
    
    print(f"\nSummary:")
    print(f"  Total examples: {len(results)}")
    print(f"  Flagged: {n_flagged}")
    print(f"  Average score: {avg_score:.3f}")


if __name__ == "__main__":
    asyncio.run(main())

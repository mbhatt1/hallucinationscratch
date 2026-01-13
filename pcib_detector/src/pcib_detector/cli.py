"""Command-line interface for PCIB detector."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .core import PCIBDetector
from .types import Config
from .eval import evaluate_dataset


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="PCIB Detector: Predictive-Coding + Information-Bottleneck hallucination detection"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # detect command
    detect_parser = subparsers.add_parser("detect", help="Detect hallucinations in a single answer")
    detect_parser.add_argument("--answer", required=True, help="Answer text to verify")
    detect_parser.add_argument("--evidence", required=True, help="Evidence/context to verify against")
    detect_parser.add_argument("--provider", default="openai", choices=["openai", "anthropic", "gemini"], help="LLM provider")
    detect_parser.add_argument("--model", help="Model to use (uses provider default if not specified)")
    detect_parser.add_argument("--threshold", type=float, default=0.5, help="Score threshold for flagging")
    detect_parser.add_argument("--n-ensemble", type=int, default=1, help="Number of verification samples")
    detect_parser.add_argument("--ensemble-temperature", type=float, default=0.7, help="Temperature for ensemble")
    detect_parser.add_argument("--output", "-o", help="Output file (JSON)")
    detect_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    # eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate on hallucination benchmark")
    eval_parser.add_argument("--dataset", default="PatronusAI/HaluBench", help="HuggingFace dataset")
    eval_parser.add_argument("--split", default="", help="Dataset split (auto-detect)")
    eval_parser.add_argument("--limit", type=int, default=300, help="Max examples")
    eval_parser.add_argument("--provider", default="openai", choices=["openai", "anthropic", "gemini"], help="LLM provider")
    eval_parser.add_argument("--model", help="Model to use (uses provider default if not specified)")
    eval_parser.add_argument("--n-ensemble", type=int, default=1, help="Ensemble size")
    eval_parser.add_argument("--output", "-o", default="eval_results.jsonl", help="Output file")
    
    # interactive command
    interactive_parser = subparsers.add_parser("interactive", help="Interactive detection mode")
    interactive_parser.add_argument("--provider", default="openai", choices=["openai", "anthropic", "gemini"], help="LLM provider")
    interactive_parser.add_argument("--model", help="Model to use (uses provider default if not specified)")
    interactive_parser.add_argument("--threshold", type=float, default=0.5, help="Score threshold")
    
    args = parser.parse_args()
    
    if args.command == "detect":
        asyncio.run(detect_command(args))
    elif args.command == "eval":
        asyncio.run(eval_command(args))
    elif args.command == "interactive":
        asyncio.run(interactive_command(args))
    else:
        parser.print_help()
        sys.exit(1)


async def detect_command(args):
    """Run single detection."""
    config = Config(
        provider=args.provider,
        model=args.model,
        n_ensemble=args.n_ensemble,
        ensemble_temperature=args.ensemble_temperature
    )
    
    detector = PCIBDetector(config)
    model_name = config.model or detector.backend.get_default_model()
    
    print(f"Detecting hallucinations with {args.provider}/{model_name}...")
    if args.n_ensemble > 1:
        print(f"Using ensemble verification (n={args.n_ensemble})")
    
    result = await detector.detect_hallucination(
        answer=args.answer,
        evidence=args.evidence,
        return_details=True,
        threshold=args.threshold
    )
    
    # Print results
    status = "🚨 HALLUCINATION DETECTED" if result.flagged else "✅ GROUNDED"
    print(f"\n{status}")
    print(f"Overall Score: {result.score:.3f} (threshold: {args.threshold})")
    
    if result.claims:
        print(f"\n📋 Claims ({len(result.claims)}):")
        for i, claim in enumerate(result.claims, 1):
            flag = "🚨" if claim.flagged else "✅"
            print(f"\n{i}. {flag} {claim.text}")
            print(f"   Score: {claim.score:.3f}")
            
            if args.verbose:
                print(f"   Signals:")
                print(f"     Contradict: {claim.signals.post.contradict:.3f}")
                print(f"     Entail: {claim.signals.post.entail:.3f}")
                print(f"     Unknown: {claim.signals.post.unknown:.3f}")
                print(f"     Uptake KL: {claim.signals.uptake_kl:.3f}")
                print(f"     Stress JS: {claim.signals.stress_js:.3f}")
                print(f"     Conflict JS: {claim.signals.conflict_js:.3f}")
    
    # Save to file if requested
    if args.output:
        output_data = {
            "flagged": result.flagged,
            "score": result.score,
            "threshold": args.threshold,
            "answer": result.answer,
            "evidence": result.evidence,
            "claims": [
                {
                    "text": c.text,
                    "score": c.score,
                    "flagged": c.flagged,
                    "signals": {
                        "contradict": c.signals.post.contradict,
                        "entail": c.signals.post.entail,
                        "unknown": c.signals.post.unknown,
                        "uptake_kl": c.signals.uptake_kl,
                        "stress_js": c.signals.stress_js,
                        "conflict_js": c.signals.conflict_js,
                    }
                }
                for c in result.claims
            ]
        }
        
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n💾 Results saved to {args.output}")


async def eval_command(args):
    """Run evaluation on benchmark dataset."""
    config = Config(
        provider=args.provider,
        model=args.model,
        n_ensemble=args.n_ensemble,
    )
    
    detector = PCIBDetector(config)
    
    await evaluate_dataset(
        detector=detector,
        dataset_name=args.dataset,
        split=args.split,
        limit=args.limit,
        output_file=args.output,
        verbose=True,
    )


async def interactive_command(args):
    """Run interactive detection mode."""
    config = Config(provider=args.provider, model=args.model)
    detector = PCIBDetector(config)
    model_name = config.model or detector.backend.get_default_model()
    
    print(f"🔍 PCIB Detector (Interactive Mode)")
    print(f"Provider: {args.provider}")
    print(f"Model: {model_name}")
    print(f"Threshold: {args.threshold}")
    print(f"\nEnter 'quit' or 'exit' to stop\n")
    
    while True:
        try:
            print("=" * 60)
            answer = input("\n📝 Answer: ").strip()
            
            if answer.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            
            if not answer:
                print("❌ Answer cannot be empty")
                continue
            
            evidence = input("📚 Evidence: ").strip()
            
            if not evidence:
                print("❌ Evidence cannot be empty")
                continue
            
            print("\n🔄 Analyzing...")
            
            result = await detector.detect_hallucination(
                answer=answer,
                evidence=evidence,
                return_details=True,
                threshold=args.threshold
            )
            
            status = "🚨 HALLUCINATION" if result.flagged else "✅ GROUNDED"
            print(f"\n{status} (score: {result.score:.3f})")
            
            if result.claims:
                print(f"\nClaims:")
                for i, claim in enumerate(result.claims, 1):
                    flag = "🚨" if claim.flagged else "✅"
                    print(f"  {i}. {flag} {claim.text} ({claim.score:.3f})")
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except EOFError:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()

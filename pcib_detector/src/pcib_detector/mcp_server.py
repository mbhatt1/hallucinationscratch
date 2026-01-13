"""MCP server for PCIB detector integration with Claude Desktop."""

import asyncio
import json
import logging
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from .core import PCIBDetector
from .types import Config


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pcib-detector-mcp")

# Initialize detector
detector = None


def get_detector() -> PCIBDetector:
    """Lazy initialize detector."""
    global detector
    if detector is None:
        config = Config(model="gpt-4o-mini")  # Can be overridden via env vars
        detector = PCIBDetector(config)
    return detector


# Create MCP server
app = Server("pcib-detector")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="detect_hallucination",
            description=(
                "Detect hallucinations in an answer using Predictive-Coding + Information-Bottleneck analysis. "
                "Tests whether the model actually uses evidence by measuring: "
                "(1) evidence uptake (did evidence update beliefs?), "
                "(2) bottleneck stress (is judgment stable under noise?), "
                "(3) conflict sensitivity (does model resist contradictions?). "
                "Returns flagged status, confidence score, and per-claim analysis."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The answer text to verify for hallucinations"
                    },
                    "evidence": {
                        "type": "string",
                        "description": "The evidence/context to verify against"
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Score threshold for flagging (0-1, default 0.5)",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "default": 0.5
                    }
                },
                "required": ["answer", "evidence"]
            }
        ),
        Tool(
            name="explain_signals",
            description=(
                "Explain the PC+IB signals used for hallucination detection. "
                "Useful for understanding what each signal means and how to interpret the results."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool calls."""
    
    if name == "detect_hallucination":
        answer = arguments.get("answer", "")
        evidence = arguments.get("evidence", "")
        threshold = arguments.get("threshold", 0.5)
        
        if not answer:
            return [TextContent(
                type="text",
                text="Error: 'answer' parameter is required"
            )]
        
        if not evidence:
            return [TextContent(
                type="text",
                text="Error: 'evidence' parameter is required"
            )]
        
        try:
            det = get_detector()
            result = await det.detect_hallucination(
                answer=answer,
                evidence=evidence,
                return_details=True,
                threshold=threshold
            )
            
            # Format response
            status = "🚨 HALLUCINATION DETECTED" if result.flagged else "✅ GROUNDED"
            
            response = f"""
## {status}

**Overall Score:** {result.score:.3f} (threshold: {threshold})

### Claims Analysis

"""
            
            for i, claim in enumerate(result.claims, 1):
                flag_emoji = "🚨" if claim.flagged else "✅"
                response += f"""
#### {flag_emoji} Claim {i} (score: {claim.score:.3f})
**Text:** {claim.text}

**Signals:**
- Contradict: {claim.signals.post.contradict:.3f}
- Entail: {claim.signals.post.entail:.3f}
- Unknown: {claim.signals.post.unknown:.3f}
- Uptake KL: {claim.signals.uptake_kl:.3f}
- Stress JS: {claim.signals.stress_js:.3f}
- Conflict JS: {claim.signals.conflict_js:.3f}
- BF Contradict: {claim.signals.bayes_factor_contradict():.3f}
- BF Entail: {claim.signals.bayes_factor_entail():.3f}

"""
            
            if result.signals:
                response += f"""
### Aggregate Signals

- Mean Contradict: {result.signals['mean_contradict']:.3f}
- Mean Entail: {result.signals['mean_entail']:.3f}
- Mean Uptake KL: {result.signals['mean_uptake_kl']:.3f}
- Mean Stress JS: {result.signals['mean_stress_js']:.3f}
- Mean Conflict JS: {result.signals['mean_conflict_js']:.3f}
- Flagged Claims: {result.signals['n_flagged']}/{len(result.claims)}

"""
            
            response += """
### Interpretation Guide

**Score Ranges:**
- 0.0-0.3: Likely grounded
- 0.3-0.5: Uncertain (manual review)
- 0.5-0.7: Likely hallucination
- 0.7-1.0: High confidence hallucination

**Key Signals:**
- High contradict → Evidence refutes claim
- Low entail → Evidence doesn't support claim
- Low uptake KL → Evidence didn't update beliefs
- High stress JS → Judgment unstable under noise
- Low conflict JS → Doesn't resist contradictions
- Positive BF contradict → Strong evidence for contradiction
- Negative BF entail → Evidence doesn't support
"""
            
            return [TextContent(type="text", text=response)]
            
        except Exception as e:
            logger.error(f"Error in detect_hallucination: {e}", exc_info=True)
            return [TextContent(
                type="text",
                text=f"Error: {str(e)}"
            )]
    
    elif name == "explain_signals":
        explanation = """
# PC+IB Signal Explanation

## 1. Evidence Uptake (Predictive Coding)

**Signal:** `uptake_kl = KL(p(y|evidence) || p(y|∅))`

Measures how much the evidence updated the model's beliefs from a neutral prior.

- **Low uptake + high confidence** = Hallucination signal
  - Model made confident claim without using evidence
  - Likely drawing from training data instead

- **High uptake** = Evidence-grounded
  - Model's beliefs changed substantially after seeing evidence

## 2. Bottleneck Stress (Information Theory)

**Signal:** `stress_js = JS(p(y|evidence), p(y|evidence+distractor))`

Tests stability by adding irrelevant noise (distractor text).

- **High stress** = Unstable judgment
  - Judgment changes when irrelevant text is added
  - Indicates uncertainty or weak grounding

- **Low stress** = Stable judgment
  - Robust to noise

## 3. Conflict Sensitivity

**Signal:** `conflict_js = JS(p(y|evidence), p(y|evidence+conflict))`

Tests whether model resists contradictory information.

- **Low conflict sensitivity** = Ungrounded
  - Judgment doesn't change even when contradicted
  - Model not actually using evidence

- **High conflict sensitivity** = Evidence-dependent
  - Model changes judgment when evidence conflicts
  - Properly grounded in evidence

## 4. Posterior Distribution

**Signals:** `p(ENTAIL)`, `p(CONTRADICT)`, `p(UNKNOWN)`

Direct assessment of claim-evidence relationship.

- **High CONTRADICT** = Evidence refutes claim
- **Low ENTAIL** = Evidence doesn't support claim
- **High UNKNOWN** = Evidence insufficient or ambiguous

## 5. Bayes Factors

**Signals:** `BF_contradict`, `BF_entail`

Log-ratio of posterior odds to prior odds.

- **Positive BF_contradict** = Evidence supports contradiction
- **Negative BF_entail** = Evidence doesn't support claim

## Integration

All signals are combined via calibrated logistic regression:
- Log-odds transformations (amplify extremes)
- Interaction terms (non-linear patterns)
- Ensemble averaging (reduce noise)

**Target:** AUROC ≈ 0.86 on HaluBench with GPT-5.2
"""
        return [TextContent(type="text", text=explanation)]
    
    else:
        return [TextContent(
            type="text",
            text=f"Unknown tool: {name}"
        )]


async def main():
    """Run the MCP server."""
    logger.info("Starting PCIB Detector MCP server")
    
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())

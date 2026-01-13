# PCIB Detector MCP Server Setup

This guide explains how to set up the PCIB Detector as an MCP (Model Context Protocol) server for Claude Desktop.

## What is MCP?

MCP (Model Context Protocol) allows Claude Desktop to access external tools and resources. The PCIB Detector MCP server provides Claude with the ability to detect hallucinations in AI-generated text using predictive coding and information bottleneck theory.

## Prerequisites

1. **Claude Desktop** installed on your system
2. **Python 3.10+** with pip
3. **OpenAI API Key**

## Installation

### 1. Install the Package

```bash
# Clone or download the repository
cd pcib_detector

# Install with MCP dependencies
pip install -e ".[mcp]"
```

### 2. Set Up OpenAI API Key

The detector uses OpenAI's API for verification. Set your API key:

```bash
export OPENAI_API_KEY=your_api_key_here
```

Or create a `.env` file in your home directory:

```bash
echo "OPENAI_API_KEY=your_api_key_here" >> ~/.env
```

## Claude Desktop Configuration

### 1. Locate Configuration File

The Claude Desktop configuration file is located at:

**macOS/Linux:**
```
~/Library/Application Support/Claude/claude_desktop_config.json
```

**Windows:**
```
%APPDATA%\Claude\claude_desktop_config.json
```

### 2. Add MCP Server Configuration

Edit the configuration file and add the PCIB detector server:

```json
{
  "mcpServers": {
    "pcib-detector": {
      "command": "python",
      "args": [
        "-m",
        "pcib_detector.mcp_server"
      ],
      "env": {
        "OPENAI_API_KEY": "your_api_key_here"
      }
    }
  }
}
```

**Important:** Replace `your_api_key_here` with your actual OpenAI API key.

### Alternative: Use System Environment Variables

If you've already set `OPENAI_API_KEY` in your shell environment:

```json
{
  "mcpServers": {
    "pcib-detector": {
      "command": "python",
      "args": [
        "-m",
        "pcib_detector.mcp_server"
      ]
    }
  }
}
```

### 3. Restart Claude Desktop

After saving the configuration:
1. Quit Claude Desktop completely
2. Restart Claude Desktop
3. The PCIB detector tools should now be available

## Available Tools

### 1. `detect_hallucination`

Detects hallucinations in AI-generated answers by comparing against evidence.

**Parameters:**
- `answer` (required): The AI-generated answer to verify
- `evidence` (required): The source evidence/context
- `threshold` (optional): Score threshold for flagging (default: 0.5)

**Example Claude prompt:**
```
Please check if this answer contains hallucinations:

Answer: "Python was created by Dennis Ritchie in 1985."
Evidence: "Python is a programming language created by Guido van Rossum in 1991."
```

### 2. `explain_signals`

Explains the PC+IB methodology and signals.

**Example Claude prompt:**
```
Explain how the PCIB detector signals work.
```

## Usage Examples in Claude

### Basic Detection

```
I have an AI answer that I want to verify. Can you check if it's grounded?

Answer: "The Eiffel Tower was completed in 1895 and stands 500 meters tall."

Evidence: "The Eiffel Tower, completed in 1889, is an iron lattice tower located in Paris, France. It stands approximately 300 meters (984 feet) tall."
```

### Batch Analysis

```
I have several answers to verify. Can you check each one?

1. Answer: "Shakespeare was born in 1564 in Stratford-upon-Avon."
   Evidence: "William Shakespeare was born in April 1564 in Stratford-upon-Avon, England."

2. Answer: "Einstein published his theory of relativity in 1920."
   Evidence: "Einstein published his special theory of relativity in 1905 and general relativity in 1915."
```

### Understanding Results

The detector provides:
- **Overall Score**: 0-10+ risk score (higher = more likely hallucination)
- **Flagged**: ✅ (grounded) or 🚨 (potential hallucination)
- **Per-Claim Analysis**: Individual claims and their signals
- **PC+IB Signals**: Evidence uptake, bottleneck stress, conflict sensitivity

## Interpreting Signals

### 📈 Evidence Uptake (KL divergence)
- **Low (<0.15)**: Evidence barely changed belief → potential hallucination
- **Normal (0.15-0.60)**: Healthy evidence integration
- **High (>0.60)**: Strong evidence update

### 🔊 Bottleneck Stress (JS divergence)
- **Low (<0.05)**: Stable under noise → good
- **High (>0.12)**: Unstable judgment → potential issue

### ⚔️ Conflict Sensitivity (JS divergence)
- **Low (<0.08)**: Ignores contradictions → potential issue
- **Normal (0.08-0.20)**: Responds to conflicts appropriately
- **High (>0.20)**: Strong conflict response

### 🎯 Posterior Probabilities
- **High Contradict (>0.30)**: Evidence contradicts claim
- **High Unknown (>0.45)**: Insufficient evidence
- **High Entail + Low Uptake**: Prior-driven (not evidence-driven)

## Troubleshooting

### Server Not Appearing in Claude

1. Check the configuration file path is correct
2. Ensure the JSON syntax is valid (use a JSON validator)
3. Verify Python is in your PATH: `which python`
4. Check logs in Claude Desktop (Help → View Logs)

### API Key Issues

1. Verify your OpenAI API key is valid
2. Check you have API credits available
3. Test the key manually: `curl https://api.openai.com/v1/models -H "Authorization: Bearer $OPENAI_API_KEY"`

### Performance

- **First request is slow**: The server initializes on first use
- **Subsequent requests**: Should be faster due to connection pooling
- **Cost**: Typical cost is ~$0.001-0.003 per answer with gpt-4o-mini

### Rate Limits

The detector automatically handles OpenAI rate limits with exponential backoff. If you hit rate limits:
1. Wait a few seconds and try again
2. Consider upgrading your OpenAI tier
3. Use a lower volume of requests

## Advanced Configuration

### Custom Model

To use a different OpenAI model, modify the server startup in `mcp_server.py`:

```python
config = Config(model="gpt-4o")  # Use GPT-4o instead of gpt-4o-mini
```

### Ensemble Verification

For higher accuracy (at increased cost), enable ensemble verification:

```python
config = Config(
    model="gpt-4o-mini",
    n_ensemble=3,  # Average over 3 samples
    ensemble_temperature=0.7
)
```

### Custom Thresholds

Tune detection thresholds for your use case:

```python
config = Config(
    entail_conf=0.75,      # Confidence threshold
    uptake_low=0.15,       # Low uptake threshold
    uptake_high=0.60,      # High uptake threshold
    stress_js_hi=0.12,     # Stress threshold
    conflict_js_low=0.08,  # Conflict sensitivity threshold
    contradict_hi=0.30,    # Contradiction threshold
)
```

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check the examples in `examples/`
- Read the main README.md for technical details

## References

- [Model Context Protocol (MCP)](https://modelcontextprotocol.io/)
- [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses)
- [OpenAI Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs)

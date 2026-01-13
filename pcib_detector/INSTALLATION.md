# Installation Guide

Complete installation instructions for the PCIB Detector package.

## Requirements

- Python 3.10 or higher
- pip package manager
- OpenAI API key

## Basic Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/yourusername/pcib-detector.git
cd pcib-detector

# Install the package
pip install -e .
```

### From PyPI (when published)

```bash
pip install pcib-detector
```

## Optional Dependencies

### MCP Server Support

For Claude Desktop integration:

```bash
pip install -e ".[mcp]"
```

### Evaluation Tools

For running benchmark evaluations:

```bash
pip install -e ".[eval]"
```

### Development Tools

For contributing to the project:

```bash
pip install -e ".[dev]"
```

### All Optional Dependencies

```bash
pip install -e ".[mcp,eval,dev]"
```

## OpenAI API Setup

### 1. Get an API Key

1. Sign up at [OpenAI Platform](https://platform.openai.com/)
2. Navigate to API Keys section
3. Create a new API key

### 2. Set Environment Variable

**macOS/Linux:**
```bash
export OPENAI_API_KEY=your_api_key_here
```

Add to `~/.bashrc` or `~/.zshrc` for persistence:
```bash
echo 'export OPENAI_API_KEY=your_api_key_here' >> ~/.zshrc
source ~/.zshrc
```

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="your_api_key_here"
```

For persistence, add to user environment variables through System Properties.

### 3. Verify Installation

```bash
python -c "import pcib_detector; print(pcib_detector.__version__)"
```

Expected output: `0.1.0`

## Usage Modes

### 1. Python Library

```python
import asyncio
from pcib_detector import PCIBDetector, Config

async def main():
    detector = PCIBDetector(Config(model="gpt-4o-mini"))
    result = await detector.detect_hallucination(
        answer="...",
        evidence="..."
    )
    print(result.flagged)

asyncio.run(main())
```

### 2. Command-Line Interface

```bash
# Single detection
pcib detect \
  --answer "Python was created in 1991" \
  --evidence "Python was first released in 1991 by Guido van Rossum"

# Batch evaluation
pcib eval \
  --dataset PatronusAI/HaluBench \
  --limit 500 \
  --output results.jsonl

# Interactive mode
pcib interactive
```

### 3. MCP Server

See [`MCP_SETUP.md`](MCP_SETUP.md) for detailed Claude Desktop integration instructions.

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError`:

```bash
# Ensure you're in the correct directory
pip install -e .

# Or use absolute path
pip install -e /path/to/pcib-detector
```

### OpenAI API Errors

**Issue:** `AuthenticationError`
- Verify your API key is correct
- Check you have API credits available
- Test with: `curl https://api.openai.com/v1/models -H "Authorization: Bearer $OPENAI_API_KEY"`

**Issue:** `RateLimitError`
- Wait a few seconds and retry
- The detector automatically handles rate limits with backoff
- Consider upgrading your OpenAI tier

**Issue:** `APIConnectionError`
- Check your internet connection
- Verify OpenAI services are operational: https://status.openai.com/

### Python Version Issues

If you encounter compatibility errors:

```bash
# Check your Python version
python --version

# Should be 3.10 or higher
# If not, install Python 3.10+:

# macOS (using Homebrew)
brew install python@3.10

# Ubuntu/Debian
sudo apt-get install python3.10

# Windows
# Download from https://www.python.org/downloads/
```

### Permission Issues

**Linux/macOS:**
```bash
# If you get permission errors
pip install --user -e .

# Or use a virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install -e .
```

**Windows:**
```powershell
# Use a virtual environment
python -m venv venv
.\venv\Scripts\activate
pip install -e .
```

## Virtual Environment (Recommended)

Using a virtual environment prevents package conflicts:

```bash
# Create virtual environment
python -m venv pcib-env

# Activate it
# macOS/Linux:
source pcib-env/bin/activate
# Windows:
pcib-env\Scripts\activate

# Install package
pip install -e .

# When done
deactivate
```

## Docker Setup (Advanced)

Create a `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install package
COPY . /app
RUN pip install -e ".[mcp,eval]"

# Set API key (or pass at runtime)
ENV OPENAI_API_KEY=${OPENAI_API_KEY}

CMD ["python", "-m", "pcib_detector.cli"]
```

Build and run:

```bash
docker build -t pcib-detector .
docker run -e OPENAI_API_KEY=$OPENAI_API_KEY pcib-detector
```

## Upgrading

### From Source

```bash
cd pcib-detector
git pull
pip install -e . --upgrade
```

### From PyPI

```bash
pip install --upgrade pcib-detector
```

## Uninstallation

```bash
pip uninstall pcib-detector
```

## Next Steps

- Read [`README.md`](README.md) for usage examples
- Try the examples in [`examples/`](examples/)
- Set up MCP server: [`MCP_SETUP.md`](MCP_SETUP.md)
- Run benchmark evaluation: `pcib eval --help`

## Support

For issues or questions:
- GitHub Issues: https://github.com/yourusername/pcib-detector/issues
- Documentation: [`README.md`](README.md)
- Examples: [`examples/`](examples/)

# Contributing to PCIB Detector

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

### 1. Fork and Clone

```bash
git clone https://github.com/yourusername/pcib-detector.git
cd pcib-detector
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Development Dependencies

```bash
pip install -e ".[dev,mcp,eval]"
```

### 4. Set Up Pre-commit Hooks (Optional)

```bash
pip install pre-commit
pre-commit install
```

## Code Style

We use:
- **Black** for code formatting (line length: 120)
- **Ruff** for linting
- **Type hints** for all functions

### Format Code

```bash
black src/ examples/
```

### Lint Code

```bash
ruff check src/ examples/
```

## Project Structure

```
pcib_detector/
├── src/pcib_detector/
│   ├── __init__.py         # Package exports
│   ├── core.py             # Main detector class
│   ├── types.py            # Data structures
│   ├── math_utils.py       # Mathematical functions
│   ├── openai_backend.py   # OpenAI API wrapper
│   ├── perturbations.py    # Evidence perturbations
│   ├── calibration.py      # Weight learning
│   ├── eval.py             # Evaluation tools
│   ├── cli.py              # Command-line interface
│   └── mcp_server.py       # MCP server
├── examples/               # Usage examples
├── tests/                  # Unit tests (TODO)
└── docs/                   # Documentation (TODO)
```

## Making Changes

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

- Write clear, documented code
- Add type hints
- Follow existing code style
- Update documentation if needed

### 3. Test Your Changes

```bash
# Run examples
python examples/basic_usage.py

# Run eval (if applicable)
pcib eval --limit 50
```

### 4. Commit Changes

```bash
git add .
git commit -m "feat: add feature description"
```

Use conventional commit messages:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `refactor:` Code refactoring
- `test:` Adding tests
- `chore:` Maintenance tasks

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Types of Contributions

### Bug Reports

When reporting bugs, include:
- Python version
- Package version
- Minimal reproduction example
- Error message/traceback
- Expected vs actual behavior

### Feature Requests

When requesting features:
- Clear description of the feature
- Use case and motivation
- Example API if applicable
- Potential implementation approach

### Code Contributions

Areas where contributions are welcome:

#### 1. Core Detection
- Improve signal extraction
- Add new perturbation strategies
- Optimize performance
- Better conflict generation

#### 2. Evaluation
- Add more benchmark datasets
- Improve calibration methods
- Cross-validation utilities
- Performance profiling

#### 3. MCP Server
- Additional tools
- Better error handling
- Streaming responses
- Caching optimizations

#### 4. Testing
- Unit tests for all modules
- Integration tests
- End-to-end tests
- Benchmark regression tests

#### 5. Documentation
- API documentation
- Tutorials and guides
- Jupyter notebooks
- Video tutorials

#### 6. Tooling
- CLI improvements
- Web interface
- Jupyter widget
- VS Code extension

## Testing Guidelines

### Writing Tests

```python
# tests/test_core.py
import pytest
from pcib_detector import PCIBDetector, Config

@pytest.mark.asyncio
async def test_detect_grounded():
    detector = PCIBDetector(Config())
    result = await detector.detect_hallucination(
        answer="The Eiffel Tower is in Paris.",
        evidence="The Eiffel Tower is located in Paris, France."
    )
    assert result.score < 0.5  # Should be grounded
```

### Running Tests

```bash
pytest tests/ -v
```

## Performance Considerations

- The detector makes multiple API calls per answer
- Use `detect_batch()` for processing multiple examples
- Enable ensemble verification only when needed (increases cost)
- Consider caching for repeated verifications

## Benchmarking

When making performance changes:

```python
import time

start = time.time()
result = await detector.detect_hallucination(answer, evidence)
elapsed = time.time() - start
print(f"Time: {elapsed:.2f}s")
```

Compare before/after changes.

## Documentation

Update documentation when:
- Adding new features
- Changing APIs
- Adding configuration options
- Fixing bugs that affect usage

Documentation files:
- `README.md` - Main documentation
- `INSTALLATION.md` - Installation guide
- `MCP_SETUP.md` - MCP server setup
- `examples/README.md` - Example scripts
- Docstrings in code

## Release Process

(For maintainers)

### 1. Update Version

Edit `pyproject.toml` and `src/pcib_detector/__init__.py`:

```python
__version__ = "0.2.0"
```

### 2. Update Changelog

Document changes in `CHANGELOG.md`.

### 3. Build Package

```bash
python -m build
```

### 4. Test Package

```bash
pip install dist/pcib_detector-0.2.0-py3-none-any.whl
```

### 5. Publish to PyPI

```bash
python -m twine upload dist/*
```

### 6. Tag Release

```bash
git tag v0.2.0
git push origin v0.2.0
```

## Questions?

- Open an issue for questions
- Check existing issues and PRs
- Read the documentation
- Try the examples

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Provide constructive feedback
- Focus on improving the project

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Acknowledgments

Thank you for contributing to PCIB Detector! Your contributions help make AI systems more reliable and trustworthy.

# Publishing Guide

**Complete guide for publishing PCIB Detector to PyPI**

## Prerequisites

### 1. PyPI Account Setup

1. **Create accounts:**
   - Production PyPI: https://pypi.org/account/register/
   - Test PyPI: https://test.pypi.org/account/register/

2. **Enable 2FA** (required for API token creation)

3. **Create API tokens:**
   
   **Production PyPI:**
   - Go to https://pypi.org/manage/account/token/
   - Click "Add API token"
   - Name: `pcib-detector-github-actions`
   - Scope: Project `pcib-detector` (or entire account for first publish)
   - Copy token (starts with `pypi-`)
   
   **Test PyPI:**
   - Go to https://test.pypi.org/manage/account/token/
   - Same process as above
   - Copy token

### 2. GitHub Secrets Setup

Add secrets to your GitHub repository:

1. Go to repository Settings → Secrets and variables → Actions
2. Add these secrets:
   - `PYPI_API_TOKEN`: Your production PyPI token
   - `TEST_PYPI_API_TOKEN`: Your Test PyPI token

## Version Management

### Versioning Scheme

We use [Semantic Versioning](https://semver.org/):

```
MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]
```

- **MAJOR**: Incompatible API changes
- **MINOR**: New features, backwards compatible
- **PATCH**: Bug fixes, backwards compatible
- **PRERELEASE**: alpha, beta, rc1, etc.
- **BUILD**: build metadata

Examples:
- `1.0.0` - First stable release
- `1.1.0` - New feature added
- `1.1.1` - Bug fix
- `2.0.0` - Breaking changes
- `1.2.0-beta.1` - Pre-release

### Update Version

Edit `pcib_detector/pyproject.toml`:

```toml
[project]
name = "pcib-detector"
version = "1.0.0"  # ← Update this
```

And update `pcib_detector/src/pcib_detector/__init__.py`:

```python
__version__ = "1.0.0"  # ← Update this
```

## Publishing Process

### Option 1: Automatic (via GitHub Release) - RECOMMENDED

1. **Update version numbers** (see above)

2. **Update CHANGELOG.md:**
   ```markdown
   ## [1.0.1] - 2026-01-15
   
   ### Fixed
   - Bug fix description
   
   ### Changed
   - Change description
   ```

3. **Commit and push:**
   ```bash
   git add pcib_detector/pyproject.toml pcib_detector/src/pcib_detector/__init__.py CHANGELOG.md
   git commit -m "Bump version to 1.0.1"
   git push origin main
   ```

4. **Create and push tag:**
   ```bash
   git tag -a v1.0.1 -m "Release version 1.0.1"
   git push origin v1.0.1
   ```

5. **Create GitHub Release:**
   - Go to GitHub → Releases → "Draft a new release"
   - Choose tag: `v1.0.1`
   - Release title: `v1.0.1`
   - Description: Copy from CHANGELOG.md
   - Click "Publish release"

6. **GitHub Actions automatically:**
   - Runs tests
   - Builds distribution
   - Publishes to PyPI
   - Attaches artifacts to release

7. **Verify publication:**
   ```bash
   # Wait a few minutes, then:
   pip install --upgrade pcib-detector
   python -c "import pcib_detector; print(pcib_detector.__version__)"
   ```

### Option 2: Test PyPI First (Manual)

For testing before production release:

1. **Build distribution:**
   ```bash
   cd pcib_detector
   python -m build
   ```

2. **Check distribution:**
   ```bash
   twine check dist/*
   ```

3. **Upload to Test PyPI:**
   ```bash
   twine upload --repository testpypi dist/* --verbose
   ```
   
   Or trigger via GitHub Actions:
   - Go to Actions → "Publish to PyPI"
   - Click "Run workflow"
   - This uploads to Test PyPI

4. **Test installation from Test PyPI:**
   ```bash
   pip install --index-url https://test.pypi.org/simple/ \
       --extra-index-url https://pypi.org/simple/ \
       pcib-detector
   ```

5. **If successful, create GitHub Release** (Option 1)

### Option 3: Manual Production Publish

Only use if GitHub Actions is unavailable:

1. **Build distribution:**
   ```bash
   cd pcib_detector
   rm -rf dist/  # Clean old builds
   python -m build
   ```

2. **Check distribution:**
   ```bash
   twine check dist/*
   ```

3. **Upload to PyPI:**
   ```bash
   twine upload dist/*
   ```
   
   You'll be prompted for:
   - Username: `__token__`
   - Password: Your PyPI API token

## Pre-Release Checklist

Before creating a release:

- [ ] All tests pass locally
- [ ] All CI checks pass on GitHub
- [ ] Version bumped in `pyproject.toml` and `__init__.py`
- [ ] CHANGELOG.md updated
- [ ] Documentation updated if needed
- [ ] Examples tested and working
- [ ] No API keys or secrets in code
- [ ] Dependencies up to date and secure

```bash
# Run local checks
cd pcib_detector

# Install dev dependencies
pip install -e ".[dev]"

# Format code
black src/ examples/
ruff check src/ examples/ --fix

# Type check
mypy src/pcib_detector --ignore-missing-imports

# Build and check
python -m build
twine check dist/*
```

## Post-Release Checklist

After publishing:

- [ ] Verify installation: `pip install --upgrade pcib-detector`
- [ ] Test basic functionality
- [ ] Update documentation website (if applicable)
- [ ] Announce on social media / relevant channels
- [ ] Close related GitHub issues
- [ ] Update project board

## Common Issues

### Issue: "File already exists"

**Cause:** Trying to upload same version again

**Solution:**
1. Bump version number
2. Clean and rebuild:
   ```bash
   rm -rf dist/ build/ *.egg-info
   python -m build
   ```

### Issue: "Invalid or non-existent authentication"

**Cause:** Wrong API token or expired token

**Solution:**
1. Generate new API token on PyPI
2. Update GitHub secret
3. For manual upload: `export TWINE_PASSWORD=pypi-...`

### Issue: "Package name already exists"

**Cause:** Name conflict with existing package

**Solution:**
1. Choose different name in `pyproject.toml`
2. Update all references in code
3. Or contact PyPI to transfer/claim name

### Issue: "Metadata validation failed"

**Cause:** Invalid metadata in pyproject.toml

**Solution:**
1. Check `pyproject.toml` syntax
2. Run `twine check dist/*`
3. Fix reported issues

## Release Types

### Patch Release (1.0.0 → 1.0.1)

Bug fixes only:

```bash
# Fix bugs
git commit -m "Fix: description"

# Update version
vim pcib_detector/pyproject.toml  # 1.0.0 → 1.0.1
vim pcib_detector/src/pcib_detector/__init__.py

# Update changelog
vim CHANGELOG.md

# Create release
git tag -a v1.0.1 -m "Patch release: bug fixes"
git push origin v1.0.1
```

### Minor Release (1.0.0 → 1.1.0)

New features (backwards compatible):

```bash
# Develop features
git commit -m "feat: new feature"

# Update version
# 1.0.0 → 1.1.0

# Update changelog with "Added" section

# Create release
git tag -a v1.1.0 -m "Minor release: new features"
git push origin v1.1.0
```

### Major Release (1.0.0 → 2.0.0)

Breaking changes:

```bash
# Make breaking changes
git commit -m "BREAKING: API change description"

# Update version
# 1.0.0 → 2.0.0

# Update changelog with "Changed" and migration guide

# Create release
git tag -a v2.0.0 -m "Major release: breaking changes"
git push origin v2.0.0
```

### Pre-Release (1.1.0-beta.1)

Testing before stable:

```bash
# Update version to pre-release
# 1.0.0 → 1.1.0-beta.1

# Mark as pre-release in GitHub
# Check "This is a pre-release" when creating release
```

## Rollback Procedure

If a release has critical issues:

1. **Yank the release on PyPI:**
   - Go to PyPI project page
   - Click on the version → "Options" → "Yank"
   - Provide reason

2. **Create hotfix:**
   ```bash
   git revert <bad-commit>
   # Or fix directly
   ```

3. **Release new patch version:**
   ```bash
   # 1.0.1 (broken) → 1.0.2 (fixed)
   ```

4. **Communicate:**
   - Update GitHub release notes
   - Announce the fix
   - Update issue trackers

## Automation Details

### GitHub Actions Workflow

The `.github/workflows/publish.yml` workflow:

1. **Triggers:**
   - Automatically on GitHub Release creation
   - Manually via workflow_dispatch

2. **Steps:**
   - Checks out code
   - Extracts version from release tag
   - Updates pyproject.toml with version
   - Builds source and wheel distributions
   - Validates with twine
   - Publishes to PyPI (release) or Test PyPI (manual)
   - Attaches artifacts to GitHub release

3. **Required Secrets:**
   - `PYPI_API_TOKEN`
   - `TEST_PYPI_API_TOKEN`

### CI Workflow

The `.github/workflows/ci.yml` runs on all pushes/PRs:

1. **Lint:** Black, Ruff, MyPy
2. **Test:** Python 3.8-3.12
3. **Build:** Distribution validation
4. **Security:** Safety, Bandit scans

## Package Structure

```
pcib_detector/
├── pyproject.toml          # Package metadata and dependencies
├── README.md               # PyPI project description
├── LICENSE                 # License file (MIT)
├── src/
│   └── pcib_detector/
│       ├── __init__.py     # Version and exports
│       ├── core.py         # Main detector
│       └── ...
├── examples/               # Example scripts
└── tests/                  # Test suite (if added)
```

## PyPI Project Page

After publishing, your project appears at:
- https://pypi.org/project/pcib-detector/

The page shows:
- README content
- Installation instructions
- Version history
- Download statistics
- GitHub link
- License

## Maintenance Schedule

**Regular tasks:**

- **Weekly:** Check for security updates in dependencies
- **Monthly:** Review open issues and PRs
- **Quarterly:** Dependency updates and testing
- **Annually:** Major version planning

## Support

For publishing issues:

- **PyPI Help:** https://pypi.org/help/
- **Packaging Guide:** https://packaging.python.org/
- **GitHub Actions:** https://docs.github.com/actions

---

## Quick Reference

### Create Release (Recommended Process)

```bash
# 1. Update versions
vim pcib_detector/pyproject.toml
vim pcib_detector/src/pcib_detector/__init__.py
vim CHANGELOG.md

# 2. Commit and tag
git add .
git commit -m "Bump version to X.Y.Z"
git push

git tag -a vX.Y.Z -m "Release X.Y.Z"
git push origin vX.Y.Z

# 3. Create GitHub Release
# Go to GitHub and create release from tag

# 4. Wait for Actions to complete
# 5. Verify: pip install --upgrade pcib-detector
```

### Test on Test PyPI

```bash
# 1. Trigger manual workflow on GitHub
# Actions → Publish to PyPI → Run workflow

# 2. Test install
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    pcib-detector==X.Y.Z
```

### Emergency Hotfix

```bash
# 1. Fix bug
git commit -m "Hotfix: critical bug"

# 2. Bump patch version (e.g., 1.0.1 → 1.0.2)

# 3. Fast-track release
git tag -a v1.0.2 -m "Hotfix: critical bug"
git push origin v1.0.2

# 4. Create release immediately
```

---

**Last Updated:** January 2026  
**Maintainer:** @yourusername

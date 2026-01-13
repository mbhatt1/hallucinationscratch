# Security Policy

## Supported Versions

We currently support the following versions with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

## Reporting a Vulnerability

We take the security of PCIB Detector seriously. If you believe you have found a security vulnerability, please report it to us responsibly.

### How to Report

**Please DO NOT report security vulnerabilities through public GitHub issues.**

Instead, please email security@example.com (or your contact email) with:

1. **Description** of the vulnerability
2. **Steps to reproduce** the issue
3. **Potential impact** of the vulnerability
4. **Suggested fix** (if you have one)

### What to Expect

- **Acknowledgment**: We will acknowledge receipt within 48 hours
- **Assessment**: We will assess the vulnerability and determine severity
- **Updates**: We will keep you informed of our progress
- **Resolution**: We aim to resolve critical issues within 7 days
- **Credit**: We will credit you in the security advisory (unless you prefer to remain anonymous)

### Security Best Practices

When using PCIB Detector, follow these security guidelines:

#### 1. API Key Management

**✅ DO:**
- Store API keys in environment variables
- Use secrets management systems (AWS Secrets Manager, Azure Key Vault, etc.)
- Rotate API keys regularly
- Use separate keys for development and production
- Limit API key permissions to minimum required

**❌ DON'T:**
- Hardcode API keys in source code
- Commit API keys to version control
- Share API keys in public channels
- Use production keys in development

```python
# ✅ Good
import os
config = Config(api_key=os.getenv("OPENAI_API_KEY"))

# ❌ Bad
config = Config(api_key="sk-...actual-key...")
```

#### 2. Input Validation

Always validate user input before processing:

```python
# Validate answer and evidence lengths
MAX_LENGTH = 50000  # characters

def validate_input(answer: str, evidence: str) -> bool:
    if len(answer) > MAX_LENGTH or len(evidence) > MAX_LENGTH:
        raise ValueError("Input too long")
    return True

# Use in your code
validate_input(answer, evidence)
result = await detector.detect_hallucination(answer, evidence)
```

#### 3. Rate Limiting

Implement rate limiting to prevent abuse:

```python
from time import time
from collections import deque

class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window = window_seconds
        self.requests = deque()
    
    def allow_request(self) -> bool:
        now = time()
        # Remove old requests
        while self.requests and self.requests[0] < now - self.window:
            self.requests.popleft()
        
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True
        return False

# Usage
limiter = RateLimiter(max_requests=100, window_seconds=60)

if limiter.allow_request():
    result = await detector.detect_hallucination(...)
else:
    raise Exception("Rate limit exceeded")
```

#### 4. Logging and Monitoring

**✅ DO log:**
- Request IDs for traceability
- Error messages (sanitized)
- Performance metrics
- Rate limit hits

**❌ DON'T log:**
- API keys
- Personal information (PII)
- Full request/response payloads with sensitive data
- Stack traces in production logs

```python
import logging
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hash sensitive data
def hash_sensitive(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()[:16]

# Good logging
logger.info(f"Detection request: {request_id}, user: {hash_sensitive(user_id)}")
logger.error(f"Detection failed: {error_type} (request: {request_id})")

# Bad logging - DON'T DO THIS
# logger.info(f"API Key: {api_key}")  # ❌
# logger.info(f"User email: {email}")  # ❌
```

#### 5. Dependency Security

Keep dependencies up to date:

```bash
# Check for vulnerabilities
pip install safety
safety check

# Update dependencies
pip install --upgrade pcib-detector

# Audit dependencies regularly
pip list --outdated
```

#### 6. Network Security

**For production deployments:**

- Use HTTPS for all API calls (already enforced by providers)
- Implement certificate pinning if needed
- Use VPCs and private networks where possible
- Implement proper firewall rules
- Use API gateways for additional security

#### 7. Data Privacy

**When handling sensitive data:**

- Anonymize data before processing
- Implement data retention policies
- Comply with GDPR, CCPA, and other regulations
- Use encryption at rest and in transit
- Document data flows and processing

```python
import re

def anonymize_pii(text: str) -> str:
    """Remove common PII from text."""
    # Email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    # Phone numbers (simple)
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
    # Credit cards (simple)
    text = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '[CC]', text)
    return text

# Use before processing
answer = anonymize_pii(user_answer)
evidence = anonymize_pii(user_evidence)
```

### Known Security Considerations

#### 1. API Cost Control

The detector makes multiple API calls per detection. Implement cost controls:

```python
class CostTracker:
    def __init__(self, max_cost_usd: float):
        self.max_cost = max_cost_usd
        self.current_cost = 0.0
    
    def check_budget(self, estimated_cost: float) -> bool:
        if self.current_cost + estimated_cost > self.max_cost:
            raise Exception(f"Budget exceeded: ${self.current_cost:.2f}")
        self.current_cost += estimated_cost
        return True

# Usage
tracker = CostTracker(max_cost_usd=100.0)
```

#### 2. Prompt Injection

The detector itself uses LLMs, which could be vulnerable to prompt injection. We mitigate this by:

- Using structured outputs (JSON schema) where possible
- Validating all LLM outputs
- Using clear prompt templates
- Sanitizing user inputs

#### 3. Model Behavior

LLM outputs can be unpredictable. Always:

- Validate detection results
- Implement human review for high-stakes decisions
- Don't rely solely on automated detection
- Monitor for false positives/negatives

### Responsible Disclosure Timeline

1. **Day 0**: Vulnerability reported
2. **Day 2**: Acknowledged and assessment begins
3. **Day 7**: Fix developed for critical issues
4. **Day 14**: Fix released and advisory published
5. **Day 30**: Full details disclosed (after patch adoption)

### Security Updates

We will publish security advisories for:

- **Critical**: Immediate attention required
- **High**: Should be addressed quickly
- **Medium**: Should be addressed in normal update cycle
- **Low**: Informational

Subscribe to security advisories:
- GitHub Security Advisories: [Watch this repo]
- Email: security-announce@example.com

### Bug Bounty

We do not currently have a formal bug bounty program, but we appreciate responsible disclosure and will:

- Publicly thank security researchers (with permission)
- Provide detailed credit in security advisories
- Consider rewards for significant vulnerabilities (case by case)

### Compliance

PCIB Detector is designed to help with:

- **AI Safety**: Detecting hallucinations in critical systems
- **Regulatory Compliance**: Validating AI outputs
- **Audit Trails**: Detailed detection logs

However, users are responsible for:

- GDPR/CCPA compliance in their deployments
- Industry-specific regulations (healthcare, finance, etc.)
- Data retention and privacy policies

### Security Checklist for Production

Before deploying to production, ensure:

- [ ] API keys stored securely (environment variables/secrets manager)
- [ ] Rate limiting implemented
- [ ] Input validation in place
- [ ] Logging configured (no sensitive data)
- [ ] Monitoring and alerting set up
- [ ] Cost controls implemented
- [ ] Data anonymization for PII
- [ ] Regular security updates scheduled
- [ ] Incident response plan documented
- [ ] Backup and recovery procedures tested

### Contact

For security concerns:
- **Email**: security@example.com
- **PGP Key**: [Link to PGP key if available]
- **Response Time**: Within 48 hours

For general issues:
- **GitHub Issues**: https://github.com/yourusername/pcib-detector/issues

---

**Last Updated**: January 2026  
**Version**: 1.0

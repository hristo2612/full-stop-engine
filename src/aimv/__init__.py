"""AI music video pipeline."""

import os

# python.org Python on macOS ships without a CA bundle, so model downloads fail SSL checks.
# Point it at certifi's bundle unless the user already chose one.
try:
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except ImportError:  # pragma: no cover
    pass

# Ops MPS does not implement yet fall back to CPU instead of crashing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

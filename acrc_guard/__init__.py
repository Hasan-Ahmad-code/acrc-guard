"""ACRC-Guard: poisoning-robust, evidence-grounded Retrieval-Augmented Generation."""

try:  # load OPENROUTER_API_KEY etc. from a local .env file if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from .config import GuardConfig  # noqa: E402
from .pipeline import GuardedRAG, RAGResult, VanillaRAG  # noqa: E402

__all__ = ["GuardConfig", "GuardedRAG", "VanillaRAG", "RAGResult"]
__version__ = "0.1.0"

"""Configuration settings for MLflow MCP Server."""

import logging
import os
from typing import Optional

import mlflow


# Environment variable configuration
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
CACHE_TTL = int(os.environ.get("CACHE_TTL", "300"))  # 5 minutes default
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

# Initialize MLflow
mlflow.set_tracking_uri(uri=TRACKING_URI)

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger("mlflow-mcp-server")
logger.info(f"Using MLflow tracking server at: {TRACKING_URI}")

# Check for optional dependencies
try:
    from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
    HAS_TENACITY = True
    logger.info("Tenacity available - retry functionality enabled")
except ImportError:
    HAS_TENACITY = False
    logger.warning(
        "tenacity not installed - retry functionality disabled. Install with: pip install tenacity"
    )


def get_retry_decorator():
    """Get retry decorator if tenacity is available."""
    if HAS_TENACITY:
        return retry(
            retry=retry_if_exception_type((ConnectionError, TimeoutError)),
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=4, max=10),
        )
    else:
        # Return a no-op decorator if tenacity is not available
        def no_retry(func):
            return func
        return no_retry

"""Utility functions and decorators for MLflow MCP Server."""

import json
import logging
from functools import wraps
from typing import Any, Callable, Dict

from cachetools import TTLCache

from ..core import CACHE_TTL, get_retry_decorator
from ..core import MLflowConnectionError, MLflowServerError

logger = logging.getLogger(__name__)

# Global cache instance
cache = TTLCache(maxsize=128, ttl=CACHE_TTL)


def handle_mlflow_errors(func: Callable) -> Callable:
    """Decorator to handle MLflow errors and convert them to custom exceptions."""
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ConnectionError as e:
            logger.error(f"Connection error in {func.__name__}: {e}")
            raise MLflowConnectionError(
                f"Failed to connect to MLflow server in {func.__name__}",
                original_error=e,
            )
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            raise MLflowServerError(
                f"Unexpected error in {func.__name__}: {str(e)}",
                suggestion="Check MLflow server logs for more details",
                original_error=e,
            )
    
    return wrapper


def mcp_response(func: Callable) -> Callable:
    """Decorator to format function responses for MCP protocol."""
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return {
                "success": True,
                "data": result,
                "timestamp": str(logger.handlers[0].formatter.formatTime() if logger.handlers else ""),
            }
        except Exception as e:
            error_response = {
                "success": False,
                "error": {
                    "type": type(e).__name__,
                    "message": str(e),
                    "timestamp": str(logger.handlers[0].formatter.formatTime() if logger.handlers else ""),
                }
            }
            
            # Add suggestion if available
            if hasattr(e, "suggestion") and e.suggestion:
                error_response["error"]["suggestion"] = e.suggestion
            
            return error_response
    
    return wrapper


def cached(ttl: int = CACHE_TTL) -> Callable:
    """Decorator to cache function results with TTL."""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Try to get from cache
            if cache_key in cache:
                logger.debug(f"Cache hit for {func.__name__}")
                return cache[cache_key]
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache[cache_key] = result
            logger.debug(f"Cached result for {func.__name__}")
            
            return result
        
        return wrapper
    
    return decorator


def safe_mlflow_call(func: Callable, *args, **kwargs) -> Any:
    """Safely execute MLflow API calls with error handling and retrying."""
    retry_decorator = get_retry_decorator()
    
    @retry_decorator
    @handle_mlflow_errors
    def _safe_call():
        return func(*args, **kwargs)
    
    return _safe_call()


def format_json_response(data: Any) -> str:
    """Format data as JSON string for responses."""
    try:
        return json.dumps(data, indent=2, default=str)
    except (TypeError, ValueError) as e:
        logger.warning(f"Failed to serialize data to JSON: {e}")
        return str(data)

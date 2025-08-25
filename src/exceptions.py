"""Custom exceptions for MLflow MCP Server."""

from typing import Optional


class MLflowServerError(Exception):
    """Base exception for MLflow server errors."""
    
    def __init__(
        self,
        message: str,
        suggestion: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.suggestion = suggestion
        self.original_error = original_error


class MLflowConnectionError(MLflowServerError):
    """Raised when connection to MLflow server fails."""
    
    def __init__(
        self,
        message: str = "Failed to connect to MLflow server",
        original_error: Optional[Exception] = None,
    ):
        suggestion = (
            "Check if MLflow server is running and accessible. "
            "Verify MLFLOW_TRACKING_URI environment variable."
        )
        super().__init__(message, suggestion, original_error)


class MLflowResourceNotFoundError(MLflowServerError):
    """Raised when a requested resource is not found."""
    
    def __init__(
        self,
        resource_type: str,
        resource_id: str,
        suggestion: Optional[str] = None,
    ):
        message = f"{resource_type} '{resource_id}' not found"
        if not suggestion:
            suggestion = f"Verify that the {resource_type.lower()} ID is correct and exists in MLflow"
        super().__init__(message, suggestion)


class MLflowValidationError(MLflowServerError):
    """Raised when input validation fails."""
    
    def __init__(
        self,
        message: str,
        query_details: Optional[str] = None,
        suggestion: Optional[str] = None,
    ):
        if query_details:
            message = f"{message}. Query details: {query_details}"
        if not suggestion:
            suggestion = "Check the input parameters and ensure they meet the expected format"
        super().__init__(message, suggestion)

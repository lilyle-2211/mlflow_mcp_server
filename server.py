#!/usr/bin/env python3
"""
Enhanced MLflow MCP Server for Claude Code

This is a specialized, production-ready MLflow MCP Server designed for Claude Code integration.
It provides comprehensive functionality for:

🔧 Core Features:
1. Enhanced error handling with detailed suggestions and context
2. Retry logic with exponential backoff for transient failures
3. Pydantic data models for type safety and validation
4. TTL caching for performance optimization
5. Claude Code optimized responses and instructions

🚀 Advanced Features:
6. Health check monitoring for system status
7. Multi-run comparison with statistical analysis
8. Bulk artifact downloads with pattern matching
9. Smart error messages with troubleshooting suggestions
10. Robust file system handling for read-only environments

🛠 Tools Available:
Core Operations:
- list_models(), get_model_details(): Model registry management
- list_experiments(), search_runs(): Experiment exploration
- download_artifact(), read_artifact_content(): Artifact handling

Advanced Analytics:
- health_check(): System health and connectivity monitoring
- compare_runs(): Side-by-side analysis of multiple ML runs
- get_experiment_leaderboard(): Ranked performance analysis
- analyze_hyperparameter_impact(): Statistical parameter analysis
- export_experiment_report(): Generate comprehensive reports

Model Lifecycle:
- promote_model(): Move models through stages (Staging→Production)
- create_model_version(): Register new model versions from runs
- get_model_serving_info(): Deployment readiness and serving options
- bulk_download_artifacts(): Download multiple artifacts with patterns

Environment variables:
    MLFLOW_TRACKING_URI: URI of the MLflow tracking server (default: http://localhost:5000)
    LOG_LEVEL: Logging level - DEBUG, INFO, WARNING, ERROR (default: INFO)
    CACHE_TTL: Cache time-to-live in seconds (default: 300)

Dependencies:
    Required: mlflow, mcp, pydantic, cachetools
    Optional: tenacity (for retry functionality)

Usage:
    python mlflow_server_claude_code.py

Version: 2.0 - Enhanced Edition
"""

import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from typing import Any, Dict, List, Optional, Union

import mlflow
from cachetools import TTLCache
from mlflow import MlflowClient
from pydantic import BaseModel, ConfigDict, Field

# Add retry functionality
try:
    from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

    HAS_TENACITY = True
except ImportError:
    # Fallback if tenacity is not installed
    HAS_TENACITY = False

# Set up logging
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("mlflow-mcp-server")

# Check tenacity availability
if not HAS_TENACITY:
    logger.warning(
        "tenacity not installed - retry functionality disabled. Install with: pip install tenacity"
    )

# Configuration
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
CACHE_TTL = int(os.environ.get("CACHE_TTL", "300"))  # 5 minutes default

# Initialize MLflow
mlflow.set_tracking_uri(uri=TRACKING_URI)
logger.info(f"Using MLflow tracking server at: {TRACKING_URI}")

# Initialize cache
cache = TTLCache(maxsize=100, ttl=CACHE_TTL)

# Get MLflow client
client = MlflowClient()

try:
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP(
        name="mlflow-claude-code",
        instructions="""
        I am an MLflow MCP server specifically designed for Claude Code integration.
        I help you interact with your MLflow tracking server to manage machine learning
        experiments, models, and artifacts efficiently.

        Key capabilities:
        - List and search experiments and runs
        - Get detailed information about models, experiments, and runs
        - Download and read artifact contents
        - Search runs with complex filters
        - Process natural language queries about your ML data
        - Provide system information and health checks

        I'm optimized for Claude Code workflows with enhanced error handling,
        caching, and detailed responses perfect for development environments.
        """,
    )
except ImportError:
    logger.error("Failed to import MCP. Please install with: pip install mcp[cli]")
    sys.exit(1)


# ========== Enhanced Custom Exceptions ==========


class MLflowServerError(Exception):
    """Base exception for MLflow server errors"""

    def __init__(self, message: str, suggestion: str = None, original_error: Exception = None):
        self.message = message
        self.suggestion = suggestion
        self.original_error = original_error

        full_message = message
        if suggestion:
            full_message += f"\n💡 Suggestion: {suggestion}"
        if original_error:
            full_message += f"\n🔍 Original error: {str(original_error)}"

        super().__init__(full_message)


class MLflowConnectionError(MLflowServerError):
    """Raised when unable to connect to MLflow server"""

    def __init__(self, uri: str, original_error: Exception = None):
        suggestion = f"Check if MLflow server is running at {uri}. Try: curl {uri}"
        super().__init__(
            f"Cannot connect to MLflow server at {uri}",
            suggestion=suggestion,
            original_error=original_error,
        )


class MLflowResourceNotFoundError(MLflowServerError):
    """Raised when a requested resource is not found"""

    def __init__(self, resource_type: str, resource_id: str, suggestion: str = None):
        default_suggestion = (
            f"Verify the {resource_type} ID '{resource_id}' exists and is accessible"
        )
        super().__init__(
            f"{resource_type.title()} '{resource_id}' not found",
            suggestion=suggestion or default_suggestion,
        )


class MLflowValidationError(MLflowServerError):
    """Raised when input validation fails"""

    def __init__(self, message: str, suggestion: str = None):
        default_suggestion = "Check the parameter format and try again"
        super().__init__(message, suggestion=suggestion or default_suggestion)


class MLflowQueryError(MLflowServerError):
    """Raised when a query fails"""

    def __init__(self, message: str, query_details: str = None, suggestion: str = None):
        if query_details:
            message += f"\nQuery details: {query_details}"
        super().__init__(message, suggestion=suggestion)


# ========== Pydantic Data Models ==========


class RunInfo(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    run_id: str
    run_name: Optional[str] = None
    experiment_id: str
    status: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    artifact_uri: Optional[str] = None
    lifecycle_stage: Optional[str] = None
    user_id: Optional[str] = None


class ModelVersionInfo(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    version: str
    status: str
    current_stage: Optional[str] = None
    creation_timestamp: Optional[str] = None
    source: Optional[str] = None
    run_id: Optional[str] = None
    run: Optional[Dict[str, Any]] = None


class RegisteredModelInfo(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    name: str
    creation_timestamp: Optional[str] = None
    last_updated_timestamp: Optional[str] = None
    description: Optional[str] = None
    tags: Dict[str, str] = Field(default_factory=dict)
    latest_versions: List[ModelVersionInfo] = Field(default_factory=list)


class ExperimentInfo(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    experiment_id: str
    name: str
    artifact_location: Optional[str] = None
    lifecycle_stage: Optional[str] = None
    creation_time: Optional[str] = None
    tags: Dict[str, str] = Field(default_factory=dict)
    run_count: Union[int, str] = 0


class ArtifactInfo(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    path: str
    is_dir: bool
    file_size: Optional[int] = None


class SystemInfo(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    mlflow_version: str
    tracking_uri: str
    registry_uri: str
    artifact_uri: str
    python_version: str
    server_time: str
    experiment_count: Union[int, str]
    model_count: Union[int, str]
    active_runs: Union[int, str]


# ========== Utility Functions (Deduplication) ==========


class MLflowUtils:
    """Utility class for common MLflow operations"""

    @staticmethod
    def format_timestamp(timestamp_ms: Optional[int]) -> str:
        """Convert a millisecond timestamp to a human-readable string."""
        if not timestamp_ms:
            return "N/A"
        try:
            dt = datetime.fromtimestamp(timestamp_ms / 1000.0)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError) as e:
            logger.warning(f"Error formatting timestamp {timestamp_ms}: {e}")
            return "Invalid timestamp"

    @staticmethod
    def resolve_run_id(run_identifier: str) -> str:
        """
        Resolve run identifier to actual run ID.
        Handles both run IDs and run names.

        Args:
            run_identifier: Either a run ID or run name

        Returns:
            The actual run ID

        Raises:
            MLflowResourceNotFoundError: If run not found
        """
        cache_key = f"run_id_{run_identifier}"
        if cache_key in cache:
            return cache[cache_key]

        try:
            # First try to get by run_id directly
            run = client.get_run(run_identifier)
            cache[cache_key] = run_identifier
            return run_identifier
        except Exception:
            # If that fails, search for run by name across all experiments
            logger.debug(f"Run ID not found, searching by name: {run_identifier}")

            try:
                all_experiments = client.search_experiments()
                exp_ids = [exp.experiment_id for exp in all_experiments]

                # Search for runs with matching name
                runs = client.search_runs(experiment_ids=exp_ids, max_results=1000)
                matching_runs = [r for r in runs if r.info.run_name == run_identifier]

                if matching_runs:
                    actual_run_id = matching_runs[0].info.run_id
                    cache[cache_key] = actual_run_id
                    return actual_run_id
                else:
                    raise MLflowResourceNotFoundError(
                        "run", run_identifier, "Verify the run ID or name exists in MLflow"
                    )

            except Exception as e:
                if isinstance(e, MLflowResourceNotFoundError):
                    raise
                raise MLflowQueryError(f"Error searching for run: {str(e)}")

    @staticmethod
    def safe_get_run_count(experiment_id: str) -> Union[int, str]:
        """Safely get run count for an experiment"""
        try:
            runs = client.search_runs(experiment_ids=[experiment_id], max_results=1000)
            return len(runs)
        except Exception as e:
            logger.warning(f"Error getting run count for experiment {experiment_id}: {str(e)}")
            return "Error getting count"

    @staticmethod
    def extract_metrics_safely(metrics_dict: Dict[str, Any]) -> Dict[str, Union[float, str]]:
        """Safely extract and convert metrics to appropriate types"""
        safe_metrics = {}
        for k, v in metrics_dict.items():
            try:
                safe_metrics[k] = round(float(v), 4)
            except (ValueError, TypeError):
                safe_metrics[k] = str(v)
        return safe_metrics


# ========== Error Handling Decorator ==========


def handle_mlflow_errors(func):
    """Decorator to handle MLflow errors consistently"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except MLflowServerError:
            # Re-raise our custom exceptions
            raise
        except ConnectionError as e:
            logger.error(f"MLflow connection error in {func.__name__}: {str(e)}")
            raise MLflowConnectionError(f"Failed to connect to MLflow server: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}", exc_info=True)
            raise MLflowQueryError(f"Error in {func.__name__}: {str(e)}")

    return wrapper


def mcp_response(func):
    """Decorator to handle responses for MCP with error handling"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            if isinstance(result, BaseModel):
                return result.model_dump()
            return result
        except MLflowServerError as e:
            error_response = {"error": str(e), "error_type": e.__class__.__name__}
            return error_response
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}", exc_info=True)
            error_response = {
                "error": f"Internal server error: {str(e)}",
                "error_type": "InternalError",
            }
            return error_response

    return wrapper


# ========== Caching Decorator ==========


def cached(ttl: int = CACHE_TTL):
    """Decorator to cache function results"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}_{hash(str(args) + str(sorted(kwargs.items())))}"

            if cache_key in cache:
                logger.debug(f"Cache hit for {func.__name__}")
                return cache[cache_key]

            result = func(*args, **kwargs)
            cache[cache_key] = result
            logger.debug(f"Cached result for {func.__name__}")
            return result

        return wrapper

    return decorator


# ========== Retry and Utility Enhancements ==========


def safe_mlflow_call(func, *args, **kwargs):
    """Execute MLflow calls with retry logic if tenacity is available"""
    if HAS_TENACITY:

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        )
        def _retry_call():
            return func(*args, **kwargs)

        try:
            return _retry_call()
        except Exception as e:
            logger.error(f"MLflow call failed after retries: {func.__name__}")
            raise MLflowConnectionError(TRACKING_URI, original_error=e)
    else:
        # Fallback without retry
        try:
            return func(*args, **kwargs)
        except (ConnectionError, TimeoutError, OSError) as e:
            raise MLflowConnectionError(TRACKING_URI, original_error=e)


# ========== Enhanced MCP Tools ==========


@mcp.tool()
@mcp_response
def health_check() -> Dict[str, Any]:
    """
    Check MLflow server connectivity and MCP server health.

    Returns:
        A dictionary containing health status and system information.
    """
    logger.info("Performing health check")

    health_status = {
        "mcp_server": "healthy",
        "mlflow_server": "unknown",
        "tracking_uri": TRACKING_URI,
        "cache_info": {"size": len(cache), "max_size": cache.maxsize, "ttl": CACHE_TTL},
        "timestamp": datetime.now().isoformat(),
        "checks": [],
    }

    # Check MLflow server connectivity
    try:
        start_time = time.time()
        experiments = safe_mlflow_call(client.search_experiments)
        response_time = (time.time() - start_time) * 1000  # Convert to ms

        health_status["mlflow_server"] = "healthy"
        health_status["response_time_ms"] = round(response_time, 2)
        health_status["total_experiments"] = len(experiments)
        health_status["checks"].append(
            {
                "name": "mlflow_connectivity",
                "status": "pass",
                "details": f"Connected successfully in {response_time:.2f}ms",
            }
        )
    except Exception as e:
        health_status["mlflow_server"] = "unhealthy"
        health_status["error"] = str(e)
        health_status["checks"].append(
            {"name": "mlflow_connectivity", "status": "fail", "details": str(e)}
        )

    # Check cache health
    if len(cache) > cache.maxsize * 0.8:
        health_status["checks"].append(
            {
                "name": "cache_usage",
                "status": "warn",
                "details": f"Cache usage high: {len(cache)}/{cache.maxsize}",
            }
        )
    else:
        health_status["checks"].append(
            {
                "name": "cache_usage",
                "status": "pass",
                "details": f"Cache usage normal: {len(cache)}/{cache.maxsize}",
            }
        )

    return health_status


@mcp.tool()
@mcp_response
@handle_mlflow_errors
def compare_runs(run_ids: List[str]) -> Dict[str, Any]:
    """
    Compare metrics, parameters, and performance across multiple runs.

    Args:
        run_ids: List of run IDs or names to compare

    Returns:
        A dictionary containing detailed comparison across runs.
    """
    if len(run_ids) < 2:
        raise MLflowValidationError("At least 2 run IDs required for comparison")

    logger.info(f"Comparing {len(run_ids)} runs")

    runs_data = []
    all_metrics = set()
    all_params = set()

    # Collect data from all runs
    for run_id in run_ids:
        actual_run_id = MLflowUtils.resolve_run_id(run_id)
        run = safe_mlflow_call(client.get_run, actual_run_id)

        run_data = {
            "run_id": actual_run_id,
            "run_name": run.info.run_name or actual_run_id[:8],
            "status": run.info.status,
            "start_time": MLflowUtils.format_timestamp(run.info.start_time),
            "end_time": MLflowUtils.format_timestamp(run.info.end_time),
            "metrics": run.data.metrics,
            "params": run.data.params,
            "tags": run.data.tags,
        }

        runs_data.append(run_data)
        all_metrics.update(run.data.metrics.keys())
        all_params.update(run.data.params.keys())

    # Create comparison matrices
    metrics_comparison = {}
    for metric in all_metrics:
        metrics_comparison[metric] = {}
        values = []
        for run_data in runs_data:
            value = run_data["metrics"].get(metric, None)
            metrics_comparison[metric][run_data["run_name"]] = value
            if value is not None:
                values.append(value)

        # Add statistical info
        if values:
            metrics_comparison[metric]["_stats"] = {
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "range": max(values) - min(values) if len(values) > 1 else 0,
            }

    params_comparison = {}
    for param in all_params:
        params_comparison[param] = {}
        for run_data in runs_data:
            params_comparison[param][run_data["run_name"]] = run_data["params"].get(param, None)

    # Find best performing run for each metric
    best_runs = {}
    for metric, values in metrics_comparison.items():
        if "_stats" in values:
            best_value = values["_stats"]["max"]  # Assume higher is better
            for run_name, value in values.items():
                if run_name != "_stats" and value == best_value:
                    best_runs[metric] = run_name
                    break

    return {
        "comparison_summary": {
            "total_runs": len(run_ids),
            "metrics_compared": len(all_metrics),
            "params_compared": len(all_params),
        },
        "runs": runs_data,
        "metrics_comparison": metrics_comparison,
        "params_comparison": params_comparison,
        "best_performers": best_runs,
        "recommendations": [
            f"Consider parameters from run '{best_runs.get(list(best_runs.keys())[0], 'N/A')}' for best performance"
            if best_runs
            else "No clear best performer found"
        ],
    }


@mcp.tool()
@mcp_response
@handle_mlflow_errors
def bulk_download_artifacts(
    run_id: str, artifact_patterns: List[str], download_path: str = "downloads"
) -> Dict[str, Any]:
    """
    Download multiple artifacts matching patterns in one operation.

    Args:
        run_id: The ID or name of the MLflow run
        artifact_patterns: List of artifact path patterns (supports wildcards)
        download_path: Local path to download to (default: "downloads")

    Returns:
        A dictionary containing download results for each artifact.
    """
    if not run_id.strip():
        raise MLflowValidationError("run_id cannot be empty")
    if not artifact_patterns:
        raise MLflowValidationError("artifact_patterns cannot be empty")

    logger.info(f"Bulk downloading {len(artifact_patterns)} artifact patterns from run {run_id}")

    # Resolve run ID
    actual_run_id = MLflowUtils.resolve_run_id(run_id)

    # Get all artifacts for this run
    try:
        all_artifacts = safe_mlflow_call(client.list_artifacts, actual_run_id)
    except Exception as e:
        raise MLflowQueryError(
            f"Failed to list artifacts for run {run_id}", suggestion="Verify the run ID exists"
        )

    # Match patterns to actual artifact paths
    import fnmatch

    matched_artifacts = []

    def _match_artifacts(artifacts, pattern_list):
        """Recursively match artifact patterns"""
        matches = []
        for artifact in artifacts:
            for pattern in pattern_list:
                if fnmatch.fnmatch(artifact.path, pattern):
                    matches.append(artifact.path)
                    break
        return matches

    matched_artifacts = _match_artifacts(all_artifacts, artifact_patterns)

    if not matched_artifacts:
        return {
            "run_id": actual_run_id,
            "patterns": artifact_patterns,
            "matched_count": 0,
            "downloads": [],
            "message": "No artifacts matched the specified patterns",
        }

    # Use absolute path to project directory
    if not os.path.isabs(download_path):
        project_dir = os.path.dirname(os.path.abspath(__file__))
        download_path = os.path.join(project_dir, download_path)

    os.makedirs(download_path, exist_ok=True)

    # Download each matched artifact
    download_results = []
    successful_downloads = 0

    for artifact_path in matched_artifacts:
        try:
            downloaded_path = safe_mlflow_call(
                client.download_artifacts, actual_run_id, artifact_path, download_path
            )
            download_results.append(
                {
                    "artifact_path": artifact_path,
                    "status": "success",
                    "downloaded_to": downloaded_path,
                }
            )
            successful_downloads += 1
        except Exception as e:
            download_results.append(
                {"artifact_path": artifact_path, "status": "failed", "error": str(e)}
            )

    return {
        "run_id": actual_run_id,
        "original_run_input": run_id,
        "patterns": artifact_patterns,
        "matched_count": len(matched_artifacts),
        "successful_downloads": successful_downloads,
        "failed_downloads": len(matched_artifacts) - successful_downloads,
        "downloads": download_results,
    }


# ========== Model Lifecycle Management ==========


@mcp.tool()
@mcp_response
@handle_mlflow_errors
def promote_model(model_name: str, version: str, stage: str) -> Dict[str, Any]:
    """
    Promote model version to a specific stage (Production/Staging/Archived).

    Args:
        model_name: Name of the registered model
        version: Version number to promote
        stage: Target stage ("Production", "Staging", "Archived", "None")

    Returns:
        A dictionary containing promotion details and status.
    """
    if not model_name.strip():
        raise MLflowValidationError("model_name cannot be empty")
    if not version.strip():
        raise MLflowValidationError("version cannot be empty")

    valid_stages = ["Production", "Staging", "Archived", "None"]
    if stage not in valid_stages:
        raise MLflowValidationError(
            f"stage must be one of: {valid_stages}",
            suggestion=f"Use one of: {', '.join(valid_stages)}",
        )

    logger.info(f"Promoting model {model_name} version {version} to {stage}")

    try:
        # Get current model version info
        model_version = safe_mlflow_call(client.get_model_version, model_name, version)

        previous_stage = model_version.current_stage

        # Transition model version to new stage
        model_version = safe_mlflow_call(
            client.transition_model_version_stage, name=model_name, version=version, stage=stage
        )

        return {
            "model_name": model_name,
            "version": version,
            "previous_stage": previous_stage,
            "new_stage": stage,
            "transition_time": datetime.now().isoformat(),
            "status": "success",
            "message": f"Successfully promoted {model_name} v{version} from {previous_stage} to {stage}",
        }

    except Exception as e:
        raise MLflowResourceNotFoundError(
            "model_version",
            f"{model_name}:{version}",
            "Verify the model and version exist in the registry",
        )


@mcp.tool()
@mcp_response
@handle_mlflow_errors
def create_model_version(model_name: str, source: str, run_id: str) -> Dict[str, Any]:
    """
    Create new model version from a run.

    Args:
        model_name: Name of the registered model
        source: Source path of the model (usually artifact path)
        run_id: ID of the run containing the model

    Returns:
        A dictionary containing new model version details.
    """
    if not model_name.strip():
        raise MLflowValidationError("model_name cannot be empty")
    if not source.strip():
        raise MLflowValidationError("source cannot be empty")
    if not run_id.strip():
        raise MLflowValidationError("run_id cannot be empty")

    logger.info(f"Creating model version for {model_name} from run {run_id}")

    # Resolve run ID
    actual_run_id = MLflowUtils.resolve_run_id(run_id)

    try:
        # Verify run exists and get run info
        run = safe_mlflow_call(client.get_run, actual_run_id)

        # Create model version
        model_version = safe_mlflow_call(
            client.create_model_version, name=model_name, source=source, run_id=actual_run_id
        )

        return {
            "model_name": model_name,
            "version": model_version.version,
            "source": source,
            "run_id": actual_run_id,
            "creation_time": MLflowUtils.format_timestamp(model_version.creation_timestamp),
            "status": model_version.status,
            "current_stage": model_version.current_stage,
            "run_info": {
                "status": run.info.status,
                "start_time": MLflowUtils.format_timestamp(run.info.start_time),
                "metrics": dict(list(run.data.metrics.items())[:5]),  # Top 5 metrics
            },
            "message": f"Successfully created version {model_version.version} for model {model_name}",
        }

    except Exception as e:
        if "already exists" in str(e).lower():
            raise MLflowValidationError(
                f"Model version already exists for this source",
                suggestion="Use a different source path or check existing versions",
            )
        raise MLflowQueryError(f"Failed to create model version: {str(e)}")


@mcp.tool()
@mcp_response
@handle_mlflow_errors
def get_experiment_leaderboard(experiment_id: str, metric: str) -> Dict[str, Any]:
    """
    Get ranked leaderboard of runs by metric performance.

    Args:
        experiment_id: ID of the experiment
        metric: Metric name to rank by

    Returns:
        A dictionary containing ranked runs and statistical summary.
    """
    if not experiment_id.strip():
        raise MLflowValidationError("experiment_id cannot be empty")
    if not metric.strip():
        raise MLflowValidationError("metric cannot be empty")

    logger.info(f"Generating leaderboard for experiment {experiment_id} by {metric}")

    try:
        # Get all runs for the experiment
        runs = safe_mlflow_call(
            client.search_runs, experiment_ids=[experiment_id], max_results=1000
        )

        if not runs:
            return {
                "experiment_id": experiment_id,
                "metric": metric,
                "total_runs": 0,
                "leaderboard": [],
                "message": "No runs found in this experiment",
            }

        # Filter runs that have the specified metric
        runs_with_metric = []
        for run in runs:
            if metric in run.data.metrics:
                runs_with_metric.append(
                    {
                        "run_id": run.info.run_id,
                        "run_name": run.info.run_name or run.info.run_id[:8],
                        "status": run.info.status,
                        "start_time": MLflowUtils.format_timestamp(run.info.start_time),
                        "metric_value": run.data.metrics[metric],
                        "params": dict(list(run.data.params.items())[:5]),  # Top 5 params
                        "other_metrics": {
                            k: v
                            for k, v in run.data.metrics.items()
                            if k != metric
                            and len({k: v for k, v in run.data.metrics.items() if k != metric}) < 3
                        },
                    }
                )

        if not runs_with_metric:
            return {
                "experiment_id": experiment_id,
                "metric": metric,
                "total_runs": len(runs),
                "runs_with_metric": 0,
                "leaderboard": [],
                "message": f"No runs found with metric '{metric}'",
            }

        # Sort by metric value (descending - assume higher is better)
        leaderboard = sorted(runs_with_metric, key=lambda x: x["metric_value"], reverse=True)

        # Add rankings
        for i, run in enumerate(leaderboard):
            run["rank"] = i + 1
            run["percentile"] = round((len(leaderboard) - i) / len(leaderboard) * 100, 1)

        # Calculate statistics
        metric_values = [run["metric_value"] for run in runs_with_metric]
        stats = {
            "min": min(metric_values),
            "max": max(metric_values),
            "mean": sum(metric_values) / len(metric_values),
            "median": sorted(metric_values)[len(metric_values) // 2],
            "std": (
                sum((x - sum(metric_values) / len(metric_values)) ** 2 for x in metric_values)
                / len(metric_values)
            )
            ** 0.5,
            "range": max(metric_values) - min(metric_values),
        }

        return {
            "experiment_id": experiment_id,
            "metric": metric,
            "total_runs": len(runs),
            "runs_with_metric": len(runs_with_metric),
            "leaderboard": leaderboard[:20],  # Top 20
            "statistics": stats,
            "insights": [
                f"Best performing run: {leaderboard[0]['run_name']} ({leaderboard[0]['metric_value']:.4f})"
                if leaderboard
                else "No runs found",
                f"Performance range: {stats['range']:.4f} ({metric})",
                f"Average performance: {stats['mean']:.4f} ± {stats['std']:.4f}",
            ],
        }

    except Exception as e:
        raise MLflowQueryError(
            f"Failed to generate leaderboard: {str(e)}",
            suggestion="Verify experiment ID and metric name exist",
        )


@mcp.tool()
@mcp_response
@handle_mlflow_errors
def analyze_hyperparameter_impact(experiment_id: str) -> Dict[str, Any]:
    """
    Analyze the statistical impact of hyperparameters on metrics.

    Args:
        experiment_id: ID of the experiment to analyze

    Returns:
        A dictionary containing hyperparameter impact analysis.
    """
    if not experiment_id.strip():
        raise MLflowValidationError("experiment_id cannot be empty")

    logger.info(f"Analyzing hyperparameter impact for experiment {experiment_id}")

    try:
        # Get all runs for the experiment
        runs = safe_mlflow_call(
            client.search_runs, experiment_ids=[experiment_id], max_results=1000
        )

        if not runs or len(runs) < 3:
            return {
                "experiment_id": experiment_id,
                "total_runs": len(runs) if runs else 0,
                "analysis": {},
                "message": "Need at least 3 runs for meaningful hyperparameter analysis",
            }

        # Extract parameters and metrics
        all_params = set()
        all_metrics = set()
        run_data = []

        for run in runs:
            if run.info.status == "FINISHED":
                run_info = {
                    "run_id": run.info.run_id,
                    "params": run.data.params,
                    "metrics": run.data.metrics,
                }
                run_data.append(run_info)
                all_params.update(run.data.params.keys())
                all_metrics.update(run.data.metrics.keys())

        if len(run_data) < 3:
            return {
                "experiment_id": experiment_id,
                "total_runs": len(runs),
                "finished_runs": len(run_data),
                "analysis": {},
                "message": "Need at least 3 finished runs for analysis",
            }

        # Analyze parameter impact on each metric
        impact_analysis = {}

        for metric in all_metrics:
            metric_analysis = {"parameter_impacts": {}, "sample_size": 0}

            # Get runs that have this metric
            runs_with_metric = [run for run in run_data if metric in run["metrics"]]
            if len(runs_with_metric) < 3:
                continue

            metric_analysis["sample_size"] = len(runs_with_metric)

            # Analyze each parameter's impact
            for param in all_params:
                param_values = {}

                # Group runs by parameter value
                for run in runs_with_metric:
                    if param in run["params"]:
                        param_val = run["params"][param]
                        if param_val not in param_values:
                            param_values[param_val] = []
                        param_values[param_val].append(run["metrics"][metric])

                # Calculate statistics for each parameter value
                if len(param_values) > 1:  # Need variation in parameter
                    param_stats = {}
                    for val, metrics in param_values.items():
                        if len(metrics) >= 2:  # Need at least 2 samples
                            param_stats[val] = {
                                "count": len(metrics),
                                "mean": sum(metrics) / len(metrics),
                                "min": min(metrics),
                                "max": max(metrics),
                            }

                    if len(param_stats) > 1:
                        # Calculate impact score (range of means)
                        means = [stats["mean"] for stats in param_stats.values()]
                        impact_score = max(means) - min(means)

                        metric_analysis["parameter_impacts"][param] = {
                            "impact_score": impact_score,
                            "parameter_stats": param_stats,
                            "best_value": max(
                                param_stats.keys(), key=lambda k: param_stats[k]["mean"]
                            ),
                            "worst_value": min(
                                param_stats.keys(), key=lambda k: param_stats[k]["mean"]
                            ),
                        }

            # Rank parameters by impact
            if metric_analysis["parameter_impacts"]:
                sorted_params = sorted(
                    metric_analysis["parameter_impacts"].items(),
                    key=lambda x: x[1]["impact_score"],
                    reverse=True,
                )
                metric_analysis["ranked_parameters"] = [
                    {
                        "parameter": param,
                        "impact_score": data["impact_score"],
                        "best_value": data["best_value"],
                        "improvement_potential": data["impact_score"],
                    }
                    for param, data in sorted_params[:10]  # Top 10
                ]

            impact_analysis[metric] = metric_analysis

        # Generate insights
        insights = []
        for metric, analysis in impact_analysis.items():
            if "ranked_parameters" in analysis and analysis["ranked_parameters"]:
                top_param = analysis["ranked_parameters"][0]
                insights.append(
                    f"{metric}: '{top_param['parameter']}' has highest impact "
                    f"(improvement potential: {top_param['improvement_potential']:.4f})"
                )

        return {
            "experiment_id": experiment_id,
            "total_runs": len(runs),
            "analyzed_runs": len(run_data),
            "parameters_analyzed": len(all_params),
            "metrics_analyzed": len(all_metrics),
            "impact_analysis": impact_analysis,
            "key_insights": insights[:5],  # Top 5 insights
            "recommendations": [
                f"Focus on optimizing '{insight.split(':')[1].split(chr(39))[1]}' parameter for better {insight.split(':')[0]} performance"
                for insight in insights[:3]
            ],
        }

    except Exception as e:
        raise MLflowQueryError(f"Failed to analyze hyperparameter impact: {str(e)}")


@mcp.tool()
@mcp_response
@handle_mlflow_errors
def export_experiment_report(experiment_id: str, format: str = "html") -> Dict[str, Any]:
    """
    Generate comprehensive experiment report in various formats.

    Args:
        experiment_id: ID of the experiment
        format: Report format ("html", "json", "markdown")

    Returns:
        A dictionary containing report content and metadata.
    """
    if not experiment_id.strip():
        raise MLflowValidationError("experiment_id cannot be empty")

    valid_formats = ["html", "json", "markdown"]
    if format not in valid_formats:
        raise MLflowValidationError(f"format must be one of: {valid_formats}")

    logger.info(f"Generating {format} report for experiment {experiment_id}")

    try:
        # Get experiment info
        experiment = safe_mlflow_call(client.get_experiment, experiment_id)
        runs = safe_mlflow_call(
            client.search_runs, experiment_ids=[experiment_id], max_results=1000
        )

        # Collect experiment statistics
        if not runs:
            return {
                "experiment_id": experiment_id,
                "format": format,
                "report_content": "No runs found in this experiment",
                "metadata": {"total_runs": 0, "generation_time": datetime.now().isoformat()},
            }

        # Calculate summary statistics
        finished_runs = [run for run in runs if run.info.status == "FINISHED"]
        failed_runs = [run for run in runs if run.info.status == "FAILED"]

        all_metrics = set()
        all_params = set()
        for run in finished_runs:
            all_metrics.update(run.data.metrics.keys())
            all_params.update(run.data.params.keys())

        # Get top performers for main metrics
        top_performers = {}
        for metric in list(all_metrics)[:5]:  # Top 5 metrics
            runs_with_metric = [run for run in finished_runs if metric in run.data.metrics]
            if runs_with_metric:
                best_run = max(runs_with_metric, key=lambda r: r.data.metrics[metric])
                top_performers[metric] = {
                    "run_name": best_run.info.run_name or best_run.info.run_id[:8],
                    "value": best_run.data.metrics[metric],
                    "params": dict(list(best_run.data.params.items())[:3]),  # Top 3 params
                }

        # Generate report based on format
        report_data = {
            "experiment_info": {
                "name": experiment.name,
                "experiment_id": experiment_id,
                "creation_time": MLflowUtils.format_timestamp(experiment.creation_timestamp)
                if hasattr(experiment, "creation_timestamp")
                else None,
                "artifact_location": experiment.artifact_location,
            },
            "summary_stats": {
                "total_runs": len(runs),
                "finished_runs": len(finished_runs),
                "failed_runs": len(failed_runs),
                "success_rate": f"{len(finished_runs)/len(runs)*100:.1f}%" if runs else "0%",
                "unique_parameters": len(all_params),
                "tracked_metrics": len(all_metrics),
            },
            "top_performers": top_performers,
            "parameters_analyzed": list(all_params)[:10],  # Top 10 params
            "metrics_tracked": list(all_metrics)[:10],  # Top 10 metrics
            "generation_metadata": {
                "generated_at": datetime.now().isoformat(),
                "report_format": format,
                "mlflow_server": TRACKING_URI,
            },
        }

        # Format-specific generation
        if format == "json":
            report_content = json.dumps(report_data, indent=2)

        elif format == "markdown":
            md_content = f"""# Experiment Report: {experiment.name}

## Summary
- **Experiment ID**: {experiment_id}
- **Total Runs**: {report_data['summary_stats']['total_runs']}
- **Success Rate**: {report_data['summary_stats']['success_rate']}
- **Parameters Tested**: {report_data['summary_stats']['unique_parameters']}
- **Metrics Tracked**: {report_data['summary_stats']['tracked_metrics']}

## Top Performers
"""
            for metric, performer in top_performers.items():
                md_content += f"""
### {metric}: {performer['value']:.4f}
- **Best Run**: {performer['run_name']}
- **Key Parameters**: {', '.join([f"{k}={v}" for k, v in performer['params'].items()])}
"""

            md_content += f"""
## Parameters Analyzed
{', '.join(report_data['parameters_analyzed'])}

## Metrics Tracked
{', '.join(report_data['metrics_tracked'])}

---
*Report generated on {report_data['generation_metadata']['generated_at']}*
"""
            report_content = md_content

        else:  # HTML format
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>MLflow Experiment Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ color: #1f77b4; border-bottom: 2px solid #1f77b4; }}
        .metric {{ background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .summary {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin: 20px 0; }}
        .stat-box {{ background: #e9ecef; padding: 15px; border-radius: 5px; text-align: center; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Experiment Report: {experiment.name}</h1>
        <p><strong>ID:</strong> {experiment_id} | <strong>Generated:</strong> {report_data['generation_metadata']['generated_at']}</p>
    </div>

    <div class="summary">
        <div class="stat-box"><h3>{report_data['summary_stats']['total_runs']}</h3><p>Total Runs</p></div>
        <div class="stat-box"><h3>{report_data['summary_stats']['success_rate']}</h3><p>Success Rate</p></div>
        <div class="stat-box"><h3>{report_data['summary_stats']['unique_parameters']}</h3><p>Parameters</p></div>
    </div>

    <h2>🏆 Top Performers</h2>
"""
            for metric, performer in top_performers.items():
                html_content += f"""
    <div class="metric">
        <h3>{metric}: {performer['value']:.4f}</h3>
        <p><strong>Best Run:</strong> {performer['run_name']}</p>
        <p><strong>Key Parameters:</strong> {', '.join([f"{k}={v}" for k, v in performer['params'].items()])}</p>
    </div>
"""

            html_content += f"""
    <h2>📊 Analysis Summary</h2>
    <p><strong>Parameters:</strong> {', '.join(report_data['parameters_analyzed'])}</p>
    <p><strong>Metrics:</strong> {', '.join(report_data['metrics_tracked'])}</p>

    <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #dee2e6;">
        <p><em>Generated from MLflow server: {TRACKING_URI}</em></p>
    </footer>
</body>
</html>
"""
            report_content = html_content

        return {
            "experiment_id": experiment_id,
            "experiment_name": experiment.name,
            "format": format,
            "report_content": report_content,
            "metadata": report_data["generation_metadata"],
            "summary": report_data["summary_stats"],
            "file_size_chars": len(report_content),
            "message": f"Successfully generated {format.upper()} report for experiment {experiment.name}",
        }

    except Exception as e:
        raise MLflowQueryError(f"Failed to generate experiment report: {str(e)}")


@mcp.tool()
@mcp_response
@handle_mlflow_errors
def get_model_serving_info(model_name: str, version: str) -> Dict[str, Any]:
    """
    Get serving endpoints and deployment status for a model version.

    Args:
        model_name: Name of the registered model
        version: Model version to check

    Returns:
        A dictionary containing serving information and deployment status.
    """
    if not model_name.strip():
        raise MLflowValidationError("model_name cannot be empty")
    if not version.strip():
        raise MLflowValidationError("version cannot be empty")

    logger.info(f"Checking serving info for {model_name} version {version}")

    try:
        # Get model version details
        model_version = safe_mlflow_call(client.get_model_version, model_name, version)

        # Get run info for additional context
        run_info = None
        if model_version.run_id:
            try:
                run = safe_mlflow_call(client.get_run, model_version.run_id)
                run_info = {
                    "run_id": model_version.run_id,
                    "status": run.info.status,
                    "metrics": dict(list(run.data.metrics.items())[:5]),  # Top 5 metrics
                }
            except:
                pass

        # Check for serving-related tags
        serving_tags = {}
        if hasattr(model_version, "tags") and model_version.tags:
            for tag in model_version.tags:
                if any(
                    keyword in tag.key.lower() for keyword in ["serve", "deploy", "endpoint", "url"]
                ):
                    serving_tags[tag.key] = tag.value

        # Generate serving recommendations based on stage
        serving_recommendations = []
        deployment_readiness = "Unknown"

        if model_version.current_stage == "Production":
            serving_recommendations.extend(
                [
                    "Model is in Production stage - ready for high-traffic serving",
                    "Consider monitoring model performance in production",
                    "Set up automated retraining pipelines",
                ]
            )
            deployment_readiness = "Ready"
        elif model_version.current_stage == "Staging":
            serving_recommendations.extend(
                [
                    "Model is in Staging - suitable for testing and validation",
                    "Run A/B tests before promoting to Production",
                    "Validate model performance on production-like data",
                ]
            )
            deployment_readiness = "Testing"
        else:
            serving_recommendations.extend(
                [
                    "Promote model to Staging for deployment testing",
                    "Validate model performance before serving",
                    "Consider model monitoring and logging setup",
                ]
            )
            deployment_readiness = "Not Ready"

        # Mock serving endpoint info (would be real in production)
        potential_endpoints = {
            "rest_api": f"https://api.mlflow.com/models/{model_name}/{version}/invocations",
            "batch_inference": f"mlflow models serve -m models:/{model_name}/{version}",
            "docker_image": f"mlflow-models:{model_name}-{version}",
            "kubernetes_deployment": f"kubectl apply -f {model_name}-{version}-deployment.yaml",
        }

        return {
            "model_name": model_name,
            "version": version,
            "current_stage": model_version.current_stage,
            "deployment_readiness": deployment_readiness,
            "model_info": {
                "creation_time": MLflowUtils.format_timestamp(model_version.creation_timestamp),
                "source": model_version.source,
                "status": model_version.status,
            },
            "run_context": run_info,
            "serving_tags": serving_tags,
            "potential_endpoints": potential_endpoints,
            "serving_recommendations": serving_recommendations,
            "deployment_options": [
                {
                    "method": "MLflow Model Serving",
                    "command": f"mlflow models serve -m models:/{model_name}/{version} -p 5001",
                    "description": "Local serving for development and testing",
                },
                {
                    "method": "Docker Container",
                    "command": f"mlflow models build-docker -m models:/{model_name}/{version} -n {model_name}-{version}",
                    "description": "Containerized deployment for cloud platforms",
                },
                {
                    "method": "Cloud Serving",
                    "command": f"# Deploy to cloud platform (AWS SageMaker, Google AI Platform, etc.)",
                    "description": "Managed serving with auto-scaling and monitoring",
                },
            ],
            "monitoring_suggestions": [
                "Track prediction latency and throughput",
                "Monitor model accuracy degradation over time",
                "Set up alerts for anomalous predictions",
                "Log prediction inputs and outputs for retraining",
            ],
        }

    except Exception as e:
        raise MLflowResourceNotFoundError(
            "model_version",
            f"{model_name}:{version}",
            "Verify the model name and version exist in the registry",
        )


@mcp.tool()
@mcp_response
@handle_mlflow_errors
@cached()
def list_models(name_contains: str = "", max_results: int = 100) -> Dict[str, Any]:
    """
    List all registered models in the MLflow model registry, with optional filtering.

    Args:
        name_contains: Optional filter to only include models whose names contain this string
        max_results: Maximum number of results to return (default: 100)

    Returns:
        A dictionary containing all registered models matching the criteria.
    """
    logger.info(f"Fetching registered models (filter: '{name_contains}', max: {max_results})")

    if max_results <= 0:
        raise MLflowValidationError("max_results must be positive")

    # Get all registered models
    registered_models = client.search_registered_models(max_results=max_results)

    # Filter by name if specified
    if name_contains:
        registered_models = [
            model for model in registered_models if name_contains.lower() in model.name.lower()
        ]

    # Convert to Pydantic models
    models_info = []
    for model in registered_models:
        model_data = {
            "name": model.name,
            "creation_timestamp": MLflowUtils.format_timestamp(model.creation_timestamp),
            "last_updated_timestamp": MLflowUtils.format_timestamp(model.last_updated_timestamp),
            "description": model.description or "",
            "tags": {tag.key: tag.value for tag in model.tags} if hasattr(model, "tags") else {},
            "latest_versions": [],
        }

        # Add the latest versions if available
        if model.latest_versions and len(model.latest_versions) > 0:
            for version in model.latest_versions:
                version_info = ModelVersionInfo(
                    version=version.version,
                    status=version.status,
                    current_stage=version.current_stage,
                    creation_timestamp=MLflowUtils.format_timestamp(version.creation_timestamp),
                    source=version.source,
                    run_id=version.run_id,
                )
                model_data["latest_versions"].append(version_info.model_dump())

        model_info = RegisteredModelInfo(**model_data)
        models_info.append(model_info.model_dump())

    return {"total_models": len(models_info), "models": models_info}


@mcp.tool()
@mcp_response
@handle_mlflow_errors
@cached()
def list_experiments(name_contains: str = "", max_results: int = 100) -> Dict[str, Any]:
    """
    List all experiments in the MLflow tracking server, with optional filtering.

    Args:
        name_contains: Optional filter to only include experiments whose names contain this string
        max_results: Maximum number of results to return (default: 100)

    Returns:
        A dictionary containing all experiments matching the criteria.
    """
    logger.info(f"Fetching experiments (filter: '{name_contains}', max: {max_results})")

    if max_results <= 0:
        raise MLflowValidationError("max_results must be positive")

    # Get all experiments
    experiments = client.search_experiments()

    # Filter by name if specified
    if name_contains:
        experiments = [exp for exp in experiments if name_contains.lower() in exp.name.lower()]

    # Limit to max_results
    experiments = experiments[:max_results]

    # Convert to Pydantic models
    experiments_info = []
    for exp in experiments:
        exp_data = {
            "experiment_id": exp.experiment_id,
            "name": exp.name,
            "artifact_location": exp.artifact_location,
            "lifecycle_stage": exp.lifecycle_stage,
            "creation_time": MLflowUtils.format_timestamp(exp.creation_time)
            if hasattr(exp, "creation_time")
            else None,
            "tags": {tag.key: tag.value for tag in exp.tags} if hasattr(exp, "tags") else {},
            "run_count": MLflowUtils.safe_get_run_count(exp.experiment_id),
        }

        exp_info = ExperimentInfo(**exp_data)
        experiments_info.append(exp_info.model_dump())

    return {"total_experiments": len(experiments_info), "experiments": experiments_info}


@mcp.tool()
@mcp_response
@handle_mlflow_errors
@cached()
def get_run_details(run_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific MLflow run.

    Args:
        run_id: The ID or name of the MLflow run

    Returns:
        A dictionary containing detailed run information.
    """
    if not run_id.strip():
        raise MLflowValidationError("run_id cannot be empty")

    logger.info(f"Fetching details for run: {run_id}")

    # Resolve run ID (handles both IDs and names)
    actual_run_id = MLflowUtils.resolve_run_id(run_id)
    run = client.get_run(actual_run_id)

    run_data = {
        "run_id": run.info.run_id,
        "run_name": run.info.run_name,
        "experiment_id": run.info.experiment_id,
        "status": run.info.status,
        "start_time": MLflowUtils.format_timestamp(run.info.start_time),
        "end_time": MLflowUtils.format_timestamp(run.info.end_time) if run.info.end_time else None,
        "artifact_uri": run.info.artifact_uri,
        "lifecycle_stage": run.info.lifecycle_stage,
        "user_id": run.info.user_id,
        "metrics": MLflowUtils.extract_metrics_safely(run.data.metrics),
        "params": dict(run.data.params),
        "tags": {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")},
    }

    run_info = RunInfo(**run_data)
    return run_info.model_dump()


@mcp.tool()
@mcp_response
@handle_mlflow_errors
@cached()
def get_system_info() -> Dict[str, Any]:
    """
    Get information about the MLflow tracking server and system.

    Returns:
        A dictionary containing system information.
    """
    logger.info("Getting MLflow system information")

    system_data = {
        "mlflow_version": mlflow.__version__,
        "tracking_uri": mlflow.get_tracking_uri(),
        "registry_uri": mlflow.get_registry_uri(),
        "artifact_uri": mlflow.get_artifact_uri(),
        "python_version": sys.version,
        "server_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiment_count": "Error retrieving count",
        "model_count": "Error retrieving count",
        "active_runs": "Error retrieving count",
    }

    # Get experiment count
    try:
        experiments = client.search_experiments()
        system_data["experiment_count"] = len(experiments)
    except Exception as e:
        logger.warning(f"Error getting experiment count: {str(e)}")

    # Get model count
    try:
        models = client.search_registered_models()
        system_data["model_count"] = len(models)
    except Exception as e:
        logger.warning(f"Error getting model count: {str(e)}")

    # Get active run count
    try:
        active_runs = 0
        experiments = client.search_experiments()
        for exp in experiments:
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                filter_string="attributes.status = 'RUNNING'",
                max_results=1000,
            )
            active_runs += len(runs)
        system_data["active_runs"] = active_runs
    except Exception as e:
        logger.warning(f"Error getting active run count: {str(e)}")

    system_info = SystemInfo(**system_data)
    return system_info.model_dump()


@mcp.tool()
@mcp_response
@handle_mlflow_errors
@cached()
def get_model_details(model_name: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific registered model.

    Args:
        model_name: The name of the registered model

    Returns:
        A dictionary containing detailed information about the model.
    """
    if not model_name.strip():
        raise MLflowValidationError("model_name cannot be empty")

    logger.info(f"Fetching details for model: {model_name}")

    try:
        model = safe_mlflow_call(client.get_registered_model, model_name)
    except Exception as e:
        raise MLflowResourceNotFoundError(
            "model", model_name, "Check if the model is registered in MLflow"
        )

    model_data = {
        "name": model.name,
        "creation_timestamp": MLflowUtils.format_timestamp(model.creation_timestamp),
        "last_updated_timestamp": MLflowUtils.format_timestamp(model.last_updated_timestamp),
        "description": model.description or "",
        "tags": {tag.key: tag.value for tag in model.tags} if hasattr(model, "tags") else {},
        "versions": [],
    }

    # Get all versions for this model
    versions = client.search_model_versions(f"name='{model_name}'")

    for version in versions:
        version_data = {
            "version": version.version,
            "status": version.status,
            "current_stage": version.current_stage,
            "creation_timestamp": MLflowUtils.format_timestamp(version.creation_timestamp),
            "source": version.source,
            "run_id": version.run_id,
        }

        # Get additional information about the run if available
        if version.run_id:
            try:
                run = client.get_run(version.run_id)
                version_data["run"] = {
                    "status": run.info.status,
                    "start_time": MLflowUtils.format_timestamp(run.info.start_time),
                    "end_time": MLflowUtils.format_timestamp(run.info.end_time)
                    if run.info.end_time
                    else None,
                    "metrics": MLflowUtils.extract_metrics_safely(run.data.metrics),
                }
            except Exception as e:
                logger.warning(f"Error getting run details for {version.run_id}: {str(e)}")
                version_data["run"] = "Error retrieving run details"

        model_data["versions"].append(version_data)

    return model_data


@mcp.tool()
@mcp_response
@handle_mlflow_errors
@cached()
def get_experiment_details(experiment_id: str = "", experiment_name: str = "") -> Dict[str, Any]:
    """
    Get detailed information about a specific experiment.

    Args:
        experiment_id: The ID of the experiment (optional)
        experiment_name: The name of the experiment (optional)

    Returns:
        A dictionary containing detailed experiment information.
    """
    if not experiment_id.strip() and not experiment_name.strip():
        raise MLflowValidationError("Either experiment_id or experiment_name must be provided")

    logger.info(f"Fetching experiment details (ID: '{experiment_id}', Name: '{experiment_name}')")

    # Get experiment
    try:
        if experiment_id:
            experiment = client.get_experiment(experiment_id)
        else:
            experiment = client.get_experiment_by_name(experiment_name)
    except Exception as e:
        raise MLflowResourceNotFoundError(
            "experiment",
            experiment_id or experiment_name,
            "Check if the experiment exists in MLflow",
        )

    if not experiment:
        raise MLflowResourceNotFoundError(
            "experiment",
            experiment_id or experiment_name,
            "Experiment may have been deleted or archived",
        )

    # Get runs for this experiment
    runs = client.search_runs(experiment_ids=[experiment.experiment_id], max_results=50)

    # Prepare experiment info
    exp_data = {
        "experiment_id": experiment.experiment_id,
        "name": experiment.name,
        "artifact_location": experiment.artifact_location,
        "lifecycle_stage": experiment.lifecycle_stage,
        "creation_time": MLflowUtils.format_timestamp(experiment.creation_time)
        if hasattr(experiment, "creation_time")
        else None,
        "tags": {tag.key: tag.value for tag in experiment.tags}
        if hasattr(experiment, "tags")
        else {},
        "total_runs": len(runs),
        "runs": [],
    }

    # Add run summaries
    for run in runs[:20]:  # Limit to first 20 runs
        run_info = {
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            "status": run.info.status,
            "start_time": MLflowUtils.format_timestamp(run.info.start_time),
            "end_time": MLflowUtils.format_timestamp(run.info.end_time)
            if run.info.end_time
            else None,
            "metrics": MLflowUtils.extract_metrics_safely(run.data.metrics),
            "params": dict(run.data.params),
            "tags": {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")},
        }
        exp_data["runs"].append(run_info)

    return exp_data


@mcp.tool()
@mcp_response
@handle_mlflow_errors
def list_run_artifacts(run_id: str, artifact_path: str = "") -> Dict[str, Any]:
    """
    List artifacts for a specific MLflow run.

    Args:
        run_id: The ID or name of the MLflow run
        artifact_path: Optional path within artifacts (default: root)

    Returns:
        A dictionary containing artifact information.
    """
    if not run_id.strip():
        raise MLflowValidationError("run_id cannot be empty")

    logger.info(f"Listing artifacts for run: {run_id} at path: '{artifact_path}'")

    # Resolve run ID (handles both IDs and names)
    actual_run_id = MLflowUtils.resolve_run_id(run_id)

    artifacts = client.list_artifacts(actual_run_id, artifact_path)

    artifact_info = {
        "run_id": actual_run_id,
        "original_run_input": run_id,
        "artifact_path": artifact_path,
        "artifacts": [],
    }

    for artifact in artifacts:
        artifact_data = ArtifactInfo(
            path=artifact.path, is_dir=artifact.is_dir, file_size=artifact.file_size
        )
        artifact_info["artifacts"].append(artifact_data.model_dump())

    return artifact_info


@mcp.tool()
@mcp_response
@handle_mlflow_errors
@cached()
def search_runs(
    experiment_ids: str = "", filter_string: str = "", max_results: int = 20
) -> Dict[str, Any]:
    """
    Search for MLflow runs with optional filtering.

    Args:
        experiment_ids: Comma-separated experiment IDs to search (default: all experiments)
        filter_string: MLflow filter string (e.g., "metrics.accuracy > 0.9")
        max_results: Maximum number of results to return (default: 20)

    Returns:
        A dictionary containing matching runs.
    """
    if max_results <= 0:
        raise MLflowValidationError("max_results must be positive")

    logger.info(
        f"Searching runs (experiments: '{experiment_ids}', filter: '{filter_string}', max: {max_results})"
    )

    # Parse experiment IDs
    exp_ids = None
    if experiment_ids:
        exp_ids = [exp_id.strip() for exp_id in experiment_ids.split(",") if exp_id.strip()]

    # If no experiment IDs specified, get all experiments
    if exp_ids is None:
        try:
            all_experiments = client.search_experiments()
            exp_ids = [exp.experiment_id for exp in all_experiments]
        except Exception as e:
            logger.warning(f"Error getting all experiments: {str(e)}")
            exp_ids = ["0"]  # Default to experiment 0

    # Search runs
    runs = client.search_runs(
        experiment_ids=exp_ids,
        filter_string=filter_string if filter_string else None,
        max_results=max_results,
    )

    runs_info = {
        "total_found": len(runs),
        "experiment_ids": exp_ids,
        "filter_string": filter_string,
        "runs": [],
    }

    for run in runs:
        run_info = {
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            "experiment_id": run.info.experiment_id,
            "status": run.info.status,
            "start_time": MLflowUtils.format_timestamp(run.info.start_time),
            "end_time": MLflowUtils.format_timestamp(run.info.end_time)
            if run.info.end_time
            else None,
            "metrics": MLflowUtils.extract_metrics_safely(run.data.metrics),
            "params": dict(run.data.params),
            "tags": {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")},
        }
        runs_info["runs"].append(run_info)

    return runs_info


@mcp.tool()
@mcp_response
@handle_mlflow_errors
def download_artifact(
    run_id: str, artifact_path: str, download_path: str = "downloads"
) -> Dict[str, Any]:
    """
    Download an artifact from a specific MLflow run.

    Args:
        run_id: The ID or name of the MLflow run
        artifact_path: Path to the artifact to download
        download_path: Local path to download to (default: "downloads")

    Returns:
        A dictionary containing download information.
    """
    if not run_id.strip():
        raise MLflowValidationError("run_id cannot be empty")
    if not artifact_path.strip():
        raise MLflowValidationError("artifact_path cannot be empty")

    logger.info(f"Downloading artifact from run {run_id}: {artifact_path}")

    # Resolve run ID (handles both IDs and names)
    actual_run_id = MLflowUtils.resolve_run_id(run_id)

    # Use absolute path to project directory to avoid read-only filesystem issues
    if not os.path.isabs(download_path):
        project_dir = os.path.dirname(os.path.abspath(__file__))
        download_path = os.path.join(project_dir, download_path)

    # Create download directory if it doesn't exist
    os.makedirs(download_path, exist_ok=True)

    # Download the artifact
    downloaded_path = client.download_artifacts(actual_run_id, artifact_path, download_path)

    return {
        "run_id": actual_run_id,
        "original_run_input": run_id,
        "artifact_path": artifact_path,
        "downloaded_to": downloaded_path,
        "status": "success",
    }


@mcp.tool()
@mcp_response
@handle_mlflow_errors
def read_artifact_content(run_id: str, artifact_path: str) -> Dict[str, Any]:
    """
    Read the content of a text-based artifact from a specific MLflow run.

    Args:
        run_id: The ID or name of the MLflow run
        artifact_path: Path to the artifact to read

    Returns:
        A dictionary containing the artifact content.
    """
    if not run_id.strip():
        raise MLflowValidationError("run_id cannot be empty")
    if not artifact_path.strip():
        raise MLflowValidationError("artifact_path cannot be empty")

    logger.info(f"Reading artifact content from run {run_id}: {artifact_path}")

    # Resolve run ID (handles both IDs and names)
    actual_run_id = MLflowUtils.resolve_run_id(run_id)

    try:
        # Try to read artifact directly from MLflow server without downloading
        import io
        import tempfile
        from urllib.parse import urlparse

        # Get the artifact URI
        run = safe_mlflow_call(client.get_run, actual_run_id)
        artifact_uri = f"{run.info.artifact_uri}/{artifact_path}"

        # For local file systems, read directly
        if artifact_uri.startswith("file://"):
            file_path = artifact_uri.replace("file://", "")
            if not os.path.isfile(file_path):
                raise MLflowResourceNotFoundError(
                    "artifact", artifact_path, "Verify the artifact path exists in the run"
                )

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except UnicodeDecodeError:
                raise MLflowValidationError(
                    f"File {artifact_path} is not a text file or uses unsupported encoding"
                )

        # For remote systems (S3, Azure, GCS, etc.), use MLflow's artifact downloading
        else:
            # Use a temporary directory for remote artifacts
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = client.download_artifacts(actual_run_id, artifact_path, temp_dir)
                full_path = os.path.join(temp_dir, artifact_path)

                if not os.path.isfile(full_path):
                    raise MLflowResourceNotFoundError(
                        "artifact", artifact_path, "Verify the artifact path exists in the run"
                    )

                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        content = f.read()
                except UnicodeDecodeError:
                    raise MLflowValidationError(
                        f"File {artifact_path} is not a text file or uses unsupported encoding"
                    )

    except Exception as e:
        if isinstance(e, (MLflowResourceNotFoundError, MLflowValidationError)):
            raise
        logger.error(f"Failed to read artifact {artifact_path} from run {actual_run_id}: {str(e)}")
        raise MLflowResourceNotFoundError(
            "artifact", artifact_path, "Failed to access artifact from MLflow server"
        )

    return {
        "run_id": actual_run_id,
        "original_run_input": run_id,
        "artifact_path": artifact_path,
        "content": content,
        "content_length": len(content),
    }


if __name__ == "__main__":
    try:
        logger.info(f"Starting MLflow MCP server for Claude Code with tracking URI: {TRACKING_URI}")
        logger.info(f"Cache TTL: {CACHE_TTL} seconds")
        mcp.run(transport="stdio")
    except Exception as e:
        logger.error(f"Error running MCP server: {str(e)}", exc_info=True)
        sys.exit(1)

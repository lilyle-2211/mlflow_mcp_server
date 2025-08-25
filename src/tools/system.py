"""System health and monitoring tools."""

import logging
from datetime import datetime
from typing import Any, Dict

import mlflow

from ..core import TRACKING_URI
from ..models import SystemInfo
from ..utils import handle_mlflow_errors, mcp_response
from ..utils.mlflow_client import mlflow_client

logger = logging.getLogger(__name__)


@mcp_response
@handle_mlflow_errors
def health_check() -> Dict[str, Any]:
    """
    Check MLflow server connectivity and MCP server health.
    
    Returns:
        A dictionary containing health status and system information.
    """
    logger.info("Performing health check")
    
    health_status = "healthy"
    issues = []
    
    try:
        # Test MLflow connectivity
        experiments = mlflow_client.list_experiments(max_results=1)
        experiment_count = len(mlflow_client.list_experiments(max_results=1000))
        
        # Test model registry
        models = mlflow_client.list_registered_models(max_results=1)
        model_count = len(mlflow_client.list_registered_models(max_results=1000))
        
        server_status = "connected"
        
    except Exception as e:
        server_status = "disconnected"
        health_status = "unhealthy"
        issues.append(f"MLflow server connection failed: {str(e)}")
        experiment_count = 0
        model_count = 0
    
    # Get MLflow version
    try:
        mlflow_version = mlflow.__version__
    except Exception:
        mlflow_version = "unknown"
    
    system_info = SystemInfo(
        mlflow_version=mlflow_version,
        tracking_uri=TRACKING_URI,
        server_status=server_status,
        timestamp=datetime.now(),
        experiment_count=experiment_count,
        model_count=model_count,
        additional_info={
            "health_status": health_status,
            "issues": issues
        }
    )
    
    return {
        "health": {
            "status": health_status,
            "issues": issues,
            "last_check": datetime.now().isoformat()
        },
        "system_info": system_info.model_dump()
    }


@mcp_response
@handle_mlflow_errors
def get_system_info() -> Dict[str, Any]:
    """
    Get information about the MLflow tracking server and system.
    
    Returns:
        A dictionary containing system information.
    """
    logger.info("Getting system information")
    
    try:
        # Get basic counts
        experiments = mlflow_client.list_experiments(max_results=1000)
        models = mlflow_client.list_registered_models(max_results=1000)
        
        experiment_count = len(experiments)
        model_count = len(models)
        
        # Get some statistics
        active_experiments = sum(1 for exp in experiments if exp.lifecycle_stage == "active")
        
        server_status = "connected"
        
    except Exception as e:
        logger.error(f"Failed to get system info: {e}")
        experiment_count = 0
        model_count = 0
        active_experiments = 0
        server_status = "disconnected"
    
    system_info = SystemInfo(
        mlflow_version=mlflow.__version__,
        tracking_uri=TRACKING_URI,
        server_status=server_status,
        timestamp=datetime.now(),
        experiment_count=experiment_count,
        model_count=model_count,
        additional_info={
            "active_experiments": active_experiments,
            "deleted_experiments": experiment_count - active_experiments
        }
    )
    
    return {
        "system": system_info.model_dump(),
        "capabilities": [
            "experiment_management",
            "run_tracking", 
            "model_registry",
            "artifact_storage",
            "run_comparison",
            "bulk_operations"
        ]
    }

"""MLflow run management tools."""

import logging
from typing import Any, Dict, List

from ..models import RunInfo
from ..utils import cached, handle_mlflow_errors, mcp_response
from ..utils.mlflow_client import mlflow_client

logger = logging.getLogger(__name__)


@mcp_response
@cached()
@handle_mlflow_errors
def get_run_details(run_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific MLflow run.
    
    Args:
        run_id: The ID or name of the MLflow run
        
    Returns:
        A dictionary containing detailed run information.
    """
    logger.info(f"Getting run details for: {run_id}")
    
    run = mlflow_client.get_run(run_id)
    
    run_info = RunInfo(
        run_id=run.info.run_id,
        experiment_id=run.info.experiment_id,
        status=run.info.status,
        start_time=run.info.start_time,
        end_time=run.info.end_time,
        metrics=dict(run.data.metrics),
        params=dict(run.data.params),
        tags=dict(run.data.tags)
    )
    
    # Get artifacts list
    artifacts = mlflow_client.list_artifacts(run_id)
    artifact_paths = [artifact.path for artifact in artifacts]
    
    return {
        "run": run_info.model_dump(),
        "artifacts": artifact_paths,
        "artifact_count": len(artifact_paths),
        "experiment_name": mlflow_client.client.get_experiment(run.info.experiment_id).name
    }


@mcp_response
@cached()
@handle_mlflow_errors
def search_runs(
    experiment_ids: str = "", 
    filter_string: str = "", 
    max_results: int = 20
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
    logger.info(f"Searching runs with filter: '{filter_string}', experiments: {experiment_ids}")
    
    # Parse experiment IDs
    if experiment_ids:
        exp_ids = [exp_id.strip() for exp_id in experiment_ids.split(",")]
    else:
        # Get all experiments if none specified
        experiments = mlflow_client.list_experiments(max_results=1000)
        exp_ids = [exp.experiment_id for exp in experiments]
    
    # Search runs
    runs = mlflow_client.search_runs(
        experiment_ids=exp_ids,
        filter_string=filter_string,
        max_results=max_results
    )
    
    # Convert to our data model
    run_infos = []
    for run in runs:
        run_info = RunInfo(
            run_id=run.info.run_id,
            experiment_id=run.info.experiment_id,
            status=run.info.status,
            start_time=run.info.start_time,
            end_time=run.info.end_time,
            metrics=dict(run.data.metrics),
            params=dict(run.data.params),
            tags=dict(run.data.tags)
        )
        run_infos.append(run_info.model_dump())
    
    return {
        "runs": run_infos,
        "count": len(run_infos),
        "experiment_ids": exp_ids,
        "filter_string": filter_string,
        "max_results": max_results
    }


@mcp_response
@cached()
@handle_mlflow_errors
def compare_runs(run_ids: List[str]) -> Dict[str, Any]:
    """
    Compare metrics, parameters, and performance across multiple runs.
    
    Args:
        run_ids: List of run IDs or names to compare
        
    Returns:
        A dictionary containing detailed comparison across runs.
    """
    logger.info(f"Comparing runs: {run_ids}")
    
    if len(run_ids) < 2:
        raise ValueError("At least 2 runs are required for comparison")
    
    runs_data = []
    all_metrics = set()
    all_params = set()
    
    # Collect data from all runs
    for run_id in run_ids:
        run = mlflow_client.get_run(run_id)
        
        run_data = {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "metrics": dict(run.data.metrics),
            "params": dict(run.data.params),
            "tags": dict(run.data.tags)
        }
        
        runs_data.append(run_data)
        all_metrics.update(run.data.metrics.keys())
        all_params.update(run.data.params.keys())
    
    # Create comparison tables
    metrics_comparison = {}
    for metric in all_metrics:
        metrics_comparison[metric] = {
            run_data["run_id"]: run_data["metrics"].get(metric, "N/A")
            for run_data in runs_data
        }
    
    params_comparison = {}
    for param in all_params:
        params_comparison[param] = {
            run_data["run_id"]: run_data["params"].get(param, "N/A")
            for run_data in runs_data
        }
    
    # Calculate differences and recommendations
    recommendations = []
    for metric in all_metrics:
        values = [
            run_data["metrics"][metric] 
            for run_data in runs_data 
            if metric in run_data["metrics"]
        ]
        
        if len(values) >= 2:
            best_value = max(values) if "accuracy" in metric.lower() or "score" in metric.lower() else min(values)
            best_run = next(
                run_data["run_id"] 
                for run_data in runs_data 
                if run_data["metrics"].get(metric) == best_value
            )
            recommendations.append(f"Best {metric}: {best_value:.4f} in run {best_run}")
    
    return {
        "comparison": {
            "runs": runs_data,
            "metrics_comparison": metrics_comparison,
            "params_comparison": params_comparison,
            "recommendations": recommendations
        },
        "summary": {
            "total_runs": len(run_ids),
            "common_metrics": list(all_metrics),
            "common_params": list(all_params)
        }
    }

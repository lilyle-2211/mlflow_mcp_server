"""MLflow experiment management tools."""

import logging
from typing import Any, Dict, List

from ..models import ExperimentInfo
from ..utils import cached, handle_mlflow_errors, mcp_response
from ..utils.mlflow_client import mlflow_client

logger = logging.getLogger(__name__)


@mcp_response
@cached()
@handle_mlflow_errors
def list_experiments(name_contains: str = "", max_results: int = 100) -> Dict[str, Any]:
    """
    List all experiments in the MLflow tracking server, with optional filtering.
    
    Args:
        name_contains: Optional filter to only include experiments whose names contain this string
        max_results: Maximum number of results to return (default: 100)
    
    Returns:
        A dictionary containing all experiments matching the criteria.
    """
    logger.info(f"Listing experiments with filter: '{name_contains}', max_results: {max_results}")
    
    experiments = mlflow_client.list_experiments(max_results=max_results)
    
    # Filter by name if specified
    if name_contains:
        experiments = [
            exp for exp in experiments 
            if name_contains.lower() in exp.name.lower()
        ]
    
    # Convert to our data model
    experiment_infos = []
    for exp in experiments:
        experiment_info = ExperimentInfo(
            experiment_id=exp.experiment_id,
            name=exp.name,
            artifact_location=exp.artifact_location,
            lifecycle_stage=exp.lifecycle_stage,
            creation_time=exp.creation_time,
            last_update_time=exp.last_update_time,
            tags=exp.tags or {}
        )
        experiment_infos.append(experiment_info.model_dump())
    
    return {
        "experiments": experiment_infos,
        "count": len(experiment_infos),
        "filter_applied": name_contains,
        "max_results": max_results
    }


@mcp_response
@cached()
@handle_mlflow_errors
def get_experiment_details(experiment_id: str = "", experiment_name: str = "") -> Dict[str, Any]:
    """
    Get detailed information about a specific experiment.
    
    Args:
        experiment_id: The ID of the experiment (optional)
        experiment_name: The name of the experiment (optional)
        
    Returns:
        A dictionary containing detailed experiment information.
    """
    logger.info(f"Getting experiment details for ID: '{experiment_id}', name: '{experiment_name}'")
    
    experiment = mlflow_client.get_experiment_by_name_or_id(experiment_id, experiment_name)
    
    # Get experiment runs
    runs = mlflow_client.search_runs([experiment.experiment_id], max_results=100)
    
    # Calculate summary statistics
    run_count = len(runs)
    active_runs = sum(1 for run in runs if run.info.status == "RUNNING")
    completed_runs = sum(1 for run in runs if run.info.status == "FINISHED")
    failed_runs = sum(1 for run in runs if run.info.status == "FAILED")
    
    # Get unique metrics and parameters
    all_metrics = set()
    all_params = set()
    for run in runs:
        all_metrics.update(run.data.metrics.keys())
        all_params.update(run.data.params.keys())
    
    experiment_info = ExperimentInfo(
        experiment_id=experiment.experiment_id,
        name=experiment.name,
        artifact_location=experiment.artifact_location,
        lifecycle_stage=experiment.lifecycle_stage,
        creation_time=experiment.creation_time,
        last_update_time=experiment.last_update_time,
        tags=experiment.tags or {}
    )
    
    return {
        "experiment": experiment_info.model_dump(),
        "statistics": {
            "total_runs": run_count,
            "active_runs": active_runs,
            "completed_runs": completed_runs,
            "failed_runs": failed_runs,
            "unique_metrics": sorted(list(all_metrics)),
            "unique_parameters": sorted(list(all_params))
        },
        "recent_runs": [
            {
                "run_id": run.info.run_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "metrics": dict(run.data.metrics),
                "params": dict(run.data.params)
            }
            for run in runs[:10]  # Show last 10 runs
        ]
    }

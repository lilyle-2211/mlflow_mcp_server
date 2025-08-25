"""MLflow model registry management tools."""

import logging
from typing import Any, Dict

from ..models import ModelVersionInfo, RegisteredModelInfo
from ..utils import cached, handle_mlflow_errors, mcp_response
from ..utils.mlflow_client import mlflow_client

logger = logging.getLogger(__name__)


@mcp_response
@cached()
@handle_mlflow_errors
def list_models(name_contains: str = "", max_results: int = 100) -> Dict[str, Any]:
    """
    List all registered models in the MLflow model registry, with optional filtering.
    
    Args:
        name_contains: Optional filter to only include models whose names contain this string
        max_results: Maximum number of results to return (default: 100)
    
    Returns:
        A dictionary containing all registered models matching the criteria.
    """
    logger.info(f"Listing models with filter: '{name_contains}', max_results: {max_results}")
    
    models = mlflow_client.list_registered_models(max_results=max_results, name_contains=name_contains)
    
    # Convert to our data model
    model_infos = []
    for model in models:
        # Get latest versions
        latest_versions = []
        for version in model.latest_versions:
            version_info = ModelVersionInfo(
                name=version.name,
                version=version.version,
                creation_timestamp=version.creation_timestamp,
                current_stage=version.current_stage,
                description=version.description,
                run_id=version.run_id,
                source=version.source
            )
            latest_versions.append(version_info)
        
        model_info = RegisteredModelInfo(
            name=model.name,
            creation_timestamp=model.creation_timestamp,
            last_updated_timestamp=model.last_updated_timestamp,
            description=model.description,
            latest_versions=latest_versions
        )
        model_infos.append(model_info.model_dump())
    
    return {
        "models": model_infos,
        "count": len(model_infos),
        "filter_applied": name_contains,
        "max_results": max_results
    }


@mcp_response
@cached()
@handle_mlflow_errors
def get_model_details(model_name: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific registered model.
    
    Args:
        model_name: The name of the registered model
        
    Returns:
        A dictionary containing detailed information about the model.
    """
    logger.info(f"Getting model details for: {model_name}")
    
    model = mlflow_client.get_registered_model(model_name)
    
    # Get all versions
    all_versions = []
    for version in model.latest_versions:
        version_info = ModelVersionInfo(
            name=version.name,
            version=version.version,
            creation_timestamp=version.creation_timestamp,
            current_stage=version.current_stage,
            description=version.description,
            run_id=version.run_id,
            source=version.source
        )
        all_versions.append(version_info.model_dump())
    
    # Calculate statistics
    stage_counts = {}
    for version in model.latest_versions:
        stage = version.current_stage or "None"
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
    
    model_info = RegisteredModelInfo(
        name=model.name,
        creation_timestamp=model.creation_timestamp,
        last_updated_timestamp=model.last_updated_timestamp,
        description=model.description,
        latest_versions=all_versions
    )
    
    return {
        "model": model_info.model_dump(),
        "statistics": {
            "total_versions": len(all_versions),
            "stage_distribution": stage_counts,
            "latest_version": max([int(v.version) for v in all_versions]) if all_versions else 0
        }
    }

"""MLflow artifact management tools."""

import logging
import os
from typing import Any, Dict, List

from ..models import ArtifactInfo
from ..utils import cached, handle_mlflow_errors, mcp_response
from ..utils.mlflow_client import mlflow_client

logger = logging.getLogger(__name__)


@mcp_response
@cached()
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
    logger.info(f"Listing artifacts for run: {run_id}, path: {artifact_path}")
    
    artifacts = mlflow_client.list_artifacts(run_id, artifact_path)
    
    # Convert to our data model
    artifact_infos = []
    for artifact in artifacts:
        artifact_info = ArtifactInfo(
            path=artifact.path,
            is_dir=artifact.is_dir,
            file_size=artifact.file_size
        )
        artifact_infos.append(artifact_info.model_dump())
    
    return {
        "artifacts": artifact_infos,
        "count": len(artifact_infos),
        "run_id": run_id,
        "artifact_path": artifact_path
    }


@mcp_response
@handle_mlflow_errors
def download_artifact(
    run_id: str, 
    artifact_path: str, 
    download_path: str = "downloads"
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
    logger.info(f"Downloading artifact: {artifact_path} from run: {run_id}")
    
    # Ensure download directory exists
    os.makedirs(download_path, exist_ok=True)
    
    # Download the artifact
    local_path = mlflow_client.download_artifacts(run_id, artifact_path, download_path)
    
    # Get file information
    file_size = os.path.getsize(local_path) if os.path.isfile(local_path) else None
    is_directory = os.path.isdir(local_path)
    
    return {
        "download_info": {
            "run_id": run_id,
            "artifact_path": artifact_path,
            "local_path": local_path,
            "download_path": download_path,
            "file_size": file_size,
            "is_directory": is_directory,
            "success": True
        }
    }


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
    logger.info(f"Reading artifact content: {artifact_path} from run: {run_id}")
    
    # Download to temporary location
    temp_path = "temp_downloads"
    local_path = mlflow_client.download_artifacts(run_id, artifact_path, temp_path)
    
    try:
        # Try to read as text
        with open(local_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        content_type = "text"
        file_size = len(content)
        
    except UnicodeDecodeError:
        # If it's not text, provide basic info
        content = f"Binary file - cannot display content. File size: {os.path.getsize(local_path)} bytes"
        content_type = "binary"
        file_size = os.path.getsize(local_path)
    
    except Exception as e:
        content = f"Error reading file: {str(e)}"
        content_type = "error"
        file_size = 0
    
    # Clean up temporary file
    try:
        if os.path.exists(local_path):
            os.remove(local_path)
        # Remove temp directory if empty
        if os.path.exists(temp_path) and not os.listdir(temp_path):
            os.rmdir(temp_path)
    except Exception as e:
        logger.warning(f"Failed to clean up temporary files: {e}")
    
    return {
        "artifact_content": {
            "run_id": run_id,
            "artifact_path": artifact_path,
            "content": content,
            "content_type": content_type,
            "file_size": file_size
        }
    }


@mcp_response
@handle_mlflow_errors
def bulk_download_artifacts(
    run_id: str, 
    artifact_patterns: List[str], 
    download_path: str = "downloads"
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
    logger.info(f"Bulk downloading artifacts from run: {run_id}")
    
    # Ensure download directory exists
    os.makedirs(download_path, exist_ok=True)
    
    # Get all artifacts for the run
    all_artifacts = mlflow_client.list_artifacts(run_id)
    
    download_results = []
    
    for pattern in artifact_patterns:
        matching_artifacts = []
        
        # Simple pattern matching (could be enhanced with fnmatch)
        for artifact in all_artifacts:
            if pattern in artifact.path or pattern == "*":
                matching_artifacts.append(artifact)
        
        # Download each matching artifact
        for artifact in matching_artifacts:
            try:
                local_path = mlflow_client.download_artifacts(run_id, artifact.path, download_path)
                
                download_results.append({
                    "pattern": pattern,
                    "artifact_path": artifact.path,
                    "local_path": local_path,
                    "success": True,
                    "file_size": artifact.file_size
                })
                
            except Exception as e:
                download_results.append({
                    "pattern": pattern,
                    "artifact_path": artifact.path,
                    "success": False,
                    "error": str(e)
                })
    
    successful_downloads = sum(1 for result in download_results if result["success"])
    
    return {
        "bulk_download": {
            "run_id": run_id,
            "patterns": artifact_patterns,
            "download_path": download_path,
            "results": download_results,
            "summary": {
                "total_attempts": len(download_results),
                "successful_downloads": successful_downloads,
                "failed_downloads": len(download_results) - successful_downloads
            }
        }
    }

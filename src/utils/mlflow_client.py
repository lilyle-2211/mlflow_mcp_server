"""MLflow client wrapper with enhanced functionality."""

import logging
from typing import Any, Dict, List, Optional

from mlflow import MlflowClient

from ..core import TRACKING_URI
from ..core import MLflowResourceNotFoundError
from . import safe_mlflow_call

logger = logging.getLogger(__name__)


class EnhancedMLflowClient:
    """Enhanced MLflow client with additional utility methods."""
    
    def __init__(self, tracking_uri: str = TRACKING_URI):
        self.client = MlflowClient(tracking_uri=tracking_uri)
        self.tracking_uri = tracking_uri
        logger.info(f"Initialized MLflow client for {tracking_uri}")
    
    def get_experiment_by_name_or_id(self, experiment_id: str = "", experiment_name: str = ""):
        """Get experiment by either ID or name."""
        if experiment_id:
            return safe_mlflow_call(self.client.get_experiment, experiment_id)
        elif experiment_name:
            return safe_mlflow_call(self.client.get_experiment_by_name, experiment_name)
        else:
            raise MLflowResourceNotFoundError(
                "Experiment", 
                "No ID or name provided",
                "Provide either experiment_id or experiment_name parameter"
            )
    
    def list_experiments(self, view_type: str = "ACTIVE_ONLY", max_results: int = 100) -> List[Any]:
        """List experiments with pagination support."""
        return safe_mlflow_call(self.client.search_experiments, max_results=max_results, view_type=view_type)
    
    def list_registered_models(self, max_results: int = 100, name_contains: str = "") -> List[Any]:
        """List registered models with filtering."""
        filter_string = f"name ILIKE '%{name_contains}%'" if name_contains else None
        return safe_mlflow_call(
            self.client.search_registered_models, 
            max_results=max_results, 
            filter_string=filter_string
        )
    
    def get_run(self, run_id: str):
        """Get run details with error handling."""
        try:
            return safe_mlflow_call(self.client.get_run, run_id)
        except Exception:
            raise MLflowResourceNotFoundError("Run", run_id)
    
    def get_model_version(self, name: str, version: str):
        """Get model version with error handling."""
        try:
            return safe_mlflow_call(self.client.get_model_version, name, version)
        except Exception:
            raise MLflowResourceNotFoundError("Model Version", f"{name}:{version}")
    
    def get_registered_model(self, name: str):
        """Get registered model with error handling."""
        try:
            return safe_mlflow_call(self.client.get_registered_model, name)
        except Exception:
            raise MLflowResourceNotFoundError("Registered Model", name)
    
    def search_runs(
        self, 
        experiment_ids: List[str], 
        filter_string: str = "", 
        max_results: int = 100
    ) -> List[Any]:
        """Search runs with filtering."""
        return safe_mlflow_call(
            self.client.search_runs,
            experiment_ids=experiment_ids,
            filter_string=filter_string,
            max_results=max_results
        )
    
    def list_artifacts(self, run_id: str, path: str = "") -> List[Any]:
        """List artifacts for a run."""
        try:
            return safe_mlflow_call(self.client.list_artifacts, run_id, path)
        except Exception:
            raise MLflowResourceNotFoundError("Artifacts", f"run_id={run_id}, path={path}")
    
    def download_artifacts(self, run_id: str, path: str, dst_path: str = "downloads") -> str:
        """Download artifacts from a run."""
        return safe_mlflow_call(self.client.download_artifacts, run_id, path, dst_path)


# Global client instance
mlflow_client = EnhancedMLflowClient()

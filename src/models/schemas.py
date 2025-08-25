"""Pydantic data models for MLflow MCP Server."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class RunInfo(BaseModel):
    """Information about an MLflow run."""
    
    model_config = ConfigDict(extra="allow")
    
    run_id: str = Field(..., description="Unique identifier for the run")
    experiment_id: str = Field(..., description="ID of the experiment containing this run")
    status: str = Field(..., description="Status of the run (RUNNING, FINISHED, FAILED, etc.)")
    start_time: Optional[datetime] = Field(None, description="When the run started")
    end_time: Optional[datetime] = Field(None, description="When the run ended")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Metrics logged for this run")
    params: Dict[str, str] = Field(default_factory=dict, description="Parameters for this run")
    tags: Dict[str, str] = Field(default_factory=dict, description="Tags associated with this run")


class ModelVersionInfo(BaseModel):
    """Information about a model version."""
    
    model_config = ConfigDict(extra="allow")
    
    name: str = Field(..., description="Name of the registered model")
    version: str = Field(..., description="Version number of this model")
    creation_timestamp: Optional[datetime] = Field(None, description="When this version was created")
    current_stage: str = Field(..., description="Current stage (None, Staging, Production, Archived)")
    description: Optional[str] = Field(None, description="Description of this model version")
    run_id: Optional[str] = Field(None, description="ID of the run that created this model")
    source: Optional[str] = Field(None, description="Source path of the model artifacts")


class RegisteredModelInfo(BaseModel):
    """Information about a registered model."""
    
    model_config = ConfigDict(extra="allow")
    
    name: str = Field(..., description="Name of the registered model")
    creation_timestamp: Optional[datetime] = Field(None, description="When the model was registered")
    last_updated_timestamp: Optional[datetime] = Field(None, description="When the model was last updated")
    description: Optional[str] = Field(None, description="Description of the model")
    latest_versions: List[ModelVersionInfo] = Field(default_factory=list, description="Latest versions of the model")


class ExperimentInfo(BaseModel):
    """Information about an MLflow experiment."""
    
    model_config = ConfigDict(extra="allow")
    
    experiment_id: str = Field(..., description="Unique identifier for the experiment")
    name: str = Field(..., description="Name of the experiment")
    artifact_location: str = Field(..., description="Default artifact storage location")
    lifecycle_stage: str = Field(..., description="Lifecycle stage (active or deleted)")
    creation_time: Optional[datetime] = Field(None, description="When the experiment was created")
    last_update_time: Optional[datetime] = Field(None, description="When the experiment was last updated")
    tags: Dict[str, str] = Field(default_factory=dict, description="Tags associated with the experiment")


class ArtifactInfo(BaseModel):
    """Information about an artifact."""
    
    model_config = ConfigDict(extra="allow")
    
    path: str = Field(..., description="Path to the artifact")
    is_dir: bool = Field(..., description="Whether this is a directory")
    file_size: Optional[int] = Field(None, description="Size of the file in bytes")


class SystemInfo(BaseModel):
    """System and health information."""
    
    model_config = ConfigDict(extra="allow")
    
    mlflow_version: str = Field(..., description="Version of MLflow")
    tracking_uri: str = Field(..., description="MLflow tracking server URI")
    server_status: str = Field(..., description="Status of the tracking server")
    timestamp: datetime = Field(..., description="Timestamp of the health check")
    experiment_count: Optional[int] = Field(None, description="Total number of experiments")
    model_count: Optional[int] = Field(None, description="Total number of registered models")
    additional_info: Dict[str, Any] = Field(default_factory=dict, description="Additional system information")

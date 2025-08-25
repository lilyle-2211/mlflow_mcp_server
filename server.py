#!/usr/bin/env python3
"""
MLflow MCP Server - Refactored Edition

A well-structured, modular MLflow MCP Server for Claude Desktop integration.
This server provides comprehensive MLflow functionality through a clean,
maintainable architecture.

Features:
- Modular design with separate concerns
- Enhanced error handling and logging
- Caching for performance optimization
- Type safety with Pydantic models
- Comprehensive MLflow API coverage

Usage:
    python server.py

Environment Variables:
    MLFLOW_TRACKING_URI: MLflow server URI (default: http://localhost:5000)
    LOG_LEVEL: Logging level (default: INFO)
    CACHE_TTL: Cache TTL in seconds (default: 300)
"""

import asyncio
import logging
import sys
from typing import Any, Dict

from mcp.server.lowlevel.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
from mcp import types
from mcp.server.stdio import stdio_server
from mcp.types import Tool

from src.core import logger
from src.tools import (
    bulk_download_artifacts,
    compare_runs,
    download_artifact,
    get_experiment_details,
    get_model_details,
    get_run_details,
    get_system_info,
    health_check,
    list_experiments,
    list_models,
    list_run_artifacts,
    read_artifact_content,
    search_runs,
)

# Initialize MCP server
server = Server("mlflow-mcp-server")

# Define available tools
TOOLS = [
    Tool(
        name="health_check",
        description="Check MLflow server connectivity and system health",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": []
        }
    ),
    Tool(
        name="get_system_info", 
        description="Get MLflow system information and capabilities",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": []
        }
    ),
    Tool(
        name="list_experiments",
        description="List all experiments with optional filtering",
        inputSchema={
            "type": "object",
            "properties": {
                "name_contains": {
                    "type": "string",
                    "description": "Filter experiments by name substring"
                },
                "max_results": {
                    "type": "integer", 
                    "description": "Maximum number of results",
                    "default": 100
                }
            },
            "required": []
        }
    ),
    Tool(
        name="get_experiment_details",
        description="Get detailed information about a specific experiment",
        inputSchema={
            "type": "object",
            "properties": {
                "experiment_id": {
                    "type": "string",
                    "description": "Experiment ID"
                },
                "experiment_name": {
                    "type": "string", 
                    "description": "Experiment name"
                }
            },
            "required": []
        }
    ),
    Tool(
        name="search_runs",
        description="Search for runs with filtering",
        inputSchema={
            "type": "object",
            "properties": {
                "experiment_ids": {
                    "type": "string",
                    "description": "Comma-separated experiment IDs"
                },
                "filter_string": {
                    "type": "string",
                    "description": "MLflow filter string"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results",
                    "default": 20
                }
            },
            "required": []
        }
    ),
    Tool(
        name="get_run_details",
        description="Get detailed information about a specific run",
        inputSchema={
            "type": "object", 
            "properties": {
                "run_id": {
                    "type": "string",
                    "description": "Run ID"
                }
            },
            "required": ["run_id"]
        }
    ),
    Tool(
        name="compare_runs",
        description="Compare metrics and parameters across multiple runs",
        inputSchema={
            "type": "object",
            "properties": {
                "run_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of run IDs to compare"
                }
            },
            "required": ["run_ids"]
        }
    ),
    Tool(
        name="list_models",
        description="List registered models with optional filtering",
        inputSchema={
            "type": "object",
            "properties": {
                "name_contains": {
                    "type": "string",
                    "description": "Filter models by name substring"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results", 
                    "default": 100
                }
            },
            "required": []
        }
    ),
    Tool(
        name="get_model_details",
        description="Get detailed information about a registered model",
        inputSchema={
            "type": "object",
            "properties": {
                "model_name": {
                    "type": "string",
                    "description": "Name of the registered model"
                }
            },
            "required": ["model_name"]
        }
    ),
    Tool(
        name="list_run_artifacts",
        description="List artifacts for a specific run",
        inputSchema={
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "string",
                    "description": "Run ID"
                },
                "artifact_path": {
                    "type": "string",
                    "description": "Optional path within artifacts",
                    "default": ""
                }
            },
            "required": ["run_id"]
        }
    ),
    Tool(
        name="download_artifact",
        description="Download an artifact from a run",
        inputSchema={
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "string",
                    "description": "Run ID"
                },
                "artifact_path": {
                    "type": "string", 
                    "description": "Path to the artifact"
                },
                "download_path": {
                    "type": "string",
                    "description": "Local download path",
                    "default": "downloads"
                }
            },
            "required": ["run_id", "artifact_path"]
        }
    ),
    Tool(
        name="read_artifact_content",
        description="Read the content of a text-based artifact",
        inputSchema={
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "string",
                    "description": "Run ID"
                },
                "artifact_path": {
                    "type": "string",
                    "description": "Path to the artifact"
                }
            },
            "required": ["run_id", "artifact_path"]
        }
    ),
    Tool(
        name="bulk_download_artifacts",
        description="Download multiple artifacts with pattern matching",
        inputSchema={
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "string",
                    "description": "Run ID"
                },
                "artifact_patterns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of artifact path patterns"
                },
                "download_path": {
                    "type": "string",
                    "description": "Local download path",
                    "default": "downloads"
                }
            },
            "required": ["run_id", "artifact_patterns"]
        }
    )
]


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return TOOLS


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Handle tool calls."""
    logger.info(f"Calling tool: {name} with arguments: {arguments}")
    
    try:
        # Map tool names to functions
        tool_functions = {
            "health_check": health_check,
            "get_system_info": get_system_info,
            "list_experiments": list_experiments,
            "get_experiment_details": get_experiment_details,
            "search_runs": search_runs,
            "get_run_details": get_run_details,
            "compare_runs": compare_runs,
            "list_models": list_models,
            "get_model_details": get_model_details,
            "list_run_artifacts": list_run_artifacts,
            "download_artifact": download_artifact,
            "read_artifact_content": read_artifact_content,
            "bulk_download_artifacts": bulk_download_artifacts,
        }
        
        if name not in tool_functions:
            raise ValueError(f"Unknown tool: {name}")
        
        # Call the appropriate function
        result = tool_functions[name](**arguments)
        
        return [{"type": "text", "text": str(result)}]
        
    except Exception as e:
        logger.error(f"Error calling tool {name}: {e}")
        error_response = {
            "success": False,
            "error": {
                "type": type(e).__name__,
                "message": str(e),
                "tool": name,
                "arguments": arguments
            }
        }
        return [{"type": "text", "text": str(error_response)}]


async def main():
    """Main entry point for the MCP server."""
    logger.info("Starting MLflow MCP Server")
    
    # Run health check on startup
    try:
        health_result = health_check()
        if health_result.get("success"):
            logger.info("MLflow server health check passed")
        else:
            logger.warning("MLflow server health check failed")
    except Exception as e:
        logger.error(f"Failed to perform startup health check: {e}")
    
    # Start the server
    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="mlflow-mcp-server",
                    server_version="2.0.0",
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={}
                    )
                )
            )
    except Exception as e:
        logger.error(f"Server runtime error: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)

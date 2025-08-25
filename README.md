# MLflow MCP Server

A Model Context Protocol (MCP) server that enables Claude Desktop to interact with MLflow experiments through natural language conversations.

## Features

- **Natural Language Queries**: Ask questions about your ML experiments in plain English
- **Intelligent Interpretation**: Claude explains feature importance and model performance
- **Cross-run Analysis**: Automatic statistical comparisons with recommendations
- **Artifact Analysis**: Reads plots, JSON data, and classification reports directly
- **Troubleshooting**: Smart error messages with specific suggestions

## Quick Start

### Prerequisites

- Python 3.12+
- MLflow server running
- Claude Desktop

### Installation

1. Clone the repository:
```bash
git clone https://github.com/lilyle-2211/mlflow_mcp_server.git
cd mlflow_mcp_server
```

2. Install dependencies using uv:
```bash
uv sync
```

3. Install pre-commit hooks:
```bash
uv run pre-commit install
```

### Configuration

1. Update the MLflow tracking URI in `server.py` if needed:
```python
MLFLOW_TRACKING_URI = "http://localhost:5000"  # Update as needed
```

2. Configure Claude Desktop by adding to your `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "mlflow-mcp": {
      "command": "uv",
      "args": ["run", "python", "server.py"],
      "cwd": "/path/to/mlflow_mcp_server"
    }
  }
}
```

3. Copy the configuration template:
```bash
cp claude_desktop_config.json ~/.config/claude-desktop/claude_desktop_config.json
# Or on macOS:
cp claude_desktop_config.json ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

### Usage

Start a conversation in Claude Desktop:

- "List all my experiments"
- "Compare run_01 and run_02 feature importance"
- "Show me the best performing model from experiment X"
- "What are the key differences between these runs?"

## Development

### Setup Development Environment

```bash
# Install with dev dependencies
uv sync --extra dev

# Install pre-commit hooks
uv run pre-commit install

# Run pre-commit on all files
uv run pre-commit run --all-files
```

### Code Style

This project uses:
- Black for code formatting
- isort for import sorting
- flake8 for linting
- mypy for type checking

## Tech Stack

- **MLflow**: Experiment tracking and model registry
- **MCP (Model Context Protocol)**: Claude integration protocol
- **Python 3.12**: Runtime environment
- **uv**: Package and environment management

## License

MIT License

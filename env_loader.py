"""
Environment Variable Loader

Loads environment variables from .env.local file if it exists.
This module should be imported before config.py to ensure environment
variables are available.
"""

import os
from pathlib import Path


def load_env_file(env_file: str = ".env.local") -> None:
    """
    Load environment variables from a file.

    Args:
        env_file: Path to the environment file (relative to project root)
    """
    project_root = Path(__file__).parent
    env_path = project_root / env_file

    if not env_path.exists():
        # No .env.local file, use defaults from config.py
        return

    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue

            # Parse KEY=VALUE format
            if '=' not in line:
                continue

            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()

            # Remove quotes if present
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            elif value.startswith("'") and value.endswith("'"):
                value = value[1:-1]

            # Set environment variable if not already set
            if key not in os.environ:
                os.environ[key] = value


# Auto-load .env.local when this module is imported
load_env_file()

from pathlib import Path
import os
import yaml

def _project_root() -> Path:
    """
    Return the absolute path to the project root directory.

    Example:
        If this file is located at:
            /home/user/multi_docs_chat/utils/config_loader.py
        Then this function returns:
            /home/user/multi_docs_chat
    """
    # __file__ gives current file path; parents[1] moves up two levels to project root
    return Path(__file__).resolve().parents[1]


def load_config(config_path: str | None = None) -> dict:
    """
    Load the YAML configuration file safely and return it as a dictionary.

    Priority of locating the config file:
        1. Explicit path passed as `config_path`
        2. CONFIG_PATH environment variable
        3. Default path: <project_root>/config/config.yaml
    """
    # Read CONFIG_PATH from environment (if defined)
    env_path = os.getenv("CONFIG_PATH")

    # Determine which config path to use
    if config_path is None:
        # If no path provided, use environment variable or default config location
        config_path = env_path or str(_project_root() / "config" / "config.yaml")

    # Create a Path object from the config path
    path = Path(config_path)

    # Ensure the path is absolute; if not, resolve relative to project root
    if not path.is_absolute():
        path = _project_root() / path

    # Check if the config file actually exists
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    # Open and load the YAML configuration file
    with open(path, "r", encoding="utf-8") as f:
        # Use safe_load for security; return empty dict if file is empty
        return yaml.safe_load(f) or {}

"""Project paths and settings, loaded from configs/base.yaml."""
from pathlib import Path

import yaml


def find_project_root(start: Path | None = None) -> Path:
    """Walk upwards until a folder containing configs/base.yaml is found."""
    start = Path(start or Path.cwd()).resolve()
    for folder in [start, *start.parents]:
        if (folder / "configs" / "base.yaml").exists():
            return folder
    raise FileNotFoundError(f"Could not find configs/base.yaml above {start}")


def load_config(root: Path | None = None) -> dict:
    """Read configs/base.yaml and return it as a dict."""
    root = root or find_project_root()
    with open(root / "configs" / "base.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)

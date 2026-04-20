from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def make_run_dir(category: str, name: str, root: Path | str = "logs") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(root) / category / f"{ts}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def setup_logging(run_dir: Path) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)
    return root


def dump_config(run_dir: Path, config: Any) -> None:
    if is_dataclass(config):
        payload = asdict(config)
    elif isinstance(config, dict):
        payload = config
    else:
        payload = {"config": repr(config)}
    payload = _jsonify(payload)
    (run_dir / "config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return obj

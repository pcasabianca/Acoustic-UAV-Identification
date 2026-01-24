from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging


@dataclass
class VizResult:
    artifacts: List[Path] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    skipped: bool = False
    errors: List[str] = field(default_factory=list)


@dataclass
class VizContext:
    run_id: str
    repo_root: Path
    perf_dir: Path
    results_dir: Path
    out_dir: Path
    thresholds_path: Optional[Path]
    perfs: List[Dict[str, Any]]
    models: List[str]
    splits: List[str]
    strict: bool = False
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger('viz'))

import json
from pathlib import Path
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


def load_calibrated_thresholds(path: str) -> dict:
    p = Path(path)
    data = json.loads(p.read_text())
    return data


def list_models(data: dict) -> List[str]:
    if not isinstance(data, dict):
        return []
    if 'models' in data and isinstance(data['models'], dict):
        return list(data['models'].keys())
    return list(data.keys())


def get_threshold(data: dict, model: str) -> Optional[float]:
    if not isinstance(data, dict) or model is None:
        return None
    model_up = model.upper()
    models_node = data.get('models') if isinstance(data.get('models'), dict) else None
    if models_node:
        for key, val in models_node.items():
            if key.upper() == model_up:
                try:
                    return float(val.get('threshold'))
                except Exception:
                    logger.warning('Model %s found but threshold missing or invalid', key)
                    return None
    for key, val in data.items():
        if key.upper() == model_up:
            try:
                return float(val.get('threshold'))
            except Exception:
                logger.warning('Legacy model %s found but threshold missing or invalid', key)
                return None
    logger.debug('No threshold found for model %s', model)
    return None

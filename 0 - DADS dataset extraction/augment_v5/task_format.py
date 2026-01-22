def validate_task_structure(task):
    keys = {"sample_id", "split", "label", "scenario", "seed", "components", "output_relpath"}
    missing = keys - set(task.keys())
    if missing:
        raise ValueError(f"Task missing keys: {missing}")
    if not isinstance(task["components"], list):
        raise ValueError("components must be a list")
    # validate chains if present
    def _check_chain(ch):
        if ch is None:
            return
        if not isinstance(ch, list):
            raise ValueError("chain must be a list")
        for item in ch:
            if not isinstance(item, dict) or "op" not in item or not isinstance(item["op"], str):
                raise ValueError("each chain item must be dict with 'op' string")
            if "params" in item and not isinstance(item["params"], dict):
                raise ValueError("op 'params' must be a dict if present")

    for c in task["components"]:
        _check_chain(c.get("chain"))
    _check_chain(task.get("global_chain"))

OP_REGISTRY = {}


def register(name):
    def _decorator(fn):
        OP_REGISTRY[name] = fn
        return fn

    return _decorator

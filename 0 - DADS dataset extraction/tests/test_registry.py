from importlib import import_module


def test_modifiers_import_registers_ops():
    m = import_module('augment_v5.modifiers')
    assert hasattr(m, 'OP_REGISTRY')
    assert 'gain_db' in m.OP_REGISTRY
    assert 'distance_atten' in m.OP_REGISTRY

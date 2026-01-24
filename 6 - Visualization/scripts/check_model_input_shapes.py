#!/usr/bin/env python3
"""Load each saved model and print its expected input shape(s)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

from tensorflow import keras

models = {
    'CNN': config.CNN_MODEL_PATH,
    'RNN': config.RNN_MODEL_PATH,
    'CRNN': config.CRNN_MODEL_PATH,
    'Attention-CRNN': config.ATTENTION_CRNN_MODEL_PATH,
}

for name, path in models.items():
    print(f"Model: {name}")
    if not path.exists():
        print(f"  Path not found: {path}\n")
        continue
    try:
        m = keras.models.load_model(path, compile=False)
        print(f"  Keras model loaded from: {path}")
        try:
            print(f"  Input shape(s): {m.input_shape}")
        except Exception:
            # Some models expose multiple inputs
            if hasattr(m, 'inputs'):
                shapes = [inp.shape for inp in m.inputs]
                print(f"  Inputs: {shapes}")
        print(f"  Summary (first 5 lines):")
        s = []
        m.summary(print_fn=lambda x: s.append(x))
        for line in s[:5]:
            print("   ", line)
    except Exception as e:
        print(f"  Failed to load model: {e}")
    print()

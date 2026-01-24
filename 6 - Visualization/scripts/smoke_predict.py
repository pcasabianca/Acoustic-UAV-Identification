#!/usr/bin/env python3
"""Quick smoke test: load one audio file, compute mel as training, and run predict on each model."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
import json
import numpy as np
from tensorflow import keras

# Use precomputed MEL only
INDEX = Path(config.EXTRACTED_FEATURES_DIR) / 'mel_test_index.json'
if not INDEX.exists():
    print('mel_test_index.json not found; precomputed-only mode requires it. Aborting.')
    sys.exit(1)

data = json.loads(INDEX.read_text())
names = data.get('names', [])
mels = data.get('mel', [])
if not names:
    print('mel_test_index.json contains no entries')
    sys.exit(1)

# pick first entry
name = names[0]
mel_db = np.array(mels[0])
print('Using precomputed sample:', name)
print('precomputed mel shape:', mel_db.shape)

models = {
    'CNN': config.CNN_MODEL_PATH,
    'RNN': config.RNN_MODEL_PATH,
    'CRNN': config.CRNN_MODEL_PATH,
    'Attention-CRNN': config.ATTENTION_CRNN_MODEL_PATH,
}

for name_m, path in models.items():
    print('\n---', name_m, '---')
    if not path.exists():
        print('Model not found at', path)
        continue
    try:
        # load with custom loss fallback
        from loss_functions import get_loss_function
        fallback = get_loss_function('recall_focused', fn_penalty=50.0)
        m = keras.models.load_model(path, custom_objects={'loss_fn': fallback}, compile=False)
        print('Loaded OK. model.input_shape =', m.input_shape)
        if name_m == 'RNN':
            x = mel_db[np.newaxis, ...]
        else:
            x = mel_db[np.newaxis, ..., np.newaxis]
        print('Input provided shape:', x.shape)
        preds = m.predict(x, verbose=0)
        print('Preds shape:', np.array(preds).shape)
    except Exception as e:
        print('Error with model:', e)

print('\nDone (precomputed-only mode)')

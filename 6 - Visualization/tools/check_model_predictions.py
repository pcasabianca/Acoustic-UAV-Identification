#!/usr/bin/env python3
"""
Load saved models (without compiling) and print prediction distributions on validation set.
Usage: .venv/bin/python tools/check_model_predictions.py --model cnn
Supported models: cnn, rnn, crnn, attention_crnn
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
import json
import numpy as np
import tensorflow as tf

MODEL_MAP = {
    'cnn': config.CNN_MODEL_PATH,
    'rnn': config.RNN_MODEL_PATH,
    'crnn': config.CRNN_MODEL_PATH,
    'attention_crnn': config.ATTENTION_CRNN_MODEL_PATH,
}

def load_data():
    with open(config.MEL_VAL_DATA_PATH_STR, 'r') as f:
        data = json.load(f)
    X = np.array(data.get('mel', []))
    y = np.array(data.get('labels', []))
    return X, y

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=list(MODEL_MAP.keys()))
    args = parser.parse_args()

    model_path = MODEL_MAP[args.model]
    print('Model path:', model_path)
    if not Path(model_path).exists():
        print('Model file not found:', model_path)
        sys.exit(1)

    X, y = load_data()
    print('X shape:', X.shape, 'y shape:', y.shape, 'unique y:', np.unique(y, return_counts=True))

    # Add channel dimension ONLY for CNN/CRNN/Attention (Conv2D needs 4D)
    # RNN uses pure LSTM and expects 3D: (samples, time_steps, features)
    if X.ndim == 3 and args.model != 'rnn':
        X = X[..., np.newaxis]
        print(f'Added channel dimension for {args.model} (Conv2D models need 4D)')
    elif args.model == 'rnn':
        print(f'Keeping 3D shape for RNN (LSTM expects 3D input)')

    # Load model without compiling
    model = tf.keras.models.load_model(str(model_path), compile=False)
    print('Loaded model, model.summary:')
    model.summary()

    preds = model.predict(X, batch_size=32)
    # Softmax outputs -> positive class prob = preds[:,1]
    if preds.ndim == 2 and preds.shape[1] == 2:
        pos = preds[:,1]
    else:
        # If model outputs single prob
        pos = preds.ravel()

    import numpy as np
    print('Predictions stats: min, mean, median, max:', pos.min(), pos.mean(), np.median(pos), pos.max())
    # Fraction > 0.5
    thr = 0.5
    pred_pos = (pos > thr).astype(int)
    from collections import Counter
    print('Predicted positives at t=0.5:', int(pred_pos.sum()), '/', len(pred_pos))
    # Confusion
    tp = ((pred_pos == 1) & (y == 1)).sum()
    fp = ((pred_pos == 1) & (y == 0)).sum()
    tn = ((pred_pos == 0) & (y == 0)).sum()
    fn = ((pred_pos == 0) & (y == 1)).sum()
    print('TP, FP, TN, FN:', tp, fp, tn, fn)
    print('Precision, Recall:', tp/(tp+fp+1e-9), tp/(tp+fn+1e-9))

    # Print histogram of pos
    hist = np.histogram(pos, bins=20)
    print('Histogram bins:', hist[0])

    # Print a few examples where model predicts positive but true is negative (FP samples)
    idx_fp = np.where((pred_pos==1)&(y==0))[0]
    print('Some FP indices (up to 10):', idx_fp[:10])

    # Save summary
    out = {
        'model': args.model,
        'n_samples': int(len(y)),
        'pred_pos_count': int(pred_pos.sum()),
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
        'pred_mean': float(pos.mean()),
    }
    print('Summary:', out)

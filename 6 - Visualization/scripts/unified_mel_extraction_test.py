#!/usr/bin/env python3
"""Unified MEL extractor and tester.

Usage: run from project root under the project's .venv:
  .venv/bin/python3 scripts/unified_mel_extraction_test.py

What it does:
- Implements a single `extract_mel` function matching trainers (n_mels=44, n_fft=2048, hop_length=512,
  duration=4.0s, trim/pad to 90 frames, librosa.power_to_db with ref=np.max).
- Walks `config.DATASET_ROOT / '1'` (folder `1`) and processes up to N samples (default 30).
- For each sample: computes on-the-fly mel, tries to load precomputed mel from
  `mel_test_index.json` (if present), writes a comparison PNG (`pre/onfly/diff`).
- Attempts to load models from `config` and run a single predict to validate input shapes.
- Writes a summary CSV with prediction means/success flags.

This helps pinpoint why precomputed data may be empty or padded.
"""

import json
from pathlib import Path
import numpy as np
import librosa
import matplotlib.pyplot as plt
import os
import csv
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


OUT_DIR = Path("6 - Visualization/outputs/unified_extraction")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_COUNT = 30
N_MELS = 44
N_FFT = 2048
HOP_LENGTH = 512
DURATION = 4.0
TARGET_FRAMES = 90


def extract_mel(path, sr=22050, duration=DURATION, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH,
                target_frames=TARGET_FRAMES, to_db=True):
    # load
    y, _ = librosa.load(str(path), sr=sr, duration=duration)
    # if audio shorter than duration librosa.load already returns shortened array
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    if to_db:
        mel_db = librosa.power_to_db(mel, ref=np.max)
    else:
        mel_db = mel
    # transpose to (n_mels, time)
    mel_db = mel_db[:, :target_frames]
    if mel_db.shape[1] < target_frames:
        mel_db = np.pad(mel_db, ((0, 0), (0, target_frames - mel_db.shape[1])), mode='constant', constant_values=(mel_db.min(),))
    return mel_db


def load_test_index():
    p = Path(config.EXTRACTED_FEATURES_DIR) / "mel_test_index.json"
    if not p.exists():
        return None
    try:
        with open(p, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def find_files(dataset_root, subdir='1'):
    base = Path(dataset_root) / str(subdir)
    if not base.exists():
        # fallback: try dataset root itself
        base = Path(dataset_root)
    files = []
    for p in base.rglob('*.wav'):
        files.append(p)
    files = sorted(files)
    return files


def make_comparison_png(name, pre_mel, on_mel, out_path):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    if pre_mel is None:
        axs[0].imshow(np.zeros_like(on_mel), aspect='auto', origin='lower', cmap='viridis')
        axs[0].set_title('precomputed (missing)')
    else:
        axs[0].imshow(pre_mel, aspect='auto', origin='lower', cmap='viridis')
        axs[0].set_title('precomputed')

    axs[1].imshow(on_mel, aspect='auto', origin='lower', cmap='viridis')
    axs[1].set_title('on-the-fly')

    if pre_mel is None:
        diff = on_mel
    else:
        diff = on_mel - pre_mel
    axs[2].imshow(diff, aspect='auto', origin='lower', cmap='RdBu')
    axs[2].set_title('diff (onfly - pre)')

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(name)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def load_models():
    """Load models with a safe custom_objects mapping so saved custom losses resolve.
    Returns a dict of loaded models (may be empty if models not found).
    """
    models = {}
    mp = {
        'cnn': config.CNN_MODEL_PATH,
        'rnn': config.RNN_MODEL_PATH,
        'crnn': config.CRNN_MODEL_PATH,
        'attention_crnn': config.ATTENTION_CRNN_MODEL_PATH,
    }
    import tensorflow as tf
    # Import loss functions and register a fall-back 'loss_fn' name
    try:
        from loss_functions import get_loss_function
        # default fall-back loss function (recall_focused) matching training override
        fallback_loss = get_loss_function('recall_focused', fn_penalty=50.0)
        custom_objects = {'loss_fn': fallback_loss}
    except Exception:
        custom_objects = None

    for k, p in mp.items():
        try:
            if p.exists():
                if custom_objects:
                    models[k] = tf.keras.models.load_model(str(p), custom_objects=custom_objects)
                else:
                    models[k] = tf.keras.models.load_model(str(p))
                print(f'[OK] Loaded {k} model from {p}')
            else:
                print(f'[SKIP] Model {k} not found at {p}')
        except Exception as e:
            print(f'[ERR] Loading model {k}: {e}')
    return models


def format_for_model(mel, model, model_key):
    # mel shape expected (n_mels, time)
    arr = mel.copy()
    if model_key == 'rnn':
        # RNN trainers expect shape (batch, n_mels, time)
        X = arr[np.newaxis, ...]
    else:
        # CNN/CRNN/Attention expect channel last (n_mels, time, 1)
        X = arr[..., np.newaxis][np.newaxis, ...]
    # Try to adapt to model.input_shape if necessary
    try:
        input_shape = model.input_shape
        # input_shape example: (None, 44, 90, 1) or (None,44,90)
        expected = list(input_shape[1:])
        # reshape if mismatch in channel dimension only
        if len(expected) == 3 and X.shape[1:] != tuple(expected):
            # attempt simple reshape: if expected has 3 dims and X has 3 dims
            X = np.reshape(X, (1, expected[0], expected[1], expected[2]))
    except Exception:
        pass
    return X


def main():
    print('Unified MEL extractor/test starting...')
    test_index = load_test_index()
    files = find_files(config.DATASET_ROOT, subdir='1')
    if not files:
        print('No files found to process under', config.DATASET_ROOT)
        return 1

    to_process = files[:SAMPLE_COUNT]
    models = load_models()

    summary_rows = []
    for p in to_process:
        name = p.name
        print('Processing', name)
        # Use PRECOMPUTED mel only (per user request). If missing, mark as missing.
        pre_mel = None
        if test_index is not None:
            try:
                idx = test_index.get('names', []).index(name)
                pre_mel = np.array(test_index.get('mel', [])[idx])
            except ValueError:
                pre_mel = None

        out_png = OUT_DIR / f"{name}.png"
        # For diagnostics still generate a PNG: precomputed vs placeholder
        make_comparison_png(name, pre_mel, np.zeros((N_MELS, TARGET_FRAMES)), out_png)

        # model predictions using precomputed mel if available
        row = {'file': str(p)}
        for k, m in models.items():
            if pre_mel is None:
                row[f'{k}_prob'] = ''
                row[f'{k}_ok'] = False
                row[f'{k}_err'] = 'precomputed missing'
                continue
            try:
                X = format_for_model(pre_mel, m, k)
                pred = m.predict(X)
                prob = float(pred[0, 1]) if pred.shape[-1] > 1 else float(pred[0, 0])
                row[f'{k}_prob'] = prob
                row[f'{k}_ok'] = True
            except Exception as e:
                row[f'{k}_prob'] = ''
                row[f'{k}_ok'] = False
                row[f'{k}_err'] = str(e)
        summary_rows.append(row)

    # write summary CSV
    keys = set()
    for r in summary_rows:
        keys.update(r.keys())
    keys = sorted(keys)
    out_csv = OUT_DIR / 'model_prediction_summary.csv'
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    print('Done. PNGs + summary written to', OUT_DIR)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

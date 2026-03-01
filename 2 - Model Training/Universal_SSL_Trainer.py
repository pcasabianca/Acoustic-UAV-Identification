#!/usr/bin/env python3
"""
Universal Self-Supervised Pretraining Trainer

Two-stage training:
1) Self-supervised consistency pretraining on unlabeled MEL features
2) Supervised fine-tuning on labels

This script mirrors the spirit of Universal_Curriculum_Trainer while adding an
SSL stage before standard supervised optimization.
"""

import argparse
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from termcolor import colored

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Thread and GPU config
try:
    tf.config.threading.set_intra_op_parallelism_threads(config.TF_INTRA_OP_PARALLELISM_THREADS)
    tf.config.threading.set_inter_op_parallelism_threads(config.TF_INTER_OP_PARALLELISM_THREADS)
except Exception:
    pass

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU memory growth config failed: {e}")

sys.path.insert(0, str(Path(__file__).parent.parent / 'tools'))
from feature_loader import load_mel_features
from loss_functions import get_loss_function
from tools.safe_callbacks import SafeEarlyStopping, ConfusionMatrixLogger
from Universal_Curriculum_Trainer import build_model_from_trainer


def _prepare_data():
    train_mels, train_labels, _ = load_mel_features('train')
    val_mels, val_labels, _ = load_mel_features('val')

    if len(train_mels) == 0:
        raise RuntimeError('No training MEL features found')
    if len(val_mels) == 0:
        raise RuntimeError('No validation MEL features found')

    x_train = np.asarray(train_mels, dtype=np.float32)
    y_train = np.asarray(train_labels)
    x_val = np.asarray(val_mels, dtype=np.float32)
    y_val = np.asarray(val_labels)

    if x_train.ndim == 3:
        x_train = x_train[..., np.newaxis]
    if x_val.ndim == 3:
        x_val = x_val[..., np.newaxis]

    return x_train, y_train, x_val, y_val


def _augment_batch(x):
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=0.02, dtype=tf.float32)
    gain = tf.random.uniform([tf.shape(x)[0], 1, 1, 1], minval=0.85, maxval=1.15)
    return tf.clip_by_value(x * gain + noise, -6.0, 6.0)


def _ssl_pretrain(model, x_train, ssl_epochs=5, batch_size=32, ssl_lr=1e-4):
    optimizer = keras.optimizers.Adam(learning_rate=ssl_lr)
    dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(min(len(x_train), 8192)).batch(batch_size)

    history = {'ssl_loss': []}

    for epoch in range(ssl_epochs):
        epoch_losses = []
        for batch in dataset:
            with tf.GradientTape() as tape:
                view_a = _augment_batch(batch)
                view_b = _augment_batch(batch)

                out_a = model(view_a, training=True)
                out_b = model(view_b, training=True)

                # Consistency loss (symmetric)
                loss_ab = tf.reduce_mean(tf.square(tf.stop_gradient(out_a) - out_b))
                loss_ba = tf.reduce_mean(tf.square(tf.stop_gradient(out_b) - out_a))
                loss = 0.5 * (loss_ab + loss_ba)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_losses.append(float(loss.numpy()))

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        history['ssl_loss'].append(mean_loss)
        print(colored(f"[SSL] Epoch {epoch + 1}/{ssl_epochs} - ssl_loss={mean_loss:.6f}", 'cyan'))

    return history


def _model_paths(model_key: str):
    key = model_key.lower()
    # Model save paths (nom standard sans suffix curriculum)
    model_name = f"{key}_model"
    model_save_path = config.MODELS_DIR / f"{model_name}.keras"
    history_save_path = config.RESULTS_DIR / f"{model_name}_history.csv"
    return model_save_path, history_save_path


def _pad_history_columns(columns: dict[str, list]) -> dict[str, list]:
    max_len = max((len(v) for v in columns.values()), default=0)
    padded = {}
    for key, values in columns.items():
        if len(values) < max_len:
            padded[key] = list(values) + [np.nan] * (max_len - len(values))
        else:
            padded[key] = list(values)
    return padded


def main():
    parser = argparse.ArgumentParser(description='Universal SSL + Supervised trainer')
    parser.add_argument('--model', type=str, required=True,
                        choices=['cnn', 'rnn', 'crnn', 'attention_crnn', 'efficientnet', 'mobilenet', 'conformer', 'tcn', 'ast', 'multires'])
    parser.add_argument('--ssl_epochs', type=int, default=5)
    parser.add_argument('--max_epochs', type=int, default=config.MAX_EPOCHS)
    parser.add_argument('--min_epochs', type=int, default=config.MIN_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=getattr(config, 'BATCH_SIZE', 8))
    parser.add_argument('--learning_rate', type=float, default=getattr(config, 'LEARNING_RATE', 1e-4))
    parser.add_argument('--ssl_learning_rate', type=float, default=1e-4)
    parser.add_argument('--loss', type=str, default='focal', choices=['bce', 'weighted_bce', 'focal'])
    parser.add_argument('--focal_alpha', type=float, default=0.60)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--blocks', type=int, default=6, help='Conformer blocks (conformer only)')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--verbose', type=int, default=2)
    args = parser.parse_args()

    started = datetime.now()
    print(colored('\n' + '=' * 80, 'magenta'))
    print(colored('  UNIVERSAL SSL TRAINER', 'magenta'))
    print(colored('=' * 80 + '\n', 'magenta'))

    x_train, y_train, x_val, y_val = _prepare_data()

    input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
    model = build_model_from_trainer(
        input_shape,
        args.model,
        num_conformer_blocks=args.blocks,
        num_classes=args.num_classes,
    )

    # Stage 1: SSL pretraining
    ssl_hist = _ssl_pretrain(
        model,
        x_train,
        ssl_epochs=args.ssl_epochs,
        batch_size=args.batch_size,
        ssl_lr=args.ssl_learning_rate,
    )

    # Stage 2: supervised fine-tuning
    loss_fn = get_loss_function(args.loss, alpha=args.focal_alpha, gamma=args.focal_gamma)
    optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    output_format = config.TRAINER_OUTPUT_FORMATS.get(args.model, 'binary')
    callbacks = [
        SafeEarlyStopping(
            start_from_epoch=args.min_epochs,
            monitor='val_accuracy',
            mode='max',
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1
        ),
        ConfusionMatrixLogger(
            validation_data=(x_val, y_val),
            output_format=output_format,
            verbose=1,
            predict_batch_size=min(8, args.batch_size),
        ),
    ]

    hist = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=args.batch_size,
        epochs=args.max_epochs,
        callbacks=callbacks,
        verbose=args.verbose,
    )

    model_path, history_path = _model_paths(args.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))

    merged = {'ssl_loss': ssl_hist.get('ssl_loss', [])}
    for k, v in hist.history.items():
        merged[k] = v
    pd.DataFrame(_pad_history_columns(merged)).to_csv(history_path, index=False)

    print(colored(f"\n[SAVED] Model: {model_path}", 'green'))
    print(colored(f"[SAVED] History: {history_path}", 'green'))
    print(colored(f"[DONE] Duration: {datetime.now() - started}\n", 'green'))


if __name__ == '__main__':
    main()

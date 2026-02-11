#!/usr/bin/env python3
"""
Test rapide du curriculum learning avec early stopping.
"""

import sys
import os
from pathlib import Path

# Ajouter au path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / '2 - Model Training'))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import config
from termcolor import colored
from tools.safe_callbacks import CurriculumAwareEarlyStopping
from Universal_Curriculum_Trainer import get_model_builder

# Créer données synthétiques
print(colored("Création données synthétiques...", "cyan"))
X_train = np.random.randn(1000, 44, 90, 1).astype(np.float32)
y_train = np.random.randint(0, 2, 1000).astype(np.int32)
X_val = np.random.randn(200, 44, 90, 1).astype(np.float32)
y_val = np.random.randint(0, 2, 200).astype(np.int32)
difficulty = np.random.rand(1000).astype(np.float32)

print(colored("Construction modèle CNN...", "cyan"))
build_model = get_model_builder('cnn')
model = build_model((44, 90, 1))

print(colored("Compilation...", "cyan"))
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(colored("Configuration callback early stopping...", "cyan"))
early_stopping = CurriculumAwareEarlyStopping(
    start_from_epoch=10,  # MIN 10 epochs
    monitor='val_accuracy',
    mode='max',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

print(colored("\nDémarrage training (15 epochs max, 10 min)...\n", "green"))

# Simuler curriculum: epoch par epoch
for epoch in range(15):
    # Curriculum simple: commence avec 50%, augmente graduellement
    frac = min(0.5 + epoch * 0.05, 1.0)
    n_samples = int(len(X_train) * frac)
    
    # Trier par difficulté et prendre les plus faciles
    indices = np.argsort(difficulty)[:n_samples]
    
    print(colored(f"\n[CURRICULUM] Epoch {epoch}: {n_samples}/{len(X_train)} samples (frac={frac:.2f})", "magenta"))
    
    # Train 1 epoch
    history = model.fit(
        X_train[indices], y_train[indices],
        validation_data=(X_val, y_val),
        batch_size=32,
        epochs=1,
        callbacks=[early_stopping],
        verbose=2,
        initial_epoch=epoch
    )
    
    # Check early stopping
    if model.stop_training:
        print(colored(f"\n✓ Early stopping déclenché à l'epoch {epoch}", "yellow"))
        break

print(colored(f"\n✓ Training terminé après {epoch + 1} epochs", "green"))
print(colored(f"  Global epochs du callback: {early_stopping.global_epoch}", "cyan"))
print(colored(f"  Meilleure accuracy: {early_stopping.best:.4f}", "cyan"))

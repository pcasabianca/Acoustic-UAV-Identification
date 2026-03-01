#!/usr/bin/env python3
"""
Tests unitaires pour le système de curriculum learning

Ce fichier teste:
1. Import et configuration de chaque trainer
2. Build model de chaque trainer
3. Curriculum learning avec données synthétiques
4. Compatibilité des formats (binary vs categorical)
"""
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Add tools and trainers
sys.path.insert(0, str(Path(__file__).parent.parent / 'tools'))
sys.path.insert(0, str(Path(__file__).parent.parent / '2 - Model Training'))

print("="*80)
print("TEST 1: Import Universal_Curriculum_Trainer et test build_model")
print("="*80)

try:
    from Universal_Curriculum_Trainer import (
        get_model_builder,
        build_model_from_trainer,
        CurriculumScheduler
    )
    print("✅ Universal_Curriculum_Trainer importé\n")
except Exception as e:
    print(f"❌ Erreur import: {e}")
    sys.exit(1)

trainers_to_test = [
    ('cnn', 'categorical'),
    ('efficientnet', 'binary'),
    ('mobilenet', 'binary'),
    ('ast', 'binary'),
    ('multires', 'binary'),
]

for model_type, output_format in trainers_to_test:
    try:
        print(f"{model_type} ({output_format}):")
        
        # Build model via Universal_Curriculum_Trainer
        input_shape = (44, 90, 1)
        model = build_model_from_trainer(input_shape, model_type)
        
        # Build and get output shape
        model.build((None, 44, 90, 1))
        output_shape = model.output_shape
        
        print(f"  ✅ Build successful")
        print(f"  Output shape: {output_shape}")
        print(f"  Expected format: {output_format}")
        
        # Verify format
        if output_format == 'binary':
            assert output_shape[-1] == 1, f"Binary should have 1 output, got {output_shape[-1]}"
        else:
            assert output_shape[-1] == 2, f"Categorical should have 2 outputs, got {output_shape[-1]}"
        
        print(f"  ✅ Output format verified\n")
        
        del model
        tf.keras.backend.clear_session()
        
    except Exception as e:
        print(f"  ❌ Error: {e}\n")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("TEST 2: Test de compilation avec les bonnes loss/metrics")
print("="*80)

for model_type, output_format in trainers_to_test:
    try:
        print(f"\n{model_type} ({output_format}):")
        
        # Build model
        model = build_model_from_trainer((44, 90, 1), model_type)
        
        # Configure loss and metrics based on format
        if output_format == 'binary':
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        
        # Compile
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss=loss,
            metrics=metrics
        )
        
        print(f"  ✅ Compilation successful with {loss}")
        
        # Test with synthetic data
        X_test = np.random.randn(10, 44, 90, 1).astype('float32')
        
        if output_format == 'binary':
            y_test = np.random.randint(0, 2, size=(10,)).astype('float32')
        else:
            y_test = np.random.randint(0, 2, size=(10,)).astype('int32')
        
        # Test prediction
        y_pred = model.predict(X_test, verbose=0)
        print(f"  ✅ Prediction successful, shape: {y_pred.shape}")
        
        # Test training for 1 epoch
        history = model.fit(
            X_test, y_test,
            batch_size=5,
            epochs=1,
            verbose=0
        )
        
        print(f"  ✅ Training successful")
        print(f"  Loss: {history.history['loss'][0]:.4f}")
        
        # Clear
        del model
        tf.keras.backend.clear_session()
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("TEST 3: Curriculum Learning avec données synthétiques")
print("="*80)

try:
    print("\nTest CNN avec curriculum:")
    
    # Generate synthetic data
    n_samples = 100
    X_train = np.random.randn(n_samples, 44, 90, 1).astype('float32')
    y_train = np.random.randint(0, 2, size=(n_samples,)).astype('int32')
    difficulty_train = np.random.rand(n_samples).astype('float32')
    
    X_val = np.random.randn(20, 44, 90, 1).astype('float32')
    y_val = np.random.randint(0, 2, size=(20,)).astype('int32')
    
    # Build model
    model = build_model_from_trainer((44, 90, 1), 'cnn')
    print(f"  ✅ Model built")
    
    # Compile with correct loss for categorical
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    print(f"  ✅ Model compiled")
    
    # Create curriculum scheduler
    scheduler = CurriculumScheduler(
        difficulty_scores=difficulty_train,
        strategy='linear',
        warmup_epochs=2,
        verbose=1
    )
    
    # Train epoch by epoch with curriculum
    print(f"  Testing curriculum training (3 epochs)...")
    for epoch in range(3):
        scheduler.on_epoch_begin(epoch, None)
        curr_indices = scheduler.current_indices
        
        X_curr = X_train[curr_indices]
        y_curr = y_train[curr_indices]
        
        print(f"    Epoch {epoch}: using {len(curr_indices)}/{n_samples} samples")
        
        history = model.fit(
            X_curr, y_curr,
            batch_size=16,
            epochs=1,
            verbose=0,
            validation_data=(X_val, y_val)
        )
        
        print(f"      Loss: {history.history['loss'][0]:.4f}, Val Acc: {history.history['val_accuracy'][0]:.4f}")
    
    print(f"  ✅ Curriculum training successful")
    
    tf.keras.backend.clear_session()
    
except Exception as e:
    print(f"  ❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("RÉSUMÉ DES TESTS")
print("="*80)
print("✅ Tests terminés")
print("\nSi tous les tests passent, le curriculum learning devrait fonctionner.")
print("Si des erreurs persistent, vérifier:")
print("  1. Les callbacks qui peuvent interférer")
print("  2. Les class_weights qui peuvent causer des problèmes de dimension")
print("  3. Les metrics incompatibles avec le format categorical")
print("="*80)

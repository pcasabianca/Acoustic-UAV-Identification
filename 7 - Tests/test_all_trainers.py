#!/usr/bin/env python3
"""
Test complet de tous les trainers individuellement.
Vérifie:
1. Import du trainer
2. Construction du modèle
3. Compilation
4. Entraînement court (2 epochs)
5. Prédiction
"""

import sys
import os
from pathlib import Path
import numpy as np
import tensorflow as tf

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from termcolor import colored

# Ajouter le répertoire des trainers au path
sys.path.insert(0, str(Path(config.PROJECT_ROOT) / '2 - Model Training'))

# Import de l'Universal_Curriculum_Trainer pour utiliser get_model_builder
from Universal_Curriculum_Trainer import get_model_builder, build_model_from_trainer

# Silence TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')


def create_synthetic_data(n_samples=100, output_format='categorical'):
    """Crée des données synthétiques pour le test."""
    X = np.random.randn(n_samples, 44, 90, 1).astype(np.float32)
    
    if output_format == 'binary':
        y = np.random.randint(0, 2, size=n_samples).astype(np.float32)
    else:  # categorical
        y = np.random.randint(0, 2, size=n_samples).astype(np.int32)
    
    return X, y


def get_metrics_for_format(output_format):
    """Retourne les métriques adaptées au format de sortie."""
    if output_format == 'binary':
        return ['accuracy', 
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')]
    else:  # categorical
        # Pour categorical, utiliser des métriques simples
        return ['accuracy']


def test_trainer(model_type, trainer_module):
    """Test un trainer individuel."""
    print(colored(f"\n{'='*80}", 'cyan'))
    print(colored(f"TEST: {model_type.upper()}", 'cyan', attrs=['bold']))
    print(colored(f"{'='*80}", 'cyan'))
    
    try:
        # Utiliser get_model_builder pour obtenir la fonction de build
        print(f"  1. Getting build_model function...", end=' ')
        build_model_func = get_model_builder(model_type)
        print(colored("✅", 'green'))
        
        # Détection du format de sortie
        output_format = config.TRAINER_OUTPUT_FORMATS.get(model_type, 'categorical')
        print(f"  2. Output format: {output_format}")
        
        # Création de données synthétiques
        X_train, y_train = create_synthetic_data(100, output_format)
        X_val, y_val = create_synthetic_data(20, output_format)
        print(f"  3. Synthetic data created: X={X_train.shape}, y={y_train.shape}")
        
        # Construction du modèle
        print(f"  4. Building model...", end=' ')
        input_shape = (44, 90, 1)
        
        # Certains modèles ont des paramètres spéciaux
        if model_type == 'conformer':
            model = build_model_func(input_shape, num_blocks=2)
        elif model_type in ['attention_crnn']:
            model = build_model_func(input_shape, num_classes=2)
        else:
            model = build_model_func(input_shape)
        
        print(colored("✅", 'green'))
        
        # Vérification de la couche de sortie
        output_layer = model.layers[-1]
        print(f"     Output layer: {output_layer.name}, units={output_layer.units if hasattr(output_layer, 'units') else 'N/A'}")
        
        # Compilation
        print(f"  5. Compiling model...", end=' ')
        
        # Choix de la loss function
        if output_format == 'binary':
            loss = 'binary_crossentropy'
        else:
            loss = 'sparse_categorical_crossentropy'
        
        metrics = get_metrics_for_format(output_format)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=loss,
            metrics=metrics
        )
        print(colored("✅", 'green'))
        print(f"     Loss: {loss}")
        print(f"     Metrics: {[m if isinstance(m, str) else m.name for m in metrics]}")
        
        # Entraînement court
        print(f"  6. Training (2 epochs)...", end=' ')
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=16,
            epochs=2,
            verbose=0
        )
        print(colored("✅", 'green'))
        print(f"     Final loss: {history.history['loss'][-1]:.4f}")
        print(f"     Final val_loss: {history.history['val_loss'][-1]:.4f}")
        
        # Prédiction
        print(f"  7. Prediction test...", end=' ')
        predictions = model.predict(X_val[:5], verbose=0)
        print(colored("✅", 'green'))
        print(f"     Prediction shape: {predictions.shape}")
        if output_format == 'binary':
            print(f"     Sample predictions: {predictions[:3].flatten()}")
        else:
            print(f"     Sample predictions (argmax): {np.argmax(predictions[:3], axis=1)}")
        
        print(colored(f"\n✅ {model_type.upper()} - ALL TESTS PASSED", 'green', attrs=['bold']))
        return True
        
    except Exception as e:
        print(colored(f"\n❌ {model_type.upper()} - FAILED", 'red', attrs=['bold']))
        print(colored(f"Error: {str(e)}", 'red'))
        import traceback
        traceback.print_exc()
        return False


def main():
    """Test tous les trainers."""
    print(colored("\n" + "="*80, 'magenta'))
    print(colored("  TEST COMPLET DE TOUS LES TRAINERS", 'magenta', attrs=['bold']))
    print(colored("="*80 + "\n", 'magenta'))
    
    # Liste de tous les trainers à tester
    trainers = {
        'cnn': 'CNN_Trainer',
        'rnn': 'RNN_Trainer',
        'crnn': 'CRNN_Trainer',
        'attention_crnn': 'Attention_CRNN_Trainer',
        'efficientnet': 'EfficientNet_Trainer',
        'mobilenet': 'MobileNet_Trainer',
        'conformer': 'Conformer_Trainer',
        'tcn': 'TCN_Trainer'
    }
    
    results = {}
    
    for model_type, trainer_module in trainers.items():
        success = test_trainer(model_type, trainer_module)
        results[model_type] = success
    
    # Résumé final
    print(colored("\n" + "="*80, 'magenta'))
    print(colored("  RÉSUMÉ DES TESTS", 'magenta', attrs=['bold']))
    print(colored("="*80 + "\n", 'magenta'))
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for model_type, success in results.items():
        status = colored("✅ PASSED", 'green') if success else colored("❌ FAILED", 'red')
        output_format = config.TRAINER_OUTPUT_FORMATS.get(model_type, 'categorical')
        print(f"  {model_type.upper():20s} ({output_format:12s}): {status}")
    
    print(f"\n  Total: {passed}/{total} passed")
    
    if passed == total:
        print(colored("\n🎉 TOUS LES TESTS SONT PASSÉS !", 'green', attrs=['bold']))
    else:
        print(colored(f"\n⚠️  {total - passed} test(s) ont échoué", 'yellow', attrs=['bold']))
    
    print()


if __name__ == '__main__':
    main()

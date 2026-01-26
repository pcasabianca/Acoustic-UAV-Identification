"""
Distance-Stratified Validation Callback
Evaluates model performance separately for each distance range during training.
This reveals the true performance on extreme distances (500m, 350m) which can be
hidden by high overall validation accuracy.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import json


class DistanceStratifiedCallback(keras.callbacks.Callback):
    """
    Callback that evaluates model performance separately for each distance.
    
    During training, computes accuracy for:
    - 500m samples only
    - 350m samples only  
    - 200m samples only
    - 100m samples only
    - 50m samples only
    - Overall accuracy (for comparison)
    
    Logs results to a JSON file for analysis.
    """
    
    def __init__(self, validation_dir, model_name, log_dir="results"):
        """
        Args:
            validation_dir: Path to validation dataset directory (e.g., "dataset_val")
            model_name: Name of model (e.g., "cnn", "rnn", "crnn")
            log_dir: Directory to save stratified metrics
        """
        super().__init__()
        self.validation_dir = Path(validation_dir)
        self.model_name = model_name.lower()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Will be populated in on_train_begin
        self.distance_data = {}
        self.epoch_metrics = []
        
    def _load_validation_data_by_distance(self):
        """Load and organize validation samples by distance."""
        import librosa
        import re
        
        print(f"\n[DistanceStratified] Loading validation data by distance...")
        
        # Auto-detect available distances from filenames
        print("[DistanceStratified] Auto-detecting distance categories from filenames...")
        drone_dir = self.validation_dir / "1"
        detected_distances = set()
        
        if drone_dir.exists():
            for wav_file in drone_dir.glob("*.wav"):
                # Extract distance from filename (e.g., "drone_600m_" or "aug_drone_700m_")
                match = re.search(r'drone_(\d+m)', wav_file.name)
                if match:
                    detected_distances.add(match.group(1))
        
        # Sort distances numerically (100m, 200m, 350m, 500m, 600m, 700m, etc.)
        distances = sorted(detected_distances, key=lambda x: int(x[:-1]))
        print(f"[DistanceStratified] Detected distances: {distances}")
        
        # Initialize distance buckets
        for dist in distances:
            self.distance_data[dist] = {
                'features': [],
                'labels': [],
                'filenames': []
            }
        
        # Also store ambient/background samples separately
        self.distance_data['ambient'] = {
            'features': [],
            'labels': [],
            'filenames': []
        }
        
        # Load drone samples (label 1)
        if drone_dir.exists():
            for wav_file in drone_dir.glob("*.wav"):
                filename = wav_file.name
                
                # Extract distance from filename
                distance = None
                match = re.search(r'drone_(\d+m)', filename)
                if match:
                    distance = match.group(1)
                
                if distance and distance in distances:
                    # Load audio and extract MEL spectrogram using centralized config values
                    try:
                        sr_cfg = int(getattr(__import__('config'), 'SAMPLE_RATE', 22050))
                        n_mels = int(getattr(__import__('config'), 'MEL_N_MELS', 44))
                        n_fft = int(getattr(__import__('config'), 'MEL_N_FFT', 2048))
                        hop_length = int(getattr(__import__('config'), 'MEL_HOP_LENGTH', 512))
                        target_frames = int(getattr(__import__('config'), 'MEL_TIME_FRAMES', 90))

                        audio, sr = librosa.load(wav_file, sr=sr_cfg)
                        mel = librosa.feature.melspectrogram(
                            y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
                        )
                        mel_db = librosa.power_to_db(mel, ref=np.max)

                        # Ensure consistent shape (n_mels, target_frames)
                        if mel_db.shape[1] < target_frames:
                            mel_db = np.pad(mel_db, ((0, 0), (0, target_frames - mel_db.shape[1])), mode='constant')
                        else:
                            mel_db = mel_db[:, :target_frames]

                        self.distance_data[distance]['features'].append(mel_db)
                        self.distance_data[distance]['labels'].append(1)
                        self.distance_data[distance]['filenames'].append(filename)

                    except Exception as e:
                        print(f"Error loading {filename}: {e}")
        
        # Load ambient samples (label 0)
        ambient_dir = self.validation_dir / "0"
        if ambient_dir.exists():
            try:
                sr_cfg = int(getattr(__import__('config'), 'SAMPLE_RATE', 22050))
                n_mels = int(getattr(__import__('config'), 'MEL_N_MELS', 44))
                n_fft = int(getattr(__import__('config'), 'MEL_N_FFT', 2048))
                hop_length = int(getattr(__import__('config'), 'MEL_HOP_LENGTH', 512))
                target_frames = int(getattr(__import__('config'), 'MEL_TIME_FRAMES', 90))

                audio, sr = librosa.load(wav_file, sr=sr_cfg)
                mel = librosa.feature.melspectrogram(
                    y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
                )
                mel_db = librosa.power_to_db(mel, ref=np.max)

                if mel_db.shape[1] < target_frames:
                    mel_db = np.pad(mel_db, ((0, 0), (0, target_frames - mel_db.shape[1])), mode='constant')
                else:
                    mel_db = mel_db[:, :target_frames]

                self.distance_data['ambient']['features'].append(mel_db)
                self.distance_data['ambient']['labels'].append(0)
                self.distance_data['ambient']['filenames'].append(wav_file.name)

            except Exception as e:
                print(f"Error loading {wav_file.name}: {e}")
        
        # Convert to numpy arrays
        for key in self.distance_data:
            if self.distance_data[key]['features']:
                self.distance_data[key]['features'] = np.array(self.distance_data[key]['features'])
                self.distance_data[key]['labels'] = np.array(self.distance_data[key]['labels'])
                
                # Add channel dimension for CNN/CRNN
                if len(self.distance_data[key]['features'].shape) == 3:
                    self.distance_data[key]['features'] = np.expand_dims(
                        self.distance_data[key]['features'], axis=-1
                    )
                
                print(f"  {key}: {len(self.distance_data[key]['labels'])} samples")
        
        print("[DistanceStratified] Validation data loaded.\n")
    
    def on_train_begin(self, logs=None):
        """Load validation data once at the start of training."""
        self._load_validation_data_by_distance()
    
    def on_epoch_end(self, epoch, logs=None):
        """Evaluate performance by distance at the end of each epoch."""
        if not self.distance_data:
            return
        
        metrics = {
            'epoch': epoch + 1,
            'overall_val_acc': logs.get('val_accuracy', 0),
            'overall_val_loss': logs.get('val_loss', 0),
            'distances': {}
        }
        
        print(f"\n{'='*70}")
        print(f"  Distance-Stratified Validation - Epoch {epoch + 1}")
        print(f"{'='*70}")
        
        # Evaluate each distance separately (auto-detected during loading)
        # Get all drone distances (exclude 'ambient')
        distances = [k for k in self.distance_data.keys() if k != 'ambient']
        # Sort numerically
        distances = sorted(distances, key=lambda x: int(x[:-1]))
        
        for dist in distances:
            if dist in self.distance_data and len(self.distance_data[dist]['features']) > 0:
                X = self.distance_data[dist]['features']
                y_true = self.distance_data[dist]['labels']
                
                # Predict
                y_pred = self.model.predict(X, verbose=0)
                
                # Handle both binary and categorical output
                if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                    y_pred_class = np.argmax(y_pred, axis=1)
                else:
                    y_pred_class = (y_pred.flatten() > 0.5).astype(int)
                
                # Calculate metrics
                accuracy = np.mean(y_pred_class == y_true)
                tp = np.sum((y_pred_class == 1) & (y_true == 1))
                fn = np.sum((y_pred_class == 0) & (y_true == 1))
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                
                metrics['distances'][dist] = {
                    'accuracy': float(accuracy),
                    'recall': float(recall),
                    'samples': int(len(y_true)),
                    'true_positives': int(tp),
                    'false_negatives': int(fn)
                }
                
                print(f"  {dist:5s}: Acc={accuracy*100:5.1f}% | Recall={recall*100:5.1f}% | "
                      f"TP={tp}/{len(y_true)} | Samples={len(y_true)}")
        
        # Evaluate ambient samples
        if 'ambient' in self.distance_data and len(self.distance_data['ambient']['features']) > 0:
            X = self.distance_data['ambient']['features']
            y_true = self.distance_data['ambient']['labels']
            
            y_pred = self.model.predict(X, verbose=0)
            if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                y_pred_class = np.argmax(y_pred, axis=1)
            else:
                y_pred_class = (y_pred.flatten() > 0.5).astype(int)
            
            accuracy = np.mean(y_pred_class == y_true)
            tn = np.sum((y_pred_class == 0) & (y_true == 0))
            fp = np.sum((y_pred_class == 1) & (y_true == 0))
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            metrics['distances']['ambient'] = {
                'accuracy': float(accuracy),
                'specificity': float(specificity),
                'samples': int(len(y_true)),
                'true_negatives': int(tn),
                'false_positives': int(fp)
            }
            
            print(f"  Ambient: Acc={accuracy*100:5.1f}% | Specificity={specificity*100:5.1f}% | "
                  f"TN={tn}/{len(y_true)}")
        
        print(f"{'='*70}\n")
        
        # Save metrics
        self.epoch_metrics.append(metrics)
        
        # Write to file after each epoch
        log_file = self.log_dir / f"{self.model_name}_distance_metrics.json"
        with open(log_file, 'w') as f:
            json.dump(self.epoch_metrics, f, indent=2)
    
    def on_train_end(self, logs=None):
        """Generate summary report at end of training."""
        if not self.epoch_metrics:
            return
        
        print(f"\n{'='*70}")
        print(f"  Distance-Stratified Training Summary - {self.model_name.upper()}")
        print(f"{'='*70}")
        
        # Find best epoch for each distance (auto-detected)
        distances = [k for k in self.distance_data.keys() if k != 'ambient']
        distances = sorted(distances, key=lambda x: int(x[:-1]))
        
        for dist in distances:
            best_epoch = None
            best_acc = 0
            
            for metric in self.epoch_metrics:
                if dist in metric['distances']:
                    acc = metric['distances'][dist]['accuracy']
                    if acc > best_acc:
                        best_acc = acc
                        best_epoch = metric['epoch']
            
            if best_epoch:
                print(f"  {dist}: Best Acc={best_acc*100:.1f}% at Epoch {best_epoch}")
        
        print(f"{'='*70}\n")

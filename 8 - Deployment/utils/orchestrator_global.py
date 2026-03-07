"""Unified global orchestrator (utils).

This self-contained orchestrator manages the recorder, feature extractor
and model inference. It enforces RAM-only storage and provides both
`run_once()` and `run_continuous()` entry points. The module intentionally
avoids importing external orchestrators to remain standalone.
"""
from __future__ import annotations
import json
import threading
import time
import logging
from queue import Queue, Empty
from typing import Any, Optional, Tuple
from pathlib import Path
import csv
import io
from datetime import datetime
import numpy as np

from .audio_buffer import AudioBuffer
from .recorder import RecorderFactory
from .keras_utils import load_models_from_config
from .tflite_utils import load_tflite_models_from_config
from .predictor_utils import run_models, load_thresholds_from_config
from .features_utils import extract_mel, FeatureExtractor
from .ram_utils import RamStore


class OrchestratorGlobal:
    def __init__(self, config_path: str = 'deployment_config.json', logger: Optional[logging.Logger] = None,
                 save_to_disk: bool = False, csv_log_path: Optional[str] = None):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.logger = logger or logging.getLogger('OrchestratorGlobal')
        # ensure logger has a stream handler so INFO prints appear on stdout
        if not self.logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
            self.logger.addHandler(h)
        self.logger.setLevel(logging.INFO)

        # Save options
        self.save_to_disk = bool(save_to_disk)
        self.csv_log_path = csv_log_path

        # Initialize recorder (prefer Raspberry, fallback to laptop)
        # Pass the orchestrator logger so recorder warnings use the same logger
        self.rec = RecorderFactory.get_recorder(config_path=config_path, prefer='auto', use_ram=True, use_disk=self.save_to_disk, logger=self.logger)

        # Respect explicit save_to_disk: if not set, enforce RAM-only
        try:
            if not self.save_to_disk:
                if hasattr(self.rec, 'use_disk'):
                    self.rec.use_disk = False
                if hasattr(self.rec, 'save_debug'):
                    self.rec.save_debug = False
            else:
                if hasattr(self.rec, 'use_disk'):
                    self.rec.use_disk = True
        except Exception:
            pass

        # Models (feature extraction via utils.features_utils.extract_mel)
        # feature extractor compatible with legacy core implementation
        self.feature_extractor = FeatureExtractor(self.config, logger=self.logger)
        # Env var DRONE_INFERENCE_BACKEND overrides config.
        import os as _os
        _backend = (
            _os.environ.get('DRONE_INFERENCE_BACKEND')
            or self.config.get('inference_backend', 'keras')
        ).strip().lower()
        if _backend not in ('keras', 'tflite'):
            self.logger.warning(
                f'[Orch] Unknown inference_backend "{_backend}"; defaulting to keras'
            )
            _backend = 'keras'
        self.logger.info(f'[Orch] inference_backend={_backend}')
        self.models = self._load_models_with_fallback(_backend)

        # Determine desired n_mels from model input shapes (fallback to config)
        try:
            cfg_mel = int(self.config.get('feature_extraction', {}).get('mel_spectrogram', {}).get('n_mels', 44))
        except Exception:
            cfg_mel = 44
        model_n_mels = []
        try:
            for name, info in self.models.items():
                ish = info.get('input_shape')
                if ish is None:
                    continue
                # ish may be (None, H, W, C) or (H, W, C)
                try:
                    if len(ish) == 4:
                        model_n_mels.append(int(ish[1]))
                    elif len(ish) == 3:
                        model_n_mels.append(int(ish[0]))
                except Exception:
                    continue
        except Exception:
            model_n_mels = []

        self.desired_n_mels = int(max(model_n_mels) if model_n_mels else cfg_mel)

        # CSV logging setup
        try:
            self.model_names = list(self.models.keys())
        except Exception:
            self.model_names = []
        if self.csv_log_path:
            # ensure directory exists
            try:
                Path(self.csv_log_path).parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            # prepare fieldnames (include feature_key)
            self._csv_fieldnames = ['timestamp', 'ram_key', 'disk_path', 'feature_key'] + self.model_names

        # feature RAM store (stores serialized numpy arrays)
        try:
            max_items = int(self.config.get('recording', {}).get('ram_max_items', 8))
        except Exception:
            max_items = 8
        self._feature_store = RamStore(max_items=max_items)

        # Internal state
        self._stop_evt = threading.Event()
        self._threads = []

    # ------------------------------------------------------------------
    def _load_models_with_fallback(self, preferred_backend: str) -> dict:
        """Load models using the preferred backend; fall back per-model to the
        other backend if the preferred file is absent.

        Priority order for each model:
          - preferred backend file exists  →  load with preferred loader
          - preferred file absent, other backend file exists  →  load with
            fallback loader and emit a warning
          - neither file exists  →  skip and warn
        """
        from pathlib import Path as _Path
        from .tflite_utils import _tflite_filename_for, TFLitePredictor, _TFLiteInterpreter, _TF_AVAILABLE
        from .keras_utils import load_single_model

        models_dir = _Path(self.config['models']['directory'])
        enabled = self.config['detection']['enabled_models']
        available = self.config['models']['available_models']

        # Load thresholds once
        thresholds: dict = {}
        try:
            thresh_file = self.config['detection'].get('thresholds_file')
            if thresh_file:
                import json as _json
                with open(thresh_file, 'r') as _f:
                    _data = _json.load(_f)
                for _k, _v in _data.get('models', {}).items():
                    thresholds[_k] = _v.get('threshold', 0.5)
        except Exception:
            pass

        alt_backend = 'keras' if preferred_backend == 'tflite' else 'tflite'
        loaded: dict = {}

        for name in enabled:
            if name not in available:
                self.logger.warning(f'[Orch] Model {name} not declared in available_models; skipping')
                continue

            info = available[name]
            keras_fn  = info.get('filename', '')
            tflite_fn = _tflite_filename_for(info)
            keras_path  = models_dir / keras_fn  if keras_fn  else None
            tflite_path = models_dir / tflite_fn if tflite_fn else None

            keras_exists  = bool(keras_path  and keras_path.exists())
            tflite_exists = bool(tflite_path and tflite_path.exists())

            # Determine which backend to actually use for this model
            if preferred_backend == 'tflite':
                use_backend = 'tflite' if tflite_exists else ('keras' if keras_exists else None)
            else:
                use_backend = 'keras' if keras_exists else ('tflite' if tflite_exists else None)

            if use_backend is None:
                self.logger.warning(
                    f'[Orch] No model file found for {name} '
                    f'(tried {keras_path}, {tflite_path}); skipping'
                )
                continue

            if use_backend != preferred_backend:
                self.logger.warning(
                    f'[Orch] {name}: {preferred_backend} file not found — '
                    f'falling back to {use_backend} ({(tflite_path if use_backend=="tflite" else keras_path).name})'
                )

            calibrated_key = info.get('calibrated_key', name)
            entry: dict = {
                'threshold':    thresholds.get(calibrated_key, 0.5),
                'description':  info.get('description', ''),
                'architecture': info.get('architecture'),
                'predict_fn':   None,
            }

            if use_backend == 'tflite':
                if not _TF_AVAILABLE or _TFLiteInterpreter is None:
                    self.logger.error(f'[Orch] TFLite not available; cannot load {name}')
                    continue
                try:
                    interp = _TFLiteInterpreter(model_path=str(tflite_path))
                    predictor = TFLitePredictor(interp)
                    entry['model']       = predictor
                    entry['input_shape'] = predictor.input_shape
                    self.logger.info(f'[Orch] Loaded {name} via TFLite ({tflite_fn})')
                except Exception as exc:
                    self.logger.error(f'[Orch] Failed to load TFLite model {name}: {exc}')
                    continue
            else:  # keras
                # Skip models that require unsafe deserialization (Keras-2 Lambda)
                if not info.get('safe_mode', True):
                    self.logger.warning(
                        f'[Orch] {name}: safe_mode=False (Keras-2 Lambda layers) — skipping. '
                        'Re-save from the training environment to fix.'
                    )
                    continue
                model = load_single_model(keras_path, safe_mode=True, timeout=30)
                if model is None:
                    self.logger.error(f'[Orch] Failed to load Keras model {name} from {keras_path}')
                    continue
                entry['model']       = model
                entry['input_shape'] = getattr(model, 'input_shape', None)
                self.logger.info(f'[Orch] Loaded {name} via Keras ({keras_fn})')

            loaded[name] = entry

        return loaded

    # ------------------------------------------------------------------
    def run_once(self) -> dict:
        """Record one buffer, run features + models, return full result."""
        # Record into RAM/disk depending on settings: call record_audio(), build AudioBuffer and store WAV bytes
        ram_key = None
        disk_path = None
        # Use recorder.record_and_store() so recorder can log and return metadata
        rec_start_ts = datetime.now()
        rec_start = time.time()
        meta = self.rec.record_and_store()
        rec_end = time.time()
        # meta should include 'audio', 'ram_key', 'disk_path', and 'low_variation'
        audio = meta.get('audio')
        sr = int(getattr(self.rec, 'sample_rate', self.config['audio']['sample_rate']))
        buf = AudioBuffer.from_array(audio, samplerate=sr) if audio is not None else None
        ram_key = meta.get('ram_key') if isinstance(meta, dict) else None
        disk_path = meta.get('disk_path') if isinstance(meta, dict) else None
        low_variation = bool(meta.get('low_variation')) if isinstance(meta, dict) else False

        # Extract features (respect desired n_mels)
        features = extract_mel(audio_array=buf.array, sr=buf.sr, config_path=self.config.get('config_path', 'deployment_config.json'), mel_overrides={'n_mels': self.desired_n_mels})

        # Run models using the compatibility FeatureExtractor so features
        # are adapted exactly like the legacy `core` implementation.
        res = run_models(self.models, self.feature_extractor, features, logger=self.logger)

        # Attach thresholds
        thresholds = load_thresholds_from_config(self.config)
        details = {}
        for name, prob in res['predictions'].items():
            details[name] = {
                'probability': prob,
                'threshold': self.models[name].get('threshold', thresholds.get(name, 0.5)),
            }

        return {
            'audio_buffer': buf,
            'ram_key': ram_key,
            'disk_path': disk_path,
            'predictions': res['predictions'],
            'timings_ms': res['timings_ms'],
            'inference_time_ms': res['total_inference_ms'],
            'details': details,
        }

    def run_continuous(self):
        """Run pipelined continuous recording + processing.

        Pipeline:
          recorder -> rec->feat queue -> feature extractor -> feat->model queue -> model inference

        Ensures recordings are RAM-only and workers communicate via in-memory queues.
        """
        interval = float(self.config['detection']['scan_interval_seconds'])

        q_rec_feat: Queue = Queue(maxsize=4)
        q_feat_model: Queue = Queue(maxsize=4)

        def recorder_worker():
            self.logger.info('[Orch] Recorder started')
            while not self._stop_evt.is_set():
                try:
                    # Record audio and timestamp using recorder.record_and_store()
                    rec_start = time.time()
                    rec_start_ts = datetime.now()
                    meta = self.rec.record_and_store()
                    rec_end = time.time()
                    self.logger.info(f'[Orch] Recorder: finished record (duration_ms={(rec_end-rec_start)*1000:.0f})')

                    # meta should include 'audio', 'ram_key', 'disk_path', 'low_variation'
                    audio = None
                    if isinstance(meta, dict):
                        audio = meta.get('audio')
                        ram_key = meta.get('ram_key')
                        disk_path = meta.get('disk_path')
                        low_variation = bool(meta.get('low_variation', False))
                    else:
                        ram_key = None
                        disk_path = None
                        low_variation = False

                    if audio is None:
                        self.logger.warning('[Orch] Recorder returned no audio; skipping this cycle')
                        # schedule next
                        try:
                            elapsed = time.time() - rec_start
                            sleep_for = interval - elapsed
                            if sleep_for > 0:
                                time.sleep(sleep_for)
                        except Exception:
                            time.sleep(interval)
                        continue

                    sr = int(getattr(self.rec, 'sample_rate', self.config['audio']['sample_rate']))
                    buf = AudioBuffer.from_array(audio, samplerate=sr)

                    # Log low-variation if detected
                    if low_variation:
                        try:
                            self.logger.warning(f'[Orch] Recorder: low variation detected for ram_key={ram_key}')
                        except Exception:
                            pass

                    # push tuple (buffer, ram_key, disk_path)
                    try:
                        q_rec_feat.put((buf, ram_key, disk_path), timeout=1.0)
                        self.logger.info(f'[Orch] Recorder: queued buf ram_key={ram_key} disk_path={disk_path} low_var={low_variation}')
                    except Exception:
                        try:
                            q_rec_feat.get_nowait()
                        except Exception:
                            pass
                        try:
                            q_rec_feat.put((buf, ram_key, disk_path), block=False)
                            self.logger.info(f'[Orch] Recorder: queued buf ram_key={ram_key} after drop low_var={low_variation}')
                        except Exception:
                            self.logger.warning('[Orch] Recorder: failed to queue buffer')

                except Exception as e:
                    self.logger.error(f'[Orch] Recorder error: {e}')

                # schedule next recording to achieve start-to-start interval
                try:
                    elapsed = time.time() - rec_start
                    sleep_for = interval - elapsed
                    if sleep_for > 0:
                        time.sleep(sleep_for)
                except Exception:
                    # if rec_start is not defined or other error, fallback to simple sleep
                    time.sleep(interval)

            self.logger.info('[Orch] Recorder stopped')

        def feature_worker():
            self.logger.info('[Orch] Feature worker started')
            while not self._stop_evt.is_set() or not q_rec_feat.empty():
                try:
                    buf, ram_key, disk_path = q_rec_feat.get(timeout=0.5)
                    self.logger.info(f'[Orch] Feature: got buf ram_key={ram_key} disk_path={disk_path}')
                except Empty:
                    continue
                try:
                    t0 = time.time()
                    feats = extract_mel(audio_array=buf.array, sr=buf.sr, config_path=self.config.get('config_path', 'deployment_config.json'), mel_overrides={'n_mels': self.desired_n_mels})
                    t1 = time.time()
                    self.logger.info(f'[Orch] Feature: extracted feats (ms={(t1-t0)*1000:.0f}) ram_key={ram_key}')
                    try:
                        # Drop any older feature entries so model_worker always
                        # processes the most recent buffer (reduce pipeline lag).
                        try:
                            while not q_feat_model.empty():
                                try:
                                    _old = q_feat_model.get_nowait()
                                    try:
                                        q_feat_model.task_done()
                                    except Exception:
                                        pass
                                except Exception:
                                    break
                            self.logger.info('[Orch] Feature: dropped older queued feats to prioritize newest')
                        except Exception:
                            pass

                        # serialize features into feature_store and enqueue the feature_key
                        try:
                            bio = io.BytesIO()
                            np.save(bio, feats, allow_pickle=False)
                            bfeat = bio.getvalue()
                            feature_key = self._feature_store.store(None, bfeat)
                        except Exception as e:
                            self.logger.warning(f'[Orch] Failed to store serialized features: {e}')
                            feature_key = None

                        q_feat_model.put((feature_key, buf, ram_key, disk_path), timeout=1.0)
                        self.logger.info(f'[Orch] Feature: queued feature_key={feature_key} ram_key={ram_key} disk_path={disk_path}')
                    except Exception:
                        try:
                            q_feat_model.get_nowait()
                        except Exception:
                            pass
                        try:
                            q_feat_model.put((feature_key, buf, ram_key), block=False)
                            self.logger.info(f'[Orch] Feature: queued feature_key={feature_key} ram_key={ram_key} after drop')
                        except Exception:
                            self.logger.warning('[Orch] Feature: failed to queue feats')
                except Exception as e:
                    self.logger.error(f'[Orch] Feature extraction failed: {e}')
                finally:
                    try:
                        q_rec_feat.task_done()
                    except Exception:
                        pass

            self.logger.info('[Orch] Feature worker stopped')

        def model_worker():
            self.logger.info('[Orch] Model worker started')
            while not self._stop_evt.is_set() or not q_feat_model.empty():
                try:
                    feats, buf, ram_key, disk_path = q_feat_model.get(timeout=0.5)
                except Empty:
                    continue
                try:
                    # If feats is a feature_key (string), load serialized features from feature_store
                    feature_key = None
                    if isinstance(feats, str) or feats is None:
                        feature_key = feats
                        try:
                            b = self._feature_store.get(feature_key) if feature_key is not None else None
                            if b is None:
                                self.logger.warning(f'[Orch] Missing serialized features for key={feature_key}; skipping')
                                continue
                            bio = io.BytesIO(b)
                            feats = np.load(bio, allow_pickle=False)
                        except Exception as e:
                            self.logger.error(f'[Orch] Failed to load serialized features for key={feature_key}: {e}')
                            continue

                    res = run_models(self.models, self.feature_extractor, feats, logger=self.logger)
                    thresholds = load_thresholds_from_config(self.config)
                    details = {}
                    for name, prob in res['predictions'].items():
                        details[name] = {
                            'probability': prob,
                            'threshold': self.models[name].get('threshold', thresholds.get(name, 0.5)),
                        }
                    # Log concise inference summary
                    self.logger.info(f"[Orch] Inference: {res['total_inference_ms']} ms | ram_key={ram_key}")
                    # Log detailed per-model predictions including per-model timing
                    timings = res.get('timings_ms', {}) or {}
                    for mname, det in details.items():
                        prob = det['probability']
                        thresh = det['threshold']
                        t_ms = timings.get(mname, None)
                        flag = 'ALARM' if prob >= thresh else 'ok'
                        if t_ms is not None:
                            self.logger.info(f"  - {mname}: prob={prob:.4f} thresh={thresh:.3f} time={int(t_ms)}ms => {flag}")
                        else:
                            self.logger.info(f"  - {mname}: prob={prob:.4f} thresh={thresh:.3f} => {flag}")

                    # Summary statistics across models
                    try:
                        probs = [float(d['probability']) for d in details.values()]
                        total = len(probs)
                        alerts = 0
                        for name, d in details.items():
                            if float(d['probability']) >= float(d['threshold']):
                                alerts += 1
                        import statistics as _stats
                        pmin = min(probs) if probs else 0.0
                        pmax = max(probs) if probs else 0.0
                        pmean = _stats.mean(probs) if probs else 0.0
                        self.logger.info(f"[Orch] Summary: min={pmin:.4f} max={pmax:.4f} mean={pmean:.4f} alerts={alerts}/{total}")
                    except Exception:
                        pass

                    # Remove RAM buffer after processing if recorder supports it
                    try:
                        if ram_key and hasattr(self.rec, 'delete_ram_key'):
                            ok = self.rec.delete_ram_key(ram_key)
                            self.logger.info(f"[Orch] Deleted ram_key={ram_key}: {ok}")
                    except Exception as e:
                        self.logger.warning(f"[Orch] Failed to delete ram_key={ram_key}: {e}")
                    # Remove serialized feature after processing
                    try:
                        if feature_key:
                            okf = self._feature_store.delete(feature_key)
                            self.logger.info(f"[Orch] Deleted feature_key={feature_key}: {okf}")
                    except Exception as e:
                        self.logger.warning(f"[Orch] Failed to delete feature_key={feature_key}: {e}")

                    # Append CSV row if configured
                    try:
                        if self.csv_log_path:
                            # prepare row
                            ts = datetime.now().isoformat()
                            row = {'timestamp': ts, 'ram_key': ram_key or '', 'disk_path': disk_path or '', 'feature_key': feature_key or ''}
                            for m in self.model_names:
                                row[m] = res['predictions'].get(m, '')
                            exists = Path(self.csv_log_path).exists()
                            with open(self.csv_log_path, 'a', newline='') as cf:
                                writer = csv.DictWriter(cf, fieldnames=self._csv_fieldnames)
                                if not exists:
                                    writer.writeheader()
                                writer.writerow(row)
                    except Exception as e:
                        self.logger.warning(f"[Orch] Failed to append CSV log: {e}")
                except Exception as e:
                    self.logger.error(f'[Orch] Model inference failed: {e}')
                finally:
                    try:
                        q_feat_model.task_done()
                    except Exception:
                        pass

            self.logger.info('[Orch] Model worker stopped')

        # start threads
        t_rec = threading.Thread(target=recorder_worker, daemon=True)
        t_feat = threading.Thread(target=feature_worker, daemon=True)
        t_model = threading.Thread(target=model_worker, daemon=True)
        self._threads = [t_rec, t_feat, t_model]
        for t in self._threads:
            t.start()

        try:
            count = 0
            while True:
                time.sleep(1.0)
                count += 1
                # every 5 seconds emit a heartbeat showing thread liveness
                if count % 5 == 0:
                    try:
                        alive = [t.is_alive() for t in self._threads]
                        self.logger.info(f'[Orch] heartbeat threads_alive={alive}')
                    except Exception:
                        pass
        except KeyboardInterrupt:
            self.logger.info('[Orch] Shutdown requested')
            self.stop()

    def stop(self):
        self.logger.info('[Orch] Stop requested')
        self._stop_evt.set()
        # join threads
        for t in self._threads:
            try:
                t.join(timeout=2.0)
            except Exception:
                pass


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='OrchestratorGlobal runner (diagnostics)')
    parser.add_argument('--once', action='store_true', help='Run a single record->predict cycle and exit')
    parser.add_argument('--config', default='deployment_config.json')
    args = parser.parse_args()

    print('OrchestratorGlobal starting (diagnostic mode)')
    og = OrchestratorGlobal(config_path=args.config)

    # Basic diagnostics
    print('  Config file:', args.config)
    try:
        print('  Recorder:', type(og.rec), 'sample_rate=', getattr(og.rec, 'sample_rate', og.config['audio']['sample_rate']))
    except Exception:
        print('  Recorder: <unknown>')
    print('  Desired n_mels:', og.desired_n_mels)
    print('  Loaded models:')
    for n, info in og.models.items():
        ish = info.get('input_shape')
        print(f'    - {n}: input_shape={ish}, threshold={info.get("threshold")}, arch={info.get("architecture")}')

    if args.once:
        print('\nRunning single cycle (run_once)')
        try:
            out = og.run_once()
            print('Single run result:')
            print('  ram_key:', out.get('ram_key'))
            print('  predictions:', out.get('predictions'))
            print('  timings_ms:', out.get('timings_ms'))
        except Exception as e:
            print('run_once failed:', e)
    else:
        print('\nStarting continuous run — press Ctrl-C to stop')
        try:
            og.run_continuous()
        except Exception as e:
            print('run_continuous terminated with exception:', e)

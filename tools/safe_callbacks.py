import time
from tensorflow import keras

class SafeEarlyStopping(keras.callbacks.EarlyStopping):
    """
    Extension of Keras EarlyStopping that supports a `start_from_epoch` argument.
    Training will ignore early-stopping until the current epoch index is >= start_from_epoch.
    """
    def __init__(self, start_from_epoch=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._start_from_epoch = int(start_from_epoch or 0)

    def on_epoch_end(self, epoch, logs=None):
        # Only delegate to the parent after start_from_epoch
        if epoch < self._start_from_epoch:
            return
        return super().on_epoch_end(epoch, logs)

class TimeStopping(keras.callbacks.Callback):
    """
    Stop training after a given number of seconds have elapsed.
    """
    def __init__(self, max_seconds, verbose=0):
        super().__init__()
        self.max_seconds = float(max_seconds)
        self.verbose = verbose
        self._start_time = None

    def on_train_begin(self, logs=None):
        self._start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if self._start_time is None:
            return
        elapsed = time.time() - self._start_time
        if elapsed >= self.max_seconds:
            self.model.stop_training = True
            if self.verbose:
                print(f"[TimeStopping] Stopping training after {elapsed:.1f}s >= {self.max_seconds}s")

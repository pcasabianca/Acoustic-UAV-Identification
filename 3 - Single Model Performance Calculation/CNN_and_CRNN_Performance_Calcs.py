import os
import json
import librosa
import tensorflow as tf
import numpy as np
from termcolor import colored

# Read and save parameters.
DATASET_PATH = "..."  # Path of folder with testing audios
SAVED_MODEL_PATH = ".../model_1.h5"  # Path of trained model
SAMPLE_RATE = 22050  # Sample rate in Hz.
DURATION = 1  # Length of audio files fed. Measured in seconds.
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

# Predictions (1 or 0)
JSON_PATH = ".../predictions.json"
# Performance scores (accuracy, precision, recall, f1 score)
JSON_PERFORMANCE = ".../model_scores.json"

# Prediction of fed audio
class _Class_Predict_Service:
    """Singleton class for keyword spotting inference with trained models.
    :param model: Trained model
    """
    # Mapping so drone = 1.
    model = None
    _mapping = [
        1,
        0
    ]
    _instance = None

    # Predict hard values (1 or 0).
    def predict(self, file_path):
        """
        :param file_path (str): Path to audio file to predict
        :return predicted_keyword (str): Keyword predicted by the model
        """

        # Extract mels from testing audio.
        log_mel = self.preprocess(file_path)

        # We need a 4-dim array to feed to the model for prediction: (# samples, # time steps, # coefficients, 1).
        log_mel = log_mel[np.newaxis, ..., np.newaxis]

        # Get the predicted label.
        predictions = self.model.predict(log_mel)
        predicted_index = np.argmax(predictions)
        predicted_class = self._mapping[predicted_index]
        return predicted_class

    # Outputs certainty values for soft voting (1-0).
    def preprocess(self, file_path, n_mels=90, n_fft=2048, hop_length=512, num_segments=1):
        """Extract MFCCs from audio file.
        :param file_path (str): Path of audio file
        :param n_mels (int): # of mels to extract
        :param n_fft (int): Interval we consider to apply STFT. Measured in # of samples
        :param hop_length (int): Sliding window for STFT. Measured in # of samples
        """

        num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)

        # Load audio file.
        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

        # Process segments extracting mels and storing data.
        for s in range(num_segments):
            start_sample = num_samples_per_segment * s  # s=0 --> 0
            finish_sample = start_sample + num_samples_per_segment  # s=0 --> num_samples_per_segment

            # Extract mel specs.
            mel = librosa.feature.melspectrogram(signal[start_sample:finish_sample], sr=sr, n_mels=n_mels, n_fft=n_fft,
                                                 hop_length=hop_length)
            log_mel = librosa.power_to_db(mel)

        return log_mel.T


def Keyword_Spotting_Service():
    """Factory function for Keyword_Spotting_Service class.
    :return _Keyword_Spotting_Service._instance (_Keyword_Spotting_Service):
    """

    # Ensure an instance is created only the first time the factory function is called.
    if _Class_Predict_Service._instance is None:
        _Class_Predict_Service._instance = _Class_Predict_Service()
        _Class_Predict_Service.model = tf.keras.models.load_model(SAVED_MODEL_PATH)
    return _Class_Predict_Service._instance


# Saving results into a json file.
def save_mfcc(dataset_path, json_path):
    # Dictionary to store data.
    data = {
        "mapping": [],  # Maps different class labels --> background is mapped to 0.
        "names": [],
        "results": [],  # mels are the training input, labels are the target.
    }

    # Loop through all the classes.
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # Ensure that we're not at the root level.
        if dirpath is not dataset_path:

            # Save the semantic label.
            dirpath_components = dirpath.split("/")  # class/background => ["class", "background"]
            semantic_label = dirpath_components[-1]  # considering only the last value
            data["mapping"].append(semantic_label)
            print("\nProcessing {}".format(semantic_label))

            # Process files for a specific class.
            for f in filenames:
                file_path = os.path.join(dirpath, f)

                # Create 2 instances of the keyword spotting service.
                kss = Keyword_Spotting_Service()
                kss1 = Keyword_Spotting_Service()

                # Check that different instances of the keyword spotting service point back to the same object.
                assert kss is kss1

                # Classify unseen audio.
                keyword = kss.predict(file_path)

                # Store mel for segment if it has the expected length.
                data["names"].append(f)
                data["results"].append(keyword)
                print("{}".format(file_path))

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


# Calculating performance scores (accuracy, precision, recall, f-score).
def performance_calcs(performance_path):
    # Dictionary to store model performance results.
    performance = {
        "TP": [],
        "FN": [],
        "TN": [],
        "FP": [],
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "F1 Score": [],
    }

    with open(JSON_PATH, "r") as fp:
        data = json.load(fp)

    # Convert lists to numpy arrays.
    y = np.array(data["results"])

    a = float(sum(y[0:int(len(y) / 2)]))
    b = float(sum(y[int(len(y) / 2):int(len(y))]))

    # Calculating TP, TN, FP, FN.
    TP = a
    FN = int(len(y) / 2) - a
    FP = b
    TN = int(len(y) / 2) - b

    # Performance result calcs.
    Accuracy = (TP + TN) / (TP + TN + FN + FP)
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1 = (2 * Precision * Recall) / (Precision + Recall)

    performance["TP"].append(TP)
    performance["FN"].append(FN)
    performance["TN"].append(TN)
    performance["FP"].append(FP)
    performance["Accuracy"].append(Accuracy)
    performance["Precision"].append(Precision)
    performance["Recall"].append(Recall)
    performance["F1 Score"].append(F1)

    with open(performance_path, "w") as fp:
        json.dump(performance, fp, indent=4)


if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH)
    performance_calcs(JSON_PERFORMANCE)

    print(
        colored("CRNN model performance scores have been saved to {}.".format(JSON_PERFORMANCE), "green"))

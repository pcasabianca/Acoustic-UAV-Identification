import os
import json
import librosa
import tensorflow as tf
import numpy as np
from termcolor import colored

# Read and save parameters.
DATASET_PATH = "Unseen Testing"  # Path of testing dataset.
SAMPLE_RATE = 22050
DURATION = 1  # Measured in seconds (change depending on length of one audio file).
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

# Model testing.
SAVED_MODEL = ".../model_1.h5"  # Path of trained model (change for all models trained).
# Raw predictions: 1, 0 results (for hard voting) and certainty values (for soft voting).
RESULTS = ".../voted_1.json"  # Path to save raw predictions.
# Performance scores (accuracy, precision, recall, f-score).
MODEL_SCORES = ".../scores_1.json"


# Prediction of fed audio.
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
        mel = self.preprocess(file_path)

        # We need a 4-dim array to feed to the model for prediction: (# samples, # time steps, # coefficients, 1).
        mel = mel[np.newaxis, ..., np.newaxis]

        # Get the predicted label.
        predictions = self.model.predict(mel)
        predicted_index = np.argmax(predictions)
        predicted_class = self._mapping[predicted_index]
        return predicted_class

    # Outputs certainty values for soft voting (1-0).
    def predict_prob(self, file_path):
        """
        :param file_path (str): Path to audio file to predict
        :return predicted_keyword (str): Keyword predicted by the model
        """

        # Extract mels from testing audio.
        mel = self.preprocess(file_path)

        # We need a 4-dim array to feed to the model for prediction: (# samples, # time steps, # coefficients, 1).
        mel = mel[np.newaxis, ..., np.newaxis]

        # Get the predicted label.
        predict_prob = self.model.predict_proba(mel)[:, 0]
        return predict_prob

    # Extract mel specs from raw audio.
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
            db_mel = librosa.power_to_db(mel)

        return db_mel.T


def Keyword_Spotting_Service():
    """Factory function for Keyword_Spotting_Service class.
    :return _Keyword_Spotting_Service._instance (_Keyword_Spotting_Service):
    """

    # Ensure an instance is created only the first time the factory function is called.
    if _Class_Predict_Service._instance is None:
        _Class_Predict_Service._instance = _Class_Predict_Service()
        _Class_Predict_Service.model = tf.keras.models.load_model(SAVED_MODEL)
    return _Class_Predict_Service._instance


# Saving results into a json file.
def save_prediction(dataset_path, json_path):

    # Dictionary to store data.
    data = {
        "mapping": [],
        "names": [],  # audio file names
        "results": [],  # hard results (0 or 1)
        "certainties": [],  # soft results (0-1)
    }

    # Loop through all the classes.
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # Ensure that we're not at the root level.
        if dirpath is not dataset_path:

            # Save the semantic label.
            dirpath_components = dirpath.split("/")  # uav/uav.wav => ["uav", "uav.wav"]
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
                prob = kss.predict_prob(file_path)
                certainty = float(prob)

                # Store mel for segment if it has the expected length.
                data["names"].append(f)
                data["results"].append(keyword)
                data["certainties"].append(certainty)
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

    with open(RESULTS, "r") as fp:
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
    save_prediction(DATASET_PATH, RESULTS)
    performance_calcs(MODEL_SCORES)

    print(
        colored("Model performance scores have been saved to {}.".format(MODEL_SCORES), "green"))

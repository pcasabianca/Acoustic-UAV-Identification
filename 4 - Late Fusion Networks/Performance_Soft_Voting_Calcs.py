from termcolor import colored
import numpy as np
import json
import sys

# Soft Voting Section.

# Output results (storing voting model performance results).
RESULTS = ".../voted_results.json"
JSON_WEIGHTS = ".../weights.json"  # Weights used to make weighted predictions (created during script run).
VOTED_MODELS_SCORES = ".../FINAL_SCORES.json"  # Soft voting performance scores output.

# json files with model confidence results (10 models).
cert_1 = ".../voted_1.json"
cert_2 = ".../voted_2.json"
cert_3 = ".../voted_3.json"
cert_4 = ".../voted_4.json"
cert_5 = ".../voted_5.json"
cert_6 = ".../voted_6.json"
cert_7 = ".../voted_7.json"
cert_8 = ".../voted_8.json"
cert_9 = ".../voted_9.json"
cert_10 = ".../voted_10.json"

Accuracy = ".../all_models_acc.json"  # Accuracy results from Validation 2 subset test (to calculate weights).


# Calculating weights of each voting model.
def weights(model_accuracy_json):
    weights = {
        "weights": [],
    }

    with open(Accuracy) as f:
        acc = json.load(f)

    x = np.array(acc["model_acc"])
    average = np.average(x)
    print("\nAverage accuracy is {}".format(average))

    num = list(range(0, 10))

    # Calculating weights.
    for i in num:
        y = x[i] - average
        weight = (1 / 10) + y
        weights["weights"].append(weight)

    with open(model_accuracy_json, "w") as fp:
        json.dump(weights, fp, indent=4)


# Making voted predictions (uav or background?).
def prob_result(json_path):
    # Dictionary to store data.
    results = {
        "combined_certainties": [],
        "voted_predictions": [],
    }

    with open(cert_1) as a, open(cert_2) as b, open(cert_3) as c, open(cert_4) as d, open(cert_5) as e, open(
            cert_6) as f, open(cert_7) as g, open(cert_8) as h, open(cert_9) as j, open(cert_10) as k, open(
            JSON_WEIGHTS) as v:
        data_1 = json.load(a)
        data_2 = json.load(b)
        data_3 = json.load(c)
        data_4 = json.load(d)
        data_5 = json.load(e)
        data_6 = json.load(f)
        data_7 = json.load(g)
        data_8 = json.load(h)
        data_9 = json.load(j)
        data_10 = json.load(k)
        weights = json.load(v)

    l = np.array(data_1["certainties"])
    m = np.array(data_2["certainties"])
    n = np.array(data_3["certainties"])
    o = np.array(data_4["certainties"])
    p = np.array(data_5["certainties"])
    q = np.array(data_6["certainties"])
    r = np.array(data_7["certainties"])
    s = np.array(data_8["certainties"])
    t = np.array(data_9["certainties"])
    u = np.array(data_10["certainties"])
    w = np.array(weights["weights"])

    # Multiplying certainty values by weights to find average certainty result.
    z = (l * w[0]) + (m * w[1]) + (n * w[2]) + (o * w[3]) + (p * w[4]) + (q * w[5]) + (r * w[6]) + (s * w[7]) + (
                t * w[8]) + (u * w[9])

    # Finalising prediction output from voted models.
    for i in z:
        i = round(i, 10)
        if i > 1 or i < 0:
            print(i)
            print(colored("Error: The average value has been computed incorrectly. Please check.", "red"))
            sys.exit(1)
        elif i >= 0.5:
            vote = 1
        else:
            vote = 0
        results["combined_certainties"].append(i)
        results["voted_predictions"].append(vote)

    with open(json_path, "w") as fp:
        json.dump(results, fp, indent=4)


def performance_calcs(performance_path):
    # Dictionary to store data.
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

    # Convert lists to numpy arrays --> same process as single model performance predicts.
    y = np.array(data["voted_predictions"])

    a = float(sum(y[0:int(len(y) / 2)]))
    b = float(sum(y[int(len(y) / 2):int(len(y))]))

    TP = a
    FN = int(len(y) / 2) - a
    FP = b
    TN = int(len(y) / 2) - b

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
    weights(JSON_WEIGHTS)
    prob_result(RESULTS)
    performance_calcs(VOTED_MODELS_SCORES)

    print(colored("Process completed successfully!", "green"))
